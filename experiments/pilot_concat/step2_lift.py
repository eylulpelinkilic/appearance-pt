"""Pilot step 2: lift DINOv2 patch features onto the points of a few val scenes, with the visual checks.

For 3 val scenes (fewest, median and most vertices, from the download manifest; or --scenes) it
  1. picks 30 of the scene's kept DSLR views, evenly over the capture order (is_bad frames were never kept),
  2. on the first scene, scores the pose convention: agreement of vertex colours with the photos for the toolbox
     inverse (appearance_pt.scannetpp_cameras.dslr_c2w) and two wrong alternatives,
  3. computes DINOv2 ViT-L/14 patch features at 518 x 770 px and lifts them onto every vertex (lifting.py),
  4. reports how many vertices (and labelled vertices) no view sees, views per vertex, and time per stage,
  5. saves the visual checks to <out>: overlay_<scene>.png (photo; projected points coloured by depth; vertex colours
     rendered from that camera; DINO patch PCA of the photo; lifted features rendered with the same PCA; views per
     vertex) and topdown_<scene>.png (vertex colours | lifted DINO PCA, never-seen vertices grey).
It also records free disk and RAM for planning step 3. Results: <out>/step2_summary.json. The images are
ScanNet++-derived, so they stay on Kaggle.

Usage: python experiments/pilot_concat/step2_lift.py [--input /kaggle/input] [--out DIR] [--scenes ID ...] [--views 30]
Needs PYTHONPATH=src:experiments/pilot_concat (the notebook sets it), a GPU and Internet (Hugging Face weights).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from appearance_pt import REPO_ROOT
from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.scannetpp_cameras import dslr_c2w, dslr_frames, dslr_intrinsics, pick_views
from appearance_pt.scannetpp_store import SceneArchive, anonymised_vertices, decode_image, find_scenes, manifest_row
from lifting import DINO, DINO_HW, Z_NEAR, View, colour_agreement, dino_patches, lift, load_dino, pca3, preprocess, project, visible, zbuffer

CONVENTIONS = {
    "toolbox inverse (expected)": dslr_c2w,
    "OpenGL camera, world as stored": lambda T: np.asarray(T, dtype=np.float64) @ np.diag([1.0, -1.0, -1.0, 1.0]),
    "OpenCV camera, world as stored": lambda T: np.asarray(T, dtype=np.float64),
}
EXPECTED = "toolbox inverse (expected)"
PANEL = 0.5  # overlay panels at half resolution
GREY = np.array([110, 110, 110], np.uint8)


def sync(dev: str) -> None:
    if dev == "cuda":
        torch.cuda.synchronize()


def machine() -> dict:
    disk = {p: round(shutil.disk_usage(p).free / 1e9, 1) for p in ("/kaggle/working", "/kaggle/tmp", "/tmp") if os.path.exists(p)}
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()) if Path("/proc/meminfo").exists() else {}
    ram = {k: round(int(meminfo[k].split()[0]) / 2**20, 1) for k in ("MemTotal", "MemAvailable") if k in meminfo}
    return dict(disk_free_gb=disk, ram_gb=ram, cpus=os.cpu_count())


def mask_valid(m: np.ndarray) -> np.ndarray:
    m = m[..., 0] if m.ndim == 3 else m
    return m > 0 if m.dtype == bool else m > 127


def load_scene(path: Path, scene: str, n_views: int) -> dict:
    with SceneArchive(path) as arc:
        pc = f"pointcept/val/{scene}/"
        s = {k: arc.npy(f"{pc}{k}.npy") for k in ("coord", "color", "segment")}
        s["anonymised"] = np.unique(anonymised_vertices(arc))
        transforms = arc.json("dslr/nerfstudio/transforms_undistorted.json")
        frames = dslr_frames(transforms)
        s["K"], s["hw"] = dslr_intrinsics(transforms)
        s["views"] = []
        for name in pick_views(arc.json("dslr/kept_views.json")["views"], n_views):
            mask = frames[name].get("mask_path")
            valid = (mask_valid(decode_image(arc.read(f"dslr/resized_undistorted_masks/{Path(mask).name}"))) if mask
                     else np.ones(s["hw"], bool))
            s["views"].append(dict(name=name, T=frames[name]["transform_matrix"], valid=valid,
                                   jpg=arc.read(f"dslr/resized_undistorted_images/{name}")))
    return s


def make_views(points, s, convention, dev, indices) -> list[View]:
    K = torch.tensor(s["K"], dtype=torch.float32, device=dev)
    views = []
    for i in indices:
        v = s["views"][i]
        w2c = torch.tensor(np.linalg.inv(CONVENTIONS[convention](v["T"])), dtype=torch.float32, device=dev)
        view = View(v["name"], w2c, K, s["hw"], torch.from_numpy(v["valid"]).to(dev))
        view.zbuf = zbuffer(points, view)
        views.append(view)
    return views


def agreement(points, s, convention, dev, indices, colors, keep) -> dict:
    rows = []
    for i, view in zip(indices, make_views(points, s, convention, dev, indices)):
        image = torch.from_numpy(decode_image(s["views"][i]["jpg"]).copy()).to(dev)
        rows.append(colour_agreement(points, colors, keep, view, image))
    rs = [r["r"] for r in rows if r["r"] is not None]
    return dict(r_mean=round(float(np.mean(rs)), 3) if rs else None, r_per_view=[r["r"] for r in rows],
                visible_frac=round(float(np.mean([r["n_visible"] for r in rows])) / len(points), 4))


def jet(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1)[:, None]
    return (np.clip(1.5 - np.abs(4 * t - np.array([3.0, 2.0, 1.0])), 0, 1) * 255).astype(np.uint8)


def render(u, v, z, colours, hw, background=None, scale=PANEL) -> np.ndarray:
    """Points splatted into a (scale x hw) image, the nearest point per pixel winning."""
    h, w = int(hw[0] * scale), int(hw[1] * scale)
    img = np.zeros((h, w, 3), np.uint8) if background is None else background.copy()
    ok = (z > Z_NEAR) & (u >= 0) & (u < hw[1]) & (v >= 0) & (v < hw[0])
    flat = np.minimum(v[ok] * scale, h - 1).astype(np.int64) * w + np.minimum(u[ok] * scale, w - 1).astype(np.int64)
    order = np.lexsort((z[ok], flat))
    first = np.unique(flat[order], return_index=True)[1]
    img.reshape(-1, 3)[flat[order][first]] = colours[ok][order][first]
    return img


def topdown(coord, colours, max_px=1400, px=0.02) -> np.ndarray:
    """Seen from above, below 2 m over the floor (so the ceiling does not hide the room); the highest point wins."""
    keep = coord[:, 2] < np.percentile(coord[:, 2], 1) + 2.0
    xy, z, col = coord[keep, :2], coord[keep, 2], colours[keep]
    lo, extent = xy.min(0), xy.max(0) - xy.min(0)
    px = max(px, float(extent.max()) / max_px)
    w, h = (extent / px).astype(int) + 1
    ix = ((xy - lo) / px).astype(np.int64)
    flat = (h - 1 - ix[:, 1]) * w + ix[:, 0]
    order = np.lexsort((-z, flat))
    first = np.unique(flat[order], return_index=True)[1]
    img = np.zeros((h * w, 3), np.uint8)
    img[flat[order][first]] = col[order][first]
    return img.reshape(h, w, 3)


def labelled(img: np.ndarray, text: str) -> Image.Image:
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, 8 + 6 * len(text), 18), fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255))
    return im


def grid(panels: list[Image.Image], cols: int) -> Image.Image:
    w, h = panels[0].size
    out = Image.new("RGB", (w * cols, h * -(-len(panels) // cols)))
    for i, p in enumerate(panels):
        out.paste(p.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    return out


def run_scene(model, path, scene, args, dev, conventions: list[str]) -> dict:
    t = {}
    t0 = time.perf_counter()
    s = load_scene(path, scene, args.views)
    t["load_s"] = time.perf_counter() - t0
    n = len(s["coord"])
    points = torch.from_numpy(s["coord"]).to(dev)
    colors = torch.from_numpy(s["color"]).to(dev)
    keep = torch.ones(n, dtype=torch.bool, device=dev)
    keep[torch.from_numpy(s["anonymised"]).to(dev)] = False
    check_idx = [int(i) for i in np.unique(np.linspace(0, len(s["views"]) - 1, 5).round())]
    out = dict(scene=scene, n_vertices=n, n_views=len(s["views"]), K=np.round(s["K"], 2).tolist(), hw=list(s["hw"]),
               mask_valid_frac=round(float(np.mean([v["valid"].mean() for v in s["views"]])), 4))
    print(f"  {n:,} vertices, {len(s['views'])} views, K {out['K']}, image {s['hw']}, mask valid {out['mask_valid_frac']:.3f}")

    scores = out["pose_agreement"] = {c: agreement(points, s, c, dev, check_idx, colors, keep) for c in conventions}
    for c, a in scores.items():
        print(f"  pose convention {c:34s} colour r {a['r_mean']}  visible {a['visible_frac']:.3f}  per view {a['r_per_view']}")
    best = out["convention"] = max(scores, key=lambda c: -2 if scores[c]["r_mean"] is None else scores[c]["r_mean"])
    if best != EXPECTED:
        print(f"  !! the expected convention is not the best: using {best!r}; see the overlay")

    sync(dev)
    t0 = time.perf_counter()
    views = make_views(points, s, best, dev, range(len(s["views"])))
    sync(dev)
    t["zbuffer_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    images = [preprocess(v["jpg"]) for v in s["views"]]
    t["decode_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    feats = torch.cat([dino_patches(model, torch.stack(images[b:b + args.batch]).to(dev)) for b in range(0, len(images), args.batch)])
    sync(dev)
    t["dino_s"] = time.perf_counter() - t0
    for view, f in zip(views, feats):
        view.feat = f

    # one PCA basis from the patches of all views (mostly valid ones), used for the photos and the points alike
    ph, pw = feats.shape[-2:]
    patch_ok = torch.stack([F.adaptive_avg_pool2d(v.valid.float()[None, None], (ph, pw))[0, 0] > 0.5 for v in views])
    patches = feats.permute(0, 2, 3, 1)[patch_ok].float()
    mean, basis = pca3(patches)
    proj = (patches - mean) @ basis
    lo, hi = torch.quantile(proj, 0.01, dim=0), torch.quantile(proj, 0.99, dim=0)

    def to_rgb(x):
        return (((x - mean) @ basis - lo) / (hi - lo)).clamp(0, 1).mul(255).byte()

    t0 = time.perf_counter()
    count = torch.empty(n, dtype=torch.int32, device=dev)
    rgb = torch.empty(n, 3, dtype=torch.uint8, device=dev)
    for start, feat, cnt in lift(points, views, chunk=args.chunk):
        count[start:start + len(cnt)] = cnt
        rgb[start:start + len(cnt)] = to_rgb(feat)
    sync(dev)
    t["lift_s"] = time.perf_counter() - t0
    out["seconds"] = {k: round(v, 2) for k, v in t.items()}

    count_np, rgb_np = count.cpu().numpy(), rgb.cpu().numpy()
    seen = count_np > 0
    lab = (s["segment"][:, 0] >= 0) & keep.cpu().numpy()
    bins = [(0, 0), (1, 1), (2, 4), (5, 9), (10, 19), (20, 10**6)]
    out.update(seen=round(float(seen.mean()), 4), seen_labelled=round(float(seen[lab].mean()), 4),
               views_per_seen=dict(mean=round(float(count_np[seen].mean()), 2), median=float(np.median(count_np[seen]))),
               views_hist={f"{a}-{b}" if b < 10**6 else f"{a}+": round(float(((count_np >= a) & (count_np <= b)).mean()), 4) for a, b in bins})
    print(f"  seen by >= 1 view: {out['seen']:.3f} of vertices, {out['seen_labelled']:.3f} of labelled; views per seen vertex "
          f"{out['views_per_seen']}; histogram {out['views_hist']}")
    print(f"  seconds {out['seconds']}")

    # visual checks, from the middle view
    mid = len(views) // 2
    view = views[mid]
    photo = decode_image(s["views"][mid]["jpg"])
    u, v, z = (x.cpu().numpy() for x in project(points, view.w2c, view.K))
    vis = visible(*project(points, view.w2c, view.K), view).cpu().numpy()
    small = np.asarray(Image.fromarray(photo).resize((int(s["hw"][1] * PANEL), int(s["hw"][0] * PANEL))))
    zv = z[vis]
    depth_col = jet((zv - np.percentile(zv, 5)) / max(np.percentile(zv, 95) - np.percentile(zv, 5), 1e-6))
    rgb_seen = np.where(seen[:, None], rgb_np, GREY)
    patch_pca = to_rgb(feats[mid].permute(1, 2, 0).reshape(-1, feats.shape[1]).float()).cpu().numpy().reshape(ph, pw, 3)
    counts_col = jet(count_np / max(10, np.percentile(count_np[seen], 95) if seen.any() else 1))
    counts_col[~seen] = 0
    panels = [
        labelled(small, f"{scene} {view.name}: photo"),
        labelled(render(u[vis], v[vis], z[vis], depth_col, s["hw"], background=(small * 0.35).astype(np.uint8)),
                 "visible points by depth, on the photo"),
        labelled(render(u, v, z, s["color"], s["hw"]), "vertex colours, rendered from this camera"),
        labelled(np.asarray(Image.fromarray(patch_pca).resize(small.shape[1::-1], Image.NEAREST)), "DINO patch PCA of the photo"),
        labelled(render(u, v, z, rgb_seen, s["hw"]), "lifted DINO (all views), same PCA; grey = never seen"),
        labelled(render(u, v, z, counts_col, s["hw"]), "views per vertex (blue few, red many; black none)"),
    ]
    files = [args.out / f"overlay_{scene}.png", args.out / f"topdown_{scene}.png"]
    grid(panels, 3).save(files[0])
    grid([labelled(topdown(s["coord"], s["color"]), f"{scene}: vertex colours, from above"),
          labelled(topdown(s["coord"], rgb_seen), "lifted DINO PCA; grey = never seen")], 2).save(files[1])
    out["images"] = [f.name for f in files]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/pilot_concat/step2"))
    ap.add_argument("--scenes", nargs="*", help="val scene ids (default: fewest, median and most vertices)")
    ap.add_argument("--views", type=int, default=30)
    ap.add_argument("--batch", type=int, default=6, help="images per DINOv2 forward pass")
    ap.add_argument("--chunk", type=int, default=500_000, help="vertices lifted at once")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        print("!! no GPU visible: set Accelerator to GPU T4 x2")
        return 2
    record = collect()
    (args.out / "env_record.json").write_text(json.dumps(record, indent=2))
    box = machine()
    print(f"git {json.dumps(record['git'])}\nmachine {json.dumps(box)}\nargs {vars(args)}")

    section("1. Val scenes (core datasets below --input)")
    archives = {s: p[0] for s, p in find_scenes(args.input)["core"].items()}
    subset = json.loads((REPO_ROOT / "configs" / "scannetpp_subset.json").read_text())
    cache = {}
    val = {s: manifest_row(archives[s], cache) for s in subset["val"] if s in archives}
    if not val:
        print("!! no val scenes found: attach the private dataset scannetpp-core-val")
        return 2
    by_size = sorted(val, key=lambda s: val[s].get("n_vertices", 0))
    scenes = args.scenes or [by_size[0], by_size[len(by_size) // 2], by_size[-1]]
    print(f"  {len(val)} val scenes; checking {scenes} ({[val[s].get('n_vertices') for s in scenes]} vertices)")

    section(f"2. DINOv2 {DINO[0]} @ {DINO[1][:10]}, input {DINO_HW[0]} x {DINO_HW[1]} px")
    import transformers

    dev = "cuda"
    model = load_dino(dev)
    print(f"  transformers {transformers.__version__}, {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters, "
          f"attention {getattr(model.config, '_attn_implementation', '?')}")

    rows = []
    for i, scene in enumerate(scenes):
        section(f"3.{i + 1} Scene {scene}")
        torch.cuda.reset_peak_memory_stats()
        conventions = list(CONVENTIONS) if i == 0 else [rows[0]["convention"]]  # the first scene decides
        rows.append(run_scene(model, archives[scene], scene, args, dev, conventions))
        rows[-1]["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        torch.cuda.empty_cache()

    summary = dict(args={k: str(v) for k, v in vars(args).items()}, git=record["git"], machine=box, dino=DINO, scenes=rows)
    (args.out / "step2_summary.json").write_text(json.dumps(summary, indent=1, default=str))

    section("SUMMARY step 2 (paste this block back)")
    first = rows[0]
    print(f"pose convention (colour agreement r, first scene, 5 views): "
          f"{json.dumps({c: a['r_mean'] for c, a in first['pose_agreement'].items()})}; used: {sorted({r['convention'] for r in rows})}")
    print(f"intrinsics {first['K']}, image {first['hw']}")
    for r in rows:
        sec = r["seconds"]
        print(f"{r['scene']}: {r['n_vertices']:,} vertices, {r['n_views']} views, mask valid {r['mask_valid_frac']:.3f}, "
              f"colour r {r['pose_agreement'][r['convention']]['r_mean']}; seen {r['seen']:.3f} (labelled {r['seen_labelled']:.3f}); "
              f"views/seen vertex {r['views_per_seen']['mean']}; hist {json.dumps(r['views_hist'])}")
        print(f"   seconds: load {sec['load_s']}, decode {sec['decode_s']}, DINO {sec['dino_s']}, z-buffers {sec['zbuffer_s']}, "
              f"lift {sec['lift_s']}; peak GPU {r['peak_gpu_gb']} GB")
    per_view = np.mean([r["seconds"]["dino_s"] / r["n_views"] for r in rows])
    per_mvv = np.mean([r["seconds"]["lift_s"] / (r["n_vertices"] / 1e6 * r["n_views"]) for r in rows])
    per_zb = np.mean([r["seconds"]["zbuffer_s"] / (r["n_vertices"] / 1e6 * r["n_views"]) for r in rows])
    print(f"rates: DINO {per_view:.3f} s/view; lifting {per_mvv:.3f} s per million vertex-views; z-buffer {per_zb:.3f} s per million vertex-views")
    print(f"machine: {json.dumps(box)}")
    print(f"images (Output tab, {args.out}): {[f for r in rows for f in r['images']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
