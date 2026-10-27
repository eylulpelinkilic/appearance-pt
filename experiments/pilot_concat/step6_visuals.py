"""Pilot visuals: the segmentation maps behind the step-5 numbers, on a few val scenes.

The pilot's task is semantic segmentation: a linear probe on frozen per-point features gives every vertex of a val
scene's mesh one of the 100 benchmark classes. For each scene (default: SCENES, picked from step 5's per-scene
results) this computes the toolbox's GT and the top-1 predictions of three seed-0 probes, A (lifted DINO only),
K (Concerto) and K+ (Concerto + DINO), exactly as step 5 does, and saves to <out>:
  - <scene>_view1.png and <scene>_view2.png: two of the scene's lifted DSLR views as six panels. Top row: the photo,
    the GT, and where K and K+ differ (green: only K+ right; red: only K right; light grey: both right; dark grey:
    both wrong). Bottom row: A, K and K+. A pixel shows the label of the mesh vertex nearest to where its ray first
    hits the mesh (Open3D), at half resolution; a legend lists the classes in view. View 1 is the view where K+ fixes
    the most pixels net of those it breaks; view 2 sits at the median of that score. The captions say so.
  - <scene>_topdown.png: GT, K, K+ and the difference, seen from above (below 2 m over the floor, as in step 2).
Per scene it checks that the three probes' scene mIoU equals step 5's (seed 0, from confmats.npz), which shows these
are the evaluated predictions. A prediction counts as right if it is any of the vertex's GT labels, as in the metric.
The images are ScanNet++-derived, so they stay on Kaggle. Results: <out>/visuals.json.

Usage: python experiments/pilot_concat/step6_visuals.py [--input /kaggle/input] [--probes .../probes.pt] [--out DIR]
       [--scenes ID ...]
Needs PYTHONPATH=src:third_party/utonia:third_party/sonata:third_party/concerto, a GPU, open3d and Internet (weights).
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.scannetpp_cameras import dslr_c2w
from appearance_pt.scannetpp_store import SceneArchive, decode_image, find_scenes
from encoders import attention_backend, encode_scene, load_encoder, make_transform, scene_seed
from lifting import OCCLUSION, lift, load_dino
from metrics import confusion_topk, miou
from official import OfficialGT
from pipeline import add_dino, scene_views
from probes import NUM_CLASSES, FoldedSet
from scenes import load_core_scene
from step2_lift import grid, machine, topdown

SCENES = {  # val scene: (why it is shown, step 5's scene mIoU of the seed-0 probes, from confmats.npz)
    "1ada7a0617": ("a large gain from DINO", {"A#0": 39.06, "K#0": 37.70, "K+#0": 46.99}),
    "7b6477cb95": ("about the average gain", {"A#0": 42.02, "K#0": 48.52, "K+#0": 52.38}),
    "27dd4da69e": ("39 % of its vertices never seen by the 30 views", {"A#0": 12.34, "K#0": 27.33, "K+#0": 24.78}),
    "c4c04e6d6c": ("the largest loss from DINO", {"A#0": 20.39, "K#0": 45.84, "K+#0": 37.62}),
}
PROBES = {"A#0": "DINO only (A)", "K#0": "Concerto (K)", "K+#0": "Concerto + DINO (K+)"}
OUTCOMES = ("only K+ right", "only K right", "both right", "both wrong")
OUTCOME_RGB = np.array([(40, 190, 70), (220, 50, 50), (190, 190, 190), (95, 95, 95)], np.uint8)
NO_LABEL = np.array([20, 20, 20], np.uint8)  # unlabelled or anonymised vertex, masked pixel, or a ray that hits nothing
QUIET = {"wall": (205, 200, 190), "floor": (140, 115, 90), "ceiling": (235, 235, 230)}  # large surfaces stay calm
MATCH_TOL = 0.1  # mIoU points: GPU kernels may differ in the last bits between sessions


def palette(names: list[str]) -> np.ndarray:
    """(C, 3) uint8 class colours: hues stepped by the golden ratio, three saturations and values."""
    out = []
    for i, name in enumerate(names):
        if name in QUIET:
            out.append(QUIET[name])
            continue
        rgb = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, (0.55, 0.85, 0.7)[i % 3], (0.95, 0.75, 0.85)[(i // 3) % 3])
        out.append(tuple(int(255 * c) for c in rgb))
    return np.array(out, np.uint8)


def label_colours(labels: np.ndarray, pal: np.ndarray, keep: np.ndarray) -> np.ndarray:
    col = np.repeat(NO_LABEL[None], len(labels), 0)
    ok = keep & (labels >= 0) & (labels < len(pal))
    col[ok] = pal[labels[ok]]
    return col


def outcome(pred_k: np.ndarray, pred_kp: np.ndarray, gt: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Per vertex, an index into OUTCOMES, or -1 where the metric does not count the vertex."""
    counted = keep & (gt >= 0).any(1)
    right_k, right_kp = (pred_k[:, None] == gt).any(1), (pred_kp[:, None] == gt).any(1)
    out = np.full(len(gt), -1, np.int8)
    out[counted & right_kp & ~right_k] = 0
    out[counted & right_k & ~right_kp] = 1
    out[counted & right_k & right_kp] = 2
    out[counted & ~right_k & ~right_kp] = 3
    return out


def outcome_colours(out: np.ndarray) -> np.ndarray:
    return np.where((out >= 0)[:, None], OUTCOME_RGB[np.maximum(out, 0)], NO_LABEL)


def pixel_rays(w2c: np.ndarray, K: np.ndarray, hw, scale: float) -> tuple[np.ndarray, tuple[int, int]]:
    """(h * w, 6) float32 rays (camera centre, unit direction) through the pixel centres of the image at `scale`, in
    lifting.project's convention (pixel (0, 0) spans [0, 1)^2 at full size), and (h, w)."""
    h, w = int(hw[0] * scale), int(hw[1] * scale)
    v, u = np.meshgrid((np.arange(h) + 0.5) / scale, (np.arange(w) + 0.5) / scale, indexing="ij")
    cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    R, t = w2c[:3, :3], w2c[:3, 3]
    d = cam @ R  # camera to world: R.T @ c for each row c
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.hstack([np.broadcast_to(-R.T @ t, d.shape), d]).astype(np.float32), (h, w)


def nearest_vertex(faces: np.ndarray, prim: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Per ray, the vertex of the hit triangle with the largest barycentric weight (Open3D's primitive_uvs are the
    weights of the triangle's 2nd and 3rd vertex); -1 for a miss (primitive id INVALID_ID)."""
    prim = prim.astype(np.int64)
    hit = (prim >= 0) & (prim < len(faces))
    w = np.stack([1 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], 1)
    out = np.full(len(prim), -1, np.int64)
    out[hit] = faces[prim[hit], w[hit].argmax(1)]
    return out


class MeshRays:
    """Rays cast at a val scene's mesh_aligned_0.05.ply, whose vertices are the scene's vertices (decisions.md P6)."""

    def __init__(self, path: Path, n_vertices: int, tmp: Path):
        import open3d as o3d

        tmp.mkdir(parents=True, exist_ok=True)
        ply = tmp / "mesh_aligned_0.05.ply"
        with SceneArchive(path) as arc:
            ply.write_bytes(arc.read("scans/mesh_aligned_0.05.ply"))
        mesh = o3d.io.read_triangle_mesh(str(ply))
        ply.unlink()
        assert len(mesh.vertices) == n_vertices, f"mesh has {len(mesh.vertices)} vertices, the scene {n_vertices}"
        self.faces = np.asarray(mesh.triangles, dtype=np.int64)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def vertex_ids(self, w2c: np.ndarray, K: np.ndarray, hw, scale: float) -> np.ndarray:
        """(h, w) vertex id per pixel, -1 where the ray hits nothing."""
        import open3d as o3d

        rays, (h, w) = pixel_rays(w2c, K, hw, scale)
        ans = self.scene.cast_rays(o3d.core.Tensor(rays))
        return nearest_vertex(self.faces, ans["primitive_ids"].numpy(), ans["primitive_uvs"].numpy()).reshape(h, w)


def load_models(concerto: str, device: str):
    model, _ = load_encoder(concerto, seed=0)
    return model, make_transform(concerto), load_dino(device)


@torch.inference_mode()
def predict(model, transform, dino, folded: FoldedSet, s: dict, scene: str, occ, dev: str, chunk: int = 200_000):
    """{probe: top-1 class per vertex} and the number of views that see each vertex, computed as in step 5."""
    stages, _ = encode_scene(model, transform, s, scene_seed(scene), store="cpu")
    points = torch.from_numpy(s["coord"]).to(dev)
    normals = torch.from_numpy(s["normal"]).float().to(dev)
    views = scene_views(s, points, occ, device=dev)
    add_dino(dino, s, views, device=dev)
    n = len(s["coord"])
    preds = {k: np.zeros(n, np.int16) for k in folded.keys}
    seen = np.zeros(n, np.int32)
    for start, feat, count in lift(points, views, normals, chunk=chunk):
        idx = torch.arange(start, start + len(count))
        logits = folded.logits({"concerto": stages.gather(idx, device=dev), "dino": feat})
        for k, lg in logits.items():
            preds[k][start:start + len(count)] = lg.argmax(1).cpu().numpy()
        seen[start:start + len(count)] = count.cpu().numpy()
    return preds, seen


def scene_miou(pred: np.ndarray, gt: np.ndarray, keep: np.ndarray) -> float:
    cm = confusion_topk(torch.from_numpy(pred[keep].astype(np.int64))[:, None], torch.from_numpy(gt[keep]), NUM_CLASSES)
    return round(100 * float(miou(cm.numpy())), 2)


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: fixed-size bitmap font
        return ImageFont.load_default()


def captioned(img: np.ndarray, text: str, f) -> Image.Image:
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    box = draw.textbbox((10, 8), text, font=f)
    draw.rectangle((box[0] - 6, box[1] - 5, box[2] + 6, box[3] + 5), fill=(0, 0, 0))
    draw.text((10, 8), text, fill=(255, 255, 255), font=f)
    return im


def legend(title: str, entries: list[tuple[str, tuple]], width: int, f) -> Image.Image:
    """Colour swatches with names, wrapped to `width`, under a title line."""
    pad, sw, gap = 14, 18, 24
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    rows, row, x = [], [], pad
    for name, col in entries:
        need = sw + 8 + measure.textlength(name, font=f) + gap
        if row and x + need > width - pad:
            rows.append(row)
            row, x = [], pad
        row.append((name, col, need))
        x += need
    if row:
        rows.append(row)
    text_h = measure.textbbox((0, 0), "Ag", font=f)[3]
    line = max(sw, text_h) + 12
    im = Image.new("RGB", (width, 2 * pad + line * (len(rows) + 1)), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.text((pad, pad), title, fill=(60, 60, 60), font=f)
    for r, items in enumerate(rows, 1):
        x, y = pad, pad + line * r
        for name, col, need in items:
            top = y + (text_h - sw) // 2
            draw.rectangle((x, top, x + sw, top + sw), fill=tuple(int(c) for c in col), outline=(90, 90, 90))
            draw.text((x + sw + 8, y), name, fill=(30, 30, 30), font=f)
            x += need
    return im


def stacked(*images: Image.Image) -> Image.Image:
    out = Image.new("RGB", (max(i.width for i in images), sum(i.height for i in images)), (255, 255, 255))
    y = 0
    for im in images:
        out.paste(im, (0, y))
        y += im.height
    return out


def class_entries(counts: np.ndarray, names: list[str], pal: np.ndarray, limit: int = 40) -> list[tuple[str, tuple]]:
    order = [c for c in np.argsort(-counts, kind="stable") if counts[c] > 0][:limit]
    return [(names[c], tuple(pal[c])) for c in order]


def paint(ids: np.ndarray, colours: np.ndarray) -> np.ndarray:
    img = np.repeat(NO_LABEL[None], ids.size, 0)
    flat = ids.ravel()
    img[flat >= 0] = colours[flat[flat >= 0]]
    return img.reshape(*ids.shape, 3)


def run_scene(scene: str, path: Path, models, folded, gt_source, names, pal, occ, n_views, args, dev) -> dict:
    t0 = time.perf_counter()
    s = load_core_scene(path, scene, n_views)
    n = len(s["coord"])
    gt = gt_source.labels(scene)
    keep = np.ones(n, bool)
    keep[s["anonymised"]] = False
    preds, seen = predict(*models, folded, s, scene, occ, dev)
    m = {k: scene_miou(preds[k], gt, keep) for k in PROBES}
    expected = SCENES.get(scene, (None, None))[1]
    out = outcome(preds["K#0"], preds["K+#0"], gt, keep)
    colours = {"gt": label_colours(gt[:, 0].astype(np.int64), pal, keep), "diff": outcome_colours(out)}
    colours.update({k: label_colours(preds[k].astype(np.int64), pal, keep) for k in PROBES})
    mesh = MeshRays(path, n, args.tmp / "mesh")
    w2cs = [np.linalg.inv(dslr_c2w(v["T"])) for v in s["views"]]

    scores = []  # pixels K+ fixes minus pixels it breaks, per view, at an eighth of the image scale
    for w2c in w2cs:
        ids = mesh.vertex_ids(w2c, s["K"], s["hw"], args.scale / 8)
        o = np.where(ids >= 0, out[np.maximum(ids, 0)], -1)
        scores.append(int((o == 0).sum() - (o == 1).sum()))
    order = np.argsort(scores, kind="stable")
    picks = [(int(order[-1]), "the view where K+ fixes the most pixels net of those it breaks")]
    median = int(order[len(order) // 2])
    picks.append((median if median != picks[0][0] else int(order[0]), "a view at the median of that score"))

    f = font(max(14, round(32 * args.scale)))
    views, files = [], []
    for i, (vi, why) in enumerate(picks, 1):
        v = s["views"][vi]
        ids = mesh.vertex_ids(w2cs[vi], s["K"], s["hw"], args.scale)
        h, w = ids.shape
        rr = np.minimum(((np.arange(h) + 0.5) / args.scale).astype(int), s["hw"][0] - 1)
        cc = np.minimum(((np.arange(w) + 0.5) / args.scale).astype(int), s["hw"][1] - 1)
        ids = np.where(v["valid"][np.ix_(rr, cc)], ids, -1)  # anonymised pixels of the photo
        photo = np.asarray(Image.fromarray(decode_image(v["jpg"])).convert("RGB").resize((w, h), Image.BILINEAR))
        panels = [captioned(photo, f"Photo {v['name']}: {why}", f), captioned(paint(ids, colours["gt"]), "Ground truth", f),
                  captioned(paint(ids, colours["diff"]), "Green: only K+ right. Red: only K right", f)]
        panels += [captioned(paint(ids, colours[k]), f"{PROBES[k]}: {m[k]:.1f} mIoU in this scene", f) for k in PROBES]
        img = grid(panels, 3)
        vis = ids[ids >= 0]
        counts = np.zeros(len(names), np.int64)
        for lab in (gt[vis, 0], *(preds[k][vis] for k in PROBES)):
            lab = lab[(lab >= 0) & keep[vis]].astype(np.int64)
            counts += np.bincount(lab, minlength=len(names))
        o = out[vis]
        legends = [legend("Classes in view (ground truth and predictions; black: not evaluated)", class_entries(counts, names, pal), img.width, f),
                   legend("Where K and K+ differ", list(zip(OUTCOMES, map(tuple, OUTCOME_RGB))), img.width, f)]
        file = args.out / f"{scene}_view{i}.png"
        stacked(img, *legends).save(file, optimize=True)
        files.append(file.name)
        views.append(dict(view=v["name"], why=why, score=scores[vi], pixels={name: int((o == j).sum()) for j, name in enumerate(OUTCOMES)}))

    tops = [captioned(topdown(s["coord"], colours[key]), text, f) for key, text in
            (("gt", "Ground truth"), ("K#0", f"{PROBES['K#0']}: {m['K#0']:.1f} mIoU"), ("K+#0", f"{PROBES['K+#0']}: {m['K+#0']:.1f} mIoU"),
             ("diff", "Green: only K+ right. Red: only K right"))]
    img = grid(tops, 2)
    counts = np.bincount(gt[keep, 0][gt[keep, 0] >= 0].astype(np.int64), minlength=len(names))
    file = args.out / f"{scene}_topdown.png"
    stacked(img, legend("Classes in the scene (ground truth, by size)", class_entries(counts, names, pal), img.width, f),
            legend("Where K and K+ differ", list(zip(OUTCOMES, map(tuple, OUTCOME_RGB))), img.width, f)).save(file, optimize=True)
    files.append(file.name)
    matches = None if expected is None else all(abs(m[k] - expected[k]) <= MATCH_TOL for k in PROBES)
    return dict(scene=scene, why=SCENES.get(scene, ("chosen with --scenes", None))[0], n_vertices=n,
                seen=round(float((seen[keep] > 0).mean()), 4), miou=m, step5=expected, matches_step5=matches, views=views,
                files=files, seconds=round(time.perf_counter() - t0, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--probes", type=Path, default=Path("/kaggle/working/pilot_concat/step4/probes.pt"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/pilot_concat/visuals"))
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/pilot_visuals"))
    ap.add_argument("--scenes", nargs="+", default=list(SCENES))
    ap.add_argument("--scale", type=float, default=0.5, help="image size relative to the DSLR images")
    ap.add_argument("--device", default="cuda", help="cpu only for the local tests")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("!! no GPU visible: set Accelerator to GPU T4 x2")
        return 2
    record = collect()
    (args.out / "env_record.json").write_text(json.dumps(record, indent=2))
    print(f"git {json.dumps(record['git'])}\nmachine {json.dumps(machine())}\nargs {vars(args)}")

    saved = torch.load(args.probes, weights_only=False)
    meta = saved["features"]
    folded = FoldedSet({k: saved["probes"][k] for k in PROBES}, device=dev)
    occ, n_views = OCCLUSION[meta["occlusion"]], meta["views"]
    archives = {s: p[0] for s, p in find_scenes(args.input)["core"].items()}
    missing = [s for s in args.scenes if s not in archives]
    if missing:
        print(f"!! scenes not in the attached datasets: {missing}")
        return 1
    meta_dir = archives[args.scenes[0]].parent.parent / "meta" / "metadata"
    gt_source = OfficialGT({s: archives[s] for s in args.scenes}, meta_dir, args.tmp / "gt")
    names = gt_source.class_names
    pal = palette(names)
    models = load_models(meta["encoder_names"]["concerto"], dev)
    print(f"probes {list(PROBES)}; {n_views} views, occlusion {meta['occlusion']}, concerto {meta['encoder_names']['concerto']}")

    section("1. Scenes")
    rows = []
    with attention_backend("sdpa"):
        for scene in args.scenes:
            rows.append(run_scene(scene, archives[scene], models, folded, gt_source, names, pal, occ, n_views, args, dev))
            r = rows[-1]
            print(f"  {scene} ({r['why']}): {r['n_vertices']:,} vertices, seen {r['seen']:.3f}; scene mIoU {r['miou']} "
                  f"(step 5: {r['step5']}); views {[(v['view'], v['score']) for v in r['views']]}; {r['seconds']} s", flush=True)
    (args.out / "visuals.json").write_text(json.dumps(dict(git=record["git"], args={k: str(v) for k, v in vars(args).items()},
                                                           scenes=rows), indent=1))

    section("SUMMARY visuals (paste this block back)")
    for r in rows:
        same = {True: "same as step 5", False: "DIFFERENT from step 5", None: "no step-5 value"}[r["matches_step5"]]
        print(f"{r['scene']} ({r['why']}): mIoU A {r['miou']['A#0']:.2f}, K {r['miou']['K#0']:.2f}, K+ {r['miou']['K+#0']:.2f} "
              f"({same}); seen {r['seen']:.3f}; images {r['files']}")
    print(f"images: {args.out} (shown below in the notebook; ScanNet++-derived, so they stay on Kaggle)")
    return 0 if all(r["matches_step5"] is not False for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
