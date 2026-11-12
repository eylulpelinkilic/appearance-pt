"""Experiment 3, Run B: does the student's match to its targets hold on unseen (val) scenes? A diagnostic: the only
place val images are used. Its outputs (<out>/runB/) never reach a probe: this script imports no probe code (checked
at the end), and Runs A and C refuse probe features stored under runB/.

1. B2 baseline head: a head of h's architecture trained on FROZEN Sonata features with the student's loss and data
   (Experiment 2's 409,600 training targets: the frozen teacher over all of a point's cameras; the frozen-Sonata
   upcast feature at each target point, saved by Experiment 2's step 1 from a whole-scene pass), AdamW with the
   student head's learning rate, until the epoch loss improves by less than plateau_rel for plateau_epochs epochs.
2. Per scene (b.train_scenes train scenes, a fixed seed-0 draw from the 200; all val scenes): points_per_scene random
   non-anonymised vertices seen by >= 1 camera (seed = scene seed + 2), their per-view DINOv2 observations from
   Experiment 2's step-1 code path (same checkpoint, 30 DSLR views, occlusion v2) with step 1's checks; val
   observations are stored in runB/val_observations/, apart from every training artifact.
   - t_attn(p): the frozen teacher over all of the point's cameras (exactly as Experiment 2's step 3); t_mean(p): the
     uniform average (lift()'s feature).
   - Whole scenes (one pass per scene, the scene's seed, as in evaluation): B1 = cos(h(f_student(p)), t_attn(p)),
     B2 = cos(h_B2(f_sonata(p)), t_attn(p)), B3 = cos(t_attn(p), t_mean(p)).
   - Training-style crops on the train scenes (crops of crop_points vertices around random sample points): B1 again,
     to connect with the training loss.
3. Generalization gap = mean train cos - mean val cos with a bootstrap over scenes (train and val scenes resampled
   independently), for the student and B2, and the difference of the two gaps. B1 by camera count (1, 2, >= 3) and by
   the most frequent val classes, next to the per-class IoU change student - frozen Sonata (Experiment 2 and pilot
   confusion matrices).
Outputs: <out>/runB/summary.json, points/*.npz (per-point cosines), val_observations/, head_b2.pt.

Usage: python experiments/Experiment3_DistillationDiagnostics/stepB_val_cosine.py --mode smoke|full
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.scannetpp_store import find_scenes
from encoders import attention_backend, encode_scene, scene_seed
from lifting import OCCLUSION, lift, load_dino, observe, observed_mean, project
from official import OfficialGT
from pipeline import add_dino, prefetch, scene_views
from scenes import KEYS
from step3_student import make_targets
from xattn import (CrossAttentionTeacher, ObservationStore, PredictionHead, cosine_loss, crop_indices, gather_all, mark_mode,
                   peak_gpu_gb, save_scene_observations, train_scenes, val_scenes)
from xdiag import (CONFIG, OUT, class_iou, find_exp2, find_pilot, head_features, load_config, load_frozen_sonata,
                   load_student_head, stack_config, write_json)

PROBE_MODULES = ("probes", "step4_probes", "xprobe")


def load_teacher(exp2: Path, device: str):
    saved = torch.load(exp2 / "teacher" / "teacher.pt", weights_only=False, map_location="cpu")
    teacher = CrossAttentionTeacher(saved["query_dim"], saved["key_dim"], saved["dim"])
    teacher.load_state_dict(saved["state_dict"])
    return teacher.to(device).eval().requires_grad_(False)


def train_b2_head(query: np.ndarray, target: np.ndarray, hc: dict, device: str) -> tuple[torch.nn.Module, dict]:
    """A head of h's architecture on frozen-Sonata features, the student's loss, until the loss plateaus."""
    X, Y = torch.from_numpy(query).to(device), torch.from_numpy(target).to(device)
    torch.manual_seed(hc["seed"])
    head = PredictionHead(X.shape[1], Y.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=hc["lr"], weight_decay=hc["weight_decay"])
    g = torch.Generator().manual_seed(hc["seed"])
    curve, steps, flat_epochs = [], 0, 0
    for epoch in range(hc["max_epochs"]):
        perm = torch.randperm(len(X), generator=g).to(device)
        total, n = 0.0, 0
        for s in range(0, len(X), hc["batch"]):
            b = perm[s:s + hc["batch"]]
            for group in opt.param_groups:
                group["lr"] = hc["lr"] * min(1.0, (steps + 1) / hc["warmup"])
            loss = cosine_loss(head(X[b].float()), Y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
            total, n = total + float(loss.detach()) * len(b), n + len(b)
        curve.append(round(total / n, 5))
        if len(curve) > 1 and (curve[-2] - curve[-1]) / max(curve[-2], 1e-9) < hc["plateau_rel"]:
            flat_epochs += 1
        else:
            flat_epochs = 0
        if flat_epochs >= hc["plateau_epochs"]:
            break
    return head.eval().requires_grad_(False), dict(epochs=len(curve), steps=steps, curve=curve, final_loss=curve[-1],
                                                   plateaued=flat_epochs >= hc["plateau_epochs"])


@torch.no_grad()
def sample_and_observe(s: dict, views, points, normals, n: int, seed: int, factor: float):
    """n random non-anonymised vertices seen by >= 1 camera, their observations on exactly that point set, and step 1's
    checks (mean = lift(), pixels re-project, ids unique, cameras distinct)."""
    keep = np.ones(len(s["coord"]), bool)
    keep[s["anonymised"]] = False
    rng = np.random.default_rng(seed)
    cand = np.sort(rng.choice(np.flatnonzero(keep), min(int(n * factor), int(keep.sum())), replace=False))
    ct = torch.from_numpy(cand).to(points.device)
    _, count = observed_mean(observe(points[ct], views, normals[ct]), len(cand))
    eligible = np.flatnonzero(count.cpu().numpy() >= 1)
    vertex = cand[np.sort(rng.choice(eligible, min(n, len(eligible)), replace=False))]
    for _ in range(3):
        vt = torch.from_numpy(vertex).to(points.device)
        obs = observe(points[vt], views, normals[vt])
        mean, count = observed_mean(obs, len(vertex))
        ok = (count >= 1).cpu().numpy()
        if ok.all():
            break
        vertex = vertex[ok]
    (_, lifted, lifted_count), = lift(points[vt], views, normals[vt], chunk=len(vertex) + 1)
    reproj = 0.0
    for vi in obs["view"].unique().tolist():
        m = obs["view"] == vi
        u, v, _ = project(points[vt[obs["row"][m]]], views[vi].w2c, views[vi].K)
        reproj = max(reproj, float((torch.stack([u, v], 1) - obs["uv"][m]).abs().max()))
    pairs = obs["row"].long() * len(views) + obs["view"].long()
    checks = dict(mean_vs_lift_max_abs=float((mean - lifted).abs().max()), counts_equal=bool((count == lifted_count).all()),
                  reprojection_max_px=reproj, vertex_ids_unique=bool(len(np.unique(vertex)) == len(vertex)),
                  cameras_distinct=bool(len(pairs.unique()) == len(pairs)))
    return vertex, obs, checks


@torch.no_grad()
def scene_cosines(scene, s, split, models, cfgb, occ, dev, obs_root, gt=None) -> dict:
    student, transform, head, sonata, s_transform, head_b2, teacher, dino = models
    seed = scene_seed(scene)
    points = torch.from_numpy(s["coord"]).to(dev)
    normals = torch.from_numpy(s["normal"]).float().to(dev)
    views = scene_views(s, points, occ, device=dev)
    add_dino(dino, s, views, device=dev)
    vertex, obs, checks = sample_and_observe(s, views, points, normals, cfgb["points_per_scene"], seed + 2, cfgb["candidate_factor"])
    vidx = torch.from_numpy(vertex)
    f_sonata = encode_scene(sonata, s_transform, s, seed, store="cpu")[0].gather(vidx, device=dev)
    f_student = encode_scene(student, transform, s, seed, store="cpu")[0].gather(vidx, device=dev)
    save_scene_observations(obs_root, scene, vertex, f_sonata.cpu().numpy(), obs["row"].cpu().numpy(), obs["view"].cpu().numpy(),
                            obs["uv"].cpu().numpy(), obs["feat"].cpu().numpy(), dict(scene=scene, split=split, checks=checks))
    store = ObservationStore(obs_root, [scene])  # the stored (fp16) observations, as the teacher saw them in step 3
    q, kv, mask = gather_all(store, np.arange(len(store)), dev)
    t_attn = teacher(q, kv, mask)[0]
    t_mean = (kv * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    b1 = F.cosine_similarity(head_features(head, f_student), t_attn, dim=-1)
    b2 = F.cosine_similarity(head_features(head_b2, f_sonata), t_attn, dim=-1)
    b3 = F.cosine_similarity(t_attn, t_mean, dim=-1)
    b1_crop = torch.full_like(b1, float("nan"))
    if split == "train":  # training-style crops: the student sees a crop, as during its training
        rng = np.random.default_rng(seed + 3)
        for c in range(cfgb["crops_per_scene"]):
            idx = crop_indices(s["coord"], s["coord"][vertex[rng.integers(len(vertex))]], cfgb["crop_points"])
            pos = np.searchsorted(idx, vertex).clip(max=len(idx) - 1)
            inside = (idx[pos] == vertex) & torch.isnan(b1_crop).cpu().numpy()
            if not inside.any():
                continue
            crop = {k: s[k][idx] for k in KEYS}
            fc = encode_scene(student, transform, crop, seed + 10 + c, store="cpu")[0].gather(torch.from_numpy(pos[inside]), device=dev)
            b1_crop[torch.from_numpy(np.flatnonzero(inside)).to(dev)] = F.cosine_similarity(head_features(head, fc),
                                                                                          t_attn[torch.from_numpy(inside).to(dev)], dim=-1)
    return dict(cams=store.count.astype(np.int16), cls=(gt[vertex, 0] if gt is not None else np.full(len(vertex), -100)).astype(np.int16),
                b1=b1.cpu().numpy(), b2=b2.cpu().numpy(), b3=b3.cpu().numpy(), b1_crop=b1_crop.cpu().numpy(), checks=checks)


def gap_bootstrap(per_scene: dict, key: str, train: list, val: list, n: int = 1000, seed: int = 0):
    """Mean train cos - mean val cos (point-weighted), its 95 % interval resampling train and val scenes independently;
    also returns the resampled gaps for paired comparisons."""
    def sums(scenes):
        return np.array([np.nansum(per_scene[s][key]) for s in scenes]), np.array([np.isfinite(per_scene[s][key]).sum() for s in scenes])
    ts, tn = sums(train)
    vs, vn = sums(val)
    rng = np.random.default_rng(seed)
    it, iv = rng.integers(0, len(train), (n, len(train))), rng.integers(0, len(val), (n, len(val)))
    gaps = ts[it].sum(1) / tn[it].sum(1) - vs[iv].sum(1) / vn[iv].sum(1)
    obs = ts.sum() / tn.sum() - vs.sum() / vn.sum()
    return dict(train=float(ts.sum() / tn.sum()), val=float(vs.sum() / vn.sum()), gap=float(obs),
                lo=float(np.percentile(gaps, 2.5)), hi=float(np.percentile(gaps, 97.5)), p_le_0=float((gaps <= 0).mean())), gaps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--exp2", type=Path)
    ap.add_argument("--pilot", type=Path)
    ap.add_argument("--any-exp2", action="store_true", help="tests only")
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/exp3_runB"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run = load_config(args.config, args.mode)["run"]
    cfgb = run["b"]
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    mark_mode(args.out, args.mode)
    out = args.out / "runB"
    (out / "points").mkdir(parents=True, exist_ok=True)
    record = collect()
    t0 = time.time()
    print(f"git {json.dumps(record['git'])}\nmode {args.mode}: {json.dumps(cfgb)}")

    section("1. Inputs and the B2 head (frozen Sonata features, the student's loss and targets)")
    exp2 = find_exp2(args.input, args.exp2, args.any_exp2)
    pilot = find_pilot(args.input, args.pilot)
    student, transform, head, facts = load_student_head(exp2 / "student", dev)
    teacher = load_teacher(exp2, dev)
    obs_summary = json.loads((exp2 / "observations" / "summary.json").read_text())
    store = ObservationStore(exp2 / "observations", obs_summary["scenes"])
    targets = make_targets(store, teacher, dev)
    target = np.concatenate([targets[s][1] for s in store.scenes])  # store order
    head_b2, b2_info = train_b2_head(store.query, target, cfgb["head"], dev)
    torch.save(dict(state_dict=head_b2.state_dict(), info=b2_info), out / "head_b2.pt")
    del store, targets, target
    print(f"  B2 head: {b2_info['epochs']} epochs, {b2_info['steps']} steps, final loss {b2_info['final_loss']} "
          f"(plateaued {b2_info['plateaued']}); student head loss at the end of training: see Experiment 2")

    section("2. Scenes: observations, targets, cosines")
    archives = {s: p[0] for s, p in find_scenes(args.input)["core"].items()}
    all_train = train_scenes(archives)
    order = np.random.default_rng(0).permutation(len(all_train))
    tscenes = [all_train[i] for i in sorted(order[:cfgb["train_scenes"]].tolist())]
    vscenes = val_scenes(archives, cfgb["val_scenes"])
    meta_dir = archives[vscenes[0]].parent.parent / "meta" / "metadata"
    gt_source = OfficialGT({s: archives[s] for s in vscenes}, meta_dir, args.tmp / "gt")
    sonata, s_transform = load_frozen_sonata(dev)
    dino = load_dino(dev)
    models = (student, transform, head, sonata, s_transform, head_b2, teacher, dino)
    occ = OCCLUSION[cfgb["occlusion"]]
    per_scene, checks_all = {}, []
    with attention_backend("sdpa"):
        items = [(sc, archives[sc]) for sc in tscenes + vscenes]
        for i, (scene, s, waited) in enumerate(prefetch(items, cfgb["views"], args.workers), 1):
            if isinstance(s, Exception):
                raise s
            split = "val" if scene in vscenes else "train"
            obs_root = out / "val_observations" if split == "val" else args.tmp / "train_observations"
            gt = gt_source.labels(scene) if split == "val" else None
            r = scene_cosines(scene, s, split, models, cfgb, occ, dev, obs_root, gt)
            checks_all.append(dict(scene=scene, split=split, **r.pop("checks")))
            np.savez(out / "points" / f"{split}_{scene}.npz", **r)
            per_scene[scene] = r
            print(f"  [{i}/{len(items)}] {split} {scene}: {len(r['b1'])} points, B1 {np.mean(r['b1']):.4f}, B2 {np.mean(r['b2']):.4f}, "
                  f"B3 {np.mean(r['b3']):.4f}; waited {waited} s", flush=True)
            if split == "train":
                shutil.rmtree(args.tmp / "train_observations", ignore_errors=True)

    section("3. Generalization gaps, breakdowns")
    stud, stud_gaps = gap_bootstrap(per_scene, "b1", tscenes, vscenes)
    base, base_gaps = gap_bootstrap(per_scene, "b2", tscenes, vscenes)
    dg = stud_gaps - base_gaps
    gap_diff = dict(diff=stud["gap"] - base["gap"], lo=float(np.percentile(dg, 2.5)), hi=float(np.percentile(dg, 97.5)),
                    p_le_0=float((dg <= 0).mean()))
    cat = lambda key, scenes: np.concatenate([per_scene[s][key] for s in scenes])  # noqa: E731
    vb1, vb2, vb3, vcams, vcls = (cat(k, vscenes) for k in ("b1", "b2", "b3", "cams", "cls"))
    cams = {}
    for name, m in (("1", vcams == 1), ("2", vcams == 2), (">=3", vcams >= 3)):
        cams[name] = dict(points=int(m.sum()), b1=float(vb1[m].mean()) if m.any() else None, b2=float(vb2[m].mean()) if m.any() else None,
                          b3=float(vb3[m].mean()) if m.any() else None)
    # classes: the most frequent val classes by vertex count (GT rows of the pilot's frozen-Sonata confusion matrices)
    z = np.load(pilot / "confmats.npz")
    psel = [[str(x) for x in z["scenes"]].index(s) for s in vscenes]
    s_cms = stack_config(z, "S", 1, psel)
    e = np.load(exp2 / "evaluation" / "confmats.npz")
    esel = [[str(x) for x in e["scenes"]].index(s) for s in vscenes]
    t_cms = stack_config(e, "T", 1, esel)
    counts = s_cms[0].sum(0).sum(1)
    top = [int(c) for c in np.argsort(-counts)[:cfgb["classes"]] if counts[c] > 0]
    d_iou = class_iou(t_cms) - class_iou(s_cms)
    names = gt_source.class_names
    classes = [dict(cls=names[c], vertex_share=float(counts[c] / counts.sum()), points=int((vcls == c).sum()),
                    b1=float(vb1[vcls == c].mean()) if (vcls == c).any() else None,
                    b2=float(vb2[vcls == c].mean()) if (vcls == c).any() else None,
                    iou_change_student_minus_sonata=float(d_iou[c])) for c in top]
    tb1, tcrop = cat("b1", tscenes), cat("b1_crop", tscenes)
    covered = np.isfinite(tcrop)
    crops = dict(points=int(covered.sum()), b1_whole=float(tb1[covered].mean()) if covered.any() else None,
                 b1_crop=float(tcrop[covered].mean()) if covered.any() else None,
                 b1_crop_ge3=float(tcrop[covered & (cat("cams", tscenes) >= 3)].mean()) if (covered & (cat("cams", tscenes) >= 3)).any() else None)
    obs_checks = dict(mean_vs_lift_max_abs=max(c["mean_vs_lift_max_abs"] for c in checks_all),
                      reprojection_max_px=max(c["reprojection_max_px"] for c in checks_all),
                      counts_equal=all(c["counts_equal"] for c in checks_all), vertex_ids_unique=all(c["vertex_ids_unique"] for c in checks_all),
                      cameras_distinct=all(c["cameras_distinct"] for c in checks_all))
    no_probe = [m for m in PROBE_MODULES if m in sys.modules]
    summary = dict(
        mode=args.mode, git=record["git"], exp2=str(exp2), train_scenes=tscenes, val_scenes=vscenes, sampling=dict(
            points_per_scene=cfgb["points_per_scene"], seed="scene seed + 2", cameras=">= 1", features="whole scenes, one pass per scene"),
        b2_head=b2_info, student_head=facts, b1=stud, b2=base, gap_difference=gap_diff,
        b3_val=dict(mean=float(vb3.mean()), p05=float(np.percentile(vb3, 5)), p50=float(np.median(vb3))),
        b3_train=float(cat("b3", tscenes).mean()), by_cameras=cams, classes=classes, crops=crops,
        decision=dict(rule="B", fitted_train=bool(gap_diff["lo"] > 0),
                      text=("the student's gap is larger than B2's (interval of the difference above 0): continued training "
                            "fitted the train scenes" if gap_diff["lo"] > 0 else
                            "the student's gap is not larger than B2's: no sign that continued training fitted the train scenes")),
        checks=dict(observations=obs_checks, probe_modules_imported=no_probe, no_probe=not no_probe,
                    val_observations=str(out / "val_observations")),
        storage_gb=dict(val_observations=round(sum(p.stat().st_size for p in (out / "val_observations").rglob("*") if p.is_file()) / 1e9, 3)),
        minutes=round((time.time() - t0) / 60, 2), peak_gpu_gb=peak_gpu_gb())
    write_json(out / "summary.json", summary)
    ok = (not no_probe and obs_checks["counts_equal"] and obs_checks["cameras_distinct"] and obs_checks["vertex_ids_unique"]
          and obs_checks["mean_vs_lift_max_abs"] < 1e-3)

    section("SUMMARY Run B (paste this block back)")
    print(f"scenes: {len(tscenes)} train, {len(vscenes)} val; {cfgb['points_per_scene']} points each (>= 1 camera); whole scenes")
    print(f"B2 head: {b2_info['epochs']} epochs, {b2_info['steps']} steps, final train loss {b2_info['final_loss']}, plateaued {b2_info['plateaued']}")
    for name, g in (("B1 student", stud), ("B2 head on frozen Sonata", base)):
        print(f"  {name:26s} train {g['train']:.4f}  val {g['val']:.4f}  gap {g['gap']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]")
    print(f"  gap difference (student - B2) {gap_diff['diff']:+.4f} [{gap_diff['lo']:+.4f}, {gap_diff['hi']:+.4f}]")
    print(f"  B3 cos(t_attn, t_mean): val {summary['b3_val']['mean']:.4f} (5th percentile {summary['b3_val']['p05']:.4f}), train {summary['b3_train']:.4f}")
    print(f"  by cameras (val): {json.dumps(cams)}")
    print(f"  crops (train): {json.dumps(crops)}")
    print("  classes (val): " + "; ".join(f"{c['cls']} B1 {c['b1']:.3f} dIoU {c['iou_change_student_minus_sonata']:+.1f}"
                                          for c in classes if c["b1"] is not None))
    print(f"decision B: {summary['decision']['text']}")
    print(f"checks: observations {json.dumps(obs_checks)}; probe modules imported {no_probe}; "
          f"{summary['minutes']} min, peak GPU {summary['peak_gpu_gb']} GB, val observations {summary['storage_gb']['val_observations']} GB")
    print(f"result {'OK' if ok else 'CHECK FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
