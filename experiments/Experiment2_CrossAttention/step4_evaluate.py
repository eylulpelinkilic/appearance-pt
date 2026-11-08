"""Experiment 2, step 4: the point-only student on the pilot's terms, next to the pilot's two Sonata rows.

The teacher and the prediction head are discarded; the trained student is frozen and only its per-point feature is
used: the pilot's upcast layer (all five stages concatenated), from one fp16 pass per scene with the scene's seed.
Everything is read through xattn.PointOnlyArchive, which serves the point cloud (xyz, colour, normal, labels) and the
anonymisation list and refuses images and cameras; no image, camera, DINO cache or teacher path is an argument here.

  1. Training features: the pilot's labelled sample (10,000 points per train scene, the scene's seed, the Pointcept
     major label) with the student's features, in the pilot's feature store (probes.FeatureWriter).
  2. Linear probes with the pilot's recipe (Experiment1 step4_probes.run_phase): learning rate from the pilot's grid by
     point mIoU on the same 20 held-out train scenes (seed-0 permutation, 10 epochs), then seeds 0-2 on all train
     scenes (15 epochs, batch 4096). The probes start from the same initial weights as the pilot's Sonata probe.
  3. Every vertex of every val scene (the pilot's 50; anonymised vertices excluded, as the official evaluation does),
     the toolbox's own ground truth (official.OfficialGT), top-1 and top-3 confusion matrices per scene and seed
     (metrics.confusion_topk, equal to the toolbox's). Points with fewer than three cameras are included: the camera
     requirement only selected training targets.
  3b. Protocol check: the original frozen Sonata (sonata.pth) goes through the same steps 1-3 (its own features at the
     same points, the same probe recipe as its own configuration, so its probes start from the pilot's Sonata probe
     weights). In the full run it should reproduce the pilot's frozen-Sonata row within seed noise; in the smoke run
     (small probe recipe) it is the like-for-like reference for the student.
  4. Image-free self-check: one val scene copied without dslr/ and iphone/ gives the same student features; the
     archive reads are logged; no image module (lifting, pipeline, transformers) is imported by this step.
  5. The three rows: frozen Sonata (S) and frozen Sonata + lifted DINOv2 (S+) recomputed unrounded from the pilot's
     saved per-scene confusion matrices (pilot_04_eval: confmats.npz, results.json; nothing of the pilot is rerun),
     and the student (T, this run); differences with paired bootstrap intervals over val scenes (metrics.bootstrap_diff,
     1000 resamples), and the share of the pilot's concatenation gain the student recovers.
Outputs: <out>/evaluation/ (results.json, confmats.npz, probes.pt) and <out>/reports/metrics.json for the report.

Usage: python experiments/Experiment2_CrossAttention/step4_evaluate.py --mode smoke|full [--pilot DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import probes as probes_mod
from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.scannetpp_store import find_scenes
from encoders import attention_backend, encode_scene, load_encoder, make_transform, scene_seed
from metrics import bootstrap_diff, confusion_topk, miou
from official import OfficialGT
from probes import NUM_CLASSES, FeatureStore, FeatureWriter, FoldedSet
from step4_probes import run_phase
from xattn import (CONFIG, OUT, PointOnlyArchive, dir_gb, load_config, mark_mode, load_points_scene, peak_gpu_gb, prefetch_points,
                   sample_points, strip_to_points, train_scenes, val_scenes, walk_dirs)

IMAGE_MODULES = ("lifting", "pipeline", "transformers")
INPUTS = {"S": "point cloud", "S+": "point cloud + 30 posed DSLR photos (lifted DINOv2)", "T": "point cloud", "Sr": "point cloud"}
NAMES = {"S": "Frozen Sonata", "S+": "Frozen Sonata + lifted DINOv2 (concatenation)", "T": "Point-only Sonata student",
         "Sr": "Frozen Sonata, rerun in this evaluation"}
BLOCKS = {"T": "student", "Sr": "sonata"}  # configuration: feature block (each probed on its own, with the pilot's recipe)


def load_student(path: Path, device: str):
    """The step-3 student: Sonata's architecture with the trained weights; frozen."""
    model, _ = load_encoder("sonata", seed=0)
    saved = torch.load(path, weights_only=False)  # this experiment's own checkpoint
    model.load_state_dict(saved["state_dict"])
    model.requires_grad_(False)
    return model.to(device).eval(), make_transform("sonata"), {k: v for k, v in saved.items() if k != "state_dict"}


def load_frozen_sonata(device: str):
    """The original Sonata (sonata.pth at the pinned commit), frozen: the protocol check's encoder."""
    model, _ = load_encoder("sonata", seed=0)
    model.requires_grad_(False)
    return model.to(device).eval(), make_transform("sonata")


def find_pilot(root: Path) -> Path | None:
    """The folder of the pilot's step-5 outputs (confmats.npz with the Sonata probes, and results.json)."""
    for d in walk_dirs(root):
        if (d / "confmats.npz").exists() and (d / "results.json").exists():
            with np.load(d / "confmats.npz") as z:
                if "S#0|top1" in z.files and "S+#0|top1" in z.files:
                    return d
    return None


def train_features(encoders: dict, archives, scenes, n_points, root, dev, workers) -> dict:
    """encoders: {block: (model, transform)}; each block's features at the pilot's labelled sample of every scene."""
    writer, rows, t0 = None, [], time.time()
    for i, (scene, s, waited) in enumerate(prefetch_points([(sc, archives[sc]) for sc in scenes], workers), 1):
        if isinstance(s, Exception):
            raise s
        seed = scene_seed(scene)
        idx = sample_points(s["segment"], s["anonymised"], n_points, seed)
        blocks, tiles = {}, {}
        for block, (model, transform) in encoders.items():
            stages, tiles[block] = encode_scene(model, transform, s, seed, store="cpu")
            blocks[block] = stages.gather(torch.from_numpy(idx), device=dev).cpu().numpy()
        if writer is None:
            writer = FeatureWriter(root, n_points * len(scenes), {b: f.shape[1] for b, f in blocks.items()},
                                   {b: np.float16 for b in blocks})
        writer.add(scene, blocks, s["segment"][idx, 0].astype(np.int16), np.zeros(len(idx), np.int16))
        rows.append(dict(scene=scene, points=len(idx), tiles=tiles, waited=waited))
        if i % 20 == 0 or i == len(scenes):
            print(f"  [{i}/{len(scenes)}] train features; {(time.time() - t0) / 60:.1f} min", flush=True)
    meta = writer.close(encoders=list(encoders), points_per_scene=n_points)
    return dict(rows=meta["n"], scenes=len(rows), minutes=round((time.time() - t0) / 60, 2), store_gb=dir_gb(root))


def fit_probes(root: Path, pcfg: dict, device: str) -> tuple[dict, dict]:
    """The pilot's probe recipe, each configuration (BLOCKS) in its own phase, as the pilot trained its Sonata probe:
    learning rate from the grid on the held-out scenes, then the seeds on all scenes. A configuration alone in its
    phase gets the pilot's Sonata probe's initial weights and batch order. Configurations are registered in
    probes.CONFIGS only while their probes train."""
    store = FeatureStore(root)
    n = len(store.scenes)
    order = np.random.default_rng(0).permutation(n)
    held, train = sorted(order[:pcfg["holdout"]].tolist()), sorted(order[pcfg["holdout"]:].tolist())
    args = SimpleNamespace(batch=pcfg["batch"])
    folded, info = {}, {}
    for config, block in BLOCKS.items():
        if block not in store.widths:
            continue
        probes_mod.CONFIGS[config] = ((block,), False)
        try:
            _, _, sweep_curve, held_miou = run_phase(store, train, (block,), [config], pcfg["lrs"], lambda k: float(k.split("@")[1]),
                                                     pcfg["sweep_epochs"], args, device, eval_ids=held)
            best = max(pcfg["lrs"], key=lambda lr: held_miou[f"{config}@{lr:g}"])
            probes, stats, curve, _ = run_phase(store, list(range(n)), (block,), [config], list(range(pcfg["seeds"])),
                                                lambda k: best, pcfg["epochs"], args, device)
        finally:
            probes_mod.CONFIGS.pop(config, None)
        folded.update({k: p.fold(stats) for k, p in probes.items()})
        info[config] = dict(lr=best, held_out_miou=held_miou, held_out_scenes=[store.scenes[i]["id"] for i in held],
                            sweep_loss=sweep_curve, final_loss=curve)
    return folded, info


def evaluate_val(encoders: dict, folded, archives, scenes, gt_source, dev, workers, chunk=200_000) -> tuple[dict, list]:
    fs = FoldedSet(folded, device=dev)
    per_scene = {k: {1: [], 3: []} for k in fs.keys}
    rows, t0 = [], time.time()
    for i, (scene, s, waited) in enumerate(prefetch_points([(sc, archives[sc]) for sc in scenes], workers), 1):
        if isinstance(s, Exception):
            raise s
        n = len(s["coord"])
        gt = torch.from_numpy(gt_source.labels(scene)).to(dev)
        assert len(gt) == n, f"{scene}: GT has {len(gt)} vertices, the scene {n}"
        keep = np.ones(n, bool)
        keep[s["anonymised"]] = False
        keep_t = torch.from_numpy(keep).to(dev)
        stages, tiles = {}, {}
        for block, (model, transform) in encoders.items():
            stages[block], tiles[block] = encode_scene(model, transform, s, scene_seed(scene), store="cpu")
        cms = {k: {kk: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=dev) for kk in (1, 3)} for k in fs.keys}
        for start in range(0, n, chunk):
            idx = torch.arange(start, min(start + chunk, n))
            logits = fs.logits({block: st.gather(idx, device=dev) for block, st in stages.items()})
            m, g = keep_t[start:start + len(idx)], gt[start:start + len(idx)]
            for k, lg in logits.items():
                top = lg.topk(3, dim=1).indices
                for kk in (1, 3):
                    cms[k][kk] += confusion_topk(top[m, :kk], g[m], NUM_CLASSES)
        for k in fs.keys:
            for kk in (1, 3):
                per_scene[k][kk].append(cms[k][kk].cpu().numpy().astype(np.int32))
        rows.append(dict(scene=scene, n_vertices=n, kept=int(keep.sum()), tiles=tiles, waited=waited))
        quick = {k: round(100 * float(miou(per_scene[k][1][-1])), 1) for k in fs.keys if k.endswith("#0")}
        print(f"  [{i}/{len(scenes)}] {scene}: {n:,} vertices, scene mIoU (seed 0, top-1) {quick}; "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return {k: {kk: np.stack(v) for kk, v in d.items()} for k, d in per_scene.items()}, rows


def image_free_check(model, transform, archives, scene, dev, tmp: Path, before: set) -> dict:
    stripped = strip_to_points(archives[scene], scene, tmp / "stripped")
    files = sorted(p.relative_to(stripped).as_posix() for p in stripped.rglob("*") if p.is_file())
    a, b = load_points_scene(archives[scene], scene), load_points_scene(stripped, scene)
    idx = torch.from_numpy(np.sort(np.random.default_rng(0).choice(len(a["coord"]), min(2000, len(a["coord"])), replace=False)))
    seed = scene_seed(scene)
    fa = encode_scene(model, transform, a, seed, store="cpu")[0].gather(idx, device=dev).float()
    fb = encode_scene(model, transform, b, seed, store="cpu")[0].gather(idx, device=dev).float()
    shutil.rmtree(tmp / "stripped", ignore_errors=True)
    loaded = [m for m in IMAGE_MODULES if m in sys.modules and m not in before]
    out = dict(scene=scene, stripped_copy_files=len(files), stripped_copy_has_images=any(f.startswith(("dslr/", "iphone/")) for f in files),
               max_abs_feature_diff=float((fa - fb).abs().max()), image_modules_imported_by_this_step=loaded,
               image_modules_present_before=[m for m in IMAGE_MODULES if m in before],
               archive_reads=sorted(PointOnlyArchive.READS),
               arguments="student checkpoint, core datasets (point clouds, labels) and the pilot's saved metrics only")
    out["ok"] = bool(not out["stripped_copy_has_images"] and out["max_abs_feature_diff"] <= 1e-3 and not loaded
                     and all(r.startswith(("pointcept", "scans/mesh_aligned_0.05_mask.txt")) for r in out["archive_reads"]))
    return out


def stack_confmats(source, prefix: str, kk: int, sel=None) -> np.ndarray:
    """(seeds, scenes, C, C) from a dict or an npz with keys "<prefix>#<seed>|top<kk>"."""
    keys = sorted((k for k in (source.files if hasattr(source, "files") else source)
                   if k.startswith(f"{prefix}#") and k.endswith(f"|top{kk}")), key=lambda k: int(k.split("#")[1].split("|")[0]))
    arrs = [source[k] for k in keys]
    return np.stack([a if sel is None else a[sel] for a in arrs])


def row_stats(cms1: np.ndarray, cms3: np.ndarray) -> dict:
    """mIoU (%) per seed from confusion matrices summed over scenes, unrounded."""
    s1 = [100 * float(miou(c.sum(0))) for c in cms1]
    s3 = [100 * float(miou(c.sum(0))) for c in cms3]
    return dict(top1=float(np.mean(s1)), top1_std=float(np.std(s1)), top3=float(np.mean(s3)), top3_std=float(np.std(s3)),
                top1_seeds=s1, top3_seeds=s3, seeds=len(s1))


def compare(pilot_dir: Path, new: dict, scenes: list[str], train_scenes: list[str]) -> dict:
    """new: {"<config>#<seed>": {1: (scenes, C, C), 3: ...}} of this run (T, and Sr if the control ran)."""
    z = np.load(pilot_dir / "confmats.npz")
    pres = json.loads((pilot_dir / "results.json").read_text())
    pscenes = [str(s) for s in z["scenes"]]
    sel = [pscenes.index(s) for s in scenes]
    flat = {f"{k}|top{kk}": v for k, d in new.items() for kk, v in d.items()}
    cms = {"S": {kk: stack_confmats(z, "S", kk, sel) for kk in (1, 3)}, "S+": {kk: stack_confmats(z, "S+", kk, sel) for kk in (1, 3)}}
    for c in ("T", "Sr"):
        if any(k.startswith(f"{c}#") for k in new):
            cms[c] = {kk: stack_confmats(flat, c, kk) for kk in (1, 3)}
    rows = {c: dict(name=NAMES[c], inference_inputs=INPUTS[c], **row_stats(cms[c][1], cms[c][3])) for c in ("S", "S+", "T")}
    pgit = pres.get("git", {}).get("commit")
    fgit = pres.get("features", {}).get("git", {}).get("commit")
    for c in ("S", "S+"):
        rows[c]["provenance"] = (f"pilot, reused: probes from pilot_03_probes (commit {fgit}), evaluation pilot_04_eval "
                                 f"(commit {pgit}); recomputed unrounded from its per-scene confusion matrices")
        recorded = pres.get("miou", {}).get(c, {}).get("top1_all", [None])[0]
        rows[c]["pilot_recorded_top1"] = recorded
    rows["T"]["provenance"] = "new run (this experiment)"
    diffs = {}
    for a, b in (("T", "S"), ("T", "S+"), ("S+", "S")):
        bd = bootstrap_diff(cms[a][1], cms[b][1])
        diffs[f"{a} - {b}"] = dict(diff=rows[a]["top1"] - rows[b]["top1"], lo=100 * bd["lo"], hi=100 * bd["hi"], p_le_0=bd["p_le_0"])
    gain = rows["S+"]["top1"] - rows["S"]["top1"]
    recovered = (rows["T"]["top1"] - rows["S"]["top1"]) / gain if gain else None
    control = None
    if "Sr" in cms:  # the protocol check: frozen Sonata through this evaluation, against the pilot's row and the student
        row = dict(name=NAMES["Sr"], inference_inputs=INPUTS["Sr"], provenance="new run (this experiment): protocol check",
                   **row_stats(cms["Sr"][1], cms["Sr"][3]))
        cdiffs = {}
        for a, b, ra, rb in (("Sr", "S", row, rows["S"]), ("T", "Sr", rows["T"], row)):
            bd = bootstrap_diff(cms[a][1], cms[b][1])
            cdiffs[f"{a} - {b}"] = dict(diff=ra["top1"] - rb["top1"], lo=100 * bd["lo"], hi=100 * bd["hi"], p_le_0=bd["p_le_0"])
        control = dict(row=row, diffs=cdiffs)
    feats = pres.get("features", {})
    # the pilot's step 4 saved its train scenes as ids (probes.pt "features" -> results.json)
    pilot_train = [t if isinstance(t, str) else t["id"] for t in feats.get("scenes", [])]
    checks = dict(
        val_scenes_same_order=pscenes[:len(scenes)] == scenes if len(scenes) < len(pscenes) else pscenes == scenes,
        train_scenes_match=pilot_train[:len(train_scenes)] == train_scenes,
        all_pilot_train_scenes=len(train_scenes) == len(pilot_train),
        all_pilot_val_scenes=len(scenes) == len(pscenes),
        pilot_points_per_scene=feats.get("points_per_scene"), pilot_views=feats.get("views"),
        pilot_occlusion=feats.get("occlusion"), pilot_sonata=feats.get("encoder_names", {}).get("sonata"),
        pilot_seeds=rows["S"]["seeds"], student_seeds=rows["T"]["seeds"], classes=NUM_CLASSES,
        recorded_vs_recomputed_top1={c: (rows[c]["pilot_recorded_top1"], round(rows[c]["top1"], 2)) for c in ("S", "S+")})
    checks["recorded_matches_recomputed"] = all(r is None or abs(r - v) <= 0.005 for r, v in checks["recorded_vs_recomputed_top1"].values()) \
        if len(scenes) == len(pscenes) else None
    return dict(rows=rows, diffs=diffs, concatenation_gain=gain, recovered_fraction=recovered, protocol=checks, control=control,
                pilot_train_scenes=pilot_train, pilot_dir=str(pilot_dir))


def main() -> int:
    before = set(sys.modules)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--student", type=Path, help="default <out>/student/student.pth")
    ap.add_argument("--pilot", type=Path, help="folder with the pilot's confmats.npz and results.json (default: found under --input)")
    ap.add_argument("--features", type=Path, default=Path("/tmp/exp2_probe_features"))
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/exp2_eval"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_config(args.config, args.mode)
    run, pcfg = cfg["run"], cfg["run"]["probe"]
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    mark_mode(args.out, args.mode)
    out = args.out / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    (args.out / "reports").mkdir(parents=True, exist_ok=True)
    record = collect()
    t_start = time.time()
    print(f"git {json.dumps(record['git'])}\nmode {args.mode}: probe {json.dumps(pcfg)}")

    section("1. Inputs: the student, the core datasets, the pilot's saved metrics")
    pilot_dir = args.pilot or find_pilot(args.input)
    if pilot_dir is None:
        print("!! the pilot's step-5 outputs are not attached: add the pilot_04_eval notebook output (confmats.npz, results.json)")
        return 2
    model, transform, student_meta = load_student(args.student or args.out / "student" / "student.pth", dev)
    encoders = {"student": (model, transform), "sonata": load_frozen_sonata(dev)}  # the second is the protocol check
    archives = {s: p[0] for s, p in find_scenes(args.input)["core"].items()}
    tscenes, vscenes = train_scenes(archives, run["probe_train_scenes"]), val_scenes(archives, run["val_scenes"])
    print(f"  pilot outputs {pilot_dir}; student {student_meta.get('trainable')} after {student_meta.get('steps')} steps; "
          f"{len(tscenes)} train scenes, {len(vscenes)} val scenes")

    with attention_backend("sdpa"):
        section("2. Training features (point-only) at the pilot's labelled sample")
        if args.features.exists():
            shutil.rmtree(args.features)
        feat_info = train_features(encoders, archives, tscenes, pcfg["points_per_scene"], args.features, dev, args.workers)
        print(f"  {json.dumps(feat_info)}")

        section("3. Linear probes with the pilot's recipe")
        folded, probe_info = fit_probes(args.features, pcfg, dev)
        torch.save(dict(probes=folded, lr={c: v["lr"] for c, v in probe_info.items()}, recipe=pcfg, git=record["git"]), out / "probes.pt")
        for c, v in probe_info.items():
            print(f"  {c}: learning rate {v['lr']} (held-out mIoU {json.dumps(v['held_out_miou'])})")

        section("4. All val vertices, official ground truth")
        meta_dir = archives[vscenes[0]].parent.parent / "meta" / "metadata"
        gt_source = OfficialGT({s: archives[s] for s in vscenes}, meta_dir, args.tmp / "gt")
        student_cms, val_rows = evaluate_val(encoders, folded, archives, vscenes, gt_source, dev, args.workers)

        section("5. Image-free self-check")
        free = image_free_check(model, transform, archives, vscenes[0], dev, args.tmp, before)
        print(f"  {json.dumps(free)}")

    section("6. The three rows")
    comp = compare(Path(pilot_dir), student_cms, vscenes, tscenes)
    np.savez_compressed(out / "confmats.npz", scenes=np.array(vscenes),
                        **{f"{k}|top{kk}": v for k, d in student_cms.items() for kk, v in d.items()})
    minutes = round((time.time() - t_start) / 60, 2)
    results = dict(mode=args.mode, git=record["git"], student=student_meta, train_features=feat_info, probes=probe_info,
                   val_scenes=vscenes, per_scene=val_rows, image_free=free, comparison=comp, minutes=minutes, peak_gpu_gb=peak_gpu_gb())
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))

    # everything the report reads, in one machine-readable file
    load = lambda rel: json.loads((args.out / rel).read_text()) if (args.out / rel).exists() else None  # noqa: E731
    obs, teacher_diag, teacher_sum = load("observations/summary.json"), load("teacher/diagnostic.json"), load("teacher/summary.json")
    train_log, smoke, s_checks, t_checks = (load("student/train_log.json"), load("student/smoke.json"), load("student/checks.json"),
                                            load("teacher/checks.json"))
    choice = cfg["student_choice"]
    final = (args.mode == "full" and bool(choice.get("recorded_from_smoke")) and len(vscenes) == 50 and comp["protocol"]["all_pilot_val_scenes"]
             and comp["protocol"]["train_scenes_match"] and comp["protocol"]["all_pilot_train_scenes"]
             and train_log is not None and train_log["steps"] == choice.get("steps") and free["ok"])
    gpu_hours = sum(x for x in ((obs or {}).get("minutes", 0) / 60, (teacher_sum or {}).get("minutes", 0) / 60,
                                (train_log or {}).get("session_hours", 0), minutes / 60))
    metrics = dict(
        status="final" if final else "preliminary", mode=args.mode, title=cfg["title"], generated_utc=record.get("utc"),
        rows=comp["rows"], diffs=comp["diffs"], concatenation_gain=comp["concatenation_gain"],
        recovered_fraction=comp["recovered_fraction"], protocol=comp["protocol"], teacher_diagnostic=teacher_diag,
        teacher_checks=t_checks, teacher_summary=teacher_sum,
        observations={k: v for k, v in (obs or {}).items() if k != "per_scene"}, student=train_log and {k: v for k, v in train_log.items()},
        student_checks=s_checks, smoke=smoke, control=comp["control"],
        probes=dict(lr=probe_info["T"]["lr"], held_out_miou=probe_info["T"]["held_out_miou"], recipe=pcfg,
                    control_lr=(probe_info.get("Sr") or {}).get("lr")),
        image_free=free, val_scenes=len(vscenes), train_scenes=len(tscenes),
        reproducibility=dict(commit=record["git"].get("commit"), dirty=record["git"].get("dirty"), start_commit=cfg["start_commit"],
                             checkpoint=cfg["pilot"]["encoder"], dino=cfg["pilot"]["dino"], subset=cfg["pilot"]["subset"],
                             dev_scenes=(obs or {}).get("dev_scenes"), seeds=dict(probe=list(range(pcfg["seeds"])),
                             teacher=run["teacher"]["seed"], student=run["student"]["seed"]),
                             trainable=(train_log or {}).get("trainable"), steps=(train_log or {}).get("steps"),
                             gpu_hours=round(gpu_hours, 2), student_choice=choice),
        evaluation_minutes=minutes, peak_gpu_gb=peak_gpu_gb(),
        storage_gb=dict(observations=dir_gb(args.out / "observations"), student=dir_gb(args.out / "student"),
                        probe_features=feat_info["store_gb"]))
    (args.out / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))

    section("SUMMARY step 4 evaluation (paste this block back)")
    for c in ("S", "S+", "T"):
        r = comp["rows"][c]
        print(f"  {c:3s} {r['name']:47s} top-1 {r['top1']:.4f} ± {r['top1_std']:.4f}  top-3 {r['top3']:.4f}  "
              f"[{r['inference_inputs']}; {r['provenance'][:40]}]")
    for k, d in comp["diffs"].items():
        print(f"  {k:7s} {d['diff']:+.4f} [{d['lo']:+.2f}, {d['hi']:+.2f}]  P(diff <= 0) {d['p_le_0']}")
    rec = comp["recovered_fraction"]
    print(f"share of the concatenation gain recovered: {'n/a' if rec is None else f'{rec:.3f}'}")
    if comp["control"]:
        r = comp["control"]["row"]
        print(f"protocol check, {r['name']}: top-1 {r['top1']:.4f} ± {r['top1_std']:.4f}  top-3 {r['top3']:.4f}")
        for k, d in comp["control"]["diffs"].items():
            print(f"  {k:7s} {d['diff']:+.4f} [{d['lo']:+.2f}, {d['hi']:+.2f}]  P(diff <= 0) {d['p_le_0']}")
    print(f"protocol: {json.dumps(comp['protocol'])}")
    print(f"image-free: {json.dumps({k: free[k] for k in ('ok', 'max_abs_feature_diff', 'stripped_copy_has_images', 'image_modules_imported_by_this_step', 'archive_reads')})}")
    print(f"probe lr {json.dumps({c: v['lr'] for c, v in probe_info.items()})}; {len(vscenes)} val scenes; {minutes} min; peak GPU {metrics['peak_gpu_gb']} GB; status {metrics['status']}; "
          f"metrics {args.out / 'reports' / 'metrics.json'}")
    return 0 if free["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
