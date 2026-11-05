"""Experiment 2, step 1: per-view DINOv2 observations of training points, with their frozen Sonata features.

The pilot kept only the per-point average over cameras (lifting.lift), on the session's temporary disk, and that
cache no longer exists, so the observations are recomputed with the pilot's own pieces: the same train scenes
(configs/scannetpp_subset.json), the same 30 DSLR views per scene, Sonata (sonata.pth at the pinned commit) run once
per scene with the scene's seed, DINOv2 ViT-L/14 with registers at 518 x 770, and occlusion test v2.
lifting.observe is lift() stopped just before averaging: for every camera that sees a point it keeps the camera,
the pixel and the bilinearly sampled patch feature.

Per train scene:
  1. the pilot's labelled training sample (10,000 points, the scene's seed): how many cameras see each point;
  2. target candidates: candidate_factor x targets_per_scene random non-anonymised vertices (no labels used); those
     seen by >= min_cameras distinct cameras are eligible and targets_per_scene of them are kept;
  3. checks on the kept points: the mean of their observations equals lift()'s feature for the same points (the
     pilot's feature), every observation's pixel is where that vertex projects in that camera, vertex ids are
     unique, and a point's cameras are distinct;
  4. saved: <out>/observations/scenes/<scene>.npz (vertex ids, frozen Sonata upcast feature = the teacher's query,
     observations) and the scene's point arrays for the student (--points-cache, temporary disk).
Dev scenes (held out from teacher training) are chosen as in xattn.dev_split. Scenes already saved are skipped, so a
rerun resumes; an earlier session's saved scenes are found under /kaggle/input. Summary: <out>/observations/summary.json.

Usage: python experiments/Experiment2_CrossAttention/step1_observations.py --mode smoke|full [--input /kaggle/input]
Needs PYTHONPATH=src:third_party/utonia:third_party/sonata:third_party/concerto:experiments/Experiment1_Concatenation,
a GPU and Internet (weights).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.scannetpp_store import find_scenes
from encoders import attention_backend, encode_scene, load_encoder, make_transform, scene_seed
from lifting import DINO, OCCLUSION, lift, load_dino, observe, observed_mean, project
from pipeline import add_dino, prefetch, scene_views
from step2_lift import machine
from xattn import (CONFIG, OUT, dev_split, dir_gb, find_previous, load_config, load_points_scene, mark_mode, peak_gpu_gb, sample_points,
                   save_scene_observations, train_scenes)


def load_models(device: str):
    model, _ = load_encoder("sonata", seed=0)
    return model, make_transform("sonata"), load_dino(device)


def save_points(cache: Path, scene: str, s: dict) -> None:
    path = cache / f"{scene}.npz"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        np.savez(cache / f"{scene}.tmp.npz", coord=s["coord"].astype(np.float32), color=s["color"].astype(np.uint8),
                 normal=s["normal"].astype(np.float32))
        (cache / f"{scene}.tmp.npz").replace(path)


@torch.no_grad()
def process_scene(scene: str, s: dict, sonata, transform, dino, occ, run: dict, dev: str) -> tuple[dict, dict]:
    seed = scene_seed(scene)
    n = len(s["coord"])
    stages, tiles = encode_scene(sonata, transform, s, seed, store="cpu")  # the pilot's Sonata pass for this scene
    points = torch.from_numpy(s["coord"]).to(dev)
    normals = torch.from_numpy(s["normal"]).float().to(dev)
    views = scene_views(s, points, occ, device=dev)
    add_dino(dino, s, views, device=dev)

    # 1. the pilot's labelled training sample
    lab = torch.from_numpy(sample_points(s["segment"], s["anonymised"], run["probe"]["points_per_scene"], seed)).to(dev)
    (_, _, lab_count), = lift(points[lab], views, normals[lab], chunk=len(lab) + 1)
    lab_count = lab_count.cpu().numpy()

    # 2. target candidates (no labels) and the eligible ones
    keep = np.ones(n, bool)
    keep[s["anonymised"]] = False
    rng = np.random.default_rng(seed + 1)
    want = run["targets_per_scene"]
    cand = np.sort(rng.choice(np.flatnonzero(keep), min(run["candidate_factor"] * want, int(keep.sum())), replace=False))
    cand_t = torch.from_numpy(cand).to(dev)
    cand_obs = observe(points[cand_t], views, normals[cand_t])
    _, count = observed_mean(cand_obs, len(cand))
    eligible = np.flatnonzero(count.cpu().numpy() >= run["min_cameras"])
    vertex = cand[np.sort(rng.choice(eligible, min(want, len(eligible)), replace=False))]
    del cand_obs

    # the kept points' observations, computed on exactly the point set lift() is checked on (a projection batched
    # differently can differ in the last float bit); a point that drops below min_cameras there is removed
    for _ in range(3):
        vertex_t = torch.from_numpy(vertex).to(dev)
        obs = observe(points[vertex_t], views, normals[vertex_t])
        mean, count = observed_mean(obs, len(vertex))
        ok = (count >= run["min_cameras"]).cpu().numpy()
        if ok.all():
            break
        vertex = vertex[ok]
    assert ok.all(), f"{scene}: observations unstable"

    # 3. checks: the observations' mean is the pilot's lifted feature, pixels re-project, cameras are distinct
    (_, lifted, lifted_count), = lift(points[vertex_t], views, normals[vertex_t], chunk=len(vertex) + 1)
    row, view, uv, feat = obs["row"], obs["view"], obs["uv"], obs["feat"]
    reproj = 0.0
    for vi in view.unique().tolist():
        m = view == vi
        u, v, _ = project(points[vertex_t[row[m]]], views[vi].w2c, views[vi].K)
        reproj = max(reproj, float((torch.stack([u, v], 1) - uv[m]).abs().max()))
    pairs = row.long() * len(views) + view.long()
    checks = dict(
        mean_vs_lift_max_abs=float((mean - lifted).abs().max()) if len(vertex) else 0.0,
        mean_vs_lift_min_cos=float(F.cosine_similarity(mean, lifted).min()) if len(vertex) else 1.0,
        counts_equal=bool((count == lifted_count).all()),
        reprojection_max_px=reproj,
        vertex_ids_unique=bool(len(np.unique(vertex)) == len(vertex)),
        cameras_distinct=bool(len(pairs.unique()) == len(pairs)),
    )
    stats = dict(scene=scene, n_vertices=n, tiles=tiles, labelled=len(lab_count),
                 labelled_ge1=int((lab_count >= 1).sum()), labelled_ge3=int((lab_count >= 3).sum()),
                 candidates=len(cand), candidates_ge3=int(len(eligible)), targets=int(len(vertex)),
                 observations=int(len(row)), cameras_per_target=round(float(len(row) / max(len(vertex), 1)), 2), checks=checks)
    arrays = dict(vertex=vertex, query=stages.gather(torch.from_numpy(vertex), device=dev).cpu().numpy(),
                  row=row.cpu().numpy(), view=view.cpu().numpy(), uv=uv.cpu().numpy(), feat=feat.cpu().numpy())
    return arrays, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--points-cache", type=Path, default=Path("/tmp/exp2_points"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="cuda", help="cpu only for the local tests")
    args = ap.parse_args()
    cfg = load_config(args.config, args.mode)
    run = cfg["run"]
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("!! no GPU visible: set Accelerator to GPU T4 x2")
        return 2
    mark_mode(args.out, args.mode)
    root = args.out / "observations"
    root.mkdir(parents=True, exist_ok=True)
    record = collect()
    (root / "env_record.json").write_text(json.dumps(record, indent=2))
    print(f"git {json.dumps(record['git'])}\nmachine {json.dumps(machine())}\nmode {args.mode}: {json.dumps(run)}")

    section("1. Train scenes")
    archives = {s: p[0] for s, p in find_scenes(args.input)["core"].items()}
    scenes = train_scenes(archives, run["train_scenes"])
    if len(scenes) < run["train_scenes"]:
        print(f"!! {len(scenes)} of {run['train_scenes']} train scenes attached: attach scannetpp-core-train-a and -b")
        return 2
    dev_scenes = dev_split(scenes, run["dev_scenes"])
    print(f"  {len(scenes)} train scenes, of which {len(dev_scenes)} dev scenes (held out from teacher training)")

    todo = []
    for scene in scenes:
        if not (root / "scenes" / f"{scene}.npz").exists():
            prev = find_previous(f"observations/scenes/{scene}.npz", args.mode, roots=(args.out,), inputs=args.input)
            if prev is not None:
                (root / "scenes").mkdir(parents=True, exist_ok=True)
                shutil.copy(prev, root / "scenes" / f"{scene}.npz")
        if (root / "scenes" / f"{scene}.npz").exists():
            if not (args.points_cache / f"{scene}.npz").exists():  # a new session: the point arrays are re-read
                save_points(args.points_cache, scene, load_points_scene(archives[scene], scene))
        else:
            todo.append(scene)
    print(f"  {len(scenes) - len(todo)} scenes already saved (resumed), {len(todo)} to do")

    section(f"2. Observations (Sonata sonata.pth, {DINO[0]}, {run['views']} views, occlusion {run['occlusion']})")
    failed, t_start = [], time.time()
    if todo:
        sonata, transform, dino = load_models(dev)
        occ = OCCLUSION[run["occlusion"]]
        with attention_backend("sdpa"):
            for i, (scene, s, waited) in enumerate(prefetch([(sc, archives[sc]) for sc in todo], run["views"], args.workers), 1):
                try:
                    if isinstance(s, Exception):
                        raise s
                    t0 = time.perf_counter()
                    arrays, stats = process_scene(scene, s, sonata, transform, dino, occ, run, dev)
                    stats.update(dev=scene in dev_scenes, seconds=round(time.perf_counter() - t0, 1), waited=waited)
                    save_scene_observations(root, scene, **arrays, stats=stats)
                    save_points(args.points_cache, scene, s)
                    torch.cuda.empty_cache()
                    c = stats["checks"]
                    print(f"  [{i}/{len(todo)}] {scene}: {stats['targets']} targets ({stats['candidates_ge3']}/"
                          f"{stats['candidates']} candidates with >= {run['min_cameras']} cameras), "
                          f"{stats['cameras_per_target']} cameras per target; labelled sample >= 3 cameras "
                          f"{stats['labelled_ge3'] / max(stats['labelled'], 1):.3f}; checks: mean vs lift "
                          f"{c['mean_vs_lift_max_abs']:.2e}, re-projection {c['reprojection_max_px']:.2e} px; "
                          f"{stats['seconds']} s, waited {waited} s", flush=True)
                except Exception as e:  # noqa: BLE001  (one scene's failure must not end a long run)
                    failed.append(dict(scene=scene, error=f"{type(e).__name__}: {e}"[:300]))
                    print(f"  [{i}/{len(todo)}] {scene}: FAILED {failed[-1]['error']}\n{traceback.format_exc(limit=3)}", flush=True)
                    torch.cuda.empty_cache()

    stats = []
    for scene in scenes:
        path = root / "scenes" / f"{scene}.npz"
        if path.exists():
            st = json.loads(str(np.load(path)["stats"]))
            st["dev"] = scene in dev_scenes
            stats.append(st)
    done = [st["scene"] for st in stats]
    total = lambda key: sum(st[key] for st in stats)  # noqa: E731
    checks = dict(
        mean_vs_lift_max_abs=max((st["checks"]["mean_vs_lift_max_abs"] for st in stats), default=None),
        mean_vs_lift_min_cos=min((st["checks"]["mean_vs_lift_min_cos"] for st in stats), default=None),
        reprojection_max_px=max((st["checks"]["reprojection_max_px"] for st in stats), default=None),
        counts_equal=all(st["checks"]["counts_equal"] for st in stats),
        vertex_ids_unique=all(st["checks"]["vertex_ids_unique"] for st in stats),
        cameras_distinct=all(st["checks"]["cameras_distinct"] for st in stats),
    )
    secs = [st["seconds"] for st in stats if "seconds" in st]
    summary = dict(
        mode=args.mode, git=record["git"], scenes=done, dev_scenes=[s for s in dev_scenes if s in done], failed=failed,
        training_points=total("labelled"), training_points_ge1=total("labelled_ge1"), training_points_ge3=total("labelled_ge3"),
        fraction_training_points_ge3=round(total("labelled_ge3") / max(total("labelled"), 1), 4),
        candidates=total("candidates"), candidates_ge3=total("candidates_ge3"),
        fraction_candidates_ge3=round(total("candidates_ge3") / max(total("candidates"), 1), 4),
        targets=total("targets"), observations=total("observations"), checks=checks,
        seconds_per_scene_median=float(np.median(secs)) if secs else None, minutes=round((time.time() - t_start) / 60, 1),
        peak_gpu_gb=peak_gpu_gb(), store_gb=dir_gb(root), points_cache_gb=dir_gb(args.points_cache),
        estimate_full_minutes=round(float(np.median(secs)) * 200 / 60, 0) if secs else None, per_scene=stats)
    (root / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    ok = (not failed and len(done) == len(scenes) and checks["counts_equal"] and checks["vertex_ids_unique"]
          and checks["cameras_distinct"] and (checks["mean_vs_lift_max_abs"] or 0) < 1e-3
          and (checks["reprojection_max_px"] or 0) < 1e-2)

    section("SUMMARY step 1 observations (paste this block back)")
    print(f"{len(done)}/{len(scenes)} train scenes saved ({len(summary['dev_scenes'])} dev); failed {failed}; {summary['minutes']} min")
    print(f"pilot's labelled training points seen by >= 3 distinct cameras: {summary['fraction_training_points_ge3']:.4f} "
          f"({summary['training_points_ge3']:,}/{summary['training_points']:,}); by >= 1: "
          f"{summary['training_points_ge1'] / max(summary['training_points'], 1):.4f}")
    print(f"unlabelled target candidates with >= 3 cameras: {summary['fraction_candidates_ge3']:.4f}; targets kept "
          f"{summary['targets']:,} with {summary['observations']:,} observations")
    print(f"checks: mean of observations vs the pilot's lift(): max |diff| {checks['mean_vs_lift_max_abs']}, min cosine "
          f"{checks['mean_vs_lift_min_cos']}; camera counts equal {checks['counts_equal']}; re-projection max "
          f"{checks['reprojection_max_px']} px; vertex ids unique {checks['vertex_ids_unique']}; cameras distinct "
          f"{checks['cameras_distinct']}")
    print(f"peak GPU {summary['peak_gpu_gb']} GB; store {summary['store_gb']} GB in {root}; point cache "
          f"{summary['points_cache_gb']} GB in {args.points_cache}; median {summary['seconds_per_scene_median']} s per scene "
          f"(full run of 200 scenes ~ {summary['estimate_full_minutes']} min)")
    print("result: " + ("OK" if ok else "CHECK FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
