"""Experiment 2, step 2: train the cross-attention teacher on the step-1 observations (training scenes only, no labels).

Training points are the step-1 targets of the non-dev train scenes, each seen by >= 3 distinct cameras. Every epoch
each point draws a new camera triple (xattn.three_distinct): two input cameras and a held-out third. The teacher
(xattn.CrossAttentionTeacher) takes its query from the point's frozen original Sonata feature, its keys from the two
input cameras' DINOv2 features, and uses those DINOv2 vectors as values, so it predicts a weighted average in DINOv2
space. Loss: mean of 1 - cos(prediction, held-out camera's DINOv2 feature). Only W_q and W_k receive gradients;
Sonata and DINOv2 enter as fixed features.

Diagnostic (needs no student): on the dev scenes, fixed triples (seed 0); the mean cosine to the held-out camera of
the teacher and of the uniform average of the same two input cameras, every epoch and at the end, with a paired
bootstrap over dev scenes for the difference.

Checks before training (checks.json): every triple's three cameras are distinct and the held-out camera is not an
input; a teacher with equal attention weights reproduces the two-camera mean; one update changes W_q (a second one
W_k, which gets no gradient while W_q is still zero) and keeps everything finite.
Outputs: <out>/teacher/teacher.pt, diagnostic.json, checks.json, summary.json. An existing teacher.pt (this run's
or an attached earlier one) is reused unless --force.

Usage: python experiments/Experiment2_CrossAttention/step2_teacher.py --mode smoke|full
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from xattn import (CONFIG, OUT, CrossAttentionTeacher, ObservationStore, cosine_loss, find_previous, gather_triples, load_config,
                   peak_gpu_gb, three_distinct)


def check_triples(triples: np.ndarray, count: np.ndarray) -> dict:
    a, b, c = triples.T
    return dict(triples=len(triples), within_range=bool(((triples >= 0) & (triples < count[:, None])).all()),
                inputs_distinct=bool((a != b).all()), held_out_not_an_input=bool(((c != a) & (c != b)).all()))


@torch.no_grad()
def dev_cosines(teacher, store, points, triples, device, batch=8192) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per dev point: cosine of the teacher and of the uniform average to the held-out camera, and the teacher's
    weight on the first input camera."""
    t_cos, u_cos, w0 = [], [], []
    for s in range(0, len(points), batch):
        q, kv, target = gather_triples(store, points[s:s + batch], triples[s:s + batch], device)
        pred, w = teacher(q, kv)
        t_cos.append(torch.nn.functional.cosine_similarity(pred, target, dim=-1).cpu().numpy())
        u_cos.append(torch.nn.functional.cosine_similarity(kv.mean(1), target, dim=-1).cpu().numpy())
        w0.append(w[:, 0].cpu().numpy())
    return np.concatenate(t_cos), np.concatenate(u_cos), np.concatenate(w0)


def scene_bootstrap(diff: np.ndarray, scene: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """Mean of a per-point difference and its 95 % interval, resampling scenes (with all their points)."""
    ids = np.unique(scene)
    sums = np.array([diff[scene == i].sum() for i in ids])
    counts = np.array([(scene == i).sum() for i in ids])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), (n, len(ids)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return dict(mean=float(diff.mean()), lo=float(np.percentile(means, 2.5)), hi=float(np.percentile(means, 97.5)),
                p_le_0=float((means <= 0).mean()), scenes=len(ids))


def update_check(store, train_pts, cfg_t, device) -> dict:
    """One and two optimizer steps on a fresh teacher: which projections changed, and is everything finite."""
    rng = np.random.default_rng(123)
    pts = train_pts[rng.permutation(len(train_pts))[:min(cfg_t["batch"], len(train_pts))]]
    q, kv, target = gather_triples(store, pts, three_distinct(store.count[pts], rng), device)
    teacher = CrossAttentionTeacher(q.shape[1], kv.shape[2], cfg_t["dim"]).to(device)
    opt = torch.optim.AdamW(teacher.parameters(), lr=cfg_t["lr"], weight_decay=cfg_t["weight_decay"])
    out = {}
    for step in (1, 2):
        before = copy.deepcopy(teacher.state_dict())
        loss = cosine_loss(teacher(q, kv)[0], target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        after = teacher.state_dict()
        out[f"update{step}"] = dict(loss=float(loss.detach()), finite=bool(all(torch.isfinite(v).all() for v in after.values())),
                                    changed={k: bool((after[k] != before[k]).any()) for k in ("q.weight", "k.weight")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="retrain even if a teacher.pt exists")
    args = ap.parse_args()
    run = load_config(args.config, args.mode)["run"]
    cfg_t = run["teacher"]
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    out = args.out / "teacher"
    out.mkdir(parents=True, exist_ok=True)
    record = collect()
    obs_summary = json.loads((args.out / "observations" / "summary.json").read_text())
    scenes, dev_scenes = obs_summary["scenes"], obs_summary["dev_scenes"]
    print(f"git {json.dumps(record['git'])}\nmode {args.mode}: teacher {json.dumps(cfg_t)}, epochs {run['teacher_epochs']}")

    section("1. Observations")
    store = ObservationStore(args.out / "observations", scenes)
    dev_ids = [scenes.index(s) for s in dev_scenes]
    train_pts = store.points_of([i for i in range(len(scenes)) if i not in dev_ids])
    dev_pts = store.points_of(dev_ids)
    dev_triples = three_distinct(store.count[dev_pts], np.random.default_rng(0))
    print(f"  {len(train_pts):,} training points in {len(scenes) - len(dev_ids)} scenes, {len(dev_pts):,} dev points in "
          f"{len(dev_ids)} dev scenes; {len(store.obs_view):,} observations; query width {store.query.shape[1]}, DINO "
          f"width {store.obs_feat.shape[1]}")

    section("2. Checks: camera separation, equal weights = mean, one update")
    first = three_distinct(store.count[train_pts], np.random.default_rng(1))
    checks = dict(dev_triples=check_triples(dev_triples, store.count[dev_pts]),
                  train_triples=check_triples(first, store.count[train_pts]))
    probe = CrossAttentionTeacher(store.query.shape[1], store.obs_feat.shape[1], cfg_t["dim"]).to(dev)  # W_q = 0
    with torch.no_grad():
        q, kv, _ = gather_triples(store, dev_pts[:4096], dev_triples[:4096], dev)
        pred, w = probe(q, kv)
        checks["equal_weights"] = dict(max_weight_deviation=float((w - 0.5).abs().max()),
                                       max_abs_diff_to_mean=float((pred - kv.mean(1)).abs().max()))
    checks["update"] = update_check(store, train_pts, cfg_t, dev)
    ok_checks = (all(v for k, v in checks["dev_triples"].items() if k != "triples")
                 and all(v for k, v in checks["train_triples"].items() if k != "triples")
                 and checks["equal_weights"]["max_abs_diff_to_mean"] < 1e-4
                 and checks["update"]["update1"]["changed"]["q.weight"] and checks["update"]["update2"]["changed"]["k.weight"]
                 and checks["update"]["update1"]["finite"] and checks["update"]["update2"]["finite"])
    checks["ok"] = bool(ok_checks)
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    print(f"  {json.dumps(checks)}")
    if not ok_checks:
        print("!! a teacher check failed; not training")
        return 1

    section(f"3. Training ({run['teacher_epochs']} epochs, batch {cfg_t['batch']}, lr {cfg_t['lr']})")
    t_start = time.time()
    existing = None if args.force else find_previous("teacher/teacher.pt", roots=(args.out,), inputs=args.input)
    torch.manual_seed(cfg_t["seed"])
    teacher = CrossAttentionTeacher(store.query.shape[1], store.obs_feat.shape[1], cfg_t["dim"]).to(dev)
    curve = []
    if existing is not None:
        saved = torch.load(existing, weights_only=False)
        teacher.load_state_dict(saved["state_dict"])
        curve = saved.get("curve", [])
        if existing.resolve() != (out / "teacher.pt").resolve():
            shutil.copy(existing, out / "teacher.pt")
        print(f"  reusing {existing} (--force to retrain)")
    else:
        opt = torch.optim.AdamW(teacher.parameters(), lr=cfg_t["lr"], weight_decay=cfg_t["weight_decay"])
        steps = run["teacher_epochs"] * -(-len(train_pts) // cfg_t["batch"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + np.cos(np.pi * min(i, steps) / steps)))
        for epoch in range(run["teacher_epochs"]):
            rng = np.random.default_rng(1000 + epoch)
            order = rng.permutation(len(train_pts))
            pts = train_pts[order]
            triples = three_distinct(store.count[pts], rng)
            loss_sum, n_batches = 0.0, 0
            for s in range(0, len(pts), cfg_t["batch"]):
                q, kv, target = gather_triples(store, pts[s:s + cfg_t["batch"]], triples[s:s + cfg_t["batch"]], dev)
                loss = cosine_loss(teacher(q, kv)[0], target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                loss_sum, n_batches = loss_sum + float(loss.detach()), n_batches + 1
            t_cos, u_cos, _ = dev_cosines(teacher, store, dev_pts, dev_triples, dev)
            curve.append(dict(epoch=epoch + 1, train_loss=round(loss_sum / n_batches, 5), dev_teacher_cos=round(float(t_cos.mean()), 5),
                              dev_uniform_cos=round(float(u_cos.mean()), 5)))
            print(f"  epoch {epoch + 1}/{run['teacher_epochs']}: {json.dumps(curve[-1])}", flush=True)
        torch.save(dict(state_dict=teacher.state_dict(), query_dim=store.query.shape[1], key_dim=store.obs_feat.shape[1],
                        dim=cfg_t["dim"], config=cfg_t, epochs=run["teacher_epochs"], curve=curve, git=record["git"],
                        train_scenes=[s for s in scenes if s not in dev_scenes], dev_scenes=dev_scenes), out / "teacher.pt")

    section("4. Diagnostic on the dev triples")
    teacher.eval()
    t_cos, u_cos, w0 = dev_cosines(teacher, store, dev_pts, dev_triples, dev)
    dev_scene = store.scene[dev_pts]
    diag = dict(
        dev_scenes=dev_scenes, dev_points=int(len(dev_pts)), teacher_cos=float(t_cos.mean()), uniform_cos=float(u_cos.mean()),
        difference=scene_bootstrap(t_cos - u_cos, dev_scene), teacher_better_fraction=float((t_cos > u_cos).mean()),
        mean_abs_weight_shift=float(np.abs(w0 - 0.5).mean()),
        per_scene=[dict(scene=scenes[i], points=int((dev_scene == i).sum()), teacher_cos=float(t_cos[dev_scene == i].mean()),
                        uniform_cos=float(u_cos[dev_scene == i].mean())) for i in dev_ids],
        curve=curve, triples="three distinct cameras per dev point, seed 0; inputs = columns 0 and 1, held out = column 2")
    (out / "diagnostic.json").write_text(json.dumps(diag, indent=1))
    summary = dict(mode=args.mode, git=record["git"], train_points=int(len(train_pts)), dev_points=int(len(dev_pts)),
                   epochs=run["teacher_epochs"], minutes=round((time.time() - t_start) / 60, 2), peak_gpu_gb=peak_gpu_gb(),
                   reused=existing is not None, checks_ok=checks["ok"])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    section("SUMMARY step 2 teacher (paste this block back)")
    d = diag["difference"]
    print(f"checks: camera separation {checks['dev_triples']['held_out_not_an_input'] and checks['train_triples']['inputs_distinct']}, "
          f"equal weights = mean (max |diff| {checks['equal_weights']['max_abs_diff_to_mean']:.1e}), one update changes W_q "
          f"{checks['update']['update1']['changed']['q.weight']}, second W_k {checks['update']['update2']['changed']['k.weight']}")
    print(f"dev triples ({diag['dev_points']:,} points, {len(dev_scenes)} scenes): held-out cosine teacher {diag['teacher_cos']:.4f} "
          f"vs uniform average {diag['uniform_cos']:.4f}; difference {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]; teacher "
          f"better on {diag['teacher_better_fraction']:.3f} of triples; mean |w - 0.5| {diag['mean_abs_weight_shift']:.3f}")
    print(f"{summary['epochs']} epochs on {summary['train_points']:,} points, {summary['minutes']} min, peak GPU {summary['peak_gpu_gb']} GB"
          f"{' (reused an existing teacher)' if summary['reused'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
