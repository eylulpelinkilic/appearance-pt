"""Pilot step 5: every step-4 probe on all 50 val scenes, scored with the official ScanNet++ semantic metric.

Per val scene (loaded in background threads, with the toolbox's own top-3 GT, official.py):
  - the four encoders run once over the whole scene (seed = encoders.scene_seed, fp16, SDPA; tiles if out of memory),
    with stage features kept in CPU memory; DINO is lifted onto every vertex from the same --views views and with the
    same occlusion variant as in step 3;
  - in chunks of vertices: every probe's logits (FoldedSet), top-3 classes, and confusion matrices (metrics.py, equal
    to the toolbox's) for top-1 and top-3, over the vertices the evaluation keeps (anonymised ones excluded), split
    into vertices seen by >= 1 view and never seen.
Then per configuration: mIoU mean and std over the probe seeds; per-class IoU; per-class gains of K+ over K (and of
the other "+ DINO" pairs); paired bootstrap over scenes for the main differences. Cross-check: one probe's
predictions on --official-scenes scenes are written as the toolbox's text files and scored by its eval_semantic.
Results: <out>/results.json, <out>/confmats.npz (per scene and probe, top-1 and top-3).

Usage: python experiments/pilot_concat/step5_eval.py [--input /kaggle/input] [--probes .../step4/probes.pt] [--out DIR]
Needs PYTHONPATH=src:third_party/utonia:third_party/sonata:third_party/concerto, a GPU and Internet (weights).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch

from appearance_pt import REPO_ROOT
from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.scannetpp_store import find_scenes
from encoders import attention_backend, scene_seed
from lifting import OCCLUSION, lift
from metrics import bootstrap_diff, confusion_topk, ious, miou
from official import OfficialGT, compare_with_pointcept
from pipeline import ENCODER_BLOCKS, Models, add_dino, encode_blocks, prefetch, scene_views
from probes import NUM_CLASSES, FoldedSet
from scenes import load_core_scene, raw_features
from step2_lift import machine

SUBSETS = ("all", "seen", "unseen")
PAIRS = (("K+", "K"), ("C", "B"), ("S+", "S"), ("R+", "R"), ("K", "S"), ("K", "B"), ("K+", "C"), ("K+-eq", "K+"), ("C-eq", "C"))
PAPER = {"S": 38.9, "K": 45.6}  # Concerto paper, linear probing on ScanNet++ val (mIoU, %)


def write_labels(path: Path, labels: np.ndarray) -> None:
    np.savetxt(path, labels, fmt="%d", delimiter=",")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--probes", type=Path, default=Path("/kaggle/working/pilot_concat/step4/probes.pt"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/pilot_concat/step5"))
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/pilot_step5"))
    ap.add_argument("--chunk", type=int, default=100_000, help="vertices per prediction chunk")
    ap.add_argument("--official-probe", default="K#0", help="probe whose predictions the toolbox's eval_semantic re-scores")
    ap.add_argument("--official-scenes", type=int, default=10)
    ap.add_argument("--limit", type=int, help="only the first N val scenes (a quick test)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="cuda", help="cpu only for the local tests")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("!! no GPU visible: set Accelerator to GPU T4 x2")
        return 2
    record = collect()
    (args.out / "env_record.json").write_text(json.dumps(record, indent=2))
    box = machine()
    print(f"git {json.dumps(record['git'])}\nmachine {json.dumps(box)}\nargs {vars(args)}")

    saved = torch.load(args.probes, weights_only=False)
    feat_meta = saved["features"]
    folded = FoldedSet(saved["probes"], device=dev)
    configs = list(saved["configs"])
    n_views, occ_name, concerto = feat_meta["views"], feat_meta["occlusion"], feat_meta["concerto"]
    seeds = sorted({int(k.split("#")[1]) for k in folded.keys})
    print(f"probes: {len(folded.keys)} ({len(configs)} configurations x {len(seeds)} seeds); features: {n_views} views, "
          f"occlusion {occ_name}, concerto {concerto}, points/scene {feat_meta['points_per_scene']}")

    section("1. Val scenes and the toolbox's ground truth")
    archives = {s: p[0] for s, p in find_scenes(args.input)["core"].items()}
    subset = json.loads((REPO_ROOT / "configs" / "scannetpp_subset.json").read_text())
    scenes = [s for s in subset["val"] if s in archives][:args.limit]
    meta_dir = archives[scenes[0]].parent.parent / "meta" / "metadata"
    gt_source = OfficialGT({s: archives[s] for s in scenes}, meta_dir, args.tmp)
    class_names = gt_source.class_names
    print(f"  {len(scenes)} val scenes; {len(class_names)} classes; metadata {meta_dir}")

    def load(scene, path):
        s = load_core_scene(path, scene, n_views)
        s["gt"] = gt_source.labels(scene)
        return s

    section("2. Models")
    models = Models(concerto=feat_meta["encoder_names"]["concerto"], device=dev)  # the same weights as the training features
    assert models.names == feat_meta["encoder_names"], (models.names, feat_meta["encoder_names"])
    occ = OCCLUSION[occ_name]

    section("3. Scenes")
    keys = folded.keys
    total = {k: {kk: {sub: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=dev) for sub in SUBSETS}
                 for kk in (1, 3)} for k in keys}
    per_scene = {k: {1: [], 3: []} for k in keys}
    pred_dir, gt_dir = args.tmp / "official_pred", args.tmp / "official_gt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    rows, failed, t_start = [], [], time.time()
    with attention_backend("sdpa"):
        for i, (scene, s, waited) in enumerate(prefetch([(sc, archives[sc]) for sc in scenes], n_views, args.workers, load=load), 1):
            try:
                if isinstance(s, Exception):
                    raise s
                t0 = time.perf_counter()
                n = len(s["coord"])
                gt = s["gt"]
                assert len(gt) == n, f"GT has {len(gt)} vertices, the scene {n}"
                keep = np.ones(n, bool)
                keep[s["anonymised"]] = False
                stages, info = encode_blocks(models, s, scene_seed(scene), store="cpu")
                points = torch.from_numpy(s["coord"]).to(dev)
                normals = torch.from_numpy(s["normal"]).float().to(dev)
                views = scene_views(s, points, occ, device=dev)
                add_dino(models.dino, s, views, device=dev)
                raw = torch.from_numpy(raw_features(s)).to(dev)
                gt_t, keep_t = torch.from_numpy(gt).to(dev), torch.from_numpy(keep).to(dev)
                scene_cm = {k: {kk: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=dev) for kk in (1, 3)} for k in keys}
                official_pred = np.zeros((n, 3), np.int16) if i <= args.official_scenes else None
                seen_total = 0
                for start, dino, count in lift(points, views, normals, chunk=args.chunk):
                    idx = torch.arange(start, start + len(count))
                    feats = {b: stages[b].gather(idx, device=dev) for b in ENCODER_BLOCKS}
                    feats["dino"], feats["raw"] = dino, raw[idx.to(dev)]
                    logits = folded.logits(feats)
                    del feats
                    g, kp, seen = gt_t[start:start + len(count)], keep_t[start:start + len(count)], count > 0
                    seen_total += int((seen & kp).sum())
                    masks = {"all": kp, "seen": kp & seen, "unseen": kp & ~seen}
                    for k, lg in logits.items():
                        top = lg.topk(3, dim=1).indices
                        if official_pred is not None and k == args.official_probe:
                            official_pred[start:start + len(count)] = top.cpu().numpy()
                        for kk in (1, 3):
                            for sub, m in masks.items():
                                cm = confusion_topk(top[m, :kk], g[m], NUM_CLASSES)
                                total[k][kk][sub] += cm
                                if sub == "all":
                                    scene_cm[k][kk] += cm
                    del logits
                for k in keys:
                    for kk in (1, 3):
                        per_scene[k][kk].append(scene_cm[k][kk].cpu().numpy().astype(np.int32))
                if official_pred is not None:
                    write_labels(pred_dir / f"{scene}.txt", official_pred)
                    write_labels(gt_dir / f"{scene}.txt", gt)
                del stages, views, points, normals
                torch.cuda.empty_cache()
                cmp = compare_with_pointcept(gt, s["segment"])
                quick = {c: round(100 * float(miou(per_scene[f"{c}#0"][1][-1])), 1) for c in ("K", "K+") if f"{c}#0" in keys}
                row = dict(scene=scene, n_vertices=n, kept=int(keep.sum()), seen=round(seen_total / max(int(keep.sum()), 1), 4),
                           tiles={b: t for b, (t, _) in info.items()}, sec_encoders={b: sec for b, (_, sec) in info.items()},
                           gt_vs_pointcept=cmp, sec_scene=round(time.perf_counter() - t0, 1), sec_wait=waited)
                rows.append(row)
                eta = (time.time() - t_start) / i * (len(scenes) - i) / 60
                print(f"  [{i}/{len(scenes)}] {scene}: {n:,} vertices, seen {row['seen']:.3f}, tiles {row['tiles']}, scene mIoU "
                      f"(seed 0, top-1) {quick}; GT vs Pointcept {cmp}; {row['sec_scene']} s, waited {waited} s; "
                      f"ETA {eta:.0f} min", flush=True)
            except Exception as e:  # noqa: BLE001
                failed.append(dict(scene=scene, error=f"{type(e).__name__}: {e}"[:300]))
                print(f"  [{i}/{len(scenes)}] {scene}: FAILED {failed[-1]['error']}\n{traceback.format_exc(limit=3)}", flush=True)
                torch.cuda.empty_cache()
    done = [r["scene"] for r in rows]
    if not done:
        print("!! no scene evaluated")
        return 1

    section("4. Aggregates")
    total_np = {k: {kk: {sub: cm.cpu().numpy() for sub, cm in v.items()} for kk, v in d.items()} for k, d in total.items()}
    res = {}
    for c in configs:
        ks = [f"{c}#{s}" for s in seeds if f"{c}#{s}" in total_np]
        entry = {}
        for kk in (1, 3):
            for sub in SUBSETS:
                vals = [100 * float(miou(total_np[k][kk][sub])) for k in ks]
                entry[f"top{kk}_{sub}"] = (round(float(np.mean(vals)), 2), round(float(np.std(vals)), 2))
        with warnings.catch_warnings():  # classes absent from every seed's matrix stay NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            entry["iou_top1"] = {sub: np.nanmean([ious(total_np[k][1][sub]) for k in ks], 0) for sub in SUBSETS}
        res[c] = entry
    gains = {}
    for a, b in PAIRS:
        if a in res and b in res:
            gains[f"{a} - {b}"] = {sub: res[a]["iou_top1"][sub] - res[b]["iou_top1"][sub] for sub in SUBSETS}
    stacked = {k: np.stack(per_scene[k][1]) for k in keys}  # (scenes, C, C)
    boot = {}
    for a, b in PAIRS:
        ka, kb = [f"{a}#{s}" for s in seeds], [f"{b}#{s}" for s in seeds]
        if all(k in stacked for k in ka + kb):
            d = bootstrap_diff(np.stack([stacked[k] for k in ka]), np.stack([stacked[k] for k in kb]))
            boot[f"{a} - {b}"] = {k: (round(100 * v, 2) if k != "p_le_0" else v) for k, v in d.items()}

    section("5. Cross-check with the toolbox's eval_semantic")
    checked = [sc for sc in done[:args.official_scenes] if (pred_dir / f"{sc}.txt").exists()]
    official = {}
    if checked and args.official_probe in keys:
        theirs = gt_source.evaluate_files(checked, pred_dir, gt_dir)
        rows_idx = [done.index(sc) for sc in checked]
        ours = {kk: float(miou(np.stack(per_scene[args.official_probe][kk])[rows_idx].sum(0))) for kk in (1, 3)}
        official = dict(probe=args.official_probe, scenes=len(checked), toolbox={k: round(v, 6) for k, v in theirs.items()},
                        ours={k: round(v, 6) for k, v in ours.items()},
                        equal=all(abs(theirs[k] - ours[k]) < 1e-9 for k in (1, 3)))
        print(f"  {official}")

    def table(arr):
        return {name: (None if np.isnan(v) else round(100 * float(v), 2)) for name, v in zip(class_names, arr)}

    out = dict(args={k: str(v) for k, v in vars(args).items()}, git=record["git"], machine=box, features=feat_meta,
               lr=saved["lr"], scenes=done, failed=failed, per_scene=rows, official_check=official,
               miou={c: {k: v for k, v in e.items() if k != "iou_top1"} for c, e in res.items()},
               iou_top1={c: {sub: table(e["iou_top1"][sub]) for sub in SUBSETS} for c, e in res.items()},
               gains_top1={p: {sub: table(g[sub]) for sub in SUBSETS} for p, g in gains.items()},
               bootstrap_top1=boot, minutes=round((time.time() - t_start) / 60, 1))
    (args.out / "results.json").write_text(json.dumps(out, indent=1, default=str))
    np.savez_compressed(args.out / "confmats.npz", scenes=np.array(done),
                        **{f"{k}|top{kk}": np.stack(per_scene[k][kk]) for k in keys for kk in (1, 3)})

    section("SUMMARY step 5 (paste this block back)")
    print(f"{len(done)}/{len(scenes)} val scenes; failed {failed}; {out['minutes']} min")
    gtc = [r["gt_vs_pointcept"] for r in rows]
    print(f"toolbox GT vs Pointcept segment: same rows {np.mean([g['same_rows'] for g in gtc]):.4f}, same label sets "
          f"{np.mean([g['same_label_sets'] for g in gtc]):.4f}, multilabel vertices {np.mean([g['multilabel'] for g in gtc]):.4f}")
    print(f"official eval_semantic cross-check: {json.dumps(official)}")
    print(f"vertices seen by >= 1 view: {np.mean([r['seen'] for r in rows]):.3f} (scene mean)")
    print(f"{'config':7s} {'top-1 mIoU':>14s} {'top-3 mIoU':>14s} {'top-1 seen':>12s} {'top-1 unseen':>13s}  (mean, std over {len(seeds)} probe seeds)")
    for c, e in res.items():
        paper = f"  paper {PAPER[c]}" if c in PAPER else ""
        print(f"{c:7s} {e['top1_all'][0]:7.2f} ± {e['top1_all'][1]:4.2f} {e['top3_all'][0]:7.2f} ± {e['top3_all'][1]:4.2f} "
              f"{e['top1_seen'][0]:12.2f} {e['top1_unseen'][0]:13.2f}{paper}")
    print("paired bootstrap over scenes, top-1 mIoU difference [95% interval], P(diff <= 0):")
    for p, d in boot.items():
        print(f"  {p:12s} {d['diff']:+6.2f} [{d['lo']:+6.2f}, {d['hi']:+6.2f}]  P={d['p_le_0']}")
    if "K+ - K" in gains:
        g = gains["K+ - K"]["all"]
        order = np.argsort(np.nan_to_num(g, nan=0.0))
        fmt = lambda ix: ", ".join(f"{class_names[j]} {100 * g[j]:+.1f}" for j in ix if not np.isnan(g[j]))  # noqa: E731
        print(f"K+ - K per-class IoU (top-1), largest gains: {fmt(order[::-1][:12])}")
        print(f"K+ - K per-class IoU (top-1), largest losses: {fmt(order[:8])}")
        print(f"K+ - K mean per-class gain: all {100 * np.nanmean(g):+.2f}, seen {100 * np.nanmean(gains['K+ - K']['seen']):+.2f}, "
              f"unseen {100 * np.nanmean(gains['K+ - K']['unseen']):+.2f}")
    print(f"full results: {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
