"""Experiment 3, Run A: is the distilled information in the student but not linearly readable from f(p)?

Experiment 2's student was trained through a head h (LayerNorm, 1232 -> 1024, GELU, 1024 -> 1024) whose output was
matched to the teacher, while the linear probe read f(p), the 1232-d upcast feature before h. This run probes, with the
pilot's recipe (each feature set's learning rate chosen by the recipe's own rule):
  A1  f(p): Experiment 2's evaluation, reused (its saved per-scene confusion matrices);
  A2  h(f(p)), 1024-d;
  A3  [f(p), h(f(p))], 2256-d;
next to the pilot's frozen Sonata and Sonata + DINOv2 rows (reused). h is the head Experiment 2 saved in
student/checkpoint.pt (its SHA-256 is recorded). Checks: the head belongs to the saved student (same encoder weights
in both files) and ended training; reproduction (config.json check_rules): f(p) recomputed now, fed to Experiment 2's
own saved probes on every val scene, gives A1' within A1's probe-seed standard deviation, with every scene encoded in
as many tiles as in Experiment 2; feature widths; the image-free check of Experiment 2.
Decision rule (config.json): supported if A2 or A3 beats A1 in top-1 with a 95 % interval excluding 0.
Outputs: <out>/runA/ (summary.json, confmats.npz, probes.pt).

Usage: python experiments/Experiment3_DistillationDiagnostics/stepA_head_probes.py --mode smoke|full
Needs PYTHONPATH=src:third_party/utonia:third_party/sonata:third_party/concerto:experiments/Experiment1_Concatenation:
experiments/Experiment2_CrossAttention, a GPU, and Experiment 2's full-run output and the pilot_04_eval output attached.
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

from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.scannetpp_store import find_scenes
from encoders import attention_backend
from official import OfficialGT
from step4_evaluate import image_free_check
from xattn import mark_mode, peak_gpu_gb, train_scenes, val_scenes
from xdiag import CONFIG, OUT, find_exp2, find_pilot, load_config, load_student_head, stack_config, write_json
from xprobe import diff_of, evaluate_val, fit_configs, flat, pilot_and_exp2, row_of, table_rows, train_features

NAMES = {"S": "Frozen Sonata", "S+": "Frozen Sonata + lifted DINOv2 (concatenation)", "A1": "Student f(p)",
         "A2": "Student h(f(p))", "A3": "Student [f(p), h(f(p))]"}
INPUTS = {"S": "point cloud", "S+": "point cloud + 30 posed DSLR photos", "A1": "point cloud", "A2": "point cloud",
          "A3": "point cloud"}
PROVENANCE = {"S": "pilot (reused)", "S+": "pilot (reused)", "A1": "Experiment 2 (reused confusion matrices)",
              "A2": "new (Run A)", "A3": "new (Run A)"}
CONFIGS = {"A2": ("student_h",), "A3": ("student_f", "student_h")}


def reproduction(ref_cms: dict, a1: dict, exp2: Path, vscenes: list, val_rows: list) -> dict:
    """config.json check_rules.A_reproduction: A1' (Experiment 2's saved probes on today's f(p), every val scene) against
    A1 (Experiment 2's saved confusion matrices): the mIoU drift against A1's probe-seed standard deviation, the share of
    confusion counts that moved, and the tile count of every scene against Experiment 2's."""
    keys = sorted(ref_cms, key=lambda k: int(k.split("#")[1]))
    mine = {kk: np.stack([ref_cms[k][kk] for k in keys]) for kk in (1, 3)}  # (seeds, scenes, C, C), seeds as A1's
    assert all(mine[kk].shape == a1[kk].shape for kk in (1, 3)), (mine[1].shape, a1[1].shape)
    r_now, r_saved = row_of(mine[1], mine[3]), row_of(a1[1], a1[3])
    out = dict(scenes=len(vscenes), seeds=len(keys), a1_recomputed=r_now, diff_top1=r_now["top1"] - r_saved["top1"],
               diff_top3=r_now["top3"] - r_saved["top3"], tolerance_top1=r_saved["top1_std"], tolerance_top3=r_saved["top3_std"])
    for kk in (1, 3):
        d = np.abs(mine[kk].astype(np.int64) - a1[kk].astype(np.int64))
        out[f"moved_share_top{kk}"] = float(d.sum() / 2 / a1[kk].astype(np.int64).sum())
    d1 = np.abs(mine[1].astype(np.int64) - a1[1].astype(np.int64))
    out["max_abs_count_diff"] = int(max(d1.max(), np.abs(mine[3].astype(np.int64) - a1[3].astype(np.int64)).max()))
    out["identical_scenes"] = int((d1.reshape(d1.shape[0], d1.shape[1], -1).max(2).max(0) == 0).sum())
    saved = {r["scene"]: (r.get("tiles") or {}).get("student")
             for r in json.loads((exp2 / "evaluation" / "results.json").read_text()).get("per_scene", [])}
    now = {r["scene"]: r["tiles"]["student"] for r in val_rows}
    out["tiles_mismatch"] = {s: dict(now=now.get(s), experiment2=saved.get(s)) for s in vscenes if now.get(s) != saved.get(s)}
    out["tiles_match"] = not out["tiles_mismatch"]
    out["tiles_more_than_one"] = {s: n for s, n in now.items() if n != 1}
    out["ok"] = bool(out["tiles_match"] and abs(out["diff_top1"]) <= out["tolerance_top1"]
                     and abs(out["diff_top3"]) <= out["tolerance_top3"])
    return out


def main() -> int:
    before = set(sys.modules)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--exp2", type=Path, help="Experiment 2's output folder (default: found under --input)")
    ap.add_argument("--pilot", type=Path, help="the pilot's step-5 folder (default: found under --input)")
    ap.add_argument("--any-exp2", action="store_true", help="accept an Experiment 2 output that is not the full run (tests)")
    ap.add_argument("--features", type=Path, default=Path("/tmp/exp3_runA_features"))
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/exp3_runA"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run = load_config(args.config, args.mode)["run"]
    pcfg = run["probe"]
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    mark_mode(args.out, args.mode)
    out = args.out / "runA"
    out.mkdir(parents=True, exist_ok=True)
    record = collect()
    t0 = time.time()
    print(f"git {json.dumps(record['git'])}\nmode {args.mode}: probe {json.dumps(pcfg)}")

    section("1. Inputs: Experiment 2's student and head, the pilot's rows")
    exp2 = find_exp2(args.input, args.exp2, args.any_exp2)
    pilot = find_pilot(args.input, args.pilot)
    student, transform, head, facts = load_student_head(exp2 / "student", dev)
    archives = {s: p[0] for s, p in find_scenes(args.input)["core"].items()}
    tscenes, vscenes = train_scenes(archives, run["probe_train_scenes"]), val_scenes(archives, run["val_scenes"])
    print(f"  Experiment 2: {exp2}\n  pilot: {pilot}\n  head sha256 {facts['head_sha256'][:16]}..., encoder sha256 "
          f"{facts['encoder_sha256'][:16]}..., step {facts['step']}/{facts['steps']}; {len(tscenes)} train, {len(vscenes)} val scenes")

    enc = {"student": (student, transform, head)}
    with attention_backend("sdpa"):
        section("2. Training features f(p) and h(f(p)) at the pilot's labelled sample")
        if args.features.exists():
            shutil.rmtree(args.features)
        feat_info = train_features(enc, archives, tscenes, pcfg["points_per_scene"], args.features, dev, args.workers)
        print(f"  {json.dumps(feat_info)}")

        section("3. Probes with the pilot's recipe (learning rate per feature set)")
        folded, probe_info = fit_configs(args.features, CONFIGS, pcfg, dev, forbidden=args.out / "runB")
        torch.save(dict(probes=folded, info=probe_info, recipe=pcfg, git=record["git"]), out / "probes.pt")
        for c, v in probe_info.items():
            print(f"  {c} ({v['width']}-d): learning rate {v['lr']} (held-out mIoU {json.dumps(v['held_out_miou'])})")

        section("4. All val vertices; Experiment 2's own probes on today's f(p) (reproduction check)")
        meta_dir = archives[vscenes[0]].parent.parent / "meta" / "metadata"
        gt_source = OfficialGT({s: archives[s] for s in vscenes}, meta_dir, args.tmp / "gt")
        saved2 = torch.load(exp2 / "evaluation" / "probes.pt", weights_only=False)
        reference = {k: v for k, v in saved2["probes"].items() if k.startswith("T#")}
        cms, ref_cms, val_rows = evaluate_val(enc, folded, archives, vscenes, gt_source, dev, args.workers,
                                              reference=reference, reference_scenes=len(vscenes))

        section("5. Image-free check (Experiment 2's)")
        free = image_free_check(student, transform, archives, vscenes[0], dev, args.tmp, before)
        print(f"  {json.dumps({k: free[k] for k in ('ok', 'max_abs_feature_diff', 'image_modules_imported_by_this_step')})}")

    section("6. Rows and differences")
    allcms = pilot_and_exp2(pilot, exp2, vscenes)
    repro = reproduction(ref_cms, allcms["A1"], exp2, vscenes, val_rows)
    widths = feat_info["widths"]
    checks = dict(
        head=facts, head_matches_student=True, ended_training=facts["step"] == facts["steps"], reproduction=repro,
        widths=dict(f=widths["student_f"], h=widths["student_h"], f_plus_h=widths["student_f"] + widths["student_h"],
                    ok=widths["student_f"] == facts["head_shapes"]["net.1.weight"][1]
                    and widths["student_h"] == facts["head_shapes"]["net.3.weight"][0]),
        image_free=free)
    for c in CONFIGS:
        allcms[c] = {kk: stack_config(flat(cms), c, kk) for kk in (1, 3)}
    rows, gain = table_rows(allcms, NAMES, INPUTS, PROVENANCE)
    for c in CONFIGS:
        rows[c]["lr"] = probe_info[c]["lr"]
    rows["A1"]["lr"] = (json.loads((exp2 / "reports" / "metrics.json").read_text()).get("probes") or {}).get("lr")
    diffs = {f"{c} - A1": {f"top{kk}": diff_of(allcms[c], allcms["A1"], rows[c], rows["A1"], kk) for kk in (1, 3)}
             for c in CONFIGS}
    supported = any(diffs[f"{c} - A1"]["top1"]["lo"] > 0 for c in CONFIGS)
    np.savez_compressed(out / "confmats.npz", scenes=np.array(vscenes), **flat(cms))
    summary = dict(mode=args.mode, git=record["git"], exp2=str(exp2), pilot=str(pilot), rows=rows, gain=gain, diffs=diffs,
                   decision=dict(rule="A", supported=supported,
                                 text="supported: the distilled information is better read through h" if supported else
                                      "not supported: neither h(f(p)) nor [f(p), h(f(p))] beats f(p) with an interval above 0"),
                   probes=probe_info, train_features=feat_info, checks=checks, val_scenes=vscenes, per_scene=val_rows,
                   minutes=round((time.time() - t0) / 60, 2), peak_gpu_gb=peak_gpu_gb())
    write_json(out / "summary.json", summary)
    ok = checks["reproduction"]["ok"] and checks["widths"]["ok"] and free["ok"] and checks["ended_training"]

    section("SUMMARY Run A (paste this block back)")
    for c in ("S", "S+", "A1", "A2", "A3"):
        r = rows[c]
        vs = r.get("vs_frozen_sonata_top1")
        print(f"  {c:3s} {r['name']:46s} top-1 {r['top1']:.4f} ± {r['top1_std']:.4f}  top-3 {r['top3']:.4f}  "
              f"recovered {r['recovered_top1']:+.3f} / {r['recovered_top3']:+.3f}"
              + (f"  vs S {vs['diff']:+.2f} [{vs['lo']:+.2f}, {vs['hi']:+.2f}]" if vs else "") + f"  lr {r.get('lr', '-')}")
    for k, d in diffs.items():
        print(f"  {k}: top-1 {d['top1']['diff']:+.4f} [{d['top1']['lo']:+.2f}, {d['top1']['hi']:+.2f}] P {d['top1']['p_le_0']}; "
              f"top-3 {d['top3']['diff']:+.4f} [{d['top3']['lo']:+.2f}, {d['top3']['hi']:+.2f}]")
    print(f"decision A: {summary['decision']['text']}")
    print(f"checks: head sha256 {facts['head_sha256'][:16]}... (step {facts['step']}/{facts['steps']}); widths "
          f"{json.dumps(checks['widths'])}; image-free {free['ok']}")
    print(f"reproduction ({repro['scenes']} val scenes, Experiment 2's saved probes on today's f(p)): A1' top-1 "
          f"{repro['a1_recomputed']['top1']:.4f} vs A1 {rows['A1']['top1']:.4f} (diff {repro['diff_top1']:+.4f}, tolerance "
          f"{repro['tolerance_top1']:.4f}); top-3 {repro['a1_recomputed']['top3']:.4f} vs {rows['A1']['top3']:.4f} (diff "
          f"{repro['diff_top3']:+.4f}, tolerance {repro['tolerance_top3']:.4f}); moved counts {repro['moved_share_top1']:.2e} "
          f"(top-1) / {repro['moved_share_top3']:.2e} (top-3), max count diff {repro['max_abs_count_diff']}, identical "
          f"scenes {repro['identical_scenes']}; tiles match {repro['tiles_match']} -> {'OK' if repro['ok'] else 'FAILED'}")
    print(f"{summary['minutes']} min, peak GPU {summary['peak_gpu_gb']} GB; result {'OK' if ok else 'CHECK FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
