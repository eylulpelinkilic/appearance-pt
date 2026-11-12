"""Experiment 3, Run C, part 2: the uniform-teacher student evaluated exactly as Experiment 2's step 4, and compared
with the attention-teacher student.

Features of the frozen uniform student at the pilot's labelled sample and at every val vertex: C1 f(p), and (config
c.head_features) C2 h(f(p)) and C3 [f(p), h(f(p))], with its own head; probes with the pilot's recipe; the official
metric; Experiment 2's image-free check. Compared by paired bootstrap with the attention student: C1 with A1
(Experiment 2's f(p)), and C2/C3 with A2/A3 when Run A's output is available (this run's or an attached one).
Decision rule (config.json): if |student_mean - student_attn| < 1 mIoU or its 95 % interval includes 0, report "no
evidence that learned attention matters" and never claim equivalence. A second training seed of each student is then
proposed with a time estimate, not run.
Outputs: <out>/runC/ (summary.json, confmats.npz, probes.pt).

Usage: python experiments/Experiment3_DistillationDiagnostics/stepC2_evaluate.py --mode smoke|full
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
from xattn import mark_mode, peak_gpu_gb, run_mode, train_scenes, val_scenes
from xdiag import CONFIG, OUT, find_exp2, find_pilot, find_runs, load_config, load_student_head, stack_config, write_json
from xprobe import diff_of, evaluate_val, fit_configs, flat, pilot_and_exp2, table_rows, train_features

NAMES = {"S": "Frozen Sonata", "S+": "Frozen Sonata + lifted DINOv2 (concatenation)", "A1": "Attention student f(p)",
         "A2": "Attention student h(f(p))", "A3": "Attention student [f(p), h(f(p))]", "C1": "Uniform-teacher student f(p)",
         "C2": "Uniform-teacher student h(f(p))", "C3": "Uniform-teacher student [f(p), h(f(p))]"}
INPUTS = {k: "point cloud" for k in NAMES} | {"S+": "point cloud + 30 posed DSLR photos"}
PROVENANCE = {"S": "pilot (reused)", "S+": "pilot (reused)", "A1": "Experiment 2 (reused confusion matrices)", "A2": "Run A",
              "A3": "Run A", "C1": "new (Run C)", "C2": "new (Run C)", "C3": "new (Run C)"}


def find_run_a(out: Path, inputs: Path, mode: str) -> Path | None:
    """runA/ of this output folder, else of an attached Experiment 3 output of the same mode (notebook 1's)."""
    if (out / "runA" / "summary.json").exists():
        return out / "runA"
    for d in find_runs(inputs, "Experiment3_DistillationDiagnostics"):
        if run_mode(d) == mode and (d / "runA" / "summary.json").exists():
            return d / "runA"
    return None


def main() -> int:
    before = set(sys.modules)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--exp2", type=Path)
    ap.add_argument("--pilot", type=Path)
    ap.add_argument("--any-exp2", action="store_true", help="tests only")
    ap.add_argument("--features", type=Path, default=Path("/tmp/exp3_runC_features"))
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/exp3_runC"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run = load_config(args.config, args.mode)["run"]
    pcfg = run["probe"]
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    mark_mode(args.out, args.mode)
    out = args.out / "runC"
    record = collect()
    t0 = time.time()

    section("1. Inputs: the uniform-teacher student and its head")
    exp2 = find_exp2(args.input, args.exp2, args.any_exp2)
    pilot = find_pilot(args.input, args.pilot)
    student, transform, head, facts = load_student_head(out / "student", dev)
    archives = {s: p[0] for s, p in find_scenes(args.input)["core"].items()}
    tscenes, vscenes = train_scenes(archives, run["probe_train_scenes"]), val_scenes(archives, run["val_scenes"])
    configs = {"C1": ("mean_f",)} | ({"C2": ("mean_h",), "C3": ("mean_f", "mean_h")} if run["c"]["head_features"] else {})
    enc = {"mean": (student, transform, head if run["c"]["head_features"] else None)}
    print(f"  head sha256 {facts['head_sha256'][:16]}..., step {facts['step']}/{facts['steps']}; configurations {list(configs)}")

    with attention_backend("sdpa"):
        section("2. Training features at the pilot's labelled sample")
        if args.features.exists():
            shutil.rmtree(args.features)
        feat_info = train_features(enc, archives, tscenes, pcfg["points_per_scene"], args.features, dev, args.workers)

        section("3. Probes with the pilot's recipe")
        folded, probe_info = fit_configs(args.features, configs, pcfg, dev, forbidden=args.out / "runB")
        torch.save(dict(probes=folded, info=probe_info, recipe=pcfg, git=record["git"]), out / "probes.pt")
        for c, v in probe_info.items():
            print(f"  {c} ({v['width']}-d): learning rate {v['lr']}")

        section("4. All val vertices")
        meta_dir = archives[vscenes[0]].parent.parent / "meta" / "metadata"
        gt_source = OfficialGT({s: archives[s] for s in vscenes}, meta_dir, args.tmp / "gt")
        cms, _, val_rows = evaluate_val(enc, folded, archives, vscenes, gt_source, dev, args.workers)
        free = image_free_check(student, transform, archives, vscenes[0], dev, args.tmp, before)

    section("5. Rows and the attention-vs-uniform comparison")
    allcms = pilot_and_exp2(pilot, exp2, vscenes)
    run_a = find_run_a(args.out, args.input, args.mode)
    if run_a is not None:
        za = np.load(run_a / "confmats.npz")
        asel = [[str(x) for x in za["scenes"]].index(s) for s in vscenes]
        for c in ("A2", "A3"):
            allcms[c] = {kk: stack_config(za, c, kk, asel) for kk in (1, 3)}
    for c in configs:
        allcms[c] = {kk: stack_config(flat(cms), c, kk) for kk in (1, 3)}
    rows, gain = table_rows(allcms, NAMES, INPUTS, PROVENANCE)
    for c in configs:
        rows[c]["lr"] = probe_info[c]["lr"]
    pairs = [("C1", "A1")] + [(c, a) for c, a in (("C2", "A2"), ("C3", "A3")) if c in allcms and a in allcms]
    diffs = {f"{c} - {a}": {f"top{kk}": diff_of(allcms[c], allcms[a], rows[c], rows[a], kk) for kk in (1, 3)} for c, a in pairs}
    key = diffs["C1 - A1"]["top1"]
    no_evidence = abs(key["diff"]) < 1 or key["lo"] <= 0 <= key["hi"]
    text = ("no evidence that learned attention matters" if no_evidence else
            f"the {'uniform' if key['diff'] > 0 else 'attention'}-teacher student is ahead by {abs(key['diff']):.2f} mIoU "
            f"(interval {key['lo']:+.2f} to {key['hi']:+.2f})")
    train_log = json.loads((out / "student" / "train_log.json").read_text())
    exp2_hours = (json.loads((exp2 / "reports" / "metrics.json").read_text()).get("student") or {}).get("session_hours")
    per_student_h = (train_log["session_hours"] if args.mode == "full" else (exp2_hours or 2.6)) + 0.5
    proposal = dict(text="Train a second seed of each student (seeds 1 for both the attention and the uniform student), each "
                         "followed by the step-4 evaluation, to measure training-seed variance.",
                    estimate_hours=round(2 * per_student_h, 1), per_student_hours=round(per_student_h, 1))
    np.savez_compressed(out / "confmats.npz", scenes=np.array(vscenes), **flat(cms))
    checks = json.loads((out / "student" / "checks.json").read_text())
    summary = dict(mode=args.mode, git=record["git"], exp2=str(exp2), run_a=str(run_a) if run_a else None, rows=rows, gain=gain,
                   diffs=diffs, decision=dict(rule="C", no_evidence=no_evidence, text=text), training=train_log | dict(curve=None),
                   training_curve=train_log["curve"], checks=dict(training=checks, head=facts, image_free=free), probes=probe_info,
                   train_features=feat_info, second_seed_proposal=proposal, val_scenes=vscenes, per_scene=val_rows,
                   minutes=round((time.time() - t0) / 60, 2), peak_gpu_gb=peak_gpu_gb())
    write_json(out / "summary.json", summary)

    section("SUMMARY Run C evaluation (paste this block back)")
    for c in [k for k in ("S", "S+", "A1", "A2", "A3", "C1", "C2", "C3") if k in rows]:
        r = rows[c]
        vs = r.get("vs_frozen_sonata_top1")
        print(f"  {c:3s} {r['name']:42s} top-1 {r['top1']:.4f} ± {r['top1_std']:.4f}  top-3 {r['top3']:.4f}  recovered "
              f"{r['recovered_top1']:+.3f} / {r['recovered_top3']:+.3f}" + (f"  vs S {vs['diff']:+.2f} [{vs['lo']:+.2f}, {vs['hi']:+.2f}]" if vs else ""))
    for k, d in diffs.items():
        print(f"  {k}: top-1 {d['top1']['diff']:+.4f} [{d['top1']['lo']:+.2f}, {d['top1']['hi']:+.2f}] P {d['top1']['p_le_0']}; "
              f"top-3 {d['top3']['diff']:+.4f} [{d['top3']['lo']:+.2f}, {d['top3']['hi']:+.2f}]")
    print(f"decision C: {text}")
    print(f"proposal (not run): {proposal['text']} About {proposal['estimate_hours']} GPU-hours "
          f"({proposal['per_student_hours']} h per student with its evaluation).")
    print(f"image-free {free['ok']}; {summary['minutes']} min, peak GPU {summary['peak_gpu_gb']} GB")
    return 0 if free["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
