"""Pilot step 1b: Concerto as a frozen encoder (configs K, K+), verified as step 1 verified Utonia and Sonata.

On Pointcept's public demo room it checks
  1. the stored configs and upcast widths of concerto_large_origin (the original Concerto, used for K) and
     concerto_large (per the HF model card a pre-release of Utonia; compared only),
  2. whether concerto_large_origin holds the backbone of pretrain-concerto-v1m1-2-large-video.pth, the checkpoint
     Pointcept's Concerto README links for probing (student and teacher backbones compared tensor by tensor),
  3. which of the two the released ScanNet-20 probe (concerto_large_linear_prob_head_sc) belongs to,
  4. SDPA vs the explicit attention product, fp16 vs fp32, run-to-run variation, zeroed colour or normals,
then measures GPU time and peak memory per scene against scene size (the room tiled k x k), like step 1.
Results: <out>/step1b_summary.json and <out>/env_record.json.

Usage: python experiments/pilot_concat/step1b_concerto.py [--out DIR] [--tiles 1 2 3] [--skip-pretrain-check]
Needs PYTHONPATH=src:third_party/utonia:third_party/sonata:third_party/concerto and Internet (Hugging Face).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.smoke import load_demo_sample
from encoders import WEIGHTS, checkpoint, load_encoder, load_scannet_probe
from scenes import tile_scene
from step1_encoders import bench_one, compare, describe, predict, print_bench, run, score

PRETRAIN = "pretrain-concerto-v1m1-2-large-video.pth"  # Pointcept training checkpoint, 8.3 GB
BENCH = (("concerto", "sdpa", True), ("concerto", "manual", True), ("concerto", "sdpa", False))


def same_backbone(tmp: Path) -> dict:
    """Compare concerto_large_origin's state_dict with the student and teacher backbones of PRETRAIN."""
    origin = checkpoint("concerto", "concerto_large_origin.pth")["state_dict"]
    repo_id, revision = WEIGHTS["concerto"]
    try:
        path = hf_hub_download(repo_id, PRETRAIN, revision=revision, local_dir=tmp)
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        state = state.get("state_dict", state)
        out = {}
        for part in ("student", "teacher"):
            prefix = next((p for p in (f"module.{part}.backbone.", f"{part}.backbone.") if any(k.startswith(p) for k in state)), None)
            if prefix is None:
                out[part] = "no backbone keys"
                continue
            other = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            common = [k for k in origin if k in other and origin[k].shape == other[k].shape]
            max_abs = max((float((origin[k].float() - other[k].float()).abs().max()) for k in common), default=None)
            out[part] = dict(prefix=prefix, n_origin=len(origin), n_pretrain=len(other), n_common_same_shape=len(common),
                             only_origin=sorted(set(origin) - set(other))[:3], only_pretrain=sorted(set(other) - set(origin))[:3],
                             max_abs_diff=max_abs)
        out["top_level_keys"] = sorted(state)[:5] if not any("backbone" in k for k in state) else None
        return out
    except Exception as e:  # noqa: BLE001  (report and go on: the other checks do not depend on it)
        return dict(status=f"{type(e).__name__}: {e}"[:300])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def checks(models, head, room, gt, seed) -> dict:
    res = {}
    name = "concerto"
    section("4a. SDPA vs explicit softmax(QK^T)V attention (same seed, so same voxels and orders)")
    res["sdpa_vs_manual"] = {}
    for amp in (False, True):
        a = run(models, name, room, seed, amp=amp, backend="manual")
        b = run(models, name, room, seed, amp=amp, backend="sdpa")
        key = "fp16" if amp else "fp32"
        res["sdpa_vs_manual"][key] = dict(**compare(a, b), pred_agreement=round(float((predict(head, a) == predict(head, b)).mean()), 5))
        print(f"  {key} {res['sdpa_vs_manual'][key]}")

    section("4b. fp16 autocast vs fp32 (SDPA, same seed)")
    a, b = run(models, name, room, seed, amp=False), run(models, name, room, seed, amp=True)
    res["fp16_vs_fp32"] = dict(**compare(a, b), pred_agreement=round(float((predict(head, a) == predict(head, b)).mean()), 5),
                               miou_fp32=score(head, a, gt)["miou"], miou_fp16=score(head, b, gt)["miou"])
    print(f"  {res['fp16_vs_fp32']}")

    section("4c. Run-to-run variation (fp16, SDPA)")
    a, b, c = (run(models, name, room, s, amp=True) for s in (seed, seed, seed + 1))
    res["randomness"] = dict(same_seed_max_abs=compare(a, b)["max_abs"], other_seed=compare(a, c),
                             other_seed_pred_agreement=round(float((predict(head, a) == predict(head, c)).mean()), 5),
                             miou_seeds=[score(head, a, gt)["miou"], score(head, c, gt)["miou"]])
    print(f"  {res['randomness']}")

    section("4d. Zeroed colour or normals (anonymised ScanNet++ vertices have no colour; fp16, SDPA)")
    res["zeroed"] = {"none": score(head, a, gt)}
    for key in ("color", "normal"):
        res["zeroed"][key] = score(head, run(models, name, {**room, key: np.zeros_like(room[key])}, seed, amp=True), gt)
    print(f"  {res['zeroed']}")

    feat = a[0]
    res["upcast"] = dict(width=int(feat.shape[1]), finite=bool(torch.isfinite(feat).all()), std=round(float(feat.float().std()), 4))
    print(f"  upcast {res['upcast']}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/pilot_concat/step1b"))
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/concerto_pretrain"), help="where the 8.3 GB checkpoint goes (deleted)")
    ap.add_argument("--tiles", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6], help="k for k x k copies of the room")
    ap.add_argument("--repeats", type=int, default=2, help="timed calls per scene after the first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-pretrain-check", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        print("!! no GPU visible: set Accelerator to GPU T4 x2")
        return 2
    record = collect()
    (args.out / "env_record.json").write_text(json.dumps(record, indent=2))
    print(f"git {json.dumps(record['git'])}\ntorch {json.dumps(record['torch'])}\nargs {vars(args)}")

    section("1. Concerto checkpoints (Hugging Face, enable_flash=False)")
    models, configs = {}, {}
    for name in ("concerto", "concerto_prerelease"):
        models[name], configs[name] = load_encoder(name, seed=args.seed)
    encoders = describe(models, configs)

    section(f"2. Same backbone as {PRETRAIN}?")
    pretrain = dict(status="skipped") if args.skip_pretrain_check else same_backbone(args.tmp)
    print(f"  {json.dumps(pretrain)}")

    section("3. Released ScanNet-20 probe (concerto_large_linear_prob_head_sc) on both backbones (fp32, original attention)")
    room = load_demo_sample("sample1")
    gt = room["segment20"].astype(np.int64)
    head = load_scannet_probe("concerto")
    probe = {name: score(head, run(models, name, room, args.seed, backend="manual"), gt) for name in models}
    print(f"  {probe}")
    del models["concerto_prerelease"]
    demo = checks(models, head, room, gt, args.seed)

    section("5. Time and peak GPU memory per scene, one T4")
    rows, failed_at = [], {}
    for k in args.tiles:
        scene = tile_scene(room, k)
        for name, backend, amp in BENCH:
            if len(scene["coord"]) >= failed_at.get((name, backend, amp), float("inf")):
                continue
            row = dict(scene=f"demo_room_x{k * k}", **bench_one(models, name, backend, amp, scene, args.repeats))
            rows.append(row)
            print_bench([row])
            if row["status"] != "ok":
                failed_at[(name, backend, amp)] = row["n_input"]
    ok = [r for r in rows if r["status"] == "ok" and (r["attention"], r["precision"]) == ("sdpa", "fp16")]
    largest = max(ok, key=lambda r: r["n_grid"], default=None)
    per_point = largest and dict(kb=round(largest["peak_gb"] * 2**30 / largest["n_grid"] / 1e3, 1),
                                 us=round(largest["sec_gpu"] / largest["n_grid"] * 1e6, 2), at_n_grid=largest["n_grid"])

    summary = dict(args={k: str(v) for k, v in vars(args).items()}, git=record["git"], torch=record["torch"],
                   encoders=encoders, pretrain_check=pretrain, scannet_probe=probe, demo=demo, bench=rows, per_grid_point=per_point)
    (args.out / "step1b_summary.json").write_text(json.dumps(summary, indent=1, default=str))

    section("SUMMARY step 1b (paste this block back)")
    for name, e in encoders.items():
        print(f"{name:19s} params {e['params_m']}M, weights {e['weights_gb']} GB, enc_channels {e['enc_channels']}, "
              f"upcast width {e['upcast_width']}, head dim {e['head_dim']}")
    print(f"same backbone as {PRETRAIN}: {json.dumps(pretrain)}")
    print(f"ScanNet-20 probe mIoU on the demo room: {json.dumps({k: v['miou'] for k, v in probe.items()})}")
    print(f"SDPA vs manual: {json.dumps({k: (v['cos_p01'], v['pred_agreement']) for k, v in demo['sdpa_vs_manual'].items()})}"
          "  (cos 1st pct, pred agreement)")
    f = demo["fp16_vs_fp32"]
    print(f"fp16 vs fp32: cos 1st pct {f['cos_p01']}, agreement {f['pred_agreement']}, mIoU {f['miou_fp32']} / {f['miou_fp16']}")
    r = demo["randomness"]
    print(f"randomness: same-seed max|diff| {r['same_seed_max_abs']}, other-seed cos {r['other_seed']['cos_mean']}, "
          f"agreement {r['other_seed_pred_agreement']}")
    print(f"zeroed inputs mIoU: {json.dumps({k: v['miou'] for k, v in demo['zeroed'].items()})}; upcast {json.dumps(demo['upcast'])}")
    print_bench(rows)
    print(f"per grid point (largest fp16 SDPA run): {json.dumps(per_point)}")
    print(f"\nfull results: {args.out / 'step1b_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
