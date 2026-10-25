"""Pilot step 1: frozen point encoders (Utonia, Sonata, untrained PTv3) on a Kaggle T4.

On Pointcept's public demo room (a ScanNet scan with ScanNet-20 labels) it checks
  1. what each checkpoint holds (stored config, parameters, upcast feature width),
  2. which input scale Utonia expects, scored with its released ScanNet linear probe,
  3. that SDPA attention gives the same features as the original explicit product,
  4. fp16 against fp32, run-to-run variation (same seed, other seed), zeroed colour or normals,
  5. colour scaled to [0, 1] (both packages) against [-1, 1] (Pointcept before Sonata's release),
then measures GPU time and peak memory per scene against scene size (the room tiled k x k;
real scenes with --scenes). Results: <out>/step1_summary.json and <out>/env_record.json.

Usage: python experiments/pilot_concat/step1_encoders.py [--out DIR] [--tiles 1 2 3] [--scenes DIR]
Needs PYTHONPATH=src:third_party/utonia:third_party/sonata and Internet (Hugging Face weights).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.smoke import load_demo_sample, semseg_metrics
from encoders import attention_backend, encode, load_encoder, load_scannet_probe, make_transform
from scenes import load_scene_dir, tile_scene

GB = 2**30
UTONIA_VARIANTS = {
    "default": "scale 0.5, grid 0.01 (2 cm): utonia.transform.default(0.5), the pilot's choice",
    "pointcept_scannetpp_lin": "scale 1, grid 0.02 (2 cm): Pointcept semseg-utonia-v1m1-2a-scannetpp-lin",
    "metric_1cm": "scale 1, grid 0.01 (1 cm): utonia.transform.default(1.0)",
}
# (encoder, attention, amp): the pilot's candidates first, then the original Sonata path and fp32
BENCH = (("utonia", "sdpa", True), ("sonata", "sdpa", True), ("sonata", "manual", True),
         ("utonia", "sdpa", False), ("sonata", "sdpa", False))


def run(models, name, scene, seed, amp=False, backend="sdpa", variant="default"):
    with attention_backend(backend):
        return encode(models[name], make_transform(name, variant), scene, seed=seed, amp=amp)


@torch.inference_mode()  # encode() returns inference tensors; the head must not record autograd
def predict(head, result) -> np.ndarray:
    feat, inverse, _ = result
    return head(feat.float()).argmax(-1)[inverse].cpu().numpy()


def score(head, result, gt) -> dict:
    m = semseg_metrics(predict(head, result), gt, num_classes=20)
    return dict(miou=round(m["miou"], 4), acc=round(m["acc"], 4), n_grid=result[2]["n_grid"])


def compare(a, b, chunk: int = 100_000) -> dict:
    """Agreement of two encode() results per input point: max |diff|, relative L2 error, cosine."""
    (fa, ia, _), (fb, ib, _) = a, b
    cos, max_abs, err, ref = [], 0.0, 0.0, 0.0
    for s in range(0, len(ia), chunk):
        x, y = fa[ia[s:s + chunk]].float(), fb[ib[s:s + chunk]].float()
        cos.append(F.cosine_similarity(x, y, dim=1))
        max_abs = max(max_abs, float((x - y).abs().max()))
        err += float(((x - y) ** 2).sum())
        ref += float((x**2).sum())
    cos = torch.cat(cos)
    return dict(max_abs=round(max_abs, 5), rel_l2=round((err / ref) ** 0.5, 6),
                cos_mean=round(float(cos.mean()), 5), cos_p01=round(float(cos.kthvalue(max(1, len(cos) // 100)).values), 5))


def describe(models, configs) -> dict:
    out = {}
    for name, model in models.items():
        cfg = configs[name]
        out[name] = dict(
            params_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            weights_gb=round(sum(p.numel() * p.element_size() for p in model.parameters()) / GB, 3),
            enc_channels=list(cfg["enc_channels"]), upcast_width=sum(cfg["enc_channels"]),
            head_dim=[c // h for c, h in zip(cfg["enc_channels"], cfg["enc_num_head"])],
        )
        print(f"{name:7s} {json.dumps(out[name])}")
        if name != "random":
            out[name]["config"] = cfg
            print(f"        stored config: {json.dumps(cfg, default=str)}")
    return out


def demo_checks(models, heads, room, gt, seed) -> dict:
    res = {}
    section("2a. Utonia input scale, scored with Utonia's released ScanNet-20 linear probe (fp32)")
    res["utonia_scale"] = {}
    for variant, what in UTONIA_VARIANTS.items():
        res["utonia_scale"][variant] = score(heads["utonia"], run(models, "utonia", room, seed, variant=variant), gt)
        print(f"  {variant:24s} {res['utonia_scale'][variant]}   [{what}]")

    section("2b. Sonata with its released ScanNet-20 linear probe (fp32, original attention)")
    res["sonata_probe"] = score(heads["sonata"], run(models, "sonata", room, seed, backend="manual"), gt)
    print(f"  {res['sonata_probe']}")

    section("2c. SDPA vs explicit softmax(QK^T)V attention (same seed, so same voxels and orders)")
    res["sdpa_vs_manual"] = {}
    for name in ("utonia", "sonata"):
        for amp in (False, True):
            a = run(models, name, room, seed, amp=amp, backend="manual")
            b = run(models, name, room, seed, amp=amp, backend="sdpa")
            agree = float((predict(heads[name], a) == predict(heads[name], b)).mean())
            key = f"{name}_{'fp16' if amp else 'fp32'}"
            res["sdpa_vs_manual"][key] = dict(**compare(a, b), pred_agreement=round(agree, 5))
            print(f"  {key:12s} {res['sdpa_vs_manual'][key]}")

    section("2d. fp16 autocast vs fp32 (SDPA, same seed)")
    res["fp16_vs_fp32"] = {}
    for name in ("utonia", "sonata"):
        a = run(models, name, room, seed, amp=False)
        b = run(models, name, room, seed, amp=True)
        agree = float((predict(heads[name], a) == predict(heads[name], b)).mean())
        res["fp16_vs_fp32"][name] = dict(**compare(a, b), pred_agreement=round(agree, 5),
                                         miou_fp32=score(heads[name], a, gt)["miou"], miou_fp16=score(heads[name], b, gt)["miou"])
        print(f"  {name:7s} {res['fp16_vs_fp32'][name]}")

    section("2e. Run-to-run variation (fp16, SDPA): GridSample keeps a random point per voxel, orders are shuffled")
    res["randomness"] = {}
    for name in ("utonia", "sonata"):
        a = run(models, name, room, seed, amp=True)
        b = run(models, name, room, seed, amp=True)
        c = run(models, name, room, seed + 1, amp=True)
        agree = float((predict(heads[name], a) == predict(heads[name], c)).mean())
        res["randomness"][name] = dict(same_seed_max_abs=compare(a, b)["max_abs"], other_seed=compare(a, c),
                                       other_seed_pred_agreement=round(agree, 5),
                                       miou_seeds=[score(heads[name], a, gt)["miou"], score(heads[name], c, gt)["miou"]])
        print(f"  {name:7s} {res['randomness'][name]}")

    section("2f. Zeroed colour or normals (anonymised ScanNet++ vertices will have zero colour; fp16, SDPA)")
    res["zeroed"] = {}
    for name in ("utonia", "sonata"):
        res["zeroed"][name] = {"none": score(heads[name], run(models, name, room, seed, amp=True), gt)}
        for key in ("color", "normal"):
            scene = {**room, key: np.zeros_like(room[key])}
            res["zeroed"][name][key] = score(heads[name], run(models, name, scene, seed, amp=True), gt)
        print(f"  {name:7s} {res['zeroed'][name]}")

    section("2g. Colour range: colour/255 in [0, 1] (both packages) vs colour/127.5 - 1 in [-1, 1] "
            "(Pointcept's NormalizeColor before 2025-03-20; fp16, SDPA)")
    # NormalizeColor divides by 255, so 2c - 255 comes out as c/127.5 - 1
    signed = {**room, "color": room["color"].astype(np.float32) * 2 - 255}
    res["color_range"] = {}
    for name in ("utonia", "sonata"):
        res["color_range"][name] = {"unit": res["zeroed"][name]["none"],
                                    "signed": score(heads[name], run(models, name, signed, seed, amp=True), gt)}
        print(f"  {name:7s} {res['color_range'][name]}")

    section("2h. Upcast width and finiteness, all three encoders (fp16, SDPA)")
    res["upcast"] = {}
    for name in models:
        feat = run(models, name, room, seed, amp=True)[0]
        res["upcast"][name] = dict(width=int(feat.shape[1]), finite=bool(torch.isfinite(feat).all()),
                                   std=round(float(feat.float().std()), 4))
        print(f"  {name:7s} {res['upcast'][name]}")
    return res


def bench_one(models, name, backend, amp, scene, repeats) -> dict:
    """First call (includes spconv kernel tuning) + `repeats` timed calls, seeds 0.. on one scene."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    runs, status = [], "ok"
    try:
        for seed in range(1 + repeats):
            feat, inverse, stats = run(models, name, scene, seed, amp=amp, backend=backend)
            del feat, inverse
            runs.append(stats)
    except RuntimeError as e:  # torch.cuda.OutOfMemoryError, or spconv/cumm allocation failures
        status = "OOM" if "out of memory" in str(e).lower() else f"error: {str(e).splitlines()[0][:120]}"
    torch.cuda.empty_cache()
    warm = runs[1:] or runs
    return dict(
        encoder=name, attention=backend, precision="fp16" if amp else "fp32", status=status,
        n_input=len(scene["coord"]), n_grid=runs[0]["n_grid"] if runs else None,
        sec_first=round(runs[0]["sec_gpu"], 2) if runs else None,
        sec_gpu=round(statistics.median(r["sec_gpu"] for r in warm), 2) if warm else None,
        sec_cpu_transform=round(statistics.median(r["sec_transform"] for r in warm), 2) if warm else None,
        # beyond the weights resident on the GPU; allocations outside torch's allocator are not counted
        peak_gb=round((torch.cuda.max_memory_allocated() - resident) / GB, 2),
    )


def print_bench(rows) -> None:
    print(f"{'scene':18s} {'encoder':7s} {'attn':6s} {'prec':4s} {'n_input':>9s} {'n_grid':>9s} "
          f"{'first_s':>7s} {'gpu_s':>6s} {'cpu_s':>6s} {'peak_GB':>7s}  status")
    for r in rows:
        print(f"{r['scene']:18s} {r['encoder']:7s} {r['attention']:6s} {r['precision']:4s} {r['n_input']:>9d} "
              f"{r['n_grid'] or 0:>9d} {r['sec_first'] or 0:>7.2f} {r['sec_gpu'] or 0:>6.2f} "
              f"{r['sec_cpu_transform'] or 0:>6.2f} {r['peak_gb']:>7.2f}  {r['status']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/pilot_concat/step1"))
    ap.add_argument("--tiles", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6], help="k for k x k copies of the room")
    ap.add_argument("--scenes", type=Path, help="directory of Pointcept-format scenes (coord/color/normal.npy)")
    ap.add_argument("--repeats", type=int, default=2, help="timed calls per scene after the first")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        print("!! no GPU visible: set Accelerator to GPU T4 x2")
        return 2

    record = collect()
    (args.out / "env_record.json").write_text(json.dumps(record, indent=2))
    print(f"git {json.dumps(record['git'])}\ntorch {json.dumps(record['torch'])}\nargs {vars(args)}")

    section("1. Encoders (Hugging Face checkpoints, enable_flash=False)")
    models, configs = {}, {}
    for name in ("utonia", "sonata", "random"):
        models[name], configs[name] = load_encoder(name, seed=args.seed)
    encoders = describe(models, configs)
    heads = {name: load_scannet_probe(name) for name in ("utonia", "sonata")}

    room = load_demo_sample("sample1")
    gt = room["segment20"].astype(np.int64)
    print(f"demo room: {len(gt)} points, {np.mean(gt >= 0):.1%} labelled")
    demo = demo_checks(models, heads, room, gt, args.seed)
    del models["random"], heads

    section("3. Time and peak GPU memory per scene, one T4")
    scenes = [(f"demo_room_x{k * k}", tile_scene(room, k)) for k in args.tiles]
    if args.scenes:
        scenes += [(p.name, load_scene_dir(p)) for p in sorted(args.scenes.iterdir()) if (p / "coord.npy").exists()]
    rows, failed_at = [], {}
    for scene_name, scene in scenes:
        for name, backend, amp in BENCH:
            if len(scene["coord"]) >= failed_at.get((name, backend, amp), float("inf")):
                continue  # failed on a smaller scene already
            row = dict(scene=scene_name, **bench_one(models, name, backend, amp, scene, args.repeats))
            rows.append(row)
            print_bench([row])
            if row["status"] != "ok":
                failed_at[(name, backend, amp)] = row["n_input"]

    summary = dict(args={k: str(v) for k, v in vars(args).items()}, git=record["git"], torch=record["torch"],
                   encoders=encoders, demo=demo, bench=rows)
    (args.out / "step1_summary.json").write_text(json.dumps(summary, indent=1, default=str))

    section("SUMMARY (paste this block back)")
    for name, e in encoders.items():
        print(f"{name:7s} params {e['params_m']}M, weights {e['weights_gb']} GB, enc_channels {e['enc_channels']}, "
              f"upcast width {e['upcast_width']}, head dim {e['head_dim']}")
    print(f"Utonia scale (ScanNet-20 probe mIoU): {json.dumps({k: v['miou'] for k, v in demo['utonia_scale'].items()})}")
    print(f"Sonata probe mIoU: {demo['sonata_probe']['miou']}")
    print(f"SDPA vs manual: {json.dumps({k: (v['cos_p01'], v['pred_agreement']) for k, v in demo['sdpa_vs_manual'].items()})}"
          "  (cos 1st pct, pred agreement)")
    print(f"fp16 vs fp32: {json.dumps({k: (v['cos_p01'], v['pred_agreement'], v['miou_fp32'], v['miou_fp16']) for k, v in demo['fp16_vs_fp32'].items()})}"
          "  (cos 1st pct, agreement, mIoU fp32, fp16)")
    print(f"randomness: {json.dumps({k: (v['same_seed_max_abs'], v['other_seed']['cos_mean'], v['other_seed_pred_agreement']) for k, v in demo['randomness'].items()})}"
          "  (same-seed max|diff|, other-seed cos mean, agreement)")
    print(f"zeroed inputs mIoU: {json.dumps({k: {m: s['miou'] for m, s in v.items()} for k, v in demo['zeroed'].items()})}")
    print(f"colour range mIoU: {json.dumps({k: {m: s['miou'] for m, s in v.items()} for k, v in demo['color_range'].items()})}"
          "  (unit = colour/255, signed = colour/127.5 - 1)")
    print_bench(rows)
    print(f"\nfull results: {args.out / 'step1_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
