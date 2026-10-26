"""Pilot step 4: one linear probe per configuration (probes.CONFIGS) on the step-3 training features.

  1. Learning rate per configuration: every configuration at each of --lrs, trained on the train scenes minus
     --holdout scenes (seed 0, --sweep-epochs) and scored on the held-out scenes (point-level mIoU, top-100 classes);
     the best rate is kept.
  2. Final probes: every configuration at its rate, seeds 0..--seeds-1, on all train scenes (--epochs).
Configurations train in phases (probes.PHASES: one encoder block with DINO, then the raw inputs), each phase's
blocks held in memory (on the GPU when they fit); the probes of a phase train together on the same batches
(cross-entropy, AdamW, cosine decay, features standardised per block; "-eq" configurations first project each block
onto its leading EQ_WIDTH principal directions). The seeds
change the head's initialisation and the batch order only; the features are fixed. Each final probe is folded into
weights on the raw blocks (Probe.fold) and saved to <out>/probes.pt for step 5. Summary: <out>/step4_summary.json.

Usage: python experiments/pilot_concat/step4_probes.py [--features /tmp/pilot_features] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from probes import (CONFIGS, EQ_WIDTH, PHASES, FeatureStore, Probe, block_stats, evaluate_probes, fit_pca, phase_of, place,
                    train_probes)


def run_phase(store, scene_ids, blocks, configs, seeds_or_lrs, lr_of, epochs, args, device, eval_ids=None):
    """Load one phase's blocks, fit statistics (and PCA for -eq configs) on them, train the probes of its configs
    (one per configuration and seed or learning rate) together; returns the probes, their statistics, the loss
    curve and, with eval_ids, the held-out mIoU."""
    data, label = place(*store.read(scene_ids, blocks), device)
    stats = block_stats(data)
    proj = {b: fit_pca(data[b], stats[b], device=device) for b in blocks if any(CONFIGS[c][1] for c in configs) and b != "raw"}
    probes = {}
    for v in seeds_or_lrs:
        torch.manual_seed(v if isinstance(v, int) else 0)
        for c in configs:
            probes[f"{c}#{v}" if isinstance(v, int) else f"{c}@{v:g}"] = Probe(CONFIGS[c][0], store.widths,
                                                                               proj if CONFIGS[c][1] else None)
    curve = train_probes(data, label, probes, {k: lr_of(k) for k in probes}, stats, epochs, batch=args.batch,
                         seed=1, device=device)
    held = None
    if eval_ids is not None:
        del data, label
        torch.cuda.empty_cache()
        data, label = place(*store.read(eval_ids, blocks), device)
        held = evaluate_probes(data, label, probes, stats, device=device)
    del data, label
    torch.cuda.empty_cache()
    return probes, stats, curve, held


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("/tmp/pilot_features"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/pilot_concat/step4"))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--lrs", type=float, nargs="*", default=[3e-4, 1e-3, 3e-3, 1e-2])
    ap.add_argument("--holdout", type=int, default=20, help="train scenes held out for the learning-rate choice")
    ap.add_argument("--sweep-epochs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    record = collect()
    (args.out / "env_record.json").write_text(json.dumps(record, indent=2))
    store = FeatureStore(args.features)
    n_scenes = len(store.scenes)
    print(f"git {json.dumps(record['git'])}\nargs {vars(args)}\nstore: {store.meta['n']:,} rows, {n_scenes} scenes, "
          f"widths {store.widths}; occlusion {store.meta.get('occlusion')}, concerto {store.meta.get('concerto')}")
    t_start = time.time()

    section(f"1. Learning rate per configuration ({args.holdout} held-out scenes, seed 0, {args.sweep_epochs} epochs)")
    order = np.random.default_rng(0).permutation(n_scenes)
    held, train = sorted(order[:args.holdout].tolist()), sorted(order[args.holdout:].tolist())
    phases = [(ph, [c for c in args.configs if phase_of(c) == ph]) for ph in PHASES]
    phases = [(ph, cs) for ph, cs in phases if cs]
    held_miou, sweep_curve = {}, {}
    for ph, cs in phases:
        print(f"  phase {'+'.join(ph)}: {cs} x lr {args.lrs}")
        _, _, sweep_curve["+".join(ph)], h = run_phase(store, train, ph, cs, args.lrs, lambda k: float(k.split("@")[1]),
                                                        args.sweep_epochs, args, device, eval_ids=held)
        held_miou.update(h)
    best = {c: max(args.lrs, key=lambda lr: held_miou[f"{c}@{lr:g}"]) for c in args.configs}
    for c in args.configs:
        print(f"  {c:6s} held-out mIoU by lr {json.dumps({f'{lr:g}': held_miou[f'{c}@{lr:g}'] for lr in args.lrs})} -> {best[c]:g}")

    section(f"2. Final probes (all {n_scenes} scenes, seeds 0-{args.seeds - 1}, {args.epochs} epochs)")
    folded, curve = {}, {}
    for ph, cs in phases:
        print(f"  phase {'+'.join(ph)}: {cs} x seeds {list(range(args.seeds))}")
        probes, stats, curve["+".join(ph)], _ = run_phase(store, list(range(n_scenes)), ph, cs, list(range(args.seeds)),
                                                          lambda k: best[k.split("#")[0]], args.epochs, args, device)
        folded.update({k: p.fold(stats) for k, p in probes.items()})
    torch.save(dict(probes=folded, configs={c: CONFIGS[c] for c in args.configs}, lr=best, eq_width=EQ_WIDTH,
                    features=store.meta | {"scenes": [s["id"] for s in store.scenes]}, git=record["git"]),
               args.out / "probes.pt")

    summary = dict(args={k: str(v) for k, v in vars(args).items()}, git=record["git"], rows=store.meta["n"], scenes=n_scenes,
                   held_out=[store.scenes[i]["id"] for i in held], held_out_miou=held_miou, lr=best,
                   sweep_loss=sweep_curve, final_loss=curve, minutes=round((time.time() - t_start) / 60, 1))
    (args.out / "step4_summary.json").write_text(json.dumps(summary, indent=1))

    section("SUMMARY step 4 (paste this block back)")
    print(f"{store.meta['n']:,} training points from {n_scenes} scenes; {len(folded)} probes saved to {args.out / 'probes.pt'}")
    print("held-out point mIoU at the chosen rate (seed 0, trained without the held-out scenes):")
    for c in args.configs:
        print(f"  {c:6s} {held_miou[f'{c}@{best[c]:g}']:.4f}  (lr {best[c]:g})")
    print(f"loss per probe by epoch: sweep {sweep_curve}; final {curve}")
    print(f"total {summary['minutes']} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
