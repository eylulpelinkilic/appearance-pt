# Results

Every training/evaluation run gets one row in **Runs**. Report aggregates as mean ± std over
seeds (**Aggregates**), and don't claim differences smaller than the seed-to-seed spread.
Metrics are on the ScanNet++ v2 semantic **val** split, scored by the official toolbox per vertex
on `mesh_aligned_0.05.ply` with anonymized vertices excluded (see `configs/protocol_reduced.py`).

## Runs

| Run ID | Date | Model | Config | Git commit | Seed | Protocol | Env (torch / wheelhouse) | Iters | GPU h | Val mIoU top-1 | Val mIoU top-3 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot_04 | 2026-10-26 | linear probes on frozen features (12 configs) | `experiments/pilot_concat` | in results.json | 0–2 | pilot (P7) | torch 2.11.0+cu128 / appearance-pt-wheels | 15 epochs | ≈ 0.75 (eval) | see Aggregates | see Aggregates | probes from pilot_03 (steps 3–4) |

## Aggregates (mean ± std over seeds)

Pilot (frozen encoders, one linear probe each; `experiments/pilot_concat/README.md`). The supervised
PTv3 and DITR rows were removed when the project changed direction (decisions.md D14).

| Config | Features | Seeds | Val mIoU top-1 | Val mIoU top-3 |
|---|---|---|---|---|
| A | DINOv2 ViT-L/14 with registers, lifted | 3 | 33.89 ± 0.12 | 62.51 ± 0.13 |
| B | Utonia | 3 | 37.64 ± 0.03 | 64.68 ± 0.13 |
| C | Utonia ⊕ DINO | 3 | 41.47 ± 0.07 | 68.89 ± 0.05 |
| C-eq | Utonia ⊕ DINO, 512 + 512 dims | 3 | 40.86 ± 0.03 | 68.08 ± 0.03 |
| S | Sonata (paper: 38.9) | 3 | 30.83 ± 0.05 | 54.91 ± 0.04 |
| S+ | Sonata ⊕ DINO | 3 | 40.73 ± 0.02 | 69.07 ± 0.05 |
| K | Concerto (paper: 45.6) | 3 | 39.09 ± 0.03 | 66.20 ± 0.04 |
| K+ | Concerto ⊕ DINO | 3 | 41.76 ± 0.05 | 69.20 ± 0.04 |
| K+-eq | Concerto ⊕ DINO, 512 + 512 dims | 3 | 41.42 ± 0.05 | 69.50 ± 0.08 |
| R | untrained PTv3 | 3 | 6.97 ± 0.00 | 15.38 ± 0.00 |
| R+ | untrained PTv3 ⊕ DINO | 3 | 34.07 ± 0.02 | 60.01 ± 0.04 |
| X | xyz, colour, normal | 3 | 2.63 ± 0.00 | 7.91 ± 0.00 |

Seeds are probe seeds only (head initialisation and batch order; the features are fixed), so the ± understates
the uncertainty. Differences between configs are judged by a paired bootstrap over the 50 val scenes
(decisions.md P7 results); e.g. K+ − K = +2.67 [+1.22, +4.04]. Probing is lighter than the paper's (sampled
points, 15 epochs, no augmentation, 200 train scenes), so absolute numbers sit 6–8 points below it.

Reference (leaderboard, full protocol, **test** set, not comparable in absolute terms): PTv3 48.8, DITR 52.5 mIoU.

## Per-class IoU

Added in M6 (A vs B, with a focus on small and texture-defined classes such as socket and book).
