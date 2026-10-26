# Results

Every training/evaluation run gets one row in **Runs**. Report aggregates as mean ± std over
seeds (**Aggregates**), and don't claim differences smaller than the seed-to-seed spread.
Metrics are on the ScanNet++ v2 semantic **val** split, scored by the official toolbox per vertex
on `mesh_aligned_0.05.ply` with anonymized vertices excluded (see `configs/protocol_reduced.py`).

## Runs

| Run ID | Date | Model | Config | Git commit | Seed | Protocol | Env (torch / wheelhouse) | Iters | GPU h | Val mIoU top-1 | Val mIoU top-3 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | no runs yet (M1) |

## Aggregates (mean ± std over seeds)

Pilot (frozen encoders, one linear probe each; `experiments/pilot_concat/README.md`). The supervised
PTv3 and DITR rows were removed when the project changed direction (decisions.md D14).

| Config | Features | Seeds | Val mIoU top-1 | Val mIoU top-3 |
|---|---|---|---|---|
| A | DINOv2 ViT-L/14, lifted | — | — | — |
| B | Utonia | — | — | — |
| C | Utonia ⊕ DINO | — | — | — |
| S | Sonata (paper: 38.9) | — | — | — |
| S+ | Sonata ⊕ DINO | — | — | — |
| K | Concerto (paper: 45.6) | — | — | — |
| K+ | Concerto ⊕ DINO | — | — | — |
| R | untrained PTv3 (alone / ⊕ DINO) | — | — | — |
| X | xyz, colour, normal | — | — | — |

Reference (leaderboard, full protocol, **test** set, not comparable in absolute terms): PTv3 48.8, DITR 52.5 mIoU.

## Per-class IoU

Added in M6 (A vs B, with a focus on small and texture-defined classes such as socket and book).
