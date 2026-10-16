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

| Model | Protocol | Seeds | Val mIoU top-1 | Val mIoU top-3 |
|---|---|---|---|---|
| A: PTv3 | reduced_v1 | — | — | — |
| B: DITR (PTv3 + DINOv2-L) | reduced_v1 | — | — | — |

Reference (leaderboard, full protocol, **test** set, not comparable in absolute terms): PTv3 48.8, DITR 52.5 mIoU.

## Per-class IoU

Added in M6 (A vs B, with a focus on small and texture-defined classes such as socket and book).
