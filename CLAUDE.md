# CLAUDE.md — context and working rules for agent sessions

## Project
Research on appearance-aware point cloud segmentation on ScanNet++ v2: how colour and image (ViT)
features improve Point Transformer models. SOTA on the ScanNet++ semseg leaderboard is DITR
("DINO in the Room", 3DV 2026): frozen DINOv2 features projected onto points and injected into
PTv3's decoder (52.5 test mIoU vs 48.8 for PTv3).

**Phase 1 = reproduction only, no new research ideas.** Under ONE fixed reduced protocol:
A = PTv3 on ScanNet++ v2, B = DITR (PTv3 + DINOv2 injection) on ScanNet++ v2.

## Platform: Kaggle notebooks (the agent cannot run code there)
- Up to 2× T4 16 GB (Turing, sm_75): no FlashAttention 2, no bf16 → `enable_flash=False`, smaller
  attention patches, fp16 AMP, DDP on both GPUs.
- ~30 GPU h/week, max 12 h/session, ~20 GB persistent `/kaggle/working`, larger temp disk
  (numbers: see notebook 00 output once the user pastes it).
- Workflow: the agent writes code + notebooks here → the user pushes to a private GitHub repo,
  runs on Kaggle, pastes logs back. Every Kaggle run = one clear entry point, readable logs.
- Notebooks are generated: edit scripts, then `python scripts/make_notebooks.py`
  (`tests/test_notebooks.py` fails on stale notebooks).
- Secrets `SCANNETPP_TOKEN`, `GITHUB_TOKEN` via `kaggle_secrets.UserSecretsClient` only. **Never write
  tokens into code, configs, logs, argv or commits.** (`scripts/setup/kaggle_bootstrap.py` shows how.)
- Compiled deps are built once into a wheelhouse → private Kaggle Dataset `appearance-pt-wheels`;
  sessions install offline (`scripts/setup/install_env.py`), refusing a mismatched image.
- Training checkpoints every ~30 min and resumes automatically; one run of the reduced protocol
  must fit in one session (≤ 11 h).
- The local Mac has no NVIDIA GPU and its disk is nearly full: never download data locally.

## Data (M2)
- Raw data does not fit at once (meshes + semantics alone: 132 GB). Chunked, resumable pipeline:
  per batch of scenes, generate a download yml for just those scenes/assets → official download
  script → Pointcept preprocessing (`pointcept/datasets/preprocessing/scannetpp/preprocess_scannetpp.py`)
  → compact outputs → delete raw. Progress in a manifest so interrupted sessions resume.
- Outputs as versioned PRIVATE Kaggle Datasets (ScanNet++ forbids redistribution); check Kaggle's
  dataset size limits first. Process 5 scenes first, measure size/scene, extrapolate, report.
- Subset: all 50 val scenes + fixed random 200 train scenes (fixed seed; list committed).
- **ScanNet++ token expires 2026-10-25** (extension possible from 2026-10-18): meshes + semantics for
  the subset first; iPhone data later, only for those scenes and only needed frames.
- DITR later: per scene decode a pool of ~20–30 iPhone frames + depth, DINOv2 ViT-L patch features
  in fp16; estimate total storage before committing (PCA to fewer channels is an option).
- Anonymization: anonymized regions are magenta (255,0,255) in meshes and images, masks provided.
  Colours there = missing (zero + validity flag channel); mask those pixels in images.
- Evaluation: official ScanNet++ toolbox, per vertex on `mesh_aligned_0.05.ply`, anonymized vertices
  excluded, top-1 and top-3 multilabel IoU, per-class IoU. Sanity-check with trivial predictions.

## Milestones — STOP after each and report to the user before continuing
- **M1 Environment**: inspect machine, repo skeleton, environments (one for Pointcept; separate DITR
  env only if pins conflict), spconv + pointops (no flash-attn), Sonata/Utonia smoke test.
- **M2 Data + evaluation** (above).
- **M3 Reduced protocol**: from Pointcept's ScanNet++ PTv3 config: `enable_flash=False`, smaller patch,
  fp16 AMP, batch 1–2, capped points per crop, 2 cm grid. Run a few hundred iters; report s/iter and
  peak memory; propose a schedule that fits the user's GPU-hour budget and explain the trade-off.
  The user decides; then freeze `configs/protocol_reduced.py`.
- **M4 Baseline A**: frozen protocol, seed 0 first; after review seeds 1–2. Log to `docs/results.md`.
- **M5 Baseline B**: check DITR's ScanNet++ image support (none in the repo, see decisions.md); write a
  loader in `src/` (frames, intrinsics/extrinsics, depth); iPhone + sensor depth first, DSLR later.
  Projection test that saves an overlay image for visual pose-convention checks. Cache frozen DINOv2
  ViT-L patch features fp16 (per-view patch maps + point→(view, patch) indices). DITR indoor settings:
  10 views/scene (random train, temporally equidistant eval), depth occlusion check, 1–4 m distance
  filter, injection into all decoder levels. Same frozen protocol, 2–3 seeds.
- **M6 Sanity check**: B should clearly beat A (leaderboard gap ≈ 3.7 mIoU), also per class (socket,
  book…). If not, debug projection first.

## Working rules
- **Ask the user before:** downloading > 50 GB, starting any run > 2 h, changing the frozen protocol.
- Every run records git commit, config, seed and environment (`env_record.json`).
- Results as mean ± std over seeds; never claim differences smaller than seed variance.
- When unsure about a fact (APIs, config names, dataset formats), check the actual code/docs and say
  what was verified. Don't guess.
- Keep `docs/decisions.md` updated (dated, with evidence).
- `third_party/` = pinned submodules, never edited in place; our changes go in `src/` or `patches/`.
  DITR is AGPL-3.0 (code and weights): never copy it into `src/`. The ScanNet++ toolbox has no license:
  submodule use only.
- No data or data-derived files (images, overlays, features, predictions) in git.
