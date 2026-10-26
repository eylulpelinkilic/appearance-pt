# CLAUDE.md — context and working rules for agent sessions

## Project (direction changed 2026-10-26, docs/decisions.md D14)
Self-supervised 2D→3D representation learning on ScanNet++ v2, in the direction of Concerto: better
general point cloud encoders learned from images without labels, evaluated mainly by linear and
decoder probing of frozen encoders. **Not** a supervised segmentation method: DITR ("DINO in the
Room", 3DV 2026) is related work only.

- Reference encoders: Sonata (3D-only SSL), Concerto (joint 2D-3D SSL), Utonia. Concerto paper,
  ScanNet++ val: linear probing Sonata 38.9, Concerto 45.6 mIoU; decoder probing Concerto 48.3.
  These models saw ScanNet++ val (and test) scenes, unlabelled, during pretraining.
- **Now: the pilot** (`experiments/pilot_concat/`): Concerto-style linear probes on frozen features,
  ending in a write-up. Key question: do lifted DINO features still add information on top of
  Concerto (K+ vs K), which was itself trained to predict DINOv2 features?
- **Next phase (planned, do NOT start without the user):** continued pretraining from Sonata weights
  on ScanNet++ under a fixed GPU budget, comparing Concerto's cross-modal objective (control) with
  new objectives (likely higher-resolution cross-modal prediction using ScanNet++'s DSLR images and
  dense laser scans). Pointcept contains Concerto's pretraining code and configs (decisions.md P4).

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
- Training checkpoints every ~30 min and resumes automatically; one training run must fit in one
  session (≤ 11 h).
- The local Mac has no NVIDIA GPU and its disk is nearly full: never download data locally.

## Data (M2)
- Raw data does not fit at once (meshes + semantics alone: 132 GB). Chunked, resumable pipeline:
  per batch of scenes, generate a download yml for just those scenes/assets → official download
  script → Pointcept preprocessing (`pointcept/datasets/preprocessing/scannetpp/preprocess_scannetpp.py`)
  → compact outputs → delete raw. Progress in a manifest so interrupted sessions resume.
- Outputs as versioned PRIVATE Kaggle Datasets (ScanNet++ forbids redistribution); check Kaggle's
  dataset size limits first. Process 5 scenes first, measure size/scene, extrapolate, report.
- Subset: all 50 val scenes + fixed random 200 train scenes (fixed seed; list committed).
- **ScanNet++ token (extended on 2026-10-25, new token) expires 2026-11-01**: meshes + semantics for
  the subset first; iPhone data later, only for those scenes and only needed frames.
- Kept per scene (private datasets `scannetpp-core-*`, `scannetpp-iphone-*`): Pointcept arrays, a 60-view
  DSLR pool with poses, val raw mesh + annotations, a 60-frame iPhone pool with depth and poses.
  Kaggle unpacks the tars into directories; read them through `appearance_pt.scannetpp_store`.
- Image features (pilot): DINOv2 ViT-L/14 (with registers) patch features lifted to points from ~30 DSLR views; estimate
  storage before any feature dump (PCA to fewer channels is an option).
- Anonymization: anonymized regions are magenta (255,0,255) in meshes and images, masks provided.
  Colours there = missing (zero + validity flag channel); mask those pixels in images.
- Evaluation: official ScanNet++ toolbox, per vertex on `mesh_aligned_0.05.ply`, anonymized vertices
  excluded, top-1 and top-3 multilabel IoU, per-class IoU. Sanity-check with trivial predictions.

## Milestones — STOP after each and report to the user before continuing
- **M1 Environment**: inspect machine, repo skeleton, one environment, spconv + pointops (no
  flash-attn), Sonata/Utonia smoke test.
- **M2 Data + evaluation** (above).
- **Pilot** (`experiments/pilot_concat/README.md`): frozen encoders + linear probes. Configs A (DINO),
  B (Utonia), C (Utonia ⊕ DINO, also PCA-equalised), S (Sonata), S+ (Sonata ⊕ DINO), K (Concerto),
  K+ (Concerto ⊕ DINO), R (random PTv3, alone and ⊕ DINO), X (xyz, colour, normal). Steps 1–5; stop
  after step 1, after the step-2 visual check, and report storage before the step-3 feature dump.
  Report per-class gains of K+ over K and the visible / never-seen split.
- **Pretraining phase**: planned after the pilot (see Project); not specified yet.
- **Postponed** (supervised; only if the user brings them back): M3 reduced training protocol and
  M4 PTv3 baseline. **Dropped:** the DITR reproduction (old M5–M6, baseline B).

## Working rules
- **Ask the user before:** downloading > 50 GB, starting any run > 2 h, changing a frozen protocol.
- Every run records git commit, config, seed and environment (`env_record.json`).
- Results as mean ± std over seeds; never claim differences smaller than seed variance.
- When unsure about a fact (APIs, config names, dataset formats), check the actual code/docs and say
  what was verified. Don't guess.
- Keep `docs/decisions.md` updated (dated, with evidence).
- `third_party/` = pinned submodules, never edited in place; our changes go in `src/` or `patches/`.
  DITR is AGPL-3.0 (code and weights): never copy it into `src/`. The ScanNet++ toolbox has no license:
  submodule use only. Utonia and Concerto weights are CC-BY-NC-4.0: private, non-commercial use only.
- No data or data-derived files (images, overlays, features, predictions) in git.
