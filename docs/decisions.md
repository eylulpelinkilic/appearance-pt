# Decision log

Newest first. Each entry: what was decided, why, and what was verified (and how).
"Verified" means checked against code, docs or a run; "unverified" items are flagged.

---

## 2026-10-24 — Pilot (Concerto-style concatenation probe), step 1: point encoders

Spec: `experiments/pilot_concat/README.md`. Code: `experiments/pilot_concat/encoders.py`.

### P1. Utonia as a pinned submodule; HF weights pinned by commit
| Component | Pin | License |
|---|---|---|
| `third_party/utonia` (github.com/Pointcept/Utonia) | `f832deb77b9ea942e13dc9ef5a9e7e982b1d09ed` (2026-10-24, "Use SDPA in the non-flash attention branch") | Apache-2.0 (code) |
| HF `Pointcept/Utonia`: `utonia.pth` (549 MB), `utonia_linear_prob_head_sc.pth` | `9457f025e36c7001ee962a542f0d27a5b600e1e7` | CC-BY-NC-4.0 |
| HF `facebook/sonata`: `sonata.pth` (434 MB), `sonata_linear_prob_head_sc.pth` | `df99897472c09f91ba9288da0a034aacffc0b010` | not stated |

- Why the standalone package rather than Pointcept main's `PT-v3m3`: the HF model card calls `utonia.pth` "the simplified version for inference" for that repo, and at this commit its non-flash attention uses SDPA. Pointcept's `PT-v3m3` materialises the full attention matrix.
- Both packages' `load()` download `revision="main"`. `encoders.checkpoint()` downloads at the SHAs above and passes the local file to `load()`.
- Change to D2: the pilot uses Sonata weights as a frozen feature extractor (config S), not only for the smoke test. Utonia weights are non-commercial. Everything stays private.

### P2. What each encoder expects (verified in code at the pins above)
| | Utonia | Sonata |
|---|---|---|
| Load | `utonia.model.load` → checkpoint `{config, state_dict}` → `PointTransformerV3(**config)`, encoder only | `sonata.model.load`, same layout; HF `config.json`: `in_channels=9`, `enc_mode=True`, `traceable=True`, `shuffle_orders=True`, patch 1024 |
| Input feat | `Collect(feat_keys=("coord","color","normal"))`, 9 channels; `NormalizeColor` = colour/255 | same |
| Missing colour/normal | zeros (README, HF card); pretraining drops both to 0 (`RandomDropColor`/`RandomDropNormal`) | not in pretraining (`pretrain-sonata-v1m1-0-base.py` has no colour/normal drop), so zeroed anonymised colours are out of distribution |
| Transform | `transform.default(0.5)`: `RandomScale(0.5)` → `CenterShift` → `GridSample(0.01)` (2 cm in metres, coordinates halved) | `transform.default()`: `CenterShift` → `GridSample(0.02)`, metres |
| Upcast width | 54+108+216+432+576 = **1386** | 48+96+192+384+512 = **1232** |
| Head dim | 18 at every stage | 16 at every stage |
| Attention, `enable_flash=False` | SDPA, head dim zero-padded 18 → 24 for the memory-efficient kernel (`SerializedAttention.forward`) | explicit `softmax(QKᵀ)V` only; `encoders.sonata_sdpa_forward` swaps in SDPA (same maths) |

**Utonia's scale.** Indoor scale 0.5 comes from:
- `demo/2_sem_seg.py` (`transform.default(0.5)` with the released ScanNet probe);
- Pointcept's `semseg-utonia-v1m1-0a-scannet-lin.py` and `-3a-s3dis-lin.py` (`RandomScale([0.5, 0.5])` + grid 0.01);
- indoor pretraining (`indoor_transform` in `pretrain-utonia-v1m1-0-base_stagev2.py`: `RandomScale([0.45, 0.55])` + grid 0.01, used for ScanNet, ScanNet++, ARKitScenes, S3DIS and HM3D).

The odd one out is `semseg-utonia-v1m1-2a-scannetpp-lin.py`, which uses no rescale and grid 0.02. That file differs from the Sonata ScanNet++ config only in the backbone block, and Pointcept's Utonia README says the probing configs "have not been verified with one additional run". Both give the same 2 cm voxels: floor((0.5c − min 0.5c)/0.01) = floor((c − min c)/0.02). But the ScanNet++ config feeds coordinates 2× larger than pretraining, both into the xyz input channels and into RoPE (`Point3DRoPE`, base 10, applied to `point.coord`). **Decision: scale 0.5, grid 0.01.** Notebook check 2a scores all three variants with the released ScanNet probe.

**Per-point features** (`DefaultSegmentorV2.forward` in `pointcept/models/default.py`; both packages' `demo/2_sem_seg.py`):
- While `pooling_parent` exists: `parent.feat = cat(parent.feat, feat[pooling_inverse])`. This concatenates all five stages. The READMEs' `range(2)`-then-overwrite loop is for visualisation only.
- `point.inverse` (from `GridSample(return_inverse=True)`) then maps every input point to its voxel's row, so all points in one 2 cm voxel share a feature.
- `encode()` keeps voxel-level features plus that index rather than materialising N × 1386.

**Randomness at inference.**
- `GridSample(mode="train")` keeps a random point per voxel (`np.random`).
- `Point.serialization(shuffle_orders=True)` permutes the four orders (`torch.randperm`), in eval too.
- So `encode()` seeds numpy and torch per call. Check 2e measures how much features move across seeds.

**Data facts.**
- Points per ScanNet++ scene are the vertices of `mesh_aligned_0.05.ply`: Pointcept `preprocess_scannetpp.py` saves float32 coord, uint8 colour and open3d vertex normals. These are the same vertices the official evaluation scores.
- Pretraining saw ScanNet++ **val and test** scenes, unlabelled: Sonata (`ScanNetPPDataset` train/val/test chunks), Concerto and Utonia (`DefaultImagePointDataset` splits train/val/test).
- Utonia, like Concerto, was trained with a 2D–3D objective against **DINOv2 ViT-g/14 with registers at 518 px** (`image_weight_name="dinov2_vitg14_reg"`). Our DINO block is the same family. So C − B measures what DINO adds beyond what Utonia already learned to predict from it. S+ − S (Sonata had no image objective) is the closer analogue of Concerto's pilot.

### P3. Open until the step-1 notebook has run
- Per-scene time and peak memory on a T4, and the largest scene that fits (tiled demo room as a size proxy; `--scenes` for real ScanNet++ scenes later). Grid points per ScanNet++ scene: unknown until data exists.
- Whether SDPA and fp16 change features or predictions measurably (checks 2c, 2d).

---

## 2026-10-16 — M1: repository, environment strategy, smoke test

### D1. Third-party code as pinned, shallow submodules
| Submodule | Upstream | Pinned commit | Upstream date |
|---|---|---|---|
| `third_party/Pointcept` | github.com/Pointcept/Pointcept | `1342eda30e96cbb5fadf5374ff8eb18f6de15c71` | 2026-09-11 |
| `third_party/ditr` | github.com/RWTHVision/ditr | `f66d1dadb82e97ded7750567a043e4783305b01c` | 2026-05-18 |
| `third_party/scannetpp` | github.com/scannetpp/scannetpp | `21ba1dc724e0d09c6a1f72b0d63944f5363eb675` | 2026-07-10 |
| `third_party/sonata` | github.com/facebookresearch/sonata | `18c09ff8d713494f78a8213792262b910977a65d` | — |

- DITR moved from `VisualComputingInstitute/ditr` to `RWTHVision/ditr` (the GitHub API redirects; verified). DITR is a fork of Pointcept at `c4363c86` (its README and `3rd-party-licenses.txt`).
- Sonata was added because the M1 smoke test uses its inference package (not on PyPI).
- Shallow (`shallow = true` in `.gitmodules`) to keep clones small. The local disk had about 0.5 GB free.
- Third-party code is never edited in place. Our changes go in `src/` or as patch files under `patches/` (none so far).

### D2. Licenses (checked before copying anything into `src/`)
| Component | License | Source of the claim |
|---|---|---|
| Pointcept code | MIT | `third_party/Pointcept/LICENSE` |
| DITR code (whole repo, not only weights) | AGPL-3.0 | `third_party/ditr/LICENSE`, `pyproject.toml` |
| DITR weights (HF `rwthvision/ditr`) | AGPL-3.0 | HF model tags |
| ScanNet++ toolbox | **no license file** | repo root has no LICENSE; GitHub API reports none |
| ScanNet++ data | ScanNet++ Terms of Use: no redistribution | user's instructions |
| Sonata code | Apache-2.0 | `third_party/sonata/LICENSE` |
| Sonata weights (HF `facebook/sonata`) | not stated on the model card | HF API / model card |
| Utonia weights (HF `Pointcept/Utonia`) | CC-BY-NC-4.0 | HF model tags |
| DINOv2 ViT-L/14 reg4 (timm `vit_large_patch14_reg4_dinov2.lvd142m`) | Apache-2.0 | HF model tags |

Policy that follows from this:
- **Nothing from DITR is copied into `src/`.** We use DITR in place via the submodule. Any modification to DITR is a derivative work under AGPL-3.0. That's fine while the repo is private, but if it is ever released, those parts (and anything combined with them) must be AGPL-3.0. Clean-room code in `src/` written from the paper avoids this.
- The ScanNet++ toolbox has no license, so it is used only via the submodule and never copied.
- Sonata weights are used only for the M1 smoke test, never as a component of our models.
- Data, and anything derived from data (overlays, features, predictions, figures of scenes), never enters git (`.gitignore` plus `tests/test_repo_hygiene.py`). Processed data lives only in **private** Kaggle Datasets.

### D3. One shared Python environment on Kaggle's preinstalled PyTorch (pending the numbers from notebook 00)
Decision: a **single** environment (Kaggle GPU image plus one wheelhouse) serves Pointcept, DITR, Sonata and the ScanNet++ evaluation. No separate DITR environment unless the smoke test shows a real conflict.

Evidence:
- I traced the hard (unguarded) imports reachable from `tools/train.py` of both codebases with an AST walk. Pointcept main needs addict, einops, h5py, open3d, packaging, peft, pointops, regex, scipy, spconv, tensorboardX, termcolor, timm, torch_cluster, torch_geometric, torch_scatter, torchvision, transformers, wandb and yapf. DITR additionally needs SharedArray, albumentations and imageio. All of these fit in one environment.
- DITR's pins that conflict with T4 or with the image were relaxed. `torch==2.4.1+cu124` is replaced by the image's torch (DITR's README: "should probably also work with newer versions"). `flash-attn==2.6.3` is skipped because T4 is sm_75 and FlashAttention 2 needs sm_80; DITR guards that import (`try/except` in `point_transformer_v3m1_base.py`).
- **yapf conflict, resolved:** DITR's `Config.pretty_text` calls `FormatCode(..., verify=True)`, and yapf 0.40.2 removed `verify`, which is why DITR pins `yapf<0.40.2`. Pointcept main doesn't pass `verify`. `yapf==0.40.1` works for both.
- `pointops`: `libs/pointops/src` (the CUDA kernels) is byte-identical in Pointcept and DITR. Only the Python wrappers differ (Pointcept replaced deprecated `torch.cuda.IntTensor`). So **one wheel built from Pointcept main** serves both.
- Both codebases ship a top-level package named `pointcept`. They run in separate processes, each with its own repo directory as cwd and on `PYTHONPATH`, which is also how their `scripts/train.sh` works.
- ScanNet++ toolbox: `requirements.txt` pins `torch==1.13.1+cu116`, but `semantic/eval/eval_semantic.py` only needs numpy, torch (generic ops), munch, tqdm and pyyaml. It runs in the shared environment.

Why the preinstalled torch rather than a pinned one: it avoids shipping about 3 GB of torch and CUDA wheels per session, and Kaggle's torch is built for its own GPUs. Reproducibility comes from three things:
1. Pinning each notebook's Kaggle environment.
2. `install_env.py` refusing to install when the session's torch or Python differs from the wheelhouse's `MANIFEST.json`.
3. Every run recording its environment (`env_record.json`).

Fallback, if notebook 00 shows the image's torch lacks `sm_75` or breaks Pointcept: put pinned torch wheels (Pointcept's tested 2.5.x/cu124) in the wheelhouse.

What the Kaggle image looks like, from `Kaggle/docker-python` `Dockerfile.tmpl` on main (2026-10-16; **to be confirmed by notebook 00**): built on the Colab runtime `release-colab-external-images_20260917`, **Python 3.13**. Wheel availability checked on PyPI and data.pyg.org for that case: spconv 2.3.8 has cp313 wheels (cu118/cu121/cu124/cu126, none newer); PyG has torch_scatter/cluster wheels up to torch 2.14 (cu126/cu128/cu130); open3d 0.20.0 has cp313; SharedArray 3.2.4 is sdist-only and gets built.

### D4. Wheelhouse design (`scripts/setup/build_wheels.py`)
- Contents come from pip's own resolver (`pip install --dry-run --report`) with torch, torchvision, torchaudio, triton and numpy constrained to the image's versions. The wheelhouse therefore holds exactly what the image is missing, and nothing in the image is replaced.
- spconv variant: the newest of cu126/cu124/cu121/cu118 not newer than torch's CUDA; cu126 for CUDA 12.8 and 13.x. **Unverified:** spconv-cu126 on a CUDA 13 torch. The smoke test exercises it.
- pointops is compiled with `TORCH_CUDA_ARCH_LIST=7.5+PTX`, in a temp copy so the submodule stays clean.
- Sessions install with `pip install --no-index --no-deps -r requirements.lock.txt`: offline, exact, and with no resolver.

### D5. Smoke test = Sonata demo, not Utonia
- Sonata's README documents the flash-free path (`enable_flash=False`, `enc_patch_size=1024`), and its ScanNet linear-probe head gives a *numerical* check (accuracy/mIoU against the demo room's labels), not only a picture.
- Utonia's README says "Make sure FlashAttention is installed", and its weights are CC-BY-NC-4.0.
- The M1 smoke test also runs PTv3 training steps with the Pointcept ScanNet++ config, DITR image-injection training steps, and NCCL/DDP on 2 GPUs. M1 should derisk the whole stack, not only inference.

### D6. Secrets handling
- Tokens come only from `kaggle_secrets.UserSecretsClient`. The GitHub token reaches git only through `GIT_CONFIG_*` environment variables (an `http.extraheader`), never argv, so it can't appear in logs or `CalledProcessError` tracebacks. Submodules (public) are fetched without it. Covered by `tests/test_bootstrap.py`.
- Environment records use an allowlist of variables, because Kaggle sessions hold credentials in `KAGGLE_*` variables.
- `.gitignore` excludes `download_scannetpp*.yml` (the official download config embeds the token) and `*token*`.

### Observations for later milestones (facts checked in code; decisions deferred)
- **M2 / ScanNet++ token:** it expires **2026-10-25 (9 days from today)**, and an extension is possible from 2026-10-18. Meshes and semantics for the subset should be downloaded first.
- **M2:** the official download script (`download_scannetpp.py` plus yml) isn't in the toolbox repo. It comes from the ScanNet++ website with the token.
- **M3:** Pointcept's ScanNet++ PTv3 config uses batch 12, 800 epochs, `SphereCrop(point_max=204800)`, train split `train_grid1mm_chunk6x6_stride3x3`, `enable_flash=True`, `upcast_*=False`. With flash off, `upcast_attention/softmax=True` is the safer fp16 setting; the smoke test uses it. Pointcept's trainer supports `amp_dtype="float16"` with GradScaler (default `amp_dtype`). DITR's trainer calls `torch.autocast("cuda")` without a dtype, which defaults to fp16.
- **M3 option:** PyTorch SDPA's memory-efficient kernel supports sm_75. Replacing PTv3's manual non-flash attention with SDPA could cut memory a lot. It would be a patch to third-party code, so it's decided in M3 with numbers (notebook 00 reports which SDPA backends work).
- **M5:** DITR's repo has **no ScanNet++ image config or loader**: `configs/scannetpp` has only base PTv3/PTv2/SpUNet, and the `-image` configs exist for ScanNet, ScanNet200, S3DIS, nuScenes and others. We need our own loader. Candidate to reuse: Pointcept main's Concerto ScanNet++ image preprocessing (`pointcept/datasets/preprocessing/concerto/scannetpp/`), which extracts images and point-pixel correspondences.
- **M5:** DITR runs DINOv2 **online** (`FrozenDINOv2`, timm `vit_{size}_patch14_reg4_dinov2`, i.e. the *with-registers* variant; `requires_grad_(False)`, `eval()`) on 420×560 images (30×40 patches), with **image augmentation in training** (`ImageColorJitter`, `ImageRandomHorizontalFlip`). Caching features, as planned, removes these augmentations. That's a protocol deviation to decide in M5. A horizontal flip can't be emulated exactly by flipping cached patch maps, because ViT features aren't flip-equivariant.
- **M5:** `assign_image_feat` loops over views in order. **A point visible in several views gets the feature of the last view** (overwrite, no averaging). Visibility, occlusion and distance filtering live in `image_mask`, which the dataset builds. Our cached-feature version must reproduce this behaviour or document a deviation.
- **M5:** DITR wraps DINOv2 in `torch.compile(fullgraph=True)`. Triton/inductor support on sm_75 + Python 3.13 is tested in the smoke test, with a fallback run under `TORCHDYNAMO_DISABLE=1`. `FrozenDINOv2` isn't an `nn.Module` and calls `.cuda()` in `__init__`, so models are built on a GPU.

### Local machine (2026-10-16)
macOS 15 (Darwin 24.3, arm64), no NVIDIA GPU, so there is **no local GPU environment**. A local `.venv` (uv, Python 3.14) holds only pytest and numpy for CPU unit tests. Disk is 100% full (about 0.5 GB free), so nothing data-sized is ever put on this machine.

## 2026-10-21 — M1: first wheelhouse build on Kaggle

### D7. spconv/cumm: switch off compile-on-import with a `.pth` (`src/appearance_pt/sitefix.py`)
First `m1_01` run, image facts: Python 3.13.15, torch 2.11.0+cu128, nvcc 12.8, 2× T4 (14.56 GB each), torch arch list includes `sm_75`. All 22 wheels built and installed. pip picked spconv-cu126 2.3.8 + cumm-cu126 0.7.11, torch-scatter/cluster from PyG's `torch-2.11.0+cu128` page, and pointops compiled for 7.5+PTX in 3.5 min. Every import in the check passed **except `spconv.pytorch`**: on import, cumm started a ninja build of its own bindings and failed with `fatal error: tensorview/core/all.h: No such file or directory`. The compile flags show `-I /usr/local/lib/python3.13/dist-packages/include`.

Cause, verified in the sources of cumm v0.7.11 and spconv v2.3.8 and in pccm's `utils.py`:
- `pccm.utils.project_is_editable(name)` returns True when a `setup.py` or `.gitignore` exists in the **parent of the package dir**, i.e. in site-packages itself. Kaggle's dist-packages evidently has one. Which one is logged as `markers_in_site_dir` from the next run on.
- With that flag, `cumm/__init__.py` JIT-builds `core_cc`. The guard is `... and not CUMM_DISABLE_JIT`. `spconv/build.py` does the same, guarded by `SPCONV_DISABLE_JIT`.
- `cumm/constants.py` checks `<site-packages>/include` ("pip dev install") **before** `cumm/include` ("pip package", which the wheel ships). It then honours `CUMM_INCLUDE_PATH`. That path matters at runtime too: spconv's NVRTC fallback kernels compile against these headers.
- `spconv.constants.SPCONV_DEBUG_CPP_ONLY`/`EDITABLE_INSTALLED` also come from the misdetection, but nothing else in the spconv 2.3.8 package reads them.

Fix: after installing, write `appearance_pt_spconv.pth` next to cumm. It sets `CUMM_DISABLE_JIT=1`, `SPCONV_DISABLE_JIT=1` and `CUMM_INCLUDE_PATH=<cumm>/include` (with `setdefault`) at startup of every interpreter, which covers training runs, DDP workers and the import check. It's called from `build_wheels.py` step 6 and from `install_env.py`, and the values go into `env_record.json`. Rejected alternatives:
- Deleting the stray file from the image: it edits the image and is fragile.
- Setting the variables only in the notebook bootstrap: it would miss processes not started from it, and the include path is only known after installation.

Still to verify: the smoke test's spconv forward/backward on the GPU.
