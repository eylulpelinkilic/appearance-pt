# Decision log

Newest first. Each entry: what was decided, why, and what was verified (and how).
"Verified" means checked against code, docs or a run; "unverified" items are flagged.

---

## 2026-10-25 — M2 / pilot data: token extension, token-safe downloads, sizing first

### D8. Our own download client instead of running the official script
- **Token:** new (extended) token issued 2026-10-25, valid 7 days, so it **expires 2026-11-01** (user). It is stored only as the Kaggle secret `SCANNETPP_TOKEN`.
- **The official `download_scannetpp.py` (from the ScanNet++ website) leaks the token into logs:**
  - `print(f"Using token from config file: {cfg.token}")` (line 291);
  - with `verbose`, every URL is printed, and the URL embeds the token (`url_template.replace("TOKEN", cfg.token)`, line 263);
  - the 404 and HTTP-error boxes print the URL.

  It also blocks on `input()` for a "y" (line 306). So it is never run on Kaggle.
- **`appearance_pt.scannetpp_download`** reuses the official `download_scannetpp.yml` (URL template, `zipped_assets`, `exclude_assets`, `meta_files`). It resolves paths with the toolbox's `common/scene_release.py` (`ScannetppScene_Release`, loaded from the submodule), the same way the official script does: a zipped asset downloads as `<path>.zip`. It keeps the official retry policy (back off on 429/5xx). Errors replace the token with `***` and suppress exception chaining; `tests/test_scannetpp_download.py` checks this.
- **Licensing:** the official `.py` and `.yml` have no license, so neither goes into git (`.gitignore`: `download_scannetpp*.py`, `download_scannetpp*.yml`). Kaggle gets the `.yml` through the private dataset `scannetpp-download-config`. The local `.yml` holds only the placeholder `<YOUR_TOKEN_HERE>`, which was checked.
- **Path fix:** the `.gitignore` rule `data/` also matches `scripts/data/`, so entry points live in `scripts/scannetpp/`.

### D9. Measure sizes before downloading (notebook `m2_00_scannetpp_sizes`, CPU only)
- Downloads only `meta_files` (split and class lists).
- Fixes the subset: all `nvs_sem_val` scenes plus 200 `nvs_sem_train` scenes, `numpy.random.default_rng(0)`, sorted. It gets committed as `configs/scannetpp_subset.json`; the pilot and M2 share it.
- Sends a HEAD request for every candidate asset of those scenes: mesh + mask + segments + annotations, semantic mesh `.ply`, undistorted resized DSLR images + masks + `transforms_undistorted.json` + train/test lists, DSLR COLMAP, and iPhone video/mask/depth/poses/COLMAP.
- Verified in the toolbox: `semantic/eval/eval_semantic.py` reads `scan_mesh_mask_path` (`mesh_aligned_0.05_mask.txt`) to ignore anonymised vertices, so the mask is required.
- Unverified until data exists: the DSLR `is_bad` frame flag. No toolbox code reads it; it is expected in `transforms_undistorted.json`.

**Result (run 2026-10-25 14:54 UTC).** Splits: 856 `nvs_sem_train` scenes, 50 `nvs_sem_val`. All 2,500 files exist (0 missing, 0 unknown). Download sizes in GB (zips):

| Group | Train (200) | Val (50) | Mean per scene, train/val | Max per scene |
|---|---|---|---|---|
| mesh + mask + segments + annotations | 8.73 | 2.00 | 0.044 / 0.040 | 0.35 |
| semantic mesh `.ply` | 6.62 | 1.52 | 0.033 / 0.030 | 0.26 |
| DSLR undistorted (images, masks, transforms, lists) | 80.23 | 11.02 | 0.401 / 0.220 | 2.64 |
| DSLR COLMAP | 10.01 | 2.97 | 0.050 / 0.059 | 0.77 |
| iPhone (video, mask, depth, poses, COLMAP) | 190.50 | 39.70 | 0.952 / 0.794 | 3.98 |

- **Not needed:** the semantic mesh `.ply`. The evaluation ground truth comes from `prepare_training_data.py` (`ScannetPP_Release_Dataset`: mesh + `segments.json` + `segments_anno.json`) → `.pth` `vtx_labels` → `prepare_semantic_gt.py`, and Pointcept's preprocessing reads the same three files.
- **Probably not needed:** DSLR COLMAP, if `transforms_undistorted.json` carries poses, intrinsics and the bad-frame flag. To check on the first scenes.
- **Approved by the user (2026-10-25): the core ~102 GB download** (mesh + labels 10.7 GB, DSLR undistorted 91.3 GB) for the 250-scene subset, to start after a 5-scene check of download speed and kept size per scene. Batches are downloaded, reduced to what we keep, and the raw files deleted.
- **Pending:** the iPhone download (230.2 GB, main project M5 only). The user decides this week, before the token expires on 2026-11-01.
- **Storage:** private Kaggle datasets. Kaggle's private quota is reported as 100 GB or 200 GB per user, measured unzipped (secondary sources disagree; to be read off the dataset-creation page). Target: keep everything under 100 GB, split across datasets of at most ~20 GB, one per notebook output.
- **Subset file:** `configs/scannetpp_subset.json` is the notebook's output file (seed 0; 200 train + 50 val, sorted, no duplicates).
  - Two train IDs are not 10 hex characters: `7104910700000` and `717485935`. The notebook output has them exactly like that, and the code only reads, samples and writes strings, so they come from the server's `splits/nvs_sem_train.txt` itself. They look like scientific-notation corruption (`71049107e5` read as 7.1049107·10¹²).
  - (An earlier guess that Kaggle's log viewer did this was wrong.)
  - The size requests found every file for both scenes (0 missing). **Confirmed real by `m2_01`:** 948,191 and 4,301,110 vertices, labelled (42% and 90%), 391 and 2,157 DSLR images. They stay in the subset.

### D10. Trial download (`m2_01`, Kaggle CPU session, 2026-10-25 15:24 UTC; 5 scenes: 3 val + the two odd IDs)

| Scene | Download MB | MB/s | Vertices | Pointcept npy MB | DSLR images (mean MB) | `is_bad` |
|---|---|---|---|---|---|---|
| 09c1414f1b (val) | 1,057 | 17.0 (1 worker) | 3,383,834 | 132.0 | 2,391 (0.405) | 166 |
| 0d2ee665be (val) | 84 | 11.3 (1 worker) | 1,082,845 | 42.2 | 189 (0.289) | 98 |
| 13c3e046d7 (val) | 202 | 22.8 (4 workers) | 1,670,693 | 65.0 | 420 (0.375) | 2 |
| 7104910700000 (train) | 118 | 20.4 (4 workers) | 948,191 | 37.0 | 391 (0.241) | 82 |
| 717485935 (train) | 1,079 | 20.0 (4 workers) | 4,301,110 | 167.7 | 2,157 (0.448) | 443 |

- **Speed:** 14.1 MB/s mean with one request at a time, 21.1 MB/s with 4 in parallel. Core 102 GB ≈ 1.3 h of pure transfer at 4 workers; iPhone 230 GB ≈ 3.0 h.
- **DSLR:** 1752×1168 JPG (0.24–0.45 MB) with PNG masks (5 KB). `transforms_undistorted.json` holds `camera_model: PINHOLE`, `fl_x`, `fl_y`, `cx`, `cy`, `w`, `h`, `k1`–`k4` = 0, `has_mask`, `frames` (`file_path`, `is_bad`, `mask_path`, `transform_matrix`), `test_frames` (27) and `aabb_range`. **DSLR COLMAP is not needed.** `transform_matrix` is presumably nerfstudio's camera-to-world (OpenGL axes); the step-2 projection overlay checks the convention.
- **Anonymisation mask** (`mesh_aligned_0.05_mask.txt`): a list of vertex indices, empty in 4 of 5 scenes, 2,017 vertices in 09c1414f1b.
- **Pointcept npy:** ~39 bytes per vertex: coord f32 (12), color u8 (3), normal f32 (12), segment int16 ×3 (6), instance int16 ×3 (6). Labelled fraction 0.42–0.91.
- **Scene size:** 0.95–4.3M vertices. Grid points at 2 cm are not measured yet. Scenes above the ~1.4M-grid-point single-pass limit (P3) will need chunked encoding in step 2.

### D11. What we keep, and how the full download runs (user decisions 2026-10-25: 60 DSLR views; iPhone yes)
- **One uncompressed tar per scene** (`appearance_pt.scannetpp_reduce.pack_scene`). JPGs and npy barely compress, and per-scene tars keep the notebook output to about 100 files. (Kaggle may limit the number of output files; not verified.) Each tar holds:
  - Pointcept's npy files (unchanged Pointcept format, so its loaders work as-is);
  - `mesh_aligned_0.05_mask.txt`;
  - **val only:** the raw `mesh_aligned_0.05.ply`, `segments.json` and `segments_anno.json`, so the official ground truth (toolbox `prepare_training_data` → `prepare_semantic_gt`) can be regenerated exactly after the token expires;
  - the full `transforms_undistorted.json` and `train_test_lists.json` (poses of all frames, small);
  - a **pool of 60 DSLR views** with masks, plus `kept_views.json`.
- **View pool:** `frames` + `test_frames`, minus `is_bad`, ordered by file name (= capture order), then 60 picked evenly. The pilot's ~30 views per scene and any later re-selection (coverage, occlusion) come from this pool. The other images are deleted, so this choice is final.
- **Parts:** `val` (50 scenes), `train-a` (train[:100]), `train-b` (train[100:]). One CPU notebook run (`m2_02_scannetpp_core`) and one private dataset per part (`scannetpp-core-{val,train-a,train-b}`). Each output stays under ~20 GB; the script stops at 19 GB. Estimated kept size: 5-scene mean ≈ 100 MB per scene (Pointcept npy ≈ 39 B/vertex + 60 views × 0.39 MB), so ≈ 25–30 GB in total including the val raw files.
- **CPU sessions:** Kaggle's CPU image has its own CPU torch, so the GPU wheelhouse does not install. The only extra package is `open3d==0.20.0` from PyPI (the wheelhouse's version), installed online.
- **Val part done (2026-10-25 15:58 UTC, 15.5 min, CPU session):** 50/50 scenes; 13.0 GB downloaded, 8.36 GB kept (≈167 MB per scene including the raw val meshes). Vertices per scene: min 271,704, median 1,204,156, max 4,775,139. 3,086 `is_bad` frames in total; every scene has the full 60-view pool. Train parts expected ≈ 1 h and ≈ 10 GB kept each (no raw meshes).
- **Train parts done (2026-10-25, 16:33 and 16:32 UTC):**
  - train-a: 100/100 scenes; 44.6 GB downloaded, 8.31 GB kept; 48 min. Vertices min 217,160, median 1,152,117, max 7,717,931. 12,418 `is_bad` frames. **Smallest view pool: 31** (a scene with fewer than 60 good frames).
  - train-b: 100/100 scenes; 44.4 GB downloaded, 8.87 GB kept; 45 min. Vertices min 254,894, median 1,137,323, **max 13,710,288**. 12,094 `is_bad` frames. Every pool has 60 views.
  - **Core download complete: 250/250 scenes, 102.0 GB downloaded, 25.5 GB kept** across three notebook outputs, to be saved as the private datasets `scannetpp-core-{val,train-a,train-b}`.
  - For step 2: the largest scenes (up to 13.7M vertices) are far above the ~1.4M-grid-point single-pass encoder limit even after 2 cm sampling, so chunked encoding is required. Scenes with fewer than 60 good DSLR frames keep all of them; the pilot's ~30 views still fit.
- **iPhone (230 GB): yes, this week, reduced to a frame pool** (60 frames per scene + depth + poses). See D12.

### D12. iPhone trial (`m2_03`, CPU session, 2026-10-25 16:58 UTC; val scenes 09c1414f1b, 0d2ee665be, 13c3e046d7) and the batch
- **Videos:** `rgb.mkv` and `rgb_mask.mkv` are 1920×1440 at 60 fps, with 5,396–10,896 frames.
- **Depth:** `depth.bin` holds one record per video frame, **lz4 uint16 mm**, 256×192. In the 60 sampled frames every pixel is valid; median depth is 1.4–2.1 m.
- **Poses:** `pose_intrinsic_imu.json` has one entry per video frame (`frame_%06d`) with `timestamp`, `pose` (4×4), `aligned_pose` (4×4), `intrinsic` (3×3) and `imu`.
- **COLMAP:** registers every 10th frame (418–1,071 frames) with an **`OPENCV`** camera model 1920×1440, so the iPhone images still have lens distortion and M5's projection must apply it. Files: `cameras.txt`, `images.txt`, `points3D.txt`.
- **Download:** 27–28 MB/s with 4 parallel requests.
- **Pool extraction:** ffmpeg `select` filter on exact frame numbers, as in the toolbox's `extract_rgb_in_colmap`; it decodes the whole video. Time: RGB 39–150 s and mask 12–25 s per scene; depth 1–2 s.
- **Kept per scene (60-frame pool):** 24–45 MB uncompressed. JPG 0.15–0.27 MB, mask PNG 6 KB, depth PNG 25 KB; the rest is the full pose JSON (13–26 MB) and COLMAP text. In the batch these are gzipped and `points3D.txt` is dropped, so the estimate is ≈ 6–9 GB for 250 scenes.
- **Batch (`m2_04`, `scannetpp_iphone_fetch.py`):**
  - The pool is 60 frames spread evenly over the COLMAP-registered frames (`select_pool`).
  - The two videos decode in parallel, and the next scene downloads while the current one decodes.
  - Five parts of 50 scenes (`val`, `train-1` … `train-4`) run as parallel CPU sessions, each estimated at 1.5–3 h, each saved to a private dataset `scannetpp-iphone-*`.
- **Fix found by a unit test:** for per-frame `depth.bin` the "whole file is one zlib block" attempt can inflate garbage and fail at the reshape (`ValueError`) rather than with `zlib.error`. The toolbox falls back on any exception, and so does `read_depth` now.

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

### P3. Step-1 notebook results (Kaggle T4, 2026-10-25 12:36 UTC, run of `pilot_01_encoders`)
Demo room: Pointcept `sample1`, a ScanNet scan with 273,530 points, 187,979 grid points at 2 cm. Scored with the released ScanNet-20 probes; one room, so treat mIoU as a sanity check, not a benchmark.
- **Utonia scale:**
  - mIoU 0.780 for `default(0.5)` (scale 0.5, grid 0.01);
  - 0.773 for Pointcept's ScanNet++ config (scale 1, grid 0.02);
  - 0.517 for `default(1.0)` (scale 1, 1 cm grid).

  **Decision confirmed: scale 0.5, grid 0.01.** The 0.7-point gap to the ScanNet++-config variant comes from one room and one seed, so it isn't meaningful on its own. The choice rests on P2's evidence; the run only shows nothing contradicts it.
- **SDPA vs explicit attention:** 1st-percentile cosine 1.0 in fp32 and fp16 for both encoders; prediction agreement 1.0 (fp32), 0.9998 (Utonia fp16) and 0.9999 (Sonata fp16). **Decision: SDPA for both.** Sonata's explicit product runs out of memory already at 746k grid points.
- **fp16 vs fp32:** 1st-percentile cosine 1.0; agreement 0.9998; mIoU 0.7803 vs 0.7803 (Utonia) and 0.4912 vs 0.4913 (Sonata). **Decision: fp16 autocast, fp16 features.**
- **Randomness:** the same seed gives bitwise-identical features (max |diff| 0). Another seed (different voxel representatives and order shuffle) gives mean cosine 0.995 (Utonia) and 0.992 (Sonata), with 3.0% and 3.5% of point predictions flipping. **Decision: one seeded pass per scene, seed derived from the scene id.** Pointcept's multi-pass/TTA testing is not reproduced; this is a deviation to list in the write-up.
- **Zeroed inputs (mIoU):**
  - Utonia: 0.780 normally, 0.714 with colour zeroed, 0.777 with normals zeroed.
  - Sonata: 0.491 normally, **0.535** with colour zeroed, **0.096** with normals zeroed.
  - Utonia is robust to missing colour, which matters for anonymised vertices. Sonata collapses without normals, consistent with no normal dropout in its pretraining.
- **Sonata, open issue:** its released probe scores 0.491 on this room against Utonia's 0.780, and zeroing its colour *raises* the score.
  - Our call matches `third_party/sonata/demo/2_sem_seg.py` exactly (load, `transform.default()`, upcast loop, head).
  - Pointcept switched `NormalizeColor` from `/127.5 − 1` to `/255` in the Sonata release commit `10b749039c` (2025-03-20), and the Sonata package has used `/255` since its first commit. So a colour-convention mismatch is possible but not established.
  - **Tested (rerun 2026-10-25 12:56 UTC, check 2g):** with colours fed as `/127.5 − 1`, Sonata scores 0.487 against 0.491 with `/255`. Utonia, the control, drops from 0.780 to 0.700, so the test is sensitive. **The colour range is not the cause.** Sonata's predictions barely depend on colour on this room (0.491 with `/255`, 0.487 with `/127.5 − 1`, 0.535 with colour zeroed) and depend heavily on normals. **Decision: keep the package default `/255` for Sonata.**
  - The low score stays unexplained. The pilot's sanity check (S ≈ 38.9 mIoU on ScanNet++ val) tests it at scale.
  - The rerun reproduced every other step-1 number exactly (fixed seeds); GPU times varied by 5–10%.
- **Cost per scene, one T4, upcast features materialised.** Peak memory is GPU memory beyond resident weights (weights: Utonia 0.51 GB, Sonata 0.40 GB):

  | grid points | Utonia fp16 | Sonata fp16 (SDPA) | Sonata fp16 (explicit) |
  |---|---|---|---|
  | 188k | 0.75 s, 1.29 GB | 0.49 s, 1.15 GB | 0.82 s, 3.42 GB |
  | 746k | 3.05 s, 5.10 GB | 1.97 s, 4.57 GB | OOM |
  | 1.65M | 7.34 s, 11.33 GB; a repeat ran out of memory | 4.86 s, 10.14 GB; a repeat ran out of memory | n/a |

  - Scaling: about 7 KB and 4.4 µs per grid point for Utonia; 6 KB and 2.9 µs for Sonata. fp32 roughly doubles memory.
  - CPU transform: about 0.4 µs per input point.
  - **Safe single-pass limit ≈ 1.4M grid points per scene.** Larger ScanNet++ scenes will need chunking, or features kept per stage instead of concatenated; to be decided once real scene sizes are known (first scenes in step 2).

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

Rerun with the fix (commit `637b952`, started 2026-10-20 22:18 UTC; log pasted 2026-10-25):
- `markers_in_site_dir = [".gitignore", "include"]`. Kaggle's dist-packages has both stray markers, which confirms the diagnosis above.
- The `.pth` was written to `/usr/local/lib/python3.13/dist-packages/appearance_pt_spconv.pth`.
- All 24 modules in the import check pass, including `spconv.pytorch`.
- 22 wheels, 506 MB; the whole build took 6.7 min.
- `pip check` reports only conflicts among the image's preinstalled packages (google-colab, bigframes, dopamine-rl, moviepy), none involving the wheelhouse.

Still to verify: the smoke test's spconv forward/backward on the GPU.
