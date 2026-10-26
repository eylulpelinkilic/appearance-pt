# Decision log

Newest first. Each entry: what was decided, why, and what was verified (and how).
"Verified" means checked against code, docs or a run; "unverified" items are flagged.

---

## 2026-10-26 — Direction change; pilot: Concerto (K, K+), step 2 design; dataset check results

### D14. New direction: self-supervised 2D→3D representation learning (user decision, 2026-10-26)
- **Goal now:** better general point cloud encoders learned from images, without labels, in the direction of Concerto. They are evaluated mainly by linear and decoder probing of frozen encoders on ScanNet++. The project is no longer a supervised, DITR-style segmentation method.
- **Dropped:** the DITR reproduction (old M5–M6, baseline B). DITR is related work only. `third_party/ditr` stays pinned for reference; no code uses it.
- **Postponed:** supervised PTv3 training (old M4), together with the reduced training protocol it needed (old M3, `configs/protocol_reduced.py` draft).
- **Reference encoders:** Sonata (3D-only SSL), Concerto (joint 2D-3D SSL) and Utonia. From the Concerto paper, as given by the user, on ScanNet++ val:
  - linear probing: Sonata 38.9, Concerto 45.6 mIoU;
  - decoder probing: Concerto 48.3.

  All three saw ScanNet++ val and test scenes, unlabelled, during pretraining (P2; P4 for Concerto's dataset config).
- **Pilot additions:** K (Concerto) and K+ (Concerto ⊕ DINO). The key question is whether lifted DINO features still add information on top of Concerto, which was trained to predict DINOv2 features. The report gives per-class gains of K+ over K and the visible/never-seen split.
- **Planned next phase (not started):** continued pretraining from Sonata weights on ScanNet++ under a fixed GPU budget. Concerto's cross-modal objective is the control; the new objectives are likely higher-resolution cross-modal prediction from ScanNet++'s DSLR images and dense laser scans.
- **Carried over unchanged:** the data (M2: the same 250 scenes, DSLR and iPhone pools; the iPhone frames and depth may serve pretraining), the official evaluation and the pilot's steps.

### D15. Before the token expires: train-scene triangles and a size survey (user approved, 2026-10-26)
- **Why:** the point z-buffer reaches only 0.78–0.91 precision against mesh visibility (P6 results), because the decimated mesh leaves flat surfaces with few vertices. Exact visibility, and depth rendering for DSLR views (likely needed by the next phase), need the triangles. Val scenes kept their raw mesh; train scenes did not (D11).
- **What `m2_06_scannetpp_faces` does** (CPU, `scripts/scannetpp/scannetpp_faces.py`):
  - downloads `mesh_aligned_0.05.ply` of the 200 train scenes (part of D9's 8.73 GB mesh + labels group);
  - keeps `faces/<scene>.npz` per scene: triangles (int32), the vertex count, and the SHA-1 of the float32 vertex coordinates, i.e. of Pointcept's `coord.npy`. With the core train datasets attached, the hash is checked against `coord.npy`, so the triangles index the same vertices in the same order. Val showed they are identical (P6 results);
  - output becomes the private dataset `scannetpp-train-faces`.
- **Size survey for the next phase** (HEAD requests only, 250 subset scenes, extrapolated to the 856 + 50 scenes): dense laser scans (`pc_aligned.ply`, its mask, scanner poses), full-resolution DSLR images and masks, and the panorama (panocam) assets. Whether to download any of them is the user's decision, and must come before 2026-11-01.

### D13 (results). Dataset check `m2_05`, 2026-10-25 23:24 UTC, CPU, 4 processes
- **Mount paths.** Kaggle now mounts datasets at `/kaggle/input/datasets/<user>/<slug>/`, not `/kaggle/input/<slug>/`. `find_scenes` searches recursively, so nothing had to change.
- **Core: 250/250 scenes OK** (val 50/50, train 200/200). Every file was read and decoded, and the counts match the manifests. 25.5 GB were read in 3.3 min.
  - Vertices per scene: 217,160 / 1,152,117 / 13,710,288 (min / median / max).
  - **Grid points at 2 cm:** 46,705 / 257,693 / 3,313,384. **6 scenes exceed the 1.4M single-pass limit** (P3), so the encoders need chunking only for those.
  - Labelled fraction: min 0.19, median 0.77.
  - Anonymised vertices: 107,343 mask entries in 44 scenes, all of them magenta. Magenta vertices in total: 107,305, so the mask lists repeat some indices and no magenta vertex lies outside a mask.
  - DSLR: 14,971 views (31–60 per scene) with as many masks. All images are 1168×1752 and match `transforms_undistorted.json`; camera model PINHOLE.
- **iPhone: 0/250 OK, every scene "missing `iphone/pose_intrinsic_imu.json.gz`".**
  - The error names the archive `09c1414f1b`, not `09c1414f1b.tar`. So Kaggle unpacked each tar into a directory when it built the dataset.
  - In every scene, `pool.json` and the per-frame RGB, mask and depth counts were present and consistent. Only the gzipped members were "missing", so Kaggle most likely decompressed them too.
  - **Fix:** `SceneArchive.unzipped` reads either form, and the summary reports which one it found. A test covers an unpacked, decompressed copy.
  - **Re-run (2026-10-26 09:10 UTC, 12.3 min, 40.1 GB read): ALL SCENES OK, iPhone 250/250.** Kaggle had decompressed the `.gz` members in every scene; none was still compressed. That confirms the repackaging explanation.
    - 15,000 frames (60 per scene); RGB 1440×1920; COLMAP camera `OPENCV` in every scene.
    - Depth: per-frame lz4 uint16 mm in 240 scenes and per-frame zlib float32 m in 10 (all train). Worst frame's zero-depth fraction 0.000; median depth per scene 0.81–2.87 m (median 1.51 m).
    - iPhone mask pixels > 127: 0.975–1.000 of each frame. This fits 255 meaning "usable" (the nerfstudio convention the pilot also assumes for DSLR masks; step 2 reports the DSLR fraction).
  - **M2 data is complete and checked.** Nothing more needs the ScanNet++ token, which expires 2026-11-01: the metadata (class lists, label map, splits) is inside the core datasets.

### P4. Concerto (configs K, K+): verified in code; T4 numbers pending (`step1b_concerto.py`)
- **Code:** `third_party/concerto` = github.com/Pointcept/Concerto @ `5b41117160cdd92c3538be1911277b17e7ef5871` (2026-08-18), Apache-2.0. It is Sonata's package with the model list changed:
  - `transform.py` is byte-identical to Sonata's;
  - `SerializedAttention` is identical (diffed), with only the explicit softmax(QKᵀ)V when FlashAttention is off. `encoders.sonata_sdpa_forward` is swapped in for both packages (`attention_backend`).
- **Loading:** `concerto.model.load(path, custom_config=dict(enable_flash=False), ckpt_only=True)` gives `{config, state_dict}`, then `PointTransformerV3(**config)`, an encoder-only PT-v3m2. This is the same layout as Sonata, so the same upcast loop applies.
- **Default transform** (`concerto.transform.default()`): `CenterShift(apply_z)` → `GridSample(0.02, mode="train", fnv)` → `NormalizeColor` (/255) → `Collect(coord, color, normal)`; metres, 2 cm voxels, no rescale. Pointcept's Concerto ScanNet++ linear-probe config (`configs/concerto/semseg-ptv3-large-v1m1-2a-scannetpp-lin.py`) tests with the same steps, plus a second `CenterShift(apply_z=False)` after sampling, which is a sub-voxel xy shift.
- **Weights:** HF `Pointcept/Concerto` @ `efc130718a564eba0ce4899318f3c20b19d9c752` (2026-08-18), CC-BY-NC-4.0.

  | File | Size | What it is |
  |---|---|---|
  | `concerto_{tiny,small,base,large}.pth` | 0.02 / 0.15 / 0.43 / 0.83 GB | per the model card, "the pre-release version of our next work, Utonia" (they handle missing colour and normals) |
  | `concerto_base_origin.pth` | 0.43 GB | per the card, "the original Concerto model" |
  | `concerto_large_origin.pth` | 0.83 GB | not described in the card |
  | `concerto_base_siglip.pth` | 0.43 GB | SigLIPv2 instead of DINOv2 |
  | `concerto_large_outdoor.pth` | 0.83 GB | outdoor model |
  | `concerto_large_linear_prob_head_sc.pth` | 0.14 MB | ScanNet-20 probe head |
  | `pretrain-concerto-v1m1-2-large-video.pth` | 8.3 GB | Pointcept training checkpoint: Pointcept's Concerto README links it as "our pre-trained model weight" and probes it with the large configs (`backbone_out_channels=1728`) |

  The package's `MODELS` list (load by name) offers only the pre-release names.
- **Decision: K uses `concerto_large_origin.pth`.**
  - Large size: `enc_channels` 64-128-256-512-768, `enc_num_head` 4-8-16-32-48, so head dim 16 and upcast width 1728 (per the probe configs).
  - Why: it is the original model at the size Pointcept probes on ScanNet++.
  - **Unverified until step 1b runs:** whether it holds the same backbone as `pretrain-concerto-v1m1-2-large-video.pth`. Step 1b compares the student and teacher backbones tensor by tensor, scores the released ScanNet-20 head on both large backbones (which tells whose head it is), and reports stored config, width, memory and time per scene on a T4.
- **Step-1b run** (`pilot_02`, 2026-10-26 09:21 UTC, 2.8 min):
  - **Both large checkpoints:** 207.67M parameters, 0.774 GB, identical stored configs (`mask_token=True`, `traceable=True`, patch 1024, `enc_num_head` 4-8-16-32-48). **Upcast width 1728, head dim 16.**
  - **Released ScanNet-20 probe on the demo room:** 0.717 mIoU on the pre-release `concerto_large` (for comparison with their own heads: Utonia 0.780, Sonata 0.491), but **0.003 mIoU (accuracy 0.9 %) on `concerto_large_origin`.**
    - So the head was trained on the pre-release backbone, and the two files hold different weights.
    - The score says nothing about the origin model's quality, which is **still unchecked.**
  - **Comparison with the training checkpoint failed.** `torch.load(weights_only=True)` refuses the non-tensor objects in a Pointcept training checkpoint, and our 300-character cut removed the line naming them. Next: load it with an unpickler that replaces every non-tensor object with an inert stub, so no code from the file runs.
  - **SDPA vs the explicit product:** 1st-percentile cosine 1.0; prediction agreement 1.0 (fp32) and 0.99944 (fp16).
  - **fp16 vs fp32:** cosine 1.0; agreement 0.99925. **Decision: SDPA and fp16 for Concerto too.** (Agreement uses the released head, so for the origin model it measures prediction stability only.)
  - **Seeds:** the same seed gives bitwise-identical features. Another seed gives mean cosine 0.988 and 1st percentile 0.953, so Concerto is **more seed-sensitive than Utonia (0.995) and Sonata (0.992).**
  - **Cost, fp16 SDPA:**
    - 188k grid points: 0.68 s, 1.58 GB;
    - 746k: 2.80 s, 6.27 GB;
    - 1.65M: **out of memory** (allocator peak 8.6 GB when it failed);
    - fp32 and the explicit product already run out of memory at 746k.
    - Per grid point: 9.0 KB and 3.75 µs (Utonia: 7 KB and 4.4 µs).
  - **Most of the memory at large scenes is the upcast itself:** the final concatenation alone is N × 1728 × 2 B = 5.7 GB at 1.65M grid points, and allocating it needs its inputs alive too. Step 3 needs features at ≤ 10k sampled points per scene, and step 5 only needs logits. So step 3 gathers each stage's features at the sampled points' voxels, and step 5 upcasts per-stage logits (100 wide) instead of features (1728 wide). Whether chunking is still needed for the largest scenes (3.3M grid points) is to be measured then.
- **Distillation target:** Concerto's pretraining configs use `image_weight_name="dinov2_vitg14_reg"` (DINOv2 ViT-g/14 with registers), the same as Utonia (P2). The pilot's DINO block is ViT-L/14 without registers. So K+ − K asks whether lifted ViT-L features add anything beyond a model trained to predict ViT-g features.
- **Pretraining code in Pointcept** (pin `1342eda`), for the next phase. Reported only; nothing is set up.
  - `pointcept/models/concerto/concerto_v1m1_base.py` (`Concerto-v1m1`) and `concerto_v1m2_distill.py`.
  - `configs/concerto/pretrain-concerto-v1m1-0-base.py`, `-1-large-base.py`, `-2-large-video.py` (adds RE10K video), plus linear, decoder and fine-tuning configs for ScanNet, ScanNet200, ScanNet++, S3DIS and outdoor sets.
  - The large-base config: batch 96 in total, 100 epochs, AdamW lr 0.004. Data: ARKitScenes, ScanNet, ScanNet++ (train + val + test), S3DIS, HM3D and Structured3D through `DefaultImagePointDataset` (`pointcept/datasets/defaults.py`).
  - ScanNet++ image assets: `pointcept/datasets/preprocessing/concerto/scannetpp/` (`preprocess_scannetpp.py/.sh`, `splits.py`). It uses DSLR undistorted images with COLMAP poses, every 20th registered frame (`frame_gap=20`).
  - Extra requirements per its README: `transformers==4.50.3`, `peft`.

### P5. Pilot step 2: DSLR cameras and DINO lifting (code: `appearance_pt.scannetpp_cameras`, `experiments/pilot_concat/lifting.py`)
- **Pose convention (verified in toolbox code).**
  - `transforms_undistorted.json` frames come from `common/utils/nerfstudio.py convert_frames`, copied unchanged by `dslr/undistort.py`. That function takes COLMAP's camera-to-world, flips the camera y and z axes, swaps world rows x and y, and negates world row z.
  - `common/render.py` renders `mesh_aligned_0.05.ply` with the COLMAP poses as they are, so COLMAP's world frame is the mesh frame.
  - `dslr_c2w` undoes the three steps. A unit test checks it against the toolbox's own `convert_frames`.
- **Run-time check of the convention:** on the first scene, vertex luminance is correlated with photo luminance at the projected pixel for the expected convention and two wrong ones. The best one is used, and a warning is printed if it isn't the expected one.
  - On a synthetic room (CPU test): expected r 0.77 with 10% of points visible; wrong conventions r 0.35 / 0.02 with 0.1% visible.
  - Views that face a wall head-on see the side walls at grazing angles and score lower (about 0.6). Diagonal views reach r 0.9995.
- **Views:** 30 of the kept 60-view pool, evenly in capture order. `is_bad` frames were never kept.
- **DINOv2:** `facebook/dinov2-large` @ `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` (Apache-2.0; ViT-L/14, no registers), through HF `transformers` (in the wheelhouse). **Replaced by the variant with registers after the first run (P6).**
  - Input: the whole image resized to 518×770 (short side 518 per the spec; 770 and 784 are both 7 px from the exact 777, and the smaller was taken), ImageNet normalisation.
  - Output: last layer after the final norm, patch tokens only, a 37×55 grid, fp16.
- **Occlusion: a z-buffer of the points themselves, for every scene.** Mesh faces are kept only for val, so mesh-rendered depth is impossible for train scenes.
  - Cell 8 px (146×219 cells per view), closed with a 3×3 minimum filter.
  - A point is visible if it is ≥ 5 cm in front of the camera, inside the image, on a mask pixel > 127, and within 2 cm + 5 % of depth of the nearest depth around it.
  - **Known weakness:** at grazing angles, points up to that tolerance behind the nearest surface share its pixel's feature.
- **Per point:** bilinear sample of the patch grid at the projected pixel (`align_corners=False`), averaged over the views that see it. The view count is recorded; never-seen points get zeros and count 0.
- **Visual checks for review** (`step2_lift.py`, 3 val scenes: fewest, median and most vertices):
  - an overlay of the middle view with six panels: photo; visible points by depth on the photo; vertex colours rendered from the camera; DINO patch PCA of the photo; lifted features rendered with the same PCA; views per vertex;
  - a top-down view: vertex colours next to the lifted-feature PCA.
- **Also measured:** seen fraction (all and labelled vertices), views per vertex, and time per stage (DINO per view; z-buffer and lifting per million vertex-views), plus free disk and RAM for step 3.
- **CPU tests** (CPU torch 2.14.1 added to the local dev group; 526 MB):
  - toolbox pose round trip;
  - z-buffer occlusion (a plate in front of a wall), mask and bounds;
  - patch-centre sampling;
  - averaging over two views;
  - colour-agreement discrimination;
  - PCA;
  - `run_scene` end to end on a synthetic room packed like a Kaggle-unpacked core scene, with a stand-in for DINOv2.

- **Step-2 run** (`pilot_02`, 2026-10-26 09:26 UTC, 2.3 min; val scenes with the fewest, median and most vertices):

  | Scene | Vertices | Colour r | Seen (labelled) | Views per seen vertex | Load / DINO / lift (s) | Peak GPU |
  |---|---|---|---|---|---|---|
  | f9f95681fd | 271,704 | 0.868 | 0.947 (0.951) | 7.3 | 20.8 / 4.3 / 1.0 | 4.7 GB |
  | 1ada7a0617 | 1,224,287 | 0.879 | 0.938 (0.946) | 5.8 | 18.9 / 3.6 / 3.7 | 7.3 GB |
  | 578511c8a9 | 4,775,139 | 0.845 | 0.806 (0.816) | 4.6 | 21.0 / 3.6 / 9.5 | 7.4 GB |

  - **Pose convention confirmed:** colour agreement r 0.868 for the toolbox inverse against 0.158 (OpenGL camera, world as stored) and no visible points at all (OpenCV camera, world as stored). The vertex colours come from the laser scanner's own camera, so r ≈ 0.87 against the DSLR photos is a strong match.
  - **Intrinsics of the undistorted DSLR images (1168×1752) differ per scene:** fx 463.8, 623.3 and 464.6 in these three; cx 876, cy 584. That is a 109–124° horizontal field of view. One DINO patch (37×55 grid) covers ≈ 32 full-resolution pixels, so **≈ 5–7 cm per metre of depth (≈ 8–10 cm at 1.5 m), against 2 cm voxels.** This matters for the next phase's higher-resolution objectives.
  - **Never seen by any of the 30 views:** 5.3 %, 6.2 % and 19.4 % of vertices (the largest scene is the worst). Enough points fall on both sides for the visible/never-seen split.
  - **Masks:** 1.000 of the pixels usable in all 90 views. The convention is not inverted (that would give ≈ 0), and the undistorted images have no invalid border. Whether the DSLR masks mark the anonymised regions themselves can only show in a scene with anonymised vertices; none of these three is checked yet.
  - **Rates on a T4:** DINO 0.128 s per view (batch 6). Lifting 0.098 s and z-buffers 0.004 s per million vertex-views.
  - **Loading takes ≈ 20 s per scene whatever its size,** reading ~65 files from the dataset mount. The cause is not isolated (m2_05, in a CPU session, read all 250 core scenes, ≈ 135 files each, in 3.3 min with 4 processes). Step 3 will prefetch scenes in background threads.
  - **Machine (GPU session):** `/kaggle/working` 20.9 GB free, **`/tmp` 1,122.8 GB free** (the filesystem's report, not a documented quota), RAM 31.3 GB, 4 CPUs.
  - **Visual check** (user's screenshots, 2026-10-26):
    - **Geometry is right.** Projected points follow the edges of monitors, keyboards, chairs, utensils and door frames in all three scenes. Lifted features show the photo's DINO regions (ceiling, furniture, floor, counter).
    - **Isolated single-patch outliers** in uniform regions of the photos' DINO PCA (blinds, kitchen wall and cabinets, lab ceiling). These look like the high-norm artifact tokens of DINOv2 without registers (Darcet et al., ICLR 2024).
    - **Straight-edged regions and rays on floors in the top-down views.** The averaged feature changes with the set of views that see a point, so the per-view features depend on viewing geometry, not only on the surface. A 3-component PCA of uniform surfaces may exaggerate this.
    - **Horizontal banding on far, grazing surfaces** (lab ceiling and back wall) and **striped seen/never-seen patterns on the lab tables.** This is consistent with false self-occlusion: the 8 px cells and 3×3 minimum filter span 24 px, over which a grazing surface's depth changes by far more than the 2 cm + 5 % tolerance.
    - **Display:** averaged features drift towards the PCA mean and look grey, like the "never seen" grey.
    - Proposed changes before step 3 are in the 2026-10-26 report to the user. They are not decided yet.

### P6. Step-1b and step-2 revisions after the review (user approved, 2026-10-26)
- **DINO block: `facebook/dinov2-with-registers-large` @ `e4c89a4e05589de9b3e188688a303d0f3c04d0f3`** (Apache-2.0; ViT-L/14 with 4 register tokens; `Dinov2WithRegistersModel`, in transformers since 4.47, Kaggle has 5.16.1).
  - Replaces `facebook/dinov2-large` because of the single-patch outliers seen in the first run.
  - It is the same family as Concerto's and Utonia's teacher (ViT-g/14 with registers), so K+ − K compares like with like apart from model size.
  - `dino_patches` keeps the last 37×55 tokens, so the CLS and register tokens drop out unchanged.
- **Occlusion test with a slope term** (`lifting.Occlusion`, variants `v0`–`v3`):
  - With the Pointcept vertex normals, the tolerance adds the depth change of the point's own surface across the 3×3 neighbourhood: `slope_px × z × |(n_x/fx, n_y/fy)| / |n·r|` (camera coordinates, r = K⁻¹(u, v, 1); exact to first order for a plane).
  - |n·r| is floored at cos 80° · |r|. That bounds the term at ≈ 0.2·z for 16 px at f = 464, so no extra cap is needed: a fixed 0.5 m cap left a test floor only 36 % visible.
  - Variants: `v0` as first run; `v1` + slope over 16 px (default); `v2` 4 px cells and 8 px; `v3` = v1 + back-face rejection.
  - **CPU tests:**
    - a floor 4–9 m away seen from 1.5 m (incidence 70–80°): visible 3.1 % with v0, 100 % with v1 and v2;
    - points behind a plate stay hidden with every variant;
    - flipped normals are all rejected by v3;
    - synthetic room: coverage 59.5 % → 72.1 %.
  - **Step 2 now scores every variant against exact visibility on the val meshes:**
    - rays from the camera centre to each vertex, cast with Open3D's `RaycastingScene` (Open3D 0.20.0 is in the wheelhouse); visible if the first hit is no more than 2 cm before the vertex;
    - 5 views per scene, up to 300k candidates each; reported as precision and recall;
    - also reported: the fraction of visible points whose normal faces away (whether back-face rejection is safe), and whether the mesh vertices equal Pointcept's `coord`.
  - The default stays `v1` until those numbers are in.
- **View consistency** (`step2_lift.view_consistency`), on 50k points seen by ≥ 2 views: cosine between one view's feature and the mean of the other views, after subtracting the scene's mean patch feature. Baseline: the same, against a random other point's mean. Binned by angle from the optical axis, incidence angle and depth. The synthetic room gives 0.55 against a 0.02 baseline.
- **Display:**
  - never-seen vertices are white, and PCA colours are scaled to ≤ 220, so white can only mean "never seen";
  - the overlay gets a third row: visibility under v0, under the variant used, and from the mesh;
  - the top-down gets a third panel, views per vertex.
- **Step 1b:**
  - the training checkpoint is read with `appearance_pt.checkpoints.load_inert`: tensors and ordered dicts are rebuilt, and every other global becomes an inert stub, so nothing from the file runs. A test pickles an object whose reconstruction would call a function, and checks that it never runs;
  - a **half-room linear probe** (`probes.split_probe`: train on one side of the room's median x, score the other, both ways) for Utonia, Sonata, both Concerto files, the untrained PTv3 and the raw 9 input channels. It needs no released head, so it can check the original Concerto;
  - seed sensitivity is now measured for the pre-release Concerto too;
  - the size benchmark is skipped (`--tiles` with no values; measured in the first run).

### P6 results. Second `pilot_02` run (2026-10-26 11:45–11:49 UTC, commit `8c15f95`)
**Concerto (step 1b):**
- `concerto_large_origin` against the training checkpoint `pretrain-concerto-v1m1-2-large-video.pth`: all 453 tensors have the same names and shapes, but max |diff| is 0.97 (student) and 0.995 (teacher). **So it is not that checkpoint's backbone.** The inert loader stubbed only `_codecs.encode`, `numpy.core.multiarray.scalar` and `numpy.dtype` (plain metadata).
- **Half-room linear probe** on the demo room (ScanNet-20; mIoU, accuracy):

  | Encoder | mIoU | Accuracy |
  |---|---|---|
  | Utonia | 0.235 | 0.68 |
  | Concerto pre-release | 0.212 | 0.68 |
  | **Concerto origin** | **0.206** | **0.67** |
  | Sonata | 0.167 | 0.60 |
  | raw xyz + colour + normal | 0.098 | 0.26 |
  | untrained PTv3 | 0.085 | 0.42 |

  **The original model is healthy.** Its 0.003 with the released head (P4) only showed that the head belongs to the pre-release weights.
- **Decision: K uses `concerto_large_origin`.**
  - Open: which checkpoint the paper's ScanNet++ 45.6 comes from; step 5 compares K with it.
  - Seed sensitivity: pre-release 0.987, origin 0.988.

**Lifting (step 2):**
- **Registers:** the isolated single-patch outliers are gone from the photos' DINO PCA (user's screenshots).
- **Occlusion against mesh visibility** (5 views per scene, up to 300k points each; precision / recall, F1 in brackets):

  | Scene (vertices) | truly visible | v0 | v1 | **v2** | v3 |
  |---|---|---|---|---|---|
  | f9f95681fd (272k) | 0.83 | 0.945 / 0.841 (0.890) | 0.902 / 0.924 (0.913) | **0.908 / 0.956 (0.931)** | 0.932 / 0.858 (0.893) |
  | 1ada7a0617 (1.2M) | 0.59 | 0.914 / 0.900 (0.907) | 0.880 / 0.929 (0.904) | **0.873 / 0.958 (0.913)** | 0.930 / 0.842 (0.884) |
  | 578511c8a9 (4.8M) | 0.52 | 0.844 / 0.555 (0.668) | 0.775 / 0.810 (0.792) | **0.776 / 0.881 (0.825)** | 0.859 / 0.742 (0.796) |

  **Decision: v2** (4 px cells, slope term over 8 px): the best F1 in every scene.
- **Remaining error is precision** (0.78–0.91): points the z-buffer calls visible that the mesh hides. The mesh is decimated: flat surfaces (screens, table tops, chair backs) keep few vertices, so the z-buffer has holes there and the dense vertices behind them show through.
  - The overlays' mesh panel shows this: the undersides of monitors, chairs and tables are hidden (red), as they should be.
  - Exact visibility needs the mesh faces. Val has them; for train they were not kept (D11). Re-downloading them would be part of the 8.73 GB "mesh + mask + segments + annotations" group (D9), and the token expires on 2026-11-01.
- **Other step-2 numbers:**
  - Mesh vertices equal Pointcept's `coord` exactly (max |diff| 0) in all three scenes.
  - Back-facing normals among truly visible points: 7–9 %, so back-face rejection (v3) costs recall; not used.
  - Coverage with v1: 0.982 / 0.969 / 0.921 of vertices seen (v0: 0.947 / 0.938 / 0.806); v2 recalls more.
  - Colour agreement 0.85 / 0.86 / 0.78, slightly below v0's, as expected from the lower precision.
- **View consistency** (centred cosine, one view against the others' mean): 0.73 / 0.64 / 0.65, against 0.10 / 0.08 / 0.14 for a random other point.
  - By angle from the optical axis: 0–40° 0.65–0.78; 40–60° 0.57–0.70; **> 60° 0.51–0.55** (5–7 % of samples).
  - By incidence angle: no trend.
  - By depth (largest scene): near points are less consistent (0–1 m 0.49, > 4 m 0.68).
  - **Decision: no view weighting in the pilot**, since the inconsistent periphery is ≤ 7 % of samples. Recorded for the next phase.
- **Time:** loading 7–11 s per scene this run (20 s in the first); DINO 0.12 s per view; lifting 0.13 s per million vertex-views (with normals).

### P7. Steps 3–5: design (code and CPU tests 2026-10-26; one Kaggle notebook, `pilot_03_probes`)
- **One session, approved by the user (> 2 h):** step 3 → step 4 → step 5. The first cell sets the two choices settled by pilot_02 (P6 results): occlusion `v2`, Concerto `concerto_large_origin`.
- **Encoder features without the full upcast** (`stages.py`):
  - each stage's features are kept as they are, and the concatenation `[f_0, f_1[inv_1], …]` is built only at the vertices needed. A test checks it equals `encoders.encode`'s loop;
  - this avoids materialising N × 1728 values, which ran Concerto out of memory at 1.65M grid points (P4);
  - if a pass still runs out of memory, the scene is split at the median of its longer horizontal axis into tiles with 1 m of context, up to 8 tiles. Each vertex takes its features from the tile whose core holds it. Summaries list the scenes that needed tiles.
- **One pass per scene and encoder,** seed = CRC32 of the scene id (P3). Colours are used as Pointcept stores them, with magenta in anonymised regions, as in the encoders' pretraining and Pointcept's probing configs. Attention: SDPA; precision fp16.
- **Step 3, training features:**
  - per train scene, up to 10,000 vertices drawn uniformly from the labelled, non-anonymised ones; label = Pointcept `segment[:, 0]` (the major label);
  - blocks: Utonia 1386, Sonata 1232, Concerto 1728, untrained PTv3 1386 and lifted DINO 1024 (fp16, zeros where unseen; the same 30 views and occlusion variant as step 2); plus raw inputs, 9 channels in fp32 (x, y centred, z above the scene's 1st-percentile height, colour / 255, normal);
  - also stored: the number of views that see each vertex;
  - memory-mapped arrays on `/tmp` (≈ 27 GB; S1 option (a)). Scenes load in 3 background threads.
- **Step 4, probes** (configs A, B, C, C-eq, S, S+, K, K+, K+-eq, R, R+, X; `-eq` = each block PCA-reduced to 512 dims, so the concatenated blocks have equal width; added for K+ as well, since K+ is the key comparison):
  - features standardised per block; linear head; cross-entropy; AdamW (weight decay 1e-4); cosine decay; batch 4096;
  - learning rate per config from {3e-4, 1e-3, 3e-3, 1e-2}, chosen by point mIoU on 20 held-out train scenes (10 epochs, seed 0). Final probes: 15 epochs, seeds 0–2, all 200 scenes;
  - training runs in phases, one encoder block plus DINO at a time (≤ 11 GB), held on the GPU when it fits. Otherwise every epoch would re-read the whole 27 GB store;
  - each probe is folded into weights on the raw blocks (standardisation and PCA absorbed) and saved to `probes.pt`;
  - **the seeds change only the head's initialisation and batch order; the features are fixed.** So the seed spread understates the uncertainty, and the paired bootstrap over val scenes is the main measure for differences.
- **Step 5, evaluation on the 50 val scenes:**
  - **official GT built with the toolbox's own code in-process** (`official.py`: `ScannetPP_Release_Dataset` with prepare_training_data's transforms, multilabel `max_gt` 3, ignore −100).
  - **Finding:** in multilabel mode the toolbox keeps labels in annotation order, while Pointcept's `parse_scene` moves the smallest instance's label to the first column. The top-k metric uses the first GT label when no prediction matches, so Pointcept's `segment.npy` is not the official GT. Step 5 reports how often the two agree;
  - metric: `metrics.confusion_topk` on the GPU, equal to the toolbox's `fast_hist_topk_multilabel` (test). As a cross-check, one probe's predictions on 10 scenes are written as the toolbox's text files and re-scored by its `eval_semantic`;
  - anonymised vertices are excluded, as in `eval_semantic`;
  - reported: top-1 and top-3 mIoU (seed mean ± std), split into seen vs never-seen vertices, per-class IoU, per-class gains of each "+ DINO" pair, and paired-bootstrap 95 % intervals over scenes for K+ − K, C − B, S+ − S, R+ − R, K − S, K − B, K+ − C and the -eq pairs; sanity check against the paper (S 38.9, K 45.6).
- **Deviations from Concerto's linear probing (for the write-up):**
  - point-sampled frozen features instead of training on whole scenes (no augmentation, no Lovász loss, 15 epochs instead of 800);
  - 200 train scenes instead of all 856;
  - one encoder pass per scene, no test-time augmentation;
  - DINO averaged over 30 DSLR views.
- **CPU tests:** an end-to-end run of steps 3–5 on synthetic rooms, with stub encoder packages, fake encoders, the DINO stand-in and a stand-in for the toolbox GT. It checks the agreement with the toolbox's metric, and that the anonymised vertices are excluded.

### S1. Storage for the step-3 training features, now with Concerto (decided 2026-10-26: option (a))
200 train scenes × 10,000 labelled points = 2.0M points. Every train scene has at least 41k labelled vertices (min labelled fraction 0.19 × min 217k vertices), so the cap always binds.

| Block | Width | Bytes/point (fp16) | Total |
|---|---|---|---|
| Utonia | 1386 | 2,772 | 5.5 GB |
| Sonata | 1232 | 2,464 | 4.9 GB |
| **Concerto (new)** | 1728 (to be confirmed by step 1b) | 3,456 | **6.9 GB** |
| untrained PTv3 | 1386 | 2,772 | 5.5 GB |
| DINOv2 ViT-L | 1024 | 2,048 | 4.1 GB |
| xyz, colour, normal (fp32), labels, view count, ids | — | ≈ 51 | 0.1 GB |
| **Total** | | **≈ 13.6 KB** | **≈ 27.1 GB** (was ≈ 20.2 GB without Concerto) |

- **This does not fit the ~20 GB `/kaggle/working` in one output.** Options:
  - **(a)** Keep the dump on the session's temp disk and train the probes in the same session, persisting only the probes and metrics. **Measured in step 2: `/tmp` has 1.1 TB free in a GPU session**, so space is no constraint.
  - **(b)** Write one output and one private dataset per block, each 4–7 GB. That adds ≈ 27 GB of private storage (≈ 59 GB in total), and probes can be re-run without re-extracting.
  - **(c)** 5,000 points per scene: ≈ 13.6 GB.
- Val needs no dump: step 5 recomputes features per scene on the fly.
- **Decision (user, 2026-10-26): option (a)**, and approval to run steps 3–5 as one Kaggle session longer than 2 h. The aim is to finish the pilot within about 24–36 h.

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
- **Batch done (5 parallel CPU sessions, finished 2026-10-25 18:31–18:57 UTC):**

  | Part | Scenes | Downloaded | Kept | Time | Mean s/scene (download / reduce) |
  |---|---|---|---|---|---|
  | val | 50/50 | 39.7 GB | 0.84 GB | 1.43 h | 41 / 102 |
  | train-1 | 50/50 | 47.6 GB | 1.10 GB | 1.51 h | 44 / 107 |
  | train-2 | 50/50 | 52.8 GB | 1.70 GB | 1.79 h | 47 / 127 |
  | train-3 | 50/50 | 47.2 GB | 1.20 GB | 1.48 h | 41 / 105 |
  | train-4 | 50/50 | 42.9 GB | 1.03 GB | 1.51 h | 40 / 108 |
  | **Total** | **250/250** | **230.2 GB** | **5.87 GB** | | |

  - Every scene has the full 60-frame pool.
  - **Depth encoding:** all val scenes are per-frame lz4 uint16 mm. **10 train scenes** are instead per-frame **zlib float32 metres**, the toolbox's other per-frame format. They are converted exactly as the toolbox does (×1000, cast to uint16 mm). `m2_05` found no zero-depth pixels in any kept frame (D13).
  - Private datasets: `scannetpp-iphone-{val,train-1,train-2,train-3,train-4}`. With the core data that is **≈ 31.4 GB of ScanNet++ data in total**.

### D13. Check every saved scene before the token expires (`m2_05_scannetpp_verify`, CPU, no ScanNet++ token)
- **Why now:** a scene that is missing or damaged in a dataset can only be downloaded again until 2026-11-01. The download notebooks checked each tar as they wrote it; this checks what actually ended up in the eight Kaggle datasets.
- **What it does** (`appearance_pt.scannetpp_store`, `scripts/scannetpp/scannetpp_verify.py`):
  - Finds every `scenes/<scene>.tar` below `/kaggle/input`. It also accepts unpacked copies, `scenes/<scene>/` or `scenes/<scene>.tar/`, because it is not verified whether Kaggle unpacks tars when it builds a dataset; the layout printout will show.
  - **Reads and decodes every file:**
    - Pointcept arrays: dtype, shape (N, 3), finite values, labels in [-1, 99];
    - the anonymisation indices, which must be < N;
    - val raw files: the ply header vertex count and `segIndices` length must both equal N;
    - every DSLR view (must be a non-`is_bad` frame) and mask (same size as its image);
    - every iPhone RGB, mask and depth PNG (192×256, integer mm);
    - the pose entry (4×4 `aligned_pose`, 3×3 `intrinsic`) and COLMAP registration of every pool frame.
  - Tar sizes, vertex counts and pool sizes must match the download manifests.
  - Truncation shows up as a tar read error, or as members ending past the end of the file.
- **Numbers it also records** (per scene in `verify.json`, kept on Kaggle; not data, but not committed either):
  - 2 cm grid points, i.e. `floor(coord / 0.02)` cells, which is how many points the encoders see; needed to plan chunked encoding in pilot step 2;
  - labelled fraction;
  - anonymised vertices, and how many of them are magenta;
  - magenta vertices outside the mask;
  - image sizes against `transforms_undistorted.json`'s `w`/`h`;
  - COLMAP camera models;
  - depth encodings and zero-depth fraction.
- **Local test:** a fake input tree runs through the full script. Unit tests cover a half-written JPEG, a cut-off tar, an unregistered iPhone frame and a manifest mismatch. Pillow was added to the local dev dependencies for these tests.

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
