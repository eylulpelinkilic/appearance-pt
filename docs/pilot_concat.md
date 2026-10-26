# Pilot: do lifted DINOv2 features still add to Concerto? (ScanNet++ v2, linear probes)

2026-10-26. Code: `experiments/pilot_concat/`. Design, checks and raw numbers: `docs/decisions.md` P3–P7.
Runs: Kaggle notebooks `pilot_02_concerto_lift` (steps 1b–2), `pilot_03_probes` (steps 3–4, commit `b949425`),
`pilot_04_eval` (step 5, commit `03228a2`).

## Summary

- **Yes, but little.** Concatenating lifted DINOv2 features to frozen Concerto features improves a linear probe by
  **+2.7 mIoU (95 % interval +1.2 to +4.0)** on ScanNet++ val (Concerto 39.1 → 41.8). The same features add +9.9 to
  Sonata, Concerto's image-free counterpart. Concerto's image training already recovers about three quarters of
  what lifted DINO gives a 3D-only encoder.
- **The remaining gain is on small, appearance-defined objects.** 11 of 99 classes improve reliably and none gets
  reliably worse. The clearest gains are mouse +16, telephone +15, cup +14, socket +8 and book +6 IoU points. The
  gain grows from frequent classes (+1.3) to rare ones (+4.1).
- **With DINO added, the trained encoders end up close together.** Sonata ⊕ DINO 40.7, Utonia ⊕ DINO 41.5,
  Concerto ⊕ DINO 41.8; the last two cannot be separated. Under linear probing, most of what separates these
  encoders is image-derived information.
- **On points no camera sees, lifting cannot help, but distillation does.** On the 5.5 % of vertices outside all
  30 views, adding DINO slightly lowers every encoder's score. Concerto beats Sonata there by 8.6 points.
- **Sanity checks pass.** Our metric equals the official toolbox's on the same predictions. Concerto's lead over
  Sonata (+8.3, interval +6.5 to +9.6) agrees with the paper's +6.7. Our absolute numbers are 6–8 points below the
  paper's because our probing is much lighter (see Deviations).

## Question and setup

Concerto (NeurIPS 2025) trains a point encoder jointly on 3D self-distillation and on predicting DINOv2 image
features from points. Its pilot study showed that concatenating image and point features beats either alone. We ask
whether, after that training, lifted DINOv2 features still add information to Concerto's, and on which classes and
points.

- **Data:** ScanNet++ v2, top-100 semantic classes.
  - Training: 200 train scenes (fixed random subset), up to 10,000 labelled, non-anonymised vertices each (2.0M
    points), labelled with Pointcept's major label.
  - Evaluation: all 50 val scenes, every vertex of `mesh_aligned_0.05.ply` (75.6M vertices), anonymised vertices
    excluded.
- **Point encoders (frozen, one pass per scene, fp16):** Sonata, Concerto (`concerto_large_origin`), Utonia, and an
  untrained PTv3. Features are all five stages concatenated at full resolution (Concerto 1728 channels). All three
  trained encoders saw the ScanNet++ val scenes, unlabelled, during pretraining.
- **Image features:** DINOv2 ViT-L/14 with registers, last-layer patch tokens at 518 × 770.
  - Lifted to vertices from 30 undistorted DSLR views per scene, averaged over the views that see each vertex.
  - Occlusion comes from a point z-buffer; against the mesh, its precision is 0.78–0.91 and its recall 0.88–0.96.
  - Vertices that no view sees get zeros.
  - Concerto was trained to predict a larger DINOv2 (ViT-g/14), so K+ adds features of a smaller model than its
    own target.
- **Probes:** one linear layer per configuration, on per-block standardised features.
  - Training: cross-entropy, AdamW, cosine schedule, 15 epochs, batch 4096, 3 seeds.
  - Learning rate chosen per configuration on 20 held-out train scenes.
- **Configurations:**
  - A DINO alone, B Utonia, C B ⊕ A, S Sonata, S+ S ⊕ A, K Concerto, K+ K ⊕ A, R untrained PTv3, R+ R ⊕ A, and
    X raw xyz/colour/normal.
  - `-eq`: both blocks PCA-reduced to 512 dimensions before concatenation, to rule out a width effect.
- **Metric and uncertainty:**
  - Official ScanNet++ multilabel top-1 and top-3 mIoU per vertex. The toolbox's own GT is built in-process, and
    the toolbox's `eval_semantic` reproduces our numbers exactly (probe K, 10 scenes).
  - Seed spread only reflects the probe's initialisation and batch order. Differences are therefore judged by a
    paired bootstrap over the 50 val scenes (1000 resamples, 95 % intervals).

## Results

### mIoU per configuration (val, %)

Mean ± std over 3 probe seeds. "Seen" is the 94.5 % of evaluated vertices seen by at least one of the 30 views
(61–99 % per scene); "never seen" is the rest.

| Config | Features | Top-1 | Top-3 | Top-1 seen | Top-1 never seen |
|---|---|---|---|---|---|
| A | DINO (lifted) | 33.89 ± 0.12 | 62.51 ± 0.13 | 35.54 | 0.34 |
| B | Utonia | 37.64 ± 0.03 | 64.68 ± 0.13 | 37.73 | 29.80 |
| C | Utonia ⊕ DINO | 41.47 ± 0.07 | 68.89 ± 0.05 | 42.10 | 28.84 |
| C-eq | same, 512 + 512 dims | 40.86 ± 0.03 | 68.08 ± 0.03 | 41.52 | 28.17 |
| S | Sonata | 30.83 ± 0.05 | 54.91 ± 0.04 | 31.16 | 23.20 |
| S+ | Sonata ⊕ DINO | 40.73 ± 0.02 | 69.07 ± 0.05 | 41.68 | 22.73 |
| K | Concerto | 39.09 ± 0.03 | 66.20 ± 0.04 | 39.18 | 31.78 |
| **K+** | **Concerto ⊕ DINO** | **41.76 ± 0.05** | **69.20 ± 0.04** | **42.34** | 30.99 |
| K+-eq | same, 512 + 512 dims | 41.42 ± 0.05 | 69.50 ± 0.08 | 42.07 | 29.75 |
| R | untrained PTv3 | 6.97 ± 0.00 | 15.38 ± 0.00 | 6.99 | 6.27 |
| R+ | untrained PTv3 ⊕ DINO | 34.07 ± 0.02 | 60.01 ± 0.04 | 35.72 | 4.97 |
| X | xyz, colour, normal | 2.63 ± 0.00 | 7.91 ± 0.00 | 2.63 | 3.35 |

### Differences (top-1 mIoU, paired bootstrap over val scenes)

| Comparison | Difference | 95 % interval | Reading |
|---|---|---|---|
| K+ − K | **+2.67** | [+1.22, +4.04] | DINO still adds to Concerto |
| C − B | +3.83 | [+2.35, +5.02] | and to Utonia |
| S+ − S | +9.90 | [+7.78, +11.07] | and much more to Sonata |
| R+ − R | +27.10 | [+24.24, +28.79] | an untrained encoder adds little to DINO (R+ ≈ A) |
| K − S | +8.26 | [+6.52, +9.62] | Concerto's image training (paper: +6.7) |
| K − B | +1.45 | [+0.33, +2.74] | Concerto ahead of Utonia |
| K+ − C | +0.29 | [−0.63, +1.72] | not resolved |
| K+-eq − K+ | −0.34 | [−1.05, +0.33] | width does not explain the gain |
| C-eq − C | −0.61 | [−1.22, +0.07] | width does not explain the gain |

The probe-seed std (≤ 0.13) is much smaller than these intervals: the val scenes, not the probe, limit what can be
resolved. Differences under about 1 mIoU cannot be resolved with 50 val scenes under this protocol.

### Per class

A per-class change counts only if its 95 % interval, from the same scene bootstrap, excludes 0 (by at least 0.05
IoU points). The full table is in the appendix.

- **K+ − K: 11 classes up, none down.**
  - Up (IoU points): mouse +16.1 [+5.5, +30.3], telephone +15.3 [+4.0, +29.2], cup +13.5 [+5.2, +23.5], socket
    +7.9 [+4.1, +13.1], toilet paper +6.1, exhaust fan +6.1, book +6.1, refrigerator +3.5, bed +3.0, monitor +1.8,
    table +1.1.
  - The largest point estimates are not resolved: headphones +34, clock +26, air vent +23. Each occurs in only 2–3
    val scenes.
  - Neither are all the apparent losses: tissue box −21, blind rail −17, painting −13, shoe rack −13, each in 1–2
    scenes.
- **The gain grows as classes get rarer.** Grouping classes into thirds by val vertex count gives:

  | Third | Typical classes | K+ − K | K − S | S+ − S |
  |---|---|---|---|---|
  | Frequent | wall, floor, table, chair, door, window | +1.3 [+0.7, +2.5] | +7.3 | +7.1 |
  | Middle | towel, book, sink, bag, microwave, picture | +2.7 [+0.8, +4.6] | +13.6 | +13.3 |
  | Rare | speaker, kettle, pan, power strip, smoke detector | +4.1 [+1.0, +7.0] | +4.1 | +9.6 |

  Concerto's image training helps mid-sized objects most: computer tower, suitcase, printer, shoes, blanket and
  microwave each gain 29–55 points over Sonata. It helps the rare, small classes much less. On the rare third,
  Concerto ⊕ DINO is no better than Sonata ⊕ DINO (−1.4 [−3.1, +2.1]).
- **DINO helps the same classes whichever encoder it is added to.** The rank correlation of per-class gains with
  K+ − K is 0.60 for S+ − S and 0.68 for C − B. It is only 0.10 for K − S: the classes where Concerto beats
  Sonata are mostly not the ones where DINO still helps Concerto.

### Seen and never-seen points

- **On seen vertices,** K+ − K is +3.2 mIoU.
- **On never-seen vertices (5.5 %),** adding DINO lowers every encoder's score: K+ 31.0 vs K 31.8, S+ 22.7 vs S 23.2,
  C 28.8 vs B 29.8. There the DINO block is zero, and the probe, fitted almost entirely on seen points, has no flag
  that tells it the image features are missing.
  - The mean per-class gain on never-seen vertices (−2.9) differs from the mIoU difference (−0.8). The per-class
    mean covers only classes with an IoU in both configurations, and on this small subset, spurious predictions of
    absent classes count as IoU 0 in one configuration only.
  - Per-class never-seen gains are in the appendix without intervals; most classes have few or no never-seen
    vertices.
- **The encoders themselves transfer image knowledge to geometry alone.** On never-seen vertices Concerto scores
  31.8 against Sonata's 23.2. Caveat: "never seen" means outside our 30 views. Concerto was pretrained on these
  scenes with their images, so these points may have had image supervision during its pretraining.

### Learning rate

Every configuration except X chose the lowest rate in the grid (3e-4). The held-out score falls steadily with rate,
so a lower rate might be slightly better. From 1e-3 to 3e-4:

- the trained encoders gain 0.7–1.6 points;
- the differences between configurations hardly move (held-out K+ − K 4.1 → 4.2, S+ − S 9.5 → 9.6, K − S 6.3 → 7.0).

A lower rate would likely raise all absolute numbers by about a point without changing the comparisons, so the
pilot was not re-run.

## Comparison with the Concerto paper

Concerto reports linear probing on ScanNet++ val at 38.9 (Sonata) and 45.6 (Concerto). We get 30.8 and 39.1, 8.1
and 6.5 points lower. Our probe is much lighter (below), so the absolute numbers are not comparable. The comparison
between encoders is: our K − S = +8.3 [+6.5, +9.6] contains the paper's +6.7.

## Deviations and limitations

- **Probing:**
  - frozen features at 10,000 sampled points per scene from 200 of the 856 train scenes, not whole scenes from all
    of them;
  - 15 epochs, not about 800, and no augmentation or Lovász loss;
  - one encoder pass per scene, no test-time augmentation;
  - learning rate at the low edge of its grid.
- **Lifting:** one DINOv2 model (ViT-L/14 with registers), 30 DSLR views, last layer only, view averaging and a point
  z-buffer for occlusion. Exact mesh visibility would be cleaner; the train-scene triangles are now saved
  (`scannetpp-train-faces`) but were not used here.
- **Linear probes only;** no decoder probing.
- **Evaluation:**
  - all three trained encoders saw the val scenes, unlabelled, during pretraining;
  - differences under about 1 mIoU cannot be resolved with 50 val scenes;
  - per-class results for classes found in few scenes are not resolved, and are reported as such.
- **Seeds:** three probe seeds; the encoders and the lifting are deterministic per scene.

## What this suggests for the next phase (continued pretraining from Sonata)

These are suggestions, to be decided with the user.

- **There is still room for a better image objective, but not much at this probing level.** K+ is a rough yardstick
  for a Concerto-like encoder that fully absorbed ViT-L DINO features on seen points: +2.7 mIoU over Concerto.
  Since differences under about 1 mIoU cannot be resolved here, a new objective must beat the Concerto-objective
  control by a comparable margin, or the evaluation needs more power (more val scenes, or decoder probing).
- **Target small, appearance-defined objects.** That is where image information is still missing: mouse,
  telephone, cup, socket, book, and the rare third of the classes. Higher-resolution image targets, such as DSLR
  images at finer patch density, fit that.
- **Keep the image knowledge inside the 3D encoder.** Lifting cannot cover unseen points and slightly hurts there,
  while Concerto-style distillation carries over to them.
- **Evaluation protocol for that phase:** extend the learning-rate grid below 3e-4 (and try stronger weight decay),
  keep the paired scene bootstrap, report per-class intervals, and add decoder probing.

## Reproduce

- **Steps 3–5:** notebook `pilot_03_probes` (≈ 2 GPU hours on 2× T4: feature dump ≈ 40 min, probes 25 min,
  evaluation 45 min).
- **Step 5 alone:** `pilot_04_eval`, from a saved `probes.pt`.
- **Per-class intervals:**
  `python experiments/pilot_concat/step5_classes.py --results results.json --confmats confmats.npz` (local, CPU).

## Appendix: per-class K+ − K

Top-1 IoU (%) on all evaluated vertices, mean over 3 probe seeds.

- **Columns:**
  - "Val scenes": how many of the 50 val scenes contain the class.
  - "Vertices %": the class's share of evaluated vertices.
  - "Diff" in bold: the 95 % interval excludes 0.
  - "Seen" / "Never seen": the same difference on those vertex subsets, without intervals. "—" means the class does
    not occur there.
- Classes with no IoU in either configuration are omitted.

| Class | Val scenes | Vertices % | K | K+ | Diff | 95 % interval | Seen | Never seen |
|---|---|---|---|---|---|---|---|---|
| headphones | 2 | 0.005 | 12.3 | 46.5 | +34.2 | [-24.1, +70.2] | +34.2 | — |
| clock | 3 | 0.012 | 42.5 | 68.4 | +25.8 | [+0.0, +38.9] | +20.2 | — |
| air vent | 3 | 0.007 | 0.0 | 23.5 | +23.4 | [-0.2, +37.7] | +23.5 | +0.0 |
| clothes hanger | 1 | 0.035 | 23.3 | 40.1 | +16.8 | [+0.0, +21.9] | +17.0 | -8.0 |
| cutting board | 4 | 0.015 | 21.9 | 38.2 | +16.3 | [-4.2, +18.8] | +17.1 | -2.3 |
| mouse | 15 | 0.014 | 38.7 | 54.8 | **+16.1** | [+5.5, +30.3] | +16.1 | +0.0 |
| telephone | 14 | 0.048 | 41.9 | 57.2 | **+15.3** | [+4.0, +29.2] | +15.3 | — |
| speaker | 6 | 0.032 | 16.4 | 31.1 | +14.7 | [-1.1, +40.6] | +14.7 | +0.0 |
| cup | 17 | 0.036 | 18.6 | 32.0 | **+13.5** | [+5.2, +23.5] | +15.3 | -19.4 |
| power strip | 7 | 0.019 | 2.4 | 13.4 | +11.0 | [+0.0, +27.6] | +11.4 | +0.0 |
| basket | 4 | 0.049 | 3.4 | 13.2 | +9.8 | [-3.3, +44.6] | +13.4 | -6.6 |
| bag | 13 | 0.215 | 19.5 | 27.7 | +8.3 | [-0.6, +14.7] | +9.2 | -1.5 |
| socket | 38 | 0.084 | 23.0 | 30.9 | **+7.9** | [+4.1, +13.1] | +8.1 | -2.7 |
| blinds | 15 | 2.122 | 59.4 | 67.2 | +7.8 | [-1.1, +19.4] | +8.0 | -6.6 |
| pan | 3 | 0.019 | 3.3 | 10.9 | +7.6 | [+0.0, +9.0] | +7.6 | — |
| coat hanger | 3 | 0.052 | 15.6 | 23.0 | +7.4 | [-5.8, +21.5] | +7.2 | +12.2 |
| toilet paper | 9 | 0.014 | 47.1 | 53.2 | **+6.1** | [+1.1, +12.0] | +7.1 | -0.1 |
| exhaust fan | 9 | 0.077 | 15.0 | 21.1 | **+6.1** | [+0.4, +22.9] | +6.3 | +0.0 |
| book | 19 | 0.268 | 41.3 | 47.4 | **+6.1** | [+1.0, +9.9] | +6.5 | -1.9 |
| paper bag | 1 | 0.005 | 0.0 | 5.6 | +5.6 | [+0.0, +12.8] | +5.8 | +0.0 |
| paper towel | 5 | 0.011 | 10.7 | 15.9 | +5.2 | [+0.0, +14.5] | +5.2 | +0.0 |
| kettle | 6 | 0.023 | 19.0 | 24.0 | +4.9 | [-0.7, +22.8] | +10.9 | +0.0 |
| keyboard | 17 | 0.117 | 61.2 | 65.9 | +4.7 | [-3.9, +15.9] | +4.6 | — |
| whiteboard eraser | 6 | 0.004 | 4.8 | 9.1 | +4.3 | [-0.7, +13.2] | +4.3 | — |
| soap dispenser | 5 | 0.011 | 38.0 | 41.9 | +3.9 | [-1.0, +13.1] | +3.9 | — |
| shoes | 7 | 0.165 | 56.1 | 59.9 | +3.8 | [-0.9, +15.7] | +4.4 | +0.5 |
| crate | 3 | 0.160 | 46.5 | 50.1 | +3.6 | [-4.4, +4.7] | +3.9 | -3.3 |
| bottle | 20 | 0.088 | 27.3 | 30.8 | +3.6 | [-1.4, +8.5] | +3.6 | +1.4 |
| picture | 14 | 0.183 | 34.6 | 38.1 | +3.5 | [-0.2, +9.1] | +3.5 | +18.2 |
| smoke detector | 19 | 0.020 | 51.7 | 55.3 | +3.5 | [-5.3, +12.0] | +3.7 | +0.6 |
| refrigerator | 8 | 0.337 | 45.9 | 49.3 | **+3.5** | [+0.8, +6.2] | +4.3 | -3.6 |
| curtain | 7 | 1.621 | 75.6 | 78.9 | +3.3 | [-0.9, +11.2] | +4.0 | -0.6 |
| bed | 8 | 0.535 | 77.1 | 80.2 | **+3.0** | [+0.8, +7.3] | +3.4 | +1.2 |
| trash can | 35 | 0.809 | 70.2 | 73.2 | +3.0 | [+0.0, +6.3] | +3.6 | -3.8 |
| towel | 15 | 0.327 | 73.4 | 76.2 | +2.8 | [-0.1, +8.2] | +3.5 | -9.8 |
| container | 2 | 0.009 | 0.0 | 2.7 | +2.7 | [+0.0, +5.4] | +3.0 | +0.0 |
| window | 29 | 2.175 | 49.4 | 52.0 | +2.6 | [+0.0, +5.7] | +2.7 | -2.2 |
| microwave | 8 | 0.185 | 75.8 | 78.2 | +2.5 | [-1.6, +6.5] | +2.5 | +0.9 |
| box | 32 | 0.874 | 47.5 | 50.0 | +2.4 | [-0.7, +6.0] | +2.5 | +0.3 |
| window frame | 22 | 1.240 | 23.5 | 25.8 | +2.3 | [-0.1, +4.9] | +2.5 | -2.4 |
| computer tower | 7 | 0.561 | 85.9 | 88.1 | +2.2 | [-1.4, +21.2] | +3.2 | -4.5 |
| kitchen counter | 7 | 0.313 | 55.0 | 57.1 | +2.1 | [-0.6, +6.4] | +1.7 | +2.0 |
| whiteboard | 18 | 0.980 | 65.2 | 67.2 | +2.0 | [-0.4, +4.4] | +2.1 | -4.3 |
| plant | 6 | 0.345 | 83.3 | 85.2 | +1.9 | [-1.8, +7.4] | +1.6 | +13.6 |
| kitchen cabinet | 8 | 0.657 | 35.1 | 36.9 | +1.8 | [-2.4, +8.2] | +2.6 | -8.7 |
| monitor | 18 | 1.282 | 89.2 | 90.9 | **+1.8** | [+0.5, +3.5] | +1.8 | +3.0 |
| ceiling lamp | 40 | 5.184 | 67.5 | 69.2 | +1.8 | [-0.1, +3.8] | +1.7 | +1.8 |
| spray bottle | 3 | 0.009 | 0.0 | 1.7 | +1.7 | [+0.0, +2.4] | +1.7 | — |
| shower wall | 5 | 0.894 | 56.2 | 57.8 | +1.6 | [-1.9, +7.1] | +1.9 | -4.9 |
| blanket | 7 | 0.535 | 80.6 | 82.2 | +1.6 | [-0.6, +10.9] | +1.4 | -1.6 |
| pipe | 9 | 0.708 | 76.3 | 77.8 | +1.6 | [+0.0, +5.8] | +1.5 | +2.2 |
| bowl | 2 | 0.004 | 0.0 | 1.6 | +1.6 | [+0.0, +5.1] | +1.6 | — |
| doorframe | 25 | 0.755 | 29.8 | 31.3 | +1.5 | [-1.2, +3.9] | +2.0 | -12.2 |
| backpack | 7 | 0.129 | 46.8 | 48.3 | +1.5 | [-9.9, +8.6] | +1.6 | -2.7 |
| storage cabinet | 10 | 1.099 | 51.7 | 53.1 | +1.4 | [-0.7, +3.6] | +1.5 | -1.1 |
| toilet | 8 | 0.110 | 89.1 | 90.4 | +1.3 | [-0.2, +2.9] | +1.2 | +7.1 |
| tap | 7 | 0.018 | 21.6 | 22.9 | +1.3 | [-6.5, +12.8] | +1.3 | -19.3 |
| printer | 7 | 0.163 | 74.0 | 75.2 | +1.2 | [-0.5, +3.8] | +1.1 | +3.3 |
| table | 32 | 3.984 | 72.1 | 73.2 | **+1.1** | [+0.4, +1.7] | +1.4 | -4.0 |
| pillow | 7 | 0.115 | 37.5 | 38.5 | +1.0 | [-2.8, +6.1] | +1.1 | +9.6 |
| bucket | 3 | 0.047 | 10.9 | 11.7 | +0.8 | [-7.7, +9.4] | +0.8 | +0.5 |
| heater | 24 | 0.964 | 73.9 | 74.6 | +0.7 | [-0.9, +2.4] | +1.0 | -1.8 |
| table lamp | 2 | 0.009 | 73.4 | 74.1 | +0.7 | [-14.5, +14.5] | +0.7 | — |
| ceiling | 43 | 17.249 | 84.3 | 84.9 | +0.7 | [-0.2, +1.6] | +0.7 | +0.1 |
| bookshelf | 8 | 1.050 | 33.9 | 34.4 | +0.5 | [-1.1, +2.6] | +0.4 | +2.2 |
| windowsill | 20 | 0.492 | 60.4 | 60.9 | +0.4 | [-3.0, +3.8] | +0.4 | +1.6 |
| wall | 50 | 22.600 | 79.4 | 79.8 | +0.4 | [-0.1, +0.9] | +0.5 | -0.8 |
| office chair | 16 | 1.406 | 76.5 | 76.8 | +0.3 | [-0.6, +1.3] | +0.5 | -2.2 |
| cabinet | 29 | 1.669 | 36.0 | 36.2 | +0.2 | [-1.8, +2.5] | +0.3 | -2.6 |
| light switch | 38 | 0.068 | 11.2 | 11.3 | +0.1 | [-2.4, +4.6] | +0.1 | +0.0 |
| cushion | 1 | 0.018 | 0.0 | 0.1 | +0.1 | [+0.0, +0.2] | +0.1 | — |
| floor | 50 | 14.474 | 93.4 | 93.5 | +0.0 | [-0.2, +0.2] | +0.0 | -0.2 |
| file folder | 0 | 0.000 | 0.0 | 0.0 | +0.0 | [+0.0, +0.0] | +0.0 | — |
| slippers | 1 | 0.007 | 0.0 | 0.0 | +0.0 | [+0.0, +0.0] | +0.0 | +0.0 |
| binder | 1 | 0.004 | 0.0 | 0.0 | +0.0 | [+0.0, +0.0] | +0.0 | +0.0 |
| stapler | 1 | 0.004 | 0.0 | 0.0 | +0.0 | [+0.0, +0.0] | +0.0 | — |
| marker | 1 | 0.001 | 0.0 | 0.0 | +0.0 | [+0.0, +0.0] | +0.0 | — |
| door | 47 | 2.390 | 68.2 | 68.1 | -0.0 | [-2.0, +2.1] | +0.1 | -4.5 |
| shelf | 20 | 2.018 | 21.7 | 21.6 | -0.1 | [-1.4, +1.6] | -0.0 | -1.9 |
| board | 5 | 0.095 | 2.8 | 2.6 | -0.2 | [-1.1, +0.5] | -0.2 | +0.0 |
| toilet brush | 5 | 0.014 | 54.5 | 54.3 | -0.2 | [-9.1, +7.0] | +0.8 | -5.2 |
| chair | 25 | 3.225 | 77.4 | 77.0 | -0.4 | [-2.1, +1.1] | -0.3 | -2.4 |
| electrical duct | 13 | 0.256 | 17.6 | 17.1 | -0.5 | [-3.9, +2.2] | -0.6 | -0.3 |
| pot | 1 | 0.011 | 17.2 | 16.6 | -0.6 | [-3.2, +2.0] | -1.2 | +1.7 |
| cloth | 4 | 0.054 | 0.9 | 0.0 | -0.9 | [-2.1, +0.1] | +0.0 | -2.5 |
| paper | 16 | 0.106 | 17.7 | 16.3 | -1.4 | [-6.8, +1.7] | -0.9 | -31.9 |
| sofa | 8 | 0.732 | 75.5 | 73.6 | -1.9 | [-8.5, +2.1] | -2.0 | -0.6 |
| poster | 4 | 0.014 | 10.5 | 8.5 | -2.1 | [-11.8, +4.9] | -2.1 | +0.0 |
| sink | 18 | 0.250 | 65.7 | 63.5 | -2.2 | [-5.0, +1.2] | -2.2 | -3.2 |
| plant pot | 6 | 0.035 | 16.0 | 12.7 | -3.3 | [-16.5, +8.2] | -3.0 | -13.4 |
| jacket | 6 | 0.182 | 59.7 | 56.2 | -3.5 | [-21.6, +4.6] | -3.8 | -2.3 |
| laptop | 7 | 0.051 | 36.5 | 32.2 | -4.2 | [-26.1, +8.9] | +3.8 | -54.1 |
| tv | 6 | 0.412 | 91.0 | 86.4 | -4.6 | [-14.3, +1.7] | -0.7 | -27.4 |
| jar | 2 | 0.006 | 13.8 | 9.0 | -4.8 | [-16.4, +17.4] | -5.0 | +0.0 |
| suitcase | 5 | 0.112 | 68.3 | 61.9 | -6.4 | [-17.4, +8.7] | -6.3 | -4.8 |
| shoe rack | 2 | 0.062 | 19.0 | 6.1 | -12.9 | [-14.5, +0.0] | -13.3 | -1.9 |
| painting | 1 | 0.010 | 12.9 | 0.0 | -12.9 | [-18.0, +0.0] | -12.9 | — |
| blind rail | 2 | 0.027 | 17.9 | 0.7 | -17.3 | [-48.7, +0.0] | -18.9 | +0.0 |
| tissue box | 1 | 0.004 | 21.4 | 0.0 | -21.4 | [-27.0, +0.0] | -1.4 | -28.0 |
