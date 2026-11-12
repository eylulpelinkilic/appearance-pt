# Experiment 3: where did the distilled image information go?

[Experiment 2](../Experiment2_CrossAttention/README.md) trained a point-only Sonata student to match a cross-attention
teacher's DINOv2 targets. Probed linearly on f(p), the student gained +1.91 [+0.83, +3.38] mIoU over frozen Sonata,
which is 19 % of the concatenation gain (about 42 % at top-3). Its training loss fell from 0.675 to 0.134. The
teacher's learned attention was barely better than a uniform average of the cameras (dev cosine 0.6792 vs 0.6741).
This experiment runs three diagnostics of that student. Nothing in Experiment 2's code or outputs changes: its
modules are imported and its saved artifacts reused.

| Run | Question | What it measures |
|---|---|---|
| **A** | Is the distilled information in the student but not linearly readable from f(p)? | Linear probes (the pilot's recipe) on A2 h(f(p)) (1024-d) and A3 [f(p), h(f(p))] (2256-d), against A1 f(p), Experiment 2's row |
| **B** | Does the student's match to its targets hold on unseen scenes? | B1 = cos(h(f_student(p)), t_attn(p)) on 50 train and 50 val scenes; B2 = the same for a head of h's architecture trained on frozen Sonata until plateau; B3 = cos(t_attn, t_mean). Breakdowns by camera count (1, 2, ≥ 3) and by the 15 most frequent val classes |
| **C** | Does learned attention matter? | A second student, identical to Experiment 2's step 3 except that its target is t_mean, the uniform camera average (the pilot's lifted feature). Evaluated exactly like Experiment 2's step 4 (f(p), plus head features) and compared with the attention student |

**Decision rules:** these are fixed in `config.json` (`decision_rules`) before the full runs, and the report prints
them in its method section.
- **A:** supported if A2 or A3 beats A1 at top-1 with a 95 % paired-bootstrap interval above 0.
- **B:** compare the generalization gap (train − val cosine, bootstrap over scenes) of the student with B2's. If the
  interval of the difference of gaps is above 0, the student fitted the train scenes.
- **C:** if |uniform − attention| < 1 mIoU, or its interval includes 0, the report says "no evidence that learned
  attention matters", never "equivalent". The report then proposes a second seed of both students, with a time
  estimate; that seed is not run.

**Shared protocol.**
- Probes use the pilot's recipe: learning-rate grid [3e-4, 1e-3, 3e-3, 1e-2], 20 held-out train scenes, 10 sweep
  epochs, then seeds 0–2 × 15 epochs at batch 4096. Each feature set gets its own learning rate, chosen by the
  recipe's rule.
- Evaluation: the pilot's 10,000 labelled points per train scene, every vertex of the 50 val scenes, and the official
  metric.
- Comparisons: paired bootstrap over the 50 val scenes (1000 resamples), with unrounded differences.
- Every row reports its recovered share of the concatenation gain at top-1 and top-3.
- The frozen Sonata and Sonata + DINOv2 rows are reused from the pilot; A1 is reused from Experiment 2's saved
  confusion matrices.

Start commit: `85cf5c870c9b95f2e01cca9a7c118d1a0e2c95a9`. Experiment 2's full run: commit `0546137`.

## Files

| File | What it does |
|---|---|
| `config.json` | Decision rules, settings per mode (`smoke`, `full`) |
| `xdiag.py` | Shared pieces: config, finding Experiment 2's full output (fails early when an artifact is missing), the student with its head from `student/checkpoint.pt`, per-class IoU. No probe code |
| `xprobe.py` | Probes per feature set with the pilot's recipe, val evaluation with the head applied on the fly, rows and paired differences (Runs A and C only) |
| `stepA_head_probes.py` | Run A |
| `stepB_val_cosine.py` | Run B, the only place val images are used. It imports no probe code and checks this at the end |
| `stepC1_uniform_student.py` | Run C, part 1: the t_mean targets, the pre-training checks, the training (checkpoints every 30 min, resumes) |
| `stepC2_evaluate.py` | Run C, part 2: evaluation, the comparison with the attention student, the second-seed proposal |
| `step_report.py` | `reports/metrics.json` (every number) and `reports/distillation_diagnostics_report.pdf` |
| `notebooks/exp3_ab_diagnostics.ipynb` | Kaggle notebook 1: Runs A and B, then the report (preliminary) |
| `notebooks/exp3_c_uniform_teacher.ipynb` | Kaggle notebook 2: Run C, then the report over all three runs |

## Checks each run records

- **A:**
  - h is the head in Experiment 2's final `checkpoint.pt` (its SHA-256 is recorded). Its encoder weights equal
    `student.pth`, and it is at step 24,000 of 24,000.
  - Reproduction (`config.json` → `check_rules`). f(p) is recomputed now on every val scene and passed through
    Experiment 2's own saved probes; this gives A1′. The check passes if:
    - every scene is encoded in as many tiles as in Experiment 2;
    - |A1′ − A1| is no larger than A1's standard deviation over probe seeds, at top-1 and at top-3.

    The share of confusion counts that moved is reported too. Identical counts are not required: the GPU's
    sparse-convolution kernels are chosen per session (see decisions.md E3).
  - Feature widths (1232, 1024, 2256).
  - Experiment 2's image-free check.
- **B:**
  - The val observations pass step 1's checks: their mean equals `lift()`, pixels re-project, ids are unique and
    cameras distinct.
  - No probe module is imported.
  - The probe code refuses features stored under `runB/`.
- **C:**
  - The configuration diff with the attention student is exactly `["target"]`.
  - t_mean equals the pilot's `lift()` recomputed from the images (relative difference < 5e-3, cosine > 0.999; the
    stored observations are fp16).
  - One update changes an encoder parameter (encoder gradient norm > 0).
  - Experiment 2's image-free check on the evaluation.

## Running on Kaggle

The wheelhouse needs the pinned (original) Kaggle image, and a new notebook gets a newer one. So **import each
notebook into an existing pinned notebook** (open it → File → Import Notebook → upload the `.ipynb`):

- notebook 1 (`exp3_ab_diagnostics.ipynb`) → `pilot_01_encoders`
- notebook 2 (`exp3_c_uniform_teacher.ipynb`) → `pilot_03_probes`

Do not import into `pilot_04_eval`, or into the notebook that ran Experiment 2's full run (`pilot_02_concerto_lift`,
possibly renamed `exp2_cross_attention`). Their latest outputs are inputs here (the pilot rows and Experiment 2's full
run).

**Settings (both):** GPU T4 x2, Internet on, secret `GITHUB_TOKEN`, "Pin to original environment" kept.

**Inputs:**

| Input | Notebook | Why |
|---|---|---|
| `appearance-pt-wheels` | 1, 2 | the offline environment |
| `scannetpp-core-train-a`, `-train-b`, `scannetpp-core-val` | 1, 2 | point clouds, labels; DSLR views (Run B's observations, Run C's lift check) |
| `pilot_04_eval` output | 1, 2 | the pilot's frozen Sonata and Sonata + DINOv2 rows |
| the output of the notebook that ran Experiment 2's full run (8 Nov, about 3.6 h; its Output tab shows `Experiment2_CrossAttention/` with `student/`, `teacher/`, `observations/`, `evaluation/`, `reports/`) | 1, 2 | student, head, teacher, observations, evaluation |
| notebook 1's output (same `RUN` mode) | 2 | Run A's confusion matrices (A2/A3 comparisons); the report then covers A, B and C |
| the notebook's own earlier output | (resume only) | Run C's training resumes from its last checkpoint |

**Order:**
1. Notebook 1 smoke, then paste the SUMMARY blocks back.
2. Notebook 1 full, about 1.5 h.
3. Notebook 2 smoke.
4. Notebook 2 full, about 3.3 h (training 2.6 h, as Experiment 2's step 3). This is over 2 h, so it needs the go-ahead.

Each step prints a `SUMMARY … (paste this block back)` block. The logs are in `/kaggle/working/logs/exp3_*.log`. The
outputs are in `/kaggle/working/Experiment3_DistillationDiagnostics/`:
- `runA/`: summary, confusion matrices, probes.
- `runB/`: summary, per-point cosines, `val_observations/` (about 6–8 GB in the full run), the B2 head.
- `runC/`: the uniform student, its training log and checks, summary, confusion matrices.
- `reports/`.

A full run refuses Experiment 2 outputs that are not its full final run, and never mixes smoke and full outputs.
