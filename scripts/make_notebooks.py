"""Generate the Kaggle notebooks in notebooks/ from the scripts they wrap.

Notebooks are thin: a settings checklist, one config cell, the embedded bootstrap
(scripts/setup/kaggle_bootstrap.py) and one entry point per step. All logic lives in
versioned .py files; tests/test_notebooks.py fails if a notebook is out of date.

Usage: python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = (ROOT / "scripts/setup/kaggle_bootstrap.py").read_text()

CONFIG = '''\
# ---- run configuration: the only cell to edit ----
REPO = "eylulpelinkilic/appearance-pt"  # your private GitHub repo (owner/name)
REF = "main"                  # branch, tag or commit SHA; the resolved commit is logged
'''

CLONE = BOOTSTRAP + '''

CODE_DIR, COMMIT = clone_repo(REPO, REF)
PYTHONPATH = {"PYTHONPATH": f"{CODE_DIR}/src"}
'''

OPEN3D_CPU = '''\
# CPU session: Kaggle's CPU image has its own torch, so the GPU wheelhouse does not apply. ScanNet++
# preprocessing needs only open3d on top: the wheelhouse's pinned version, from PyPI.
run_logged([sys.executable, "-m", "pip", "install", "--quiet", "open3d==0.20.0"],
           "/kaggle/working/logs/{log}", cwd=CODE_DIR)
'''

PIP_CPU = '''\
# CPU session: small extra packages from PyPI (lz4 for depth.bin; imageio-ffmpeg in case the image has no ffmpeg)
run_logged([sys.executable, "-m", "pip", "install", "--quiet", {packages}],
           "/kaggle/working/logs/{log}", cwd=CODE_DIR)
'''

PILOT_PATH = '''\
# pilot scripts import the Utonia, Sonata and Concerto inference packages (pinned submodules) and our src/
PILOT_PYTHONPATH = {"PYTHONPATH": ":".join(f"{CODE_DIR}/{p}" for p in ("src", "third_party/utonia", "third_party/sonata",
                                                                      "third_party/concerto"))}
'''

PILOT_CHOICES = '''\
# choices from the pilot_02 run of 2026-10-26 11:47 UTC (decisions.md P6, results)
OCCLUSION = "v2"  # z-buffer variant (lifting.OCCLUSION): best F1 against the mesh in all three val scenes
CONCERTO = "concerto"  # concerto_large_origin: healthy in the half-room probe; "concerto_prerelease" is the alternative
'''

FIND_PROBES = '''\
# the step-4 probes: probes.pt from a pilot_03_probes output, or uploaded on its own as a private dataset
import pathlib
found = sorted(pathlib.Path("/kaggle/input").rglob("probes.pt"), key=lambda p: "step4" not in str(p))
assert found, "attach a dataset or notebook output that contains probes.pt (pilot_concat/step4/ of pilot_03_probes)"
PROBES = str(found[0])
print(f"{len(found)} probes.pt found; using {PROBES}")
'''

SHOW_IMAGES = '''\
# the images (ScanNet++-derived: they stay in this private notebook)
from IPython.display import Image as ShowImage, display
for png in sorted(pathlib.Path("{folder}").glob("*.png")):
    print(png.name)
    display(ShowImage(filename=str(png)))
'''

SETTINGS = {
    "gpu": "**Accelerator:** GPU T4 x2",
    "cpu": "**Accelerator:** None (CPU only; saves GPU quota)",
    "dlconfig": "**Input:** private dataset `scannetpp-download-config` (the official `download_scannetpp.yml` "
                "with the token line set back to `<YOUR_TOKEN_HERE>`)",
    "alldata": "**Input:** all eight private ScanNet++ datasets: `scannetpp-core-val`, `scannetpp-core-train-a`, "
               "`scannetpp-core-train-b`, `scannetpp-iphone-val` and `scannetpp-iphone-train-1` … `-train-4`",
    "coreval": "**Input:** private dataset `scannetpp-core-val`",
    "coreall": "**Inputs:** private datasets `scannetpp-core-val`, `scannetpp-core-train-a` and `scannetpp-core-train-b`",
    "probes": "**Input:** `probes.pt` from a finished `pilot_03_probes` run (its `pilot_concat/step4/` folder): attach that "
              "notebook's output (Add Input → Your Work → Notebooks), or upload the file as a private dataset and attach it",
    "coretrain": "**Inputs (optional, for the vertex-order check):** private datasets `scannetpp-core-train-a` and "
                 "`scannetpp-core-train-b`",
    "internet": "**Internet:** on",
    "github": "**Secrets** (Add-ons → Secrets): `GITHUB_TOKEN` attached",
    "both_secrets": "**Secrets** (Add-ons → Secrets): `GITHUB_TOKEN` and `SCANNETPP_TOKEN` attached",
    "pin": "**Environment:** pin to the same image as the wheelhouse (Settings → Environment preference); "
           "install_env.py refuses a mismatched image",
    "wheels": "**Input:** private dataset `appearance-pt-wheels` (output of m1_01)",
}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def run_step(script: str, args: str, log: str, env: str = "PYTHONPATH", keep_going: bool = False) -> dict:
    call = f'''run_logged([sys.executable, "-u", "{script}"{args}],
           "/kaggle/working/logs/{log}", cwd=CODE_DIR, extra_env={env})'''
    if keep_going:  # a failure is reported, and the next step still runs
        call = "try:\n    " + call.replace("\n", "\n    ") + "\nexcept RuntimeError as e:\n    print(f\"!! {e}; going on with the next step\")"
    return code(call)


def header(title: str, purpose: str, settings: list[str], run: str, paste: str) -> dict:
    checklist = "\n".join(f"- [ ] {SETTINGS[s]}" for s in settings)
    return md(f"""
# {title}

{purpose}

**Before running**
{checklist}

**Run:** {run}

**Paste back:** {paste}

*Generated by `scripts/make_notebooks.py`; edit the scripts, not this notebook.*
""")


def notebook(cells: list[dict]) -> dict:
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True, "language": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build() -> dict[str, dict]:
    """Notebooks by repo-relative path."""
    env_report = (ROOT / "scripts/setup/env_report.py").read_text()
    return {
        "notebooks/m1_00_env_report.ipynb": notebook([
            header(
                "M1-00 · Kaggle environment report",
                "Measures the platform before anything is installed: GPU, driver/CUDA, compilers, preinstalled "
                "PyTorch (incl. sm_75 support, SDPA backends, torch.compile), disk, network and secrets. "
                "Installs nothing and does not need the repo.",
                ["gpu", "internet", "both_secrets"],
                "Run All (interactive is fine, ~3 min).",
                "the SUMMARY block at the end (full JSON: /kaggle/working/m1_env_report.json).",
            ),
            code('import os\nos.environ["APPT_GITHUB_REPO"] = "eylulpelinkilic/appearance-pt"  # optional: checks token access to the repo'),
            code(env_report),
        ]),
        "notebooks/m1_01_build_wheels.ipynb": notebook([
            header(
                "M1-01 · Build the offline wheelhouse",
                "Builds every package the image lacks against its preinstalled torch (spconv, torch-scatter/cluster, "
                "pointops compiled for sm_75, SharedArray, …) into `/kaggle/working/wheelhouse`, then installs from it "
                "offline as a check. Afterwards create a **private** dataset `appearance-pt-wheels` from this version's output.",
                ["gpu", "internet", "github", "pin"],
                "Save Version → Save & Run All (Commit), so the output persists (~15–30 min).",
                "/kaggle/working/logs/m1_01_build_wheels.log (or the version log).",
            ),
            code(CONFIG),
            code(CLONE),
            run_step("scripts/setup/build_wheels.py", ', "--out", "/kaggle/working/wheelhouse"', "m1_01_build_wheels.log"),
        ]),
        "notebooks/m1_02_smoke_test.ipynb": notebook([
            header(
                "M1-02 · Offline install + GPU smoke test",
                "Installs the wheelhouse offline, then checks: compiled extensions, ScanNet++ toolbox import, the Sonata "
                "demo (semantic segmentation of Pointcept's demo room, fp32 and fp16), PTv3 training steps "
                "(Pointcept, ScanNet++ config, no flash-attn), DITR image-injection training steps, and 2-GPU DDP. "
                "Internet is still needed for the Sonata/DINOv2 weights and the demo room (Hugging Face).",
                ["gpu", "internet", "github", "pin", "wheels"],
                "Save Version → Save & Run All (Commit) (~15–25 min).",
                "the SUMMARY table (full log: /kaggle/working/logs/m1_02_smoke_test.log; figure: "
                "/kaggle/working/smoke/sonata_semseg_topdown.png).",
            ),
            code(CONFIG),
            code(CLONE),
            run_step("scripts/setup/install_env.py", "", "m1_02_smoke_test.log"),
            run_step("scripts/setup/smoke_test.py", "", "m1_02_smoke_test.log"),
        ]),
        "notebooks/m2_00_scannetpp_sizes.ipynb": notebook([
            header(
                "M2-00 · ScanNet++ subset and asset sizes",
                "Downloads only the ScanNet++ metadata (split and class lists), fixes the scene subset (all 50 val "
                "scenes + 200 random train scenes, seed 0) and asks the server for the size of every candidate asset "
                "of those scenes: mesh + labels, undistorted DSLR images + poses, iPhone data. No scene data is "
                "downloaded. The token comes from Kaggle Secrets and never appears in the output.",
                ["cpu", "internet", "both_secrets", "dlconfig"],
                "Save Version → Save & Run All (Commit) (about 20–40 min: one request per file, paced for the server).",
                "the SUMMARY block and the SUBSET line at the end (full log: "
                "/kaggle/working/logs/m2_00_scannetpp_sizes.log; per-file sizes: /kaggle/working/scannetpp_sizes/sizes.csv).",
            ),
            code(CONFIG),
            code(CLONE),
            run_step("scripts/scannetpp/scannetpp_sizes.py", "", "m2_00_scannetpp_sizes.log"),
        ]),
        "notebooks/m2_01_scannetpp_trial.ipynb": notebook([
            header(
                "M2-01 · ScanNet++ trial download (5 scenes)",
                "Downloads the core assets (mesh + labels; undistorted DSLR images, masks, poses) of 3 val scenes and "
                "of the two subset scenes with unusual IDs, into temporary storage. Measures download speed (sequential "
                "and parallel), unpacked size, the size after Pointcept's ScanNet++ preprocessing and the size we would "
                "keep; reports DSLR image count/resolution, the pose-file fields (bad-frame flag?) and the anonymisation "
                "mask format. Deletes every scene afterwards; only metrics are saved. About 2 GB is downloaded.",
                ["cpu", "internet", "both_secrets", "dlconfig"],
                "Save Version → Save & Run All (Commit) (roughly 10–30 min, depending on download speed).",
                "the SUMMARY block at the end (full log: /kaggle/working/logs/m2_01_scannetpp_trial.log; "
                "per-scene details: /kaggle/working/scannetpp_trial/trial.json).",
            ),
            code(CONFIG),
            code(CLONE),
            code(OPEN3D_CPU.format(log="m2_01_scannetpp_trial.log")),
            run_step("scripts/scannetpp/scannetpp_trial.py", "", "m2_01_scannetpp_trial.log"),
        ]),
        "notebooks/m2_06_scannetpp_faces.ipynb": notebook([
            header(
                "M2-06 · Train-scene mesh triangles, and sizes for the next phase (before the token expires 2026-11-01)",
                "Downloads `mesh_aligned_0.05.ply` of the 200 train scenes, keeps only its triangles per scene "
                "(`faces/<scene>.npz`, checked to index the same vertices as Pointcept's `coord.npy`) and deletes the "
                "rest; this gives exact visibility and depth rendering for DSLR views later. Then asks the server for "
                "the sizes of the dense laser scans, the full-resolution DSLR images and the panorama assets of the 250 "
                "subset scenes (no download). Afterwards create the PRIVATE dataset `scannetpp-train-faces` from the "
                "output. Runs on CPU (no GPU quota); it can run while `pilot_03` runs.",
                ["cpu", "internet", "both_secrets", "dlconfig", "coretrain"],
                "Save Version → Save & Run All (Commit) (expected about 30–60 min).",
                "the SUMMARY block at the end (full log: /kaggle/working/logs/m2_06_scannetpp_faces.log; per-scene "
                "manifest: /kaggle/working/scannetpp_faces/manifest.json; sizes: .../sizes.csv).",
            ),
            code(CONFIG),
            code(CLONE),
            code(OPEN3D_CPU.format(log="m2_06_scannetpp_faces.log")),
            run_step("scripts/scannetpp/scannetpp_faces.py", "", "m2_06_scannetpp_faces.log"),
        ]),
        "notebooks/m2_02_scannetpp_core.ipynb": notebook([
            header(
                "M2-02 · ScanNet++ core download, one part of the subset",
                "Downloads mesh + labels and the undistorted DSLR data of one part of the 250-scene subset, runs "
                "Pointcept's ScanNet++ preprocessing and keeps, per scene, one tar in `/kaggle/working/scannetpp_core/"
                "scenes/`: Pointcept arrays, anonymisation mask, a pool of 60 good DSLR views (+ masks, poses), and for "
                "val scenes the raw mesh + annotations for the official ground truth. Raw files are deleted per scene. "
                "Import this notebook three times (one per part) and set PART in each; afterwards create one PRIVATE "
                "dataset per output: `scannetpp-core-val`, `scannetpp-core-train-a`, `scannetpp-core-train-b`.",
                ["cpu", "internet", "both_secrets", "dlconfig"],
                "Save Version → Save & Run All (Commit). Expected: val ≈ 0.5 h, each train part ≈ 1–2 h.",
                "the SUMMARY block at the end (full log: /kaggle/working/logs/m2_02_scannetpp_core.log; per-scene "
                "manifest: /kaggle/working/scannetpp_core/manifest.json).",
            ),
            code(CONFIG),
            code('# ---- which part of the scene subset this run downloads (one notebook copy per part) ----\n'
                 'PART = "val"   # "val" (50 scenes), "train-a" (100), "train-b" (100)'),
            code(CLONE),
            code(OPEN3D_CPU.format(log="m2_02_scannetpp_core.log")),
            run_step("scripts/scannetpp/scannetpp_fetch.py", ', "--part", PART', "m2_02_scannetpp_core.log"),
        ]),
        "notebooks/m2_03_scannetpp_iphone_trial.ipynb": notebook([
            header(
                "M2-03 · ScanNet++ iPhone trial (3 scenes)",
                "Downloads the iPhone data (RGB and mask videos, depth.bin, ARKit poses, COLMAP) of 3 val scenes into "
                "temporary storage and reports what it contains: video resolution/fps/frames, the depth encoding, the "
                "pose-file fields, the COLMAP-registered frames. Extracts a pool of 60 frames (RGB, mask, depth) spread "
                "over the registered frames, times it and measures what we would keep. Deletes everything; only metrics "
                "are saved. About 2–4 GB is downloaded. Can run at the same time as the m2_02 train parts.",
                ["cpu", "internet", "both_secrets", "dlconfig"],
                "Save Version → Save & Run All (Commit) (roughly 10–20 min).",
                "the SUMMARY block at the end (full log: /kaggle/working/logs/m2_03_scannetpp_iphone_trial.log).",
            ),
            code(CONFIG),
            code(CLONE),
            code(PIP_CPU.format(packages='"lz4", "imageio-ffmpeg"', log="m2_03_scannetpp_iphone_trial.log")),
            run_step("scripts/scannetpp/scannetpp_iphone_trial.py", "", "m2_03_scannetpp_iphone_trial.log"),
        ]),
        "notebooks/m2_04_scannetpp_iphone.ipynb": notebook([
            header(
                "M2-04 · ScanNet++ iPhone download, one part of the subset",
                "Downloads the iPhone data of one part of the 250-scene subset and keeps, per scene, one tar in "
                "`/kaggle/working/scannetpp_iphone/scenes/`: 60 frames spread over the COLMAP-registered frames (RGB "
                "JPG, mask PNG, depth PNG in mm), the full ARKit pose file and COLMAP cameras/images (gzipped). Raw "
                "files are deleted per scene; the next scene downloads while the current one decodes. Import this "
                "notebook five times (one per part), set PART in each, run them side by side; afterwards create one "
                "PRIVATE dataset per output: `scannetpp-iphone-val`, `scannetpp-iphone-train-1` … `-train-4`.",
                ["cpu", "internet", "both_secrets", "dlconfig"],
                "Save Version → Save & Run All (Commit). Expected ≈ 1.5–3 h per part (decoding the 60 fps videos "
                "is the slow step).",
                "the SUMMARY block at the end (full log: /kaggle/working/logs/m2_04_scannetpp_iphone.log; per-scene "
                "manifest: /kaggle/working/scannetpp_iphone/manifest.json).",
            ),
            code(CONFIG),
            code('# ---- which part of the scene subset this run downloads (one notebook copy per part) ----\n'
                 'PART = "val"   # "val", "train-1", "train-2", "train-3", "train-4" (50 scenes each)'),
            code(CLONE),
            code(PIP_CPU.format(packages='"lz4", "imageio-ffmpeg"', log="m2_04_scannetpp_iphone.log")),
            run_step("scripts/scannetpp/scannetpp_iphone_fetch.py", ', "--part", PART', "m2_04_scannetpp_iphone.log"),
        ]),
        "notebooks/m2_05_scannetpp_verify.ipynb": notebook([
            header(
                "M2-05 · Check the saved ScanNet++ datasets",
                "Reads every scene archive in the eight private ScanNet++ datasets and decodes every file in it: "
                "Pointcept arrays, anonymisation mask, val raw mesh and annotations, every DSLR view and mask, every "
                "iPhone frame, mask and depth map, the poses and the COLMAP image names. Sizes and counts must match the "
                "download manifests. Downloads nothing and needs no ScanNet++ token; run it before the token expires "
                "(2026-11-01), so a damaged scene can still be downloaded again.",
                ["cpu", "internet", "github", "alldata"],
                "Save Version → Save & Run All (Commit) (expected 20–60 min: about 32 GB is read and decoded).",
                "the SUMMARY block at the end (full log: /kaggle/working/logs/m2_05_scannetpp_verify.log; per-scene "
                "numbers: /kaggle/working/scannetpp_verify/verify.json).",
            ),
            code(CONFIG),
            code(CLONE),
            run_step("scripts/scannetpp/scannetpp_verify.py", "", "m2_05_scannetpp_verify.log"),
        ]),
        "experiments/pilot_concat/notebooks/pilot_01_encoders.ipynb": notebook([
            header(
                "Pilot · Step 1 · Frozen point encoders on a T4",
                "Loads Utonia, Sonata and an untrained PTv3 (Utonia's architecture) without FlashAttention and checks, "
                "on Pointcept's public demo room: stored configs and feature widths, Utonia's input scale (with its "
                "released ScanNet probe), SDPA vs the original attention, fp16 vs fp32, run-to-run variation, zeroed "
                "colour/normals, colour range [0, 1] vs [-1, 1]. Then measures time and peak GPU memory per scene against scene size. Public data "
                "only (Hugging Face): no ScanNet++ data, no `SCANNETPP_TOKEN`.",
                ["gpu", "internet", "github", "pin", "wheels"],
                "Save Version → Save & Run All (Commit) (expected well under 1 h; the largest tiled scenes are "
                "allowed to run out of memory, that is part of the measurement).",
                "the SUMMARY block at the end (full log: /kaggle/working/logs/pilot_01_encoders.log; results: "
                "/kaggle/working/pilot_concat/step1/step1_summary.json).",
            ),
            code(CONFIG),
            code(CLONE),
            code(PILOT_PATH),
            run_step("scripts/setup/install_env.py", "", "pilot_01_encoders.log"),
            run_step("experiments/pilot_concat/step1_encoders.py", "", "pilot_01_encoders.log", env="PILOT_PYTHONPATH"),
        ]),
        "experiments/pilot_concat/notebooks/pilot_02_concerto_lift.ipynb": notebook([
            header(
                "Pilot · Step 1b (Concerto) and Step 2 (DINO lifted to points)",
                "**Step 1b:** loads Concerto (`concerto_large_origin`, the original model; and the pre-release "
                "`concerto_large`) without FlashAttention; compares the original with Pointcept's training checkpoint "
                "`pretrain-concerto-v1m1-2-large-video.pth` (8.3 GB, read so that none of its code runs, then deleted); "
                "scores the released ScanNet-20 probe on Pointcept's public demo room; trains a half-room linear probe "
                "for every encoder (a health check that needs no released head); compares SDPA with the original "
                "attention, fp16 with fp32, seeds and zeroed inputs. **Step 2:** on 3 val scenes, projects every vertex "
                "into 30 DSLR views, checks the pose convention against the photos, scores the occlusion test against "
                "the mesh, lifts DINOv2 ViT-L/14 (with registers) patch features onto the vertices, measures how "
                "consistent the views are, and saves the visual checks (shown at the end). No `SCANNETPP_TOKEN` needed.",
                ["gpu", "internet", "github", "pin", "wheels", "coreval"],
                "Save Version → Save & Run All (Commit) (expected about 10–15 min).",
                "both SUMMARY blocks (step 1b and step 2; full logs: /kaggle/working/logs/pilot_02_*.log), and look at "
                "the images at the end of the notebook (files: /kaggle/working/pilot_concat/step2/*.png).",
            ),
            code(CONFIG),
            code(CLONE),
            code(PILOT_PATH),
            run_step("scripts/setup/install_env.py", "", "pilot_02_install.log"),
            # --tiles with no values: time and memory per scene were measured in the first run (decisions.md P4)
            run_step("experiments/pilot_concat/step1b_concerto.py", ', "--tiles"', "pilot_02_step1b.log",
                     env="PILOT_PYTHONPATH", keep_going=True),
            run_step("experiments/pilot_concat/step2_lift.py", "", "pilot_02_step2.log", env="PILOT_PYTHONPATH"),
            code(SHOW_IMAGES.format(folder="/kaggle/working/pilot_concat/step2")),
        ]),
        "experiments/pilot_concat/notebooks/pilot_03_probes.ipynb": notebook([
            header(
                "Pilot · Steps 3–5 (training features, linear probes, evaluation on val)",
                "**Step 3:** for the 200 train scenes, features of up to 10k labelled vertices each from Utonia, "
                "Sonata, Concerto, an untrained PTv3, lifted DINOv2 and the raw inputs, written to the session's temp "
                "disk (~27 GB; nothing data-derived is kept). **Step 4:** a linear probe per configuration (A, B, C, "
                "C-eq, S, S+, K, K+, K+-eq, R, R+, X): learning rate chosen on 20 held-out train scenes, then 3 seeds on "
                "all 200. **Step 5:** every probe on all 50 val scenes with the official ScanNet++ metric (top-1/top-3 "
                "mIoU, per class, seen vs never-seen vertices, bootstrap over scenes), cross-checked against the "
                "toolbox's own eval_semantic. The two choices from pilot_02 are set in the first code cell. "
                "No `SCANNETPP_TOKEN` needed.",
                ["gpu", "internet", "github", "pin", "wheels", "coreall"],
                "Save Version → Save & Run All (Commit) (expected about 2.5–3 h; approved by the user on 2026-10-26).",
                "the three SUMMARY blocks (step 3, 4, 5; full logs: /kaggle/working/logs/pilot_03_*.log).",
            ),
            code(CONFIG),
            code(CLONE),
            code(PILOT_PATH),
            code(PILOT_CHOICES),
            run_step("scripts/setup/install_env.py", "", "pilot_03_install.log"),
            run_step("experiments/pilot_concat/step3_features.py", ', "--occlusion", OCCLUSION, "--concerto", CONCERTO',
                     "pilot_03_step3.log", env="PILOT_PYTHONPATH"),
            run_step("experiments/pilot_concat/step4_probes.py", "", "pilot_03_step4.log", env="PILOT_PYTHONPATH"),
            run_step("experiments/pilot_concat/step5_eval.py", "", "pilot_03_step5.log", env="PILOT_PYTHONPATH"),
        ]),
        "experiments/pilot_concat/notebooks/pilot_04_eval.ipynb": notebook([
            header(
                "Pilot · Step 5 only (evaluation on val), with the probes of a finished pilot_03 run",
                "Runs step 5 (every probe on the 50 val scenes, official ScanNet++ metric, cross-check with the "
                "toolbox's eval_semantic) from the `probes.pt` that `pilot_03_probes` saved, without redoing steps 3 "
                "and 4. No `SCANNETPP_TOKEN` needed.",
                ["gpu", "internet", "github", "pin", "wheels", "coreval", "probes"],
                "Save Version → Save & Run All (Commit) (expected about 45–60 min).",
                "the SUMMARY block of step 5 (full log: /kaggle/working/logs/pilot_04_step5.log; results: "
                "/kaggle/working/pilot_concat/step5/results.json).",
            ),
            code(CONFIG),
            code(CLONE),
            code(PILOT_PATH),
            code(FIND_PROBES),
            run_step("scripts/setup/install_env.py", "", "pilot_04_install.log"),
            run_step("experiments/pilot_concat/step5_eval.py", ', "--probes", PROBES', "pilot_04_step5.log",
                     env="PILOT_PYTHONPATH"),
        ]),
        "experiments/pilot_concat/notebooks/pilot_05_visuals.ipynb": notebook([
            header(
                "Pilot · Segmentation maps (what the probes predict on val scenes)",
                "The pilot's task is semantic segmentation: every vertex of a val scene's mesh gets one of 100 classes. "
                "This draws the predictions behind step 5's numbers on 4 val scenes (a large gain from DINO, about the "
                "average gain, many never-seen vertices, and the largest loss): the photo, the ground truth, DINO only "
                "(A), Concerto (K), Concerto + DINO (K+) and where K and K+ differ, for two camera views per scene and "
                "from above. The seed-0 predictions are recomputed as in step 5, and each scene's mIoU is checked "
                "against step 5. The images are ScanNet++-derived, so they stay in this private notebook. "
                "No `SCANNETPP_TOKEN` needed.",
                ["gpu", "internet", "github", "pin", "wheels", "coreval", "probes"],
                "Save Version → Save & Run All (Commit) (expected about 15–20 min).",
                "the SUMMARY block (full log: /kaggle/working/logs/pilot_05_visuals.log), and look at the images at the "
                "end of the notebook (files: /kaggle/working/pilot_concat/visuals/*.png).",
            ),
            code(CONFIG),
            code(CLONE),
            code(PILOT_PATH),
            code(FIND_PROBES),
            run_step("scripts/setup/install_env.py", "", "pilot_05_install.log"),
            run_step("experiments/pilot_concat/step6_visuals.py", ', "--probes", PROBES', "pilot_05_visuals.log",
                     env="PILOT_PYTHONPATH"),
            code(SHOW_IMAGES.format(folder="/kaggle/working/pilot_concat/visuals")),
        ]),
    }


def render(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main() -> None:
    for path, nb in build().items():
        (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / path).write_text(render(nb))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
