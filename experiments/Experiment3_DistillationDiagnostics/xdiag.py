"""Experiment 3 (diagnostics for the Experiment 2 student): the pieces its steps share.

- Configuration (config.json next to this file, deep-merged per mode) and the output layout under
  /kaggle/working/Experiment3_DistillationDiagnostics (runA/, runB/, runC/, reports/).
- Finding Experiment 2's full-run output among the attached inputs, and failing early when an artifact is missing.
- The Experiment 2 student with its prediction head h, which Experiment 2 saved only in student/checkpoint.pt.
- Confusion-matrix helpers (per-class IoU) shared by every run.
Probes and their evaluation are in xprobe.py, which Run B never imports: Run B's val diagnostics must not reach a probe.

Imports nothing that loads images (lifting, pipeline, transformers): Runs A and C check that.
"""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from appearance_pt import REPO_ROOT
from encoders import load_encoder, make_transform
from metrics import ious
from xattn import PredictionHead, run_mode, walk_dirs

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"
OUT = Path("/kaggle/working/Experiment3_DistillationDiagnostics")
EXP2_CONFIG = REPO_ROOT / "experiments" / "Experiment2_CrossAttention" / "config.json"
EXP2_FILES = ("student/student.pth", "student/checkpoint.pt", "teacher/teacher.pt", "observations/summary.json",
              "evaluation/confmats.npz", "evaluation/probes.pt", "evaluation/results.json", "reports/metrics.json")
IMAGE_MODULES = ("lifting", "pipeline", "transformers")


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path = CONFIG, mode: str = "full") -> dict:
    cfg = json.loads(Path(path).read_text())
    return dict(cfg, run=dict(deep_merge(cfg["shared"], cfg["modes"][mode]), mode=mode))


# ---- inputs ------------------------------------------------------------------------------------------------------

def find_runs(inputs: Path, name: str) -> list[Path]:
    """Folders called `name` among the attached inputs (the pruned walk of Experiment 2)."""
    return [d for d in walk_dirs(inputs) if d.name == name] if Path(inputs).exists() else []


def find_exp2(inputs: Path, explicit: Path | None = None, any_status: bool = False) -> Path:
    """Experiment 2's full-run output (run.json mode "full", metrics status "final") with every file Experiment 3
    reuses; exits with the reason otherwise. `any_status` is for the CPU tests only."""
    candidates = [Path(explicit)] if explicit else find_runs(inputs, "Experiment2_CrossAttention")
    reasons = []
    for d in candidates:
        missing = [f for f in EXP2_FILES if not (d / f).exists()]
        status = json.loads((d / "reports/metrics.json").read_text()).get("status") if (d / "reports/metrics.json").exists() else None
        if missing:
            reasons.append(f"{d}: missing {missing}")
        elif not any_status and (run_mode(d) != "full" or status != "final"):
            reasons.append(f"{d}: a {run_mode(d)} run with status {status}, not the full final run")
        else:
            return d
    raise SystemExit("!! Experiment 2's full-run output is not attached or incomplete (add the notebook output of the "
                     "run with RUN = \"full\"): " + ("; ".join(reasons) or "no Experiment2_CrossAttention folder found"))


def find_pilot(inputs: Path, explicit: Path | None = None) -> Path:
    """The pilot's step-5 outputs (confmats.npz with S and S+, results.json)."""
    dirs = [Path(explicit)] if explicit else list(walk_dirs(inputs))
    for d in dirs:
        if (d / "confmats.npz").exists() and (d / "results.json").exists():
            with np.load(d / "confmats.npz") as z:
                if "S#0|top1" in z.files and "S+#0|top1" in z.files:
                    return d
    raise SystemExit("!! the pilot's step-5 outputs are not attached: add the pilot_04_eval notebook output")


def sha256_state(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def head_from_state(state: dict) -> PredictionHead:
    """Experiment 2's PredictionHead (LayerNorm, Linear, GELU, Linear) with its sizes read from the weights."""
    w1, w3 = state["net.1.weight"], state["net.3.weight"]
    head = PredictionHead(w1.shape[1], w3.shape[0], hidden=w1.shape[0])
    head.load_state_dict(state)
    return head.eval().requires_grad_(False)


def load_student_head(student_dir: Path, device: str):
    """(encoder, transform, head, facts) of a trained student: encoder weights from student.pth, the head from the final
    training checkpoint (the only place it was saved). Exits if the head is missing or the checkpoint is not the end
    of training or does not hold the same encoder weights."""
    saved = torch.load(Path(student_dir) / "student.pth", weights_only=False, map_location="cpu")
    ck = torch.load(Path(student_dir) / "checkpoint.pt", weights_only=False, map_location="cpu")
    if "head" not in ck:
        raise SystemExit(f"!! {student_dir}/checkpoint.pt has no prediction head: Run A/B cannot use h")
    if ck.get("step") != ck.get("steps"):
        raise SystemExit(f"!! {student_dir}/checkpoint.pt is at step {ck.get('step')} of {ck.get('steps')}, not the end of training")
    enc_hash, ck_hash = sha256_state(saved["state_dict"]), sha256_state(ck["model"])
    if enc_hash != ck_hash:
        raise SystemExit(f"!! {student_dir}: checkpoint.pt and student.pth hold different encoders")
    model = build_student(saved["state_dict"], device)
    head = head_from_state(ck["head"]).to(device)
    facts = dict(encoder_sha256=enc_hash, head_sha256=sha256_state(ck["head"]), step=ck["step"], steps=ck["steps"],
                 trainable=ck.get("trainable"), crop_points=ck.get("crop_points"),
                 head_shapes={k: list(v.shape) for k, v in ck["head"].items()})
    return model, make_transform("sonata"), head, facts


def build_student(state: dict, device: str):
    model, _ = load_encoder("sonata", seed=0)
    model.load_state_dict(state)
    return model.requires_grad_(False).to(device).eval()


def load_frozen_sonata(device: str):
    model, _ = load_encoder("sonata", seed=0)
    return model.requires_grad_(False).to(device).eval(), make_transform("sonata")


def head_features(head, f: torch.Tensor) -> torch.Tensor:
    """h(f) in float32 from the fp16 upcast features, as during training (the head casts to float)."""
    with torch.no_grad():
        return head(f.float())


def stack_config(source, config: str, kk: int, sel=None) -> np.ndarray:
    """(seeds, scenes, C, C) from a dict or npz with keys "<config>#<seed>|top<kk>"."""
    names = source.files if hasattr(source, "files") else list(source)
    keys = sorted((k for k in names if k.startswith(f"{config}#") and k.endswith(f"|top{kk}")),
                  key=lambda k: int(k.split("#")[1].split("|")[0]))
    assert keys, f"no confusion matrices for {config}"
    return np.stack([source[k] if sel is None else source[k][sel] for k in keys])


def class_iou(cms1: np.ndarray) -> np.ndarray:
    """Per-class top-1 IoU (%) of confusion matrices summed over scenes, mean over seeds (NaN for absent classes)."""
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns: classes absent from these scenes
        return 100 * np.nanmean(np.stack([ious(c.sum(0)) for c in cms1]), 0)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
