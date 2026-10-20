"""Environment record attached to every run (git commits, image, torch, GPUs, packages).

Only allowlisted environment variables are recorded: Kaggle sessions carry credentials
in the environment (e.g. the user-secrets token), so nothing is dumped wholesale.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import REPO_ROOT

SAFE_ENV_VARS = (
    "KAGGLE_KERNEL_RUN_TYPE",
    "KAGGLE_DOCKER_IMAGE",
    "KAGGLE_CONTAINER_NAME",
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "TORCH_CUDA_ARCH_LIST",
    "NCCL_P2P_DISABLE",
    "OMP_NUM_THREADS",
    "CUMM_DISABLE_JIT",
    "SPCONV_DISABLE_JIT",
    "CUMM_INCLUDE_PATH",
)

# Modules that must import after installing the wheelhouse (build_wheels.py, install_env.py).
IMPORT_CHECK_MODULES = (
    "torch", "spconv.pytorch", "torch_scatter", "torch_cluster", "torch_geometric", "pointops",
    "SharedArray", "open3d", "addict", "yapf", "termcolor", "tensorboardX", "einops", "timm",
    "peft", "transformers", "h5py", "imageio", "plyfile", "albumentations", "wandb", "regex",
    "huggingface_hub", "munch",
)

KEY_PACKAGES = (
    "torch", "torchvision", "triton", "numpy", "scipy", "spconv-cu118", "spconv-cu121",
    "spconv-cu124", "spconv-cu126", "cumm-cu126", "torch-scatter", "torch-cluster",
    "torch-geometric", "pointops", "timm", "transformers", "peft", "open3d", "SharedArray",
    "yapf", "addict", "einops", "albumentations", "huggingface-hub", "munch", "wandb",
)


def check_imports(modules=IMPORT_CHECK_MODULES) -> bool:
    """Import each module in a fresh interpreter (sees newly installed packages); print ok/FAIL."""
    code = (
        "import importlib, sys\nbad = []\n"
        f"for m in {list(modules)!r}:\n"
        "    try:\n        importlib.import_module(m); print(f'  ok    {m}')\n"
        "    except Exception as e:\n        bad.append(m); print(f'  FAIL  {m}: {type(e).__name__}: {e}')\n"
        "sys.exit(1 if bad else 0)\n"
    )
    return subprocess.run([sys.executable, "-c", code]).returncode == 0


def _run(args: list[str], cwd: Path | None = None) -> str:
    """stdout of a successful command, else "" (e.g. `git rev-parse HEAD` before the first commit)."""
    try:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def git_info(root: Path = REPO_ROOT) -> dict:
    """Commit of this repo (with dirty flag) and of every pinned submodule."""
    status = _run(["git", "status", "--porcelain", "--ignore-submodules=none"], cwd=root)
    submodules = {}
    for line in _run(["git", "submodule", "status"], cwd=root).splitlines():
        # "<flag><sha> <path> (<describe>)"; flag '+' = checked-out commit differs from pin
        parts = line[1:].split()
        if len(parts) >= 2:
            submodules[parts[1]] = {"commit": parts[0], "matches_pin": line[0] != "+"}
    return {
        "commit": _run(["git", "rev-parse", "HEAD"], cwd=root) or None,
        "dirty": bool(status),
        "submodules": submodules,
    }


def package_versions(names=KEY_PACKAGES) -> dict:
    out = {}
    for name in names:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return out


def torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False}
    info = {
        "available": True,
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "devices": [],
    }
    if torch.cuda.is_available():
        info["arch_list"] = torch.cuda.get_arch_list()
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {"name": p.name, "capability": f"{p.major}.{p.minor}", "memory_gb": round(p.total_memory / 2**30, 2)}
            )
    return info


def collect(root: Path = REPO_ROOT) -> dict:
    return {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "env": {k: os.environ[k] for k in SAFE_ENV_VARS if k in os.environ},
        "git": git_info(root),
        "torch": torch_info(),
        "packages": package_versions(),
    }
