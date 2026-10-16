"""Shared helpers for the M1 smoke checks (scripts/setup/smoke/*)."""

from __future__ import annotations

import numpy as np

SCANNET20_NAMES = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain", "toilet", "sink",
    "bathtub", "otherfurniture",
)


def load_demo_sample(name: str = "sample1") -> dict:
    """Pointcept's public demo scene (a ScanNet room with segment20 labels), fetched from HF.

    Same file as sonata.data.load("sample1"); downloaded at run time, never stored in the repo.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id="Pointcept/demo", filename=f"{name}.npz", repo_type="dataset")
    return dict(np.load(path))


def voxel_downsample(coord: np.ndarray, grid_size: float) -> tuple[np.ndarray, np.ndarray]:
    """First point per voxel. Returns (kept indices, integer grid coords of kept points)."""
    grid = np.floor((coord - coord.min(0)) / grid_size).astype(np.int64)
    _, keep = np.unique(grid, axis=0, return_index=True)
    keep = np.sort(keep)
    return keep, grid[keep]


def confusion(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_index: int = -1) -> np.ndarray:
    valid = (gt != ignore_index) & (gt >= 0) & (gt < num_classes)
    idx = gt[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    return np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def semseg_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_index: int = -1) -> dict:
    """Point accuracy and mIoU over classes that occur in GT or prediction."""
    cm = confusion(pred, gt, num_classes, ignore_index)
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - tp
    present = union > 0
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=present)
    return {
        "acc": float(tp.sum() / max(cm.sum(), 1)),
        "miou": float(iou[present].mean()) if present.any() else 0.0,
        "n_classes_present": int(present.sum()),
        "iou": iou.tolist(),
    }


def cuda_peak_gb() -> float:
    import torch

    return round(torch.cuda.max_memory_allocated() / 2**30, 3)
