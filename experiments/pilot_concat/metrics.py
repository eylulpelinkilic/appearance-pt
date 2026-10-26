"""Official ScanNet++ semantic metric, on the GPU, and a paired bootstrap over scenes.

confusion_topk reproduces the toolbox's fast_hist_topk_multilabel (semantic/utils/confmat.py): a vertex counts if it
has at least one GT label in [0, num_classes); the first of its top-k predictions that matches any of its GT labels is
paired with the first GT label it matches; a vertex with no match pairs its top-1 prediction with its first GT label.
mIoU is the toolbox's: per-class IoU from the summed confusion matrix, NaN (skipped) for classes absent from both.
tests/test_pilot_metrics.py checks equality with the toolbox's own function.
"""

from __future__ import annotations

import numpy as np
import torch

IGNORE = -100  # the toolbox's ignore label


def confusion_topk(top_preds: torch.Tensor, multilabel: torch.Tensor, num_classes: int, ignore_label: int = IGNORE) -> torch.Tensor:
    """(num_classes, num_classes) int64 confusion matrix (rows GT, columns prediction) of top_preds (n, k) against
    multilabel GT (n, m)."""
    valid = ((multilabel != ignore_label).sum(1) > 0) & (((multilabel >= 0) & (multilabel < num_classes)).sum(1) > 0)
    p, g = top_preds[valid].long(), multilabel[valid].long()
    matches = p[:, :, None] == g[:, None, :]  # (n, k, m)
    hit = matches.any(2)  # (n, k)
    has_hit = hit.any(1)
    rows = torch.arange(len(p), device=p.device)
    first_pred = hit.int().argmax(1)  # first matching prediction (0 if none)
    first_gt = matches[rows, first_pred].int().argmax(1)  # first GT label it matches
    pred = torch.where(has_hit, p[rows, first_pred], p[:, 0])
    gt = torch.where(has_hit, g[rows, first_gt], g[:, 0])
    assert len(gt) == 0 or (int(gt.min()) >= 0 and int(pred.min()) >= 0)
    return torch.bincount(num_classes * gt + pred, minlength=num_classes**2).view(num_classes, num_classes)


def ious(cm: np.ndarray) -> np.ndarray:
    """Per-class IoU of one confusion matrix (C, C) or a batch (..., C, C); NaN where a class has no GT and no
    prediction."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    union = cm.sum(-1) + cm.sum(-2) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        return tp / union


def miou(cm: np.ndarray) -> np.ndarray:
    return np.nanmean(ious(cm), axis=-1)


def bootstrap_diff(cms_a: np.ndarray, cms_b: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """Paired bootstrap over scenes of mIoU(a) - mIoU(b). cms_*: (seeds, scenes, C, C) per-scene confusion matrices,
    the same scenes in the same order; each resample draws scenes with replacement, sums their matrices per seed and
    averages mIoU over seeds. Returns the observed difference, the 2.5 / 97.5 percentiles and P(diff <= 0)."""
    rng = np.random.default_rng(seed)
    n_scenes = cms_a.shape[1]
    weights = np.stack([np.bincount(rng.integers(0, n_scenes, n_scenes), minlength=n_scenes) for _ in range(n)])  # (n, S)

    def resampled(cms):
        flat = cms.reshape(cms.shape[0], n_scenes, -1).astype(np.float64)  # (seeds, S, C*C)
        summed = np.einsum("rs,ksc->krc", weights, flat).reshape(cms.shape[0], n, *cms.shape[2:])
        return miou(summed).mean(0)  # (n,)

    def observed(cms):
        return float(miou(cms.sum(1)).mean())

    diff = resampled(cms_a) - resampled(cms_b)
    return dict(diff=round(observed(cms_a) - observed(cms_b), 4), lo=round(float(np.percentile(diff, 2.5)), 4),
                hi=round(float(np.percentile(diff, 97.5)), 4), p_le_0=round(float((diff <= 0).mean()), 4))
