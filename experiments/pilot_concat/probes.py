"""Linear probes on frozen per-point features (pilot steps 1b and 4)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from appearance_pt.smoke import semseg_metrics


def split_probe(feat: torch.Tensor, gt: np.ndarray, x: np.ndarray, num_classes: int = 20, steps: int = 300,
                seed: int = 0) -> dict:
    """Per-point features (N, C): a linear head trained on the points on one side of the median x and scored on the
    other side, both ways (features standardised on the training half; AdamW, full batch). A health check of a
    feature set on one room, not a benchmark."""
    dev = feat.device
    y = torch.from_numpy(gt).to(dev)
    right = torch.from_numpy(x >= np.median(x)).to(dev)
    folds = []
    for train in (~right, right):
        tr = train & (y >= 0)
        xt = feat[tr].float()  # a normal tensor even if feat came out of inference mode
        mu, sd = xt.mean(0), xt.std(0) + 1e-6
        xt = (xt - mu) / sd
        torch.manual_seed(seed)
        head = torch.nn.Linear(xt.shape[1], num_classes).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
        for _ in range(steps):
            opt.zero_grad()
            F.cross_entropy(head(xt), y[tr]).backward()
            opt.step()
        with torch.no_grad():
            pred = head((feat[~train].float() - mu) / sd).argmax(1).cpu().numpy()
        folds.append(semseg_metrics(pred, gt[(~train).cpu().numpy()], num_classes=num_classes))
    return dict(miou=round(float(np.mean([f["miou"] for f in folds])), 4), acc=round(float(np.mean([f["acc"] for f in folds])), 4),
                miou_folds=[round(f["miou"], 4) for f in folds])
