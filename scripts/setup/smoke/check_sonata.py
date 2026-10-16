"""Sonata demo, headless: pretrained PTv3 encoder + ScanNet linear-probe head on the demo room.

The M1 smoke test (facebookresearch/sonata demo/2_sem_seg.py without the Open3D window):
flash-attn disabled exactly as the Sonata README documents, run in fp32 and fp16 autocast,
scored against the room's segment20 labels, with top-down PNGs for a visual check.
Run from the repo root with PYTHONPATH=third_party/sonata:src.
"""

import os
import time
from pathlib import Path

import numpy as np
import torch

import sonata
from appearance_pt.runlog import emit_result, section
from appearance_pt.smoke import SCANNET20_NAMES, cuda_peak_gb, semseg_metrics

OUT = Path(os.environ.get("SMOKE_OUT", "/kaggle/working/smoke"))
OUT.mkdir(parents=True, exist_ok=True)


class SegHead(torch.nn.Module):  # same module layout as the demo, so the probe checkpoint loads
    def __init__(self, backbone_out_channels, num_classes):
        super().__init__()
        self.seg_head = torch.nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


section("load Sonata (enable_flash=False, enc_patch_size=1024) + linear probe head")
sonata.utils.set_seed(24525867)
model = sonata.load(
    "sonata", repo_id="facebook/sonata",
    custom_config=dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False),
).cuda().eval()
ckpt = sonata.load("sonata_linear_prob_head_sc", repo_id="facebook/sonata", ckpt_only=True)
head = SegHead(**ckpt["config"]).cuda().eval()
head.load_state_dict(ckpt["state_dict"])

section("demo room")
raw = sonata.data.load("sample1")
gt = raw.pop("segment20").astype(np.int64)
raw.pop("segment200", None)
coord_full = raw["coord"].copy()
print(f"{len(gt)} points, labelled {np.mean(gt >= 0):.1%}")


def infer(amp: bool):
    point = sonata.transform.default()({k: v.copy() for k, v in raw.items()})
    point = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in point.items()}
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        point = model(point)
        while "pooling_parent" in point.keys():  # un-pool to full resolution (as in the demo)
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        logits = head(point.feat)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t
    inverse = point.inverse  # grid-sampled point -> original point
    return logits.argmax(-1)[inverse].cpu().numpy(), point.feat[inverse].float(), seconds, int(point.feat.shape[0])


results = {}
for amp in (False, True):
    torch.cuda.reset_peak_memory_stats()
    pred, feat, seconds, n_grid = infer(amp)
    m = semseg_metrics(pred, gt, num_classes=20)
    tag = "fp16" if amp else "fp32"
    results[tag] = dict(pred=pred, feat=feat, seconds=seconds, metrics=m, peak_gb=cuda_peak_gb())
    print(f"{tag}: {n_grid} grid points, {seconds:.2f}s, peak {results[tag]['peak_gb']} GB, "
          f"acc {m['acc']:.3f}, mIoU {m['miou']:.3f} over {m['n_classes_present']} classes")
agreement = float(np.mean(results["fp16"]["pred"] == results["fp32"]["pred"]))
print(f"fp16 vs fp32 prediction agreement: {agreement:.4f}")
print("per-class IoU (fp32): " + ", ".join(
    f"{n}={i:.2f}" for n, i in zip(SCANNET20_NAMES, results["fp32"]["metrics"]["iou"]) if i > 0))

section(f"visual check -> {OUT}")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

palette = plt.get_cmap("tab20")(np.arange(20))[:, :3]
feat = results["fp32"]["feat"]
u, _, _ = torch.pca_lowrank(feat - feat.mean(0), q=3, center=False)
pca = u.cpu().numpy()
pca = (pca - pca.min(0)) / (pca.max(0) - pca.min(0) + 1e-8)
panels = {
    "input color": np.clip(raw["color"] / 255.0, 0, 1),
    "ground truth (segment20)": np.where(gt[:, None] >= 0, palette[np.clip(gt, 0, 19)], 0.0),
    "Sonata linear probe (fp32)": palette[results["fp32"]["pred"]],
    "feature PCA": pca,
}
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
order = np.argsort(coord_full[:, 2])  # draw low points first so the top-down view shows the top surface
for ax, (title, colors) in zip(axes, panels.items()):
    ax.scatter(coord_full[order, 0], coord_full[order, 1], c=colors[order], s=0.2, linewidths=0)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "sonata_semseg_topdown.png", dpi=110)
print(f"saved {OUT / 'sonata_semseg_topdown.png'}")

emit_result(
    acc_fp32=round(results["fp32"]["metrics"]["acc"], 4), miou_fp32=round(results["fp32"]["metrics"]["miou"], 4),
    acc_fp16=round(results["fp16"]["metrics"]["acc"], 4), miou_fp16=round(results["fp16"]["metrics"]["miou"], 4),
    fp16_fp32_agreement=round(agreement, 4), sec_fp32=round(results["fp32"]["seconds"], 2),
    sec_fp16=round(results["fp16"]["seconds"], 2), peak_gb_fp32=results["fp32"]["peak_gb"],
)
