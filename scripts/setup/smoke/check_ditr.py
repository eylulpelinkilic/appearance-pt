"""DITR image-injection PTv3 on T4 in the shared environment: build + fp16 training steps.

Exercises what is DITR-specific: FrozenDINOv2 (timm, wrapped in torch.compile), the
PT-v3m1-image decoder injection, fp16 AMP without flash-attn. Views and point<->pixel
correspondences are random, so projection is NOT tested here (that is M5). DINOv2 ViT-S by
default to keep the download small; set SMOKE_DITR_DINO=large to try the real encoder.
Run with cwd=third_party/ditr and PYTHONPATH=.:<repo>/src (DITR's own `pointcept` fork).
"""

import os
import time

import numpy as np
import torch

from appearance_pt.runlog import emit_result, section
from appearance_pt.smoke import cuda_peak_gb, load_demo_sample, voxel_downsample
from pointcept.models import build_model
from pointcept.utils.config import Config

DINO = os.environ.get("SMOKE_DITR_DINO", "small")
CAMS, H, W = 4, 420, 560  # DITR's ScanNet image size (ImageResize [420, 560] -> 30x40 patches)
POINT_MAX = 60000
STEPS = 5

section(f"config: configs/scannet/semseg-pt-v3m1-0-image.py, flash off, dinov2={DINO}")
print(f"TORCHDYNAMO_DISABLE={os.environ.get('TORCHDYNAMO_DISABLE', '0')} (1 = DITR's torch.compile of DINOv2 is off)")
cfg = Config.fromfile("configs/scannet/semseg-pt-v3m1-0-image.py")
bb = cfg.model.backbone
bb.enable_flash = False
bb.upcast_attention = True
bb.upcast_softmax = True
bb.dinov2 = DINO
print(cfg.pretty_text[:400] + "\n...")  # DITR's dump uses FormatCode(verify=True): needs yapf<0.40.2

section("model")
t = time.perf_counter()
model = build_model(cfg.model).cuda().train()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"built in {time.perf_counter() - t:.1f}s; trainable {n_params / 1e6:.1f}M (frozen DINOv2 excluded)")

section(f"data: demo room voxelized at 2 cm, {POINT_MAX}-point crop, {CAMS} random {H}x{W} views")
s = load_demo_sample("sample1")
keep, grid = voxel_downsample(s["coord"], 0.02)
rng = np.random.default_rng(0)
center = s["coord"][keep][rng.integers(len(keep))]
crop = np.argsort(np.linalg.norm(s["coord"][keep] - center, axis=1))[:POINT_MAX]
idx, grid = keep[crop], grid[crop]
n = len(idx)
batch = dict(
    coord=torch.from_numpy(s["coord"][idx].astype(np.float32)),
    grid_coord=torch.from_numpy(grid - grid.min(0)).int(),
    feat=torch.from_numpy(np.concatenate([s["color"][idx] / 127.5 - 1, s["normal"][idx]], 1).astype(np.float32)),
    segment=torch.from_numpy(s["segment20"][idx].astype(np.int64)),
    offset=torch.tensor([n]),
    image=torch.randn(1, CAMS, 3, H, W),
    image_coord=torch.stack([torch.rand(n, CAMS) * W, torch.rand(n, CAMS) * H], -1),  # (x, y) pixels
    image_mask=torch.rand(n, CAMS) < 0.5,
)
batch = {k: v.cuda() for k, v in batch.items()}
print({k: tuple(v.shape) for k, v in batch.items()})

section(f"{STEPS} AdamW steps, autocast fp16 + GradScaler (step 0 includes compilation)")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
scaler = torch.amp.GradScaler("cuda")
torch.cuda.reset_peak_memory_stats()
losses, times = [], []
for step in range(STEPS):
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(dict(batch))["loss"]
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t)
    losses.append(loss.item())
    print(f"  step {step}: loss {losses[-1]:.4f}  {times[-1]:.2f}s")
assert all(np.isfinite(losses)), "non-finite loss"

emit_result(
    dinov2=DINO, points=n, cams=CAMS, first_step_s=round(times[0], 1),
    sec_per_step=round(float(np.mean(times[2:])), 3), peak_gb=cuda_peak_gb(),
    compile_disabled=os.environ.get("TORCHDYNAMO_DISABLE") == "1",
)
