"""Pointcept PTv3 (ScanNet++ config) training steps on T4: no flash-attn, fp16 AMP + GradScaler.

Run with cwd=third_party/Pointcept and PYTHONPATH=.:<repo>/src (Pointcept's own `pointcept`).
Uses Pointcept's public demo room (ScanNet) as input: this tests the software stack, it is
not the M3 throughput/memory profile (which uses ScanNet++ crops).
"""

import time

import numpy as np
import torch

from appearance_pt.runlog import emit_result, section
from appearance_pt.smoke import cuda_peak_gb, load_demo_sample
from pointcept.datasets.transform import Compose
from pointcept.datasets.utils import collate_fn
from pointcept.models import build_model
from pointcept.utils.config import Config

POINT_MAX = 80000  # smoke-test crop; the protocol value is chosen in M3
STEPS = 6

section("config: configs/scannetpp/semseg-pt-v3m1-0-base.py with flash disabled")
cfg = Config.fromfile("configs/scannetpp/semseg-pt-v3m1-0-base.py")
bb = cfg.model.backbone
bb.enable_flash = False
bb.upcast_attention = True  # fp32 attention logits/softmax in the non-flash path (fp16 stability)
bb.upcast_softmax = True
print(cfg.pretty_text[:600] + "\n...")  # pretty_text goes through yapf
print(f"enc_patch_size={bb.enc_patch_size} dec_patch_size={bb.dec_patch_size} amp_dtype={cfg.amp_dtype}")

section("model")
model = build_model(cfg.model).cuda().train()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{type(model).__name__} / {bb.type}: {n_params / 1e6:.1f}M parameters")

section(f"data: demo room -> GridSample 0.02 -> SphereCrop {POINT_MAX}")
s = load_demo_sample("sample1")
print({k: v.shape for k, v in s.items()})
transform = Compose([
    dict(type="CenterShift", apply_z=True),
    dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="SphereCrop", point_max=POINT_MAX, mode="random"),
    dict(type="CenterShift", apply_z=False),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("color", "normal")),
])
sample = dict(coord=s["coord"].astype(np.float32), color=s["color"].astype(np.float32),
              normal=s["normal"].astype(np.float32), segment=s["segment20"].astype(np.int64))
# 2 random crops = batch size 2 per GPU; copies because some transforms work in place
batch = collate_fn([transform({k: v.copy() for k, v in sample.items()}) for _ in range(2)])
batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
print(f"points per batch {batch['coord'].shape[0]}, offset {batch['offset'].tolist()}, feat {tuple(batch['feat'].shape)}")

section(f"{STEPS} AdamW steps, autocast fp16 + GradScaler")
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
    print(f"  step {step}: loss {losses[-1]:.4f}  {times[-1]:.2f}s  scale {scaler.get_scale():.0f}")
assert all(np.isfinite(losses)), "non-finite loss"

model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    logits = model(dict(batch))["seg_logits"]
print(f"eval forward: seg_logits {tuple(logits.shape)}")

emit_result(
    points=int(batch["coord"].shape[0]), params_m=round(n_params / 1e6, 1),
    sec_per_step=round(float(np.mean(times[2:])), 3), peak_gb=cuda_peak_gb(),
    loss_first=round(losses[0], 4), loss_last=round(losses[-1], 4),
)
