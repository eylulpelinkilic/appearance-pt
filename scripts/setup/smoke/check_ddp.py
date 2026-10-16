"""Multi-GPU check (run under torchrun): NCCL all_reduce correctness + bandwidth, DDP fp16 steps.

Usage: torchrun --standalone --nproc_per_node=2 scripts/setup/smoke/check_ddp.py
"""

import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from appearance_pt.runlog import emit_result

dist.init_process_group("nccl", timeout=timedelta(seconds=120))
rank, world = dist.get_rank(), dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dev = torch.device("cuda", local_rank)


def log(msg):
    if rank == 0:
        print(msg, flush=True)


log(f"world size {world}, NCCL {'.'.join(map(str, torch.cuda.nccl.version()))}, "
    f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', 'unset')}")

x = torch.full((4,), float(rank + 1), device=dev)
dist.all_reduce(x)
expected = world * (world + 1) / 2
assert torch.allclose(x, torch.full_like(x, expected)), f"all_reduce gave {x.tolist()}, expected {expected}"
log(f"all_reduce correct ({x[0].item():.0f})")

buf = torch.randn(64 * 2**20, device=dev)  # 256 MiB fp32
for _ in range(2):
    dist.all_reduce(buf)
torch.cuda.synchronize()
t = time.perf_counter()
iters = 5
for _ in range(iters):
    dist.all_reduce(buf)
torch.cuda.synchronize()
algbw = buf.numel() * 4 * iters / (time.perf_counter() - t) / 1e9
busbw = algbw * 2 * (world - 1) / world
log(f"all_reduce 256 MiB: algbw {algbw:.2f} GB/s, busbw {busbw:.2f} GB/s")

torch.manual_seed(0)
model = DDP(torch.nn.Sequential(torch.nn.Linear(512, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 20)).to(dev), device_ids=[local_rank])
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda")
torch.manual_seed(rank)  # different data per rank
for _ in range(5):
    with torch.autocast("cuda", dtype=torch.float16):
        loss = torch.nn.functional.cross_entropy(model(torch.randn(256, 512, device=dev)), torch.randint(0, 20, (256,), device=dev))
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
checksum = torch.stack([p.detach().double().sum() for p in model.parameters()]).sum().reshape(1)
gathered = [torch.zeros_like(checksum) for _ in range(world)]
dist.all_gather(gathered, checksum)
in_sync = all(torch.allclose(g, gathered[0]) for g in gathered)
log(f"DDP fp16 steps done; parameters identical across ranks: {in_sync}")
assert in_sync

if rank == 0:
    emit_result(world=world, algbw_gbps=round(algbw, 2), busbw_gbps=round(busbw, 2), p2p_disabled=os.environ.get("NCCL_P2P_DISABLE") == "1")
dist.barrier()
dist.destroy_process_group()
