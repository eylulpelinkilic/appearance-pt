"""Compiled extensions on the GPU: spconv, torch_scatter, torch_cluster, pointops, SharedArray.

Run from the repo root with PYTHONPATH=src. Numerical checks compare against plain torch.
"""

import os
import time

import numpy as np
import torch

from appearance_pt.runlog import emit_result, section

dev = torch.device("cuda")
metrics = {}

section("spconv: SubMConv3d / SparseConv3d forward + backward (fp32, fp16 autocast, fp16 weights)")
import spconv.pytorch as spconv  # noqa: E402

g = torch.Generator().manual_seed(0)
coords = torch.unique(torch.randint(0, 128, (80000, 3), generator=g), dim=0)
indices = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32), coords.int()], 1).to(dev)
net = spconv.SparseSequential(
    spconv.SubMConv3d(16, 32, 3, padding=1, bias=False, indice_key="subm0"),
    torch.nn.ReLU(),
    spconv.SparseConv3d(32, 64, 2, stride=2, bias=False, indice_key="down0"),
).to(dev)
for mode in ("fp32", "fp16_autocast", "fp16_weights"):
    m = net.half() if mode == "fp16_weights" else net.float()
    feats = torch.randn(len(coords), 16, device=dev, dtype=torch.half if mode == "fp16_weights" else torch.float, requires_grad=True)
    t = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16, enabled=mode == "fp16_autocast"):
        y = m(spconv.SparseConvTensor(feats, indices, [128, 128, 128], 1))
    loss = y.features.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(loss) and torch.isfinite(feats.grad).all())
    print(f"  {mode:14s} out {tuple(y.features.shape)} {y.features.dtype}  loss {loss.item():.4f}  finite {finite}  {1000 * (time.perf_counter() - t):.0f} ms")
    assert finite, f"spconv {mode} produced non-finite values"
    metrics[f"spconv_{mode}_ms"] = round(1000 * (time.perf_counter() - t))
net.float()

section("torch_scatter.segment_csr vs torch reference")
import torch_scatter  # noqa: E402

src = torch.randn(1000, 8, device=dev)
indptr = torch.tensor([0, 100, 350, 1000], device=dev)
out = torch_scatter.segment_csr(src, indptr, reduce="mean")
ref = torch.stack([src[a:b].mean(0) for a, b in zip(indptr[:-1].tolist(), indptr[1:].tolist())])
err = (out - ref).abs().max().item()
print(f"  max abs err {err:.2e}")
assert err < 1e-5
metrics["scatter_err"] = err

section("torch_cluster.fps")
import torch_cluster  # noqa: E402

pts = torch.rand(20000, 3, device=dev)
idx = torch_cluster.fps(pts, torch.zeros(len(pts), dtype=torch.long, device=dev), ratio=0.05)
print(f"  sampled {len(idx)} of {len(pts)} points, unique {len(torch.unique(idx))}")
assert len(torch.unique(idx)) == len(idx)

section("pointops.knn_query vs brute force (two batches)")
import pointops  # noqa: E402

xyz = torch.rand(30000, 3, device=dev)
offset = torch.tensor([12000, 30000], device=dev, dtype=torch.int32)
k = 16
_, dist = pointops.knn_query(k, xyz, offset)  # returns (idx, euclidean distance)
ref = []
for a, b in ((0, 12000), (12000, 30000)):
    ref.append(torch.cdist(xyz[a:b], xyz[a:b]).topk(k, largest=False).values)
ref = torch.cat(ref)
err = (dist.sort(1).values - ref).abs().max().item()
print(f"  max abs distance err {err:.2e}")
assert err < 1e-4
metrics["knn_err"] = err

section("SharedArray in /dev/shm (Pointcept/DITR dataset cache)")
import SharedArray  # noqa: E402

name = f"shm://appt_smoke_{os.getpid()}"
a = SharedArray.create(name, (1000,), dtype=np.float32)
a[:] = 1.5
assert float(SharedArray.attach(name).sum()) == 1500.0
SharedArray.delete(name)
print("  create/attach/delete ok")

section("open3d import")
import open3d  # noqa: E402

print(f"  open3d {open3d.__version__}")
metrics["open3d"] = open3d.__version__

emit_result(**metrics)
