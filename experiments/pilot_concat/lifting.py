"""DINOv2 patch features lifted onto the points of a scene (pilot configs A, C, S+, K+, R+DINO).

Each point (a vertex of mesh_aligned_0.05.ply) is projected into the DSLR views of its scene (undistorted PINHOLE
images; cameras from appearance_pt.scannetpp_cameras). Occlusion uses a z-buffer of the points themselves, since the
pilot keeps no mesh faces for train scenes: the nearest depth per CELL x CELL pixel cell, closed with a 3 x 3 minimum
filter so that gaps between points do not let hidden points through. A point is visible from a view if it is in front
of the camera, inside the image, on a valid mask pixel (masks mark anonymised and invalid regions) and no farther than
DEPTH_TOL behind the nearest depth around it. Its feature is the bilinear sample of the view's patch grid at the
projected pixel, averaged over every view that sees it; points no view sees get zeros and a view count of 0.
Choices and their reasons: docs/decisions.md P5.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

DINO = ("facebook/dinov2-large", "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c")  # ViT-L/14, no registers; checked 2026-10-26
DINO_HW = (518, 770)  # short side 518 px; 770 and 784 are both 7 px off 518 * 1752 / 1168 = 777, the smaller is taken
PATCH = 14
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet, as DINOv2 was trained
CELL = 8  # z-buffer cell in full-resolution pixels
DEPTH_TOL = (0.02, 0.05)  # visible if depth <= nearest depth + 2 cm + 5 % of depth
Z_NEAR = 0.05  # metres in front of the camera


@dataclass
class View:
    name: str
    w2c: torch.Tensor  # (4, 4) world (mesh frame) to OpenCV camera
    K: torch.Tensor  # (3, 3)
    hw: tuple[int, int]  # full-resolution image size
    valid: torch.Tensor  # (H, W) bool, mask pixel usable
    zbuf: torch.Tensor | None = None  # (ceil(H / CELL), ceil(W / CELL)) nearest depth, inf where empty
    feat: torch.Tensor | None = None  # (C, h, w) patch features


def load_dino(device: str = "cuda"):
    from transformers import AutoModel

    return AutoModel.from_pretrained(DINO[0], revision=DINO[1]).to(device).eval()


def preprocess(jpg: bytes) -> torch.Tensor:
    """(3, 518, 770) normalised DINOv2 input from JPEG bytes; the whole image is resized (no crop)."""
    from PIL import Image

    with Image.open(io.BytesIO(jpg)) as im:
        x = np.asarray(im.convert("RGB").resize(DINO_HW[::-1], Image.BICUBIC), dtype=np.float32) / 255
    x = torch.from_numpy(x).permute(2, 0, 1)
    return (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]


@torch.inference_mode()
def dino_patches(model, images: torch.Tensor) -> torch.Tensor:
    """(B, C, H/14, W/14) fp16 patch features of normalised images (B, 3, H, W): last layer, after the final norm."""
    b, _, h, w = images.shape
    h, w = h // PATCH, w // PATCH
    with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=images.is_cuda):
        tokens = model(pixel_values=images).last_hidden_state
    return tokens[:, -h * w:].reshape(b, h, w, -1).permute(0, 3, 1, 2).half()  # drops the CLS (and register) tokens


def project(points: torch.Tensor, w2c: torch.Tensor, K: torch.Tensor):
    """Pixel coordinates (u, v; pixel (0, 0) spans [0, 1)^2) and depth z of world points (N, 3)."""
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    zs = z.clamp(min=1e-6)
    return K[0, 0] * cam[:, 0] / zs + K[0, 2], K[1, 1] * cam[:, 1] / zs + K[1, 2], z


def in_image(u, v, z, hw) -> torch.Tensor:
    return (z > Z_NEAR) & (u >= 0) & (u < hw[1]) & (v >= 0) & (v < hw[0])


def zbuffer(points: torch.Tensor, view: View) -> torch.Tensor:
    """Nearest depth per cell over all points, closed with a 3 x 3 minimum filter."""
    u, v, z = project(points, view.w2c, view.K)
    ok = in_image(u, v, z, view.hw)
    hz, wz = -(-view.hw[0] // CELL), -(-view.hw[1] // CELL)
    idx = (v[ok] / CELL).long() * wz + (u[ok] / CELL).long()
    buf = torch.full((hz * wz,), float("inf"), device=points.device).scatter_reduce_(0, idx, z[ok], reduce="amin")
    return -F.max_pool2d(-buf.view(1, 1, hz, wz), 3, stride=1, padding=1)[0, 0]


def visible(u, v, z, view: View) -> torch.Tensor:
    ok = in_image(u, v, z, view.hw)
    ui, vi = u.long().clamp(0, view.hw[1] - 1), v.long().clamp(0, view.hw[0] - 1)
    ok &= view.valid[vi, ui]
    return ok & (z <= view.zbuf[vi // CELL, ui // CELL] + DEPTH_TOL[0] + DEPTH_TOL[1] * z)


def sample(feat: torch.Tensor, u, v, hw) -> torch.Tensor:
    """(P, C) float32 bilinear samples of a (C, h, w) patch grid that covers the whole (H, W) image."""
    grid = torch.stack([2 * u / hw[1] - 1, 2 * v / hw[0] - 1], dim=-1).view(1, 1, -1, 2).float()
    out = F.grid_sample(feat[None].float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out[0, :, 0].T


def lift(points: torch.Tensor, views: list[View], chunk: int = 500_000, max_sample: int = 200_000):
    """Yields (start, mean feature (n, C) float32, view count (n,) int32) for consecutive chunks of `points`.

    Every view needs `feat` and a `zbuf` built from all points of the scene (not only these)."""
    channels = views[0].feat.shape[0]
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk]
        acc = torch.zeros(len(p), channels, device=p.device)
        count = torch.zeros(len(p), dtype=torch.int32, device=p.device)
        for view in views:
            u, v, z = project(p, view.w2c, view.K)
            idx = visible(u, v, z, view).nonzero().squeeze(1)
            for s in range(0, len(idx), max_sample):
                j = idx[s:s + max_sample]
                acc.index_add_(0, j, sample(view.feat, u[j], v[j], view.hw))
            count[idx] += 1
        yield start, acc / count.clamp(min=1).unsqueeze(1), count


def colour_agreement(points, colors, keep, view: View, image: torch.Tensor) -> dict:
    """Pearson r between vertex luminance and image luminance at the projected pixel, over the visible points:
    high for the right pose convention, near zero for a wrong one."""
    u, v, z = project(points, view.w2c, view.K)
    vis = visible(u, v, z, view) & keep
    n = int(vis.sum())
    if n < 100:
        return dict(n_visible=n, r=None)
    lum = torch.tensor([0.299, 0.587, 0.114], device=points.device)
    a = colors[vis].float() @ lum
    b = image[v[vis].long(), u[vis].long()].float() @ lum
    r = torch.corrcoef(torch.stack([a, b]))[0, 1]
    return dict(n_visible=n, r=round(float(r), 4))


def pca3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean (C,) and the 3 leading principal directions (C, 3) of rows x (M, C)."""
    x = x.float()
    mean = x.mean(0)
    xc = x - mean
    _, vecs = torch.linalg.eigh(xc.T @ xc / max(len(x) - 1, 1))
    return mean, vecs[:, -3:].flip(1)
