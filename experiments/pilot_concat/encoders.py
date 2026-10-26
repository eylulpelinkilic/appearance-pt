"""Frozen point encoders for the pilot: Utonia, Sonata, Concerto and an untrained PTv3 with Utonia's architecture.

Loads each encoder for a T4 (no FlashAttention), applies the input transform it was pretrained and
probed with, and returns the "upcast" per-point features used for linear probing. Where each fact
was verified: docs/decisions.md (2026-10-24, P1). In short:

- Utonia (third_party/utonia, utonia/model.py `load`): the checkpoint stores `config` + `state_dict`
  and PointTransformerV3(**config) is the encoder-only PTv3. Input feat = (coord, color/255, normal),
  9 channels; a missing modality is set to zero (README). Indoor transform = transform.default(0.5):
  RandomScale(0.5) -> CenterShift -> GridSample(0.01), i.e. 2 cm voxels in metres, as in
  demo/2_sem_seg.py, Pointcept's Utonia ScanNet/S3DIS probe configs and indoor pretraining
  (RandomScale [0.45, 0.55]).
- Sonata (third_party/sonata, sonata/model.py `load`): same checkpoint layout and 9 input channels;
  transform.default() = CenterShift -> GridSample(0.02), coordinates in metres.
- Concerto (third_party/concerto, concerto/model.py `load`): Sonata's package with the model list changed;
  transform.py and SerializedAttention are identical to Sonata's, so the same transform and SDPA swap apply.
  Weights: concerto_large_origin.pth, the original Concerto (HF model card: the default concerto_large/base/small
  are a pre-release of Utonia); docs/decisions.md P4.
- Upcast (pointcept/models/default.py DefaultSegmentorV2.forward, both demo/2_sem_seg.py): walk the
  pooling chain back to the first stage, concatenating every stage's features (width =
  sum(enc_channels)); point.inverse then maps grid-sampled points to the input points.
"""

from __future__ import annotations

import contextlib
import time

import concerto
import numpy as np
import sonata
import torch
import torch.nn.functional as F
import utonia
from huggingface_hub import hf_hub_download

from scenes import KEYS
from stages import Stages, TiledStages, encode_tiled

# Hugging Face repo and commit of the weights (the packages' load() would take "main"), checked 2026-10-24
WEIGHTS = {
    "utonia": ("Pointcept/Utonia", "9457f025e36c7001ee962a542f0d27a5b600e1e7"),
    "sonata": ("facebook/sonata", "df99897472c09f91ba9288da0a034aacffc0b010"),
    "concerto": ("Pointcept/Concerto", "efc130718a564eba0ce4899318f3c20b19d9c752"),  # checked 2026-10-26
}
PACKAGES = {"utonia": utonia, "sonata": sonata, "concerto": concerto}
# encoder name: (weights family, checkpoint file); "random" = Utonia's config with untrained weights
ENCODERS = {
    "utonia": ("utonia", "utonia.pth"),
    "sonata": ("sonata", "sonata.pth"),
    "concerto": ("concerto", "concerto_large_origin.pth"),
    "concerto_prerelease": ("concerto", "concerto_large.pth"),  # step-1b comparison only
    "random": ("utonia", "utonia.pth"),
}
UTONIA_INDOOR_SCALE = 0.5


def seed_everything(seed: int) -> None:
    # GridSample(mode="train") keeps a random point per voxel (np.random), and
    # serialization(shuffle_orders=True) permutes the four orders (torch.randperm), in eval too
    np.random.seed(seed)
    torch.manual_seed(seed)


def checkpoint(family: str, filename: str, **kwargs) -> dict:
    """A {config, state_dict} checkpoint of the "utonia", "sonata" or "concerto" HF repo at its pinned commit."""
    repo_id, revision = WEIGHTS[family]
    path = hf_hub_download(repo_id, filename, revision=revision)
    return PACKAGES[family].model.load(path, ckpt_only=True, **kwargs)


def load_encoder(name: str, seed: int = 0) -> tuple[torch.nn.Module, dict]:
    """name: a key of ENCODERS; "random" is Utonia's config with weights initialised from `seed`."""
    family, filename = ENCODERS[name]
    pkg = PACKAGES[family]
    # T4 is sm_75; FlashAttention 2 needs sm_80
    ckpt = checkpoint(family, filename, custom_config=dict(enable_flash=False))
    torch.manual_seed(seed)
    model = pkg.model.PointTransformerV3(**ckpt["config"])
    if name != "random":
        model.load_state_dict(ckpt["state_dict"])
    return model.cuda().eval(), ckpt["config"]


class SegHead(torch.nn.Module):  # module layout of the released *_linear_prob_head_sc checkpoints
    def __init__(self, backbone_out_channels, num_classes):
        super().__init__()
        self.seg_head = torch.nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


def load_scannet_probe(name: str) -> SegHead:
    """The released ScanNet-20 linear probe for "utonia", "sonata" or "concerto" (concerto_large_linear_prob_head_sc)."""
    filename = "concerto_large_linear_prob_head_sc.pth" if name == "concerto" else f"{name}_linear_prob_head_sc.pth"
    ckpt = checkpoint(name, filename)
    head = SegHead(**ckpt["config"])
    head.load_state_dict(ckpt["state_dict"])
    return head.cuda().eval()


def make_transform(name: str, variant: str = "default"):
    """Input transform for an encoder. Variants other than "default" exist to test Utonia's scale:
    "pointcept_scannetpp_lin": val transform of Pointcept configs/utonia/semseg-utonia-v1m1-2a-scannetpp-lin.py
    (metres, 2 cm grid; copied from the Sonata config); "metric_1cm": transform.default(1.0)."""
    if name in ("sonata", "concerto", "concerto_prerelease"):
        assert variant == "default"
        return PACKAGES[ENCODERS[name][0]].transform.default()
    if variant == "default":
        return utonia.transform.default(scale=UTONIA_INDOOR_SCALE)
    if variant == "metric_1cm":
        return utonia.transform.default(scale=1.0)
    assert variant == "pointcept_scannetpp_lin", variant
    return utonia.transform.Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True,
             return_inverse=True),
        dict(type="CenterShift", apply_z=False),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse"), feat_keys=("coord", "color", "normal")),
    ])


def sonata_sdpa_forward(self, point):
    """sonata.model.SerializedAttention.forward (non-flash branch) with the attention computed by
    torch SDPA instead of materialising the (patches, heads, K, K) matrix; the memory-efficient kernel
    runs on sm_75. Same maths as the original for inference (no dropout, no RPE). Concerto's
    SerializedAttention is a verbatim copy of Sonata's, so this serves both.
    Adapted from facebookresearch/sonata (Apache-2.0)."""
    assert not (self.enable_flash or self.enable_rpe or self.training)
    self.patch_size = min(sonata.utils.offset2bincount(point.offset).min().tolist(), self.patch_size_max)
    H, K, C = self.num_heads, self.patch_size, self.channels
    pad, unpad, _ = self.get_padding_and_inverse(point)
    order = point.serialized_order[self.order_index][pad]
    inverse = unpad[point.serialized_inverse[self.order_index]]
    qkv = self.qkv(point.feat)[order]
    q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
    if self.upcast_attention:
        q, k = q.float(), k.float()
    feat = F.scaled_dot_product_attention(q, k, v.to(q.dtype), scale=self.scale)
    feat = feat.to(qkv.dtype).transpose(1, 2).reshape(-1, C)[inverse]
    point.feat = self.proj_drop(self.proj(feat))
    return point


@contextlib.contextmanager
def attention_backend(backend: str):
    """"sdpa" or "manual" (the explicit softmax(QK^T)V product) for all encoders' non-flash branch.

    Utonia uses SDPA since Pointcept/Utonia@f832deb unless utonia.model.SDPA_AVAILABLE is False;
    Sonata and Concerto only have the manual product, so "sdpa" swaps in sonata_sdpa_forward.
    """
    assert backend in ("sdpa", "manual"), backend
    classes = (sonata.model.SerializedAttention, concerto.model.SerializedAttention)
    saved = utonia.model.SDPA_AVAILABLE, [c.forward for c in classes]
    utonia.model.SDPA_AVAILABLE = backend == "sdpa"
    if backend == "sdpa":
        for c in classes:
            c.forward = sonata_sdpa_forward
    try:
        yield
    finally:
        utonia.model.SDPA_AVAILABLE = saved[0]
        for c, forward in zip(classes, saved[1]):
            c.forward = forward


@torch.inference_mode()
def encode(model, transform, scene: dict, seed: int, amp: bool = True) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Upcast features of one scene.

    Returns (feat, inverse, stats): feat (M, sum(enc_channels)) on the GPU for the M grid-sampled
    points, in fp16 with `amp` else fp32; inverse (N,) maps each input point to its row of feat.
    """
    out_dtype = torch.float16 if amp else torch.float32
    seed_everything(seed)
    t0 = time.perf_counter()
    point = transform({key: scene[key].copy() for key in KEYS})
    t1 = time.perf_counter()
    point = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in point.items()}
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        point = model(point)
        point.feat = point.feat.to(out_dtype)
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat.to(out_dtype), point.feat[inverse]], dim=-1)
            point = parent
    torch.cuda.synchronize()
    stats = dict(n_input=len(scene["coord"]), n_grid=int(point.feat.shape[0]),
                 sec_transform=t1 - t0, sec_gpu=time.perf_counter() - t1)
    return point.feat, point.inverse, stats


@torch.inference_mode()
def encode_stages(model, transform, scene: dict, seed: int, amp: bool = True) -> Stages:
    """Per-stage features of one forward pass (no upcast); Stages.gather gives encode()'s values at chosen points."""
    seed_everything(seed)
    point = transform({key: scene[key].copy() for key in KEYS})
    point = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in point.items()}
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        point = model(point)
    return Stages.from_point(point, torch.float16 if amp else torch.float32)


def encode_scene(model, transform, scene: dict, seed: int, store: str = "cuda", margin: float = 1.0) -> tuple[TiledStages, int]:
    """encode_stages for a whole scene (fp16, the attention backend in force), split into spatial tiles with `margin`
    metres of context if a single pass runs out of memory. Stage features are kept on `store` ("cpu" to free the GPU).
    Returns the tiles and their number (1 = one pass)."""
    def run(inputs):
        sub = {key: scene[key][inputs] for key in KEYS}
        return encode_stages(model, transform, sub, seed).to(store)

    return encode_tiled(run, scene["coord"], margin=margin)


def scene_seed(scene_id: str) -> int:
    """The seed of a scene's one encoder pass (voxel representatives and order shuffle; decisions.md P3)."""
    import zlib

    return zlib.crc32(scene_id.encode()) % 2**31
