"""Per-scene pieces shared by pilot steps 3 and 5: the frozen models, background scene loading, DSLR views with
DINOv2 features, and the encoders' per-stage features of a whole scene."""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from appearance_pt.scannetpp_cameras import dslr_c2w
from encoders import encode_scene, load_encoder, make_transform
from lifting import View, dino_patches, load_dino, preprocess, zbuffer
from scenes import load_core_scene

ENCODER_BLOCKS = ("utonia", "sonata", "concerto", "random")  # feature blocks from point encoders (pilot B, S, K, R)


class Models:
    """The four frozen point encoders (no FlashAttention; run them inside encoders.attention_backend("sdpa")) and
    DINOv2. `concerto` names the encoders.ENCODERS entry used for the concerto block."""

    def __init__(self, concerto: str = "concerto", seed: int = 0, device: str = "cuda"):
        self.encoders, self.transforms, self.widths, self.names = {}, {}, {}, {}
        for block in ENCODER_BLOCKS:
            name = concerto if block == "concerto" else block
            self.encoders[block], cfg = load_encoder(name, seed=seed)
            self.transforms[block] = make_transform(name)
            self.widths[block] = sum(cfg["enc_channels"])
            self.names[block] = name
        self.dino = load_dino(device)
        self.dino_width = self.dino.config.hidden_size


def encode_blocks(models: Models, s: dict, seed: int, store: str) -> tuple[dict, dict]:
    """{block: TiledStages} for every encoder on the whole scene, and {block: (tiles, seconds)}."""
    out, info = {}, {}
    for block, model in models.encoders.items():
        t0 = time.perf_counter()
        out[block], tiles = encode_scene(model, models.transforms[block], s, seed, store=store)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        info[block] = (tiles, round(time.perf_counter() - t0, 2))
    return out, info


def scene_views(s: dict, points: torch.Tensor, occ, device="cuda") -> list[View]:
    """The scene's loaded DSLR views (toolbox pose convention, decisions.md P5) with z-buffers over all points."""
    K = torch.tensor(s["K"], dtype=torch.float32, device=device)
    views = []
    for v in s["views"]:
        w2c = torch.tensor(np.linalg.inv(dslr_c2w(v["T"])), dtype=torch.float32, device=device)
        view = View(v["name"], w2c, K, s["hw"], torch.from_numpy(v["valid"]).to(device), occ=occ)
        view.zbuf = zbuffer(points, view)
        views.append(view)
    return views


def add_dino(model, s: dict, views: list[View], batch: int = 6, device="cuda") -> None:
    """DINOv2 patch features of every view (view.feat)."""
    images = [preprocess(v["jpg"]) for v in s["views"]]
    for b in range(0, len(images), batch):
        feats = dino_patches(model, torch.stack(images[b:b + batch]).to(device))
        for view, f in zip(views[b:b + batch], feats):
            view.feat = f


def prefetch(items: list[tuple[str, object]], n_views: int, workers: int = 3, ahead: int = 4, load=None):
    """Yields (scene, loaded scene or the exception, seconds waited) in order, loading up to `ahead` scenes in
    background threads (loading is I/O on the dataset mount: ~20 s per scene, whatever its size). load(scene, path)
    defaults to scenes.load_core_scene with n_views."""
    load = load or (lambda scene, path: load_core_scene(path, scene, n_views))
    with ThreadPoolExecutor(workers) as pool:
        queue, it = deque(), iter(items)

        def submit():
            nxt = next(it, None)
            if nxt is not None:
                queue.append((nxt[0], pool.submit(load, nxt[0], nxt[1])))

        for _ in range(ahead):
            submit()
        while queue:
            scene, future = queue.popleft()
            submit()
            t0 = time.perf_counter()
            try:
                s = future.result()
            except Exception as e:  # noqa: BLE001  (reported per scene by the caller)
                s = e
            yield scene, s, round(time.perf_counter() - t0, 2)
