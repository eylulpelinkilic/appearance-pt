"""Per-stage encoder features, gathered at chosen points, and spatial tiles for scenes too large for one pass.

The pilot's per-point feature is the upcast of DefaultSegmentorV2 (encoders.encode): every stage's feature carried
back to the grid points through the pooling inverses and concatenated, [f_0, f_1, ..., f_K]. Materialised for a whole
scene that is N_grid x 1728 values for Concerto (5.7 GB in fp16 at 1.65M grid points, its out-of-memory point on a
T4). Stages keeps the stage features as they are (far smaller) and builds the upcast only for the points asked for.
Torch only, no encoder packages, so it runs in the CPU tests.
"""

from __future__ import annotations

import numpy as np
import torch


class Stages:
    """feats[k]: (N_k, C_k) stage-k features (stage 0 = grid points); invs[k]: (N_k,) maps stage-k rows to stage-k+1
    rows (the pooling_inverse of stage k+1); inverse: (N_input,) maps input points to grid points."""

    def __init__(self, feats: list[torch.Tensor], invs: list[torch.Tensor], inverse: torch.Tensor):
        assert len(invs) == len(feats) - 1
        self.feats, self.invs, self.inverse = feats, invs, inverse

    @classmethod
    def from_point(cls, point, dtype=torch.float16) -> Stages:
        """From the encoder's output Point (deepest stage, with the pooling_parent chain)."""
        feats, invs = [], []
        while "pooling_parent" in point.keys():
            feats.append(point.feat.to(dtype))
            invs.append(point.pooling_inverse)
            point = point.pooling_parent
        feats.append(point.feat.to(dtype))
        return cls(feats[::-1], invs[::-1], point.inverse)

    @property
    def width(self) -> int:
        return sum(f.shape[1] for f in self.feats)

    @property
    def n_grid(self) -> int:
        return len(self.feats[0])

    def to(self, device) -> Stages:
        return Stages([f.to(device) for f in self.feats], [i.to(device) for i in self.invs], self.inverse.to(device))

    def gather(self, idx: torch.Tensor) -> torch.Tensor:
        """(len(idx), width) upcast features of input points idx, on the device the stages are on."""
        g = self.inverse[idx.to(self.inverse.device)]
        out = [self.feats[0][g]]
        for inv, feat in zip(self.invs, self.feats[1:]):
            g = inv[g]
            out.append(feat[g])
        return torch.cat(out, dim=1)


def tile_inputs(coord: np.ndarray, core: np.ndarray, margin: float) -> np.ndarray:
    """Indices of all points within `margin` (xy) of the bounding box of the core points: a tile's encoder input."""
    lo, hi = coord[core, :2].min(0) - margin, coord[core, :2].max(0) + margin
    return np.flatnonzero(((coord[:, :2] >= lo) & (coord[:, :2] <= hi)).all(1))


def split_core(coord: np.ndarray, core: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The core points cut in two at the median of their longer horizontal extent."""
    axis = int(np.argmax(np.ptp(coord[core, :2], axis=0)))
    below = coord[core, axis] < np.median(coord[core, axis])
    if below.all() or not below.any():  # degenerate (many equal coordinates): split by order
        below = np.arange(len(core)) < len(core) // 2
    return core[below], core[~below]


class TiledStages:
    """Stages of one or more spatial tiles; each input point belongs to the core of exactly one tile, whose
    features (computed with `margin` of context around the core) it takes."""

    def __init__(self, n_points: int):
        self.tile_of = np.full(n_points, -1, np.int64)
        self.local = np.full(n_points, -1, np.int64)
        self.tiles: list[Stages] = []

    def add(self, stages: Stages, inputs: np.ndarray, core: np.ndarray) -> None:
        """inputs: global indices fed to the encoder for this tile (stages.inverse is over them); core: the global
        indices this tile answers for (a subset of inputs)."""
        pos = np.searchsorted(inputs, core)
        assert (inputs[pos] == core).all() and (self.tile_of[core] == -1).all()
        self.tile_of[core] = len(self.tiles)
        self.local[core] = pos
        self.tiles.append(stages)

    @property
    def complete(self) -> bool:
        return bool((self.tile_of >= 0).all())

    @property
    def width(self) -> int:
        return self.tiles[0].width

    def gather(self, idx: torch.Tensor, device="cuda") -> torch.Tensor:
        idx_np = idx.cpu().numpy()
        if len(self.tiles) == 1:
            return self.tiles[0].gather(torch.from_numpy(self.local[idx_np])).to(device)
        out = torch.empty(len(idx_np), self.width, dtype=self.tiles[0].feats[0].dtype, device=device)
        tile = self.tile_of[idx_np]
        for t, stages in enumerate(self.tiles):
            sel = np.flatnonzero(tile == t)
            if len(sel):
                out[torch.from_numpy(sel).to(device)] = stages.gather(torch.from_numpy(self.local[idx_np[sel]])).to(device)
        return out


def is_oom(e: BaseException) -> bool:
    """CUDA out of memory from torch, or from spconv / cumm (which raise plain RuntimeErrors)."""
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def encode_tiled(encode, coord: np.ndarray, margin: float = 1.0, max_depth: int = 3) -> tuple[TiledStages, int]:
    """encode(inputs: global indices) -> Stages over those inputs. Tries the whole scene, and on out-of-memory splits
    the failing tile in two (up to 2**max_depth tiles). Returns the tiles and how many there are."""
    import gc

    n = len(coord)
    tiled = TiledStages(n)
    pending = [(np.arange(n), 0)]  # (core, depth)
    while pending:
        core, depth = pending.pop()
        inputs = np.arange(n) if depth == 0 else tile_inputs(coord, core, margin)
        try:
            stages = encode(inputs)
        except RuntimeError as e:
            if not is_oom(e) or depth >= max_depth:
                raise
            stages = None
        if stages is None:  # outside the except block, so the traceback no longer holds the failed pass's tensors
            gc.collect()
            torch.cuda.empty_cache()
            pending += [(half, depth + 1) for half in split_core(coord, core)]
            continue
        tiled.add(stages, inputs, core)
    assert tiled.complete
    return tiled, len(tiled.tiles)
