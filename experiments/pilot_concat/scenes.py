"""Scene inputs for the pilot encoders, numpy only (dicts with coord, color 0-255, normal)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

KEYS = ("coord", "color", "normal")


def tile_scene(scene: dict, k: int, gap: float = 0.5) -> dict:
    """k x k copies of a scene side by side in xy, `gap` metres apart.

    A stand-in for a larger scene with realistic local structure, to measure how encoder
    memory and time grow with the number of points before real ScanNet++ scenes exist.
    """
    coord = scene["coord"]
    step = coord.max(0) - coord.min(0) + gap
    shifts = [np.array([i * step[0], j * step[1], 0.0], dtype=coord.dtype) for i in range(k) for j in range(k)]
    return {
        "coord": np.concatenate([coord + s for s in shifts]),
        "color": np.concatenate([scene["color"]] * len(shifts)),
        "normal": np.concatenate([scene["normal"]] * len(shifts)),
    }


def load_scene_dir(path: Path) -> dict:
    """One scene as written by Pointcept's ScanNet++ preprocessing (coord/color/normal.npy:
    vertices of mesh_aligned_0.05.ply, uint8 colours, open3d vertex normals)."""
    return {key: np.load(Path(path) / f"{key}.npy") for key in KEYS}
