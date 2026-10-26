"""ScanNet++ DSLR cameras in the frame of the scan mesh (mesh_aligned_0.05.ply, i.e. the points we segment).

dslr/nerfstudio/transforms_undistorted.json holds PINHOLE intrinsics of the undistorted images (fl_x, fl_y, cx, cy, w, h)
and, per frame, a `transform_matrix` that the toolbox wrote from COLMAP (common/utils/nerfstudio.py convert_frames;
dslr/undistort.py copies the frames unchanged): COLMAP's camera-to-world with the camera y and z axes flipped, then the
world x and y rows swapped and the world z row negated (instant-ngp / nerfstudio convention). The toolbox renders
mesh_aligned_0.05.ply with the COLMAP poses as they are (common/render.py), so undoing those three steps gives the
camera-to-world in the mesh frame with OpenCV axes (x right, y down, z forward). docs/decisions.md P5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def dslr_c2w(transform_matrix) -> np.ndarray:
    """COLMAP camera-to-world (OpenCV axes, mesh frame) from a transform_matrix of transforms_undistorted.json."""
    c2w = np.array(transform_matrix, dtype=np.float64)
    c2w[2, :] *= -1  # undo: world z row negated
    c2w = c2w[[1, 0, 2, 3], :]  # undo: world x and y rows swapped
    c2w[0:3, 1:3] *= -1  # undo: camera y and z axes flipped
    return c2w


def dslr_intrinsics(transforms: dict) -> tuple[np.ndarray, tuple[int, int]]:
    """(K 3x3, (h, w)) of the undistorted images; the projection model must be PINHOLE."""
    if transforms.get("camera_model", "PINHOLE") != "PINHOLE":
        raise ValueError(f"expected undistorted PINHOLE images, got {transforms.get('camera_model')}")
    K = np.array([[transforms["fl_x"], 0, transforms["cx"]], [0, transforms["fl_y"], transforms["cy"]], [0, 0, 1]], dtype=np.float64)
    return K, (int(transforms["h"]), int(transforms["w"]))


def dslr_frames(transforms: dict) -> dict[str, dict]:
    """All frames (train and test) by image file name."""
    return {Path(f["file_path"]).name: f for f in transforms.get("frames", []) + transforms.get("test_frames", [])}


def pick_views(views: list[str], n: int) -> list[str]:
    """n views spread evenly over a pool that is already in capture order (all of them if the pool is smaller)."""
    if len(views) <= n:
        return list(views)
    return [views[i] for i in np.unique(np.round(np.linspace(0, len(views) - 1, n)).astype(int))]
