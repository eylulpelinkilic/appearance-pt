"""Scene inputs for the pilot (dicts with coord, color 0-255, normal; numpy), and the loader for packed ScanNet++
core scenes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

KEYS = ("coord", "color", "normal")


def mask_valid(m: np.ndarray) -> np.ndarray:
    """Usable pixels of a DSLR anonymisation mask (255 = usable, the nerfstudio convention)."""
    m = m[..., 0] if m.ndim == 3 else m
    return m > 0 if m.dtype == bool else m > 127


def load_core_scene(path: Path, scene: str, n_views: int) -> dict:
    """A packed core scene (train or val): Pointcept arrays (coord, color, normal, segment), the anonymised vertex
    ids, the undistorted DSLR intrinsics and n_views of the kept views (evenly over the capture order), each with its
    nerfstudio transform_matrix, usable-pixel mask and JPEG bytes."""
    from appearance_pt.scannetpp_cameras import dslr_frames, dslr_intrinsics, pick_views
    from appearance_pt.scannetpp_store import SceneArchive, anonymised_vertices, decode_image

    with SceneArchive(path) as arc:
        split = arc.names("pointcept/")[0].split("/")[1]
        pc = f"pointcept/{split}/{scene}/"
        s = {k: arc.npy(f"{pc}{k}.npy") for k in ("coord", "color", "normal", "segment")}
        s["split"] = split
        s["anonymised"] = np.unique(anonymised_vertices(arc))
        transforms = arc.json("dslr/nerfstudio/transforms_undistorted.json")
        frames = dslr_frames(transforms)
        s["K"], s["hw"] = dslr_intrinsics(transforms)
        s["views"] = []
        for name in pick_views(arc.json("dslr/kept_views.json")["views"], n_views):
            mask = frames[name].get("mask_path")
            valid = (mask_valid(decode_image(arc.read(f"dslr/resized_undistorted_masks/{Path(mask).name}"))) if mask
                     else np.ones(s["hw"], bool))
            s["views"].append(dict(name=name, T=frames[name]["transform_matrix"], valid=valid,
                                   jpg=arc.read(f"dslr/resized_undistorted_images/{name}")))
    return s


def raw_features(s: dict) -> np.ndarray:
    """Config X: (N, 9) float32 xyz (x, y centred on the scene, z above its 1st-percentile height), colour / 255 and
    normal."""
    c = s["coord"].astype(np.float32)
    xyz = np.c_[c[:, :2] - c[:, :2].mean(0), c[:, 2] - np.percentile(c[:, 2], 1)]
    return np.c_[xyz, s["color"] / 255.0, s["normal"]].astype(np.float32)


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
