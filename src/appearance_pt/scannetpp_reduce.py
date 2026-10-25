"""Reduce a downloaded ScanNet++ scene to what the project keeps, packed into one uncompressed tar.

Kept per scene (docs/decisions.md D11):
- pointcept/<split>/<scene>/*.npy: Pointcept's ScanNet++ preprocessing (parse_scene) of mesh_aligned_0.05.ply
  (coord, color, normal, segment and instance as top-3 int16);
- scans/mesh_aligned_0.05_mask.txt: anonymised vertex indices (the official evaluation ignores them);
- val scenes only: scans/mesh_aligned_0.05.ply, segments.json, segments_anno.json, so the official
  ground truth (toolbox semantic/prep) can be regenerated exactly after the token expires;
- dslr/: a pool of views (images + masks), the full transforms_undistorted.json and train_test_lists.json,
  plus kept_views.json naming the pool.
"""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import numpy as np

from . import THIRD_PARTY

PREPROCESS = THIRD_PARTY / "Pointcept/pointcept/datasets/preprocessing/scannetpp/preprocess_scannetpp.py"
DSLR_POOL = 60  # views kept per scene (user decision 2026-10-25; the pilot uses ~30 of them)
VAL_RAW = ("mesh_aligned_0.05.ply", "segments.json", "segments_anno.json")


def load_preprocess():
    """Pointcept's preprocess_scannetpp.py as a module (it is a script, not part of the package)."""
    spec = importlib.util.spec_from_file_location("preprocess_scannetpp", PREPROCESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def label_setup(pp, root: Path):
    """Label mapping and class indices exactly as preprocess_scannetpp.py's __main__ builds them."""
    import pandas as pd

    bench = Path(root) / "metadata" / "semantic_benchmark"
    names = np.loadtxt(bench / "top100.txt", dtype=str, delimiter=".")
    mapping = pp.filter_map_classes(pd.read_csv(bench / "map_benchmark.csv"), mapping_type="semantic")
    return mapping, {name: i for i, name in enumerate(names)}


def select_views(transforms: dict, n: int = DSLR_POOL) -> list[dict]:
    """Up to n frames spread evenly along the capture order, never an is_bad one.

    Candidates are the "frames" and "test_frames" of transforms_undistorted.json, ordered by image
    file name (the DSLR's running number, i.e. capture order).
    """
    frames = [f for f in transforms.get("frames", []) + transforms.get("test_frames", []) if not f.get("is_bad", False)]
    frames.sort(key=lambda f: Path(f["file_path"]).name)
    if len(frames) <= n:
        return frames
    idx = np.unique(np.round(np.linspace(0, len(frames) - 1, n)).astype(int))
    return [frames[i] for i in idx]


def pack_scene(scene_dir: Path, pointcept_dir: Path, split: str, archive: Path, n_views: int = DSLR_POOL) -> dict:
    """Write the kept files of one downloaded scene to `archive` (uncompressed tar: JPGs and npy barely
    compress). Returns what was kept."""
    scene_dir, scene = Path(scene_dir), Path(scene_dir).name
    dslr = scene_dir / "dslr"
    transforms = json.loads((dslr / "nerfstudio" / "transforms_undistorted.json").read_text())
    views = select_views(transforms, n_views)
    kept = dict(scene=scene, split=split, n_frames=len(transforms.get("frames", [])) + len(transforms.get("test_frames", [])),
                n_bad=sum(bool(f.get("is_bad")) for f in transforms.get("frames", []) + transforms.get("test_frames", [])),
                views=[Path(f["file_path"]).name for f in views], masks=[Path(f["mask_path"]).name for f in views if f.get("mask_path")])
    part = archive.with_name(archive.name + ".part")
    with tarfile.open(part, "w") as tar:
        for path in sorted(Path(pointcept_dir).iterdir()):
            tar.add(path, arcname=f"pointcept/{split}/{scene}/{path.name}")
        tar.add(scene_dir / "scans" / "mesh_aligned_0.05_mask.txt", arcname="scans/mesh_aligned_0.05_mask.txt")
        if split == "val":
            for name in VAL_RAW:
                tar.add(scene_dir / "scans" / name, arcname=f"scans/{name}")
        tar.add(dslr / "nerfstudio" / "transforms_undistorted.json", arcname="dslr/nerfstudio/transforms_undistorted.json")
        tar.add(dslr / "train_test_lists.json", arcname="dslr/train_test_lists.json")
        for name in kept["views"]:
            tar.add(dslr / "resized_undistorted_images" / name, arcname=f"dslr/resized_undistorted_images/{name}")
        for name in kept["masks"]:
            tar.add(dslr / "resized_undistorted_masks" / name, arcname=f"dslr/resized_undistorted_masks/{name}")
        info = json.dumps(kept, indent=1).encode()
        entry = tarfile.TarInfo("dslr/kept_views.json")
        entry.size = len(info)
        tar.addfile(entry, io.BytesIO(info))
    part.rename(archive)
    return kept
