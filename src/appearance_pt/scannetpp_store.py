"""Packed ScanNet++ scenes in the private Kaggle datasets: find them, read them, check them.

The download notebooks wrote one uncompressed tar per scene to <output>/scenes/<scene>.tar, next to manifest.json:
"core" scenes (m2_02; layout of scannetpp_reduce.pack_scene) and "iphone" scenes (m2_04; scannetpp_iphone.reduce_scene).
Kaggle may unpack archives when it builds a dataset, so a scene may also arrive as a directory (scenes/<scene>/ or
scenes/<scene>.tar/); SceneArchive reads both the same way.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import tarfile
from pathlib import Path

import numpy as np

from .scannetpp_iphone import DEPTH_HW

POINTCEPT = dict(coord=np.float32, color=np.uint8, normal=np.float32, segment=np.int16, instance=np.int16)  # (N, 3) each
N_CLASSES = 100  # top-100 semantic benchmark; Pointcept's ignore index is -1
MAGENTA = (255, 0, 255)  # anonymised regions in meshes and images


class SceneError(RuntimeError):
    """A packed scene is incomplete or unreadable."""


class SceneArchive:
    """Read-only view of one packed scene: an uncompressed tar, or the directory it was unpacked into."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.bytes_read = 0
        if self.path.is_dir():
            self._tar = None
            self._members = {p.relative_to(self.path).as_posix(): p for p in self.path.rglob("*") if p.is_file()}
            return
        self._tar = tarfile.open(self.path)
        self._members = {m.name: m for m in self._tar.getmembers() if m.isfile()}
        end = max((m.offset_data + m.size for m in self._members.values()), default=0)
        if end > self.path.stat().st_size:  # a cut-off copy: the last member's data is incomplete
            raise SceneError(f"{self.path.name}: truncated ({self.path.stat().st_size} bytes, members end at {end})")

    def __enter__(self) -> SceneArchive:
        return self

    def __exit__(self, *exc) -> None:
        if self._tar is not None:
            self._tar.close()

    def names(self, prefix: str = "") -> list[str]:
        return sorted(n for n in self._members if n.startswith(prefix))

    def read(self, name: str) -> bytes:
        if name not in self._members:
            raise SceneError(f"{self.path.name}: missing {name}")
        if self._tar is None:
            data = self._members[name].read_bytes()
        else:
            with self._tar.extractfile(self._members[name]) as f:  # tarfile raises ReadError on a short read
                data = f.read()
        self.bytes_read += len(data)
        return data

    def json(self, name: str):
        data = self.read(name)
        return json.loads(gzip.decompress(data) if name.endswith(".gz") else data)

    def npy(self, name: str) -> np.ndarray:
        return np.load(io.BytesIO(self.read(name)))


def kind_of(path: Path) -> str:
    """'core' or 'iphone', from the archive's first entry (a tar is not read further)."""
    path = Path(path)
    if path.is_dir():
        return "iphone" if (path / "iphone").is_dir() else "core"
    with tarfile.open(path) as tar:
        first = tar.next()
    return "iphone" if first is not None and first.name.startswith("iphone/") else "core"


def find_scenes(root: Path) -> dict[str, dict[str, list[Path]]]:
    """{kind: {scene: [archives]}} for every .../scenes/<scene>[.tar] below root; more than one archive = a duplicate."""
    found, seen = {"core": {}, "iphone": {}}, set()
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        if Path(dirpath).name != "scenes":
            continue
        dirnames[:] = []
        for entry in sorted(Path(dirpath).iterdir()):
            if not (entry.is_dir() or entry.suffix == ".tar") or entry.resolve() in seen:
                continue
            seen.add(entry.resolve())
            found[kind_of(entry)].setdefault(entry.name.removesuffix(".tar"), []).append(entry)
    return found


def decode_image(data: bytes) -> np.ndarray:
    """Every pixel decoded (PIL raises on a truncated or corrupt file)."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return np.asarray(im)


def grid_points(coord: np.ndarray, size: float) -> int:
    """Occupied cells of a `size` grid: the points GridSample keeps (up to where the grid starts)."""
    g = np.floor(coord / size).astype(np.int64)
    g -= g.min(0)
    return len(np.unique((g[:, 0] << 42) | (g[:, 1] << 21) | g[:, 2]))


def check_core(arc: SceneArchive, scene: str, split: str) -> dict:
    """Read and decode every file of a core scene; raise SceneError on the first problem, else return its numbers."""
    pc = f"pointcept/{split}/{scene}/"
    a = {k: arc.npy(f"{pc}{k}.npy") for k in POINTCEPT}
    n = len(a["coord"])
    for k, dtype in POINTCEPT.items():
        if a[k].shape != (n, 3) or a[k].dtype != dtype:
            raise SceneError(f"{k}.npy is {a[k].dtype} {a[k].shape}, expected {np.dtype(dtype)} ({n}, 3)")
    if not (np.isfinite(a["coord"]).all() and np.isfinite(a["normal"]).all()):
        raise SceneError("non-finite coord or normal")
    if a["segment"].min() < -1 or a["segment"].max() >= N_CLASSES:
        raise SceneError(f"segment labels outside [-1, {N_CLASSES - 1}]")
    anon = np.array([int(t) for t in re.split(rb"[\s,]+", arc.read("scans/mesh_aligned_0.05_mask.txt")) if t], np.int64)
    if len(anon) and (anon.min() < 0 or anon.max() >= n):
        raise SceneError("anonymisation mask indexes past the vertices")
    magenta = (a["color"] == MAGENTA).all(1)

    if split == "val":
        ply = arc.read("scans/mesh_aligned_0.05.ply")
        vertices = re.search(rb"element vertex (\d+)", ply[:ply.find(b"end_header")])
        if not vertices or int(vertices.group(1)) != n:
            raise SceneError(f"mesh_aligned_0.05.ply header does not have {n} vertices")
        if len(arc.json("scans/segments.json")["segIndices"]) != n:
            raise SceneError(f"segments.json does not have {n} vertices")
        if "segGroups" not in arc.json("scans/segments_anno.json"):
            raise SceneError("segments_anno.json has no segGroups")

    transforms = arc.json("dslr/nerfstudio/transforms_undistorted.json")
    arc.json("dslr/train_test_lists.json")
    kept = arc.json("dslr/kept_views.json")
    frames = {Path(f["file_path"]).name: f for f in transforms.get("frames", []) + transforms.get("test_frames", [])}
    image_hw = set()
    for name in kept["views"]:
        if name not in frames or frames[name].get("is_bad"):
            raise SceneError(f"view {name} is not a good frame of transforms_undistorted.json")
        image_hw.add(decode_image(arc.read(f"dslr/resized_undistorted_images/{name}")).shape[:2])
    for name in kept["masks"]:
        if decode_image(arc.read(f"dslr/resized_undistorted_masks/{name}")).shape[:2] not in image_hw:
            raise SceneError(f"mask {name} does not match the image size")
    return dict(n_vertices=n, n_grid_2cm=grid_points(a["coord"], 0.02),
                labelled=round(float((a["segment"][:, 0] >= 0).mean()), 4),
                n_anonymised=len(anon), n_anonymised_magenta=int(magenta[anon].sum()), n_magenta=int(magenta.sum()),
                n_views=len(kept["views"]), n_masks=len(kept["masks"]), camera_model=transforms.get("camera_model"),
                image_hw=sorted(f"{h}x{w}" for h, w in image_hw), transforms_hw=f"{transforms.get('h')}x{transforms.get('w')}")


def check_iphone(arc: SceneArchive, scene: str, split: str) -> dict:
    """Read and decode every file of an iPhone scene; raise SceneError on the first problem, else return its numbers."""
    info = arc.json("iphone/pool.json")
    pool = info["pool"]
    if not pool or len(set(pool)) != len(pool):
        raise SceneError("empty or repeated frame pool")
    for sub in ("rgb", "rgb_masks", "depth"):
        if len(arc.names(f"iphone/{sub}/")) != len(pool):
            raise SceneError(f"iphone/{sub} has {len(arc.names(f'iphone/{sub}/'))} files for {len(pool)} pool frames")
    poses = arc.json("iphone/pose_intrinsic_imu.json.gz")
    registered = set(re.findall(r"frame_\d{6}\.jpg", gzip.decompress(arc.read("iphone/colmap/images.txt.gz")).decode()))
    cameras = gzip.decompress(arc.read("iphone/colmap/cameras.txt.gz")).decode()
    models = sorted({line.split()[1] for line in cameras.splitlines() if line.strip() and not line.startswith("#")})
    rgb_hw, zero, median, mask_on = set(), [], [], []
    for frame in pool:
        key = f"frame_{frame:06d}"
        entry = poses.get(key, {})
        if np.shape(entry.get("aligned_pose")) != (4, 4) or np.shape(entry.get("intrinsic")) != (3, 3):
            raise SceneError(f"{key}: no aligned_pose (4x4) and intrinsic (3x3) in pose_intrinsic_imu.json")
        if f"{key}.jpg" not in registered:
            raise SceneError(f"{key} is not in COLMAP images.txt")
        rgb = decode_image(arc.read(f"iphone/rgb/{key}.jpg"))
        mask = decode_image(arc.read(f"iphone/rgb_masks/{key}.png"))
        depth = decode_image(arc.read(f"iphone/depth/{key}.png"))
        if rgb.ndim != 3 or mask.shape[:2] != rgb.shape[:2]:
            raise SceneError(f"{key}: rgb {rgb.shape}, mask {mask.shape}")
        if depth.shape != DEPTH_HW or depth.dtype.kind not in "iu":
            raise SceneError(f"{key}: depth {depth.dtype} {depth.shape}, expected integer mm {DEPTH_HW}")
        rgb_hw.add(rgb.shape[:2])
        zero.append(float((depth == 0).mean()))
        median.append(float(np.median(depth[depth > 0])) if (depth > 0).any() else 0.0)
        mask_on.append(float((mask > 127).mean()))
    return dict(n_pool=len(pool), n_video_frames=info.get("n_video_frames"), n_registered=len(registered),
                depth_encoding=info.get("depth_encoding"), camera_models=models,
                rgb_hw=sorted(f"{h}x{w}" for h, w in rgb_hw), depth_zero_max=round(max(zero), 4),
                depth_median_mm=int(np.median(median)), mask_on=round(float(np.mean(mask_on)), 4))
