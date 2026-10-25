"""ScanNet++ iPhone data: pick a frame pool, decode only those frames, pack what we keep.

Formats (measured by notebook m2_03, docs/decisions.md D12): rgb.mkv and rgb_mask.mkv are 1920x1440 at 60 fps;
depth.bin holds one lz4-compressed uint16 (mm) 256x192 frame per video frame (the toolbox's
iphone/prepare_iphone_data.py also handles a zlib float32 variant); pose_intrinsic_imu.json has, per frame,
timestamp, pose, aligned_pose (4x4), intrinsic (3x3) and imu; COLMAP (OPENCV camera model, so with lens
distortion) registers every 10th frame, named frame_%06d.jpg after the video frame number.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
import shutil
import subprocess
import tarfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import THIRD_PARTY

IPHONE_POOL = 60  # frames kept per scene (user decision 2026-10-25)
DEPTH_HW = (192, 256)  # toolbox prepare_iphone_data.py: height, width = 192, 256


def toolbox_colmap():
    """The toolbox's common/utils/colmap.py (read_cameras_text, read_images_text), loaded from the submodule."""
    path = THIRD_PARTY / "scannetpp" / "common" / "utils" / "colmap.py"
    spec = importlib.util.spec_from_file_location("scannetpp_colmap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def registered_frames(colmap, colmap_dir: Path) -> list[int]:
    """Video frame numbers of the COLMAP-registered images (frame_000120.jpg -> 120), sorted."""
    images = colmap.read_images_text(Path(colmap_dir) / "images.txt")
    return sorted(int(Path(im.name).stem.split("_")[-1]) for im in images.values())


def select_pool(frames: list[int], n: int = IPHONE_POOL) -> list[int]:
    """Up to n frame numbers spread evenly over the given (registered) frames."""
    frames = sorted(frames)
    if len(frames) <= n:
        return frames
    return [frames[i] for i in np.unique(np.round(np.linspace(0, len(frames) - 1, n)).astype(int))]


def read_depth(path: Path, wanted: set[int]) -> tuple[str, int, dict[int, np.ndarray]]:
    """(encoding, number of frames, {frame number: depth in mm, uint16}) for the wanted frames.

    Same decoding as the toolbox: first try one zlib block of float32 metres, else per-frame records
    (4-byte little-endian size + lz4 uint16 mm, or zlib float32 metres).
    """
    h, w = DEPTH_HW
    data = Path(path).read_bytes()
    try:
        depth = np.frombuffer(zlib.decompress(data, wbits=-zlib.MAX_WBITS), dtype=np.float32).reshape(-1, h, w)
        return "zlib float32 (whole file)", len(depth), {i: (depth[i] * 1000).astype(np.uint16) for i in wanted if i < len(depth)}
    except (zlib.error, ValueError):  # per-frame records can inflate to garbage that does not reshape
        pass
    out, frame, offset, encodings = {}, 0, 0, set()
    while offset < len(data):
        size = int.from_bytes(data[offset:offset + 4], "little")
        if frame in wanted:
            chunk = data[offset + 4:offset + 4 + size]
            try:
                import lz4.block

                d = np.frombuffer(lz4.block.decompress(chunk, uncompressed_size=h * w * 2), dtype=np.uint16).reshape(h, w)
                encodings.add("lz4 uint16 mm")
            except Exception:  # noqa: BLE001  (the toolbox falls back to zlib float32 per frame)
                d = (np.frombuffer(zlib.decompress(chunk, wbits=-zlib.MAX_WBITS), dtype=np.float32).reshape(h, w) * 1000).astype(np.uint16)
                encodings.add("zlib float32 m")
            out[frame] = d
        offset += 4 + size
        frame += 1
    return "per frame: " + ", ".join(sorted(encodings)), frame, out


def extract_frames(ffmpeg: str, video: Path, frames: list[int], out_dir: Path, ext: str, gray: bool = False) -> float:
    """Decode only the given frame numbers with ffmpeg's select filter (exact frame numbers, as the toolbox
    does) into out_dir/frame_%06d<ext>. Returns seconds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / "select.txt"
    script.write_text("select='" + "+".join(f"eq(n,{i})" for i in frames) + "'")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-filter_script:v", str(script), "-vsync", "vfr"]
    cmd += ["-pix_fmt", "gray"] if gray else ["-q:v", "2"]
    t0 = time.perf_counter()
    subprocess.run(cmd + [str(out_dir / f"tmp_%06d{ext}")], check=True)
    seconds = time.perf_counter() - t0
    script.unlink()
    produced = sorted(out_dir.glob(f"tmp_*{ext}"))
    if len(produced) != len(frames):
        raise RuntimeError(f"{video.name}: ffmpeg produced {len(produced)} frames, expected {len(frames)}")
    for tmp, index in zip(produced, sorted(frames)):
        tmp.rename(out_dir / f"frame_{index:06d}{ext}")
    return seconds


def reduce_scene(colmap, ffmpeg: str, iphone_dir: Path, archive: Path, n: int = IPHONE_POOL) -> dict:
    """Extract the frame pool of one downloaded scene's iPhone data and pack it into `archive` (uncompressed tar;
    JSON and COLMAP text are gzipped inside). The large points3D.txt is not kept."""
    import imageio.v2 as iio

    iphone_dir = Path(iphone_dir)
    registered = registered_frames(colmap, iphone_dir / "colmap")
    pool = select_pool(registered, n)
    work = iphone_dir / "pool"
    with ThreadPoolExecutor(2) as decode:  # the two videos decode in parallel
        rgb = decode.submit(extract_frames, ffmpeg, iphone_dir / "rgb.mkv", pool, work / "rgb", ".jpg")
        mask = decode.submit(extract_frames, ffmpeg, iphone_dir / "rgb_mask.mkv", pool, work / "rgb_masks", ".png", True)
        seconds_rgb, seconds_mask = rgb.result(), mask.result()
    encoding, n_depth, depth = read_depth(iphone_dir / "depth.bin", set(pool))
    (work / "depth").mkdir(parents=True, exist_ok=True)
    for i, d in depth.items():
        iio.imwrite(work / "depth" / f"frame_{i:06d}.png", d)
    info = dict(pool=pool, n_video_frames=len(json.loads((iphone_dir / "pose_intrinsic_imu.json").read_text())),
                n_registered=len(registered), n_depth_frames=n_depth,
                depth_encoding=encoding, seconds_rgb=round(seconds_rgb, 1), seconds_mask=round(seconds_mask, 1))
    if len(depth) != len(pool):
        raise RuntimeError(f"depth frames missing for {sorted(set(pool) - set(depth))[:5]}")

    def add_bytes(tar, name, data: bytes):
        entry = tarfile.TarInfo(name)
        entry.size = len(data)
        tar.addfile(entry, io.BytesIO(data))

    part = archive.with_name(archive.name + ".part")
    with tarfile.open(part, "w") as tar:
        for sub in ("rgb", "rgb_masks", "depth"):
            for path in sorted((work / sub).iterdir()):
                tar.add(path, arcname=f"iphone/{sub}/{path.name}")
        add_bytes(tar, "iphone/pose_intrinsic_imu.json.gz", gzip.compress((iphone_dir / "pose_intrinsic_imu.json").read_bytes()))
        for name in ("cameras.txt", "images.txt"):
            add_bytes(tar, f"iphone/colmap/{name}.gz", gzip.compress((iphone_dir / "colmap" / name).read_bytes()))
        add_bytes(tar, "iphone/pool.json", json.dumps(info, indent=1).encode())
    part.rename(archive)
    return info
