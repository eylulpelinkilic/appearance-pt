"""Trial of the iPhone data for a few ScanNet++ scenes, before the full (230 GB) iPhone download.

Per scene: downloads the iPhone assets (rgb.mkv, rgb_mask.mkv, depth.bin, pose_intrinsic_imu.json, colmap/)
to temporary storage, then reports what they contain: video size/fps/frames, which depth.bin encoding is
used (one zlib block of float32, or per-frame lz4 uint16 / zlib float32, as handled by the toolbox's
iphone/prepare_iphone_data.py), the pose file's fields, and the COLMAP-registered frames. It extracts a pool
of up to 60 frames spread evenly over the COLMAP-registered frames (RGB as JPG, mask and depth as PNG),
times this and measures what we would keep. Deletes everything; only metrics are written to --out.

Usage: python scripts/scannetpp/scannetpp_iphone_trial.py [--scenes ID ...] [--out DIR] [--tmp DIR]
Needs the Kaggle secret SCANNETPP_TOKEN, the token-free download config, opencv, lz4 and ffmpeg
(system ffmpeg, else the imageio-ffmpeg binary).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from appearance_pt import REPO_ROOT, THIRD_PARTY
from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.scannetpp_download import (ASSET_GROUPS, SPLITS, ScannetppClient, find_config, kaggle_secret,
                                              load_config, scene_files)

MB = 1e6
POOL = 60
DEPTH_HW = (192, 256)  # toolbox: height, width = 192, 256


def toolbox_colmap():
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


def video_info(path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    info = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=round(cap.get(cv2.CAP_PROP_FPS), 2), frames_reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def read_depth(path: Path, wanted: set[int]) -> tuple[str, int, dict[int, np.ndarray]]:
    """(encoding, number of frames, {frame index: depth in mm as uint16}) for the wanted frame indices."""
    h, w = DEPTH_HW
    data = path.read_bytes()
    try:  # one zlib block of float32 metres
        depth = np.frombuffer(zlib.decompress(data, wbits=-zlib.MAX_WBITS), dtype=np.float32).reshape(-1, h, w)
        return "zlib float32 (whole file)", len(depth), {i: (depth[i] * 1000).astype(np.uint16) for i in wanted if i < len(depth)}
    except zlib.error:
        pass
    import lz4.block

    out, frame, offset, encodings = {}, 0, 0, set()
    while offset < len(data):
        size = int.from_bytes(data[offset:offset + 4], "little")
        chunk = data[offset + 4:offset + 4 + size]
        offset += 4 + size
        if frame in wanted or frame == 0:
            try:
                d = np.frombuffer(lz4.block.decompress(chunk, uncompressed_size=h * w * 2), dtype=np.uint16).reshape(h, w)
                encodings.add("lz4 uint16 mm")
            except Exception:  # noqa: BLE001  (the toolbox falls back to zlib float32 per frame)
                d = (np.frombuffer(zlib.decompress(chunk, wbits=-zlib.MAX_WBITS), dtype=np.float32).reshape(h, w) * 1000).astype(np.uint16)
                encodings.add("zlib float32 m")
            if frame in wanted:
                out[frame] = d
        frame += 1
    return "per frame: " + ", ".join(sorted(encodings)), frame, out


def extract_frames(ffmpeg: str, video: Path, indices: list[int], out_dir: Path, ext: str, gray: bool = False) -> float:
    """Decode only the given frame numbers (ffmpeg select filter, as the toolbox does); returns seconds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / "select.txt"
    script.write_text("select='" + "+".join(f"eq(n,{i})" for i in indices) + "'")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-filter_script:v", str(script), "-vsync", "vfr"]
    cmd += ["-pix_fmt", "gray"] if gray else ["-q:v", "2"]
    t0 = time.perf_counter()
    subprocess.run(cmd + [str(out_dir / f"tmp_%06d{ext}")], check=True)
    seconds = time.perf_counter() - t0
    script.unlink()
    for tmp, index in zip(sorted(out_dir.glob(f"tmp_*{ext}")), indices):
        tmp.rename(out_dir / f"frame_{index:06d}{ext}")
    return seconds


def run_scene(client, cfg, colmap, ffmpeg, scene: str, split: str, root: Path) -> dict:
    files = scene_files(cfg, scene, SPLITS[split], list(ASSET_GROUPS["iphone"]))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(4) as pool:
        fetched = list(pool.map(lambda f: client.fetch(f, root), files))
    download_s = time.perf_counter() - t0
    iphone = root / "data" / scene / "iphone"
    row = dict(scene=scene, split=split, download_mb=round(sum(f["download_bytes"] for f in fetched) / MB, 1),
               mb_per_s=round(sum(f["download_bytes"] for f in fetched) / MB / download_s, 2),
               files_mb={f["asset"]: round(f["unpacked_bytes"] / MB, 1) for f in fetched},
               colmap_files=sorted(p.name for p in (iphone / "colmap").iterdir()))
    row["rgb"], row["mask"] = video_info(iphone / "rgb.mkv"), video_info(iphone / "rgb_mask.mkv")

    poses = json.loads((iphone / "pose_intrinsic_imu.json").read_text())
    first = poses[sorted(poses)[0]]
    row["pose_file"] = dict(n=len(poses), first_key=sorted(poses)[0],
                            fields={k: list(np.shape(v)) for k, v in first.items()})
    images = colmap.read_images_text(iphone / "colmap" / "images.txt")
    registered = sorted(int(Path(im.name).stem.split("_")[-1]) for im in images.values())
    camera = list(colmap.read_cameras_text(iphone / "colmap" / "cameras.txt").values())[0]
    row["colmap"] = dict(n_registered=len(registered), first=registered[:3], last=registered[-1],
                         camera_model=camera.model, camera_wh=[camera.width, camera.height])

    idx = np.unique(np.round(np.linspace(0, len(registered) - 1, min(POOL, len(registered)))).astype(int))
    pool = [registered[i] for i in idx]
    out = root / "pool" / scene
    row["seconds_rgb"] = round(extract_frames(ffmpeg, iphone / "rgb.mkv", pool, out / "rgb", ".jpg"), 1)
    row["seconds_mask"] = round(extract_frames(ffmpeg, iphone / "rgb_mask.mkv", pool, out / "rgb_masks", ".png", gray=True), 1)
    t1 = time.perf_counter()
    encoding, n_depth, depth = read_depth(iphone / "depth.bin", set(pool))
    import imageio.v2 as iio

    (out / "depth").mkdir(parents=True, exist_ok=True)
    for i, d in depth.items():
        iio.imwrite(out / "depth" / f"frame_{i:06d}.png", d)
    row["depth"] = dict(encoding=encoding, n_frames=n_depth, seconds=round(time.perf_counter() - t1, 1),
                        valid_fraction=round(float(np.mean([np.mean(d > 0) for d in depth.values()])), 3) if depth else None,
                        median_m=round(float(np.median([np.median(d[d > 0]) for d in depth.values() if (d > 0).any()])) / 1000, 2) if depth else None)
    sizes = {sub: [p.stat().st_size for p in (out / sub).iterdir()] for sub in ("rgb", "rgb_masks", "depth")}
    row["pool"] = {sub: dict(n=len(s), mean_mb=round(statistics.mean(s) / MB, 3) if s else 0) for sub, s in sizes.items()}
    row["kept_mb"] = round(sum(sum(s) for s in sizes.values()) / MB
                           + (iphone / "pose_intrinsic_imu.json").stat().st_size / MB
                           + sum(p.stat().st_size for p in (iphone / "colmap").iterdir() if p.name != "points3D.txt") / MB, 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", help="scene IDs (default: the first 3 val scenes of the subset)")
    ap.add_argument("--config", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/scannetpp_iphone_trial"))
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/scannetpp_iphone_trial"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "env_record.json").write_text(json.dumps(collect(), indent=2))

    subset = json.loads((REPO_ROOT / "configs" / "scannetpp_subset.json").read_text())
    split_of = {**{s: "train" for s in subset["train"]}, **{s: "val" for s in subset["val"]}}
    scenes = args.scenes or subset["val"][:3]
    ffmpeg = ffmpeg_exe()
    print(f"scenes {scenes}; ffmpeg {ffmpeg}; free disk {shutil.disk_usage(args.tmp.parent).free / 1e9:.0f} GB")
    cfg = load_config(find_config(args.config))
    client = ScannetppClient(cfg, kaggle_secret("SCANNETPP_TOKEN"))
    colmap = toolbox_colmap()

    section("iPhone scenes: download, inspect, extract a 60-frame pool, delete")
    rows = []
    for scene in scenes:
        try:
            row = run_scene(client, cfg, colmap, ffmpeg, scene, split_of.get(scene, "val"), args.tmp)
        except Exception as e:  # noqa: BLE001  (record and go on)
            row = dict(scene=scene, error=f"{type(e).__name__}: {e}"[:300])
        rows.append(row)
        print(json.dumps(row), flush=True)
        shutil.rmtree(args.tmp / "data" / scene, ignore_errors=True)
        shutil.rmtree(args.tmp / "pool" / scene, ignore_errors=True)
        (args.out / "iphone_trial.json").write_text(json.dumps(rows, indent=1))

    section("SUMMARY (paste this block back)")
    for r in rows:
        if "error" in r:
            print(f"{r['scene']}: ERROR {r['error']}")
            continue
        print(f"{r['scene']} ({r['split']}): download {r['download_mb']} MB at {r['mb_per_s']} MB/s, files MB {r['files_mb']}")
        print(f"  rgb {r['rgb']} | mask {r['mask']}")
        print(f"  depth {r['depth']}")
        print(f"  pose file: {r['pose_file']}")
        print(f"  colmap {r['colmap']}, files {r['colmap_files']}")
        print(f"  pool {r['pool']}; extraction s: rgb {r['seconds_rgb']}, mask {r['seconds_mask']}, depth {r['depth']['seconds']}; "
              f"kept {r['kept_mb']} MB")
    return 0 if all("error" not in r for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
