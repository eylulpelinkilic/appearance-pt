"""Trial download of a few ScanNet++ scenes before the full download: speed, sizes, file formats.

For each scene: downloads the core assets (mesh + labels; undistorted DSLR images, masks, poses) to a
temporary directory and measures download speed (the first two scenes one file at a time, the rest
with parallel requests, to see whether concurrency helps). It records unpacked sizes, runs Pointcept's
ScanNet++ preprocessing (parse_scene) and measures what we would keep. It also reports what the DSLR
files contain (image count and resolution, pose-file fields such as a bad-frame flag) and what the
anonymisation mask looks like, then deletes the scene. Only metrics are written to --out.

Default scenes: the first 3 val scenes of configs/scannetpp_subset.json, plus every subset scene whose
ID is not 10 hex characters (checks that those scenes' files are real).

Usage: python scripts/scannetpp/scannetpp_trial.py [--scenes ID ...] [--workers 4] [--out DIR] [--tmp DIR]
Needs the Kaggle secret SCANNETPP_TOKEN, the token-free download config (input dataset) and open3d.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from appearance_pt import REPO_ROOT, THIRD_PARTY
from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.scannetpp_download import (CORE_ASSETS, SPLITS, ScannetppClient, find_config, kaggle_secret,
                                              load_config, scene_files)

MB = 1e6
VIEWS_KEPT = 30  # DSLR views per scene the pilot keeps
CORE_DOWNLOAD_GB, IPHONE_DOWNLOAD_GB = 102.0, 230.2  # whole subset, from m2_00 (docs/decisions.md D9)
PREPROCESS = THIRD_PARTY / "Pointcept/pointcept/datasets/preprocessing/scannetpp/preprocess_scannetpp.py"


def load_preprocess():
    spec = importlib.util.spec_from_file_location("preprocess_scannetpp", PREPROCESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def label_setup(pp, root: Path):
    """Label mapping and class indices exactly as preprocess_scannetpp.py's __main__ builds them."""
    import pandas as pd

    bench = root / "metadata" / "semantic_benchmark"
    names = np.loadtxt(bench / "top100.txt", dtype=str, delimiter=".")
    mapping = pp.filter_map_classes(pd.read_csv(bench / "map_benchmark.csv"), mapping_type="semantic")
    return mapping, {name: i for i, name in enumerate(names)}


def dir_stats(path: Path) -> dict:
    files = [p for p in path.rglob("*") if p.is_file()]
    sizes = [p.stat().st_size for p in files] or [0]
    return dict(n=len(files), ext=dict(Counter(p.suffix for p in files)), mb=round(sum(sizes) / MB, 1),
                mean_mb=round(statistics.mean(sizes) / MB, 3))


def inspect_dslr(scene_dir: Path) -> dict:
    from PIL import Image

    dslr = scene_dir / "dslr"
    images, masks = dslr / "resized_undistorted_images", dslr / "resized_undistorted_masks"
    first = next(p for p in sorted(images.rglob("*")) if p.is_file())
    with Image.open(first) as im:
        image_size = list(im.size)
    tf = json.loads((dslr / "nerfstudio" / "transforms_undistorted.json").read_text())
    frames = tf.get("frames", [])
    frame_keys = sorted(set().union(*(f.keys() for f in frames))) if frames else []
    lists = json.loads((dslr / "train_test_lists.json").read_text())
    return dict(
        images=dir_stats(images), masks=dir_stats(masks), image_size_wh=image_size,
        transforms_scalar_keys={k: v for k, v in tf.items() if not isinstance(v, (list, dict))},
        transforms_list_lengths={k: len(v) for k, v in tf.items() if isinstance(v, list)},
        frame_keys=frame_keys,
        frame_true_counts={k: sum(bool(f.get(k)) for f in frames) for k in frame_keys if k.startswith("is_")},
        train_test_lists={k: len(v) if isinstance(v, list) else type(v).__name__ for k, v in lists.items()},
        pose_file_mb=round((dslr / "nerfstudio" / "transforms_undistorted.json").stat().st_size / MB, 3),
    )


def inspect_mask(path: Path, n_vertices: int) -> dict:
    """mesh_aligned_0.05_mask.txt: eval_semantic.py loads it as a LongTensor; is it 0/1 per vertex or indices?"""
    values = np.loadtxt(path, dtype=np.int64, ndmin=1)
    per_vertex = len(values) == n_vertices and set(np.unique(values).tolist()) <= {0, 1}
    return dict(length=int(len(values)), n_unique=int(len(np.unique(values))), max=int(values.max()) if len(values) else None,
                looks_like="0/1 per vertex" if per_vertex else "vertex indices",
                anonymised_vertices=int(values.sum()) if per_vertex else int(len(np.unique(values))))


def preprocess(pp, labels, name: str, split: str, root: Path, out: Path) -> dict:
    t0 = time.perf_counter()
    pp.parse_scene(name, split, root, out, labels[0], labels[1], -1)
    scene_out = out / split / name
    segment = np.load(scene_out / "segment.npy")
    files = {p.name: round(p.stat().st_size / MB, 1) for p in sorted(scene_out.iterdir())}
    return dict(seconds=round(time.perf_counter() - t0, 1), n_vertices=int(len(segment)), files_mb=files,
                mb=round(sum(files.values()), 1), labelled=round(float((segment[:, 0] >= 0).mean()), 3),
                n_classes=int(len(np.unique(segment[:, 0][segment[:, 0] >= 0]))))


def run_scene(client, cfg, pp, labels, scene: str, split: str, root: Path, workers: int) -> dict:
    files = scene_files(cfg, scene, SPLITS[split], CORE_ASSETS)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        fetched = list(pool.map(lambda f: client.fetch(f, root), files))
    wall = time.perf_counter() - t0
    download = sum(f["download_bytes"] for f in fetched)
    scene_dir = root / "data" / scene
    row = dict(scene=scene, split=split, workers=workers, download_mb=round(download / MB, 1),
               unpacked_mb=round(sum(f["unpacked_bytes"] for f in fetched) / MB, 1), seconds=round(wall, 1),
               mb_per_s=round(download / MB / wall, 2),
               per_asset_mb={f["asset"]: round(f["unpacked_bytes"] / MB, 1) for f in fetched})
    row["dslr"] = inspect_dslr(scene_dir)
    row["points"] = preprocess(pp, labels, scene, split, root, root / "processed")
    row["mask"] = inspect_mask(scene_dir / "scans" / "mesh_aligned_0.05_mask.txt", row["points"]["n_vertices"])
    d = row["dslr"]
    row["kept_mb"] = round(row["points"]["mb"] + VIEWS_KEPT * (d["images"]["mean_mb"] + d["masks"]["mean_mb"])
                           + d["pose_file_mb"] + (scene_dir / "scans" / "mesh_aligned_0.05_mask.txt").stat().st_size / MB, 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", help="scene IDs (default: 3 val scenes + odd-looking IDs)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests from the third scene on")
    ap.add_argument("--config", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/scannetpp_trial"))
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/scannetpp_trial"), help="downloads (deleted afterwards)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "env_record.json").write_text(json.dumps(collect(), indent=2))

    subset = json.loads((REPO_ROOT / "configs" / "scannetpp_subset.json").read_text())
    split_of = {**{s: "train" for s in subset["train"]}, **{s: "val" for s in subset["val"]}}
    odd = [s for s in split_of if not re.fullmatch(r"[0-9a-f]{10}", s)]
    scenes = args.scenes or subset["val"][:3] + odd
    print(f"scenes: {scenes} (IDs that are not 10 hex characters: {odd})")
    print(f"free disk at {args.tmp.parent}: {shutil.disk_usage(args.tmp.parent).free / 1e9:.0f} GB")

    cfg = load_config(find_config(args.config))
    client = ScannetppClient(cfg, kaggle_secret("SCANNETPP_TOKEN"))
    section("1. Metadata files")
    for path in cfg["meta_files"]:
        client.download(path, args.tmp / path)
    for key, split in SPLITS.items():
        ids = (args.tmp / "splits" / f"{split}.txt").read_text().split()
        bad = [s for s in ids if not re.fullmatch(r"[0-9a-f]{10}", s)]
        print(f"  {split}: {len(ids)} scenes, not 10-hex: {bad}")
    pp = load_preprocess()
    labels = label_setup(pp, args.tmp)
    print(f"  label mapping: {len(labels[1])} classes")

    section("2. Scenes: download, inspect, preprocess, delete")
    rows = []
    for i, scene in enumerate(scenes):
        workers = 1 if i < 2 else args.workers
        try:
            row = run_scene(client, cfg, pp, labels, scene, split_of.get(scene, "val"), args.tmp, workers)
        except Exception as e:  # noqa: BLE001  (record and go on: one bad scene must not end the trial)
            row = dict(scene=scene, split=split_of.get(scene), workers=workers, error=f"{type(e).__name__}: {e}")
        rows.append(row)
        print(json.dumps(row))
        shutil.rmtree(args.tmp / "data" / scene, ignore_errors=True)
        shutil.rmtree(args.tmp / "processed", ignore_errors=True)
        (args.out / "trial.json").write_text(json.dumps(rows, indent=1))

    section("SUMMARY (paste this block back)")
    ok = [r for r in rows if "error" not in r]
    print(f"{'scene':14s} {'split':5s} {'wrk':>3s} {'down_MB':>8s} {'unpk_MB':>8s} {'sec':>6s} {'MB/s':>6s} "
          f"{'verts':>9s} {'pts_MB':>7s} {'images':>6s} {'img_MB':>6s} {'kept_MB':>7s}")
    for r in rows:
        if "error" in r:
            print(f"{r['scene']:14s} {r['split'] or '?':5s} ERROR {r['error'][:150]}")
            continue
        print(f"{r['scene']:14s} {r['split']:5s} {r['workers']:>3d} {r['download_mb']:>8.1f} {r['unpacked_mb']:>8.1f} "
              f"{r['seconds']:>6.1f} {r['mb_per_s']:>6.2f} {r['points']['n_vertices']:>9d} {r['points']['mb']:>7.1f} "
              f"{r['dslr']['images']['n']:>6d} {r['dslr']['images']['mean_mb']:>6.3f} {r['kept_mb']:>7.1f}")
    for workers in sorted({r["workers"] for r in ok}):
        speeds = [r["mb_per_s"] for r in ok if r["workers"] == workers]
        mbs = statistics.mean(speeds)
        print(f"workers={workers}: {mbs:.2f} MB/s mean -> core {CORE_DOWNLOAD_GB:.0f} GB in "
              f"{CORE_DOWNLOAD_GB * 1e3 / mbs / 3600:.1f} h, iPhone {IPHONE_DOWNLOAD_GB:.0f} GB in "
              f"{IPHONE_DOWNLOAD_GB * 1e3 / mbs / 3600:.1f} h")
    if ok:
        d = ok[0]["dslr"]
        print(f"DSLR ({ok[0]['scene']}): image {d['image_size_wh']}, images {d['images']}, masks {d['masks']}")
        print(f"  pose file: scalar keys {json.dumps(d['transforms_scalar_keys'])}")
        print(f"  pose file: lists {d['transforms_list_lengths']}, frame keys {d['frame_keys']}")
        print(f"  bad-frame style flags (count true) per scene: "
              f"{json.dumps({r['scene']: r['dslr']['frame_true_counts'] for r in ok})}")
        print(f"  train_test_lists: {d['train_test_lists']}")
        print(f"mask file per scene: {json.dumps({r['scene']: r['mask'] for r in ok})}")
        print(f"points per scene (labelled fraction, classes, files MB): "
              f"{json.dumps({r['scene']: [r['points']['labelled'], r['points']['n_classes'], r['points']['files_mb']] for r in ok})}")
    print(f"\nfull results: {args.out / 'trial.json'}")
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
