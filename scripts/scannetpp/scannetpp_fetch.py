"""Download one part of the ScanNet++ subset and keep only what the project needs.

Parts of configs/scannetpp_subset.json: "val" (50 scenes), "train-a" (first 100 train scenes), "train-b" (last
100). Per scene: download mesh + labels and undistorted DSLR data to temporary storage (parallel requests), run
Pointcept's ScanNet++ preprocessing, pack the kept files into <out>/scenes/<scene>.tar
(appearance_pt.scannetpp_reduce), delete the raw files. <out>/manifest.json records every scene; scenes whose
tar exists are skipped, so a rerun resumes. Stops before the output exceeds --max-out-gb (Kaggle keeps about
20 GB of notebook output).

Usage: python scripts/scannetpp/scannetpp_fetch.py --part val [--out DIR] [--tmp DIR] [--workers 4]
Needs the Kaggle secret SCANNETPP_TOKEN, the token-free download config (input dataset) and open3d.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from appearance_pt import REPO_ROOT
from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.scannetpp_download import (CORE_ASSETS, SPLITS, ScannetppClient, find_config, kaggle_secret,
                                              load_config, scene_files)
from appearance_pt.scannetpp_reduce import label_setup, load_preprocess, pack_scene

PARTS = {"val": ("val", slice(None)), "train-a": ("train", slice(0, 100)), "train-b": ("train", slice(100, None))}


def dir_gb(path: Path) -> float:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 1e9 if Path(path).exists() else 0.0


def process(client, cfg, pp, labels, scene: str, split: str, tmp: Path, out: Path, workers: int) -> dict:
    t0 = time.perf_counter()
    try:
        files = scene_files(cfg, scene, SPLITS[split], CORE_ASSETS)
        with ThreadPoolExecutor(workers) as pool:
            fetched = list(pool.map(lambda f: client.fetch(f, tmp), files))
        t1 = time.perf_counter()
        pp.parse_scene(scene, split, tmp, tmp / "processed", labels[0], labels[1], -1)
        processed = tmp / "processed" / split / scene
        archive = out / "scenes" / f"{scene}.tar"
        archive.parent.mkdir(parents=True, exist_ok=True)
        kept = pack_scene(tmp / "data" / scene, processed, split, archive)
        return dict(scene=scene, split=split, status="ok",
                    download_mb=round(sum(f["download_bytes"] for f in fetched) / 1e6, 1),
                    seconds_download=round(t1 - t0, 1), seconds=round(time.perf_counter() - t0, 1),
                    n_vertices=int(len(np.load(processed / "segment.npy", mmap_mode="r"))),
                    n_frames=kept["n_frames"], n_bad=kept["n_bad"], n_views=len(kept["views"]),
                    tar_mb=round(archive.stat().st_size / 1e6, 1))
    finally:
        shutil.rmtree(tmp / "data" / scene, ignore_errors=True)
        shutil.rmtree(tmp / "processed", ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=sorted(PARTS))
    ap.add_argument("--config", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/scannetpp_core"))
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/scannetpp"), help="raw downloads (deleted per scene)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests per scene")
    ap.add_argument("--max-out-gb", type=float, default=19.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "env_record.json").write_text(json.dumps(collect(), indent=2))

    subset = json.loads((REPO_ROOT / "configs" / "scannetpp_subset.json").read_text())
    split, sl = PARTS[args.part]
    scenes = subset[split][sl]
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    print(f"part {args.part}: {len(scenes)} {split} scenes; free disk at {args.tmp.parent}: "
          f"{shutil.disk_usage(args.tmp.parent).free / 1e9:.0f} GB")

    cfg = load_config(find_config(args.config))
    client = ScannetppClient(cfg, kaggle_secret("SCANNETPP_TOKEN"))
    section("1. Metadata files (labels for preprocessing)")
    for path in cfg["meta_files"]:
        client.download(path, args.tmp / path)
    pp = load_preprocess()
    labels = label_setup(pp, args.tmp)
    (args.out / "meta").mkdir(exist_ok=True)
    shutil.copytree(args.tmp / "metadata", args.out / "meta" / "metadata", dirs_exist_ok=True)
    shutil.copytree(args.tmp / "splits", args.out / "meta" / "splits", dirs_exist_ok=True)

    section(f"2. Scenes ({args.part})")
    t0 = time.time()
    for i, scene in enumerate(scenes, 1):
        if (args.out / "scenes" / f"{scene}.tar").exists() and manifest.get(scene, {}).get("status") == "ok":
            continue
        if dir_gb(args.out) > args.max_out_gb - 0.5:
            manifest[scene] = dict(scene=scene, split=split, status="not started: output limit reached")
            continue
        for attempt in (1, 2):
            try:
                manifest[scene] = process(client, cfg, pp, labels, scene, split, args.tmp, args.out, args.workers)
                break
            except Exception as e:  # noqa: BLE001  (record, retry once, then go on with the next scene)
                manifest[scene] = dict(scene=scene, split=split, status=f"failed (attempt {attempt}): {type(e).__name__}: {e}"[:300])
        manifest_path.write_text(json.dumps(manifest, indent=1))
        done = sum(m.get("status") == "ok" for m in manifest.values())
        print(f"  [{i}/{len(scenes)}] {json.dumps(manifest[scene])}  | ok {done}, output {dir_gb(args.out):.1f} GB, "
              f"{(time.time() - t0) / 60:.0f} min", flush=True)

    section(f"SUMMARY {args.part} (paste this block back)")
    rows = [manifest.get(s, dict(scene=s, status="missing")) for s in scenes]
    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"scenes ok {len(ok)}/{len(scenes)}; downloaded {sum(r['download_mb'] for r in ok) / 1e3:.1f} GB; "
          f"kept {sum(r['tar_mb'] for r in ok) / 1e3:.2f} GB in {args.out / 'scenes'}; output dir {dir_gb(args.out):.2f} GB; "
          f"{(time.time() - t0) / 3600:.2f} h this run")
    if ok:
        print(f"vertices per scene: min {min(r['n_vertices'] for r in ok)}, median {int(np.median([r['n_vertices'] for r in ok]))}, "
              f"max {max(r['n_vertices'] for r in ok)}; DSLR views kept: min {min(r['n_views'] for r in ok)}, "
              f"bad frames total {sum(r['n_bad'] for r in ok)}")
    for r in rows:
        if r.get("status") != "ok":
            print(f"NOT OK {r['scene']}: {r.get('status')}")
    return 0 if len(ok) == len(scenes) else 1


if __name__ == "__main__":
    sys.exit(main())
