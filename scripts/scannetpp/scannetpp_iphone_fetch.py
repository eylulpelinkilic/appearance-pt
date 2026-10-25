"""Download one part of the subset's ScanNet++ iPhone data and keep a 60-frame pool per scene.

Parts of configs/scannetpp_subset.json: "val" (50 scenes), "train-1" ... "train-4" (50 train scenes each), so
five sessions can run side by side (decoding the full 60 fps videos is the slow step: ~1-3 min per scene).
Per scene: download rgb.mkv, rgb_mask.mkv, depth.bin, pose_intrinsic_imu.json and colmap/ (the next scene
downloads while this one decodes), extract the frame pool (appearance_pt.scannetpp_iphone.reduce_scene) into
<out>/scenes/<scene>.tar, delete the raw files. <out>/manifest.json records every scene; scenes whose tar
exists are skipped on a rerun.

Usage: python scripts/scannetpp/scannetpp_iphone_fetch.py --part val [--out DIR] [--tmp DIR]
Needs the Kaggle secret SCANNETPP_TOKEN, the token-free download config, lz4, imageio and ffmpeg (system
ffmpeg, else the imageio-ffmpeg binary).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from appearance_pt import REPO_ROOT
from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.scannetpp_download import (ASSET_GROUPS, SPLITS, ScannetppClient, find_config, kaggle_secret,
                                              load_config, scene_files)
from appearance_pt.scannetpp_iphone import ffmpeg_exe, reduce_scene, toolbox_colmap

PARTS = {"val": ("val", slice(None)), **{f"train-{k}": ("train", slice(50 * (k - 1), 50 * k)) for k in (1, 2, 3, 4)}}


def download(client, cfg, scene: str, split: str, tmp: Path, workers: int = 4) -> dict:
    t0 = time.perf_counter()
    files = scene_files(cfg, scene, SPLITS[split], list(ASSET_GROUPS["iphone"]))
    with ThreadPoolExecutor(workers) as pool:
        fetched = list(pool.map(lambda f: client.fetch(f, tmp), files))
    return dict(download_mb=round(sum(f["download_bytes"] for f in fetched) / 1e6, 1),
                seconds_download=round(time.perf_counter() - t0, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=sorted(PARTS))
    ap.add_argument("--config", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/scannetpp_iphone"))
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/scannetpp_iphone"), help="raw downloads (deleted per scene)")
    args = ap.parse_args()
    (args.out / "scenes").mkdir(parents=True, exist_ok=True)
    (args.out / "env_record.json").write_text(json.dumps(collect(), indent=2))

    subset = json.loads((REPO_ROOT / "configs" / "scannetpp_subset.json").read_text())
    split, sl = PARTS[args.part]
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    todo = [s for s in subset[split][sl]
            if not ((args.out / "scenes" / f"{s}.tar").exists() and manifest.get(s, {}).get("status") == "ok")]
    ffmpeg = ffmpeg_exe()
    print(f"part {args.part}: {len(subset[split][sl])} {split} scenes, {len(todo)} to do; ffmpeg {ffmpeg}; "
          f"free disk at {args.tmp.parent}: {shutil.disk_usage(args.tmp.parent).free / 1e9:.0f} GB")
    cfg = load_config(find_config(args.config))
    client = ScannetppClient(cfg, kaggle_secret("SCANNETPP_TOKEN"))
    colmap = toolbox_colmap()

    section(f"iPhone scenes ({args.part}): download (one ahead), extract pool, pack, delete")
    t0 = time.time()
    with ThreadPoolExecutor(1) as prefetch:
        ahead = prefetch.submit(download, client, cfg, todo[0], split, args.tmp) if todo else None
        for i, scene in enumerate(todo):
            current, ahead = ahead, (prefetch.submit(download, client, cfg, todo[i + 1], split, args.tmp)
                                     if i + 1 < len(todo) else None)
            row = dict(scene=scene, split=split)
            for attempt in (1, 2):
                try:
                    row.update(current.result() if attempt == 1 else download(client, cfg, scene, split, args.tmp))
                    t1 = time.perf_counter()
                    archive = args.out / "scenes" / f"{scene}.tar"
                    info = reduce_scene(colmap, ffmpeg, args.tmp / "data" / scene / "iphone", archive)
                    row.update(status="ok", seconds_reduce=round(time.perf_counter() - t1, 1),
                               tar_mb=round(archive.stat().st_size / 1e6, 1),
                               **{k: v for k, v in info.items() if k != "pool"}, n_pool=len(info["pool"]))
                    break
                except Exception as e:  # noqa: BLE001  (record, retry once with a fresh download, then go on)
                    row["status"] = f"failed (attempt {attempt}): {type(e).__name__}: {e}"[:300]
                    shutil.rmtree(args.tmp / "data" / scene, ignore_errors=True)
            shutil.rmtree(args.tmp / "data" / scene, ignore_errors=True)
            manifest[scene] = row
            manifest_path.write_text(json.dumps(manifest, indent=1))
            print(f"  [{i + 1}/{len(todo)}] {json.dumps(row)} | {(time.time() - t0) / 60:.0f} min", flush=True)

    section(f"SUMMARY iPhone {args.part} (paste this block back)")
    rows = [manifest.get(s, dict(scene=s, status="missing")) for s in subset[split][sl]]
    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"scenes ok {len(ok)}/{len(rows)}; downloaded {sum(r['download_mb'] for r in ok) / 1e3:.1f} GB; "
          f"kept {sum(r['tar_mb'] for r in ok) / 1e3:.2f} GB; {(time.time() - t0) / 3600:.2f} h this run")
    if ok:
        print(f"pool frames: min {min(r['n_pool'] for r in ok)}; depth encodings {sorted({r['depth_encoding'] for r in ok})}; "
              f"mean s/scene: download {sum(r['seconds_download'] for r in ok) / len(ok):.0f}, "
              f"reduce {sum(r['seconds_reduce'] for r in ok) / len(ok):.0f}")
    for r in rows:
        if r.get("status") != "ok":
            print(f"NOT OK {r['scene']}: {r.get('status')}")
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
