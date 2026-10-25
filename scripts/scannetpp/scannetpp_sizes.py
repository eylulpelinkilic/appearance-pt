"""Sizes of the ScanNet++ assets the project may need, for the fixed scene subset, without downloading them.

Downloads only the metadata files (split and class lists), picks the subset (all nvs_sem_val scenes
plus a fixed random sample of nvs_sem_train), then asks the server for the size of every candidate
asset of every subset scene (HEAD requests). Writes subset.json, sizes.csv and a summary to --out.

Usage: python scripts/scannetpp/scannetpp_sizes.py [--config YML_OR_DIR] [--out DIR] [--n-train 200] [--seed 0]
Needs the Kaggle secret SCANNETPP_TOKEN and a token-free download_scannetpp.yml (private dataset input).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.scannetpp_download import PLACEHOLDER_TOKEN, ScannetppClient, choose_subset, load_config, scene_files

GB = 1e9
SPLITS = {"train": "nvs_sem_train", "val": "nvs_sem_val"}
# candidate assets (names from download_scannetpp.yml), grouped by what they are for
ASSET_GROUPS = {
    "mesh+labels": ("scan_mesh_path", "scan_mesh_mask_path", "scan_mesh_segs_path", "scan_anno_json_path"),
    "semantic_mesh_ply": ("scan_sem_mesh_path",),
    "dslr_undistorted": ("dslr_resized_undistorted_dir", "dslr_resized_undistorted_mask_dir",
                         "dslr_nerfstudio_transform_undistorted_path", "dslr_train_test_lists_path"),
    "dslr_colmap": ("dslr_colmap_dir",),
    "iphone": ("iphone_video_path", "iphone_video_mask_path", "iphone_depth_path",
               "iphone_pose_intrinsic_imu_path", "iphone_colmap_dir"),
}
GROUP_OF = {asset: group for group, assets in ASSET_GROUPS.items() for asset in assets}


def find_config(path: Path) -> Path:
    if path.is_file():
        return path
    found = sorted(path.glob("**/download_scannetpp*.yml"))
    if not found:
        sys.exit(f"!! no download_scannetpp*.yml under {path}: attach the private dataset with the token-free config")
    return found[0]


def kaggle_secret(name: str) -> str:
    from kaggle_secrets import UserSecretsClient

    return UserSecretsClient().get_secret(name)


def summarize(rows: list[dict]) -> list[dict]:
    """Per (group, split): files, missing, unknown sizes, total GB, per-scene mean and max GB."""
    per_scene = defaultdict(lambda: defaultdict(int))
    stats = defaultdict(lambda: dict(files=0, missing=0, unknown=0))
    for r in rows:
        key = (GROUP_OF[r["asset"]], r["split"])
        stats[key]["files"] += 1
        if r["bytes"] is None:
            stats[key]["missing"] += 1
        elif r["bytes"] < 0:
            stats[key]["unknown"] += 1
        else:
            per_scene[key][r["scene"]] += r["bytes"]
    out = []
    for (group, split), s in sorted(stats.items()):
        sizes = list(per_scene[(group, split)].values()) or [0]
        out.append(dict(group=group, split=split, **s, total_gb=round(sum(sizes) / GB, 2),
                        mean_scene_gb=round(sum(sizes) / len(sizes) / GB, 3), max_scene_gb=round(max(sizes) / GB, 3)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("/kaggle/input"), help="yml file, or a dir to search")
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/scannetpp_sizes"))
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pause", type=float, default=0.1, help="seconds between requests")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "env_record.json").write_text(json.dumps(collect(), indent=2))

    config_path = find_config(args.config)
    cfg = load_config(config_path)
    print(f"config: {config_path}\nURL template: {cfg['root_url'].split('?')[0]}?...")
    if cfg.get("token", PLACEHOLDER_TOKEN) != PLACEHOLDER_TOKEN:
        print("!! the config file contains a token; it is ignored (the token comes from the Kaggle secret)")
    client = ScannetppClient(cfg, kaggle_secret("SCANNETPP_TOKEN"), pause=args.pause)

    section("1. Metadata files (split and class lists)")
    for path in cfg["meta_files"]:
        n = client.download(path, args.out / "meta" / path)
        print(f"  {path} ({n} bytes)")
    splits = {k: (args.out / "meta" / "splits" / f"{v}.txt").read_text().split() for k, v in SPLITS.items()}
    print("  scenes per split: " + ", ".join(f"{SPLITS[k]} {len(v)}" for k, v in splits.items()))
    subset = choose_subset(splits["train"], splits["val"], args.n_train, args.seed)
    (args.out / "subset.json").write_text(json.dumps(subset, indent=1) + "\n")
    print(f"  subset: {len(subset['val'])} val + {len(subset['train'])} train (seed {args.seed}) -> {args.out / 'subset.json'}")

    section("2. Asset sizes (HEAD requests, nothing downloaded)")
    assets = list(GROUP_OF)
    scenes = [(s, "val") for s in subset["val"]] + [(s, "train") for s in subset["train"]]
    rows, t0 = [], time.time()
    for i, (scene, split) in enumerate(scenes, 1):
        for f in scene_files(cfg, scene, SPLITS[split], assets):
            rows.append(dict(scene=scene, split=split, asset=f["asset"], remote=f["remote"], bytes=client.size(f["remote"])))
        if i % 25 == 0 or i == len(scenes):
            print(f"  {i}/{len(scenes)} scenes, {(time.time() - t0) / 60:.1f} min")
    with open(args.out / "sizes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))

    section("SUMMARY (paste this block back)")
    print(f"scenes per split: {json.dumps({SPLITS[k]: len(v) for k, v in splits.items()})}")
    print(f"{'group':18s} {'split':5s} {'files':>5s} {'missing':>7s} {'unknown':>7s} {'total_GB':>9s} {'mean_GB':>8s} {'max_GB':>7s}")
    for s in summary:
        print(f"{s['group']:18s} {s['split']:5s} {s['files']:>5d} {s['missing']:>7d} {s['unknown']:>7d} "
              f"{s['total_gb']:>9.2f} {s['mean_scene_gb']:>8.3f} {s['max_scene_gb']:>7.3f}")
    print(f"\nSUBSET (paste too; it gets committed as configs/scannetpp_subset.json)\n{json.dumps(subset)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
