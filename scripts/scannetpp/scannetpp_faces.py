"""Before the ScanNet++ token expires (2026-11-01): the train meshes' triangles, and the sizes of the assets the next
phase may want.

1. Faces. For every train scene of the subset: download mesh_aligned_0.05.ply (zipped), read it with open3d as
   Pointcept's preprocessing does, and keep <out>/faces/<scene>.npz with the triangles (F, 3) int32, the vertex count
   and the SHA-1 of the vertex coordinates as float32, i.e. of Pointcept's coord.npy. If the core train datasets are
   attached, that hash is checked against coord.npy, which proves the triangles index the same vertices in the same
   order. The raw mesh is deleted. Val scenes need nothing: their raw mesh is in scannetpp-core-val. With the
   triangles, visibility from a DSLR camera can be computed exactly (rays against the mesh) instead of with the
   point z-buffer (docs/decisions.md P6 results), and depth can be rendered for every view.
2. Sizes (HEAD requests; nothing downloaded) of the dense laser scans (pc_aligned.ply, its anonymisation mask, scanner
   poses), the full-resolution DSLR images and masks, and the panorama (panocam) assets, for the 250 subset scenes,
   with a per-scene mean to extrapolate to the whole dataset.
<out>/manifest.json records every scene, so a rerun resumes; the run stops before the output exceeds --max-out-gb.

Usage: python scripts/scannetpp/scannetpp_faces.py [--config /kaggle/input] [--core /kaggle/input] [--out DIR]
Needs the Kaggle secret SCANNETPP_TOKEN, the token-free download config (input dataset) and open3d.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from appearance_pt import REPO_ROOT
from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.scannetpp_download import SPLITS, ScannetppClient, find_config, kaggle_secret, load_config, scene_files
from appearance_pt.scannetpp_store import SceneArchive, find_scenes

SIZE_GROUPS = {
    "laser_scan": ("scan_pc_path", "scan_pc_mask_path", "scan_transformed_poses_path"),
    "dslr_original": ("dslr_original_dir", "dslr_original_mask_dir"),
    "panocam": ("pano_rgb_dir", "pano_anon_mask_dir", "pano_depth_dir", "pano_azim_dir", "pano_elev_dir",
                "pano_resized_rgb_dir", "pano_resized_depth_dir", "pano_resized_mask_dir", "pano_resized_azim_dir",
                "pano_resized_elev_dir"),
}
DATASET_SCENES = {"train": 856, "val": 50}  # nvs_sem_train, nvs_sem_val (D9)


def vertex_sha1(coord: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(coord, dtype=np.float32).tobytes()).hexdigest()


def faces_of(client, cfg, scene: str, tmp: Path, out: Path, core: dict) -> dict:
    import open3d as o3d

    t0 = time.perf_counter()
    try:
        (entry,) = scene_files(cfg, scene, SPLITS["train"], ["scan_mesh_path"])
        got = client.fetch(entry, tmp)
        mesh = o3d.io.read_triangle_mesh(str(tmp / entry["local"]))
        coord = np.asarray(mesh.vertices).astype(np.float32)  # as preprocess_scannetpp.parse_scene
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        sha = vertex_sha1(coord)
        matches = None
        if scene in core:
            with SceneArchive(core[scene]) as arc:
                matches = vertex_sha1(arc.npy(f"pointcept/train/{scene}/coord.npy")) == sha
        path = out / "faces" / f"{scene}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, faces=faces, n_vertices=np.int64(len(coord)), vertex_sha1=np.array(sha))
        return dict(scene=scene, status="ok", n_vertices=len(coord), n_faces=len(faces), matches_coord=matches,
                    zip_mb=round(got["download_bytes"] / 1e6, 1), npz_mb=round(path.stat().st_size / 1e6, 1),
                    seconds=round(time.perf_counter() - t0, 1))
    except Exception as e:  # noqa: BLE001  (recorded; a rerun retries the scene)
        return dict(scene=scene, status=f"failed: {client.redact(f'{type(e).__name__}: {e}')}"[:300])
    finally:
        shutil.rmtree(tmp / "data" / scene, ignore_errors=True)


def measure_sizes(client, cfg, subset: dict, workers: int) -> list[dict]:
    jobs = [(scene, split, group, entry) for split in ("train", "val") for scene in subset[split]
            for group, assets in SIZE_GROUPS.items() for entry in scene_files(cfg, scene, SPLITS[split], list(assets))]

    def one(job):
        scene, split, group, entry = job
        try:
            size = client.size(entry["remote"])
        except Exception as e:  # noqa: BLE001
            size = f"error: {client.redact(e)}"[:120]
        return dict(scene=scene, split=split, group=group, asset=entry["asset"], remote_zip=entry["zipped"], bytes=size)

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, jobs))


def size_table(rows: list[dict]) -> dict:
    per_scene = defaultdict(lambda: defaultdict(int))
    counts = defaultdict(lambda: dict(files=0, missing=0, unknown=0))
    for r in rows:
        key = (r["group"], r["split"])
        counts[key]["files"] += 1
        if r["bytes"] is None:
            counts[key]["missing"] += 1
        elif not isinstance(r["bytes"], int) or r["bytes"] < 0:
            counts[key]["unknown"] += 1
        else:
            per_scene[key][r["scene"]] += r["bytes"]
    out = {}
    for key in sorted(counts):
        sizes = list(per_scene[key].values()) or [0]
        group, split = key
        out[f"{group}/{split}"] = dict(**counts[key], scenes_with_data=len(per_scene[key]), total_gb=round(sum(sizes) / 1e9, 2),
                                       mean_gb=round(float(np.mean(sizes)) / 1e9, 3), max_gb=round(max(sizes) / 1e9, 2),
                                       full_split_gb=round(float(np.mean(sizes)) * DATASET_SCENES[split] / 1e9, 0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--core", type=Path, default=Path("/kaggle/input"), help="where the core train datasets are (optional)")
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/scannetpp_faces"))
    ap.add_argument("--tmp", type=Path, default=Path("/tmp/scannetpp_faces"), help="raw meshes (deleted per scene)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-out-gb", type=float, default=19.0)
    ap.add_argument("--skip-sizes", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "env_record.json").write_text(json.dumps(collect(), indent=2))
    subset = json.loads((REPO_ROOT / "configs" / "scannetpp_subset.json").read_text())
    cfg = load_config(find_config(args.config))
    client = ScannetppClient(cfg, kaggle_secret("SCANNETPP_TOKEN"))
    core = {s: p[0] for s, p in find_scenes(args.core)["core"].items()} if args.core.exists() else {}
    print(f"{len(subset['train'])} train scenes; core archives found for {sum(s in core for s in subset['train'])} of them "
          f"(vertex-order check); free disk at {args.tmp.parent}: {shutil.disk_usage(args.tmp.parent).free / 1e9:.0f} GB")

    section("1. Train-scene triangles (mesh_aligned_0.05.ply -> faces/<scene>.npz)")
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    todo = [s for s in subset["train"] if manifest.get(s, {}).get("status") != "ok"]
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {}
        for scene in todo:
            futures[scene] = pool.submit(faces_of, client, cfg, scene, args.tmp, args.out, core)
        for i, scene in enumerate(todo, 1):
            row = futures[scene].result()
            manifest[scene] = row
            manifest_path.write_text(json.dumps(manifest, indent=1))
            out_gb = sum(p.stat().st_size for p in (args.out / "faces").glob("*.npz")) / 1e9 if (args.out / "faces").exists() else 0
            print(f"  [{i}/{len(todo)}] {scene}: {json.dumps(row)}; output {out_gb:.2f} GB", flush=True)
            if out_gb > args.max_out_gb:
                print(f"!! output above {args.max_out_gb} GB: stopping; rerun with a new output to continue")
                for f in futures.values():
                    f.cancel()
                break
    rows = [manifest[s] for s in subset["train"] if s in manifest]
    ok = [r for r in rows if r["status"] == "ok"]

    sizes = {}
    if not args.skip_sizes:
        section("2. Sizes of the laser-scan, full-resolution DSLR and panocam assets (HEAD requests)")
        size_rows = measure_sizes(client, cfg, subset, args.workers)
        with open(args.out / "sizes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(size_rows[0]))
            w.writeheader()
            w.writerows(size_rows)
        sizes = size_table(size_rows)
        (args.out / "sizes.json").write_text(json.dumps(sizes, indent=1))

    section("SUMMARY faces + sizes (paste this block back)")
    print(f"faces: {len(ok)}/{len(subset['train'])} train scenes ok; failed {[r['scene'] for r in rows if r['status'] != 'ok']}; "
          f"{(time.time() - t0) / 60:.1f} min")
    if ok:
        checked = [r for r in ok if r["matches_coord"] is not None]
        print(f"  vertex order = Pointcept coord.npy: {sum(r['matches_coord'] for r in checked)}/{len(checked)} checked scenes")
        print(f"  triangles per scene: min {min(r['n_faces'] for r in ok):,}, median {int(np.median([r['n_faces'] for r in ok])):,}, "
              f"max {max(r['n_faces'] for r in ok):,}; downloaded {sum(r['zip_mb'] for r in ok) / 1e3:.1f} GB, "
              f"kept {sum(r['npz_mb'] for r in ok) / 1e3:.2f} GB in {args.out / 'faces'}")
    for key, v in sizes.items():
        print(f"  {key:22s} files {v['files']}, missing {v['missing']}, unknown {v['unknown']}; subset {v['total_gb']} GB "
              f"(mean {v['mean_gb']} GB, max {v['max_gb']} GB per scene); whole split at the mean ≈ {v['full_split_gb']:.0f} GB")
    print("next: create the PRIVATE dataset scannetpp-train-faces from this output")
    return 0 if len(ok) == len(subset["train"]) else 1


if __name__ == "__main__":
    sys.exit(main())
