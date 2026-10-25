"""Check every packed ScanNet++ scene in the attached private datasets, while a damaged scene can still be downloaded
again (token valid until 2026-11-01).

Finds .../scenes/<scene>.tar below /kaggle/input (core: m2_02 outputs, iPhone: m2_04 outputs) and, for each scene of
configs/scannetpp_subset.json, reads and decodes every file (appearance_pt.scannetpp_store.check_core / check_iphone):
Pointcept arrays, anonymisation mask, val raw mesh and annotations, every DSLR view and mask, every iPhone frame, mask
and depth map, the poses and the COLMAP image names. Archive sizes and counts must match the download manifests.
Writes <out>/verify.json (per-scene numbers, no data).

Usage: python scripts/scannetpp/scannetpp_verify.py [--input /kaggle/input] [--out DIR] [--workers N]
No ScanNet++ token needed; nothing is downloaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from appearance_pt import REPO_ROOT
from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.scannetpp_store import SceneArchive, check_core, check_iphone, find_scenes

CHECKS = {"core": check_core, "iphone": check_iphone}
MANIFEST_KEYS = {"core": ("n_vertices", "n_views"), "iphone": ("n_pool",)}  # must equal the download manifest's row
GRID_LIMIT = 1.4e6  # single-pass encoder limit on a T4 (docs/decisions.md P3)


def manifest_row(archive: Path, cache: dict) -> dict:
    """The download manifest's row for this archive (manifest.json sits next to the scenes/ directory)."""
    path = archive.parent.parent / "manifest.json"
    if path not in cache:
        cache[path] = json.loads(path.read_text()) if path.exists() else {}
    return cache[path].get(archive.name.removesuffix(".tar"), {})


def check(kind: str, scene: str, split: str, path: Path, expected: dict) -> dict:
    t0 = time.perf_counter()
    row = dict(kind=kind, scene=scene, split=split, path=str(path))
    try:
        with SceneArchive(path) as arc:
            row.update(CHECKS[kind](arc, scene, split), mb_read=round(arc.bytes_read / 1e6, 1))
        problems = [f"{k} {row[k]} here, {expected[k]} in the manifest" for k in MANIFEST_KEYS[kind]
                    if k in expected and row[k] != expected[k]]
        if path.is_file() and "tar_mb" in expected and round(path.stat().st_size / 1e6, 1) != expected["tar_mb"]:
            problems.append(f"tar {path.stat().st_size / 1e6:.1f} MB here, {expected['tar_mb']} MB in the manifest")
        row["status"] = "FAIL: " + "; ".join(problems) if problems else "ok"
    except Exception as e:  # noqa: BLE001  (record and go on: every scene gets checked)
        row["status"] = f"FAIL: {type(e).__name__}: {e}"[:300]
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def layout(root: Path, depth: int = 2) -> None:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        level = len(Path(dirpath).relative_to(root).parts)
        print(f"  {'  ' * level}{Path(dirpath).name}/  ({len(dirnames)} dirs, {len(filenames)} files)")
        if level >= depth:
            dirnames[:] = []
        dirnames.sort()


def spread(values) -> str:
    return f"min {min(values):,}, median {int(np.median(values)):,}, max {max(values):,}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/scannetpp_verify"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "env_record.json").write_text(json.dumps(collect(), indent=2))
    subset = json.loads((REPO_ROOT / "configs" / "scannetpp_subset.json").read_text())
    splits = {scene: split for split in ("val", "train") for scene in subset[split]}

    section(f"Attached data ({args.input})")
    layout(args.input)
    found = find_scenes(args.input)
    for (kind, where), count in sorted(Counter((k, str(p.parent)) for k, scenes in found.items()
                                               for paths in scenes.values() for p in paths).items()):
        print(f"  {kind:6s} {count:4d} archives in {where}")
    tasks, cache = [], {}
    for kind, scenes in found.items():
        extra = sorted(set(scenes) - set(splits))
        if extra:
            print(f"  {kind}: {len(extra)} archives of scenes outside the subset (not checked): {extra[:5]}")
        for scene, split in splits.items():
            paths = scenes.get(scene, [])
            if len(paths) > 1:
                print(f"  {kind} {scene}: {len(paths)} copies, checking each: {[str(p) for p in paths]}")
            tasks += [(kind, scene, split, p, manifest_row(p, cache)) for p in paths]

    section(f"Reading and decoding every file ({len(tasks)} archives, {args.workers} processes)")
    t0, rows = time.time(), []
    with ProcessPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(check, *task) for task in tasks]):
            rows.append(future.result())
            if rows[-1]["status"] != "ok":
                print(f"  {rows[-1]['kind']} {rows[-1]['scene']}: {rows[-1]['status']}", flush=True)
            if len(rows) % 25 == 0 or len(rows) == len(tasks):
                print(f"  [{len(rows)}/{len(tasks)}] {sum(r.get('mb_read', 0) for r in rows) / 1e3:.1f} GB read | "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    rows.sort(key=lambda r: (r["kind"], r["split"], r["scene"]))
    (args.out / "verify.json").write_text(json.dumps(rows, indent=1))

    section("SUMMARY verify (paste this block back)")
    not_ok = []
    for kind in ("core", "iphone"):
        good = {r["scene"]: r for r in rows if r["kind"] == kind and r["status"] == "ok"}
        missing = [s for s in splits if s not in found[kind]]
        failed = sorted({r["scene"] for r in rows if r["kind"] == kind and r["status"] != "ok"} - set(good))
        print(f"{kind}: ok {len(good)}/{len(splits)} (val {sum(splits[s] == 'val' for s in good)}/{len(subset['val'])}, "
              f"train {sum(splits[s] == 'train' for s in good)}/{len(subset['train'])}); missing {len(missing)}; "
              f"failed {len(failed)}")
        not_ok += [f"{kind} {s} ({splits[s]}): missing" for s in missing]
        not_ok += [f"{kind} {r['scene']} ({r['split']}): {r['status']}" for r in rows
                   if r["kind"] == kind and r["scene"] in failed]
        g = list(good.values())
        if g and kind == "core":
            grid = [r["n_grid_2cm"] for r in g]
            print(f"  vertices {spread([r['n_vertices'] for r in g])}; 2 cm grid points {spread(grid)}; "
                  f"{sum(x > GRID_LIMIT for x in grid)} scenes above {GRID_LIMIT / 1e6:.1f}M")
            print(f"  labelled fraction min {min(r['labelled'] for r in g):.2f}, median "
                  f"{np.median([r['labelled'] for r in g]):.2f}; anonymised vertices {sum(r['n_anonymised'] for r in g):,} "
                  f"in {sum(r['n_anonymised'] > 0 for r in g)} scenes ({sum(r['n_anonymised_magenta'] for r in g):,} "
                  f"magenta); magenta vertices in total {sum(r['n_magenta'] for r in g):,}")
            print(f"  DSLR views {sum(r['n_views'] for r in g):,} (min {min(r['n_views'] for r in g)} per scene), "
                  f"masks {sum(r['n_masks'] for r in g):,}; image sizes (h x w) {dict(Counter(s for r in g for s in r['image_hw']))}; "
                  f"transforms h x w {dict(Counter(r['transforms_hw'] for r in g))}; camera {dict(Counter(r['camera_model'] for r in g))}")
        if g and kind == "iphone":
            print(f"  frames {sum(r['n_pool'] for r in g):,} (min {min(r['n_pool'] for r in g)} per scene); rgb sizes (h x w) "
                  f"{dict(Counter(s for r in g for s in r['rgb_hw']))}; COLMAP cameras "
                  f"{dict(Counter(m for r in g for m in r['camera_models']))}")
            print(f"  depth encodings {dict(Counter(r['depth_encoding'] for r in g))}; per-scene median depth "
                  f"{spread([r['depth_median_mm'] for r in g])} mm; worst frame's zero-depth fraction "
                  f"{max(r['depth_zero_max'] for r in g):.3f}; mask > 127 fraction "
                  f"{min(r['mask_on'] for r in g):.3f}–{max(r['mask_on'] for r in g):.3f}")
    print(f"read {sum(r.get('mb_read', 0) for r in rows) / 1e3:.1f} GB in {(time.time() - t0) / 60:.1f} min; "
          f"per-scene numbers: {args.out / 'verify.json'}")
    for line in not_ok:
        print(f"NOT OK {line}")
    print("ALL SCENES OK" if not not_ok else f"{len(not_ok)} PROBLEMS: these scenes need a fresh download before 2026-11-01")
    return 0 if not not_ok else 1


if __name__ == "__main__":
    sys.exit(main())
