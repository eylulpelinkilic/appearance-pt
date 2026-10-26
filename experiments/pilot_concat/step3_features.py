"""Pilot step 3: training features of the 200 train scenes, for the linear probes of step 4.

Per scene (loaded in background threads): up to --points labelled, non-anonymised vertices, drawn uniformly with the
scene's seed (encoders.scene_seed); their features from
  - utonia, sonata, concerto, random: one fp16 SDPA pass of each frozen encoder over the whole scene (the same seed),
    upcast gathered at the drawn vertices only (stages.py); a scene that runs out of memory is split into tiles;
  - dino: DINOv2 ViT-L/14 (with registers) patch features lifted from --views DSLR views (lifting.py, occlusion
    variant --occlusion), zeros where no view sees the vertex;
  - raw: xyz, colour, normal (config X);
with the Pointcept label (segment[:, 0], the major label) and the number of views that see the vertex. Written as
memory-mapped arrays to --features on the session's temp disk (probes.FeatureWriter; ~27 GB for 2M rows), which
step 4 reads in the same session. Colours are as Pointcept stores them (anonymised regions magenta), as in the
encoders' pretraining and Pointcept's probing configs. Summary: <out>/step3_summary.json.

Usage: python experiments/pilot_concat/step3_features.py [--input /kaggle/input] [--features /tmp/pilot_features]
       [--points 10000] [--views 30] [--occlusion v2] [--concerto concerto] [--limit N]
Needs PYTHONPATH=src:third_party/utonia:third_party/sonata:third_party/concerto, a GPU and Internet (weights).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from appearance_pt import REPO_ROOT
from appearance_pt.envinfo import collect
from appearance_pt.runlog import section
from appearance_pt.scannetpp_store import find_scenes
from encoders import ENCODERS, attention_backend, scene_seed
from lifting import DINO, OCCLUSION, lift
from pipeline import ENCODER_BLOCKS, Models, add_dino, encode_blocks, prefetch, scene_views
from probes import FeatureWriter
from scenes import raw_features
from step2_lift import machine


def sample_points(s: dict, n: int, seed: int) -> np.ndarray:
    keep = s["segment"][:, 0] >= 0
    keep[s["anonymised"]] = False
    cand = np.flatnonzero(keep)
    return np.sort(np.random.default_rng(seed).choice(cand, min(n, len(cand)), replace=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--features", type=Path, default=Path("/tmp/pilot_features"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/pilot_concat/step3"))
    ap.add_argument("--points", type=int, default=10_000, help="labelled vertices per scene")
    ap.add_argument("--views", type=int, default=30)
    ap.add_argument("--occlusion", choices=sorted(OCCLUSION), default="v2")
    ap.add_argument("--concerto", choices=[k for k in ENCODERS if k.startswith("concerto")], default="concerto",
                    help="Concerto weights for the concerto block (decisions.md P4)")
    ap.add_argument("--limit", type=int, help="only the first N train scenes (a quick test)")
    ap.add_argument("--workers", type=int, default=3, help="scene-loading threads")
    ap.add_argument("--device", default="cuda", help="cpu only for the local tests")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("!! no GPU visible: set Accelerator to GPU T4 x2")
        return 2
    record = collect()
    (args.out / "env_record.json").write_text(json.dumps(record, indent=2))
    box = machine()
    print(f"git {json.dumps(record['git'])}\nmachine {json.dumps(box)}\nargs {vars(args)}")

    section("1. Train scenes (core datasets below --input)")
    archives = {s: p[0] for s, p in find_scenes(args.input)["core"].items()}
    subset = json.loads((REPO_ROOT / "configs" / "scannetpp_subset.json").read_text())
    scenes = [s for s in subset["train"] if s in archives][:args.limit]
    missing = [s for s in subset["train"] if s not in archives]
    print(f"  {len(scenes)} train scenes to process; {len(missing)} of the subset not found{': ' + str(missing[:5]) if missing else ''}")
    if not scenes:
        print("!! no train scenes found: attach scannetpp-core-train-a and scannetpp-core-train-b")
        return 2

    section(f"2. Models (concerto block: {ENCODERS[args.concerto][1]}; DINO {DINO[0]}; occlusion {args.occlusion})")
    models = Models(concerto=args.concerto, device=dev)
    widths = {**models.widths, "dino": models.dino_width, "raw": 9}
    dtypes = {**{b: np.float16 for b in ENCODER_BLOCKS}, "dino": np.float16, "raw": np.float32}
    print(f"  widths {widths}; {sum(widths.values()) * 2 * args.points * len(scenes) / 1e9:.1f} GB at most")
    writer = FeatureWriter(args.features, args.points * len(scenes), widths, dtypes)

    section("3. Scenes")
    occ, rows, failed, t_start = OCCLUSION[args.occlusion], [], [], time.time()
    with attention_backend("sdpa"):
        for i, (scene, s, waited) in enumerate(prefetch([(sc, archives[sc]) for sc in scenes], args.views, args.workers), 1):
            try:
                if isinstance(s, Exception):
                    raise s
                t0 = time.perf_counter()
                seed = scene_seed(scene)
                idx = sample_points(s, args.points, seed)
                idx_t = torch.from_numpy(idx)
                stages, info = encode_blocks(models, s, seed, store="cpu")  # the GPU stays free for each pass
                blocks = {}
                for b in ENCODER_BLOCKS:
                    blocks[b] = stages[b].gather(idx_t, device=dev).cpu().numpy()
                del stages
                torch.cuda.empty_cache()
                t1 = time.perf_counter()
                points = torch.from_numpy(s["coord"]).to(dev)
                normals = torch.from_numpy(s["normal"]).float().to(dev)
                views = scene_views(s, points, occ, device=dev)
                add_dino(models.dino, s, views, device=dev)
                (_, dino, count), = lift(points[idx_t.to(dev)], views, normals[idx_t.to(dev)], chunk=len(idx) + 1)
                blocks["dino"] = dino.half().cpu().numpy()
                blocks["raw"] = raw_features(s)[idx]
                count = count.cpu().numpy()
                del views, points, normals
                writer.add(scene, blocks, s["segment"][idx, 0].astype(np.int16), count)
                t2 = time.perf_counter()
                row = dict(scene=scene, n_vertices=len(s["coord"]), n_points=len(idx), seen=round(float((count > 0).mean()), 4),
                           tiles={b: t for b, (t, _) in info.items()}, sec_encoders={b: sec for b, (_, sec) in info.items()},
                           sec_dino_lift=round(t2 - t1, 2), sec_scene=round(t2 - t0, 2), sec_wait=waited)
                rows.append(row)
                eta = (time.time() - t_start) / i * (len(scenes) - i) / 60
                print(f"  [{i}/{len(scenes)}] {scene}: {row['n_vertices']:,} vertices, {row['n_points']} points, seen "
                      f"{row['seen']:.3f}, tiles {row['tiles']}, encoders {row['sec_encoders']} s, DINO+lift "
                      f"{row['sec_dino_lift']} s, waited {waited} s; ETA {eta:.0f} min", flush=True)
            except Exception as e:  # noqa: BLE001  (one scene's failure must not end a long run)
                failed.append(dict(scene=scene, error=f"{type(e).__name__}: {e}"[:300]))
                print(f"  [{i}/{len(scenes)}] {scene}: FAILED {failed[-1]['error']}\n{traceback.format_exc(limit=3)}", flush=True)
                torch.cuda.empty_cache()

    meta = writer.close(occlusion=args.occlusion, concerto=ENCODERS[args.concerto], dino=DINO, points_per_scene=args.points,
                        views=args.views, encoder_names=models.names, git=record["git"])
    size_gb = sum(f.stat().st_size for f in args.features.glob("*.npy")) / 1e9
    summary = dict(args={k: str(v) for k, v in vars(args).items()}, git=record["git"], machine=box, rows=meta["n"],
                   scenes_done=len(rows), failed=failed, files_gb=round(size_gb, 1), per_scene=rows,
                   minutes=round((time.time() - t_start) / 60, 1))
    (args.out / "step3_summary.json").write_text(json.dumps(summary, indent=1, default=str))

    section("SUMMARY step 3 (paste this block back)")
    print(f"scenes {len(rows)}/{len(scenes)} done, {len(failed)} failed{': ' + json.dumps(failed) if failed else ''}")
    print(f"rows {meta['n']:,} ({np.mean([r['n_points'] for r in rows]):.0f} per scene); files {size_gb:.1f} GB in {args.features} "
          f"(max {meta['n'] * sum(widths[b] * np.dtype(dtypes[b]).itemsize for b in widths) / 1e9:.1f} GB used)")
    print(f"seen by >= 1 view: {np.mean([r['seen'] for r in rows]):.3f} of sampled points (min {min(r['seen'] for r in rows):.3f})")
    tiled = {r["scene"]: r["tiles"] for r in rows if max(r["tiles"].values()) > 1}
    print(f"scenes encoded in tiles (out of memory in one pass): {len(tiled)} {json.dumps(tiled)}")
    for b in ENCODER_BLOCKS:
        print(f"  {b:9s} seconds per scene: median {np.median([r['sec_encoders'][b] for r in rows]):.1f}, "
              f"max {max(r['sec_encoders'][b] for r in rows):.1f}")
    print(f"DINO + lift per scene: median {np.median([r['sec_dino_lift'] for r in rows]):.1f} s; waiting for loads: "
          f"total {sum(r['sec_wait'] for r in rows) / 60:.1f} min; total {summary['minutes']} min")
    return 0 if rows else 1  # failed scenes are listed above; step 4 trains on the rest


if __name__ == "__main__":
    sys.exit(main())
