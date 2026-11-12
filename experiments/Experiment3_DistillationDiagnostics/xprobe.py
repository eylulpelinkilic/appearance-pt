"""Experiment 3: probes for configurations of one or more feature blocks with the pilot's recipe (Experiment 1's
step4_probes.run_phase), their val evaluation with the head applied on the fly, and the rows, paired-bootstrap
differences and recovered shares of the segmentation table. Runs A and C only; Run B never imports this module."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import probes as probes_mod
from encoders import encode_scene, scene_seed
from metrics import bootstrap_diff, confusion_topk, miou
from probes import NUM_CLASSES, FeatureStore, FeatureWriter, FoldedSet
from step4_probes import run_phase
from xattn import dir_gb, prefetch_points, sample_points
from xdiag import head_features, stack_config


# ---- probes ------------------------------------------------------------------------------------------------------

def train_features(encoders: dict, archives, scenes, n_points, root, dev, workers, log=print) -> dict:
    """encoders: {name: (model, transform, head or None)}; blocks "<name>_f" (and "<name>_h") at the pilot's labelled
    sample of every scene, in the pilot's feature store."""
    writer, rows, t0 = None, [], time.time()
    for i, (scene, s, waited) in enumerate(prefetch_points([(sc, archives[sc]) for sc in scenes], workers), 1):
        if isinstance(s, Exception):
            raise s
        seed = scene_seed(scene)
        idx = torch.from_numpy(sample_points(s["segment"], s["anonymised"], n_points, seed))
        blocks = {}
        for name, (model, transform, head) in encoders.items():
            f = encode_scene(model, transform, s, seed, store="cpu")[0].gather(idx, device=dev)
            blocks[f"{name}_f"] = f.cpu().numpy()
            if head is not None:
                blocks[f"{name}_h"] = head_features(head, f).half().cpu().numpy()
        if writer is None:
            writer = FeatureWriter(root, n_points * len(scenes), {b: x.shape[1] for b, x in blocks.items()},
                                   {b: np.float16 for b in blocks})
        writer.add(scene, blocks, s["segment"][idx.numpy(), 0].astype(np.int16), np.zeros(len(idx), np.int16))
        rows.append(dict(scene=scene, waited=waited))
        if i % 20 == 0 or i == len(scenes):
            log(f"  [{i}/{len(scenes)}] train features; {(time.time() - t0) / 60:.1f} min")
    meta = writer.close(encoders=list(encoders), points_per_scene=n_points)
    return dict(rows=meta["n"], scenes=len(rows), widths=meta["widths"], minutes=round((time.time() - t0) / 60, 2),
                store_gb=dir_gb(root))


def fit_configs(root: Path, configs: dict, pcfg: dict, device: str, forbidden: Path | None = None) -> tuple[dict, dict]:
    """The pilot's recipe (Experiment 1 step4_probes.run_phase) for each configuration {name: blocks} in its own
    phase: the learning rate from the grid by point mIoU on the held-out train scenes, then the seeds on all scenes.
    `forbidden`: a folder whose files must never reach a probe (Run B's val diagnostics)."""
    if forbidden is not None:
        assert not Path(root).resolve().is_relative_to(Path(forbidden).resolve()), "probe features inside Run B's outputs"
    store = FeatureStore(root)
    n = len(store.scenes)
    order = np.random.default_rng(0).permutation(n)
    held, train = sorted(order[:pcfg["holdout"]].tolist()), sorted(order[pcfg["holdout"]:].tolist())
    args = SimpleNamespace(batch=pcfg["batch"])
    folded, info = {}, {}
    for config, blocks in configs.items():
        probes_mod.CONFIGS[config] = (tuple(blocks), False)
        try:
            _, _, sweep, held_miou = run_phase(store, train, tuple(blocks), [config], pcfg["lrs"], lambda k: float(k.split("@")[1]),
                                               pcfg["sweep_epochs"], args, device, eval_ids=held)
            best = max(pcfg["lrs"], key=lambda lr: held_miou[f"{config}@{lr:g}"])
            probes, stats, curve, _ = run_phase(store, list(range(n)), tuple(blocks), [config], list(range(pcfg["seeds"])),
                                                lambda k: best, pcfg["epochs"], args, device)
        finally:
            probes_mod.CONFIGS.pop(config, None)
        folded.update({k: p.fold(stats) for k, p in probes.items()})
        info[config] = dict(blocks=list(blocks), lr=best, held_out_miou=held_miou, sweep_loss=sweep, final_loss=curve,
                            width=int(sum(store.widths[b] for b in blocks)))
    return folded, info


def evaluate_val(encoders: dict, folded: dict, archives, scenes, gt_source, dev, workers, reference: dict | None = None,
                 reference_scenes: int = 0, chunk: int = 200_000, log=print) -> tuple[dict, dict, list]:
    """Per-scene top-1/top-3 confusion matrices of every folded probe on every val vertex (anonymised excluded), with
    blocks "<name>_f" and "<name>_h" computed per chunk. `reference`: Experiment 2's folded probes on block "student",
    applied to `<first encoder>_f` on the first `reference_scenes` scenes (the reproduction check). Rows record each
    encoder's tile count per scene (1 = one pass; more when a single pass ran out of GPU memory)."""
    fs = FoldedSet(folded, device=dev)
    ref = FoldedSet(reference, device=dev) if reference else None
    first = next(iter(encoders))
    per_scene = {k: {1: [], 3: []} for k in fs.keys}
    ref_cms = {k: {1: [], 3: []} for k in (ref.keys if ref else [])}
    rows, t0 = [], time.time()
    for i, (scene, s, waited) in enumerate(prefetch_points([(sc, archives[sc]) for sc in scenes], workers), 1):
        if isinstance(s, Exception):
            raise s
        n = len(s["coord"])
        gt = torch.from_numpy(gt_source.labels(scene)).to(dev)
        assert len(gt) == n, f"{scene}: GT has {len(gt)} vertices, the scene {n}"
        keep = np.ones(n, bool)
        keep[s["anonymised"]] = False
        keep_t = torch.from_numpy(keep).to(dev)
        stages, tiles = {}, {}
        for name, (model, transform, _) in encoders.items():
            stages[name], tiles[name] = encode_scene(model, transform, s, scene_seed(scene), store="cpu")
        use_ref = ref is not None and i <= reference_scenes
        sets = [(fs, per_scene)] + ([(ref, ref_cms)] if use_ref else [])
        cms = [{k: {kk: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=dev) for kk in (1, 3)} for k in f.keys}
               for f, _ in sets]
        for start in range(0, n, chunk):
            idx = torch.arange(start, min(start + chunk, n))
            feats = {}
            for name, (_, _, head) in encoders.items():
                feats[f"{name}_f"] = stages[name].gather(idx, device=dev)
                if head is not None:
                    feats[f"{name}_h"] = head_features(head, feats[f"{name}_f"])
            if use_ref:
                feats["student"] = feats[f"{first}_f"]
            m, g = keep_t[start:start + len(idx)], gt[start:start + len(idx)]
            for (fset, _), cm in zip(sets, cms):
                for k, lg in fset.logits({b: feats[b] for b in fset.blocks}).items():
                    top = lg.topk(3, dim=1).indices
                    for kk in (1, 3):
                        cm[k][kk] += confusion_topk(top[m, :kk], g[m], NUM_CLASSES)
        for (_, store), cm in zip(sets, cms):
            for k in cm:
                for kk in (1, 3):
                    store[k][kk].append(cm[k][kk].cpu().numpy().astype(np.int32))
        rows.append(dict(scene=scene, n_vertices=n, kept=int(keep.sum()), tiles=tiles, waited=waited))
        log(f"  [{i}/{len(scenes)}] {scene}: {n:,} vertices; {(time.time() - t0) / 60:.1f} min")
    stack = lambda d: {k: {kk: np.stack(v) for kk, v in x.items()} for k, x in d.items() if x[1]}  # noqa: E731
    return stack(per_scene), stack(ref_cms), rows


# ---- rows and differences ----------------------------------------------------------------------------------------

def flat(cms: dict) -> dict:
    return {f"{k}|top{kk}": v for k, d in cms.items() for kk, v in d.items()}


def row_of(cms1: np.ndarray, cms3: np.ndarray) -> dict:
    s1 = [100 * float(miou(c.sum(0))) for c in cms1]
    s3 = [100 * float(miou(c.sum(0))) for c in cms3]
    return dict(top1=float(np.mean(s1)), top1_std=float(np.std(s1)), top3=float(np.mean(s3)), top3_std=float(np.std(s3)),
                seeds=len(s1))


def diff_of(a: dict, b: dict, ra: dict, rb: dict, kk: int = 1) -> dict:
    """Unrounded difference of means, 95 % interval and share of resamples <= 0 (paired bootstrap over scenes)."""
    bd = bootstrap_diff(a[kk], b[kk])
    key = "top1" if kk == 1 else "top3"
    return dict(diff=ra[key] - rb[key], lo=100 * bd["lo"], hi=100 * bd["hi"], p_le_0=bd["p_le_0"])


def table_rows(cms: dict, names: dict, inputs: dict, provenance: dict, ref_s: str = "S", ref_c: str = "S+") -> tuple[dict, dict]:
    """rows {config: stats + recovered shares}, diffs vs frozen Sonata for every row. cms: {config: {1: ..., 3: ...}}."""
    rows = {c: dict(name=names[c], inference_inputs=inputs[c], provenance=provenance[c], **row_of(cms[c][1], cms[c][3]))
            for c in cms}
    gain1 = rows[ref_c]["top1"] - rows[ref_s]["top1"]
    gain3 = rows[ref_c]["top3"] - rows[ref_s]["top3"]
    for c, r in rows.items():
        r["recovered_top1"] = (r["top1"] - rows[ref_s]["top1"]) / gain1 if gain1 else None
        r["recovered_top3"] = (r["top3"] - rows[ref_s]["top3"]) / gain3 if gain3 else None
        if c != ref_s:
            r["vs_frozen_sonata_top1"] = diff_of(cms[c], cms[ref_s], r, rows[ref_s], 1)
            r["vs_frozen_sonata_top3"] = diff_of(cms[c], cms[ref_s], r, rows[ref_s], 3)
    return rows, dict(top1=gain1, top3=gain3)


def pilot_and_exp2(pilot_dir: Path, exp2: Path, scenes: list[str]) -> dict:
    """{config: {1, 3}} for S and S+ (the pilot) and A1 (Experiment 2's student f(p)), restricted to `scenes` in order."""
    z = np.load(pilot_dir / "confmats.npz")
    pscenes = [str(s) for s in z["scenes"]]
    e = np.load(exp2 / "evaluation" / "confmats.npz")
    escenes = [str(s) for s in e["scenes"]]
    psel, esel = [pscenes.index(s) for s in scenes], [escenes.index(s) for s in scenes]
    return {"S": {kk: stack_config(z, "S", kk, psel) for kk in (1, 3)},
            "S+": {kk: stack_config(z, "S+", kk, psel) for kk in (1, 3)},
            "A1": {kk: stack_config(e, "T", kk, esel) for kk in (1, 3)}}

