"""Pilot step 5 per class, with intervals: runs locally on CPU from step 5's results.json and confmats.npz.

For each pair of configurations (K+ - K first), per class: in how many val scenes it occurs, its share of the
evaluated vertices, both IoUs (top-1, all kept vertices, mean over probe seeds) and a paired bootstrap over scenes of
the difference, with the same resamples as step 5's mIoU bootstrap. A per-class difference counts only if its 95 %
interval excludes 0 (by more than CLEAR): a class that occurs in two val scenes can move by 20 IoU points with one object. Classes are
also grouped into thirds by vertex count, and the pairs' per-class gains are rank-correlated, to see whether lifted
DINO helps the same classes on top of every encoder. The seen / never-seen per-class gains are taken from
results.json without intervals (step 5 keeps those confusion matrices only summed over scenes).

Usage: python experiments/pilot_concat/step5_classes.py --results results.json --confmats confmats.npz [--out DIR]
Writes <out>/classes.json and <out>/classes.md (default: next to results.json).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

PAIRS = (("K+", "K"), ("S+", "S"), ("C", "B"), ("K", "S"), ("K+", "S+"))
CLEAR = 0.0005  # an interval counts as excluding 0 only if it clears it by 0.05 IoU points (one printed decimal): a class
#                 absent from many resamples puts a bound at exactly 0 or a hair from it
GROUPS = ("frequent", "middle", "rare")


def load_confmats(path: Path) -> tuple[list[str], dict]:
    """Scene ids and {config: (seeds, scenes, C, C)} top-1 confusion matrices per scene."""
    by_config = {}
    with np.load(path) as z:
        scenes = [str(s) for s in z["scenes"]]
        for key in z.files:
            if key.endswith("|top1"):
                config, seed = key.removesuffix("|top1").split("#")
                by_config.setdefault(config, {})[int(seed)] = z[key]
    return scenes, {c: np.stack([d[s] for s in sorted(d)]) for c, d in by_config.items()}


def per_class_iou(cms: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """IoU per class of the scene-weighted sums, averaged over seeds: cms (seeds, S, C, C), weights (R, S) -> (R, C)."""
    cms = cms.astype(np.float64)
    tp = np.einsum("rs,ksc->krc", weights, np.diagonal(cms, axis1=-2, axis2=-1))
    gt = np.einsum("rs,ksc->krc", weights, cms.sum(-1))
    pred = np.einsum("rs,ksc->krc", weights, cms.sum(-2))
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # classes absent from a resample stay NaN
        return np.nanmean(tp / (gt + pred - tp), 0)


def resample_weights(n_scenes: int, n: int = 1000, seed: int = 0) -> np.ndarray:
    """The scene resamples of metrics.bootstrap_diff (same generator, same order)."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_scenes, n_scenes), minlength=n_scenes) for _ in range(n)]).astype(np.float64)


def interval(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2.5 / 97.5 percentiles and P(diff <= 0) over the resamples (axis 0), ignoring NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(samples, [2.5, 97.5], axis=0)
        p = np.nanmean(np.where(np.isnan(samples), np.nan, samples <= 0), 0)
    return lo, hi, p


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = ~(np.isnan(a) | np.isnan(b))
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def analyse(results: dict, cms: dict, n_boot: int = 1000) -> dict:
    names = list(results["iou_top1"]["K"]["all"])
    n_scenes = next(iter(cms.values())).shape[1]
    ones, weights = np.ones((1, n_scenes)), resample_weights(n_scenes, n_boot)
    gt_rows = cms["K"][0].sum(-1)  # (S, C); the GT class of a multilabel vertex can differ by probe (< 0.5 % of vertices)
    vertices, scenes_with = gt_rows.sum(0), (gt_rows > 0).sum(0)
    present = np.flatnonzero(vertices > 0)
    groups = dict(zip(GROUPS, np.array_split(present[np.argsort(-vertices[present], kind="stable")], 3)))

    iou_obs = {c: per_class_iou(cms[c], ones)[0] for c in {c for pair in PAIRS for c in pair} if c in cms}
    for c, v in iou_obs.items():  # the same numbers as step 5's per-class table
        ref = np.array([np.nan if x is None else x for x in results["iou_top1"][c]["all"].values()])
        assert np.allclose(100 * v, ref, atol=0.006, equal_nan=True), c
    out = dict(classes=names, vertices=vertices.tolist(), scenes_with=scenes_with.tolist(),
               groups={g: [names[i] for i in ix] for g, ix in groups.items()}, pairs={})
    boot = {}
    for a, b in PAIRS:
        if a not in cms or b not in cms:
            continue
        diff = iou_obs[a] - iou_obs[b]
        boot[(a, b)] = samples = per_class_iou(cms[a], weights) - per_class_iou(cms[b], weights)
        lo, hi, p = interval(samples)
        group_stats = {}
        for g, ix in groups.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                s = np.nanmean(samples[:, ix], 1)
            group_stats[g] = dict(mean=float(np.nanmean(diff[ix])), lo=float(np.nanpercentile(s, 2.5)), hi=float(np.nanpercentile(s, 97.5)))
        key = f"{a} - {b}"
        gains = results.get("gains_top1", {}).get(key, {})
        out["pairs"][key] = dict(
            diff=diff.tolist(), lo=lo.tolist(), hi=hi.tolist(), p_le_0=p.tolist(), groups=group_stats,
            iou_a=iou_obs[a].tolist(), iou_b=iou_obs[b].tolist(),
            n_up=int((lo > CLEAR).sum()), n_down=int((hi < -CLEAR).sum()), n_classes=int((~np.isnan(diff)).sum()),
            seen=[gains.get("seen", {}).get(n) for n in names], unseen=[gains.get("unseen", {}).get(n) for n in names])
    first = "K+ - K"
    if first in out["pairs"]:
        out["rank_correlation_with_K+_-_K"] = {k: spearman(np.array(out["pairs"][first]["diff"]), np.array(v["diff"]))
                                               for k, v in out["pairs"].items() if k != first}
    return out


def pct(x, signed=True):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}")


def markdown(out: dict, pair: str = "K+ - K") -> str:
    d, names = out["pairs"][pair], out["classes"]
    total = sum(out["vertices"])
    a, b = pair.split(" - ")
    lines = [f"Per class, {pair} (top-1 IoU, %; 95 % interval from a paired bootstrap over the 50 val scenes; "
             "seen / never-seen from step 5 without intervals). Sorted by the difference.", "",
             f"| Class | Val scenes | Vertices % | {b} | {a} | Diff | 95 % interval | Seen | Never seen |",
             "|---|---|---|---|---|---|---|---|---|"]
    order = sorted((i for i in range(len(names)) if not np.isnan(d["diff"][i])), key=lambda i: -d["diff"][i])
    for i in order:
        sure = d["lo"][i] > CLEAR or d["hi"][i] < -CLEAR
        diff = f"**{pct(d['diff'][i])}**" if sure else pct(d["diff"][i])
        seen = None if d["seen"][i] is None else d["seen"][i] / 100
        unseen = None if d["unseen"][i] is None else d["unseen"][i] / 100
        lines.append(f"| {names[i]} | {out['scenes_with'][i]} | {100 * out['vertices'][i] / total:.3f} | "
                     f"{pct(d['iou_b'][i], False)} | {pct(d['iou_a'][i], False)} | {diff} | "
                     f"[{pct(d['lo'][i])}, {pct(d['hi'][i])}] | {pct(seen)} | {pct(unseen)} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--confmats", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    out_dir = args.out or args.results.parent
    results = json.loads(args.results.read_text())
    scenes, cms = load_confmats(args.confmats)
    assert scenes == results["scenes"], "confmats.npz and results.json come from different runs"
    out = analyse(results, cms, args.boot)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.json").write_text(json.dumps(out, indent=1))
    (out_dir / "classes.md").write_text(markdown(out))

    names = out["classes"]
    print(f"{len(scenes)} val scenes; {sum(v > 0 for v in out['vertices'])} classes present; groups by vertex count: "
          + ", ".join(f"{g} {len(v)}" for g, v in out["groups"].items()))
    print("pair          classes up / down (95 % interval excludes 0)   mean per-class diff by group [95 % interval]")
    for key, d in out["pairs"].items():
        groups = "; ".join(f"{g} {100 * s['mean']:+.1f} [{100 * s['lo']:+.1f}, {100 * s['hi']:+.1f}]" for g, s in d["groups"].items())
        print(f"  {key:10s}  {d['n_up']:3d} / {d['n_down']:3d} of {d['n_classes']}    {groups}")
    if out.get("rank_correlation_with_K+_-_K"):
        print("rank correlation of per-class diffs with K+ - K: "
              + ", ".join(f"{k} {v:+.2f}" for k, v in out["rank_correlation_with_K+_-_K"].items()))
    d = out["pairs"].get("K+ - K")
    if d:
        for label, ix in (("up", [i for i in range(len(names)) if d["lo"][i] > CLEAR]), ("down", [i for i in range(len(names)) if d["hi"][i] < -CLEAR])):
            ix.sort(key=lambda i: -abs(d["diff"][i]))
            print(f"K+ - K, classes {label} ({len(ix)}): " + ", ".join(
                f"{names[i]} {pct(d['diff'][i])} [{pct(d['lo'][i])}, {pct(d['hi'][i])}] ({out['scenes_with'][i]} scenes)" for i in ix))
    print(f"tables: {out_dir / 'classes.md'}, {out_dir / 'classes.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
