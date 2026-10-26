"""Per-class intervals of pilot step 5 (step5_classes.py) on synthetic confusion matrices."""

import json
import sys

import numpy as np
import pytest

from conftest import load_script

metrics = load_script("experiments/pilot_concat/metrics.py")
classes = load_script("experiments/pilot_concat/step5_classes.py")


def synthetic(rng, n_scenes=20, c=6, seeds=3):
    """K and K+ per-scene matrices: K+ gains on class 0 in every scene, on class 1 in one scene only; class 5 absent."""
    base = rng.integers(0, 20, (seeds, n_scenes, c, c)) + np.eye(c, dtype=np.int64) * 300
    base[..., 5, :] = base[..., :, 5] = 0
    base[..., 1, :] = 0
    base[:, :2, 1, 1] = 50  # class 1 occurs in two scenes
    better = base.copy()
    better[..., 0, 0] += 300
    better[:, 0, 1, 1] += 500
    return {"K": base, "K+": better}


@pytest.mark.filterwarnings("ignore:Mean of empty slice")  # class 5 in the reference table
def test_intervals_separate_steady_gains_from_one_scene(tmp_path, monkeypatch):
    cms = synthetic(np.random.default_rng(0))
    names = [f"class{i}" for i in range(6)]
    scenes = [f"scene{i:02d}" for i in range(20)]
    table = lambda cm: {n: (None if np.isnan(v) else round(100 * float(v), 2))  # noqa: E731  (as step 5 writes it)
                        for n, v in zip(names, np.nanmean(metrics.ious(cm.sum(1)), 0))}
    results = dict(scenes=scenes, iou_top1={c: {"all": table(cm)} for c, cm in cms.items()},
                   gains_top1={"K+ - K": {"seen": {n: 1.0 for n in names}, "unseen": {n: -1.0 for n in names}}})
    (tmp_path / "results.json").write_text(json.dumps(results))
    np.savez_compressed(tmp_path / "confmats.npz", scenes=np.array(scenes),
                        **{f"{c}#{s}|top{k}": cm[s] for c, cm in cms.items() for s in range(3) for k in (1, 3)})
    monkeypatch.setattr(sys, "argv", ["prog", "--results", str(tmp_path / "results.json"),
                                      "--confmats", str(tmp_path / "confmats.npz"), "--boot", "300"])
    assert classes.main() == 0

    out = json.loads((tmp_path / "classes.json").read_text())
    d = out["pairs"]["K+ - K"]
    assert d["lo"][0] > 0 and d["p_le_0"][0] == 0  # steady gain
    assert d["diff"][1] > 0.1 and d["lo"][1] <= 0  # large, but from one of two scenes
    assert np.isnan(d["diff"][5]) and out["scenes_with"][1] == 2 and out["scenes_with"][5] == 0
    assert d["n_up"] >= 1 and d["n_classes"] == 5 and list(out["pairs"]) == ["K+ - K"]
    md = (tmp_path / "classes.md").read_text()
    assert md.index("| class0 |") < md.index("| class4 |") and "+1.0 | -1.0 |" in md
