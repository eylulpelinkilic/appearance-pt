"""The inert checkpoint loader (no code from the file runs) and the half-room probe, on CPU."""

import collections
import json

import numpy as np
import pytest

from conftest import load_script

torch = pytest.importorskip("torch")
from appearance_pt.checkpoints import Inert, InertUnpickler, load_inert  # noqa: E402

probes = load_script("experiments/pilot_concat/probes.py")
CALLED = []


class Hostile:
    """Unpickling this with a normal unpickler would call CALLED.append (stand-in for arbitrary code)."""

    def __reduce__(self):
        return (CALLED.append, ("ran",))


class Config:
    def __init__(self):
        self.lr = 0.004


def test_load_inert_keeps_tensors_and_runs_nothing(tmp_path):
    w = torch.arange(12, dtype=torch.float32).view(3, 4)
    state = {"epoch": 99, "state_dict": collections.OrderedDict([("module.student.backbone.w", w),
                                                                  ("module.teacher.backbone.w", w + 1)]),
             "optimizer": {"state": {0: {"exp_avg": torch.ones(3)}}, "cfg": Config()}, "hook": Hostile()}
    path = tmp_path / "ckpt.pth"
    torch.save(state, path)
    loaded = load_inert(path)
    assert CALLED == []
    torch.testing.assert_close(loaded["state_dict"]["module.student.backbone.w"], w)
    torch.testing.assert_close(loaded["state_dict"]["module.teacher.backbone.w"], w + 1)
    assert loaded["epoch"] == 99 and isinstance(loaded["optimizer"]["cfg"], Inert) and isinstance(loaded["hook"], Inert)
    assert any(g.endswith("Config") for g in InertUnpickler.stubbed)
    with pytest.raises(Exception):  # the stock safe loader refuses this file, which is why load_inert exists
        torch.load(path, weights_only=True)


def test_split_probe_separates_informative_from_random_features():
    rng = np.random.default_rng(0)
    n, k = 4000, 4
    x = rng.uniform(0, 10, n)
    gt = rng.integers(0, k, n)
    gt[:100] = -1  # unlabelled points are ignored
    informative = np.eye(k)[np.clip(gt, 0, None)] * 3 + rng.normal(size=(n, k))
    noise = rng.normal(size=(n, 16))
    good = probes.split_probe(torch.tensor(informative, dtype=torch.float32), gt, x, num_classes=k, steps=100)
    bad = probes.split_probe(torch.tensor(noise, dtype=torch.float32), gt, x, num_classes=k, steps=100)
    assert good["miou"] > 0.8 and bad["miou"] < 0.3 and len(good["miou_folds"]) == 2


def synthetic_store(root, n_scenes=6, rows=300, classes=5, seed=0):
    """Two blocks: 'enc' (informative), 'img' (half informative), raw features in float16 like step 3."""
    rng = np.random.default_rng(seed)
    w = probes.FeatureWriter(root, n_scenes * rows, {"enc": 12, "img": 8}, {"enc": np.float16, "img": np.float16})
    centres = rng.normal(size=(classes, 12)) * 3
    for s in range(n_scenes):
        y = rng.integers(0, classes, rows)
        enc = centres[y] + rng.normal(size=(rows, 12)) + 5  # offset and scale: standardisation must handle them
        img = np.c_[centres[y, :4] * 0.5, rng.normal(size=(rows, 4))] * 10
        w.add(f"scene{s}", {"enc": enc.astype(np.float16), "img": img.astype(np.float16)}, y, rng.integers(0, 5, rows))
    w.close(note="test")
    return probes.FeatureStore(root)


def test_probe_training_folding_and_joint_application(tmp_path):
    store = synthetic_store(tmp_path / "store")
    assert store.meta["n"] == 1800 and len(store.scenes) == 6
    train, held = [0, 1, 2, 3, 4], [5]
    data, label = probes.place(*store.read(train, ["enc", "img"]), "cpu")
    stats = probes.block_stats(data)
    proj = {b: probes.fit_pca(data[b], stats[b], width=4, device="cpu") for b in ("enc", "img")}
    probe_set = {"enc": probes.Probe(("enc",), store.widths, num_classes=5),
                 "both": probes.Probe(("enc", "img"), store.widths, num_classes=5),
                 "both-eq": probes.Probe(("enc", "img"), store.widths, proj=proj, num_classes=5)}
    curve = probes.train_probes(data, label, probe_set, {k: 1e-2 for k in probe_set}, stats, epochs=5, batch=256,
                                device="cpu", log=lambda *a: None)
    assert curve[-1] < curve[0]
    held_miou = probes.evaluate_probes(*probes.place(*store.read(held, ["enc", "img"]), "cpu"), probe_set, stats, device="cpu")
    assert all(m > 0.9 for m in held_miou.values()), held_miou

    folded = probes.FoldedSet({k: p.fold(stats) for k, p in probe_set.items()}, device="cpu")
    data, _ = probes.place(*store.read(held, ["enc", "img"]), "cpu")
    raw = {b: x.float() for b, x in data.items()}
    std = probes.standardise(data, slice(None), {b: tuple(torch.from_numpy(a) for a in stats[b]) for b in data}, "cpu")
    joint = folded.logits(raw)
    for k, p in probe_set.items():
        with torch.no_grad():
            torch.testing.assert_close(joint[k], p.eval()(std), rtol=1e-3, atol=1e-3)


def test_step4_end_to_end_on_a_tiny_store(tmp_path, monkeypatch):
    import sys

    rng = np.random.default_rng(1)
    widths = {"utonia": 10, "sonata": 9, "concerto": 12, "random": 10, "dino": 8, "raw": 9}
    w = probes.FeatureWriter(tmp_path / "f", 8 * 200, widths, {b: np.float16 for b in widths})
    centres = {b: rng.normal(size=(6, d)) * 3 for b, d in widths.items()}
    for s in range(8):
        y = rng.integers(0, 6, 200)
        w.add(f"s{s}", {b: (centres[b][y] + rng.normal(size=(200, d))).astype(np.float16) for b, d in widths.items()},
              y, rng.integers(0, 4, 200))
    w.close()
    step4 = load_script("experiments/pilot_concat/step4_probes.py")
    monkeypatch.setattr(sys, "argv", ["step4", "--features", str(tmp_path / "f"), "--out", str(tmp_path / "out"),
                                      "--lrs", "0.003", "0.03", "--holdout", "2", "--sweep-epochs", "2", "--epochs", "2",
                                      "--seeds", "2", "--batch", "256"])
    assert step4.main() == 0
    saved = torch.load(tmp_path / "out" / "probes.pt", weights_only=False)
    assert len(saved["probes"]) == 2 * len(probes.CONFIGS) and set(saved["lr"]) == set(probes.CONFIGS)
    k_plus = saved["probes"]["K+#0"]
    assert set(k_plus["blocks"]) == {"concerto", "dino"} and k_plus["blocks"]["concerto"].shape == (100, 12)
    summary = json.loads((tmp_path / "out" / "step4_summary.json").read_text())
    assert summary["held_out_miou"]["K+@0.03"] > 0.5
