"""The inert checkpoint loader (no code from the file runs) and the half-room probe, on CPU."""

import collections

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
