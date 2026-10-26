"""The GPU confusion matrix against the ScanNet++ toolbox's own function, and the scene bootstrap."""

import numpy as np
import pytest

from conftest import load_script

torch = pytest.importorskip("torch")
metrics = load_script("experiments/pilot_concat/metrics.py")
toolbox = load_script("third_party/scannetpp/semantic/utils/confmat.py")


def random_case(rng, n=5000, num_classes=12):
    gt = rng.integers(0, num_classes, (n, 3))
    n_labels = rng.choice([0, 1, 2, 3], n, p=[0.1, 0.6, 0.2, 0.1])  # labels fill from the first column
    gt[np.arange(3)[None, :] >= n_labels[:, None]] = metrics.IGNORE
    pred = np.stack([rng.permutation(num_classes)[:3] for _ in range(n)])
    near = rng.random(n) < 0.4  # make hits common, in any prediction column
    pred[near, rng.integers(0, 3, near.sum())] = gt[near, 0].clip(0)
    return pred, gt, num_classes


@pytest.mark.parametrize("k", [1, 3])
def test_confusion_topk_matches_the_toolbox(k):
    rng = np.random.default_rng(k)
    pred, gt, c = random_case(rng)
    ours = metrics.confusion_topk(torch.from_numpy(pred[:, :k]), torch.from_numpy(gt), c).numpy()
    theirs = toolbox.fast_hist_topk_multilabel(pred[:, :k], gt, c, metrics.IGNORE)
    np.testing.assert_array_equal(ours, theirs)
    cm = toolbox.ConfMat(c, top_k_pred=k, ignore_label=metrics.IGNORE)
    cm.update(torch.from_numpy(pred), torch.from_numpy(gt))
    assert np.isclose(metrics.miou(ours), cm.miou) and np.allclose(metrics.ious(ours), cm.ious, equal_nan=True)


def test_bootstrap_is_paired_and_centred():
    rng = np.random.default_rng(0)
    c, scenes = 5, 30
    base = rng.integers(0, 50, (1, scenes, c, c)) + np.eye(c, dtype=int) * 200
    better = base + np.eye(c, dtype=int) * 100  # more true positives in every scene
    same = metrics.bootstrap_diff(base, base, n=200)
    assert same["diff"] == 0 and same["lo"] == 0 == same["hi"]
    up = metrics.bootstrap_diff(better, base, n=200)
    assert up["diff"] > 0 and up["lo"] > 0 and up["p_le_0"] == 0
