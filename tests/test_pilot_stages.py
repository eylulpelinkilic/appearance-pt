"""Stages.gather against encode()'s upcast loop, and the out-of-memory tiling, on fake encoder outputs (CPU)."""

import numpy as np
import pytest

from conftest import load_script

torch = pytest.importorskip("torch")
stages = load_script("experiments/pilot_concat/stages.py")


class FakePoint:
    """The parts of a Pointcept Point that encode() touches."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def keys(self):
        return list(self.__dict__)

    def pop(self, key):
        return self.__dict__.pop(key)


def fake_output(n_input=200, sizes=(60, 25, 9), widths=(4, 6, 8), seed=0):
    """A 3-stage encoder output: stage 0 has sizes[0] grid points; each stage pools into the next."""
    g = torch.Generator().manual_seed(seed)
    feats = [torch.randn(n, w, generator=g) for n, w in zip(sizes, widths)]
    invs = [torch.randint(0, sizes[k + 1], (sizes[k],), generator=g) for k in range(len(sizes) - 1)]
    point = FakePoint(feat=feats[0], inverse=torch.randint(0, sizes[0], (n_input,), generator=g))
    for k in range(1, len(sizes)):
        point = FakePoint(feat=feats[k], pooling_parent=point, pooling_inverse=invs[k - 1])
    return point


def upcast_like_encode(point):
    """encoders.encode's loop (DefaultSegmentorV2): every stage concatenated onto its parent, back to stage 0."""
    while "pooling_parent" in point.keys():
        parent, inverse = point.pop("pooling_parent"), point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    return point.feat, point.inverse


def test_gather_equals_the_upcast():
    st = stages.Stages.from_point(fake_output(), dtype=torch.float32)
    assert st.width == 18 and st.n_grid == 60
    feat, inverse = upcast_like_encode(fake_output())
    idx = torch.tensor([0, 5, 199, 5, 77])
    torch.testing.assert_close(st.gather(idx), feat[inverse[idx]])


def test_tiling_covers_every_point_once_and_falls_back_on_oom():
    rng = np.random.default_rng(0)
    coord = rng.uniform(0, [10, 4, 3], (2000, 3)).astype(np.float32)
    calls = []

    def encode(inputs):  # "runs out of memory" above 700 inputs; features = the global index, to check routing
        calls.append(len(inputs))
        if len(inputs) > 700:
            raise RuntimeError("CUDA error: out of memory")
        f = torch.from_numpy(inputs.astype(np.float32))[:, None]
        return stages.Stages([f], [], torch.arange(len(inputs)))

    tiled, n_tiles = stages.encode_tiled(encode, coord, margin=0.5)
    assert n_tiles >= 4 and calls[0] == 2000 and tiled.complete
    idx = torch.from_numpy(rng.choice(2000, 300, replace=False))
    np.testing.assert_array_equal(tiled.gather(idx, device="cpu")[:, 0].numpy(), idx.numpy())  # each point from its own tile

    with pytest.raises(RuntimeError, match="other"):
        stages.encode_tiled(lambda inputs: (_ for _ in ()).throw(RuntimeError("other failure")), coord)
