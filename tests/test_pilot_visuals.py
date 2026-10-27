"""Pilot visuals (step6_visuals.py) on CPU: ray geometry, the K vs K+ outcomes, and an end-to-end run on synthetic
rooms with the stand-ins of test_pilot_steps345 (encoder, DINO, toolbox GT) and a point z-buffer in place of Open3D's
rays at the mesh."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import ROOT, load_script

torch = pytest.importorskip("torch")
sys.path.insert(0, str(ROOT / "experiments" / "pilot_concat"))
e2e = load_script("tests/test_pilot_steps345.py")
scenes_test = e2e.scenes_test


@pytest.fixture
def step6(monkeypatch):
    e2e.stub_packages(monkeypatch)
    return load_script("experiments/pilot_concat/step6_visuals.py")


def test_rays_pass_through_pixel_centres(step6):
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = q * np.sign(np.linalg.det(q)), rng.normal(size=3)
    K, hw, scale = np.array([[300.0, 0, 160], [0, 310, 120], [0, 0, 1]]), (240, 320), 0.25
    rays, (h, w) = step6.pixel_rays(w2c, K, hw, scale)
    pts = rays[:, :3].astype(np.float64) + 2.5 * rays[:, 3:]
    u, v, z = sys.modules["lifting"].project(torch.from_numpy(pts), torch.from_numpy(w2c), torch.from_numpy(K))
    vv, uu = np.meshgrid((np.arange(h) + 0.5) / scale, (np.arange(w) + 0.5) / scale, indexing="ij")
    np.testing.assert_allclose(u.numpy(), uu.ravel(), atol=0.01)
    np.testing.assert_allclose(v.numpy(), vv.ravel(), atol=0.01)
    assert (z.numpy() > 0).all() and (h, w) == (60, 80)


def test_nearest_vertex_and_outcomes(step6):
    faces = np.array([[3, 5, 7], [1, 2, 4]])
    uv = np.array([[0.1, 0.8], [0.1, 0.1], [0.0, 0.0]], np.float32)
    assert step6.nearest_vertex(faces, np.array([0, 1, 2**32 - 1], np.uint32), uv).tolist() == [7, 1, -1]
    gt = np.array([[2, 5, -100], [3, -100, -100], [4, -100, -100], [6, -100, -100], [-100] * 3, [1, -100, -100]], np.int16)
    k, kp = np.array([2, 0, 4, 0, 0, 1], np.int16), np.array([5, 3, 0, 0, 0, 1], np.int16)
    keep = np.array([1, 1, 1, 1, 1, 0], bool)
    assert step6.outcome(k, kp, gt, keep).tolist() == [2, 0, 1, 3, -1, -1]  # a hit on any GT label counts


class ZBufferRays:
    """Stand-in for MeshRays without Open3D: each pixel takes the nearest vertex that projects into it."""

    def __init__(self, path, n_vertices, tmp):
        self.coord = np.load(next(Path(path).glob("pointcept/*/*/coord.npy"))).astype(np.float64)
        assert len(self.coord) == n_vertices

    def vertex_ids(self, w2c, K, hw, scale):
        h, w = int(hw[0] * scale), int(hw[1] * scale)
        cam = self.coord @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        ok = z > 0.05
        zs = np.where(ok, z, 1.0)
        u, v = (K[0, 0] * cam[:, 0] / zs + K[0, 2]) * scale, (K[1, 1] * cam[:, 1] / zs + K[1, 2]) * scale
        ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = np.flatnonzero(ok)
        flat = v[ok].astype(np.int64) * w + u[ok].astype(np.int64)
        order = np.lexsort((z[ok], flat))
        first = np.unique(flat[order], return_index=True)[1]
        ids = np.full(h * w, -1, np.int64)
        ids[flat[order][first]] = idx[order][first]
        return ids.reshape(h, w)


def fake_probes(path):
    """A (random weights on DINO), K (reads the label channel of the fake Concerto features), K+ (K plus zeros)."""
    g = torch.Generator().manual_seed(0)
    concerto = torch.zeros(100, e2e.WIDTHS["concerto"])
    concerto[torch.arange(12), torch.arange(12)] = 1.0
    probes = {"A#0": dict(blocks={"dino": torch.randn(100, 9, generator=g)}, bias=torch.zeros(100)),
              "K#0": dict(blocks={"concerto": concerto}, bias=torch.zeros(100)),
              "K+#0": dict(blocks={"concerto": concerto, "dino": torch.zeros(100, 9)}, bias=torch.zeros(100))}
    torch.save(dict(probes=probes, features=dict(views=4, occlusion="v2", encoder_names={"concerto": "concerto"})), path)


def test_visuals_on_synthetic_rooms(step6, tmp_path, monkeypatch):
    val = json.loads((ROOT / "configs" / "scannetpp_subset.json").read_text())["val"][:2]
    scenes_test.write_scene(tmp_path / "input" / "core_val", val[0], anonymised=range(0, 40_000, 7))
    scenes_test.write_scene(tmp_path / "input" / "core_val", val[1])
    fake_probes(tmp_path / "probes.pt")
    monkeypatch.setattr(step6, "load_models", lambda concerto, device: ("concerto", None, scenes_test.FakeDino()))
    monkeypatch.setattr(step6, "encode_scene", e2e.fake_encode_scene(sys.modules["stages"]))
    monkeypatch.setattr(step6, "OfficialGT", e2e.FakeGT)
    monkeypatch.setattr(step6, "MeshRays", ZBufferRays)
    out = tmp_path / "visuals"
    monkeypatch.setattr(sys, "argv", ["prog", "--input", str(tmp_path / "input"), "--probes", str(tmp_path / "probes.pt"),
                                      "--out", str(out), "--tmp", str(tmp_path / "tmp"), "--scenes", *val,
                                      "--scale", "1.0", "--device", "cpu"])
    assert step6.main() == 0

    res = json.loads((out / "visuals.json").read_text())["scenes"]
    assert [r["scene"] for r in res] == val and all(r["matches_step5"] is None for r in res)
    for r in res:
        assert r["miou"]["K#0"] > r["miou"]["A#0"] + 20, r["miou"]  # spurious classes count as IoU 0, as in the metric
        assert r["files"] == [f"{r['scene']}_view1.png", f"{r['scene']}_view2.png", f"{r['scene']}_topdown.png"]
        assert len({v["view"] for v in r["views"]}) == 2 and r["views"][0]["score"] >= r["views"][1]["score"]
        with Image.open(out / r["files"][0]) as im:
            assert im.width == 3 * scenes_test.HW[1] and im.height > 2 * scenes_test.HW[0]
