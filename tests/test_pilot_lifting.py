"""DSLR cameras (against the toolbox's own pose export) and the lifting geometry, on synthetic scenes (CPU torch)."""

import sys
import types

import numpy as np
import pytest

from appearance_pt import scannetpp_cameras as cams
from conftest import ROOT, load_script

torch = pytest.importorskip("torch")
lifting = load_script("experiments/pilot_concat/lifting.py")


def random_w2c(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = q, rng.normal(size=3)
    return w2c


def test_dslr_c2w_undoes_the_toolbox_export():
    sys.path.insert(0, str(ROOT / "third_party" / "scannetpp"))
    try:
        from common.utils.nerfstudio import convert_frames
    finally:
        sys.path.pop(0)
    rng = np.random.default_rng(0)
    w2cs = [random_w2c(rng) for _ in range(5)]
    images = {i: types.SimpleNamespace(world_to_camera=w, name=f"DSC{i:05d}.JPG") for i, w in enumerate(w2cs)}
    for frame, w2c in zip(convert_frames(images), w2cs):
        np.testing.assert_allclose(cams.dslr_c2w(frame["transform_matrix"]), np.linalg.inv(w2c), atol=1e-12)


def test_intrinsics_frames_and_view_pick():
    tf = dict(camera_model="PINHOLE", fl_x=900.0, fl_y=901.0, cx=876.0, cy=584.0, w=1752, h=1168,
              frames=[dict(file_path="DSC00001.JPG")], test_frames=[dict(file_path="DSC00002.JPG")])
    K, hw = cams.dslr_intrinsics(tf)
    assert hw == (1168, 1752) and K[0, 0] == 900 and K[1, 2] == 584
    assert sorted(cams.dslr_frames(tf)) == ["DSC00001.JPG", "DSC00002.JPG"]
    pool = [f"v{i:02d}" for i in range(60)]
    picked = cams.pick_views(pool, 30)
    assert len(picked) == 30 and picked[0] == "v00" and picked[-1] == "v59"
    assert cams.pick_views(pool[:20], 30) == pool[:20]


def scene_with_occluder():
    """A wall at z = 4 m and a 1 m square plate at z = 2 m in front of it; camera at the origin looking along +z."""
    g = np.arange(-3, 3, 0.01)
    wall = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    wall = np.c_[wall, np.full(len(wall), 4.0)]
    p = np.arange(-0.5, 0.5, 0.005)
    plate = np.stack(np.meshgrid(p, p, indexing="ij"), -1).reshape(-1, 2)
    plate = np.c_[plate, np.full(len(plate), 2.0)]
    return torch.tensor(np.r_[wall, plate], dtype=torch.float32), len(wall)


def make_view(hw=(100, 200), f=100.0):
    K = torch.tensor([[f, 0, hw[1] / 2], [0, f, hw[0] / 2], [0, 0, 1]])
    return lifting.View("v", torch.eye(4), K, hw, torch.ones(hw, dtype=torch.bool))


def test_zbuffer_hides_points_behind_the_plate():
    points, n_wall = scene_with_occluder()
    view = make_view()
    view.zbuf = lifting.zbuffer(points, view)
    u, v, z = lifting.project(points, view.w2c, view.K)
    vis = lifting.visible(u, v, z, view)
    inside = lifting.in_image(u, v, z, view.hw)
    wall, plate = slice(0, n_wall), slice(n_wall, None)
    assert vis[plate].float().mean() > 0.99  # the plate is in front of everything
    x, y = points[wall, 0], points[wall, 1]  # the plate's shadow on the wall: |x|, |y| < 1 m (scale 4/2)
    shadow = (x.abs() < 0.9) & (y.abs() < 0.9)
    # the shadow's edge (u = 125 px) rounds up to a cell edge (128 px) and the 3 x 3 filter adds a cell (136 px = 1.44 m)
    clear = inside[wall] & ((x.abs() > 1.5) | (y.abs() > 1.5))
    assert vis[wall][shadow].sum() == 0
    assert vis[wall][clear].float().mean() > 0.99


def test_mask_and_image_bounds():
    points, _ = scene_with_occluder()
    view = make_view()
    view.valid[:, :100] = False  # left half of the image unusable
    view.zbuf = lifting.zbuffer(points, view)
    u, v, z = lifting.project(points, view.w2c, view.K)
    vis = lifting.visible(u, v, z, view)
    assert vis.any() and (u[vis] >= 100).all()
    behind = torch.tensor([[0.0, 0.0, -1.0]])
    assert not lifting.visible(*lifting.project(behind, view.w2c, view.K), view).any()


def test_sample_hits_patch_centres():
    hw, grid = (28, 56), (2, 4)  # 2 x 4 patches of 14 px
    feat = torch.arange(8, dtype=torch.float32).view(1, *grid).repeat(3, 1, 1)  # channel value = patch index
    u = torch.tensor([7.0, 21.0, 35.0, 49.0, 7.0])
    v = torch.tensor([7.0, 7.0, 7.0, 7.0, 21.0])
    out = lifting.sample(feat, u, v, hw)
    assert out.shape == (5, 3)
    np.testing.assert_allclose(out[:, 0].numpy(), [0, 1, 2, 3, 4], atol=1e-5)
    mid = lifting.sample(feat, torch.tensor([14.0]), torch.tensor([7.0]), hw)  # between patches 0 and 1
    np.testing.assert_allclose(mid[0, 0].item(), 0.5, atol=1e-5)


def test_lift_averages_over_the_views_that_see_a_point():
    points, n_wall = scene_with_occluder()
    views = []
    for value, shift in ((1.0, 0.0), (3.0, 0.1)):  # two cameras, constant feature maps 1 and 3
        view = make_view()
        view.w2c[0, 3] = shift
        view.zbuf = lifting.zbuffer(points, view)
        view.feat = torch.full((4, 7, 14), value)
        views.append(view)
    starts, feats, counts = zip(*lifting.lift(points, views, chunk=50_000))
    assert starts[0] == 0 and starts[1] == 50_000
    feat, count = torch.cat(feats), torch.cat(counts)
    both, none = count == 2, count == 0
    assert both.any() and none.any()
    np.testing.assert_allclose(feat[both].numpy(), 2.0, atol=1e-5)
    assert (feat[none] == 0).all()
    assert (feat[count == 1] != 2.0).all()


def test_colour_agreement_prefers_the_right_pose():
    rng = np.random.default_rng(0)
    points, _ = scene_with_occluder()
    hw = (100, 200)
    image = torch.tensor(rng.integers(0, 256, (*hw, 3)), dtype=torch.uint8)
    view = make_view(hw)
    u, v, z = lifting.project(points, view.w2c, view.K)
    ok = lifting.in_image(u, v, z, hw)
    colors = torch.zeros(len(points), 3, dtype=torch.uint8)
    colors[ok] = image[v[ok].long(), u[ok].long()]  # vertex colours that agree with the photo
    keep = torch.ones(len(points), dtype=torch.bool)
    view.zbuf = lifting.zbuffer(points, view)
    good = lifting.colour_agreement(points, colors, keep, view, image)
    wrong = make_view(hw)
    wrong.w2c[:3, :3] = torch.tensor([[-1.0, 0, 0], [0, -1, 0], [0, 0, 1]])  # rotated 180 degrees about the axis
    wrong.zbuf = lifting.zbuffer(points, wrong)
    bad = lifting.colour_agreement(points, colors, keep, wrong, image)
    assert good["r"] > 0.99 and abs(bad["r"]) < 0.2


def test_pca3_finds_the_main_directions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 6)) * [10, 5, 2, 0.1, 0.1, 0.1]
    mean, basis = lifting.pca3(torch.tensor(x, dtype=torch.float32))
    assert basis.shape == (6, 3)
    np.testing.assert_allclose(np.abs(basis.numpy()[:3]), np.eye(3), atol=0.05)
