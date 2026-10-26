"""Pilot step 2 end to end on CPU: a synthetic room whose photos are rendered from known COLMAP poses and stored the
way the toolbox stores them (nerfstudio transform_matrix), packed like a Kaggle-unpacked core scene; a stand-in for
DINOv2. The pose check must pick the toolbox inverse, and the visual checks must be written."""

import io
import json
import sys
import types

import numpy as np
import pytest

from conftest import ROOT, load_script

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "pilot_concat"))
step2 = load_script("experiments/pilot_concat/step2_lift.py")
HW, F = (117, 176), 100.0


def room(spacing=0.02):
    """Floor and four walls of a 4 x 4 x 2.5 m room, with a smooth colour pattern."""
    a, h = np.arange(0, 4, spacing), np.arange(0, 2.5, spacing)
    faces = [np.stack(np.meshgrid(a, a, indexing="ij"), -1).reshape(-1, 2)]
    faces = [np.c_[faces[0], np.zeros(len(faces[0]))]]
    for fixed, axis in ((0.0, 0), (4.0, 0), (0.0, 1), (4.0, 1)):
        s, z = (g.ravel() for g in np.meshgrid(a, h, indexing="ij"))
        xyz = np.zeros((len(s), 3))
        xyz[:, axis], xyz[:, 1 - axis], xyz[:, 2] = fixed, s, z
        faces.append(xyz)
    coord = np.concatenate(faces).astype(np.float32)
    color = (127 + 120 * np.sin(np.c_[coord[:, 0] * 3, coord[:, 1] * 2 + coord[:, 2] * 4, coord[:, 2] * 5 + 1])).astype(np.uint8)
    return coord, color


def room_normals(coord):
    """Normals pointing into the room: up on the floor, inwards on the walls."""
    n = np.zeros_like(coord)
    n[coord[:, 2] < 1e-6, 2] = 1
    walls = coord[:, 2] >= 1e-6
    for axis in (0, 1):
        n[walls & (coord[:, axis] < 1e-6), axis] = 1
        n[walls & (coord[:, axis] > 4 - 1e-6), axis] = -1
    return n


def w2c_looking(yaw, centre=(2.0, 2.0, 1.25)):
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    R = np.stack([np.cross(down, fwd), down, fwd])  # rows: camera x (right), y (down), z (forward)
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = R, -R @ np.array(centre)
    return w2c


def toolbox_transform(w2c):
    """What the toolbox's common/utils/nerfstudio.py convert_frames writes for a COLMAP pose."""
    c2w = np.linalg.inv(w2c)
    c2w[0:3, 1:3] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[2, :] *= -1
    return c2w.tolist()


def photo(coord, color, w2c):
    K = np.array([[F, 0, HW[1] / 2], [0, F, HW[0] / 2], [0, 0, 1]])
    cam = coord @ w2c[:3, :3].T + w2c[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):  # points in the camera's plane; render() drops them
        u, v = (K[:2, :2] @ (cam[:, :2] / cam[:, 2:]).T) + K[:2, 2:3]
    return step2.render(u, v, cam[:, 2], color, HW, scale=1.0)


def png(array):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG" if array.ndim == 2 else "JPEG", quality=95)
    return buf.getvalue()


def room_labels(coord):
    """Floor 0, the four walls 1-4, unlabelled above 2.3 m."""
    label = np.zeros(len(coord), np.int16)
    walls = coord[:, 2] >= 1e-6
    for i, (axis, value) in enumerate(((0, 0.0), (0, 4.0), (1, 0.0), (1, 4.0)), 1):
        label[walls & (np.abs(coord[:, axis] - value) < 1e-6)] = i
    label[coord[:, 2] > 2.3] = -1
    return label


def write_scene(root, scene, split="val", anonymised=()):
    coord, color = room()
    d = root / "scenes" / scene
    pc = d / "pointcept" / split / scene
    pc.mkdir(parents=True)
    n = len(coord)
    for key, arr in dict(coord=coord, color=color, normal=room_normals(coord),
                         segment=np.c_[room_labels(coord), -np.ones((n, 2))].astype(np.int16),
                         instance=np.full((n, 3), -1, np.int16)).items():
        np.save(pc / f"{key}.npy", arr)
    (d / "scans").mkdir()
    (d / "scans" / "mesh_aligned_0.05_mask.txt").write_text("\n".join(map(str, anonymised)))
    for sub in ("nerfstudio", "resized_undistorted_images", "resized_undistorted_masks"):
        (d / "dslr" / sub).mkdir(parents=True)
    frames, names = [], []
    for i, yaw in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
        w2c = w2c_looking(yaw)
        name = f"DSC{i:05d}.JPG"
        (d / "dslr" / "resized_undistorted_images" / name).write_bytes(png(photo(coord, color, w2c)))
        (d / "dslr" / "resized_undistorted_masks" / name.replace(".JPG", ".png")).write_bytes(png(np.full(HW, 255, np.uint8)))
        frames.append(dict(file_path=name, mask_path=name.replace(".JPG", ".png"), transform_matrix=toolbox_transform(w2c), is_bad=False))
        names.append(name)
    tf = dict(camera_model="PINHOLE", fl_x=F, fl_y=F, cx=HW[1] / 2, cy=HW[0] / 2, w=HW[1], h=HW[0], frames=frames, test_frames=[])
    (d / "dslr" / "nerfstudio" / "transforms_undistorted.json").write_text(json.dumps(tf))
    (d / "dslr" / "kept_views.json").write_text(json.dumps(dict(views=names)))
    manifest = root / "manifest.json"
    rows = json.loads(manifest.read_text()) if manifest.exists() else {}
    manifest.write_text(json.dumps({**rows, scene: dict(n_vertices=n)}))
    return d


class FakeDino(torch.nn.Module):
    """Patch features that depend on the image content, shaped like a ViT's output (CLS + patch tokens)."""

    def forward(self, pixel_values):
        x = torch.nn.functional.avg_pool2d(pixel_values, 14)
        tokens = torch.cat([x, x**2, torch.sin(3 * x)], 1).flatten(2).transpose(1, 2)
        return types.SimpleNamespace(last_hidden_state=torch.cat([tokens.mean(1, keepdim=True), tokens], 1))


def test_run_scene_on_a_synthetic_room(tmp_path):
    scene = json.loads((ROOT / "configs" / "scannetpp_subset.json").read_text())["val"][0]
    path = write_scene(tmp_path / "input" / "scannetpp_core", scene)
    args = types.SimpleNamespace(views=30, batch=4, chunk=40_000, out=tmp_path / "out", occlusion="v1",
                                 consistency_points=5_000, tmp=tmp_path / "tmp")
    args.out.mkdir()
    out = step2.run_scene(FakeDino(), path, scene, args, "cpu", list(step2.CONVENTIONS))
    # views facing a wall also see the side walls at grazing angles, where several points of one wall fall within the
    # depth tolerance of a pixel; diagonal views reach r = 0.999. Wrong conventions barely see the room at all.
    scores = {c: a["r_mean"] for c, a in out["pose_agreement"].items()}
    # the wrong cameras see few points; with the room's smooth colours their r can reach ~0.5 by chance
    assert out["convention"] == step2.EXPECTED and scores[step2.EXPECTED] > 0.7, scores
    assert all(r is None or r < scores[step2.EXPECTED] - 0.2 for c, r in scores.items() if c != step2.EXPECTED), scores
    seen = {c: a["visible_frac"] for c, a in out["pose_agreement"].items()}
    assert all(f < seen[step2.EXPECTED] / 10 for c, f in seen.items() if c != step2.EXPECTED), seen
    assert out["n_views"] == 8 and out["seen"] > 0.5 and out["views_per_seen"]["mean"] >= 1
    assert out["seen"] > out["seen_v0"] + 0.05  # the slope term stops walls seen at a slant from hiding themselves
    assert out["consistency"]["n"] > 0 and set(out["consistency"]) >= {"cos", "baseline", "axis_deg", "incidence_deg"}
    assert out["mesh"].startswith("skipped")  # no open3d locally; Kaggle runs the mesh check
    for name in out["images"]:
        with Image.open(args.out / name) as im:
            assert im.size[0] > 100 and im.size[1] > 100
