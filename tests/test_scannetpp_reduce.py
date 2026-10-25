"""Reducing a downloaded ScanNet++ scene: view pool selection and the packed archive."""

import json
import tarfile

import numpy as np

from appearance_pt import scannetpp_reduce as red


def frames(n, bad=()):
    return [dict(file_path=f"DSC{i:05d}.JPG", mask_path=f"DSC{i:05d}.png", is_bad=i in bad,
                 transform_matrix=np.eye(4).tolist()) for i in range(n)]


def test_select_views_spread_and_skip_bad():
    tf = dict(frames=frames(200, bad={0, 1, 2, 199}), test_frames=[])
    views = red.select_views(tf, 60)
    names = [v["file_path"] for v in views]
    assert len(views) == 60 and names == sorted(names) and len(set(names)) == 60
    assert not any(v["is_bad"] for v in views)
    assert names[0] == "DSC00003.JPG" and names[-1] == "DSC00198.JPG"  # first and last good frames
    gaps = np.diff([int(n[3:8]) for n in names])
    assert gaps.max() - gaps.min() <= 1  # evenly spread over the capture order


def test_select_views_includes_test_frames_and_small_scenes():
    tf = dict(frames=frames(10, bad={4}), test_frames=[dict(file_path="DSC00099.JPG", mask_path="m.png")])
    names = [v["file_path"] for v in red.select_views(tf, 60)]
    assert len(names) == 10 and "DSC00004.JPG" not in names and names[-1] == "DSC00099.JPG"


def test_pack_scene(tmp_path):
    scene = tmp_path / "raw" / "abc"
    (scene / "scans").mkdir(parents=True)
    for name in ("mesh_aligned_0.05_mask.txt", *red.VAL_RAW):
        (scene / "scans" / name).write_text("x")
    dslr = scene / "dslr"
    for d in ("nerfstudio", "resized_undistorted_images", "resized_undistorted_masks"):
        (dslr / d).mkdir(parents=True)
    tf = dict(fl_x=1.0, frames=frames(5, bad={1}), test_frames=[])
    (dslr / "nerfstudio" / "transforms_undistorted.json").write_text(json.dumps(tf))
    (dslr / "train_test_lists.json").write_text("{}")
    for f in tf["frames"]:
        (dslr / "resized_undistorted_images" / f["file_path"]).write_bytes(b"jpg")
        (dslr / "resized_undistorted_masks" / f["mask_path"]).write_bytes(b"png")
    processed = tmp_path / "processed"
    processed.mkdir()
    np.save(processed / "coord.npy", np.zeros((3, 3), np.float32))

    for split, has_raw in (("val", True), ("train", False)):
        archive = tmp_path / f"{split}.tar"
        kept = red.pack_scene(scene, processed, split, archive, n_views=3)
        assert kept["n_frames"] == 5 and kept["n_bad"] == 1 and len(kept["views"]) == 3
        with tarfile.open(archive) as tar:
            names = set(tar.getnames())
            assert f"pointcept/{split}/abc/coord.npy" in names and "scans/mesh_aligned_0.05_mask.txt" in names
            assert ("scans/mesh_aligned_0.05.ply" in names) == has_raw
            assert sum(n.startswith("dslr/resized_undistorted_images/") for n in names) == 3
            assert "dslr/resized_undistorted_images/DSC00001.JPG" not in names  # is_bad
            assert json.load(tar.extractfile("dslr/kept_views.json"))["views"] == kept["views"]
        assert not archive.with_name(archive.name + ".part").exists()
