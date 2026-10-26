"""Packed scenes: finding them below /kaggle/input, reading tars or unpacked copies, and the per-scene checks."""

import gzip
import io
import json
import tarfile

import numpy as np
import pytest
from PIL import Image

from appearance_pt import scannetpp_reduce as red
from appearance_pt import scannetpp_store as st
from conftest import load_script

N = 500


def image_bytes(array, fmt):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format=fmt)
    return buf.getvalue()


def core_tar(tmp_path, scene, split, corrupt_view=False):
    """A core archive written by the real pack_scene, from a raw scene with real arrays and images."""
    raw = tmp_path / "raw" / scene
    (raw / "scans").mkdir(parents=True)
    (raw / "scans" / "mesh_aligned_0.05_mask.txt").write_text("3\n4\n")
    (raw / "scans" / "mesh_aligned_0.05.ply").write_bytes(f"ply\nelement vertex {N}\nend_header\n".encode())
    (raw / "scans" / "segments.json").write_text(json.dumps(dict(segIndices=[0] * N)))
    (raw / "scans" / "segments_anno.json").write_text(json.dumps(dict(segGroups=[])))
    dslr = raw / "dslr"
    for d in ("nerfstudio", "resized_undistorted_images", "resized_undistorted_masks"):
        (dslr / d).mkdir(parents=True)
    frames = [dict(file_path=f"DSC{i:05d}.JPG", mask_path=f"DSC{i:05d}.png", is_bad=i == 1) for i in range(4)]
    (dslr / "nerfstudio" / "transforms_undistorted.json").write_text(json.dumps(dict(w=24, h=16, frames=frames)))
    (dslr / "train_test_lists.json").write_text("{}")
    for i, f in enumerate(frames):
        jpg = image_bytes(np.full((16, 24, 3), 90, np.uint8), "JPEG")
        (dslr / "resized_undistorted_images" / f["file_path"]).write_bytes(jpg[:len(jpg) // 2] if corrupt_view and i == 2 else jpg)
        (dslr / "resized_undistorted_masks" / f["mask_path"]).write_bytes(image_bytes(np.full((16, 24), 255, np.uint8), "PNG"))
    processed = tmp_path / "processed" / scene
    processed.mkdir(parents=True)
    rng = np.random.default_rng(0)
    color = rng.integers(0, 255, (N, 3), dtype=np.uint8)
    color[[3, 4]] = st.MAGENTA
    arrays = dict(coord=rng.uniform(0, 1, (N, 3)).astype(np.float32), color=color,
                  normal=rng.normal(size=(N, 3)).astype(np.float32),
                  segment=rng.integers(-1, 100, (N, 3)).astype(np.int16), instance=np.full((N, 3), -1, np.int16))
    for k, a in arrays.items():
        np.save(processed / f"{k}.npy", a)
    archive = tmp_path / "out" / "scenes" / f"{scene}.tar"
    archive.parent.mkdir(parents=True, exist_ok=True)
    red.pack_scene(raw, processed, split, archive, n_views=60)
    return archive


def iphone_tar(path, pool=(10, 20, 30)):
    """An iPhone archive with the layout of scannetpp_iphone.reduce_scene."""
    path.parent.mkdir(parents=True, exist_ok=True)
    members = {}
    for f in pool:
        members[f"iphone/rgb/frame_{f:06d}.jpg"] = image_bytes(np.full((12, 16, 3), 50, np.uint8), "JPEG")
        members[f"iphone/rgb_masks/frame_{f:06d}.png"] = image_bytes(np.full((12, 16), 255, np.uint8), "PNG")
        members[f"iphone/depth/frame_{f:06d}.png"] = image_bytes(np.full((192, 256), 1500, np.uint16), "PNG")
    poses = {f"frame_{f:06d}": dict(aligned_pose=np.eye(4).tolist(), intrinsic=np.eye(3).tolist()) for f in range(40)}
    members["iphone/pose_intrinsic_imu.json.gz"] = gzip.compress(json.dumps(poses).encode())
    members["iphone/colmap/cameras.txt.gz"] = gzip.compress(b"# Camera list\n1 OPENCV 16 12 1 1 8 6 0 0 0 0\n")
    images = "".join(f"{i} 1 0 0 0 0 0 0 1 frame_{f:06d}.jpg\n\n" for i, f in enumerate(range(0, 40, 10)))
    members["iphone/colmap/images.txt.gz"] = gzip.compress(images.encode())
    members["iphone/pool.json"] = json.dumps(dict(pool=list(pool), n_video_frames=40, depth_encoding="per frame: lz4 uint16 mm")).encode()
    with tarfile.open(path, "w") as tar:
        for name, data in members.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    return path


@pytest.mark.parametrize("split", ["val", "train"])
def test_check_core_reads_every_file(tmp_path, split):
    archive = core_tar(tmp_path, "abc", split)
    with st.SceneArchive(archive) as arc:
        stats = st.check_core(arc, "abc", split)
        assert arc.bytes_read == sum(m.size for m in arc._members.values())  # every file was read
    assert stats["n_vertices"] == N and stats["n_views"] == 3 and stats["n_masks"] == 3
    assert stats["n_anonymised"] == 2 and stats["n_anonymised_magenta"] == 2 and stats["n_magenta"] >= 2
    assert stats["image_hw"] == ["16x24"] and stats["transforms_hw"] == "16x24"
    assert 0 < stats["n_grid_2cm"] <= N


def test_check_core_rejects_damage(tmp_path):
    with st.SceneArchive(core_tar(tmp_path / "a", "abc", "val", corrupt_view=True)) as arc, pytest.raises(OSError):
        st.check_core(arc, "abc", "val")  # half a JPEG: PIL refuses to decode it
    archive = core_tar(tmp_path / "b", "abc", "val")
    with tarfile.open(archive) as tar:
        last = tar.getmembers()[-1]
    archive.write_bytes(archive.read_bytes()[:last.offset_data + last.size // 2])  # a copy cut off inside the last file
    with pytest.raises((st.SceneError, tarfile.ReadError)):  # tarfile itself may notice first
        st.SceneArchive(archive)
    with st.SceneArchive(core_tar(tmp_path / "c", "abc", "val")) as arc, pytest.raises(st.SceneError, match="missing"):
        st.check_core(arc, "abc", "train")  # arrays are under pointcept/val/


def test_check_iphone(tmp_path):
    with st.SceneArchive(iphone_tar(tmp_path / "x.tar")) as arc:
        stats = st.check_iphone(arc, "x", "val")
    assert stats["n_pool"] == 3 and stats["n_registered"] == 4 and stats["camera_models"] == ["OPENCV"]
    assert stats["gz_as_packed"] == 3
    assert stats["depth_median_mm"] == 1500 and stats["depth_zero_max"] == 0 and stats["mask_on"] == 1
    with st.SceneArchive(iphone_tar(tmp_path / "y.tar", pool=(10, 15))) as arc, pytest.raises(st.SceneError, match="COLMAP"):
        st.check_iphone(arc, "y", "val")  # frame 15 is not registered


def test_find_scenes_tars_and_unpacked_copies(tmp_path):
    core = core_tar(tmp_path / "work", "abc", "val")
    root = tmp_path / "input"
    (root / "core-val" / "scannetpp_core" / "scenes").mkdir(parents=True)
    core.rename(root / "core-val" / "scannetpp_core" / "scenes" / "abc.tar")
    iphone_tar(root / "iphone-val" / "scenes" / "abc.tar")
    for name, src in (("def", root / "core-val" / "scannetpp_core" / "scenes" / "abc.tar"),
                      ("ghi.tar", root / "iphone-val" / "scenes" / "abc.tar")):  # as if Kaggle had unpacked them
        with tarfile.open(src) as tar:
            tar.extractall(root / "unpacked" / "scenes" / name, filter="data")
    for gz in (root / "unpacked").rglob("*.gz"):  # Kaggle also decompresses .gz files when it unpacks
        gz.with_suffix("").write_bytes(gzip.decompress(gz.read_bytes()))
        gz.unlink()
    found = st.find_scenes(root)
    assert {k: sorted(v) for k, v in found.items()} == {"core": ["abc", "def"], "iphone": ["abc", "ghi"]}
    with st.SceneArchive(found["core"]["abc"][0]) as a, st.SceneArchive(found["core"]["def"][0]) as b:
        assert a.names() == b.names() and a.read("dslr/kept_views.json") == b.read("dslr/kept_views.json")
    with st.SceneArchive(found["iphone"]["ghi"][0]) as arc:
        stats = st.check_iphone(arc, "ghi", "val")
    assert stats["n_pool"] == 3 and stats["gz_as_packed"] == 0  # read from the decompressed copies


def test_verify_compares_with_the_download_manifest(tmp_path):
    verify = load_script("scripts/scannetpp/scannetpp_verify.py")
    archive = core_tar(tmp_path, "abc", "val")
    good = dict(n_vertices=N, n_views=3, tar_mb=round(archive.stat().st_size / 1e6, 1))
    assert verify.check("core", "abc", "val", archive, good)["status"] == "ok"
    row = verify.check("core", "abc", "val", archive, {**good, "n_vertices": N + 1})
    assert row["status"].startswith("FAIL") and "n_vertices" in row["status"]
