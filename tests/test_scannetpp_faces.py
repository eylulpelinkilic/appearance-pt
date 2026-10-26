"""scannetpp_faces.py: the size survey with a fake server, and the vertex fingerprint (the mesh part needs open3d)."""

import numpy as np

from conftest import load_script

faces = load_script("scripts/scannetpp/scannetpp_faces.py")


class FakeClient:
    """Laser scans 2 GB, full-resolution DSLR 3 GB, panocam missing for val scenes."""

    def size(self, remote):
        if "pano" in remote:
            return None if "val" in remote else 1_000
        return 2_000_000_000 if "scans" in remote else 3_000_000_000

    def redact(self, text):
        return str(text)


def test_size_survey_groups_and_extrapolates(monkeypatch):
    subset = {"train": ["t1", "t2"], "val": ["v1"]}
    cfg = {"exclude_assets": {}, "zipped_assets": ["scan_pc_path", "dslr_original_dir"]}
    monkeypatch.setattr(faces, "scene_files", lambda cfg, scene, split, assets: [
        dict(asset=a, zipped=a in cfg["zipped_assets"], remote=f"{split}/{scene}/{'scans' if a.startswith('scan') else a}")
        for a in assets])
    rows = faces.measure_sizes(FakeClient(), cfg, subset, workers=2)
    assert len(rows) == 3 * (3 + 2 + 10)
    table = faces.size_table(rows)
    laser = table["laser_scan/train"]
    assert laser["files"] == 6 and laser["mean_gb"] == 6.0 and laser["full_split_gb"] == 6.0 * 856
    assert table["panocam/val"]["missing"] == 10 and table["panocam/val"]["scenes_with_data"] == 0


def test_vertex_fingerprint_matches_pointcepts_float32_coord():
    coord64 = np.random.default_rng(0).normal(size=(100, 3))  # open3d vertices are float64
    assert faces.vertex_sha1(coord64) == faces.vertex_sha1(coord64.astype(np.float32))
    assert faces.vertex_sha1(coord64) != faces.vertex_sha1(coord64[::-1])
