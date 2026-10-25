"""Token-safe ScanNet++ client: paths as the official script resolves them, subset, sizes, redaction."""

import io
import urllib.error

import pytest

from appearance_pt import scannetpp_download as dl

TOKEN = "tok-abc123"  # fake
CFG = {
    "root_url": "https://example.org/download?version=v1&token=TOKEN&file=FILEPATH",
    "zipped_assets": ["scan_mesh_path", "dslr_resized_undistorted_dir"],
    "exclude_assets": {"sem_test": ["scan_anno_json_path"]},
}


class FakeResponse(io.BytesIO):
    def __init__(self, body=b"", headers=None, status=200):
        super().__init__(body)
        self.headers, self.status = headers or {}, status


def http_error(url, code):
    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(f"server says {url}".encode()))


def test_scene_files_match_official_layout():
    files = dl.scene_files(CFG, "abc123", "nvs_sem_val",
                           ["scan_mesh_path", "scan_mesh_mask_path", "dslr_resized_undistorted_dir"])
    by_asset = {f["asset"]: f for f in files}
    assert by_asset["scan_mesh_path"]["remote"] == "data/abc123/scans/mesh_aligned_0.05.zip"
    assert by_asset["scan_mesh_path"]["local"] == "data/abc123/scans/mesh_aligned_0.05.ply"
    assert by_asset["scan_mesh_mask_path"]["remote"] == "data/abc123/scans/mesh_aligned_0.05_mask.txt"
    assert by_asset["dslr_resized_undistorted_dir"]["remote"] == "data/abc123/dslr/resized_undistorted_images.zip"


def test_scene_files_skip_assets_excluded_for_the_split():
    assets = ["scan_mesh_path", "scan_anno_json_path"]
    assert len(dl.scene_files(CFG, "s", "sem_test", assets)) == 1
    assert len(dl.scene_files(CFG, "s", "nvs_sem_train", assets)) == 2


def test_choose_subset_is_fixed_and_sorted():
    train = [f"t{i:03d}" for i in range(300)]
    a = dl.choose_subset(train, ["v2", "v1"], n_train=200, seed=0)
    b = dl.choose_subset(list(reversed(train)), ["v1", "v2"], n_train=200, seed=0)
    assert a == b
    assert a["train"] == sorted(a["train"]) and len(set(a["train"])) == 200 and set(a["train"]) <= set(train)
    assert a["val"] == ["v1", "v2"]
    assert dl.choose_subset(train, [], 200, seed=1)["train"] != a["train"]


def test_client_refuses_placeholder_token():
    with pytest.raises(ValueError):
        dl.ScannetppClient(CFG, dl.PLACEHOLDER_TOKEN)


def test_size_from_head_and_missing_file(monkeypatch):
    client = dl.ScannetppClient(CFG, TOKEN, pause=0)

    def urlopen(request, timeout):
        assert request.get_method() == "HEAD"
        if "missing" in request.full_url:
            raise http_error(request.full_url, 404)
        return FakeResponse(headers={"Content-Length": "1234"})

    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    assert client.size("data/s/scans/mesh.zip") == 1234
    assert client.size("data/missing.zip") is None


def test_size_falls_back_to_ranged_get(monkeypatch):
    client = dl.ScannetppClient(CFG, TOKEN, pause=0)

    def urlopen(request, timeout):
        if request.get_method() == "HEAD":
            return FakeResponse()
        assert request.headers["Range"] == "bytes=0-0"
        return FakeResponse(b"x", {"Content-Range": "bytes 0-0/987654"}, status=206)

    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    assert client.size("f") == 987654


def test_errors_never_contain_the_token(monkeypatch):
    client = dl.ScannetppClient(CFG, TOKEN, pause=0)
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    calls = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        raise http_error(request.full_url, 401 if "auth" in request.full_url else 503)

    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    for remote, code, tries in (("auth", 401, 1), ("busy", 503, dl.RETRIES)):
        calls.clear()
        with pytest.raises(dl.RemoteError) as e:
            client.size(remote)
        assert e.value.code == code and len(calls) == tries
        assert TOKEN in calls[0]  # the request itself carries it ...
        assert TOKEN not in str(e.value) and "***" in str(e.value)  # ... the error does not
        assert e.value.__cause__ is None and e.value.__suppress_context__


def test_fetch_unpacks_zips_next_to_them(monkeypatch, tmp_path):
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("resized_undistorted_images/DSC0001.JPG", b"jpegbytes")
    payload = buf.getvalue()
    client = dl.ScannetppClient(CFG, TOKEN, pause=0)
    monkeypatch.setattr(dl.urllib.request, "urlopen", lambda request, timeout: FakeResponse(payload))
    entry = dl.scene_files(CFG, "s1", "nvs_sem_val", ["dslr_resized_undistorted_dir"])[0]
    stats = client.fetch(entry, tmp_path)
    assert (tmp_path / "data/s1/dslr/resized_undistorted_images/DSC0001.JPG").read_bytes() == b"jpegbytes"
    assert not (tmp_path / "data/s1/dslr/resized_undistorted_images.zip").exists()
    assert stats["download_bytes"] == len(payload) and stats["unpacked_bytes"] == 9


def test_download_writes_via_part_file(monkeypatch, tmp_path):
    client = dl.ScannetppClient(CFG, TOKEN, pause=0)
    monkeypatch.setattr(dl.urllib.request, "urlopen", lambda request, timeout: FakeResponse(b"payload"))
    target = tmp_path / "meta" / "splits" / "nvs_sem_val.txt"
    assert client.download("splits/nvs_sem_val.txt", target) == 7
    assert target.read_bytes() == b"payload" and not list(tmp_path.rglob("*.part"))
