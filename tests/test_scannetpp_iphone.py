"""iPhone helpers: frame pool selection and depth.bin decoding (both encodings the toolbox handles)."""

import zlib

import numpy as np
import pytest

from appearance_pt import scannetpp_iphone as ip

H, W = ip.DEPTH_HW


def raw_deflate(data: bytes) -> bytes:  # zlib.decompress(wbits=-MAX_WBITS) expects a raw deflate stream
    c = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return c.compress(data) + c.flush()


def test_select_pool_spreads_over_registered_frames():
    registered = list(range(0, 10_900, 10))  # COLMAP registers every 10th video frame
    pool = ip.select_pool(registered, 60)
    assert len(pool) == 60 and pool == sorted(pool) and set(pool) <= set(registered)
    assert pool[0] == 0 and pool[-1] == registered[-1]
    assert ip.select_pool([30, 10, 20], 60) == [10, 20, 30]


def test_read_depth_whole_file_zlib(tmp_path):
    depth_m = np.random.default_rng(0).uniform(0.5, 4, (5, H, W)).astype(np.float32)
    (tmp_path / "depth.bin").write_bytes(raw_deflate(depth_m.tobytes()))
    encoding, n, out = ip.read_depth(tmp_path / "depth.bin", {1, 3})
    assert encoding.startswith("zlib float32") and n == 5 and sorted(out) == [1, 3]
    np.testing.assert_array_equal(out[3], (depth_m[3] * 1000).astype(np.uint16))


def test_read_depth_per_frame_zlib_records(tmp_path):
    frames = [np.full((H, W), 1.0 + i, np.float32) for i in range(4)]
    records = b"".join(len(c).to_bytes(4, "little") + c for c in (raw_deflate(f.tobytes()) for f in frames))
    (tmp_path / "depth.bin").write_bytes(records)
    encoding, n, out = ip.read_depth(tmp_path / "depth.bin", {2})
    assert n == 4 and list(out) == [2] and "zlib float32" in encoding
    assert out[2].dtype == np.uint16 and (out[2] == 3000).all()


def test_read_depth_per_frame_lz4(tmp_path):
    lz4_block = pytest.importorskip("lz4.block")
    frames = [np.full((H, W), 1000 + i, np.uint16) for i in range(3)]
    chunks = [lz4_block.compress(f.tobytes(), store_size=False) for f in frames]
    (tmp_path / "depth.bin").write_bytes(b"".join(len(c).to_bytes(4, "little") + c for c in chunks))
    encoding, n, out = ip.read_depth(tmp_path / "depth.bin", {0, 2})
    assert encoding == "per frame: lz4 uint16 mm" and n == 3 and (out[2] == 1002).all()
