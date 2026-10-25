"""Pilot (experiments/pilot_concat): helpers that run without a GPU."""

import itertools

import numpy as np

from conftest import load_script

scenes = load_script("experiments/pilot_concat/scenes.py")


def fake_room(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "coord": rng.uniform(0, [4, 3, 2.5], (n, 3)).astype(np.float32),
        "color": rng.integers(0, 256, (n, 3), dtype=np.uint8),
        "normal": rng.normal(size=(n, 3)).astype(np.float32),
    }


def test_tile_scene_makes_separate_copies():
    room, n, gap = fake_room(), 500, 0.5
    tiled = scenes.tile_scene(room, 3, gap=gap)
    assert all(len(tiled[key]) == 9 * n for key in scenes.KEYS)
    assert tiled["coord"].dtype == np.float32 and tiled["color"].dtype == np.uint8
    copies = [tiled["coord"][c * n:(c + 1) * n] for c in range(9)]
    for a, b in itertools.combinations(copies, 2):  # bounding boxes at least `gap` apart in x or y
        apart = [max(b.min(0)[d] - a.max(0)[d], a.min(0)[d] - b.max(0)[d]) for d in (0, 1)]
        assert max(apart) >= gap - 1e-4
    for c in range(9):
        np.testing.assert_allclose(copies[c] - copies[c].min(0), room["coord"] - room["coord"].min(0), atol=1e-5)
        np.testing.assert_array_equal(tiled["color"][c * n:(c + 1) * n], room["color"])
        np.testing.assert_array_equal(tiled["normal"][c * n:(c + 1) * n], room["normal"])


def test_load_scene_dir(tmp_path):
    room = fake_room()
    for key in scenes.KEYS:
        np.save(tmp_path / f"{key}.npy", room[key])
    loaded = scenes.load_scene_dir(tmp_path)
    for key in scenes.KEYS:
        np.testing.assert_array_equal(loaded[key], room[key])
