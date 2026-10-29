"""The frozen map's pieces and its cache on synthetic data, and find_goal on the tiny maze
(conftest.tiny_maze, skipped without the simulator)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

import frozen_map  # noqa: E402
import mapping  # noqa: E402
from conftest import SEED  # noqa: E402
from geometry import CAMERA, FloorMask, OccupancyMap  # noqa: E402
from goal import find_goal  # noqa: E402
from map_agent import MapAgent  # noqa: E402
from my_agent import Float32VLAD  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402


def test_vlad_matches_the_extractor(tmp_path):
    rng = np.random.default_rng(0)
    des = np.sqrt(rng.dirichlet(np.ones(128), 300)).astype(np.float32)
    extractor = Float32VLAD(n_clusters=8, cache_dir=tmp_path)
    extractor.codebook = KMeans(8, n_init=1, random_state=0).fit(des)
    expected = extractor._vlad(des)
    got = frozen_map.vlad(des, extractor.codebook.cluster_centers_.astype(np.float32))
    assert got == pytest.approx(expected, abs=1e-4)


def test_vlad_without_descriptors_is_zero():
    assert not frozen_map.vlad(None, np.zeros((4, 128), np.float32)).any()


def test_every_frame_poses_compose_from_the_nearest_earlier_node():
    n = mapping.NODE_EVERY
    dead = np.array([[0.03 * i, 0.0, 0.0] for i in range(2 * n + 1)])
    optimised = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, np.pi / 2], [5.0, 5.0, 0.0]])
    variant = mapping.Variant([optimised], None)
    result = mapping.MapResult({"graph": variant}, [], 0, dead_reckoned=[dead])
    poses = frozen_map.every_frame_poses(result)[0]
    assert poses[0] == pytest.approx(optimised[0])
    assert poses[n] == pytest.approx(optimised[1])
    # Two ticks past the second node: 0.06 m forward along its yaw (+y).
    assert poses[n + 2] == pytest.approx([1.0, 2.06, np.pi / 2], abs=1e-9)


def test_a_frozen_map_loads_back_as_it_was_saved(tmp_path, monkeypatch, capsys):
    n = mapping.NODE_EVERY
    rng = np.random.default_rng(1)
    dead = [np.cumsum(rng.normal(0, 0.01, (2 * n + 1, 3)), axis=0) for _ in range(2)]
    grid = OccupancyMap(half_extent=1.0)
    grid.add_scan((0.0, 0.0, 0.0), np.array([[0.5, 0.0], [0.5, 0.1]]))
    centers = rng.normal(size=(4, 128)).astype(np.float32)
    result = mapping.MapResult(
        {"graph": mapping.Variant([d[::n] + 0.1 for d in dead], grid)},
        [],
        0,
        dead_reckoned=dead,
        codebook=centers,
    )
    monkeypatch.setattr(frozen_map.mapping, "build", lambda *args, **kwargs: result)
    noise = [
        cv2.GaussianBlur(rng.integers(0, 255, (240, 320, 3), np.uint8), (5, 5), 0)
        for _ in range(2 * n + 1)
    ]
    routes = [mapping.Route(noise, [None] * len(noise)) for _ in dead]
    floor, path = FloorMask(), tmp_path / frozen_map.MAP_NAME
    frozen_map.freeze(tmp_path, routes, dict(CAMERA), floor, path)
    store = frozen_map.load(path, routes, floor)

    for got, expected in zip(store.poses, frozen_map.every_frame_poses(result), strict=True):
        np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(store.grid.log_odds, grid.log_odds)
    np.testing.assert_array_equal(store.grid.origin, grid.origin)
    assert store.grid.res == grid.res
    np.testing.assert_array_equal(store.centers, centers)
    assert store.vlad.shape == (2 * len(noise), 4 * 128)
    assert store.sift.camera.cy == CAMERA["cy"] + mapping.HORIZON_SHIFT_PX
    assert "WARNING" not in capsys.readouterr().out

    # The SIFT sidecar: a view computed once, flushed, and read back by the next load.
    view = store.view(3)
    assert view.des is not None and len(view.des)
    store.sift.flush()
    again = frozen_map.load(path, routes, floor).sift
    assert list(again.cache) == [(0, 3)]
    np.testing.assert_array_equal(again.view(0, 3).pixels, view.pixels)
    np.testing.assert_allclose(again.view(0, 3).des, view.des, atol=1e-3)  # kept as float16


@pytest.fixture(scope="module")
def tiny_agent(tiny_maze) -> MapAgent:
    return MapAgent(tiny_maze)


def _spawn_views(textures) -> list[np.ndarray]:
    """The four goal views of a goal at maze SEED's spawn, facing its first heading."""
    import vis_nav_sim as sim

    world = sim.Simulator(textures, SEED)
    views = [world.initial_frame()]
    for ticks in (26, 26, 27):  # about 90 degrees each
        views.append(world.step(sim.LEFT, ticks)[-1])
    return views


def test_find_goal_places_a_goal_at_a_mapped_place(tiny_agent, textures):
    a = tiny_agent
    goal = find_goal(_spawn_views(textures), a.store, a.camera, a.floor, photographed=False)
    assert goal is not None and (goal.frame, goal.k) == (0, 0)
    assert (goal.x, goal.y) == pytest.approx((0.0, 0.0), abs=0.01)
    assert goal.yaw == pytest.approx(0.0, abs=np.radians(1.0))
    assert goal.layout >= 0.6 and goal.view_agreement >= 1


def test_find_goal_without_features_finds_nothing(tiny_agent):
    a = tiny_agent
    blank = [np.full((240, 320, 3), 128, np.uint8)] * 4
    assert find_goal(blank, a.store, a.camera, a.floor) is None
