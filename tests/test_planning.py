"""The planner and localiser on synthetic data: no simulator, no recorded frames."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "eval"))

from eval_planning import segment_hits_rects  # noqa: E402
from frozen_map import FrozenMap  # noqa: E402
from geometry import Camera, FloorMask, OccupancyMap, layout_score  # noqa: E402
from localiser import Localiser  # noqa: E402
from planner import NODE_EVERY, PlaceGraph, bresenham, crosses_occupied  # noqa: E402
from planner import Path as Route  # noqa: E402

CAMERA = {"width": 64, "height": 48, "fx": 30.0, "fy": 30.0, "cx": 32.0, "cy": 24.0}


def line(start, end, nodes: int, yaw: float = 0.0) -> np.ndarray:
    """A route of ``nodes`` nodes (NODE_EVERY frames apart) from ``start`` to ``end``."""
    t = np.linspace(0, 1, (nodes - 1) * NODE_EVERY + 1)[:, None]
    xy = np.array(start) + t * (np.array(end) - np.array(start))
    return np.column_stack([xy, np.full(len(t), yaw)])


def test_plan_follows_a_route():
    graph = PlaceGraph(
        SimpleNamespace(poses=[line((0, 0), (1, 0), 11)]),
        link_m=0.12,
        filter_walls=False,
        min_inliers=0,
    )
    assert graph.n == 11
    path = graph.plan((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    # The ends join every node within 0.15 m; which one the cheapest path takes is a tie.
    assert path.nodes == list(range(path.nodes[0], path.nodes[-1] + 1))
    assert path.nodes[0] <= 1
    assert path.nodes[-1] >= 9
    assert path.length == pytest.approx(1.0)
    assert path.waypoints[0] == pytest.approx([0, 0, 0])
    assert path.waypoints[-1] == pytest.approx([1, 0, 0])


def test_plan_takes_the_shortcut_between_routes():
    # Route 0 goes round three sides of a square; route 1 crosses it half way up, its ends 0.05 m
    # from route 0's sides, so within the link distance.
    around = np.concatenate(
        [line((0, 0), (0, 1), 11), line((0, 1), (1, 1), 11), line((1, 1), (1, 0), 11)]
    )
    across = line((0.05, 0.5), (0.95, 0.5), 10)
    graph = PlaceGraph(
        SimpleNamespace(poses=[around, across]), link_m=0.12, filter_walls=False, min_inliers=0
    )
    path = graph.plan((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert path.length < 2.1  # round the square: 3
    assert set(graph.route[path.nodes]) == {0, 1}
    unlinked = PlaceGraph(
        SimpleNamespace(poses=[around, across]), link_m=0.0, filter_walls=False, min_inliers=0
    )
    assert unlinked.plan((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)).length > 2.9


def test_turn_cost_prefers_fewer_turns():
    # Two routes with the same length between the ends, one turning back and forth.
    straight = line((0, 0), (1, 0), 11)
    wiggle = line((0, 0.1), (1, 0.1), 11)
    wiggle[:, 2] = np.where(np.arange(len(wiggle)) % 2, 1.5, -1.5)
    graph = PlaceGraph(
        SimpleNamespace(poses=[straight, wiggle]),
        link_m=0.0,
        filter_walls=False,
        min_inliers=0,
        turn_cost=0.05,
    )
    start, goal = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    assert set(graph.route[graph.plan(start, goal).nodes]) == {0}


def test_unconnected_poses_have_no_path():
    poses = [line((0, 0), (0.5, 0), 6), line((3, 3), (4, 3), 6)]
    graph = PlaceGraph(
        SimpleNamespace(poses=poses), connect_m=0.15, filter_walls=False, min_inliers=0
    )
    # Each end joins its nearest node, but the two pieces never touch.
    assert graph.plan((0, 0, 0), (4, 3, 0)) is None


def test_cross_track_geometry():
    path = Route(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], float), [])
    assert path.length == pytest.approx(2.0)
    assert path.cross_track((0.5, 0.1, 0.0)) == pytest.approx((0.1, 0.5))
    assert path.cross_track((1.2, 0.5, 0.0)) == pytest.approx((0.2, 1.5))
    assert path.cross_track((-0.3, 0.0, 0.0)) == pytest.approx((0.3, 0.0))  # before the start
    assert path.cross_track((1.0, 1.5, 0.0)) == pytest.approx((0.5, 2.0))  # past the end
    assert path.cross_track((1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0))  # on a corner


def test_waypoint_after_looks_ahead():
    path = Route(line((0, 0), (1, 0), 11), [])
    assert path.waypoint_after(0.0, 0.1)[0] == pytest.approx(0.1)
    assert path.waypoint_after(0.52, 0.1)[0] == pytest.approx(0.62)
    assert path.waypoint_after(0.99, 0.1)[0] == pytest.approx(1.0)


def test_localiser_without_motion_stays_put():
    n = 6
    store = FrozenMap(
        [np.zeros((n, 3))],
        None,
        np.zeros((4, 128), np.float32),
        np.zeros((n, 4 * 128), np.float32),
        SimpleNamespace(routes=[]),
    )
    localiser = Localiser(store, Camera(CAMERA), FloorMask(), motion={})
    blank = np.full((CAMERA["height"], CAMERA["width"], 3), 128, np.uint8)
    for _ in range(20):
        estimate = localiser.step(blank, None)
        assert (estimate.x, estimate.y, estimate.yaw) == (0.0, 0.0, 0.0)
        assert not estimate.fixed
    assert estimate.sigma_m == pytest.approx(0.02 + 19 * 0.01)


def grid_with_wall_at(x: float) -> SimpleNamespace:
    """A 0.05 m grid over [-1, 1]^2 with a vertical wall one cell wide at ``x``."""
    occupied = np.zeros((40, 40), bool)
    occupied[:, int(np.floor((x + 1.0) / 0.05))] = True
    return SimpleNamespace(occupied=lambda: occupied, origin=np.array([-1.0, -1.0]), res=0.05)


def test_bresenham_runs_between_the_ends():
    cells = list(bresenham(0, 0, 2, 4))
    assert cells[0] == (0, 0)
    assert cells[-1] == (2, 4)
    assert len(cells) == 5
    assert list(bresenham(3, 3, 3, 3)) == [(3, 3)]


def test_segment_meets_a_wall_only_across_it():
    grid = grid_with_wall_at(0.05)
    assert crosses_occupied(grid, np.array([0.0, 0.0]), np.array([0.1, 0.0]))
    assert not crosses_occupied(grid, np.array([0.1, 0.0]), np.array([0.5, 0.5]))
    assert not crosses_occupied(grid, np.array([-5.0, 0.0]), np.array([-4.0, 0.0]))  # off the grid


def test_links_through_a_wall_are_dropped():
    left, right = line((0, 0), (0, 0.9), 10), line((0.1, 0), (0.1, 0.9), 10)
    frozen = SimpleNamespace(poses=[left, right], grid=grid_with_wall_at(0.05))
    open_ = SimpleNamespace(poses=[left, right], grid=grid_with_wall_at(0.5))
    walled = PlaceGraph(frozen, min_inliers=0)
    free = PlaceGraph(open_, min_inliers=0)
    assert walled.link_counts["candidates"] == free.link_counts["candidates"] > 0
    assert walled.link_counts["after_walls"] == 0
    assert free.link_counts["after_walls"] == free.link_counts["candidates"]
    assert walled.plan((0, 0, 0), (0.1, 0, 0)) is not None  # each end joins its own side


def test_segment_hits_rects():
    rects = np.array([[1.0, 1.025, 0.0, 1.0]])
    assert segment_hits_rects(rects, np.array([0.5, 0.5]), np.array([1.5, 0.5]))
    assert not segment_hits_rects(rects, np.array([0.5, 1.5]), np.array([1.5, 1.5]))
    assert not segment_hits_rects(rects, np.array([0.5, 0.5]), np.array([0.9, 0.5]))
    assert segment_hits_rects(rects, np.array([1.01, -1.0]), np.array([1.01, 2.0]))  # vertical


def blank_localiser() -> Localiser:
    store = FrozenMap(
        [np.zeros((6, 3))],
        None,
        np.zeros((4, 128), np.float32),
        np.zeros((6, 4 * 128), np.float32),
        SimpleNamespace(routes=[]),
    )
    return Localiser(store, Camera(CAMERA), FloorMask(), motion={})


def test_a_near_fix_is_accepted_and_blended():
    localiser = blank_localiser()
    status, moved, dropped = localiser._gate((0.1, 0.0, 0.0))
    assert (status, dropped) == ("accepted", 0)
    assert 0 < moved < 0.1  # blended by the variances, not taken whole


def test_a_far_fix_is_held_until_the_next_agrees():
    localiser = blank_localiser()
    assert localiser._gate((1.0, 0.0, 0.0))[0] == "held"
    assert localiser.pose[:2] == (0.0, 0.0)
    status, moved, dropped = localiser._gate((1.05, 0.02, 0.0))
    assert (status, dropped) == ("accepted", 0)
    assert localiser.pose[:2] == pytest.approx((1.05, 0.02))  # replaced, not blended
    assert localiser.sigma == pytest.approx(0.05)


def test_disagreeing_far_fixes_are_never_taken():
    localiser = blank_localiser()
    assert localiser._gate((1.0, 0.0, 0.0))[0] == "held"
    status, _, dropped = localiser._gate((-1.0, 0.5, 0.0))
    assert (status, dropped) == ("held", 1)  # the first is dropped, the second held
    assert localiser._gate((1.0, 0.0, 0.3))[0] == "held"  # same place, yaw 17 degrees off
    assert localiser.pose == (0.0, 0.0, 0.0)


def wall(x0, x1, y, step=0.02) -> np.ndarray:
    xs = np.arange(x0, x1, step)
    return np.column_stack([xs, np.full(len(xs), y)])


def test_layout_score_agrees_on_the_same_wall():
    a = wall(-1, 1, 0.5)
    b = wall(-1, 1, 0.3)  # the same wall, seen from 0.2 m to the left of a's origin
    agree, n = layout_score(a, b, (0.0, 0.2, 0.0))
    assert agree == pytest.approx(1.0)
    assert n > 0


def test_layout_score_disagrees_on_another_wall_and_ignores_far_points():
    a = wall(-1, 1, 0.5)
    b = wall(-1, 1, 0.3)
    assert layout_score(a, b, (0.0, 0.0, 0.0))[0] == 0.0  # b's wall lands 0.2 m off a's
    far = np.vstack([b, [[3.0, 0.0], [np.nan, np.nan]]])  # beyond the range, and NaN
    assert layout_score(a, far, (0.0, 0.2, 0.0))[0] == pytest.approx(1.0)


def test_layout_score_is_symmetric_over_the_two_scans():
    a = np.vstack([wall(-1, 1, 0.5), wall(-1, 1, -0.5)])
    b = wall(-1, 1, 0.5)  # sees one of a's two walls: a's other wall has no match in b
    agree, _ = layout_score(a, b, (0.0, 0.0, 0.0))
    assert 0.6 < agree < 0.9


def test_layout_score_without_points():
    only_b = layout_score(np.empty((0, 2)), wall(0, 1, 0.5), (0, 0, 0))
    assert only_b[0] == 0.0
    assert only_b[1] > 0
    assert layout_score(np.empty((0, 2)), np.empty((0, 2)), (0, 0, 0)) == (0.0, 0)


def test_frozen_map_neighbours_stay_on_the_route():
    store = FrozenMap(
        [np.zeros((4, 3)), np.zeros((6, 3))],
        None,
        np.zeros((4, 128), np.float32),
        np.zeros((10, 4 * 128), np.float32),
        SimpleNamespace(routes=[]),
    )
    assert list(store.neighbours(1, 5)) == [0, 1, 2, 3]
    assert list(store.neighbours(7, 2)) == [5, 6, 7, 8, 9]


def shortcut_store(wall_x: float | None = None) -> SimpleNamespace:
    """Mapped frames along an L (x then y, 0.02 m apart), and a grid with a wall at ``wall_x``."""
    t = np.arange(0.0, 1.0001, 0.02)
    xy = np.concatenate([np.column_stack([t, 0 * t]), np.column_stack([0 * t + 1, t])])
    log_odds = np.zeros((40, 40), np.float32)
    if wall_x is not None:
        log_odds[:, int(np.floor((wall_x + 1.0) / 0.05))] = 5.0
    grid = OccupancyMap(half_extent=1.0, res=0.05)
    grid.log_odds = log_odds
    return SimpleNamespace(xy=xy, grid=grid)


def test_shortcut_keeps_the_corner_of_an_l():
    # The diagonal of the L is far from every frame: the corner stays.
    path = Route(
        np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [1, 0.5, 0], [1, 1, 0]], float), [1, 2, 3]
    )
    short = path.shortcut(shortcut_store())
    assert short.waypoints[:, :2].tolist() == [[0, 0], [1, 0], [1, 1]]
    assert short.nodes == [2]
    assert short.length_before == pytest.approx(2.0)
    assert short.length == pytest.approx(2.0)
    assert len(short.pulled) == 2


def test_shortcut_cuts_a_corner_within_the_corridor():
    path = Route(
        np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [1, 0.5, 0], [1, 1, 0]], float), [1, 2, 3]
    )
    short = path.shortcut(shortcut_store(), corridor_m=0.5)
    assert short.waypoints[:, :2].tolist() == [[0, 0], [1, 1]]
    assert short.length < short.length_before


def test_shortcut_stops_at_a_wall():
    path = Route(np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0]], float), [1])
    assert len(path.shortcut(shortcut_store()).waypoints) == 2
    assert len(path.shortcut(shortcut_store(wall_x=0.5)).waypoints) == 3
