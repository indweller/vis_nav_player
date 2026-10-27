"""The pose-free map's pieces on synthetic data: no simulator, no recorded frames."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "eval"))

import mapping  # noqa: E402
from geometry import (  # noqa: E402
    Camera,
    FloorMask,
    WallView,
    rigid_fit,
    rigid_ransac,
    transform,
    visual_odometry,
    wall_view,
)

CAMERA = {"width": 320, "height": 240, "fx": 92.0, "fy": 92.0, "cx": 160.0, "cy": 120.0}


def _view(wall: np.ndarray, des: np.ndarray) -> WallView:
    return WallView(
        np.zeros((len(wall), 2), np.float32), des, wall, np.full((CAMERA["width"], 2), np.nan)
    )


def test_rigid_fit_recovers_a_pose():
    rng = np.random.default_rng(0)
    b = rng.uniform(-1, 1, (40, 2))
    pose = (0.3, -0.2, 0.4)
    x, y, yaw = rigid_fit(transform(pose, b), b)
    assert (x, y, yaw) == pytest.approx(pose, abs=1e-9)


def test_rigid_ransac_ignores_outliers():
    rng = np.random.default_rng(1)
    b = rng.uniform(-1, 1, (60, 2))
    pose = (0.05, 0.02, -0.1)
    a = transform(pose, b)
    a[:20] = rng.uniform(-1, 1, (20, 2))
    fit = rigid_ransac(a, b, rng)
    assert fit is not None
    assert fit[0] == pytest.approx(pose, abs=1e-6)
    assert fit[1][20:].all()
    assert fit[1].sum() >= 40


def test_rigid_ransac_gives_nothing_for_unrelated_points():
    rng = np.random.default_rng(2)
    assert rigid_ransac(rng.uniform(-1, 1, (30, 2)), rng.uniform(-1, 1, (30, 2)), rng) is None


def test_visual_odometry_is_the_pose_of_the_new_view_in_the_old():
    rng = np.random.default_rng(3)
    des = rng.random((50, 128)).astype(np.float32)
    des /= des.sum(axis=1, keepdims=True)
    previous = rng.uniform(0.2, 1.0, (50, 2))
    motion = (0.03, 0.0, 0.05)
    # The same walls seen from the new pose: previous points in the new robot frame.
    x, y, yaw = motion
    c, s = math.cos(yaw), math.sin(yaw)
    d = previous - [x, y]
    current = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
    step = visual_odometry(_view(previous, des), _view(current, des))
    assert step == pytest.approx(motion, abs=1e-6)


def test_visual_odometry_fails_without_walls():
    des = np.random.default_rng(4).random((20, 128)).astype(np.float32)
    nowhere = np.full((20, 2), np.nan)
    assert visual_odometry(_view(nowhere, des), _view(nowhere, des)) is None


def test_wall_view_ranges_keypoints_on_the_wall_only():
    camera = Camera(CAMERA)
    image = np.full((240, 320, 3), 200, np.uint8)  # the floor, light
    image[:150] = (40, 60, 80)  # the wall, dark, its foot at row 150
    rng = np.random.default_rng(5)
    for _ in range(80):
        x, y = int(rng.integers(10, 310)), int(rng.integers(10, 140))
        cv2.circle(
            image,
            (x, y),
            int(rng.integers(2, 6)),
            tuple(int(c) for c in rng.integers(0, 255, 3)),
            -1,
        )
    floor = FloorMask()
    truth = np.zeros((240, 320), int)
    truth[160:] = 1
    clean = np.full((240, 320, 3), 200, np.uint8)
    clean[:150] = (40, 60, 80)
    floor.fit([(clean, truth)])
    view = wall_view(image, camera, floor)
    on_wall = np.isfinite(view.wall).all(axis=1)
    assert on_wall.any()
    assert (view.pixels[on_wall, 1] < 150).all()
    expected = (
        CAMERA["fy"]
        * 0.2
        / 30
        / np.cos(np.radians(camera.bearing[view.pixels[on_wall, 0].astype(int)]))
    )
    assert np.hypot(*view.wall[on_wall].T) == pytest.approx(expected, rel=0.05)


def test_compose_and_between_are_inverse():
    a, b = (1.0, 2.0, 0.7), (-0.3, 0.9, -2.0)
    assert mapping.compose(a, mapping.between(a, b)) == pytest.approx(b)


def test_optimise_pulls_a_loop_closed():
    # A square driven with a 5 % scale error in the odometry, its last node measured to be
    # the first: the closure edge moves the nodes back.
    sides = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    poses = [(x, y, k * math.pi / 2) for k, (x, y) in enumerate(sides)]
    poses.append((0.0, 0.0, 2 * math.pi))
    initial = np.array([(0.95 * x, 0.95 * y, yaw) for x, y, yaw in poses])
    edges = [(k, k + 1) for k in range(4)] + [(4, 0)]
    measured = [mapping.between(poses[k], poses[k + 1]) for k in range(4)] + [(0.0, 0.0, 0.0)]
    sigma = np.array([(0.05, 0.05)] * 4 + [(0.01, 0.01)])
    solved = mapping.optimise(
        initial,
        np.array(edges),
        np.array(measured, float),
        sigma,
        np.array([0]),
        np.array([[0.0, 0.0, 0.0]]),
    )
    assert math.hypot(*(solved[4, :2] - solved[0, :2])) < 0.01
    assert math.dist(solved[2, :2], (1.0, 1.0)) < math.dist(initial[2, :2], (1.0, 1.0))


def test_load_routes_reads_actions_and_frames_lazily(tmp_path):
    route = tmp_path / "traj_0"
    route.mkdir()
    records = [
        {"step": 0, "image": "0.jpg", "action": ["IDLE"]},
        {"step": 1, "image": "1.jpg", "action": ["FORWARD"]},
        {"step": 2, "image": "2.jpg", "action": ["FORWARD", "LEFT"]},
    ]
    (route / "data_info.json").write_text(json.dumps(records))
    cv2.imwrite(str(route / "1.jpg"), np.full((4, 4, 3), 9, np.uint8))
    (loaded,) = mapping.load_routes(tmp_path)
    assert loaded.actions == [None, "FORWARD", None]
    assert len(loaded.frames) == 3
    assert loaded.frames[1].shape == (4, 4, 3)


def test_a_closure_is_consistent_with_the_poses_it_joins():
    poses = [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
    near = mapping.Closure((0, 0), (0, mapping.NODE_EVERY), (1.5, 0.0, 0.1), 40, 30)
    far = mapping.Closure((0, 0), (0, mapping.NODE_EVERY), (3.0, 0.0, 0.0), 40, 30)
    turned = mapping.Closure((0, 0), (0, mapping.NODE_EVERY), (1.0, 0.0, 2.0), 40, 30)
    assert mapping.consistent(near, poses)
    assert not mapping.consistent(far, poses)
    assert not mapping.consistent(turned, poses)


def test_the_evaluation_aligns_by_best_fit_and_scores_errors():
    import eval_map

    truth = [np.column_stack([np.linspace(0, 3, 20), np.linspace(0, 1, 20), np.zeros(20)])]
    estimated = [
        np.column_stack([transform((0.2, -0.1, 0.3), truth[0][:, :2]), truth[0][:, 2] + 0.3])
    ]
    (spawn,) = eval_map.trajectory_errors(estimated, truth)
    (fit,) = eval_map.trajectory_errors(estimated, eval_map.best_fit(estimated, truth))
    assert spawn[0].max() > 0.2 and spawn[1].max() == pytest.approx(math.degrees(0.3))
    assert fit[0].max() < 1e-9 and fit[1].max() < 1e-6
