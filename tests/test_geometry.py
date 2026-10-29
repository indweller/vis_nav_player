"""geometry's basics: the camera's ranges, free space, the Manhattan yaw, the snap, and the motion
calibration (on the simulator, skipped without it)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

import mapping  # noqa: E402
from conftest import SEED  # noqa: E402
from geometry import (  # noqa: E402
    CAMERA,
    FAR,
    FLOOR_MASK_PATH,
    Camera,
    FloorMask,
    calibrate_motion,
    free_space,
    manhattan_yaw,
    snap,
)

TICK_M = 0.03
TICK_DEG = 3.4375
"""The simulator's tick: metres forward, degrees of turn."""


def test_ranges_invert_floor_rows():
    camera = Camera(CAMERA)
    distance = np.linspace(0.3, 2.0, camera.width)
    assert camera.ranges(camera.floor_rows(distance)) == pytest.approx(distance)


def test_ranges_are_far_at_the_horizon():
    camera = Camera(CAMERA)
    assert (camera.ranges(np.full(camera.width, camera.cy)) == FAR).all()


def test_free_space_reads_each_column_from_the_bottom():
    camera = Camera(CAMERA)
    mask = np.zeros((camera.rows, camera.width), bool)
    mask[180:] = True  # floor from row 180 down: the wall's foot at row 180
    mask[:, :10] = True  # floor to the horizon: no wall
    mask[-1, 20:30] = False  # a wall in the bottom row: its foot is below the frame
    distance, seen = free_space(camera, mask)
    assert distance[50] == pytest.approx(camera.ranges(np.full(camera.width, 180.0))[50])
    assert seen[50] and seen[:10].all()
    assert (distance[:10] == FAR).all()
    assert not seen[20:30].any()


def _wall(yaw_deg: float, offset: tuple[float, float] = (0.6, 0.0)) -> np.ndarray:
    """Robot-frame points along a wall that lies along a world axis, the robot at ``yaw_deg``."""
    a = math.radians(-yaw_deg)
    t = np.linspace(-0.4, 0.4, 80)[:, None]
    return np.array(offset) + t * np.array([math.cos(a), math.sin(a)])


@pytest.mark.parametrize("yaw_deg", [0.4, 20.0, 47.0, 89.0])
def test_manhattan_yaw_of_one_wall(yaw_deg):
    yaw, confidence = manhattan_yaw(_wall(yaw_deg))
    assert (yaw - yaw_deg + 45) % 90 - 45 == pytest.approx(0.0, abs=0.5)
    assert confidence == pytest.approx(1.0)


def test_manhattan_yaw_of_a_corner_is_its_walls():
    corner = np.concatenate([_wall(20.0), _wall(110.0, (0.6, 0.5))])
    assert manhattan_yaw(corner)[0] == pytest.approx(20.0, abs=0.5)


def test_manhattan_yaw_without_walls():
    yaw, confidence = manhattan_yaw(np.full((320, 2), np.nan))
    assert math.isnan(yaw) and confidence == 0.0


def test_snap_takes_the_nearest_axis():
    assert math.degrees(snap(math.radians(47.0), 0.0)) == pytest.approx(90.0)
    assert math.degrees(snap(math.radians(43.0), 0.0)) == pytest.approx(0.0)
    assert math.degrees(snap(math.radians(-100.0), 10.0)) == pytest.approx(-80.0)
    assert math.degrees(snap(math.radians(190.0), 5.0)) == pytest.approx(185.0)


@pytest.fixture(scope="module")
def corridor_drive(textures) -> tuple[list[np.ndarray], list[str | None]]:
    """Frames and actions of a drive along the corridor at maze SEED's spawn, without noise."""
    import vis_nav_sim as sim

    world = sim.Simulator(textures, SEED)
    frames, actions = [world.initial_frame()], [None]
    for name, ticks in [("LEFT", 26), ("FORWARD", 20), ("RIGHT", 8), ("LEFT", 8), ("BACKWARD", 15)]:
        for _ in range(ticks):
            frames.append(world.step(getattr(sim, name), 1)[-1])
            actions.append(name)
    return frames, actions


def test_calibrate_motion_measures_the_tick(corridor_drive):
    frames, actions = corridor_drive
    camera = Camera({**CAMERA, "cy": CAMERA["cy"] + mapping.HORIZON_SHIFT_PX})
    motion = calibrate_motion(frames, actions, camera, FloorMask.load(FLOOR_MASK_PATH))
    assert motion["FORWARD"]["dx"] == pytest.approx(TICK_M, abs=0.003)
    assert motion["BACKWARD"]["dx"] == pytest.approx(-TICK_M, abs=0.003)
    assert math.degrees(motion["LEFT"]["dyaw"]) == pytest.approx(TICK_DEG, abs=0.2)
    for name in ("FORWARD", "LEFT"):
        assert set(motion[name]) == {"dx", "dy", "dyaw", "std", "n"}
        assert min(motion[name]["n"]) > 0
