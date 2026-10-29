"""The localiser's prediction (visual odometry, the calibrated and the online fallbacks, the
Manhattan snap) and the navigator's replans, on stubs: no map, no frames."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

import localiser as localiser_module  # noqa: E402
from geometry import CAMERA, Camera, FloorMask  # noqa: E402
from localiser import DRIFT_M, ONLINE_MIN, START_SIGMA_M, Estimate, Localiser  # noqa: E402
from mapping import compose  # noqa: E402
from navigator import Navigator  # noqa: E402
from planner import Path as Route  # noqa: E402

MOTION = {
    "FORWARD": {"dx": 0.03, "dy": 0.0, "dyaw": 0.0},
    "LEFT": {"dx": 0.0, "dy": 0.0, "dyaw": math.radians(3.4)},
}
VIEW = SimpleNamespace(points=np.empty((0, 2)))


@pytest.fixture
def predict(monkeypatch):
    """A Localiser and ``step(action, vo=None, manhattan=(nan, 0))``: one ``_predict`` with that
    visual odometry result and manhattan_yaw."""
    localiser = Localiser(
        SimpleNamespace(), Camera(CAMERA), FloorMask(), motion=MOTION, online_motion=True
    )

    def step(action, vo=None, manhattan=(math.nan, 0.0)):
        monkeypatch.setattr(localiser_module, "visual_odometry", lambda *a: vo)
        monkeypatch.setattr(localiser_module, "manhattan_yaw", lambda points: manhattan)
        localiser._predict(VIEW, VIEW, action)
        return localiser.pose

    return localiser, step


def test_a_visual_odometry_step_moves_the_pose_and_is_kept(predict):
    localiser, step = predict
    vo = (0.02, 0.01, 0.1)
    assert step("FORWARD", vo) == pytest.approx(compose((0.0, 0.0, 0.0), vo))
    assert localiser.source == "vo"
    assert localiser.samples["FORWARD"] == [vo]
    assert localiser.sigma == pytest.approx(START_SIGMA_M + DRIFT_M)


def test_without_visual_odometry_the_calibrated_motion_stands_in(predict):
    localiser, step = predict
    assert step("FORWARD") == pytest.approx((0.03, 0.0, 0.0))
    assert localiser.source == "calibrated"
    assert "FORWARD" not in localiser.samples


def test_an_implausible_step_is_not_taken(predict):
    localiser, step = predict
    assert step("FORWARD", (0.5, 0.0, 0.0)) == pytest.approx((0.03, 0.0, 0.0))
    assert localiser.source == "calibrated"


def test_no_action_does_not_move(predict):
    localiser, step = predict
    assert step(None) == (0.0, 0.0, 0.0)
    assert step("BACKWARD") == (0.0, 0.0, 0.0)  # no calibration of it either


def test_the_run_median_replaces_the_calibration_once_there_are_enough(predict):
    localiser, step = predict
    for _ in range(ONLINE_MIN - 1):
        step("FORWARD", (0.05, 0.0, 0.0))
    x = localiser.pose[0]
    assert step("FORWARD")[0] == pytest.approx(x + 0.03)
    assert localiser.source == "calibrated"
    step("FORWARD", (0.05, 0.0, 0.0))
    x = localiser.pose[0]
    assert step("FORWARD")[0] == pytest.approx(x + 0.05)
    assert localiser.source == "online"


def test_the_turn_rate_is_the_run_median(predict):
    localiser, step = predict
    assert localiser.turn_rate() is None
    for _ in range(ONLINE_MIN):
        step("LEFT", (0.0, 0.0, math.radians(3.5)))
    assert localiser.turn_rate() == pytest.approx(3.5)
    for _ in range(ONLINE_MIN):
        step("RIGHT", (0.0, 0.0, math.radians(-3.3)))
    assert localiser.turn_rate() == pytest.approx(3.4)


def test_a_confident_manhattan_yaw_snaps_the_heading(predict):
    _, step = predict
    turn = (0.0, 0.0, math.radians(12.0))
    assert math.degrees(step("LEFT", turn, (10.0, 0.9))[2]) == pytest.approx(10.0)
    assert math.degrees(step("LEFT", turn, (10.0, 0.5))[2]) == pytest.approx(22.0)


# -- the navigator


class Planner:
    """Plans a straight path from the start to the goal (None while ``connected`` is False)."""

    def __init__(self) -> None:
        self.calls, self.connected = 0, True

    def plan(self, start, goal):
        self.calls += 1
        return Route([start, goal], []) if self.connected else None


def _estimate(x: float, y: float, jump: float = 0.0) -> Estimate:
    return Estimate(x, y, 0.0, 0.05, jump > 0, jump, None, 0, False)


def _navigator(**kwargs) -> tuple[Navigator, Planner]:
    planner = Planner()
    return Navigator(planner, None, (2.0, 0.0, 0.0), **kwargs), planner


def test_the_first_update_plans():
    navigator, planner = _navigator()
    update = navigator.update(None, None, _estimate(0.0, 0.0))
    assert update.replan == "initial" and planner.calls == 1
    assert update.path.waypoints[-1] == pytest.approx((2.0, 0.0, 0.0))


def test_on_the_path_nothing_is_replanned():
    navigator, planner = _navigator()
    navigator.update(None, None, _estimate(0.0, 0.0))
    assert navigator.update(None, None, _estimate(0.5, 0.1)).replan is None
    assert planner.calls == 1


def test_off_the_path_replans():
    navigator, _ = _navigator()
    navigator.update(None, None, _estimate(0.0, 0.0))
    assert navigator.update(None, None, _estimate(0.5, 0.2)).replan == "cross_track"


def test_a_fix_jump_replans():
    navigator, _ = _navigator()
    navigator.update(None, None, _estimate(0.0, 0.0))
    assert navigator.update(None, None, _estimate(0.5, 0.0, jump=0.11)).replan == "jump"


def test_an_old_path_is_replanned():
    navigator, _ = _navigator(replan_every=3)
    reasons = [navigator.update(None, None, _estimate(0.1 * i, 0.0)).replan for i in range(5)]
    assert reasons == ["initial", None, None, "periodic", None]


def test_without_a_new_path_the_old_one_is_kept():
    navigator, planner = _navigator()
    first = navigator.update(None, None, _estimate(0.0, 0.0)).path
    planner.connected = False
    update = navigator.update(None, None, _estimate(0.5, 0.3))
    assert update.replan is None and update.path is first


def test_the_waypoint_is_lookahead_m_along_the_path():
    navigator, _ = _navigator(lookahead_m=0.15)
    navigator.path = Route([(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0), (2.0, 0.0, 0.0)], [])
    navigator.since_plan = 0
    update = navigator.update(None, None, _estimate(0.0, 0.0))
    assert update.replan is None
    assert update.waypoint == pytest.approx((0.2, 0.0, 0.0))
