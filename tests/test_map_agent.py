"""MapAgent on a tiny recorded maze (conftest.tiny_maze): no error, and a CHECKIN at the end.
Needs the simulator's texture pack and data/floor_mask.npz, and is skipped without them."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from conftest import SEED  # noqa: E402
from map_agent import Config, MapAgent  # noqa: E402
from vis_nav_sdk import run  # noqa: E402

MAX_STEPS = 60


@pytest.mark.parametrize("override", [None, (0.0, 0.0, 0.0)])
def test_reaches_checkin(tiny_maze, override):
    agent = MapAgent(tiny_maze, goal_override=override)
    result = run(
        agent,
        f"local:{SEED}",
        viewer=False,
        quiet=True,
        check=False,
        confirm=False,
        browser=False,
        motion_noise=False,
        max_steps=MAX_STEPS,
    )
    assert result is not None
    assert agent.checkin_reason in ("low_steps", "homed", "homing_max")
    assert result.nav_steps <= MAX_STEPS


def test_config_defaults():
    config = Config()
    assert config.align_deg == 8.0
    assert config.link_m == 0.25


def test_steering_bands():
    agent = MapAgent.__new__(MapAgent)
    agent.config = Config(max_turn_ticks=26)
    agent.turn_rate, agent.localiser = 3.4375, SimpleNamespace(turn_rate=lambda: None)
    assert agent._steer(5.0) == ["FORWARD"] * 2
    assert agent._steer(30.0) == ["LEFT"] * 9
    assert agent._steer(-60.0) == ["RIGHT"] * 17
    assert agent._steer(-170.0) == ["RIGHT"] * 26
    agent.config = Config()
    assert agent._steer(30.0) == ["LEFT"] * 3


class Estimate(NamedTuple):
    x: float
    y: float
    yaw: float


def _homing_agent(steps: int = 0, view=None, yaw_deg: float = -20.0):
    """A MapAgent with the state ``_home`` reads, the goal at the origin facing 0 and the estimate
    there facing ``yaw_deg``; ``view`` is the goal view match."""
    agent = MapAgent.__new__(MapAgent)
    agent.config, agent.homing_steps = Config(), steps
    agent.turn_rate, agent.localiser = 3.4375, SimpleNamespace(turn_rate=lambda: None)
    agent.approaching, agent.timed_out, agent.view = False, False, view
    agent.goal = (0.0, 0.0, 0.0)
    agent.update = SimpleNamespace(estimate=Estimate(0.0, 0.0, math.radians(yaw_deg)))
    agent._yaw_to_view0 = lambda image: 0.0
    return agent


def test_homing_timeout_turns_by_the_estimate_before_checking_in():
    agent = _homing_agent(steps=60)
    assert (
        agent._home(SimpleNamespace(image=None)) == ["LEFT"] * 3
    )  # 20 degrees off: turn, don't check in
    assert agent.timed_out
    agent.update.estimate = Estimate(0.0, 0.0, 0.05)  # 2.9 degrees
    assert agent._home(SimpleNamespace(image=None)) == ["CHECKIN"]
    assert agent.checkin_reason == "homing_max"


def test_estimate_alignment_is_tighter_than_the_view_alignment():
    # No view matched: 6 degrees is over the 5 degree estimate tolerance.
    agent = _homing_agent(yaw_deg=-6.0)
    assert agent._home(SimpleNamespace(image=None))[0] == "LEFT"
    # A matched view: 6 degrees is under the 8 degree tolerance.
    agent = _homing_agent(view=(0, 6.0), yaw_deg=-6.0)
    assert agent._home(SimpleNamespace(image=None)) == ["CHECKIN"]


def test_test_local_never_reaches_the_session(tiny_maze, monkeypatch, capsys):
    import map_agent

    def refuse(*args, **kwargs):
        raise AssertionError("the SDK's session code was reached")

    monkeypatch.setattr(map_agent, "run", refuse)
    monkeypatch.setattr(map_agent.MapAgent, "setup", refuse)
    monkeypatch.setattr(map_agent.cli, "challenge", refuse)
    monkeypatch.setattr(map_agent.cli, "run_options", refuse)
    map_agent.main(["--test-local", "--data", str(tiny_maze)])
    out = capsys.readouterr().out
    for stage in ("map load/build", "motion calibration", "planner", "find_goal"):
        assert stage in out
    assert "frames per route:" in out and "goal pose:" in out and "motion per action" in out


def test_homing_turns_by_the_estimate_not_the_view():
    agent = _homing_agent(view=(0, 0.0))  # the view says aligned; the estimate is 20 deg off
    assert agent._home(SimpleNamespace(image=None))[0] == "LEFT"
