"""MapAgent on a tiny recorded maze: no error, and a CHECKIN at the end. Needs the simulator's
texture pack and data/floor_mask.npz, and is skipped without them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from map_agent import FLOOR_MASK_PATH, MapAgent  # noqa: E402
from vis_nav_sdk import run  # noqa: E402

SEED = 7
DRIVE = [("FORWARD", 40), ("LEFT", 15), ("FORWARD", 30), ("RIGHT", 15), ("FORWARD", 20)]
MAX_STEPS = 60


def record_tiny(root: Path) -> None:
    """One short drive from the spawn of maze 7 in the format of ``Simulator.record``."""
    import vis_nav_sim as sim

    world = sim.Simulator(sim.Textures.find(), SEED)
    route = root / "traj_0"
    route.mkdir(parents=True)
    records = []

    def save(frame, action) -> None:
        name = f"{len(records)}.jpg"
        (route / name).write_bytes(world.jpeg(frame))
        records.append({"step": len(records), "image": name, "action": [action]})

    save(world.initial_frame(), "IDLE")
    for action, ticks in DRIVE:
        for _ in range(ticks):
            save(world.step(getattr(sim, action), 1)[-1], action)
    (route / "data_info.json").write_text(json.dumps(records))
    (root / "target.jpg").write_bytes(world.jpeg(world.targets()[0]))


@pytest.fixture(scope="module")
def tiny_maze(tmp_path_factory) -> Path:
    pytest.importorskip("vis_nav_sim")
    if not FLOOR_MASK_PATH.exists():
        pytest.skip("no data/floor_mask.npz")
    import vis_nav_sim as sim

    try:
        sim.Textures.find()
    except FileNotFoundError:
        pytest.skip("no texture pack")
    root = tmp_path_factory.mktemp("tiny")
    record_tiny(root)
    return root


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


def test_arc_motion_sums_the_means_unless_the_measure_differs():
    from map_agent import arc_motion

    motion = {
        "FORWARD": {"dx": 0.03, "dy": 0.0, "dyaw": 0.0002},
        "LEFT": {"dx": 0.0, "dy": 0.0, "dyaw": 0.058},
        "RIGHT": {"dx": 0.0, "dy": 0.0, "dyaw": -0.057},
    }
    arcs, off = arc_motion(motion)
    assert arcs["FORWARD|LEFT"]["dx"] == pytest.approx(0.03)
    assert arcs["FORWARD|LEFT"]["dyaw"] == pytest.approx(0.0582)
    assert off["FORWARD|LEFT"] != off["FORWARD|LEFT"]  # NaN
    close = {"FORWARD|LEFT": {"dx": 0.03, "dy": 0.0, "dyaw": 0.06}}
    assert arc_motion(motion, close)[0]["FORWARD|LEFT"]["dyaw"] == pytest.approx(0.0582)
    far = {"FORWARD|LEFT": {"dx": 0.03, "dy": 0.0, "dyaw": 0.03}}
    arcs, off = arc_motion(motion, far)
    assert arcs["FORWARD|LEFT"]["dyaw"] == pytest.approx(0.03)
    assert off["FORWARD|LEFT"] > 0.2


def test_steering_bands():
    agent = MapAgent.__new__(MapAgent)
    agent.turn_tol, agent.arc_deg, agent.max_turn_ticks = 12.0, 45.0, 26
    agent.forward_ticks, agent.turn_rate = 2, 3.4375
    agent.arcs, agent.arc_radius = True, 0.5
    assert agent._steer(5.0) == ["FORWARD"] * 2
    assert agent._steer(-30.0) == ["FORWARD|RIGHT"]
    assert agent._steer(30.0) == ["FORWARD|LEFT"]
    assert agent._steer(90.0) == ["LEFT"] * 26
    assert agent._steer(-60.0) == ["RIGHT"] * 17
    assert agent._steer(-170.0) == ["RIGHT"] * 26
    # A waypoint 0.3 m away at 40 degrees is inside the turning circle (0.64 m): turn first.
    assert agent._steer(40.0, 0.3) == ["LEFT"] * 12
    assert agent._steer(40.0, 0.7) == ["FORWARD|LEFT"]
    agent.arcs = False
    assert agent._steer(30.0, 1.0) == ["LEFT"] * 9
