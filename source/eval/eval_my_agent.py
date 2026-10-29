"""my_agent.MyAgent on the local simulator, maze 42, against the ground truth.

    uv run source/eval/eval_my_agent.py
    uv run source/eval/eval_my_agent.py --tag verify5 --verify-candidates 5

The agent is built as my_agent.py's __main__ builds it with its default arguments, from the
exploration data of cli.record_exploration_data(42), and drives a local session with motion noise
on and a motion_seed other than the exploration's, as eval_closed_loop.drive does. It stops on
CHECKIN or when the step budget (3000) is spent.

Writes to results/my_agent/seed42/<tag>/ (--tag, default "default"): ticks.csv (per tick the true
pose, the raw, filtered and current node, the confidence, low_conf, the move's source and action
and the hops from the current node to the goal), metrics.json (distance_m and heading_deg at
arrival, checked_in, checkin_reason, nav_steps, oracle_steps, spl -- oracle_steps /
max(oracle_steps, nav_steps) for a check-in within SUCCESS_M and SUCCESS_DEG of the goal, else 0 --
and the SDK's sdk_points, sdk_tier and sdk_spl of the check-in, and the ticks where the filter's
node differs from the raw one and where the confidence was low) and path.png (true walls, true path
in black, start in blue, goal in green, check-in position in red).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import vis_nav_sim as sim
from common import textures, write_csv
from eval_closed_loop import MOTION_SEED_OFFSET, SUCCESS_DEG, SUCCESS_M
from eval_cues import Walls
from eval_map import MAP_COLOUR, Canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cli  # noqa: E402
from my_agent import SHORTCUTS_PER_1K, MyAgent  # noqa: E402
from vis_nav_sdk import Action  # noqa: E402
from vis_nav_sdk.local import local_session  # noqa: E402

SEED = 42
MAX_STEPS = 3000
PROGRESS_EVERY = 100
TICK_FIELDS = [
    "tick", "x", "y", "yaw", "raw_node", "filter_node", "current", "confidence", "low_conf",
    "source", "last", "hops",
]  # fmt: skip
OUT = Path("results") / "my_agent" / f"seed{SEED}"


def split(returned) -> tuple[Action, int]:
    """An agent's return as (action, ticks): a bare Action is one tick."""
    return returned if isinstance(returned, tuple) else (returned, 1)


def tick_row(agent: MyAgent, tick: int, pose) -> dict:
    try:
        hops = nx.shortest_path_length(agent.graph, agent.current, agent.goal)
    except (nx.NetworkXException, KeyError, TypeError):
        hops = -1
    return {
        "tick": tick,
        "x": pose[0],
        "y": pose[1],
        "yaw": pose[2],
        "raw_node": agent.raw_node,
        "filter_node": agent.filter_node,
        "current": agent.current,
        "confidence": agent.confidence,
        "low_conf": agent.low_conf,
        "source": agent.source,
        "last": agent.last,
        "hops": hops,
    }


def drive(agent: MyAgent) -> dict:
    """One run on a local session, the true pose logged every tick."""
    session = local_session(
        f"local:{SEED}",
        textures=textures(),
        motion_noise=sim.DEFAULT_NOISE,
        motion_seed=SEED + MOTION_SEED_OFFSET,
        max_steps=MAX_STEPS,
    )
    agent.setup(session.info)
    obs = session.initial_observation
    poses, ticks, result, arrival = [], [], None, None
    while True:
        decision, repeat = split(agent.act(obs))
        pose = session.pose()
        poses.append(pose)
        ticks.append(tick_row(agent, obs.step, pose))
        if obs.step % PROGRESS_EVERY < repeat:
            print(f"step {obs.step}/{MAX_STEPS}", flush=True)
        if decision & Action.CHECKIN:
            arrival = session.distance_heading()
            result = session.checkin()
            break
        if obs.steps_left <= 1:
            arrival = session.distance_heading()
            break
        obs = session.step(decision, repeat)
    return {
        "poses": np.array(poses),
        "ticks": ticks,
        "result": result,
        "arrival": arrival,
        "nav_steps": session.simulator.steps,
    }


def draw(world: sim.Simulator, run_: dict, metrics: dict, path: Path) -> None:
    canvas = Canvas(world)
    canvas.walls(Walls(world), MAP_COLOUR)
    poses = run_["poses"]
    line = np.round(canvas.pixels(poses[:, :2])).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas.image, [line], False, (0, 0, 0), 1, cv2.LINE_AA)
    canvas.marker(tuple(world.target), (0, 170, 0), arrow=True)
    canvas.marker(tuple(poses[0]), (220, 100, 0), arrow=True)
    if metrics["checked_in"]:
        canvas.marker(tuple(poses[-1]), (0, 0, 220), arrow=True)
    canvas.label(
        f"maze {SEED}: {metrics['distance_m']:.2f} m {metrics['heading_deg']:.0f} deg, "
        f"{metrics['nav_steps']} steps, spl {metrics['spl']:.2f}, {metrics['checkin_reason']}"
    )
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="default", help="output subdirectory")
    parser.add_argument(
        "--data", default=None, help="exploration data directory (default: the local recording)"
    )
    parser.add_argument("--verify-candidates", type=int, default=0, help="see my_agent.py")
    parser.add_argument("--top-k", type=int, default=30, help="see my_agent.py")
    parser.add_argument(
        "--shortcuts-per-1k", type=float, default=SHORTCUTS_PER_1K, help="see my_agent.py"
    )
    parser.add_argument("--geom-weight", type=float, default=0.001, help="see my_agent.py")
    args = parser.parse_args()
    agent = MyAgent(
        Path(args.data) if args.data else cli.record_exploration_data(SEED),
        cache_dir=Path("cache") / (Path(args.data).name if args.data else f"local-{SEED}"),
        verify_candidates=args.verify_candidates,
        geom_weight=args.geom_weight,
        top_k_shortcuts=args.top_k,
        shortcuts_per_1k=args.shortcuts_per_1k,
    )
    out = OUT / args.tag
    run_ = drive(agent)
    world = sim.Simulator(textures(), SEED)
    distance_m, heading_deg = run_["arrival"]
    result = run_["result"]
    checked_in = result is not None
    nav_steps = run_["nav_steps"]
    oracle = world.oracle_steps
    success = checked_in and distance_m <= SUCCESS_M and abs(heading_deg) <= SUCCESS_DEG
    metrics = {
        "distance_m": float(distance_m),
        "heading_deg": float(heading_deg),
        "checked_in": checked_in,
        "checkin_reason": agent.checkin_reason,
        "nav_steps": nav_steps,
        "oracle_steps": oracle,
        "spl": oracle / max(oracle, nav_steps) if success else 0.0,
        "sdk_points": result.points if checked_in else None,
        "sdk_tier": result.goal_tier if checked_in else None,
        "sdk_spl": result.spl if checked_in else None,
        "filter_differs_ticks": sum(t["filter_node"] != t["raw_node"] for t in run_["ticks"]),
        "low_conf_ticks": sum(bool(t["low_conf"]) for t in run_["ticks"]),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    draw(world, run_, metrics, out / "path.png")
    write_csv(out / "ticks.csv", TICK_FIELDS, run_["ticks"])
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out / 'metrics.json'}, {out / 'path.png'} and {out / 'ticks.csv'}")


if __name__ == "__main__":
    main()
