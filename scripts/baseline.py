"""Record the refactoring baseline into results/baseline/: what scripts/check_baseline.py
recomputes and compares after a change.

    uv run scripts/baseline.py

For each of DATA (the --test-local path: MapAgent's defaults, its frozen map and caches):
find_goal's pose on target.jpg, a SHA-256 of the map's poses (the poses themselves go to
poses_<data>.npz), and the plan length from the spawn to that goal at each of LINK_MS. Then the
midterm-like evaluation (eval_dev_like.py with EVAL_ARGS): per goal and in total, from its runs.csv.
Writes baseline.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

import numpy as np  # noqa: E402
import planner as planner_module  # noqa: E402
from goal import find_goal  # noqa: E402
from map_agent import MapAgent, goal_views  # noqa: E402

OUT = ROOT / "results" / "baseline"
DATA = ("515b23a3c14a", "da90a443e215")
LINK_MS = (0.12, 0.25)
SPAWN = (0.0, 0.0, 0.0)
EVAL_SEED = 10
EVAL_ARGS = [
    "--seed", str(EVAL_SEED), "--maze-size", "31", "--routes", "3", "--capture-every", "1",
    "--goals", "10", "--repeats", "1", "--agent-args", "--link-m 0.25",
]  # fmt: skip
EVAL_FIELDS = (
    "outcome",
    "checkin_reason",
    "points",
    "success",
    "nav_steps",
    "final_distance_m",
    "heading_err_deg",
    "found_x",
    "found_y",
    "found_yaw_deg",
)
"""The runs.csv columns kept per goal."""


def poses_sha256(poses: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for p in poses:
        digest.update(np.ascontiguousarray(p, np.float64).tobytes())
    return digest.hexdigest()


def data_folder(name: str) -> tuple[dict, list[np.ndarray]]:
    """The goal, poses hash and plan lengths of ``data/<name>``, and the map's poses."""
    root = ROOT / "data" / name
    agent = MapAgent(root)
    found = find_goal(goal_views(root), agent.store, agent.camera, agent.floor, photographed=True)
    goal = (found.x, found.y, found.yaw)
    plan_m = {}
    for link_m in LINK_MS:
        graph = planner_module.PlaceGraph(agent.store, link_m=link_m, cache_path=root / "links.npz")
        path = graph.plan(SPAWN, goal)
        plan_m[str(link_m)] = None if path is None else path.length
    summary = {
        "goal": [found.x, found.y, math.degrees(found.yaw)],
        "poses_sha256": poses_sha256(agent.store.poses),
        "plan_m": plan_m,
    }
    return summary, agent.store.poses


def number(text: str) -> float | str:
    try:
        return float(text)
    except ValueError:
        return text


def midterm_eval(tag: str) -> dict:
    """Run eval_dev_like.py with EVAL_ARGS into results/eval/dev_like/<seed>-<tag>/ and
    summarise its runs.csv: EVAL_FIELDS per goal, and the totals."""
    script = ROOT / "source" / "eval" / "eval_dev_like.py"
    subprocess.run([sys.executable, str(script), *EVAL_ARGS, "--tag", tag], cwd=ROOT, check=True)
    runs = ROOT / "results" / "eval" / "dev_like" / f"{EVAL_SEED}-{tag}" / "runs.csv"
    with runs.open() as f:
        rows = list(csv.DictReader(f))
    goals = {row["goal"]: {k: number(row[k]) for k in EVAL_FIELDS} for row in rows}
    totals = {
        "runs": len(rows),
        "points": sum(g["points"] for g in goals.values()),
        "successes": sum(g["success"] for g in goals.values()),
        "nav_steps": sum(g["nav_steps"] for g in goals.values()),
    }
    return {"goals": goals, "totals": totals}


def record(tag: str) -> tuple[dict, dict[str, list[np.ndarray]]]:
    """The baseline summary, and the map poses per data folder."""
    summary, poses = {"data": {}}, {}
    for name in DATA:
        summary["data"][name], poses[name] = data_folder(name)
    summary["eval"] = midterm_eval(tag)
    return summary, poses


def main() -> None:
    summary, poses = record("baseline")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, p in poses.items():
        np.savez(OUT / f"poses_{name}.npz", *p)
    (OUT / "baseline.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
