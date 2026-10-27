"""
Record an exploration dataset with ground-truth poses: the robot sweeps the maze by BFS to the
nearest unvisited free cell, one frame, pose and action every --capture-every ticks.

    uv run source/eval/record_gt.py
    uv run source/eval/record_gt.py --mazes 7 --capture-every 1 --coverage 0.95

Writes data/gt/<seed>-c<k>/ in Simulator.record's layout (traj_<i>/<k>.jpg, data_info.json,
target.jpg), each record with "x", "y", "yaw" added, so load_frames reads it unchanged.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as sim
from common import (
    CACHE_ROOT,
    Cell,
    bfs,
    cell_centre,
    cell_of,
    check_mapping,
    gt_dir,
    gt_name,
    ints,
    parser,
    textures,
    wrap,
    write_csv,
)
from vlad import load_frames

ROUTES = 3
COVERAGE = 0.9
MAX_TICKS = 200_000
STUCK_MOVES = 3
"""FORWARD moves in a row that go nowhere before one BACKWARD."""
STUCK_METRES = 0.005
ACTIONS = {sim.FORWARD: "FORWARD", sim.BACKWARD: "BACKWARD", sim.LEFT: "LEFT", sim.RIGHT: "RIGHT"}

FIELDS = [
    "seed",
    "capture_every",
    "records",
    "ticks",
    "routes",
    "free_cells",
    "reachable_cells",
    "visited_cells",
    "coverage",
]


class Driver:
    """Follows a path of cells, one action per decision of ``ticks`` ticks: turns in place
    towards the next cell, drives forward, and backs off when FORWARD goes nowhere."""

    def __init__(self, world: sim.Simulator, ticks: int) -> None:
        # Turn in place while the bearing is off by more than ~half a move's turn, so a turn
        # never overshoots into turning back; a cell is reached within ~half a move's travel.
        self.world = world
        self.turn_tol = max(0.1, 0.6 * sim.ANGULAR_SPEED * sim.TICK_SECONDS * ticks)
        self.reach = max(0.04, 0.6 * sim.LINEAR_SPEED * sim.TICK_SECONDS * ticks)
        self.stuck = 0

    def advance(self, path: list[Cell], x: float, y: float) -> None:
        """Drop the cells at the front of ``path`` that ``(x, y)`` has reached."""
        while path and math.dist(cell_centre(self.world, path[0]), (x, y)) < self.reach:
            path.pop(0)

    def action(self, cell: Cell, x: float, y: float, yaw: float) -> int:
        """The action towards ``cell`` from ``(x, y, yaw)``."""
        if self.stuck >= STUCK_MOVES:
            self.stuck = 0
            return sim.BACKWARD
        wx, wy = cell_centre(self.world, cell)
        error = wrap(math.atan2(wy - y, wx - x) - yaw)
        if abs(error) > self.turn_tol:
            return sim.LEFT if error > 0 else sim.RIGHT
        return sim.FORWARD

    def moved(self, action: int, x: float, y: float) -> None:
        """After stepping ``action`` from ``(x, y)``: count a FORWARD that went nowhere."""
        if action == sim.FORWARD:
            moved = math.dist(self.world.pose()[:2], (x, y))
            self.stuck = self.stuck + 1 if moved < STUCK_METRES else 0


def explore(
    world: sim.Simulator, capture_every: int, coverage: float, max_ticks: int
) -> tuple[list[dict], set[Cell], set[Cell]]:
    """Drive the sweep. Returns the records (``jpeg`` bytes, pose and action), and the
    reachable and visited cells."""
    free = ~world.occupancy
    reachable = set(bfs(free, tuple(world.spawn_cell)))
    driver = Driver(world, capture_every)

    frame = world.initial_frame()
    records: list[dict] = []
    visited: set[Cell] = set()
    path: list[Cell] = []
    while world.steps + capture_every <= max_ticks:
        x, y, yaw = world.pose()
        cell = cell_of(world, free, x, y)
        visited.add(cell)
        if len(visited & reachable) >= coverage * len(reachable):
            break
        driver.advance(path, x, y)
        if not path or path[-1] in visited:
            path = _route_to_unvisited(free, cell, visited)
            if not path:
                break

        action = driver.action(path[0], x, y, yaw)
        records.append(
            {"jpeg": world.jpeg(frame), "x": x, "y": y, "yaw": yaw, "action": ACTIONS[action]}
        )
        frame = world.step(action, capture_every)[-1]
        driver.moved(action, x, y)
    return records, reachable, visited


def _route_to_unvisited(free: np.ndarray, start: Cell, visited: set[Cell]) -> list[Cell]:
    """The cells after ``start`` on a shortest path to the nearest unvisited cell."""
    parents = bfs(free, start)
    return route(parents, start, next((cell for cell in parents if cell not in visited), None))


def route(parents: dict[Cell, Cell | None], start: Cell, goal: Cell | None) -> list[Cell]:
    """The cells after ``start`` on the BFS tree ``parents`` (from ``start``) to ``goal``."""
    path: list[Cell] = []
    while goal is not None and goal != start:
        path.append(goal)
        goal = parents[goal]
    return path[::-1]


def save(out: Path, world: sim.Simulator, records: list[dict], routes: int) -> None:
    """Simulator.record's layout. Every record is one action held for the same number of
    ticks, so equal record counts are equal tick counts."""
    if out.exists():
        shutil.rmtree(out)
    for index, chunk in enumerate(np.array_split(np.arange(len(records)), routes)):
        directory = out / f"traj_{index}"
        directory.mkdir(parents=True)
        info = []
        for step, i in enumerate(chunk):
            r = records[int(i)]
            (directory / f"{step}.jpg").write_bytes(r["jpeg"])
            info.append(
                {
                    "step": step,
                    "image": f"{step}.jpg",
                    "action": [r["action"]],
                    "x": r["x"],
                    "y": r["y"],
                    "yaw": r["yaw"],
                }
            )
        (directory / "data_info.json").write_text(json.dumps(info, indent=2))
    # Last, so its presence marks a complete recording. world.jpeg takes single frames only;
    # this is its quality and chroma subsampling for the four views side by side.
    ok, target = cv2.imencode(
        ".jpg",
        np.hstack(world.targets()),
        [
            cv2.IMWRITE_JPEG_QUALITY,
            90,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
        ],
    )
    assert ok
    (out / "target.jpg").write_bytes(target.tobytes())


def record(
    seed: int,
    capture_every: int,
    *,
    routes: int = ROUTES,
    coverage: float = COVERAGE,
    max_ticks: int = MAX_TICKS,
) -> dict:
    out = gt_dir(seed, capture_every)
    print(f"recording {out} ...", flush=True)
    world = sim.Simulator(textures(), seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=seed)
    check_mapping(world)
    records, reachable, visited = explore(world, capture_every, coverage, max_ticks)
    save(out, world, records, routes)
    # The SIFT cache is keyed by image path, which a re-recording keeps.
    shutil.rmtree(CACHE_ROOT / gt_name(seed, capture_every), ignore_errors=True)

    frames, bounds = load_frames(out)
    if len(frames) != len(records) or len(bounds) != routes:
        raise RuntimeError(f"load_frames read {len(frames)} of {len(records)} records")
    row = {
        "seed": seed,
        "capture_every": capture_every,
        "records": len(records),
        "ticks": world.steps,
        "routes": routes,
        "free_cells": int((~world.occupancy).sum()),
        "reachable_cells": len(reachable),
        "visited_cells": len(visited & reachable),
        "coverage": f"{len(visited & reachable) / len(reachable):.3f}",
    }
    print(
        f"maze {seed}: {row['records']} records, {row['ticks']} ticks, coverage "
        f"{row['coverage']} ({row['visited_cells']}/{row['reachable_cells']} reachable cells)"
    )
    return row


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--coverage", type=float, default=COVERAGE, help="stop at this coverage")
    p.add_argument("--max-ticks", type=int, default=MAX_TICKS, help="stop after this many")
    p.add_argument("--routes", type=int, default=ROUTES, help="trajectories to split into")
    p.add_argument("--force", action="store_true", help="re-record an existing dataset")
    args = p.parse_args()

    rows = []
    for seed in ints(args.mazes):
        if (gt_dir(seed, args.capture_every) / "target.jpg").exists() and not args.force:
            print(f"maze {seed}: {gt_dir(seed, args.capture_every)} exists (--force to redo)")
            continue
        rows.append(
            record(
                seed,
                args.capture_every,
                routes=args.routes,
                coverage=args.coverage,
                max_ticks=args.max_ticks,
            )
        )
    write_csv(Path(args.out_dir) / "record_gt.csv", FIELDS, rows)


if __name__ == "__main__":
    main()
