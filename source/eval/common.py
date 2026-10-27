"""
Shared pieces of the ground-truth evaluations: the arguments every script takes, the grid
<-> metre mapping, the recorded poses, and MyAgent's index built over a ground-truth dataset.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import vis_nav_sim as sim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from my_agent import MyAgent  # noqa: E402
from vlad import Frame  # noqa: E402

GT_ROOT = Path("data") / "gt"
CACHE_ROOT = Path("cache") / "gt"

METRES_PER_RADIAN = sim.LINEAR_SPEED / sim.ANGULAR_SPEED
"""Exchange rate for :func:`pose_distance`: a tick of turning weighs as much as a tick of
driving."""

Cell = tuple[int, int]
"""``(row, col)``, as ``Simulator.spawn_cell``: row along y, col along x."""


def parser(doc: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mazes", default="7,8,9", help="comma-separated maze seeds")
    parser.add_argument("--capture-every", type=int, default=1, help="ticks per record")
    parser.add_argument("--subsample", type=int, default=5, help="keep every Nth frame")
    parser.add_argument("--out-dir", default="results/eval", help="where the CSVs go")
    return parser


def ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


@functools.cache
def textures() -> sim.Textures:
    return sim.Textures.find()


def gt_name(seed: int, capture_every: int) -> str:
    return f"{seed}-c{capture_every}"


def gt_dir(seed: int, capture_every: int) -> Path:
    return GT_ROOT / gt_name(seed, capture_every)


# -- the grid


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def yaw_error_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Absolute angle between yaws, degrees in [0, 180]."""
    return np.degrees(np.abs((np.asarray(a) - b + np.pi) % (2 * np.pi) - np.pi))


def pose_distance(
    position_m: np.ndarray | float, yaw_deg: np.ndarray | float
) -> np.ndarray | float:
    """Position and yaw error as one number, in metres."""
    return position_m + np.radians(yaw_deg) * METRES_PER_RADIAN


def turn_deg_per_tick(seed: int = 7, ticks: int = 50) -> float:
    """Degrees one LEFT tick turns, without motion noise: ``ticks`` single ticks, the wrapped
    yaw change summed, so a full turn is not lost. MyAgent's TURN_DEG_PER_TICK is this."""
    world = sim.Simulator(textures(), seed)
    world.initial_frame()
    total = 0.0
    yaw = world.pose()[2]
    for _ in range(ticks):
        world.step(sim.LEFT)
        now = world.pose()[2]
        total += wrap(now - yaw)
        yaw = now
    return math.degrees(total) / ticks


def cell_centre(world: sim.Simulator, cell: Cell) -> tuple[float, float]:
    """Cells are centred on multiples of the cell size: ``x = col * size``, ``y = row * size``
    (checked by :func:`check_mapping`). Walls are thinner than a cell, so the robot can stand
    nearer a ``#`` cell's centre than any free one's."""
    return cell[1] * world.cell_size, cell[0] * world.cell_size


def cell_of(world: sim.Simulator, free: np.ndarray, x: float, y: float) -> Cell:
    """The free cell whose centre is nearest ``(x, y)``."""
    size = world.cell_size
    row, col = round(y / size), round(x / size)
    if 0 <= row < free.shape[0] and 0 <= col < free.shape[1] and free[row, col]:
        return row, col
    candidates = [
        (r, c)
        for r in range(row - 1, row + 2)
        for c in range(col - 1, col + 2)
        if 0 <= r < free.shape[0] and 0 <= c < free.shape[1] and free[r, c]
    ]
    return min(candidates, key=lambda cell: math.dist(cell_centre(world, cell), (x, y)))


def check_mapping(world: sim.Simulator) -> None:
    """The cell <-> metre mapping agrees with the simulator's own spawn and target."""
    free = ~world.occupancy
    for name, pose, cell in [
        ("spawn", world.spawn, world.spawn_cell),
        ("target", world.target, world.target_cell),
    ]:
        centre = cell_centre(world, cell)
        if math.dist(centre, pose[:2]) > 1e-3 or not free[cell]:
            raise RuntimeError(f"maze {world.seed}: {name} {pose[:2]} is not at cell {cell}")
        if cell_of(world, free, *pose[:2]) != tuple(cell):
            raise RuntimeError(f"maze {world.seed}: {name} {pose[:2]} does not map to {cell}")


def neighbours(free: np.ndarray, cell: Cell) -> list[Cell]:
    row, col = cell
    return [
        (r, c)
        for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
        if 0 <= r < free.shape[0] and 0 <= c < free.shape[1] and free[r, c]
    ]


def bfs(free: np.ndarray, start: Cell) -> dict[Cell, Cell | None]:
    """Parents of every free cell reachable from ``start``, in BFS order (4-connected)."""
    parents: dict[Cell, Cell | None] = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for nxt in neighbours(free, cell):
            if nxt not in parents:
                parents[nxt] = cell
                queue.append(nxt)
    return parents


def geodesic(world: sim.Simulator, goal: Cell) -> np.ndarray:
    """Metres along the grid from every cell to ``goal``; ``inf`` where unreachable."""
    free = ~world.occupancy
    distance = np.full(free.shape, np.inf)
    distance[goal] = 0.0
    queue = deque([goal])
    while queue:
        cell = queue.popleft()
        for nxt in neighbours(free, cell):
            if np.isinf(distance[nxt]):
                distance[nxt] = distance[cell] + world.cell_size
                queue.append(nxt)
    return distance


# -- the recorded data and the index


def build_agent(seed: int, args: argparse.Namespace, **kwargs) -> MyAgent:
    """MyAgent over ``data/gt/<seed>-c<k>``, recorded first if it is not there yet."""
    data_dir = gt_dir(seed, args.capture_every)
    if not (data_dir / "target.jpg").exists():
        from record_gt import record

        record(seed, args.capture_every)
    return MyAgent(
        data_dir,
        random_every=0,
        subsample=args.subsample,
        capture_every=args.capture_every,
        cache_dir=CACHE_ROOT / gt_name(seed, args.capture_every),
        **kwargs,
    )


def node_poses(frames: list[Frame]) -> np.ndarray:
    """``(N, 3)`` ``x, y, yaw`` per node, from ``data_info.json`` matched by image path."""
    poses: dict[Path, tuple[float, float, float]] = {}
    for directory in {frame.path.parent for frame in frames}:
        for r in json.loads((directory / "data_info.json").read_text()):
            poses[directory / r["image"]] = (r["x"], r["y"], r["yaw"])
    return np.array([poses[frame.path] for frame in frames])


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {path}")


def fmt(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"
