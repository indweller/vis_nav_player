"""
mapping.py's pose-free map against the ground truth, offline: no controller, no online runs.

    uv run source/eval/eval_map.py
    uv run source/eval/eval_map.py --mazes 7

Each maze is recorded into data/eval/<seed>/ as a challenge does (cli.record_exploration_data's
settings: DEFAULT_NOISE, motion_seed=seed, three routes, a frame a tick), with student_files=False
so that each route also has a poses.json of true poses. mapping.py never sees poses.json or the
world; only this script does. Its two variants, "odom" and "graph", are compared with the truth
in the map frame, which is the world frame moved so the spawn pose is (0, 0, 0) ("spawn" below);
"fit" is the same after the best rigid alignment of all the routes' node positions together.

Per maze, in results/eval/map/ (--out-dir/map):
  <seed>_truth.png            true walls and the three true routes, spawn and target
  <seed>_odom.png, <seed>_graph.png
                              the built occupancy grid and the estimated routes
  <seed>_overlay_<variant>.png
                              true walls (black) over the built map, aligned by the spawn pose
  <seed>_<variant>.npz        the occupancy grid
  metrics.csv                 per maze and variant: the routes' position and yaw error of the
                              nodes (median, max), wall precision and recall of the occupied
                              cells at 0.05 and 0.10 m, loop closures (candidates, proposed =
                              verified, accepted, false = accepted with the two frames more than
                              0.3 m apart in truth; wrong = accepted with a relative pose more
                              than 0.3 m or 15 degrees from the true one, which a valid closure
                              between two nearby frames that see the same wall is not), seconds
                              per stage

The images are in the world frame, the map moved by the spawn alignment.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as sim
from common import fmt, ints, textures, wrap, write_csv
from eval_cues import Walls, stored_floor
from eval_geometry import Frame, wall_cells, wall_distance
from geometry import MAP_RES, OccupancyMap, rigid_fit, transform
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mapping  # noqa: E402

DATA_ROOT = Path("data") / "eval"
TOLS = (0.05, 0.10)
FALSE_CLOSURE_M = 0.3
WRONG_CLOSURE_DEG = 15.0
SCALE = 100
"""Pixels per metre of the images."""
MARGIN = 0.3
ROUTE_COLOURS = [(200, 80, 0), (0, 140, 230), (60, 160, 0)]
"""BGR, one per route."""
MAP_COLOUR = (60, 60, 60)
OVERLAY_COLOUR = (240, 170, 80)
VARIANTS = ("odom", "graph")
STAGES = ("calibration", "odometry", "vlad", "closures", "optimise", "grid")


def record(seed: int) -> Path:
    """data/eval/<seed>/ with its poses.json files, recorded if it is not there yet."""
    root = DATA_ROOT / str(seed)
    if (root / "target.jpg").exists() and (root / "traj_0" / "poses.json").exists():
        return root
    print(f"recording {root} ...", flush=True)
    world = sim.Simulator(textures(), seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=seed)
    summary = world.record(root, routes=3, seed=seed, capture_every=1, student_files=False)
    print(f"  {summary['frames']} frames on {summary['routes']} routes")
    return root


def true_poses(root: Path, routes: list[mapping.Route]) -> list[np.ndarray]:
    """Per route, the true ``(frames, 3)`` world poses after each record's action."""
    out = []
    for t, route in enumerate(routes):
        poses = np.array(json.loads((root / f"traj_{t}" / "poses.json").read_text()))
        if len(poses) != len(route.frames):
            raise RuntimeError(f"traj_{t}: {len(poses)} poses for {len(route.frames)} frames")
        out.append(poses)
    return out


# -- metrics


def trajectory_errors(
    estimated: list[np.ndarray], truth: list[np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per route, the position (m) and yaw (degrees) error at each node, the ``truth`` poses
    being in the same frame."""
    out = []
    for est, true in zip(estimated, truth, strict=True):
        position = np.hypot(*(est[:, :2] - true[:, :2]).T)
        yaw = np.degrees(np.abs(wrap(est[:, 2] - true[:, 2])))
        out.append((position, yaw))
    return out


def best_fit(estimated: list[np.ndarray], truth: list[np.ndarray]) -> list[np.ndarray]:
    """``truth`` moved by the one rigid motion that puts the routes' node positions nearest
    the ``estimated`` ones."""
    pose = rigid_fit(np.concatenate(estimated)[:, :2], np.concatenate(truth)[:, :2])
    return [np.column_stack([transform(pose, t[:, :2]), t[:, 2] + pose[2]]) for t in truth]


def wall_scores(
    grid: OccupancyMap, frame: Frame, walls: Walls, truth_xy: np.ndarray
) -> dict[str, float]:
    """Precision (occupied cells within tolerance of a true wall) and recall (true wall cells
    within tolerance of an occupied cell) at each of TOLS."""
    occupied = grid.occupied()
    precision = wall_distance(walls, frame.to_world(grid.centres()[occupied]))
    to_map = distance_transform_edt(~occupied, sampling=grid.res)
    row, col, inside = grid.cells(frame.to_map(truth_xy))
    recall = np.full(len(truth_xy), np.inf)
    recall[inside] = to_map[row[inside], col[inside]]
    out = {"map_cells": float(occupied.sum())}
    for tol in TOLS:
        out[f"precision_{tol:.2f}"] = float(np.mean(precision <= tol)) if occupied.any() else 0.0
        out[f"recall_{tol:.2f}"] = float(np.mean(recall <= tol))
    return out


def closure_counts(result: mapping.MapResult, truth: list[np.ndarray], variant: str) -> dict:
    if variant != "graph":
        return {
            "closure_candidates": 0,
            "closures_proposed": 0,
            "closures_accepted": 0,
            "closures_false": 0,
            "closures_wrong": 0,
        }
    accepted = [c for c in result.closures if c.accepted]
    false = sum(
        math.dist(truth[c.a[0]][c.a[1], :2], truth[c.b[0]][c.b[1], :2]) > FALSE_CLOSURE_M
        for c in accepted
    )
    wrong = 0
    for c in accepted:
        measured = mapping.between(tuple(truth[c.a[0]][c.a[1]]), tuple(truth[c.b[0]][c.b[1]]))
        wrong += (
            math.hypot(measured[0] - c.pose[0], measured[1] - c.pose[1]) > FALSE_CLOSURE_M
            or abs(math.degrees(wrap(measured[2] - c.pose[2]))) > WRONG_CLOSURE_DEG
        )
    return {
        "closure_candidates": result.candidates,
        "closures_proposed": len(result.closures),
        "closures_accepted": len(accepted),
        "closures_false": false,
        "closures_wrong": wrong,
    }


# -- images


class Canvas:
    """A white image of the world, +y up."""

    def __init__(self, world: sim.Simulator) -> None:
        extent = (np.array(world.occupancy.shape[::-1]) - 1) * world.cell_size
        self.low = -MARGIN
        size = np.ceil((extent + 2 * MARGIN) * SCALE).astype(int)
        self.image = np.full((size[1], size[0], 3), 255, np.uint8)

    def pixels(self, xy: np.ndarray) -> np.ndarray:
        px = (np.asarray(xy, float) - self.low) * SCALE
        return np.stack([px[..., 0], self.image.shape[0] - 1 - px[..., 1]], axis=-1)

    def cells(self, xy: np.ndarray, colour: tuple[int, int, int]) -> None:
        """Occupied map cells (their world centres), each as a square of the grid's size."""
        pixels = np.round(self.pixels(xy)).astype(int)
        inside = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < self.image.shape[1])
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < self.image.shape[0])
        )
        mask = np.zeros(self.image.shape[:2], np.uint8)
        mask[pixels[inside, 1], pixels[inside, 0]] = 1
        size = max(int(round(MAP_RES * SCALE)), 1)
        mask = cv2.dilate(mask, np.ones((size, size), np.uint8))
        self.image[mask > 0] = colour

    def walls(self, walls: Walls, colour: tuple[int, int, int]) -> None:
        for x0, x1, y0, y1 in walls.rects:
            (u0, v0), (u1, v1) = self.pixels(np.array([[x0, y0], [x1, y1]]))
            cv2.rectangle(
                self.image,
                (int(round(u0)), int(round(v1))),
                (max(int(round(u1)), int(round(u0)) + 1), max(int(round(v0)), int(round(v1)) + 1)),
                colour,
                -1,
            )

    def routes(self, routes: list[np.ndarray]) -> None:
        for t, poses in enumerate(routes):
            line = np.round(self.pixels(poses[:, :2])).astype(np.int32)
            cv2.polylines(
                self.image,
                [line.reshape(-1, 1, 2)],
                False,
                ROUTE_COLOURS[t % len(ROUTE_COLOURS)],
                1,
                cv2.LINE_AA,
            )

    def marker(self, pose: tuple[float, float, float], colour, *, arrow: bool = False) -> None:
        centre = tuple(int(v) for v in np.round(self.pixels(np.array(pose[:2]))))
        cv2.circle(self.image, centre, 5, colour, -1, cv2.LINE_AA)
        if arrow:
            tip = self.pixels(
                np.array(pose[:2]) + 0.3 * np.array([math.cos(pose[2]), math.sin(pose[2])])
            )
            cv2.arrowedLine(
                self.image, centre, tuple(int(v) for v in np.round(tip)), colour, 2, tipLength=0.3
            )

    def label(self, text: str) -> None:
        cv2.putText(self.image, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.image)


def world_poses(frame: Frame, poses: list[np.ndarray]) -> list[np.ndarray]:
    return [np.column_stack([frame.to_world(p[:, :2]), p[:, 2] - frame.phi]) for p in poses]


def draw_truth(
    world: sim.Simulator, walls: Walls, truth: list[np.ndarray], seed: int, path: Path
) -> None:
    canvas = Canvas(world)
    canvas.walls(walls, (0, 0, 0))
    canvas.routes([t[:: mapping.NODE_EVERY] for t in truth])
    canvas.marker(world.spawn, (0, 160, 0))
    canvas.marker(world.target, (0, 0, 220), arrow=True)
    canvas.label(f"maze {seed}: truth (spawn green, target red)")
    canvas.save(path)


def draw_variant(
    world: sim.Simulator,
    walls: Walls,
    frame: Frame,
    variant: mapping.Variant,
    seed: int,
    name: str,
    out: Path,
) -> None:
    grid = variant.grid
    centres = frame.to_world(grid.centres()[grid.occupied()])
    canvas = Canvas(world)
    canvas.cells(centres, MAP_COLOUR)
    canvas.routes(world_poses(frame, variant.poses))
    canvas.marker(world.spawn, (0, 160, 0))
    canvas.label(f"maze {seed}: {name}")
    canvas.save(out / f"{seed}_{name}.png")

    overlay = Canvas(world)
    overlay.cells(centres, OVERLAY_COLOUR)
    overlay.walls(walls, (0, 0, 0))
    overlay.label(f"maze {seed}: true walls over the {name} map")
    overlay.save(out / f"{seed}_overlay_{name}.png")


# -- the run


def evaluate(seed: int, floor, out: Path) -> list[dict]:
    root = record(seed)
    world = sim.Simulator(textures(), seed)
    walls = Walls(world)
    routes = mapping.load_routes(root)
    truth = true_poses(root, routes)
    print(f"maze {seed}: {sum(len(r.frames) for r in routes)} frames on {len(routes)} routes")

    result = mapping.build(routes, world.camera, floor)

    result.save(out, str(seed))
    frame = Frame((0.0, 0.0, 0.0), world.spawn)
    truth_nodes = [frame.to_map(t[:: mapping.NODE_EVERY]) for t in truth]
    truth_xy = wall_cells(world, walls)
    draw_truth(world, walls, truth, seed, out / f"{seed}_truth.png")

    rows = []
    for name, variant in result.variants.items():
        row: dict = {"seed": seed, "variant": name}
        spawn = trajectory_errors(variant.poses, truth_nodes)
        fit = trajectory_errors(variant.poses, best_fit(variant.poses, truth_nodes))
        for label, errors in (("spawn", spawn), ("fit", fit)):
            for r, (position, yaw) in enumerate(errors):
                prefix = f"ate_{label}_r{r}"
                row[f"{prefix}_pos_med_m"] = float(np.median(position))
                row[f"{prefix}_pos_max_m"] = float(position.max())
                row[f"{prefix}_yaw_med_deg"] = float(np.median(yaw))
                row[f"{prefix}_yaw_max_deg"] = float(yaw.max())
        row.update(wall_scores(variant.grid, frame, walls, truth_xy))
        row.update(closure_counts(result, truth, name))
        shared = {s: result.seconds.get(s, 0.0) for s in STAGES}
        if name == "odom":
            shared.update(vlad=0.0, closures=0.0, optimise=0.0)
        row.update({f"seconds_{s}": v for s, v in shared.items()})
        row["odometry_visual"] = result.counts["visual"]
        row["odometry_fallback"] = result.counts["fallback"]
        rows.append(row)
        draw_variant(world, walls, frame, variant, seed, name, out)
    return rows


def route_values(row: dict, label: str, key: str) -> list[float]:
    return [row[f"ate_{label}_r{r}_{key}"] for r in range(3)]


def summarise(rows: list[dict]) -> None:
    head = (
        f"{'maze':>4} {'variant':<6} | {'spawn med m (r0/r1/r2)':<23} {'max m':>6} "
        f"{'yaw max':>7} | {'fit med m (r0/r1/r2)':<21} {'max m':>6} | "
        f"{'P/R 0.05':>11} {'P/R 0.10':>11} | "
        f"{'cand':>5} {'prop':>5} {'acc':>4} {'far':>4} {'wrong':>5} | {'sec':>6}"
    )
    print("\n" + head)
    print("-" * len(head))
    for row in rows:
        routes = functools.partial(route_values, row)
        spawn_med = "/".join(fmt(v, 2) for v in routes("spawn", "pos_med_m"))
        fit_med = "/".join(fmt(v, 2) for v in routes("fit", "pos_med_m"))
        seconds = sum(v for k, v in row.items() if k.startswith("seconds_"))
        print(
            f"{row['seed']:>4} {row['variant']:<6} | {spawn_med:<23} "
            f"{fmt(max(routes('spawn', 'pos_max_m')), 2):>6} "
            f"{fmt(max(routes('spawn', 'yaw_max_deg')), 1):>7} | {fit_med:<21} "
            f"{fmt(max(routes('fit', 'pos_max_m')), 2):>6} | "
            f"{fmt(row['precision_0.05'], 2):>5}/{fmt(row['recall_0.05'], 2):<5} "
            f"{fmt(row['precision_0.10'], 2):>5}/{fmt(row['recall_0.10'], 2):<5} | "
            f"{row['closure_candidates']:>5} {row['closures_proposed']:>5} "
            f"{row['closures_accepted']:>4} {row['closures_false']:>4} "
            f"{row['closures_wrong']:>5} | {seconds:>6.0f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mazes", default="7,8,9", help="comma-separated maze seeds")
    p.add_argument("--out-dir", default="results/eval", help="where map/ goes")
    args = p.parse_args()

    out = Path(args.out_dir) / "map"
    floor = stored_floor()
    rows = []
    for seed in ints(args.mazes):
        rows.extend(evaluate(seed, floor, out))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(out / "metrics.csv", fields, rows)
    summarise(rows)


if __name__ == "__main__":
    main()
