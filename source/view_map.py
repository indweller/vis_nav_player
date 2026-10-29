"""
Plot a frozen map from an exploration data directory, offline (no server, no session):

- the occupancy grid (log-odds of at least 2 as walls),
- each route's trajectory, in its own colour,
- the spawn (0, 0, 0) and the pose ``find_goal`` gives for ``target.jpg``, each with a heading
  arrow,
- the planner's path from the spawn to that goal (with ``--compare-shortcut``, both the plain
  plan, dashed black, and the shortcut plan, solid red).

``--goal X,Y,YAW_DEG`` plans to that pose instead of the one ``find_goal`` gives.

The map is loaded from ``map_graph.npz`` when the directory has one (built and saved otherwise).

    uv run source/view_map.py --data data/<id> [--goal-camera auto|frame|photo]
        [--goal X,Y,YAW_DEG] [--compare-shortcut] [--link-m M] [--out FILE] [--show]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import frozen_map
import geometry
import map_agent
import mapping
import matplotlib
import numpy as np
import planner as planner_module
from geometry import Camera, FloorMask
from goal import find_goal

WALL_LOG_ODDS = 2.0
ARROW_M = 0.25
"""Length of a heading arrow."""
Pose = tuple[float, float, float]


def load(root: Path):
    """``(store, camera, floor)`` of the data directory, as MapAgent builds them."""
    floor = FloorMask.load(map_agent.FLOOR_MASK_PATH)
    camera_dict = dict(geometry.CAMERA)
    camera = Camera({**camera_dict, "cy": camera_dict["cy"] + mapping.HORIZON_SHIFT_PX})
    store, _ = frozen_map.get(root, camera_dict, floor)
    return store, camera, floor


def parse_pose(text: str) -> Pose:
    """``X,Y,YAW_DEG`` as ``(x, y, yaw)`` with the yaw in radians."""
    try:
        x, y, yaw_deg = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected X,Y,YAW_DEG, got {text!r}") from None
    return x, y, math.radians(yaw_deg)


def draw(ax, store, spawn: Pose, goal: Pose | None, paths: list[tuple[object, str, str]]) -> None:
    """``paths`` are ``(path, label, line style)``; the label gets the path's length."""
    grid = store.grid
    walls = grid.log_odds >= WALL_LOG_ODDS
    size = grid.log_odds.shape[0]
    low, high = grid.origin, grid.origin + size * grid.res
    ax.imshow(
        np.where(walls, 0.0, 1.0),
        cmap="gray",
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(low[0], high[0], low[1], high[1]),
        interpolation="nearest",
    )
    for r, poses in enumerate(store.poses):
        ax.plot(poses[:, 0], poses[:, 1], lw=1.2, label=f"route {r}")
    for path, label, style in paths:
        ax.plot(
            path.waypoints[:, 0],
            path.waypoints[:, 1],
            style,
            lw=2,
            label=f"{label} ({path.length:.1f} m)",
        )
    for name, pose, colour in (("spawn", spawn, "green"), ("goal", goal, "red")):
        if pose is None:
            continue
        x, y, yaw = pose
        ax.plot(x, y, "o", color=colour, ms=8, label=f"{name} ({x:.2f}, {y:.2f})")
        ax.arrow(
            x,
            y,
            ARROW_M * math.cos(yaw),
            ARROW_M * math.sin(yaw),
            color=colour,
            width=0.01,
            head_width=0.07,
            length_includes_head=True,
            zorder=5,
        )
    used = np.concatenate([p[:, :2] for p in store.poses])
    pad = 0.5
    ax.set_xlim(used[:, 0].min() - pad, used[:, 0].max() + pad)
    ax.set_ylim(used[:, 1].min() - pad, used[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=8)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="exploration data directory")
    parser.add_argument("--goal-camera", choices=("auto", "frame", "photo"), default="auto")
    parser.add_argument(
        "--goal",
        type=parse_pose,
        default=None,
        metavar="X,Y,YAW_DEG",
        help="plan to this pose instead of the one find_goal gives",
    )
    parser.add_argument(
        "--compare-shortcut",
        action="store_true",
        help="draw the plain plan (dashed black) and the shortcut plan (solid red)",
    )
    parser.add_argument(
        "--link-m",
        type=float,
        default=planner_module.LINK_M,
        help="the planner's link radius (m)",
    )
    parser.add_argument("--out", default=None, help="default: results/maps/<data dir name>.png")
    parser.add_argument("--show", action="store_true", help="also open a window")
    args = parser.parse_args(argv)
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.data)
    out = (
        Path(args.out) if args.out else map_agent.cli.ROOT / "results" / "maps" / f"{root.name}.png"
    )

    store, camera, floor = load(root)
    spawn: Pose = (0.0, 0.0, 0.0)
    if args.goal is not None:
        goal: Pose | None = args.goal
    else:
        found = find_goal(
            map_agent.goal_views(root),
            store,
            camera,
            floor,
            photographed=args.goal_camera != "frame",
        )
        goal = None if found is None else (found.x, found.y, found.yaw)
    path = None
    paths = []
    if goal is not None:
        planner = planner_module.PlaceGraph(
            store, link_m=args.link_m, cache_path=root / "links.npz"
        )
        path = planner.plan(spawn, goal)
        if path is not None and args.compare_shortcut:
            short = path.shortcut(store)
            paths = [(path, "plain plan", "k--"), (short, "shortcut plan", "r-")]
        elif path is not None:
            paths = [(path, "planned path", "k--")]
    goal_text = (
        "none"
        if goal is None
        else f"({goal[0]:.2f}, {goal[1]:.2f}) {math.degrees(goal[2]):.0f} deg"
    )
    path_text = ", ".join(f"{label} {p.length:.2f} m" for p, label, _ in paths) or "none"
    print(f"goal: {goal_text}; path: {path_text}")

    fig, ax = plt.subplots(figsize=(9, 9))
    draw(ax, store, spawn, goal, paths)
    ax.set_title(f"{root.name}: frozen map")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
