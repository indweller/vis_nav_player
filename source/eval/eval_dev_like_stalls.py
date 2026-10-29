"""Diagnose stalls and first contacts in dev-like runs (ticks.csv, plans_*.json).

    uv run source/eval/eval_dev_like_stalls.py --tags d_base,d_floor,d_zero,d_online

Reads results/eval/dev_like/<seed>-<tag>/ (ticks.csv, plans_<goal>_r<repeat>.json) of runs made
with eval_dev_like.py, and writes stalls.md and stall_<tag>_g<goal>_<n>.png beside them in
results/eval/dev_like/<seed>-stalls/. All distances are true: the true poses and the true walls
(eval_cues.Walls, rectangles with a 0.0125 m half thickness).

1. Stalls: a run of at least STALL_TICKS ticks whose true position stays within STALL_M of the
   run's first tick (greedy: from tick a, extend while the true position is within STALL_M of
   tick a's). Turning in place counts (no progress); the pattern column tells it apart. Per stall
   the tick range, the action pattern (FORWARD / guard-BACKWARD loop, repeated FORWARD, turn
   oscillation, turning one way, turn mix, other) and the ticks each branch fired on: the floor
   guard (``guard_cause`` floor), the centre-range guard, the zero motion of --blocked-zero
   (``vo_source`` zero), the online median of --online-motion (``vo_source`` online), the
   calibrated fallback, and plain visual odometry.
2. First contact of each run (the first tick whose FORWARD / BACKWARD moved under 20% of a
   tick), at the tick before it, when the action was chosen: the planned path's true distance to
   the nearest wall at the point the agent tracks (the nearest point of the path, in the map
   frame, to its estimate) and the least over the next 0.3 m of it; the true cross-track error
   (the true position's distance to the nearest point of the path); its distance to the tracked
   point; the true heading error against the path direction there and against the waypoint
   0.15 m ahead (what the agent steers to); the estimate's position and yaw error against the
   map-consistent truth (``cm_*``).
3. The distribution of the true wall distance along every planned path (sampled every 1 cm, each
   plan of each run): the share of path length within 0.1 / 0.15 / 0.2 m of a wall.
4. A picture per stall: the true walls, the plan active at the stall's start, the true trajectory
   from 100 ticks before the stall to its end (the stall in red; black before it) and the
   estimated one (blue, placed by each tick's map-consistent transform, so its offset from the
   true one is the localisation error).

A plan is placed in the world by its nodes (``plans_*.json``: the mapped frames the planner
routes through): each node's true position, joined by straight lines. The frozen map is
distorted (frames some hundreds of ticks in are metres from where their true poses put them), so
no map-to-world transform exists; the nodes' true positions are where the plan physically sends
the robot. A position along the plan, tracked in the map frame by its arc length, is found in the
world by the same arc length between the nodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from common import textures
from eval_cues import Walls

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_dev_like import COLS, DATA_ROOT, MSAA, ROWS, DevSimulator  # noqa: E402
from mapping import wrap  # noqa: E402

ROOT = Path("results") / "eval" / "dev_like"
STALL_TICKS = 20
STALL_M = 0.05
BEFORE = 100
SAMPLE_M = 0.01
LOOKAHEAD_M = 0.15
BANDS = (0.1, 0.15, 0.2)
PIXELS_PER_M = 400
JUMP_M = 0.5


def md(header: list[str], rows: list[list]) -> str:
    sep = "|" + "|".join("---" for _ in header) + "|"
    body = ("| " + " | ".join(map(str, r)) + " |" for r in rows)
    return "\n".join(["| " + " | ".join(header) + " |", sep, *body])


def clearance(walls: Walls, points: np.ndarray) -> np.ndarray:
    """Distance from each of ``points`` (n, 2) to the nearest wall rectangle."""
    x0, x1, y0, y1 = (walls.rects[:, i][None, :] for i in range(4))
    x, y = points[:, 0:1], points[:, 1:2]
    dx = np.maximum(np.maximum(x0 - x, x - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - y, y - y1), 0.0)
    return np.hypot(dx, dy).min(axis=1)


def resample(path: np.ndarray, step: float = SAMPLE_M) -> np.ndarray:
    """Points every ``step`` metres along the polyline ``path`` (n, 2)."""
    if len(path) < 2:
        return path.copy()
    seg = np.hypot(*np.diff(path, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, arc[-1], step)
    return np.column_stack([np.interp(s, arc, path[:, 0]), np.interp(s, arc, path[:, 1])])


def project(path: np.ndarray, p: np.ndarray) -> tuple[float, float, np.ndarray]:
    """``(distance, arc length at the nearest point, unit direction there)`` of ``p`` to the
    polyline ``path``."""
    a, b = path[:-1], path[1:]
    d = b - a
    length2 = (d**2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(length2 > 0, ((p - a) * d).sum(axis=1) / length2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    near = a + t[:, None] * d
    dist = np.hypot(*(near - p).T)
    i = int(dist.argmin())
    arc = np.concatenate([[0.0], np.cumsum(np.sqrt(length2))])
    direction = d[i] / math.sqrt(length2[i]) if length2[i] > 0 else np.array([1.0, 0.0])
    return float(dist[i]), float(arc[i] + t[i] * math.sqrt(length2[i])), direction


def point_at(path: np.ndarray, arc_at: float) -> np.ndarray:
    seg = np.hypot(*np.diff(path, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    s = min(arc_at, arc[-1])
    return np.array([np.interp(s, arc, path[:, 0]), np.interp(s, arc, path[:, 1])])


def read_run(directory: Path) -> dict[tuple[int, int], list[dict]]:
    runs: dict[tuple[int, int], list[dict]] = {}
    with (directory / "ticks.csv").open() as handle:
        for row in csv.DictReader(handle):
            runs.setdefault((int(row["goal"]), int(row["repeat"])), []).append(row)
    return runs


def find_stalls(xy: np.ndarray) -> list[tuple[int, int]]:
    """``(first, last)`` index pairs of the stalls in the positions ``xy``."""
    stalls, a, n = [], 0, len(xy)
    while a < n:
        b = a
        while b + 1 < n and math.dist(xy[b + 1], xy[a]) < STALL_M:
            b += 1
        if b - a + 1 >= STALL_TICKS:
            stalls.append((a, b))
            a = b + 1
        else:
            a += 1
    return stalls


def pattern(rows: list[dict]) -> tuple[str, str]:
    """``(label, detail)`` of the actions of a stall's ticks."""
    actions = [r["action"] for r in rows]
    n = len(actions)
    count = Counter(actions)
    guards = sum(r["guard"] == "1" for r in rows)
    turns = count["LEFT"] + count["RIGHT"]
    flips = sum(
        a != b for a, b in zip(actions, actions[1:], strict=False) if {a, b} == {"LEFT", "RIGHT"}
    )
    detail = ", ".join(f"{a} {c}" for a, c in count.most_common())
    if guards and count["BACKWARD"] >= guards:
        label = "FORWARD / guard-BACKWARD loop"
    elif count["FORWARD"] >= 0.8 * n:
        label = "repeated FORWARD"
    elif turns >= 0.6 * n and flips >= 0.1 * turns:
        label = "turn oscillation"
    elif turns >= 0.6 * n:
        label = (
            "turning one way" if min(count["LEFT"], count["RIGHT"]) < 0.1 * turns else "turn mix"
        )
    else:
        label = "other"
    return label, detail


def branches(rows: list[dict]) -> dict[str, int]:
    out = Counter()
    for r in rows:
        if r["guard"] == "1":
            out["guard: " + (r["guard_cause"] or "?")] += 1
        if r["vo_source"]:
            out["step: " + r["vo_source"]] += 1
    return dict(out)


class Plan:
    """A plan of ``plans_*.json``: its map-frame waypoints (start, nodes, goal) and, in the
    world, the true positions of its nodes."""

    def __init__(self, entry: dict, truth: np.ndarray) -> None:
        self.map = np.array(entry["map"])
        self.world_nodes = truth[entry["rows"], :2]
        steps = np.hypot(*np.diff(self.map, axis=0).T) if len(self.map) > 1 else np.zeros(0)
        self.arc = np.concatenate([[0.0], np.cumsum(steps)])
        self.node_arc = self.arc[1:-1]
        self.dense_map = resample(self.map) if len(self.map) > 1 else self.map
        self.dense = resample(self.world_nodes) if len(self.world_nodes) > 1 else self.world_nodes

    def world_at(self, arc: np.ndarray | float) -> np.ndarray:
        """World points ``(n, 2)`` at map arc lengths ``arc``, between the nodes' true positions."""
        arc = np.atleast_1d(arc)
        return np.column_stack(
            [np.interp(arc, self.node_arc, self.world_nodes[:, k]) for k in (0, 1)]
        )


def place(points: np.ndarray, row: dict) -> np.ndarray:
    """Map-frame ``points`` (n, 2) in the world by the rigid motion that takes the tick's
    map-consistent truth (``cm_*``) to its true pose: exact near the robot only."""
    cm = np.array([float(row["cm_x"]), float(row["cm_y"])])
    true = np.array([float(row["true_x"]), float(row["true_y"])])
    turn = float(row["true_yaw"]) - float(row["cm_yaw"])
    c, s = math.cos(turn), math.sin(turn)
    d = points - cm
    return true + np.column_stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]])


def draw(walls, rows, plan, window, out: Path, title: str) -> None:
    """The true walls, ``plan``, and the trajectories of ``rows`` (the stall ``window`` red)."""
    pts = np.array([[float(r["true_x"]), float(r["true_y"])] for r in rows])
    est = np.vstack([place(np.array([[float(r["est_x"]), float(r["est_y"])]]), r) for r in rows])
    path = None
    if plan is not None and len(plan.dense_map) > 1:
        # The plan near where the stall starts, placed by that tick's transform (local only).
        row = rows[window[0]]
        near = np.hypot(*(plan.dense_map - [float(row["cm_x"]), float(row["cm_y"])]).T) <= 1.5
        if near.any():
            path = place(plan.dense_map[near], row)
    lo = np.minimum(pts.min(axis=0), est.min(axis=0)) - 0.3
    hi = np.maximum(pts.max(axis=0), est.max(axis=0)) + 0.3
    if path is not None:
        lo, hi = np.minimum(lo, path.min(axis=0) - 0.1), np.maximum(hi, path.max(axis=0) + 0.1)
    scale = min(PIXELS_PER_M, 900 / max(hi - lo))
    size = np.ceil((hi - lo) * scale).astype(int)
    image = np.full((size[1] + 30, size[0], 3), 255, np.uint8)

    def line(points, colour, thickness: int) -> None:
        pixels = np.array([px(p) for p in points], np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pixels], False, colour, thickness, cv2.LINE_AA)

    def px(p) -> tuple[int, int]:
        p = (np.asarray(p) - lo) * scale
        return int(round(p[0])), int(round(size[1] - p[1])) + 30

    for x0, x1, y0, y1 in walls.rects:
        if x1 < lo[0] or x0 > hi[0] or y1 < lo[1] or y0 > hi[1]:
            continue
        cv2.rectangle(image, px((x0, y1)), px((x1, y0)), (120, 120, 120), -1)
    if path is not None:
        for q in path[::5]:
            cv2.circle(image, px(q), 1, (0, 140, 255), -1)
    line(est, (220, 100, 0), 1)
    first, last = window
    line(pts[: first + 1], (0, 0, 0), 2)
    line(pts[first : last + 1], (0, 0, 220), 2)
    for p, colour in ((pts[0], (0, 170, 0)), (pts[-1], (200, 0, 200))):
        cv2.drawMarker(image, px(p), colour, cv2.MARKER_TILTED_CROSS, 12, 2)
    yaw = float(rows[last]["true_yaw"])
    tip = pts[last] + 0.08 * np.array([math.cos(yaw), math.sin(yaw)])
    cv2.arrowedLine(image, px(pts[last]), px(tip), (0, 0, 220), 2, tipLength=0.4)
    cv2.putText(image, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), image)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--tags", default="d_base,d_floor,d_zero,d_online")
    args = p.parse_args()
    out = ROOT / f"{args.seed}-stalls"
    out.mkdir(parents=True, exist_ok=True)
    world = DevSimulator(textures(), args.seed, msaa=MSAA)
    assert world.occupancy.shape == (ROWS, COLS)
    walls = Walls(world)
    root = DATA_ROOT / str(args.seed)
    truth = np.concatenate(
        [np.array(json.loads(f.read_text())) for f in sorted(root.glob("traj_*/poses.json"))]
    )

    stall_lines, contact_rows, share_rows, jump_rows = [], [], [], []
    for tag in args.tags.split(","):
        directory = ROOT / f"{args.seed}-{tag}"
        if not (directory / "ticks.csv").exists():
            continue
        shares, first_shares = {b: 0.0 for b in BANDS}, {b: 0.0 for b in BANDS}
        total = first_total = 0.0
        n_plans = jump_plans = jump_edges = edges = 0
        attach: list[float] = []
        for (goal, repeat), rows in read_run(directory).items():
            xy = np.array([[float(r["true_x"]), float(r["true_y"])] for r in rows])
            plans = {
                int(k): Plan(v, truth)
                for k, v in json.loads(
                    (directory / f"plans_{goal}_r{repeat}.json").read_text()
                ).items()
            }
            # 3. wall distance along the traversed stretches of the plans: as the agent's progress
            # (its estimate projected on the plan, in the map frame) advances, the plan's points it
            # passes are placed by that tick's transform.
            reached: dict[int, int] = {}
            for row in rows:
                pid = int(row["path_id"])
                plan = plans.get(pid)
                if plan is None or len(plan.map) < 2:
                    continue
                est_xy = np.array([float(row["est_x"]), float(row["est_y"])])
                index = min(int(project(plan.map, est_xy)[1] / SAMPLE_M), len(plan.dense_map) - 1)
                seen = reached.setdefault(pid, index)
                if index <= seen:
                    continue
                near = clearance(walls, place(plan.dense_map[seen : index + 1], row))
                reached[pid] = index
                total += len(near) * SAMPLE_M
                for b in BANDS:
                    shares[b] += float((near <= b).sum() * SAMPLE_M)
                if pid == 0:
                    first_total += len(near) * SAMPLE_M
                    for b in BANDS:
                        first_shares[b] += float((near <= b).sum() * SAMPLE_M)
            # 3b. plan edges that send the robot to another place: consecutive nodes whose true
            # positions are more than JUMP_M apart.
            for plan in plans.values():
                if len(plan.world_nodes) < 2:
                    continue
                n_plans += 1
                gaps = np.hypot(*np.diff(plan.world_nodes, axis=0).T)
                jump_edges += int((gaps > JUMP_M).sum())
                jump_plans += int((gaps > JUMP_M).any())
                edges += len(gaps)
            # The attachment: how far the plan's first node is, truly, from the robot where the plan
            # was made (its first tick).
            made: dict[int, dict] = {}
            for row in rows:
                made.setdefault(int(row["path_id"]), row)
            for i, plan in plans.items():
                if i in made and len(plan.world_nodes):
                    here = [float(made[i]["true_x"]), float(made[i]["true_y"])]
                    attach.append(math.dist(plan.world_nodes[0], here))
            # 2. first contact.
            contact = next((i for i, r in enumerate(rows) if r["blocked"] == "1"), None)
            if contact is None or contact == 0:
                contact_rows.append([tag, goal, "none", *[""] * 8])
            else:
                before = rows[contact - 1]
                plan = plans.get(int(before["path_id"]))
                pose = np.array([float(before[k]) for k in ("true_x", "true_y")])
                yaw = float(before["true_yaw"])
                if plan is not None and len(plan.map) >= 2:
                    est_xy = np.array([float(before["est_x"]), float(before["est_y"])])
                    arc = project(plan.map, est_xy)[1]
                    index = min(int(arc / SAMPLE_M), len(plan.dense_map) - 1)
                    step = int(LOOKAHEAD_M / SAMPLE_M)
                    stretch = place(plan.dense_map[max(index - 100, 0) : index + 101], before)
                    tracked = place(plan.dense_map[index : index + 1], before)[0]
                    ahead = place(plan.dense_map[index : index + 31], before)
                    wp = place(
                        plan.dense_map[min(index + step, len(plan.dense_map) - 1)][None], before
                    )[0]
                    to_wp = math.atan2(wp[1] - pose[1], wp[0] - pose[0])
                    tangent = math.atan2(wp[1] - tracked[1], wp[0] - tracked[0])
                    path_cols = [
                        f"{clearance(walls, tracked[None])[0]:.3f}",
                        f"{clearance(walls, ahead).min():.3f}",
                        f"{project(stretch, pose)[0]:.3f}",
                        f"{math.dist(tracked, pose):.3f}",
                        f"{math.degrees(wrap(tangent - yaw)):+.0f}",
                        f"{math.degrees(wrap(to_wp - yaw)):+.0f}",
                    ]
                else:
                    path_cols = ["n/a"] * 6
                est_err = math.hypot(
                    float(before["est_x"]) - float(before["cm_x"]),
                    float(before["est_y"]) - float(before["cm_y"]),
                )
                yaw_err = math.degrees(wrap(float(before["est_yaw"]) - float(before["cm_yaw"])))
                contact_rows.append(
                    [
                        tag,
                        goal,
                        f"{rows[contact]['tick']} (the {before['action']} chosen at "
                        f"{before['tick']})",
                        *path_cols,
                        f"{est_err:.3f}",
                        f"{yaw_err:+.0f}",
                    ]
                )
            # 1. stalls and 4. their pictures.
            for n, (a, b) in enumerate(find_stalls(xy), 1):
                label, detail = pattern(rows[a : b + 1])
                fired = branches(rows[a : b + 1])
                start = max(a - BEFORE, 0)
                png = f"stall_{tag}_g{goal}_{n}.png"
                draw(
                    walls,
                    rows[start : b + 1],
                    plans.get(int(rows[a]["path_id"])),
                    (a - start, b - start),
                    out / png,
                    f"{tag} goal {goal} ticks {rows[a]['tick']}-{rows[b]['tick']}: {label}",
                )
                stall_lines.append(
                    [
                        tag,
                        goal,
                        f"{rows[a]['tick']}-{rows[b]['tick']}",
                        b - a + 1,
                        label,
                        detail,
                        ", ".join(f"{k} {v}" for k, v in sorted(fired.items())),
                        rows[a]["mode"],
                        f"[png]({png})",
                    ]
                )
        far = sum(a > JUMP_M for a in attach)
        jump_rows.append(
            [
                tag,
                n_plans,
                jump_plans,
                f"{jump_edges} of {edges}",
                f"{far} of {len(attach)}",
                f"{np.median(attach):.2f}" if attach else "n/a",
            ]
        )
        share_rows.append(
            [tag, f"{total:.1f}"]
            + [f"{shares[b] / total:.1%}" if total else "n/a" for b in BANDS]
            + [f"{first_total:.1f}"]
            + [f"{first_shares[b] / first_total:.1%}" if first_total else "n/a" for b in BANDS]
        )
    lines = [
        "# Stalls and first contacts",
        "",
        f"Stall: at least {STALL_TICKS} ticks with the true position within {STALL_M} m of the "
        "first (turning in place counts). 'step: ...' counts the ticks whose localiser step came "
        "from visual odometry (vo), the calibrated fallback, the online median or zero motion; "
        "'guard: ...' the wall guard's ticks by cause.",
        "",
        "## 1. Stalls",
        "",
        md(
            ["run", "goal", "ticks", "n", "pattern", "actions", "branches fired", "mode", "plot"],
            stall_lines or [["none", "", "", "", "", "", "", "", ""]],
        ),
        "",
        "## 2. First contact (state when the action was chosen)",
        "",
        md(
            [
                "run",
                "goal",
                "contact tick",
                "wall dist at tracked point m",
                "min wall dist next 0.3 m m",
                "true cross-track m",
                "true dist to tracked point m",
                "path dir - yaw deg",
                "waypoint bearing - yaw deg",
                "est pos err m",
                "est yaw err deg",
            ],
            contact_rows,
        ),
        "",
        "Heading errors are positive when the path is to the left of the true yaw. Estimate "
        "errors are against the map-consistent truth.",
        "",
        "## 3. True wall distance along the traversed stretches of the planned paths",
        "",
        md(
            [
                "run",
                "traversed m (all plans)",
                "<= 0.1 m",
                "<= 0.15 m",
                "<= 0.2 m",
                "traversed m (first plans)",
                "<= 0.1 m",
                "<= 0.15 m",
                "<= 0.2 m",
            ],
            share_rows,
        ),
        "",
        "The stretches the agent's progress passed (its estimate projected on the plan), placed "
        "in the world tick by tick; a plan's later stretches cannot be placed (see below).",
        "",
        "## 3b. Plan edges between nodes that are not in one place",
        "",
        md(
            [
                "run",
                "plans",
                "plans with a jump",
                f"node-to-node edges > {JUMP_M} m apart (true positions)",
                f"plans whose first node is > {JUMP_M} m from the robot",
                "median robot-to-first-node m",
            ],
            jump_rows,
        ),
        "",
        "Consecutive nodes of a plan are 'neighbours' on the map, but their true positions can "
        "be metres apart: the map overlays places the robot cannot go between.",
    ]
    text = "\n".join(lines) + "\n"
    (out / "stalls.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
