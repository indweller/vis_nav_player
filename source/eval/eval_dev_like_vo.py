"""Why the localiser falls back from visual odometry, from a dev-like run's ticks.csv.

    uv run source/eval/eval_dev_like_vo.py
    uv run source/eval/eval_dev_like_vo.py --seed 101

Reads results/eval/dev_like/<seed>/ticks.csv. A step frame equals ``render()`` at the true pose
(the logged one) and the agent sees it through the simulator's JPEG, so the run's frames are
re-rendered from ticks.csv and put through the localiser's own stages (geometry.wall_view,
match_features, relative_pose) here, without the agent:

- the true distance to the wall ahead: a ray from the true pose along the yaw to the true walls
  (eval_cues.Walls);
- the centre column's wall range (the wall guard's input) and whether it is NaN;
- per fallback tick (``vo_fallback`` of ticks.csv) the first stage that failed: no features (a view
  with under 2 descriptors), too few matches (under FIT_MIN_INLIERS), matches without a wall range
  (the keypoints' wall point is NaN), wall points beyond FIT_RANGE, RANSAC (the fit returns None)
  or an implausible step. RANSAC is run here with its own rng, not the run's, so a tick the run
  fell back on and the replay does not is counted as ``rng`` (RANSAC's draw).

Tables: the fallback rate by wall distance and by the previous tick's action; the reasons; the
centre column's NaN rate by wall distance and the distance below which it is always NaN; per
run the first wall contact (ticks.csv ``blocked``) and the fallback rate in the 50 ticks before
and after it. Writes vo_diagnostics.md and vo_ticks.csv beside ticks.csv. No agent code is used
or changed.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as sim
from common import textures, write_csv
from eval_cues import Walls

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geometry  # noqa: E402
import map_agent  # noqa: E402
import mapping  # noqa: E402
from eval_dev_like import COLS, MSAA, ROWS, DevSimulator  # noqa: E402
from geometry import (  # noqa: E402
    FIT_MIN_INLIERS,
    FIT_RANGE,
    Camera,
    FloorMask,
    match_features,
    visual_odometry,
    wall_view,
)

OUT_ROOT = Path("results") / "eval" / "dev_like"
BINS = ((0.0, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.3), (0.3, math.inf))
NAN_STEP_M = 0.01
NAN_MAX_M = 0.5
WINDOW = 50
REASONS = (
    "no features",
    "too few matches",
    "matches lack a wall range",
    f"wall points beyond {FIT_RANGE} m",
    "RANSAC fail",
    "implausible step",
    "rng (run only)",
)
FIELDS = [
    "goal",
    "tick",
    "taken",
    "wall_ahead_m",
    "centre_range_m",
    "centre_nan",
    "fallback",
    "reason",
]


def read_ticks(path: Path) -> dict[int, list[dict]]:
    by_goal: dict[int, list[dict]] = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            by_goal[int(row["goal"])].append(row)
    return by_goal


def wall_ahead(rects: np.ndarray, x: float, y: float, yaw: float) -> float:
    """Metres along the ray from ``(x, y)`` at ``yaw`` to the nearest wall rectangle (inf: none);
    0 inside one."""
    d = np.array([math.cos(yaw), math.sin(yaw)])
    with np.errstate(divide="ignore", invalid="ignore"):
        t0, t1 = [], []
        for axis, lo, hi in ((0, rects[:, 0], rects[:, 1]), (1, rects[:, 2], rects[:, 3])):
            o = (x, y)[axis]
            if d[axis] == 0:
                inside = (lo <= o) & (o <= hi)
                t0.append(np.where(inside, -np.inf, np.inf))
                t1.append(np.where(inside, np.inf, -np.inf))
            else:
                a, b = (lo - o) / d[axis], (hi - o) / d[axis]
                t0.append(np.minimum(a, b))
                t1.append(np.maximum(a, b))
    near, far = np.maximum(t0[0], t0[1]), np.minimum(t1[0], t1[1])
    hit = (near <= far) & (far >= 0)
    return float(np.maximum(near[hit], 0.0).min()) if hit.any() else math.inf


def classify(prev, cur) -> str:
    """The first stage of visual_odometry that fails between two views ('' when none)."""
    if prev.des is None or cur.des is None or len(prev.des) < 2 or len(cur.des) < 2:
        return REASONS[0]
    matches = match_features(prev, cur)
    if len(matches) < FIT_MIN_INLIERS:
        return REASONS[1]
    pa, pb = prev.wall[matches[:, 0]], cur.wall[matches[:, 1]]
    valid = np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1)
    if valid.sum() < FIT_MIN_INLIERS:
        return REASONS[2]
    with np.errstate(invalid="ignore"):
        usable = valid & (np.hypot(*pa.T) <= FIT_RANGE) & (np.hypot(*pb.T) <= FIT_RANGE)
    if usable.sum() < FIT_MIN_INLIERS:
        return REASONS[3]
    step = visual_odometry(prev, cur, np.random.default_rng(0))
    if step is None:
        return REASONS[4]
    return "" if mapping.plausible(step) else REASONS[5]


def replay(goal: int, rows: list[dict], world: sim.Simulator, camera, floor, rects) -> list[dict]:
    sift = cv2.SIFT_create()
    out, previous, taken = [], None, None
    for row in rows:
        x, y, yaw = (float(row[k]) for k in ("true_x", "true_y", "true_yaw"))
        frame = cv2.imdecode(np.frombuffer(world.jpeg(world.render(x, y, yaw)), np.uint8), 1)
        view = wall_view(frame, camera, floor, sift)
        centre = view.points[len(view.points) // 2]
        centre_range = math.hypot(*centre) if np.isfinite(centre).all() else math.nan
        fallback = row["vo_fallback"]
        reason = ""
        if fallback != "" and previous is not None:
            found = classify(previous, view)
            reason = (found or REASONS[6]) if fallback == "1" else ""
        out.append(
            {
                "goal": goal,
                "tick": int(row["tick"]),
                "taken": taken or "",
                "wall_ahead_m": wall_ahead(rects, x, y, yaw),
                "centre_range_m": centre_range,
                "centre_nan": int(math.isnan(centre_range)),
                "fallback": "" if fallback == "" else int(fallback),
                "reason": reason,
            }
        )
        previous = view
        taken = None if row["action"] == "CHECKIN" else row["action"]
    return out


def rate(rows: list[dict]) -> str:
    v = [r["fallback"] for r in rows if r["fallback"] != ""]
    return f"{np.mean(v):.2f} ({len(v)})" if v else "- (0)"


def md(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def bin_label(lo: float, hi: float) -> str:
    return f"> {lo:g}" if math.isinf(hi) else f"{lo:g}-{hi:g}"


def report(rows: list[dict], ticks: dict[int, list[dict]]) -> str:
    out = ["# Visual odometry fallback diagnostics", ""]
    in_bin = lambda r, lo, hi: lo <= r["wall_ahead_m"] < hi  # noqa: E731
    out += ["## 1. Fallback rate (n ticks with a VO attempt) by true wall distance ahead", ""]
    out.append(
        md(
            ["wall ahead m", "all", *map_agent_actions()],
            [
                [bin_label(lo, hi)]
                + [rate([r for r in rows if in_bin(r, lo, hi)])]
                + [
                    rate([r for r in rows if in_bin(r, lo, hi) and r["taken"] == a])
                    for a in map_agent_actions()
                ]
                for lo, hi in BINS
            ]
            + [
                ["all", rate(rows)]
                + [rate([r for r in rows if r["taken"] == a]) for a in map_agent_actions()]
            ],
        )
    )
    out += ["", "Columns are the action taken since the previous frame.", ""]
    out += ["## 2. Fallback ticks by failure reason", ""]
    fall = [r for r in rows if r["fallback"] == 1]
    counts = Counter(r["reason"] for r in fall)
    out.append(
        md(
            ["reason", "ticks", "share", *[bin_label(lo, hi) + " m" for lo, hi in BINS]],
            [
                [
                    reason,
                    counts[reason],
                    f"{counts[reason] / max(len(fall), 1):.2f}",
                    *[
                        sum(1 for r in fall if r["reason"] == reason and in_bin(r, lo, hi))
                        for lo, hi in BINS
                    ],
                ]
                for reason in REASONS
            ],
        )
    )
    out += [f"\n{len(fall)} fallback ticks of {sum(r['fallback'] != '' for r in rows)}.", ""]
    out += ["## 3. Centre-column range NaN by true wall distance ahead", ""]
    edges = np.arange(0.0, NAN_MAX_M + NAN_STEP_M, NAN_STEP_M)
    table = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        chosen = [r for r in rows if lo <= r["wall_ahead_m"] < hi]
        if chosen:
            table.append(
                [
                    f"{lo:.2f}-{hi:.2f}",
                    len(chosen),
                    f"{np.mean([r['centre_nan'] for r in chosen]):.2f}",
                ]
            )
    beyond = [r for r in rows if r["wall_ahead_m"] >= NAN_MAX_M]
    table.append(
        [f">= {NAN_MAX_M}", len(beyond), f"{np.mean([r['centre_nan'] for r in beyond]):.2f}"]
    )
    out.append(md(["wall ahead m", "ticks", "centre NaN"], table))
    ordered = sorted(rows, key=lambda r: r["wall_ahead_m"])
    always = 0.0
    for r in ordered:
        if not r["centre_nan"]:
            break
        always = r["wall_ahead_m"]
    nan_ahead = [r for r in rows if r["centre_nan"] and math.isfinite(r["wall_ahead_m"])]
    out += [
        "",
        f"The centre column is NaN at every tick with the true wall ahead under "
        f"**{next((r['wall_ahead_m'] for r in ordered if not r['centre_nan']), math.nan):.3f} m** "
        f"(first non-NaN tick; the last NaN before it was at {always:.3f} m). "
        f"NaN ticks with a wall ahead: {len(nan_ahead)}; the farthest at "
        f"{max((r['wall_ahead_m'] for r in nan_ahead), default=math.nan):.2f} m.",
        "",
    ]
    out += [f"## 4. First wall contact: fallback rate in the {WINDOW} ticks before and after", ""]
    table = []
    for goal, rs in ticks.items():
        mine = [r for r in rows if r["goal"] == goal]
        first = next((i for i, r in enumerate(rs) if r["blocked"] == "1"), None)
        if first is None:
            table.append([goal, "none", "-", "-", rate(mine)])
            continue
        before = mine[max(0, first - WINDOW) : first]
        after = mine[first + 1 : first + 1 + WINDOW]
        table.append([goal, rs[first]["tick"], rate(before), rate(after), rate(mine[first + 1 :])])
    out.append(
        md(
            ["goal", "first contact tick", f"{WINDOW} before", f"{WINDOW} after", "all after"],
            table,
        )
    )
    return "\n".join(out) + "\n"


def map_agent_actions() -> tuple[str, ...]:
    return ("FORWARD", "BACKWARD", "LEFT", "RIGHT")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--goals", default="", help="comma-separated goal ids (default: all)")
    args = p.parse_args()
    directory = OUT_ROOT / str(args.seed)
    ticks = read_ticks(directory / "ticks.csv")
    if args.goals:
        ticks = {g: ticks[g] for g in (int(v) for v in args.goals.split(",") if v)}
    world = DevSimulator(textures(), args.seed, msaa=MSAA)
    assert world.occupancy.shape == (ROWS, COLS)
    rects = np.array(Walls(world).rects)
    camera_dict = dict(geometry.CAMERA)
    camera = Camera({**camera_dict, "cy": camera_dict["cy"] + mapping.HORIZON_SHIFT_PX})
    floor = FloorMask.load(map_agent.FLOOR_MASK_PATH)
    rows: list[dict] = []
    for goal, rs in ticks.items():
        rows.extend(replay(goal, rs, world, camera, floor, rects))
        print(f"goal {goal}: {len(rs)} ticks replayed", flush=True)
    write_csv(directory / "vo_ticks.csv", FIELDS, rows)
    text = report(rows, ticks)
    (directory / "vo_diagnostics.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
