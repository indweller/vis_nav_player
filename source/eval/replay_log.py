"""Offline replay of a hosted run log against its frozen map (no server, no session).

    uv run source/eval/replay_log.py                      # the default log and data
    uv run source/eval/replay_log.py --log data/logs/<name> --data data/<id>

``--log`` is an extracted run log (events.jsonl and images/); the zip is extracted to
data/logs/<name>/ by hand (ignore __MACOSX). Four checks, written to
results/eval/replay/<name>/ (summary.md, ticks.csv, replay.png):

1. Check-in quality: the last logged frame (the frame camera) against the four goal views of
   <data>/target.jpg (goal.target_camera): per view the SIFT epipolar inliers and the rigid
   relative pose; the best view gives the distance to the goal and the heading error (degrees to
   turn left to the goal's heading) the frame had at check-in.
2. Replay: the current Localiser (as MapAgent builds it) on the frozen map, fed the logged
   frames and, as the move since the previous frame, the previous tick's returned action (WAIT /
   CHECKIN: no move), from the spawn (0, 0, 0). Per tick: VO source, fix status, lost, the
   online medians; and the distance to the logged estimate.
3. Contacts: FORWARD ticks whose next frame is identical or near-identical
   (localiser.grey_difference under --contact-diff).
4. Path: the logged estimated path length and step count against the current planner's
   spawn-to-goal plan (metres; steps estimated from the calibrated motion).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import frozen_map
import geometry
import map_agent
import mapping
import matplotlib
import numpy as np
import planner as planner_module
from geometry import Camera, FloorMask, between, compose, wrap
from goal import find_goal, target_camera
from localiser import Localiser, grey_difference, grey_small
from view_map import draw as draw_map

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
LOG = "20261029T051014_783729Z_pid37387_62b9cea0"
DATA = "da90a443e215"
CONTACT_DIFF = 2.0
"""Mean absolute grey difference (localiser.grey_small) under which frames are near-identical."""
MOVES = geometry.MOTION_ACTIONS
ONLINE_ACTIONS = ("FORWARD", "LEFT", "RIGHT", "BACKWARD")


def load_events(log: Path) -> list[dict]:
    return [json.loads(line) for line in (log / "events.jsonl").read_text().splitlines() if line]


def checkin(frame, views, camera, floor) -> dict:
    """Match ``frame`` to the four goal ``views``: each view's inliers and relative pose."""
    sift = cv2.SIFT_create()
    rng = np.random.default_rng(0)
    seen = geometry.wall_view(frame, camera, floor, sift)
    photo = target_camera(camera)
    rows = []
    for k, image in enumerate(views):
        view = geometry.wall_view(image, photo, floor, sift)
        matches = geometry.match_features(seen, view)
        inliers = geometry.epipolar_inliers(seen, view, matches)
        fit = geometry.relative_pose(seen, view, matches[inliers], rng) if inliers.any() else None
        row = {"k": k, "matches": len(matches), "inliers": int(inliers.sum()), "fit": None}
        if fit is not None:
            x, y, yaw = fit[0]
            row.update(
                fit=fit[0],
                fit_inliers=fit[1],
                distance_m=math.hypot(x, y),
                heading_deg=math.degrees(wrap(yaw - k * math.pi / 2)),
            )
        rows.append(row)
    fitted = [r for r in rows if r["fit"] is not None]
    return {"views": rows, "best": max(fitted, key=lambda r: r["inliers"], default=None)}


def replay(events, log: Path, localiser: Localiser, contact_diff: float) -> list[dict]:
    rows, previous, previous_grey, last = [], None, None, None
    for e in events:
        frame = cv2.imread(str(log / e["image"]))
        est = localiser.step(frame, last)
        grey = grey_small(frame)
        logged = e["estimated_pose_xyyaw_rad"]
        row = {
            "tick": e["index"],
            "action": e["returned_action"],
            "mode": e["controller_mode"],
            "source": localiser.source or "",
            "status": est.status or "",
            "rejected": est.rejected,
            "lost": int(est.lost),
            "sigma": est.sigma_m,
            "x": est.x,
            "y": est.y,
            "yaw": est.yaw,
            "logged_x": logged[0],
            "logged_y": logged[1],
            "logged_yaw": logged[2],
            "logged_status": e["localization_status"] or "",
            "distance_to_logged": math.hypot(est.x - logged[0], est.y - logged[1]),
            "yaw_to_logged_deg": math.degrees(wrap(est.yaw - logged[2])),
            "grey_to_next": math.nan,
        }
        for a in ONLINE_ACTIONS:
            m = localiser.online(a)
            row[f"online_{a}"] = "" if m is None else json.dumps([round(v, 5) for v in m])
        if previous is not None:
            previous["grey_to_next"] = grey_difference(previous_grey, grey)
        rows.append(row)
        previous, previous_grey = row, grey
        last = e["returned_action"] if e["returned_action"] in MOVES else None
    return rows


def contacts(rows, threshold: float) -> list[int]:
    return [
        r["tick"]
        for r in rows
        if r["action"] == "FORWARD" and r["grey_to_next"] < threshold  # NaN (last tick): False
    ]


def steps_for(path, motion: dict, turn_deg: float) -> tuple[int, int, int]:
    """(FORWARD ticks, turn ticks, total) a MapAgent-like driver takes along ``path``, simulated
    without noise (map_agent.Config's defaults): heading toward the point lookahead_m ahead on the
    path, a turn of up to max_turn_ticks ticks when the error is over turn_tol, else
    forward_ticks FORWARD ticks; then the turn to the goal's yaw."""
    step = abs(motion["FORWARD"]["dx"])
    pose = [float(v) for v in path.waypoints[0]]
    forward = turns = progress = 0.0
    for _ in range(20000):
        if path.length - progress < map_agent.Config.offset_m:
            break
        a = path.arc
        i = int(np.searchsorted(a, min(progress + map_agent.Config.lookahead_m, path.length)))
        target = np.array(path.waypoints[min(i, len(a) - 1), :2])
        error = math.degrees(wrap(math.atan2(target[1] - pose[1], target[0] - pose[0]) - pose[2]))
        if abs(error) > map_agent.Config.turn_tol:
            n = int(np.clip(round(abs(error) / turn_deg), 1, map_agent.Config.max_turn_ticks))
            pose[2] = wrap(pose[2] + math.copysign(n * math.radians(turn_deg), error))
            turns += n
        else:
            for _ in range(map_agent.Config.forward_ticks):
                pose[0] += step * math.cos(pose[2])
                pose[1] += step * math.sin(pose[2])
                forward += 1
            progress = path.cross_track(pose)[1]
    turns += abs(math.degrees(wrap(path.waypoints[-1][2] - pose[2]))) / turn_deg
    return int(forward), int(round(turns)), int(round(forward + turns))


def runs(ticks: list[int]) -> str:
    """``ticks`` as ranges, e.g. 3-5, 9."""
    out, i = [], 0
    while i < len(ticks):
        j = i
        while j + 1 < len(ticks) and ticks[j + 1] == ticks[j] + 1:
            j += 1
        out.append(str(ticks[i]) if i == j else f"{ticks[i]}-{ticks[j]}")
        i = j + 1
    return ", ".join(out)


def plot(store, spawn, goal, plans, rows, check, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_map(ax, store, spawn, goal, plans.get("plain"))
    ax.plot([r["logged_x"] for r in rows], [r["logged_y"] for r in rows], color="magenta",
            lw=1.5, label="logged estimate")  # fmt: skip
    ax.plot([r["x"] for r in rows], [r["y"] for r in rows], color="cyan", lw=1.2,
            ls="--", label="replayed estimate")  # fmt: skip
    for name, r, colour in (
        ("logged", rows[-1], "magenta"),
        ("replayed", rows[-1], "cyan"),
    ):
        x, y = (r["logged_x"], r["logged_y"]) if name == "logged" else (r["x"], r["y"])
        ax.plot(x, y, "s", color=colour, ms=7, label=f"{name} end ({x:.2f}, {y:.2f})")
    if check is not None:
        x, y, yaw = check
        ax.plot(x, y, "*", color="red", ms=16, zorder=6, label=f"check-in pose ({x:.2f}, {y:.2f})")
        ax.arrow(x, y, 0.25 * math.cos(yaw), 0.25 * math.sin(yaw), color="red", width=0.01,
                 head_width=0.07, length_includes_head=True, zorder=6)  # fmt: skip
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("log replay: " + out.parent.name)
    fig.savefig(out, dpi=150, bbox_inches="tight")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", default=f"data/logs/{LOG}", help="extracted run log directory")
    parser.add_argument("--data", default=f"data/{DATA}", help="exploration data directory")
    parser.add_argument("--contact-diff", type=float, default=CONTACT_DIFF)
    args = parser.parse_args(argv)
    log, root = ROOT / args.log, ROOT / args.data
    out = ROOT / "results" / "eval" / "replay" / log.name
    out.mkdir(parents=True, exist_ok=True)

    events = load_events(log)
    floor = FloorMask.load(map_agent.FLOOR_MASK_PATH)
    camera_dict = dict(geometry.CAMERA)
    camera = Camera({**camera_dict, "cy": camera_dict["cy"] + mapping.HORIZON_SHIFT_PX})
    store, routes = frozen_map.get(root, camera_dict, floor)
    motion = map_agent.load_motion(root, routes, camera, floor)
    views = map_agent.goal_views(root)
    spawn = (0.0, 0.0, 0.0)

    # 1. check-in
    last = cv2.imread(str(log / events[-1]["image"]))
    check = checkin(last, views, camera, floor)
    best = check["best"]

    # 2. replay
    localiser = Localiser(store, camera, floor, motion=motion)
    rows = replay(events, log, localiser, args.contact_diff)
    with (out / "ticks.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 3. contacts
    hits = contacts(rows, args.contact_diff)
    identical = contacts(rows, 1e-9)
    forwards = sum(r["action"] == "FORWARD" for r in rows)
    smallest = sorted(
        (r["grey_to_next"], r["tick"])
        for r in rows
        if r["action"] == "FORWARD" and not math.isnan(r["grey_to_next"])
    )[:5]

    # 4. path
    found = find_goal(views, store, camera, floor)
    goal = None if found is None else (found.x, found.y, found.yaw)
    logged_goal = tuple(events[0]["estimated_goal_xyyaw_rad"])
    planner = planner_module.PlaceGraph(store, cache_path=root / "links.npz")
    plans = {}
    for name, target in (("plain", goal), ("logged goal", logged_goal)):
        if target is not None:
            plans[name] = planner.plan(spawn, target)
    turn_deg = float(np.mean([abs(math.degrees(motion[a]["dyaw"])) for a in ("LEFT", "RIGHT")]))
    xy = np.array([r["logged_x"] for r in rows]), np.array([r["logged_y"] for r in rows])
    logged_len = float(np.hypot(np.diff(xy[0]), np.diff(xy[1])).sum())
    replay_len = float(
        np.hypot(np.diff([r["x"] for r in rows]), np.diff([r["y"] for r in rows])).sum()
    )
    actions = Counter(r["action"] for r in rows)

    # the check-in pose in the map, from the best view's match
    pose = None
    if best is not None and goal is not None:
        k = best["k"]
        goal_in_frame = (best["fit"][0], best["fit"][1], wrap(best["fit"][2] - k * math.pi / 2))
        pose = compose(goal, between(goal_in_frame, (0.0, 0.0, 0.0)))
    plot(store, spawn, goal, plans, rows, pose, out / "replay.png")

    # summary
    d = np.array([r["distance_to_logged"] for r in rows])
    sources = Counter(r["source"] or "none" for r in rows)
    vo = sources["vo"]
    fallbacks = sum(v for k, v in sources.items() if k not in ("vo", "none"))
    by_action = {
        a: Counter(r["source"] for p, r in zip(rows, rows[1:], strict=False) if p["action"] == a)
        for a in MOVES
    }
    status = Counter(r["status"] for r in rows)
    rejected = sum(r["rejected"] for r in rows)
    lost_ticks = sum(r["lost"] for r in rows)
    logged_status = Counter(r["logged_status"] or "none" for r in rows)
    final = {a: localiser.online(a) for a in ONLINE_ACTIONS}
    when = {}
    for a in ONLINE_ACTIONS:
        for r in rows:
            if r[f"online_{a}"] != "":
                when[a] = r["tick"]
                break
    marks = [t for t in (0, 100, 250, 500, 750, 1000, 1250, len(rows) - 1) if t < len(rows)]
    fmt_pose = lambda p: f"({p[0]:.2f}, {p[1]:.2f}, {math.degrees(p[2]):.0f} deg)"  # noqa: E731

    lines = [
        f"# Replay of {log.name}",
        "",
        f"Map `{root.name}`; {len(rows)} ticks; actions {dict(actions)}.",
        "",
    ]
    lines += ["## 1. Check-in quality", ""]
    lines += [
        "| view k | matches | inliers | rigid inliers | distance (m) | heading (deg) |",
        "|---|---|---|---|---|---|",
    ]
    for r in check["views"]:
        if r["fit"] is None:
            lines.append(f"| {r['k']} | {r['matches']} | {r['inliers']} | - | - | - |")
        else:
            lines.append(
                f"| {r['k']} | {r['matches']} | {r['inliers']} | {r['fit_inliers']} | "
                f"{r['distance_m']:.3f} | {r['heading_deg']:+.1f} |"
            )
    if best is None:
        lines += ["", "No goal view gave a relative pose: check-in quality unknown."]
    else:
        end = events[-1]["estimated_pose_xyyaw_rad"]
        lines += [
            "",
            f"Best view {best['k']} ({best['inliers']} inliers): estimated distance to goal "
            f"**{best['distance_m']:.3f} m**, heading error **{best['heading_deg']:+.1f} deg** "
            f"(turn left to the goal's heading). Logged estimate at check-in "
            f"{fmt_pose(end)}, logged goal "
            f"{fmt_pose(logged_goal)}: logged distance "
            f"{math.hypot(end[0] - logged_goal[0], end[1] - logged_goal[1]):.3f} m, heading error "
            f"{math.degrees(wrap(logged_goal[2] - end[2])):+.1f} deg.",
        ]
        if len({r["k"] for r in check["views"] if r["inliers"] >= 15}) > 1:
            lines.append("Several views match with 15+ inliers: the single best is used.")
    lines += ["", "## 2. Replay (current Localiser)", ""]
    lines += [
        f"- VO source: {vo} VO ticks, {fallbacks} fallback ({dict(sources)}); first tick has none.",
        "- Source by the action before the tick: "
        + "; ".join(f"{a} {dict(c)}" for a, c in by_action.items() if c),
        f"- Fixes: accepted {status['accepted']}, held {status['held']}, "
        f"rejected (held then dropped) {rejected}; "
        f"layout-failed candidates {localiser.counts['layout']}.",
        f"- Lost: {localiser.lost_events} event(s), {lost_ticks} ticks lost; final sigma "
        f"{rows[-1]['sigma']:.3f} m.",
        f"- Logged run status counts: {dict(logged_status)}.",
        "- Online motion medians (dx, dy, dyaw) at the end ("
        f"{', '.join(f'{a} n={len(localiser.samples.get(a, []))}' for a in ONLINE_ACTIONS)}): "
        + "; ".join(
            f"{a} {tuple(round(v, 4) for v in m)} (from tick {when[a]})"
            for a, m in final.items()
            if m is not None
        ),
        "  Calibrated: "
        + "; ".join(
            f"{a} ({motion[a]['dx']:.3f}, {motion[a]['dy']:.3f}, {motion[a]['dyaw']:.4f})"
            for a in ONLINE_ACTIONS
            if a in motion
        ),
        f"- Replayed vs logged estimate (m): mean {d.mean():.3f}, median {np.median(d):.3f}, "
        f"max {d.max():.3f} at tick {int(d.argmax())}, final {d[-1]:.3f}. "
        f"Replayed end {fmt_pose((rows[-1]['x'], rows[-1]['y'], rows[-1]['yaw']))}.",
        "",
        "| tick | replayed (x, y) | logged (x, y) | distance (m) | yaw diff (deg) |",
        "|---|---|---|---|---|",
    ]
    for t in marks:
        r = rows[t]
        lines.append(
            f"| {t} | ({r['x']:.2f}, {r['y']:.2f}) | ({r['logged_x']:.2f}, {r['logged_y']:.2f}) | "
            f"{r['distance_to_logged']:.3f} | {r['yaw_to_logged_deg']:+.1f} |"
        )
    lines += ["", "## 3. Contacts", ""]
    lines += [
        f"FORWARD ticks: {forwards}. Followed by an identical frame: {len(identical)}; "
        f"by a frame with grey difference under {args.contact_diff:g}: **{len(hits)}**.",
        f"Indices (under {args.contact_diff:g}): {runs(hits) or 'none'}.",
        "Smallest FORWARD-to-next grey differences: "
        + ", ".join(f"{t}: {g:.2f}" for g, t in smallest),
        "The log's actions are returned, not server-confirmed, so an identical frame may also be "
        "a dropped or delayed action.",
        "",
        "## 4. Path",
        "",
        f"- Logged: estimated path length {logged_len:.2f} m (replayed {replay_len:.2f} m), "
        f"{len(rows)} ticks ({actions['FORWARD']} FORWARD, {actions['LEFT'] + actions['RIGHT']} "
        f"turns, {actions['WAIT']} WAIT); homing ticks {sum(r['mode'] == 'home' for r in rows)}.",
    ]
    for name, p in plans.items():
        if p is None:
            lines.append(f"- Plan ({name}): none")
            continue
        f_, t_, n_ = steps_for(p, motion, turn_deg)
        lines.append(
            f"- Plan ({name}): {p.length:.2f} m, about {n_} steps "
            f"({f_} FORWARD + {t_} turn ticks at {turn_deg:.2f} deg, simulated with the agent's "
            f"steering rule, no noise)."
        )
    lines += [
        "",
        f"find_goal now: {'none' if goal is None else fmt_pose(goal)}; "
        f"logged goal {fmt_pose(logged_goal)}.",
        "",
        "![replay](replay.png)",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
