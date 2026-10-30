"""MapAgent driving the local simulator, closed loop, against the ground truth.

    uv run source/eval/eval_closed_loop.py
    uv run source/eval/eval_closed_loop.py --mazes 7 --goals found

Per maze (7-12, the frozen maps of eval_goal_map) and goal variant -- "found": goal.find_goal on
the target views; "true": the true target moved into the map frame by the one rigid frame of the
first spawn, and "map-true": the true target as the map has it (the mapped pose of the exploration
frame nearest it composed with the true relative pose, as MapTruth does for localisation), both
passed as MapAgent's ``goal_override`` -- the agent drives a local session with motion noise on and
a motion_seed other than the exploration's, frames JPEG-encoded as the server sends them. Every
tick is logged: the true pose (world and map frame), the map-consistent truth (the true pose as the
map has it: the mapped pose of the frame nearest the robot composed with the true relative pose
from that frame), the estimate, sigma, the agent's mode and action, the navigator's replan, the
fix, the wall guard and the true geodesic distance (cells, 4-connected) to the true goal.

The run is scored by the simulator's check-in (distance, heading, tier, SPL, ticks), and then
again by the SDK's own scorer, run(MapAgent(...), "local:<seed>") on the same noise and seed
(score_*: points, max, the Distance / Heading / SPL blocks, tier). A run that does not check
in scores 0 points, tier "none". Per run also:
  - localisation error against the world truth (loc_) and the map-consistent truth (cm_);
  - wall-guard events, replans (by reason), accepted fixes and false fixes (accepted, then over
    0.3 m from the map-consistent truth), lost events;
  - the final error decomposition (found and map-true runs): the goal's error against the
    map-consistent goal (goal_cm_err_m), the final estimate against the goal the agent drove to
    (final_to_goal_m), against the robot's map-consistent truth (final_cm_err_m), the heading
    error at check-in (heading_deg) and the homing outcome (home_outcome: aligned = checked in
    by the homing test, timed_out = --homing-max, no_homing; home_ticks, home_fixes);
  - the longest stretch of ticks with no new minimum of the geodesic distance to the goal;
  - the fraction of ticks in each mode, and the ticks by mode and by action (FORWARD, BACKWARD,
    LEFT, RIGHT; LEFT + RIGHT are the ticks turning in place), the oracle's steps and
    ``spl_ticks`` = oracle_steps / ticks; the plans made.

--seeds N (default 3) drives each maze and goal with N motion seeds (the maze's plus
MOTION_SEED_OFFSET, plus 1000 per extra). The agent is built from MapAgent's flags
(map_agent.Config). Per goal variant the summary gives mean points +- std, the success rate
(check-in within SUCCESS_M and SUCCESS_DEG of the goal), mean SPL (the simulator's) and the
failure stage counts of the runs that failed (the first that applies: goal error > 0.2 m /
localisation lost / homing timeout / heading > 20 deg / distance > 0.2 m).

Writes to results/eval/closed_loop/: runs.csv, ticks.csv and <seed>_<goal>.png (true walls, true
trajectory in black, estimated in blue, goal true in green, map-true in blue and found in red,
replans as orange dots and wall-guard events as red crosses, both at the robot's true position)."""

from __future__ import annotations

import argparse
import dataclasses
import math
from collections import Counter
from pathlib import Path

import cv2
import map_agent
import numpy as np
import vis_nav_sim as sim
from common import cell_of, fmt, geodesic, ints, textures, write_csv
from eval_cues import Walls, stored_floor
from eval_map import MAP_COLOUR, Canvas, Frame, MapTruth, record, true_poses
from vis_nav_sdk import Action, Observation, run
from vis_nav_sdk.local import local_session

GOALS = ("found", "true", "map-true")
MODES = ("nav", "guard", "home", "search")
REASONS = ("initial", "cross_track", "jump", "periodic")
MOTION_SEED_OFFSET = 1000
"""The driven run's motion_seed is the maze's plus this, not the exploration's."""
MAX_STEPS = 3000
FALSE_FIX_M = 0.3
HELD_OUT = (10, 11, 12)
SEEDS = 3
"""Motion seeds per maze and goal."""
SEED_STEP = 1000
"""Between one motion seed and the next."""
GOAL_ERROR_M = 0.2
SUCCESS_M = 0.2
SUCCESS_DEG = 20.0
STAGES = (
    "goal error",
    "localisation lost",
    "homing timeout",
    "heading",
    "distance",
)
BLOCKS = ("Distance", "Heading", "SPL")
ACTIONS = ("FORWARD", "BACKWARD", "LEFT", "RIGHT")
FIELDS = [
    "config",
    "seed",
    "motion",
    "goal",
    "ticks",
    "checkin_reason",
    "distance_m",
    "heading_deg",
    "tier",
    "spl",
    "goal_err_m",
    "goal_cm_err_m",
    *(
        f"{p}{k}"
        for p in ("loc_", "cm_")
        for k in ("median_m", "p90_m", "max_m", "within_0.1", "within_0.2")
    ),
    "wall_guard",
    "replans",
    *(f"replans_{r}" for r in REASONS),
    "fixes_accepted",
    "false_fixes",
    "lost_events",
    "stall_ticks",
    *(f"mode_{m}" for m in MODES),
    "oracle_steps",
    "spl_ticks",
    *(f"ticks_{m}" for m in MODES),
    *(f"act_{a.lower().replace('|', '_')}" for a in ACTIONS),
    "turn_ticks",
    "plans",
    "final_err_m",
    "final_to_goal_m",
    "final_cm_err_m",
    "home_outcome",
    "home_ticks",
    "home_fixes",
    "score_points",
    "score_max",
    "score_distance",
    "score_heading",
    "score_spl",
    "score_tier",
    "sdk_nav_steps",
    "sdk_efficiency",
    "sdk_arrival_m",
    "sdk_arrival_deg",
]
TICK_FIELDS = [
    "config",
    "seed",
    "motion",
    "goal",
    "tick",
    "true_x",
    "true_y",
    "true_yaw",
    "map_x",
    "map_y",
    "map_yaw",
    "cm_x",
    "cm_y",
    "cm_yaw",
    "est_x",
    "est_y",
    "est_yaw",
    "sigma",
    "mode",
    "action",
    "replan",
    "fix",
    "false_fix",
    "lost",
    "guard",
    "geodesic",
    "home_view",
    "home_yaw",
]


class ServerFrames(map_agent.MapAgent):
    """MapAgent on JPEG-encoded frames, as the server sends them (the local session hands out raw
    ones). ``encode`` is a Simulator's ``jpeg``."""

    encode = None

    def act(self, obs: Observation):
        image = cv2.imdecode(np.frombuffer(self.encode(obs.image), np.uint8), cv2.IMREAD_COLOR)
        return super().act(dataclasses.replace(obs, image=image))


class Maze:
    """One maze's truth against the agent's map."""

    def __init__(self, seed: int, agent: ServerFrames, root: Path) -> None:
        self.seed = seed
        self.world = sim.Simulator(textures(), seed)
        self.walls = Walls(self.world)
        truth = true_poses(root, agent.routes)
        self.truth = MapTruth(agent.store, truth)
        self.frame = Frame((0.0, 0.0, 0.0), tuple(truth[0][0]))
        self.free = ~self.world.occupancy
        self.geodesic = geodesic(self.world, tuple(self.world.target_cell))
        self.true_goal = tuple(
            float(v) for v in self.frame.to_map(np.array([self.world.target]))[0]
        )
        self.consistent_goal = self.truth.consistent(self.world.target)


def motion_seed(maze: Maze, motion: int) -> int:
    return maze.seed + MOTION_SEED_OFFSET + SEED_STEP * motion


def drive(agent: ServerFrames, maze: Maze, args, motion: int = 0) -> dict:
    """One run on a local session, logged tick by tick: ``{"log": [...], "result": ...}``."""
    session = local_session(
        f"local:{maze.seed}",
        textures=textures(),
        motion_noise=sim.DEFAULT_NOISE,
        motion_seed=motion_seed(maze, motion),
        max_steps=args.max_steps,
    )
    agent.setup(session.info)
    obs = session.initial_observation
    log, result, arrival = [], None, None
    while True:
        decision, _ = agent.act(obs)
        update = agent.update
        pose = session.pose()
        estimate = update.estimate
        in_map = maze.frame.to_map(np.array([pose]))[0]
        consistent = maze.truth.consistent(pose)
        log.append(
            {
                "tick": obs.step,
                "true": pose,
                "map": tuple(in_map),
                "consistent": consistent,
                "estimate": tuple(estimate[:3]),
                "sigma": estimate.sigma_m,
                "mode": agent.mode,
                "action": "CHECKIN" if decision & Action.CHECKIN else agent.last,
                "replan": update.replan,
                "fixed": estimate.fixed,
                "lost": estimate.lost,
                "guard": agent.guarded,
                "geodesic": float(maze.geodesic[cell_of(maze.world, maze.free, *pose[:2])]),
                "view": agent.view,
                "path": update.path if update.replan else None,
            }
        )
        if decision & Action.CHECKIN:
            arrival = session.distance_heading()
            result = session.checkin()
            break
        if obs.steps_left <= 1:
            arrival = session.distance_heading()
            break
        obs = session.step(decision, 1)
    return {
        "log": log,
        "result": result,
        "arrival": arrival,
        "nav_steps": session.simulator.steps,
        "lost_events": agent.localiser.lost_events,
        "reason": agent.checkin_reason or "budget",
        "goal": agent.goal,
        "oracle_steps": maze.world.oracle_steps,
    }


def sdk_score(agent: ServerFrames, maze: Maze, args, motion: int = 0):
    """The SDK's own local scorer on the same session."""
    return run(
        agent,
        f"local:{maze.seed}",
        viewer=False,
        quiet=True,
        check=False,
        confirm=False,
        browser=False,
        textures=textures(),
        motion_noise=sim.DEFAULT_NOISE,
        motion_seed=motion_seed(maze, motion),
        max_steps=args.max_steps,
    )


# -- scoring


def spread(prefix: str, error: np.ndarray) -> dict:
    return {
        f"{prefix}median_m": float(np.median(error)),
        f"{prefix}p90_m": float(np.percentile(error, 90)),
        f"{prefix}max_m": float(error.max()),
        **{f"{prefix}within_{m}": float((error <= m).mean()) for m in (0.1, 0.2)},
    }


def longest_stall(geodesic_m: list[float]) -> int:
    """The longest run of ticks without a new minimum of the geodesic distance."""
    best, since, longest = math.inf, 0, 0
    for g in geodesic_m:
        if g < best:
            best, since = g, 0
        else:
            since += 1
            longest = max(longest, since)
    return longest


def score_run(
    maze: Maze, goal: str, run_: dict, sdk, config: str = "", motion: int = 0
) -> tuple[dict, list[dict]]:
    log = run_["log"]
    column = lambda key: np.array([t[key] for t in log], float)  # noqa: E731
    estimate, consistent = column("estimate"), column("consistent")
    world_error = np.hypot(*(column("map")[:, :2] - estimate[:, :2]).T)
    cm_error = np.hypot(*(consistent[:, :2] - estimate[:, :2]).T)
    false_fix = [t["fixed"] and cm_error[i] > FALSE_FIX_M for i, t in enumerate(log)]
    reasons = Counter(t["replan"] for t in log if t["replan"])
    modes = Counter(t["mode"] for t in log)
    result, arrival = run_["result"], run_["arrival"]
    actions = Counter(t["action"] for t in log)
    paths = [t["path"] for t in log if t["path"] is not None]
    row = {
        "config": config,
        "seed": maze.seed,
        "motion": motion,
        "goal": goal,
        "ticks": run_["nav_steps"],
        "checkin_reason": run_["reason"],
        "distance_m": arrival[0],
        "heading_deg": arrival[1],
        "tier": "none" if result is None else result.goal_tier,
        "spl": 0.0 if result is None else result.spl,
        "goal_err_m": math.dist(run_["goal"][:2], maze.true_goal[:2]),
        "goal_cm_err_m": math.dist(run_["goal"][:2], maze.consistent_goal[:2]),
        **spread("loc_", world_error),
        **spread("cm_", cm_error),
        "wall_guard": sum(t["guard"] for t in log),
        "replans": sum(reasons.values()),
        **{f"replans_{r}": reasons[r] for r in REASONS},
        "fixes_accepted": sum(t["fixed"] for t in log),
        "false_fixes": sum(false_fix),
        "lost_events": run_["lost_events"],
        "stall_ticks": longest_stall([t["geodesic"] for t in log]),
        **{f"mode_{m}": modes[m] / len(log) for m in MODES},
        "oracle_steps": run_["oracle_steps"],
        "spl_ticks": min(1.0, run_["oracle_steps"] / max(run_["nav_steps"], 1)),
        **{f"ticks_{m}": modes[m] for m in MODES},
        **{f"act_{a.lower().replace('|', '_')}": actions[a] for a in ACTIONS},
        "turn_ticks": actions["LEFT"] + actions["RIGHT"],
        "plans": len(paths),
        "final_err_m": float(world_error[-1]),
        "final_to_goal_m": math.dist(log[-1]["estimate"][:2], run_["goal"][:2]),
        "final_cm_err_m": float(cm_error[-1]),
        "home_outcome": {"homed": "aligned", "homing_max": "timed_out"}.get(
            run_["reason"], "no_homing" if modes["home"] == 0 else "cut_short"
        ),
        "home_ticks": modes["home"],
        "home_fixes": sum(t["fixed"] for t in log if t["mode"] == "home"),
        "score_points": 0.0 if sdk is None else sdk.points,
        "score_max": 10.0 if sdk is None else sdk.max_points,
        "score_distance": 0.0 if sdk is None else sdk.awarded.get("Distance", 0.0),
        "score_heading": 0.0 if sdk is None else sdk.awarded.get("Heading", 0.0),
        "score_spl": 0.0 if sdk is None else sdk.awarded.get("SPL", 0.0),
        "score_tier": "none" if sdk is None else sdk.goal_tier,
        "sdk_nav_steps": math.nan if sdk is None else sdk.nav_steps,
        "sdk_efficiency": 0.0 if sdk is None else sdk.spl,
        "sdk_arrival_m": math.nan if sdk is None or sdk.arrival is None else sdk.arrival.distance_m,
        "sdk_arrival_deg": (
            math.nan if sdk is None or sdk.arrival is None else sdk.arrival.heading_deg
        ),
    }
    ticks = []
    for i, t in enumerate(log):
        ticks.append(
            {
                "config": config,
                "seed": maze.seed,
                "motion": motion,
                "goal": goal,
                "tick": t["tick"],
                **dict(zip(("true_x", "true_y", "true_yaw"), t["true"], strict=True)),
                **dict(zip(("map_x", "map_y", "map_yaw"), t["map"], strict=True)),
                **dict(zip(("cm_x", "cm_y", "cm_yaw"), t["consistent"], strict=True)),
                **dict(zip(("est_x", "est_y", "est_yaw"), t["estimate"], strict=True)),
                "sigma": t["sigma"],
                "mode": t["mode"],
                "action": t["action"],
                "replan": t["replan"] or "",
                "fix": int(t["fixed"]),
                "false_fix": int(false_fix[i]),
                "lost": int(t["lost"]),
                "guard": int(t["guard"]),
                "geodesic": t["geodesic"],
                "home_view": "" if t["view"] is None else t["view"][0],
                "home_yaw": "" if t["view"] is None else t["view"][1],
            }
        )
    return row, ticks


# -- images


def draw(maze: Maze, run_: dict, found: tuple, row: dict, path: Path) -> None:
    canvas = Canvas(maze.world)
    canvas.walls(maze.walls, MAP_COLOUR)
    log = run_["log"]
    true_xy = np.array([t["true"][:2] for t in log])
    estimate_xy = maze.frame.to_world(np.array([t["estimate"][:2] for t in log]))
    for xy, colour in ((estimate_xy, (220, 100, 0)), (true_xy, (0, 0, 0))):
        line = np.round(canvas.pixels(xy)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas.image, [line], False, colour, 1, cv2.LINE_AA)
    for t in log:
        centre = tuple(int(v) for v in np.round(canvas.pixels(np.array(t["true"][:2]))))
        if t["replan"]:
            cv2.circle(canvas.image, centre, 3, (0, 140, 255), -1, cv2.LINE_AA)
        if t["guard"]:
            cv2.drawMarker(canvas.image, centre, (0, 0, 220), cv2.MARKER_TILTED_CROSS, 8, 2)
    canvas.marker(tuple(maze.world.target), (0, 170, 0), arrow=True)
    canvas.marker(maze.frame.pose_to_world(tuple(maze.consistent_goal)), (220, 100, 0), arrow=True)
    canvas.marker(maze.frame.pose_to_world(found), (0, 0, 220), arrow=True)
    canvas.label(
        f"maze {maze.seed} goal={row['goal']}: {row['tier']} {row['distance_m']:.2f} m "
        f"{row['heading_deg']:.0f} deg, {row['ticks']} ticks, {row['checkin_reason']}, "
        f"{row['wall_guard']} guards, {row['replans']} replans"
    )
    canvas.save(path)


# -- the run


def evaluate(seed: int, args, out: Path) -> tuple[list[dict], list[dict]]:
    root = record(seed)
    agents = {"": ServerFrames(root, config=map_agent.config_from(args))}
    first = next(iter(agents.values()))
    maze = Maze(seed, first, root)
    print(
        f"maze {seed}: {first.planner.n} nodes, true goal ({maze.true_goal[0]:.2f}, "
        f"{maze.true_goal[1]:.2f}) in the map frame",
        flush=True,
    )
    rows, ticks, runs = [], [], {}
    found = maze.true_goal
    for config, agent in agents.items():
        agent.encode = maze.world.jpeg
        for goal in args.goals:
            agent.goal_override = {
                "found": None,
                "true": maze.true_goal,
                "map-true": tuple(maze.consistent_goal),
            }[goal]
            for motion in range(args.seeds):
                run_ = drive(agent, maze, args, motion)
                if goal == "found" and motion == 0 and run_["goal"] is not None:
                    found = run_["goal"]
                sdk = None if args.no_sdk else sdk_score(agent, maze, args, motion)
                row, tick_rows = score_run(maze, goal, run_, sdk, config, motion)
                if sdk is not None and sdk.nav_steps != row["ticks"]:
                    print(
                        f"  note: the SDK's run took {sdk.nav_steps} steps, "
                        f"the logged one {row['ticks']}"
                    )
                rows.append(row)
                ticks.extend(tick_rows)
                if motion == 0:
                    runs[config, goal] = run_
                print(
                    f"  {config or 'flags':<8} {goal:<8} #{motion} {row['tier']:<7} "
                    f"{row['distance_m']:.2f} m {row['heading_deg']:>5.0f} deg "
                    f"{row['ticks']:>5} ticks ({row['checkin_reason']})  "
                    f"points {row['score_points']:g}",
                    flush=True,
                )
        agent.store.sift.flush()
    for row in rows:
        if row["motion"] == 0:
            label = f"{seed}_{row['goal']}" + (f"_{row['config']}" if row["config"] else "")
            draw(maze, runs[row["config"], row["goal"]], found, row, out / f"{label}.png")
    return rows, ticks


def failure_stage(row: dict) -> str | None:
    """The first stage a run failed at (None: it succeeded)."""
    if (
        row["distance_m"] <= SUCCESS_M
        and abs(row["heading_deg"]) <= SUCCESS_DEG
        and row["goal_cm_err_m"] <= GOAL_ERROR_M
    ):
        return None
    if row["goal_cm_err_m"] > GOAL_ERROR_M:
        return STAGES[0]
    if row["lost_events"] > 0:
        return STAGES[1]
    if row["checkin_reason"] == "homing_max":
        return STAGES[2]
    if abs(row["heading_deg"]) > SUCCESS_DEG:
        return STAGES[3]
    return STAGES[4]


def summarise_configs(rows: list[dict]) -> None:
    """Per config and goal: points, success rate, SPL and the failure stages."""
    print(
        f"\n{'config':<9} {'goal':<8} {'runs':>4} | {'points':>13} {'success':>7} {'spl':>5} | "
        + " ".join(f"{s:>17}" for s in STAGES)
    )
    for config in dict.fromkeys(r["config"] for r in rows):
        for goal in dict.fromkeys(r["goal"] for r in rows if r["config"] == config):
            chosen = [r for r in rows if r["config"] == config and r["goal"] == goal]
            points = np.array([r["score_points"] for r in chosen])
            stages = Counter(failure_stage(r) for r in chosen)
            print(
                f"{config:<9} {goal:<8} {len(chosen):>4} | "
                f"{points.mean():>6.2f} +- {points.std():<4.2f} "
                f"{np.mean([failure_stage(r) is None for r in chosen]):>7.0%} "
                f"{np.mean([r['spl'] for r in chosen]):>5.2f} | "
                + " ".join(f"{stages[s]:>17}" for s in STAGES)
            )
    print(
        f"(points: mean +- std over runs; success: distance <= {SUCCESS_M} m, heading <= "
        f"{SUCCESS_DEG:g} deg and goal error <= {GOAL_ERROR_M} m; stages count the failures)"
    )


def summarise(rows: list[dict]) -> None:
    head = (
        f"{'maze':>4} {'goal':<8} | {'tier':<7} {'dist':>5} {'head':>5} {'spl':>5} {'ticks':>5} "
        f"{'check-in':<10} | {'points':>6} {'D/H/S':>7} | {'loc med':>7} {'cm med':>6} "
        f"{'false':>5} {'lost':>4} | {'guard':>5} {'replans':>7} {'stall':>5} | "
        + " ".join(f"{m:>6}" for m in MODES)
    )
    print("\n" + head + "\n" + "-" * len(head))
    for r in rows:
        blocks = "/".join(f"{r[f'score_{b.lower()}']:g}" for b in BLOCKS)
        print(
            f"{r['seed']:>4} {r['goal']:<8} | {r['tier']:<7} {fmt(r['distance_m'], 2):>5} "
            f"{fmt(r['heading_deg'], 0):>5} {r['spl']:>5.2f} {r['ticks']:>5} "
            f"{r['checkin_reason']:<10} | {r['score_points']:>3g}/{r['score_max']:<2g} "
            f"{blocks:>7} | {fmt(r['loc_median_m'], 2):>7} {fmt(r['cm_median_m'], 2):>6} "
            f"{r['false_fixes']:>5} {r['lost_events']:>4} | {r['wall_guard']:>5} "
            f"{r['replans']:>7} {r['stall_ticks']:>5} | "
            + " ".join(f"{r[f'mode_{m}']:>6.2f}" for m in MODES)
        )
    print("\nticks by action, mode, and SPL (oracle_steps / ticks):")
    print(
        f"{'maze':>4} {'goal':<8} | {'ticks':>5} {'oracle':>6} {'spl':>5} | "
        + " ".join(f"{a.replace('FORWARD', 'FWD'):>9}" for a in ACTIONS)
        + f" | {'turning':>7} | "
        + " ".join(f"{m:>5}" for m in MODES)
        + f" | {'D':>4} {'H':>4} {'S':>5}"
    )
    for r in rows:
        print(
            f"{r['seed']:>4} {r['goal']:<8} | {r['ticks']:>5} {r['oracle_steps']:>6} "
            f"{r['spl_ticks']:>5.2f} | "
            + " ".join(f"{r['act_' + a.lower().replace('|', '_')]:>9}" for a in ACTIONS)
            + f" | {r['turn_ticks']:>7} | "
            + " ".join(f"{r[f'ticks_{m}']:>5}" for m in MODES)
            + f" | {r['score_distance']:>4g} {r['score_heading']:>4g} {r['score_spl']:>5.2f}"
        )
    print("\nfinal error decomposition (metres; heading at check-in in degrees):")
    print(
        f"{'maze':>4} {'goal':<8} | {'goal vs cm':>10} {'est vs goal':>11} {'est vs cm':>9} "
        f"{'heading':>7} | {'homing':<9} {'ticks':>5} {'fixes':>5} | {'arrival':>7}"
    )
    for r in (r for r in rows if r["goal"] != "true"):
        print(
            f"{r['seed']:>4} {r['goal']:<8} | {fmt(r['goal_cm_err_m'], 2):>10} "
            f"{fmt(r['final_to_goal_m'], 2):>11} {fmt(r['final_cm_err_m'], 2):>9} "
            f"{fmt(r['heading_deg'], 0):>7} | {r['home_outcome']:<9} {r['home_ticks']:>5} "
            f"{r['home_fixes']:>5} | {fmt(r['distance_m'], 2):>7}"
        )
    print("\nmean points / max points:")
    for goal in GOALS:
        for label, keep in (
            ("mazes 7-9", lambda s: s < HELD_OUT[0]),
            ("mazes 10-12", lambda s: s >= HELD_OUT[0]),
        ):
            chosen = [r for r in rows if r["goal"] == goal and keep(r["seed"])]
            if chosen:
                points = np.mean([r["score_points"] for r in chosen])
                best = np.mean([r["score_max"] for r in chosen])
                print(
                    f"  goal={goal:<5} {label:<11} {points:.2f} / {best:g}  ({len(chosen)} mazes)"
                )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mazes", default="7,8,9,10,11,12", help="comma-separated maze seeds")
    p.add_argument("--goals", default=",".join(GOALS), help="found, true or both")
    p.add_argument("--seeds", type=int, default=SEEDS, help="motion seeds per maze and goal")
    p.add_argument("--out-dir", default="results/eval", help="where closed_loop/ goes")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS, help="the session's step budget")
    p.add_argument("--no-sdk", action="store_true", help="skip the SDK scorer's second run")
    map_agent.add_config_arguments(p)
    args = p.parse_args()
    args.goals = [g for g in args.goals.split(",") if g]

    stored_floor()
    out = Path(args.out_dir) / "closed_loop"
    rows: list[dict] = []
    ticks: list[dict] = []
    for seed in ints(args.mazes):
        maze_rows, maze_ticks = evaluate(seed, args, out)
        rows.extend(maze_rows)
        ticks.extend(maze_ticks)
    write_csv(out / "runs.csv", FIELDS, rows)
    write_csv(out / "ticks.csv", TICK_FIELDS, ticks)
    summarise(rows)
    summarise_configs(rows)


if __name__ == "__main__":
    main()
