"""A local evaluation that mimics the development challenge (515b23a3c14a) as closely as
vis_nav_sim allows: MapAgent through the SDK's run() path, ten goals in one 31x31 maze.

    uv run source/eval/eval_dev_like.py
    uv run source/eval/eval_dev_like.py --seed 102 --goals 4 --agent-args "--align-deg 6"
    uv run source/eval/eval_dev_like.py --seed 301 --maze-size 51 --routes 3

MapAgent's flags are --agent-args, by default "" (MapAgent's own defaults, map_agent.Config).

Environment: a 31x31 maze of 0.2 m cells (Simulator rows / cols), msaa 4, camera 320x240 with
fx 92, cx 160, cy 120 (asserted), a 10,000 step budget and the SDK's default motion noise. The
exploration data is Simulator.record(): 2 routes, student_files=False (poses.json), at
CAPTURE_EVERY = 3, because the simulator cannot change its 0.03 m / 3.44 degree tick (the dev
challenge's frames move 0.08 m and 9-10 degrees) and three ticks per frame is the nearest. It is
recorded to data/eval/dev_like/<seed>/ once (dev_like.json holds the settings) and reused; the
frozen map is frozen_map.get's, reused too (--rebuild rebuilds it).

Goals: goal 0 is the simulator's own target; goals 1.. are random free cell centres with a random
yaw, each within GOAL_POSE_M of a recorded pose, spread over short / medium / long shortest paths
(terciles of the candidates' distance from the spawn). The simulator cannot move its target, so
goals 1.. are photographed with render() at the goal pose turned 90 degrees * k, from the camera
of height_offset --goal-photo-offset (default 0.10, TARGET_HEIGHT_OFFSET, targets()' camera; 0.15
is the frame camera's). Goal 0 keeps targets(), whose camera is the 0.10 one; a render() from that
camera is checked against it (pixel and geometry.ranges differences are printed and go in
summary.md).

Each run is vis_nav_sdk.run(agent, "local:<seed>") on a LocalSession subclass (31x31 maze, msaa 4,
the goal's photos as info.targets, the goal's pose for distance_heading() and the check-in). The
frames are JPEG-encoded as the server sends them (eval_closed_loop.ServerFrames). Goal 0's result
is the SDK's untouched; for the others the SDK's own score() is applied to the custom goal.

Scoring is the challenge's card, implemented here and compared with the SDK's: a check-in scores
only within 0.2 m and 20 degrees; Distance 3 / 2 / 1 at 0.1 / 0.15 / 0.2 m, Heading 3 / 2 / 1 at
10 / 15 / 20 degrees, SPL 4 / 3 / 2 / 1 at 0.95 / 0.85 / 0.75 / 0.5; the first matching tier of a
block pays and the blocks add up. SPL = success * shortest / max(shortest, taken) in steps, the
shortest being the simulator's oracle steps (for goals 1.. the same route search, checked against
the oracle on goal 0).

Writes to results/eval/dev_like/<seed>/: runs.csv (a row per run), ticks.csv (a row per tick),
run_<goal>[_r<k>].png (grey: the agent's occupancy grid; thin: the true walls; black: the true
path; blue: the estimated path; orange / purple: the first / last planned path; green: the true
goal; red: the found goal; magenta cross: the check-in pose) and summary.md.

The localiser's choice of visual odometry or the calibrated motion is read by wrapping
localiser.visual_odometry (it returns what it returns); wall contact is not reported by the
simulator, so a FORWARD / BACKWARD tick that moved under 20% of a tick is counted as one.
"""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import json
import math
import os
import shutil
import sys
import time
import types
from collections import Counter
from pathlib import Path
from unittest import mock

import cv2
import frozen_map
import geometry
import goal as goal_module
import localiser as localiser_module
import map_agent
import mapping
import numpy as np
import vis_nav_sim as sim
from common import cell_centre, cell_of, fmt, geodesic, textures, write_csv
from eval_closed_loop import ServerFrames, longest_stall, spread
from eval_cues import Walls, stored_floor
from eval_goal_photos import range_difference
from eval_map import MAP_COLOUR, Canvas, Frame, MapTruth, true_poses
from geometry import ranges
from scipy.spatial import cKDTree
from vis_nav_sdk import Action, Observation, run
from vis_nav_sdk import local as sdk_local
from vis_nav_sdk.session import Arrival, Result

ROWS = COLS = 31
MSAA = 4
MAX_STEPS = 10_000
ROUTES = 2
CAPTURE_EVERY = 3
CAMERA = {"width": 320, "height": 240, "fx": 92.0, "cx": 160.0, "cy": 120.0}
TARGET_HEIGHT_OFFSET = 0.10
"""The camera height of targets(): render() with this height_offset reproduces it (checked)."""
GOAL_POSE_M = 0.1
"""A goal lies within this of a recorded exploration pose."""
MIN_GOAL_GAP_M = 0.6
MIN_GOAL_PATH_M = 1.0
MOTION_SEED_OFFSET = 1000
SUCCESS_M, SUCCESS_DEG = 0.2, 20.0
DISTANCE_TIERS = ((0.1, 3), (0.15, 2), (0.2, 1))
HEADING_TIERS = ((10, 3), (15, 2), (20, 1))
SPL_TIERS = ((0.95, 4), (0.85, 3), (0.75, 2), (0.5, 1))
GOAL_ERROR_M = 0.2
LOC_ERROR_M = 0.2
BLOCKED_FRACTION = 0.2
"""A FORWARD / BACKWARD tick that moved under this fraction of a tick touched a wall."""
SPAWN_M, SPAWN_RAD = 0.2, 0.25
ROUTE_M, ROUTE_RAD = 0.4, 0.5
"""How far a route's first frame (CAPTURE_EVERY ticks in) may be from the spawn and from the other
route's: the most a few noisy ticks move the robot."""
TICK_M = sim.LINEAR_SPEED * sim.TICK_SECONDS
TICK_DEG = math.degrees(sim.ANGULAR_SPEED * sim.TICK_SECONDS)
DATA_ROOT = Path("data") / "eval" / "dev_like"
OUT_ROOT = Path("results") / "eval" / "dev_like"
BUCKETS = ("short", "medium", "long")
FAILURES = ("wrong goal", "localisation", "out of steps", "other")

FIELDS = [
    "goal",
    "repeat",
    "motion_seed",
    "bucket",
    "true_x",
    "true_y",
    "true_yaw_deg",
    "shortest_m",
    "shortest_steps",
    "found_x",
    "found_y",
    "found_yaw_deg",
    "goal_err_m",
    "goal_cm_err_m",
    "goal_yaw_cm_err_deg",
    "checkin_reason",
    "outcome",
    "final_distance_m",
    "heading_err_deg",
    "agent_heading_err_deg",
    "view_k",
    "view_inliers",
    "view_yaw_deg",
    "points_distance",
    "points_heading",
    "points_spl",
    "points",
    "success",
    "nav_steps",
    "spl",
    "homing_decisions",
    "loc_err_checkin_m",
    "loc_median_m",
    "loc_max_m",
    "cm_err_checkin_m",
    "cm_median_m",
    "cm_max_m",
    "fixes_accepted",
    "fixes_held",
    "fixes_rejected",
    "fixes_layout_failed",
    "lost_events",
    "wall_guard_events",
    "online_fallback_ticks",
    "online_forward_dx_m",
    "online_backward_dx_m",
    "online_left_deg",
    "online_right_deg",
    "online_turn_rate_deg",
    "online_samples_fb_lr",
    "wall_contact_ticks",
    "stall_ticks",
    "act_ms_mean",
    "act_ms_p95",
    "turn_rate_deg",
    "fallback_forward_dx_m",
    "fallback_left_deg",
    "fallback_right_deg",
    "applied_forward_dx_mean_m",
    "applied_forward_dx_median_m",
    "applied_forward_n",
    "applied_left_deg",
    "applied_right_deg",
    "vo_calls",
    "vo_fallback_frac",
    "sdk_points",
    "sdk_tier",
    "sdk_spl",
    "sdk_nav_steps",
    "sdk_agrees",
    "failure",
]
TICK_FIELDS = [
    "goal",
    "repeat",
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
    "guard",
    "path_id",
    "estw_x",
    "estw_y",
    "estw_yaw",
    "fix",
    "rejected",
    "lost",
    "vo_fallback",
    "vo_source",
    "grey_diff",
    "blocked",
    "geodesic",
]


# -- the simulator

SIM_MODULE = types.SimpleNamespace(**{n: getattr(sim, n) for n in dir(sim)})


class DevSimulator(sim.Simulator):
    """vis_nav_sim.Simulator on the dev-like maze."""

    def __init__(self, textures_, seed: int, **kwargs) -> None:
        super().__init__(textures_, seed, rows=ROWS, cols=COLS, **kwargs)


SIM_MODULE.Simulator = DevSimulator


def pose_error(pose, goal) -> tuple[float, float]:
    """``(metres, degrees)`` from ``pose`` to ``goal``, as the simulator's distance_heading()."""
    heading = abs(math.degrees(geometry.wrap(pose[2] - goal[2])))
    return math.hypot(pose[0] - goal[0], pose[1] - goal[1]), heading


@dataclasses.dataclass
class Goal:
    id: int
    pose: tuple[float, float, float]
    cell: tuple[int, int]
    bucket: str
    photos: list[np.ndarray]
    shortest_m: float
    oracle_steps: int
    photo_source: str = ""


class DevSession(sdk_local.LocalSession):
    """A LocalSession on the 31x31 maze with a goal's photos, pose and oracle."""

    def __init__(self, seed: int, *, goal: Goal, **kwargs) -> None:
        with mock.patch.object(sdk_local, "_simulator_module", lambda: SIM_MODULE):
            super().__init__(seed, **kwargs)
        self.goal = goal
        self._info = dataclasses.replace(self._info, targets=goal.photos)

    def distance_heading(self) -> tuple[float, float]:
        if self.goal.id == 0:
            return self.simulator.distance_heading()
        return pose_error(self.simulator.pose(), self.goal.pose)

    def checkin(self) -> Result:
        """Goal 0: the SDK's own. Otherwise its code, with the goal's distance and oracle."""
        if self.goal.id == 0:
            return super().checkin()
        self._require_open()
        distance_m, heading_deg = self.distance_heading()
        nav_steps = self.simulator.steps
        oracle = self.goal.oracle_steps
        efficiency = oracle / max(nav_steps, oracle)
        tier, points, awarded = sdk_local.score(distance_m, heading_deg, efficiency)
        reached = tier != "failed"
        self._result = Result(
            goal_tier=tier,
            nav_steps=nav_steps,
            spl=efficiency if reached else 0.0,
            arrival=Arrival(distance_m=distance_m, heading_deg=heading_deg) if reached else None,
            points=points,
            max_points=float(sdk_local.RUBRIC_SPEC["max_points"]),
            awarded=awarded,
            job_id=f"dev-like-{self.seed}-{self.goal.id}",
            session_id=self.session_id,
            think=self.telemetry.summary(),
        )
        self._closed = True
        return self._result


class SessionFactory:
    """Stands in for vis_nav_sdk.local.local_session while run() is called."""

    goal: Goal | None = None
    session: DevSession | None = None

    def __call__(self, target: str, **options) -> DevSession:
        self.session = DevSession(sdk_local.seed_of(target), goal=self.goal, **options)
        return self.session


# -- the shortest path


def oracle_route(world: sim.Simulator, cell: tuple[int, int], yaw: float) -> tuple[float, float]:
    """``(metres, turn degrees)`` of the cheapest 4-connected route from the spawn (facing its
    yaw) to ``cell`` and then ``yaw``: the simulator's oracle, which it reproduces exactly."""
    free = ~world.occupancy
    heading0 = math.degrees(world.spawn[2])
    final = math.degrees(yaw)

    def turn(a: float, b: float) -> float:
        return abs((b - a + 180.0) % 360.0 - 180.0)

    queue = [(0.0, 0.0, 0.0, *world.spawn_cell, -1)]
    seen = set()
    while queue:
        cost, metres, turns, row, col, k = heapq.heappop(queue)
        if (row, col, k) in seen:
            continue
        seen.add((row, col, k))
        heading = heading0 if k < 0 else 90.0 * k
        if (row, col) == tuple(cell):
            return metres, turns + turn(heading, final)
        for k2, (dr, dc) in enumerate(((0, 1), (1, 0), (0, -1), (-1, 0))):
            r, c = row + dr, col + dc
            if 0 <= r < free.shape[0] and 0 <= c < free.shape[1] and free[r, c]:
                t = turn(heading, 90.0 * k2)
                step = t / TICK_DEG + world.cell_size / TICK_M
                heapq.heappush(queue, (cost + step, metres + world.cell_size, turns + t, r, c, k2))
    raise RuntimeError(f"no route to cell {cell}")


def oracle_steps(metres: float, turn_deg: float) -> int:
    return math.ceil(metres / TICK_M + turn_deg / TICK_DEG)


# -- the scoring card


def card(distance: float, heading: float, spl: float) -> tuple[int, int, int]:
    """Points for the Distance, Heading and SPL blocks of a check-in."""
    if not (distance <= SUCCESS_M and heading <= SUCCESS_DEG):
        return 0, 0, 0

    def block(value: float, tiers, *, at_most: bool = True) -> int:
        return next((p for t, p in tiers if (value <= t if at_most else value >= t)), 0)

    return (
        block(distance, DISTANCE_TIERS),
        block(heading, HEADING_TIERS),
        block(spl, SPL_TIERS, at_most=False),
    )


# -- the exploration data


def explore(seed: int, world: sim.Simulator) -> tuple[Path, dict]:
    """data/eval/dev_like/<seed>/ with its poses.json files, recorded if it is not there with
    these settings; ``(root, record summary)``."""
    size = "" if ROWS == 31 else f"-m{ROWS}"
    size += "" if CAPTURE_EVERY == 3 else f"-c{CAPTURE_EVERY}"
    root = DATA_ROOT / (str(seed) + size + os.environ.get("DEV_LIKE_CACHE_SUFFIX", ""))
    want = {
        "rows": ROWS,
        "cols": COLS,
        "msaa": MSAA,
        "routes": ROUTES,
        "capture_every": CAPTURE_EVERY,
        "seed": seed,
        "noise": list(sim.DEFAULT_NOISE),
    }
    meta = root / "dev_like.json"
    if meta.exists() and (root / "traj_0" / "poses.json").exists():
        saved = json.loads(meta.read_text())
        if saved["settings"] == want:
            return root, saved["summary"]
    if root.exists():
        shutil.rmtree(root)
    print(f"recording {root} ...", flush=True)
    summary = world.record(
        root,
        routes=ROUTES,
        seed=seed,
        capture_every=CAPTURE_EVERY,
        student_files=False,
    )
    meta.write_text(json.dumps({"settings": want, "summary": summary}))
    return root, summary


def check_spawn(world: sim.Simulator, truth: list[np.ndarray]) -> dict:
    """Both routes start at the spawn (to within the CAPTURE_EVERY ticks before the first frame);
    exit if not."""
    first = [tuple(t[0]) for t in truth]
    spawn = world.spawn
    report = {"spawn": spawn, "first": first}
    problems = []
    for i, pose in enumerate(first):
        d, h = pose_error(pose, spawn)
        report[f"route{i}_vs_spawn"] = (d, h)
        if d > SPAWN_M or math.radians(h) > SPAWN_RAD:
            problems.append(f"route {i}'s first frame is {d:.3f} m, {h:.1f} deg from the spawn")
    d, h = pose_error(first[0], first[1])
    report["route0_vs_route1"] = (d, h)
    if d > ROUTE_M or math.radians(h) > ROUTE_RAD:
        problems.append(f"the routes' first frames are {d:.3f} m, {h:.1f} deg apart")
    if problems:
        print("the routes do not start at the spawn pose: " + "; ".join(problems))
        sys.exit(1)
    return report


# -- the goals


def goal_photos(photo_world: sim.Simulator, pose) -> list[np.ndarray]:
    x, y, yaw = pose
    return [photo_world.render(x, y, yaw + k * math.pi / 2) for k in range(4)]


def choose_goals(
    world: sim.Simulator, photo_world: sim.Simulator, truth: list[np.ndarray], args
) -> list[Goal]:
    """Goal 0, the simulator's, and ``args.goals - 1`` random ones (see the module doc)."""
    rng = np.random.default_rng(args.seed)
    free = ~world.occupancy
    tree = cKDTree(np.concatenate(truth)[:, :2])
    distance = geodesic(world, tuple(world.spawn_cell))
    target = world.target
    taken = [target[:2]]
    candidates = []
    for cell in np.argwhere(free):
        cell = tuple(int(v) for v in cell)
        centre = cell_centre(world, cell)
        if not (MIN_GOAL_PATH_M <= distance[cell] < np.inf):
            continue
        if tree.query(centre)[0] > GOAL_POSE_M or math.dist(centre, target[:2]) < MIN_GOAL_GAP_M:
            continue
        candidates.append(cell)
    metres = np.array([distance[c] for c in candidates])
    edges = np.quantile(metres, [1 / 3, 2 / 3])
    pools = [
        [c for c, m in zip(candidates, metres, strict=True) if np.searchsorted(edges, m) == b]
        for b in range(3)
    ]
    sp = world.shortest_path
    goals = [
        Goal(
            0,
            tuple(target),
            tuple(world.target_cell),
            "sim target",
            world.targets(),
            sp["metres"],
            world.oracle_steps,
            f"targets() (height_offset {TARGET_HEIGHT_OFFSET})",
        )
    ]
    for n in range(1, args.goals):
        bucket = (n - 1) % 3
        pool = [
            c
            for c in pools[bucket]
            if all(math.dist(cell_centre(world, c), t) >= MIN_GOAL_GAP_M for t in taken)
        ]
        cell = pool[int(rng.integers(len(pool)))]
        centre = cell_centre(world, cell)
        pose = (centre[0], centre[1], float(rng.uniform(-math.pi, math.pi)))
        metres_, turn = oracle_route(world, cell, pose[2])
        taken.append(centre)
        goals.append(
            Goal(
                n,
                pose,
                cell,
                BUCKETS[bucket],
                goal_photos(photo_world, pose),
                metres_,
                oracle_steps(metres_, turn),
                f"render() (height_offset {args.goal_photo_offset})",
            )
        )
    return goals


def verify_replica(world: sim.Simulator) -> dict:
    """The route search against the simulator's oracle, and pose_error against its
    distance_heading() along a short drive."""
    metres, turn = oracle_route(world, tuple(world.target_cell), world.target[2])
    sp = world.shortest_path
    steps = oracle_steps(metres, turn)
    probe = DevSimulator(textures(), world.seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=5)
    worst = 0.0
    rng = np.random.default_rng(0)
    for _ in range(40):
        probe.step(int(rng.choice([sim.FORWARD, sim.LEFT, sim.RIGHT])), int(rng.integers(1, 6)))
        mine = pose_error(probe.pose(), probe.target)
        theirs = probe.distance_heading()
        worst = max(worst, abs(mine[0] - theirs[0]), abs(mine[1] - theirs[1]))
    return {
        "oracle_steps_sim": world.oracle_steps,
        "oracle_steps_replica": steps,
        "metres_sim": sp["metres"],
        "metres_replica": metres,
        "turn_sim": sp["turn_deg"],
        "turn_replica": turn,
        "distance_heading_gap": worst,
        "ok": steps == world.oracle_steps and worst < 1e-6,
    }


def verify_photos(world, photo_world, agent) -> dict:
    """Goal 0: render() at the target pose from the targets() camera against targets()."""
    photos = world.targets()
    fallback = goal_photos(photo_world, world.target)
    camera = goal_module.target_camera(agent.camera)
    pixel, worst, gaps, disagree = [], [], [], []
    for a, b in zip(fallback, photos, strict=True):
        diff = np.abs(a.astype(float) - b)
        pixel.append(float(diff.mean()))
        worst.append(float(diff.max()))
        gap, _, only_one = range_difference(
            ranges(a, camera, agent.floor)[1], ranges(b, camera, agent.floor)[1]
        )
        gaps.append(gap)
        disagree.append(only_one)
    return {
        "pixel_mean": float(np.mean(pixel)),
        "pixel_max": float(np.max(worst)),
        "range_gap_m": float(np.nanmean(gaps)) if not np.all(np.isnan(gaps)) else 0.0,
        "range_disagree": float(np.mean(disagree)),
        "exact": float(np.max(worst)) == 0.0,
    }


# -- the agent under test

VO = {"called": False, "fallback": False}
_visual_odometry = localiser_module.visual_odometry


def counted_visual_odometry(previous, view, rng):
    step = _visual_odometry(previous, view, rng)
    VO["called"], VO["fallback"] = True, step is None or not mapping.plausible(step)
    return step


class Truth:
    """The maze's truth against the agent's map, for the tick log."""

    def __init__(self, world, agent: map_agent.MapAgent, truth: list[np.ndarray]) -> None:
        self.world = world
        self.walls = Walls(world)
        self.truth = MapTruth(agent.store, truth)
        self.frame = Frame((0.0, 0.0, 0.0), tuple(truth[0][0]))
        self.free = ~world.occupancy


class DevAgent(ServerFrames):
    """MapAgent on JPEG frames that logs every tick against the truth."""

    factory: SessionFactory
    maze: Truth
    goal_info: Goal
    geodesic: np.ndarray

    def setup(self, info) -> None:
        super().setup(info)
        self.log: list[dict] = []
        self.plans: list = []
        self.act_s: list[float] = []
        self.arrival: tuple[float, float] | None = None
        self.previous_pose: tuple | None = None
        self.own_heading_deg = math.nan
        self.view_inliers: int | None = None

    def _yaw_to_view0(self, image):
        """MapAgent's, plus the inlier count of the goal view it chose (``view_inliers``)."""
        offsets: list[tuple[object, int]] = []
        servo = self.servo
        original = servo.offset

        def offset(current, features, cx):
            result = original(current, features, cx)
            offsets.append((features, result[1]))
            return result

        servo.offset = offset
        try:
            yaw = super()._yaw_to_view0(image)
        finally:
            del servo.offset
        self.view_inliers = None
        if self.view:
            chosen = self.goal_features[self.view[0]]
            self.view_inliers = next(n for f, n in offsets if f is chosen)
        return yaw

    def act(self, obs: Observation):
        session = self.factory.session
        taken = self.last
        VO["called"] = VO["fallback"] = False
        started = time.perf_counter()
        decision, ticks = super().act(obs)
        self.act_s.append(time.perf_counter() - started)
        pose = session.pose()
        maze = self.maze
        estimate = self.update.estimate
        if self.update.replan:
            self.plans.append(self.update.path)
        applied = (
            geometry.between(self.previous_pose, pose)
            if self.previous_pose is not None and taken is not None
            else None
        )
        blocked = bool(
            applied is not None
            and taken in ("FORWARD", "BACKWARD")
            and abs(applied[0]) < BLOCKED_FRACTION * TICK_M
        )
        self.log.append(
            {
                "tick": obs.step,
                "true": pose,
                "map": tuple(maze.frame.to_map(np.array([pose]))[0]),
                "consistent": maze.truth.consistent(pose),
                "estimate": tuple(estimate[:3]),
                "sigma": estimate.sigma_m,
                "mode": self.mode,
                "action": "CHECKIN" if decision & Action.CHECKIN else self.last,
                "taken": taken,
                "applied": applied,
                "guard": self.guarded,
                "path": self.update.path,
                "status": estimate.status,
                "rejected": estimate.rejected,
                "lost": estimate.lost,
                "vo_fallback": VO["fallback"] if VO["called"] else None,
                "vo_source": self.localiser.source,
                "grey_diff": self.localiser.grey_diff,
                "blocked": blocked,
                "geodesic": float(self.geodesic[cell_of(maze.world, maze.free, *pose[:2])]),
            }
        )
        self.previous_pose = pose
        if decision & Action.CHECKIN:
            self.arrival = session.distance_heading()
            if self.goal is not None:
                self.own_heading_deg = (math.degrees(self.goal[2] - estimate.yaw) + 180) % 360 - 180
        return decision, ticks


def agent_parser() -> argparse.ArgumentParser:
    """MapAgent's own command-line flags (map_agent.py's __main__), for --agent-args."""
    p = argparse.ArgumentParser(prog="--agent-args", add_help=False)
    map_agent.add_config_arguments(p)
    return p


def agent_config(text: str) -> map_agent.Config:
    return map_agent.config_from(agent_parser().parse_args(text.split()))


# -- one run


def drive(agent: DevAgent, factory: SessionFactory, goal: Goal, repeat: int, args) -> tuple:
    """One run() of the agent on ``goal``: ``(sdk result, session)``."""
    factory.goal = agent.goal_info = goal
    agent.geodesic = geodesic(agent.maze.world, goal.cell)
    with mock.patch.object(sdk_local, "local_session", factory):
        result = run(
            agent,
            f"local:{args.seed}",
            viewer=False,
            quiet=True,
            check=False,
            confirm=False,
            browser=False,
            textures=textures(),
            motion_noise=sim.DEFAULT_NOISE,
            motion_seed=args.seed + MOTION_SEED_OFFSET + 1000 * repeat,
            max_steps=args.max_steps,
            msaa=MSAA,
        )
    return result, factory.session


def nan_stat(f, values, default: float = math.nan) -> float:
    return float(f(values)) if len(values) else default


def turn_mean(steps: np.ndarray) -> float:
    """Mean degrees turned by a tick of the given ``(dx, dy, dyaw)`` steps."""
    return float(np.degrees(steps[:, 2].mean())) if len(steps) else math.nan


def online_values(localiser) -> dict:
    """The online medians of the run (NaN where an action has under ONLINE_MIN samples), whether
    or not online motion used them."""

    def median(action: str, i: int, scale: float = 1.0) -> float:
        m = localiser.online(action)
        return math.nan if m is None else scale * m[i]

    rate = localiser.turn_rate()
    n = localiser.samples
    return {
        "online_forward_dx_m": median("FORWARD", 0),
        "online_backward_dx_m": median("BACKWARD", 0),
        "online_left_deg": median("LEFT", 2, 180 / math.pi),
        "online_right_deg": median("RIGHT", 2, 180 / math.pi),
        "online_turn_rate_deg": math.nan if rate is None else rate,
        "online_samples_fb_lr": "/".join(str(len(n.get(a, []))) for a in motion_actions()),
    }


def motion_actions() -> tuple[str, ...]:
    return ("FORWARD", "BACKWARD", "LEFT", "RIGHT")


def score_run(agent: DevAgent, goal: Goal, repeat: int, result, session, args) -> tuple:
    maze, log = agent.maze, agent.log
    column = lambda key: np.array([t[key] for t in log], float)  # noqa: E731
    estimate, consistent = column("estimate"), column("consistent")
    world_error = np.hypot(*(column("map")[:, :2] - estimate[:, :2]).T)
    cm_error = np.hypot(*(consistent[:, :2] - estimate[:, :2]).T)
    true_goal = maze.frame.to_map(np.array([goal.pose]))[0]
    consistent_goal = maze.truth.consistent(goal.pose)
    found = agent.found_goal
    nan3 = (math.nan,) * 3
    found_world = maze.frame.pose_to_world(found) if found is not None else nan3
    goal_err = math.dist(found[:2], true_goal[:2]) if found is not None else math.nan
    goal_cm_err = math.dist(found[:2], consistent_goal[:2]) if found is not None else math.nan
    goal_yaw_err = (
        abs(math.degrees(geometry.wrap(found[2] - consistent_goal[2])))
        if found is not None
        else math.nan
    )

    checked_in = result is not None
    distance, heading = agent.arrival if agent.arrival else session.distance_heading()
    nav_steps = session.simulator.steps
    spl = goal.oracle_steps / max(goal.oracle_steps, nav_steps)
    success = distance <= SUCCESS_M and heading <= SUCCESS_DEG
    points = card(distance, heading, spl) if checked_in else (0, 0, 0)
    success = success and checked_in
    outcome = "success" if success else ("miss" if checked_in else "no check-in")
    reason = agent.checkin_reason or "budget"

    counts = agent.localiser.counts
    applied = {
        a: np.array([t["applied"] for t in log if t["taken"] == a and t["applied"] is not None])
        for a in ("FORWARD", "LEFT", "RIGHT")
    }
    moved = applied["FORWARD"][:, 0] if len(applied["FORWARD"]) else np.empty(0)
    moved = moved[moved >= BLOCKED_FRACTION * TICK_M]
    vo = [t["vo_fallback"] for t in log if t["vo_fallback"] is not None]
    act_ms = np.array(agent.act_s) * 1e3
    motion = agent.motion

    sdk_points = 0.0 if result is None else result.points
    sdk_agrees = result is None or (
        abs(result.points - sum(points)) < 1e-9
        and result.nav_steps == nav_steps
        and (
            result.arrival is None
            or (
                abs(result.arrival.distance_m - distance) < 1e-6
                and abs(result.arrival.heading_deg - heading) < 1e-6
            )
        )
    )
    if not success:
        if not (goal_cm_err <= GOAL_ERROR_M):
            failure = FAILURES[0]
        elif cm_error[-1] > LOC_ERROR_M:
            failure = FAILURES[1]
        elif reason in ("low_steps", "budget"):
            failure = FAILURES[2]
        else:
            failure = FAILURES[3]
    else:
        failure = ""
    row = {
        "goal": goal.id,
        "repeat": repeat,
        "motion_seed": args.seed + MOTION_SEED_OFFSET + 1000 * repeat,
        "bucket": goal.bucket,
        "true_x": goal.pose[0],
        "true_y": goal.pose[1],
        "true_yaw_deg": math.degrees(goal.pose[2]),
        "shortest_m": goal.shortest_m,
        "shortest_steps": goal.oracle_steps,
        "found_x": found_world[0],
        "found_y": found_world[1],
        "found_yaw_deg": math.degrees(found_world[2]),
        "goal_err_m": goal_err,
        "goal_cm_err_m": goal_cm_err,
        "goal_yaw_cm_err_deg": goal_yaw_err,
        "checkin_reason": reason,
        "outcome": outcome,
        "final_distance_m": distance,
        "heading_err_deg": heading,
        "agent_heading_err_deg": agent.own_heading_deg,
        "view_k": agent.view[0] if agent.view else "",
        "view_inliers": agent.view_inliers if agent.view else "",
        "view_yaw_deg": agent.view[1] if agent.view else "",
        "points_distance": points[0],
        "points_heading": points[1],
        "points_spl": points[2],
        "points": sum(points),
        "success": int(success),
        "nav_steps": nav_steps,
        "spl": spl if success else 0.0,
        "homing_decisions": agent.homing_steps,
        "loc_err_checkin_m": float(world_error[-1]),
        "loc_median_m": spread("", world_error)["median_m"],
        "loc_max_m": float(world_error.max()),
        "cm_err_checkin_m": float(cm_error[-1]),
        "cm_median_m": spread("", cm_error)["median_m"],
        "cm_max_m": float(cm_error.max()),
        "fixes_accepted": counts["accepted"],
        "fixes_held": counts["held"],
        "fixes_rejected": counts["rejected"],
        "fixes_layout_failed": counts["layout"],
        "lost_events": agent.localiser.lost_events,
        "wall_guard_events": agent.guard_events,
        "online_fallback_ticks": sum(t["vo_source"] == "online" for t in log),
        **online_values(agent.localiser),
        "wall_contact_ticks": sum(t["blocked"] for t in log),
        "stall_ticks": longest_stall([t["geodesic"] for t in log]),
        "act_ms_mean": float(act_ms.mean()),
        "act_ms_p95": float(np.percentile(act_ms, 95)),
        "turn_rate_deg": agent.turn_rate,
        "fallback_forward_dx_m": motion["FORWARD"]["dx"],
        "fallback_left_deg": math.degrees(motion["LEFT"]["dyaw"]),
        "fallback_right_deg": math.degrees(motion["RIGHT"]["dyaw"]),
        "applied_forward_dx_mean_m": nan_stat(np.mean, moved),
        "applied_forward_dx_median_m": nan_stat(np.median, moved),
        "applied_forward_n": len(moved),
        "applied_left_deg": turn_mean(applied["LEFT"]),
        "applied_right_deg": turn_mean(applied["RIGHT"]),
        "vo_calls": len(vo),
        "vo_fallback_frac": float(np.mean(vo)) if vo else math.nan,
        "sdk_points": sdk_points,
        "sdk_tier": "none" if result is None else result.goal_tier,
        "sdk_spl": 0.0 if result is None else result.spl,
        "sdk_nav_steps": math.nan if result is None else result.nav_steps,
        "sdk_agrees": int(sdk_agrees),
        "failure": failure,
    }
    ticks = []
    plan_ids = {id(p): i for i, p in enumerate(agent.plans)}
    for t in log:
        estw = maze.frame.pose_to_world(t["estimate"][:3])
        ticks.append(
            {
                "goal": goal.id,
                "repeat": repeat,
                "tick": t["tick"],
                **dict(zip(("true_x", "true_y", "true_yaw"), t["true"], strict=True)),
                **dict(zip(("map_x", "map_y", "map_yaw"), t["map"], strict=True)),
                **dict(zip(("cm_x", "cm_y", "cm_yaw"), t["consistent"], strict=True)),
                **dict(zip(("est_x", "est_y", "est_yaw"), t["estimate"], strict=True)),
                "sigma": t["sigma"],
                "mode": t["mode"],
                "action": t["action"],
                "guard": int(t["guard"]),
                "path_id": plan_ids.get(id(t["path"]), -1),
                **dict(zip(("estw_x", "estw_y", "estw_yaw"), estw, strict=True)),
                "fix": t["status"] or "",
                "rejected": t["rejected"],
                "lost": int(t["lost"]),
                "vo_fallback": "" if t["vo_fallback"] is None else int(t["vo_fallback"]),
                "vo_source": t["vo_source"] or "",
                "grey_diff": t["grey_diff"],
                "blocked": int(t["blocked"]),
                "geodesic": t["geodesic"],
            }
        )
    return row, ticks


def draw(agent: DevAgent, goal: Goal, row: dict, path: Path) -> None:
    maze = agent.maze
    canvas = Canvas(maze.world)
    grid = agent.store.grid
    canvas.cells(maze.frame.to_world(grid.centres()[grid.occupied()]), MAP_COLOUR)
    for x0, x1, y0, y1 in maze.walls.rects:
        (u0, v0), (u1, v1) = canvas.pixels(np.array([[x0, y0], [x1, y1]]))
        cv2.rectangle(
            canvas.image,
            (int(round(u0)), int(round(v1))),
            (int(round(u1)), int(round(v0))),
            (170, 170, 170),
            1,
        )

    def line(xy: np.ndarray, colour, thickness: int = 1) -> None:
        pixels = np.round(canvas.pixels(xy)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas.image, [pixels], False, colour, thickness, cv2.LINE_AA)

    log = agent.log
    for plan, colour in ((agent.plans[:1], (0, 140, 255)), (agent.plans[-1:], (200, 0, 160))):
        for p in plan:
            line(maze.frame.to_world(p.waypoints[:, :2]), colour, 2)
    line(maze.frame.to_world(np.array([t["estimate"][:2] for t in log])), (220, 100, 0))
    line(np.array([t["true"][:2] for t in log]), (0, 0, 0))
    canvas.marker(goal.pose, (0, 170, 0), arrow=True)
    if agent.found_goal is not None:
        canvas.marker(maze.frame.pose_to_world(agent.found_goal), (0, 0, 220), arrow=True)
    end = tuple(int(v) for v in np.round(canvas.pixels(np.array(log[-1]["true"][:2]))))
    cv2.drawMarker(canvas.image, end, (200, 0, 200), cv2.MARKER_TILTED_CROSS, 14, 2)
    canvas.label(
        f"goal {goal.id} ({goal.bucket}): {row['outcome']} {row['final_distance_m']:.2f} m "
        f"{row['heading_err_deg']:.0f} deg, {row['nav_steps']} steps, {row['checkin_reason']}, "
        f"{row['points']:g} pts"
    )
    canvas.save(path)


# -- the summary


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def write_summary(rows: list[dict], env: dict, args, path: Path) -> str:
    def mean(key: str, subset=None) -> float:
        v = [r[key] for r in (subset if subset is not None else rows)]
        return float(np.mean(v))

    photos, replica, spawn = env["photos"], env["replica"], env["spawn"]
    lines = ["# Dev-like evaluation", ""]
    lines += [
        "## Environment",
        "",
        f"- Seed {args.seed}, maze {ROWS}x{COLS} cells of 0.2 m (default generator knobs: "
        f"room_density 0.1, max_room_size 3), msaa {MSAA}, camera 320x240 fx 92 cx 160 cy 120 "
        "(asserted against the simulator), step budget "
        f"{MAX_STEPS}, motion noise {sim.DEFAULT_NOISE} (the SDK default), motion seeds "
        f"{args.seed}+{MOTION_SEED_OFFSET}+1000*repeat (not the exploration's {args.seed}).",
        f"- Exploration: {env['summary']['routes']} routes, frames per route "
        f"{[t['frames'] for t in env['summary']['trajectories']]} "
        f"(total {env['summary']['frames']}; the dev challenge has 4,966), capture_every "
        f"{CAPTURE_EVERY}, student_files=False.",
        f"- Agent: MapAgent via vis_nav_sdk.run(), `--agent-args {args.agent_args}`; turn rate "
        f"{rows[0]['turn_rate_deg']:.2f} deg, fallback FORWARD dx "
        f"{rows[0]['fallback_forward_dx_m']:.3f} m.",
        f"- Spawn check: route 0's first frame is {spawn['route0_vs_spawn'][0]:.3f} m, "
        f"{spawn['route0_vs_spawn'][1]:.1f} deg from the spawn pose, route 1's "
        f"{spawn['route1_vs_spawn'][0]:.3f} m, {spawn['route1_vs_spawn'][1]:.1f} deg; the routes' "
        f"first frames are {spawn['route0_vs_route1'][0]:.3f} m, "
        f"{spawn['route0_vs_route1'][1]:.1f} deg apart (each is {CAPTURE_EVERY} ticks in).",
        "",
        "### Goal photos",
        "",
        f"- Goal photo offset used for goals 1+: --goal-photo-offset {args.goal_photo_offset} "
        "(the frame camera's height_offset is 0.15; 0.10 reproduces targets()).",
        "- Goal 0: targets() (height_offset 0.10) as is (raw frames, as the SDK's local session "
        "hands them out; targets() applies no JPEG). Goals 1+: the simulator cannot set its "
        "target, so render() at the goal pose turned 90 degrees * k, k = 0..3, with "
        f"height_offset {args.goal_photo_offset}, msaa {MSAA}. FLAGGED: these are not targets() "
        "output.",
        f"- Fallback against targets() on goal 0: mean pixel diff {photos['pixel_mean']:.4f}, "
        f"max {photos['pixel_max']:.0f}, geometry.ranges mean gap {photos['range_gap_m']:.4f} m, "
        f"columns where only one sees a wall {photos['range_disagree']:.4f} "
        f"({'exact' if photos['exact'] else 'NOT EXACT'}).",
        f"- Shortest path steps: the simulator's oracle for goal 0; for goals 1+ a replica "
        f"(min-turn route search, ceil(metres / {TICK_M} + degrees / {TICK_DEG:.4f})). On goal 0 "
        f"it gives {replica['oracle_steps_replica']} steps against the simulator's "
        f"{replica['oracle_steps_sim']}; my distance_heading differs from the simulator's by at "
        f"most {replica['distance_heading_gap']:.1e} ({'ok' if replica['ok'] else 'MISMATCH'}).",
        "",
        "### Deviations from the dev challenge",
        "",
        f"- Tick size: the simulator's tick is {TICK_M} m / {TICK_DEG:.2f} deg and cannot change; "
        f"the dev challenge's frames move 0.08 m / 9-10 deg. Exploration frames use "
        f"capture_every {CAPTURE_EVERY} (about {CAPTURE_EVERY * TICK_M:.2f} m / "
        f"{CAPTURE_EVERY * TICK_DEG:.1f} deg); the runs act and see a frame per tick "
        f"({TICK_M} m / {TICK_DEG:.2f} deg), unmodified agent.",
        f"- Frames: {env['summary']['frames']} against 4,966; no generator knob "
        "(room_density, max_room_size) gets 31x31 within 20% of that, so the defaults are used.",
        "- Wall contact is not reported by the simulator: counted as FORWARD / BACKWARD ticks "
        f"that moved under {BLOCKED_FRACTION:.0%} of a tick (inferred).",
        "- Goals 1+ are scored by the SDK's score() on the custom goal; only goal 0 is the SDK's "
        "check-in end to end.",
        "",
        "## Goals",
        "",
    ]
    per_goal = []
    for r in rows:
        per_goal.append(
            [
                f"{r['goal']}" + (f" r{r['repeat']}" if args.repeats > 1 else ""),
                r["bucket"],
                f"{r['shortest_m']:.1f} / {r['shortest_steps']}",
                f"{r['goal_cm_err_m']:.2f}",
                r["checkin_reason"],
                r["outcome"],
                f"{r['final_distance_m']:.2f}",
                f"{r['heading_err_deg']:.0f}",
                f"{r['points_distance']}/{r['points_heading']}/{r['points_spl']}",
                f"{r['points']:g}",
                r["nav_steps"],
                f"{r['spl']:.2f}",
                f"{r['cm_err_checkin_m']:.2f}",
                r["failure"] or "-",
            ]
        )
    lines.append(
        table(
            [
                "goal",
                "path",
                "shortest m / steps",
                "goal err (cm) m",
                "check-in",
                "outcome",
                "dist m",
                "head deg",
                "D/H/S",
                "pts",
                "steps",
                "SPL",
                "loc err (cm) m",
                "failure",
            ],
            per_goal,
        )
    )
    n = len(rows)
    lines += [
        "",
        "## Means",
        "",
        f"{n} runs: Distance {mean('points_distance'):.2f} / 3, Heading "
        f"{mean('points_heading'):.2f} / 3, SPL {mean('points_spl'):.2f} / 4, total "
        f"**{mean('points'):.2f} / 10**, success rate {mean('success'):.0%} "
        f"(mean SPL of the successes "
        f"{np.mean([r['spl'] for r in rows if r['success']] or [math.nan]):.2f}).",
        "",
    ]
    agree = [r for r in rows if not r["sdk_agrees"]]
    lines.append(
        "SDK score vs this card: all runs agree."
        if not agree
        else "SDK DISAGREEMENT on goals "
        + ", ".join(f"{r['goal']} (mine {r['points']:g}, SDK {r['sdk_points']:g})" for r in agree)
    )
    failed = [r for r in rows if not r["success"]]
    counts = Counter(r["failure"] for r in failed)
    lines += [
        "",
        "## Failures",
        "",
        table(
            ["cause", "runs"],
            [[c, counts[c]] for c in FAILURES],
        ),
        "",
        "Causes in order, the first that applies: wrong goal (found goal over 0.2 m from the "
        "map-consistent truth), localisation error at check-in over 0.2 m (against the "
        "map-consistent truth), out of steps (check-in `low_steps` or none), other (check-ins "
        f"by {sorted({r['checkin_reason'] for r in failed if r['failure'] == 'other'})}).",
        "",
        "## Motion and localisation",
        "",
        table(
            [
                "goal",
                "turn rate deg",
                "fallback dx m",
                "applied dx m (mean)",
                "fallback L/R deg",
                "applied L/R deg",
                "VO fallback",
                "contact ticks",
                "ms/step mean (p95)",
                "fixes a/h/r",
                "lost",
                "guards",
            ],
            [
                [
                    r["goal"],
                    f"{r['turn_rate_deg']:.2f}",
                    f"{r['fallback_forward_dx_m']:.3f}",
                    fmt(r["applied_forward_dx_mean_m"], 4),
                    f"{r['fallback_left_deg']:.1f}/{r['fallback_right_deg']:.1f}",
                    f"{fmt(r['applied_left_deg'], 2)}/{fmt(r['applied_right_deg'], 2)}",
                    fmt(r["vo_fallback_frac"], 3),
                    r["wall_contact_ticks"],
                    f"{r['act_ms_mean']:.0f} ({r['act_ms_p95']:.0f})",
                    f"{r['fixes_accepted']}/{r['fixes_held']}/{r['fixes_rejected']}",
                    r["lost_events"],
                    r["wall_guard_events"],
                ]
                for r in rows
            ],
        ),
        "",
    ]
    text = "\n".join(lines)
    path.write_text(text)
    return text


# -- the run


def main() -> None:
    global ROWS, COLS, ROUTES, CAPTURE_EVERY
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--seed", type=int, default=101, help="the maze (10: the midterm-like regression maze)"
    )
    p.add_argument("--maze-size", type=int, default=ROWS, help="maze rows and columns (cells)")
    p.add_argument("--routes", type=int, default=ROUTES, help="exploration routes recorded")
    p.add_argument(
        "--capture-every", type=int, default=CAPTURE_EVERY, help="ticks per exploration frame"
    )
    p.add_argument("--goals", type=int, default=10, help="goals, the simulator's included")
    p.add_argument("--repeats", type=int, default=1, help="runs per goal, other motion seeds")
    p.add_argument(
        "--agent-args",
        default="",
        help="MapAgent's flags (map_agent.py --help), as one string; replaces the default",
    )
    p.add_argument("--max-steps", type=int, default=MAX_STEPS, help="step budget (diagnosis only)")
    p.add_argument(
        "--goal-photo-offset",
        type=float,
        default=TARGET_HEIGHT_OFFSET,
        help="height_offset of the render() goal photos (goals 1+): "
        f"{TARGET_HEIGHT_OFFSET} reproduces targets(), 0.15 is the frame camera's",
    )
    p.add_argument("--rebuild", action="store_true", help="rebuild the frozen map")
    p.add_argument("--only", default="", help="comma-separated goal ids to run (default: all)")
    p.add_argument("--tag", default="", help="suffix of the output directory, <seed>-<tag>")
    args = p.parse_args()
    ROWS = COLS = args.maze_size
    ROUTES = args.routes
    CAPTURE_EVERY = args.capture_every
    out = OUT_ROOT / (f"{args.seed}-{args.tag}" if args.tag else str(args.seed))
    out.mkdir(parents=True, exist_ok=True)

    stored_floor()
    world = DevSimulator(
        textures(), args.seed, msaa=MSAA, motion_noise=sim.DEFAULT_NOISE, motion_seed=args.seed
    )
    assert (world.occupancy.shape, world.cell_size) == ((ROWS, COLS), 0.2), world.occupancy.shape
    for key, value in CAMERA.items():
        assert world.camera[key] == value, f"camera {key}: {world.camera[key]} != {value}"
    root, summary = explore(args.seed, world)
    routes = mapping.load_routes(root)
    truth = true_poses(root, routes)
    spawn = check_spawn(world, truth)
    print(
        f"maze {args.seed}: {summary['frames']} frames on {summary['routes']} routes "
        f"{[t['frames'] for t in summary['trajectories']]}",
        flush=True,
    )

    if args.rebuild:
        for name in ("motion.json", "links.npz"):
            (root / name).unlink(missing_ok=True)
        floor = map_agent.FloorMask.load(map_agent.FLOOR_MASK_PATH)
        frozen_map.get(root, dict(geometry.CAMERA), floor, rebuild=True)
    localiser_module.visual_odometry = counted_visual_odometry
    factory = SessionFactory()
    agent = DevAgent(root, config=agent_config(args.agent_args))
    agent.factory = factory
    agent.maze = Truth(world, agent, truth)
    agent.encode = world.jpeg
    print(
        f"agent: turn rate {agent.turn_rate:.2f} deg, fallback FORWARD dx "
        f"{agent.motion['FORWARD']['dx']:.3f} m, {agent.planner.n} nodes, links "
        f"{agent.planner.link_counts}",
        flush=True,
    )

    photo_world = DevSimulator(
        textures(),
        args.seed,
        msaa=MSAA,
        camera={**world.camera, "height_offset": args.goal_photo_offset},
    )
    targets_world = DevSimulator(
        textures(),
        args.seed,
        msaa=MSAA,
        camera={**world.camera, "height_offset": TARGET_HEIGHT_OFFSET},
    )
    replica = verify_replica(world)
    photos = verify_photos(world, targets_world, agent)
    print(f"route search vs oracle: {replica}\nrender vs targets(): {photos}", flush=True)
    goals = choose_goals(world, photo_world, truth, args)
    tree = cKDTree(np.concatenate(truth)[:, :2])
    for g in goals:
        near = tree.query(g.pose[:2])[0]
        print(
            f"goal {g.id} {g.bucket:<10} ({g.pose[0]:.2f}, {g.pose[1]:.2f}, "
            f"{math.degrees(g.pose[2]):.0f} deg) shortest {g.shortest_m:.1f} m / "
            f"{g.oracle_steps} steps, {near:.3f} m from a recorded pose, photos: {g.photo_source}"
            + ("" if near <= GOAL_POSE_M else "  (OUTSIDE the recorded poses)"),
            flush=True,
        )

    rows, ticks = [], []
    env = {"summary": summary, "photos": photos, "replica": replica, "spawn": spawn}
    only = {int(g) for g in args.only.split(",") if g}
    for goal in goals:
        if only and goal.id not in only:
            continue
        for repeat in range(args.repeats):
            result, session = drive(agent, factory, goal, repeat, args)
            row, tick_rows = score_run(agent, goal, repeat, result, session, args)
            rows.append(row)
            ticks.extend(tick_rows)
            name = f"run_{goal.id}" + (f"_r{repeat}" if repeat else "") + ".png"
            draw(agent, goal, row, out / name)
            plans = {
                i: {
                    "map": p.waypoints[:, :2].tolist(),
                    "rows": [int(agent.planner.row[n]) for n in p.nodes],
                }
                for i, p in enumerate(agent.plans)
            }
            (out / f"plans_{goal.id}_r{repeat}.json").write_text(json.dumps(plans))
            print(
                f"goal {goal.id} #{repeat}: {row['outcome']} {row['final_distance_m']:.2f} m "
                f"{row['heading_err_deg']:.0f} deg, {row['nav_steps']} steps "
                f"({row['checkin_reason']}), {row['points']:g} pts, goal err "
                f"{fmt(row['goal_cm_err_m'], 2)} m, SDK "
                f"{'agrees' if row['sdk_agrees'] else 'DISAGREES'}",
                flush=True,
            )
            agent.store.sift.flush()
    write_csv(out / "runs.csv", FIELDS, rows)
    write_csv(out / "ticks.csv", TICK_FIELDS, ticks)
    print("\n" + write_summary(rows, env, args, out / "summary.md"))


if __name__ == "__main__":
    main()
