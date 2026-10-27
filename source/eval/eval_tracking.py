"""
Offline poses for the exploration frames, a metric place graph over them, and online tracking
of a fresh run along a path in it, with visual fixes: no agent.

    uv run source/eval/eval_tracking.py
    uv run source/eval/eval_tracking.py --mazes 7 --ticks 500

Data: Simulator.record() output, data/eval/<seed>-r3-c1, with the true poses of poses.json
(eval_geometry.record_routes). The map frame has its origin at the spawn pose, which is where
every route starts; the truth in it is the world pose moved and turned to the spawn.

1. Offline poses, per route: calibrate_motion on that route's frames, then from (0, 0, 0) at the
   spawn dead reckoning ("dr"); with the yaw snapped to the manhattan candidate nearest it, at
   every frame with confidence ("snap"); and then scan_match, search +-
   MATCH_SEARCH_M / MATCH_SEARCH_DEG, at every SCAN_EVERY-th frame to the points of the scans of
   the previous WINDOW frames
   ("scan"; a match is taken as in eval_geometry). Final and largest position error and final
   yaw error against the truth.

2. Place graph over the "scan" poses: a node per exploration frame; an edge between consecutive
   frames of a route, and between any two frames within EDGE_M and EDGE_DEG (any route), weighed
   by metres plus a half metre a radian. The goal is eval_geometry.find_goal over all four
   target views; the path is the shortest from route 0's first frame. Nodes, edges, components,
   the goal frame, its view and inliers.

3. Online tracking: a fresh run from the spawn with another motion_seed, driven open-loop for
   --ticks ticks by the path's recorded actions (an edge forward is the actions of the frames it
   spans, backward their reverses; a switch to another route's frame none, and an edge off
   the recorded sequence weighs an extra JUMP_M). Each tick the tracker
   dead-reckons with the motion calibrated on the routes (their median), snaps the yaw to the
   manhattan candidate when confident, and estimates the turn rate: the running median of the
   manhattan yaw difference per tick over confident turn ticks (reported; --online-turn dead-reckons
   with it once it has MIN_RATE_SAMPLES). With fixes, it also matches the view with Servo
   (SIFT) against up to
   FIX_CANDIDATES exploration frames within FIX_RADIUS_M of the tracked pose (distinct poses,
   nearest first); at FIX_INLIERS inliers it resets the pose to that frame's, its yaw turned by
   the view's offset. Tracked-pose error against the truth over time, with and without fixes.

Writes results/eval/eval_tracking.csv.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as sim
from common import METRES_PER_RADIAN, ints, parser, textures, write_csv, yaw_error_deg
from eval_cues import stored_floor
from eval_geometry import (
    MATCH_AFTER,
    MIN_GAIN,
    SCAN_ACCEPT,
    Images,
    Route,
    complete_motion,
    data_root,
    estimate,
    find_goal,
    record_routes,
)
from geometry import (
    Camera,
    FloorMask,
    OccupancyMap,
    advance,
    calibrate_motion,
    pose_score,
    scan_match,
    snap,
)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from servo import Servo
from vlad import REVERSE

SCAN_EVERY = 5
"""Frames between the scans of the offline poses. Every frame snaps: a wall contact turns the
robot ten degrees a tick (poses.json), and a yaw dead-reckoned past 45 degrees between snaps
snaps to the wrong quadrant."""
WINDOW = 50
"""scan_match is to the points of the scans within this many frames before."""
MATCH_SEARCH_M, MATCH_SEARCH_DEG = 0.05, 2.0
EDGE_M, EDGE_DEG = 0.08, 45.0
JUMP_M = 0.5
"""Metres added to the weight of an edge that is not between consecutive frames: a hop between
routes is a switch at a crossing, not a way to travel, so a path keeps to the recorded
drives."""
REPLAY_MAX = 50
"""Frames of one route an edge may span and still be driven by the recorded actions between."""
FIX_RADIUS_M = 0.3
FIX_CANDIDATES = 16
FIX_INLIERS = 30
FIX_DISTINCT = (0.05, 10.0)
"""Candidates are distinct by at least this many metres or degrees from those taken."""
MIN_RATE_SAMPLES = 10
"""Turn-rate samples before the online rate replaces the calibrated one."""
MAX_RATE_DEG = 10.0
"""A turn tick's manhattan yaw difference per tick beyond this is a bad reading."""
REPORT_TICKS = 5
"""Rows of the error-over-time table."""
Pose = tuple[float, float, float]


def to_map(spawn: Pose, world: np.ndarray) -> np.ndarray:
    """``(n, 3)`` world poses in the frame whose origin is ``spawn``."""
    c, s = math.cos(-spawn[2]), math.sin(-spawn[2])
    d = world[:, :2] - np.array(spawn[:2])
    xy = np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=1)
    return np.column_stack([xy, world[:, 2] - spawn[2]])


# -- offline poses


def offline_poses(
    route: Route, estimates: list[dict], motion: dict[str, dict], mode: str
) -> np.ndarray:
    """``(N, 3)`` map-frame poses of the route's frames, ``mode`` one of "dr", "snap", "scan"."""
    pose: Pose = (0.0, 0.0, 0.0)
    window: deque[tuple[int, Pose, np.ndarray]] = deque()
    scans = 0
    poses = []
    for i in range(len(route.images)):
        pose = advance(pose, motion.get(route.motions[i]))
        e = estimates[i]
        if mode != "dr":
            local, manhattan = e["points"], e["manhattan"]
            if manhattan is not None:
                pose = (pose[0], pose[1], snap(pose[2], manhattan))
            if mode == "scan" and i % SCAN_EVERY == 0:
                while window and window[0][0] <= i - WINDOW:
                    window.popleft()
                if scans >= MATCH_AFTER and window:
                    grid = OccupancyMap()
                    for _, scan_pose, scan_points in window:
                        grid.add_scan(scan_pose, scan_points)
                    field = grid.likelihood_field()
                    matched, score = scan_match(
                        local,
                        field,
                        pose,
                        search=(MATCH_SEARCH_M, MATCH_SEARCH_M, MATCH_SEARCH_DEG),
                        manhattan_deg=manhattan,
                    )
                    if score >= SCAN_ACCEPT and score - pose_score(field, pose, local) >= MIN_GAIN:
                        pose = matched
                window.append((i, pose, local))
                scans += 1
        poses.append(pose)
    return np.array(poses)


# -- the place graph


def build_graph(poses: np.ndarray, route_of: np.ndarray) -> tuple[coo_matrix, int]:
    """Symmetric sparse weights over the nodes, and the edge count."""
    n = len(poses)
    pairs = cKDTree(poses[:, :2]).query_pairs(EDGE_M, output_type="ndarray")
    near = yaw_error_deg(poses[pairs[:, 0], 2], poses[pairs[:, 1], 2]) <= EDGE_DEG
    consecutive = np.nonzero(route_of[:-1] == route_of[1:])[0]
    edges = np.concatenate([pairs[near], np.stack([consecutive, consecutive + 1], axis=1)])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    a, b = edges[:, 0], edges[:, 1]
    turn = np.abs((poses[a, 2] - poses[b, 2] + np.pi) % (2 * np.pi) - np.pi)
    weight = np.hypot(*(poses[a, :2] - poses[b, :2]).T) + METRES_PER_RADIAN * turn + 1e-6
    weight = weight + JUMP_M * ~((route_of[a] == route_of[b]) & (b - a == 1))
    graph = coo_matrix(
        (np.concatenate([weight, weight]), (np.concatenate([a, b]), np.concatenate([b, a]))),
        shape=(n, n),
    ).tocsr()
    return graph, len(edges)


def plan_actions(
    path: list[int], route_of: np.ndarray, local_index: np.ndarray, routes: list[Route]
) -> list[str]:
    """The recorded actions along the node ``path``; "IDLE" for a record without a movement."""
    actions = []
    for a, b in zip(path, path[1:], strict=False):
        if route_of[a] != route_of[b] or abs(b - a) > REPLAY_MAX:
            continue  # a switch to another drive at the same place
        route = routes[route_of[a]]
        if b > a:
            actions += [route.motions[local_index[j]] or "IDLE" for j in range(a + 1, b + 1)]
        else:
            led = [route.motions[local_index[j]] for j in range(a, b, -1)]
            actions += [REVERSE[name] if name else "IDLE" for name in led]
    return actions


# -- online tracking


class Tracker:
    """A pose tracked from the spawn: odometry, a manhattan snap, an online turn rate, and
    visual fixes against the exploration frames (none without ``fixes``)."""

    def __init__(
        self,
        motion: dict[str, dict],
        camera: Camera,
        floor: FloorMask,
        servo: Servo,
        frames: list[Path],
        poses: np.ndarray,
        fixes: bool,
        use_rate: bool = False,
    ) -> None:
        self.motion = {name: dict(m) for name, m in motion.items()}
        self.camera, self.floor, self.servo = camera, floor, servo
        self.frames, self.poses, self.use_fixes = frames, poses, fixes
        self.use_rate = use_rate
        self.tree = cKDTree(poses[:, :2])
        self.pose: Pose = (0.0, 0.0, 0.0)
        self.previous: float | None = None  # the last tick's manhattan yaw, if confident
        self.rates: list[float] = []
        self.fixes = 0

    @property
    def turn_rate_deg(self) -> float:
        """The running median of the turn samples, degrees a tick (NaN before any)."""
        return float(np.median(self.rates)) if self.rates else math.nan

    def update(self, action: str, image: np.ndarray) -> Pose:
        self.pose = advance(self.pose, self.motion.get(action))
        e = estimate(image, self.camera, self.floor)
        manhattan = e["manhattan"]
        if action in ("LEFT", "RIGHT") and manhattan is not None and self.previous is not None:
            sign = 1.0 if action == "LEFT" else -1.0
            sample = sign * ((manhattan - self.previous + 45.0) % 90.0 - 45.0)
            if 0.0 < sample <= MAX_RATE_DEG:
                self.rates.append(sample)
                if self.use_rate and len(self.rates) >= MIN_RATE_SAMPLES:
                    rate = math.radians(self.turn_rate_deg)
                    self.motion["LEFT"]["dyaw"], self.motion["RIGHT"]["dyaw"] = rate, -rate
        if manhattan is not None:
            self.pose = (self.pose[0], self.pose[1], snap(self.pose[2], manhattan))
        self.previous = manhattan
        if self.use_fixes:
            self._fix(image)
        return self.pose

    def _fix(self, image: np.ndarray) -> None:
        near = self.tree.query_ball_point(self.pose[:2], FIX_RADIUS_M)
        if not near:
            return
        near = np.array(near)
        offset = np.hypot(
            *(self.poses[near, :2] - self.pose[:2]).T
        ) + METRES_PER_RADIAN * np.radians(
            yaw_error_deg(self.poses[near, 2], np.full(len(near), self.pose[2]))
        )
        chosen: list[int] = []
        for j in near[np.argsort(offset)]:
            if all(
                math.dist(self.poses[j, :2], self.poses[c, :2]) >= FIX_DISTINCT[0]
                or yaw_error_deg(self.poses[j, 2], self.poses[c, 2]) >= FIX_DISTINCT[1]
                for c in chosen
            ):
                chosen.append(int(j))
                if len(chosen) == FIX_CANDIDATES:
                    break
        best, best_inliers, best_yaw = None, 0, 0.0
        for j in chosen:
            yaw, inliers = self.servo.yaw_offset(image, self.frames[j])
            if yaw is not None and inliers > best_inliers:
                best, best_inliers, best_yaw = j, inliers, yaw
        if best is not None and best_inliers >= FIX_INLIERS:
            x, y, yaw = self.poses[best]
            # The recorded view lies to our right when the offset is positive: it faces less
            # counter-clockwise than we do.
            self.pose = (float(x), float(y), float(yaw + math.radians(best_yaw)))
            self.fixes += 1


def drive(world: sim.Simulator, actions: list[str]) -> tuple[list[bytes], np.ndarray]:
    """The run's JPEGs and true poses (map frame) per tick; the robot starts at the spawn."""
    spawn = world.spawn
    images, truth = [], []
    for name in actions:
        frame = world.step(getattr(sim, name))[-1]
        images.append(world.jpeg(frame))
        truth.append(world.pose())
    return images, to_map(spawn, np.array(truth))


def track_run(tracker: Tracker, actions: list[str], images: list[bytes], truth: np.ndarray) -> dict:
    estimated = []
    for name, data in zip(actions, images, strict=True):
        estimated.append(tracker.update(name, cv2.imdecode(np.frombuffer(data, np.uint8), 1)))
    estimated = np.array(estimated)
    error = np.hypot(*(estimated[:, :2] - truth[:, :2]).T)
    return {
        "error_m": error,
        "error_deg": yaw_error_deg(estimated[:, 2], truth[:, 2]),
        "fixes": tracker.fixes,
        "turn_rate_deg": tracker.turn_rate_deg,
        "rate_samples": len(tracker.rates),
    }


# -- the evaluation


def evaluate(seed: int, args: argparse.Namespace, floor: FloorMask) -> list[dict]:
    rows: list[dict] = []
    seconds: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        seconds[name] = now - clock
        clock = now

    world = sim.Simulator(textures(), seed)
    camera = Camera(world.camera)
    routes = record_routes(seed, 1)
    spawn = world.spawn
    print(f"\nmaze {seed}: {[len(r.images) for r in routes]} records on {len(routes)} routes")
    lap("data")

    # 1. offline poses
    motions, estimates, offline, truth = [], [], {}, []
    for route in routes:
        names = [None, *route.motions[1:]]
        motions.append(
            complete_motion(calibrate_motion(Images(route.images), names, camera, floor))
        )
        estimates.append([estimate(cv2.imread(str(path)), camera, floor) for path in route.images])
        truth.append(to_map(spawn, route.truth))
    lap("calibration")
    for mode in ("dr", "snap", "scan"):
        offline[mode] = [
            offline_poses(route, estimates[t], motions[t], mode) for t, route in enumerate(routes)
        ]
        for t in range(len(routes)):
            error = np.hypot(*(offline[mode][t][:, :2] - truth[t][:, :2]).T)
            yaw = yaw_error_deg(offline[mode][t][:, 2], truth[t][:, 2])
            rows.append(
                {
                    "section": "offline",
                    "seed": seed,
                    "trajectory": t,
                    "mode": mode,
                    "records": len(error),
                    "final_err_m": f"{error[-1]:.3f}",
                    "max_err_m": f"{error.max():.3f}",
                    "median_err_m": f"{np.median(error):.3f}",
                    "final_yaw_err_deg": f"{yaw[-1]:.2f}",
                }
            )
    lap("offline poses")

    # 2. the place graph and the goal
    poses = np.concatenate(offline["scan"])
    truth_all = np.concatenate(truth)
    route_of = np.concatenate([np.full(len(r.images), t) for t, r in enumerate(routes)])
    local_index = np.concatenate([np.arange(len(r.images)) for r in routes])
    first = np.cumsum([0] + [len(r.images) for r in routes])
    images = [path for r in routes for path in r.images]
    graph, edges = build_graph(poses, route_of)
    labels, _ = connected_components(graph, directed=False)
    t, i, view, _, inliers = find_goal(
        seed, data_root(seed, 1), routes, world.camera, 1, args.subsample, views=4
    )
    goal = int(first[t] + i)
    distance, predecessors = dijkstra(graph, directed=False, indices=0, return_predecessors=True)
    reachable = bool(np.isfinite(distance[goal]))
    path = []
    if reachable:
        node = goal
        while node >= 0:
            path.append(node)
            node = predecessors[node]
        path.reverse()
    actions = plan_actions(path, route_of, local_index, routes)
    goal_error = math.dist(truth_all[goal, :2], to_map(spawn, np.array([world.target]))[0, :2])
    route_text = (
        f"path {len(path)} nodes, {len(actions)} actions, {distance[goal]:.1f} weighed"
        if reachable
        else "unreachable from route 0"
    )
    print(
        f"  graph: {len(poses)} nodes, {edges} edges, {labels} components; goal frame route "
        f"{t} record {i} (view {view}, {inliers} inliers, {goal_error:.2f} m from the target), "
        + route_text
    )
    rows.append(
        {
            "section": "graph",
            "seed": seed,
            "nodes": len(poses),
            "edges": edges,
            "components": labels,
            "goal_route": t,
            "goal_record": i,
            "goal_view": view,
            "goal_inliers": inliers,
            "goal_to_target_m": f"{goal_error:.3f}",
            "reachable": reachable,
            "path_nodes": len(path),
            "path_actions": len(actions),
        }
    )
    lap("graph and goal")
    if not actions:
        print("  no path to follow: tracking skipped")
        return rows

    # 3. online tracking
    ticks = min(args.ticks, len(actions))
    actions = actions[:ticks]
    run = sim.Simulator(
        textures(), seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=seed + args.motion_seed
    )
    run.initial_frame()
    frames, run_truth = drive(run, actions)
    calibrated = {
        name: {
            key: float(np.median([m[name][key] for m in motions if name in m]))
            for key in ("dx", "dy", "dyaw")
        }
        for name in motions[0]
    }
    offline_rate = float(
        np.mean([abs(math.degrees(calibrated[name]["dyaw"])) for name in ("LEFT", "RIGHT")])
    )
    servo = Servo(camera.fx, cx=camera.cx, cache_size=2048)
    results = {}
    for label, fixes in (("fixes", True), ("no fixes", False)):
        tracker = Tracker(calibrated, camera, floor, servo, images, poses, fixes, args.online_turn)
        started = time.perf_counter()
        results[label] = track_run(tracker, actions, frames, run_truth)
        results[label]["seconds"] = time.perf_counter() - started
    lap("tracking")
    true_rate = math.degrees(sim.ANGULAR_SPEED * sim.TICK_SECONDS)
    for label, r in results.items():
        e, d = r["error_m"], r["error_deg"]
        rows.append(
            {
                "section": "tracking",
                "seed": seed,
                "variant": label,
                "ticks": ticks,
                "median_err_m": f"{np.median(e):.3f}",
                "p90_err_m": f"{np.percentile(e, 90):.3f}",
                "max_err_m": f"{e.max():.3f}",
                "final_err_m": f"{e[-1]:.3f}",
                "median_err_deg": f"{np.median(d):.2f}",
                "p90_err_deg": f"{np.percentile(d, 90):.2f}",
                "max_err_deg": f"{d.max():.2f}",
                "fixes": r["fixes"],
                "online_turn_deg": f"{r['turn_rate_deg']:.3f}",
                "offline_turn_deg": f"{offline_rate:.3f}",
                "true_turn_deg": f"{true_rate:.4f}",
                "rate_samples": r["rate_samples"],
                "seconds": f"{r['seconds']:.1f}",
            }
        )
        for tick in np.linspace(0, ticks - 1, REPORT_TICKS).astype(int):
            rows.append(
                {
                    "section": "timeline",
                    "seed": seed,
                    "variant": label,
                    "tick": int(tick) + 1,
                    "err_m": f"{e[tick]:.3f}",
                    "err_deg": f"{d[tick]:.2f}",
                }
            )
    rows.append({"section": "time", "seed": seed, **{k: f"{v:.1f}" for k, v in seconds.items()}})
    return rows


def section(rows: list[dict], name: str) -> list[dict]:
    return [r for r in rows if r["section"] == name]


def summarise(rows: list[dict]) -> None:
    print("\noffline poses against the truth (map frame at the spawn), metres / degrees")
    print(
        f"{'maze':>4} {'route':>5} {'records':>7} | "
        + " | ".join(f"{m:>22}" for m in ("dr end/max/yaw", "snap end/max/yaw", "scan end/max/yaw"))
    )
    offline = section(rows, "offline")
    for seed, t in dict.fromkeys((r["seed"], r["trajectory"]) for r in offline):
        group = {r["mode"]: r for r in offline if r["seed"] == seed and r["trajectory"] == t}
        cells = [
            f"{g['final_err_m']:>7} {g['max_err_m']:>7} {g['final_yaw_err_deg']:>6}"
            for g in group.values()
        ]
        print(
            f"{seed:>4} {t:>5} {group['dr']['records']:>7} | "
            + " | ".join(f"{c:>22}" for c in cells)
        )

    print("\nplace graph and goal")
    print(
        f"{'maze':>4} {'nodes':>6} {'edges':>7} {'comps':>5} | {'route':>5} {'record':>6} "
        f"{'view':>4} {'inl':>4} {'to target m':>11} | {'reach':>5} {'path':>5} {'actions':>7}"
    )
    for r in section(rows, "graph"):
        print(
            f"{r['seed']:>4} {r['nodes']:>6} {r['edges']:>7} {r['components']:>5} | "
            f"{r['goal_route']:>5} {r['goal_record']:>6} "
            f"{r['goal_view']:>4} {r['goal_inliers']:>4} "
            f"{r['goal_to_target_m']:>11} | {str(r['reachable']):>5} {r['path_nodes']:>5} "
            f"{r['path_actions']:>7}"
        )

    print("\nonline tracking against world.pose(): position error m / yaw error deg")
    print(
        f"{'maze':>4} {'variant':>8} {'ticks':>5} | {'med m':>6} {'p90 m':>6} {'max m':>6} | "
        f"{'med deg':>7} {'p90 deg':>7} {'max deg':>7} | {'fixes':>5}"
    )
    for r in section(rows, "tracking"):
        print(
            f"{r['seed']:>4} {r['variant']:>8} {r['ticks']:>5} | {r['median_err_m']:>6} "
            f"{r['p90_err_m']:>6} {r['max_err_m']:>6} | {r['median_err_deg']:>7} "
            f"{r['p90_err_deg']:>7} {r['max_err_deg']:>7} | {r['fixes']:>5}"
        )
    print("\nturn rate, degrees a tick: online (running median), offline (calibrated), true")
    for r in section(rows, "tracking"):
        if r["variant"] == "fixes":
            print(
                f"  maze {r['seed']}: online {r['online_turn_deg']} ({r['rate_samples']} samples), "
                f"offline {r['offline_turn_deg']}, true {r['true_turn_deg']}"
            )
    print("\nerror over time (m / deg)")
    timeline = section(rows, "timeline")
    for seed, variant in dict.fromkeys((r["seed"], r["variant"]) for r in timeline):
        cells = [
            f"{r['tick']}: {r['err_m']}/{r['err_deg']}"
            for r in timeline
            if r["seed"] == seed and r["variant"] == variant
        ]
        print(f"  maze {seed} {variant:>8}: " + "  ".join(cells))
    print("\ntime, seconds")
    for r in section(rows, "time"):
        print(
            "  maze "
            + str(r["seed"])
            + ": "
            + ", ".join(f"{k} {v}" for k, v in r.items() if k not in ("section", "seed"))
        )


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--ticks", type=int, default=2000, help="ticks of the tracked run")
    p.add_argument("--motion-seed", type=int, default=100, help="added to the maze seed")
    p.add_argument(
        "--online-turn",
        action="store_true",
        help="dead-reckon turns with the online rate once it has MIN_RATE_SAMPLES",
    )
    args = p.parse_args()
    if args.capture_every != 1:
        raise SystemExit("eval_tracking works on capture_every 1 data")
    floor = stored_floor()
    rows = []
    for seed in ints(args.mazes):
        rows += evaluate(seed, args, floor)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(Path(args.out_dir) / "eval_tracking.csv", fields, rows)
    summarise(rows)


if __name__ == "__main__":
    main()
