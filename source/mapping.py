"""
A pose-free map from exploration data: frames, the action behind each frame, and the camera.
No true poses, no simulator.

1. Odometry. Per tick, geometry.visual_odometry between consecutive frames (SIFT matches on
   the walls, ranged by the floor-wall boundary, a RANSAC rigid fit). Where it fails, or gives
   more than MAX_STEP_M / MAX_STEP_DEG, the action's calibrate_motion mean stands in. The
   integrated yaw is snapped to manhattan_yaw when that is confident and within SNAP_DEG of it.
   Every route starts at the spawn, (0, 0, 0) for route 0; the others take route 0's position
   and the yaw of the SIFT match of their first frame to route 0's first START_FRAMES. The
   spawn's heading is taken to lie along a maze axis.
2. Loop closures. VLAD (MyAgent's Float32VLAD, codebook fitted on these frames) of every
   NODE_EVERY-th frame; for every QUERY_EVERY-th, the TOP_K nearest nodes, not within
   EXCLUDE_TICKS of it on its route; kept with MIN_SIFT_INLIERS epipolar SIFT inliers, and a
   relative pose from the wall keypoints' rigid fit.
3. Pose graph over the nodes (every NODE_EVERY-th frame): odometry edges composed between
   them, closure edges, and a prior on each route's start; scipy least_squares with a Huber
   loss. Closures more than PRE_GATE_M / PRE_GATE_DEG from the dead reckoning are left out;
   after a solve, those that disagree with the poses by MAX_CLOSURE_M / MAX_CLOSURE_DEG are
   dropped and the graph solved again, up to ROUNDS times.
4. An OccupancyMap of the nodes' wall points within GRID_MAX_RANGE at the poses.

``build`` returns the "odom" variant (the dead reckoning, no closures) and the "graph" one.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from geometry import (
    MANHATTAN_CONFIDENCE,
    Camera,
    FloorMask,
    OccupancyMap,
    WallView,
    calibrate_motion,
    clip_range,
    epipolar_inliers,
    manhattan_yaw,
    match_features,
    relative_pose,
    snap,
    visual_odometry,
    wall_view,
)
from my_agent import Float32VLAD
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from servo import Servo
from sklearn.cluster import KMeans
from tqdm import tqdm

NODE_EVERY = 5
"""Frames between pose-graph nodes (and map scans)."""
QUERY_EVERY = 10
TOP_K = 10
EXCLUDE_TICKS = 100
MIN_SIFT_INLIERS = 30
MAX_CLOSURE_M = 1.0
MAX_CLOSURE_DEG = 45.0
PRE_GATE_M, PRE_GATE_DEG = 1.0, 30.0
"""A closure further than this from the dead reckoning is left out of the first solve: the
yaw is locked to the maze's axes and the position drifts ~1 %, while a closure between
perpendicular or repeated walls of one texture fits well and can be 90 degrees out; a few
dozen of those, agreeing, bend a graph they are part of past what the check after it can see."""
ROUNDS = 3
"""Solves, each without the closures the last left more than MAX_CLOSURE_M / _DEG out."""
GRID_MAX_RANGE = 0.8
"""Wall points beyond this range, metres, are left out of the map: a row of the wall's foot is
~0.05 m of range at 0.8 m, ~0.12 m at 1.5 m."""
SNAP_DEG = 20.0
MAX_STEP_M, MAX_STEP_DEG = 0.1, 20.0
"""A visual odometry step further than this is a failure: a tick drives 0.03 m or turns 3.4
degrees, or, against a wall, turns the robot ten degrees without moving it, so the action's
mean cannot gate it."""
START_FRAMES = 60
"""Route 0's frames a later route's first frame is matched against, for its start yaw."""
START_MIN_INLIERS = 15
CODEBOOK_DESCRIPTORS = 100_000
HORIZON_SHIFT_PX = 1.0
"""Rows added to the camera's cy for the wall ranges. geometry.ranges reads the floor-wall
boundary as a pixel-corner coordinate against a pixel-centre cy, and the wall's foot is on
average half a row above the boundary: unshifted, ranges (and so visual odometry) read ~5%
short of a forward tick's 0.03 m on mazes 7-9 (0.99 shifted)."""
ODOM_SIGMA_M, ODOM_SIGMA_DEG = 0.01, 1.0
"""A node-to-node odometry edge's spread: the visual odometry drifts ~1 %. A looser spread lets
a few mutually consistent aliased closures (the same wall texture elsewhere) bend the graph."""
CLOSURE_SIGMA_M, CLOSURE_SIGMA_DEG = 0.05, 2.0
START_SIGMA_M, START_SIGMA_DEG = 0.001, 3.0
"""A route's start prior: the position is exact; the yaw is as sure as the SIFT match."""
HUBER_SCALE = 2.0
"""The Huber loss's knee, in spreads."""

Pose = tuple[float, float, float]


class Images:
    """A route's frames, read from ``paths`` on indexing."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> np.ndarray:
        return cv2.imread(str(self.paths[i]))


@dataclass
class Route:
    frames: Sequence[np.ndarray]
    """Indexable by frame."""
    actions: list[str | None]
    """Per frame, the single movement that led to it (None: none or several, or the start)."""


def load_routes(root: Path) -> list[Route]:
    """The routes of a Simulator.record() directory, from ``traj_<i>/data_info.json`` and the
    frames only."""
    directories = sorted(
        (d for d in Path(root).iterdir() if d.is_dir() and d.name.startswith("traj_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    routes = []
    for directory in directories:
        records = json.loads((directory / "data_info.json").read_text())
        moves = ("FORWARD", "BACKWARD", "LEFT", "RIGHT")
        actions = [r["action"][0] if len(r["action"]) == 1 else None for r in records]
        routes.append(
            Route(
                Images([directory / r["image"] for r in records]),
                [a if a in moves else None for a in actions],
            )
        )
    return routes


@dataclass
class Closure:
    a: tuple[int, int]
    """``(route, frame)`` of the earlier node."""
    b: tuple[int, int]
    pose: Pose
    """The pose of b in a's frame."""
    sift_inliers: int
    wall_inliers: int
    accepted: bool = False


@dataclass
class Variant:
    poses: list[np.ndarray]
    """Per route, ``(nodes, 3)`` map-frame poses at frames 0, NODE_EVERY, ..."""
    grid: OccupancyMap


@dataclass
class MapResult:
    variants: dict[str, Variant]
    closures: list[Closure]
    candidates: int
    """VLAD candidate pairs looked at."""
    seconds: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    dead_reckoned: list[np.ndarray] = field(default_factory=list)
    """Per route, the ``(frames, 3)`` odometry poses (every frame, before the graph)."""
    codebook: np.ndarray | None = None
    """The VLAD codebook's ``(clusters, 128)`` centres, when the graph variant was built."""

    def save(self, directory: Path, prefix: str) -> None:
        """Each variant's occupancy grid to ``<directory>/<prefix>_<variant>.npz``."""
        for name, variant in self.variants.items():
            variant.grid.save(Path(directory) / f"{prefix}_{name}.npz")


class Timer:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def compose(a: Pose, b: Pose) -> Pose:
    """The pose ``b`` (given in a's frame) in the frame ``a`` is in."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def between(a: Pose, b: Pose) -> Pose:
    """The pose of ``b`` in ``a``'s frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return (c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2]))


# -- motion


class Concatenated:
    def __init__(self, parts: Sequence[Sequence[np.ndarray]]) -> None:
        self.parts = parts
        self.offsets = np.cumsum([0, *(len(p) for p in parts)])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, i: int) -> np.ndarray:
        part = int(np.searchsorted(self.offsets, i, side="right")) - 1
        return self.parts[part][i - int(self.offsets[part])]


def calibrate(routes: list[Route], camera: Camera, floor: FloorMask) -> dict[str, dict]:
    """calibrate_motion over all the routes; BACKWARD, if there is none, as FORWARD reversed."""
    frames = Concatenated([r.frames for r in routes])
    actions = [a for r in routes for a in [None, *r.actions[1:]]]
    motion = calibrate_motion(frames, actions, camera, floor)
    if "FORWARD" in motion and "BACKWARD" not in motion:
        f = motion["FORWARD"]
        motion["BACKWARD"] = {**f, "dx": -f["dx"], "dy": -f["dy"], "dyaw": -f["dyaw"]}
    return motion


# -- odometry


@dataclass
class Node:
    """What a node frame keeps for the later stages."""

    view: WallView
    local: np.ndarray
    """Its wall points within GRID_MAX_RANGE."""


@dataclass
class Tracked:
    poses: np.ndarray
    """``(frames, 3)`` dead-reckoned poses."""
    nodes: list[Node]
    """Per node (frame 0, NODE_EVERY, ...)."""
    counts: dict[str, int]


def odometry(
    route: Route,
    start: Pose,
    camera: Camera,
    floor: FloorMask,
    motion: dict[str, dict],
    *,
    progress: str = "",
) -> Tracked:
    """Dead reckoning along ``route`` from ``start``: visual odometry per tick, the action's
    mean motion where it fails, and the yaw snapped to the maze's axes."""
    sift = cv2.SIFT_create()
    rng = np.random.default_rng(0)
    counts = {"visual": 0, "fallback": 0, "snapped": 0}
    poses = np.empty((len(route.frames), 3))
    nodes: list[Node] = []
    pose = start
    previous = None
    for i in tqdm(range(len(route.frames)), desc=progress, leave=False):
        view = wall_view(route.frames[i], camera, floor, sift)
        yaw_m, confidence = manhattan_yaw(view.points)
        if previous is not None:
            action = route.actions[i]
            expected = motion.get(action) if action else None
            step = visual_odometry(previous, view, rng)
            if step is not None and not plausible(step):
                step = None
            if step is None:
                counts["fallback"] += 1
                step = (expected["dx"], expected["dy"], expected["dyaw"]) if expected else (0, 0, 0)
            else:
                counts["visual"] += 1
            pose = compose(pose, step)
            if confidence >= MANHATTAN_CONFIDENCE:
                yaw = snap(pose[2], yaw_m)
                if abs(wrap(yaw - pose[2])) <= math.radians(SNAP_DEG):
                    pose = (pose[0], pose[1], yaw)
                    counts["snapped"] += 1
        poses[i] = pose
        if i % NODE_EVERY == 0:
            nodes.append(Node(view, clip_range(view.points, GRID_MAX_RANGE)))
        previous = view
    return Tracked(poses, nodes, counts)


def plausible(step: Pose) -> bool:
    """Whether a visual odometry ``step`` is within MAX_STEP_M and MAX_STEP_DEG."""
    return math.hypot(step[0], step[1]) <= MAX_STEP_M and abs(step[2]) <= math.radians(MAX_STEP_DEG)


def start_yaw(first: WallView, reference: list[tuple[WallView, float]], cx: float, fx: float):
    """The yaw of a route's first frame ``first`` from the SIFT match to the reference frames
    ``(view, yaw)``: the one with most inliers, its yaw plus the bearing offset; None below
    START_MIN_INLIERS."""
    servo = Servo(fx, cx=cx, min_matches=START_MIN_INLIERS)
    best = None
    for view, yaw in reference:
        offset, inliers = servo.offset((first.pixels, first.des), (view.pixels, view.des), cx)
        if offset is not None and (best is None or inliers > best[0]):
            best = inliers, yaw + math.radians(offset)
    return None if best is None else best[1]


# -- loop closures


def fit_codebook(nodes: list[Node], extractor: Float32VLAD) -> None:
    descriptors = np.vstack([n.view.des for n in nodes if n.view.des is not None])
    rng = np.random.default_rng(0)
    if len(descriptors) > CODEBOOK_DESCRIPTORS:
        descriptors = descriptors[rng.choice(len(descriptors), CODEBOOK_DESCRIPTORS, False)]
    extractor.codebook = KMeans(
        n_clusters=extractor.n_clusters, n_init=1, max_iter=100, random_state=42
    ).fit(descriptors)


def find_closures(tracked: list[Tracked], timer: Timer) -> tuple[list[Closure], int, Float32VLAD]:
    """Verified loop closures between the routes' nodes, the candidates looked at, and the
    VLAD extractor (codebook) fitted on the nodes."""
    index = [(r, k) for r, t in enumerate(tracked) for k in range(len(t.nodes))]
    nodes = [tracked[r].nodes[k] for r, k in index]
    with timer.stage("vlad"):
        extractor = Float32VLAD(cache_dir=Path("cache") / "mapping")
        fit_codebook(nodes, extractor)
        database = np.array([extractor._vlad(n.view.des) for n in nodes], np.float32)
    route = np.array([r for r, _ in index])
    frame = np.array([k * NODE_EVERY for _, k in index])
    queries = [q for q, (_, k) in enumerate(index) if (k * NODE_EVERY) % QUERY_EVERY == 0]
    rng = np.random.default_rng(0)
    closures: list[Closure] = []
    candidates = 0
    seen: set[tuple[int, int]] = set()
    with timer.stage("closures"):
        similarity = database[queries] @ database.T
        for row, q in enumerate(tqdm(queries, desc="closures", leave=False)):
            same_route = (route == route[q]) & (np.abs(frame - frame[q]) < EXCLUDE_TICKS)
            scores = np.where(same_route | ~database.any(axis=1), -np.inf, similarity[row])
            if not database[q].any():
                continue
            top = np.argsort(-scores)[:TOP_K]
            for c in top[np.isfinite(scores[top])]:
                a, b = (int(c), q) if (route[c], frame[c]) < (route[q], frame[q]) else (q, int(c))
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                candidates += 1
                closure = verify(nodes[a].view, nodes[b].view, index[a], index[b], rng)
                if closure is not None:
                    closures.append(closure)
    return closures, candidates, extractor


def verify(
    va: WallView, vb: WallView, ia: tuple[int, int], ib: tuple[int, int], rng
) -> Closure | None:
    """A Closure from node ``ia`` to ``ib`` if their views have MIN_SIFT_INLIERS epipolar
    inliers and the wall keypoints among them fit a rigid motion."""
    matches = match_features(va, vb)
    if len(matches) < MIN_SIFT_INLIERS:
        return None
    inliers = epipolar_inliers(va, vb, matches)
    if inliers.sum() < MIN_SIFT_INLIERS:
        return None
    fit = relative_pose(va, vb, matches[inliers], rng)
    if fit is None:
        return None
    return Closure(
        (ia[0], ia[1] * NODE_EVERY), (ib[0], ib[1] * NODE_EVERY), fit[0], int(inliers.sum()), fit[1]
    )


# -- the pose graph


def optimise(
    initial: np.ndarray,
    edges: np.ndarray,
    measured: np.ndarray,
    sigma: np.ndarray,
    anchors: np.ndarray,
    anchor_poses: np.ndarray,
) -> np.ndarray:
    """Poses ``(n, 3)`` minimising the Huber loss of the whitened residuals of the edges
    ``(i, j)`` against the ``measured`` pose of j in i's frame (``sigma``: ``(m, 2)``,
    metres and radians) and of the ``anchors`` (node indices) against ``anchor_poses``."""
    n = len(initial)
    i, j = edges.T
    anchor_sigma = np.array([START_SIGMA_M, START_SIGMA_M, math.radians(START_SIGMA_DEG)])

    def residuals(x: np.ndarray) -> np.ndarray:
        p = x.reshape(n, 3)
        c, s = np.cos(p[i, 2]), np.sin(p[i, 2])
        d = p[j, :2] - p[i, :2]
        local = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
        edge = np.column_stack(
            [
                (local - measured[:, :2]) / sigma[:, :1],
                wrap(p[j, 2] - p[i, 2] - measured[:, 2]) / sigma[:, 1],
            ]
        )
        start = (p[anchors] - anchor_poses) / anchor_sigma
        start[:, 2] = wrap(p[anchors, 2] - anchor_poses[:, 2]) / anchor_sigma[2]
        return np.concatenate([edge.ravel(), start.ravel()])

    m = len(edges)
    e_rows = (
        3 * np.arange(m)[:, None, None] + np.arange(3)[None, :, None] + np.zeros((1, 1, 6), int)
    )
    e_cols = np.concatenate([3 * i[:, None] + np.arange(3), 3 * j[:, None] + np.arange(3)], axis=1)
    e_cols = e_cols[:, None, :] + np.zeros((1, 3, 1), int)
    a_rows = 3 * m + 3 * np.arange(len(anchors))[:, None] + np.arange(3)
    a_cols = 3 * anchors[:, None] + np.arange(3)
    sparsity = coo_matrix(
        (
            np.ones(e_rows.size + a_rows.size),
            (
                np.concatenate([e_rows.ravel(), a_rows.ravel()]),
                np.concatenate([e_cols.ravel(), a_cols.ravel()]),
            ),
        ),
        shape=(3 * m + 3 * len(anchors), 3 * n),
    )
    result = least_squares(
        residuals,
        initial.ravel(),
        jac_sparsity=sparsity,
        loss="huber",
        f_scale=HUBER_SCALE,
        x_scale="jac",
        max_nfev=60,
    )
    return result.x.reshape(n, 3)


def solve(tracked: list[Tracked], starts: list[Pose], closures: list[Closure]) -> list[np.ndarray]:
    """Per route, the node poses optimised with ``closures``."""
    sizes = [len(t.nodes) for t in tracked]
    first = np.concatenate([[0], np.cumsum(sizes)])
    initial = np.concatenate([t.poses[::NODE_EVERY] for t in tracked])
    edges, measured, sigma = [], [], []
    odom_sigma = (ODOM_SIGMA_M, math.radians(ODOM_SIGMA_DEG))
    for r, t in enumerate(tracked):
        nodes = t.poses[::NODE_EVERY]
        for k in range(len(nodes) - 1):
            edges.append((first[r] + k, first[r] + k + 1))
            measured.append(between(tuple(nodes[k]), tuple(nodes[k + 1])))
            sigma.append(odom_sigma)
    for c in closures:
        edges.append((first[c.a[0]] + c.a[1] // NODE_EVERY, first[c.b[0]] + c.b[1] // NODE_EVERY))
        measured.append(c.pose)
        sigma.append((CLOSURE_SIGMA_M, math.radians(CLOSURE_SIGMA_DEG)))
    solved = optimise(
        initial,
        np.array(edges),
        np.array(measured),
        np.array(sigma),
        first[:-1],
        np.array(starts),
    )
    return [solved[first[r] : first[r + 1]] for r in range(len(tracked))]


def consistent(
    closure: Closure,
    poses: list[np.ndarray],
    max_m: float = MAX_CLOSURE_M,
    max_deg: float = MAX_CLOSURE_DEG,
) -> bool:
    """Whether ``closure`` agrees with the node ``poses`` within MAX_CLOSURE_M / _DEG."""
    pa = tuple(poses[closure.a[0]][closure.a[1] // NODE_EVERY])
    pb = tuple(poses[closure.b[0]][closure.b[1] // NODE_EVERY])
    x, y, yaw = between(pa, pb)
    return math.hypot(x - closure.pose[0], y - closure.pose[1]) <= max_m and abs(
        wrap(yaw - closure.pose[2])
    ) <= math.radians(max_deg)


def solve_rounds(
    tracked: list[Tracked], starts: list[Pose], closures: list[Closure]
) -> tuple[list[np.ndarray], list[Closure]]:
    """The node poses and the closures kept: those within PRE_GATE_M / _DEG of the dead
    reckoning, then solved and filtered by ``consistent`` for up to ROUNDS rounds."""
    dead_reckoned = [t.poses[::NODE_EVERY] for t in tracked]
    kept = [c for c in closures if consistent(c, dead_reckoned, PRE_GATE_M, PRE_GATE_DEG)]
    for _ in range(ROUNDS):
        poses = solve(tracked, starts, kept)
        agreeing = [c for c in kept if consistent(c, poses)]
        if len(agreeing) == len(kept):
            return poses, kept
        kept = agreeing
    return solve(tracked, starts, kept), kept


def make_grid(tracked: list[Tracked], poses: list[np.ndarray]) -> OccupancyMap:
    grid = OccupancyMap()
    for t, route_poses in zip(tracked, poses, strict=True):
        for node, pose in zip(t.nodes, route_poses, strict=True):
            grid.add_scan(tuple(pose), node.local)
    return grid


# -- all of it


def build(
    routes: list[Route],
    camera_dict: dict,
    floor: FloorMask | None = None,
    *,
    variants: Sequence[str] = ("odom", "graph"),
) -> MapResult:
    """The maps of ``variants`` ("odom", "graph") from the exploration ``routes``."""
    camera = Camera({**camera_dict, "cy": camera_dict["cy"] + HORIZON_SHIFT_PX})
    floor = floor or FloorMask.load()
    timer = Timer()

    with timer.stage("calibration"):
        motion = calibrate(routes, camera, floor)

    tracked: list[Tracked] = []
    starts: list[Pose] = []
    with timer.stage("odometry"):
        for r, route in enumerate(routes):
            yaw = 0.0
            if r:
                first = wall_view(route.frames[0], camera, floor)
                reference = [
                    (tracked[0].nodes[k].view, float(tracked[0].poses[k * NODE_EVERY, 2]))
                    for k in range(START_FRAMES // NODE_EVERY + 1)
                ]
                found = start_yaw(first, reference, camera.cx, camera.fx)
                yaw = 0.0 if found is None else found
                yaw_m, confidence = manhattan_yaw(first.points)
                if confidence >= MANHATTAN_CONFIDENCE:
                    snapped = snap(yaw, yaw_m)
                    if abs(wrap(snapped - yaw)) <= math.radians(SNAP_DEG):
                        yaw = snapped
            starts.append((0.0, 0.0, yaw))
            tracked.append(
                odometry(route, starts[-1], camera, floor, motion, progress=f"route {r}")
            )

    result = MapResult({}, [], 0)
    result.counts = {k: sum(t.counts[k] for t in tracked) for k in tracked[0].counts}
    result.dead_reckoned = [t.poses for t in tracked]
    if "odom" in variants:
        with timer.stage("grid"):
            poses = [t.poses[::NODE_EVERY] for t in tracked]
            result.variants["odom"] = Variant(poses, make_grid(tracked, poses))
    if "graph" in variants:
        closures, result.candidates, extractor = find_closures(tracked, timer)
        result.codebook = extractor.codebook.cluster_centers_.astype(np.float32)
        with timer.stage("optimise"):
            poses, kept = solve_rounds(tracked, starts, closures)
            for c in kept:
                c.accepted = True
        with timer.stage("grid"):
            result.variants["graph"] = Variant(poses, make_grid(tracked, poses))
        result.closures = closures
    result.seconds = timer.seconds
    return result
