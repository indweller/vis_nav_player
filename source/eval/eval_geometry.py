"""
A map built from exploration data with calibrated motion and scan matching, the goal's pose in
it, and localisation on it, offline: no agent.

    uv run source/eval/eval_geometry.py
    uv run source/eval/eval_geometry.py --mazes 7 --localise 50

The data is what a challenge hands out: Simulator.record() output, three routes of a frame a
tick (data/eval/<seed>-r3-c1; --capture-every). A record's action is the one that led to its
frame. The floor mask is eval_cues.stored_floor: data/floor_mask.npz, fitted on mazes 1-6 and
scored here on 7-9, through the simulator's JPEG.

1. Data and ground truth. Simulator.record(student_files=False) adds poses.json to each route:
   the true [x, y, yaw] after each record's action (same frames and data_info.json as the
   student files). record_data re-records into data/eval/ if they are missing.
   The map frame's origin is (0, 0, 0) at route 0's first frame. For record() data, whether
   each route's first frame is the spawn view, and route 0's first frame, is printed (pixel
   difference); a route that starts elsewhere than route 0's first frame takes its start yaw
   from the manhattan candidate nearest 0, at the origin's position.

2. Motion: geometry.calibrate_motion on the exploration data, printed next to the simulator's
   true per-tick motion. Nothing about the motion is hard-coded.

3. Mapping: the routes in order into one OccupancyMap, from wall points within MAP_MAX_RANGE,
   three ways. "dr": dead reckoning with the calibrated motion every frame. "snap": every
   SCAN_EVERY-th frame with manhattan confidence >= MANHATTAN_CONFIDENCE snaps the yaw to its
   manhattan candidate nearest the dead-reckoned one. "scan": as snap, then (from the
   MATCH_AFTER-th scan) scan_match against the map so far, its yaw held at the manhattan
   candidate when there is one; a match scoring >= SCAN_ACCEPT and MIN_GAIN above the predicted
   pose's score sets the pose. A snap that turns the yaw more than MAX_SNAP_TURN_DEG from the
   dead reckoning since the last accepted match is rejected, for the candidate nearest that
   dead reckoning, unless scan_match confirms it. Only every SCAN_EVERY-th frame adds a scan.
   Per route, the final and largest position error (and the final yaw error) of each way;
   per map, wall precision (occupied cells within TOLS of a true wall) and recall (true wall
   cells within TOLS of an occupied cell).

4. Goal: the exploration frame (every --subsample-th) that best matches the first target view:
   VLAD's top GOAL_CANDIDATES, then the most SIFT (Servo) inliers. Its pose error against
   world.target, using its mapped pose in each way's map.

5. Localisation: --localise renders per maze at random clear poses and yaws, through the
   simulator's JPEG; the prior is the true pose moved PRIOR_M in a random direction and turned
   PRIOR_DEG either way. scan_match against the true walls (rasterised into an OccupancyMap)
   and against the scan-built map: success (< SUCCESS_M and < SUCCESS_DEG).

Each stage is timed. Maps are saved as results/eval/eval_geometry_<seed>.png in the map frame:
the truth, then the dr, snap and scan maps; true walls light grey under the map's black cells,
the true path green, the estimated one blue.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as sim
from common import CACHE_ROOT, fmt, ints, parser, textures, wrap, write_csv
from eval_cues import CLEARANCE, Walls, floor_samples, stored_floor
from geometry import (
    LOG_ODDS_MAX,
    MANHATTAN_CONFIDENCE,
    MAP_RES,
    MOTION_ACTIONS,
    SEARCH,
    Camera,
    FloorMask,
    OccupancyMap,
    advance,
    calibrate_motion,
    clip_range,
    manhattan_yaw,
    points,
    pose_score,
    ranges,
    scan_match,
    snap,
)
from my_agent import Float32VLAD
from scipy.ndimage import distance_transform_edt
from servo import Servo
from vlad import Frame as Record

MAP_MAX_RANGE = 0.8
"""Wall points beyond this range, metres, are left out of maps and matches: a row of the wall's
foot is ~0.05 m of range at 0.8 m, ~0.12 m at 1.5 m."""
SCAN_EVERY = 5
"""Frames between the scans a map takes (and snaps and matches are made at)."""
MATCH_AFTER = 10
"""Scans mapped before scan matching starts."""
SCAN_ACCEPT = 0.9
"""scan_match's score at and above which its pose can be taken."""
MIN_GAIN = 0.15
"""... and by which it must beat the predicted pose's score: a match that barely helps is noise."""
MAX_SNAP_TURN_DEG = 45.0
TOLS = (0.05, 0.10)
PRIOR_M, PRIOR_DEG = 0.3, 20.0
"""Localisation priors: the true pose moved this far in a random direction and turned this far
either way."""
SEARCH_M = PRIOR_M + 0.05
"""scan_match's window for localisation, +- metres along x and y: the prior's error and a cell's
margin."""
SUCCESS_M, SUCCESS_DEG = 0.1, 3.0
SAME_START_TOLERANCE = 0.5
"""Two first frames are the same view within this mean |pixel difference|."""
FLOOR_SAMPLES = 80
GOAL_CANDIDATES = 5
MODES = ("dr", "snap", "scan")

Pose = tuple[float, float, float]


@dataclass
class Route:
    images: list[Path]
    motions: list[str | None]
    """Per frame, the movement that led to it from the previous frame (None: none, or the
    route's start)."""
    truth: np.ndarray
    """``(N, 3)`` world poses at the frames."""


@dataclass
class Track:
    poses: list[np.ndarray]
    """Per route, ``(N, 3)`` map-frame poses at the frames."""
    grid: OccupancyMap
    counts: dict[str, int] = field(default_factory=dict)


class Timer:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start


class Images:
    """A route's, or routes', images read on indexing."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> np.ndarray:
        return cv2.imread(str(self.paths[i]))


# -- the data


def routes_in(root: Path) -> list[tuple[Path, list[dict]]]:
    """Each route's directory and its ``data_info.json`` records."""
    directories = sorted(
        (d for d in root.iterdir() if d.is_dir() and d.name.startswith("traj_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    return [(d, json.loads((d / "data_info.json").read_text())) for d in directories]


def movement(record: dict) -> str | None:
    """The record's action when it is one movement; None for IDLE or several actions."""
    names = record["action"]
    return names[0] if len(names) == 1 and names[0] in MOTION_ACTIONS else None


def data_root(seed: int, capture_every: int) -> Path:
    return Path("data") / "eval" / f"{seed}-r3-c{capture_every}"


def record_data(root: Path, seed: int, capture_every: int) -> None:
    """Simulator.record into ``root`` unless it is there with its poses.json files (written
    by ``student_files=False``; the frames and data_info.json are the same bytes as without)."""
    if (root / "target.jpg").exists() and (root / "traj_0" / "poses.json").exists():
        return
    print(f"recording {root} ...", flush=True)
    world = sim.Simulator(textures(), seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=seed)
    summary = world.record(
        root, routes=3, seed=seed, capture_every=capture_every, student_files=False
    )
    print(f"  {summary['frames']} frames on {summary['routes']} routes")


def record_routes(seed: int, capture_every: int) -> list[Route]:
    """The record() data's routes, with the true poses of poses.json (``[x, y, yaw]`` after
    each record's action)."""
    root = data_root(seed, capture_every)
    record_data(root, seed, capture_every)
    routes = []
    for directory, recs in routes_in(root):
        truth = np.array(json.loads((directory / "poses.json").read_text()))
        if len(truth) != len(recs):
            raise RuntimeError(f"{directory}: {len(truth)} poses for {len(recs)} records")
        routes.append(
            Route([directory / r["image"] for r in recs], [movement(r) for r in recs], truth)
        )
    return routes


def start_check(world: sim.Simulator, first_frames: list[Path]) -> tuple[list[float], list[float]]:
    """Per route, the mean |pixel difference| of its first frame from the spawn view and from
    route 0's first frame (both through the JPEG)."""
    spawn = jpeg(world, world.initial_frame()).astype(int)
    firsts = [cv2.imread(str(path)).astype(int) for path in first_frames]
    return (
        [float(np.abs(first - spawn).mean()) for first in firsts],
        [float(np.abs(first - firsts[0]).mean()) for first in firsts],
    )


# -- per-frame estimates


def estimate(image: np.ndarray, camera: Camera, floor: FloorMask) -> dict:
    """The wall points within MAP_MAX_RANGE, and manhattan_yaw (None below the confidence)."""
    wall = points(ranges(image, camera, floor))
    yaw, confidence = manhattan_yaw(wall)
    return {
        "points": clip_range(wall, MAP_MAX_RANGE),
        "manhattan": yaw if confidence >= MANHATTAN_CONFIDENCE else None,
    }


def jpeg(world: sim.Simulator, image: np.ndarray) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(world.jpeg(image), np.uint8), cv2.IMREAD_COLOR)


# -- tracking


def track(
    routes: list[Route],
    estimates: list[dict[int, dict]],
    motion: dict[str, dict],
    starts: list[bool],
    capture_every: int,
    mode: str,
) -> Track:
    """The routes in order into one map, ``mode`` one of MODES. ``estimates`` are per route the
    scan frames' by index; ``starts`` say which routes start at the origin."""
    matching = mode == "scan"
    grid = OccupancyMap()
    out = Track([], grid, {"snapped": 0, "matched": 0, "confirmed": 0, "rejected": 0})
    pose: Pose = (0.0, 0.0, 0.0)
    anchor = 0.0  # the dead-reckoned yaw since the last accepted match
    scans = 0
    for t, route in enumerate(routes):
        poses = []
        for i in range(len(route.images)):
            e = estimates[t].get(i)
            if i == 0:
                yaw = 0.0
                if t and not starts[t] and e["manhattan"] is not None:
                    yaw = snap(0.0, e["manhattan"])
                pose, anchor = (0.0, 0.0, yaw), yaw
            else:
                step = motion.get(route.motions[i])
                pose = advance(pose, step, capture_every)
                anchor = advance((0.0, 0.0, anchor), step, capture_every)[2]
            if e is not None:
                local, manhattan = e["points"], e["manhattan"]
                field_ = grid.likelihood_field() if matching and scans >= MATCH_AFTER else None
                if i and manhattan is not None and mode != "dr":
                    yaw = snap(pose[2], manhattan)
                    if matching and abs(math.degrees(wrap(yaw - anchor))) > MAX_SNAP_TURN_DEG:
                        yaw = confirm(local, field_, pose, yaw, snap(anchor, manhattan), out)
                    pose = (pose[0], pose[1], yaw)
                    out.counts["snapped"] += 1
                if field_ is not None:
                    matched, score = scan_match(local, field_, pose, manhattan_deg=manhattan)
                    if score >= SCAN_ACCEPT and score - pose_score(field_, pose, local) >= MIN_GAIN:
                        pose, anchor = matched, matched[2]
                        out.counts["matched"] += 1
                grid.add_scan(pose, local)
                scans += 1
            poses.append(pose)
        out.poses.append(np.array(poses))
    return out


def confirm(local, field_, pose: Pose, snapped: float, along: float, out: Track) -> float:
    """``snapped`` if scan_match confirms it over ``along`` (the candidate nearest the dead
    reckoning), else ``along``; without a map yet, ``along``."""
    if field_ is not None:
        scores = [
            scan_match(local, field_, (pose[0], pose[1], yaw), search=(0.25, 0.25, 0.0))[1]
            for yaw in (snapped, along)
        ]
        if scores[0] >= SCAN_ACCEPT and scores[0] >= scores[1]:
            out.counts["confirmed"] += 1
            return snapped
    out.counts["rejected"] += 1
    return along


# -- the ground truth


class Frame:
    """The map frame against the world: the map pose ``origin`` is the world pose ``truth``."""

    def __init__(self, origin: Pose, truth: Pose) -> None:
        self.p0 = np.array(truth[:2])
        self.q0 = np.array(origin[:2])
        self.phi = origin[2] - truth[2]
        c, s = math.cos(self.phi), math.sin(self.phi)
        self.rotation = np.array([[c, -s], [s, c]])

    def to_map(self, world: np.ndarray) -> np.ndarray:
        """``(n, 2)`` world points, or ``(n, 3)`` poses, in the map frame."""
        xy = (world[:, :2] - self.p0) @ self.rotation.T + self.q0
        if world.shape[1] == 2:
            return xy
        return np.column_stack([xy, world[:, 2] + self.phi])

    def to_world(self, xy: np.ndarray) -> np.ndarray:
        return (xy - self.q0) @ self.rotation + self.p0

    def pose_to_world(self, pose: Pose) -> Pose:
        x, y = self.to_world(np.array([pose[:2]]))[0]
        return float(x), float(y), pose[2] - self.phi


def wall_cells(world: sim.Simulator, walls: Walls) -> np.ndarray:
    """World centres of the MAP_RES cells (aligned on the world origin) a true wall passes
    through."""
    lo, hi = -0.5, (max(world.occupancy.shape) - 1) * world.cell_size + 0.5
    axis = np.arange(lo, hi, MAP_RES) + MAP_RES / 2
    x, y = np.meshgrid(axis, axis)
    hit = np.zeros(x.shape, bool)
    for x0, x1, y0, y1 in walls.rects:
        gap_x = np.maximum(np.maximum(x0 - x, x - x1), 0.0)
        gap_y = np.maximum(np.maximum(y0 - y, y - y1), 0.0)
        hit |= (gap_x <= MAP_RES / 2) & (gap_y <= MAP_RES / 2)
    return np.stack([x[hit], y[hit]], axis=1)


def wall_distance(walls: Walls, xy: np.ndarray) -> np.ndarray:
    """Metres from each world point to the nearest true wall."""
    distance = np.full(len(xy), np.inf)
    for x0, x1, y0, y1 in walls.rects:
        gap_x = np.maximum(np.maximum(x0 - xy[:, 0], xy[:, 0] - x1), 0.0)
        gap_y = np.maximum(np.maximum(y0 - xy[:, 1], xy[:, 1] - y1), 0.0)
        np.minimum(distance, np.hypot(gap_x, gap_y), out=distance)
    return distance


def map_scores(grid: OccupancyMap, frame: Frame, walls: Walls, truth_xy: np.ndarray) -> dict:
    """Precision and recall of the occupied cells at each of TOLS."""
    occupied = grid.occupied()
    precision_distance = wall_distance(walls, frame.to_world(grid.centres()[occupied]))
    to_map = distance_transform_edt(~occupied, sampling=grid.res)
    row, col, inside = grid.cells(frame.to_map(truth_xy))
    recall_distance = np.full(len(truth_xy), np.inf)
    recall_distance[inside] = to_map[row[inside], col[inside]]
    out = {"map_cells": int(occupied.sum())}
    for tol in TOLS:
        p = float(np.mean(precision_distance <= tol)) if occupied.any() else 0.0
        out[f"precision_{tol:.2f}"] = f"{p:.3f}"
        out[f"recall_{tol:.2f}"] = f"{np.mean(recall_distance <= tol):.3f}"
    return out


# -- the images


def panel(
    grid: OccupancyMap,
    truth_map: np.ndarray,
    box: tuple[slice, slice],
    occupied: np.ndarray | None,
    paths: list[tuple[np.ndarray, tuple[int, int, int]]],
    label: str,
) -> np.ndarray:
    """A BGR panel of the map frame's ``box``, +y up."""
    image = np.full((*grid.log_odds.shape, 3), 255, np.uint8)
    image[truth_map] = (210, 210, 210)
    if occupied is not None:
        image[occupied] = (0, 0, 0)
    scale = 2
    image = cv2.resize(image[box], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    corner = grid.origin + MAP_RES * np.array([box[1].start, box[0].start])
    for path, colour in paths:
        pixels = np.round((path[:, :2] - corner) / MAP_RES * scale).astype(np.int32)
        cv2.polylines(image, [pixels.reshape(-1, 1, 2)], False, colour, 1, cv2.LINE_AA)
    image = image[::-1].copy()
    cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
    return image


def save_maps(
    path: Path,
    seed: int,
    tracks: dict[str, Track],
    frames: dict[str, Frame],
    truth_xy: np.ndarray,
    truth_poses: np.ndarray,
) -> None:
    grid = tracks["scan"].grid
    frame = frames["scan"]
    truth_map = np.zeros(grid.log_odds.shape, bool)
    row, col, inside = grid.cells(frame.to_map(truth_xy))
    truth_map[row[inside], col[inside]] = True
    seen = truth_map.copy()
    for t in tracks.values():
        seen |= t.grid.occupied()
    rows, cols = np.nonzero(seen)
    margin = 10
    box = (
        slice(max(rows.min() - margin, 0), rows.max() + margin),
        slice(max(cols.min() - margin, 0), cols.max() + margin),
    )
    true_path = (frame.to_map(truth_poses), (60, 160, 0))
    panels = [panel(grid, truth_map, box, None, [true_path], f"maze {seed}: truth")]
    for mode, t in tracks.items():
        estimated = (np.concatenate(t.poses), (200, 80, 0))
        panels.append(panel(t.grid, truth_map, box, t.grid.occupied(), [estimated], mode))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.hstack(panels))


# -- the goal


def find_goal(
    seed: int,
    root: Path,
    routes: list[Route],
    camera: dict,
    capture_every: int,
    subsample: int,
    views: int = 1,
) -> tuple[int, int, int, int, int]:
    """``(route, record, view, vlad rank, inliers)`` of the exploration frame best matching one
    of the first ``views`` target views (target.jpg holds four side by side): per view VLAD's
    top GOAL_CANDIDATES over every ``subsample``-th record, then the most Servo inliers over
    the views' candidates."""
    target = cv2.imread(str(root / "target.jpg"))
    width = target.shape[1] // 4
    frames, where = [], []
    for t, route in enumerate(routes):
        for i in range(0, len(route.images), subsample):
            frames.append(Record(route.images[i], route.motions[i] or "", t))
            where.append((t, i))
    extractor = Float32VLAD(128, CACHE_ROOT / f"geometry-{seed}-c{capture_every}-s{subsample}")
    database = extractor.fit(frames)
    servo = Servo(camera["fx"], cx=camera["cx"])
    best = None
    for k in range(views):
        view = target[:, k * width : (k + 1) * width]
        top = np.argsort(-(database @ extractor.extract(view)))[:GOAL_CANDIDATES]
        for rank, j in enumerate(top):
            inliers = servo.yaw_offset(view, frames[j].path)[1]
            if best is None or inliers > best[3]:
                best = (*where[int(j)], k, rank, inliers)
    return best[0], best[1], best[2], best[3], best[4]


def complete_motion(motion: dict[str, dict]) -> dict[str, dict]:
    """``motion`` with BACKWARD, if the data has none, as FORWARD reversed (an assumption)."""
    if "BACKWARD" in motion or "FORWARD" not in motion:
        return motion
    f = motion["FORWARD"]
    back = {**f, "dx": -f["dx"], "dy": -f["dy"], "dyaw": -f["dyaw"], "n": (0, 0, 0)}
    return {**motion, "BACKWARD": back}


# -- localisation


def localise(
    seed: int,
    world: sim.Simulator,
    walls: Walls,
    camera: Camera,
    floor: FloorMask,
    maps: dict[str, tuple[OccupancyMap, Frame | None]],
    count: int,
    rng: np.random.Generator,
) -> list[dict]:
    """``count`` rows per map: scan_match from a noisy prior at a random clear pose."""
    fields = {name: grid.likelihood_field() for name, (grid, _) in maps.items()}
    free = np.argwhere(~world.occupancy)
    rows = []
    for _ in range(count):
        while True:
            r, c = free[rng.integers(len(free))]
            x = c * world.cell_size + rng.uniform(-0.1, 0.1)
            y = r * world.cell_size + rng.uniform(-0.1, 0.1)
            if walls.clearance(x, y) >= CLEARANCE:
                break
        yaw = rng.uniform(-math.pi, math.pi)
        e = estimate(jpeg(world, world.render(x, y, yaw)), camera, floor)
        local, manhattan = e["points"], e["manhattan"]
        direction = rng.uniform(0, 2 * math.pi)
        noise = np.array(
            [
                PRIOR_M * math.cos(direction),
                PRIOR_M * math.sin(direction),
                math.radians(PRIOR_DEG) * rng.choice([-1, 1]),
            ]
        )
        for name, (_, frame) in maps.items():
            true = np.array([(x, y, yaw)])
            true = (true if frame is None else frame.to_map(true))[0]
            prior = tuple(true + noise)
            pose, score = scan_match(
                local,
                fields[name],
                prior,
                search=(SEARCH_M, SEARCH_M, SEARCH[2]),
                manhattan_deg=manhattan,
            )
            error_m = math.dist(pose[:2], true[:2])
            error_deg = abs(math.degrees(wrap(pose[2] - true[2])))
            rows.append(
                {
                    "section": "localise",
                    "seed": seed,
                    "map": name,
                    "score": f"{score:.3f}",
                    "err_m": f"{error_m:.3f}",
                    "err_deg": f"{error_deg:.2f}",
                    "success": error_m < SUCCESS_M and error_deg < SUCCESS_DEG,
                }
            )
    return rows


def true_walls_map(truth_xy: np.ndarray) -> OccupancyMap:
    """An OccupancyMap in the world frame, occupied where a true wall passes."""
    grid = OccupancyMap()
    row, col, inside = grid.cells(truth_xy)
    grid.log_odds[row[inside], col[inside]] = LOG_ODDS_MAX
    return grid


# -- the evaluation


def evaluate(
    seed: int,
    args,
    world: sim.Simulator,
    walls: Walls,
    camera: Camera,
    floor: FloorMask,
    rng: np.random.Generator,
    timer: Timer,
) -> list[dict]:
    rows = []
    k = args.capture_every

    samples = floor_samples(world, walls, camera, rng, FLOOR_SAMPLES, jpeg=True)
    precision, recall = floor.score(samples)
    rows.append(
        {
            "section": "floor",
            "seed": seed,
            "precision": f"{precision:.3f}",
            "recall": f"{recall:.3f}",
        }
    )

    # 1. the data, its ground truth and its starts
    with timer.stage("data"):
        routes = record_routes(seed, k)
        root = data_root(seed, k)
    note = f"{root}: {sum(len(r.images) for r in routes)} records with poses.json"
    print(f"maze {seed}: {note}")
    rows.append({"section": "data", "seed": seed, "source": note})
    from_spawn, from_first = start_check(
        world, [d / recs[0]["image"] for d, recs in routes_in(data_root(seed, k))]
    )
    same = [d <= SAME_START_TOLERANCE for d in from_first]
    for t, (a, b) in enumerate(zip(from_spawn, from_first, strict=True)):
        print(
            f"  record() route {t} first frame: mean |diff| {a:.2f} from the spawn view "
            f"({'identical' if a <= SAME_START_TOLERANCE else 'differs'}), {b:.2f} from "
            f"route 0's ({'identical' if same[t] else 'differs'})"
        )
        rows.append(
            {
                "section": "start",
                "seed": seed,
                "trajectory": t,
                "spawn_diff": f"{a:.3f}",
                "route0_diff": f"{b:.3f}",
            }
        )
    starts = same

    # 2. motion, from the data
    with timer.stage("calibration"):
        paths = [path for route in routes for path in route.images]
        names = [name for route in routes for name in [None, *route.motions[1:]]]
        motion = calibrate_motion(Images(paths), names, camera, floor, ticks=k)
        motion = complete_motion(motion)
    for name, m in motion.items():
        rows.append(
            {
                "section": "motion",
                "seed": seed,
                "action": name,
                "dx_m": f"{m['dx']:.4f}",
                "dy_m": f"{m['dy']:.4f}",
                "dyaw_deg": f"{math.degrees(m['dyaw']):.3f}",
                "std_dx": f"{m['std'][0]:.4f}",
                "std_dy": f"{m['std'][1]:.4f}",
                "std_dyaw_deg": f"{math.degrees(m['std'][2]):.3f}",
                "n": "/".join(str(v) for v in m["n"]),
            }
        )

    # 3. mapping
    with timer.stage("mapping"):
        estimates = [
            {
                i: estimate(cv2.imread(str(route.images[i])), camera, floor)
                for i in range(0, len(route.images), SCAN_EVERY)
            }
            for route in routes
        ]
        tracks = {mode: track(routes, estimates, motion, starts, k, mode) for mode in MODES}
        truth_xy = wall_cells(world, walls)
        frames = {
            mode: Frame(tuple(t.poses[0][0]), tuple(routes[0].truth[0]))
            for mode, t in tracks.items()
        }
        for mode, t in tracks.items():
            for r, (route, poses) in enumerate(zip(routes, t.poses, strict=True)):
                truth = frames[mode].to_map(route.truth)
                error = np.hypot(*(poses[:, :2] - truth[:, :2]).T)
                rows.append(
                    {
                        "section": "route",
                        "seed": seed,
                        "trajectory": r,
                        "mode": mode,
                        "records": len(poses),
                        "final_err_m": f"{error[-1]:.3f}",
                        "max_err_m": f"{error.max():.3f}",
                        "final_yaw_err_deg": (
                            f"{abs(math.degrees(wrap(poses[-1, 2] - truth[-1, 2]))):.2f}"
                        ),
                    }
                )
            rows.append(
                {
                    "section": "map",
                    "seed": seed,
                    "mode": mode,
                    **t.counts,
                    **map_scores(t.grid, frames[mode], walls, truth_xy),
                }
            )
        save_maps(
            Path(args.out_dir) / f"eval_geometry_{seed}.png",
            seed,
            tracks,
            frames,
            truth_xy,
            np.concatenate([route.truth for route in routes]),
        )

    # 4. the goal
    with timer.stage("goal"):
        t, i, _, rank, inliers = find_goal(seed, root, routes, world.camera, k, args.subsample)
        target = np.array(world.target)
        found = routes[t].truth[i]
        for mode, track_ in tracks.items():
            x, y, yaw = frames[mode].pose_to_world(tuple(track_.poses[t][i]))
            rows.append(
                {
                    "section": "goal",
                    "seed": seed,
                    "mode": mode,
                    "trajectory": t,
                    "record": i,
                    "vlad_rank": rank,
                    "inliers": inliers,
                    "frame_to_target_m": f"{math.dist(found[:2], target[:2]):.3f}",
                    "frame_to_target_deg": (f"{abs(math.degrees(wrap(found[2] - target[2]))):.2f}"),
                    "err_m": f"{math.dist((x, y), target[:2]):.3f}",
                    "err_deg": f"{abs(math.degrees(wrap(yaw - target[2]))):.2f}",
                }
            )

    # 5. localisation on the true walls and the scan-built map
    with timer.stage("localisation"):
        maps = {
            "true": (true_walls_map(truth_xy), None),
            "built": (tracks["scan"].grid, frames["scan"]),
        }
        rows += localise(seed, world, walls, camera, floor, maps, args.localise, rng)
    return rows


# -- the report


def section(rows: list[dict], name: str) -> list[dict]:
    return [r for r in rows if r["section"] == name]


def summarise(rows: list[dict], timer: Timer, per_maze: dict[int, dict[str, float]]) -> None:
    print("\nfloor mask (stored, fitted on mazes 1-6) on renders through the simulator's JPEG")
    for r in section(rows, "floor"):
        print(f"  maze {r['seed']}: precision {r['precision']}, recall {r['recall']}")
    print("\ndata")
    for r in section(rows, "data"):
        print(f"  maze {r['seed']}: {r['source']}")

    true_m = sim.LINEAR_SPEED * sim.TICK_SECONDS
    true_deg = math.degrees(sim.ANGULAR_SPEED * sim.TICK_SECONDS)
    print(
        f"\ncalibrated motion per tick (median, 1.4826 MAD); the simulator's: {true_m:.3f} m, "
        f"{true_deg:.4f} deg"
    )
    print(
        f"{'maze':>4} {'action':>8} | {'dx m':>8} {'dy m':>8} {'dyaw deg':>9} | "
        f"{'sd dx':>6} {'sd dy':>6} {'sd yaw':>6} | {'n x/y/yaw':>12}"
    )
    for r in section(rows, "motion"):
        print(
            f"{r['seed']:>4} {r['action']:>8} | {r['dx_m']:>8} {r['dy_m']:>8} {r['dyaw_deg']:>9} | "
            f"{r['std_dx']:>6} {r['std_dy']:>6} {r['std_dyaw_deg']:>6} | {r['n']:>12}"
        )

    print("\nroutes: position error, metres, and final yaw error, degrees, in the map frame")
    print(f"{'maze':>4} {'route':>5} | " + " | ".join(f"{m + ' end/max/yaw':>22}" for m in MODES))
    routes = section(rows, "route")
    for seed, trajectory in dict.fromkeys((r["seed"], r["trajectory"]) for r in routes):
        cells = []
        for mode in MODES:
            r = next(
                r
                for r in routes
                if r["seed"] == seed and r["trajectory"] == trajectory and r["mode"] == mode
            )
            cells.append(f"{r['final_err_m']:>7} {r['max_err_m']:>7} {r['final_yaw_err_deg']:>6}")
        print(f"{seed:>4} {trajectory:>5} | " + " | ".join(f"{c:>22}" for c in cells))

    print("\nmaps: occupied cells within 0.05 / 0.10 m of a true wall, true walls within that")
    print(
        f"{'maze':>4} {'mode':>5} {'cells':>6} | {'P.05':>5} {'R.05':>5} {'P.10':>5} {'R.10':>5}"
        f" | {'snapped':>7} {'matched':>7} {'confirm':>7} {'reject':>6}"
    )
    for r in section(rows, "map"):
        print(
            f"{r['seed']:>4} {r['mode']:>5} {r['map_cells']:>6} | {r['precision_0.05']:>5} "
            f"{r['recall_0.05']:>5} {r['precision_0.10']:>5} {r['recall_0.10']:>5} | "
            f"{r['snapped']:>7} {r['matched']:>7} {r['confirmed']:>7} {r['rejected']:>6}"
        )

    print(
        "\ngoal: the exploration frame matching target view 0 (VLAD top "
        f"{GOAL_CANDIDATES}, then most SIFT inliers), its mapped pose against world.target"
    )
    print(
        f"{'maze':>4} {'mode':>5} {'route':>5} {'record':>6} {'rank':>4} {'inl':>4} | "
        f"{'frame m':>7} {'frame deg':>9} | {'map err m':>9} {'deg':>6}"
    )
    for r in section(rows, "goal"):
        print(
            f"{r['seed']:>4} {r['mode']:>5} {r['trajectory']:>5} {r['record']:>6} "
            f"{r['vlad_rank']:>4} {r['inliers']:>4} | {r['frame_to_target_m']:>7} "
            f"{r['frame_to_target_deg']:>9} | {r['err_m']:>9} {r['err_deg']:>6}"
        )

    local = section(rows, "localise")
    print(
        f"\nlocalisation: scan_match from the true pose moved {PRIOR_M} m and turned "
        f"{PRIOR_DEG} deg; success < {SUCCESS_M} m and < {SUCCESS_DEG} deg"
    )
    for name in dict.fromkeys(r["map"] for r in local):
        group = [r for r in local if r["map"] == name]
        ok = [float(r["score"]) for r in group if r["success"]]
        bad = [float(r["score"]) for r in group if not r["success"]]
        print(
            f"{name:>6} {len(group):>4} renders: success "
            f"{np.mean([r['success'] for r in group]):.2f}; median error "
            f"{np.median([float(r['err_m']) for r in group]):.3f} m, "
            f"{np.median([float(r['err_deg']) for r in group]):.2f} deg; median score "
            f"{fmt(np.median(ok) if ok else math.nan)} on successes, "
            f"{fmt(np.median(bad) if bad else math.nan)} on failures"
        )

    stages = list(timer.seconds)
    print("\ntime, seconds")
    print(f"{'maze':>6} | " + " ".join(f"{s:>12}" for s in stages) + f" | {'total':>8}")
    for seed, seconds in per_maze.items():
        print(
            f"{seed:>6} | "
            + " ".join(f"{seconds.get(s, 0.0):>12.1f}" for s in stages)
            + f" | {sum(seconds.values()):>8.1f}"
        )
    print(
        f"{'total':>6} | "
        + " ".join(f"{timer.seconds[s]:>12.1f}" for s in stages)
        + f" | {sum(timer.seconds.values()):>8.1f}"
    )


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--localise", type=int, default=300, help="localisation renders per maze")
    p.add_argument("--seed", type=int, default=0, help="sampling seed")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    floor = stored_floor()
    rows, timer, per_maze = [], Timer(), {}
    for seed in ints(args.mazes):
        world = sim.Simulator(textures(), seed)
        before = dict(timer.seconds)
        rows.extend(
            evaluate(seed, args, world, Walls(world), Camera(world.camera), floor, rng, timer)
        )
        per_maze[seed] = {s: v - before.get(s, 0.0) for s, v in timer.seconds.items()}
        print(f"maze {seed}: done", flush=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(Path(args.out_dir) / "eval_geometry.csv", fields, rows)
    summarise(rows, timer, per_maze)


if __name__ == "__main__":
    main()
