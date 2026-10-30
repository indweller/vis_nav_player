"""
Steering cues under position offset, offline: no agent. Per maze, random record pairs (r, r + k)
from data/gt; the robot's view is rendered at r's recorded pose moved each of OFFSETS metres along
or across its heading, with up to --yaw-noise degrees of yaw noise, and compared with r + k's
recorded image.

    uv run source/eval/eval_cues.py
    uv run source/eval/eval_cues.py --mazes 7 --pairs 20

Cue A (whole-image yaw): grey and |Sobel-x| column profiles (per-column means over the middle 60%
of rows), resampled to bearing; the bearing shift within +-60 degrees maximising their summed
normalised cross-correlation is the yaw to r + k's heading. The SIFT Servo runs on the same pairs
(it can return None, which counts as a miss).

Cue B (free space, robot view only): a floor mask by colour, fitted on renders where the floor is
known from the wall geometry (stored_floor: mazes 1-6, through the simulator's JPEG, kept in
data/floor_mask.npz). Per column the lowest non-floor row gives the distance to the wall
over a flat floor, the free distance per bearing. The corridor heading is the centre of the
max-free sector nearest a prior bearing (cue A's yaw; the "oracle" columns use the true bearing
to r + k). The lateral offset is half the difference between the left and right wall distances
across the camera's axis, each side's distance the SIDE_PERCENTILE of its wall points' (beyond
SIDE_DEG either side): the nearest wall, not the far side of an opening. Columns whose wall foot
is below the frame (a wall nearer than the bottom row sees) give no wall point.

Ground truth comes from the poses: the yaw to r + k's heading, the bearing to its position, and
the lateral offset from the centreline of the nearest free cell across the grid axis nearest the
robot's heading (common.cell_of; at a side opening the nearest cell can be the side corridor's).
Bearings and yaws are degrees, positive to the right (the Servo's sign); lateral offsets are
metres, positive right of the centreline.

Walls are thin slabs, WALL_HALF either side of the '#' cells' centres, joining neighbouring '#'
cells, and the camera is CAMERA_HEIGHT above a flat floor with no pitch (the horizon is at cy):
both measured from renders, the height by stepping toward a wall and differencing the distances
of the floor's edge. The camera dict's height_offset (0.15) is not the camera's height. An offset
pose must keep CLEARANCE from every wall and see r's pose; otherwise the other sign, then the other
axis, is tried, and the sample is dropped when none fits. One-cell corridors are 0.4 m wide, so
most offsets of 0.2 m and more end up longitudinal: the CSV's axis column says which.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as sim
from common import cell_of, fmt, gt_dir, ints, parser, textures, write_csv
from geometry import FAR, FLOOR_MASK_PATH, Camera, FloorMask, free_space, wrap
from servo import Servo

WALL_HALF = 0.0125
"""Half a wall's thickness, metres."""
CLEARANCE = 0.05
"""An offset pose keeps this far from every wall, metres."""
KS = (1, 2, 4)
OFFSETS = (0.0, 0.1, 0.2, 0.3, 0.4)
PROFILE_ROWS = 0.6
"""Cue A averages each column over this middle fraction of the rows."""
MAX_SHIFT_DEG = 60.0
BEARING_STEP_DEG = 0.5
FREE_FRAC = 0.6
"""Cue B's sectors: runs of columns with at least this fraction of the largest free distance."""
SIDE_DEG = 25.0
"""Cue B's side walls: points at least this far either side of the camera's axis."""
SIDE_PERCENTILE = 25.0
"""Cue B's side wall distance: this percentile of the side's wall points' distances."""
FIT_FRAMES = 40
"""Renders per maze the floor mask is fitted on."""
FLOOR_MAZES = range(1, 7)
"""The mazes the stored floor mask is fitted on, none of them evaluated."""
MIN_TARGET_DISTANCE = 0.05
"""Bearing errors are left out when r + k is nearer than this, metres: its bearing is noise."""
FIELDS = [
    "seed",
    "k",
    "trajectory",
    "record",
    "offset_m",
    "axis",
    "yaw_noise_deg",
    "true_yaw_deg",
    "a_yaw_deg",
    "a_err_deg",
    "sift_yaw_deg",
    "sift_inliers",
    "sift_err_deg",
    "true_bearing_deg",
    "target_distance_m",
    "b_heading_deg",
    "b_bearing_err_deg",
    "b_oracle_heading_deg",
    "b_oracle_bearing_err_deg",
    "true_lateral_m",
    "b_lateral_m",
    "b_lateral_err_m",
]


# -- the wall geometry


class Walls:
    """The maze's walls as axis-aligned rectangles ``(x0, x1, y0, y1)``: a square at every
    '#' cell's centre and a slab to each '#' neighbour's."""

    def __init__(self, world: sim.Simulator, half: float = WALL_HALF) -> None:
        occupied = world.occupancy
        size = world.cell_size
        rects = []
        for row, col in np.argwhere(occupied):
            x, y = col * size, row * size
            rects.append((x - half, x + half, y - half, y + half))
            if col + 1 < occupied.shape[1] and occupied[row, col + 1]:
                rects.append((x, x + size, y - half, y + half))
            if row + 1 < occupied.shape[0] and occupied[row + 1, col]:
                rects.append((x - half, x + half, y, y + size))
        self.rects = np.array(rects)

    def cast(self, x: float, y: float, angles: np.ndarray, far: float = FAR) -> np.ndarray:
        """Distance from ``(x, y)`` to the first wall along each world angle, at most ``far``."""
        dx, dy = np.cos(angles)[:, None], np.sin(angles)[:, None]
        x0, x1, y0, y1 = (self.rects[:, i][None, :] for i in range(4))
        with np.errstate(divide="ignore", invalid="ignore"):
            tx0, tx1 = (x0 - x) / dx, (x1 - x) / dx
            ty0, ty1 = (y0 - y) / dy, (y1 - y) / dy
        # A ray parallel to a slab's sides hits it at any t if it starts between them.
        inside_x = (x0 <= x) & (x <= x1)
        inside_y = (y0 <= y) & (y <= y1)
        near_x = np.where(dx == 0, np.where(inside_x, -np.inf, np.inf), np.minimum(tx0, tx1))
        far_x = np.where(dx == 0, np.where(inside_x, np.inf, -np.inf), np.maximum(tx0, tx1))
        near_y = np.where(dy == 0, np.where(inside_y, -np.inf, np.inf), np.minimum(ty0, ty1))
        far_y = np.where(dy == 0, np.where(inside_y, np.inf, -np.inf), np.maximum(ty0, ty1))
        enter = np.maximum(near_x, near_y)
        leave = np.minimum(far_x, far_y)
        hit = (enter <= leave) & (leave >= 0)
        t = np.where(hit, np.maximum(enter, 0.0), np.inf)
        return np.minimum(t.min(axis=1), far)

    def clearance(self, x: float, y: float) -> float:
        """Distance from ``(x, y)`` to the nearest wall."""
        x0, x1, y0, y1 = self.rects.T
        dx = np.maximum(np.maximum(x0 - x, x - x1), 0.0)
        dy = np.maximum(np.maximum(y0 - y, y - y1), 0.0)
        return float(np.hypot(dx, dy).min())

    def sees(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        """Whether the segment from ``a`` to ``b`` crosses no wall."""
        length = math.dist(a, b)
        if length == 0:
            return True
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        return float(self.cast(*a, np.array([angle]), far=length + 1)[0]) > length


# -- the camera (geometry.Camera)


def floor_truth(camera: Camera, walls: Walls, x: float, y: float, yaw: float) -> np.ndarray:
    """``(rows, width)`` -1 unknown, 1 floor, 0 wall: the pixels below the horizon a margin
    of a pixel either side of the walls' feet."""
    edge = camera.floor_rows(walls.cast(x, y, camera.angles(yaw)))
    rows = np.arange(camera.rows)[:, None] + 0.5
    truth = np.full((camera.rows, camera.width), -1, np.int8)
    below = rows > camera.cy + 1
    truth[below & (rows > edge[None, :] + 1)] = 1
    truth[below & (rows < edge[None, :] - 1)] = 0
    return truth


# -- cue B: the free space (geometry.FloorMask, geometry.free_space)


def corridor_heading(camera: Camera, ranges: np.ndarray, prior_deg: float) -> float | None:
    """The centre bearing of the run of columns with at least FREE_FRAC of the largest free
    range whose centre is nearest ``prior_deg``."""
    smooth = np.convolve(ranges, np.ones(5) / 5, mode="same")
    open_ = smooth >= FREE_FRAC * smooth.max()
    centres = []
    start = None
    for u, flag in enumerate([*open_, False]):
        if flag and start is None:
            start = u
        elif not flag and start is not None:
            if u - start >= 3:
                centres.append(float(np.mean(camera.bearing[start:u])))
            start = None
    if not centres:
        return None
    return min(centres, key=lambda c: abs(c - prior_deg))


def lateral_offset(camera: Camera, ranges: np.ndarray, seen: np.ndarray) -> float | None:
    """Half of (left wall distance - right wall distance) across the camera's axis: positive
    when the robot is right of the centreline. A side's distance is the SIDE_PERCENTILE of its
    wall points' distances across the axis, over the ``seen`` columns; None without wall
    points on both sides."""
    across = ranges * np.sin(np.radians(camera.bearing))
    wall = seen & (ranges < FAR * 0.95)
    left = -across[wall & (camera.bearing <= -SIDE_DEG)]
    right = across[wall & (camera.bearing >= SIDE_DEG)]
    if not len(left) or not len(right):
        return None
    return (
        float(np.percentile(left, SIDE_PERCENTILE)) - float(np.percentile(right, SIDE_PERCENTILE))
    ) / 2


# -- cue A: whole-image yaw


def profiles(camera: Camera, image: np.ndarray) -> np.ndarray:
    """``(2, n)`` grey and |Sobel-x| column means over the middle rows, resampled to a uniform
    bearing grid and standardised."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    margin = int(round(camera.rows * (1 - PROFILE_ROWS) / 2))
    middle = slice(margin, camera.rows - margin)
    edges = np.abs(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3))
    grid = bearing_grid(camera)
    out = []
    for channel in (grey, edges):
        column = channel[middle].mean(axis=0)
        resampled = np.interp(grid, camera.bearing, column)
        out.append((resampled - resampled.mean()) / (resampled.std() + 1e-6))
    return np.array(out)


def bearing_grid(camera: Camera) -> np.ndarray:
    lo, hi = camera.bearing[0], camera.bearing[-1]
    return np.arange(lo, hi + 1e-9, BEARING_STEP_DEG)


def whole_image_yaw(camera: Camera, current: np.ndarray, record: np.ndarray) -> float:
    """The bearing shift d maximising the summed NCC of current(theta) and record(theta - d):
    the record's view lies d degrees to the right."""
    cur, rec = profiles(camera, current), profiles(camera, record)
    n = cur.shape[1]
    steps = int(round(MAX_SHIFT_DEG / BEARING_STEP_DEG))
    best, best_score = 0, -np.inf
    for s in range(-steps, steps + 1):
        a = cur[:, max(s, 0) : n + min(s, 0)]
        b = rec[:, max(-s, 0) : n - max(s, 0)]
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        score = np.sum(
            (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        )
        if score > best_score:
            best, best_score = s, score
    return best * BEARING_STEP_DEG


# -- the samples


def records(seed: int, capture_every: int) -> list[tuple[Path, list[dict]]]:
    """Each trajectory's directory and its ``data_info.json`` records."""
    root = gt_dir(seed, capture_every)
    trajectories = sorted(
        (d for d in root.iterdir() if d.is_dir() and d.name.startswith("traj_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    return [(d, json.loads((d / "data_info.json").read_text())) for d in trajectories]


def offset_pose(
    walls: Walls,
    x: float,
    y: float,
    yaw: float,
    offset: float,
    rng: np.random.Generator,
) -> tuple[float, float, str] | None:
    """``(x, y, axis)`` ``offset`` metres along ("longitudinal") or across ("lateral") the
    heading, the axis and sign drawn at random, the other sign and axis tried in turn; None
    when none keeps CLEARANCE and sees ``(x, y)``."""
    axes = ["longitudinal", "lateral"]
    if rng.random() < 0.5:
        axes.reverse()
    signs = [1, -1] if rng.random() < 0.5 else [-1, 1]
    for axis in axes:
        for sign in signs:
            if axis == "longitudinal":
                dx, dy = math.cos(yaw), math.sin(yaw)
            else:  # to the right
                dx, dy = math.sin(yaw), -math.cos(yaw)
            px, py = x + sign * offset * dx, y + sign * offset * dy
            if walls.clearance(px, py) >= CLEARANCE and walls.sees((x, y), (px, py)):
                return px, py, axis if offset else "none"
    return None


def true_lateral(world: sim.Simulator, free: np.ndarray, x: float, y: float, yaw: float) -> float:
    """The displacement to the right of the centreline through the nearest free cell's centre
    along the grid axis nearest ``yaw``."""
    axis = round(yaw / (math.pi / 2)) * (math.pi / 2)
    row, col = cell_of(world, free, x, y)
    dx, dy = x - col * world.cell_size, y - row * world.cell_size
    return dx * math.sin(axis) - dy * math.cos(axis)


def right_deg(angle: float) -> float:
    """A counter-clockwise angle in radians as degrees positive to the right."""
    return -math.degrees(wrap(angle))


def floor_samples(
    world: sim.Simulator,
    walls: Walls,
    camera: Camera,
    rng: np.random.Generator,
    count: int,
    *,
    jpeg: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``count`` renders at random clear poses and their floor truth; with ``jpeg``, through
    the simulator's JPEG, as records are."""
    free = np.argwhere(~world.occupancy)
    samples = []
    for _ in range(count):
        while True:
            row, col = free[rng.integers(len(free))]
            x = col * world.cell_size + rng.uniform(-0.1, 0.1)
            y = row * world.cell_size + rng.uniform(-0.1, 0.1)
            if walls.clearance(x, y) >= CLEARANCE:
                break
        yaw = rng.uniform(-math.pi, math.pi)
        image = world.render(x, y, yaw)
        if jpeg:
            image = cv2.imdecode(np.frombuffer(world.jpeg(image), np.uint8), cv2.IMREAD_COLOR)
        samples.append((image, floor_truth(camera, walls, x, y, yaw)))
    return samples


def fit_floor(
    worlds: dict[int, tuple[sim.Simulator, Walls, Camera]],
    rng: np.random.Generator,
    *,
    jpeg: bool = False,
) -> FloorMask:
    """The floor mask, fitted on FIT_FRAMES renders per maze at random clear poses and scored
    on as many more."""
    fitted, held_out = [], []
    for world, walls, camera in worlds.values():
        samples = floor_samples(world, walls, camera, rng, 2 * FIT_FRAMES, jpeg=jpeg)
        fitted += samples[::2]
        held_out += samples[1::2]
    mask = FloorMask()
    mask.fit(fitted)
    precision, recall = mask.score(held_out)
    print(f"floor mask: held-out precision {precision:.3f}, recall {recall:.3f}")
    return mask


def stored_floor(path: Path = FLOOR_MASK_PATH) -> FloorMask:
    """FloorMask.load(path); first fitted on FLOOR_MAZES through the simulator's JPEG and
    saved there if it is missing."""
    if not path.exists():
        print(f"fitting the floor mask on mazes {list(FLOOR_MAZES)} into {path} ...")
        worlds = {}
        for seed in FLOOR_MAZES:
            world = sim.Simulator(textures(), seed)
            worlds[seed] = world, Walls(world), Camera(world.camera)
        fit_floor(worlds, np.random.default_rng(0), jpeg=True).save(path)
    return FloorMask.load(path)


def evaluate(
    seed: int,
    args,
    world: sim.Simulator,
    walls: Walls,
    camera: Camera,
    floor: FloorMask,
    servo: Servo,
    rng: np.random.Generator,
) -> tuple[list[dict], int]:
    """One row per sample, and the samples dropped for want of a clear offset pose."""
    trajectories = [t for t in records(seed, args.capture_every) if len(t[1]) > max(KS)]
    free = ~world.occupancy
    rows, dropped = [], 0
    for k in KS:
        for _ in range(args.pairs):
            t = int(rng.integers(len(trajectories)))
            directory, recs = trajectories[t]
            r = int(rng.integers(len(recs) - k))
            here, there = recs[r], recs[r + k]
            record_image = cv2.imread(str(directory / there["image"]))
            record_features = servo.image_features(directory / there["image"])
            for offset in OFFSETS:
                pose = offset_pose(walls, here["x"], here["y"], here["yaw"], offset, rng)
                if pose is None:
                    dropped += 1
                    continue
                x, y, axis = pose
                noise = rng.uniform(-args.yaw_noise, args.yaw_noise)
                yaw = here["yaw"] + math.radians(noise)
                view = world.render(x, y, yaw)

                true_yaw = right_deg(there["yaw"] - yaw)
                distance = math.hypot(there["x"] - x, there["y"] - y)
                true_bearing = right_deg(math.atan2(there["y"] - y, there["x"] - x) - yaw)
                bearing_known = distance >= MIN_TARGET_DISTANCE

                a_yaw = whole_image_yaw(camera, view, record_image)
                sift_yaw, inliers = servo.offset(servo.features(view), record_features, camera.cx)

                ranges, seen = free_space(camera, floor(view))
                heading = corridor_heading(camera, ranges, a_yaw)
                oracle = corridor_heading(camera, ranges, true_bearing)
                lateral = lateral_offset(camera, ranges, seen)
                truth_lateral = true_lateral(world, free, x, y, yaw)

                def err(estimate: float | None, truth: float, known: bool = True) -> str:
                    if estimate is None or not known:
                        return ""
                    return f"{abs(math.degrees(wrap(math.radians(estimate - truth)))):.2f}"

                rows.append(
                    {
                        "seed": seed,
                        "k": k,
                        "trajectory": t,
                        "record": r,
                        "offset_m": offset,
                        "axis": axis,
                        "yaw_noise_deg": f"{noise:.2f}",
                        "true_yaw_deg": f"{true_yaw:.2f}",
                        "a_yaw_deg": f"{a_yaw:.2f}",
                        "a_err_deg": err(a_yaw, true_yaw),
                        "sift_yaw_deg": "" if sift_yaw is None else f"{sift_yaw:.2f}",
                        "sift_inliers": inliers,
                        "sift_err_deg": err(sift_yaw, true_yaw),
                        "true_bearing_deg": f"{true_bearing:.2f}" if bearing_known else "",
                        "target_distance_m": f"{distance:.3f}",
                        "b_heading_deg": "" if heading is None else f"{heading:.2f}",
                        "b_bearing_err_deg": err(heading, true_bearing, bearing_known),
                        "b_oracle_heading_deg": "" if oracle is None else f"{oracle:.2f}",
                        "b_oracle_bearing_err_deg": err(oracle, true_bearing, bearing_known),
                        "true_lateral_m": f"{truth_lateral:.3f}",
                        "b_lateral_m": "" if lateral is None else f"{lateral:.3f}",
                        "b_lateral_err_m": (
                            "" if lateral is None else f"{abs(lateral - truth_lateral):.3f}"
                        ),
                    }
                )
    return rows, dropped


# -- the report


def values(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows if r[key] != ""])


def summarise(rows: list[dict]) -> None:
    """One line per (k, offset)."""
    print(
        f"\n{'k':>2} {'offset':>6} {'n':>5} | {'A med':>6} {'A<10':>5} | {'SIFT ok':>7} "
        f"{'S med':>6} {'S<10':>5} | {'B bear':>6} {'B orc':>6} {'B lat m':>7} {'lat n':>5}"
    )
    for k in KS:
        for offset in OFFSETS:
            group = [r for r in rows if r["k"] == k and r["offset_m"] == offset]
            if not group:
                print(f"{k:>2} {offset:>6.1f} {0:>5}")
                continue
            a = values(group, "a_err_deg")
            s = values(group, "sift_err_deg")
            b = values(group, "b_bearing_err_deg")
            oracle = values(group, "b_oracle_bearing_err_deg")
            lateral = values(group, "b_lateral_err_m")

            def med(v: np.ndarray, digits: int = 1) -> str:
                return fmt(float(np.median(v)) if len(v) else math.nan, digits)

            print(
                f"{k:>2} {offset:>6.1f} {len(group):>5} | {med(a):>6} "
                f"{fmt(float(np.mean(a < 10)), 2):>5} | {fmt(len(s) / len(group), 2):>7} "
                f"{med(s):>6} {fmt(float((s < 10).sum()) / len(group), 2):>5} | "
                f"{med(b):>6} {med(oracle):>6} {med(lateral, 3):>7} {len(lateral):>5}"
            )
    print(
        "\nA, S: |yaw error| to r + k's heading, degrees (S<10 counts None as a miss); "
        "B bear / B orc: |corridor heading - bearing to r + k| with cue A's / the true bearing "
        "as the prior; B lat: |lateral offset error|, metres, over lat n samples."
    )


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--pairs", type=int, default=50, help="record pairs per maze and k")
    p.add_argument("--yaw-noise", type=float, default=10.0, help="uniform yaw noise, +- degrees")
    p.add_argument("--seed", type=int, default=0, help="sampling seed")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    worlds = {}
    for seed in ints(args.mazes):
        world = sim.Simulator(textures(), seed)
        worlds[seed] = world, Walls(world), Camera(world.camera)
    floor = stored_floor()

    rows = []
    for seed, (world, walls, camera) in worlds.items():
        servo = Servo(camera.fx, cx=camera.cx)
        maze, dropped = evaluate(seed, args, world, walls, camera, floor, servo, rng)
        print(f"maze {seed}: {len(maze)} samples, {dropped} dropped (no clear offset pose)")
        rows.extend(maze)
    write_csv(Path(args.out_dir) / "eval_cues.csv", FIELDS, rows)
    summarise(rows)


if __name__ == "__main__":
    main()
