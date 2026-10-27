"""
Geometry from one image, for a level camera CAMERA_HEIGHT above a flat floor (the horizon is at
cy). A floor mask by colour gives, per column, the row of the wall's foot and so the range to the
wall; the ranges give wall points in the robot's frame, and the walls' directions give the
robot's yaw modulo 90 degrees in a maze whose walls are axis-aligned. Wall points accumulate into
an OccupancyMap, and scan_match places a scan on the map's likelihood field.

Bearings are degrees, positive to the right; the robot's frame has x forward and y to the left,
and yaws are counter-clockwise, as the simulator's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

CAMERA_HEIGHT = 0.20
"""Metres from the floor to the camera, measured from renders. The camera dict's height_offset
(0.15) is not the camera's height."""
FAR = 5.0
"""The renderer's far plane: ranges are capped at it."""
LUT_BINS = 16
"""Floor mask: per-channel bins of the colour lookup table."""
FLOOR_MASK_PATH = Path("data") / "floor_mask.npz"
"""FloorMask.load's default: fitted on mazes 1-6 (eval_cues.stored_floor)."""
SEGMENT_SPAN = 16
"""manhattan_yaw: a segment joins columns this far apart. Neighbouring columns (1) see mostly the
range's row quantisation: a run of columns on one row is a line across the camera's axis, so
their histogram peaks at the yaw that makes it a wall (eval_geometry, mazes 7-9: a median error of
21-25 degrees on renders and 11 on record frames at span 1; 0.6-1.0 and 0.7 at 16)."""
SEGMENT_MAX_GAP = 0.02
"""manhattan_yaw: segments longer than this per column of span, metres, cross a depth step (a
corner or an opening) and are left out."""
SEGMENT_MAX_RANGE = 1.5
"""manhattan_yaw: points beyond this range, metres, are too coarse to give a direction."""
HIST_BIN_DEG = 1.0
PEAK_HALF_WIDTH_DEG = 3.0
"""manhattan_yaw's confidence: the weight within this of the peak over the total weight."""
MAP_RES = 0.05
MAP_HALF_EXTENT = 7.0
"""An OccupancyMap spans this many metres either side of its origin: a 31-cell maze (6.2 m) from
any start inside it."""
LOG_ODDS_HIT = math.log(0.7 / 0.3)
"""Added to a cell once per scan with a point in it."""
LOG_ODDS_MAX = 5.0
OCCUPIED_LOG_ODDS = 2.0
"""A cell is occupied from this log-odds: hits in three scans."""
FIELD_SIGMA = 0.05
"""The likelihood field's Gaussian, metres."""
SEARCH = (0.25, 0.25, 4.0)
"""scan_match's window: +- metres along x and y, +- degrees of yaw."""
PRIOR_WEIGHT = 0.3
"""scan_match prefers poses near the prior: it maximises the score less this times the squared
distance from the prior over the window's half-width, so a flat score along a corridor keeps
the prior's position rather than wandering (eval_geometry's mapping: 0.05 drifts metres)."""
FINE_STEPS = 5
"""scan_match's fine level divides a coarse step (MAP_RES, 1 degree) into this many."""
MANHATTAN_CONFIDENCE = 0.6
"""manhattan_yaw's confidence at and above which a frame's yaw is taken."""
CALIBRATION_RANGE = 0.8
"""calibrate_motion matches wall points within this range, metres: a row of the wall's foot is
~0.05 m of range at 0.8 m, ~0.12 m at 1.5 m."""
CALIBRATION_MIN_SCORE = 0.9
"""calibrate_motion keeps a frame pair whose match scores at least this."""
CONDITIONING_M = 0.05
CONDITIONING_DROP = 0.05
"""calibrate_motion takes a pair's x (or y) where the score drops this much a CONDITIONING_M
shift along it."""
MIN_MANHATTAN_PAIRS = 5
"""calibrate_motion takes an action's yaw from the manhattan_yaw differences given this many."""
MOTION_ACTIONS = ("FORWARD", "BACKWARD", "LEFT", "RIGHT")
KEYPOINT_MARGIN_PX = 1.0
"""wall_view: a keypoint is on the wall when its blob (radius size / 2) plus this many pixels
ends above the wall's foot in its column."""
MATCH_RATIO = 0.8
"""match_features: Lowe's ratio."""
FIT_RANGE = 1.2
"""rigid_ransac fits wall keypoints within this range, metres."""
FIT_INLIER_M = 0.06
"""... counting a pair an inlier within this distance of its fit."""
FIT_ITERATIONS = 200
FIT_MIN_SEPARATION = 0.05
"""A rigid_ransac hypothesis is drawn from two pairs this far apart (metres) in both views."""
FIT_MIN_INLIERS = 6
"""visual_odometry and relative_pose give nothing with fewer inliers."""


class Camera:
    """Bearings of the columns (degrees, positive right) and the flat-floor range of a row,
    from a camera dict (``width``, ``height``, ``fx``, ``fy``, ``cx``, ``cy``)."""

    def __init__(self, camera: dict, height: float = CAMERA_HEIGHT) -> None:
        self.width, self.rows = camera["width"], camera["height"]
        self.fx, self.fy = camera["fx"], camera["fy"]
        self.cx, self.cy = camera["cx"], camera["cy"]
        self.height = height
        self.bearing = np.degrees(np.arctan((np.arange(self.width) + 0.5 - self.cx) / self.fx))

    def angles(self, yaw: float) -> np.ndarray:
        """World angles of the column rays from a camera at ``yaw``: right is clockwise."""
        return yaw - np.radians(self.bearing)

    def floor_rows(self, ranges: np.ndarray) -> np.ndarray:
        """Per column, the first image row below the horizon that sees the floor nearer than
        the wall at ``ranges`` (rows below it see the floor)."""
        depth = ranges * np.cos(np.radians(self.bearing))
        return self.cy + self.fy * self.height / depth

    def ranges(self, edge_rows: np.ndarray) -> np.ndarray:
        """Per column, the range along the floor to a wall whose foot is at ``edge_rows``
        (image rows of the floor/wall boundary), capped at FAR."""
        below = np.maximum(edge_rows - self.cy, 1e-6)
        depth = self.fy * self.height / below
        return np.minimum(depth / np.cos(np.radians(self.bearing)), FAR)


class FloorMask:
    """A colour lookup table: a BGR bin is floor where floor pixels outnumber wall pixels in
    the images it was fitted on."""

    def __init__(self) -> None:
        self.table = np.zeros((LUT_BINS,) * 3, bool)

    @staticmethod
    def _bins(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        index = (image.astype(np.int32) * LUT_BINS) // 256
        return index[..., 0], index[..., 1], index[..., 2]

    def fit(self, samples: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """``samples``: images and per-pixel truth, 1 floor, 0 wall, -1 unknown."""
        floor = np.zeros((LUT_BINS,) * 3)
        wall = np.zeros((LUT_BINS,) * 3)
        for image, truth in samples:
            b, g, r = self._bins(image)
            np.add.at(floor, (b[truth == 1], g[truth == 1], r[truth == 1]), 1)
            np.add.at(wall, (b[truth == 0], g[truth == 0], r[truth == 0]), 1)
        self.table = floor > wall

    def __call__(self, image: np.ndarray) -> np.ndarray:
        mask = self.table[self._bins(image)].astype(np.uint8)
        # A stray wall pixel on the floor would end a column's scan early.
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)

    def save(self, path: Path = FLOOR_MASK_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, table=self.table)

    @classmethod
    def load(cls, path: Path = FLOOR_MASK_PATH) -> FloorMask:
        mask = cls()
        with np.load(path) as data:
            mask.table = data["table"]
        if mask.table.shape != (LUT_BINS,) * 3:
            raise ValueError(f"{path}: a {mask.table.shape} table, not {LUT_BINS} bins a channel")
        return mask

    def score(self, samples: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
        """Precision and recall of the floor on ``samples``."""
        hits = predicted = actual = 0
        for image, truth in samples:
            mask = self(image)
            known = truth >= 0
            hits += int((mask & (truth == 1)).sum())
            predicted += int((mask & known).sum())
            actual += int((truth == 1).sum())
        return hits / max(predicted, 1), hits / max(actual, 1)


def free_space(camera: Camera, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per column, the range to the wall: from the bottom row up, the first non-floor row
    below the horizon (none: FAR); and whether the wall's foot is in the frame (a bottom row
    that is not floor only bounds the range)."""
    top = int(math.floor(camera.cy)) + 1
    below = mask[top:, :][::-1]  # bottom row first
    wall = ~below
    first = np.where(wall.any(axis=0), wall.argmax(axis=0), below.shape[0])
    edge = camera.rows - first  # the boundary between the wall row and the floor row under it
    ranges = camera.ranges(edge.astype(float))
    ranges[first == below.shape[0]] = FAR
    return ranges, first > 0


def ranges(image: np.ndarray, camera: Camera, floor: FloorMask) -> tuple[np.ndarray, np.ndarray]:
    """Per column, ``(bearing_deg, range_m)``: the range is NaN where the wall's foot is below
    the frame or no wall is nearer than the far plane."""
    distance, seen = free_space(camera, floor(image))
    return camera.bearing.copy(), np.where(seen & (distance < FAR * 0.95), distance, np.nan)


def points(column_ranges: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """``(width, 2)`` wall points ``(x forward, y left)`` in the robot's frame, one per column
    in column order, NaN where the range is."""
    bearing, distance = column_ranges
    b = np.radians(bearing)
    return np.stack([distance * np.cos(b), -distance * np.sin(b)], axis=1)


def manhattan_yaw(
    wall_points: np.ndarray,
    *,
    span: int = SEGMENT_SPAN,
    max_gap: float = SEGMENT_MAX_GAP,
    max_range: float = SEGMENT_MAX_RANGE,
) -> tuple[float, float]:
    """The robot's yaw modulo 90 degrees, in ``[0, 90)``, and a confidence in ``[0, 1]``, from
    column-ordered robot-frame ``wall_points`` (as :func:`points`). A wall segment at robot-frame
    angle a lies along a world axis, so the yaw is -a modulo 90: the length-weighted histogram of
    -a over segments joining columns ``span`` apart, folded modulo 90 and smoothed, peaks there,
    refined by a parabola through the peak bin and its neighbours. The confidence is the weight
    within PEAK_HALF_WIDTH_DEG of the peak over the total; (NaN, 0) without segments."""
    p = np.where(np.hypot(*wall_points.T)[:, None] <= max_range, wall_points, np.nan)
    d = p[span:] - p[:-span]
    length = np.hypot(*d.T)
    ok = np.isfinite(length) & (length > 0) & (length <= max_gap * span)
    if not ok.any():
        return math.nan, 0.0
    folded = (-np.degrees(np.arctan2(d[ok, 1], d[ok, 0]))) % 90.0
    weight = length[ok]

    bins = int(round(90.0 / HIST_BIN_DEG))
    hist = np.bincount((folded / HIST_BIN_DEG).astype(int) % bins, weight, bins)
    smooth = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]), [1, 2, 3, 2, 1], "valid")
    peak = int(np.argmax(smooth))
    left, centre, right = smooth[peak - 1], smooth[peak], smooth[(peak + 1) % bins]
    curvature = left - 2 * centre + right
    shift = 0.5 * (left - right) / curvature if curvature < 0 else 0.0
    yaw = ((peak + 0.5 + shift) * HIST_BIN_DEG) % 90.0

    near = np.abs((folded - yaw + 45.0) % 90.0 - 45.0) <= PEAK_HALF_WIDTH_DEG
    return float(yaw), float(weight[near].sum() / weight.sum())


def snap(yaw: float, manhattan_deg: float) -> float:
    """The yaw, radians, of the form ``manhattan_deg`` + k 90 degrees nearest ``yaw``."""
    k = round((math.degrees(yaw) - manhattan_deg) / 90.0)
    return math.radians(manhattan_deg + 90.0 * k)


def transform(pose: tuple[float, float, float], local: np.ndarray) -> np.ndarray:
    """``(n, 2)`` robot-frame points placed at ``pose`` in the map frame."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack(
        [x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]], 1
    )


class LikelihoodField:
    """Per cell of an OccupancyMap, exp(-d^2 / 2 sigma^2) of the distance d from its centre to
    the nearest occupied cell's: 1 on a wall, whatever the walls' density. Bilinear between
    cell centres, 0 off the map."""

    def __init__(self, values: np.ndarray, origin: np.ndarray, res: float) -> None:
        self.values, self.origin, self.res = values, origin, res

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        u = (xy[..., 0] - self.origin[0]) / self.res - 0.5  # column, between centres
        v = (xy[..., 1] - self.origin[1]) / self.res - 0.5  # row
        rows, cols = self.values.shape
        inside = (u >= 0) & (u < cols - 1) & (v >= 0) & (v < rows - 1)
        u, v = np.where(inside, u, 0.0), np.where(inside, v, 0.0)
        c0, r0 = u.astype(int), v.astype(int)
        fu, fv = u - c0, v - r0
        f = self.values
        top = f[r0, c0] * (1 - fu) + f[r0, c0 + 1] * fu
        bottom = f[r0 + 1, c0] * (1 - fu) + f[r0 + 1, c0 + 1] * fu
        return np.where(inside, top * (1 - fv) + bottom * fv, 0.0)


class OccupancyMap:
    """Log-odds occupancy at ``res`` metres, ``half_extent`` either side of the origin; cell
    ``(row, col)`` covers ``y`` from ``origin[1] + row res`` and ``x`` from ``origin[0] + col
    res``. Scans add hits only: free space is never carved."""

    def __init__(self, half_extent: float = MAP_HALF_EXTENT, res: float = MAP_RES) -> None:
        self.res = res
        self.origin = np.array([-half_extent, -half_extent])
        size = int(round(2 * half_extent / res))
        self.log_odds = np.zeros((size, size), np.float32)

    def cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows and columns of the map points ``xy``, and whether each is on the map."""
        col, row = np.floor((xy - self.origin) / self.res).astype(int).T
        size = self.log_odds.shape[0]
        return row, col, (row >= 0) & (row < size) & (col >= 0) & (col < size)

    def centres(self) -> np.ndarray:
        """``(rows, cols, 2)`` map coordinates of the cells' centres."""
        size = self.log_odds.shape[0]
        axis = (np.arange(size) + 0.5) * self.res
        x, y = np.meshgrid(self.origin[0] + axis, self.origin[1] + axis)
        return np.stack([x, y], axis=-1)

    def add_scan(self, pose: tuple[float, float, float], wall_points: np.ndarray) -> None:
        """One hit per cell holding any of the robot-frame ``wall_points`` (NaN rows skipped)
        placed at ``pose``."""
        local = wall_points[np.isfinite(wall_points).all(axis=1)]
        row, col, inside = self.cells(transform(pose, local))
        hit = np.unique(np.stack([row[inside], col[inside]]), axis=1)
        cells = (hit[0], hit[1])
        self.log_odds[cells] = np.minimum(self.log_odds[cells] + LOG_ODDS_HIT, LOG_ODDS_MAX)

    def occupied(self) -> np.ndarray:
        return self.log_odds >= OCCUPIED_LOG_ODDS

    def likelihood_field(self, sigma: float = FIELD_SIGMA) -> LikelihoodField:
        occupied = self.occupied()
        if not occupied.any():
            return LikelihoodField(np.zeros(occupied.shape), self.origin, self.res)
        distance = distance_transform_edt(~occupied, sampling=self.res)
        return LikelihoodField(np.exp(-(distance**2) / (2 * sigma**2)), self.origin, self.res)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, log_odds=self.log_odds, origin=self.origin, res=self.res)

    @classmethod
    def load(cls, path: Path) -> OccupancyMap:
        grid = cls.__new__(cls)
        with np.load(path) as data:
            grid.log_odds = data["log_odds"]
            grid.origin = data["origin"]
            grid.res = float(data["res"])
        return grid


def scan_match(
    wall_points: np.ndarray,
    grid: OccupancyMap | LikelihoodField,
    prior_pose: tuple[float, float, float],
    search: tuple[float, float, float] = SEARCH,
    manhattan_deg: float | None = None,
    prior_weight: float = PRIOR_WEIGHT,
) -> tuple[tuple[float, float, float], float]:
    """The pose within ``search`` of ``prior_pose`` (+- x, y metres, +- yaw degrees) where the
    robot-frame ``wall_points`` score highest on ``grid``'s likelihood field, less
    ``prior_weight`` (distance from the prior / the window's half-width)^2; and the score: the
    field's mean over the points, in [0, 1]. Correlative and coarse-to-fine: every pose on a
    grid of the field's cell size and 1 degree, then FINE_STEPS times finer around the best.
    With ``manhattan_deg`` the yaw is fixed at its candidate nearest the prior's (as
    :func:`snap`) whatever the yaw window. (prior_pose, 0) without points."""
    field = grid if isinstance(grid, LikelihoodField) else grid.likelihood_field()
    local = wall_points[np.isfinite(wall_points).all(axis=1)]
    if not len(local):
        return prior_pose, 0.0
    x0, y0, yaw0 = prior_pose
    step = field.res
    if manhattan_deg is not None and math.isfinite(manhattan_deg):
        yaws, yaw_step = np.array([snap(yaw0, manhattan_deg)]), 0.0
    else:
        yaw_step = math.radians(1.0)
        half = math.floor(math.radians(search[2]) / yaw_step + 1e-9)
        yaws = yaw0 + yaw_step * np.arange(-half, half + 1)
    dx = step * np.arange(-math.floor(search[0] / step), math.floor(search[0] / step) + 1)
    dy = step * np.arange(-math.floor(search[1] / step), math.floor(search[1] / step) + 1)
    scale = max(search[0], search[1], step)

    def penalty(shifts: np.ndarray) -> np.ndarray:
        return prior_weight * ((shifts[:, 0] - x0) ** 2 + (shifts[:, 1] - y0) ** 2) / scale**2

    bx, by, byaw, _ = _best_pose(field, local, x0 + dx, y0 + dy, yaws, penalty)
    fine = np.arange(-FINE_STEPS + 1, FINE_STEPS) / FINE_STEPS
    fine_yaws = byaw + yaw_step * fine if yaw_step else np.array([byaw])
    bx, by, byaw, score = _best_pose(
        field, local, bx + step * fine, by + step * fine, fine_yaws, penalty
    )
    return (bx, by, byaw), score


def _best_pose(
    field: LikelihoodField,
    local: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    yaws: np.ndarray,
    penalty,
) -> tuple[float, float, float, float]:
    """The ``(x, y, yaw, score)`` over the grid of ``xs`` x ``ys`` x ``yaws`` maximising the
    score less ``penalty(shifts)``."""
    tx, ty = np.meshgrid(xs, ys)
    shifts = np.stack([tx.ravel(), ty.ravel()], axis=1)  # (m, 2)
    cost = penalty(shifts)
    best, best_objective = (float(xs[0]), float(ys[0]), float(yaws[0]), 0.0), -np.inf
    for yaw in yaws:
        rotated = transform((0.0, 0.0, float(yaw)), local)  # (n, 2)
        scores = field(shifts[:, None, :] + rotated[None, :, :]).mean(axis=1)
        objective = scores - cost
        i = int(np.argmax(objective))
        if objective[i] > best_objective:
            best_objective = objective[i]
            best = (float(shifts[i, 0]), float(shifts[i, 1]), float(yaw), float(scores[i]))
    return best


def pose_score(
    field: LikelihoodField, pose: tuple[float, float, float], local: np.ndarray
) -> float:
    """The likelihood field's mean over the robot-frame ``local`` placed at ``pose``."""
    local = local[np.isfinite(local).all(axis=1)]
    return float(field(transform(pose, local)).mean()) if len(local) else 0.0


def clip_range(wall_points: np.ndarray, max_range: float) -> np.ndarray:
    """``wall_points`` beyond ``max_range`` set to NaN."""
    far = ~(np.hypot(*wall_points.T) <= max_range)
    return np.where(far[:, None], np.nan, wall_points)


def advance(
    pose: tuple[float, float, float], motion: dict | None, ticks: int = 1
) -> tuple[float, float, float]:
    """``pose`` after ``ticks`` ticks of an action whose ``motion`` is a calibrate_motion entry
    (dx, dy in the robot's frame, dyaw in radians, per tick); None: it does not move."""
    if motion is None:
        return pose
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (
        x + ticks * (c * motion["dx"] - s * motion["dy"]),
        y + ticks * (s * motion["dx"] + c * motion["dy"]),
        yaw + ticks * motion["dyaw"],
    )


def calibrate_motion(
    frames: Sequence[np.ndarray],
    actions: Sequence[str | Sequence[str] | None],
    camera: Camera,
    floor: FloorMask,
    *,
    ticks: int = 1,
    max_pairs: int = 400,
    max_range: float = CALIBRATION_RANGE,
) -> dict[str, dict]:
    """What each action does, per tick, from consecutive ``frames`` of one route:
    ``actions[i]`` led to ``frames[i]`` (``ticks`` ticks held), and only single-action steps
    count (a list of one name, or a name). Per action in MOTION_ACTIONS with pairs:
    ``{"dx", "dy", "dyaw", "std", "n"}``, the motion of the robot's pose in its previous frame
    (metres forward and left, radians counter-clockwise), ``std`` those three's spreads, and
    ``n`` the pairs kept for each. Up to ``max_pairs`` per action, evenly spread over the route.

    Each pair is scan_match of the later frame's wall points to the earlier's (three hits
    each, an occupied map without a prior pulling the match to zero motion). The yaw is the
    difference of the two manhattan_yaws when both are confident (MANHATTAN_CONFIDENCE), and
    that holds the match's yaw fixed; the action's dyaw is those differences' when there are
    MIN_MANHATTAN_PAIRS, else the matches' own. A pair scoring below
    CALIBRATION_MIN_SCORE is dropped, and a pair gives an axis only if it pins it down
    (CONDITIONING_DROP). The mean is the median, the spread 1.4826 MAD, over the
    pairs within 3 spreads of the median."""
    steps: dict[str, list[int]] = {name: [] for name in MOTION_ACTIONS}
    for i, action in enumerate(actions):
        name = (
            action
            if isinstance(action, str) or action is None
            else (action[0] if len(action) == 1 else None)
        )
        if i and name in steps:
            steps[name].append(i)

    cache: dict[int, tuple[np.ndarray, float, float]] = {}

    def view(i: int) -> tuple[np.ndarray, float, float]:
        if i not in cache:
            wall = points(ranges(frames[i], camera, floor))
            yaw, confidence = manhattan_yaw(wall)
            cache[i] = clip_range(wall, max_range), yaw, confidence
        return cache[i]

    def robust(values: np.ndarray) -> tuple[float, float, int]:
        """Median and 1.4826 MAD of the values within 3 MADs of the median, and how many."""
        middle = float(np.median(values))
        spread = 1.4826 * float(np.median(np.abs(values - middle)))
        kept = values[np.abs(values - middle) <= 3 * max(spread, 1e-9)]
        if not len(kept):
            kept = values
        middle = float(np.median(kept))
        return middle, 1.4826 * float(np.median(np.abs(kept - middle))), len(kept)

    out: dict[str, dict] = {}
    for name, indices in steps.items():
        chosen = sorted(
            {indices[j] for j in np.linspace(0, len(indices) - 1, max_pairs).astype(int)}
            if indices
            else ()
        )
        # x, y, yaw per kept pair; NaN where the match does not pin it down.
        deltas, manhattan = [], []
        for i in chosen:
            before, yaw0, conf0 = view(i - 1)
            after, yaw1, conf1 = view(i)
            reference = OccupancyMap(half_extent=max_range + 2 * SEARCH[0])
            for _ in range(3):
                reference.add_scan((0.0, 0.0, 0.0), before)
            field = reference.likelihood_field()
            sure = conf0 >= MANHATTAN_CONFIDENCE and conf1 >= MANHATTAN_CONFIDENCE
            dyaw = math.radians((yaw1 - yaw0 + 45.0) % 90.0 - 45.0) if sure else 0.0
            window = (SEARCH[0], SEARCH[1], 0.0 if sure else SEARCH[2])
            pose, score = scan_match(after, field, (0.0, 0.0, dyaw), window, prior_weight=0.0)
            if score < CALIBRATION_MIN_SCORE:
                continue
            # A corridor's walls score alike at any shift along it: an axis counts only where
            # moving the pose CONDITIONING_M along it loses CONDITIONING_DROP of the score.
            delta = [pose[0], pose[1], pose[2]]
            for axis in (0, 1):
                moved = [
                    pose_score(
                        field,
                        (
                            pose[0] + sign * CONDITIONING_M * (axis == 0),
                            pose[1] + sign * CONDITIONING_M * (axis == 1),
                            pose[2],
                        ),
                        after,
                    )
                    for sign in (-1, 1)
                ]
                if score - max(moved) < CONDITIONING_DROP:
                    delta[axis] = math.nan
            deltas.append(delta)
            if sure:
                manhattan.append(dyaw)
        if not deltas:
            continue
        deltas = np.array(deltas) / ticks
        columns = [deltas[np.isfinite(deltas[:, k]), k] for k in range(3)]
        if len(manhattan) >= MIN_MANHATTAN_PAIRS:
            # A match's own yaw is quantised by the wall points' rows and reads low.
            columns[2] = np.array(manhattan) / ticks
        if not all(len(c) for c in columns):
            continue
        stats = [robust(c) for c in columns]
        out[name] = {
            "dx": stats[0][0],
            "dy": stats[1][0],
            "dyaw": stats[2][0],
            "std": tuple(st[1] for st in stats),
            "n": tuple(st[2] for st in stats),
        }
    return out


@dataclass
class WallView:
    """One image for matching: SIFT keypoints, their RootSIFT descriptors, and the wall points
    of its columns and of its keypoints on a wall."""

    pixels: np.ndarray
    """``(n, 2)`` keypoint positions (column, row)."""
    des: np.ndarray | None
    """``(n, 128)`` RootSIFT descriptors, float32; None without keypoints."""
    wall: np.ndarray
    """``(n, 2)`` robot-frame ``(x, y)`` of each keypoint's column wall point: NaN where the
    keypoint is not on a wall (on the floor, or the column's wall is out of range)."""
    points: np.ndarray
    """``(width, 2)`` wall points of the columns, as :func:`points`."""


def wall_view(
    image: np.ndarray, camera: Camera, floor: FloorMask, sift: cv2.SIFT | None = None
) -> WallView:
    """The :class:`WallView` of a BGR ``image``. A keypoint sees the wall at the column's
    floor-wall range if it lies above the wall's foot: the wall is vertical, so the ray through
    the keypoint hits it at that range whatever the keypoint's height."""
    sift = sift or cv2.SIFT_create()
    column_points = points(ranges(image, camera, floor))
    keypoints, des = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    if des is None or not len(keypoints):
        return WallView(np.empty((0, 2), np.float32), None, np.empty((0, 2)), column_points)
    des = np.sqrt(des / (np.sum(des, axis=1, keepdims=True) + 1e-7)).astype(np.float32)
    pixels = np.array([k.pt for k in keypoints], np.float32)
    radius = np.array([k.size for k in keypoints], np.float32) / 2
    column = np.clip(pixels[:, 0].astype(int), 0, camera.width - 1)
    wall = column_points[column]
    with np.errstate(invalid="ignore"):
        foot = camera.floor_rows(np.hypot(*column_points.T))[column]
        above = pixels[:, 1] + radius + KEYPOINT_MARGIN_PX < foot
    wall[~above] = np.nan
    return WallView(pixels, des, wall, column_points)


_MATCHER = cv2.BFMatcher(cv2.NORM_L2)


def match_features(a: WallView, b: WallView, ratio: float = MATCH_RATIO) -> np.ndarray:
    """``(m, 2)`` indices ``(i in a, j in b)`` of the descriptor pairs passing Lowe's ratio."""
    if a.des is None or b.des is None or len(a.des) < 2 or len(b.des) < 2:
        return np.empty((0, 2), int)
    good = [
        (m.queryIdx, m.trainIdx)
        for m, n in (p for p in _MATCHER.knnMatch(a.des, b.des, k=2) if len(p) == 2)
        if m.distance < ratio * n.distance
    ]
    return np.array(good, int).reshape(-1, 2)


def epipolar_inliers(
    a: WallView, b: WallView, matches: np.ndarray, threshold: float = 2.0
) -> np.ndarray:
    """Boolean mask over ``matches`` consistent with one fundamental matrix (RANSAC)."""
    if len(matches) < 8:
        return np.zeros(len(matches), bool)
    _, mask = cv2.findFundamentalMat(
        a.pixels[matches[:, 0]], b.pixels[matches[:, 1]], cv2.FM_RANSAC, threshold, 0.99
    )
    return np.zeros(len(matches), bool) if mask is None else mask.ravel().astype(bool)


def rigid_fit(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """The least-squares ``(x, y, yaw)`` placing the points ``b`` on ``a`` (``a ~ R b + t``):
    the pose of b's frame in a's."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    da, db = a - ca, b - cb
    yaw = math.atan2(
        float((db[:, 0] * da[:, 1] - db[:, 1] * da[:, 0]).sum()),
        float((db * da).sum()),
    )
    c, s = math.cos(yaw), math.sin(yaw)
    return (
        float(ca[0] - c * cb[0] + s * cb[1]),
        float(ca[1] - s * cb[0] - c * cb[1]),
        yaw,
    )


def rigid_ransac(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    *,
    threshold: float = FIT_INLIER_M,
    iterations: int = FIT_ITERATIONS,
) -> tuple[tuple[float, float, float], np.ndarray] | None:
    """RANSAC rigid fit of corresponding 2D points ``a[k] ~ pose(b[k])``: hypotheses from two
    pairs (rigid motions preserve their separation), the most inliers within ``threshold``,
    then refitted on the inliers twice. ``(pose, inlier mask)``; None with fewer than
    FIT_MIN_INLIERS."""
    n = len(a)
    if n < FIT_MIN_INLIERS:
        return None
    i, j = rng.integers(0, n, (2, iterations))
    da, db = a[j] - a[i], b[j] - b[i]
    la, lb = np.hypot(*da.T), np.hypot(*db.T)
    ok = (la > FIT_MIN_SEPARATION) & (lb > FIT_MIN_SEPARATION) & (np.abs(la - lb) < 2 * threshold)
    if not ok.any():
        return None
    i, da, db = i[ok], da[ok], db[ok]
    yaw = np.arctan2(da[:, 1], da[:, 0]) - np.arctan2(db[:, 1], db[:, 0])
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    t = a[i] - np.concatenate([c * b[i, :1] - s * b[i, 1:], s * b[i, :1] + c * b[i, 1:]], axis=1)
    placed = np.stack(
        [
            t[:, :1] + c * b[None, :, 0] - s * b[None, :, 1],
            t[:, 1:] + s * b[None, :, 0] + c * b[None, :, 1],
        ],
        axis=-1,
    )  # (hypotheses, n, 2)
    inliers = np.hypot(*(placed - a[None]).transpose(2, 0, 1)) <= threshold
    mask = inliers[int(np.argmax(inliers.sum(axis=1)))]
    pose = (0.0, 0.0, 0.0)
    for _ in range(3):
        if mask.sum() < FIT_MIN_INLIERS:
            return None
        pose = rigid_fit(a[mask], b[mask])
        mask = np.hypot(*(transform(pose, b) - a).T) <= threshold
    if mask.sum() < FIT_MIN_INLIERS:
        return None
    return rigid_fit(a[mask], b[mask]), mask


def relative_pose(
    a: WallView,
    b: WallView,
    matches: np.ndarray,
    rng: np.random.Generator,
    *,
    max_range: float = FIT_RANGE,
) -> tuple[tuple[float, float, float], int] | None:
    """The pose of ``b`` in ``a``'s frame and its inlier count, from the wall keypoints among
    ``matches`` within ``max_range`` in both views (:func:`rigid_ransac`); None without a fit."""
    pa, pb = a.wall[matches[:, 0]], b.wall[matches[:, 1]]
    with np.errstate(invalid="ignore"):
        usable = (np.hypot(*pa.T) <= max_range) & (np.hypot(*pb.T) <= max_range)
    if usable.sum() < FIT_MIN_INLIERS:
        return None
    fit = rigid_ransac(pa[usable], pb[usable], rng)
    return None if fit is None else (fit[0], int(fit[1].sum()))


def visual_odometry(
    prev: WallView, cur: WallView, rng: np.random.Generator | None = None
) -> tuple[float, float, float] | None:
    """The motion ``(dx, dy, dyaw)`` of the robot's pose in its previous frame (metres forward
    and left, radians counter-clockwise) from ``prev`` to ``cur``: SIFT matches on the wall,
    their range from the floor-wall boundary, and a RANSAC rigid fit. None if too few matches
    fit."""
    rng = rng or np.random.default_rng(0)
    fit = relative_pose(prev, cur, match_features(prev, cur), rng)
    return None if fit is None else fit[0]
