"""
Online localisation on the frozen map, without ground truth: the pose ``(x, y, yaw)`` in the map
frame and its uncertainty ``sigma_m``, from the camera frames and the actions taken.

Each tick the pose is predicted by visual odometry against the previous frame (where that fails,
the median of the run's own visual odometry steps of the action once there are ONLINE_MIN, else
its calibrated mean motion) and snapped to the maze's axes, and sigma grows. Every
``fix_every`` ticks a fix is tried: the VLAD top-10 of the mapped frames near the estimate,
reranked by SIFT inliers, and with enough of them the rigid relative pose gives an absolute pose,
blended with the estimate by their variances. A fix also needs the current frame's wall points to
lie on those of the matched frame and of the FIX_NEIGHBOURS frames either side of it, placed by
their mapped poses (geometry.layout_score >= ``layout_min``): that drops look-alike places. A fix
further from the estimate than max(GATE_M, 3 sigma) is not believed at once: it is held, and taken
only if the next fix agrees with it (HOLD_AGREE_M, HOLD_AGREE_DEG), as a replacement of the
estimate. With a ``focus`` set a fix is tried every tick, against the frames near the focus.
With no accepted fix for ``lost_ticks`` ticks, or sigma over ``lost_sigma``, the robot
is lost: the search covers every mapped frame, and every fix needs that confirmation.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import cv2
import mapping
import numpy as np
from frozen_map import FrozenMap
from geometry import (
    LAYOUT_MIN,
    MANHATTAN_CONFIDENCE,
    Camera,
    FloorMask,
    Pose,
    compose,
    epipolar_inliers,
    layout_score,
    manhattan_yaw,
    match_features,
    relative_pose,
    snap,
    visual_odometry,
    vlad,
    wall_view,
    wrap,
)

START_SIGMA_M = 0.02
DRIFT_M = 0.01
"""Sigma grows this much per tick."""
FIX_SIGMA_M = 0.05
FIX_EVERY = 5
FIX_TOP_K = 10
FIX_INLIERS = 30
FIX_RADIUS_M = 0.5
FIX_NEIGHBOURS = 5
"""Frames either side of the matched one whose wall points the layout check uses."""
LOST_TICKS = 100
LOST_SIGMA_M = 0.5
GATE_M = 0.3
HOLD_AGREE_M = 0.1
HOLD_AGREE_DEG = 5.0
HOLD_TICKS = 20
"""A held fix is dropped when no fix confirms it within this many ticks."""
GREY_SIZE = (40, 30)
"""``grey_difference`` compares frames downsampled to this (width, height)."""
ONLINE_MIN = 20
"""Successful visual odometry steps of an action before the online median replaces the
calibrated motion."""


def grey_small(frame: np.ndarray) -> np.ndarray:
    """``frame`` as a small grey image, for ``grey_difference``."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, GREY_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def grey_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute grey difference of two ``grey_small`` images."""
    return float(np.abs(a - b).mean())


class Estimate(NamedTuple):
    x: float
    y: float
    yaw: float
    sigma_m: float
    fixed: bool
    """A fix was applied this tick."""
    jump: float
    """Metres the fix moved the estimate (0 without one)."""
    status: str | None
    """This tick's fix: "accepted", "held" (awaiting confirmation), or None."""
    rejected: int
    """Held fixes dropped this tick, confirmed by none."""
    lost: bool
    """The robot is lost (it is searching all frames)."""
    layout: float = math.nan
    """The layout score of this tick's fix candidate (NaN without one)."""
    candidate: Pose | None = None
    """This tick's fix candidate, whether or not the layout check passed it."""


class Localiser:
    def __init__(
        self,
        frozen_map: FrozenMap,
        camera: Camera,
        floor: FloorMask,
        *,
        motion: dict[str, dict] | None = None,
        fix_every: int = FIX_EVERY,
        lost_ticks: int = LOST_TICKS,
        lost_sigma: float = LOST_SIGMA_M,
        layout_min: float = LAYOUT_MIN,
    ) -> None:
        """``camera`` is the one the map was built with (the horizon shift included). ``motion``
        is mapping.calibrate's result, calibrated over the map's exploration frames if None."""
        self.map, self.camera, self.floor = frozen_map, camera, floor
        self.motion = motion or mapping.calibrate(frozen_map.sift.routes, camera, floor)
        self.fix_every, self.lost_ticks, self.lost_sigma = fix_every, lost_ticks, lost_sigma
        self.layout_min = layout_min
        self.samples: dict[str, list[tuple[float, float, float]]] = {}
        """Successful visual odometry steps by action."""
        self.grey: np.ndarray | None = None
        self.grey_diff = math.nan
        """The grey difference of the latest frame to the one before (NaN: none)."""
        self.source: str | None = None
        """How the latest step came: "vo", "calibrated" or "online" (None: no step)."""
        self.sift = cv2.SIFT_create()
        self.rng = np.random.default_rng(0)
        self.pose: Pose = (0.0, 0.0, 0.0)
        self.sigma = START_SIGMA_M
        self.ticks = 0
        self.since_fix = 0
        self.lost = False
        self._was_lost = False
        self.lost_events = 0
        self.counts = {"accepted": 0, "held": 0, "rejected": 0, "layout": 0}
        self.held: tuple[float, float, float, int] | None = None
        """The held fix's gap to the estimate (dx, dy, dyaw) and the tick it was made."""
        self.previous = None
        self.focus: tuple[float, float] | None = None
        """A map position: while set, a fix is tried every tick against the mapped frames within
        FIX_RADIUS_M of it (rather than of the estimate)."""

    def step(self, frame: np.ndarray, action: str | None) -> Estimate:
        """Advance one tick: ``frame`` is the image after ``action`` (a name from
        geometry.MOTION_ACTIONS, None for the first frame or no movement)."""
        view = wall_view(frame, self.camera, self.floor, self.sift)
        grey = grey_small(frame)
        self.grey_diff = math.nan if self.grey is None else grey_difference(self.grey, grey)
        self.grey = grey
        self.source = None
        self.ticks += 1
        self.since_fix += 1
        if self.previous is not None:
            self._predict(self.previous, view, action)
        self.previous = view

        jump, status, rejected = 0.0, None, 0
        candidate, layout = None, math.nan
        self.lost = self.since_fix >= self.lost_ticks or self.sigma > self.lost_sigma
        if self.lost and not self._was_lost:
            self.lost_events += 1
        self._was_lost = self.lost
        if self.focus is not None or self.ticks % self.fix_every == 0:
            if self.held is not None and self.ticks - self.held[3] > HOLD_TICKS:
                self.held, rejected = None, 1
            found = self._fix(view) if view.des is not None else None
            if found is not None:
                candidate, layout = found
                if layout >= self.layout_min:
                    status, jump, dropped = self._gate(candidate)
                    rejected += dropped
                else:
                    self.counts["layout"] += 1
            self.counts["rejected"] += rejected
        return Estimate(
            *self.pose,
            self.sigma,
            status == "accepted",
            jump,
            status,
            rejected,
            self.lost,
            layout,
            candidate,
        )

    def _gate(self, fix: Pose) -> tuple[str, float, int]:
        """Take or hold ``fix``: ``(status, metres moved, held fixes dropped)``."""
        x, y, yaw = self.pose
        gap = (fix[0] - x, fix[1] - y, wrap(fix[2] - yaw))
        held, self.held = self.held, None
        if held is not None and (
            math.hypot(gap[0] - held[0], gap[1] - held[1]) <= HOLD_AGREE_M
            and abs(wrap(gap[2] - held[2])) <= math.radians(HOLD_AGREE_DEG)
        ):
            return "accepted", self._apply(fix, replace=True), 0
        dropped = int(held is not None)
        if not self.lost and math.hypot(gap[0], gap[1]) <= max(GATE_M, 3 * self.sigma):
            return "accepted", self._apply(fix, replace=False), dropped
        self.held = (*gap, self.ticks)
        self.counts["held"] += 1
        return "held", 0.0, dropped

    def _predict(self, previous, view, action: str | None) -> None:
        step = visual_odometry(previous, view, self.rng)
        if step is not None and not mapping.plausible(step):
            step = None
        if step is not None:
            self.source = "vo"
            if action:
                self.samples.setdefault(action, []).append(step)
        else:
            step = self.fallback(action)
        x, y, yaw = compose(self.pose, step)
        manhattan, confidence = manhattan_yaw(view.points)
        if confidence >= MANHATTAN_CONFIDENCE:
            yaw = snap(yaw, manhattan)
        self.pose = (x, y, yaw)
        self.sigma += DRIFT_M

    def online(self, action: str) -> tuple[float, float, float] | None:
        """The median of the successful visual odometry steps of ``action`` this run (each of
        dx, dy, dyaw), None under ONLINE_MIN of them."""
        steps = self.samples.get(action, [])
        return (
            tuple(float(v) for v in np.median(steps, axis=0)) if len(steps) >= ONLINE_MIN else None
        )

    def fallback(self, action: str | None) -> tuple[float, float, float]:
        """The motion assumed for ``action`` when visual odometry fails (and sets ``source``)."""
        if action and (median := self.online(action)) is not None:
            self.source = "online"
            return median
        self.source = "calibrated"
        expected = self.motion.get(action) if action else None
        return (expected["dx"], expected["dy"], expected["dyaw"]) if expected else (0, 0, 0)

    def turn_rate(self) -> float | None:
        """Degrees a LEFT or RIGHT tick turns, from the online medians (the mean of the two
        absolute values, or of the one there is); None without either."""
        turns = [abs(math.degrees(m[2])) for a in ("LEFT", "RIGHT") if (m := self.online(a))]
        return float(np.mean(turns)) if turns else None

    def _fix(self, view) -> tuple[Pose, float] | None:
        """The absolute pose of ``view`` from the map and its layout score, or None: near the
        estimate, or over every frame when lost."""
        rows = np.arange(len(self.map.xy))
        if self.focus is not None:
            rows = rows[np.hypot(*(self.map.xy - self.focus).T) <= FIX_RADIUS_M]
        elif not self.lost:
            radius = max(FIX_RADIUS_M, 3 * self.sigma)
            rows = rows[np.hypot(*(self.map.xy - self.pose[:2]).T) <= radius]
        if not len(rows):
            return None
        similarity = self.map.vlad[rows] @ vlad(view.des, self.map.centers)
        top = rows[np.argsort(-similarity)[:FIX_TOP_K]]
        best = None
        for n in top:
            mapped = self.map.view(int(n))
            matches = match_features(mapped, view)
            inliers = epipolar_inliers(mapped, view, matches)
            if best is None or inliers.sum() > best[0]:
                best = (int(inliers.sum()), int(n), matches[inliers])
        if best is None or best[0] < FIX_INLIERS:
            return None
        _, n, matches = best
        fit = relative_pose(self.map.view(n), view, matches, self.rng)
        if fit is None:
            return None
        around = self.map.surroundings(self.map.neighbours(n, FIX_NEIGHBOURS), self.map.pose(n))
        return compose(self.map.pose(n), fit[0]), layout_score(around, view.points, fit[0])[0]

    def _apply(self, fix: Pose, *, replace: bool) -> float:
        """Blend the fix into the estimate (``replace`` it: a confirmed or lost-search fix);
        the metres it moved."""
        x, y, yaw = self.pose
        if replace:
            weight, self.sigma = 1.0, FIX_SIGMA_M
        else:
            weight = self.sigma**2 / (self.sigma**2 + FIX_SIGMA_M**2)
            self.sigma *= math.sqrt(1.0 - weight)
        dx, dy = weight * (fix[0] - x), weight * (fix[1] - y)
        self.pose = (x + dx, y + dy, wrap(yaw + weight * wrap(fix[2] - yaw)))
        self.counts["accepted"] += 1
        self.since_fix = 0
        self.lost = self._was_lost = False
        return math.hypot(dx, dy)
