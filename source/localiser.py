"""
Online localisation on the frozen map, without ground truth: the pose ``(x, y, yaw)`` in the map
frame and its uncertainty ``sigma_m``, from the camera frames and the actions taken.

Each tick the pose is predicted by visual odometry against the previous frame (the action's
calibrated mean motion where that fails) and snapped to the maze's axes, and sigma grows. Every
``fix_every`` ticks a fix is tried: the VLAD top-10 of the mapped frames near the estimate,
reranked by SIFT inliers, and with enough of them the rigid relative pose gives an absolute pose,
blended with the estimate by their variances. A fix also needs the current frame's wall points to
lie on those of the matched frame and of the FIX_NEIGHBOURS frames either side of it, placed by
their mapped poses (geometry.layout_score >= ``layout_min``): that drops look-alike places. A fix
further from the estimate than max(GATE_M, 3 sigma) is not believed at once: it is held, and taken
only if the next fix agrees with it (HOLD_AGREE_M, HOLD_AGREE_DEG), as a replacement of the
estimate. With no accepted fix for ``lost_ticks`` ticks, or sigma over ``lost_sigma``, the robot
is lost: the search covers every mapped frame, and every fix needs that confirmation.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import cv2
import mapping
import numpy as np
from frozen_map import FrozenMap, vlad
from geometry import (
    LAYOUT_MIN,
    MANHATTAN_CONFIDENCE,
    Camera,
    FloorMask,
    epipolar_inliers,
    layout_score,
    manhattan_yaw,
    match_features,
    relative_pose,
    snap,
    visual_odometry,
    wall_view,
)
from mapping import compose, wrap

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
    candidate: mapping.Pose | None = None
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
        self.sift = cv2.SIFT_create()
        self.rng = np.random.default_rng(0)
        self.pose: mapping.Pose = (0.0, 0.0, 0.0)
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

    def step(self, frame: np.ndarray, action: str | None) -> Estimate:
        """Advance one tick: ``frame`` is the image after ``action`` (a name from
        geometry.MOTION_ACTIONS, None for the first frame or no movement)."""
        view = wall_view(frame, self.camera, self.floor, self.sift)
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
        if self.ticks % self.fix_every == 0:
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

    def _gate(self, fix: mapping.Pose) -> tuple[str, float, int]:
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
        if step is None:
            expected = self.motion.get(action) if action else None
            step = (expected["dx"], expected["dy"], expected["dyaw"]) if expected else (0, 0, 0)
        x, y, yaw = compose(self.pose, step)
        manhattan, confidence = manhattan_yaw(view.points)
        if confidence >= MANHATTAN_CONFIDENCE:
            snapped = snap(yaw, manhattan)
            if abs(wrap(snapped - yaw)) <= math.radians(mapping.SNAP_DEG):
                yaw = snapped
        self.pose = (x, y, yaw)
        self.sigma += DRIFT_M

    def _fix(self, view) -> tuple[mapping.Pose, float] | None:
        """The absolute pose of ``view`` from the map and its layout score, or None: near the
        estimate, or over every frame when lost."""
        rows = np.arange(len(self.map.xy))
        if not self.lost:
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

    def _apply(self, fix: mapping.Pose, *, replace: bool) -> float:
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
