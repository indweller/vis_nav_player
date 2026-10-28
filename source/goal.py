"""
Goal identification on a frozen map, without ground truth: where, in the map frame, was a goal
photographed from, given its four views (the goal seen at yaw + 0, 90, 180, 270 degrees).

For each view, VLAD retrieval of the TOP_K nearest mapped frames, reranked by the SIFT
matches' epipolar inliers. Each of the best CANDIDATES (frame, view) pairs gives a goal: the
frame's pose composed with the view's camera pose in the frame's camera frame (the rigid fit of
the wall keypoints; the servo's yaw alone, at the frame's own position, when that fails), turned
back by 90 degrees per view.

Look-alike places match well, so each candidate is also checked against the walls: the four views'
wall points, turned by 90 degrees per view into the goal's frame and merged, are placed at the
candidate's pose and scored (geometry.layout_score) against the mapped wall points of the frames
within NEAR_M of it. A candidate scoring LAYOUT_MIN beats any that does not; then the inliers
decide.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import cv2
import numpy as np
from frozen_map import FrozenMap, vlad
from geometry import (
    LAYOUT_MIN,
    Camera,
    FloorMask,
    WallView,
    epipolar_inliers,
    layout_score,
    match_features,
    relative_pose,
    transform,
    wall_view,
)
from mapping import compose, wrap
from servo import Servo

TOP_K = 20
"""Frames kept per view by VLAD, to be reranked."""
CANDIDATES = 10
"""(frame, view) pairs, best by inliers, whose layout is scored."""
NEAR_M = 1.0
"""The layout is scored against the mapped frames this near a candidate."""
DISTINCT_M = 0.3
"""Two candidate locations closer than this are one place."""
SERVO_MIN_MATCHES = 8


class Candidate(NamedTuple):
    inliers: int
    frame: int
    k: int
    x: float
    y: float
    yaw: float
    layout: float
    """The goal scan's layout score at this pose against the map (-1: not scored)."""
    relative: tuple[float, float, float] | None
    """The view's camera in the frame's camera frame; None if only the servo's yaw placed it."""


class Goal(NamedTuple):
    x: float
    y: float
    yaw: float
    """The goal's pose in the map frame: the heading the first view faces."""
    frame: int
    """Row of the chosen frame in the FrozenMap (``map.route[frame]``, ``map.frame[frame]``)."""
    k: int
    """Which view matched."""
    inliers: int
    confidence: int
    """The best's inliers less the best at a distinct location (> DISTINCT_M away)."""
    relative: tuple[float, float, float] | None = None
    """The matched view's camera in the frame's camera frame; None if only the servo's yaw
    placed it."""
    layout: float = 0.0
    """The layout score of the goal scan at the chosen pose."""
    view_agreement: int = 0
    """The views (the chosen one included) whose own best candidate, by inliers, is within
    DISTINCT_M of the chosen one."""


def goal_scan(seen: list[WallView]) -> np.ndarray:
    """The four views' wall points in the goal's frame: view k looks 90 degrees * k to the left
    of the goal's heading, from the same place."""
    return np.concatenate(
        [transform((0.0, 0.0, k * math.pi / 2), v.points) for k, v in enumerate(seen)]
    )


def find_goal(
    views: list[np.ndarray],
    store: FrozenMap,
    camera: Camera,
    floor: FloorMask,
    ranking: list[Candidate] | None = None,
    layout_min: float = LAYOUT_MIN,
) -> Goal | None:
    """The goal from its four BGR ``views``; None when no frame matches any of them. A list
    passed as ``ranking`` is filled with the scored candidates, best by inliers first."""
    sift = cv2.SIFT_create()
    rng = np.random.default_rng(0)
    seen = [wall_view(v, camera, floor, sift) for v in views]
    scored = []  # (inliers, frame row, k, matches[inliers])
    for k, view in enumerate(seen):
        if view.des is None:
            continue
        similarity = store.vlad @ vlad(view.des, store.centers)
        top = np.argpartition(-similarity, TOP_K)[:TOP_K]
        for n in top:
            mapped = store.view(int(n))
            matches = match_features(mapped, view)
            inliers = epipolar_inliers(mapped, view, matches)
            scored.append((int(inliers.sum()), int(n), k, matches[inliers]))
    if not scored or max(s[0] for s in scored) == 0:
        return None
    scored.sort(key=lambda s: -s[0])

    def place(entry: tuple) -> Candidate:
        """The goal pose a (frame, view) pair gives."""
        inliers, n, k, matches = entry
        pose = store.pose(n)
        mapped, view = store.view(n), seen[k]
        fit = relative_pose(mapped, view, matches, rng)
        if fit is not None:
            x, y, yaw = compose(pose, fit[0])
            relative = fit[0]
        else:
            servo = Servo(camera.fx, cx=camera.cx, min_matches=SERVO_MIN_MATCHES)
            offset, _ = servo.offset(
                (view.pixels, view.des), (mapped.pixels, mapped.des), camera.cx
            )
            x, y = pose[:2]
            yaw = pose[2] + (0.0 if offset is None else math.radians(offset))
            relative = None
        return Candidate(inliers, n, k, x, y, wrap(yaw - k * math.pi / 2), -1.0, relative)

    scan = goal_scan(seen)
    candidates = []
    for entry in scored[:CANDIDATES]:
        c = place(entry)
        near = np.flatnonzero(np.hypot(*(store.xy - (c.x, c.y)).T) <= NEAR_M)
        mapped_points = store.surroundings(near)
        candidates.append(c._replace(layout=layout_score(mapped_points, scan, c[3:6])[0]))
    if ranking is not None:
        ranking.extend(candidates)
    best = max(candidates, key=lambda c: (c.layout >= layout_min, c.inliers))

    # Each view's own best candidate (the first of its entries in `scored`).
    own = {}
    for entry in scored:
        own.setdefault(entry[2], entry)
    placed = {(c.frame, c.k): c for c in candidates}
    agreeing = sum(
        math.hypot(c.x - best.x, c.y - best.y) <= DISTINCT_M
        for c in (placed.get((e[1], k)) or place(e) for k, e in own.items())
    )

    far = [s[0] for s in scored if np.hypot(*(store.xy[s[1]] - store.xy[best.frame])) > DISTINCT_M]
    confidence = best.inliers - max(far, default=0)
    return Goal(
        best.x,
        best.y,
        best.yaw,
        best.frame,
        best.k,
        best.inliers,
        confidence,
        best.relative,
        best.layout,
        agreeing,
    )
