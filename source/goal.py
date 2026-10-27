"""
Goal identification on a frozen map, without ground truth: where, in the map frame, was a goal
photographed from, given its four views (the goal seen at yaw + 0, 90, 180, 270 degrees).

For each view, VLAD retrieval of the TOP_K nearest mapped frames, reranked by the SIFT
matches' epipolar inliers; the best (frame, view) pair gives the goal: the frame's pose
composed with the view's camera pose in the frame's camera frame (the rigid fit of the wall
keypoints; the servo's yaw alone, at the frame's own position, when that fails), turned back
by 90 degrees per view.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import cv2
import numpy as np
from frozen_map import FrozenMap, vlad
from geometry import (
    Camera,
    FloorMask,
    epipolar_inliers,
    match_features,
    relative_pose,
    wall_view,
)
from mapping import compose, wrap
from servo import Servo

TOP_K = 20
"""Frames kept per view by VLAD, to be reranked."""
DISTINCT_M = 0.3
"""Two candidate locations closer than this are one place."""
SERVO_MIN_MATCHES = 8


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


def find_goal(
    views: list[np.ndarray], store: FrozenMap, camera: Camera, floor: FloorMask
) -> Goal | None:
    """The goal from its four BGR ``views``; None when no frame matches any of them."""
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
    if not scored:
        return None
    inliers, n, k, matches = max(scored, key=lambda s: s[0])
    if inliers == 0:
        return None

    far = [s[0] for s in scored if np.hypot(*(store.xy[s[1]] - store.xy[n])) > DISTINCT_M]
    confidence = inliers - max(far, default=0)

    pose = store.pose(n)
    mapped, view = store.view(n), seen[k]
    fit = relative_pose(mapped, view, matches, rng)
    if fit is not None:
        x, y, yaw = compose(pose, fit[0])
        relative = fit[0]
    else:
        servo = Servo(camera.fx, cx=camera.cx, min_matches=SERVO_MIN_MATCHES)
        offset, _ = servo.offset((view.pixels, view.des), (mapped.pixels, mapped.des), camera.cx)
        x, y = pose[:2]
        yaw = pose[2] + (0.0 if offset is None else math.radians(offset))
        relative = None
    return Goal(x, y, wrap(yaw - k * math.pi / 2), n, k, inliers, confidence, relative)
