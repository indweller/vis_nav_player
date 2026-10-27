"""
Visual servoing along an edge: how far to turn so the camera faces the way a recorded view
faces, from the horizontal shift of RootSIFT matches between the two images.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import cv2
import numpy as np


class Servo:
    """``yaw_offset(image, path)`` is the yaw from the current view to the recorded image at
    ``path``, positive when that view lies to the right. Recorded images' features are
    LRU-cached by path; the current view's are kept for the next call on the same array."""

    def __init__(
        self,
        fx: float,
        *,
        cx: float | None = None,
        ratio: float = 0.75,
        min_matches: int = 15,
        cache_size: int = 256,
    ) -> None:
        self.fx = fx
        self.cx = cx
        """Principal point; None: the image's centre column."""
        self.ratio = ratio
        self.min_matches = min_matches
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.image_features = functools.lru_cache(maxsize=cache_size)(self._image_features)
        self._view: tuple[np.ndarray, tuple[np.ndarray, np.ndarray | None]] | None = None
        """The last current view and its features: locating matches one view many times."""

    def features(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Keypoint positions ``(N, 2)`` and RootSIFT descriptors ``(N, 128)`` (None: none)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        keypoints, des = self.sift.detectAndCompute(gray, None)
        if des is None or len(des) == 0:
            return np.empty((0, 2), np.float32), None
        des = des / (np.sum(des, axis=1, keepdims=True) + 1e-7)
        points = np.array([k.pt for k in keypoints], np.float32)
        return points, np.sqrt(des).astype(np.float32)

    def _image_features(self, path: Path) -> tuple[np.ndarray, np.ndarray | None]:
        return self.features(cv2.imread(str(path)))

    def yaw_offset(self, image: np.ndarray, path: Path) -> tuple[float | None, int]:
        """``(degrees, inliers)``; degrees is None with fewer than ``min_matches`` inliers."""
        if self._view is None or self._view[0] is not image:
            self._view = image, self.features(image)
        cx = self.cx if self.cx is not None else (image.shape[1] - 1) / 2
        return self.offset(self._view[1], self.image_features(Path(path)), cx)

    def offset(
        self,
        current: tuple[np.ndarray, np.ndarray | None],
        target: tuple[np.ndarray, np.ndarray | None],
        cx: float,
    ) -> tuple[float | None, int]:
        """``yaw_offset`` from two ``features``, ``cx`` being the principal point's column."""
        (cur_points, cur_des), (rec_points, rec_des) = current, target
        if cur_des is None or rec_des is None or len(cur_des) < 2 or len(rec_des) < 2:
            return None, 0
        good = [
            m
            for m, n in (p for p in self.matcher.knnMatch(cur_des, rec_des, k=2) if len(p) == 2)
            if m.distance < self.ratio * n.distance
        ]
        if len(good) < self.min_matches:
            return None, len(good)
        cur = cur_points[[m.queryIdx for m in good]]
        rec = rec_points[[m.trainIdx for m in good]]
        # A feature at u in the recorded view sits at u + dx in ours: that view faces to the
        # right of us when dx > 0. RANSAC on a similarity transform drops the mismatches.
        _, mask = cv2.estimateAffinePartial2D(cur, rec, method=cv2.RANSAC)
        if mask is not None and mask.any():
            keep = mask.ravel().astype(bool)
            cur, rec = cur[keep], rec[keep]
        inliers = len(cur)
        if inliers < self.min_matches:
            return None, inliers
        # Bearings rather than one pixel shift: across a wide field of view a feature off the
        # centre moves up to sec^2 of its bearing more pixels than one on it.
        bearing = np.arctan((cur[:, 0] - cx) / self.fx) - np.arctan((rec[:, 0] - cx) / self.fx)
        return math.degrees(float(np.median(bearing))), inliers
