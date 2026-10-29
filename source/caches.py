"""
What a data folder's caches were made with, so that a stale one is noticed.

``map_graph.npz`` (frozen_map), ``motion.json`` (map_agent.load_motion) and ``links.npz``
(planner.PlaceGraph) each keep a fingerprint: a hash of the constants listed here for it and, for
the first two, of the floor mask's table. Loading a cache compares its fingerprint with the current
code's and warns, once per file, when it differs or is missing; the cache is used either way.
``map_agent.py --rebuild`` makes all three again.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType

import geometry
import mapping
import numpy as np

Settings = dict[ModuleType, tuple[str, ...]]

VIEW = (
    "CAMERA",
    "CAMERA_HEIGHT",
    "FAR",
    "LUT_BINS",
    "SEGMENT_SPAN",
    "SEGMENT_MAX_GAP",
    "SEGMENT_MAX_RANGE",
    "HIST_BIN_DEG",
    "PEAK_HALF_WIDTH_DEG",
    "MANHATTAN_CONFIDENCE",
)
"""geometry's settings of a frame's wall ranges and Manhattan yaw."""
CALIBRATION = (
    "MAP_RES",
    "LOG_ODDS_HIT",
    "LOG_ODDS_MAX",
    "OCCUPIED_LOG_ODDS",
    "FIELD_SIGMA",
    "SEARCH",
    "FINE_STEPS",
    "CALIBRATION_RANGE",
    "CALIBRATION_MIN_SCORE",
    "CONDITIONING_M",
    "CONDITIONING_DROP",
    "MIN_MANHATTAN_PAIRS",
)
"""geometry's settings of calibrate_motion's scan matching."""
INLIERS = ("MATCH_RATIO", "EPIPOLAR_PX")
"""geometry's settings of the SIFT epipolar inliers between two frames."""
FIT = (
    "KEYPOINT_MARGIN_PX",
    "FIT_RANGE",
    "FIT_INLIER_M",
    "FIT_ITERATIONS",
    "FIT_MIN_SEPARATION",
    "FIT_MIN_INLIERS",
)
"""geometry's settings of visual odometry's and the closures' rigid fits."""

LINKS: Settings = {geometry: INLIERS}
MOTION: Settings = {geometry: VIEW + CALIBRATION, mapping: ("HORIZON_SHIFT_PX",)}
MAP: Settings = {
    geometry: VIEW + CALIBRATION + INLIERS + FIT + ("MAP_HALF_EXTENT",),
    mapping: (
        "NODE_EVERY",
        "QUERY_EVERY",
        "TOP_K",
        "EXCLUDE_TICKS",
        "MIN_SIFT_INLIERS",
        "MAX_CLOSURE_M",
        "MAX_CLOSURE_DEG",
        "PRE_GATE_M",
        "PRE_GATE_DEG",
        "ROUNDS",
        "GRID_MAX_RANGE",
        "MAX_STEP_M",
        "MAX_STEP_DEG",
        "START_FRAMES",
        "START_MIN_INLIERS",
        "CODEBOOK_DESCRIPTORS",
        "HORIZON_SHIFT_PX",
        "ODOM_SIGMA_M",
        "ODOM_SIGMA_DEG",
        "CLOSURE_SIGMA_M",
        "CLOSURE_SIGMA_DEG",
        "START_SIGMA_M",
        "START_SIGMA_DEG",
        "HUBER_SCALE",
    ),
}


def fingerprint(settings: Settings, floor: geometry.FloorMask | None = None) -> str:
    """A hash of the current values of ``settings`` and of ``floor``'s table."""
    values = [(m.__name__, n, getattr(m, n)) for m, names in settings.items() for n in names]
    digest = hashlib.sha256(repr(values).encode())
    if floor is not None:
        digest.update(np.ascontiguousarray(floor.table).tobytes())
    return digest.hexdigest()[:16]


_warned: set[Path] = set()


def check(path: Path, saved: str | None, expected: str) -> None:
    """Warn, once per file, when the cache at ``path`` has no fingerprint (``saved`` None) or
    another than ``expected``."""
    path = Path(path).resolve()
    if saved == expected or path in _warned:
        return
    _warned.add(path)
    why = (
        "has no settings fingerprint (it predates cache versioning)"
        if saved is None
        else "was made with other settings than the current code's"
    )
    print(
        f"WARNING: cache {path} {why}; using it anyway (map_agent.py --rebuild makes it again)",
        flush=True,
    )
