"""
The planner on the localiser's estimate, without ground truth: keeps a path from the estimated
pose to the goal and says where to head next. No controller: it only reports the path and the
next waypoint.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from localiser import Estimate, Localiser
from planner import Path, PlaceGraph, Pose
from scipy.spatial import cKDTree

REPLAN_M = 0.15
"""Replan when the estimate is this far from the path."""
JUMP_M = 0.1
"""Replan when a fix moves the estimate this far."""
REPLAN_EVERY = 200
LOOKAHEAD_M = 0.1


class Update(NamedTuple):
    estimate: Estimate
    path: Path | None
    replan: str | None
    """Why a new path was made this tick: "initial", "cross_track", "jump" or "periodic"."""
    waypoint: Pose | None
    """The first waypoint at least LOOKAHEAD_M ahead along the path."""


class Navigator:
    def __init__(
        self,
        planner: PlaceGraph,
        localiser: Localiser,
        goal_pose: Pose,
        *,
        replan_m: float = REPLAN_M,
        replan_every: int = REPLAN_EVERY,
        jump_m: float = JUMP_M,
        store=None,
        corridor_m: float | None = None,
    ) -> None:
        """With a ``store`` (the FrozenMap) and ``corridor_m``, every new path is shortcut
        (``Path.shortcut``) before it is used."""
        self.planner, self.localiser, self.goal = planner, localiser, goal_pose
        self.replan_m, self.replan_every, self.jump_m = replan_m, replan_every, jump_m
        self.store, self.corridor_m = store, corridor_m
        self.tree = None if store is None or corridor_m is None else cKDTree(store.xy)
        self.path: Path | None = None
        self.since_plan = 0

    def update(
        self, frame: np.ndarray, action: str | None, estimate: Estimate | None = None
    ) -> Update:
        """Localise on ``frame`` (or take ``estimate``, the localiser's step on it, already made
        for another navigator), replan if need be, and say where to go."""
        if estimate is None:
            estimate = self.localiser.step(frame, action)
        pose = estimate[:3]
        self.since_plan += 1
        reason = None
        if self.path is None:
            reason = "initial"
        elif self.path.cross_track(pose)[0] > self.replan_m:
            reason = "cross_track"
        elif estimate.jump > self.jump_m:
            reason = "jump"
        elif self.since_plan >= self.replan_every:
            reason = "periodic"
        if reason is not None:
            planned = self.planner.plan(pose, self.goal)
            if planned is None:
                reason = None
            else:
                if self.tree is not None:
                    planned = planned.shortcut(self.store, self.corridor_m, self.tree)
                self.path, self.since_plan = planned, 0
        waypoint = None
        if self.path is not None:
            waypoint = self.path.waypoint_after(self.path.cross_track(pose)[1], LOOKAHEAD_M)
        return Update(estimate, self.path, reason, waypoint)
