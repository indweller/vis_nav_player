"""
Map-based agent: follows a path on the frozen map to the goal, then homes in on the goal views.

At construction it builds (or loads) the frozen map of the exploration data (``map_graph.npz``
and the planner's ``links.npz`` cached in the data directory, the motion calibration in
``motion.json``). At setup the goal is ``goal.find_goal`` on the four target views (or the
``goal_override`` pose, for evaluations), the localiser starts at the map's origin (the spawn),
and a navigator plans to the goal.

Each tick the navigator localises (given the move just made), replans if it must, and the agent
aims at the point --lookahead-m ahead on the path. Moves are held as queues of single ticks, so
the localiser sees every frame:
  - heading error to the waypoint over --turn-tol: turn toward it for clip(round(error / turn
    rate), 1, 3) ticks; else FORWARD for --forward-ticks;
  - wall guard: a FORWARD with the centre column's wall range under --guard-m becomes BACKWARD
    for 2 ticks, and the path is replanned;
  - homing, once the estimate is within --near-m of the goal (and from then on): the frame is
    matched against the four goal views by SIFT (servo.py) to find which view it is looking at
    and so the yaw to view 0; turn until that is under --align-deg; then, if the rigid relative
    pose to that view puts the goal over --offset-m away, step toward it, a tick at a time (along
    the heading if it is off that way only; else turn to it and drive, aligning again once
    there); CHECKIN when aligned and within --offset-m, or after --homing-max homing decisions.
    The views match only within about 0.1 m of the goal; without a match, the estimate stands
    in for the yaw and the offset.
  - CHECKIN when fewer than 5 steps are left.

    uv run source/map_agent.py --local 7
    uv run source/map_agent.py --challenge <id>
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cli
import cv2
import frozen_map
import mapping
import navigator as navigator_module
import numpy as np
import planner as planner_module
from geometry import Camera, FloorMask, epipolar_inliers, match_features, relative_pose, wall_view
from goal import find_goal
from localiser import Localiser
from mapping import between, wrap
from servo import Servo
from vis_nav_sdk import Action, Agent, Observation, Result, SessionInfo, run

FLOOR_MASK_PATH = cli.ROOT / "data" / "floor_mask.npz"
MOTION_NAME = "motion.json"
FOV_DEG = 120.0
"""The camera's horizontal field of view; with the frame size it gives fx."""
DEFAULT_TURN_DEG = 3.4375

LOOKAHEAD_M = 0.15
TURN_TOL_DEG = 12.0
FORWARD_TICKS = 2
MAX_TURN_TICKS = 3
GUARD_M = 0.07
GUARD_TICKS = 2
NEAR_M = 0.15
ALIGN_DEG = 8.0
OFFSET_M = 0.08
HOMING_MAX = 60
VIEW_GATE_DEG = 30.0
"""Homing believes a goal view's yaw only within this of the estimate's."""
SERVO_MIN_MATCHES = 8
"""Inliers the servo needs between the frame and a goal view (the views match only near)."""
LOW_STEPS = 5
"""CHECKIN when fewer steps than this are left."""

Pose = tuple[float, float, float]


def camera_dict(frame: np.ndarray) -> dict:
    """The camera of the challenge's frames: the size of ``frame``, a 120 degree field of view
    and the principal point at the centre."""
    height, width = frame.shape[:2]
    fx = float(round(width / 2 / math.tan(math.radians(FOV_DEG / 2))))
    return {"width": width, "height": height, "fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2}


def load_motion(root: Path, routes, camera: Camera, floor: FloorMask) -> dict[str, dict]:
    """mapping.calibrate over the exploration routes, kept in ``root/motion.json``."""
    path = root / MOTION_NAME
    if path.exists():
        return json.loads(path.read_text())
    motion = mapping.calibrate(routes, camera, floor)
    path.write_text(json.dumps(motion, default=lambda v: np.asarray(v).tolist()))
    return motion


class MapAgent(Agent):
    def __init__(
        self,
        data_dir: str | Path,
        *,
        goal_override: Pose | None = None,
        lookahead_m: float = LOOKAHEAD_M,
        turn_tol: float = TURN_TOL_DEG,
        forward_ticks: int = FORWARD_TICKS,
        guard_m: float = GUARD_M,
        near_m: float = NEAR_M,
        align_deg: float = ALIGN_DEG,
        offset_m: float = OFFSET_M,
        homing_max: int = HOMING_MAX,
    ) -> None:
        """``goal_override``: a goal pose in the map frame, used instead of ``find_goal``."""
        self.root = Path(data_dir)
        self.goal_override = goal_override
        self.lookahead_m, self.turn_tol, self.forward_ticks = lookahead_m, turn_tol, forward_ticks
        self.guard_m, self.near_m, self.align_deg = guard_m, near_m, align_deg
        self.offset_m, self.homing_max = offset_m, homing_max

        self.floor = FloorMask.load(FLOOR_MASK_PATH)
        routes = mapping.load_routes(self.root)
        self.camera_dict = camera_dict(routes[0].frames[0])
        self.camera = Camera(
            {**self.camera_dict, "cy": self.camera_dict["cy"] + mapping.HORIZON_SHIFT_PX}
        )
        self.store, self.routes = frozen_map.get(self.root, self.camera_dict, self.floor)
        self.motion = load_motion(self.root, self.routes, self.camera, self.floor)
        self.planner = planner_module.PlaceGraph(self.store, cache_path=self.root / "links.npz")
        turns = [
            abs(math.degrees(self.motion[a]["dyaw"])) for a in ("LEFT", "RIGHT") if a in self.motion
        ]
        self.turn_rate = float(np.mean(turns)) if turns else DEFAULT_TURN_DEG
        """Degrees one LEFT or RIGHT tick turns, from the exploration data."""
        self.reset()

    def reset(self) -> None:
        self.goal: Pose | None = None
        self.found_goal: Pose | None = None
        self.localiser: Localiser | None = None
        self.navigator: navigator_module.Navigator | None = None
        self.queue: list[str] = []
        self.last: str | None = None
        """The move taken since the previous frame (None: none)."""
        self.mode = "start"
        self.update: navigator_module.Update | None = None
        self.checkin_reason: str | None = None
        self.homing = False
        self.homing_steps = 0
        self.approaching = False
        """Homing: driving to the goal sideways-on, not holding the alignment."""
        self.guard_events = 0
        self.replan_pending = False
        self.guarded = False
        """The wall guard fired on this tick."""
        self.view: tuple[int, float, float] | None = None
        """Homing: the goal view the frame matched, the yaw to view 0 and the offset in metres."""
        self.result: Result | None = None

    # -- the session

    def setup(self, info: SessionInfo) -> None:
        self.reset()
        self.localiser = Localiser(self.store, self.camera, self.floor, motion=self.motion)
        if self.goal_override is not None:
            self.goal = tuple(float(v) for v in self.goal_override)
        else:
            found = find_goal(info.targets, self.store, self.camera, self.floor)
            self.goal = None if found is None else (found.x, found.y, found.yaw)
        self.found_goal = self.goal
        if self.goal is not None:
            self.navigator = navigator_module.Navigator(self.planner, self.localiser, self.goal)
        sift = cv2.SIFT_create()
        self.goal_views = [wall_view(t, self.camera, self.floor, sift) for t in info.targets]
        self.servo = Servo(self.camera.fx, cx=self.camera.cx, min_matches=SERVO_MIN_MATCHES)
        self.goal_features = [self.servo.features(t) for t in info.targets]
        self.rng = np.random.default_rng(0)

    def finish(self, result: Result | None) -> None:
        self.result = result

    def act(self, obs: Observation) -> tuple[Action, int]:
        self.guarded = False
        self.view = None
        if self.navigator is None:
            estimate = self.localiser.step(obs.image, self.last)
            self.update = navigator_module.Update(estimate, None, None, None)
        else:
            if self.replan_pending and not self.queue:
                self.navigator.path, self.replan_pending = None, False
            self.update = self.navigator.update(obs.image, self.last)
        if obs.steps_left < LOW_STEPS:
            return self._checkin("low_steps")
        if self.navigator is None:
            self.mode = "search"
            return self._move("LEFT", obs)
        if not self.queue:
            self.mode, self.queue = self._decide(obs)
            if self.queue == ["CHECKIN"]:
                return self._checkin(self.checkin_reason or "homed")
        return self._move(self.queue.pop(0), obs)

    def _move(self, name: str, obs: Observation) -> tuple[Action, int]:
        """One tick of ``name``, unless the wall guard turns a FORWARD into a retreat."""
        if name == "FORWARD" and self._centre_range() < self.guard_m:
            self.guard_events += 1
            self.guarded = True
            self.mode, self.replan_pending = "guard", True
            self.queue = ["BACKWARD"] * (GUARD_TICKS - 1)
            name = "BACKWARD"
        self.last = name
        return getattr(Action, name), 1

    def _checkin(self, reason: str) -> tuple[Action, int]:
        self.checkin_reason, self.last, self.queue = reason, None, []
        return Action.CHECKIN, 1

    def _centre_range(self) -> float:
        """Metres to the wall in the centre column of the latest frame (inf: none in range)."""
        previous = self.localiser.previous
        if previous is None:
            return math.inf
        x, y = previous.points[len(previous.points) // 2]
        return math.inf if math.isnan(x) else math.hypot(x, y)

    # -- the decisions

    def _decide(self, obs: Observation) -> tuple[str, list[str]]:
        """``(mode, queue of single ticks)``; ["CHECKIN"] to check in."""
        estimate = self.update.estimate
        gx, gy, _ = self.goal
        if not self.homing and math.hypot(estimate.x - gx, estimate.y - gy) <= self.near_m:
            self.homing = True
        if self.homing:
            return "home", self._home(obs)
        path = self.update.path
        if path is None:
            return "search", ["LEFT"]
        waypoint = path.waypoint_after(path.cross_track(estimate[:3])[1], self.lookahead_m)
        error = math.degrees(
            wrap(math.atan2(waypoint[1] - estimate.y, waypoint[0] - estimate.x) - estimate.yaw)
        )
        return "nav", self._steer(error)

    def _steer(self, error: float) -> list[str]:
        """Turn toward a waypoint ``error`` degrees to the left, or drive at it."""
        if abs(error) > self.turn_tol:
            return self._turn(error)
        return ["FORWARD"] * self.forward_ticks

    def _turn(self, error: float) -> list[str]:
        ticks = int(np.clip(round(abs(error) / self.turn_rate), 1, MAX_TURN_TICKS))
        return ["LEFT" if error > 0 else "RIGHT"] * ticks

    def _home(self, obs: Observation) -> list[str]:
        self.homing_steps += 1
        if self.homing_steps > self.homing_max:
            self.checkin_reason = "homing_max"
            return ["CHECKIN"]
        yaw, (x, y) = self._goal_error(obs.image)
        distance = math.hypot(x, y)
        if self.approaching:
            if distance > self.offset_m:
                return self._approach(x, y)
            self.approaching = False
        if abs(yaw) >= self.align_deg:
            return self._turn(yaw)
        if distance > self.offset_m:
            if abs(y) <= self.offset_m / 2:
                # Off along the heading only: keep the alignment and drive it out.
                return ["FORWARD" if x > 0 else "BACKWARD"]
            # Off to the side: turn and drive, and align again once there.
            self.approaching = True
            return self._approach(x, y)
        self.checkin_reason = "homed"
        return ["CHECKIN"]

    def _approach(self, x: float, y: float) -> list[str]:
        """One tick toward the goal at ``(x, y)`` in the robot frame, front or back end first."""
        angle = math.degrees(math.atan2(y, x))
        if abs(angle) <= self.turn_tol:
            return ["FORWARD"]
        if abs(angle) >= 180 - self.turn_tol:
            return ["BACKWARD"]
        return self._turn(angle if abs(angle) < 90 else angle - math.copysign(180.0, angle))[:1]

    def _goal_error(self, image: np.ndarray) -> tuple[float, tuple[float, float]]:
        """``(degrees to turn left to face view 0, goal position in the robot frame)``. The
        estimate gives both; the goal view the frame matches best (SIFT, servo.py) gives the yaw,
        and the rigid relative pose to it the position, where they exist. A view is only
        believed if its yaw is within VIEW_GATE_DEG of the estimate's: look-alikes match too."""
        estimate = self.update.estimate
        error = math.degrees(wrap(self.goal[2] - estimate.yaw))
        there = between(estimate[:3], self.goal)[:2]
        current = self.servo.features(image)
        best = None
        for k, features in enumerate(self.goal_features):
            degrees, inliers = self.servo.offset(current, features, self.camera.cx)
            if degrees is None:
                continue
            seen = math.degrees(wrap(-math.radians(k * 90.0 + degrees)))
            if abs(wrap(math.radians(seen - error))) > math.radians(VIEW_GATE_DEG):
                continue
            if best is None or inliers > best[0]:
                best = inliers, k, seen
        if best is None:
            return error, there
        _, k, error = best
        view, goal_view = self.localiser.previous, self.goal_views[k]
        matches = match_features(goal_view, view)
        fit = relative_pose(
            goal_view, view, matches[epipolar_inliers(goal_view, view, matches)], self.rng
        )
        if fit is not None:
            there = between(fit[0], (0.0, 0.0, 0.0))[:2]
        self.view = k, error, math.hypot(*there)
        return error, there

    # -- the viewer

    def hud(self) -> list[str]:
        if self.update is None:
            return ["map agent: starting"]
        e = self.update.estimate
        left = (
            math.nan
            if self.update.path is None
            else self.update.path.length - self.update.path.cross_track(e[:3])[1]
        )
        return [
            f"est ({e.x:.2f}, {e.y:.2f}) {math.degrees(e.yaw):.0f} deg",
            f"sigma {e.sigma_m:.2f} m{'  LOST' if e.lost else ''}",
            f"path left {left:.2f} m",
            f"mode {self.mode}",
        ]


if __name__ == "__main__":
    parser = cli.parser(__doc__)
    parser.add_argument("--data", default=None, help="exploration data directory")
    parser.add_argument("--lookahead-m", type=float, default=LOOKAHEAD_M)
    parser.add_argument("--turn-tol", type=float, default=TURN_TOL_DEG, help="degrees")
    parser.add_argument("--forward-ticks", type=int, default=FORWARD_TICKS)
    parser.add_argument("--guard-m", type=float, default=GUARD_M)
    parser.add_argument("--near-m", type=float, default=NEAR_M)
    parser.add_argument("--align-deg", type=float, default=ALIGN_DEG)
    parser.add_argument("--offset-m", type=float, default=OFFSET_M)
    parser.add_argument("--homing-max", type=int, default=HOMING_MAX)
    args = parser.parse_args()

    agent = MapAgent(
        cli.exploration_data(args, args.data),
        lookahead_m=args.lookahead_m,
        turn_tol=args.turn_tol,
        forward_ticks=args.forward_ticks,
        guard_m=args.guard_m,
        near_m=args.near_m,
        align_deg=args.align_deg,
        offset_m=args.offset_m,
        homing_max=args.homing_max,
    )
    result = run(agent, cli.challenge(args), **cli.run_options(args))
    print(f"result: {result}")
