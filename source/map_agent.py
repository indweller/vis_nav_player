"""
Map-based agent: follows a path on the frozen map to the goal, then homes in on the goal views.

At construction it builds (or loads) the frozen map of the exploration data (``map_graph.npz``
and the planner's ``links.npz`` cached in the data directory, the motion calibration in
``motion.json``). Each cache keeps a fingerprint of the settings it was made with and warns when
stale (caches.py); --rebuild makes all three again. At setup the goal is ``goal.find_goal`` on
the four target views (or the ``goal_override`` pose, for evaluations), the localiser starts at
the map's origin (the spawn), and a navigator plans to the goal.

Each tick the navigator localises (given the move just made), replans if it must, and the agent
aims at the point --lookahead-m ahead on the path. Moves are held as queues of single ticks, so
the localiser sees every frame:
  - heading error to the waypoint at most --turn-tol: FORWARD for --forward-ticks; over it: turn
    toward it for clip(round(error / turn rate), 1, --max-turn-ticks) ticks (3; 26 with --arcs).
    With --arcs, an error up to --arc-deg is one tick of an arc (FORWARD with LEFT or RIGHT)
    instead.
    Arcs have no calibration of their own: the localiser's fallback motion for one is the FORWARD
    and turn means summed (``arc_motion``). An arc turns on a circle of radius R = step / turn per
    tick and never reaches a waypoint inside the circle (closer than 2 R sin(error)): it orbits
    it, so such an error is turned in place instead;
  - with --shortcut, every plan is string-pulled (``Path.shortcut``, --corridor-m);
  - wall guard: a FORWARD with the centre column's wall range under --guard-m becomes BACKWARD
    for 2 ticks, and the path is replanned;
  - with --floor-guard, the same when the floor fills under --floor-guard-frac of the bottom
    rows of the central columns (a wall closer than about 0.15 m, which the wall range cannot read);
  - with --blocked-zero, a FORWARD / BACKWARD tick on which visual odometry fails and the frame
    differs from the previous one by under --blocked-diff (grey, downsampled) is assumed to have
    gone nowhere, rather than to have moved by the calibrated mean;
  - unless --no-online-motion, the localiser's fallback motion and the turn rate are the medians of
    the run's own visual odometry steps per action, once there are 20;
  - homing, once the estimate is within --near-m of the goal (and from then on): the localiser
    tries a fix every tick against the mapped frames within 0.5 m of the goal (its SIFT and layout
    checks). The target is the goal pose. The frame is matched against the four goal views by
    SIFT (servo.py) to find which view it is looking at and so the yaw to view 0 (the estimate's
    when none matches; the views match only within about 0.1 m of the goal). By default
    (--align-source estimate) the turn is the estimate's yaw error alone: the view match is still
    computed (and kept in ``view``) but does not set it; --align-source view lets a matched view
    set it. Turn until that is under --align-deg; then, if the estimate is over --offset-m from
    the goal, step toward it, a tick at a time (along the heading if it is off that way only; else
    turn to it and drive, aligning again once there); CHECKIN when aligned and within --offset-m.
    Unless --no-fixes: the goal view overrides the estimate's yaw only with at least
    15 inliers (else 8) and within 30 degrees of it. Without a match the estimate's yaw (against
    the goal's from find_goal) must be under 5 degrees. After --homing-max homing decisions the
    agent turns by the estimate alone to that yaw error and then checks in, wherever it is;
    with --no-fixes it checks in at once.
  - CHECKIN when fewer than 5 steps are left (the only check-in before alignment).
  - At check-in the homing outcome and the heading error (the estimate's, against the goal's yaw)
    are printed.

    uv run source/map_agent.py --local 7
    uv run source/map_agent.py --challenge <id>
    uv run source/map_agent.py --test-local --data data/<id>   # offline check, no session
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import caches
import cli
import cv2
import frozen_map
import geometry
import mapping
import navigator as navigator_module
import numpy as np
import planner as planner_module
from geometry import Camera, FloorMask, floor_fraction
from goal import find_goal
from localiser import Localiser
from mapping import between, wrap
from servo import Servo
from vis_nav_sdk import Action, Agent, Observation, Result, SessionInfo, run

FLOOR_MASK_PATH = geometry.FLOOR_MASK_PATH
MOTION_NAME = "motion.json"
LINKS_NAME = "links.npz"
DEFAULT_TURN_DEG = 3.4375

LOOKAHEAD_M = 0.15
TURN_TOL_DEG = 12.0
ARC_DEG = 45.0
FORWARD_TICKS = 2
MAX_TURN_TICKS = 3
ARC_MAX_TURN_TICKS = 26
"""Most ticks of a turn in place when arcs are on."""
HOMING_TURN_TICKS = 3
"""Most ticks of a turn in place while homing."""
ARC_TOLERANCE = 0.2
"""``arc_motion`` replaces the summed means with a measurement that differs by more than this."""
ARCS = {"LEFT": "FORWARD|LEFT", "RIGHT": "FORWARD|RIGHT"}
GUARD_M = 0.07
GUARD_TICKS = 2
FLOOR_GUARD_FRAC = 0.49
"""``--floor-guard``: the floor fraction of the central bottom rows under which a wall is under
0.15 m away (fitted on a dev-like run, F1 0.98: eval_dev_like_flags.py)."""
BLOCKED_DIFF = 17.3
"""``--blocked-zero``: the grey difference (localiser.grey_difference) under which a failed-VO
FORWARD / BACKWARD tick is taken as blocked (fitted on a dev-like run, F1 0.97)."""
NEAR_M = 0.15
ALIGN_DEG = 8.0
ALIGN_SOURCES = ("estimate", "view")
OFFSET_M = 0.05
HOMING_MAX = 60
VIEW_GATE_DEG = 30.0
"""Homing believes a goal view's yaw only within this of the estimate's."""
ESTIMATE_ALIGN_DEG = 5.0
"""Without a goal view match, the estimate's yaw error must be under this to check in."""
LEGACY_MIN_MATCHES = 8
"""SERVO_MIN_MATCHES before the alignment fixes (``fixes=False``, i.e. --no-fixes)."""
SERVO_MIN_MATCHES = 15
"""Inliers the servo needs between the frame and a goal view (the views match only near)."""
LOW_STEPS = 5
"""CHECKIN when fewer steps than this are left."""

GOAL_CAMERAS = ("auto", "frame", "photo")
GOAL_VIEW_PX = 320
"""Width of each of the four goal views split from target.jpg."""

Pose = tuple[float, float, float]


def load_motion(root: Path, routes, camera: Camera, floor: FloorMask) -> dict[str, dict]:
    """mapping.calibrate over the exploration routes, kept in ``root/motion.json`` beside a
    "fingerprint" key (caches.MOTION), which is checked on loading and not returned."""
    path = root / MOTION_NAME
    expected = caches.fingerprint(caches.MOTION, floor)
    if path.exists():
        motion = json.loads(path.read_text())
        caches.check(path, motion.pop("fingerprint", None), expected)
        return motion
    motion = mapping.calibrate(routes, camera, floor)
    saved = {**motion, "fingerprint": expected}
    path.write_text(json.dumps(saved, default=lambda v: np.asarray(v).tolist()))
    return motion


def arc_motion(
    motion: dict[str, dict], measured: dict[str, dict] | None = None
) -> tuple[dict[str, dict], dict[str, float]]:
    """The motion of the arcs (FORWARD with a turn, one tick), by action name: the FORWARD and
    turn means summed. A ``measured`` arc (``{name: {"dx", "dy", "dyaw"}}``) replaces it where its
    dx or dyaw is off the sum by more than ARC_TOLERANCE. Also each arc's worst relative
    difference between the two (NaN without a measurement)."""
    arcs, off = {}, {}
    for turn, name in ARCS.items():
        if "FORWARD" not in motion or turn not in motion:
            continue
        f, t = motion["FORWARD"], motion[turn]
        summed = {k: f[k] + t[k] for k in ("dx", "dy", "dyaw")}
        off[name] = math.nan
        arcs[name] = summed
        if measured and name in measured:
            m = measured[name]
            off[name] = max(abs(m[k] - summed[k]) / abs(summed[k]) for k in ("dx", "dyaw"))
            if off[name] > ARC_TOLERANCE:
                arcs[name] = {k: float(m[k]) for k in ("dx", "dy", "dyaw")}
    return arcs, off


class MapAgent(Agent):
    def __init__(
        self,
        data_dir: str | Path,
        *,
        goal_override: Pose | None = None,
        lookahead_m: float = LOOKAHEAD_M,
        turn_tol: float = TURN_TOL_DEG,
        arc_deg: float = ARC_DEG,
        arcs: bool = False,
        max_turn_ticks: int | None = None,
        shortcut: bool = False,
        corridor_m: float = planner_module.CORRIDOR_M,
        link_m: float = planner_module.LINK_M,
        arc_measured: dict[str, dict] | None = None,
        forward_ticks: int = FORWARD_TICKS,
        guard_m: float = GUARD_M,
        near_m: float = NEAR_M,
        align_deg: float = ALIGN_DEG,
        offset_m: float = OFFSET_M,
        homing_max: int = HOMING_MAX,
        fixes: bool = True,
        align_source: str = "estimate",
        goal_camera: str = "auto",
        floor_guard: bool = False,
        floor_guard_frac: float = FLOOR_GUARD_FRAC,
        blocked_zero: bool = False,
        blocked_diff: float = BLOCKED_DIFF,
        online_motion: bool = True,
        rebuild: bool = False,
    ) -> None:
        """``goal_override``: a goal pose in the map frame, used instead of ``find_goal``.
        ``arcs``: steer with arcs; ``max_turn_ticks`` is then ARC_MAX_TURN_TICKS unless given.
        ``shortcut``: shortcut every plan, within ``corridor_m`` of the mapped frames.
        ``link_m``: the planner's link radius (metres) between mapped poses.
        ``arc_measured``: measured arc motion for ``arc_motion``.
        ``fixes``: True (the default) turns the final alignment on (the 5 degree estimate
        alignment, the 15 inlier view gate, the turn after a homing timeout).
        ``align_source``: "estimate" (the default) turns by the estimate's yaw error only (the
        view is still matched and logged); "view" turns by the goal view's yaw when one matches.
        In "estimate" mode the goal views are still matched every homing tick, and a match
        changes one thing: the alignment tolerance widens from ESTIMATE_ALIGN_DEG (5 degrees) to
        ``align_deg`` (8).
        ``goal_camera``: the camera the goal photos are read with: "frame" (the exploration
        frames' own camera), "photo" (the lower goal.TARGET_CAMERA_HEIGHT camera, which
        targets() renders with), or "auto": "photo", on local and server challenges alike.
        ``online_motion``: the fallback motion and turn rate from the run's own visual odometry
        (on by default; see the module docstring).
        ``floor_guard``, ``blocked_zero``: the options of the module docstring, each off by
        default; ``floor_guard_frac`` and ``blocked_diff`` are their thresholds.
        ``rebuild``: make the three caches (map, motion, links) again rather than load them."""
        if goal_camera not in GOAL_CAMERAS:
            raise ValueError(f"goal_camera must be one of {GOAL_CAMERAS}, not {goal_camera!r}")
        self.goal_camera = goal_camera
        self.floor_guard, self.floor_guard_frac = floor_guard, floor_guard_frac
        self.blocked_zero = blocked_diff if blocked_zero else None
        self.online_motion = online_motion
        self.root = Path(data_dir)
        self.goal_override = goal_override
        self.lookahead_m, self.turn_tol, self.forward_ticks = lookahead_m, turn_tol, forward_ticks
        self.arc_deg, self.arcs = arc_deg, arcs
        if max_turn_ticks is None:
            max_turn_ticks = ARC_MAX_TURN_TICKS if arcs else MAX_TURN_TICKS
        self.max_turn_ticks = max_turn_ticks
        self.corridor_m = corridor_m if shortcut else None
        self.guard_m, self.near_m, self.align_deg = guard_m, near_m, align_deg
        self.offset_m, self.homing_max, self.fixes = offset_m, homing_max, fixes
        if align_source not in ALIGN_SOURCES:
            raise ValueError(f"align_source {align_source!r} not in {ALIGN_SOURCES}")
        self.align_source = align_source

        self.floor = FloorMask.load(FLOOR_MASK_PATH)
        routes = mapping.load_routes(self.root)
        shape = routes[0].frames[0].shape[:2]
        size = (geometry.CAMERA["height"], geometry.CAMERA["width"])
        assert shape == size, f"exploration frames are {shape} (rows, cols), CAMERA is {size}"
        self.camera_dict = dict(geometry.CAMERA)
        self.camera = Camera(
            {**self.camera_dict, "cy": self.camera_dict["cy"] + mapping.HORIZON_SHIFT_PX}
        )
        self.timings: dict[str, float] = {}
        """Seconds of each construction stage: map, motion, planner."""
        if rebuild:
            for name in (MOTION_NAME, LINKS_NAME):
                (self.root / name).unlink(missing_ok=True)
        t = time.perf_counter()
        self.store, self.routes = frozen_map.get(
            self.root, self.camera_dict, self.floor, rebuild=rebuild
        )
        self.timings["map load/build"] = time.perf_counter() - t
        t = time.perf_counter()
        self.motion = load_motion(self.root, self.routes, self.camera, self.floor)
        self.timings["motion calibration"] = time.perf_counter() - t
        t = time.perf_counter()
        self.planner = planner_module.PlaceGraph(
            self.store, link_m=link_m, cache_path=self.root / LINKS_NAME
        )
        self.timings["planner"] = time.perf_counter() - t
        turns = [
            abs(math.degrees(self.motion[a]["dyaw"])) for a in ("LEFT", "RIGHT") if a in self.motion
        ]
        self.turn_rate = float(np.mean(turns)) if turns else DEFAULT_TURN_DEG
        """Degrees one LEFT or RIGHT tick turns, from the exploration data."""
        arcs, self.arc_off = arc_motion(self.motion, arc_measured)
        self.motion = {**self.motion, **arcs}
        arc = arcs.get("FORWARD|LEFT")
        self.arc_radius = math.inf if arc is None else abs(arc["dx"] / arc["dyaw"])
        """Metres: the radius of the circle an arc drives."""
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
        self.timed_out = False
        """Homing: --homing-max decisions made; turning by the estimate to check in."""
        self.heading_error: float | None = None
        """Degrees to turn left to the goal's yaw, by the estimate, at check-in."""
        self.approaching = False
        """Homing: driving to the goal sideways-on, not holding the alignment."""
        self.guard_events = 0
        self.floor_guard_events = 0
        """Guard events caused by the floor fraction alone (counted in ``guard_events`` too)."""
        self.floor_frac = math.nan
        """The floor fraction of the latest frame (NaN: --floor-guard is off)."""
        self.replan_pending = False
        self.guarded = False
        """The wall guard fired on this tick."""
        self.guard_cause: str | None = None
        """"centre" (the centre range) or "floor" (the floor fraction alone) on a guard tick."""
        self.view: tuple[int, float] | None = None
        """Homing: the goal view the frame matched and the yaw to view 0 it gave."""
        self.result: Result | None = None

    # -- the session

    @staticmethod
    def check_camera(camera) -> None:
        """Warn (never fail) if the session's camera differs from geometry.CAMERA."""
        if camera is None:
            return
        seen = {k: getattr(camera, k) for k in geometry.CAMERA}
        off = {k: (seen[k], v) for k, v in geometry.CAMERA.items() if seen[k] != v}
        if off:
            print(
                "WARNING: the session's camera differs from geometry.CAMERA (session, expected): "
                + ", ".join(f"{k} {a} vs {b}" for k, (a, b) in off.items()),
                flush=True,
            )

    def setup(self, info: SessionInfo) -> None:
        self.reset()
        self.check_camera(getattr(info, "camera", None))
        self.localiser = Localiser(
            self.store,
            self.camera,
            self.floor,
            motion=self.motion,
            blocked_zero=self.blocked_zero,
            online_motion=self.online_motion,
        )
        if self.goal_override is not None:
            self.goal = tuple(float(v) for v in self.goal_override)
        else:
            mode = "photo" if self.goal_camera == "auto" else self.goal_camera
            found = find_goal(
                info.targets, self.store, self.camera, self.floor, photographed=mode == "photo"
            )
            self.goal = None if found is None else (found.x, found.y, found.yaw)
            pose = (
                "none"
                if found is None
                else f"({found.x:.2f}, {found.y:.2f}) {math.degrees(found.yaw):.0f} deg"
            )
            print(f"goal camera: {mode}; goal pose: {pose}", flush=True)
        self.found_goal = self.goal
        if self.goal is not None:
            self.navigator = navigator_module.Navigator(
                self.planner,
                self.localiser,
                self.goal,
                lookahead_m=self.lookahead_m,
                store=self.store,
                corridor_m=self.corridor_m,
            )
        min_matches = SERVO_MIN_MATCHES if self.fixes else LEGACY_MIN_MATCHES
        self.servo = Servo(self.camera.fx, cx=self.camera.cx, min_matches=min_matches)
        self.goal_features = [self.servo.features(t) for t in info.targets]

    def finish(self, result: Result | None) -> None:
        self.result = result

    def act(self, obs: Observation) -> tuple[Action, int]:
        self.guarded = False
        self.guard_cause = None
        self.view = None
        if self.floor_guard:
            self.floor_frac = floor_fraction(self.floor(obs.image), self.camera)
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
        close = self.floor_guard and self.floor_frac < self.floor_guard_frac
        if name.startswith("FORWARD") and (self._centre_range() < self.guard_m or close):
            self.guard_events += 1
            self.guard_cause = "centre" if self._centre_range() < self.guard_m else "floor"
            self.floor_guard_events += int(self._centre_range() >= self.guard_m)
            self.guarded = True
            self.mode, self.replan_pending = "guard", True
            self.queue = ["BACKWARD"] * (GUARD_TICKS - 1)
            name = "BACKWARD"
        self.last = name
        action = Action(0)
        for part in name.split("|"):
            action |= getattr(Action, part)
        return action, 1

    def _checkin(self, reason: str) -> tuple[Action, int]:
        self.checkin_reason, self.last, self.queue = reason, None, []
        estimate = self.update.estimate
        self.heading_error = math.degrees(wrap(self.goal[2] - estimate.yaw)) if self.goal else None
        outcome = {"homed": "aligned", "homing_max": "timed out"}.get(reason, reason)
        if self.homing and reason == "low_steps":
            outcome = "out of steps"
        elif not self.homing:
            outcome = "not homing"
        error = "n/a" if self.heading_error is None else f"{self.heading_error:+.1f} deg"
        print(
            f"check-in: {reason} ({outcome}, {self.homing_steps} homing decisions), "
            f"heading error {error}"
        )
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
            self.localiser.focus = (gx, gy)
        if self.homing:
            return "home", self._home(obs)
        waypoint = self.update.waypoint
        if waypoint is None:
            return "search", ["LEFT"]
        error = math.degrees(
            wrap(math.atan2(waypoint[1] - estimate.y, waypoint[0] - estimate.x) - estimate.yaw)
        )
        distance = math.hypot(waypoint[0] - estimate.x, waypoint[1] - estimate.y)
        return "nav", self._steer(error, distance)

    def _steer(self, error: float, distance: float = math.inf) -> list[str]:
        """Drive at a waypoint ``distance`` metres away and ``error`` degrees to the left, in an
        arc or after turning."""
        if abs(error) <= self.turn_tol:
            return ["FORWARD"] * self.forward_ticks
        reachable = distance >= 2 * self.arc_radius * math.sin(math.radians(abs(error)))
        if self.arcs and abs(error) <= self.arc_deg and reachable:
            return [ARCS["LEFT" if error > 0 else "RIGHT"]]
        return self._turn(error, self.max_turn_ticks)

    def _turn(self, error: float, most: int = HOMING_TURN_TICKS) -> list[str]:
        rate = (self.localiser.turn_rate() if self.online_motion else None) or self.turn_rate
        ticks = int(np.clip(round(abs(error) / rate), 1, most))
        return ["LEFT" if error > 0 else "RIGHT"] * ticks

    def _home(self, obs: Observation) -> list[str]:
        self.homing_steps += 1
        if self.homing_steps > self.homing_max:
            if not self.fixes:
                self.checkin_reason = "homing_max"
                return ["CHECKIN"]
            self.timed_out = True
        if self.timed_out:
            yaw = math.degrees(wrap(self.goal[2] - self.update.estimate.yaw))
            if abs(yaw) < ESTIMATE_ALIGN_DEG:
                self.checkin_reason = "homing_max"
                return ["CHECKIN"]
            return self._turn(yaw)
        yaw = self._yaw_to_view0(obs.image)
        if self.align_source == "estimate":
            yaw = math.degrees(wrap(self.goal[2] - self.update.estimate.yaw))
        # Without a view the estimate alone aligns, to ESTIMATE_ALIGN_DEG.
        align_deg = self.align_deg if self.view or not self.fixes else ESTIMATE_ALIGN_DEG
        x, y = between(self.update.estimate[:3], self.goal)[:2]
        distance = math.hypot(x, y)
        if self.approaching:
            if distance > self.offset_m:
                return self._approach(x, y)
            self.approaching = False
        if abs(yaw) >= align_deg:
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

    def _yaw_to_view0(self, image: np.ndarray) -> float:
        """Degrees to turn left to face view 0. The estimate gives it; the goal view the frame
        matches best (SIFT, servo.py, SERVO_MIN_MATCHES inliers) overrides it, if within
        VIEW_GATE_DEG of the estimate's: look-alikes match too."""
        estimate = self.update.estimate
        error = math.degrees(wrap(self.goal[2] - estimate.yaw))
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
            return error
        self.view = best[1], best[2]
        return best[2]

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


def local_data_dir(args) -> Path:
    """The exploration data for ``--test-local``, never downloaded: ``--data``, else
    ``data/<challenge>`` (or the local maze, recorded offline on first use)."""
    if args.data:
        return Path(args.data)
    if args.local is not None:
        return cli.exploration_data(args, None)
    path = cli.ROOT / "data" / str(args.challenge)
    if not args.challenge or not (path / "target.jpg").exists():
        raise SystemExit(f"--test-local needs --data, or downloaded data in {path}")
    return path


def goal_views(root: Path) -> list[np.ndarray]:
    """The four goal views of ``root/target.jpg`` (side by side), GOAL_VIEW_PX wide each."""
    target = cv2.imread(str(root / "target.jpg"))
    if target is None:
        raise SystemExit(f"cannot read {root / 'target.jpg'}")
    width = target.shape[1] // 4
    views = [target[:, k * width : (k + 1) * width] for k in range(4)]
    views = [
        cv2.resize(v, (GOAL_VIEW_PX, v.shape[0])) if v.shape[1] != GOAL_VIEW_PX else v
        for v in views
    ]
    return views


def report_local(agent: MapAgent, root: Path) -> None:
    """Print what ``MapAgent`` built and what ``find_goal`` makes of ``root/target.jpg`` (four
    views side by side), touching no server and creating no session."""
    views = goal_views(root)
    mode = "photo" if agent.goal_camera == "auto" else agent.goal_camera
    t = time.perf_counter()
    found = find_goal(views, agent.store, agent.camera, agent.floor, photographed=mode == "photo")
    timings = {**agent.timings, "find_goal": time.perf_counter() - t}
    print("stage wall time:")
    for name, seconds in timings.items():
        print(f"  {name}: {seconds:.2f} s")
    print("motion per action (dx m, dy m, dyaw deg):")
    for name, m in agent.motion.items():
        print(f"  {name}: {m['dx']:.4f} {m['dy']:.4f} {math.degrees(m['dyaw']):.3f}")
    print("frames per route: " + ", ".join(str(len(r.frames)) for r in agent.routes))
    pose = (
        "none"
        if found is None
        else f"({found.x:.2f}, {found.y:.2f}) {math.degrees(found.yaw):.0f} deg"
    )
    print(f"goal camera: {mode}; goal pose: {pose}")


def agent_options(args) -> dict:
    return {
        "lookahead_m": args.lookahead_m,
        "turn_tol": args.turn_tol,
        "arc_deg": args.arc_deg,
        "arcs": args.arcs,
        "max_turn_ticks": args.max_turn_ticks,
        "shortcut": args.shortcut,
        "corridor_m": args.corridor_m,
        "link_m": args.link_m,
        "forward_ticks": args.forward_ticks,
        "guard_m": args.guard_m,
        "near_m": args.near_m,
        "align_deg": args.align_deg,
        "offset_m": args.offset_m,
        "homing_max": args.homing_max,
        "fixes": not args.no_fixes,
        "align_source": args.align_source,
        "goal_camera": args.goal_camera,
        "floor_guard": args.floor_guard,
        "floor_guard_frac": args.floor_guard_frac,
        "blocked_zero": args.blocked_zero,
        "blocked_diff": args.blocked_diff,
        "online_motion": not args.no_online_motion,
        "rebuild": args.rebuild,
    }


def main(argv: list[str] | None = None) -> None:
    parser = cli.parser(__doc__)
    parser.add_argument(
        "--test-local",
        action="store_true",
        help="build or load everything offline, print timings, and exit before any server call",
    )
    parser.add_argument("--data", default=None, help="exploration data directory")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="make the map, motion and links caches again (they warn when stale)",
    )
    parser.add_argument("--lookahead-m", type=float, default=LOOKAHEAD_M)
    parser.add_argument("--turn-tol", type=float, default=TURN_TOL_DEG, help="degrees")
    parser.add_argument("--arc-deg", type=float, default=ARC_DEG, help="degrees")
    parser.add_argument("--arcs", action="store_true", help="steer with FORWARD+turn arcs")
    parser.add_argument(
        "--max-turn-ticks",
        type=int,
        default=None,
        help=f"{MAX_TURN_TICKS}; {ARC_MAX_TURN_TICKS} with --arcs",
    )
    parser.add_argument("--shortcut", action="store_true", help="string-pull every plan")
    parser.add_argument("--corridor-m", type=float, default=planner_module.CORRIDOR_M)
    parser.add_argument("--link-m", type=float, default=planner_module.LINK_M)
    parser.add_argument("--forward-ticks", type=int, default=FORWARD_TICKS)
    parser.add_argument("--guard-m", type=float, default=GUARD_M)
    parser.add_argument("--near-m", type=float, default=NEAR_M)
    parser.add_argument("--align-deg", type=float, default=ALIGN_DEG)
    parser.add_argument("--offset-m", type=float, default=OFFSET_M)
    parser.add_argument("--homing-max", type=int, default=HOMING_MAX)
    parser.add_argument("--floor-guard", action="store_true", help="wall guard from the floor")
    parser.add_argument("--floor-guard-frac", type=float, default=FLOOR_GUARD_FRAC)
    parser.add_argument("--blocked-zero", action="store_true", help="zero motion when blocked")
    parser.add_argument("--blocked-diff", type=float, default=BLOCKED_DIFF)
    parser.add_argument(
        "--no-online-motion",
        action="store_true",
        help="calibrated fallback motion instead of the medians of this run's VO",
    )
    parser.add_argument(
        "--align-source",
        choices=ALIGN_SOURCES,
        default="estimate",
        help="yaw homing turns by: the estimate's error (the view is still matched and logged), "
        "or the matched goal view's",
    )
    parser.add_argument("--no-fixes", action="store_true", help="the final alignment off")
    parser.add_argument(
        "--goal-camera",
        choices=GOAL_CAMERAS,
        default="auto",
        help="camera the goal photos are read with; auto: photo (frame is explicit)",
    )
    args = parser.parse_args(argv)
    if args.test_local:
        root = local_data_dir(args)
        report_local(MapAgent(root, **agent_options(args)), root)
        return

    root = cli.exploration_data(args, args.data)
    agent = MapAgent(root, **agent_options(args))
    result = run(agent, cli.challenge(args), **cli.run_options(args))
    print(f"result: {result}")


if __name__ == "__main__":
    main()
