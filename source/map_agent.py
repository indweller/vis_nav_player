"""
Map-based agent: follows a path on the frozen map to the goal, then homes in on the goal views.

At construction it builds (or loads) the frozen map of the exploration data (``map_graph.npz``
and the planner's ``links.npz`` cached in the data directory, the motion calibration in
``motion.json``). Each cache keeps a fingerprint of the settings it was made with and warns when
stale (caches.py); --rebuild makes all three again. At setup the goal is ``goal.find_goal`` on
the four target views (or the ``goal_override`` pose, for evaluations), the localiser starts at
the map's origin (the spawn), and a navigator plans to the goal.

The goal photos are read with goal.TARGET_CAMERA_HEIGHT's lower camera (``photographed``). The
tuning parameters are a ``Config``, each one a flag of the same name (--lookahead-m and so on).

Each tick the navigator localises (given the move just made), replans if it must, and the agent
aims at the point --lookahead-m ahead on the path. Moves are held as queues of single ticks, so
the localiser sees every frame:
  - heading error to the waypoint at most --turn-tol: FORWARD for --forward-ticks; over it: turn
    toward it for clip(round(error / turn rate), 1, --max-turn-ticks) ticks;
  - wall guard: a FORWARD with the centre column's wall range under --guard-m becomes BACKWARD
    for 2 ticks, and the path is replanned;
  - the localiser's fallback motion and the turn rate are the medians of the run's own visual
    odometry steps per action, once there are 20;
  - homing, once the estimate is within --near-m of the goal (and from then on): the localiser
    tries a fix every tick against the mapped frames within 0.5 m of the goal (its SIFT and layout
    checks). The target is the goal pose. The turn is the estimate's yaw error (against the goal's
    from find_goal). The frame is also matched against the four goal views by SIFT (servo.py,
    at least 15 inliers, within 30 degrees of the estimate's yaw; the views match only within
    about 0.1 m of the goal), and the match (kept in ``view``) changes one thing: turn until the
    yaw error is under --align-deg with a match, under 5 degrees without. Then, if the estimate
    is over --offset-m from the goal, step toward it, a tick at a time (along the heading if it is
    off that way only; else turn to it and drive, aligning again once there); CHECKIN when
    aligned and within --offset-m. After --homing-max homing decisions the agent turns by the
    estimate alone to under 5 degrees and then checks in, wherever it is.
  - CHECKIN when fewer than 5 steps are left (the only check-in before alignment).
  - At check-in the homing outcome and the heading error (the estimate's, against the goal's yaw)
    are printed.

    uv run source/map_agent.py --local 7
    uv run source/map_agent.py --challenge <id>
    uv run source/map_agent.py --test-local --data data/<id>   # offline check, no session
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, fields
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
from geometry import Camera, FloorMask
from goal import find_goal
from localiser import Localiser
from mapping import between, wrap
from servo import Servo
from vis_nav_sdk import Action, Agent, Observation, Result, SessionInfo, run

FLOOR_MASK_PATH = geometry.FLOOR_MASK_PATH
MOTION_NAME = "motion.json"
LINKS_NAME = "links.npz"
DEFAULT_TURN_DEG = 3.4375

HOMING_TURN_TICKS = 3
"""Most ticks of a turn in place while homing."""
GUARD_TICKS = 2
VIEW_GATE_DEG = 30.0
"""Homing believes a goal view's yaw only within this of the estimate's."""
ESTIMATE_ALIGN_DEG = 5.0
"""Without a goal view match, the estimate's yaw error must be under this to check in."""
SERVO_MIN_MATCHES = 15
"""Inliers the servo needs between the frame and a goal view (the views match only near)."""
LOW_STEPS = 5
"""CHECKIN when fewer steps than this are left."""

GOAL_VIEW_PX = 320
"""Width of each of the four goal views split from target.jpg."""

Pose = tuple[float, float, float]


@dataclass(frozen=True)
class Config:
    """MapAgent's tuning parameters; each is a command-line flag of the same name."""

    lookahead_m: float = 0.15
    """Metres ahead on the path of the point the agent aims at."""
    turn_tol: float = 12.0
    """Degrees of heading error to the waypoint up to which the agent drives rather than turns."""
    max_turn_ticks: int = 3
    """Most ticks of a turn in place toward the waypoint."""
    forward_ticks: int = 2
    """FORWARD ticks per drive decision."""
    link_m: float = planner_module.LINK_M
    """The planner's link radius (metres) between mapped poses."""
    guard_m: float = 0.07
    """The wall guard's range: metres to the wall in the centre column."""
    near_m: float = 0.15
    """Metres from the goal at which homing starts."""
    align_deg: float = 8.0
    """Homing's yaw tolerance, degrees, when a goal view matches (else ESTIMATE_ALIGN_DEG)."""
    offset_m: float = 0.05
    """Homing's distance tolerance, metres."""
    homing_max: int = 60
    """Homing decisions before the agent turns by the estimate alone and checks in."""


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """A flag per ``Config`` field (--lookahead-m for ``lookahead_m``), its default the field's."""
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        parser.add_argument(flag, type=type(f.default), default=f.default)


def config_from(args: argparse.Namespace) -> Config:
    """The ``Config`` of the flags ``add_config_arguments`` added."""
    return Config(**{f.name: getattr(args, f.name) for f in fields(Config)})


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


class MapAgent(Agent):
    def __init__(
        self,
        data_dir: str | Path,
        *,
        config: Config | None = None,
        goal_override: Pose | None = None,
        rebuild: bool = False,
    ) -> None:
        """``config``: the tuning parameters (``Config()`` if None).
        ``goal_override``: a goal pose in the map frame, used instead of ``find_goal``.
        ``rebuild``: make the three caches (map, motion, links) again rather than load them."""
        self.config = config or Config()
        self.root = Path(data_dir)
        self.goal_override = goal_override

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
            self.store, link_m=self.config.link_m, cache_path=self.root / LINKS_NAME
        )
        self.timings["planner"] = time.perf_counter() - t
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
        self.timed_out = False
        """Homing: --homing-max decisions made; turning by the estimate to check in."""
        self.heading_error: float | None = None
        """Degrees to turn left to the goal's yaw, by the estimate, at check-in."""
        self.approaching = False
        """Homing: driving to the goal sideways-on, not holding the alignment."""
        self.guard_events = 0
        self.replan_pending = False
        self.guarded = False
        """The wall guard fired on this tick."""
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
            online_motion=True,
        )
        if self.goal_override is not None:
            self.goal = tuple(float(v) for v in self.goal_override)
        else:
            found = find_goal(info.targets, self.store, self.camera, self.floor, photographed=True)
            self.goal = None if found is None else (found.x, found.y, found.yaw)
            print(f"goal pose: {pose_text(found)}", flush=True)
        self.found_goal = self.goal
        if self.goal is not None:
            self.navigator = navigator_module.Navigator(
                self.planner, self.localiser, self.goal, lookahead_m=self.config.lookahead_m
            )
        self.servo = Servo(self.camera.fx, cx=self.camera.cx, min_matches=SERVO_MIN_MATCHES)
        self.goal_features = [self.servo.features(t) for t in info.targets]

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
        if name == "FORWARD" and self._centre_range() < self.config.guard_m:
            self.guard_events += 1
            self.guarded = True
            self.mode, self.replan_pending = "guard", True
            self.queue = ["BACKWARD"] * (GUARD_TICKS - 1)
            name = "BACKWARD"
        self.last = name
        return getattr(Action, name), 1

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
        if not self.homing and math.hypot(estimate.x - gx, estimate.y - gy) <= self.config.near_m:
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
        return "nav", self._steer(error)

    def _steer(self, error: float) -> list[str]:
        """Drive at a waypoint ``error`` degrees to the left, after turning to it if need be."""
        if abs(error) <= self.config.turn_tol:
            return ["FORWARD"] * self.config.forward_ticks
        return self._turn(error, self.config.max_turn_ticks)

    def _turn(self, error: float, most: int = HOMING_TURN_TICKS) -> list[str]:
        rate = self.localiser.turn_rate() or self.turn_rate
        ticks = int(np.clip(round(abs(error) / rate), 1, most))
        return ["LEFT" if error > 0 else "RIGHT"] * ticks

    def _home(self, obs: Observation) -> list[str]:
        self.homing_steps += 1
        if self.homing_steps > self.config.homing_max:
            self.timed_out = True
        if self.timed_out:
            yaw = math.degrees(wrap(self.goal[2] - self.update.estimate.yaw))
            if abs(yaw) < ESTIMATE_ALIGN_DEG:
                self.checkin_reason = "homing_max"
                return ["CHECKIN"]
            return self._turn(yaw)
        self._yaw_to_view0(obs.image)  # sets ``view``; the turn is by the estimate regardless
        yaw = math.degrees(wrap(self.goal[2] - self.update.estimate.yaw))
        # Without a view the estimate alone aligns, to ESTIMATE_ALIGN_DEG.
        align_deg = self.config.align_deg if self.view else ESTIMATE_ALIGN_DEG
        x, y = between(self.update.estimate[:3], self.goal)[:2]
        distance = math.hypot(x, y)
        offset_m = self.config.offset_m
        if self.approaching:
            if distance > offset_m:
                return self._approach(x, y)
            self.approaching = False
        if abs(yaw) >= align_deg:
            return self._turn(yaw)
        if distance > offset_m:
            if abs(y) <= offset_m / 2:
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
        turn_tol = self.config.turn_tol
        if abs(angle) <= turn_tol:
            return ["FORWARD"]
        if abs(angle) >= 180 - turn_tol:
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
    t = time.perf_counter()
    found = find_goal(views, agent.store, agent.camera, agent.floor, photographed=True)
    timings = {**agent.timings, "find_goal": time.perf_counter() - t}
    print("stage wall time:")
    for name, seconds in timings.items():
        print(f"  {name}: {seconds:.2f} s")
    print("motion per action (dx m, dy m, dyaw deg):")
    for name, m in agent.motion.items():
        print(f"  {name}: {m['dx']:.4f} {m['dy']:.4f} {math.degrees(m['dyaw']):.3f}")
    print("frames per route: " + ", ".join(str(len(r.frames)) for r in agent.routes))
    print(f"goal pose: {pose_text(found)}")


def pose_text(found) -> str:
    """``find_goal``'s pose for printing ("none" if it found none)."""
    if found is None:
        return "none"
    return f"({found.x:.2f}, {found.y:.2f}) {math.degrees(found.yaw):.0f} deg"


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
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    if args.test_local:
        root = local_data_dir(args)
        report_local(MapAgent(root, config=config_from(args), rebuild=args.rebuild), root)
        return

    root = cli.exploration_data(args, args.data)
    agent = MapAgent(root, config=config_from(args), rebuild=args.rebuild)
    result = run(agent, cli.challenge(args), **cli.run_options(args))
    print(f"result: {result}")


if __name__ == "__main__":
    main()
