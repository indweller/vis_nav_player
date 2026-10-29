"""
Autonomous baseline: takes the baseline's shortest-path move every step and checks in at the
goal node. With no path to the goal it turns; the visual-shortcut gap and count grow with the
index. Its node is a Bayes filter's estimate (node_filter.py); while the filter's confidence is
below --conf-threshold it keeps its last node and turns. --no-filter takes the raw best match.

Each trajectory edge replays the raw recorded actions between its two nodes (reversed when walking
the trajectory backwards), --capture-every ticks per record and at most --max-segment-ticks per
move. Visual edges on the path are skipped: the robot is taken to be at their far end. It
localises after every move and keeps the queued segment while the estimate is within one hop of
the edge's ends; otherwise it replans. When the hop count to the goal has not beaten its best in
--stall-window confident estimates, the path's first edge costs --stall-penalty times more.

With --servo (the default) it follows the edge closed-loop instead, against every raw record the
edge spans in data_info.json (reversed when walking backwards). Each confident step it locates
itself at the span's record with the most Servo inliers (servo.py), searching from the last one
located to LOCATE_WINDOW past it, and aims at the record --lookahead-records further on. At the
span's end it follows the path's next trajectory edge. While the yaw offset to the aim exceeds
--yaw-tol it turns toward it, else it drives the edge's recorded translation (FORWARD, or
BACKWARD walking a FORWARD edge backwards) --forward-ticks; an edge without one keeps turning.
Without a located record, or with too few matches to the aim, it replays the queued segment.
--no-servo always replays.

    uv run source/my_agent.py --challenge <id>      # on the server, one attempt
    uv run source/my_agent.py --local 7             # on your machine, maze 7, free
    uv run source/my_agent.py --local 7 --no-filter # raw argmax localisation
    uv run source/my_agent.py --local 7 --no-servo  # replay the recorded segments only
"""

from __future__ import annotations

import argparse
import cProfile
import itertools
import json
import pstats
import random
from collections import Counter, deque
from pathlib import Path

import cli
import networkx as nx
import numpy as np
from baseline_agent import (
    MIN_SHORTCUT_GAP,
    TEMPORAL_WEIGHT,
    VISUAL_WEIGHT_BASE,
    VISUAL_WEIGHT_SCALE,
    BaselineAgent,
    _distance,
)
from node_filter import NodeFilter
from servo import Servo
from vis_nav_sdk import Action, Observation, Result, SessionInfo, run
from vis_nav_sdk import protocol as P
from vlad import MOVEMENT, REVERSE, Frame, VLADExtractor, load_frames

Record = tuple[int, int]
"""``(trajectory, record index in its data_info.json)``."""

FALLBACK = "LEFT"
LOW_CONF_MOVE = "LEFT"
"""What MyAgent does while the filter is unsure where it is: turn and look around."""
CONF_THRESHOLD = 0.5
STALL_WINDOW = 30
"""Confident estimates without a new best hop count to the goal before a stall."""
STALL_PENALTY = 5.0
"""A stall multiplies the weight of the path's first edge by this."""
MAX_SEGMENT_TICKS = 15
SHORTCUT_CHUNK = 1024
SHORTCUT_GAP_FRAC = 0.02
"""Shortcuts join nodes at least this fraction of the index apart (and MIN_SHORTCUT_GAP)."""
SHORTCUTS_PER_1K = 10
"""Shortcut edges per 1000 nodes (at least top_k_shortcuts)."""
LOOKAHEAD_RECORDS = 1
"""Raw records from the located one to the one the servo aims at."""
LOCATE_WINDOW = 4
"""Records past the last located one the servo searches."""
YAW_TOL = 8.0
"""Degrees of yaw offset the servo drives forward through rather than turning."""
FORWARD_TICKS = 5
TURN_DEG_PER_TICK = 3.4375
"""Degrees one LEFT or RIGHT tick turns, measured by eval/common.py's turn_deg_per_tick()."""
GOAL_VIEWS = 4
CHECKIN_VIEWS = 3
CHECKIN_YAW = 5.0
CHECKIN_INLIERS = 15
STUCK_FORWARDS = 3
STUCK_BACKWARD_TICKS = 2
LOST_STEPS = 5
OSCILLATION_WINDOW = 6
OSCILLATION_FLIPS = 3
MAX_SPINS = 2
PROFILE_STEPS = 1000


class Float32VLAD(VLADExtractor):
    """VLADExtractor whose database and query vectors are float32: half the memory, and the
    similarity products run in single precision."""

    def _vlad(self, des: np.ndarray | None) -> np.ndarray:
        return super()._vlad(des).astype(np.float32)


class MyAgent(BaselineAgent):
    def __init__(
        self,
        data_dir: Path,
        *,
        ticks: int = 5,
        random_every: int = 10,
        n_clusters: int = 128,
        subsample: int = 5,
        top_k_shortcuts: int = 30,
        shortcut_gap_frac: float = SHORTCUT_GAP_FRAC,
        shortcuts_per_1k: float = SHORTCUTS_PER_1K,
        cache_dir: Path = Path("cache"),
        filter: bool = True,
        hops: int = 2,
        p_stay: float = 0.5,
        action_boost: float = 3.0,
        eps: float = 1e-3,
        tau: float = 0.02,
        conf_threshold: float = CONF_THRESHOLD,
        verify_candidates: int = 0,
        geom_weight: float = 0.001,
        capture_every: int = 1,
        max_segment_ticks: int = MAX_SEGMENT_TICKS,
        stall_window: int = STALL_WINDOW,
        stall_penalty: float = STALL_PENALTY,
        servo: bool = True,
        lookahead_records: int = LOOKAHEAD_RECORDS,
        yaw_tol: float = YAW_TOL,
        forward_ticks: int = FORWARD_TICKS,
        turn_deg_per_tick: float = TURN_DEG_PER_TICK,
        goal_views: int = GOAL_VIEWS,
        checkin_views: int = CHECKIN_VIEWS,
        checkin_yaw: float = CHECKIN_YAW,
    ) -> None:
        # BaselineAgent.__init__, with a float32 extractor.
        self.frames, bounds = load_frames(data_dir, subsample)
        print(f"{len(self.frames)} frames from {len(bounds)} trajectories")
        self.raw_index, self.segments = _segments(self.frames, bounds, subsample)
        self.capture_every = capture_every
        self.max_segment_ticks = max_segment_ticks
        self.stall_window = stall_window
        self.stall_penalty = stall_penalty
        self.use_servo = servo
        self.lookahead_records = lookahead_records
        self.record_paths = _record_paths(self.frames)
        """Every raw record's image per trajectory, by record index."""
        self.yaw_tol = yaw_tol
        self.forward_ticks = forward_ticks
        self.turn_deg_per_tick = turn_deg_per_tick
        self.goal_views = goal_views
        self.checkin_views = checkin_views
        self.checkin_yaw = checkin_yaw
        self.goal_features: tuple[np.ndarray, np.ndarray | None] | None = None
        self.verified = 0
        self.stuck_count = 0
        self.stuck = False
        self.lost_count = 0
        self.spin: list[tuple[str, int]] = []
        self.turn_signs: deque[int] = deque(maxlen=OSCILLATION_WINDOW)
        self.spin_count = 0
        self._previous_located: Record | None = None
        self.servo: Servo | None = None
        """Built by setup(), which knows the camera."""
        self.extractor = Float32VLAD(n_clusters, cache_dir)
        self.database = self.extractor.fit(self.frames)
        self.shortcut_gap_frac = shortcut_gap_frac
        self.shortcuts_per_1k = shortcuts_per_1k
        self.graph = self._build_graph(bounds, top_k_shortcuts)
        self.filter = (
            NodeFilter(
                self.graph,
                self.frames,
                hops=hops,
                p_stay=p_stay,
                action_boost=action_boost,
                eps=eps,
                tau=tau,
            )
            if filter
            else None
        )
        self.conf_threshold = conf_threshold
        self.verify_candidates = verify_candidates
        self.geom_weight = geom_weight

        self.goal: int | None = None
        self.current: int | None = None
        self.match_distance = 0.0
        self.path: list[int] = []
        self._panel: list[tuple[np.ndarray, str, tuple[int, int, int]]] = []
        self._localised_step = -1

        self.ticks = ticks
        self.random_every = random_every
        self.moves = 0
        self.last = "-"
        self.no_path = False
        self.decisions = 0
        self.no_path_decisions = 0
        self.checkin_reason: str | None = None
        self.raw_node: int | None = None
        self.filter_node: int | None = None
        self.confidence = float("nan")
        self.low_conf = False
        self.low_conf_decisions = 0
        self.queue: list[tuple[str, int]] = []
        """Moves still to take on the current trajectory edge: ``(action, ticks)``."""
        self.source = "-"
        """Where the last move came from: servo-turn, servo-forward, segment, low-conf, no-path,
        random or hint."""
        self.estimate: int | None = None
        """The last confident node estimate (current also moves along visual edges)."""
        self.edge: tuple[int, int] | None = None
        """The trajectory edge the queue walks."""
        self.best_hops = float("inf")
        self.since_best = 0
        self.stalls: list[dict] = []
        """One ``{"step", "edge", "hops"}`` per stall, ``edge`` being the penalised one."""
        self._weights: dict[tuple[int, int], float] = {}
        """Penalised edges' original weights."""
        self.follow: tuple[int, int] | None = None
        """The trajectory edge the servo follows."""
        self.span: list[Record] = []
        """Its raw records, in walking order."""
        self.at = 0
        """The span position located last."""
        self.aimed = False
        """Whether the servo tried this step."""
        self.located: Record | None = None
        """The record the robot was located at this step (None: none reached min_matches)."""
        self.located_inliers = 0
        """Inliers against it, or the most any searched record had when none located."""
        self.target: Record | None = None
        """The record the servo aimed at this step."""
        self.target_inliers = 0
        self.yaw: float | None = None
        """Degrees to turn to face the target, positive to the right (None: too few matches)."""

    def setup(self, info: SessionInfo) -> None:
        super().setup(info)
        views = info.targets[: max(1, self.goal_views)]
        similarity = sum(self.database @ self.extractor.extract(view) for view in views)
        self.goal = int(np.argmax(similarity))
        print(f"goal: node {self.goal} from {len(views)} views")
        self.goal_features = None
        self.verified = self.stuck_count = self.lost_count = 0
        self.stuck = False
        self.spin = []
        self.turn_signs = deque(maxlen=OSCILLATION_WINDOW)
        self.spin_count = 0
        self._previous_located = None
        self.moves = 0
        self.last = "-"
        self.no_path = False
        self.decisions = 0
        self.no_path_decisions = 0
        self.checkin_reason = None
        self.raw_node = None
        self.filter_node = None
        self.confidence = float("nan")
        self.low_conf = False
        self.low_conf_decisions = 0
        self.queue = []
        self.source = "-"
        self.estimate = None
        self.edge = None
        self.best_hops = float("inf")
        self.since_best = 0
        self.stalls = []
        for (a, b), weight in self._weights.items():
            self.graph[a][b]["weight"] = weight
        self._weights = {}
        self.follow, self.span, self.at = None, [], 0
        self._clear_aim()
        if self.use_servo:
            camera = np.asarray(info.camera.intrinsic_matrix, float)
            fx, cx = float(camera[0, 0]), float(camera[0, 2])
            if self.servo is None:
                self.servo = Servo(
                    fx, cx=cx, cache_size=sum(len(paths) for paths in self.record_paths.values())
                )
            self.servo.fx, self.servo.cx = fx, cx
            self.goal_features = self.servo.features(info.targets[0])
        if self.filter is not None:
            self.filter.reset()

    def act(self, obs: Observation) -> Action | tuple[Action, int]:
        if obs.step != self._localised_step:
            # self.last is the move taken since the previous view ("-" or CHECKIN: none).
            if self.filter is not None and self.last in MOVEMENT:
                self.filter.predict(self.last)
            self._localise(obs.image)
            if not self.low_conf:
                if self.queue and not self._on_edge():
                    self.queue.clear()
                if not self.queue:
                    self._plan()
                self._guard(obs.step)
            self._aim(obs.image)
            self._verify(obs.image)
            self._recover(obs.step)
            self._localised_step = obs.step
        self.decisions += 1
        at_goal = (
            self.verified >= self.checkin_views
            if self.goal_features is not None
            else not self.low_conf and self.path == [self.goal]
        )
        if at_goal or obs.steps_left <= self.ticks:
            self.checkin_reason = "at_goal" if at_goal else "out_of_steps"
            print(f"check-in: {self.checkin_reason} (node {self.current}, step {obs.step})")
            self.last = "CHECKIN"
            return Action.CHECKIN
        self.moves += 1
        ticks = self.ticks
        if self.stuck:
            self.stuck = False
            self.last, ticks, self.source = "BACKWARD", STUCK_BACKWARD_TICKS, "stuck"
            ticks = min(ticks, obs.steps_left - 1)
        elif self.spin:
            self.queue.clear()
            self.last, ticks = self.spin.pop(0)
            self.source = "lost"
            ticks = min(ticks, obs.steps_left - 1)
        elif self.low_conf:
            self.queue.clear()
            self.low_conf_decisions += 1
            self.last, self.source = LOW_CONF_MOVE, "low-conf"
        elif self.no_path:
            self.queue.clear()
            self.no_path_decisions += 1
            self.last, self.source = FALLBACK, "no-path"
        elif self.random_every > 0 and self.moves % self.random_every == 0:
            self.queue.clear()
            self.last = random.choice(sorted(MOVEMENT - {self.last}))
            self.source = "random"
        elif self.yaw is not None and not self._oscillating(obs.step):
            assert self.follow is not None
            translation = self._translation(self.follow)
            if abs(self.yaw) > self.yaw_tol or translation is None:
                self.last = "RIGHT" if self.yaw > 0 else "LEFT"
                turn = round(abs(self.yaw) / self.turn_deg_per_tick)
                ticks = int(np.clip(turn, 1, self.max_segment_ticks))
                self.source = "servo-turn"
            else:
                self.last, ticks, self.source = translation, self.forward_ticks, "servo-forward"
            ticks = min(ticks, obs.steps_left - 1)
        elif self.queue:
            self.last, ticks = self.queue.pop(0)
            ticks = min(ticks, obs.steps_left - 1)
            self.source = "segment"
        else:
            hint = self._hint()
            self.last = hint if hint in MOVEMENT else FALLBACK
            self.source = "hint"
        return getattr(Action, self.last), ticks

    def finish(self, result: Result | None) -> None:
        print(f"result: {result}")

    def hud(self) -> list[str]:
        lines = [f"auto: {self.last}"] + BaselineAgent.hud(self)[1:]
        if self.filter is not None:
            low = "  (low: looking around)" if self.low_conf else ""
            lines.append(f"confidence {self.confidence:.2f} / {self.conf_threshold:.2f}{low}")
        if self.no_path:
            lines.append(f"no path from node {self.current} to the goal")
        if self.aimed:
            if self.located is None:
                lines.append(f"servo: lost ({self.located_inliers} inliers at best)")
            else:
                yaw = "-" if self.yaw is None else f"{self.yaw:+.1f} deg"
                lines.append(
                    f"servo: at {_name(self.located)} ({self.located_inliers}), aim "
                    f"{_name(self.target)} ({self.target_inliers}), yaw {yaw}"
                )
        if self.queue:
            lines.append(f"segment: {len(self.queue)} moves left")
        return lines

    def _localise(self, image: np.ndarray) -> None:
        # BaselineAgent._localise without the planning (_plan). With the filter, the node is
        # its estimate, and a low confidence keeps the previous node and path.
        similarity = self.database @ self.extractor.extract(image)
        self.raw_node = int(np.argmax(similarity))
        node = self.raw_node
        if self.filter is not None:
            observed = similarity
            if self.servo is not None and self.verify_candidates > 0:
                # Geometric verification: the top candidates gain geom_weight per inlier.
                observed = similarity.copy()
                for n in np.argsort(-similarity)[: self.verify_candidates]:
                    record = (self.frames[n].trajectory, self.raw_index[n])
                    yaw, inliers = self.servo.yaw_offset(image, self._record_image(record))
                    observed[n] += self.geom_weight * (0 if yaw is None else inliers)
            self.filter.update(observed)
            node, self.confidence = self.filter.estimate()
            self.filter_node = node
            self.low_conf = self.confidence < self.conf_threshold
            if self.low_conf:
                return
        self.current = self.estimate = node
        self.match_distance = _distance(float(similarity[self.current]))

    def _verify(self, image: np.ndarray) -> None:
        """Count consecutive steps at the goal whose view matches the first target."""
        self.verified += 1
        if (
            self.servo is None
            or self.goal_features is None
            or self.low_conf
            or self.path != [self.goal]
        ):
            self.verified = 0
            return
        current = self.servo.features(image)
        yaw, inliers = self.servo.offset(current, self.goal_features, self.servo.cx)
        if yaw is None or inliers < CHECKIN_INLIERS or abs(yaw) > self.checkin_yaw:
            self.verified = 0

    def _oscillating(self, step: int) -> bool:
        """Record a servo turn's direction; after OSCILLATION_FLIPS changes of direction within
        the last OSCILLATION_WINDOW turns, drop the servo's yaw so this step falls through to the
        queue and the hint."""
        assert self.yaw is not None and self.follow is not None
        if abs(self.yaw) <= self.yaw_tol and self._translation(self.follow) is not None:
            return False
        self.turn_signs.append(1 if self.yaw > 0 else -1)
        signs = self.turn_signs
        flips = sum(a != b for a, b in itertools.pairwise(signs))
        if len(signs) < OSCILLATION_WINDOW or flips < OSCILLATION_FLIPS:
            return False
        self.turn_signs.clear()
        print(f"turn oscillation: skipping servo turn (step {step})")
        self.yaw = None
        return True

    def _recover(self, step: int) -> None:
        """Back off after STUCK_FORWARDS FORWARD moves that left the located record unchanged;
        spin after LOST_STEPS steps without a located record, at most MAX_SPINS in a row."""
        if self.located is not None:
            self.spin_count = 0
        if self.spin:
            self.lost_count = 0
        elif self.aimed and self.located is None:
            self.lost_count += 1
        else:
            self.lost_count = 0
        if self.last == "FORWARD" and self.located is not None:
            self.stuck_count = self.stuck_count + 1 if self.located == self._previous_located else 0
        else:
            self.stuck_count = 0
        self._previous_located = self.located
        if self.stuck_count >= STUCK_FORWARDS:
            self.stuck_count = 0
            self.stuck = True
            self.queue.clear()
            self._clear_aim()
        elif self.lost_count >= LOST_STEPS:
            self.lost_count = 0
            self.queue.clear()
            self._clear_aim()
            if self.spin_count + 1 > MAX_SPINS:
                self.spin_count = 0
                self.stuck = True
                print(f"spin limit: backing off (step {step})")
                return
            self.spin_count += 1
            remaining = round(360 / self.turn_deg_per_tick)
            cap = max(1, min(self.max_segment_ticks, P.MAX_REPEAT))
            while remaining > 0:
                self.spin.append(("LEFT", min(remaining, cap)))
                remaining -= cap

    def _plan(self) -> None:
        """The path from current to the goal (empty, and no_path set, in another component),
        walked to its first trajectory edge."""
        self.queue.clear()
        self.edge = None
        try:
            self.path = nx.shortest_path(self.graph, self.current, self.goal, weight="weight")
            self.no_path = False
        except nx.NetworkXNoPath:
            self.path = []
            self.no_path = True
        if not self.no_path:
            self._walk()
        self._panel = self._render_panel()

    def _clear_aim(self) -> None:
        self.aimed, self.located, self.target, self.yaw = False, None, None, None
        self.located_inliers = self.target_inliers = 0

    def _aim(self, image: np.ndarray) -> None:
        """Locate the robot in the followed edge's span, moving on to the path's next
        trajectory edge from the span's end, then the yaw offset to the record
        lookahead_records further on. Nothing without a servo, while low on confidence, or
        with no trajectory edge on the path."""
        self._clear_aim()
        if self.servo is None or self.low_conf or self.no_path:
            return
        edges = [
            (a, b)
            for a, b in itertools.pairwise(self.path)
            if not self.graph[a][b].get("visual", False)
        ]
        if not edges:
            return
        if self.follow not in edges:
            self._follow(edges[0])
        self.aimed = True
        while True:
            k, self.located_inliers = self._locate(image)
            if k is None:
                return
            i = edges.index(self.follow)
            if k < len(self.span) - 1 or i + 1 == len(edges):
                break
            self._follow(edges[i + 1])
        self.at, self.located = k, self.span[k]
        self.target = self.span[min(k + self.lookahead_records, len(self.span) - 1)]
        self.yaw, self.target_inliers = self.servo.yaw_offset(
            image, self._record_image(self.target)
        )

    def _follow(self, edge: tuple[int, int]) -> None:
        """Follow trajectory edge ``a -> b``: records ``raw_index[a]..raw_index[b]``."""
        a, b = edge
        start, end = self.raw_index[a], self.raw_index[b]
        step = 1 if end >= start else -1
        trajectory = self.frames[a].trajectory
        self.follow, self.at = edge, 0
        self.span = [(trajectory, r) for r in range(start, end + step, step)]

    def _locate(self, image: np.ndarray) -> tuple[int | None, int]:
        """The span position from the last located to LOCATE_WINDOW past it with the most
        inliers (None: none reached min_matches), and those inliers (or the most seen)."""
        assert self.servo is not None
        best, best_inliers, most = None, 0, 0
        for k in range(self.at, min(self.at + LOCATE_WINDOW + 1, len(self.span))):
            yaw, inliers = self.servo.yaw_offset(image, self._record_image(self.span[k]))
            most = max(most, inliers)
            if yaw is not None and inliers > best_inliers:
                best, best_inliers = k, inliers
        return best, best_inliers if best is not None else most

    def _record_image(self, record: Record) -> Path:
        trajectory, index = record
        return self.record_paths[trajectory][index]

    def _translation(self, edge: tuple[int, int]) -> str | None:
        """FORWARD or BACKWARD, whichever the edge's segment drives more (None: it only
        turns); walking a trajectory backwards, its recorded translation reversed."""
        counts = Counter(a for a in self.segments[edge] if a in {"FORWARD", "BACKWARD"})
        return counts.most_common(1)[0][0] if counts else None

    def _on_edge(self) -> bool:
        """Whether the estimate is the queued edge's start or end node, or a neighbour of one."""
        assert self.edge is not None
        a, b = self.edge
        return (
            self.current in {a, b} or self.current in self.graph[a] or self.current in self.graph[b]
        )

    def _guard(self, step: int) -> None:
        """Count confident estimates since the hop count to the goal last beat its best; after
        stall_window of them, penalise the path's first edge and replan."""
        if self.no_path:
            return
        hops = nx.shortest_path_length(self.graph, self.estimate, self.goal)
        if hops < self.best_hops:
            self.best_hops, self.since_best = hops, 0
            return
        self.since_best += 1
        if self.since_best < self.stall_window or len(self.path) < 2:
            return
        a, b = self.path[0], self.path[1]
        key = (min(a, b), max(a, b))
        self._weights.setdefault(key, self.graph[a][b]["weight"])
        self.graph[a][b]["weight"] *= self.stall_penalty
        self.stalls.append({"step": step, "edge": (a, b), "hops": hops})
        print(f"stall {len(self.stalls)}: edge {a}-{b} penalised (hops {hops}, step {step})")
        self._plan()
        self.best_hops, self.since_best = hops, 0

    def _walk(self) -> None:
        """Skip the path's leading visual edges (current becomes their far node), then queue the
        recorded moves of its first trajectory edge. Only visual edges to the goal: at the goal."""
        k = 0
        while k + 1 < len(self.path) and self.graph[self.path[k]][self.path[k + 1]].get("visual"):
            k += 1
        self.path = self.path[k:]
        self.current = self.path[0]
        if len(self.path) >= 2:
            self.edge = (self.path[0], self.path[1])
            self.queue = _moves(
                self.segments[self.edge], self.capture_every, self.max_segment_ticks
            )

    def _build_graph(self, bounds: list[tuple[int, int]], top_k: int) -> nx.Graph:
        n = len(self.database)
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for start, end in bounds:
            for i in range(start, end - 1):
                graph.add_edge(i, i + 1, weight=TEMPORAL_WEIGHT)

        gap = max(MIN_SHORTCUT_GAP, round(self.shortcut_gap_frac * n))
        top_k = max(top_k, round(self.shortcuts_per_1k * n / 1000))
        for i, j, s in _top_shortcuts(self.database, top_k, gap):
            d = _distance(s)
            graph.add_edge(i, j, weight=VISUAL_WEIGHT_BASE + VISUAL_WEIGHT_SCALE * d, visual=True)
        components = [len(c) for c in nx.connected_components(graph)]
        print(
            f"graph: {n} nodes, {graph.number_of_edges()} edges ({top_k} visual, gap {gap}); "
            f"{len(components)} components, largest {max(components, default=0)} nodes"
        )
        return graph


def _name(record: Record | None) -> str:
    """``trajectory:record``, as eval_control logs it."""
    return "" if record is None else f"{record[0]}:{record[1]}"


def _record_paths(frames: list[Frame]) -> dict[int, list[Path]]:
    """Every record's image in each loaded trajectory's ``data_info.json``, IDLE and
    combined actions included."""
    paths = {}
    for frame in frames:
        if frame.trajectory not in paths:
            directory = frame.path.parent
            records = json.loads((directory / "data_info.json").read_text())
            paths[frame.trajectory] = [directory / r["image"] for r in records]
    return paths


def _single(record: dict) -> bool:
    return len(record["action"]) == 1 and record["action"][0] in MOVEMENT


def _segments(
    frames: list[Frame], bounds: list[tuple[int, int]], subsample: int
) -> tuple[list[int], dict[tuple[int, int], list[str]]]:
    """Each node's record index within its trajectory's ``data_info.json``, and per temporal
    edge ``(i, j)`` the single movement actions recorded from node ``i`` up to node ``j``: for
    ``i -> i + 1`` as recorded, for ``i + 1 -> i`` reversed and each one undone."""
    raw_index: list[int] = []
    segments: dict[tuple[int, int], list[str]] = {}
    for start, end in bounds:
        if start == end:
            continue
        records = json.loads((frames[start].path.parent / "data_info.json").read_text())
        raw = [k for k, r in enumerate(records) if _single(r)][::subsample]
        assert len(raw) == end - start, "data_info.json does not match the loaded frames"
        raw_index.extend(raw)
        for i, (a, b) in enumerate(itertools.pairwise(raw), start):
            actions = [r["action"][0] for r in records[a:b] if _single(r)]
            segments[i, i + 1] = actions
            segments[i + 1, i] = [REVERSE[action] for action in reversed(actions)]
    return raw_index, segments


def _moves(
    actions: list[str], capture_every: int, max_ticks: int = P.MAX_REPEAT
) -> list[tuple[str, int]]:
    """Runs of equal actions as ``(action, ticks)``, ``capture_every`` ticks per record and at
    most ``max_ticks`` (and ``P.MAX_REPEAT``) ticks per move."""
    cap = min(max_ticks, P.MAX_REPEAT)
    moves = []
    for action, group in itertools.groupby(actions):
        ticks = len(list(group)) * capture_every
        while ticks > 0:
            moves.append((action, min(ticks, cap)))
            ticks -= cap
    return moves


def _top_shortcuts(
    database: np.ndarray, top_k: int, gap: int = MIN_SHORTCUT_GAP, chunk: int = SHORTCUT_CHUNK
) -> list[tuple[int, int, float]]:
    """The ``top_k`` most similar pairs ``i < j`` more than ``gap`` apart, as
    ``(i, j, similarity)``. Scans ``database @ database.T`` a block of rows at a time with a
    running top-k, so the full N x N matrix never exists. Masked pairs score -2, as in the
    baseline."""
    if top_k <= 0:
        return []
    n = len(database)
    top_k = min(top_k, n * n)
    best = np.empty(0, database.dtype)
    best_index = np.empty(0, np.int64)
    columns = np.arange(n)
    for start in range(0, n, chunk):
        block = database[start : start + chunk] @ database.T
        rows = np.arange(start, start + len(block))[:, None]
        block[columns[None, :] <= rows + gap] = -2
        flat = block.ravel()
        keep = np.argpartition(flat, -min(top_k, len(flat)))[-top_k:]
        best = np.concatenate([best, flat[keep]])
        best_index = np.concatenate([best_index, keep + start * n])
        if len(best) > top_k:
            keep = np.argpartition(best, -top_k)[-top_k:]
            best, best_index = best[keep], best_index[keep]
    return [(*divmod(int(index), n), float(s)) for index, s in zip(best_index, best, strict=True)]


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    """The NodeFilter flags, shared with the evaluation scripts."""
    parser.add_argument(
        "--no-filter", dest="filter", action="store_false", help="localise by raw argmax"
    )
    parser.add_argument("--hops", type=int, default=2, help="filter: reach of one move")
    parser.add_argument("--p-stay", type=float, default=0.5, help="filter: weight of staying")
    parser.add_argument(
        "--action-boost", type=float, default=3.0, help="filter: weight of the taken move's edge"
    )
    parser.add_argument("--eps", type=float, default=1e-3, help="filter: uniform mixing per move")
    parser.add_argument("--tau", type=float, default=0.02, help="filter: similarity temperature")
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=CONF_THRESHOLD,
        help="filter: below this, keep the previous node and look around",
    )


def add_control_args(parser: argparse.ArgumentParser) -> None:
    """The segment, stall and servo flags, shared with eval_control."""
    parser.add_argument(
        "--max-segment-ticks",
        type=int,
        default=MAX_SEGMENT_TICKS,
        help="cap on each queued segment move's ticks",
    )
    parser.add_argument(
        "--stall-window",
        type=int,
        default=STALL_WINDOW,
        help="confident estimates without a new best hop count before a stall",
    )
    parser.add_argument(
        "--stall-penalty",
        type=float,
        default=STALL_PENALTY,
        help="a stall multiplies the path's first edge weight by this",
    )
    parser.add_argument(
        "--servo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="follow edges by matching the view (--no-servo: replay the recorded segments)",
    )
    parser.add_argument(
        "--lookahead-records",
        type=int,
        default=LOOKAHEAD_RECORDS,
        help="servo: raw records from the located one to the aim",
    )
    parser.add_argument(
        "--yaw-tol", type=float, default=YAW_TOL, help="servo: degrees off before turning"
    )
    parser.add_argument(
        "--forward-ticks", type=int, default=FORWARD_TICKS, help="servo: ticks per FORWARD"
    )
    parser.add_argument(
        "--turn-deg-per-tick",
        type=float,
        default=TURN_DEG_PER_TICK,
        help="servo: degrees one turn tick turns",
    )


def control_kwargs(args: argparse.Namespace) -> dict:
    return {
        "max_segment_ticks": args.max_segment_ticks,
        "stall_window": args.stall_window,
        "stall_penalty": args.stall_penalty,
        "servo": args.servo,
        "lookahead_records": args.lookahead_records,
        "yaw_tol": args.yaw_tol,
        "forward_ticks": args.forward_ticks,
        "turn_deg_per_tick": args.turn_deg_per_tick,
    }


def filter_kwargs(args: argparse.Namespace) -> dict:
    return {
        "filter": args.filter,
        "hops": args.hops,
        "p_stay": args.p_stay,
        "action_boost": args.action_boost,
        "eps": args.eps,
        "tau": args.tau,
        "conf_threshold": args.conf_threshold,
    }


if __name__ == "__main__":
    parser = cli.parser(__doc__)
    parser.add_argument("--data", default=None, help="exploration data directory")
    parser.add_argument("--subsample", type=int, default=5, help="keep every Nth frame")
    parser.add_argument("--n-clusters", type=int, default=128, help="VLAD codebook size")
    parser.add_argument("--top-k", type=int, default=30, help="visual shortcut edges, at least")
    parser.add_argument(
        "--shortcut-gap-frac",
        type=float,
        default=SHORTCUT_GAP_FRAC,
        help="shortcuts join nodes at least this fraction of the index apart",
    )
    parser.add_argument(
        "--shortcuts-per-1k",
        type=float,
        default=SHORTCUTS_PER_1K,
        help="visual shortcut edges per 1000 nodes",
    )
    add_filter_args(parser)
    parser.add_argument(
        "--verify-candidates",
        type=int,
        default=0,
        help="filter: top visual matches checked by feature matching (0: off)",
    )
    parser.add_argument(
        "--geom-weight",
        type=float,
        default=0.001,
        help="filter: similarity added per inlier of a verified candidate",
    )
    add_control_args(parser)
    parser.add_argument(
        "--goal-views", type=int, default=GOAL_VIEWS, help="targets summed to pick the goal node"
    )
    parser.add_argument(
        "--checkin-views",
        type=int,
        default=CHECKIN_VIEWS,
        help="consecutive verified decisions at the goal before checking in",
    )
    parser.add_argument(
        "--checkin-yaw",
        type=float,
        default=CHECKIN_YAW,
        help="degrees of yaw to the first target a verified check-in allows",
    )
    parser.add_argument(
        "--profile", action="store_true", help="print the top 15 functions of the first 1000 steps"
    )
    parser.add_argument("--ticks", type=int, default=5, help="ticks per move")
    parser.add_argument(
        "--capture-every", type=int, default=1, help="ticks per recorded exploration step"
    )
    parser.add_argument(
        "--random-every", type=int, default=10, help="random move every N moves (0: never)"
    )
    parser.add_argument("--max-steps", type=int, default=10_000, help="step budget for this run")
    parser.add_argument("--no-viewer", action="store_true", help="do not open the viewer window")
    args = parser.parse_args()

    challenge = cli.challenge(args)
    agent = MyAgent(
        cli.exploration_data(args, args.data),
        ticks=args.ticks,
        capture_every=args.capture_every,
        random_every=args.random_every,
        n_clusters=args.n_clusters,
        subsample=args.subsample,
        top_k_shortcuts=args.top_k,
        shortcut_gap_frac=args.shortcut_gap_frac,
        shortcuts_per_1k=args.shortcuts_per_1k,
        cache_dir=Path("cache") / challenge.replace(":", "-"),
        goal_views=args.goal_views,
        checkin_views=args.checkin_views,
        checkin_yaw=args.checkin_yaw,
        **filter_kwargs(args),
        verify_candidates=args.verify_candidates,
        geom_weight=args.geom_weight,
        **control_kwargs(args),
    )
    if args.profile:
        profiler, taken, act = cProfile.Profile(), [0], agent.act

        def profiled(obs: Observation) -> Action | tuple[Action, int]:
            if taken[0] >= PROFILE_STEPS:
                return act(obs)
            profiler.enable()
            try:
                return act(obs)
            finally:
                profiler.disable()
                taken[0] += 1
                if taken[0] == PROFILE_STEPS:
                    pstats.Stats(profiler).sort_stats("cumulative").print_stats(15)

        agent.act = profiled
    budget = {"max_steps": args.max_steps} if args.local is not None else {}
    opts = cli.run_options(args)
    opts["viewer"] = not args.no_viewer
    run(agent, challenge, **budget, **opts)
