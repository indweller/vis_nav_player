"""
Path planning and online localisation on the frozen map, against the ground truth. No controller:
an oracle drives.

    uv run source/eval/eval_planning.py
    uv run source/eval/eval_planning.py --mazes 7 --variants direct

Per maze (the frozen maps of eval_goal_map, mazes 7-12) the goal is goal.find_goal on
world.targets(), and the true target moved into the map frame, so the planning error can be told
from the goal error: each run has a navigator for each goal, sharing one localiser (the oracle
drives the same either way).

The oracle is record_gt's Driver following a shortest path of cells (common.bfs, the oracle's
route: world.shortest_path only counts it) from the spawn to the target, with motion noise on
and a motion_seed other than the exploration data's. Variants: "direct", and "detour": at a
third of the route it drives to a random free cell 1-2 m off the route, then resumes to the
target.

Every tick, navigator.update(frame, action) runs on the JPEG-encoded frame, and the true pose,
estimate, sigma, fix, jump, lost state, replans (with reasons) and paths are logged. Per maze,
variant and goal (results/eval/planning/planning.csv):
  - localisation error over time (median, p90, max) and the fractions of ticks within 0.1 / 0.2 m,
    against the world truth (loc_) and against the map-consistent truth (cm_): the true pose as
    the map has it, the mapped pose of the frame nearest the true pose composed with the true
    relative pose from that frame to the robot (what a perfect localiser against this map says);
  - fixes accepted / held / rejected, and false fixes (accepted, then more than 0.3 m from the
    map-consistent truth); lost events;
  - replans, by reason;
  - plan quality at each replan: whether any link edge on the path (an edge that is not a route
    step) crosses a true wall, judged on the two frames' true positions against the walls as thin
    slabs (Walls); and the path's length along the nodes' true positions, from the robot's true
    position, over the true grid geodesic to the goal's cell;
  - the final estimate's error, and its distance from the goal.
  - the goal's error also against the map-consistent goal (goal_cm_err_m), and find_goal's chosen
    candidate's layout score and view agreement (printed, with the layout scores of its true
    candidates, within 0.3 m of the map-consistent goal, against the false);
  - the layout check of the fixes (geometry.layout_score of the frame against the matched
    frames, --layout-min): every tick's fix candidate (one that passed the inlier test) is true
    or false by its distance, 0.3 m, from the map-consistent truth. Per run: the candidates, the
    false ones before the gate and after it (layout >= --layout-min), the layout scores of true
    and false candidates, and the fixes the gate dropped. false_fixes is still the accepted ones.
Also per maze (results/eval/planning/links.csv): the link edges before and after each of the
planner's filters (the occupied-cell test, then the SIFT inlier test) and the fraction of them
that cross a true wall; and find_goal's top candidates.

Plots, in results/eval/planning/: <seed>_<variant>.png (the map; the true route in black, the
estimated one in blue, the estimated goal's path at each replan thin and coloured by time, the
goal true green and estimated red) and <seed>_<variant>_error.png (error against time, with fix
and replan markers).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as sim
from common import cell_of, fmt, geodesic, ints, textures, write_csv
from eval_cues import Walls, stored_floor
from eval_geometry import Frame
from eval_goal_map import jpeg
from eval_map import MAP_COLOUR, Canvas, MapTruth, record, true_poses
from record_gt import ACTIONS, Driver, route

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frozen_map  # noqa: E402
import goal  # noqa: E402
import localiser as localiser_module  # noqa: E402
import mapping  # noqa: E402
import navigator as navigator_module  # noqa: E402
import planner as planner_module  # noqa: E402
from common import bfs  # noqa: E402
from geometry import LAYOUT_MIN, Camera  # noqa: E402

VARIANTS = ("direct", "detour")
GOALS = ("found", "true")
REASONS = ("initial", "cross_track", "jump", "periodic")
DETOUR_M = (1.0, 2.0)
MAX_TICKS = 6000
MOTION_SEED_OFFSET = 1000
"""The driven run's motion_seed is the maze's plus this, not the exploration's."""
WITHIN_M = (0.1, 0.2)
FALSE_FIX_M = 0.3
TRUE_CANDIDATE_M = 0.3
"""A goal candidate within this of the map-consistent goal is true."""
FIELDS = [
    "seed",
    "variant",
    "goal",
    "ticks",
    "goal_err_m",
    "goal_cm_err_m",
    "goal_layout",
    "goal_views",
    *(
        f"{p}{k}"
        for p in ("loc_", "cm_")
        for k in ("median_m", "p90_m", "max_m", "within_0.1", "within_0.2")
    ),
    "fixes_per_100",
    "fixes_accepted",
    "fixes_held",
    "fixes_rejected",
    "false_fixes",
    "candidates",
    "candidates_false",
    "candidates_false_gated",
    "layout_true_median",
    "layout_false_median",
    "fixes_layout_dropped",
    "lost_events",
    "replans",
    *(f"replans_{r}" for r in REASONS),
    "plans_crossing",
    "ratio_map_median",
    "ratio_true_median",
    "ratio_true_max",
    "final_err_m",
    "final_to_goal_m",
]
LINK_FIELDS = ["seed", "stage", "links", "crossing_wall", "fraction"]


# -- the oracle


def detour_cell(world: sim.Simulator, cells: list, rng: np.random.Generator):
    """A free cell reachable from the route 1-2 m from all of it."""
    free = ~world.occupancy
    reachable = np.array(sorted(bfs(free, tuple(world.spawn_cell))))
    size = world.cell_size
    near = np.min(
        np.hypot(*(reachable[:, None, :] - np.array(cells)[None, :, :]).transpose(2, 0, 1)),
        axis=1,
    )
    ok = reachable[(near * size >= DETOUR_M[0]) & (near * size <= DETOUR_M[1])]
    if not len(ok):
        return None
    return tuple(int(v) for v in ok[rng.integers(len(ok))])


def stages(world: sim.Simulator, variant: str, seed: int) -> list:
    """The cells the oracle drives to, in order, the last being the target."""
    free = ~world.occupancy
    start, target = tuple(world.spawn_cell), tuple(world.target_cell)
    cells = [start, *route(bfs(free, start), start, target)]
    expected = world.shortest_path["cells"]
    if len(cells) != expected:
        print(f"  note: BFS route has {len(cells)} cells, world.shortest_path {expected}")
    if variant == "direct":
        return [target]
    side = detour_cell(world, cells, np.random.default_rng(seed))
    third = cells[len(cells) // 3]
    return [third, *([side] if side else []), target]


def drive(world: sim.Simulator, targets: list, max_ticks: int):
    """Yield ``(frame, action name, true world pose)`` for each tick of the oracle's drive,
    after the first frame (action None, at the spawn)."""
    free = ~world.occupancy
    driver = Driver(world, 1)
    frame = world.initial_frame()
    yield jpeg(world, frame), None, world.pose()
    path: list = []
    queue = list(targets)
    for _ in range(max_ticks):
        x, y, yaw = world.pose()
        driver.advance(path, x, y)
        while not path and queue:
            here = cell_of(world, free, x, y)
            path = route(bfs(free, here), here, queue.pop(0))
        if not path:
            return
        action = driver.action(path[0], x, y, yaw)
        frame = world.step(action, 1)[-1]
        driver.moved(action, x, y)
        yield jpeg(world, frame), ACTIONS[action], world.pose()


# -- scoring


def segment_hits_rects(rects: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """Whether the segment ``a`` to ``b`` meets any of the axis-aligned ``rects`` (x0, x1, y0,
    y1): the slab test, the segment's parameter interval inside both slabs non-empty."""
    d = b - a
    enter, leave = np.zeros(len(rects)), np.ones(len(rects))
    for axis, (lo, hi) in enumerate([(rects[:, 0], rects[:, 1]), (rects[:, 2], rects[:, 3])]):
        if abs(d[axis]) < 1e-12:
            outside = (a[axis] < lo) | (a[axis] > hi)
            leave = np.where(outside, -1.0, leave)
            continue
        t0, t1 = (lo - a[axis]) / d[axis], (hi - a[axis]) / d[axis]
        enter = np.maximum(enter, np.minimum(t0, t1))
        leave = np.minimum(leave, np.maximum(t0, t1))
    return bool((enter <= leave).any())


class Truth(MapTruth):
    """The ground truth of a maze against its map: MapTruth, the true walls as thin slabs, and
    the questions asked of them."""

    def __init__(self, store, truth: list[np.ndarray], walls: Walls) -> None:
        super().__init__(store, truth)
        self.walls = walls

    def crossings(self, rows: np.ndarray) -> np.ndarray:
        """Per ``(row, row)`` pair, whether the segment between the two frames' true positions
        meets a true wall."""
        return np.array(
            [
                segment_hits_rects(self.walls.rects, self.poses[a, :2], self.poses[b, :2])
                for a, b in rows
            ],
            bool,
        )


class Run:
    """What one drive logged, in the map frame."""

    def __init__(self) -> None:
        self.true: list[np.ndarray] = []
        self.consistent: list[np.ndarray] = []
        self.estimate: list[np.ndarray] = []
        self.sigma: list[float] = []
        self.fixed: list[bool] = []
        self.jump: list[float] = []
        self.lost: list[bool] = []
        self.status: list[str | None] = []
        self.rejected: list[int] = []
        self.candidate: list[np.ndarray | None] = []
        self.layout: list[float] = []
        self.replans: dict[str, list[dict]] = {g: [] for g in GOALS}

    def error(self, against: str = "true") -> np.ndarray:
        truth = np.array(self.true if against == "true" else self.consistent)
        return np.hypot(*(truth[:, :2] - np.array(self.estimate)[:, :2]).T)

    def candidates(self) -> tuple[np.ndarray, np.ndarray]:
        """The layout scores of the fix candidates, true ones and false ones: a candidate is
        false when over FALSE_FIX_M from the map-consistent truth of its tick."""
        true, false = [], []
        for candidate, layout, truth in zip(
            self.candidate, self.layout, self.consistent, strict=True
        ):
            if candidate is not None:
                wrong = math.dist(candidate[:2], truth[:2]) > FALSE_FIX_M
                (false if wrong else true).append(layout)
        return np.array(true), np.array(false)

    def false_fixes(self) -> int:
        """Accepted fixes after which the estimate is over FALSE_FIX_M from the map-consistent
        truth."""
        error = self.error("consistent")
        return int(sum(error[i] > FALSE_FIX_M for i, f in enumerate(self.fixed) if f))


def run_variant(
    world, driven, frame, store, planner, truth, camera, floor, motion, goals, targets, args, edges
) -> Run:
    localiser = localiser_module.Localiser(
        store,
        camera,
        floor,
        motion=motion,
        fix_every=args.fix_every,
        lost_ticks=args.lost_ticks,
        layout_min=args.layout_min,
    )
    navigators = {
        name: navigator_module.Navigator(
            planner,
            localiser,
            pose,
            replan_m=args.replan_m,
            replan_every=args.replan_every,
        )
        for name, pose in goals.items()
    }
    free = ~world.occupancy
    distances = {
        name: geodesic(world, cell_of(world, free, *frame.to_world(np.array([pose[:2]]))[0]))
        for name, pose in goals.items()
    }
    run = Run()
    for tick, (image, action, true_world) in enumerate(drive(driven, targets, args.max_ticks)):
        updates = {}
        estimate = None
        for name, nav in navigators.items():
            updates[name] = nav.update(image, action, estimate)
            estimate = updates[name].estimate
        run.true.append(frame.to_map(np.array([true_world]))[0])
        run.consistent.append(np.array(truth.consistent(true_world)))
        run.estimate.append(np.array(estimate[:3]))
        run.sigma.append(estimate.sigma_m)
        run.fixed.append(estimate.fixed)
        run.jump.append(estimate.jump)
        run.lost.append(estimate.lost)
        run.status.append(estimate.status)
        run.rejected.append(estimate.rejected)
        run.candidate.append(None if estimate.candidate is None else np.array(estimate.candidate))
        run.layout.append(estimate.layout)
        for name, update in updates.items():
            if update.replan is None:
                continue
            geo = distances[name][cell_of(world, free, *true_world[:2])]
            path = update.path
            ids = [(int(a), int(b)) for a, b in zip(path.nodes[:-1], path.nodes[1:], strict=True)]
            links = [e for e in ids if tuple(sorted(e)) not in planner.route_edges]
            on_truth = np.array([*true_world[:2]])
            along = [on_truth, *truth.poses[planner.row[path.nodes], :2]]
            true_length = float(
                sum(math.dist(p, q) for p, q in zip(along[:-1], along[1:], strict=True))
            )
            run.replans[name].append(
                {
                    "tick": tick,
                    "reason": update.replan,
                    "path": path,
                    "ratio": path.length / geo if geo > 0.2 else math.nan,
                    "ratio_true": true_length / geo if geo > 0.2 else math.nan,
                    "crosses": any(edges[tuple(sorted(e))] for e in links),
                }
            )
    run.layout_min = localiser.layout_min
    run.lost_events = localiser.lost_events
    run.counts = dict(localiser.counts)
    return run


def spread(prefix: str, error: np.ndarray) -> dict:
    return {
        f"{prefix}median_m": float(np.median(error)),
        f"{prefix}p90_m": float(np.percentile(error, 90)),
        f"{prefix}max_m": float(error.max()),
        **{f"{prefix}within_{m}": float((error <= m).mean()) for m in WITHIN_M},
    }


def score(
    seed: int, variant: str, name: str, run: Run, goals: dict, true_goal, consistent_goal, found
) -> dict:
    ticks = len(run.true)
    replans = run.replans[name]
    ratios = np.array([r["ratio"] for r in replans if math.isfinite(r["ratio"])])
    true_ratios = np.array([r["ratio_true"] for r in replans if math.isfinite(r["ratio_true"])])
    reasons = Counter(r["reason"] for r in replans)
    final = np.array(run.estimate[-1])
    goal_pose = goals[name]
    true_layout, false_layout = run.candidates()
    median = lambda v: float(np.median(v)) if len(v) else math.nan  # noqa: E731
    error = run.error()
    return {
        "seed": seed,
        "variant": variant,
        "goal": name,
        "ticks": ticks,
        "goal_err_m": math.dist(goal_pose[:2], true_goal[:2]),
        "goal_cm_err_m": math.dist(goal_pose[:2], consistent_goal[:2]),
        "goal_layout": math.nan if found is None else found.layout,
        "goal_views": math.nan if found is None else found.view_agreement,
        **spread("loc_", error),
        **spread("cm_", run.error("consistent")),
        "fixes_per_100": 100.0 * run.counts["accepted"] / ticks,
        "fixes_accepted": run.counts["accepted"],
        "fixes_held": run.counts["held"],
        "fixes_rejected": run.counts["rejected"],
        "false_fixes": run.false_fixes(),
        "candidates": len(true_layout) + len(false_layout),
        "candidates_false": len(false_layout),
        "candidates_false_gated": int((false_layout >= run.layout_min).sum()),
        "layout_true_median": median(true_layout),
        "layout_false_median": median(false_layout),
        "fixes_layout_dropped": run.counts["layout"],
        "lost_events": run.lost_events,
        "replans": len(replans),
        **{f"replans_{r}": reasons[r] for r in REASONS},
        "plans_crossing": float(np.mean([r["crosses"] for r in replans])) if replans else math.nan,
        "ratio_map_median": float(np.median(ratios)) if len(ratios) else math.nan,
        "ratio_true_median": float(np.median(true_ratios)) if len(true_ratios) else math.nan,
        "ratio_true_max": float(true_ratios.max()) if len(true_ratios) else math.nan,
        "final_err_m": float(error[-1]),
        "final_to_goal_m": math.dist(final[:2], goal_pose[:2]),
    }


# -- images


def time_colour(t: float) -> tuple[int, int, int]:
    bgr = cv2.applyColorMap(np.array([[int(255 * t)]], np.uint8), cv2.COLORMAP_VIRIDIS)[0, 0]
    return tuple(int(v) for v in bgr)


def polyline(canvas: Canvas, xy: np.ndarray, colour, width: int = 1) -> None:
    line = np.round(canvas.pixels(xy)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas.image, [line], False, colour, width, cv2.LINE_AA)


def draw_map(world, walls, frame, store, run: Run, goals, path: Path, label: str) -> None:
    canvas = Canvas(world)
    grid = store.grid
    canvas.cells(frame.to_world(grid.centres()[grid.occupied()]), (235, 235, 235))
    canvas.walls(walls, MAP_COLOUR)
    n = max(len(run.true) - 1, 1)
    for replan in run.replans["found"]:
        polyline(
            canvas,
            frame.to_world(replan["path"].waypoints[:, :2]),
            time_colour(replan["tick"] / n),
        )
    polyline(canvas, frame.to_world(np.array(run.estimate)[:, :2]), (220, 100, 0), 1)
    polyline(canvas, frame.to_world(np.array(run.true)[:, :2]), (0, 0, 0), 1)
    canvas.marker(frame.pose_to_world(tuple(goals["true"])), (0, 170, 0), arrow=True)
    canvas.marker(frame.pose_to_world(tuple(goals["found"])), (0, 0, 220), arrow=True)
    canvas.label(label)
    canvas.save(path)


def draw_error(run: Run, name: str, label: str, path: Path) -> None:
    width, height, left, bottom, top = 1100, 340, 50, 30, 24
    image = np.full((height, width, 3), 255, np.uint8)
    error = run.error()
    ticks = len(error)
    top_m = max(0.3, min(float(error.max()), 2.0))
    plot_h, plot_w = height - bottom - top, width - left - 10

    def at(i: float, metres: float) -> tuple[int, int]:
        metres = min(metres, top_m)
        return int(left + plot_w * i / max(ticks - 1, 1)), int(top + plot_h * (1 - metres / top_m))

    for m in WITHIN_M:
        cv2.line(image, at(0, m), at(ticks - 1, m), (200, 200, 200), 1)
        cv2.putText(image, f"{m} m", (4, at(0, m)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
    cv2.putText(image, f"{top_m:.1f}", (4, top + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
    in_lost = None
    for i in range(ticks + 1):
        lost = i < ticks and run.lost[i]
        if lost and in_lost is None:
            in_lost = i
        if not lost and in_lost is not None:
            overlay = image.copy()
            cv2.rectangle(
                overlay, (at(in_lost, 0)[0], top), (at(i, 0)[0], top + plot_h), (200, 230, 255), -1
            )
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            in_lost = None
    consistent = run.error("consistent")
    for i in range(1, ticks):
        cv2.line(image, at(i - 1, run.sigma[i - 1]), at(i, run.sigma[i]), (190, 190, 190), 1)
        cv2.line(
            image,
            at(i - 1, consistent[i - 1]),
            at(i, consistent[i]),
            (170, 60, 170),
            1,
            cv2.LINE_AA,
        )
        cv2.line(image, at(i - 1, error[i - 1]), at(i, error[i]), (220, 100, 0), 1, cv2.LINE_AA)
    for replan in run.replans[name]:
        x = at(replan["tick"], 0)[0]
        cv2.line(image, (x, top), (x, top + plot_h), (0, 0, 220), 1)
    for i in (i for i, status in enumerate(run.status) if status == "held"):
        x = at(i, 0)[0]
        cv2.line(image, (x, top + plot_h), (x, top + plot_h - 10), (0, 140, 255), 1)
    for i in np.flatnonzero(run.fixed):
        x = at(int(i), 0)[0]
        cv2.line(image, (x, top + plot_h), (x, top + plot_h - 6), (0, 150, 0), 1)
    cv2.putText(image, label, (left, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
    cv2.putText(
        image,
        "blue: error vs world  purple: vs map-consistent  grey: sigma  green: fixes"
        "  orange: held  red: replans",
        (left, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        0,
        1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


# -- the run


def link_report(seed: int, planner, truth: Truth) -> tuple[list[dict], dict]:
    """The link edges' truth at each filter stage, printed; and every edge's wall-crossing flag."""
    counts = planner.link_counts
    stages = {
        "candidates": planner.candidates,
        "after_walls": planner.after_walls,
        "after_inliers": planner.links,
    }
    rows = []
    for stage, pairs in stages.items():
        crossing = int(truth.crossings(planner.row[pairs]).sum()) if len(pairs) else 0
        rows.append(
            {
                "seed": seed,
                "stage": stage,
                "links": len(pairs),
                "crossing_wall": crossing,
                "fraction": crossing / len(pairs) if len(pairs) else math.nan,
            }
        )
        print(
            f"  links {stage:<13} {len(pairs):>6}   crossing a true wall {rows[-1]['fraction']:.3f}"
        )
    print(f"  link counts {counts}")
    edges = dict(
        zip(map(tuple, planner.pairs), truth.crossings(planner.row[planner.pairs]), strict=True)
    )
    return rows, edges


def goal_candidates(seed, ranking, found, store, frame, world, truth) -> None:
    """find_goal's candidates, best by inliers first: the frame, the view, its inliers and
    layout, the goal pose it gives and its distance from the map-consistent goal; and the layout
    scores of the true candidates against the false."""
    target = truth.consistent(world.target)
    chosen = "none" if found is None else f"layout {found.layout:.2f} views {found.view_agreement}"
    print(f"maze {seed}: find_goal's {len(ranking)} candidates (chosen: {chosen})")
    for c in ranking:
        mapped = store.pose(c.frame)
        true_frame = frame.to_map(np.array([truth.poses[c.frame]]))[0]
        miss = math.dist(c[3:5], target[:2])
        print(
            f"  r{store.route[c.frame]} #{store.frame[c.frame]:<5} view {c.k}  inliers "
            f"{c.inliers:>3}  layout {c.layout:.2f}  goal ({c.x:6.2f} {c.y:6.2f}) off by "
            f"{miss:.2f} m {'TRUE ' if miss <= TRUE_CANDIDATE_M else 'false'}  frame mapped "
            f"({mapped[0]:6.2f} {mapped[1]:6.2f}) true ({true_frame[0]:6.2f} {true_frame[1]:6.2f})"
        )
    for label, want in (("true", True), ("false", False)):
        scores = [
            c.layout for c in ranking if (math.dist(c[3:5], target[:2]) <= TRUE_CANDIDATE_M) is want
        ]
        print(f"  {label} candidates: {len(scores)}, layouts {[round(v, 2) for v in scores]}")


def evaluate(seed: int, args, floor, out: Path) -> tuple[list[dict], list[dict]]:
    root = record(seed)
    world = sim.Simulator(textures(), seed)
    walls = Walls(world)
    store, routes = frozen_map.get(root, world.camera, floor)
    truth_poses = true_poses(root, routes)
    truth = Truth(store, truth_poses, walls)
    frame = Frame((0.0, 0.0, 0.0), tuple(truth_poses[0][0]))
    camera = Camera({**world.camera, "cy": world.camera["cy"] + mapping.HORIZON_SHIFT_PX})
    motion = mapping.calibrate(routes, camera, floor)
    planner = planner_module.PlaceGraph(
        store,
        link_m=args.link_m,
        turn_cost=args.turn_cost,
        connect_m=args.connect_m,
        min_inliers=args.min_inliers,
        cache_path=root / "links.npz",
    )

    true_goal = tuple(frame.to_map(np.array([world.target]))[0])
    ranking: list = []
    found = goal.find_goal(
        world.targets(), store, camera, floor, ranking, layout_min=args.layout_min
    )
    consistent_goal = truth.consistent(world.target)
    goals = {"true": true_goal}
    if found is not None:
        goals["found"] = (found.x, found.y, found.yaw)
    else:
        print(f"maze {seed}: no goal found; planning to the true one only")
        goals["found"] = true_goal
    print(
        f"maze {seed}: {planner.n} nodes, {len(planner.pairs)} edges; goal error "
        f"{math.dist(goals['found'][:2], true_goal[:2]):.2f} m",
        flush=True,
    )
    goal_candidates(seed, ranking, found, store, frame, world, truth)
    link_rows, edges = link_report(seed, planner, truth)

    rows = []
    for variant in args.variants:
        driven = sim.Simulator(
            textures(),
            seed,
            motion_noise=sim.DEFAULT_NOISE,
            motion_seed=seed + MOTION_SEED_OFFSET,
        )
        targets = stages(driven, variant, seed)
        run = run_variant(
            world,
            driven,
            frame,
            store,
            planner,
            truth,
            camera,
            floor,
            motion,
            goals,
            targets,
            args,
            edges,
        )
        for name in GOALS:
            rows.append(score(seed, variant, name, run, goals, true_goal, consistent_goal, found))
        row = rows[-2]
        label = (
            f"maze {seed} {variant}: median error {row['loc_median_m']:.2f} m "
            f"(map-consistent {row['cm_median_m']:.2f}), {row['replans']} replans, "
            f"goal error {row['goal_err_m']:.2f} m"
        )
        draw_map(world, walls, frame, store, run, goals, out / f"{seed}_{variant}.png", label)
        draw_error(run, "found", label, out / f"{seed}_{variant}_error.png")
        print(f"  {variant}: {len(run.true)} ticks", flush=True)
    store.sift.flush()
    return rows, link_rows


def summarise(rows: list[dict], links: list[dict]) -> None:
    head = (
        f"{'maze':>4} {'variant':<7} {'goal':<5} {'ticks':>5} | {'world: med':>10} {'p90':>5} "
        f"{'max':>5} {'<.2':>5} | {'map-cons: med':>13} {'p90':>5} {'max':>5} {'<.2':>5} | "
        f"{'acc':>4} {'held':>4} {'rej':>4} {'false':>5} {'lost':>4} | {'replans':>7} "
        f"{'i/c/j/p':>11} | {'cross':>5} {'ratio':>5} | {'final':>5} {'to goal':>7}"
    )
    print("\n" + head + "\n" + "-" * len(head))
    for r in rows:
        by = "/".join(str(r[f"replans_{x}"]) for x in REASONS)
        print(
            f"{r['seed']:>4} {r['variant']:<7} {r['goal']:<5} {r['ticks']:>5} | "
            f"{fmt(r['loc_median_m'], 2):>10} {fmt(r['loc_p90_m'], 2):>5} "
            f"{fmt(r['loc_max_m'], 2):>5} {r['loc_within_0.2']:>5.2f} | "
            f"{fmt(r['cm_median_m'], 2):>13} {fmt(r['cm_p90_m'], 2):>5} "
            f"{fmt(r['cm_max_m'], 2):>5} {r['cm_within_0.2']:>5.2f} | "
            f"{r['fixes_accepted']:>4} {r['fixes_held']:>4} {r['fixes_rejected']:>4} "
            f"{r['false_fixes']:>5} {r['lost_events']:>4} | {r['replans']:>7} {by:>11} | "
            f"{fmt(r['plans_crossing'], 2):>5} {fmt(r['ratio_true_median'], 2):>5} | "
            f"{fmt(r['final_err_m'], 2):>5} {fmt(r['final_to_goal_m'], 2):>7}"
        )
    print("\ngoals and the layout check (per maze and variant; the goal rows are the found goal):")
    print(
        f"{'maze':>4} {'variant':<7} | {'goal err':>8} {'cm err':>7} {'layout':>6} {'views':>5} | "
        f"{'cands':>5} {'false':>5} {'false gated':>11} {'dropped':>7} | "
        f"{'layout true':>11} {'false':>6} | {'false fixes':>11}"
    )
    for r in (r for r in rows if r["goal"] == "found"):
        print(
            f"{r['seed']:>4} {r['variant']:<7} | {fmt(r['goal_err_m'], 2):>8} "
            f"{fmt(r['goal_cm_err_m'], 2):>7} {fmt(r['goal_layout'], 2):>6} "
            f"{fmt(r['goal_views'], 0):>5} | {r['candidates']:>5} {r['candidates_false']:>5} "
            f"{r['candidates_false_gated']:>11} {r['fixes_layout_dropped']:>7} | "
            f"{fmt(r['layout_true_median'], 2):>11} {fmt(r['layout_false_median'], 2):>6} | "
            f"{r['false_fixes']:>11}"
        )
    print("\nlink edges crossing a true wall (all mazes pooled):")
    for stage in ("candidates", "after_walls", "after_inliers"):
        pooled = [r for r in links if r["stage"] == stage]
        n = sum(r["links"] for r in pooled)
        crossing = sum(r["crossing_wall"] for r in pooled)
        print(f"  {stage:<13} {n:>7} links, {crossing / max(n, 1):.3f} cross a wall")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mazes", default="7,8,9,10,11,12", help="comma-separated maze seeds")
    p.add_argument("--variants", default=",".join(VARIANTS), help="direct, detour or both")
    p.add_argument("--out-dir", default="results/eval", help="where planning/ goes")
    p.add_argument("--max-ticks", type=int, default=MAX_TICKS, help="stop a drive after this many")
    p.add_argument("--link-m", type=float, default=planner_module.LINK_M)
    p.add_argument("--turn-cost", type=float, default=planner_module.TURN_COST)
    p.add_argument("--connect-m", type=float, default=planner_module.CONNECT_M)
    p.add_argument("--min-inliers", type=int, default=planner_module.LINK_INLIERS)
    p.add_argument("--layout-min", type=float, default=LAYOUT_MIN, help="layout score to pass")
    p.add_argument("--fix-every", type=int, default=localiser_module.FIX_EVERY)
    p.add_argument("--lost-ticks", type=int, default=localiser_module.LOST_TICKS)
    p.add_argument("--replan-m", type=float, default=navigator_module.REPLAN_M)
    p.add_argument("--replan-every", type=int, default=navigator_module.REPLAN_EVERY)
    args = p.parse_args()
    args.variants = [v for v in args.variants.split(",") if v]

    out = Path(args.out_dir) / "planning"
    floor = stored_floor()
    rows: list[dict] = []
    links: list[dict] = []
    for seed in ints(args.mazes):
        maze_rows, maze_links = evaluate(seed, args, floor, out)
        rows.extend(maze_rows)
        links.extend(maze_links)
    write_csv(out / "planning.csv", FIELDS, rows)
    write_csv(out / "links.csv", LINK_FIELDS, links)
    summarise(rows, links)


if __name__ == "__main__":
    main()
