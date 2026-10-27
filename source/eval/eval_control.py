"""
MyAgent's act() loop on the simulator with the truth alongside: every decision is logged with the
true pose, the matched node and its pose error, the next node on the path, the action and where it
came from, and whether the move brought the robot nearer that node's recorded pose.

    uv run source/eval/eval_control.py
    uv run source/eval/eval_control.py --mazes 7 --max-decisions 500 --stop-at-checkin
    uv run source/eval/eval_control.py --tau-sweep 0.005,0.01,0.02,0.05 --conf-sweep 0.3,0.5,0.7

Action sources: "servo-turn" and "servo-forward" (MyAgent's servo turned toward its target
record, or drove the edge's recorded translation toward it), "segment" (a move queued from a
trajectory edge's raw recorded actions; queue is the number of moves still queued after it),
"temporal" (the recorded action along a trajectory edge), "reverse" (its reverse, walking a
trajectory backwards), "visual-fallback" (a visual-shortcut edge, which has no action, so MyAgent
turns), "no-path" (the goal is in another graph component; MyAgent turns), "low-conf" (the node
filter is unsure where the robot is; MyAgent turns), and "checkin". A check-in does not end the
loop unless --stop-at-checkin: the robot turns instead (MyAgent's fallback) so the rule keeps
being tested. Distances to a node are pose distances (position error plus yaw error at 0.5 m
per radian); progress to the goal is the geodesic grid distance from the robot's cell to the
target's.

Each decision also logs the raw argmax node and the filter's estimate with their position errors,
the filter's confidence, and whether it was below the threshold, plus MyAgent's stalls so far and
the edge a stall penalised that decision; each maze's summary prints its stall count. Each also logs
the servo's located and target records as "trajectory:record" with the inliers against each, and
the yaw offset (blank located: no record reached min_matches, its inliers then the most any had;
blank yaw: too few matches to the target). The summary prints how often the servo tried, failed
to locate the robot, and got no yaw. With
--tau-sweep or --conf-sweep, every combination of the two runs on every maze and filter_sweep.csv
gets one summary row per combination (the other CSVs are not written).
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import vis_nav_sim as sim
from common import (
    build_agent,
    cell_of,
    floats,
    fmt,
    geodesic,
    ints,
    node_poses,
    parser,
    pose_distance,
    textures,
    write_csv,
    yaw_error_deg,
)
from my_agent import (
    FALLBACK,
    _name,
    add_control_args,
    add_filter_args,
    control_kwargs,
    filter_kwargs,
)
from vis_nav_sdk import Action
from vis_nav_sdk import protocol as P
from vis_nav_sdk.session import Camera, Limits, Observation, SessionInfo

SOURCES = (
    "servo-turn",
    "servo-forward",
    "segment",
    "temporal",
    "reverse",
    "visual-fallback",
    "no-path",
    "low-conf",
)
SWEEP_QUANTILES = np.linspace(0.5, 1.0, 11)
FIELDS = [
    "seed",
    "decision",
    "step",
    "x",
    "y",
    "yaw_deg",
    "node",
    "node_pos_err_m",
    "node_yaw_err_deg",
    "raw_node",
    "raw_pos_err_m",
    "filter_node",
    "filter_pos_err_m",
    "confidence",
    "low_conf",
    "next_node",
    "action",
    "source",
    "queue",
    "stalls",
    "stall_edge",
    "servo",
    "servo_located",
    "servo_located_inliers",
    "servo_target",
    "servo_target_inliers",
    "servo_yaw_deg",
    "ticks",
    "next_dist_before",
    "next_dist_after",
    "next_pos_before_m",
    "next_pos_after_m",
    "goal_node",
    "goal_similarity",
    "goal_distance_m",
    "goal_heading_deg",
    "goal_geodesic_m",
    "checkin",
    "at_goal",
]
SWEEP_FIELDS = ["seed", "threshold", "fired", "precision", "recall"]
FILTER_SWEEP_FIELDS = [
    "mazes",
    "tau",
    "conf_threshold",
    "decisions",
    "raw_pos_err_med_m",
    "filter_pos_err_med_m",
    "low_conf_frac",
    "filter_err_confident_med_m",
    "filter_err_low_conf_med_m",
    "checkins",
    "checkin_precision",
    "final_geodesic_mean_m",
]


def session_info(world: sim.Simulator) -> SessionInfo:
    """What a local session would hand ``setup()``."""
    return SessionInfo(
        session_id=f"eval-{world.seed}",
        challenge={"id": f"local:{world.seed}", "name": f"eval maze {world.seed}", "local": True},
        camera=Camera(
            width=world.width,
            height=world.height,
            color_order=P.COLOR_ORDER,
            encoding="raw",
            intrinsic_matrix=world.intrinsic_matrix,
        ),
        limits=Limits(
            max_steps=P.DEFAULT_MAX_STEPS,
            max_repeat=min(P.MAX_REPEAT, sim.MAX_REPEAT),
            attempts_used=0,
            attempts_allowed=None,
            silence_timeout_s=P.DEFAULT_SILENCE_TIMEOUT_S,
            grace_s=P.DEFAULT_GRACE_S,
            session_timeout_s=P.DEFAULT_SESSION_TIMEOUT_S,
        ),
        targets=world.targets(),
    )


def source_of(agent, path: list[int]) -> str:
    if agent.low_conf:
        return "low-conf"
    if agent.no_path:
        return "no-path"
    if agent.source in {"segment", "servo-turn", "servo-forward"}:
        return agent.source
    a, b = path[0], path[1]
    if agent.graph[a][b].get("visual", False):
        return "visual-fallback"
    return "temporal" if b == a + 1 else "reverse"


def evaluate(seed: int, args, agent) -> list[dict]:
    poses = node_poses(agent.frames)
    world = sim.Simulator(
        textures(), seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=args.motion_seed
    )
    free = ~world.occupancy
    to_goal = geodesic(world, tuple(world.target_cell))
    max_steps = P.DEFAULT_MAX_STEPS

    # act() localises with extractor.extract; keep its vector for the goal similarity. The
    # class's method is wrapped, so an agent evaluated again is not wrapped twice.
    last: dict[str, np.ndarray] = {}
    extract = type(agent.extractor).extract.__get__(agent.extractor)

    def remembering(image: np.ndarray) -> np.ndarray:
        last["query"] = extract(image)
        return last["query"]

    agent.extractor.extract = remembering
    agent.setup(session_info(world))

    def to_node(node: int | None, x: float, y: float, yaw: float) -> tuple[float, float]:
        """Pose distance and position distance from ``(x, y, yaw)`` to ``node``'s pose."""
        if node is None:
            return math.nan, math.nan
        position = math.hypot(poses[node, 0] - x, poses[node, 1] - y)
        return float(pose_distance(position, yaw_error_deg(poses[node, 2], yaw))), position

    def position_error(node: int | None, x: float, y: float) -> str:
        return "" if node is None else f"{math.hypot(poses[node, 0] - x, poses[node, 1] - y):.3f}"

    frame = world.initial_frame()
    rows = []
    for decision in range(args.max_decisions):
        if world.steps >= max_steps:
            break
        x, y, yaw = world.pose()
        distance_m, heading_deg = world.distance_heading()
        step = world.steps
        stalls = len(agent.stalls)
        choice = agent.act(Observation(frame, step=step, steps_left=max_steps - step))
        node, path = agent.current, list(agent.path)
        next_node = path[1] if len(path) >= 2 else None

        if choice == Action.CHECKIN:
            action, ticks, source, checkin = getattr(Action, FALLBACK), agent.ticks, "checkin", 1
            agent.last = FALLBACK  # the move the filter predicts from next
        else:
            (action, ticks), source, checkin = choice, source_of(agent, path), 0

        before, before_pos = to_node(next_node, x, y, yaw)
        frame = world.step(int(action), ticks)[-1]
        after, after_pos = to_node(next_node, *world.pose())

        rows.append(
            {
                "seed": seed,
                "decision": decision,
                "step": step,
                "x": f"{x:.3f}",
                "y": f"{y:.3f}",
                "yaw_deg": f"{math.degrees(yaw):.1f}",
                "node": "" if node is None else node,
                "node_pos_err_m": position_error(node, x, y),
                "node_yaw_err_deg": (
                    "" if node is None else f"{yaw_error_deg(poses[node, 2], yaw):.1f}"
                ),
                "raw_node": agent.raw_node,
                "raw_pos_err_m": position_error(agent.raw_node, x, y),
                "filter_node": "" if agent.filter_node is None else agent.filter_node,
                "filter_pos_err_m": position_error(agent.filter_node, x, y),
                "confidence": fmt(agent.confidence, 4) if agent.filter is not None else "",
                "low_conf": int(agent.low_conf),
                "next_node": "" if next_node is None else next_node,
                "action": "CHECKIN" if checkin else action.name,
                "source": source,
                "queue": len(agent.queue),
                "stalls": len(agent.stalls),
                "stall_edge": (
                    "-".join(map(str, agent.stalls[-1]["edge"]))
                    if len(agent.stalls) > stalls
                    else ""
                ),
                "servo": int(agent.aimed),
                "servo_located": _name(agent.located),
                "servo_located_inliers": agent.located_inliers if agent.aimed else "",
                "servo_target": _name(agent.target),
                "servo_target_inliers": "" if agent.target is None else agent.target_inliers,
                "servo_yaw_deg": "" if agent.yaw is None else f"{agent.yaw:.2f}",
                "ticks": ticks,
                "next_dist_before": "" if next_node is None else f"{before:.4f}",
                "next_dist_after": "" if next_node is None else f"{after:.4f}",
                "next_pos_before_m": "" if next_node is None else f"{before_pos:.4f}",
                "next_pos_after_m": "" if next_node is None else f"{after_pos:.4f}",
                "goal_node": agent.goal,
                "goal_similarity": f"{float(agent.database[agent.goal] @ last['query']):.4f}",
                "goal_distance_m": f"{distance_m:.3f}",
                "goal_heading_deg": f"{heading_deg:.1f}",
                "goal_geodesic_m": f"{to_goal[cell_of(world, free, x, y)]:.2f}",
                "checkin": checkin,
                "at_goal": int(
                    distance_m <= args.checkin_distance and heading_deg <= args.checkin_heading
                ),
            }
        )
        if checkin and args.stop_at_checkin:
            break
    return rows


def precision_recall(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    hits = float((predicted & truth).sum())
    precision = hits / predicted.sum() if predicted.any() else math.nan
    recall = hits / truth.sum() if truth.any() else math.nan
    return precision, recall


def sweep(seed: int, rows: list[dict]) -> list[dict]:
    """Check in whenever the goal similarity reaches a threshold: precision and recall per
    threshold, the thresholds being quantiles of the similarities seen."""
    similarity = np.array([float(r["goal_similarity"]) for r in rows])
    truth = np.array([r["at_goal"] for r in rows], bool)
    out = []
    for threshold in np.unique(np.round(np.quantile(similarity, SWEEP_QUANTILES), 4)):
        predicted = similarity >= threshold
        precision, recall = precision_recall(predicted, truth)
        out.append(
            {
                "seed": seed,
                "threshold": f"{threshold:.4f}",
                "fired": int(predicted.sum()),
                "precision": fmt(precision),
                "recall": fmt(recall),
            }
        )
    return out


def longest_stall(rows: list[dict]) -> tuple[int, int, int, int]:
    """The longest run of decisions that never got nearer the goal than before it:
    ``(decisions, ticks, first, last)``."""
    best = math.inf
    start = 0
    longest = (0, 0, 0, 0)
    for i, r in enumerate(rows):
        distance = float(r["goal_geodesic_m"])
        if distance < best - 1e-9:
            best, start = distance, i + 1
        elif i - start + 1 > longest[0]:
            ticks = sum(int(rows[j]["ticks"]) for j in range(start, i + 1))
            longest = (i - start + 1, ticks, start, i)
    return longest


def median(rows: list[dict], key: str) -> float:
    values = [float(r[key]) for r in rows if r[key] != ""]
    return float(np.median(values)) if values else math.nan


def filter_stats(rows: list[dict]) -> dict[str, float]:
    """Raw vs filtered median position error, how often the filter was unsure, and the
    filtered error when it was sure and when it was not."""
    low = [r for r in rows if r["low_conf"]]
    sure = [r for r in rows if not r["low_conf"]]
    return {
        "raw": median(rows, "raw_pos_err_m"),
        "filter": median(rows, "filter_pos_err_m"),
        "low_conf_frac": len(low) / len(rows) if rows else math.nan,
        "confident": median(sure, "filter_pos_err_m"),
        "low_conf": median(low, "filter_pos_err_m"),
    }


def summarise(seed: int, rows: list[dict], swept: list[dict]) -> None:
    geodesic_m = [float(r["goal_geodesic_m"]) for r in rows]
    node_err = median(rows, "node_pos_err_m")
    print(
        f"\nmaze {seed}: {len(rows)} decisions, {rows[-1]['step']} ticks; geodesic to goal "
        f"{fmt(geodesic_m[0], 2)} -> {fmt(geodesic_m[-1], 2)} m (best {fmt(min(geodesic_m), 2)});"
        f" matched node median position error {fmt(node_err)} m"
    )
    stats = filter_stats(rows)
    print(
        f"  median position error: raw argmax {fmt(stats['raw'])} m, filtered "
        f"{fmt(stats['filter'])} m; low confidence {fmt(stats['low_conf_frac'])} of decisions; "
        f"filtered when confident {fmt(stats['confident'])} m, when low {fmt(stats['low_conf'])} m"
    )
    print(f"  {'source':<16} {'moves':>6} {'nearer':>7} {'pos nearer':>10}")
    for source in SOURCES:
        moves = [r for r in rows if r["source"] == source]
        if not moves:
            print(f"  {source:<16} {0:>6}")
            continue
        nearer = np.mean(
            [float(r["next_dist_after"]) < float(r["next_dist_before"]) for r in moves]
        )
        pos_nearer = np.mean(
            [float(r["next_pos_after_m"]) < float(r["next_pos_before_m"]) for r in moves]
        )
        print(f"  {source:<16} {len(moves):>6} {fmt(nearer):>7} {fmt(pos_nearer):>10}")

    aimed = [r for r in rows if r["servo"]]
    if aimed:
        located = [r for r in aimed if r["servo_located"]]
        lost = len(aimed) - len(located)
        blind = sum(r["servo_yaw_deg"] == "" for r in located)
        print(
            f"  servo: tried {len(aimed)} decisions; location failed {lost} "
            f"({fmt(lost / len(aimed))}); located but yaw None {blind} "
            f"({fmt(blind / len(located) if located else math.nan)}); median inliers located "
            f"{fmt(median(located, 'servo_located_inliers'), 0)}, target "
            f"{fmt(median(located, 'servo_target_inliers'), 0)}"
        )

    edges = [r["stall_edge"] for r in rows if r["stall_edge"]]
    print(f"  stalls: {len(edges)}" + (f" (penalised edges {' '.join(edges)})" if edges else ""))

    decisions, ticks, first, last = longest_stall(rows)
    print(
        f"  longest stretch without progress: {decisions} decisions, {ticks} ticks "
        f"(decisions {first}-{last})"
    )

    fired = np.array([r["checkin"] for r in rows], bool)
    truth = np.array([r["at_goal"] for r in rows], bool)
    precision, recall = precision_recall(fired, truth)
    print(
        f"  check-in rule: fired {int(fired.sum())}, at goal {int(truth.sum())}, "
        f"precision {fmt(precision)}, recall {fmt(recall)}"
    )
    print(f"  {'goal-sim threshold':>18} {'fired':>6} {'precision':>9} {'recall':>7}")
    for row in swept:
        print(
            f"  {row['threshold']:>18} {row['fired']:>6} {row['precision']:>9} {row['recall']:>7}"
        )


def sweep_row(seeds: list[int], tau: float, conf: float, rows: list[dict]) -> dict:
    stats = filter_stats(rows)
    fired = np.array([r["checkin"] for r in rows], bool)
    truth = np.array([r["at_goal"] for r in rows], bool)
    precision, _ = precision_recall(fired, truth)
    final = [
        float(maze[-1]["goal_geodesic_m"])
        for seed in seeds
        if (maze := [r for r in rows if r["seed"] == seed])
    ]
    return {
        "mazes": " ".join(map(str, seeds)),
        "tau": tau,
        "conf_threshold": conf,
        "decisions": len(rows),
        "raw_pos_err_med_m": fmt(stats["raw"]),
        "filter_pos_err_med_m": fmt(stats["filter"]),
        "low_conf_frac": fmt(stats["low_conf_frac"]),
        "filter_err_confident_med_m": fmt(stats["confident"]),
        "filter_err_low_conf_med_m": fmt(stats["low_conf"]),
        "checkins": int(fired.sum()),
        "checkin_precision": fmt(precision),
        "final_geodesic_mean_m": fmt(float(np.mean(final)) if final else math.nan, 2),
    }


def filter_sweep(args, seeds: list[int]) -> None:
    """Every (tau, conf_threshold) on every maze, one agent per maze; one summary row per pair."""
    taus = floats(args.tau_sweep) if args.tau_sweep else [args.tau]
    confs = floats(args.conf_sweep) if args.conf_sweep else [args.conf_threshold]
    combos = list(itertools.product(taus, confs))
    rows: dict[tuple[float, float], list[dict]] = {combo: [] for combo in combos}
    for seed in seeds:
        agent = build_agent(
            seed, args, ticks=args.ticks, **filter_kwargs(args), **control_kwargs(args)
        )
        for tau, conf in combos:
            print(f"maze {seed}: tau {tau}, conf_threshold {conf}", flush=True)
            agent.filter.tau = tau
            agent.conf_threshold = conf
            rows[tau, conf].extend(evaluate(seed, args, agent))
    out = [sweep_row(seeds, tau, conf, rows[tau, conf]) for tau, conf in combos]
    write_csv(Path(args.out_dir) / "filter_sweep.csv", FILTER_SWEEP_FIELDS, out)
    print(f"  {'tau':>6} {'conf':>5} {'raw':>6} {'filter':>6} {'low':>6} {'sure':>6} {'unsure':>6}")
    for row in out:
        print(
            f"  {row['tau']:>6} {row['conf_threshold']:>5} {row['raw_pos_err_med_m']:>6} "
            f"{row['filter_pos_err_med_m']:>6} {row['low_conf_frac']:>6} "
            f"{row['filter_err_confident_med_m']:>6} {row['filter_err_low_conf_med_m']:>6}"
        )


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--max-decisions", type=int, default=2000, help="act() calls per maze")
    p.add_argument("--ticks", type=int, default=5, help="ticks per move")
    p.add_argument("--motion-seed", type=int, default=1, help="motion noise seed")
    p.add_argument("--checkin-distance", type=float, default=0.2, help="at goal within, metres")
    p.add_argument("--checkin-heading", type=float, default=30.0, help="and within, degrees")
    p.add_argument("--stop-at-checkin", action="store_true", help="end at the first check-in")
    add_filter_args(p)
    add_control_args(p)
    p.add_argument("--tau-sweep", default="", help='e.g. "0.005,0.01,0.02,0.05"')
    p.add_argument("--conf-sweep", default="", help='e.g. "0.3,0.5,0.7"')
    args = p.parse_args()

    if args.tau_sweep or args.conf_sweep:
        if not args.filter:
            p.error("--tau-sweep and --conf-sweep need the filter")
        filter_sweep(args, ints(args.mazes))
        return

    rows, swept = [], []
    summaries = []
    for seed in ints(args.mazes):
        agent = build_agent(
            seed, args, ticks=args.ticks, **filter_kwargs(args), **control_kwargs(args)
        )
        maze = evaluate(seed, args, agent)
        maze_sweep = sweep(seed, maze)
        summaries.append((seed, maze, maze_sweep))
        rows.extend(maze)
        swept.extend(maze_sweep)
    out = Path(args.out_dir)
    write_csv(out / "eval_control.csv", FIELDS, rows)
    write_csv(out / "eval_control_sweep.csv", SWEEP_FIELDS, swept)
    for seed, maze, maze_sweep in summaries:
        summarise(seed, maze, maze_sweep)


if __name__ == "__main__":
    main()
