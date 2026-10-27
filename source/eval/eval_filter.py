"""
Offline localisation on a fixed test route: the robot drives through --n-waypoints random free
cells (record_gt's driver, motion noise seeded by --motion-seed), and every (tau, conf_threshold)
in the sweeps replays the same route through the node filter, against the raw best match.

    uv run source/eval/eval_filter.py
    uv run source/eval/eval_filter.py --mazes 7 --tau-sweep 0.01,0.02 --conf-sweep 0.5

The route is cached in data/gt/<seed>-test.npz (frames as JPEG, as record_gt saves them) and
re-recorded when its settings change. At each decision the filter predicts with the action
taken since the previous one, then updates with the view's similarities. A node's error is
the distance from its recorded pose to the true pose; it is within tolerance within --pos-tol
and --yaw-tol. The filtered node is the filter's estimate whatever its confidence; a decision
is low_conf when the confidence is below the threshold. Results are pooled over mazes.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as sim
from common import (
    GT_ROOT,
    bfs,
    build_agent,
    cell_of,
    floats,
    fmt,
    ints,
    node_poses,
    parser,
    textures,
    write_csv,
    yaw_error_deg,
)
from record_gt import ACTIONS, Driver, route
from tqdm import tqdm

MAX_DECISIONS_PER_WAYPOINT = 300
"""A waypoint the robot has not reached in this many decisions is given up on."""
FIELDS = [
    "tau",
    "conf_threshold",
    "mazes",
    "decisions",
    "raw_err_med_m",
    "filter_err_med_m",
    "raw_yaw_med_deg",
    "filter_yaw_med_deg",
    "raw_within",
    "filter_within",
    "low_conf_frac",
    "err_confident_med_m",
    "err_low_conf_med_m",
    "confident_precision",
]


def route_settings(args) -> np.ndarray:
    return np.array([args.n_waypoints, args.seed, args.motion_seed, args.ticks])


def record_route(seed: int, args) -> dict[str, np.ndarray]:
    """Drive to each waypoint in turn: per decision the JPEG frame, the true pose, and the
    action taken from it."""
    world = sim.Simulator(
        textures(), seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=args.motion_seed
    )
    free = ~world.occupancy
    cells = list(bfs(free, tuple(world.spawn_cell)))
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(cells), min(args.n_waypoints, len(cells)), replace=False)
    driver = Driver(world, args.ticks)

    frame = world.initial_frame()
    jpegs: list[bytes] = []
    poses: list[tuple[float, float, float]] = []
    actions: list[str] = []
    for waypoint in (cells[i] for i in picks):
        x, y, _ = world.pose()
        start = cell_of(world, free, x, y)
        path = route(bfs(free, start), start, waypoint)
        for _ in range(MAX_DECISIONS_PER_WAYPOINT):
            x, y, yaw = world.pose()
            driver.advance(path, x, y)
            if not path:
                break
            action = driver.action(path[0], x, y, yaw)
            jpegs.append(world.jpeg(frame))
            poses.append((x, y, yaw))
            actions.append(ACTIONS[action])
            frame = world.step(action, args.ticks)[-1]
            driver.moved(action, x, y)
        else:
            print(f"maze {seed}: gave up on waypoint {waypoint}")
    return {
        "jpeg": np.frombuffer(b"".join(jpegs), np.uint8),
        "offsets": np.cumsum([0, *map(len, jpegs)]),
        "poses": np.array(poses),
        "actions": np.array(actions),
        "settings": route_settings(args),
    }


def load_route(seed: int, args) -> dict[str, np.ndarray]:
    path = GT_ROOT / f"{seed}-test.npz"
    if path.exists():
        with np.load(path) as cached:
            if np.array_equal(cached["settings"], route_settings(args)):
                return dict(cached)
    print(f"recording {path} ...", flush=True)
    recorded = record_route(seed, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **recorded)
    return recorded


def evaluate(seed: int, args, taus: list[float]) -> dict:
    """Position and yaw errors of the raw best match, and of the filter's estimate with its
    confidence per tau, at every decision of the route."""
    agent = build_agent(
        seed, args, hops=args.hops, p_stay=args.p_stay, action_boost=args.action_boost, eps=args.eps
    )
    poses = node_poses(agent.frames)
    test = load_route(seed, args)
    jpeg, offsets, actions = test["jpeg"], test["offsets"], test["actions"]
    truth = test["poses"]
    print(f"maze {seed}: {len(truth)} decisions on the test route")

    queries = np.stack(
        [
            agent.extractor.extract(cv2.imdecode(jpeg[a:b], cv2.IMREAD_COLOR))
            for a, b in tqdm(zip(offsets[:-1], offsets[1:], strict=True), desc="queries")
        ]
    )
    similarity = queries @ agent.database.T

    def errors(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position = np.hypot(poses[nodes, 0] - truth[:, 0], poses[nodes, 1] - truth[:, 1])
        return position, yaw_error_deg(poses[nodes, 2], truth[:, 2])

    out = {"raw": errors(similarity.argmax(axis=1)), "filter": {}}
    node_filter = agent.filter
    for tau in taus:
        node_filter.tau = tau
        node_filter.reset()
        nodes, confidence = [], []
        for decision, row in enumerate(similarity):
            if decision:
                node_filter.predict(str(actions[decision - 1]))
            node_filter.update(row)
            node, conf = node_filter.estimate()
            nodes.append(node)
            confidence.append(conf)
        out["filter"][tau] = (*errors(np.array(nodes)), np.array(confidence))
    return out


def median(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else math.nan


def summarise(seeds, taus, confs, results: list[dict], args) -> list[dict]:
    raw_pos = np.concatenate([r["raw"][0] for r in results])
    raw_yaw = np.concatenate([r["raw"][1] for r in results])
    raw_within = (raw_pos <= args.pos_tol) & (raw_yaw <= args.yaw_tol)
    rows = []
    for tau, conf in itertools.product(taus, confs):
        pos, yaw, confidence = (
            np.concatenate([r["filter"][tau][k] for r in results]) for k in range(3)
        )
        within = (pos <= args.pos_tol) & (yaw <= args.yaw_tol)
        low = confidence < conf
        rows.append(
            {
                "tau": tau,
                "conf_threshold": conf,
                "mazes": " ".join(map(str, seeds)),
                "decisions": len(pos),
                "raw_err_med_m": fmt(median(raw_pos)),
                "filter_err_med_m": fmt(median(pos)),
                "raw_yaw_med_deg": fmt(median(raw_yaw), 1),
                "filter_yaw_med_deg": fmt(median(yaw), 1),
                "raw_within": fmt(raw_within.mean()),
                "filter_within": fmt(within.mean()),
                "low_conf_frac": fmt(low.mean()),
                "err_confident_med_m": fmt(median(pos[~low])),
                "err_low_conf_med_m": fmt(median(pos[low])),
                "confident_precision": fmt(within[~low].mean() if (~low).any() else math.nan),
            }
        )
    return rows


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--n-waypoints", type=int, default=15, help="random free cells to visit")
    p.add_argument("--seed", type=int, default=1, help="waypoint seed")
    p.add_argument("--motion-seed", type=int, default=2, help="motion noise seed")
    p.add_argument("--ticks", type=int, default=5, help="ticks per decision")
    p.add_argument("--tau-sweep", default="0.005,0.01,0.02,0.05", help="filter temperatures")
    p.add_argument("--conf-sweep", default="0.3,0.5,0.7", help="confidence thresholds")
    p.add_argument("--pos-tol", type=float, default=0.4, help="within tolerance, metres")
    p.add_argument("--yaw-tol", type=float, default=45.0, help="and degrees")
    p.add_argument("--hops", type=int, default=2, help="filter: reach of one move")
    p.add_argument("--p-stay", type=float, default=0.5, help="filter: weight of staying")
    p.add_argument(
        "--action-boost", type=float, default=3.0, help="filter: weight of the taken move's edge"
    )
    p.add_argument("--eps", type=float, default=1e-3, help="filter: uniform mixing per move")
    args = p.parse_args()

    seeds, taus, confs = ints(args.mazes), floats(args.tau_sweep), floats(args.conf_sweep)
    results = [evaluate(seed, args, taus) for seed in seeds]
    rows = summarise(seeds, taus, confs, results, args)
    write_csv(Path(args.out_dir) / "eval_filter.csv", FIELDS, rows)

    columns = [
        ("tau", "tau"),
        ("conf", "conf_threshold"),
        ("raw err", "raw_err_med_m"),
        ("filt err", "filter_err_med_m"),
        ("raw in", "raw_within"),
        ("filt in", "filter_within"),
        ("low", "low_conf_frac"),
        ("sure err", "err_confident_med_m"),
        ("low err", "err_low_conf_med_m"),
        ("sure prec", "confident_precision"),
    ]
    print(f"\n{rows[0]['decisions']} decisions on mazes {rows[0]['mazes']}")
    print("  " + " ".join(f"{name:>9}" for name, _ in columns))
    for row in sorted(rows, key=lambda r: float(r["filter_within"]), reverse=True):
        print("  " + " ".join(f"{row[key]!s:>9}" for _, key in columns))


if __name__ == "__main__":
    main()
