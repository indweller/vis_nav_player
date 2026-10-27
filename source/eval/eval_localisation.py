"""
Place recognition against ground truth: render --n random poses (a random free cell, jittered by
up to a quarter cell, with a random yaw), query the database, and check the recorded poses of the
top-k nodes.

    uv run source/eval/eval_localisation.py
    uv run source/eval/eval_localisation.py --mazes 7 --n 2000 --pos-tol 0.3

A query is a hit at k when any of its top k nodes is within --pos-tol and --yaw-tol of it;
"possible" counts queries with any such node in the database at all. The descriptor AUC is of
query-node similarity: pairs within tolerance against pairs more than FAR_M apart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import vis_nav_sim as sim
from common import (
    bfs,
    build_agent,
    cell_centre,
    fmt,
    ints,
    node_poses,
    parser,
    textures,
    write_csv,
    yaw_error_deg,
)
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

KS = (1, 5, 20)
FAR_M = 2.0
FIELDS = [
    "seed",
    "query",
    "x",
    "y",
    "yaw_deg",
    "top1",
    "top1_pos_err_m",
    "top1_yaw_err_deg",
    "top1_similarity",
    *(f"hit@{k}" for k in KS),
    "possible",
    "n_within_tol",
]


def evaluate(seed: int, args) -> tuple[list[dict], float]:
    agent = build_agent(seed, args)
    poses = node_poses(agent.frames)
    world = sim.Simulator(textures(), seed)
    cells = list(bfs(~world.occupancy, tuple(world.spawn_cell)))
    rng = np.random.default_rng(args.seed)
    jitter = world.cell_size / 4

    rows = []
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    for query in tqdm(range(args.n), desc=f"maze {seed}"):
        cx, cy = cell_centre(world, cells[rng.integers(len(cells))])
        x, y = cx + rng.uniform(-jitter, jitter), cy + rng.uniform(-jitter, jitter)
        yaw = rng.uniform(-np.pi, np.pi)
        similarity = agent.database @ agent.extractor.extract(world.render(x, y, yaw))

        pos_err = np.hypot(poses[:, 0] - x, poses[:, 1] - y)
        yaw_err = yaw_error_deg(poses[:, 2], yaw)
        within = (pos_err <= args.pos_tol) & (yaw_err <= args.yaw_tol)
        positives.append(similarity[within])
        negatives.append(similarity[pos_err > FAR_M])

        order = np.argsort(-similarity)[: max(KS)]
        top1 = int(order[0])
        rows.append(
            {
                "seed": seed,
                "query": query,
                "x": f"{x:.3f}",
                "y": f"{y:.3f}",
                "yaw_deg": f"{np.degrees(yaw):.1f}",
                "top1": top1,
                "top1_pos_err_m": f"{pos_err[top1]:.3f}",
                "top1_yaw_err_deg": f"{yaw_err[top1]:.1f}",
                "top1_similarity": f"{similarity[top1]:.4f}",
                **{f"hit@{k}": int(within[order[:k]].any()) for k in KS},
                "possible": int(within.any()),
                "n_within_tol": int(within.sum()),
            }
        )

    positive, negative = np.concatenate(positives), np.concatenate(negatives)
    auc = float("nan")
    if len(positive) and len(negative):
        labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
        auc = float(roc_auc_score(labels, np.r_[positive, negative]))
    return rows, auc


def summarise(seed: int, rows: list[dict], auc: float) -> None:
    def rate(key: str) -> str:
        return fmt(np.mean([r[key] for r in rows]))

    recalls = "  ".join(f"recall@{k}={rate(f'hit@{k}')}" for k in KS)
    median = np.median([float(r["top1_pos_err_m"]) for r in rows])
    print(
        f"\nmaze {seed}: {len(rows)} queries  {recalls}  (possible {rate('possible')})\n"
        f"  top-1 median position error {fmt(median)} m   descriptor AUC "
        f"(within tol vs > {FAR_M:g} m) {fmt(auc)}"
    )


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--n", type=int, default=500, help="queries per maze")
    p.add_argument("--pos-tol", type=float, default=0.4, help="hit radius, metres")
    p.add_argument("--yaw-tol", type=float, default=45.0, help="hit yaw tolerance, degrees")
    p.add_argument("--seed", type=int, default=0, help="query sampling seed")
    args = p.parse_args()

    rows = []
    summaries = []
    for seed in ints(args.mazes):
        maze, auc = evaluate(seed, args)
        summaries.append((seed, maze, auc))
        rows.extend(maze)
    write_csv(Path(args.out_dir) / "eval_localisation.csv", FIELDS, rows)
    for seed, maze, auc in summaries:
        summarise(seed, maze, auc)


if __name__ == "__main__":
    main()
