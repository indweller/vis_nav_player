"""
How good is the goal node? Scores every node against the target views, three ways: the front
view only (what MyAgent does), the best of the four views, and the sum over the four, and
compares the chosen node's recorded pose to world.target.

    uv run source/eval/eval_goal.py
    uv run source/eval/eval_goal.py --mazes 7 --subsample 1

"Closest" is the node with the smallest pose distance to the target (position error plus yaw
error at 0.5 m per radian); its rank is its place in the method's similarity ordering.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import vis_nav_sim as sim
from common import (
    build_agent,
    fmt,
    ints,
    node_poses,
    parser,
    pose_distance,
    textures,
    write_csv,
    yaw_error_deg,
)

FIELDS = [
    "seed",
    "method",
    "n_nodes",
    "chosen",
    "chosen_view",
    "pos_err_m",
    "yaw_err_deg",
    "chosen_similarity",
    "closest",
    "closest_pos_err_m",
    "closest_yaw_err_deg",
    "closest_rank",
    "closest_similarity",
]


def evaluate(seed: int, args) -> list[dict]:
    agent = build_agent(seed, args)
    poses = node_poses(agent.frames)
    world = sim.Simulator(textures(), seed)
    tx, ty, tyaw = world.target

    views = np.stack([agent.extractor.extract(view) for view in world.targets()])
    similarity = agent.database @ views.T
    methods = {
        "front": (similarity[:, 0], np.zeros(len(similarity), int)),
        "max4": (similarity.max(axis=1), similarity.argmax(axis=1)),
        "sum4": (similarity.sum(axis=1), np.full(len(similarity), -1)),
    }

    pos_err = np.hypot(poses[:, 0] - tx, poses[:, 1] - ty)
    yaw_err = yaw_error_deg(poses[:, 2], tyaw)
    closest = int(np.argmin(pose_distance(pos_err, yaw_err)))

    rows = []
    for method, (score, view) in methods.items():
        chosen = int(np.argmax(score))
        rank = int(np.flatnonzero(np.argsort(-score, kind="stable") == closest)[0]) + 1
        rows.append(
            {
                "seed": seed,
                "method": method,
                "n_nodes": len(poses),
                "chosen": chosen,
                "chosen_view": int(view[chosen]),
                "pos_err_m": f"{pos_err[chosen]:.3f}",
                "yaw_err_deg": f"{yaw_err[chosen]:.1f}",
                "chosen_similarity": f"{score[chosen]:.4f}",
                "closest": closest,
                "closest_pos_err_m": f"{pos_err[closest]:.3f}",
                "closest_yaw_err_deg": f"{yaw_err[closest]:.1f}",
                "closest_rank": rank,
                "closest_similarity": f"{score[closest]:.4f}",
            }
        )
    return rows


def summarise(seed: int, rows: list[dict]) -> None:
    first = rows[0]
    print(
        f"\nmaze {seed}: {first['n_nodes']} nodes; closest node {first['closest']} is "
        f"{first['closest_pos_err_m']} m / {first['closest_yaw_err_deg']} deg off the target"
    )
    print(f"  {'method':<6} {'node':>5} {'pos m':>7} {'yaw deg':>8} {'rank':>6} sim chosen/closest")
    for r in rows:
        print(
            f"  {r['method']:<6} {r['chosen']:>5} {r['pos_err_m']:>7} {r['yaw_err_deg']:>8} "
            f"{r['closest_rank']:>6} {r['chosen_similarity']} / {r['closest_similarity']}"
        )


def main() -> None:
    args = parser(__doc__).parse_args()
    rows = []
    for seed in ints(args.mazes):
        maze = evaluate(seed, args)
        summarise(seed, maze)
        rows.extend(maze)
    write_csv(Path(args.out_dir) / "eval_goal.csv", FIELDS, rows)
    for method in ("front", "max4", "sum4"):
        errors = [float(r["pos_err_m"]) for r in rows if r["method"] == method]
        if errors:
            print(f"{method}: mean position error {fmt(sum(errors) / len(errors))} m")


if __name__ == "__main__":
    main()
