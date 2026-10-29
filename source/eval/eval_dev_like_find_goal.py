"""Why find_goal picks the place it does, on the dev-like maze's goals.

    uv run source/eval/eval_dev_like_find_goal.py --goals 1,4
    uv run source/eval/eval_dev_like_find_goal.py --goals 1,4 --goal-camera frame

For each goal (the same goals as eval_dev_like.py: same seed, same choice) the goal's four photos
go through goal.find_goal with ``--goal-camera`` (photo: the lower camera of the goal photos), and
this writes results/eval/dev_like/<seed>-find-goal/goal_<id>.md and candidates_<id>.csv:

- the candidates find_goal scored (its CANDIDATES best (frame, view) pairs by inliers): rank,
  route, record (the frame's index in its route), view, inliers, layout score, and the pose error
  against the map-consistent truth (MapTruth.consistent: the goal's true pose as the map has it)
  in metres and degrees; the one find_goal chose (layout >= LAYOUT_MIN first, then inliers) is
  marked;
- per view k, the VLAD ranking of all mapped frames by the photo's descriptors: the rank of the
  mapped frame nearest the true goal position, and the best rank of a frame that truly sees the
  view (true position within SEES_M of the goal's, true yaw within SEES_DEG of the view's
  direction, goal yaw + 90 degrees * k), against TOP_K, the frames that go on to the SIFT rerank;
- whether the true place appears among the candidates at all (a candidate whose pose is within
  PLACE_M of the consistent truth, whatever its yaw), and among the VLAD top-K of any view.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import vis_nav_sim as sim
from common import textures, write_csv
from eval_map import MapTruth, true_poses

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frozen_map  # noqa: E402
import geometry  # noqa: E402
import goal as goal_module  # noqa: E402
import map_agent  # noqa: E402
import mapping  # noqa: E402
from eval_dev_like import (  # noqa: E402
    MSAA,
    OUT_ROOT,
    TARGET_HEIGHT_OFFSET,
    DevSimulator,
    choose_goals,
    explore,
)
from geometry import Camera, FloorMask, wall_view  # noqa: E402

SEES_M = 0.3
SEES_DEG = 45.0
PLACE_M = goal_module.DISTINCT_M
FIELDS = [
    "rank",
    "chosen",
    "route",
    "record",
    "view",
    "inliers",
    "layout",
    "x",
    "y",
    "yaw_deg",
    "pos_err_m",
    "yaw_err_deg",
]


def md(header: list[str], rows: list[list]) -> str:
    sep = "|" + "|".join("---" for _ in header) + "|"
    body = ("| " + " | ".join(map(str, r)) + " |" for r in rows)
    return "\n".join(["| " + " | ".join(header) + " |", sep, *body])


def vlad_ranks(views, store, camera, floor, truth: MapTruth, goal_pose, consistent) -> tuple:
    """Per view k: ``(rank of the nearest frame, its (m, deg) from the view's pose, best rank of
    a frame that sees the view, that frame's row, the rank-1 frame's row)``, ranks from 1; and
    the rows of each view's VLAD top-K."""
    sift = cv2.SIFT_create()
    seeing = goal_module.target_camera(camera)
    positions, yaws = truth.poses[:, :2], truth.poses[:, 2]
    out, tops = [], []
    for k, photo in enumerate(views):
        view = wall_view(photo, seeing, floor, sift)
        if view.des is None:
            out.append(None)
            tops.append(set())
            continue
        similarity = store.vlad @ frozen_map.vlad(view.des, store.centers)
        order = np.argsort(-similarity)
        rank = np.empty(len(order), int)
        rank[order] = np.arange(1, len(order) + 1)
        look = goal_pose[2] + k * math.pi / 2
        gap = np.hypot(*(positions - goal_pose[:2]).T)
        turn = np.degrees(np.abs(mapping.wrap(yaws - look)))
        nearest = int(gap.argmin())
        sees = np.flatnonzero((gap <= SEES_M) & (turn <= SEES_DEG))
        best = int(sees[rank[sees].argmin()]) if len(sees) else None
        out.append(
            {
                "nearest": nearest,
                "nearest_rank": int(rank[nearest]),
                "nearest_gap": (float(gap[nearest]), float(turn[nearest])),
                "sees": len(sees),
                "best": best,
                "best_rank": None if best is None else int(rank[best]),
                "best_gap": None if best is None else (float(gap[best]), float(turn[best])),
                "top1": int(order[0]),
                "top1_gap": (float(gap[order[0]]), float(turn[order[0]])),
                "similarity": (
                    float(similarity[order[0]]),
                    None if best is None else float(similarity[best]),
                ),
            }
        )
        tops.append(set(order[: goal_module.TOP_K].tolist()))
    return out, tops


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--goals", default="1,4", help="comma-separated goal ids")
    p.add_argument("--count", type=int, default=10, help="goals the maze has (as eval_dev_like)")
    p.add_argument("--goal-camera", choices=("photo", "frame"), default="photo")
    args = p.parse_args()
    wanted = [int(g) for g in args.goals.split(",") if g]
    out = OUT_ROOT / f"{args.seed}-find-goal"
    out.mkdir(parents=True, exist_ok=True)

    world = DevSimulator(
        textures(), args.seed, msaa=MSAA, motion_noise=sim.DEFAULT_NOISE, motion_seed=args.seed
    )
    root, _ = explore(args.seed, world)
    routes = mapping.load_routes(root)
    truth_poses = true_poses(root, routes)
    floor = FloorMask.load(map_agent.FLOOR_MASK_PATH)
    camera_dict = dict(geometry.CAMERA)
    camera = Camera({**camera_dict, "cy": camera_dict["cy"] + mapping.HORIZON_SHIFT_PX})
    store, _ = frozen_map.get(root, camera_dict, floor)
    truth = MapTruth(store, truth_poses)
    photo_world = DevSimulator(
        textures(),
        args.seed,
        msaa=MSAA,
        camera={**world.camera, "height_offset": TARGET_HEIGHT_OFFSET},
    )
    goals = choose_goals(
        world, photo_world, truth_poses, SimpleNamespace(seed=args.seed, goals=args.count)
    )

    for goal in goals:
        if goal.id not in wanted:
            continue
        consistent = truth.consistent(goal.pose)
        ranking: list[goal_module.Candidate] = []
        found = goal_module.find_goal(
            goal.photos,
            store,
            camera,
            floor,
            ranking=ranking,
            photographed=args.goal_camera == "photo",
        )
        chosen = None if found is None else (found.frame, found.k)
        rows = []
        for rank, c in enumerate(ranking, 1):
            pos = math.hypot(c.x - consistent[0], c.y - consistent[1])
            yaw = abs(math.degrees(mapping.wrap(c.yaw - consistent[2])))
            rows.append(
                {
                    "rank": rank,
                    "chosen": int((c.frame, c.k) == chosen),
                    "route": int(store.route[c.frame]),
                    "record": int(store.frame[c.frame]),
                    "view": c.k,
                    "inliers": c.inliers,
                    "layout": c.layout,
                    "x": c.x,
                    "y": c.y,
                    "yaw_deg": math.degrees(c.yaw),
                    "pos_err_m": pos,
                    "yaw_err_deg": yaw,
                }
            )
        write_csv(out / f"candidates_{goal.id}.csv", FIELDS, rows)
        ranks, tops = vlad_ranks(
            goal.photos, store, camera, floor, truth, np.array(goal.pose), consistent
        )

        lines = [
            f"# find_goal ranking, goal {goal.id} ({goal.bucket}), seed {args.seed}, "
            f"--goal-camera {args.goal_camera}",
            "",
            f"True goal pose (world) ({goal.pose[0]:.2f}, {goal.pose[1]:.2f}) "
            f"{math.degrees(goal.pose[2]):.0f} deg; as the map has it (map-consistent) "
            f"({consistent[0]:.2f}, {consistent[1]:.2f}) {math.degrees(consistent[2]):.0f} deg.",
            "",
        ]
        if found is None:
            lines.append("find_goal returned None.")
        else:
            lines.append(
                f"find_goal chose ({found.x:.2f}, {found.y:.2f}) {math.degrees(found.yaw):.0f} deg: "  # noqa: E501
                f"route {int(store.route[found.frame])} record {int(store.frame[found.frame])} "
                f"view {found.k}, {found.inliers} inliers, layout {found.layout:.2f}, confidence "
                f"{found.confidence}, {found.view_agreement} views agree; error "
                f"{math.hypot(found.x - consistent[0], found.y - consistent[1]):.2f} m, "
                f"{abs(math.degrees(mapping.wrap(found.yaw - consistent[2]))):.0f} deg."
            )
        lines += [
            "",
            "## Candidates (best by inliers first; layout >= "
            f"{goal_module.LAYOUT_MIN} beats any that is under)",
            "",
            md(
                [
                    "#",
                    "chosen",
                    "route",
                    "record",
                    "view",
                    "inliers",
                    "layout",
                    "pose (x, y, yaw)",
                    "pos err m",
                    "yaw err deg",
                ],
                [
                    [
                        r["rank"],
                        "**yes**" if r["chosen"] else "",
                        r["route"],
                        r["record"],
                        r["view"],
                        r["inliers"],
                        f"{r['layout']:.2f}",
                        f"({r['x']:.2f}, {r['y']:.2f}, {r['yaw_deg']:.0f})",
                        f"{r['pos_err_m']:.2f}",
                        f"{r['yaw_err_deg']:.0f}",
                    ]
                    for r in rows
                ],
            ),
            "",
            f"## VLAD rank of the true place, per view (of {len(store.xy)} mapped frames; "
            f"top {goal_module.TOP_K} go on to the SIFT rerank)",
            "",
            f"'Nearest' is the mapped frame nearest the true goal position, with its true gap "
            f"(m / deg off the view's direction); 'sees' is a frame within {SEES_M} m of it and "
            f"{SEES_DEG:.0f} deg of the view's direction (how many there are; its best rank). "
            "'Top 1' is the frame VLAD put first, with its true gap to the goal.",
            "",
        ]
        table = []
        for k, r in enumerate(ranks):
            if r is None:
                table.append([k, "no features", "", "", "", ""])
                continue
            table.append(
                [
                    k,
                    f"{r['nearest_rank']} ({r['nearest_gap'][0]:.2f} m / {r['nearest_gap'][1]:.0f} deg)",  # noqa: E501
                    "none"
                    if r["best"] is None
                    else f"{r['best_rank']} of {r['sees']} ({r['best_gap'][0]:.2f} m / {r['best_gap'][1]:.0f} deg)",  # noqa: E501
                    "yes"
                    if r["best_rank"] is not None and r["best_rank"] <= goal_module.TOP_K
                    else "no",
                    f"{r['top1_gap'][0]:.2f} m / {r['top1_gap'][1]:.0f} deg",
                    f"{r['similarity'][0]:.3f}"
                    + ("" if r["similarity"][1] is None else f" vs {r['similarity'][1]:.3f}"),
                ]
            )
        lines.append(
            md(
                [
                    "view",
                    "nearest frame's rank",
                    "best 'sees' rank",
                    "in top K",
                    "top 1 true gap",
                    "similarity top 1 vs best sees",
                ],
                table,
            )
        )
        near = [c for c in rows if c["pos_err_m"] <= PLACE_M]
        in_top = [
            k
            for k, r in enumerate(ranks)
            if r and r["best_rank"] is not None and r["best_rank"] <= goal_module.TOP_K
        ]
        lines += [
            "",
            "## Does the true place appear?",
            "",
            f"- Among the {len(rows)} candidates, within {PLACE_M} m of the map-consistent truth "
            f"(any yaw): **{'yes, ' + str(len(near)) + ' (ranks ' + ', '.join(str(c['rank']) for c in near) + ')' if near else 'no'}**."  # noqa: E501
            + (
                f" Best yaw error among them {min(c['yaw_err_deg'] for c in near):.0f} deg."
                if near
                else ""
            ),
            f"- In a view's VLAD top {goal_module.TOP_K} (a frame that sees the view): "
            f"**{'views ' + ', '.join(map(str, in_top)) if in_top else 'no view'}**.",
        ]
        text = "\n".join(lines) + "\n"
        (out / f"goal_{goal.id}.md").write_text(text)
        print(text)


if __name__ == "__main__":
    main()
