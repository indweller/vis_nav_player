"""
Goal identification on the frozen pose-free map, against the ground truth. No controller.

    uv run source/eval/eval_goal_map.py
    uv run source/eval/eval_goal_map.py --mazes 7 --n 10
    uv run source/eval/eval_goal_map.py --rebuild

Per maze (7-9 are the development mazes, 10-12 held out) the exploration data is recorded into
data/eval/<seed>/ as eval_map does, mapping.build's graph variant is frozen to
data/eval/<seed>/map_graph.npz (frozen_map.py; --rebuild rebuilds it), and goal.find_goal is
run on the views of
  - the maze's real target (the four images of world.targets()), and
  - --n goals at random free cell centres, yaw one of 0/90/180/270 degrees, their views
    rendered by world.render at yaw + 90 k degrees and JPEG-encoded as the server does.
find_goal sees the views, the frozen map and the camera only. Truth is moved into the map frame
by route 0's first true pose (poses.json), the map frame being the spawn's.

Per sample (results/eval/goal/goals.csv): position and yaw error; success at 0.2 m & 30 degrees
(the scoring one) and at 0.4 m & 30; the chosen view k; the retrieval error (true distance from
the chosen frame to the goal); the mapping error (the chosen frame's mapped pose against its
true one); the relative pose's error (against the true pose of view k's camera in the frame's
camera frame); the confidence.

Plots, in results/eval/goal/: <seed>_goals.png (the map with the true walls, a true goal arrow
in green and an estimated one in red per sample, joined by a line) and <seed>_examples.png (six
samples, two each of the best, median and worst by position error: target view k | the matched
frame | the errors).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as sim
from common import cell_centre, fmt, ints, textures, wrap, write_csv, yaw_error_deg
from eval_cues import Walls, stored_floor
from eval_geometry import Frame
from eval_map import MAP_COLOUR, Canvas, record, true_poses

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frozen_map  # noqa: E402
import goal  # noqa: E402
import mapping  # noqa: E402
from geometry import Camera  # noqa: E402

SUCCESS_M, SUCCESS_DEG = 0.2, 30.0
LOOSE_M = 0.4
YAWS = (0.0, 90.0, 180.0, 270.0)
SPLITS = {"7-9": (7, 8, 9), "10-12": (10, 11, 12)}
CONFIDENCE_BINS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 10**9)]
FIELDS = [
    "seed",
    "sample",
    "real_target",
    "true_x",
    "true_y",
    "true_yaw_deg",
    "est_x",
    "est_y",
    "est_yaw_deg",
    "pos_err_m",
    "yaw_err_deg",
    "success_0.2",
    "success_0.4",
    "k",
    "inliers",
    "confidence",
    "chosen_route",
    "chosen_frame",
    "retrieval_err_m",
    "mapping_err_m",
    "mapping_yaw_err_deg",
    "relative",
    "rel_pos_err_m",
    "rel_yaw_err_deg",
]


def jpeg(world: sim.Simulator, image: np.ndarray) -> np.ndarray:
    """``image`` as the server would send it, decoded again."""
    return cv2.imdecode(np.frombuffer(world.jpeg(image), np.uint8), cv2.IMREAD_COLOR)


def check_render(world: sim.Simulator) -> float:
    """Mean absolute pixel difference of world.render at world.target (JPEG-encoded) from
    world.targets()[0]. The two are not the same image (the walls' texture detail and the
    floor differ though the geometry is exact), so the real target's samples use targets()."""
    target = world.targets()[0].astype(int)
    return float(np.abs(jpeg(world, world.render(*world.target)).astype(int) - target).mean())


def samples(world: sim.Simulator, n: int, rng: np.random.Generator) -> list[dict]:
    """The real target, then ``n`` goals: world pose and the four views."""
    out = [{"real": True, "pose": tuple(world.target), "views": world.targets()}]
    free = np.argwhere(~world.occupancy)
    for index in rng.choice(len(free), size=n, replace=False):
        x, y = cell_centre(world, tuple(free[index]))
        yaw = math.radians(float(rng.choice(YAWS)))
        views = [jpeg(world, world.render(x, y, yaw + k * math.pi / 2)) for k in range(4)]
        out.append({"real": False, "pose": (x, y, yaw), "views": views})
    return out


def pose_error(est: tuple, true: tuple) -> tuple[float, float]:
    """Position (m) and yaw (degrees) error."""
    return math.hypot(est[0] - true[0], est[1] - true[1]), float(yaw_error_deg(est[2], true[2]))


def score(
    sample: dict, found: goal.Goal | None, store, frame: Frame, truth: list[np.ndarray]
) -> dict:
    true = tuple(frame.to_map(np.array([sample["pose"]]))[0])
    row = {
        "real_target": int(sample["real"]),
        "true_x": true[0],
        "true_y": true[1],
        "true_yaw_deg": math.degrees(wrap(true[2])),
    }
    if found is None:
        return {
            **row,
            "pos_err_m": math.inf,
            "yaw_err_deg": 180.0,
            "success_0.2": 0,
            "success_0.4": 0,
        }
    est = (found.x, found.y, found.yaw)
    position, yaw = pose_error(est, true)
    route, index = int(store.route[found.frame]), int(store.frame[found.frame])
    true_frame = tuple(frame.to_map(truth[route][index : index + 1])[0])
    mapped = store.pose(found.frame)
    row.update(
        est_x=est[0],
        est_y=est[1],
        est_yaw_deg=math.degrees(wrap(est[2])),
        pos_err_m=position,
        yaw_err_deg=yaw,
        **{
            "success_0.2": int(position <= SUCCESS_M and yaw <= SUCCESS_DEG),
            "success_0.4": int(position <= LOOSE_M and yaw <= SUCCESS_DEG),
        },
        k=found.k,
        inliers=found.inliers,
        confidence=found.confidence,
        chosen_route=route,
        chosen_frame=index,
        retrieval_err_m=math.hypot(true_frame[0] - true[0], true_frame[1] - true[1]),
    )
    row["mapping_err_m"], row["mapping_yaw_err_deg"] = pose_error(mapped, true_frame)
    row["relative"] = int(found.relative is not None)
    if found.relative is not None:
        camera = (true[0], true[1], true[2] + found.k * math.pi / 2)
        row["rel_pos_err_m"], row["rel_yaw_err_deg"] = pose_error(
            found.relative, mapping.between(true_frame, camera)
        )
    return row


# -- images


def draw_goals(world, walls, frame, store, results: list[tuple[dict, dict]], seed, path) -> None:
    canvas = Canvas(world)
    grid = store.grid
    canvas.cells(frame.to_world(grid.centres()[grid.occupied()]), MAP_COLOUR)
    canvas.walls(walls, (0, 0, 0))
    for sample, row in results:
        if "est_x" not in row:
            continue
        a = canvas.pixels(np.array(sample["pose"][:2]))
        estimate = frame.pose_to_world(
            (row["est_x"], row["est_y"], math.radians(row["est_yaw_deg"]))
        )
        b = canvas.pixels(np.array(estimate[:2]))
        cv2.line(
            canvas.image,
            tuple(int(v) for v in np.round(a)),
            tuple(int(v) for v in np.round(b)),
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
    for sample, row in results:
        canvas.marker(sample["pose"], (0, 170, 0), arrow=True)
        if "est_x" in row:
            canvas.marker(
                frame.pose_to_world((row["est_x"], row["est_y"], math.radians(row["est_yaw_deg"]))),
                (0, 0, 220),
                arrow=True,
            )
    wins = sum(row["success_0.2"] for _, row in results)
    canvas.label(f"maze {seed}: true goals green, estimates red; {wins}/{len(results)} at 0.2 m")
    canvas.save(path)


def panel(lines: list[str], size: tuple[int, int] = (320, 240)) -> np.ndarray:
    image = np.full((size[1], size[0], 3), 255, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(image, text, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return image


def draw_examples(store, results: list[tuple[dict, dict]], seed: int, path: Path) -> None:
    found = [(s, r) for s, r in results if "est_x" in r]
    if not found:
        return
    found.sort(key=lambda sr: sr[1]["pos_err_m"])
    mid = len(found) // 2
    chosen = found[:2] + found[max(mid - 1, 0) : mid + 1] + found[-2:]
    rows = []
    for sample, row in chosen:
        target = sample["views"][row["k"]]
        matched = store.image(
            int(
                np.flatnonzero(
                    (store.route == row["chosen_route"]) & (store.frame == row["chosen_frame"])
                )[0]
            )
        )
        lines = [
            f"maze {seed}{' (real target)' if sample['real'] else ''}  view k={row['k']}",
            f"pos {row['pos_err_m']:.2f} m  yaw {row['yaw_err_deg']:.0f} deg",
            f"retrieval {row['retrieval_err_m']:.2f} m  mapping {row['mapping_err_m']:.2f} m",
            f"rel pose {fmt(row.get('rel_pos_err_m'), 2)} m "
            f"{fmt(row.get('rel_yaw_err_deg'), 0)} deg",
            f"inliers {row['inliers']}  confidence {row['confidence']}",
            f"frame r{row['chosen_route']} #{row['chosen_frame']}",
        ]
        rows.append(np.hstack([target, matched, panel(lines)]))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))


# -- the run


def evaluate(seed: int, args, floor, out: Path) -> tuple[list[dict], list[tuple[dict, dict]]]:
    root = record(seed)
    world = sim.Simulator(textures(), seed)
    walls = Walls(world)
    diff = check_render(world)
    print(f"maze {seed}: render at target vs targets()[0]: mean pixel difference {diff:.1f}")
    store, routes = frozen_map.get(root, world.camera, floor, rebuild=args.rebuild)
    truth = true_poses(root, routes)
    frame = Frame((0.0, 0.0, 0.0), tuple(truth[0][0]))
    camera = Camera({**world.camera, "cy": world.camera["cy"] + mapping.HORIZON_SHIFT_PX})

    rng = np.random.default_rng(seed)
    results = []
    for number, sample in enumerate(samples(world, args.n, rng)):
        found = goal.find_goal(sample["views"], store, camera, floor)
        row = {"seed": seed, "sample": number, **score(sample, found, store, frame, truth)}
        results.append((sample, row))
        print(
            f"  {number:>3} {'T' if sample['real'] else ' '} pos {fmt(row['pos_err_m'], 2):>5} m "
            f"yaw {fmt(row['yaw_err_deg'], 0):>4} deg  conf {row.get('confidence', '-')}",
            flush=True,
        )
    store.sift.flush()
    draw_goals(world, walls, frame, store, results, seed, out / f"{seed}_goals.png")
    draw_examples(store, results, seed, out / f"{seed}_examples.png")
    return [row for _, row in results], results


def column(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows if key in r and np.isfinite(r[key])], float)


def median(rows: list[dict], key: str) -> str:
    values = column(rows, key)
    return fmt(float(np.median(values)), 2) if len(values) else "n/a"


def summarise(label: str, rows: list[dict]) -> None:
    if not rows:
        return
    n = len(rows)
    print(
        f"{label:<8} {n:>4} | {sum(r['success_0.2'] for r in rows) / n:>8.2f} "
        f"{sum(r['success_0.4'] for r in rows) / n:>8.2f} | "
        f"{median(rows, 'pos_err_m'):>7} {median(rows, 'yaw_err_deg'):>7} | "
        f"{median(rows, 'retrieval_err_m'):>7} {median(rows, 'mapping_err_m'):>7} "
        f"{median(rows, 'rel_pos_err_m'):>7}"
    )


def report(rows: list[dict]) -> None:
    head = (
        f"{'':<8} {'n':>4} | {'ok 0.2m':>8} {'ok 0.4m':>8} | {'pos m':>7} {'yaw deg':>7} | "
        f"{'retr m':>7} {'map m':>7} {'rel m':>7}   (medians)"
    )
    print("\n" + head + "\n" + "-" * len(head))
    for seed in sorted({r["seed"] for r in rows}):
        summarise(f"maze {seed}", [r for r in rows if r["seed"] == seed])
    for name, seeds in SPLITS.items():
        summarise(name, [r for r in rows if r["seed"] in seeds])
    summarise("all", rows)
    for name, seeds in SPLITS.items():
        pooled = [r for r in rows if r["seed"] in seeds and "confidence" in r]
        if not pooled:
            continue
        print(f"\nconfidence vs success, mazes {name}:")
        for low, high in CONFIDENCE_BINS:
            binned = [r for r in pooled if low <= r["confidence"] < high]
            if binned:
                rate = sum(r["success_0.2"] for r in binned) / len(binned)
                label = f"{low}-{high - 1}" if high < 10**9 else f"{low}+"
                print(f"  {label:>6}: {len(binned):>4} samples, {rate:.2f} at 0.2 m")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mazes", default="7,8,9,10,11,12", help="comma-separated maze seeds")
    p.add_argument("--n", type=int, default=40, help="random goals per maze")
    p.add_argument("--rebuild", action="store_true", help="rebuild the frozen maps")
    p.add_argument("--out-dir", default="results/eval", help="where goal/ goes")
    args = p.parse_args()

    out = Path(args.out_dir) / "goal"
    floor = stored_floor()
    rows: list[dict] = []
    for seed in ints(args.mazes):
        maze, _ = evaluate(seed, args, floor, out)
        rows.extend(maze)
    write_csv(out / "goals.csv", FIELDS, rows)
    report(rows)


if __name__ == "__main__":
    main()
