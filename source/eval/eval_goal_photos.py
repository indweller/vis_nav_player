"""
The real goal photos against renders: do world.targets() look like renders at world.target?

    uv run source/eval/eval_goal_photos.py
    uv run source/eval/eval_goal_photos.py --mazes 7,8

Per maze (7-12) and view k (the target pose turned 90 degrees * k to the left), the photo is
compared with a render at that pose with msaa=1 and msaa=4, raw and through world.jpeg (as the
server sends frames): the mean absolute pixel difference, and the difference in
geometry.ranges (the per-column wall range, on the camera the agent uses): the mean absolute
range difference and the median photo / render ratio over the columns both see a wall in, and
the fraction of columns where one sees a wall and the other does not. The floor mask
(data/floor_mask.npz) is also scored on the photo and on each render against the true floor of
that pose (precision, recall; eval_cues.floor_truth).
Writes results/eval/goal_photos.csv.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import vis_nav_sim as sim
from common import fmt, ints, textures, write_csv
from eval_cues import Walls, floor_truth

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mapping  # noqa: E402
from geometry import FLOOR_MASK_PATH, Camera, FloorMask, ranges  # noqa: E402

VARIANTS = [(m, j) for m in (1, 4) for j in (False, True)]
FIELDS = [
    "seed",
    "k",
    "variant",
    "pixel_diff",
    "range_diff_m",
    "range_ratio",
    "wall_disagree",
    "precision",
    "recall",
]


def name(msaa: int, jpeg: bool) -> str:
    return f"msaa{msaa}{'-jpeg' if jpeg else '-raw'}"


def through_jpeg(world: sim.Simulator, image: np.ndarray) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(world.jpeg(image), np.uint8), cv2.IMREAD_COLOR)


def range_difference(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Mean absolute gap and median ratio ``a / b`` over the columns both see a wall in, and the
    fraction of columns where only one does."""
    both = np.isfinite(a) & np.isfinite(b) & (b > 0)
    gap = float(np.abs(a[both] - b[both]).mean()) if both.any() else math.nan
    ratio = float(np.median(a[both] / b[both])) if both.any() else math.nan
    return gap, ratio, float((np.isfinite(a) != np.isfinite(b)).mean())


def evaluate(seed: int, floor: FloorMask) -> list[dict]:
    worlds = {m: sim.Simulator(textures(), seed, msaa=m) for m in (1, 4)}
    world = worlds[1]
    photos = world.targets()
    walls = Walls(world)
    plain = Camera(world.camera)
    shifted = Camera({**world.camera, "cy": world.camera["cy"] + mapping.HORIZON_SHIFT_PX})
    x, y, yaw = world.target
    rows = []
    for k, photo in enumerate(photos):
        pose = (x, y, yaw + k * math.pi / 2)
        truth = floor_truth(plain, walls, *pose)
        photo_ranges = ranges(photo, shifted, floor)[1]
        candidates = {"photo": (photo, photo_ranges)}
        for msaa, jpeg in VARIANTS:
            image = worlds[msaa].render(*pose)
            if jpeg:
                image = through_jpeg(worlds[msaa], image)
            candidates[name(msaa, jpeg)] = (image, ranges(image, shifted, floor)[1])
        for label, (image, r) in candidates.items():
            gap, ratio, disagree = range_difference(photo_ranges, r)
            precision, recall = floor.score([(image, truth)])
            rows.append(
                {
                    "seed": seed,
                    "k": k,
                    "variant": label,
                    "pixel_diff": float(np.abs(image.astype(float) - photo).mean()),
                    "range_diff_m": gap,
                    "range_ratio": ratio,
                    "wall_disagree": disagree,
                    "precision": precision,
                    "recall": recall,
                }
            )
    return rows


def summarise(rows: list[dict]) -> None:
    print(
        f"\n{'vs the photo':<14} {'pixel diff':>10} {'range diff m':>13} {'photo/render':>12} "
        f"{'wall disagree':>14} {'precision':>10} {'recall':>7}"
    )
    for label in ["photo", *(name(m, j) for m, j in VARIANTS)]:

        def mean(key: str, label: str = label) -> float:
            return float(np.nanmean([r[key] for r in rows if r["variant"] == label]))

        print(
            f"{label:<14} {fmt(mean('pixel_diff'), 2):>10} {fmt(mean('range_diff_m'), 3):>13} "
            f"{fmt(mean('range_ratio'), 2):>12} {fmt(mean('wall_disagree'), 3):>14} "
            f"{fmt(mean('precision'), 3):>10} {fmt(mean('recall'), 3):>7}"
        )
    print("(the photo row's precision / recall are the floor mask on the photos themselves)")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mazes", default="7,8,9,10,11,12")
    p.add_argument("--out-dir", default="results/eval")
    args = p.parse_args()
    floor = FloorMask.load(FLOOR_MASK_PATH)
    rows = []
    for seed in ints(args.mazes):
        rows.extend(evaluate(seed, floor))
    write_csv(Path(args.out_dir) / "goal_photos.csv", FIELDS, rows)
    summarise(rows)


if __name__ == "__main__":
    main()
