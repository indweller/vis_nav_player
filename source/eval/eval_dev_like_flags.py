"""Fit and score the thresholds of MapAgent's --floor-guard and --blocked-zero from a dev-like
run's ticks.csv.

    uv run source/eval/eval_dev_like_flags.py --dir 101
    uv run source/eval/eval_dev_like_flags.py --dir 101-baseline --floor-frac 0.5 --blocked-diff 1.0

Every frame of ticks.csv is re-rendered at its true pose (and JPEG-encoded, as the agent sees it:
see eval_dev_like_vo.py) and the agent's own functions are applied to it, without the agent:

- floor guard: geometry.floor_fraction (floor pixels in the bottom 10 rows of the columns within
  20 px of cx) against the truth "a wall ahead is under WALL_M (0.15 m)": the nearest true wall
  (eval_cues.Walls) over the rays of those columns, from the true pose. A single ray along the yaw
  is unstable at a wall's end (it slips past the corner while the band still sees the wall), so
  the label is the band's; the single ray's is reported too. The wall is predicted close when the
  fraction is under the threshold;
- blocked-zero: localiser.grey_difference between the frame and the previous one, on the
  FORWARD / BACKWARD ticks where visual odometry failed (ticks.csv ``vo_fallback`` = 1), against
  the truth "the tick moved under 20% of a tick" (ticks.csv ``blocked``, from the true poses).
  The tick is predicted blocked when the difference is under the threshold.

Without ``--floor-frac`` / ``--blocked-diff`` the thresholds are fitted: the one maximising F1 of
the positive class (the wall under 0.15 m; blocked), over a grid of the observed values. With
them, they are scored as given. Writes flag_ticks.csv and flags.md into the run's directory.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from common import textures, write_csv
from eval_cues import Walls

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geometry  # noqa: E402
import map_agent  # noqa: E402
import mapping  # noqa: E402
from eval_dev_like import BLOCKED_FRACTION, COLS, MSAA, ROWS, TICK_M, DevSimulator  # noqa: E402
from eval_dev_like_vo import wall_ahead  # noqa: E402
from geometry import FLOOR_BAND_PX, Camera, FloorMask, floor_fraction  # noqa: E402
from localiser import CONTACT_ACTIONS, grey_difference, grey_small  # noqa: E402

OUT_ROOT = Path("results") / "eval" / "dev_like"
WALL_M = 0.15
FIELDS = [
    "goal",
    "repeat",
    "tick",
    "taken",
    "wall_ahead_m",
    "band_wall_m",
    "floor_frac",
    "grey_diff",
    "applied_dx_m",
    "blocked",
    "vo_fallback",
]


def read_ticks(path: Path) -> dict[tuple[int, int], list[dict]]:
    runs: dict[tuple[int, int], list[dict]] = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            runs[int(row["goal"]), int(row["repeat"])].append(row)
    return runs


def replay(rows, world, camera, floor, walls, goal: int, repeat: int) -> list[dict]:
    out, previous_grey, previous_pose, taken = [], None, None, None
    columns = np.arange(camera.width)
    band = np.abs(columns - camera.cx) <= FLOOR_BAND_PX
    for row in rows:
        pose = tuple(float(row[k]) for k in ("true_x", "true_y", "true_yaw"))
        frame = cv2.imdecode(np.frombuffer(world.jpeg(world.render(*pose)), np.uint8), 1)
        grey = grey_small(frame)
        applied = mapping.between(previous_pose, pose)[0] if previous_pose and taken else math.nan
        out.append(
            {
                "goal": goal,
                "repeat": repeat,
                "tick": int(row["tick"]),
                "taken": taken or "",
                "wall_ahead_m": wall_ahead(walls.rects, *pose),
                "band_wall_m": float(walls.cast(*pose[:2], camera.angles(pose[2])[band]).min()),
                "floor_frac": floor_fraction(floor(frame), camera),
                "grey_diff": math.nan
                if previous_grey is None
                else grey_difference(previous_grey, grey),
                "applied_dx_m": applied,
                "blocked": int(row["blocked"]),
                "vo_fallback": row["vo_fallback"],
            }
        )
        previous_grey, previous_pose = grey, pose
        taken = None if row["action"] == "CHECKIN" else row["action"]
    return out


def scores(predicted: np.ndarray, truth: np.ndarray) -> dict:
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    tn = int((~predicted & ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def fit(values: np.ndarray, truth: np.ndarray) -> tuple[float, dict]:
    """The threshold (``values < threshold`` predicts the positive class) of best F1, among the
    midpoints of the sorted distinct values."""
    distinct = np.unique(values)
    grid = np.concatenate(
        [[distinct[0] - 1e-3], (distinct[1:] + distinct[:-1]) / 2, [distinct[-1] + 1e-3]]
    )
    best = max((scores(values < t, truth)["f1"], -abs(t), t) for t in grid)
    return best[2], scores(values < best[2], truth)


def md(header: list[str], rows: list[list]) -> str:
    sep = "|" + "|".join("---" for _ in header) + "|"
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            sep,
            *("| " + " | ".join(map(str, r)) + " |" for r in rows),
        ]
    )


def curve(values: np.ndarray, truth: np.ndarray, grid: list[float]) -> str:
    rows = []
    for t in grid:
        s = scores(values < t, truth)
        rows.append(
            [
                f"{t:.3g}",
                s["tp"],
                s["fp"],
                s["fn"],
                s["tn"],
                f"{s['precision']:.3f}",
                f"{s['recall']:.3f}",
                f"{s['f1']:.3f}",
            ]
        )
    return md(["threshold", "TP", "FP", "FN", "TN", "precision", "recall", "F1"], rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dir", default="101", help="the run's directory under results/eval/dev_like")
    p.add_argument("--seed", type=int, default=101, help="the maze")
    p.add_argument("--floor-frac", type=float, default=None, help="score this, do not fit")
    p.add_argument("--blocked-diff", type=float, default=None, help="score this, do not fit")
    args = p.parse_args()
    directory = OUT_ROOT / args.dir
    runs = read_ticks(directory / "ticks.csv")
    world = DevSimulator(textures(), args.seed, msaa=MSAA)
    assert world.occupancy.shape == (ROWS, COLS)
    walls = Walls(world)
    camera_dict = dict(geometry.CAMERA)
    camera = Camera({**camera_dict, "cy": camera_dict["cy"] + mapping.HORIZON_SHIFT_PX})
    floor = FloorMask.load(map_agent.FLOOR_MASK_PATH)
    rows: list[dict] = []
    for (goal, repeat), rs in runs.items():
        rows.extend(replay(rs, world, camera, floor, walls, goal, repeat))
        print(f"goal {goal} #{repeat}: {len(rs)} ticks replayed", flush=True)
    write_csv(directory / "flag_ticks.csv", FIELDS, rows)

    wall = np.array([r["band_wall_m"] for r in rows])
    ray = np.array([r["wall_ahead_m"] for r in rows])
    frac = np.array([r["floor_frac"] for r in rows])
    close = wall < WALL_M
    lines = [f"# Flag thresholds, run `{args.dir}`", ""]

    # Floor guard.
    lines += [
        "## --floor-guard",
        "",
        f"{len(rows)} replayed frames, {int(close.sum())} with a true wall under {WALL_M} m in the "
        f"central band ({close.mean():.1%}); by the single yaw ray {int((ray < WALL_M).sum())}.",
        "",
    ]
    if args.floor_frac is None:
        t_floor, s = fit(frac, close)
        lines.append(f"Fitted threshold (best F1): **{t_floor:.4f}**.")
    else:
        t_floor, s = args.floor_frac, scores(frac < args.floor_frac, close)
        lines.append(f"Threshold as given: **{t_floor:.4f}**.")
    lines += [
        "",
        f"At it: precision {s['precision']:.3f}, recall {s['recall']:.3f}, F1 {s['f1']:.3f} "
        f"(TP {s['tp']}, FP {s['fp']}, FN {s['fn']}, TN {s['tn']}).",
        "",
        "Floor fraction by the true wall distance in the band (median, 5th-95th percentile):",
        "",
    ]
    bins = [
        (0, 0.05),
        (0.05, 0.1),
        (0.1, 0.15),
        (0.15, 0.2),
        (0.2, 0.3),
        (0.3, 0.5),
        (0.5, math.inf),
    ]
    lines.append(
        md(
            ["band wall m", "frames", "floor fraction"],
            [
                [
                    f"{lo:g}-{hi:g}",
                    int(m.sum()),
                    "n/a"
                    if not m.any()
                    else f"{np.median(frac[m]):.2f} ({np.percentile(frac[m], 5):.2f}-{np.percentile(frac[m], 95):.2f})",  # noqa: E501
                ]
                for lo, hi in bins
                for m in [(wall >= lo) & (wall < hi)]
            ],
        )
    )
    lines += [
        "",
        "Threshold sweep:",
        "",
        curve(
            frac, close, sorted({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, round(t_floor, 3)})
        ),
    ]
    s3 = scores(frac < t_floor, ray < WALL_M)
    lines += [
        "",
        f"Against the single yaw ray's label at the same threshold: precision {s3['precision']:.3f}, "  # noqa: E501
        f"recall {s3['recall']:.3f} (TP {s3['tp']}, FP {s3['fp']}, FN {s3['fn']}, TN {s3['tn']}).",
    ]
    forward = np.array([r["taken"] == "FORWARD" for r in rows])
    # The guard is read on the frame a FORWARD is about to follow; the next row's `taken` is it.
    ahead = np.append(forward[1:], False)
    if ahead.any():
        s2 = scores((frac < t_floor)[ahead], close[ahead])
        lines += [
            "",
            f"Only the {int(ahead.sum())} frames a FORWARD followed: precision {s2['precision']:.3f}, "  # noqa: E501
            f"recall {s2['recall']:.3f} (TP {s2['tp']}, FP {s2['fp']}, FN {s2['fn']}, TN {s2['tn']}).",  # noqa: E501
        ]

    # Blocked zero.
    sel = [r for r in rows if r["taken"] in CONTACT_ACTIONS and r["vo_fallback"] == "1"]
    diff = np.array([r["grey_diff"] for r in sel])
    blocked = np.array([bool(r["blocked"]) for r in sel])
    n_contact = sum(r["taken"] in CONTACT_ACTIONS for r in rows)
    lines += [
        "",
        "## --blocked-zero",
        "",
        f"{n_contact} FORWARD / BACKWARD ticks, {len(sel)} of them with a visual odometry fallback, "  # noqa: E501
        f"{int(blocked.sum())} of those truly blocked (moved under {BLOCKED_FRACTION:.0%} of "
        f"{TICK_M} m) and {int((~blocked).sum())} moving.",
        "",
    ]
    if len(sel) and blocked.any() and (~blocked).any():
        if args.blocked_diff is None:
            t_diff, s = fit(diff, blocked)
            lines.append(f"Fitted threshold (best F1): **{t_diff:.4f}**.")
        else:
            t_diff, s = args.blocked_diff, scores(diff < args.blocked_diff, blocked)
            lines.append(f"Threshold as given: **{t_diff:.4f}**.")
        lines += [
            "",
            "Confusion matrix at it (rows: truth; columns: predicted):",
            "",
            md(
                ["", "predicted blocked (zero)", "predicted moving (calibrated)"],
                [["truly blocked", s["tp"], s["fn"]], ["truly moving", s["fp"], s["tn"]]],
            ),
            "",
            f"Precision {s['precision']:.3f}, recall {s['recall']:.3f}, F1 {s['f1']:.3f}.",
            "",
            "Grey difference (median, 5th-95th percentile): blocked "
            f"{np.median(diff[blocked]):.2f} ({np.percentile(diff[blocked], 5):.2f}-{np.percentile(diff[blocked], 95):.2f}), "  # noqa: E501
            f"moving {np.median(diff[~blocked]):.2f} ({np.percentile(diff[~blocked], 5):.2f}-{np.percentile(diff[~blocked], 95):.2f}).",  # noqa: E501
            "",
            "Threshold sweep:",
            "",
            curve(diff, blocked, sorted({0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, round(t_diff, 3)})),
        ]
        every = np.array(
            [
                r["grey_diff"]
                for r in rows
                if r["taken"] in CONTACT_ACTIONS and r["vo_fallback"] != ""
            ]
        )
        lines += [
            "",
            f"(All {len(every)} contact ticks with a VO call, fallback or not, are not used to fit.)",  # noqa: E501
        ]
    else:
        lines.append("Not enough of both classes to fit.")
    text = "\n".join(lines) + "\n"
    (directory / "flags.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
