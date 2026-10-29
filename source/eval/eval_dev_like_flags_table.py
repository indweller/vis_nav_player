"""Compare dev-like runs of MapAgent's flags: one summary table per config, from runs.csv.

    uv run source/eval/eval_dev_like_flags_table.py
    uv run source/eval/eval_dev_like_flags_table.py --configs base,floor,zero,online,all

Reads results/eval/dev_like/<seed>-<tag>/runs.csv and ticks.csv for each tag and writes
results/eval/dev_like/<seed>-flags.md: per config the mean over its runs of points, success rate,
the visual odometry fallback rate, wall contact ticks, guard events and the localisation error at
check-in (against the map-consistent truth, ``cm``; the world-frame error too), with the runs'
standard deviation of points between repeats; then per goal points; for --blocked-zero the
confusion matrix of its zeroed ticks against the true blocked ticks (of the ticks.csv of the
runs: a tick where visual odometry failed on a FORWARD / BACKWARD); for --online-motion the
converged medians.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("results") / "eval" / "dev_like"
TAGS = {
    "base": "baseline",
    "floor": "--floor-guard",
    "zero": "--blocked-zero",
    "online": "--online-motion",
    "all": "all three",
}


def read(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def num(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows])


def md(header: list[str], rows: list[list]) -> str:
    sep = "|" + "|".join("---" for _ in header) + "|"
    body = ("| " + " | ".join(map(str, r)) + " |" for r in rows)
    return "\n".join(["| " + " | ".join(header) + " |", sep, *body])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--configs", default=",".join(TAGS))
    args = p.parse_args()
    tags = [t for t in args.configs.split(",") if (ROOT / f"{args.seed}-{t}" / "runs.csv").exists()]
    runs = {t: read(ROOT / f"{args.seed}-{t}" / "runs.csv") for t in tags}
    lines = [f"# Flags on seed {args.seed}, 10 goals, --goal-camera photo", ""]
    summary = []
    for t in tags:
        r = runs[t]
        repeats = sorted({int(x["repeat"]) for x in r})
        by_repeat = [num([x for x in r if int(x["repeat"]) == k], "points").mean() for k in repeats]
        summary.append(
            [
                TAGS.get(t, t),
                len(r),
                f"{num(r, 'points').mean():.2f}",
                f"{np.std(by_repeat):.2f}",
                f"{num(r, 'success').mean():.0%}",
                f"{np.nanmean(num(r, 'vo_fallback_frac')):.3f}",
                f"{num(r, 'wall_contact_ticks').mean():.0f}",
                f"{num(r, 'wall_guard_events').mean():.0f}",
                f"{num(r, 'floor_guard_events').mean():.0f}",
                f"{np.mean(num(r, 'cm_err_checkin_m')):.2f}",
                f"{np.median(num(r, 'cm_err_checkin_m')):.2f}",
                f"{np.mean(num(r, 'loc_err_checkin_m')):.2f}",
                f"{num(r, 'nav_steps').mean():.0f}",
                f"{(num(r, 'nav_steps') >= 9900).sum()}",
            ]
        )
    lines += [
        "## Summary (mean per run)",
        "",
        md(
            [
                "config",
                "runs",
                "points /10",
                "sd of repeat means",
                "success",
                "VO fallback",
                "contact ticks",
                "guard events",
                "(floor)",
                "loc err cm m (mean)",
                "(median)",
                "loc err world m",
                "steps",
                "out of steps",
            ],
            summary,
        ),
        "",
        "Guard events count every wall guard firing (the floor guard's included); `(floor)` is "
        "those the centre range alone would not have fired. `loc err cm` is the localisation "
        "error at check-in against the map-consistent truth; `world` against the world truth. "
        "`out of steps`: runs of at least 9,900 steps.",
        "",
        "## Points per goal (mean of the repeats)",
        "",
    ]
    goals = sorted({int(x["goal"]) for t in tags for x in runs[t]})
    per_goal = []
    for g in goals:
        row = [g]
        for t in tags:
            v = [float(x["points"]) for x in runs[t] if int(x["goal"]) == g]
            ok = [int(x["success"]) for x in runs[t] if int(x["goal"]) == g]
            row.append(f"{np.mean(v):.1f} ({sum(ok)}/{len(ok)})" if v else "-")
        per_goal.append(row)
    lines += [
        md(["goal", *[TAGS.get(t, t) for t in tags]], per_goal),
        "",
        "(points; successes/runs)",
        "",
    ]

    lines += ["## Failures", ""]
    causes = ["wrong goal", "localisation", "out of steps", "other"]
    lines.append(
        md(
            ["config", *causes],
            [
                [TAGS.get(t, t), *[Counter(x["failure"] for x in runs[t])[c] for c in causes]]
                for t in tags
            ],
        )
    )

    for t in ("zero", "all"):
        if t not in runs:
            continue
        counts = Counter()
        ticks = read(ROOT / f"{args.seed}-{t}" / "ticks.csv")
        previous = {}
        for x in ticks:
            key = (x["goal"], x["repeat"])
            before = previous.get(key)
            previous[key] = x
            if (
                before is None
                or x["vo_fallback"] != "1"
                or before["action"] not in ("FORWARD", "BACKWARD")
            ):
                continue
            counts[(x["vo_source"] == "zero", x["blocked"] == "1")] += 1
        lines += [
            "",
            f"## --blocked-zero in `{t}`: zeroed ticks against the truth",
            "",
            "Ticks where visual odometry failed after a FORWARD / BACKWARD.",
            "",
            md(
                ["", "zeroed", "calibrated / online"],
                [
                    ["truly blocked", counts[(True, True)], counts[(False, True)]],
                    ["truly moving", counts[(True, False)], counts[(False, False)]],
                ],
            ),
        ]
    for t in ("online", "all"):
        if t not in runs:
            continue
        r = runs[t]
        lines += ["", f"## --online-motion in `{t}`: converged medians (end of each run)", ""]
        rows = []
        for key, label, truth in (
            ("online_forward_dx_m", "FORWARD dx m", 0.03),
            ("online_backward_dx_m", "BACKWARD dx m", -0.03),
            ("online_left_deg", "LEFT deg", 3.4375),
            ("online_right_deg", "RIGHT deg", -3.4375),
            ("online_turn_rate_deg", "turn rate deg", 3.4375),
        ):
            v = num(r, key)
            v = v[~np.isnan(v)]
            rows.append(
                [
                    label,
                    f"{np.median(v):.4f}" if len(v) else "n/a",
                    f"{v.min():.4f} .. {v.max():.4f}" if len(v) else "",
                    len(v),
                    truth,
                ]
            )
        base = runs.get("base", r)
        rows.append(
            [
                "calibrated FORWARD dx m",
                f"{num(base, 'fallback_forward_dx_m').mean():.4f}",
                "",
                "",
                "",
            ]
        )
        rows.append(
            ["calibrated turn rate deg", f"{num(base, 'turn_rate_deg').mean():.4f}", "", "", ""]
        )
        lines += [
            md(["value", "median over runs", "range", "runs with it", "simulator's tick"], rows),
            "",
            f"Fallback ticks that used the online median: {int(num(r, 'online_fallback_ticks').sum())} "  # noqa: E501
            f"of {int(sum(float(x['vo_calls']) * float(x['vo_fallback_frac']) for x in r if not math.isnan(float(x['vo_fallback_frac']))))} VO fallbacks.",  # noqa: E501
        ]
    text = "\n".join(lines) + "\n"
    (ROOT / f"{args.seed}-flags.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
