"""
Headless benchmark of MyAgent against exploration-data size on local mazes: every maze seed x
routes x capture_every, recorded on first use, one CSV row per run and a summary at the end.

    uv run source/eval_data_scale.py
    uv run source/eval_data_scale.py --mazes 7,8 --routes 3,10 --capture-every 5
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
import traceback
from collections import defaultdict
from pathlib import Path

from my_agent import SHORTCUT_GAP_FRAC, SHORTCUTS_PER_1K, MyAgent, add_filter_args, filter_kwargs
from vis_nav_sdk import run

FIELDS = [
    "seed",
    "routes",
    "capture_every",
    "n_frames",
    "build_s",
    "run_s",
    "goal_tier",
    "points",
    "max_points",
    "spl",
    "nav_steps",
    "distance_m",
    "heading_deg",
    "checkin_reason",
    "no_path_frac",
    "low_conf_frac",
]
REACHED = {"perfect", "partial"}


def ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def record(path: Path, seed: int, routes: int, capture_every: int) -> None:
    if (path / "target.jpg").exists():
        return
    import vis_nav_sim as sim

    print(f"recording {path} ...", flush=True)
    world = sim.Simulator(
        sim.Textures.find(), seed, motion_noise=sim.DEFAULT_NOISE, motion_seed=seed
    )
    summary = world.record(path, routes=routes, seed=seed, capture_every=capture_every)
    print(f"  {summary['frames']} frames on {summary['routes']} routes")


def evaluate(seed: int, routes: int, capture_every: int, args: argparse.Namespace) -> dict:
    name = f"{seed}-r{routes}-c{capture_every}"
    data_dir = Path("data") / "scale" / name
    record(data_dir, seed, routes, capture_every)

    started = time.perf_counter()
    agent = MyAgent(
        data_dir,
        ticks=args.ticks,
        capture_every=capture_every,
        random_every=0,
        subsample=args.subsample,
        shortcut_gap_frac=args.shortcut_gap_frac,
        shortcuts_per_1k=args.shortcuts_per_1k,
        cache_dir=Path("cache") / "scale" / name,
        **filter_kwargs(args),
    )
    build_s = time.perf_counter() - started

    started = time.perf_counter()
    result = run(
        agent, f"local:{seed}", viewer=False, quiet=True, check=False, confirm=False, browser=False
    )
    run_s = time.perf_counter() - started

    row = dict.fromkeys(FIELDS, "")
    row.update(
        seed=seed,
        routes=routes,
        capture_every=capture_every,
        n_frames=len(agent.frames),
        build_s=f"{build_s:.2f}",
        run_s=f"{run_s:.2f}",
        checkin_reason=agent.checkin_reason or "",
        no_path_frac=f"{agent.no_path_decisions / max(1, agent.decisions):.3f}",
        low_conf_frac=f"{agent.low_conf_decisions / max(1, agent.decisions):.3f}",
    )
    if result is not None:
        row.update(
            goal_tier=result.goal_tier,
            points=result.points,
            max_points=result.max_points,
            spl=f"{result.spl:.4f}",
            nav_steps=result.nav_steps,
        )
        if result.arrival is not None:
            row.update(
                distance_m=f"{result.arrival.distance_m:.3f}",
                heading_deg=f"{result.arrival.heading_deg:.1f}",
            )
    return row


def mean(rows: list[dict], key: str) -> str:
    values = [float(r[key]) for r in rows if r[key] != ""]
    return f"{sum(values) / len(values):.3f}" if values else "n/a"


def summarise(rows: list[dict]) -> None:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["routes"], row["capture_every"])].append(row)
    print()
    for (routes, capture_every), group in sorted(groups.items()):
        success = sum(r["goal_tier"] in REACHED for r in group) / len(group)
        print(
            f"r{routes} c{capture_every}: runs={len(group)} n_frames={mean(group, 'n_frames')} "
            f"success={success:.2f} points={mean(group, 'points')} spl={mean(group, 'spl')} "
            f"distance_m={mean(group, 'distance_m')} no_path_frac={mean(group, 'no_path_frac')} "
            f"low_conf_frac={mean(group, 'low_conf_frac')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mazes", default="7,8,9,10,11", help="comma-separated maze seeds")
    parser.add_argument("--routes", default="3,10,30", help="comma-separated route counts")
    parser.add_argument("--capture-every", default="1", help="comma-separated tick intervals")
    parser.add_argument("--subsample", type=int, default=5, help="keep every Nth frame")
    parser.add_argument("--ticks", type=int, default=5, help="ticks per move")
    parser.add_argument(
        "--shortcut-gap-frac",
        type=float,
        default=SHORTCUT_GAP_FRAC,
        help="shortcuts join nodes at least this fraction of the index apart",
    )
    parser.add_argument(
        "--shortcuts-per-1k",
        type=float,
        default=SHORTCUTS_PER_1K,
        help="visual shortcut edges per 1000 nodes",
    )
    add_filter_args(parser)
    parser.add_argument("--out", default="results/data_scale.csv", help="CSV to write")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        grid = itertools.product(ints(args.mazes), ints(args.routes), ints(args.capture_every))
        for seed, routes, capture_every in grid:
            print(f"\n== maze {seed}, routes {routes}, capture_every {capture_every}", flush=True)
            try:
                row = evaluate(seed, routes, capture_every, args)
            except Exception:
                traceback.print_exc()
                continue
            rows.append(row)
            writer.writerow(row)
            handle.flush()
    print(f"\nwrote {len(rows)} rows to {out}")
    summarise(rows)


if __name__ == "__main__":
    main()
