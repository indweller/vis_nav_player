"""Visual odometry error per frame at capture_every=1, against capture_every=3.

    uv run source/eval/eval_dev_like_vo_c1.py

Records route 0 of the seed-101 maze at capture_every=1 (data/eval/dev_like_c1/<seed>/, one route,
student_files=False; a different route from the capture_every=3 recording's route 0, because the
exploration depends on the capture rate), runs the odometry of eval_dev_like_map.trace_odometry
(mapping.odometry with each tick logged) over its first FRAMES frames and compares each frame's
step with the true relative pose. The capture_every=3 numbers are those of the first FRAMES / 3
frames of results/eval/dev_like/<seed>-map/cache.pkl (about the same ticks).
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import vis_nav_sim as sim
from common import textures

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geometry  # noqa: E402
import map_agent  # noqa: E402
import mapping  # noqa: E402
from eval_dev_like import DATA_ROOT, MSAA, DevSimulator  # noqa: E402
from eval_dev_like_map import OUT_ROOT, VO, trace_odometry  # noqa: E402
from eval_geometry import Frame  # noqa: E402
from geometry import Camera, FloorMask  # noqa: E402
from mapping import between, wrap  # noqa: E402

FRAMES = 1000


def stats(used, source, truth, label: str) -> list:
    rel = np.array([between(tuple(truth[i - 1]), tuple(truth[i])) for i in range(1, len(truth))])
    step = used[1:]
    t_err = np.hypot(step[:, 0] - rel[:, 0], step[:, 1] - rel[:, 1])
    y_err = np.degrees(np.abs(wrap(step[:, 2] - rel[:, 2])))
    vo = source[1:] == VO
    return [
        label,
        len(rel),
        f"{vo.mean():.1%}",
        f"{np.median(t_err[vo]):.4f} / {np.percentile(t_err[vo], 90):.4f}",
        f"{np.median(y_err[vo]):.3f} / {np.percentile(y_err[vo], 90):.3f}",
        f"{np.sum(step[vo, 2] - rel[vo, 2]) * 180 / math.pi:+.1f}",
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=101)
    args = p.parse_args()
    root = Path("data") / "eval" / "dev_like_c1" / str(args.seed)
    if not (root / "traj_0" / "poses.json").exists():
        world = DevSimulator(
            textures(), args.seed, msaa=MSAA, motion_noise=sim.DEFAULT_NOISE, motion_seed=args.seed
        )
        if root.exists():
            shutil.rmtree(root)
        world.record(root, routes=1, seed=args.seed, capture_every=1, student_files=False)
    route = mapping.load_routes(root)[0]
    route = mapping.Route(mapping.Images(route.frames.paths[:FRAMES]), route.actions[:FRAMES])
    truth = np.array(json.loads((root / "traj_0" / "poses.json").read_text()))[:FRAMES]
    floor = FloorMask.load(map_agent.FLOOR_MASK_PATH)
    camera_dict = dict(geometry.CAMERA)
    camera = Camera({**camera_dict, "cy": camera_dict["cy"] + mapping.HORIZON_SHIFT_PX})
    motion = mapping.calibrate([route], camera, floor)
    out = trace_odometry(route, (0.0, 0.0, 0.0), camera, floor, motion, "c1")
    frame = Frame((0.0, 0.0, 0.0), tuple(truth[0]))
    truth_map = frame.to_map(truth)
    err = np.hypot(*(out["poses"][:, :2] - truth_map[:, :2]).T)
    yaw = np.degrees(np.abs(wrap(out["poses"][:, 2] - truth_map[:, 2])))
    length = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(truth[:, :2], axis=0).T))])

    cache = pickle.loads((OUT_ROOT / f"{args.seed}-map" / "cache.pkl").read_bytes())
    t3 = np.array(json.loads((DATA_ROOT / str(args.seed) / "traj_0" / "poses.json").read_text()))
    n3 = FRAMES // 3
    rows = [
        stats(out["used"], out["source"], truth, f"capture_every=1, first {FRAMES} frames"),
        stats(
            cache["used"][0][:n3],
            cache["source"][0][:n3],
            t3[:n3],
            f"capture_every=3, first {n3} frames",
        ),
    ]
    sep = "|---|---|---|---|---|---|"
    lines = [
        "| run | frames | VO used | translation err m (median / p90) | yaw err deg (median / p90) "
        "| summed signed yaw err deg |",
        sep,
        *("| " + " | ".join(map(str, r)) + " |" for r in rows),
        "",
        f"capture_every=1 over {FRAMES} frames ({length[-1]:.2f} m true path, true step per frame "
        f"{np.median(np.hypot(*np.diff(truth[:, :2], axis=0).T)[np.hypot(*np.diff(truth[:, :2], axis=0).T) > 1e-4]):.4f} m): "  # noqa: E501
        f"dead-reckoned position error {err[-1]:.2f} m, yaw error {yaw[-1]:.1f} deg at the end "
        f"(max {err.max():.2f} m / {yaw.max():.1f} deg).",
    ]
    c3 = cache["dead"][0][:n3]
    f3 = Frame((0.0, 0.0, 0.0), tuple(t3[0]))
    m3 = f3.to_map(t3[:n3])
    l3 = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(t3[:n3, :2], axis=0).T))])
    lines += [
        f"capture_every=3 over its first {n3} frames ({l3[-1]:.2f} m): position error "
        f"{np.hypot(*(c3[-1, :2] - m3[-1, :2])):.2f} m, yaw error "
        f"{abs(math.degrees(wrap(c3[-1, 2] - m3[-1, 2]))):.1f} deg.",
    ]
    text = "\n".join(lines) + "\n"
    (OUT_ROOT / f"{args.seed}-map" / "vo_c1.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
