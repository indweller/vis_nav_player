"""Does the turn over-read, through the snap rule, explain the yaw drift of the dev-like map?
No change to mapping.py, localiser.py or frozen_map.py: the odometry is repeated here with its
parts separated (the visual odometry pass, which does not depend on the pose; the integration with
the snap rule, which does), so snap variants cost nothing.

    uv run source/eval/eval_dev_like_yaw.py          # seed 101; passes are cached in the output

Writes results/eval/dev_like/<seed>-yaw/: summary.md, CSVs and route0_yaw.png.

1. Route 0 yaw timeline (shipped odometry): true yaw error per frame, with the frame-0 gauge offset
   ("frame" error, the convention of eval_dev_like_map: the map's yaw 0 is route 0's first frame)
   and without it ("world" error: map yaw against the true world yaw, the frame the snap targets
   live in, since manhattan_yaw is the yaw against the maze's axes), the Manhattan confidence and
   the snap outcome.
2. Turn bias: the sums of VO dyaw and true dyaw over LEFT / RIGHT frames; the camera mapping.py
   uses against the simulator's (fx fitted from renders turned by known angles); the sum again
   with the fitted fx.
3. Dead reckoning (a) shipped, (b) SNAP_DEG 45, (c) true fx, (d) both.
4. Pose graph on the best variant with the default closure rule, solved to convergence with the
   odometry sigmas from the measured per-frame VO error.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import vis_nav_sim as sim
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geometry  # noqa: E402
import map_agent  # noqa: E402
import mapping  # noqa: E402
from common import textures  # noqa: E402
from eval_dev_like import DATA_ROOT, MSAA, DevSimulator  # noqa: E402
from eval_geometry import Frame  # noqa: E402
from geometry import (  # noqa: E402
    MANHATTAN_CONFIDENCE,
    Camera,
    FloorMask,
    clip_range,
    manhattan_yaw,
    snap,
    visual_odometry,
    wall_view,
)
from mapping import NODE_EVERY, between, compose, wrap  # noqa: E402

OUT_ROOT = Path("results") / "eval" / "dev_like"
PATH_MARKS = (5, 10, 20, 30, 50, 75, 100)
FOLD_NEAR_M, FOLD_FAR_M = 0.15, 0.5
CHECK_FRAMES = (2000, 2390)
NONE, APPLIED, REFUSED, UNSURE = 0, 1, 2, 3
OUTCOME = {NONE: "none", APPLIED: "applied", REFUSED: "refused>SNAP_DEG", UNSURE: "not confident"}
SHIPPED_FX = 92.0
CONVERGED_NFEV = 2000
"""The shipped cap is 60; least_squares status 0 means this one was reached too."""


def md(header, rows) -> str:
    sep = "|" + "|".join("---" for _ in header) + "|"
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            sep,
            *("| " + " | ".join(map(str, r)) + " |" for r in rows),
        ]
    )


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else math.nan


def write_csv(path: Path, header, rows) -> None:
    path.write_text("\n".join([",".join(header), *(",".join(map(str, r)) for r in rows)]) + "\n")


def camera_for(fx: float) -> Camera:
    d = dict(geometry.CAMERA)
    return Camera({**d, "fx": fx, "cy": d["cy"] + mapping.HORIZON_SHIFT_PX})  # fy stays 92


# -- true intrinsics: fx fitted from renders turned by a known angle


def fit_fx(seed: int, truth0: np.ndarray, out: Path) -> dict:
    path = out / "fx_fit.json"
    if path.exists():
        return json.loads(path.read_text())
    world = DevSimulator(
        textures(), seed, msaa=MSAA, motion_noise=sim.DEFAULT_NOISE, motion_seed=seed
    )
    cam = world.camera
    cx, cy = cam["cx"], cam["cy"]
    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    rows = []
    for frame in (0, 500, 1000, 2000, 3000):
        x, y, yaw = (float(v) for v in truth0[frame])
        a = world.render(x, y, yaw)
        ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
        for deg in (3.0, 6.0, 10.0):
            b = world.render(x, y, yaw + math.radians(deg))
            kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
            good = [
                m
                for m, n in (p for p in matcher.knnMatch(da, db, k=2) if len(p) == 2)
                if m.distance < 0.75 * n.distance
            ]
            if len(good) < 30:
                continue
            pa = np.array([ka[m.queryIdx].pt for m in good])
            pb = np.array([kb[m.trainIdx].pt for m in good])
            d = math.radians(deg)

            def predict(f, g, pa=pa, d=d):
                ra, rb = (pa[:, 0] - cx) / f, (pa[:, 1] - cy) / g
                c, s = math.cos(d), math.sin(d)
                z = c - s * ra
                return np.column_stack([cx + f * (c * ra + s) / z, cy + g * rb / z])

            keep = np.hypot(*(predict(92.0, 92.0) - pb).T) < 4.0
            if keep.sum() < 20:
                continue

            def res_tied(p, keep=keep, pb=pb):
                return (predict(p[0], p[0])[keep] - pb[keep]).ravel()

            def res_free(p, keep=keep, pb=pb):
                return (predict(p[0], p[1])[keep] - pb[keep]).ravel()

            tied = least_squares(res_tied, [92.0], loss="soft_l1", f_scale=1.0).x[0]
            free = least_squares(res_free, [92.0, 92.0], loss="soft_l1", f_scale=1.0).x
            rows.append((frame, deg, int(keep.sum()), tied, free[0], free[1]))
    arr = np.array(rows)
    result = {
        "reported": cam,
        "fov120_fx": 160 / math.tan(math.radians(60)),
        "fx_tied_median": float(np.median(arr[:, 3])),
        "fx_free_median": float(np.median(arr[:, 4])),
        "fy_free_median": float(np.median(arr[:, 5])),
        "fx_tied_min_max": [float(arr[:, 3].min()), float(arr[:, 3].max())],
        "pairs": len(arr),
        "rows": arr.tolist(),
    }
    path.write_text(json.dumps(result, indent=1))
    return result


# -- pass: everything that does not depend on the pose


def run_pass(args: tuple[int, float, str, str]) -> None:
    r, fx, tag, out = args
    path = Path(out) / f"pass_{tag}_r{r}.npz"
    if path.exists():
        return
    route = mapping.load_routes(DATA_ROOT / "101")[r]
    floor = FloorMask.load(map_agent.FLOOR_MASK_PATH)
    camera = camera_for(fx)
    sift = cv2.SIFT_create()
    rng = np.random.default_rng(0)
    n = len(route.frames)
    raw = np.full((n, 3), np.nan)
    yaw_m, conf = np.zeros(n), np.zeros(n)
    previous = None
    for i in tqdm(range(n), desc=f"{tag} r{r}", position=r, leave=False):
        view = wall_view(route.frames[i], camera, floor, sift)
        yaw_m[i], conf[i] = manhattan_yaw(view.points)
        if previous is not None:
            step = visual_odometry(previous, view, rng)
            if step is not None:
                raw[i] = step
        previous = view
    np.savez(path, raw=raw, yaw_m=yaw_m, conf=conf)


# -- integration: the pose, the snap rule


def integrate(
    p, actions, motion, start, snap_deg: float, max_deg: float = mapping.MAX_STEP_DEG
) -> dict:
    n = len(p["conf"])
    poses, used = np.empty((n, 3)), np.zeros((n, 3))
    outcome = np.zeros(n, int)
    vo = np.zeros(n, bool)
    pose = start
    free_yaw = start[2]
    free = np.empty(n)
    for i in range(n):
        if i:
            step = tuple(p["raw"][i]) if np.isfinite(p["raw"][i, 0]) else None
            if step is not None and not (
                math.hypot(step[0], step[1]) <= mapping.MAX_STEP_M
                and abs(step[2]) <= math.radians(max_deg)
            ):
                step = None
            vo[i] = step is not None
            if step is None:
                e = motion.get(actions[i]) if actions[i] else None
                step = (e["dx"], e["dy"], e["dyaw"]) if e else (0.0, 0.0, 0.0)
            used[i] = step
            pose = compose(pose, step)
            free_yaw += step[2]
            if p["conf"][i] >= MANHATTAN_CONFIDENCE:
                yaw = snap(pose[2], p["yaw_m"][i])
                if abs(wrap(yaw - pose[2])) <= math.radians(snap_deg):
                    pose = (pose[0], pose[1], yaw)
                    outcome[i] = APPLIED
                else:
                    outcome[i] = REFUSED
            else:
                outcome[i] = UNSURE
        poses[i] = pose
        free[i] = free_yaw
    return {"poses": poses, "used": used, "outcome": outcome, "vo": vo, "free_yaw": free}


def start_yaw_route1(fx: float, poses0: np.ndarray, snap_deg: float, routes, floor) -> float:
    camera = camera_for(fx)
    first = wall_view(routes[1].frames[0], camera, floor)
    reference = [
        (
            wall_view(routes[0].frames[k * NODE_EVERY], camera, floor),
            float(poses0[k * NODE_EVERY, 2]),
        )
        for k in range(mapping.START_FRAMES // NODE_EVERY + 1)
    ]
    found = mapping.start_yaw(first, reference, camera.cx, camera.fx)
    yaw = 0.0 if found is None else found
    yaw_m, confidence = manhattan_yaw(first.points)
    if confidence >= MANHATTAN_CONFIDENCE:
        snapped = snap(yaw, yaw_m)
        if abs(wrap(snapped - yaw)) <= math.radians(snap_deg):
            yaw = snapped
    return yaw


# -- metrics


def path_length(truth: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(truth[:, :2], axis=0).T))])


def evaluate(nodes: list[np.ndarray], truth_nodes: list[np.ndarray], lengths: list[np.ndarray]):
    """Node poses per route against the truth (map frame): per-route error series, the marks
    table rows, the fold rate and the map error at CHECK_FRAMES (route 0)."""
    out = {"routes": []}
    for p, t, length in zip(nodes, truth_nodes, lengths, strict=True):
        pos = np.hypot(*(p[:, :2] - t[:, :2]).T)
        yaw = np.degrees(np.abs(wrap(p[:, 2] - t[:, 2])))
        marks = {}
        for m in PATH_MARKS:
            if length[-1] >= m:
                k = int(np.searchsorted(length, m))
                marks[m] = (pos[k], yaw[k])
        out["routes"].append({"pos": pos, "yaw": yaw, "marks": marks})
    allp, allt = np.concatenate(nodes), np.concatenate(truth_nodes)
    pairs = cKDTree(allp[:, :2]).query_pairs(FOLD_NEAR_M, output_type="ndarray")
    far = np.hypot(*(allt[pairs[:, 0], :2] - allt[pairs[:, 1], :2]).T) > FOLD_FAR_M
    out["fold"] = float(far.mean()) if len(pairs) else math.nan
    out["pairs"] = len(pairs)
    out["check"] = {
        f: (out["routes"][0]["pos"][f // NODE_EVERY], out["routes"][0]["yaw"][f // NODE_EVERY])
        for f in CHECK_FRAMES
    }
    return out


def metric_rows(name: str, ev: dict) -> tuple[list, list]:
    """(summary rows per route, marks rows per route)."""
    summary, marks = [], []
    for r, d in enumerate(ev["routes"]):
        summary.append(
            [
                name,
                r,
                *(
                    f"{f(d[k]):.2f}"
                    for k in ("pos", "yaw")
                    for f in (np.median, lambda v: pct(v, 90), np.max)
                ),
            ]
        )
        for m, (pe, ye) in d["marks"].items():
            marks.append([name, r, m, f"{pe:.2f}", f"{ye:.1f}"])
    return summary, marks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()
    seed = args.seed
    root = DATA_ROOT / str(seed)
    out = OUT_ROOT / f"{seed}-yaw"
    out.mkdir(parents=True, exist_ok=True)
    truth = [np.array(json.loads((root / f"traj_{t}" / "poses.json").read_text())) for t in (0, 1)]
    frame0 = Frame((0.0, 0.0, 0.0), tuple(truth[0][0]))
    truth_map = [frame0.to_map(t) for t in truth]
    gauge = math.degrees(wrap(truth[0][0, 2]))  # true world yaw of route 0's first frame
    routes = mapping.load_routes(root)
    floor = FloorMask.load(map_agent.FLOOR_MASK_PATH)
    motion = json.loads((root / "motion.json").read_text())
    lengths = [path_length(t) for t in truth]

    fit = fit_fx(seed, truth[0], out)
    fx_true = round(fit["fx_tied_median"], 3)

    jobs = [
        (r, fx, tag, str(out))
        for fx, tag in ((SHIPPED_FX, "fx92"), (fx_true, "fxT"))
        for r in (0, 1)
    ]
    with mp.get_context("spawn").Pool(4) as pool:
        pool.map(run_pass, jobs, chunksize=1)
    passes = {
        tag: [dict(np.load(out / f"pass_{tag}_r{r}.npz")) for r in (0, 1)]
        for tag in ("fx92", "fxT")
    }

    def dead_reckon(
        tag: str, fx: float, snap_deg: float, max_deg: float = mapping.MAX_STEP_DEG
    ) -> dict:
        res = [
            integrate(passes[tag][0], routes[0].actions, motion, (0.0, 0.0, 0.0), snap_deg, max_deg)
        ]
        yaw1 = start_yaw_route1(fx, res[0]["poses"], snap_deg, routes, floor)
        res.append(
            integrate(
                passes[tag][1], routes[1].actions, motion, (0.0, 0.0, yaw1), snap_deg, max_deg
            )
        )
        return {"routes": res, "start_yaw1": yaw1, "fx": fx}

    variants = {
        "(a) shipped": ("fx92", SHIPPED_FX, 20.0),
        "(b) SNAP_DEG 45": ("fx92", SHIPPED_FX, 45.0),
        "(c) true fx": ("fxT", fx_true, 20.0),
        "(d) SNAP_DEG 45 + true fx": ("fxT", fx_true, 45.0),
        # not asked for: the yaw budget of section 1 points at the step gate
        "(e) MAX_STEP_DEG 45 (extra)": ("fx92", SHIPPED_FX, 20.0, 45.0),
        "(f) MAX_STEP_DEG 45 + SNAP_DEG 45 (extra)": ("fx92", SHIPPED_FX, 45.0, 45.0),
    }
    dr = {name: dead_reckon(*v) for name, v in variants.items()}

    lines = [f"# Yaw drift and the turn bias, seed {seed}", ""]

    # check against the shipped cache
    cache_path = OUT_ROOT / f"{seed}-map" / "cache.pkl"
    if cache_path.exists():
        import pickle

        cache = pickle.loads(cache_path.read_bytes())
        gap = max(
            float(np.abs(a["poses"] - b).max())
            for a, b in zip(dr["(a) shipped"]["routes"], cache["dead"], strict=True)
        )
        lines += [
            f"Variant (a) against the cached shipped dead reckoning: max pose difference {gap:.3g} "
            f"({'reproduced' if gap < 1e-6 else 'NOT reproduced'}).",
            "",
        ]

    # ---- 1. route 0 yaw timeline
    a = dr["(a) shipped"]["routes"][0]
    t0 = truth_map[0]
    world0 = truth[0][:, 2]
    err_frame = np.degrees(wrap(a["poses"][:, 2] - t0[:, 2]))
    err_world = np.degrees(wrap(a["poses"][:, 2] - world0))
    err_free = np.degrees(wrap(a["free_yaw"] - t0[:, 2]))
    conf0 = passes["fx92"][0]["conf"]
    write_csv(
        out / "yaw_timeline_r0.csv",
        [
            "frame",
            "path_m",
            "action",
            "true_yaw_world_deg",
            "map_yaw_deg",
            "err_frame_deg",
            "err_world_deg",
            "err_unsnapped_frame_deg",
            "manhattan_conf",
            "snap",
        ],
        [
            [
                i,
                f"{lengths[0][i]:.3f}",
                routes[0].actions[i] or "",
                f"{math.degrees(world0[i]):.2f}",
                f"{math.degrees(a['poses'][i, 2]):.2f}",
                f"{err_frame[i]:.2f}",
                f"{err_world[i]:.2f}",
                f"{err_free[i]:.2f}",
                f"{conf0[i]:.3f}",
                OUTCOME[a["outcome"][i]],
            ]
            for i in range(len(err_frame))
        ],
    )
    lines += [
        "## 1. Route 0 yaw timeline (shipped)",
        "",
        f"Route 0's first frame has true world yaw {gauge:+.2f} deg; the map gives it yaw 0, so the "  # noqa: E501
        "'frame' error (map yaw against the true yaw in the map frame) is the gauge-offset convention of "  # noqa: E501
        "the earlier analysis, and 'world' = frame error minus that offset is the map yaw against the "  # noqa: E501
        "true yaw against the maze axes, which is what the snap compares ("
        "manhattan_yaw is measured against the axes).",
        "",
    ]
    rows = []
    n = len(err_frame)
    for label, e in (("frame (gauge counted)", err_frame), ("world (gauge removed)", err_world)):
        over = np.nonzero(np.abs(e) > 20)[0]
        first = int(over[0]) if len(over) else None
        row = [
            label,
            first if first is not None else "never",
            f"{lengths[0][first]:.1f}" if first is not None else "",
        ]
        for sl in (slice(1, first), slice(first, n)) if first else (slice(1, n), slice(0, 0)):
            o = a["outcome"][sl]
            if len(o) == 0:
                row += ["", ""]
                continue
            conf = (o == APPLIED) | (o == REFUSED)
            row += [
                f"{(o == APPLIED).mean():.1%}",
                f"{(o == APPLIED).sum() / max(conf.sum(), 1):.1%}",
            ]
        rows.append(row)
    lines += [
        md(
            [
                "error",
                "first frame |err| > 20 deg",
                "path m",
                "applied, before (all frames)",
                "applied, before (of confident)",
                "applied, after (all frames)",
                "applied, after (of confident)",
            ],
            rows,
        ),
        "",
    ]
    counts = {OUTCOME[k]: int((a["outcome"][1:] == k).sum()) for k in (APPLIED, REFUSED, UNSURE)}
    lines += [f"Snap outcomes over route 0: {counts}.", ""]
    rows = []
    for m in (5, 10, 20, 30, 40, 50):
        if lengths[0][-1] >= m:
            k = int(np.searchsorted(lengths[0], m))
            rows.append(
                [
                    m,
                    k,
                    f"{err_frame[k]:+.1f}",
                    f"{err_world[k]:+.1f}",
                    f"{err_free[k]:+.1f}",
                    OUTCOME[a["outcome"][k]],
                ]
            )
    lines += [
        md(
            [
                "path m",
                "frame",
                "err frame deg",
                "err world deg",
                "unsnapped integrated err (frame) deg",
                "snap at frame",
            ],
            rows,
        ),
        "",
    ]

    # yaw error budget: where the unsnapped yaw error comes from
    step_err = np.degrees(a["used"][:, 2]) - np.degrees(
        [0.0, *(wrap(world0[i] - world0[i - 1]) for i in range(1, n))]
    )
    budget = []
    for lim_frame, label in (
        (int(np.searchsorted(lengths[0], 20)), "20 m"),
        (int(np.searchsorted(lengths[0], 30)), "30 m"),
    ):
        for src, vo in (("VO step used", True), ("fallback (calibrated mean)", False)):
            for act in ("FORWARD", "BACKWARD", "LEFT", "RIGHT", None):
                sel = np.array([x == act for x in routes[0].actions]) & (a["vo"] == vo)
                sel[lim_frame:] = False
                sel[0] = False
                if sel.sum():
                    budget.append(
                        [
                            label,
                            src,
                            act or "none/several",
                            int(sel.sum()),
                            f"{step_err[sel].sum():+.1f}",
                        ]
                    )
    lines += [
        "Unsnapped yaw error budget (sum of used dyaw minus true dyaw, deg) over route 0's first 20 / 30 m:",  # noqa: E501
        "",
        md(["to", "step source", "action", "frames", "sum error deg"], budget),
        "",
    ]
    big = np.nonzero((np.abs(step_err) > 3) & ~a["vo"])[0]
    rawdeg = np.degrees(passes["fx92"][0]["raw"][:, 2])
    lines += [
        "Fallback frames whose true turn is over 3 deg off the fallback (first 12): frame, action, true dyaw deg, "  # noqa: E501
        "raw VO dyaw deg (rejected by plausible: |dyaw| > MAX_STEP_DEG = 20):",
        "",
        md(
            ["frame", "action", "true dyaw", "raw VO dyaw"],
            [
                [
                    i,
                    routes[0].actions[i],
                    f"{-step_err[i] + np.degrees(a['used'][i, 2]):+.1f}",
                    f"{rawdeg[i]:+.1f}",
                ]
                for i in big[:12]
            ],
        ),
        "",
    ]

    # ---- 2. turn bias
    lines += ["## 2. Turn bias", ""]
    d = camera_for(SHIPPED_FX)
    rep = fit["reported"]
    lines += [
        f"Camera mapping.py uses (geometry.CAMERA + HORIZON_SHIFT_PX): width {d.width}, height {d.rows}, "  # noqa: E501
        f"fx {d.fx}, fy {d.fy}, cx {d.cx}, cy {d.cy} (120 deg FOV rounded: 160 / tan 60 = {fit['fov120_fx']:.3f}). "  # noqa: E501
        f"The simulator reports fx {rep['fx']}, fy {rep['fy']}, cx {rep['cx']}, cy {rep['cy']}. "
        f"Fitted from {fit['pairs']} render pairs turned by a known 3 / 6 / 10 deg (pure rotation, "
        f"SIFT matches): fx = {fit['fx_tied_median']:.3f} (fx = fy tied; range "
        f"{fit['fx_tied_min_max'][0]:.3f}-{fit['fx_tied_min_max'][1]:.3f}); free fit fx {fit['fx_free_median']:.3f}, "  # noqa: E501
        f"fy {fit['fy_free_median']:.3f}. Variants (c) / (d) use fx = {fx_true} and leave fy at 92 "
        "(the camera dict has one fy used for the ranges, whose HORIZON_SHIFT_PX was tuned with it).",  # noqa: E501
        "",
    ]
    rows, csv = [], []
    for tag, label in (("fx92", "fx 92 (shipped)"), ("fxT", f"fx {fx_true}")):
        for r in (0, 1):
            dr_ = dr["(a) shipped"] if tag == "fx92" else dr["(c) true fx"]
            res = dr_["routes"][r]
            tw = truth[r][:, 2]
            for which in ("LEFT", "RIGHT", "both"):
                sel = np.array(
                    [
                        (x in ("LEFT", "RIGHT")) if which == "both" else x == which
                        for x in routes[r].actions
                    ]
                )
                sel &= res["vo"]
                sel[0] = False
                idx = np.nonzero(sel)[0]
                vo_sum = math.degrees(res["used"][idx, 2].sum())
                true_sum = math.degrees(sum(wrap(tw[i] - tw[i - 1]) for i in idx))
                row = [
                    label,
                    r,
                    which,
                    len(idx),
                    f"{vo_sum:+.1f}",
                    f"{true_sum:+.1f}",
                    f"{vo_sum / true_sum:.4f}" if true_sum else "",
                    f"{vo_sum - true_sum:+.2f}",
                ]
                rows.append(row)
                csv.append(row)
    hdr = [
        "camera",
        "route",
        "frames",
        "n",
        "sum VO dyaw deg",
        "sum true dyaw deg",
        "ratio",
        "diff deg",
    ]
    write_csv(out / "turn_bias.csv", hdr, csv)
    lines += [
        md(hdr, rows),
        "",
        "Frames are single-action LEFT / RIGHT frames on which the VO step was used "
        "(not the calibrated fallback).",
        "",
    ]

    # ---- 3. dead-reckoning variants
    lines += [
        "## 3. Dead-reckoning variants",
        "",
        "Errors are against the true poses in the map frame, at the nodes (every 5th frame). "
        "Position in m, yaw in deg (gauge counted, as in the earlier analysis). The fold rate is "
        f"the share of node pairs the map puts within {FOLD_NEAR_M} m that are over {FOLD_FAR_M} m "
        "apart in truth (both routes).",
        "",
    ]
    truth_nodes = [t[::NODE_EVERY] for t in truth_map]
    len_nodes = [length[::NODE_EVERY] for length in lengths]

    def nodes_of(v):
        return [r_["poses"][::NODE_EVERY] for r_ in v["routes"]]

    evs = {name: evaluate(nodes_of(v), truth_nodes, len_nodes) for name, v in dr.items()}
    sum_rows, mark_rows, head_rows = [], [], []
    for name, ev in evs.items():
        s, m = metric_rows(name, ev)
        sum_rows += s
        mark_rows += m
        head_rows.append(
            [
                name,
                f"{ev['fold']:.1%} ({ev['pairs']} pairs)",
                *(f"{ev['check'][f][0]:.2f} m / {ev['check'][f][1]:.1f} deg" for f in CHECK_FRAMES),
            ]
        )
    shdr = [
        "variant",
        "route",
        "pos median",
        "pos p90",
        "pos max",
        "yaw median",
        "yaw p90",
        "yaw max",
    ]
    mhdr = ["variant", "route", "path m", "pos err m", "yaw err deg"]
    write_csv(out / "dr_summary.csv", shdr, sum_rows)
    write_csv(out / "dr_by_path.csv", mhdr, mark_rows)
    lines += [
        md(
            ["variant", "fold rate", *(f"map error at frame {f} (route 0)" for f in CHECK_FRAMES)],
            head_rows,
        ),
        "",
        "Error over all nodes of a route:",
        "",
        md(shdr, sum_rows),
        "",
        "Error at path-length marks:",
        "",
    ]
    table = {}
    for name, route, m, pe, ye in mark_rows:
        table.setdefault((route, m), {})[name] = f"{pe} m / {ye} deg"
    lines += [
        md(
            ["route", "path m", *variants],
            [[r, m, *(v.get(n, "") for n in variants)] for (r, m), v in sorted(table.items())],
        ),
        "",
    ]
    best = min(
        evs, key=lambda k: np.percentile(np.concatenate([d_["pos"] for d_ in evs[k]["routes"]]), 90)
    )
    lines += [f"Best variant by all-node position p90: **{best}**.", ""]

    # ---- 4. pose graph on the best variant
    import eval_dev_like_map as edm

    tag, fx = variants[best][:2]
    v = dr[best]
    used_err = []
    for r in (0, 1):
        res = v["routes"][r]
        rel = np.array(
            [between(tuple(truth[r][i - 1]), tuple(truth[r][i])) for i in range(1, len(truth[r]))]
        )
        sel = res["vo"][1:]
        e = res["used"][1:][sel] - rel[sel]
        e[:, 2] = wrap(e[:, 2])
        used_err.append(e)
    e = np.concatenate(used_err)
    rms_xy = math.sqrt(float((e[:, :2] ** 2).sum(axis=1).mean() / 2))
    rms_yaw = math.degrees(math.sqrt(float((e[:, 2] ** 2).mean())))
    sig_m, sig_deg = rms_xy * math.sqrt(NODE_EVERY), rms_yaw * math.sqrt(NODE_EVERY)
    lines += [
        "## 4. Pose graph on the best variant",
        "",
        f"Per-frame VO error of {best} (used steps, {len(e)} frames): per-axis translation RMS "
        f"{rms_xy * 1000:.1f} mm, yaw RMS {rms_yaw:.3f} deg. Node edges span {NODE_EVERY} frames, so the sigmas are "  # noqa: E501
        f"x sqrt({NODE_EVERY}): {sig_m * 1000:.1f} mm, {sig_deg:.2f} deg (shipped: "
        f"{mapping.ODOM_SIGMA_M * 1000:.0f} mm, {mapping.ODOM_SIGMA_DEG:.0f} deg).",
        "",
    ]

    cache_closures = cache["closures"] if tag == "fx92" else None
    if cache_closures is None:
        camera = camera_for(fx)
        sift = cv2.SIFT_create()
        tracked = []
        for r, route in enumerate(routes):
            nodes = []
            for i in tqdm(range(0, len(route.frames), NODE_EVERY), desc=f"nodes r{r}", leave=False):
                view = wall_view(route.frames[i], camera, floor, sift)
                nodes.append(mapping.Node(view, clip_range(view.points, mapping.GRID_MAX_RANGE)))
            tracked.append(SimpleNamespace(nodes=nodes))
        closures, cand, _ = mapping.find_closures(tracked, mapping.Timer())
    else:
        closures = cache_closures
    lines += [f"{len(closures)} verified closures.", ""]

    dead = [r_["poses"] for r_ in v["routes"]]
    starts = [(0.0, 0.0, 0.0), (0.0, 0.0, v["start_yaw1"])]
    stub = edm.stub(dead)
    real = mapping.least_squares
    log: list = []

    def converge(*a_, **kw):
        kw.update(max_nfev=CONVERGED_NFEV)
        res = real(*a_, **kw)
        log.append((res.status, res.nfev, float(res.cost), res.message))
        return res

    def capped(*a_, **kw):
        res = real(*a_, **kw)
        log.append((res.status, res.nfev, float(res.cost), res.message))
        return res

    configs = [
        (
            "shipped solver (cap 60), shipped sigmas",
            capped,
            (mapping.ODOM_SIGMA_M, mapping.ODOM_SIGMA_DEG),
        ),
        ("converged, shipped sigmas", converge, (mapping.ODOM_SIGMA_M, mapping.ODOM_SIGMA_DEG)),
        ("converged, measured sigmas", converge, (sig_m, sig_deg)),
    ]
    saved = (mapping.ODOM_SIGMA_M, mapping.ODOM_SIGMA_DEG)
    g_sum, g_marks, g_head = [], [], []
    for label, fn, sig in configs:
        mapping.least_squares = fn
        mapping.ODOM_SIGMA_M, mapping.ODOM_SIGMA_DEG = sig
        log.clear()
        poses, kept, _ = edm.solve_rounds(stub, starts, closures, pregate=True)
        mapping.least_squares = real
        mapping.ODOM_SIGMA_M, mapping.ODOM_SIGMA_DEG = saved
        ev = evaluate(poses, truth_nodes, len_nodes)
        s, m = metric_rows(label, ev)
        g_sum += s
        g_marks += m
        status = "; ".join(f"status {x[0]}, nfev {x[1]}" for x in log)
        g_head.append(
            [
                label,
                len(kept),
                f"{ev['fold']:.1%}",
                *(f"{ev['check'][f][0]:.2f} m / {ev['check'][f][1]:.1f} deg" for f in CHECK_FRAMES),
                status,
            ]
        )
    write_csv(out / "graph_summary.csv", shdr, g_sum)
    write_csv(out / "graph_by_path.csv", mhdr, g_marks)
    lines += [
        md(
            [
                "solve",
                "closures kept",
                "fold rate",
                *(f"map error at frame {f}" for f in CHECK_FRAMES),
                "least_squares (per round)",
            ],
            g_head,
        ),
        "",
        md(shdr, g_sum),
        "",
    ]
    table = {}
    for name, route, m, pe, ye in g_marks:
        table.setdefault((route, m), {})[name] = f"{pe} m / {ye} deg"
    lines += [
        md(
            ["route", "path m", *(c[0] for c in configs)],
            [[r, m, *(v_.get(c[0], "") for c in configs)] for (r, m), v_ in sorted(table.items())],
        ),
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")

    # ---- plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, grid = "#222222", "#dddddd"
    colors = {
        "a": "#2a6fbb",
        "b": "#d9731a",
        "c": "#1b9e8a",
        "d": "#7b52ab",
        "e": "#c0392b",
        "f": "#555555",
    }
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    x = lengths[0]
    ax1.plot(x, err_frame, color=colors["a"], lw=1.2, label="map yaw, gauge counted")
    ax1.plot(
        x, err_world, color=colors["b"], lw=1.2, label="map yaw, gauge removed (what the snap sees)"
    )
    ax1.plot(x, err_free, color="#888888", lw=1.0, label="unsnapped integrated yaw (gauge counted)")
    for s_ in (20, -20):
        ax1.axhline(s_, color=ink, lw=0.8, ls="--")
    ref = a["outcome"] == REFUSED
    ax1.plot(
        x[ref],
        np.full(ref.sum(), ax1.get_ylim()[0]),
        "|",
        color="#c0392b",
        ms=6,
        label="snap refused",
    )
    ax1.set_ylabel("yaw error, deg")
    ax1.set_title(
        "Route 0, shipped dead reckoning: yaw error against path length", loc="left", color=ink
    )
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    for key, name in zip("abcdef", variants, strict=True):
        e_ = np.degrees(wrap(dr[name]["routes"][0]["poses"][:, 2] - t0[:, 2]))
        ax2.plot(x, e_, color=colors[key], lw=1.2, label=name)
    ax2.axhline(0, color=ink, lw=0.6)
    ax2.set_ylabel("yaw error, deg (gauge counted)")
    ax2.set_xlabel("true path length, m")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in (ax1, ax2):
        ax.grid(color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "route0_yaw.png", dpi=140)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
