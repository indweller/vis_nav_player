"""Why the dev-like map folds, and whether other closure rules would have fixed it. No change to
mapping.py: its stages are repeated here with the intermediate results kept.

    uv run source/eval/eval_dev_like_map.py            # builds the cache once, then analyses
    uv run source/eval/eval_dev_like_map.py --rebuild  # redo the cache

Stage 1 (cached in results/eval/dev_like/<seed>-map/cache.pkl): mapping.build's odometry with every
tick logged (the visual odometry step or the calibrated fallback, snapping), the start yaw of the
later routes, and ``find_closures`` with every candidate kept (matches, epipolar inliers, the rigid
fit, whether it verified) instead of only the verified closures. The same rng draws are made in
the same order as mapping.py's, so the result is its map (checked against the frozen map).

Stage 2 (map.md, candidates.csv, vo_frames.csv beside the cache):
1. every closure candidate: the true distance between its two frames, its verification, its fate
   in mapping.solve_rounds' rule (failed verification, pre-gate rejected, dropped in consistency
   round 1 / 2 / 3, kept), and the confusion of "true closure" (frames under TRUE_M apart)
   against fate;
2. drift: the dead-reckoned and optimised position and yaw error against the path length along
   each route, and the per-frame visual odometry error (translation, yaw) against the truth;
3. the pose graph solved again from the cached closures: (a) pre-gate off, (b) pre-gate off with
   a consistency threshold that grows with the path distance of the closure's nearer node from
   its route's start, (c) only the true closures. The node position error (median, p90, max) and
   the fold rate: node pairs the map puts within FOLD_NEAR_M that are over FOLD_FAR_M apart;
4. the map error at frames 2390 and 2395 of route 0.

The map frame is route 0's: its first frame is (0, 0, 0). Errors are against the true poses
placed in that frame by the rigid motion that takes route 0's first true pose to (0, 0, 0)
(eval_geometry.Frame), so a route's start yaw error counts.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frozen_map  # noqa: E402
import geometry  # noqa: E402
import map_agent  # noqa: E402
import mapping  # noqa: E402
from eval_dev_like import DATA_ROOT  # noqa: E402
from eval_geometry import Frame  # noqa: E402
from geometry import (  # noqa: E402
    MANHATTAN_CONFIDENCE,
    Camera,
    FloorMask,
    clip_range,
    epipolar_inliers,
    manhattan_yaw,
    match_features,
    relative_pose,
    snap,
    visual_odometry,
    wall_view,
)
from mapping import (  # noqa: E402
    GRID_MAX_RANGE,
    MIN_SIFT_INLIERS,
    NODE_EVERY,
    PRE_GATE_DEG,
    PRE_GATE_M,
    QUERY_EVERY,
    ROUNDS,
    TOP_K,
    between,
    compose,
    plausible,
    wrap,
)
from my_agent import Float32VLAD  # noqa: E402

OUT_ROOT = Path("results") / "eval" / "dev_like"
TRUE_M = 0.3
FOLD_NEAR_M, FOLD_FAR_M = 0.15, 0.5
PATH_MARKS = (5, 10, 20, 30, 50, 75, 100, 150)
FRAMES = (2390, 2395)
NO_SOURCE, VO, FALLBACK = 0, 1, 2


# -- stage 1: the pipeline, with its intermediate results kept


def trace_odometry(route, start, camera, floor, motion, label: str) -> dict:
    """mapping.odometry, keeping per frame the step used, where it came from and the raw visual
    odometry step."""
    sift = cv2.SIFT_create()
    rng = np.random.default_rng(0)
    n = len(route.frames)
    poses = np.empty((n, 3))
    used, raw = np.zeros((n, 3)), np.full((n, 3), np.nan)
    source, snapped = np.zeros(n, int), np.zeros(n, bool)
    views = []
    pose, previous = start, None
    for i in tqdm(range(n), desc=label, leave=False):
        view = wall_view(route.frames[i], camera, floor, sift)
        yaw_m, confidence = manhattan_yaw(view.points)
        if previous is not None:
            action = route.actions[i]
            expected = motion.get(action) if action else None
            step = visual_odometry(previous, view, rng)
            if step is not None:
                raw[i] = step
            if step is not None and not plausible(step):
                step = None
            if step is None:
                source[i] = FALLBACK
                step = (expected["dx"], expected["dy"], expected["dyaw"]) if expected else (0, 0, 0)
            else:
                source[i] = VO
            used[i] = step
            pose = compose(pose, step)
            if confidence >= MANHATTAN_CONFIDENCE:
                pose = (pose[0], pose[1], snap(pose[2], yaw_m))
                snapped[i] = True
        poses[i] = pose
        if i % NODE_EVERY == 0:
            views.append(mapping.Node(view, clip_range(view.points, GRID_MAX_RANGE)))
        previous = view
    return {
        "poses": poses,
        "used": used,
        "raw": raw,
        "source": source,
        "snapped": snapped,
        "nodes": views,
    }


def trace_closures(tracked: list[dict]) -> tuple[list[dict], list[mapping.Closure]]:
    """mapping.find_closures with every candidate kept: ``(candidates, verified closures)``."""
    index = [(r, k) for r, t in enumerate(tracked) for k in range(len(t["nodes"]))]
    nodes = [tracked[r]["nodes"][k] for r, k in index]
    extractor = Float32VLAD(cache_dir=Path("cache") / "mapping")
    mapping.fit_codebook(nodes, extractor)
    database = np.array([extractor._vlad(n.view.des) for n in nodes], np.float32)
    route = np.array([r for r, _ in index])
    frame = np.array([k * NODE_EVERY for _, k in index])
    queries = [q for q, (_, k) in enumerate(index) if (k * NODE_EVERY) % QUERY_EVERY == 0]
    rng = np.random.default_rng(0)
    candidates: list[dict] = []
    closures: list[mapping.Closure] = []
    seen: set[tuple[int, int]] = set()
    similarity = database[queries] @ database.T
    for row, q in enumerate(tqdm(queries, desc="closures", leave=False)):
        same_route = (route == route[q]) & (np.abs(frame - frame[q]) < mapping.EXCLUDE_TICKS)
        scores = np.where(same_route | ~database.any(axis=1), -np.inf, similarity[row])
        if not database[q].any():
            continue
        top = np.argsort(-scores)[:TOP_K]
        for c in top[np.isfinite(scores[top])]:
            a, b = (int(c), q) if (route[c], frame[c]) < (route[q], frame[q]) else (q, int(c))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            va, vb = nodes[a].view, nodes[b].view
            record = {
                "a": (int(route[a]), int(frame[a])),
                "b": (int(route[b]), int(frame[b])),
                "similarity": float(similarity[row, c]),
                "matches": 0,
                "inliers": 0,
                "fit": False,
                "wall_inliers": 0,
                "pose": None,
                "verified": False,
            }
            matches = match_features(va, vb)
            record["matches"] = len(matches)
            if len(matches) >= MIN_SIFT_INLIERS:
                inliers = epipolar_inliers(va, vb, matches)
                record["inliers"] = int(inliers.sum())
                if inliers.sum() >= MIN_SIFT_INLIERS:
                    fit = relative_pose(va, vb, matches[inliers], rng)
                    if fit is not None:
                        record["fit"] = True
                        record["wall_inliers"] = fit[1]
                        record["pose"] = tuple(float(v) for v in fit[0])
                        record["verified"] = True
                        closures.append(
                            mapping.Closure(
                                record["a"], record["b"], fit[0], int(inliers.sum()), fit[1]
                            )
                        )
            record["closure"] = len(closures) - 1 if record["verified"] else -1
            candidates.append(record)
    return candidates, closures


def build_cache(seed: int, path: Path) -> dict:
    root = DATA_ROOT / str(seed)
    routes = mapping.load_routes(root)
    floor = FloorMask.load(map_agent.FLOOR_MASK_PATH)
    camera_dict = dict(geometry.CAMERA)
    camera = Camera({**camera_dict, "cy": camera_dict["cy"] + mapping.HORIZON_SHIFT_PX})
    motion = json.loads((root / "motion.json").read_text())
    tracked, starts = [], []
    for r, route in enumerate(routes):
        yaw = 0.0
        if r:
            first = wall_view(route.frames[0], camera, floor)
            reference = [
                (tracked[0]["nodes"][k].view, float(tracked[0]["poses"][k * NODE_EVERY, 2]))
                for k in range(mapping.START_FRAMES // NODE_EVERY + 1)
            ]
            found = mapping.start_yaw(first, reference, camera.cx, camera.fx)
            yaw = 0.0 if found is None else found
            yaw_m, confidence = manhattan_yaw(first.points)
            if confidence >= MANHATTAN_CONFIDENCE:
                yaw = snap(yaw, yaw_m)
        starts.append((0.0, 0.0, yaw))
        tracked.append(trace_odometry(route, starts[-1], camera, floor, motion, f"route {r}"))
    candidates, closures = trace_closures(tracked)
    cache = {
        "starts": starts,
        "dead": [t["poses"] for t in tracked],
        "used": [t["used"] for t in tracked],
        "raw": [t["raw"] for t in tracked],
        "source": [t["source"] for t in tracked],
        "snapped": [t["snapped"] for t in tracked],
        "actions": [r.actions for r in routes],
        "candidates": candidates,
        "closures": closures,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(cache))
    return cache


# -- stage 2


def stub(dead: list[np.ndarray]) -> list:
    """What mapping.solve needs of ``tracked``: ``poses`` and ``nodes`` (counted)."""
    return [SimpleNamespace(poses=d, nodes=range(len(d[::NODE_EVERY]))) for d in dead]


def solve_rounds(
    tracked,
    starts,
    closures: list[mapping.Closure],
    *,
    pregate: bool,
    limits=None,
):
    """mapping.solve_rounds with the fate of every closure: ``(poses, kept, fates)``, fates the
    index -> "pre-gate" | "dropped r<k>" | "kept". ``limits(closure)`` gives the consistency
    thresholds (m, deg); the default is MAX_CLOSURE_M / _DEG."""
    limits = limits or (lambda c: (mapping.MAX_CLOSURE_M, mapping.MAX_CLOSURE_DEG))
    dead = [t.poses[::NODE_EVERY] for t in tracked]
    fates: dict[int, str] = {}
    if pregate:
        kept = []
        for i, c in enumerate(closures):
            if mapping.consistent(c, dead, PRE_GATE_M, PRE_GATE_DEG):
                kept.append(i)
            else:
                fates[i] = "pre-gate"
    else:
        kept = list(range(len(closures)))
    for round_ in range(1, ROUNDS + 1):
        poses = mapping.solve(tracked, starts, [closures[i] for i in kept])
        agreeing = [i for i in kept if mapping.consistent(closures[i], poses, *limits(closures[i]))]
        for i in set(kept) - set(agreeing):
            fates[i] = f"dropped r{round_}"
        if len(agreeing) == len(kept):
            break
        kept = agreeing
    else:
        poses = mapping.solve(tracked, starts, [closures[i] for i in kept])
    for i in kept:
        fates[i] = "kept"
    return poses, kept, fates


def md(header: list[str], rows: list[list]) -> str:
    sep = "|" + "|".join("---" for _ in header) + "|"
    body = ("| " + " | ".join(map(str, r)) + " |" for r in rows)
    return "\n".join(["| " + " | ".join(header) + " |", sep, *body])


def pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else math.nan


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()
    out = OUT_ROOT / f"{args.seed}-map"
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "cache.pkl"
    if cache_path.exists() and not args.rebuild:
        cache = pickle.loads(cache_path.read_bytes())
    else:
        cache = build_cache(args.seed, cache_path)
    root = DATA_ROOT / str(args.seed)
    truth = [np.array(json.loads((root / f"traj_{t}" / "poses.json").read_text())) for t in (0, 1)]
    frame0 = Frame((0.0, 0.0, 0.0), tuple(truth[0][0]))
    truth_map = [frame0.to_map(t) for t in truth]  # (frames, 3) in the map frame
    dead, starts = cache["dead"], cache["starts"]
    tracked = stub(dead)
    closures, candidates = cache["closures"], cache["candidates"]
    node_truth = [t[::NODE_EVERY] for t in truth_map]
    lines = [f"# Map fidelity, seed {args.seed}", ""]

    # The default rule, and the check against the frozen map.
    poses0, kept0, fates0 = solve_rounds(tracked, starts, closures, pregate=True)
    floor = FloorMask.load(map_agent.FLOOR_MASK_PATH)
    store, _ = frozen_map.get(root, dict(geometry.CAMERA), floor)
    frozen = [np.asarray(p)[::NODE_EVERY] for p in store.poses]
    gap = max(
        float(np.hypot(*(a[:, :2] - b[:, :2]).T).max()) for a, b in zip(poses0, frozen, strict=True)
    )
    lines += [
        f"Rebuilt: {len(candidates)} candidates, {len(closures)} verified, {len(kept0)} kept by "
        f"the default rule; its node poses differ from the frozen map's by at most {gap:.3g} m "
        f"({'reproduced' if gap < 1e-2 else 'NOT REPRODUCED'}).",
        "",
    ]

    # Truth helpers.
    def true_pose(route: int, frame: int) -> np.ndarray:
        return truth[route][frame]

    for c in candidates:
        ta, tb = true_pose(*c["a"]), true_pose(*c["b"])
        c["true_m"] = float(math.hypot(*(ta[:2] - tb[:2])))
        rel_true = between(tuple(ta), tuple(tb))
        c["true_yaw_deg"] = math.degrees(abs(wrap(tb[2] - ta[2])))
        if c["pose"] is not None:
            c["pose_err_m"] = float(
                math.hypot(c["pose"][0] - rel_true[0], c["pose"][1] - rel_true[1])
            )
            c["pose_err_deg"] = math.degrees(abs(wrap(c["pose"][2] - rel_true[2])))
        else:
            c["pose_err_m"] = c["pose_err_deg"] = math.nan

    def fate_of(c: dict, fates: dict[int, str]) -> str:
        return fates.get(c["closure"], "?") if c["verified"] else "failed verification"

    # 1. Candidates.
    lines += ["## 1. Closure candidates", ""]
    fate_names = [
        "failed verification",
        "pre-gate",
        "dropped r1",
        "dropped r2",
        "dropped r3",
        "kept",
    ]
    for c in candidates:
        c["fate"] = fate_of(c, fates0)
        c["true"] = c["true_m"] < TRUE_M
    table = []
    for label, flag in (("true (< 0.3 m apart)", True), ("false", False)):
        sub = [c for c in candidates if c["true"] == flag]
        count = Counter(c["fate"] for c in sub)
        table.append([label, len(sub), *[count[f] for f in fate_names]])
    lines += [
        "Default rule (pre-gate, then consistency rounds), fate by true closure:",
        "",
        md(["", "candidates", *fate_names], table),
        "",
    ]
    verified = [c for c in candidates if c["verified"]]
    kept_true = sum(c["true"] for c in verified if c["fate"] == "kept")
    lines += [
        f"Verified {len(verified)} of {len(candidates)} candidates; {sum(c['true'] for c in verified)} "  # noqa: E501
        f"of the verified are true. Kept: {sum(c['fate'] == 'kept' for c in verified)}, "
        f"{kept_true} of them true.",
        "",
        "Relative pose of the verified closures against the truth (true closures / false):",
        "",
    ]
    rows = []
    for label, flag in (("true", True), ("false", False)):
        sub = [c for c in verified if c["true"] == flag]
        if not sub:
            continue
        pm = np.array([c["pose_err_m"] for c in sub])
        pd = np.array([c["pose_err_deg"] for c in sub])
        ins = np.array([c["inliers"] for c in sub])
        rows.append(
            [
                label,
                len(sub),
                f"{np.median(ins):.0f} ({ins.min()}-{ins.max()})",
                f"{np.median(pm):.2f} / {pct(pm, 90):.2f}",
                f"{np.median(pd):.1f} / {pct(pd, 90):.1f}",
                f"{np.mean(pm <= 0.3):.0%}",
            ]
        )
    lines += [
        md(
            [
                "",
                "verified",
                "inliers median (range)",
                "pose error m (median / p90)",
                "yaw error deg (median / p90)",
                "pose within 0.3 m",
            ],
            rows,
        ),
        "",
    ]
    # Why true closures were not kept.
    why = Counter()
    for c in candidates:
        if c["true"]:
            why[c["fate"]] += 1
    miss = Counter()
    for c in candidates:
        if c["true"] and not c["verified"]:
            if c["matches"] < MIN_SIFT_INLIERS:
                miss["matches < 30"] += 1
            elif c["inliers"] < MIN_SIFT_INLIERS:
                miss["epipolar inliers < 30"] += 1
            else:
                miss["rigid fit failed"] += 1
    lines += [
        "True candidates that failed verification, by stage: "
        + (", ".join(f"{k} {v}" for k, v in miss.items()) or "none")
        + ".",
        "",
    ]
    # Fate of the dropped true closures: did their pose disagree with the solved poses by much,
    # was their own relative pose wrong?
    dropped_true = [
        c
        for c in verified
        if c["true"] and c["fate"] in ("pre-gate", "dropped r1", "dropped r2", "dropped r3")
    ]
    if dropped_true:
        pm = np.array([c["pose_err_m"] for c in dropped_true])
        lines += [
            f"True closures the rule dropped: {len(dropped_true)}; their own relative pose error "
            f"median {np.median(pm):.2f} m, {np.mean(pm <= 0.3):.0%} within 0.3 m (so the rest "
            "were dropped with a wrong fit, the rest with a right one).",
            "",
        ]
    # Write candidates.csv.
    import csv

    with (out / "candidates.csv").open("w", newline="") as handle:
        fields = [
            "a_route",
            "a_frame",
            "b_route",
            "b_frame",
            "true_m",
            "true_yaw_deg",
            "matches",
            "inliers",
            "fit",
            "wall_inliers",
            "verified",
            "pose_err_m",
            "pose_err_deg",
            "fate",
        ]
        writer = csv.writer(handle)
        writer.writerow(fields)
        for c in candidates:
            writer.writerow(
                [
                    c["a"][0],
                    c["a"][1],
                    c["b"][0],
                    c["b"][1],
                    f"{c['true_m']:.3f}",
                    f"{c['true_yaw_deg']:.1f}",
                    c["matches"],
                    c["inliers"],
                    int(c["fit"]),
                    c["wall_inliers"],
                    int(c["verified"]),
                    f"{c['pose_err_m']:.3f}",
                    f"{c['pose_err_deg']:.1f}",
                    c["fate"],
                ]
            )

    # 2. Drift.
    lines += ["## 2. Drift", ""]
    rows, err_ratio, yaw_ratio = [], [], []
    path_len = []
    for r in range(2):
        steps = np.hypot(*np.diff(truth[r][:, :2], axis=0).T)
        path_len.append(np.concatenate([[0.0], np.cumsum(steps)]))
        d = dead[r]
        e_pos = np.hypot(*(d[:, :2] - truth_map[r][:, :2]).T)
        e_yaw = np.degrees(np.abs(wrap(d[:, 2] - truth_map[r][:, 2])))
        tdyaw = wrap(np.diff(truth[r][:, 2]))
        e_int = np.degrees(
            np.concatenate([[0.0], np.cumsum(wrap(cache["used"][r][1:, 2] - tdyaw))])
        )
        g = np.asarray(frozen[r])
        # graph poses at nodes only: compare at node frames.
        e_graph = np.hypot(*(g[:, :2] - node_truth[r][:, :2]).T)
        pl_nodes = path_len[r][::NODE_EVERY]
        for mark in PATH_MARKS:
            if mark > path_len[r][-1]:
                continue
            i = int(np.searchsorted(path_len[r], mark))
            j = int(np.searchsorted(pl_nodes, mark))
            rows.append(
                [
                    r,
                    mark,
                    i,
                    f"{e_pos[i]:.2f}",
                    f"{e_pos[i] / mark:.3f}",
                    f"{e_yaw[i]:.1f}",
                    f"{e_int[i]:+.1f}",
                    f"{e_graph[min(j, len(e_graph) - 1)]:.2f}",
                ]
            )
        far = path_len[r] > 5.0
        err_ratio.append(e_pos[far] / path_len[r][far])
        yaw_ratio.append(e_yaw[far] / path_len[r][far])
        rows.append(
            [
                r,
                "end",
                len(d) - 1,
                f"{e_pos[-1]:.2f}",
                f"{e_pos[-1] / path_len[r][-1]:.3f}",
                f"{e_yaw[-1]:.1f}",
                f"{e_int[-1]:+.1f}",
                f"{e_graph[-1]:.2f}",
            ]
        )
    lines += [
        "Dead-reckoned (visual odometry + snapping, no closures) position and yaw error against "
        "the true path length (m, from the true poses), and the frozen graph's error at the same "
        "path length:",
        "",
        md(
            [
                "route",
                "true path m",
                "frame",
                "DR pos err m",
                "DR err / path",
                "DR yaw err deg",
                "unsnapped integrated yaw err deg",
                "graph pos err m",
            ],
            rows,
        ),
        "",
    ]
    ratio = np.concatenate(err_ratio)
    lines += [
        f"DR position error / path length (path > 5 m): median {np.median(ratio):.3f}, p90 "
        f"{pct(ratio, 90):.3f}, max {ratio.max():.3f}.",
        "",
    ]
    rate_m = pct(ratio, 90)
    yaw_rate = pct(np.concatenate(yaw_ratio), 90)

    # Per-frame visual odometry error.
    vo_csv = []
    for r in range(2):
        used, src = cache["used"][r], cache["source"][r]
        for i in range(1, len(used)):
            rel = between(tuple(truth[r][i - 1]), tuple(truth[r][i]))
            step = used[i]
            vo_csv.append(
                (
                    r,
                    i,
                    cache["actions"][r][i] or "",
                    int(src[i]),
                    *rel,
                    *step,
                    float(math.hypot(step[0] - rel[0], step[1] - rel[1])),
                    math.degrees(wrap(step[2] - rel[2])),
                )
            )
    arr = np.array([(v[0], v[1], v[3], *v[4:]) for v in vo_csv], float)
    actions = np.array([v[2] for v in vo_csv])
    t_err, y_err = arr[:, 9], np.abs(arr[:, 10])
    true_t = np.hypot(arr[:, 3], arr[:, 4])
    rows = []
    for label, mask in (
        ("all frames", np.ones(len(arr), bool)),
        ("visual odometry used", arr[:, 2] == VO),
        ("calibrated fallback used", arr[:, 2] == FALLBACK),
    ):
        rows.append(
            [
                label,
                int(mask.sum()),
                f"{mask.mean():.1%}",
                f"{np.median(t_err[mask]):.4f} / {pct(t_err[mask], 90):.4f}",
                f"{np.median(y_err[mask]):.2f} / {pct(y_err[mask], 90):.2f}",
                f"{np.mean(arr[mask][:, 3 + 6 - 6 + 0]):.4f}",
            ]
        )
    # replace the last column: mean signed along-track error
    for row, mask in zip(
        rows,
        (np.ones(len(arr), bool), arr[:, 2] == VO, arr[:, 2] == FALLBACK),
        strict=True,
    ):
        row[-1] = f"{np.mean(arr[mask][:, 6] - arr[mask][:, 3]):+.4f}"
    lines += [
        "Per captured frame (3 ticks), the step the odometry integrated (visual or fallback) "
        "against the true relative pose of the two frames:",
        "",
        md(
            [
                "",
                "frames",
                "share",
                "translation err m (median / p90)",
                "yaw err deg (median / p90)",
                "mean signed dx err m",
            ],
            rows,
        ),
        "",
    ]
    rows = []
    for name in ("FORWARD", "BACKWARD", "LEFT", "RIGHT", ""):
        mask = actions == name
        if not mask.any():
            continue
        vo = mask & (arr[:, 2] == VO)
        rows.append(
            [
                name or "none / several",
                int(mask.sum()),
                f"{np.mean(arr[mask][:, 2] == VO):.0%}",
                f"{np.median(true_t[mask]):.3f}",
                f"{np.median(arr[vo][:, 6] - arr[vo][:, 3]) if vo.any() else math.nan:+.4f}",
                f"{np.median(arr[vo][:, 8] - arr[vo][:, 5]) if vo.any() else math.nan:+.3f}",
                f"{np.median(arr[mask][:, 6] - arr[mask][:, 3]):+.4f}",
                f"{np.median(arr[mask][:, 8] - arr[mask][:, 5]):+.3f}",
            ]
        )
    lines += [
        "By the frame's action label (the capture tick's only; a frame is 3 ticks): median true "
        "translation, and the median signed error of the integrated dx (m) and dyaw (rad) for "
        "the frames where visual odometry was used and for all:",
        "",
        md(
            [
                "action",
                "frames",
                "VO share",
                "true step m",
                "VO dx err",
                "VO dyaw err",
                "all dx err",
                "all dyaw err",
            ],
            rows,
        ),
        "",
        f"Snapped to the maze axes: {np.mean(np.concatenate(cache['snapped'])):.0%} of frames.",
        "",
    ]
    with (out / "vo_frames.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "route",
                "frame",
                "action",
                "source",
                "true_dx",
                "true_dy",
                "true_dyaw",
                "used_dx",
                "used_dy",
                "used_dyaw",
                "t_err_m",
                "yaw_err_deg",
            ]
        )
        writer.writerows(vo_csv)

    # 3. Re-solves.
    lines += ["## 3. The pose graph solved again", ""]
    drift_floor = mapping.MAX_CLOSURE_M
    node_path = [
        np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(dead[r][:, :2], axis=0).T))])[
            ::NODE_EVERY
        ]
        for r in range(2)
    ]

    def near_path(c: mapping.Closure, rule: str) -> float:
        da = node_path[c.a[0]][c.a[1] // NODE_EVERY]
        db = node_path[c.b[0]][c.b[1] // NODE_EVERY]
        return min(da, db) if rule == "near" else da + db

    def scaled(rule: str):
        def limits(c: mapping.Closure) -> tuple[float, float]:
            d = near_path(c, rule)
            return drift_floor + rate_m * d, mapping.MAX_CLOSURE_DEG + yaw_rate * d

        return limits

    true_ids = {c["closure"] for c in verified if c["true"]}
    good_ids = {
        c["closure"]
        for c in verified
        if c["true"] and c["pose_err_m"] <= 0.3 and c["pose_err_deg"] <= 20
    }
    solves: list = []
    original = mapping.least_squares

    def configured(**overrides):
        """mapping.least_squares with ``overrides``, logging each result."""

        def run(*a, **k):
            result = original(*a, **{**k, **overrides})
            solves.append(result)
            return result

        return run

    def solver_note() -> str:
        r = solves[-1]
        return f"{'converged' if r.status > 0 else 'stopped at the cap'} ({r.nfev} evals, cost {r.cost:.0f})"  # noqa: E501

    def make_variants(**overrides) -> dict[str, tuple]:
        mapping.least_squares = configured(**overrides)
        out: dict[str, tuple] = {}

        def add(name: str, fn) -> None:
            solves.clear()
            result = fn()
            out[name] = (*result, solver_note())

        add("default rule", lambda: solve_rounds(tracked, starts, closures, pregate=True))
        add("(a) pre-gate off", lambda: solve_rounds(tracked, starts, closures, pregate=False))
        add(
            "(b) pre-gate off, threshold grows with path (nearer node)",
            lambda: solve_rounds(tracked, starts, closures, pregate=False, limits=scaled("near")),
        )
        add(
            "(b') as (b), path of both nodes summed",
            lambda: solve_rounds(tracked, starts, closures, pregate=False, limits=scaled("sum")),
        )
        ids = sorted(true_ids)
        add(
            "(c) oracle: true closures only",
            lambda: (mapping.solve(tracked, starts, [closures[i] for i in ids]), ids, {}),
        )
        ids = sorted(good_ids)
        add(
            "(c') oracle: true and well-fitted (<= 0.3 m, 20 deg)",
            lambda: (mapping.solve(tracked, starts, [closures[i] for i in ids]), ids, {}),
        )
        return out

    results: dict[str, dict[str, tuple]] = {}
    results["As mapping.py solves it (Huber, max_nfev 60)"] = make_variants()
    results["Solver run longer (Huber, max_nfev 3000)"] = make_variants(max_nfev=3000)
    # Oracle with a linear loss (no outliers among true closures) and with looser odometry.
    mapping.least_squares = configured(max_nfev=1000, loss="linear")
    ids = sorted(true_ids)
    extra: dict[str, tuple] = {}
    solves.clear()
    extra["(c) oracle, linear loss, max_nfev 1000"] = (
        mapping.solve(tracked, starts, [closures[i] for i in ids]),
        ids,
        {},
        solver_note(),
    )
    saved = (mapping.ODOM_SIGMA_M, mapping.ODOM_SIGMA_DEG)
    mapping.ODOM_SIGMA_M, mapping.ODOM_SIGMA_DEG = saved[0] * 10, saved[1] * 10
    solves.clear()
    extra["(c) oracle, linear loss, max_nfev 1000, odometry sigma x10"] = (
        mapping.solve(tracked, starts, [closures[i] for i in ids]),
        ids,
        {},
        solver_note(),
    )
    mapping.ODOM_SIGMA_M, mapping.ODOM_SIGMA_DEG = saved
    mapping.least_squares = original
    results["Diagnostics: the oracle in other solver settings"] = extra
    reference = {
        "frozen map (cache file)": (frozen, None, None, ""),
        "no closures (dead reckoning)": ([d[::NODE_EVERY] for d in dead], [], {}, ""),
    }
    lines += [
        f"Rule (b): the consistency thresholds are MAX_CLOSURE_M + {rate_m:.3f} m per metre and "
        f"MAX_CLOSURE_DEG + {yaw_rate:.3f} deg per metre of the closure's path distance (the DR "
        "path from its route's start; the nearer of its two nodes), the p90 of the measured "
        "drift ratios above. The 'aligned' error is after the one rigid motion that best fits "
        "all nodes to the truth (the map's gauge removed).",
        "",
    ]
    all_truth = np.concatenate(node_truth)

    def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """``a`` moved by the rigid motion that best fits it to ``b`` (both (n, 2))."""
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
        d = np.diag([1.0, np.sign(np.linalg.det(vt.T @ u.T))])
        rot = vt.T @ d @ u.T
        return (a - ca) @ rot.T + cb

    def table(group: dict[str, tuple]) -> str:
        rows = []
        for name, (poses, kept, _fates, note) in group.items():
            allp = np.concatenate(poses)
            pos = np.hypot(*(allp[:, :2] - all_truth[:, :2]).T)
            aligned = np.hypot(*(kabsch(allp[:, :2], all_truth[:, :2]) - all_truth[:, :2]).T)
            yaw = np.degrees(np.abs(wrap(allp[:, 2] - all_truth[:, 2])))
            pairs = cKDTree(allp[:, :2]).query_pairs(FOLD_NEAR_M, output_type="ndarray")
            far = (
                np.hypot(*(all_truth[pairs[:, 0], :2] - all_truth[pairs[:, 1], :2]).T) > FOLD_FAR_M
            )
            in_fold = np.zeros(len(allp), bool)
            in_fold[pairs[far].ravel()] = True
            rows.append(
                [
                    name,
                    ""
                    if kept is None
                    else f"{len(kept)} ({sum(i in true_ids for i in kept)} true)",
                    f"{np.median(pos):.2f} / {pct(pos, 90):.2f} / {pos.max():.2f}",
                    f"{np.median(aligned):.2f} / {pct(aligned, 90):.2f} / {aligned.max():.2f}",
                    f"{np.median(yaw):.1f} / {pct(yaw, 90):.1f} / {yaw.max():.1f}",
                    f"{far.mean():.1%} ({int(far.sum())} of {len(pairs)})",
                    f"{in_fold.mean():.1%}",
                    note,
                ]
            )
        return md(
            [
                "variant",
                "closures kept",
                "node pos error m (median / p90 / max)",
                "aligned (median / p90 / max)",
                "yaw error deg (median / p90 / max)",
                f"fold rate: map pairs < {FOLD_NEAR_M} m that are > {FOLD_FAR_M} m apart",
                "nodes in a fold",
                "last solve",
            ],
            rows,
        )

    lines += ["### Reference", "", table(reference), ""]
    for title, group in results.items():
        lines += [f"### {title}", "", table(group), ""]
    for title in list(results)[:2]:
        for name in (
            "(a) pre-gate off",
            "(b) pre-gate off, threshold grows with path (nearer node)",
        ):
            fates = results[title][name][2]
            names = ["failed verification", "dropped r1", "dropped r2", "dropped r3", "kept"]
            rows = []
            for label, flag in (("true", True), ("false", False)):
                sub = [c for c in candidates if c["true"] == flag]
                count = Counter(fate_of(c, fates) for c in sub)
                rows.append([label, len(sub), *[count[f] for f in names]])
            lines += [
                f"Fates under {name} ({title}):",
                "",
                md(["", "candidates", *names], rows),
                "",
            ]
    variants = {**reference, **results["As mapping.py solves it (Huber, max_nfev 60)"]}
    variants.update(
        {
            f"{k} [longer solve]": v
            for k, v in results["Solver run longer (Huber, max_nfev 3000)"].items()
        }
    )
    variants.update(extra)

    # 4. Frames 2390 and 2395.
    lines += ["## 4. Map error at frames 2390 and 2395 of route 0", ""]
    rows = []
    for name, (poses, *_rest) in variants.items():
        row = [name]
        for f in FRAMES:
            n = f // NODE_EVERY
            row.append(
                f"{np.hypot(*(np.asarray(poses[0])[n, :2] - node_truth[0][n, :2])):.2f} "
                f"(map ({poses[0][n][0]:.2f}, {poses[0][n][1]:.2f}))"
            )
        rows.append(row)
    lines += [
        "True positions in the map frame: "
        + "; ".join(
            f"frame {f} ({truth_map[0][f][0]:.2f}, {truth_map[0][f][1]:.2f}), world "
            f"({truth[0][f][0]:.2f}, {truth[0][f][1]:.2f}), true path length "
            f"{path_len[0][f]:.1f} m"
            for f in FRAMES
        )
        + ".",
        "",
        md(["variant", *[f"frame {f}: error m (map pose)" for f in FRAMES]], rows),
        "",
    ]
    text = "\n".join(lines) + "\n"
    (out / "map.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
