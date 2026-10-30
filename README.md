# Visual Navigation Challenge — vis_nav_player

Fork of the [ai4ce starter kit](https://github.com/ai4ce/vis_nav_player) for NYU ROB-GY 6203 Robot
Perception. A robot is placed in a maze it has explored before. The agent gets its camera frames, four
photos of the goal and an exploration dataset. It picks each move and checks in when it thinks it has
arrived. Runs are scored on distance to the goal (≤ 0.2 m), heading (≤ 30°) and path length (SPL).

Course tutorial: <https://visual-navigation-challenge.ai4ce.dev/>

## Status

| Part | State |
|---|---|
| Pose-free metric map (`mapping.py`) | **Working, frozen.** Trajectories are within about 0.05–0.2 m of truth. The occupancy grid is noisy. |
| Goal identification (`goal.py`) | **Working.** The right exploration frame is found, with an accurate relative pose. Map-frame error is dominated by mapping error. |
| Planning and localisation (`planner.py`, `localiser.py`, `navigator.py`) | **Built, evaluated with an oracle driver.** Online localisation on the frozen map and Dijkstra over a place graph. See `eval_planning.py`. |
| Map-based agent (`map_agent.py`) | **The agent.** Path follower, wall guard and visual homing on the navigator. On the midterm-like maze (`eval_dev_like.py --seed 10`) it reaches all 10 goals. |

The approach: a metric map built from exploration data only (no poses), goal identification in that
map, then a controller that tracks the robot from the known spawn pose and homes in on the goal.
The earlier place-recognition agent (`my_agent.py`) and its evaluations are archived in
`source/legacy/`.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                     # core + dev + local simulator (vis-nav-sim)
```

`uv sync` installs the project in editable mode with `source/` and `source/eval/` on the path, so
their modules import each other by name (`import geometry`, `from common import ...`) from any script.

Copy `.env.example` to `.env` and set `VIS_NAV_API_KEY` and `VIS_NAV_CHALLENGE`. Every agent also takes
`--challenge`, `--api-key`, or `--local <seed>` to run on the local simulator (no attempt spent).

```bash
uv run source/map_agent.py --local 7                          # the agent
uv run source/map_agent.py --test-local --data data/<id>      # build or load the caches, find the goal; no session
uv run source/keyboard_agent.py --local 7                     # drive by hand
uv run pytest -q && uv run ruff check . && uv run ruff format --check .
```

`map_agent.py`'s tuning parameters are its `Config` (`--lookahead-m`, `--turn-tol`, `--max-turn-ticks`,
`--forward-ticks`, `--link-m`, `--guard-m`, `--near-m`, `--align-deg`, `--offset-m`, `--homing-max`).
It caches the frozen map (`map_graph.npz`), the motion calibration (`motion.json`) and the planner's
links (`links.npz`) in the data directory. Each cache keeps a fingerprint of the settings it was made
with and warns when it is stale; `--rebuild` makes all three again.

## Layout

| Path | What it does |
|---|---|
| `source/map_agent.py` | `MapAgent`: frozen map + `PlaceGraph` + localiser + navigator; drives toward a waypoint 0.15 m along the path (FORWARD under 12° off, else turns), backs off a wall, homes in near the goal (turning by the estimate; a goal-view match widens the tolerance from 5° to 8°), and checks in. |
| `source/geometry.py` | Single-image geometry and poses. A floor mask gives per-column wall ranges and wall points; Manhattan yaw (heading mod 90°); visual odometry (SIFT wall points + rigid RANSAC); VLAD vectors; occupancy map; scan matching; motion calibration; `Pose`, `wrap`, `compose`, `between`. |
| `source/mapping.py` | Pose-free map from exploration data: visual odometry + Manhattan snap → VLAD/SIFT loop closures → pose graph (robust least squares) → occupancy grid. No poses or simulator used. |
| `source/frozen_map.py` | Caches a `mapping.build` result in `map_graph.npz` (every frame's pose, grid, VLAD) plus lazily filled SIFT sidecars. |
| `source/goal.py` | `find_goal`: for each of the 4 views, VLAD top-20 → rerank by SIFT inliers → relative pose → goal pose and heading (−90°·k for view k). |
| `source/planner.py` | `PlaceGraph` over every 5th mapped frame: route edges plus links within `link_m` (0.25 m) kept only if they cross no occupied grid cell and share ≥ 15 SIFT inliers (cached in `links.npz`); metric + turn cost; Dijkstra `plan`; `Path.cross_track`. |
| `source/localiser.py` | Online pose on the frozen map: visual odometry + Manhattan snap (the run's own median motion, else the calibration, where VO fails), VLAD/SIFT fixes blended by sigma. A fix beyond max(0.3 m, 3 sigma) is held until the next fix agrees (0.1 m / 5°); global search when lost uses the same confirmation. |
| `source/navigator.py` | Localise, replan (no path, cross-track, fix jump, periodic), next waypoint. |
| `source/caches.py` | The settings fingerprints of the three caches. |
| `source/servo.py` | SIFT yaw offset between two views (bearing-based). |
| `source/view_map.py` | Plot a frozen map, the goal and the planned path. |
| `source/cli.py`, `source/paths.py` | Shared CLI and `.env` loading; the repository root. |
| `source/keyboard_agent.py` | Drive by hand. |
| `source/eval/` | Evaluations on the local simulator (below). |
| `source/legacy/` | Archived: the place-recognition agent, the upstream baseline and RL demo, and their evaluations. See its README. |
| `scripts/` | `baseline.py` records the regression baseline (find_goal poses, map hashes, plan lengths, the seed-10 midterm-like eval) to `results/baseline/baseline.json`; `check_baseline.py` compares the current code with it. |
| `data/floor_mask.npz` | Floor colour lookup table, fitted on mazes 1–6. |

## Evaluations (`source/eval/`)

All of these run on the local simulator and use ground truth only for scoring.

| Script | Measures |
|---|---|
| `eval_dev_like.py` | **The midterm-like evaluation:** MapAgent through the SDK's `run()` on a dev-challenge-like maze (seed 10 is the regression maze), per goal and tick against the truth. Writes `runs.csv`, `ticks.csv` and plots to `results/eval/dev_like/<seed>/`. |
| `eval_closed_loop.py` | **MapAgent closed loop** on mazes 7–12, goal found / true / map-true: per-tick log, localisation error, wall guards, stalls, mode fractions, the simulator's check-in and the SDK's local score. Writes plots, `runs.csv` and `ticks.csv` to `results/eval/closed_loop/`. |
| `eval_map.py` | **Pose-free map vs truth:** trajectory error, wall precision/recall, closures. Writes plots to `results/eval/map/`. |
| `eval_goal_map.py` | **Goal identification on the frozen map:** the real target + 40 random goals per maze; mazes 7–9 for development, 10–12 held out. Writes plots to `results/eval/goal/`. |
| `eval_planning.py` | **Planning and localisation with an oracle driver** (direct and detour) on mazes 7–12: localisation error, fixes, lost events, replans, plan quality. Writes plots and `planning.csv` to `results/eval/planning/`. |
| `eval_goal_photos.py` | **Goal photos vs renders** at `world.target` (msaa 1 and 4, raw and JPEG): pixel diff, `geometry.ranges` diff and floor-mask precision / recall on `world.targets()`. Writes `results/eval/goal_photos.csv`. |
| `eval_cues.py` | Steering cues (whole-image yaw, free space, SIFT) under position offset; also the true walls the other scripts score against. |
| `record_gt.py` | Our own exploration recorder with poses (BFS coverage). Its routes don't start at the spawn. |
| `replay_log.py` | Offline replay of a hosted run log against its frozen map: check-in quality, the localiser, contacts, the path. |

```bash
uv run source/eval/eval_dev_like.py --seed 10 --maze-size 31 --routes 3 --capture-every 1 --agent-args "--link-m 0.25"
uv run source/eval/eval_map.py --mazes 7,8,9
uv run source/eval/eval_goal_map.py --mazes 7,8,9,10,11,12   # --rebuild to rebuild frozen maps
uv run source/eval/eval_planning.py                          # mazes 7-12, direct and detour
uv run scripts/check_baseline.py                             # the current code against the baseline
```

Evaluation data goes to `data/eval/<seed>/`: `record()` output with the challenge's settings, 3 routes,
a frame every tick and `student_files=False`, so `poses.json` gives ground truth.

## Results so far

- **Map agent, midterm-like maze (seed 10, 31×31, 3 routes):** 10 of 10 goals, 90 points, 5,993
  steps (the regression baseline).
- **VLAD agent (archived):** 0 successes. Descriptors are fine (AUC 0.85–0.89). Single-frame
  localisation jumps around, and open-loop replay of recorded actions drifts.
- **Geometry:** Manhattan yaw is accurate to about 1°, even at 0.4 m offset. Free-space ranges are
  within about 1% median.
- **Mapping (graph variant):** trajectory median error 0.05–0.22 m over 35–65 m routes. Loop closures
  add little over odometry. Grid wall precision at 0.05 m is 0.25–0.43: walls are thick, with blobs,
  because the grid records hits only.
- **Goal identification:** 34% of goals within 0.2 m / 30°, 68% within 0.4 m; held-out mazes only
  slightly worse. Retrieval error is about 0.08 m and relative-pose error about 0; the miss is the
  mapped pose of the matched frame (about 0.25 m).

## Challenge facts used

- Exploration data has a frame every tick, covers every cell, and contains the goal.
- The spawn is a fixed cell and heading. Every route starts there; the map origin is (0, 0, 0).
- The server's motion model isn't published, so motion comes from the data (visual odometry), not
  simulator constants. Wall contact distorts commanded motion.
- Goal view k is the goal seen at target_yaw + 90°·k; view 0 is the arrival heading.

## Known issues

- Map builds are not bit-repeatable (multithreaded k-means, and OpenCV's fundamental-matrix mask on
  degenerate matches), so a rebuilt cache differs from a saved one by millimetres.
- `render()` doesn't reproduce `world.targets()` exactly (pixel diff 23–48); possibly anti-aliasing
  (`msaa=4` on the server). Random-goal views may not match real goal photos.
- `find_goal`'s confidence doesn't predict success.
- Some mapping constants (`HORIZON_SHIFT_PX`, pose-graph sigmas, closure gates) were tuned on mazes 7–9.
- The occupancy grid never clears free space.
- `data/local-7` and `data/scale/*` are stale recordings from older settings and SDK versions.

## Next steps

1. Check the render mismatch (`msaa=4`) and add view-agreement as a goal confidence.
2. Optionally refine mapping: a lattice map (walls on the 0.4 m grid) or free-space clearing.
