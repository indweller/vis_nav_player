"""The graph MyAgent builds on the seed-42 local exploration data, in a spring layout.

    uv run source/eval/plot_my_agent_graph.py
    uv run source/eval/plot_my_agent_graph.py --data data/gt/42-main

Writes results/my_agent/seed42/graph/graph.png: node positions are networkx.spring_layout(seed=0),
nodes coloured by trajectory, trajectory edges blue, visual edges orange, the goal green and the
node localised at the start black.

With --data the graph is drawn over the true maze instead, to seed42/<data directory name>/
graph_overlay.png: each node at the ground-truth pose record_gt.py wrote for its record
(frames[n].trajectory, raw_index[n]), over the walls eval_closed_loop draws, in the world frame.

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import vis_nav_sim as sim
from common import textures
from eval_cues import Walls
from eval_map import MAP_COLOUR, Canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cli  # noqa: E402
from my_agent import MyAgent  # noqa: E402
from vis_nav_sdk.local import local_session  # noqa: E402

SEED = 42
OUT = Path("results") / "my_agent" / f"seed{SEED}"
SIZE = 1600
MARGIN = 40
TOP = 50
BLUE = (200, 120, 0)  # BGR
ORANGE = (0, 140, 255)
GREEN = (0, 170, 0)
BLACK = (0, 0, 0)
DRIVES = [(0, 0, 220), (200, 0, 200), (0, 160, 160), (120, 120, 120)]
"""Node colour per trajectory id (BGR)."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data", default=None, help="exploration data directory (default: the local recording)"
    )
    args = parser.parse_args()
    data = Path(args.data) if args.data else None
    agent = MyAgent(
        data or cli.record_exploration_data(SEED),
        cache_dir=Path("cache") / (data.name if data else f"local-{SEED}"),
    )
    out = OUT / data.name / "graph_overlay.png" if data else OUT / "graph" / "graph.png"
    session = local_session(f"local:{SEED}", textures=textures())
    agent.setup(session.info)  # picks the goal node
    graph = agent.graph.copy()
    agent.act(session.initial_observation)  # localises the start node
    start, goal = agent.current, agent.goal
    assert graph.number_of_nodes() == len(agent.database)

    visual = [(a, b) for a, b, d in graph.edges(data=True) if d.get("visual")]
    trajectory = [(a, b) for a, b, d in graph.edges(data=True) if not d.get("visual")]
    components = nx.number_connected_components(graph)
    summary = (
        f"{graph.number_of_nodes()} nodes, {len(trajectory)} trajectory edges, "
        f"{len(visual)} visual edges, {components} connected components"
    )

    if data:
        world = sim.Simulator(textures(), SEED)
        poses = [
            json.loads((data / f"traj_{t}" / "data_info.json").read_text())
            for t in range(len(set(f.trajectory for f in agent.frames)))
        ]
        xy = np.array(
            [
                [r["x"], r["y"]]
                for r in (
                    poses[agent.frames[n].trajectory][agent.raw_index[n]]
                    for n in range(graph.number_of_nodes())
                )
            ]
        )
        canvas = Canvas(world)
        canvas.walls(Walls(world), MAP_COLOUR)
        px = np.round(canvas.pixels(xy)).astype(int)
        px[:, 1] += TOP
        image = np.vstack([np.full((TOP, canvas.image.shape[1], 3), 255, np.uint8), canvas.image])
    else:
        layout = nx.spring_layout(graph, seed=0)
        xy = np.array([layout[n] for n in range(graph.number_of_nodes())])
        xy = (xy - xy.min(0)) / np.ptp(xy, axis=0)
        px = np.round(xy * (SIZE - 2 * MARGIN) + MARGIN).astype(int)
        px[:, 1] += TOP
        image = np.full((SIZE + TOP, SIZE, 3), 255, np.uint8)

    def point(n: int) -> tuple[int, int]:
        return int(px[n, 0]), int(px[n, 1])

    for edges, colour, width in ((trajectory, BLUE, 1), (visual, ORANGE, 2)):
        for a, b in edges:
            cv2.line(image, point(a), point(b), colour, width, cv2.LINE_AA)
    for n in graph.nodes:
        cv2.circle(image, point(n), 2, DRIVES[agent.frames[n].trajectory % len(DRIVES)], -1)
    for node, colour in ((goal, GREEN), (start, BLACK)):
        if node is not None:
            cv2.circle(image, point(node), 9, colour, -1, cv2.LINE_AA)
    title = f"maze {SEED}: {summary}"
    scale = min(
        0.8,
        0.9
        * (image.shape[1] - 2 * MARGIN / 2)
        / cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0],
    )
    cv2.putText(image, title, (MARGIN // 2, 32), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image)
    print(summary)
    print(f"goal node {goal}, start node {start}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
