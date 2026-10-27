"""
A discrete Bayes filter over the graph's nodes: the belief moves along the graph with each
action taken and is reweighted by how much the camera view looks like each node.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp
from vlad import MOVEMENT, REVERSE, Frame


class NodeFilter:
    """``predict(action)`` spreads the belief to nodes within ``hops`` hops, ``action_boost``
    times more towards those whose shortest path starts with a temporal edge recorded with
    that action; ``update(similarity)`` multiplies it by ``exp(similarity / tau)``."""

    def __init__(
        self,
        graph: nx.Graph,
        frames: list[Frame],
        *,
        hops: int = 2,
        p_stay: float = 0.5,
        action_boost: float = 3.0,
        eps: float = 1e-3,
        tau: float = 0.02,
    ) -> None:
        self.graph = graph
        self.n = graph.number_of_nodes()
        self.eps = eps
        self.tau = tau
        self.transitions = _transitions(graph, frames, hops, p_stay, action_boost)
        self.belief = np.full(self.n, 1 / self.n)

    def reset(self) -> None:
        self.belief = np.full(self.n, 1 / self.n)

    def predict(self, action: str) -> None:
        belief = self.transitions[action].T @ self.belief
        self.belief = (1 - self.eps) * belief + self.eps / self.n

    def update(self, similarity: np.ndarray) -> None:
        belief = self.belief * np.exp((similarity - similarity.max()) / self.tau)
        self.belief = belief / belief.sum()

    def estimate(self) -> tuple[int, float]:
        """The most likely node and the belief mass on it and its neighbours."""
        node = int(np.argmax(self.belief))
        mass = self.belief[node] + sum(self.belief[j] for j in self.graph[node])
        return node, float(mass)


def _move(graph: nx.Graph, frames: list[Frame], i: int, j: int) -> str | None:
    """The recorded move along temporal edge ``i -> j``; None for a visual edge."""
    if graph[i][j].get("visual", False):
        return None
    if j == i + 1:
        return frames[i].action
    if j == i - 1:
        return REVERSE[frames[j].action]
    return None


def _transitions(
    graph: nx.Graph, frames: list[Frame], hops: int, p_stay: float, action_boost: float
) -> dict[str, sp.csr_matrix]:
    """Row-stochastic ``T_a`` per move ``a``: ``T_a[i, j]`` is P(at j after a | at i)."""
    rows, cols, first_moves = [], [], []
    for i in graph.nodes:
        for j, path in nx.single_source_shortest_path(graph, i, cutoff=hops).items():
            rows.append(i)
            cols.append(j)
            first_moves.append(None if j == i else _move(graph, frames, i, path[1]))
    rows, cols = np.array(rows), np.array(cols)
    n = graph.number_of_nodes()
    transitions = {}
    for action in sorted(MOVEMENT):
        weights = np.array(
            [
                p_stay if i == j else action_boost if move == action else 1.0
                for i, j, move in zip(rows, cols, first_moves, strict=True)
            ]
        )
        matrix = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
        transitions[action] = sp.diags(1 / np.asarray(matrix.sum(axis=1)).ravel()) @ matrix
    return transitions
