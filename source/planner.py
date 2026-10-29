"""
Path planning on the frozen map, without ground truth: a place graph over the mapped poses and
Dijkstra over it.

The nodes are every NODE_EVERY-th mapped frame (pose x, y, yaw), per route. Two nodes are linked
if they are consecutive on a route (the robot drove between them) or are a *link*: within
``link_m`` of each other, whatever their headings (the robot can turn in place, and another
route's pass through the same place is a shortcut), and surviving two filters, since nearness in
the map is not nearness in the maze:
  - the segment between their mapped poses crosses no occupied cell of the frozen grid;
  - their frames share at least ``min_inliers`` SIFT epipolar inliers (cached).
An edge weighs its metric length plus ``turn_cost`` metres per radian of heading change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from multiprocessing.pool import ThreadPool
from pathlib import Path as FilePath

import caches
import numpy as np
from geometry import epipolar_inliers, match_features
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

NODE_EVERY = 5
LINK_M = 0.12
TURN_COST = 0.05
"""Metres per radian of heading change."""
LINK_INLIERS = 15
"""SIFT epipolar inliers two frames need to share for a link."""
CONNECT_M = 0.15
"""Start and goal poses join every node this near (the nearest one if none is)."""
CORRIDOR_M = 0.08
"""A shortcut stays this near some mapped frame's position."""
CORRIDOR_STEP_M = 0.02
"""Spacing of the points a shortcut segment is checked at."""
SHORTCUT_LOG_ODDS = 4.0
"""A shortcut crosses no cell of the grid with at least this log-odds."""

Pose = tuple[float, float, float]


@dataclass
class Path:
    waypoints: np.ndarray
    """``(n, 3)`` ``x, y, yaw``: the start, the nodes, the goal."""
    nodes: list[int]
    """The PlaceGraph node ids passed through, in order."""
    length: float = field(init=False)
    """Metres along the waypoints."""
    arc: np.ndarray = field(init=False, repr=False)
    """Metres from the start to each waypoint."""
    length_before: float = field(default=math.nan, repr=False)
    """Metres of the path before ``shortcut`` (NaN if it was not shortcut)."""
    pulled: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list, repr=False)
    """The segments (start, end points) ``shortcut`` made from more than one hop."""

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, float).reshape(-1, 3)
        steps = np.hypot(*np.diff(self.waypoints[:, :2], axis=0).T)
        self.arc = np.concatenate([[0.0], np.cumsum(steps)])
        self.length = float(self.arc[-1])

    def cross_track(self, pose: Pose | np.ndarray) -> tuple[float, float]:
        """``(distance, progress)``: the distance from ``pose``'s position to the nearest point
        of the path, and the metres along the path to that point."""
        p = np.asarray(pose[:2], float)
        a, b = self.waypoints[:-1, :2], self.waypoints[1:, :2]
        if not len(a):
            return float(np.hypot(*(p - self.waypoints[0, :2]))), 0.0
        d = b - a
        length2 = (d**2).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(length2 > 0, ((p - a) * d).sum(axis=1) / length2, 0.0)
        t = np.clip(t, 0.0, 1.0)
        distance = np.hypot(*(a + t[:, None] * d - p).T)
        i = int(np.argmin(distance))
        return float(distance[i]), float(self.arc[i] + t[i] * math.sqrt(length2[i]))

    def shortcut(
        self,
        store,
        corridor_m: float = CORRIDOR_M,
        tree: cKDTree | None = None,
        log_odds: float = SHORTCUT_LOG_ODDS,
    ) -> Path:
        """Greedy string-pulling: from each kept waypoint, jump to the farthest later one whose
        straight segment is safe, and drop those between. A segment is safe if every point on it
        (CORRIDOR_STEP_M apart) is within ``corridor_m`` of some mapped frame position of
        ``store`` (``tree`` is a cKDTree of ``store.xy``) and it crosses no cell of ``store.grid``
        with log-odds of at least ``log_odds``. The start and the goal stay."""
        tree = cKDTree(store.xy) if tree is None else tree
        blocked = store.grid.log_odds >= log_odds
        xy = self.waypoints[:, :2]
        keep, pulled = [0], []
        i = 0
        while i < len(xy) - 1:
            j = next(
                (
                    j
                    for j in range(len(xy) - 1, i + 1, -1)
                    if safe(store, tree, blocked, xy[i], xy[j], corridor_m)
                ),
                i + 1,
            )
            if j > i + 1:
                pulled.append((xy[i].copy(), xy[j].copy()))
            keep.append(j)
            i = j
        nodes = [self.nodes[k - 1] for k in keep[1:-1]]
        path = Path(self.waypoints[keep], nodes)
        path.length_before, path.pulled = self.length, pulled
        return path

    def waypoint_after(self, progress: float, ahead: float) -> Pose:
        """The first waypoint at least ``ahead`` metres beyond ``progress`` (the last if none)."""
        i = int(np.searchsorted(self.arc, progress + ahead - 1e-9))
        x, y, yaw = self.waypoints[min(i, len(self.waypoints) - 1)]
        return float(x), float(y), float(yaw)


def bresenham(r0: int, c0: int, r1: int, c1: int):
    """The grid cells ``(row, col)`` on the line ``(r0, c0)`` to ``(r1, c1)``, both included."""
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    error = dc - dr
    r, c = r0, c0
    while True:
        yield r, c
        if r == r1 and c == c1:
            return
        step = 2 * error
        if step > -dr:
            error -= dr
            c += sc
        if step < dc:
            error += dc
            r += sr


def crosses_occupied(grid, a: np.ndarray, b: np.ndarray) -> bool:
    """Whether the segment ``a`` to ``b`` (map points) passes an occupied cell of ``grid``
    (an OccupancyMap; cells off the grid count as free)."""
    occupied = grid.occupied()
    (r0, r1), (c0, c1) = (
        (int(v) for v in np.floor((np.array([a, b])[:, i] - grid.origin[i]) / grid.res))
        for i in (1, 0)
    )
    size = occupied.shape[0]
    return any(
        0 <= r < size and 0 <= c < size and occupied[r, c] for r, c in bresenham(r0, c0, r1, c1)
    )


def safe(
    store, tree: cKDTree, blocked: np.ndarray, a: np.ndarray, b: np.ndarray, corridor_m: float
) -> bool:
    """Whether the segment ``a`` to ``b`` stays within ``corridor_m`` of the mapped frame
    positions (KD-tree ``tree``) and off the ``blocked`` cells of ``store.grid``."""
    n = max(2, int(math.ceil(math.dist(a, b) / CORRIDOR_STEP_M)) + 1)
    points = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
    if (tree.query(points)[0] > corridor_m).any():
        return False
    row, col, inside = store.grid.cells(points)
    return not blocked[row[inside], col[inside]].any()


class PlaceGraph:
    """The graph over ``frozen_map.poses`` (per route ``(frames, 3)``); with the filters on,
    also its ``grid`` and ``view(row)`` (a FrozenMap).

    ``link_counts`` is how many links there were before and after each filter; ``candidates``,
    ``after_walls`` and ``links`` are those link sets as ``(m, 2)`` node ids, ``consecutive``
    the route edges. ``cache_path`` keeps the inlier counts between runs."""

    def __init__(
        self,
        frozen_map,
        *,
        node_every: int = NODE_EVERY,
        link_m: float = LINK_M,
        turn_cost: float = TURN_COST,
        connect_m: float = CONNECT_M,
        filter_walls: bool = True,
        min_inliers: int = LINK_INLIERS,
        cache_path: FilePath | None = None,
    ) -> None:
        self.turn_cost, self.connect_m = turn_cost, connect_m
        poses, route, frame, row = [], [], [], []
        offset = 0
        for r, p in enumerate(frozen_map.poses):
            keep = np.arange(0, len(p), node_every)
            poses.append(np.asarray(p)[keep])
            route.append(np.full(len(keep), r))
            frame.append(keep)
            row.append(offset + keep)
            offset += len(p)
        self.poses = np.concatenate(poses)
        self.route, self.frame, self.row = (np.concatenate(v) for v in (route, frame, row))
        self.tree = cKDTree(self.poses[:, :2])
        self.n = len(self.poses)

        near = np.sort(self.tree.query_pairs(link_m, output_type="ndarray").reshape(-1, 2), axis=1)
        step = np.flatnonzero(self.route[1:] == self.route[:-1])
        self.consecutive = np.stack([step, step + 1], axis=1)
        consecutive = {(int(a), int(b)) for a, b in self.consecutive}
        self.route_edges = consecutive
        candidates = np.array(
            [p for p in near.tolist() if tuple(p) not in consecutive], int
        ).reshape(-1, 2)
        self.candidates = candidates
        self.link_counts = {"candidates": len(candidates)}
        links = candidates
        if filter_walls:
            keep = [
                not crosses_occupied(frozen_map.grid, self.poses[a, :2], self.poses[b, :2])
                for a, b in links
            ]
            links = links[np.array(keep, bool)]
            self.link_counts["after_walls"] = len(links)
        self.after_walls = links
        if min_inliers > 0:
            inliers = self._inliers(frozen_map, links, cache_path)
            links = links[inliers >= min_inliers]
            self.link_counts["after_inliers"] = len(links)
        self.links = links

        pairs = np.unique(np.concatenate([self.consecutive, links]), axis=0)
        self.pairs = pairs
        self.weights = self._weight(self.poses[pairs[:, 0]], self.poses[pairs[:, 1]])

    def _inliers(self, frozen_map, links: np.ndarray, cache_path: FilePath | None) -> np.ndarray:
        """SIFT epipolar inliers between the two frames of each link, from the cache or
        computed (and cached). The cache keeps the fingerprint (caches.LINKS) it was made with: a
        stale cache is used, warned about, and keeps its fingerprint when entries are added."""
        cache: dict[tuple[int, int], int] = {}
        expected = caches.fingerprint(caches.LINKS)
        stamp: str | None = expected
        if cache_path is not None and FilePath(cache_path).exists():
            with np.load(cache_path) as data:
                cache = {
                    (int(a), int(b)): int(n)
                    for (a, b), n in zip(data["pairs"], data["inliers"], strict=True)
                }
                stamp = str(data["fingerprint"]) if "fingerprint" in data.files else None
            caches.check(cache_path, stamp, expected)
        rows = self.row
        keys = [(int(rows[a]), int(rows[b])) for a, b in links]
        todo = [k for k in keys if k not in cache]
        if todo:
            # SIFT of every frame needed, in threads (cv2 releases the GIL); then match.
            needed = sorted({r for k in todo for r in k})
            with ThreadPool() as pool:
                pool.map(frozen_map.view, needed, chunksize=16)
            for a, b in todo:
                va, vb = frozen_map.view(a), frozen_map.view(b)
                cache[(a, b)] = int(epipolar_inliers(va, vb, match_features(va, vb)).sum())
            if cache_path is not None:
                FilePath(cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    cache_path,
                    pairs=np.array(list(cache), int).reshape(-1, 2),
                    inliers=np.array(list(cache.values()), int),
                    **({} if stamp is None else {"fingerprint": stamp}),
                )
        return np.array([cache[k] for k in keys], int)

    def _weight(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Metric distance plus the turn cost, between rows of poses ``a`` and ``b``."""
        yaw = np.abs((a[:, 2] - b[:, 2] + np.pi) % (2 * np.pi) - np.pi)
        return np.hypot(*(a[:, :2] - b[:, :2]).T) + self.turn_cost * yaw

    def _attach(self, pose: Pose) -> np.ndarray:
        """Nodes within ``connect_m`` of ``pose`` (the nearest if none)."""
        near = self.tree.query_ball_point(pose[:2], self.connect_m)
        return np.array(near if near else [self.tree.query(pose[:2])[1]], int)

    def plan(self, start: Pose, goal: Pose) -> Path | None:
        """The cheapest path from the ``start`` pose to the ``goal`` pose, joined to the graph
        as above; None if they are not connected."""
        s, g = self.n, self.n + 1
        ends = []
        for index, pose in ((s, start), (g, goal)):
            nodes = self._attach(pose)
            cost = self._weight(np.tile(np.array(pose, float), (len(nodes), 1)), self.poses[nodes])
            ends.append((np.full(len(nodes), index), nodes, cost))
        rows = np.concatenate([self.pairs[:, 0], *(e[0] for e in ends)])
        cols = np.concatenate([self.pairs[:, 1], *(e[1] for e in ends)])
        data = np.concatenate([self.weights, *(e[2] for e in ends)])
        graph = csr_matrix((data, (rows, cols)), shape=(self.n + 2,) * 2)
        distance, before = dijkstra(graph, directed=False, indices=s, return_predecessors=True)
        if not math.isfinite(distance[g]):
            return None
        chain = [g]
        while chain[-1] != s:
            chain.append(int(before[chain[-1]]))
        nodes = chain[-2:0:-1]
        waypoints = np.array([start, *self.poses[nodes], goal], float)
        return Path(waypoints, nodes)
