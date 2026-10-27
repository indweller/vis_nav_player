"""
A mapping.build result frozen to disk, so goal identification can be evaluated (and later used)
without rebuilding the map: ``data/eval/<seed>/map_graph.npz`` holds

- the optimised pose of every frame, per route. A node frame (every NODE_EVERY-th) has the
  graph's pose; any other frame composes its dead-reckoned motion since the nearest earlier node
  onto that node's pose;
- the occupancy grid;
- the VLAD codebook and the VLAD vector of every frame (float16);

and ``sift_r<route>.npz`` beside it, a lazily filled sidecar store of per-frame SIFT: RootSIFT
descriptors (float16), keypoints and wall points. mapping.py's logic is untouched.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import mapping
import numpy as np
from geometry import Camera, FloorMask, OccupancyMap, WallView, wall_view
from tqdm import tqdm

MAP_NAME = "map_graph.npz"


def vlad(des: np.ndarray | None, centers: np.ndarray) -> np.ndarray:
    """The VLADExtractor's vector (intra- and power-normalised) of RootSIFT ``des`` against the
    codebook ``centers``: float32, zeros without descriptors."""
    k, d = centers.shape
    if des is None or not len(des):
        return np.zeros(k * d, np.float32)
    des = des.astype(np.float32)
    distance = (centers**2).sum(axis=1)[None, :] - 2.0 * des @ centers.T
    labels = distance.argmin(axis=1)
    sums = np.zeros((k, d), np.float32)
    np.add.at(sums, labels, des - centers[labels])
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    sums = np.divide(sums, norms, out=sums, where=norms > 0)
    flat = sums.ravel()
    flat = np.sign(flat) * np.sqrt(np.abs(flat))
    norm = np.linalg.norm(flat)
    return (flat / norm if norm > 0 else flat).astype(np.float32)


class SiftStore:
    """Per-frame ``WallView``s (without the column points), computed from the frames on first
    use and kept in ``<directory>/sift_r<route>.npz`` by :meth:`flush`."""

    def __init__(
        self,
        routes: Sequence[mapping.Route],
        camera: Camera,
        floor: FloorMask,
        directory: Path,
    ) -> None:
        self.routes, self.camera, self.floor = routes, camera, floor
        self.directory = Path(directory)
        self.cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray | None, np.ndarray]] = {}
        self.dirty = False
        self._local = threading.local()
        for r in range(len(routes)):
            path = self._path(r)
            if not path.exists():
                continue
            with np.load(path) as data:
                edges = np.concatenate([[0], np.cumsum(data["counts"])])
                for i, frame in enumerate(data["frames"]):
                    a, b = edges[i], edges[i + 1]
                    des = data["des"][a:b] if b > a else None
                    self.cache[(r, int(frame))] = (data["pixels"][a:b], des, data["wall"][a:b])

    def _path(self, route: int) -> Path:
        return self.directory / f"sift_r{route}.npz"

    def raw(self, route: int, frame: int) -> WallView:
        """The frame's view, computed (thread-safe), not stored."""
        sift = getattr(self._local, "sift", None)
        if sift is None:
            sift = self._local.sift = cv2.SIFT_create()
        return wall_view(self.routes[route].frames[frame], self.camera, self.floor, sift)

    def compute(self, route: int, frame: int) -> tuple:
        view = self.raw(route, frame)
        des = None if view.des is None else view.des.astype(np.float16)
        return view.pixels, des, view.wall.astype(np.float32)

    def view(self, route: int, frame: int) -> WallView:
        entry = self.cache.get((route, frame))
        if entry is None:
            entry = self.cache[(route, frame)] = self.compute(route, frame)
            self.dirty = True
        pixels, des, wall = entry
        return WallView(
            pixels,
            None if des is None else des.astype(np.float32),
            wall.astype(np.float64),
            np.empty((0, 2)),
        )

    def flush(self) -> None:
        if not self.dirty:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        for r in range(len(self.routes)):
            keys = sorted(k for k in self.cache if k[0] == r)
            if not keys:
                continue
            entries = [self.cache[k] for k in keys]
            des = [e[1] for e in entries if e[1] is not None]
            np.savez(
                self._path(r),
                frames=np.array([k[1] for k in keys]),
                counts=np.array([len(e[0]) for e in entries]),
                pixels=np.concatenate([e[0] for e in entries]),
                des=np.concatenate(des) if des else np.empty((0, 128), np.float16),
                wall=np.concatenate([e[2] for e in entries]),
            )
        self.dirty = False


class FrozenMap:
    """The loaded map: ``poses[r]`` ``(frames, 3)``, ``grid``, ``centers`` ``(K, 128)``, and
    ``vlad`` ``(N, K*128)`` float32 over every frame, ``route[n]`` / ``frame[n]`` naming row n.
    ``sift`` is the lazily filled :class:`SiftStore`."""

    def __init__(
        self,
        poses: list[np.ndarray],
        grid: OccupancyMap,
        centers: np.ndarray,
        vlads: np.ndarray,
        sift: SiftStore,
    ) -> None:
        self.poses = poses
        self.grid = grid
        self.centers = centers
        self.vlad = vlads
        self.sift = sift
        self.route = np.concatenate([np.full(len(p), r) for r, p in enumerate(poses)])
        self.frame = np.concatenate([np.arange(len(p)) for p in poses])
        self.xy = np.concatenate([p[:, :2] for p in poses])

    def pose(self, n: int) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.poses[self.route[n]][self.frame[n]])

    def view(self, n: int) -> WallView:
        return self.sift.view(int(self.route[n]), int(self.frame[n]))

    def image(self, n: int) -> np.ndarray:
        return self.sift.routes[self.route[n]].frames[int(self.frame[n])]


def every_frame_poses(result: mapping.MapResult) -> list[np.ndarray]:
    """Per route, the graph pose of every frame (see the module doc)."""
    out = []
    for nodes, dead in zip(result.variants["graph"].poses, result.dead_reckoned, strict=True):
        poses = np.empty_like(dead)
        for f in range(len(dead)):
            n = f // mapping.NODE_EVERY
            base = n * mapping.NODE_EVERY
            step = mapping.between(tuple(dead[base]), tuple(dead[f]))
            poses[f] = mapping.compose(tuple(nodes[n]), step)
        out.append(poses)
    return out


def freeze(
    root: Path, routes: list[mapping.Route], camera_dict: dict, floor: FloorMask, path: Path
) -> None:
    """Build the graph variant over ``routes`` and write ``path`` and the SIFT sidecar."""
    result = mapping.build(routes, camera_dict, floor, variants=("graph",))
    poses = every_frame_poses(result)
    camera = Camera({**camera_dict, "cy": camera_dict["cy"] + mapping.HORIZON_SHIFT_PX})
    store = SiftStore(routes, camera, floor, path.parent)
    centers = result.codebook
    assert centers is not None

    def describe(job: tuple[int, int]) -> np.ndarray:
        return vlad(store.raw(*job).des, centers).astype(np.float16)

    jobs = [(r, f) for r, route in enumerate(routes) for f in range(len(route.frames))]
    with ThreadPool(os.cpu_count() or 1) as pool:
        vlads = np.stack(
            list(tqdm(pool.imap(describe, jobs, chunksize=16), total=len(jobs), desc="VLAD"))
        )
    grid = result.variants["graph"].grid
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{f"poses_{r}": p for r, p in enumerate(poses)},
        routes=len(routes),
        log_odds=grid.log_odds,
        origin=grid.origin,
        res=grid.res,
        centers=centers,
        vlad=vlads,
        camera=np.array([camera_dict[k] for k in CAMERA_KEYS]),
    )
    store.flush()


CAMERA_KEYS = ("width", "height", "fx", "fy", "cx", "cy")


def load(path: Path, routes: list[mapping.Route], floor: FloorMask) -> FrozenMap:
    with np.load(path) as data:
        n = int(data["routes"])
        poses = [data[f"poses_{r}"] for r in range(n)]
        grid = OccupancyMap.__new__(OccupancyMap)
        grid.log_odds, grid.origin, grid.res = data["log_odds"], data["origin"], float(data["res"])
        camera_dict = dict(zip(CAMERA_KEYS, data["camera"].tolist(), strict=True))
        camera_dict.update(width=int(camera_dict["width"]), height=int(camera_dict["height"]))
        centers, vlads = data["centers"], data["vlad"].astype(np.float32)
    camera = Camera({**camera_dict, "cy": camera_dict["cy"] + mapping.HORIZON_SHIFT_PX})
    return FrozenMap(poses, grid, centers, vlads, SiftStore(routes, camera, floor, path.parent))


def get(
    root: Path, camera_dict: dict, floor: FloorMask, *, rebuild: bool = False
) -> tuple[FrozenMap, list[mapping.Route]]:
    """The frozen map of the recording at ``root``: loaded from ``root/map_graph.npz`` when
    present, else (or with ``rebuild``) built and saved first."""
    routes = mapping.load_routes(root)
    path = root / MAP_NAME
    if rebuild or not path.exists():
        for sidecar in root.glob("sift_r*.npz"):
            sidecar.unlink()
        freeze(root, routes, camera_dict, floor, path)
    return load(path, routes, floor), routes
