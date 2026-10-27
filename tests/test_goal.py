"""The frozen map's pieces on synthetic data: no simulator, no recorded frames."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

import frozen_map  # noqa: E402
import mapping  # noqa: E402
from my_agent import Float32VLAD  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402


def test_vlad_matches_the_extractor(tmp_path):
    rng = np.random.default_rng(0)
    des = np.sqrt(rng.dirichlet(np.ones(128), 300)).astype(np.float32)
    extractor = Float32VLAD(n_clusters=8, cache_dir=tmp_path)
    extractor.codebook = KMeans(8, n_init=1, random_state=0).fit(des)
    expected = extractor._vlad(des)
    got = frozen_map.vlad(des, extractor.codebook.cluster_centers_.astype(np.float32))
    assert got == pytest.approx(expected, abs=1e-4)


def test_vlad_without_descriptors_is_zero():
    assert not frozen_map.vlad(None, np.zeros((4, 128), np.float32)).any()


def test_every_frame_poses_compose_from_the_nearest_earlier_node():
    n = mapping.NODE_EVERY
    dead = np.array([[0.03 * i, 0.0, 0.0] for i in range(2 * n + 1)])
    optimised = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, np.pi / 2], [5.0, 5.0, 0.0]])
    variant = mapping.Variant([optimised], None)
    result = mapping.MapResult({"graph": variant}, [], 0, dead_reckoned=[dead])
    poses = frozen_map.every_frame_poses(result)[0]
    assert poses[0] == pytest.approx(optimised[0])
    assert poses[n] == pytest.approx(optimised[1])
    # Two ticks past the second node: 0.06 m forward along its yaw (+y).
    assert poses[n + 2] == pytest.approx([1.0, 2.06, np.pi / 2], abs=1e-9)
