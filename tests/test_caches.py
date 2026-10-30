"""Cache fingerprints: stale or unversioned caches warn once and are used anyway."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import caches
import map_agent
import numpy as np
from geometry import FloorMask, WallView
from planner import PLACE_EVERY, PlaceGraph

MOTION = {"FORWARD": {"dx": 0.03, "dy": 0.0, "dyaw": 0.0, "std": [0, 0, 0], "n": [1, 1, 1]}}


def test_check_warns_once_per_file_and_not_when_current(tmp_path, capsys):
    caches.check(tmp_path / "a", "x", "x")
    assert capsys.readouterr().out == ""
    caches.check(tmp_path / "b", None, "x")
    caches.check(tmp_path / "b", None, "x")
    assert capsys.readouterr().out.count("predates cache versioning") == 1
    caches.check(tmp_path / "c", "y", "x")
    assert "other settings" in capsys.readouterr().out


def test_fingerprint_follows_the_settings(monkeypatch):
    before = caches.fingerprint(caches.MOTION, FloorMask())
    assert caches.fingerprint(caches.MOTION, FloorMask()) == before
    monkeypatch.setattr(caches.mapping, "HORIZON_SHIFT_PX", caches.mapping.HORIZON_SHIFT_PX + 1)
    assert caches.fingerprint(caches.MOTION, FloorMask()) != before


def test_fingerprint_follows_the_floor_mask():
    floor = FloorMask()
    before = caches.fingerprint(caches.MOTION, floor)
    floor.table[0, 0, 0] = True
    assert caches.fingerprint(caches.MOTION, floor) != before


def test_motion_keeps_its_fingerprint_out_of_the_motion(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(map_agent.mapping, "calibrate", lambda *a: dict(MOTION))
    floor = FloorMask()
    assert map_agent.load_motion(tmp_path, [], None, floor) == MOTION
    assert "fingerprint" in json.loads((tmp_path / "motion.json").read_text())
    assert map_agent.load_motion(tmp_path, [], None, floor) == MOTION
    assert capsys.readouterr().out == ""


def test_legacy_motion_warns_and_loads(tmp_path, capsys):
    (tmp_path / "motion.json").write_text(json.dumps(MOTION))
    assert map_agent.load_motion(tmp_path, [], None, FloorMask()) == MOTION
    assert "predates cache versioning" in capsys.readouterr().out


def _links_graph(cache_path: Path) -> PlaceGraph:
    """Two parallel routes 0.1 m apart: links between them, each with no inliers."""
    t = np.linspace(0, 0.9, 9 * PLACE_EVERY + 1)
    left = np.column_stack([0 * t, t, 0 * t])
    right = np.column_stack([0 * t + 0.1, t, 0 * t])
    empty = WallView(np.empty((0, 2), np.float32), None, np.empty((0, 2)), np.empty((0, 2)))
    store = SimpleNamespace(poses=[left, right], view=lambda row: empty)
    return PlaceGraph(store, filter_walls=False, cache_path=cache_path)


def test_links_are_stamped_and_a_stale_stamp_is_kept(tmp_path, capsys):
    path = tmp_path / "links.npz"
    _links_graph(path)
    with np.load(path) as data:
        assert str(data["fingerprint"]) == caches.fingerprint(caches.LINKS)
        pairs, inliers = data["pairs"], data["inliers"]
    assert len(pairs) > 1
    np.savez(path, pairs=pairs[:1], inliers=inliers[:1], fingerprint="stale")
    _links_graph(path)
    assert "other settings" in capsys.readouterr().out
    with np.load(path) as data:
        assert str(data["fingerprint"]) == "stale"  # entries added, still marked stale
        assert len(data["pairs"]) == len(pairs)
