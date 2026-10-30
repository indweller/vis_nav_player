"""Fixtures on the local simulator, skipped without vis_nav_sim, its texture pack or
data/floor_mask.npz."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from geometry import FLOOR_MASK_PATH

SEED = 7
DRIVE = [("FORWARD", 40), ("LEFT", 15), ("FORWARD", 30), ("RIGHT", 15), ("FORWARD", 20)]


@pytest.fixture(scope="session")
def textures():
    """The simulator's texture pack."""
    pytest.importorskip("vis_nav_sim")
    if not FLOOR_MASK_PATH.exists():
        pytest.skip("no data/floor_mask.npz")
    import vis_nav_sim as sim

    try:
        return sim.Textures.find()
    except FileNotFoundError:
        pytest.skip("no texture pack")


def record_tiny(root: Path, textures) -> None:
    """One short drive from the spawn of maze SEED in the format of ``Simulator.record``."""
    import vis_nav_sim as sim

    world = sim.Simulator(textures, SEED)
    route = root / "traj_0"
    route.mkdir(parents=True)
    records = []

    def save(frame, action) -> None:
        name = f"{len(records)}.jpg"
        (route / name).write_bytes(world.jpeg(frame))
        records.append({"step": len(records), "image": name, "action": [action]})

    save(world.initial_frame(), "IDLE")
    for action, ticks in DRIVE:
        for _ in range(ticks):
            save(world.step(getattr(sim, action), 1)[-1], action)
    (route / "data_info.json").write_text(json.dumps(records))
    (root / "target.jpg").write_bytes(world.jpeg(world.targets()[0]))


@pytest.fixture(scope="session")
def tiny_maze(tmp_path_factory, textures) -> Path:
    """The recording of ``record_tiny`` (its caches are made on first use)."""
    root = tmp_path_factory.mktemp("tiny")
    record_tiny(root, textures)
    return root
