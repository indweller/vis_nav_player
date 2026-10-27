"""Both agents against a stand-in session: no server, no window (SDL dummy driver)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from baseline_agent import BaselineAgent  # noqa: E402
from keyboard_agent import KeyboardAgent  # noqa: E402
from vis_nav_sdk import Action, run  # noqa: E402
from vis_nav_sdk import protocol as P  # noqa: E402
from vis_nav_sdk.session import (  # noqa: E402
    Arrival,
    Camera,
    Limits,
    Observation,
    Result,
    SessionInfo,
)
from vis_nav_sdk.telemetry import Telemetry  # noqa: E402
from vis_nav_sdk.viewer import Keys, Viewer  # noqa: E402


def _image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros((240, 320, 3), np.uint8)
    for _ in range(12):
        x, y = rng.integers(0, 300), rng.integers(0, 220)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(image, (x, y), (x + 40, y + 30), color, -1)
    return image


def _info() -> SessionInfo:
    return SessionInfo(
        session_id="s",
        challenge={"id": "c", "name": "maze"},
        camera=Camera(320, 240, "bgr", "jpeg", np.eye(3)),
        limits=Limits(
            max_steps=200,
            max_repeat=P.MAX_REPEAT,
            attempts_used=0,
            attempts_allowed=3,
            silence_timeout_s=120,
            grace_s=300,
            session_timeout_s=3600,
        ),
        targets=[_image(100 + i) for i in range(4)],
    )


class FakeSession:
    def __init__(self) -> None:
        self.info = _info()
        self.limits = self.info.limits
        self.session_id = self.info.session_id
        self.initial_observation = Observation(_image(0), 0, 200)
        self.calls: list[tuple[int, int]] = []
        self.aborted = None
        self.result = None
        self.telemetry = Telemetry()

    def step(self, action, repeat=1):
        self.calls.append((int(action), repeat))
        used = sum(r for _, r in self.calls)
        return Observation(_image(used), used, 200 - used)

    def checkin(self):
        self.result = Result(
            goal_tier="partial",
            nav_steps=sum(r for _, r in self.calls),
            spl=0.8,
            points=3.0,
            max_points=5.0,
            awarded={},
            job_id="j",
            session_id="s",
            arrival=Arrival(distance_m=0.1, heading_deg=5.0),
        )
        return self.result

    def abort(self, reason=""):
        self.aborted = reason

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class FakeClient:
    server = "http://fake"

    def __init__(self, api_key=None, *, server=None) -> None:
        self.api_key = api_key

    def challenge(self, challenge_id):
        return {"id": challenge_id, "name": "maze"}

    def quota(self, challenge_id):
        return {"attempts_used": 0, "attempts_allowed": 3, "running": 0, "reservation": None}

    def start_session(self, challenge_id):
        return {
            "session_id": "s",
            "token": "vns_s_t",
            "expires_at": 2**53,
            "attempts_used": 0,
            "attempts_allowed": 3,
            "max_steps": 200,
            "final_submission_link": None,
            "page_url": "https://site/challenges/c?session=s",
        }


@pytest.fixture
def session(monkeypatch, tmp_path) -> FakeSession:
    from vis_nav_sdk import rest
    from vis_nav_sdk import session as session_module

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    fake = FakeSession()
    monkeypatch.setattr(rest, "Client", FakeClient)
    monkeypatch.setattr(session_module, "redeem", lambda *a, **k: fake)
    return fake


@pytest.fixture
def keys(monkeypatch):
    """Script the keyboard: one set of held key names per act() call, then nothing."""
    script: list[set[str]] = []

    def fake_keys(self):
        held = frozenset(script.pop(0) if script else set())
        return Keys(held=held, tapped=frozenset())

    monkeypatch.setattr(Viewer, "keys", fake_keys)
    return script


def test_keyboard_agent_moves_one_tick_per_frame(session, keys):
    keys += [
        set(),  # pre-flight
        {"up"},
        {"up", "left"},
        set(),  # released: wait, no tick burnt
        {"space"},
    ]
    viewer = Viewer(scale=1, hold_s=0)
    result = run(KeyboardAgent(), "vns_c_x", viewer=viewer, fps=None, quiet=True)
    assert result is not None
    assert session.calls == [(int(Action.FORWARD), 1), (int(Action.FORWARD | Action.LEFT), 1)]


def test_the_runner_paces_a_window(session, keys):
    keys += [set()] + [{"up"}] * 6 + [{"space"}]
    viewer = Viewer(scale=1, hold_s=0)
    started = time.monotonic()
    run(KeyboardAgent(), "vns_c_x", viewer=viewer, fps=20, quiet=True)
    assert len(session.calls) == 6
    assert time.monotonic() - started >= 6 / 20 * 0.8


def test_keyboard_agent_escape_quits(session, keys):
    keys += [set(), {"escape"}]
    viewer = Viewer(scale=1, hold_s=0)
    assert run(KeyboardAgent(), "vns_c_x", viewer=viewer, quiet=True) is None
    assert session.aborted == "agent quit"


def test_keyboard_agent_needs_a_window(session):
    with pytest.raises(Exception, match="viewer"):
        run(KeyboardAgent(), "vns_c_x", viewer=False, quiet=True)
    assert session.calls == []


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    for t in range(2):
        traj = data / f"traj_{t}"
        traj.mkdir(parents=True)
        records = []
        for k in range(12):
            cv2.imwrite(str(traj / f"{k}.jpg"), _image(1000 * t + k))
            records.append(
                {"step": k, "image": f"{k}.jpg", "action": ["FORWARD" if k % 3 else "LEFT"]}
            )
        records[5]["action"] = ["FORWARD", "LEFT"]  # a combined action is skipped
        (traj / "data_info.json").write_text(json.dumps(records))
    return data


def test_baseline_indexes_localises_and_hints(dataset, tmp_path, session, keys):
    agent = BaselineAgent(
        dataset, n_clusters=4, subsample=1, top_k_shortcuts=3, cache_dir=tmp_path / "cache"
    )
    assert len(agent.frames) == 22
    assert agent.graph.number_of_nodes() == 22

    keys += [set(), {"up"}, {"space"}]
    result = run(agent, "vns_c_x", viewer=Viewer(scale=1, hold_s=0), fps=None, quiet=True)
    assert result is not None
    assert agent.goal is not None and agent.current is not None
    assert any("node" in line for line in agent.hud())
    panel = agent.panel()
    assert panel and panel[0][1].startswith("best match") and panel[0][0].shape == (240, 320, 3)

    # A second construction hits the caches.
    again = BaselineAgent(
        dataset, n_clusters=4, subsample=1, top_k_shortcuts=3, cache_dir=tmp_path / "cache"
    )
    assert np.allclose(again.database, agent.database)


@pytest.mark.parametrize("use_filter", [True, False])
def test_my_agent_drives_itself(dataset, tmp_path, session, use_filter):
    from my_agent import MyAgent

    agent = MyAgent(
        dataset,
        n_clusters=4,
        subsample=1,
        top_k_shortcuts=3,
        random_every=0,
        cache_dir=tmp_path / "cache",
        filter=use_filter,
    )
    result = run(agent, "vns_c_x", viewer=False, fps=None, quiet=True)
    assert result is not None
    assert session.calls
    moves = {int(getattr(Action, name)) for name in ("FORWARD", "BACKWARD", "LEFT", "RIGHT")}
    for action, repeat in session.calls:
        assert action in moves
        assert 0 < repeat <= P.MAX_REPEAT


def test_my_agent_replays_the_raw_actions_between_nodes(dataset, tmp_path):
    from my_agent import MyAgent, _moves

    agent = MyAgent(
        dataset,
        n_clusters=4,
        subsample=2,
        top_k_shortcuts=3,
        cache_dir=tmp_path / "cache",
        capture_every=3,
    )
    # Records 0-11 with 5 combined: kept 0 1 2 3 4 6 7 8 9 10 11, then every 2nd.
    assert agent.raw_index[:6] == [0, 2, 4, 7, 9, 11]
    assert agent.segments[0, 1] == ["LEFT", "FORWARD"]
    assert agent.segments[2, 3] == ["FORWARD", "LEFT"]  # record 5 dropped
    assert agent.segments[3, 2] == ["RIGHT", "BACKWARD"]
    assert _moves(["FORWARD"] * 4 + ["LEFT"], 3) == [("FORWARD", 12), ("LEFT", 3)]
    long = _moves(["FORWARD"] * P.MAX_REPEAT, 2)
    assert [t for _, t in long] == [P.MAX_REPEAT, P.MAX_REPEAT]
    assert _moves(["LEFT"] * 7, 3, max_ticks=15) == [("LEFT", 15), ("LEFT", 6)]


def test_my_agent_penalises_a_stalled_edge_and_resets(dataset, tmp_path):
    import networkx as nx
    from my_agent import MyAgent

    agent = MyAgent(
        dataset,
        n_clusters=4,
        subsample=1,
        top_k_shortcuts=3,
        cache_dir=tmp_path / "cache",
        stall_window=2,
        stall_penalty=5.0,
    )
    agent.setup(_info())
    agent.goal, agent.estimate, agent.current = 5, 1, 1
    agent.path = [1, 2, 3, 4, 5]
    weight = agent.graph[1][2]["weight"]
    for step in range(3):  # a first best, then two estimates without a better one
        agent._guard(step)
    hops = nx.shortest_path_length(agent.graph, 1, 5)
    assert agent.stalls == [{"step": 2, "edge": (1, 2), "hops": hops}]
    assert agent.graph[1][2]["weight"] == pytest.approx(5 * weight)
    assert agent.queue and agent.edge is not None

    agent.setup(_info())
    assert agent.stalls == [] and agent.graph[1][2]["weight"] == weight


def test_node_filter_follows_the_action_and_the_view():
    import networkx as nx
    from node_filter import NodeFilter
    from vlad import Frame

    graph = nx.path_graph(10)
    frames = [Frame(Path(f"{i}.jpg"), "FORWARD", 0) for i in range(10)]
    node_filter = NodeFilter(graph, frames, tau=0.01)
    for matrix in node_filter.transitions.values():
        assert np.allclose(matrix.sum(axis=1), 1)

    node_filter.belief = np.eye(10)[3]
    node_filter.predict("FORWARD")
    assert node_filter.belief[4] > node_filter.belief[2]

    node_filter.update(np.eye(10)[7])
    node, confidence = node_filter.estimate()
    assert node == 7
    assert 0 < confidence <= 1


def test_dotenv_fills_the_environment_without_overriding_it(tmp_path, monkeypatch):
    """Two lines in .env are how the key and the challenge reach the scripts. Quotes are
    allowed, comments and blanks skipped, and anything already in the environment wins."""
    import cli

    env = tmp_path / ".env"
    env.write_text(
        '# course\nVIS_NAV_API_KEY="k-from-file"\nVIS_NAV_CHALLENGE=abc123\n\nBROKEN LINE\n'
    )
    monkeypatch.delenv("VIS_NAV_API_KEY", raising=False)
    monkeypatch.setenv("VIS_NAV_CHALLENGE", "from-shell")
    cli.load_dotenv(env)
    assert os.environ["VIS_NAV_API_KEY"] == "k-from-file"
    assert os.environ["VIS_NAV_CHALLENGE"] == "from-shell"
    cli.load_dotenv(tmp_path / "missing.env")  # absent is fine


def test_servo_finds_the_sign_and_size_of_a_shift(tmp_path):
    import math

    from servo import Servo

    rng = np.random.default_rng(3)
    scene = np.zeros((240, 480, 3), np.uint8)
    for _ in range(150):
        x, y = int(rng.integers(0, 470)), int(rng.integers(0, 230))
        w, h = (int(v) for v in rng.integers(6, 30, 2))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(scene, (x, y), (x + w, y + h), color, -1)
    left, right = scene[:, 40:360], scene[:, 80:400]  # right's view lies 40 px to the right
    cv2.imwrite(str(tmp_path / "left.jpg"), left)
    cv2.imwrite(str(tmp_path / "right.jpg"), right)
    fx = 300.0
    servo = Servo(fx)
    expected = math.degrees(math.atan(40 / fx))

    yaw, inliers = servo.yaw_offset(left, tmp_path / "right.jpg")
    assert yaw is not None and inliers >= servo.min_matches
    assert yaw == pytest.approx(expected, abs=1.0)
    yaw, _ = servo.yaw_offset(right, tmp_path / "left.jpg")
    assert yaw == pytest.approx(-expected, abs=1.0)
    servo.yaw_offset(left, tmp_path / "right.jpg")
    assert servo.image_features.cache_info().misses == 2  # cached by path

    blank = np.zeros_like(left)
    assert servo.yaw_offset(blank, tmp_path / "left.jpg") == (None, 0)


def test_my_agent_servo_follows_the_raw_records_of_an_edge(dataset, tmp_path):
    from my_agent import MyAgent

    agent = MyAgent(
        dataset, n_clusters=4, subsample=2, top_k_shortcuts=3, cache_dir=tmp_path / "cache"
    )
    # raw_index starts 0 2 4 7: edge 3 -> 2 walks records 7 down to 4, combined record 5 kept.
    agent._follow((3, 2))
    assert agent.span == [(0, 7), (0, 6), (0, 5), (0, 4)]
    assert agent._record_image((0, 5)) == dataset / "traj_0" / "5.jpg"
    assert agent._translation((2, 3)) == "FORWARD"
    assert agent._translation((3, 2)) == "BACKWARD"

    agent.setup(_info())
    assert agent.servo is not None
    agent.path, agent.goal = [3, 2, 1], 1
    image = cv2.imread(str(agent._record_image((0, 6))))
    agent._aim(image)  # the view is record 6's own: located there, aiming at record 5
    assert agent.aimed and agent.located == (0, 6) and agent.target == (0, 5)
    assert agent.follow == (3, 2) and agent.at == 1
