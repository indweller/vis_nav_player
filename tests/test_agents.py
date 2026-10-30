"""The keyboard agent and the runner against a stand-in session (no server, no window: SDL's
dummy driver), the .env loading and the servo."""

from __future__ import annotations

import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import cv2
import numpy as np
import pytest
from keyboard_agent import KeyboardAgent
from vis_nav_sdk import Action, run
from vis_nav_sdk import protocol as P
from vis_nav_sdk.session import (
    Arrival,
    Camera,
    Limits,
    Observation,
    Result,
    SessionInfo,
)
from vis_nav_sdk.telemetry import Telemetry
from vis_nav_sdk.viewer import Keys, Viewer


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
