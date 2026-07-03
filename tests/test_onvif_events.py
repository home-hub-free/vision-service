"""EventPuller — edge extraction, dedupe, resubscribe-on-drop, gate fail-open.

The puller consumes the camera's own CellMotion/Tamper (plan §3): a motion edge
must flip the worker's YOLO pre-gate state + publish exactly once per transition
(channel `motion` = pull-lane/memory-only; `tamper` rides to the agent), and a
dead camera/subscription must resubscribe forever without crashing — with
`events_attached` false meanwhile so the detect gate fails OPEN.
"""
import os
import time

import pytest

import app.ingest as ingest
import app.onvif_events as ev_mod
from app.config import cfg
from app.onvif import OnvifError, parse_notifications
from app.onvif_events import EventPuller

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "onvif")


def fx(name):
    with open(os.path.join(FIX, name + ".xml")) as f:
        return f.read()


class FakeWorker:
    def __init__(self):
        class Cam:
            id = "mc200"
            zone = "entrance"
        self.cam = Cam()
        self.events_attached = False
        self.motion_active = False
        self.last_motion_ts = 0.0


@pytest.fixture()
def published(monkeypatch):
    out = []
    monkeypatch.setattr(ingest, "publish_signal",
                        lambda zone, cam, ch, val, meta=None: out.append((zone, cam, ch, val, meta)))
    return out


def make_puller(client=None):
    w = FakeWorker()
    return EventPuller(w, client), w


# ── dispatch: edges, provenance, dedupe ───────────────────────────────────────
def test_motion_edge_flips_gate_and_publishes_once(published):
    p, w = make_puller()
    notes = parse_notifications(fx("pull_messages_motion"))
    now = time.time()
    for n in notes:
        p._dispatch(n, now)
    assert w.motion_active is True and w.last_motion_ts == now
    assert published == [
        ("entrance", "mc200", "motion", True, {"provenance": "camera_motion"}),
        ("entrance", "mc200", "tamper", True, {"provenance": "camera_tamper"}),
    ]
    # Same state again → no duplicate publish (edges, not levels).
    for n in notes:
        p._dispatch(n, now + 1)
    assert len(published) == 2


def test_motion_clear_publishes_false_edge(published):
    p, w = make_puller()
    p._dispatch({"topic": "tns1:RuleEngine/CellMotionDetector/Motion",
                 "data": {"IsMotion": "true"}}, 100.0)
    p._dispatch({"topic": "tns1:RuleEngine/CellMotionDetector/Motion",
                 "data": {"IsMotion": "false"}}, 105.0)
    assert w.motion_active is False
    assert w.last_motion_ts == 100.0  # stamped on the rising edge (linger anchor)
    assert [e[3] for e in published] == [True, False]


def test_unknown_topic_and_missing_item_are_ignored(published):
    p, _ = make_puller()
    p._dispatch({"topic": "tns1:Something/Else", "data": {"X": "1"}}, 1.0)
    p._dispatch({"topic": "tns1:RuleEngine/CellMotionDetector/Motion", "data": {}}, 1.0)
    assert published == []


# ── loop: resubscribe on drop, gate fails open ────────────────────────────────
class FlakyClient:
    """First create_pullpoint raises (camera rebooting); then a subscription that
    yields one empty pull, then dies — forcing a full resubscribe cycle."""

    def __init__(self):
        self.creates = 0
        self.pulls = 0

    def create_pullpoint(self, term_s=120):
        self.creates += 1
        if self.creates == 1:
            raise OnvifError("unreachable: down")
        return "http://cam:1024/event"

    def pull_messages(self, sub, timeout_s=10.0, limit=32):
        self.pulls += 1
        if self.pulls > 1:
            raise OnvifError("subscription gone", fault=True)
        return []

    def renew(self, sub, term_s=120):
        pass

    def unsubscribe(self, sub):
        pass


def test_puller_resubscribes_and_never_dies(monkeypatch, published):
    monkeypatch.setattr(cfg, "onvif_pull_timeout_s", 0.01)
    client = FlakyClient()
    p, w = make_puller(client)
    # Shrink the backoff so the test runs in ms, keeping the logic intact.
    orig_wait = p._stop_evt.wait
    monkeypatch.setattr(p._stop_evt, "wait", lambda t=None: orig_wait(min(t or 0, 0.02)))
    p.start()
    try:
        deadline = time.time() + 3.0
        while client.creates < 3 and time.time() < deadline:
            time.sleep(0.01)
        assert client.creates >= 3      # failed once, then kept resubscribing
        assert p.is_alive()
    finally:
        p.stop()
        p.join(timeout=2.0)
    assert w.events_attached is False   # detached on stop → detect gate fails open


# ── the worker-side gate (camera.py) ──────────────────────────────────────────
def test_motion_gate_defaults_open_and_respects_linger(monkeypatch):
    from app.camera import CameraWorker
    from app.hub_client import Camera

    w = CameraWorker(Camera({"id": "g", "zone": "z", "ip": "1.2.3.4",
                             "stream": {"port": 81, "path": "/s"}}))
    now = 1000.0
    assert w._motion_gate_open(now)                      # knob off → open
    monkeypatch.setattr(cfg, "detect_on_motion", True)
    assert w._motion_gate_open(now)                      # no subscription → fail OPEN
    w.events_attached = True
    assert not w._motion_gate_open(now)                  # attached + no motion → gated
    w.motion_active = True
    assert w._motion_gate_open(now)                      # motion → open
    w.motion_active = False
    w.last_motion_ts = now
    assert w._motion_gate_open(now + cfg.motion_linger_s - 1)   # linger holds it open
    assert not w._motion_gate_open(now + cfg.motion_linger_s + 1)
