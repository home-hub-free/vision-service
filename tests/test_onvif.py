"""ONVIF client — verb→SOAP-body shapes, token traps, clamps, capability caching.

Fixtures under fixtures/onvif/ are REAL responses captured from the MC200
(2026-07-03) via tools/onvif_cli.py — the client is asserted against the exact
XML the camera actually speaks (pull_messages_motion.xml is the one synthetic,
spec-shaped exception; see its header).
"""
import os
import time

import pytest

from app.config import cfg
from app.hub_client import Camera
from app.onvif import (OnvifClient, OnvifError, client_for_camera,
                       notification_bool, parse_notifications)

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "onvif")


def fx(name: str) -> str:
    with open(os.path.join(FIX, name + ".xml")) as f:
        return f.read()


class FakeTransport:
    """Replaces OnvifClient._post: records (url, envelope) and plays back queued
    responses (a str, an Exception to raise, or a (matcher → str) callable)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, envelope, timeout):
        self.calls.append((url, envelope.decode()))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def sent(self, i=-1) -> str:
        return self.calls[i][1]


def make_client(*responses):
    c = OnvifClient("192.168.1.251", "u", "pw")
    t = FakeTransport(responses)
    c._post = t
    return c, t


# ── parsing against the real capture ──────────────────────────────────────────
def test_presets_parse_real_fixture():
    c, _ = make_client(fx("get_profiles"), fx("ptz_get_presets"))
    presets = c.get_presets()
    assert presets == [{"token": "1", "name": "hub-home",
                        "x": pytest.approx(0.047055), "y": pytest.approx(-0.571451)}]


def test_status_parse_real_fixture():
    c, _ = make_client(fx("get_profiles"), fx("ptz_get_status"))
    st = c.get_status()
    assert st["x"] == pytest.approx(-0.120058)
    assert st["y"] == pytest.approx(-0.022954)
    assert st["move_status"] == "UNKNOWN"


def test_profile_token_discovered_and_cached():
    c, t = make_client(fx("get_profiles"), fx("ptz_get_status"), fx("ptz_get_status"))
    assert c.profile_token() == "profile_1"
    c.get_status()
    c.get_status()
    # GetProfiles fetched exactly once; both status calls carry the cached token.
    assert len(t.calls) == 3
    assert "profile_1" in t.sent(1) and "profile_1" in t.sent(2)


def test_video_source_token_from_fixture():
    c, _ = make_client(fx("get_video_sources"))
    assert c.video_source_token() == "raw_vs1"


def test_soap_fault_raises_with_code():
    c, _ = make_client(fx("soap_fault"))
    with pytest.raises(OnvifError) as ei:
        c.call(c.dev, "<GetAudioOutputs/>")
    assert ei.value.fault is True
    assert ei.value.code == "ActionNotSupported"


# ── the profile-vs-video-source token trap (§0) ───────────────────────────────
def test_ptz_uses_profile_token_imaging_uses_video_source_token():
    c, t = make_client(fx("get_profiles"), "<ok/>",  # goto: profiles + verb
                       fx("get_video_sources"), fx("imaging_get_settings"))  # imaging
    c.goto_preset("1")
    assert "<tptz:ProfileToken>profile_1</tptz:ProfileToken>" in t.sent(1)
    c.get_imaging()
    assert "<timg:VideoSourceToken>raw_vs1</timg:VideoSourceToken>" in t.sent(3)


# ── PTZ move discipline ───────────────────────────────────────────────────────
def test_move_timed_clamps_velocity_and_ttl_and_auto_stops(monkeypatch):
    monkeypatch.setattr(cfg, "ptz_max_ttl_s", 0.1)
    c, t = make_client(fx("get_profiles"), "<ok/>", "<ok/>")  # profiles, move, stop
    used = c.move_timed(5.0, -3.0, ttl_s=60.0)
    assert used == pytest.approx(0.1)          # ttl clamped to the cap
    move_body = t.sent(1)
    assert 'x="1.0"' in move_body and 'y="-1.0"' in move_body  # velocity clamped
    time.sleep(0.4)                            # let the auto-stop timer fire
    assert any("<tptz:Stop>" in sent for _, sent in t.calls)


def test_new_move_supersedes_previous_timer(monkeypatch):
    monkeypatch.setattr(cfg, "ptz_max_ttl_s", 5.0)
    c, t = make_client(fx("get_profiles"), "<ok/>", "<ok/>", "<ok/>")
    c.move_timed(0.5, 0.0, ttl_s=5.0)
    first_timer = c._move_timer
    c.move_timed(-0.5, 0.0, ttl_s=5.0)
    assert first_timer is not c._move_timer
    assert first_timer.finished.is_set()  # cancelled, not left to fire a double-stop
    c._move_timer.cancel()


# ── capability probe caching ──────────────────────────────────────────────────
def test_capabilities_fault_caches_false():
    # nodes → fault (no PTZ); imaging (vsrc+settings) ok; events ok.
    c, t = make_client(fx("soap_fault"), fx("get_video_sources"),
                       fx("imaging_get_settings"), "<TopicSet>CellMotionDetector</TopicSet>")
    caps = c.capabilities()
    assert caps == {"ptz": False, "imaging": True, "events": True}
    n = len(t.calls)
    assert c.capabilities() == caps  # cached — no new transport traffic
    assert len(t.calls) == n
    assert c.capabilities_cached() == caps


def test_capabilities_transport_error_backs_off_not_cached():
    c, t = make_client(OSError("down"))
    with pytest.raises(OnvifError):
        c.capabilities()
    assert c.capabilities_cached() is None
    with pytest.raises(OnvifError):   # inside the backoff window: no new call
        c.capabilities()
    assert len(t.calls) == 1


# ── imaging merge/clamp semantics ─────────────────────────────────────────────
def test_set_imaging_merges_clamps_and_orders_elements():
    c, t = make_client(fx("get_video_sources"), fx("imaging_get_settings"), "<ok/>")
    merged = c.set_imaging({"brightness": 150.0, "contrast": None})
    assert merged["brightness"] == 100.0       # clamped
    assert merged["contrast"] == 50.0          # merged from current settings
    body = t.sent(2)
    # Schema order: Brightness → ColorSaturation → Contrast → Sharpness.
    assert body.index("tt:Brightness") < body.index("tt:ColorSaturation") \
        < body.index("tt:Contrast") < body.index("tt:Sharpness")


def test_set_imaging_ignores_ir_cut_when_camera_lacks_it():
    # The MC200's current fw has no IrCutFilter in GetImagingSettings (verified
    # 2026-07-03) — asking for ir_cut must not write the element (strict stacks 400).
    c, t = make_client(fx("get_video_sources"), fx("imaging_get_settings"), "<ok/>")
    merged = c.set_imaging({"ir_cut": "ON", "sharpness": 60})
    assert "ir_cut" not in merged
    assert "IrCutFilter" not in t.sent(2)


# ── events plumbing ───────────────────────────────────────────────────────────
def test_create_pullpoint_returns_subscription_address():
    c, _ = make_client(fx("create_pullpoint"))
    assert c.create_pullpoint() == "http://192.168.1.251:1024/event-1024_1024"


def test_parse_notifications_empty_and_motion():
    assert parse_notifications(fx("pull_messages_empty")) == []
    notes = parse_notifications(fx("pull_messages_motion"))
    assert len(notes) == 2
    assert "CellMotionDetector/Motion" in notes[0]["topic"]
    assert notification_bool(notes[0]["data"], "IsMotion") is True
    assert "Tamper" in notes[1]["topic"]
    assert notification_bool(notes[1]["data"], "IsTamper") is True
    assert notes[0]["ts"] == "2026-07-03T06:20:00Z"


# ── credentials come from the existing stream URL (plan §1 rule 2) ────────────
def test_client_for_camera_parses_rtsp_credentials():
    cam = Camera({"id": "x", "zone": "z"},
                 stream_url_override="rtsp://user:secret@10.0.0.9:554/stream2")
    c = client_for_camera(cam)
    assert c is not None
    assert (c.host, c.user, c.passwd) == ("10.0.0.9", "user", "secret")
    assert c.port == cfg.onvif_port


def test_client_for_camera_none_for_mjpeg_or_credless():
    assert client_for_camera(Camera({"id": "esp", "ip": "10.0.0.5",
                                     "stream": {"port": 81, "path": "/s"}})) is None
    assert client_for_camera(Camera({"id": "y"}, stream_url_override="rtsp://10.0.0.9/live")) is None
