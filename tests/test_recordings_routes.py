"""Footage-review surface: segment index reads, signed clip tokens, and the
/recordings/* routes (list bearer-gated, clip token-gated + Range-seekable).

A temp EventIndex + a temp recordings dir with a real .mp4 are swapped into the
routes module; the hub-session gate (require_user) is stubbed per-test.
"""
import os
import tempfile
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.config import cfg
from app.index_db import EventIndex
from app.main import app
from app.media_token import sign_clip, verify_clip
from app.routes import recordings as rec_routes

client = TestClient(app)

_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"video-bytes" * 64  # enough to Range-slice


class _FakeCam:
    def __init__(self, name):
        self.name = name


class _FakeWorker:
    """Stands in for a CameraWorker: only .status() + .cam.name are read by the routes."""
    def __init__(self, cam_id, zone, records):
        self.cam = _FakeCam("camera")
        self._st = {"id": cam_id, "zone": zone, "records": records}

    def status(self):
        return self._st


@pytest.fixture()
def env(monkeypatch):
    tmp = tempfile.mkdtemp()
    rec_dir = os.path.join(tmp, "recordings", "cam1")
    os.makedirs(rec_dir, exist_ok=True)
    idx = EventIndex(os.path.join(tmp, "index.db"))

    # One closed segment (a real file on disk) + one open (still recording) segment.
    seg_file = os.path.join(rec_dir, "20260704-120000.mp4")
    with open(seg_file, "wb") as fh:
        fh.write(_MP4)
    t0 = time.time() - 3600
    seg_id = idx.open_segment("cam1", "sala", seg_file, start_ts=t0)
    idx.close_segment(seg_id, end_ts=t0 + 300)

    monkeypatch.setattr(rec_routes, "index", idx)
    monkeypatch.setattr(rec_routes, "workers", {
        "cam1": _FakeWorker("cam1", "sala", records=True),
        "desk": _FakeWorker("desk", "oficina", records=False),  # face-ID cam, no footage
    })
    monkeypatch.setattr(rec_routes, "require_user", lambda authorization: {"id": "u1"})
    monkeypatch.setattr(cfg, "rec_dir", os.path.join(tmp, "recordings"))
    yield {"idx": idx, "seg_id": seg_id, "seg_file": seg_file, "t0": t0, "rec_dir": os.path.join(tmp, "recordings")}


# ── index reads ──────────────────────────────────────────────────────────────
def test_segments_between_and_days(env):
    idx, t0 = env["idx"], env["t0"]
    segs = idx.segments_between("cam1", t0 - 10, t0 + 400)
    assert len(segs) == 1 and segs[0]["duration"] == pytest.approx(300, abs=1)
    assert idx.segments_between("cam1", t0 + 10_000, t0 + 20_000) == []  # window past footage
    days = idx.recording_days("cam1")
    assert len(days) == 1 and days[0] == time.strftime("%Y-%m-%d", time.localtime(t0))


def test_segment_by_id_resolves_and_misses(env):
    seg = env["idx"].segment_by_id(env["seg_id"])
    assert seg and seg["cam_id"] == "cam1" and seg["file"] == env["seg_file"]
    assert env["idx"].segment_by_id(9999) is None


# ── signed clip token ────────────────────────────────────────────────────────
def test_clip_token_roundtrip_and_tamper():
    tok = sign_clip(7)
    assert verify_clip(7, tok)
    assert not verify_clip(8, tok)                       # bound to the seg id
    assert not verify_clip(7, tok + "x")                 # tampered sig
    assert not verify_clip(7, "")                        # empty
    expired = sign_clip(7, ttl_s=-1)
    assert not verify_clip(7, expired)                   # past expiry


# ── list routes (bearer-gated) ───────────────────────────────────────────────
def test_cameras_lists_only_recording_cams(env):
    body = client.get("/recordings/cameras").json()
    ids = {c["id"] for c in body["cameras"]}
    assert ids == {"cam1"}                               # the face-ID "desk" cam is excluded
    assert body["cameras"][0]["days"]


def test_segments_route_carries_signed_clip(env):
    t0 = env["t0"]
    body = client.get(f"/recordings/cam1/segments?start={t0 - 10}&end={t0 + 400}").json()
    assert len(body["segments"]) == 1
    clip = body["segments"][0]["clip"]
    assert clip.startswith("recordings/cam1/clip/") and "token=" in clip


def test_list_routes_require_auth(env, monkeypatch):
    def _deny(authorization):
        raise HTTPException(status_code=401, detail="missing bearer token")
    monkeypatch.setattr(rec_routes, "require_user", _deny)
    assert client.get("/recordings/cameras").status_code == 401
    assert client.get("/recordings/cam1/segments?start=0&end=1").status_code == 401


# ── clip route (token-gated, Range-seekable) ─────────────────────────────────
def test_clip_streams_with_valid_token_and_range(env):
    seg_id = env["seg_id"]
    tok = sign_clip(seg_id)
    r = client.get(f"/recordings/cam1/clip/{seg_id}?token={tok}")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert r.content == _MP4
    # Range request → 206 partial (the <video> element seeks this way).
    r2 = client.get(f"/recordings/cam1/clip/{seg_id}?token={tok}", headers={"Range": "bytes=0-15"})
    assert r2.status_code == 206 and len(r2.content) == 16


def test_clip_rejects_bad_token_and_unknown_segment(env):
    seg_id = env["seg_id"]
    assert client.get(f"/recordings/cam1/clip/{seg_id}?token=nope").status_code == 403
    good = sign_clip(9999)
    assert client.get(f"/recordings/cam1/clip/9999?token={good}").status_code == 404
    # Right seg id, wrong camera in the path → 404 (id/camera must agree).
    tok = sign_clip(seg_id)
    assert client.get(f"/recordings/other/clip/{seg_id}?token={tok}").status_code == 404


def test_clip_blocks_path_traversal(env, monkeypatch):
    """A segment row whose file escapes rec_dir must be refused even with a valid token."""
    idx = env["idx"]
    escaped = os.path.join(os.path.dirname(env["rec_dir"]), "..", "etc", "passwd")
    bad_id = idx.open_segment("cam1", "sala", escaped)
    tok = sign_clip(bad_id)
    assert client.get(f"/recordings/cam1/clip/{bad_id}?token={tok}").status_code in (403, 404)
