"""CameraWorker reader/processor decoupling.

The camera reader must NEVER be throttled by perception: the ESP32-CAM's MJPEG server is
single-consumer and stalls (then times out → reconnect backoff) if the reader pauses to
run inference. So a slow — even stuck — pipeline must not stop the reader from draining
the stream. We prove that with a fake fast frame source and a pipeline that blocks.
"""
import threading
import time

import numpy as np

import app.camera as cam_mod
from app.camera import CameraWorker
from app.config import cfg
from app.hub_client import Camera
from app.occupancy import Identity
from app.perception import DetectedTrack

_FAKE_JPEG = b"\xff\xd8" + b"x" * 512 + b"\xff\xd9"


class _FakeResp:
    def read(self, n=4096):
        return b""

    def close(self):
        pass


def _worker_with_fast_source():
    def fake_open_stream(url, timeout=10.0):
        return _FakeResp()

    def fake_iter(read, chunk=4096):
        while True:
            yield _FAKE_JPEG
            time.sleep(0.002)  # ~500 fps source

    cam_mod.open_stream = fake_open_stream
    cam_mod.iter_jpeg_frames = fake_iter
    return CameraWorker(Camera({"id": "t", "zone": "z", "ip": "1.2.3.4",
                                "stream": {"port": 81, "path": "/s"}}))


def test_records_only_when_rtsp_main_present():
    """Record scope: a camera archives footage ONLY when it declares a full-quality
    RTSP main stream (record_url). A face-ID cam (MJPEG only, no record_url) gets a
    hard-off recorder — no gated JPEG-pipe recording on a desk/entrance sensor."""
    face_id = CameraWorker(Camera({"id": "desk", "zone": "z", "ip": "1.2.3.4",
                                   "stream": {"port": 81, "path": "/s"}}))
    assert face_id.recorder.mode == "off"
    assert face_id.status()["records"] is False

    ip_cam = CameraWorker(Camera(
        {"id": "mc200", "zone": "sala", "ip": "1.2.3.5"},
        stream_url_override="rtsp://h/stream2",
        record_url_override="rtsp://h/stream1"))
    assert ip_cam.recorder.mode != "off"   # passthrough → continuous
    assert ip_cam.status()["records"] is True


def test_reader_not_blocked_by_stuck_pipeline():
    orig_open, orig_iter = cam_mod.open_stream, cam_mod.iter_jpeg_frames
    w = _worker_with_fast_source()
    entered = threading.Event()
    release = threading.Event()

    def blocking_pipeline(jpeg, now):
        entered.set()
        release.wait(timeout=3.0)  # block the processor on this one frame

    w._run_pipeline = blocking_pipeline
    w.start()
    try:
        assert entered.wait(2.0)   # processor picked up a frame and is now stuck
        time.sleep(0.3)            # ...while the reader keeps draining
        assert w.frames_seen > 20  # reader advanced far past the 1 stuck pipeline frame
    finally:
        release.set()
        w.stop()
        w.join(timeout=2.0)
        cam_mod.open_stream, cam_mod.iter_jpeg_frames = orig_open, orig_iter


# ── SMART_FACE_ID: guest re-verify + overlap taint ───────────────────────────
# These drive `_run_pipeline` directly (no threads) with fake collaborators, so the
# `need`/recheck decision and the overlap-taint pass are exercised end-to-end.

def _plain_worker():
    return CameraWorker(Camera({"id": "t", "zone": "z", "ip": "1.2.3.4",
                                "stream": {"port": 81, "path": "/s"}}))


class _FakeGallery:
    """Records recheck/resolve calls; recheck stays indecisive (None) so the CALL — not
    a relabel — is what a test pins."""
    def __init__(self):
        self.recheck_calls = []
        self.resolve_calls = []

    def recheck(self, emb):
        self.recheck_calls.append(emb)
        return None

    def resolve(self, emb, thumb=None, thumb_box=None):
        self.resolve_calls.append(emb)
        return Identity(id=None, name=None, cls="unknown", confidence=0.0)

    def default_label(self, guest_id):
        return "Person 1"


class _FakeTracker:
    def __init__(self):
        self.updates = []

    def update(self, cam_id, zone, observations, now, zone_occupied=None):
        self.updates.append((observations, zone_occupied))
        return []

    def snapshot(self, zone=None, now=None):
        return {}


class _FakeDetector:
    backend = "fake"

    def __init__(self, tracks):
        self._tracks = tracks

    def detect_and_track(self, frame):
        return self._tracks


def _wire_pipeline(monkeypatch, w, tracks, embeds):
    """Patch the pipeline's collaborators to fakes; return the fake gallery + tracker."""
    fg, ft = _FakeGallery(), _FakeTracker()
    w.detector = _FakeDetector(tracks)
    monkeypatch.setattr(cam_mod, "decode_jpeg", lambda jpeg: np.zeros((480, 640, 3), dtype=np.uint8))
    monkeypatch.setattr(cam_mod, "gallery", fg)
    monkeypatch.setattr(cam_mod, "tracker", ft)
    monkeypatch.setattr(cam_mod, "draw_overlay", lambda *a, **k: b"")
    monkeypatch.setattr(w, "_embed_tracks", lambda frame, trks, need: dict(embeds))
    return fg, ft


def test_guest_labeled_track_is_rechecked_after_reverify(monkeypatch):
    """The real bug: camera.py only re-verified `household` tracks, so a track mislabeled
    `guest` kept it for life. A guest past face_reverify_s must now get a recheck (the
    ledger's upgrade path merges it back into the member on a decisive household match)."""
    monkeypatch.setattr(cfg, "face_reverify_s", 20.0)
    w = _plain_worker()
    tid = "g1"
    w._idents[tid] = Identity(id="guest:1", name=None, cls="guest", confidence=0.4)
    w._ident_ts[tid] = 0.0
    tracks = [DetectedTrack(track_id=tid, bbox=(100, 100, 200, 300))]
    fg, _ = _wire_pipeline(monkeypatch, w, tracks, {tid: ([0.1] * 512, None, None)})
    w._run_pipeline(b"x", now=cfg.face_reverify_s + 5)
    assert fg.recheck_calls, "a guest-labelled track past face_reverify_s must be re-verified"


def test_overlapping_tracks_are_tainted_and_force_rechecked(monkeypatch):
    """Two person boxes crossing past overlap_taint_iou taint BOTH tracks → their cached
    labels are FORCE-rechecked within ~1s (not the 20s cadence), and the observations
    handed to the ledger are flagged tainted (barred from ghost adoption)."""
    monkeypatch.setattr(cfg, "face_reverify_s", 20.0)
    monkeypatch.setattr(cfg, "overlap_taint_iou", 0.35)
    monkeypatch.setattr(cfg, "taint_max_s", 300.0)
    w = _plain_worker()
    a, b = "a", "b"
    w._idents[a] = Identity(id="u1", name="David", cls="household", confidence=0.9)
    w._idents[b] = Identity(id="u2", name="Ana", cls="household", confidence=0.9)
    w._ident_ts[a] = w._ident_ts[b] = 100.0   # checked only 2s ago — periodic would NOT fire
    tracks = [DetectedTrack(track_id=a, bbox=(100, 100, 200, 300)),   # IoU ≈ 0.43
              DetectedTrack(track_id=b, bbox=(140, 100, 240, 300))]
    embeds = {a: ([0.1] * 512, None, None), b: ([0.2] * 512, None, None)}
    fg, ft = _wire_pipeline(monkeypatch, w, tracks, embeds)
    w._run_pipeline(b"x", now=102.0)
    # Both tracks force-rechecked despite the recent check — that's the taint path.
    assert len(fg.recheck_calls) == 2
    assert w._is_tainted(a, 102.0) and w._is_tainted(b, 102.0)
    obs, _ = ft.updates[-1]
    assert obs and all(o.tainted for o in obs)
