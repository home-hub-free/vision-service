"""On-demand high-res sampling — far faces re-embedded from the main stream.

The substream keeps detect/track cheap, but a 40–70px face there is inside ArcFace's
noisy zone. When a NEW track's face is found-but-small, ONE full-res frame is fetched
(rate-limited, degradable) and the face re-embedded from it. These tests drive the
worker's upgrade orchestration and the sampler's failure posture with fakes — no
camera, no cv2 decode (fake frames only carry .shape).
"""
import time

from app.camera import CameraWorker
from app.config import cfg
from app.highres import HighResSampler, make_highres_sampler
from app.hub_client import Camera
from app.perception import DetectedTrack


class _Empty:
    size = 0


class _Frame:
    """Just enough ndarray for the scaling math (shape[0]=h, shape[1]=w) and for
    face_crop_jpeg's slicing (an empty crop → thumb stays None, which is fine here)."""

    def __init__(self, w, h):
        self.shape = (h, w, 3)

    def __getitem__(self, key):
        return _Empty()


LO, HI = _Frame(640, 360), _Frame(2560, 1440)  # 4× substream → main


class _ScaleAwareFace:
    """Fake engine: on the LO frame the face is small (40px); on the HI frame the
    same face is 160px. Keyed by frame width so both single- and multi-track paths
    exercise the same behavior."""

    backend = "fake"

    def embed_face(self, frame, bbox):
        if frame.shape[1] == LO.shape[1]:
            return [0.1] * 4, (bbox[0] + 10, bbox[1] + 10, bbox[0] + 50, bbox[1] + 50)
        return [0.9] * 4, (bbox[0] + 40, bbox[1] + 40, bbox[0] + 200, bbox[1] + 200)

    def faces(self, frame):
        return []


class _FakeSampler:
    def __init__(self, frame=None, degraded=False):
        self.frame = frame
        self.degraded = degraded
        self.calls = 0

    def get_frame(self):
        self.calls += 1
        return self.frame


def _worker():
    w = CameraWorker(Camera({"id": "t", "zone": "z", "ip": "1.2.3.4",
                             "stream": {"port": 81, "path": "/s"}}))
    w.face = _ScaleAwareFace()
    return w


def test_small_face_upgraded_from_one_highres_frame():
    cfg.highres_min_face_px = 90
    w = _worker()
    w.highres = _FakeSampler(frame=HI)
    tracks = [DetectedTrack(track_id="a", bbox=(0, 0, 100, 200))]
    out = w._embed_tracks(LO, tracks, {"a": "resolve"})
    assert out["a"][0] == [0.9] * 4          # the HI embedding won
    assert w.highres.calls == 1


def test_large_face_never_triggers_sampler():
    cfg.highres_min_face_px = 30             # 40px lo face is now "big enough"
    w = _worker()
    w.highres = _FakeSampler(frame=HI)
    out = w._embed_tracks(LO, [DetectedTrack(track_id="a", bbox=(0, 0, 100, 200))],
                          {"a": "resolve"})
    assert out["a"][0] == [0.1] * 4
    assert w.highres.calls == 0


def test_rate_limited_resolve_holds_back_noisy_embedding():
    """Sampler healthy but returns None (rate limit): a small-face RESOLVE is held
    back for this pass — the track retries within ms — instead of seeding a noisy
    far-face cluster. A RECHECK keeps its substream embedding path (an indecisive
    recheck is already a no-op)."""
    cfg.highres_min_face_px = 90
    w = _worker()
    w.highres = _FakeSampler(frame=None, degraded=False)
    tracks = [DetectedTrack(track_id="a", bbox=(0, 0, 100, 200))]
    out = w._embed_tracks(LO, tracks, {"a": "resolve"})
    assert out["a"] == (None, None, None)
    out = w._embed_tracks(LO, tracks, {"a": "recheck"})
    assert out["a"][0] == [0.1] * 4


def test_degraded_sampler_passes_substream_through():
    """A broken high-res source (session cap, camera quirk) must never starve
    recognition: degraded → the substream embedding is used as before."""
    cfg.highres_min_face_px = 90
    w = _worker()
    w.highres = _FakeSampler(frame=None, degraded=True)
    out = w._embed_tracks(LO, [DetectedTrack(track_id="a", bbox=(0, 0, 100, 200))],
                          {"a": "resolve"})
    assert out["a"][0] == [0.1] * 4


def test_sampler_rate_limit_and_degrade_then_selfheal():
    cfg.highres_interval_s = 60.0            # only the explicit clock moves it
    cam = Camera({"id": "t", "zone": "z", "ip": "1.2.3.4"},
                 stream_url_override="rtsp://u:p@h/stream2",
                 record_url_override="rtsp://u:p@h/stream1")
    s = HighResSampler(cam)
    fetches = {"n": 0, "result": None}

    def fake_fetch():
        fetches["n"] += 1
        return fetches["result"]

    s._fetch = fake_fetch
    assert s.get_frame() is None and fetches["n"] == 1   # attempt 1: fails
    assert s.get_frame() is None and fetches["n"] == 1   # rate-limited: no attempt
    for _ in range(2):                                    # 2 more failures → degraded
        s._last_attempt = 0.0
        s.get_frame()
    assert s.degraded and fetches["n"] == 3
    # Degraded → retry pace slows to 10× the interval...
    s._last_attempt = time.monotonic() - cfg.highres_interval_s * 2
    assert s.get_frame() is None and fetches["n"] == 3   # 2× interval: still held
    # ...but a successful fetch fully self-heals.
    fetches["result"] = HI
    s._last_attempt = 0.0
    assert s.get_frame() is HI
    assert not s.degraded


def test_sampler_only_for_dual_stream_cameras():
    cfg.highres_enabled = True
    esp32 = Camera({"id": "sat", "zone": "z", "ip": "1.2.3.4",
                    "stream": {"port": 81, "path": "/s"}})
    assert make_highres_sampler(esp32) is None            # one low-res stream only
    ipcam = Camera({"id": "c110", "zone": "z", "ip": "1.2.3.5"},
                   stream_url_override="rtsp://u:p@h/stream2",
                   record_url_override="rtsp://u:p@h/stream1")
    assert make_highres_sampler(ipcam) is not None
    cfg.highres_enabled = False
    try:
        assert make_highres_sampler(ipcam) is None        # kill switch
    finally:
        cfg.highres_enabled = True


def test_snapshot_creds_come_from_the_rtsp_url():
    cam = Camera({"id": "c", "zone": "z", "ip": "h"},
                 stream_url_override="rtsp://david:secret@h/stream2",
                 record_url_override="rtsp://david:secret@h/stream1")
    assert HighResSampler(cam)._creds() == ("david", "secret")
