"""Privacy mode — the per-camera "stop watching NOW" switch.

Three layers under test:
  * PrivacyStore — persisted across restarts (a reboot must never silently resume
    surveillance the household switched off).
  * Routes — POST /privacy/{cam} applies immediately (worker teardown called before
    the response), GET /privacy lists every worker, streams/snapshot answer 423.
  * CameraWorker — a live reader goes dark on toggle (frames stop, relay slots
    cleared, occupancy withdrawn) and resumes when lifted.
"""
import time

import pytest
from fastapi.testclient import TestClient

import app.camera as cam_mod
from app.camera import CameraWorker
from app.hub_client import Camera
from app.main import app
from app.occupancy import Identity, Observation, OccupancyTracker
from app.privacy import PrivacyStore
from app.state import privacy, tracker, workers

client = TestClient(app)

_FAKE_JPEG = b"\xff\xd8" + b"x" * 512 + b"\xff\xd9"


@pytest.fixture(autouse=True)
def clean_privacy(tmp_path):
    """Point the shared store at a scratch file so tests never touch data/privacy.json."""
    privacy.path = str(tmp_path / "privacy.json")
    privacy._private.clear()
    yield
    privacy._private.clear()


# ── store persistence ─────────────────────────────────────────────────────────

def test_store_persists_across_restart(tmp_path):
    path = str(tmp_path / "p.json")
    store = PrivacyStore(path)
    assert store.is_private("cam1") is False
    store.set("cam1", True)
    assert PrivacyStore(path).is_private("cam1") is True   # a fresh boot keeps it
    store.set("cam1", False)
    assert PrivacyStore(path).is_private("cam1") is False


def test_store_survives_corrupt_file(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{not json")
    store = PrivacyStore(str(path))  # must not raise
    assert store.all() == {}


# ── routes ────────────────────────────────────────────────────────────────────

class FakeWorker:
    def __init__(self, cam_id, zone="sala"):
        class Cam:
            pass
        self.cam = Cam()
        self.cam.id = cam_id
        self.cam.zone = zone
        self.latest_raw = _FAKE_JPEG
        self.latest_annotated = None
        self.paused = 0

    def pause_for_privacy(self):
        self.paused += 1


@pytest.fixture
def fake_workers():
    workers.clear()
    workers["c110"] = FakeWorker("c110")
    workers["esp"] = FakeWorker("esp", zone="oficina")
    yield workers
    workers.clear()


def test_unknown_camera_404s(fake_workers):
    assert client.post("/privacy/nope", json={"on": True}).status_code == 404


def test_toggle_applies_immediately_and_lists(fake_workers):
    r = client.post("/privacy/c110", json={"on": True})
    assert r.status_code == 200
    assert r.json() == {"cam_id": "c110", "zone": "sala", "privacy": True}
    assert fake_workers["c110"].paused == 1        # teardown ran before the response
    assert privacy.is_private("c110") is True

    r = client.get("/privacy")
    assert r.json()["cameras"] == {"c110": True, "esp": False}

    r = client.post("/privacy/c110", json={"on": False})
    assert r.json()["privacy"] is False
    assert fake_workers["c110"].paused == 1        # OFF never calls the teardown
    assert client.get("/privacy").json()["cameras"] == {"c110": False, "esp": False}


def test_off_roster_private_camera_still_listed(fake_workers):
    """A privacy flag can't hide by unplugging the camera: persisted ids without a
    live worker still show up (private) in the listing."""
    privacy.set("ghost", True)
    assert client.get("/privacy").json()["cameras"]["ghost"] is True


def test_stream_and_snapshot_refuse_while_private(fake_workers):
    assert client.get("/snapshot/c110").status_code == 200
    client.post("/privacy/c110", json={"on": True})
    assert client.get("/snapshot/c110").status_code == 423
    assert client.get("/stream/c110").status_code == 423
    assert client.get("/stream/c110/raw").status_code == 423
    # the other camera is untouched
    assert client.get("/snapshot/esp").status_code == 200


# ── occupancy withdrawal ──────────────────────────────────────────────────────

def test_drop_camera_withdraws_tracks_silently():
    trk = OccupancyTracker()
    david = Identity(cls="household", id="u1", name="David", confidence=0.9)
    ana = Identity(cls="household", id="u2", name="Ana", confidence=0.9)
    now = time.time()
    # enough consecutive sightings to cross "present" on both cameras (one person
    # per camera — the ledger dedupes the SAME person across cameras to one entry)
    for i in range(5):
        trk.update("camA", "sala", [Observation("t1", david)], now + i)
        trk.update("camB", "sala", [Observation("t2", ana)], now + i)
    assert len(trk.snapshot("sala").get("sala", [])) == 2

    trk.drop_camera("camA")
    remaining = trk.snapshot("sala").get("sala", [])
    assert len(remaining) == 1                      # camB's observation survives
    assert remaining[0]["track"].startswith("camB:")

    trk.drop_camera("camB")
    assert trk.snapshot("sala") == {}               # zone cleanly empty, no crash


# ── live worker goes dark + resumes ───────────────────────────────────────────

def _fast_source_worker():
    def fake_open_stream(url, timeout=10.0):
        class R:
            def read(self, n=4096):
                return b""

            def close(self):
                pass
        return R()

    def fake_iter(read, chunk=4096):
        while True:
            yield _FAKE_JPEG
            time.sleep(0.002)

    cam_mod.open_stream = fake_open_stream
    cam_mod.iter_jpeg_frames = fake_iter
    return CameraWorker(Camera({"id": "pv", "zone": "z", "ip": "1.2.3.4",
                                "stream": {"port": 81, "path": "/s"}}))


def _wait(cond, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_worker_goes_dark_on_toggle_and_resumes():
    orig_open, orig_iter = cam_mod.open_stream, cam_mod.iter_jpeg_frames
    w = _fast_source_worker()
    workers["pv"] = w
    w.start()
    try:
        assert _wait(lambda: w.frames_seen > 5)
        assert w.latest_raw is not None
        assert w.status()["privacy"] is False

        client.post("/privacy/pv", json={"on": True})
        assert w.status()["privacy"] is True
        # relay slot cleared (the reader's 1s sweep catches a frame that was
        # mid-flight through _on_frame when the toggle landed)
        assert _wait(lambda: w.latest_raw is None)
        assert _wait(lambda: not w.connected)       # reader disconnected from the cam
        seen = w.frames_seen
        time.sleep(0.3)
        assert w.frames_seen == seen                # and no frames flow while private
        assert "pv" not in {k.split(":")[0] for k in tracker._tracks}

        client.post("/privacy/pv", json={"on": False})
        assert _wait(lambda: w.frames_seen > seen)  # reader reconnected on its own
        assert _wait(lambda: w.latest_raw is not None)
        assert w.status()["privacy"] is False
    finally:
        w.stop()
        w.join(timeout=2.0)
        workers.pop("pv", None)
        cam_mod.open_stream, cam_mod.iter_jpeg_frames = orig_open, orig_iter
