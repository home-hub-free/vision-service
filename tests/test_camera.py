"""CameraWorker reader/processor decoupling.

The camera reader must NEVER be throttled by perception: the ESP32-CAM's MJPEG server is
single-consumer and stalls (then times out → reconnect backoff) if the reader pauses to
run inference. So a slow — even stuck — pipeline must not stop the reader from draining
the stream. We prove that with a fake fast frame source and a pipeline that blocks.
"""
import threading
import time

import app.camera as cam_mod
from app.camera import CameraWorker
from app.hub_client import Camera

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
