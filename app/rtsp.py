"""RTSP/H.264 transport — the reader half for `rtsp://` cameras, companion to mjpeg.py.

Most prosumer/PoE cameras (Reolink, Amcrest, Dahua, Tapo, any ONVIF cam) publish
RTSP/H.264, not HTTP MJPEG (see CAMERA_BRINGUP_PLAN §2 / DECISIONS #3). This module
opens an `rtsp://` stream with OpenCV's FFmpeg backend, pulls decoded frames, and
re-encodes each to JPEG so the rest of the worker (dashboard relay, recorder, the
processor's `decode_jpeg`) is byte-for-byte identical to the MJPEG path — i.e. this is
a *transport swap*, not a pipeline change. `camera._pump` auto-selects this vs the MJPEG
reader by URL scheme (`is_rtsp`).

cv2 (opencv) is REQUIRED for RTSP (it's already the perception decode dependency). If it
is missing we raise; the worker's reconnect loop logs + backs off exactly as it does for
a network error — so on a null/CPU-without-opencv install RTSP cams simply don't stream
while MJPEG cams still do.

Dual-stream: point the reader/detect stream (`Camera.stream_url`) at the cheap SUBSTREAM
and let recorder.py record the full-quality MAIN stream by codec-copy
(`Camera.record_url`). That keeps H.264 decode + YOLO off the 4K main stream (DECISIONS
#1 GPU contention) while recording stays full quality. NB: re-encoding each frame to JPEG
here (then decoding it again in the processor) is the cost of keeping the seam unchanged;
a later optimization can carry ndarrays end-to-end to skip both (DECISIONS).
"""
from __future__ import annotations

import os
import time
from typing import Callable, Iterator
from urllib.parse import urlparse, urlunparse

from .config import cfg
from .perception import encode_jpeg

RTSP_SCHEMES = ("rtsp", "rtsps")


def is_rtsp(url: str | None) -> bool:
    if not url:
        return False
    try:
        return urlparse(url).scheme.lower() in RTSP_SCHEMES
    except Exception:  # noqa: BLE001
        return False


def redact_url(url: str) -> str:
    """`rtsp://user:pass@host:554/path` -> `rtsp://***@host:554/path` (safe for logs)."""
    try:
        p = urlparse(url)
    except Exception:  # noqa: BLE001
        return "<url>"
    if p.username or p.password:
        host = p.hostname or ""
        if p.port:
            host = f"{host}:{p.port}"
        p = p._replace(netloc=f"***@{host}")
    return urlunparse(p)


def _capture_options() -> str:
    # TCP transport is the single most important RTSP-reliability knob: UDP drops/reorders
    # on busy Wi-Fi and corrupts H.264. `stimeout` (microseconds) bounds a dead-stream read
    # so the worker reconnects instead of blocking forever.
    return f"rtsp_transport;{cfg.rtsp_transport}|stimeout;{int(cfg.rtsp_timeout_s * 1_000_000)}"


def iter_rtsp_frames(url: str, should_stop: Callable[[], bool]) -> Iterator[bytes]:
    """Yield JPEG frames from an RTSP stream until `should_stop()` or the stream drops.

    Owns its `cv2.VideoCapture` and releases it on any exit (return, raise, or generator
    close — so `contextlib.closing(...)` in the caller guarantees cleanup on stop). Raises
    (not returns) on cv2-missing / open failure so the caller's reconnect+backoff handles
    RTSP and MJPEG uniformly.
    """
    try:
        import cv2  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"RTSP needs opencv (pip install opencv-python-headless) to pull {redact_url(url)}"
        ) from e

    # Must be set before VideoCapture is constructed; the FFmpeg backend reads it from env.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _capture_options()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the freshest frame, never a backlog
    except Exception:  # noqa: BLE001 — not every backend honors it
        pass
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"could not open RTSP stream {redact_url(url)}")
    try:
        misses = 0
        while not should_stop():
            ok, frame = cap.read()
            if not ok or frame is None:
                # Tolerate a few transient read misses, then surface to the reconnect loop.
                misses += 1
                if misses > cfg.rtsp_max_read_misses:
                    raise RuntimeError(f"RTSP read failed repeatedly {redact_url(url)}")
                time.sleep(0.05)
                continue
            misses = 0
            jpeg = encode_jpeg(frame)
            if jpeg:
                yield jpeg
    finally:
        cap.release()
