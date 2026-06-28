"""MJPEG transport — pull a camera's multipart stream, re-serve it to the dashboard.

The ESP32-CAM serves stock `multipart/x-mixed-replace` MJPEG on :81 and reliably
handles ~1 client (§3.2), so the **vision-service is the only thing that opens the
cam's /stream** and the dashboard views via us, never the cam directly. This module
is the pull half (parse the boundary stream into JPEG frames) and a re-serve helper
(wrap frames back into a multipart response for the dashboard).

Pure stdlib + an injectable byte source, so it's testable without a real camera
(see ../tests/test_mjpeg.py).
"""
from __future__ import annotations

import re
import urllib.request
from typing import Callable, Iterator, Optional

_BOUNDARY_RE = re.compile(rb"boundary=([^\s;]+)", re.I)
_CONTENT_LEN_RE = re.compile(rb"Content-Length:\s*(\d+)", re.I)
_SOI = b"\xff\xd8"  # JPEG start-of-image
_EOI = b"\xff\xd9"  # JPEG end-of-image


def iter_jpeg_frames(read: Callable[[int], bytes], chunk: int = 4096) -> Iterator[bytes]:
    """Yield complete JPEG frames from a multipart MJPEG byte stream.

    `read(n)` returns up to n bytes (e.g. `response.read`). Robust to cameras that
    omit Content-Length: we scan for SOI..EOI markers rather than trusting headers,
    which is what makes the stock esp_camera stream parse reliably.
    """
    buf = bytearray()
    while True:
        data = read(chunk)
        if not data:
            if buf:
                # flush any trailing complete frame
                start = buf.find(_SOI)
                end = buf.find(_EOI, start + 2) if start >= 0 else -1
                if start >= 0 and end >= 0:
                    yield bytes(buf[start:end + 2])
            return
        buf += data
        while True:
            start = buf.find(_SOI)
            if start < 0:
                if len(buf) > chunk:  # nothing useful; don't grow unbounded
                    del buf[:-2]
                break
            end = buf.find(_EOI, start + 2)
            if end < 0:
                if start > 0:
                    del buf[:start]  # drop pre-SOI noise
                break
            yield bytes(buf[start:end + 2])
            del buf[:end + 2]


def open_stream(url: str, timeout: float = 10.0):
    """Open a camera MJPEG stream; returns the urllib response (an object with
    `.read(n)`). Caller iterates with iter_jpeg_frames(resp.read)."""
    req = urllib.request.Request(url, headers={"User-Agent": "home-hub-vision/1"})
    return urllib.request.urlopen(req, timeout=timeout)


def parse_boundary(content_type: Optional[str]) -> Optional[str]:
    if not content_type:
        return None
    m = _BOUNDARY_RE.search(content_type.encode())
    return m.group(1).decode() if m else None


MJPEG_BOUNDARY = "homehubframe"


def multipart_chunk(jpeg: bytes) -> bytes:
    """Wrap one JPEG as a multipart part for re-serving to the dashboard."""
    return (
        f"--{MJPEG_BOUNDARY}\r\n"
        f"Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(jpeg)}\r\n\r\n"
    ).encode() + jpeg + b"\r\n"
