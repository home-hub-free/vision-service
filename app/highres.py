"""On-demand high-res frame sampling — the far-face accuracy lever.

Detect/track/embed runs on the cheap SUBSTREAM (720p on the Tapo/MC200 fleet), where
a face at 3–4 m is ~40–70px across — inside ArcFace's noisy zone (it normalizes every
face to 112×112, so accuracy tops out once a face is ~110px wide and degrades fast
below ~60px; that noise is what seeded 146 guest clusters for 3 people). The main
stream is 1.5–2× the substream, but decoding it continuously just to ID an
occasionally-far face is a permanent CPU tax paid mostly on empty rooms.

So: sample it. When the camera worker finds a face that is present-but-SMALL on the
substream, it asks this sampler for ONE full-res frame and re-embeds from that. Cost
profile is near-zero continuous + a burst per event (embedding only happens on new
tracks and periodic re-verifies, and the sampler is rate-limited per camera). The RTSP
grab measured ~2.1s per frame on the live C110s (2304×1296, verified 2026-07-06 with
the reader+recorder sessions BOTH held — no session-cap problem) and runs on the
processor thread, so detection pauses ~2s during a grab; the reader keeps draining, so
the camera never stalls. Make the grab async if that pause ever matters.

Source order, probed once and cached:
  1. a declared snapshot URL (`Camera.snapshot_url`),
  2. the ONVIF Media GetSnapshotUri (a single JPEG over HTTP — no extra RTSP session;
     the C110 faults on it, verified 2026-07-06, so this mostly serves future cams),
  3. a one-frame RTSP grab of the main stream (`Camera.record_url`) — works on the
     whole IP-cam fleet, but is a THIRD session next to the reader (substream) and
     recorder (main): some firmware caps concurrent sessions, which shows up here as
     open failures.

Failure posture: `degraded` after 3 consecutive misses — the worker then passes
substream embeddings through unchanged (never starves recognition on a broken
source) and retries at 10× the interval so a recovered camera self-heals.
"""
from __future__ import annotations

import base64
import os
import time
import urllib.request
from typing import Callable, Optional
from urllib.parse import urlparse

from .config import cfg
from .hub_client import Camera
from .rtsp import _capture_options, redact_url

_UNPROBED = object()


class HighResSampler:
    def __init__(self, cam: Camera,
                 get_onvif: Optional[Callable[[], object]] = None) -> None:
        self.cam = cam
        self._get_onvif = get_onvif  # lazy: the supervisor attaches ONVIF after start
        self._onvif_uri = _UNPROBED  # None once probed-and-unsupported
        self._last_attempt = 0.0
        self._fails = 0

    @property
    def degraded(self) -> bool:
        return self._fails >= 3

    def status(self) -> dict:
        return {"degraded": self.degraded, "fails": self._fails}

    def get_frame(self):
        """One full-res BGR frame, or None (rate-limited / source down). Never raises.
        Degraded sources keep retrying at 10× the interval so they self-heal."""
        now = time.monotonic()
        interval = cfg.highres_interval_s * (10 if self.degraded else 1)
        if now - self._last_attempt < interval:
            return None
        self._last_attempt = now
        try:
            frame = self._fetch()
        except Exception as e:  # noqa: BLE001 — sampling must never break the pipeline
            print(f"[vision] cam {self.cam.id} highres fetch error: {e!r}", flush=True)
            frame = None
        if frame is None:
            self._fails += 1
            if self._fails == 3:
                print(f"[vision] cam {self.cam.id} highres sampler degraded "
                      f"(3 misses) — substream embeddings pass through", flush=True)
        else:
            self._fails = 0
        return frame

    # ── source resolution ─────────────────────────────────────────────────────
    def _fetch(self):
        uri = self.cam.snapshot_url or self._probe_onvif_uri()
        if uri:
            frame = self._fetch_snapshot(uri)
            if frame is not None:
                return frame
        if self.cam.record_url:
            return self._grab_rtsp(self.cam.record_url)
        return None

    def _probe_onvif_uri(self) -> Optional[str]:
        if self._onvif_uri is not _UNPROBED:
            return self._onvif_uri
        client = self._get_onvif() if self._get_onvif else None
        if client is None or not hasattr(client, "snapshot_uri"):
            return None  # not attached yet — stay unprobed, try again next time
        try:
            self._onvif_uri = client.snapshot_uri()
        except Exception:  # noqa: BLE001
            self._onvif_uri = None
        if self._onvif_uri:
            print(f"[vision] cam {self.cam.id} highres via ONVIF snapshot "
                  f"{redact_url(self._onvif_uri)}", flush=True)
        return self._onvif_uri

    def _creds(self):
        """user/pass for the snapshot HTTP GET — reuse the RTSP URL's userinfo (the
        one credential store, per the ONVIF-seam decision)."""
        for url in (self.cam.record_url, self.cam.stream_url):
            if not url:
                continue
            p = urlparse(url)
            if p.username:
                return p.username, p.password or ""
        return None

    def _fetch_snapshot(self, uri: str):
        from .perception import decode_jpeg
        creds = self._creds()
        for auth in (None, "basic"):
            req = urllib.request.Request(uri)
            if auth == "basic":
                if not creds:
                    break
                token = base64.b64encode(f"{creds[0]}:{creds[1]}".encode()).decode()
                req.add_header("Authorization", f"Basic {token}")
            try:
                with urllib.request.urlopen(req, timeout=cfg.rtsp_timeout_s) as r:
                    data = r.read()
            except Exception:  # noqa: BLE001 — 401 falls through to basic, else give up
                continue
            frame = decode_jpeg(data)
            if frame is not None:
                return frame
        return None

    def _grab_rtsp(self, url: str):
        """Open the main stream, decode one frame, release. ~2.1s on the live C110s.
        The session exists only for the grab — a third concurrent session next to the
        reader + recorder, which the C110s accept (verified live); a firmware session
        cap elsewhere surfaces as an open failure and the degraded posture handles it."""
        try:
            import cv2  # type: ignore
        except Exception:  # noqa: BLE001
            return None
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _capture_options()
        timeout_ms = int(cfg.rtsp_timeout_s * 1000)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms,
        ])
        try:
            if not cap.isOpened():
                return None
            for _ in range(3):  # the first read after open can be a partial frame
                ok, frame = cap.read()
                if ok and frame is not None:
                    return frame
            return None
        finally:
            cap.release()


def make_highres_sampler(cam: Camera,
                         get_onvif: Optional[Callable[[], object]] = None
                         ) -> Optional[HighResSampler]:
    """A sampler only for cameras that HAVE a higher-res source than their detect
    stream — keyed on `record_url` (the dual-stream IP cams), exactly like record
    scope. Satellites/ESP32 cams have one low-res stream; upscaling it teaches
    nothing."""
    if not cfg.highres_enabled or not cam.record_url:
        return None
    return HighResSampler(cam, get_onvif=get_onvif)
