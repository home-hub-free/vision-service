"""CameraWorker — the per-camera pipeline (§4.1/§4.2), one daemon thread per stream.

    pull MJPEG → (capped FPS) decode → person-detect+track → (on new/unmatched track)
    face-detect+embed → gallery.resolve → occupancy.update → publish edges + index
    + drive recorder.

Two FPS budgets keep the GPU honest (§11.1): every frame is relayed for the live view
(cheap), but the perception pipeline runs at `detect_fps`, and face-embed is gated on
NEW/unmatched tracks — we embed a face once per person, not per frame (ByteTrack ids
make that possible). The agent never sees frames; only the digested edges leave here.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

from . import ingest
from .config import cfg
from .hub_client import Camera
from .mjpeg import iter_jpeg_frames, multipart_chunk, open_stream
from .occupancy import Identity, Observation, UNKNOWN
from .perception import crop_jpeg, decode_jpeg, draw_overlay, make_detector, make_face_engine
from .recorder import Recorder
from .state import gallery, index, tracker


class CameraWorker(threading.Thread):
    def __init__(self, cam: Camera) -> None:
        super().__init__(daemon=True, name=f"vision-cam-{cam.id}")
        self.cam = cam
        self.detector = make_detector()
        self.face = make_face_engine()
        self.recorder = Recorder(cam.id, cam.zone, index)
        self._stop = threading.Event()

        # latest frames for the dashboard re-serve.
        self.latest_raw: Optional[bytes] = None
        self.latest_annotated: Optional[bytes] = None
        self.last_frame_ts: float = 0.0
        self.frames_seen: int = 0
        self.connected: bool = False

        # per-track resolved identity cache (embed once per person, §4.1).
        self._ident: Dict[str, Identity] = {}
        self._last_pipeline = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def run(self) -> None:
        self.recorder.start()
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._pump()
                backoff = 1.0
            except Exception as e:  # noqa: BLE001 — reconnect, never die
                self.connected = False
                print(f"[vision] cam {self.cam.id} stream error: {e}; retry in {backoff:.0f}s", flush=True)
                if self._stop.wait(backoff):
                    break
                backoff = min(30.0, backoff * 2)
        self.recorder.stop()

    def stop(self) -> None:
        self._stop.set()

    def _pump(self) -> None:
        url = self.cam.stream_url
        if not url:
            raise RuntimeError("no stream url")
        resp = open_stream(url)
        self.connected = True
        print(f"[vision] cam {self.cam.id} streaming from {url}", flush=True)
        try:
            for jpeg in iter_jpeg_frames(resp.read):
                if self._stop.is_set():
                    break
                self._on_frame(jpeg)
        finally:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass
            self.connected = False

    # ── per-frame ─────────────────────────────────────────────────────────────
    def _on_frame(self, jpeg: bytes) -> None:
        now = time.time()
        self.latest_raw = jpeg
        self.last_frame_ts = now
        self.frames_seen += 1
        self.recorder.write_frame(jpeg)
        self.recorder.tick()

        # Throttle the heavy pipeline to detect_fps (relay stays full-rate).
        if cfg.detect_fps <= 0 or (now - self._last_pipeline) < (1.0 / cfg.detect_fps):
            return
        self._last_pipeline = now
        self._run_pipeline(jpeg, now)

    def _run_pipeline(self, jpeg: bytes, now: float) -> None:
        frame = decode_jpeg(jpeg)
        if frame is None:
            return  # null build (no cv2): presence-less M0 relay/record only
        tracks = self.detector.detect_and_track(frame)
        labels: Dict[str, str] = {}
        observations: List[Observation] = []
        for t in tracks:
            ident = self._ident.get(t.track_id)
            # Embed + resolve once per track, then re-try only while still unknown.
            if ident is None or ident.cls == "unknown":
                emb = self.face.embed(frame, t.bbox)
                if emb is not None:
                    # Capture a face/person crop so every default-id'd person carries a
                    # thumbnail the dashboard can show (admin labels from the face).
                    ident = gallery.resolve(emb, crop_jpeg(frame, t.bbox))
                else:
                    ident = ident or UNKNOWN
                self._ident[t.track_id] = ident
            observations.append(Observation(track_id=t.track_id, identity=ident))
            # Live overlay label: real name if known, else the default "Person N" for a
            # guest cluster (every detected person is labelled by default), else "person".
            if ident.name:
                labels[t.track_id] = ident.name
            elif ident.cls == "guest" and ident.id:
                labels[t.track_id] = gallery.default_label(ident.id)
            else:
                labels[t.track_id] = "person"

        edges = tracker.update(self.cam.id, self.cam.zone, observations, now)
        count = len(tracker.snapshot(self.cam.zone).get(self.cam.zone, []))
        self.recorder.on_presence(count > 0)
        for edge in edges:
            ingest.publish_edge(edge, count)
            index.record_event(edge)

        # Annotated frame for the dashboard's "who is here" view (§6).
        if tracks:
            self.latest_annotated = draw_overlay(frame, tracks, labels) or self.latest_annotated
        else:
            self.latest_annotated = None
        # Forget identities of tracks that aged out so the cache can't grow unbounded.
        self._prune_idents({t.track_id for t in tracks})

    def _prune_idents(self, live: set) -> None:
        if len(self._ident) > 256:
            self._ident = {k: v for k, v in self._ident.items() if k in live}

    # ── dashboard re-serve (§3.2 — we are the ONLY client of the cam) ─────────
    def mjpeg_generator(self, annotated: bool = True):
        """Yield multipart MJPEG of the latest frame (annotated when available). The
        dashboard views THIS, never the camera directly (a 2nd client stalls the cam)."""
        last_sent = 0.0
        while not self._stop.is_set():
            frame = (self.latest_annotated if annotated else None) or self.latest_raw
            if frame is not None and self.last_frame_ts != last_sent:
                last_sent = self.last_frame_ts
                yield multipart_chunk(frame)
            time.sleep(0.05)

    def status(self) -> dict:
        return {
            "id": self.cam.id, "zone": self.cam.zone, "ip": self.cam.ip,
            "connected": self.connected, "frames_seen": self.frames_seen,
            "last_frame_age_s": round(time.time() - self.last_frame_ts, 1) if self.last_frame_ts else None,
            "detector": self.detector.backend, "face": self.face.backend,
            "rec_mode": self.recorder.mode,
        }
