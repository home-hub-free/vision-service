"""CameraWorker — the per-camera pipeline (§4.1/§4.2).

    pull MJPEG → (capped FPS) decode → person-detect+track → (on new/unmatched track)
    face-detect+embed → gallery.resolve → occupancy.update → publish edges + index
    + drive recorder.

TWO THREADS, decoupled (this matters — see below):
  * READER (`run`): drains the camera's MJPEG stream CONTINUOUSLY, full-rate, doing only
    cheap work (relay slot, frames_seen, recorder feed). It hands the freshest frame to
    the processor and immediately reads the next.
  * PROCESSOR (`_process_loop`): runs the heavy perception pipeline on the LATEST frame,
    paced at `detect_fps`, dropping stale frames.

Why split them: the ESP32-CAM's MJPEG server is single-consumer and needs a consumer
that drains continuously. If the reader pauses to run inference (YOLO + SCRFD/ArcFace,
hundreds of ms), the camera's send blocks, the sensor pipeline stalls, the read times
out (>10s), and the worker drops into a multi-second reconnect backoff — which
collapsed the effective frame rate to ~0.25 fps and broke ByteTrack continuity /
occupancy. Keeping the reader free of inference keeps the camera streaming.

Two FPS budgets keep the GPU honest (§11.1): every frame is relayed for the live view
(cheap), but the perception pipeline runs at `detect_fps`, and face-embed is gated on
NEW/unmatched tracks — we embed a face once per person, not per frame (ByteTrack ids
make that possible). The agent never sees frames; only the digested edges leave here.
"""
from __future__ import annotations

import contextlib
import threading
import time
from typing import Dict, List, Optional

from . import hub_push, ingest
from .config import cfg
from .hub_client import Camera
from .mjpeg import iter_jpeg_frames, multipart_chunk, open_stream
from .rtsp import is_rtsp, iter_rtsp_frames, redact_url
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
        self.recorder = Recorder(cam.id, cam.zone, index, record_url=cam.record_url)
        # NB: not `_stop` — threading.Thread._stop is an internal METHOD it calls when a
        # thread finishes/joins; shadowing it with an Event breaks join() ("'Event' object
        # is not callable"). Same trap as _ident/_idents above.
        self._stop_evt = threading.Event()

        # latest frames for the dashboard re-serve.
        self.latest_raw: Optional[bytes] = None
        self.latest_annotated: Optional[bytes] = None
        self.last_frame_ts: float = 0.0
        self.frames_seen: int = 0
        self.connected: bool = False

        # Reader→processor handoff: a single latest-frame slot (we always process the
        # freshest, never a backlog). The reader fills it; the processor drains it.
        self._frame_cv = threading.Condition()
        self._pending_frame: Optional[bytes] = None
        self._proc_thread: Optional[threading.Thread] = None
        self._present = False  # set by the processor, read by the reader to gate recording

        # per-track resolved identity cache (embed once per person, §4.1). Touched ONLY
        # by the processor thread. NB: must NOT be named `_ident` — this class subclasses
        # threading.Thread, which uses `self._ident` for the thread id and overwrites it
        # on .start() (an int), breaking len() in _prune_idents. Use `_idents`.
        self._idents: Dict[str, Identity] = {}
        self._last_pipeline = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def run(self) -> None:
        """READER thread: drain the camera continuously; never run inference here."""
        self.recorder.start()
        self._proc_thread = threading.Thread(
            target=self._process_loop, name=f"vision-proc-{self.cam.id}", daemon=True)
        self._proc_thread.start()
        backoff = 1.0
        while not self._stop_evt.is_set():
            try:
                self._pump()
                backoff = 1.0
            except Exception as e:  # noqa: BLE001 — reconnect, never die
                self.connected = False
                # repr (not str) so a code bug (e.g. TypeError) is distinguishable from a
                # transient network error at a glance — the difference matters for triage.
                print(f"[vision] cam {self.cam.id} stream error: {e!r}; retry in {backoff:.0f}s", flush=True)
                if self._stop_evt.wait(backoff):
                    break
                # Cap low: a transient stall shouldn't blacken the feed for 30s. With the
                # reader decoupled, stalls should be rare anyway.
                backoff = min(5.0, backoff * 2)
        # Wake + join the processor so we shut down cleanly.
        with self._frame_cv:
            self._frame_cv.notify_all()
        if self._proc_thread is not None:
            self._proc_thread.join(timeout=2.0)
        self.recorder.stop()

    def stop(self) -> None:
        self._stop_evt.set()
        with self._frame_cv:  # unblock the processor's wait()
            self._frame_cv.notify_all()

    def _pump(self) -> None:
        url = self.cam.stream_url
        if not url:
            raise RuntimeError("no stream url")
        # Auto-select transport by URL scheme: rtsp:// (Reolink/Amcrest/Dahua/Tapo/ONVIF)
        # decodes via OpenCV's FFmpeg backend; everything else is HTTP-MJPEG (ESP32-CAM).
        if is_rtsp(url):
            self.connected = True
            print(f"[vision] cam {self.cam.id} streaming (rtsp) from {redact_url(url)}", flush=True)
            try:
                # closing() guarantees cap.release() when the reader stops/reconnects.
                with contextlib.closing(iter_rtsp_frames(url, self._stop_evt.is_set)) as frames:
                    for jpeg in frames:
                        if self._stop_evt.is_set():
                            break
                        self._on_frame(jpeg)
            finally:
                self.connected = False
            return
        resp = open_stream(url)
        self.connected = True
        print(f"[vision] cam {self.cam.id} streaming (mjpeg) from {url}", flush=True)
        try:
            for jpeg in iter_jpeg_frames(resp.read):
                if self._stop_evt.is_set():
                    break
                self._on_frame(jpeg)
        finally:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass
            self.connected = False

    # ── reader: per-frame, CHEAP only (no inference — see module docstring) ─────
    def _on_frame(self, jpeg: bytes) -> None:
        now = time.time()
        self.latest_raw = jpeg          # full-rate relay slot for the dashboard
        self.last_frame_ts = now
        self.frames_seen += 1
        # Recording is driven from the reader so it sees every frame at full rate; the
        # presence gate (`_present`) is computed by the processor. write_frame just feeds
        # an already-running ffmpeg (or the pre-roll ring) and never blocks meaningfully.
        self.recorder.write_frame(jpeg)
        self.recorder.on_presence(self._present)
        self.recorder.tick()
        # Hand the freshest frame to the processor, overwriting any unprocessed one — we
        # never want a backlog; perception always works on the latest frame.
        with self._frame_cv:
            self._pending_frame = jpeg
            self._frame_cv.notify()

    # ── processor: the heavy pipeline on its OWN thread, paced at detect_fps ─────
    def _process_loop(self) -> None:
        while not self._stop_evt.is_set():
            with self._frame_cv:
                while self._pending_frame is None and not self._stop_evt.is_set():
                    self._frame_cv.wait(timeout=1.0)
                jpeg = self._pending_frame
                self._pending_frame = None
            if jpeg is None or self._stop_evt.is_set():
                continue
            # Pace the heavy pipeline at detect_fps (the reader/relay stays full-rate).
            now = time.time()
            if cfg.detect_fps > 0:
                wait = (1.0 / cfg.detect_fps) - (now - self._last_pipeline)
                if wait > 0 and self._stop_evt.wait(wait):
                    break
                with self._frame_cv:  # grab a fresher frame if one arrived while pacing
                    if self._pending_frame is not None:
                        jpeg = self._pending_frame
                        self._pending_frame = None
                now = time.time()
            self._last_pipeline = now
            try:
                self._run_pipeline(jpeg, now)
            except Exception as e:  # noqa: BLE001 — a bad frame must never kill perception
                print(f"[vision] cam {self.cam.id} pipeline error: {e!r}", flush=True)

    def _run_pipeline(self, jpeg: bytes, now: float) -> None:
        frame = decode_jpeg(jpeg)
        if frame is None:
            return  # null build (no cv2): presence-less M0 relay/record only
        tracks = self.detector.detect_and_track(frame)
        labels: Dict[str, str] = {}
        observations: List[Observation] = []
        for t in tracks:
            ident = self._idents.get(t.track_id)
            # Embed + resolve once per track, then re-try only while still unknown.
            if ident is None or ident.cls == "unknown":
                emb = self.face.embed(frame, t.bbox)
                if emb is not None:
                    # Capture a face/person crop so every default-id'd person carries a
                    # thumbnail the dashboard can show (admin labels from the face).
                    ident = gallery.resolve(emb, crop_jpeg(frame, t.bbox))
                else:
                    ident = ident or UNKNOWN
                self._idents[t.track_id] = ident
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
        zone_people = tracker.snapshot(self.cam.zone).get(self.cam.zone, [])
        count = len(zone_people)
        self._present = count > 0  # reader reads this to drive gated recording
        for edge in edges:
            ingest.publish_edge(edge, count)
            index.record_event(edge)
        # On any salient change, push the per-zone occupancy+identity digest to the hub so it
        # FUSES it into the `rooms` world-model the agent reads (§3.1). Edges already debounce —
        # so this fires once per arrival/leave, not per frame. Best-effort; never throws.
        if edges:
            hub_push.push_room(self.cam.zone, zone_people)

        # Annotated frame for the dashboard's "who is here" view (§6).
        if tracks:
            self.latest_annotated = draw_overlay(frame, tracks, labels) or self.latest_annotated
        else:
            self.latest_annotated = None
        # Forget identities of tracks that aged out so the cache can't grow unbounded.
        self._prune_idents({t.track_id for t in tracks})

    def _prune_idents(self, live: set) -> None:
        if len(self._idents) > 256:
            self._idents = {k: v for k, v in self._idents.items() if k in live}

    # ── dashboard re-serve (§3.2 — we are the ONLY client of the cam) ─────────
    def mjpeg_generator(self, annotated: bool = True):
        """Yield multipart MJPEG of the latest frame (annotated when available). The
        dashboard views THIS, never the camera directly (a 2nd client stalls the cam)."""
        last_sent = 0.0
        while not self._stop_evt.is_set():
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
