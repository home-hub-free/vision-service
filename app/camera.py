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
from .highres import make_highres_sampler
from .hub_client import Camera
from .mjpeg import iter_jpeg_frames, multipart_chunk, open_stream
from .rtsp import is_rtsp, iter_rtsp_frames, redact_url
from .occupancy import Identity, Observation, UNKNOWN
from .perception import (DetectedTrack, assign_faces_to_tracks, classify_posture,
                         crop_jpeg, decode_jpeg, draw_overlay, face_crop_jpeg,
                         make_detector, make_face_engine, make_pose_engine,
                         match_poses_to_tracks)
from .recorder import Recorder
from .state import gallery, index, privacy, tracker


class CameraWorker(threading.Thread):
    def __init__(self, cam: Camera) -> None:
        super().__init__(daemon=True, name=f"vision-cam-{cam.id}")
        self.cam = cam
        self.detector = make_detector()
        self.face = make_face_engine()
        self.pose = make_pose_engine()  # T1 (§3): null unless VISION_POSE_BACKEND is set
        # Record scope (DECISIONS): a camera records ONLY when it has a full-quality
        # RTSP MAIN stream (`record_url`) — the IP-cam fleet (Tapo/MC200). MJPEG-only
        # cams (the ESP32-CAM entrance cam + the face-ID desk cams on satellites) get a
        # hard-off recorder so we never spend disk archiving a face-ID sensor. `off` is
        # a no-op for both intake paths (write_frame/on_presence/tick all early-return).
        records = bool(cam.record_url)
        self.recorder = Recorder(cam.id, cam.zone, index,
                                 mode=(cfg.rec_mode_default if records else "off"),
                                 record_url=cam.record_url)
        # On-demand high-res sampling (highres.py): dual-stream IP cams only. The
        # ONVIF getter is lazy — the supervisor attaches `_onvif` after start.
        self.highres = make_highres_sampler(
            cam, get_onvif=lambda: getattr(self, "_onvif", None))
        # NB: not `_stop` — threading.Thread._stop is an internal METHOD it calls when a
        # thread finishes/joins; shadowing it with an Event breaks join() ("'Event' object
        # is not callable"). Same trap as _ident/_idents above.
        self._stop_evt = threading.Event()

        # Privacy mode (app/privacy.py): while private the reader never connects, so
        # nothing downstream (stream/record/perception/occupancy) can see the camera.
        # `_privacy_applied` makes the teardown idempotent (reader loop + the route's
        # immediate `pause_for_privacy` may both run it); `_rec_started` defers the
        # recorder's first start out of run()'s head so a camera that BOOTS private
        # never opens ffmpeg at all.
        self._privacy_applied = False
        self._rec_started = False

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

        # ONVIF motion pre-gate state (CAMERA_ONVIF_CONTROL_PLAN §3), written by the
        # camera's EventPuller (supervisor-owned), read by the processor loop. When
        # `detect_on_motion` is on AND events are attached, the heavy pipeline only
        # runs while the camera's own motion detector is active (+ linger); if the
        # event subscription drops, `events_attached` falls false and the gate fails
        # OPEN (back to always-on detection — never blind because a subscription died).
        self.events_attached = False
        self.motion_active = False
        self.last_motion_ts = 0.0

        # per-track resolved identity cache (embed once per person, §4.1). Touched ONLY
        # by the processor thread. NB: must NOT be named `_ident` — this class subclasses
        # threading.Thread, which uses `self._ident` for the thread id and overwrites it
        # on .start() (an int), breaking len() in _prune_idents. Use `_idents`.
        self._idents: Dict[str, Identity] = {}
        # per-track last identity-check timestamp (drives the periodic household
        # re-verify that heals tracker id-switches; pruned alongside _idents).
        self._ident_ts: Dict[str, float] = {}
        self._last_pipeline = 0.0
        # T0/T1 digest cadence: pose sub-sampling counter (§3 cost-gate lever) and the
        # occupied-zone heartbeat push clock (§2 — activity changes with no salient
        # edge still reach the hub within digest_heartbeat_s, never per-frame).
        self._pose_counter = 0
        self._last_digest_push = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def run(self) -> None:
        """READER thread: drain the camera continuously; never run inference here."""
        self._proc_thread = threading.Thread(
            target=self._process_loop, name=f"vision-proc-{self.cam.id}", daemon=True)
        self._proc_thread.start()
        backoff = 1.0
        while not self._stop_evt.is_set():
            if self.is_private():
                self.pause_for_privacy()  # idempotent — usually already applied
                if self._stop_evt.wait(1.0):
                    break
                continue
            self._resume_from_privacy()  # also the recorder's FIRST start
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

    # ── privacy mode ──────────────────────────────────────────────────────────
    def is_private(self) -> bool:
        return privacy.is_private(self.cam.id)

    def pause_for_privacy(self) -> None:
        """Go dark NOW (also called from the toggle route's thread, so the recorder
        stops writing before the HTTP response returns — the reader converges within
        a frame). Closes the recorder (passthrough ffmpeg included), drops the
        pre-roll ring and the relay frames, and silently withdraws this camera's
        occupancy so /occupancy can't show people frozen at the pause instant."""
        # Relay slots clear OUTSIDE the idempotence gate: a frame mid-flight through
        # _on_frame when the toggle lands can repopulate them just after the teardown —
        # the reader loop re-calls this every second while private, sweeping it out.
        self.latest_raw = None
        self.latest_annotated = None
        self.last_frame_ts = 0.0
        if self._privacy_applied:
            return
        self._privacy_applied = True
        self.recorder.stop()
        self.recorder.clear_ring()
        self._present = False
        with self._frame_cv:  # a queued frame must not be processed post-toggle
            self._pending_frame = None
        tracker.drop_camera(self.cam.id)
        print(f"[vision] cam {self.cam.id} privacy ON — stream, recording and "
              f"perception paused", flush=True)

    def _resume_from_privacy(self) -> None:
        """(Re)arm the recorder — the first start after boot, and every resume."""
        if self._rec_started and not self._privacy_applied:
            return
        self._rec_started = True
        self.recorder.start()
        if self._privacy_applied:
            self._privacy_applied = False
            print(f"[vision] cam {self.cam.id} privacy OFF — resuming", flush=True)

    def _pump(self) -> None:
        url = self.cam.stream_url
        if not url:
            raise RuntimeError("no stream url")
        # Reader bail-out: shutdown OR privacy — both must break the drain loop fast
        # (privacy takes effect within one frame, not one reconnect).
        halt = lambda: self._stop_evt.is_set() or self.is_private()  # noqa: E731
        # Auto-select transport by URL scheme: rtsp:// (Reolink/Amcrest/Dahua/Tapo/ONVIF)
        # decodes via OpenCV's FFmpeg backend; everything else is HTTP-MJPEG (ESP32-CAM).
        if is_rtsp(url):
            self.connected = True
            print(f"[vision] cam {self.cam.id} streaming (rtsp) from {redact_url(url)}", flush=True)
            try:
                # closing() guarantees cap.release() when the reader stops/reconnects.
                with contextlib.closing(iter_rtsp_frames(url, halt)) as frames:
                    for jpeg in frames:
                        if halt():
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
                if halt():
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
        if self.is_private():
            return  # belt-and-braces: the pump loop is about to break anyway
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
            if jpeg is None or self._stop_evt.is_set() or self.is_private():
                continue  # a frame queued just before a privacy toggle is dropped
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
            if not self._motion_gate_open(now):
                continue  # empty scene per the camera's own motion detector — skip inference
            try:
                self._run_pipeline(jpeg, now)
            except Exception as e:  # noqa: BLE001 — a bad frame must never kill perception
                print(f"[vision] cam {self.cam.id} pipeline error: {e!r}", flush=True)

    def _run_pipeline(self, jpeg: bytes, now: float) -> None:
        frame = decode_jpeg(jpeg)
        if frame is None:
            return  # null build (no cv2): presence-less M0 relay/record only
        tracks = self.detector.detect_and_track(frame)
        frame_w = int(frame.shape[1])
        # T1 pose pass (§3): only on frames that already have person tracks, same
        # motion gate (we're past it), sub-sampled by pose_every_n (posture changes
        # slowly — the cost-gate fallback runs pose every ~3rd detect frame).
        postures: Dict[str, str] = {}
        # Satellite/ESP32 cams (not context_capable) skip pose entirely: their frames
        # can't support full-body inference, and the CPU is better spent elsewhere.
        if tracks and self.cam.context_capable and getattr(self.pose, "backend", "null") != "null":
            self._pose_counter += 1
            if self._pose_counter % max(1, cfg.pose_every_n) == 0:
                matched = match_poses_to_tracks(self.pose.detect(frame), tracks)
                for tid, kps in matched.items():
                    tbox = next(t.bbox for t in tracks if t.track_id == tid)
                    posture = classify_posture(kps, tbox)
                    if posture:
                        postures[tid] = posture
        # Which tracks need a fresh embedding this frame: `resolve` = no identity yet
        # (embed once per track, then re-try only while still unknown); `recheck` = the
        # periodic household re-verify that heals tracker id-switches.
        need: Dict[str, str] = {}
        for t in tracks:
            ident = self._idents.get(t.track_id)
            if ident is None or ident.cls == "unknown":
                need[t.track_id] = "resolve"
            elif (cfg.face_reverify_s > 0 and ident.cls == "household"
                  and now - self._ident_ts.get(t.track_id, 0.0) >= cfg.face_reverify_s):
                need[t.track_id] = "recheck"
        embeds = self._embed_tracks(frame, tracks, need)
        idents: Dict[str, Identity] = {}
        for t in tracks:
            ident = self._idents.get(t.track_id)
            mode = need.get(t.track_id)
            if mode == "resolve":
                emb, thumb, thumb_box = embeds.get(t.track_id, (None, None, None))
                if emb is not None:
                    try:
                        ident = gallery.resolve(
                            emb,
                            thumb or crop_jpeg(frame, t.bbox, max_dim=cfg.capture_crop_px),
                            thumb_box=thumb_box)
                    except Exception as e:  # noqa: BLE001 — a gallery failure on ONE face
                        # must not abort the frame (it would also drop occupancy, edges
                        # and every other track — a DB hiccup blinded whole frames once).
                        print(f"[vision] cam {self.cam.id} resolve error: {e!r}", flush=True)
                        ident = ident or UNKNOWN
                else:
                    ident = ident or UNKNOWN
                self._idents[t.track_id] = ident
                self._ident_ts[t.track_id] = now
            elif mode == "recheck":
                # Heal tracker id-switches: when two people cross, ByteTrack can swap
                # their track ids — each then wears the OTHER's cached label for the
                # track's whole life. Periodically re-embed and, only on a decisive
                # fresh match (recheck is side-effect-free and margin-gated), replace
                # the cached identity. An indecisive frame keeps the cached label.
                self._ident_ts[t.track_id] = now  # pace re-checks even when indecisive
                emb, _, _ = embeds.get(t.track_id, (None, None, None))
                fresh = None
                if emb is not None:
                    try:
                        fresh = gallery.recheck(emb)
                    except Exception as e:  # noqa: BLE001 — same posture as resolve
                        print(f"[vision] cam {self.cam.id} recheck error: {e!r}", flush=True)
                if fresh is not None and fresh.id != ident.id:
                    print(f"[vision] cam {self.cam.id} track {t.track_id} relabelled "
                          f"{ident.name or ident.id} -> {fresh.name or fresh.id} "
                          f"(id-switch heal)", flush=True)
                    ident = fresh
                    self._idents[t.track_id] = fresh
            idents[t.track_id] = ident
        self._dedupe_household(idents)
        labels: Dict[str, str] = {}
        observations: List[Observation] = []
        for t in tracks:
            ident = idents[t.track_id]
            observations.append(Observation(track_id=t.track_id, identity=ident,
                                            bbox=t.bbox, frame_w=frame_w,
                                            posture=postures.get(t.track_id),
                                            context=self.cam.context_capable))
            # Live overlay label: real name if known, else the default "Person N" for a
            # guest cluster (every detected person is labelled by default), else "person".
            if ident.name:
                labels[t.track_id] = ident.name
            elif ident.cls == "guest" and ident.id:
                labels[t.track_id] = gallery.default_label(ident.id)
            else:
                labels[t.track_id] = "person"

        edges = tracker.update(self.cam.id, self.cam.zone, observations, now)
        zone_people = tracker.snapshot(self.cam.zone, now=now).get(self.cam.zone, [])
        count = len(zone_people)
        self._present = count > 0  # reader reads this to drive gated recording
        for edge in edges:
            ingest.publish_edge(edge, count)
            index.record_event(edge)
        # On any salient change, push the per-zone occupancy+identity digest to the hub so it
        # FUSES it into the `rooms` world-model the agent reads (§3.1). Edges already debounce —
        # so this fires once per arrival/leave, not per frame. Best-effort; never throws.
        # T0 (§2): while the zone stays occupied with no edges, a heartbeat re-push keeps the
        # hub's dwell/activity fresh (and its vision TTL alive) — still never per-frame.
        if edges or (zone_people and now - self._last_digest_push >= cfg.digest_heartbeat_s):
            hub_push.push_room(self.cam.zone, zone_people)
            self._last_digest_push = now

        # Annotated frame for the dashboard's "who is here" view (§6).
        if tracks:
            self.latest_annotated = draw_overlay(frame, tracks, labels) or self.latest_annotated
        else:
            self.latest_annotated = None
        # Forget identities of tracks that aged out so the cache can't grow unbounded.
        self._prune_idents({t.track_id for t in tracks})

    def _motion_gate_open(self, now: float) -> bool:
        """The opt-in ONVIF-motion YOLO pre-gate (plan §3). Open unless the knob is
        on AND a live event subscription says the scene is empty. Linger keeps the
        pipeline running long enough after the last motion to see people leave
        (keep it > leave_grace_s if you tighten it)."""
        if not cfg.detect_on_motion or not self.events_attached:
            return True
        return self.motion_active or (now - self.last_motion_ts) < cfg.motion_linger_s

    def _embed_tracks(self, frame, tracks, wanted) -> Dict[str, tuple]:
        """(emb, thumb, thumb_box) per wanted track id — one substream pass, then an
        optional HIGH-RES upgrade for faces the substream found but too SMALL to embed
        cleanly (under highres_min_face_px — ArcFace tops out at 112px, and far-face
        noise is what seeded 146 clusters for 3 people). The sampler fetches ONE
        full-res frame shared by every small face in the pass; track boxes scale
        linearly between the streams. If the sampler is healthy but momentarily
        rate-limited, small-face RESOLVES are held back a pass (the track retries in
        ~200 ms and catches the grab) rather than seeding a noisy guest cluster; a
        degraded sampler passes substream embeddings through unchanged."""
        out = self._embed_pass(frame, tracks, wanted)
        min_px = cfg.highres_min_face_px
        if self.highres is not None and min_px > 0:
            # Upgrade candidates: a face that embedded but SMALL, and equally a face
            # the engine FOUND but abstained on (emb None, px known) — det/sharpness
            # improve on the main-stream frame just like size does (measured live:
            # det 0.49 sub → 0.77 main). The faces most in need of rescue are
            # exactly the ones the quality gate held back.
            small = [tid for tid, (emb, _t, _tb, px) in out.items()
                     if 0 < px < min_px or (emb is None and px > 0)]
            if small:
                hi = self.highres.get_frame()
                if hi is not None:
                    sx = hi.shape[1] / frame.shape[1]
                    sy = hi.shape[0] / frame.shape[0]
                    hi_tracks = [DetectedTrack(track_id=t.track_id,
                                               bbox=(int(t.bbox[0] * sx), int(t.bbox[1] * sy),
                                                     int(t.bbox[2] * sx), int(t.bbox[3] * sy)))
                                 for t in tracks]
                    hi_out = self._embed_pass(hi, hi_tracks, {tid: wanted[tid] for tid in small})
                    for tid in small:
                        emb, thumb, thumb_box, px = hi_out.get(tid, (None, None, None, 0))
                        if emb is not None:  # no face on the hi frame → keep the lo result
                            out[tid] = (emb, thumb, thumb_box, px)
                elif not self.highres.degraded:
                    for tid in small:
                        if wanted.get(tid) == "resolve":
                            out[tid] = (None, None, None, 0)
        # Identity size floor, applied AFTER the high-res rescue had its shot: a face
        # still under face_min_px is below what ArcFace can embed meaningfully (the
        # noise that smeared member centroids together) — the track keeps counting for
        # occupancy, but identity abstains rather than guessing. Entries that stayed
        # abstained (emb None) collapse to the plain "no face" shape.
        floor = max(0, cfg.face_min_px)
        return {tid: ((None, None, None)
                      if (v[0] is None or (floor and 0 < v[3] < floor))
                      else v[:3])
                for tid, v in out.items()}

    def _embed_pass(self, frame, tracks, wanted) -> Dict[str, tuple]:
        """(emb, thumb, thumb_box, face_px) per wanted track id, on ONE frame. With one
        person in frame the per-crop path (`_embed_track`) stands; with SEVERAL, faces
        are detected once at frame level and assigned to tracks EXCLUSIVELY by
        containment (`assign_faces_to_tracks`) — the per-crop "largest face in my box"
        rule embeds the other person's face when person boxes overlap, which is how two
        people in one room end up wearing each other's names. `face_px` is the detected
        face's longest side (0 when unknown) — the high-res upgrade trigger."""
        out: Dict[str, tuple] = {}
        if not wanted:
            return out
        if len(tracks) > 1 and hasattr(self.face, "faces"):
            # No size filter here: small faces must flow through so the high-res
            # upgrade in _embed_tracks can rescue them — the face_min_px floor is
            # applied there, after the rescue. Intrinsic quality (blur/pose/conf)
            # is already gated inside the engine.
            faces = self.face.faces(frame)
            assigned = assign_faces_to_tracks(faces, tracks)
            for tid in wanted:
                hit = assigned.get(tid)
                if hit is None:
                    out[tid] = (None, None, None, 0)
                    continue
                emb, fbox = hit
                packed = face_crop_jpeg(frame, fbox, max_dim=cfg.capture_crop_px)
                thumb, thumb_box = packed if packed is not None else (None, None)
                out[tid] = (emb, thumb, thumb_box,
                            max(fbox[2] - fbox[0], fbox[3] - fbox[1]))
            return out
        for t in tracks:
            if t.track_id in wanted:
                out[t.track_id] = self._embed_track(frame, t.bbox)
        return out

    def _dedupe_household(self, idents: Dict[str, Identity]) -> None:
        """One member cannot be two people in one frame: when several live tracks carry
        the SAME household id, keep the highest-confidence one and reset the rest to
        unknown (dropped from the cache too, so they re-embed next frame — with
        exclusive face assignment the retry converges on the right person instead of
        ping-ponging)."""
        by_member: Dict[str, List[str]] = {}
        for tid, ident in idents.items():
            if ident is not None and ident.cls == "household" and ident.id:
                by_member.setdefault(ident.id, []).append(tid)
        for tids in by_member.values():
            if len(tids) < 2:
                continue
            tids.sort(key=lambda tid: idents[tid].confidence, reverse=True)
            for tid in tids[1:]:
                idents[tid] = UNKNOWN
                self._idents.pop(tid, None)
                self._ident_ts.pop(tid, None)

    def _embed_track(self, frame, bbox):
        """(embedding, thumb, thumb_box, face_px) for one track's person box.
        Prefer the face-box-aware path: the thumbnail is then a face-CENTERED crop of
        exactly the matched face (never a full-body sliver, never an ambiguous
        two-person box), and the face's position within it rides along so the review
        card can ring it. face_px = the face's longest side (0 unknown) — the high-res
        upgrade trigger. All None-able (no face found / engine w/o API)."""
        emb, thumb, thumb_box, face_px = None, None, None, 0
        if hasattr(self.face, "embed_face"):
            hit = self.face.embed_face(frame, bbox)
            if hit is not None:
                emb, fbox = hit
                face_px = max(fbox[2] - fbox[0], fbox[3] - fbox[1])
                packed = face_crop_jpeg(frame, fbox, max_dim=cfg.capture_crop_px)
                if packed is not None:
                    thumb, thumb_box = packed
        else:  # engines without the face-box API — old person-crop behavior
            emb = self.face.embed(frame, bbox)
        return emb, thumb, thumb_box, face_px

    def _prune_idents(self, live: set) -> None:
        if len(self._idents) > 256:
            self._idents = {k: v for k, v in self._idents.items() if k in live}
            self._ident_ts = {k: v for k, v in self._ident_ts.items() if k in live}

    # ── dashboard re-serve (§3.2 — we are the ONLY client of the cam) ─────────
    def mjpeg_generator(self, annotated: bool = True):
        """Yield multipart MJPEG of the latest frame (annotated when available). The
        dashboard views THIS, never the camera directly (a 2nd client stalls the cam)."""
        last_sent = 0.0
        while not self._stop_evt.is_set():
            if self.is_private():
                return  # end the multipart response — a viewer must not hold a frozen feed
            frame = (self.latest_annotated if annotated else None) or self.latest_raw
            if frame is not None and self.last_frame_ts != last_sent:
                last_sent = self.last_frame_ts
                yield multipart_chunk(frame)
            time.sleep(0.05)

    def status(self) -> dict:
        # Cached-only ONVIF capability summary (never a network probe from a poll
        # path): None until the supervisor's probe succeeds, then e.g.
        # {"ptz": true, "imaging": true, "events": true} — the dashboard uses it to
        # decide which camera controls to draw (fixed cams get no D-pad).
        onvif_client = getattr(self, "_onvif", None)
        caps = onvif_client.capabilities_cached() if onvif_client else None
        return {
            "id": self.cam.id, "zone": self.cam.zone, "ip": self.cam.ip,
            "connected": self.connected, "frames_seen": self.frames_seen,
            "last_frame_age_s": round(time.time() - self.last_frame_ts, 1) if self.last_frame_ts else None,
            "detector": self.detector.backend, "face": self.face.backend,
            "pose": self.pose.backend,
            "rec_mode": self.recorder.mode,
            "records": self.recorder.mode != "off",
            "privacy": self.is_private(),
            "highres": self.highres.status() if self.highres else None,
            "onvif": caps,
            "events_attached": self.events_attached,
            "motion_active": self.motion_active if self.events_attached else None,
        }
