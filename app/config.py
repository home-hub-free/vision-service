"""vision-service configuration — all env-driven, with decision-stub defaults.

Every value here that corresponds to an open call in CAMERA_VISION_PLAN §11 is a
**config knob with a safe default**, so the service runs end-to-end today and the
human decision becomes "flip an env var", not "write code". See ../DECISIONS.md for
the running list and where each plugs in.

The service ships runnable with the NULL perception backends (no torch / no ROCm),
exactly like the speaker-service's `stub` backend: the streaming, recording,
retention, roster-sync, MQTT-producer and dashboard plumbing all work; identity
lights up when a real backend is installed and selected. That makes M0 (solid
stream + view) shippable with zero ML dependencies.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _b(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")


@dataclass
class Config:
    # ── service ──────────────────────────────────────────────────────────────
    port: int = field(default_factory=lambda: _i("VISION_PORT", 8130))

    # ── hub (roster + identity broker) ───────────────────────────────────────
    # The vision-service reads the camera roster (GET /get-devices) and the person
    # roster (GET /auth/users) using HUB_SERVICE_TOKEN — the same internal-caller
    # pattern voice uses (CLAUDE.md "Service token"). The hub never sees frames.
    hub_url: str = field(default_factory=lambda: os.getenv("HUB_URL", "http://127.0.0.1:8088").rstrip("/"))
    hub_service_token: str = field(default_factory=lambda: os.getenv("HUB_SERVICE_TOKEN", ""))
    roster_poll_s: float = field(default_factory=lambda: _f("VISION_ROSTER_POLL_S", 30.0))

    # ── §2 validation escape hatch (CAMERA_BRINGUP_PLAN §2, DECISIONS #3 go/no-go) ─
    # Pull a camera that has NOT declared to the hub — an MJPEG-HTTP IP cam, a laptop/USB
    # webcam served as MJPEG, or the bare ESP32-CAM before its firmware declares — so
    # the WHOLE box pipeline (detect → track → face → ID) can be validated for the
    # image-quality go/no-go BEFORE any firmware exists. Comma-list of `id@zone@url`
    # entries, e.g. "lab@sala@http://192.168.1.50:81/stream". The roster stays the
    # default source; these only fill ids the roster doesn't already carry. Empty by
    # default → roster is the only camera source.
    static_cameras: str = field(default_factory=lambda: os.getenv("VISION_STATIC_CAMERAS", ""))

    # ── RTSP transport (rtsp:// cameras — Reolink/Amcrest/Dahua/Tapo/ONVIF) ──────
    # The reader auto-selects RTSP vs HTTP-MJPEG by URL scheme (`app/rtsp.is_rtsp`). TCP
    # is the reliable default (UDP corrupts H.264 on busy Wi-Fi); raise the timeout for
    # flaky links. These apply only to rtsp:// sources. RTSP cameras need opencv installed
    # (the reader decodes via OpenCV's FFmpeg backend) + ffmpeg on PATH for codec-copy
    # recording of the main stream — see requirements.txt.
    rtsp_transport: str = field(default_factory=lambda: os.getenv("VISION_RTSP_TRANSPORT", "tcp"))
    rtsp_timeout_s: float = field(default_factory=lambda: _f("VISION_RTSP_TIMEOUT_S", 5.0))
    rtsp_max_read_misses: int = field(default_factory=lambda: _i("VISION_RTSP_MAX_READ_MISSES", 30))
    # Wall-clock deadman: if no frame decodes for this long the reader raises so the worker
    # reconnects — the count-based escape alone scales with the read timeout (each miss can
    # block a full rtsp_timeout_s inside cap.read()), not with real time.
    rtsp_stall_s: float = field(default_factory=lambda: _f("VISION_RTSP_STALL_S", 15.0))

    # ── ingestion (this service is its OWN MQTT producer — §5.2) ─────────────
    # Mirrors the hub's seam: publishes to homehub/<zone>/<camId>/<channel>, gated on
    # a live broker, fire-and-forget. Default ON; flip off for an isolated bring-up.
    ingestion_enabled: bool = field(default_factory=lambda: _b("VISION_INGESTION_ENABLED", True))
    mqtt_url: str = field(default_factory=lambda: os.getenv("MQTT_URL", "mqtt://127.0.0.1:1883"))

    # ── ONVIF control seam (CAMERA_ONVIF_CONTROL_PLAN — PTZ/imaging/events/clock) ─
    # The control seam lives HERE (plan §1): credentials are parsed from each camera's
    # existing rtsp:// stream URL (never a second secret store), capabilities are probed
    # per-camera (the C110s are fixed — events+imaging, no PTZ), and everything degrades
    # per-capability. Port 2020 is the Tapo/Mercusys family default; override globally
    # here (a per-camera override only gets built when a camera actually differs).
    onvif_enabled: bool = field(default_factory=lambda: _b("VISION_ONVIF_ENABLED", True))
    onvif_port: int = field(default_factory=lambda: _i("VISION_ONVIF_PORT", 2020))
    onvif_timeout_s: float = field(default_factory=lambda: _f("VISION_ONVIF_TIMEOUT_S", 6.0))
    # A continuous PTZ move is auto-stopped after its ttl; this caps the ttl a caller
    # may request (plan §2: never leave a continuous move running).
    ptz_max_ttl_s: float = field(default_factory=lambda: _f("VISION_PTZ_MAX_TTL_S", 2.0))
    # In-camera motion/tamper via PullPoint (plan §3). Default on; per-camera it only
    # activates when the capability probe says the camera actually emits events.
    onvif_events_enabled: bool = field(default_factory=lambda: _b("VISION_ONVIF_EVENTS_ENABLED", True))
    # 5s, not 10: the MC200 kills PullMessages sockets held ~10s+ on an idle scene
    # (verified live 2026-07-03 — 10s pulls churned the subscription every ~21s;
    # 3–5s pulls answer cleanly with an empty response).
    onvif_pull_timeout_s: float = field(default_factory=lambda: _f("VISION_ONVIF_PULL_TIMEOUT_S", 5.0))
    # Opt-in GPU/CPU saver: run the heavy perception pipeline only while the camera's
    # own motion detector says something moved (+ linger). Default OFF — always-on
    # detection stays the baseline; YOLO remains authoritative for *person*.
    detect_on_motion: bool = field(default_factory=lambda: _b("VISION_DETECT_ON_MOTION", False))
    motion_linger_s: float = field(default_factory=lambda: _f("VISION_MOTION_LINGER_S", 10.0))
    # WAN-blocked cameras can't NTP → push the box clock daily (plan §6). 0 disables.
    onvif_time_sync_h: float = field(default_factory=lambda: _f("VISION_ONVIF_TIME_SYNC_H", 24.0))

    # ── hub room-digest push (PERCEPTION_TO_AGENT_PLAN §3.1 — the agent-facing fusion) ─
    # Besides the MQTT producer (above, which feeds memory + the agent WAKE lane), we PUSH a
    # small per-zone occupancy+identity digest straight to the hub on every salient change, so
    # the hub can FUSE it (with ambient mic + PIR) into the `rooms` world-model the agent reads
    # on GET /state. Best-effort like ingestion: POST /perception, fire-and-forget, never throws
    # into perception, a no-op when the hub is down. Only resolved {id,name,class,confidence}
    # crosses — NEVER an embedding (biometrics stay on the box). Default ON; flip off to isolate.
    hub_push_enabled: bool = field(default_factory=lambda: _b("VISION_HUB_PUSH_ENABLED", True))

    # ── perception backends (§4.2, §11.1/§11.2 — the GPU/ROCm decisions) ─────
    # person/track: "null" (no detection — M0) | "ultralytics" (YOLO+ByteTrack).
    # face: "null" (no ID) | "insightface" (SCRFD detect + ArcFace embed).
    # DECISION (§11.2): onnxruntime-ROCm vs torch-ROCm runtime — selected inside the
    # backend module; default install runs CPU/null so nothing is blocked on ROCm.
    backend: str = field(default_factory=lambda: os.getenv("VISION_BACKEND", "null"))
    face_backend: str = field(default_factory=lambda: os.getenv("VISION_FACE_BACKEND", "null"))
    detect_fps: float = field(default_factory=lambda: _f("VISION_DETECT_FPS", 5.0))  # §11.1 GPU lever
    person_conf: float = field(default_factory=lambda: _f("VISION_PERSON_CONF", 0.4))
    face_match_threshold: float = field(default_factory=lambda: _f("VISION_FACE_THRESHOLD", 0.35))
    # A live match must also BEAT the 2nd-best member by this margin — a 0.36-vs-0.34
    # face is ambiguous, not a match (ambiguity is how members cross-contaminate).
    face_match_margin: float = field(default_factory=lambda: _f("VISION_FACE_MATCH_MARGIN", 0.05))
    # Re-verify a track's cached household label every N seconds (0 disables): heals
    # tracker id-switches (two people cross → their labels swap and stick otherwise).
    face_reverify_s: float = field(default_factory=lambda: _f("VISION_FACE_REVERIFY_S", 20.0))

    # ── SCRFD face-detector sizing (the far/small-face levers) ────────────────
    # `det_size` is the square SCRFD works at: 640 (the insightface default) throws away
    # the detail that lets it find a DISTANT face; 1024–1280 recovers range at the cost of
    # more CPU per detected NEW track (SCRFD only runs on new/unmatched tracks at
    # detect_fps, so it's affordable). `det_thresh` is SCRFD's own confidence bar (its
    # default is 0.5) — lower it to keep weak far-face detections. `face_min_px` is an
    # optional post-detect gate: skip embedding a face whose bbox longest side is under
    # this many pixels, so tiny low-detail faces don't seed noisy guest clusters. 0 = off.
    face_det_size: int = field(default_factory=lambda: _i("VISION_FACE_DET_SIZE", 1024))
    face_det_thresh: float = field(default_factory=lambda: _f("VISION_FACE_DET_THRESH", 0.4))
    face_min_px: int = field(default_factory=lambda: _i("VISION_FACE_MIN_PX", 0))

    # Online reinforcement (§4.3): on a confident, UNAMBIGUOUS household match, fold the
    # live embedding into that member's centroid so passive recognition self-improves
    # day-to-day (no manual re-enroll). Gated strictly to prevent drift: the match must
    # clear `reinforce_threshold` (deliberately > the match threshold) AND beat the
    # 2nd-best member by `reinforce_margin` (so a look-alike can't pull a centroid), and
    # the running-mean weight is capped at `reinforce_cap` so no single frame dominates
    # (it becomes a gentle EMA once a member is well-established).
    face_reinforce: bool = field(default_factory=lambda: _b("VISION_FACE_REINFORCE", True))
    face_reinforce_threshold: float = field(default_factory=lambda: _f("VISION_FACE_REINFORCE_THRESHOLD", 0.5))
    face_reinforce_margin: float = field(default_factory=lambda: _f("VISION_FACE_REINFORCE_MARGIN", 0.08))
    face_reinforce_cap: int = field(default_factory=lambda: _i("VISION_FACE_REINFORCE_CAP", 50))

    # ── T0 activity: dwell + speed (VISION_CONTEXT_TIERS_PLAN §2 — no model) ──
    # Zone activity is classified from data the tracker already has: `passing` when
    # dwell is short OR the speed EMA is above the bar; `settled` past the settle
    # dwell at low speed; `lingering` between. Speed is in frame-widths/s.
    activity_pass_dwell_s: float = field(default_factory=lambda: _f("VISION_ACTIVITY_PASS_DWELL_S", 20.0))
    activity_settle_dwell_s: float = field(default_factory=lambda: _f("VISION_ACTIVITY_SETTLE_DWELL_S", 60.0))
    activity_speed_fws: float = field(default_factory=lambda: _f("VISION_ACTIVITY_SPEED_FWS", 0.25))
    # Occupied-zone digest heartbeat: activity/dwell changes with no salient edge still
    # reach the hub within this period (and keep the hub's vision TTL fresh). Edges
    # remain the primary push; this is NOT per-frame spam.
    digest_heartbeat_s: float = field(default_factory=lambda: _f("VISION_DIGEST_HEARTBEAT_S", 30.0))

    # ── T2a activity hints: context rules (plan §4.2a — no model, digest-build only) ─
    # zone-kind × dwell × posture × hour → a hedged activity HINT on the digest
    # ("making breakfast or coffee"). Pure code over T0/T1 fields, evaluated only when
    # a digest is built (never per frame). Flag off = the tier's §8 kill switch.
    hints_enabled: bool = field(default_factory=lambda: _b("VISION_ACTIVITY_HINTS", True))
    # Zone-name → kind overrides, "zone=kind,zone2=kind" (kinds: kitchen|dining|living|
    # office|bedroom|entrance|bathroom). The built-in es/en synonym table covers the
    # common names; this knob exists for zones the table can't guess ("cueva=office").
    zone_kinds: str = field(default_factory=lambda: os.getenv("VISION_ZONE_KINDS", ""))
    # Hint hysteresis: a fired hint survives brief rule dropouts (a posture flicker, one
    # `moving` snapshot) for this long while the zone stays occupied. An emptied zone or
    # a DIFFERENT fired hint clears/replaces it immediately.
    hint_hold_s: float = field(default_factory=lambda: _f("VISION_HINT_HOLD_S", 30.0))

    # ── T1 posture: pose → body state (plan §3 — CPU, motion-gated, cost-gated) ─
    # "null" ships the tier dark; "ultralytics" runs yolov8n-pose on the SAME cadence
    # as detect, only on frames that already have person tracks. `pose_every_n` is the
    # §3 cost-gate lever: N>1 runs pose every Nth detect frame (posture changes slowly).
    pose_backend: str = field(default_factory=lambda: os.getenv("VISION_POSE_BACKEND", "null"))
    pose_every_n: int = field(default_factory=lambda: _i("VISION_POSE_EVERY_N", 1))
    pose_min_kp_conf: float = field(default_factory=lambda: _f("VISION_POSE_MIN_KP_CONF", 0.3))
    # Posture debounce: a NEW posture must hold this long before it replaces the
    # committed one. A partial bbox at frame-exit reads "lying" for a few frames and
    # would otherwise flap the digest (and drop a T2a hint). The first read commits
    # immediately (nothing to protect yet).
    posture_stable_s: float = field(default_factory=lambda: _f("VISION_POSTURE_STABLE_S", 10.0))
    # Fall-shaped alert (§3): `lying` outside these zones for longer than the dwell bar
    # emits one posture_alert edge per episode. Alert-only — no autonomy attached.
    lying_ok_zones: str = field(default_factory=lambda: os.getenv(
        "VISION_LYING_OK_ZONES", "bedroom,recamara,cuarto,living,sala"))
    lying_alert_dwell_s: float = field(default_factory=lambda: _f("VISION_LYING_ALERT_DWELL_S", 60.0))

    # ── occupancy debounce (§4.3 / §8 — fire once per arrival, not per frame) ─
    enter_frames: int = field(default_factory=lambda: _i("VISION_ENTER_FRAMES", 3))   # consecutive hits → present
    leave_grace_s: float = field(default_factory=lambda: _f("VISION_LEAVE_GRACE_S", 6.0))  # gone this long → track expires
    rewake_cooldown_s: float = field(default_factory=lambda: _f("VISION_REWAKE_COOLDOWN_S", 30.0))  # re-arm window
    # Identity-level hysteresis (the presence LEDGER — see occupancy.py). Measured
    # 2026-07-06: detector dropouts on seated people re-formed tracks 30–170s later,
    # emitting ~700 false enter/leave edges in 6h. `leave_confirm_s`: a vanished
    # person goes pending-left silently and person_left only emits after this much
    # continuous absence (a return inside the window heals with ZERO edges; 0 = old
    # per-track behaviour). `identify_settle_s`: a NEW unresolved (unknown) presence
    # holds its entered edge this long so the face can resolve first — a flap-heal or
    # a short-lived detection blip then never wakes anyone (0 = announce instantly).
    leave_confirm_s: float = field(default_factory=lambda: _f("VISION_LEAVE_CONFIRM_S", 120.0))
    identify_settle_s: float = field(default_factory=lambda: _f("VISION_IDENTIFY_SETTLE_S", 8.0))

    # ── recording (§9) ───────────────────────────────────────────────────────
    # mode: "off" | "continuous" | "gated" | "hybrid" (§9.2; hybrid is the rec default).
    # encoder: "libx264" (CPU, today's recipe) | "vaapi"/"amf" (GPU — DECISION §9.1/§11.4).
    rec_mode_default: str = field(default_factory=lambda: os.getenv("VISION_REC_MODE", "hybrid"))
    rec_encoder: str = field(default_factory=lambda: os.getenv("VISION_REC_ENCODER", "libx264"))
    rec_dir: str = field(default_factory=lambda: os.getenv("VISION_REC_DIR", os.path.join(DATA, "recordings")))
    hls_dir: str = field(default_factory=lambda: os.getenv("VISION_HLS_DIR", os.path.join(DATA, "hls")))
    segment_seconds: int = field(default_factory=lambda: _i("VISION_SEGMENT_SECONDS", 300))  # 5-min archive segments
    preroll_seconds: float = field(default_factory=lambda: _f("VISION_PREROLL_SECONDS", 12.0))  # gated ring buffer
    rec_fps: int = field(default_factory=lambda: _i("VISION_REC_FPS", 10))

    # ── retention janitor (§9.3 — mandatory; never wedge the disk) ───────────
    # DECISION (§9.6/§11.4): exact numbers from a MEASURED day. Defaults are placeholders.
    retention_days: int = field(default_factory=lambda: _i("VISION_RETENTION_DAYS", 5))
    disk_cap_gb: float = field(default_factory=lambda: _f("VISION_DISK_CAP_GB", 0.0))  # 0 = no cap (age-only)
    janitor_interval_s: float = field(default_factory=lambda: _f("VISION_JANITOR_INTERVAL_S", 600.0))

    # ── privacy mode (per-camera "stop watching NOW" — see app/privacy.py) ───
    # Persisted set of private cam ids; must survive restarts, so it's a file in
    # DATA next to the gallery/index DBs, not an env flag.
    privacy_file: str = field(default_factory=lambda: os.getenv("VISION_PRIVACY_FILE", os.path.join(DATA, "privacy.json")))

    # ── gallery + guests (§4.3 / §11.6 / §11.7 — biometrics stay on the box) ─
    gallery_db: str = field(default_factory=lambda: os.getenv("VISION_GALLERY_DB", os.path.join(DATA, "gallery.db")))
    index_db: str = field(default_factory=lambda: os.getenv("VISION_INDEX_DB", os.path.join(DATA, "index.db")))
    face_dim: int = field(default_factory=lambda: _i("VISION_FACE_DIM", 512))  # ArcFace buffalo_l = 512-d
    guest_cluster_threshold: float = field(default_factory=lambda: _f("VISION_GUEST_CLUSTER_THRESHOLD", 0.5))
    guest_min_sightings: int = field(default_factory=lambda: _i("VISION_GUEST_MIN_SIGHTINGS", 3))  # surface to review
    guest_ttl_days: int = field(default_factory=lambda: _i("VISION_GUEST_TTL_DAYS", 30))

    # ── on-demand high-res sampling (the far-face accuracy lever) ─────────────
    # Detect/track stays on the cheap substream; when a NEW track's face is found but
    # SMALL (under min_face_px on the substream — the noisy-embedding zone that seeded
    # 146 clusters for 3 people), ONE full-res frame is fetched from the camera's main
    # source and the face re-embedded from it. ArcFace normalizes to 112×112, so
    # ~110px-wide faces gain nothing (near cams skip this entirely) while a 50px face
    # doubling to 100px is the difference between cluster fodder and a clean match.
    # Near-zero continuous cost: fires only on new/re-verified tracks, rate-limited per
    # camera, and only cameras with a dual-stream main (`record_url`) participate.
    # Source order: declared snapshot URL → ONVIF GetSnapshotUri (probed once; the
    # C110 faults on it — verified 2026-07-06) → one-frame RTSP grab of the main
    # stream. Repeated failures (e.g. the camera caps concurrent RTSP sessions) mark
    # the sampler degraded: substream embeddings pass through unchanged and retries
    # slow to 10× the interval, so a broken source can never starve recognition.
    highres_enabled: bool = field(default_factory=lambda: _b("VISION_HIGHRES_ENABLED", True))
    highres_min_face_px: int = field(default_factory=lambda: _i("VISION_HIGHRES_MIN_FACE_PX", 90))
    highres_interval_s: float = field(default_factory=lambda: _f("VISION_HIGHRES_INTERVAL_S", 3.0))

    # ── capture ledger (identity-pollution insurance) ─────────────────────────
    # The gallery centroids are running means: once an embedding is folded into a
    # member it can never be exactly recovered (reinforce folds especially). This
    # ledger permanently archives every face crop + embedding behind an identity
    # decision — plain JPEGs on disk grouped per identity, indexed (with the exact
    # embedding) in the gallery DB — so the household can always review the raw
    # ingredients and REBUILD any member's profile from curated ones
    # (tools/rebuild_profile.py). No retention on purpose (crops are ~10 KB; a busy
    # week is a few MB). Dir default "" = a `captures/` folder next to the gallery DB.
    captures_enabled: bool = field(default_factory=lambda: _b("VISION_CAPTURES_ENABLED", True))
    captures_dir: str = field(default_factory=lambda: os.getenv("VISION_CAPTURES_DIR", ""))
    # Longest side of the stored face/person crops (ledger + review cards). Display
    # and archive only — embeddings are computed from the live frame BEFORE the crop
    # is downscaled, so this knob buys reviewability/re-embeddability, never accuracy.
    # 480 ≈ 30 KB/crop vs 220px ≈ 10 KB: cheap insurance that the archived ingredient
    # is comfortably judgeable by a human and re-embeddable if its ledger row is lost.
    capture_crop_px: int = field(default_factory=lambda: _i("VISION_CAPTURE_CROP_PX", 480))

    # ── review tiers (self-healing gallery — how a guest cluster resolves) ───
    # Each unpromoted cluster is scored against the household centroids and lands in
    # one of three tiers:
    #   score ≥ autoheal_threshold (+margin) → "definitely them": silently merged into
    #     that member, never shown to anyone (same strictness posture as reinforce).
    #   score ≥ suggest_threshold            → "probably them": surfaced as an
    #     "Is this you?" card addressed to that member only.
    #   below                                 → unknown: surfaced to every member.
    # A member answering "No" is recorded per-cluster and permanently blocks both
    # suggesting AND auto-healing that cluster into them.
    face_autoheal_threshold: float = field(default_factory=lambda: _f("VISION_FACE_AUTOHEAL_THRESHOLD", 0.5))
    face_autoheal_margin: float = field(default_factory=lambda: _f("VISION_FACE_AUTOHEAL_MARGIN", 0.08))
    face_suggest_threshold: float = field(default_factory=lambda: _f("VISION_FACE_SUGGEST_THRESHOLD", 0.2))

    def __post_init__(self) -> None:
        for d in (DATA, self.rec_dir, self.hls_dir):
            os.makedirs(d, exist_ok=True)


cfg = Config()
