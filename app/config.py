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

    # ── ingestion (this service is its OWN MQTT producer — §5.2) ─────────────
    # Mirrors the hub's seam: publishes to homehub/<zone>/<camId>/<channel>, gated on
    # a live broker, fire-and-forget. Default ON; flip off for an isolated bring-up.
    ingestion_enabled: bool = field(default_factory=lambda: _b("VISION_INGESTION_ENABLED", True))
    mqtt_url: str = field(default_factory=lambda: os.getenv("MQTT_URL", "mqtt://127.0.0.1:1883"))

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

    # ── occupancy debounce (§4.3 / §8 — fire once per arrival, not per frame) ─
    enter_frames: int = field(default_factory=lambda: _i("VISION_ENTER_FRAMES", 3))   # consecutive hits → present
    leave_grace_s: float = field(default_factory=lambda: _f("VISION_LEAVE_GRACE_S", 6.0))  # gone this long → left
    rewake_cooldown_s: float = field(default_factory=lambda: _f("VISION_REWAKE_COOLDOWN_S", 30.0))  # re-arm window

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
    retention_days: int = field(default_factory=lambda: _i("VISION_RETENTION_DAYS", 14))
    disk_cap_gb: float = field(default_factory=lambda: _f("VISION_DISK_CAP_GB", 0.0))  # 0 = no cap (age-only)
    janitor_interval_s: float = field(default_factory=lambda: _f("VISION_JANITOR_INTERVAL_S", 600.0))

    # ── gallery + guests (§4.3 / §11.6 / §11.7 — biometrics stay on the box) ─
    gallery_db: str = field(default_factory=lambda: os.getenv("VISION_GALLERY_DB", os.path.join(DATA, "gallery.db")))
    index_db: str = field(default_factory=lambda: os.getenv("VISION_INDEX_DB", os.path.join(DATA, "index.db")))
    face_dim: int = field(default_factory=lambda: _i("VISION_FACE_DIM", 512))  # ArcFace buffalo_l = 512-d
    guest_cluster_threshold: float = field(default_factory=lambda: _f("VISION_GUEST_CLUSTER_THRESHOLD", 0.5))
    guest_min_sightings: int = field(default_factory=lambda: _i("VISION_GUEST_MIN_SIGHTINGS", 3))  # surface to review
    guest_ttl_days: int = field(default_factory=lambda: _i("VISION_GUEST_TTL_DAYS", 30))

    def __post_init__(self) -> None:
        for d in (DATA, self.rec_dir, self.hls_dir):
            os.makedirs(d, exist_ok=True)


cfg = Config()
