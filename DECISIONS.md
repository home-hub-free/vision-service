# Camera Vision — open decisions (stubbed, wired, ready to flip)

Every human decision from CAMERA_VISION_PLAN §11 (and the §9 recording residuals) was
left as a **config knob with a safe default** or a **clearly-marked code stub**, so the
rest of the system was built against it and nothing is blocked. This is the list to
resolve later; each row says *where it plugs in* so flipping it is a config/edit, not a
re-architecture.

| # | Decision (plan ref) | Default shipped | Where it plugs in | What to do to resolve |
|---|---|---|---|---|
| 1 | **GPU contention** — vision vs voice TTFA (§11.1) | `VISION_DETECT_FPS=5`, face-embed gated on new tracks, null backend | `app/camera.py` throttle + `app/config.py` `detect_fps` | Measure voice TTFA with a real backend on; set the FPS cap / decide time-slice. Pick a target "vision must not regress TTFA beyond X". |
| 2 | **ROCm runtime** — torch-ROCm vs onnxruntime-ROCm (§11.2) | CPU/null (`VISION_DEVICE=cpu`) | `app/perception.py` `_UltralyticsDetector.device`, `_InsightFaceEngine` providers; `requirements.txt` | Spike both; set `VISION_DEVICE=cuda` and/or `VISION_ORT_PROVIDERS=ROCMExecutionProvider`. Face stages run fine on CPU as a fallback. |
| 3 | **ESP32-CAM image quality for ID** — M2 go/no-go (§11.3) | **escape hatch shipped, verdict OPEN** (presence works regardless) | `VISION_STATIC_CAMERAS` env (`hub_client.parse_static_cameras`) — pull any MJPEG/RTSP source with NO firmware; `Camera.stream_url` is camera-agnostic | **[HUMAN/HW]** Run the §2 spike (see README "Validation spike") against a known-good cam, stand at room distance, record GO/NO-GO below. If OV2640 ID is poor → point the roster (or a static entry) at an **RTSP/IP cam** (1080p) — only the URL changes; pipeline is identical. |
| 4 | **Recording encode** — CPU libx264 vs GPU VAAPI/AMF (§9.1/§11.4) | `VISION_REC_ENCODER=libx264` | `app/recorder.py` `_encode_args` | Measure CPU under load; set `VISION_REC_ENCODER=vaapi` (or `amf`) if it saturates AND doesn't starve the vision/LLM GPU. |
| 5 | **Retention numbers / disk cap** (§9.3/§11.4) | `VISION_RETENTION_DAYS=14`, `VISION_DISK_CAP_GB=0` (age-only) | `app/config.py` + `app/retention.py` | Measure one real day/camera; set days + cap from the measured GB. |
| 6 | **At-rest encryption** of recordings (§9.3/§11.4) | off (playback gated behind dashboard auth) | `app/recorder.py` output path | Decide if raw video at rest needs encryption; if so, encrypt the `recordings/` volume. |
| 7 | **Gallery storage** — vision-local vs memory-service (§11.6) | vision-local sqlite (`app/gallery.py`) | `app/gallery.py` db path | Keep biometrics on the box (recommended). Only move if there's a reason; don't put embeddings in the hub. |
| 8 | **Guest lifecycle** — cluster threshold, TTL, prompt-to-name (§11.7) | `GUEST_CLUSTER_THRESHOLD=0.5`, `GUEST_MIN_SIGHTINGS=3`, `GUEST_TTL_DAYS=30` | `app/config.py` + `app/gallery.py` cluster + `guests` route | Tune on real footage; add a TTL janitor for stale unnamed guests if needed. |
| 9 | **Enrollment endpoint owner** (§5.3/§11.8) | **vision-service** (`POST /vision/faces/enroll`), hub stays biometrics-free | `app/routes/enroll.py` + dashboard Face-ID control | Confirmed = vision-service. Hub only brokers identity (roster + token). No change needed unless reversed. |
| 10 | **Multi-camera scaling** (§11.9) | one daemon thread per stream; perception at `detect_fps` | `app/supervisor.py` + `app/camera.py` | If N grows large, consider an async task model or a worker pool; back-pressure already lives in the worker. |
| 11 | **Identity fusion (face × voice)** (§11.10) | **stub** — not yet fused | `app/occupancy.Identity` (shared envelope) + voice resolver (separate repo) | Design the rule: same person seen + heard in one zone → boost confidence / face confirms a low-confidence voiceprint. Both already fill the same `data.user` shape, so fusion is a reconcile step, not new plumbing. |
| 12 | **Dashboard stream delivery** — MJPEG proxy vs HLS vs WebRTC (§11.5) | **MJPEG proxy** (`/vision/stream/<id>`); HLS also served (`/vision/hls/<id>/live.m3u8`) | `app/routes/streams.py` + `app/main.py` static mount; dashboard tile | Pick per deployment; both are wired. WebRTC is the future low-latency option (most work). |
| 13 | **Event index → memory-service?** (§9.6/§11.4) | vision-local only (events already reach memory via MQTT) | `app/index_db.EventIndex._to_memory` (empty stub) | If the segment pointer must live in memory-service too, implement the POST in `_to_memory`. |
| 14 | **Camera zone assignment** — flash-time vs dashboard (§3.3) | dashboard-assigned (recommended; units interchangeable) | hub `/devices-data-set` merges `zone`; roster carries it to the worker | No code change — assign zone in the dashboard after declare. |

## Firmware (separate `devices/` repo — now BUILT, 2026-06-28)
The ESP32-CAM firmware (§3) lives in the standalone `devices` repo (`devices/camera`).
It is **written + compiles green** against `FIRMWARE_CONTRACT.md`: declares with the
`stream` block via the shared core's new additive `HomeHubDevice::setDeclareExtra()`
hook; `/stream`+`/capture` on an `esp_http_server` (:81), `/status`+`/control` on the
shared `WebServer` (:80); the retired UDP/:82→`192.168.1.199` scheme is gone; mDNS hub
discovery; brownout detector disabled. `pio run` → `firmware.bin` (RAM 16%, Flash 31%).
**Remaining is [HUMAN/HW] only:** USB-flash the unit and the §2 stand-in-front go/no-go
(see `devices/camera/README.md`). The hub already accepts the declare
(`captureStreamDeclare`).

## Resolved decision verdicts (record as you run the spikes)
- **#3 ESP32-CAM image quality — GO / NO-GO:** _OPEN_ — run the §2 validation spike
  (README) and record here: ⬜ GO (OV2640 adequate at room distance) · ⬜ NO-GO (use
  RTSP/IP cam for ID rooms, ESP32-CAM presence-only).
- **#1 GPU contention — voice-TTFA regression target:** _OPEN_ — measure TTFA with a
  real backend on vs idle; record the chosen `VISION_DETECT_FPS` cap + the target
  ("vision must not regress TTFA beyond X ms").
- **#2 ROCm runtime — torch-ROCm vs onnxruntime-ROCm:** _OPEN_ — spike both in a
  throwaway venv (README "M1/M2 ROCm spike"); record the winner + the exact working
  `pip install` line in `requirements.txt`.
