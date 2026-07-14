---
title: Camera Vision — open decisions
summary: vision-service open decisions as config knobs with safe defaults (GPU/ROCm, retention, fusion, …) plus the identity-quality + review-loop overhauls and record-scope work.
status: LIVE
owner: vision-service
updated: 2026-07-09
tags: [vision, camera, decisions]
---

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
| 3 | **ESP32-CAM image quality for ID** — M2 go/no-go (§11.3) | **escape hatch shipped, verdict OPEN** (presence works regardless) | `VISION_STATIC_CAMERAS` env (`hub_client.parse_static_cameras`) — pull any MJPEG-HTTP **or RTSP** source with NO firmware (reader auto-selects by scheme, `app/rtsp.py`); dual-stream via a 2nd record URL | **[HUMAN/HW]** Run the §2 spike (see README "Validation spike") against a known-good cam, stand at room distance, record GO/NO-GO below. If OV2640 ID is poor → point the roster (or a static entry) at a higher-res IP cam (1080p): both **MJPEG-HTTP** and **RTSP/H.264** are config-only now (RTSP wired in `app/rtsp.py`); use the dual-stream form for ID rooms (detect on substream, record main by codec-copy). |
| 4 | **Recording encode** — CPU libx264 vs GPU VAAPI/AMF (§9.1/§11.4) | `VISION_REC_ENCODER=libx264` | `app/recorder.py` `_encode_args` | Measure CPU under load; set `VISION_REC_ENCODER=vaapi` (or `amf`) if it saturates AND doesn't starve the vision/LLM GPU. |
| 5 | **Retention numbers / disk cap** (§9.3/§11.4) | **RESOLVED 2026-07-04: `VISION_RETENTION_DAYS=5`** (age-only, `DISK_CAP_GB=0`) — user call pending a NAS | `app/config.py` + `app/retention.py` + `.env` | Revisit once a NAS exists; add a disk cap (or point `rec_dir` at the mount) if 5-day volume outgrows the box. |
| 6 | **At-rest encryption** of recordings (§9.3/§11.4) | off (playback gated: **list = hub bearer `require_user`; clip bytes = signed short-TTL token**, `app/media_token.py`) | `app/recorder.py` output path | Decide if raw video at rest needs encryption; if so, encrypt the `recordings/` volume. |
| 7 | **Gallery storage** — vision-local vs memory-service (§11.6) | vision-local sqlite (`app/gallery.py`) | `app/gallery.py` db path | Keep biometrics on the box (recommended). Only move if there's a reason; don't put embeddings in the hub. |
| 8 | **Guest lifecycle** — cluster threshold, TTL, prompt-to-name (§11.7) | `GUEST_CLUSTER_THRESHOLD=0.5`, `GUEST_MIN_SIGHTINGS=3`, `GUEST_TTL_DAYS=30` | `app/config.py` + `app/gallery.py` cluster + `guests` route | Tune on real footage; add a TTL janitor for stale unnamed guests if needed. |
| 9 | **Enrollment endpoint owner** (§5.3/§11.8) | **vision-service** (`POST /vision/faces/enroll`), hub stays biometrics-free | `app/routes/enroll.py` + dashboard Face-ID control | Confirmed = vision-service. Hub only brokers identity (roster + token). No change needed unless reversed. |
| 10 | **Multi-camera scaling** (§11.9) | one daemon thread per stream; perception at `detect_fps` | `app/supervisor.py` + `app/camera.py` | If N grows large, consider an async task model or a worker pool; back-pressure already lives in the worker. |
| 11 | **Identity fusion (face × voice)** (§11.10) | **stub** — not yet fused | `app/occupancy.Identity` (shared envelope) + voice resolver (separate repo) | Design the rule: same person seen + heard in one zone → boost confidence / face confirms a low-confidence voiceprint. Both already fill the same `data.user` shape, so fusion is a reconcile step, not new plumbing. |
| 12 | **Dashboard stream delivery** — MJPEG proxy vs HLS vs WebRTC (§11.5) | **MJPEG proxy** (`/vision/stream/<id>`); HLS also served (`/vision/hls/<id>/live.m3u8`) | `app/routes/streams.py` + `app/main.py` static mount; dashboard tile | Pick per deployment; both are wired. WebRTC is the future low-latency option (most work). |
| 13 | **Event index → memory-service?** (§9.6/§11.4) | vision-local only (events already reach memory via MQTT) | `app/index_db.EventIndex._to_memory` (empty stub) | If the segment pointer must live in memory-service too, implement the POST in `_to_memory`. |
| 14 | **Camera zone assignment** — flash-time vs dashboard (§3.3) | dashboard-assigned (recommended; units interchangeable) — **now covers static/.env IP cams too**: they are proxy-declared to the hub each roster sync (`hub_client.declare_camera`), so their zone is a dashboard dropdown like any device; the `@zone@` in `VISION_STATIC_CAMERAS` is only the first-boot/hub-down fallback | hub `/devices-data-set` merges `zone`; roster carries it to the worker | No code change — assign zone in the dashboard after declare. |

## Identity quality overhaul (BUILT 2026-07-07)
Both david↔ana pollution incidents traced to one root: **embeddings of small/blurry/
turned-away faces are noise** (a member's own enroll burst self-agreed at cos ~0.2,
measured on the capture ledger), and every runtime fold into a shared running mean
(reinforce, promote→enroll) averaged that noise until two members read cos 0.702
apart and swapped names. Four decisions, all shipped:
- **Identity abstains below the quality bar.** Engine-enforced det/pose/sharpness
  gates (`face_quality_reason`) + a size floor (`face_min_px`, applied AFTER the
  high-res rescue so far faces still get their main-stream upgrade). A gated-out
  face still counts for occupancy. Enrollment is stricter (`assess_enroll`:
  exactly-one-face, 110px, yaw≤30) and 422s coach the guided flow.
- **Profiles are immutable anchor sets.** `anchors` table = individually-stored
  gated enroll embeddings; matching = top-2 anchor mean (one rogue anchor can't
  impersonate). Runtime never mutates anchors; reinforce survives for legacy
  anchor-less members only; first gated enroll RESETS a legacy centroid.
- **Silent folds must be earned.** Promotion = routing only (never writes `faces`);
  autoheal needs maturity (`min_sightings`/`min_span_s`/`min_coherence`) — never a
  single frame. Human review answers stay ungated.
- **A tripwire watches the one number that mattered.** `app/face_audit.py` (boot +
  every 6h, `GET /faces/health`, `POST /faces/audit`): member-vs-member max
  cross-anchor cosine ≥ 0.45 → SMEAR ALARM, all silent folds freeze (self-clears on
  a healthy pass); promotions re-scored against anchors (detach < 0.30); 24h
  cluster-churn signal.

## Review-loop health fixes (2026-07-08)
A numeric audit of the review flow found the household's "yes, it's me" answers were
being fought by the system, not compounded. Five fixes (all `app/gallery.py` +
`app/face_audit.py`), motivated by measured numbers (184 human promotes vs 118
auto-detaches on the same cards; 37/52 promotions were junk-embedding clusters; 11%
of promoted-path IDs were noise faces < 0.20 vs the member's anchors):
- **Human confirms are sticky.** A `guests.promoted_by` column tags each promotion
  `human` (review flow) or `auto` (autoheal); NULL legacy rows read as human. The
  auditor now detaches ONLY `auto` promotions — silently reverting a person's answer
  and re-queuing the card was a loop they could not win.
- **A promotion must cohere to speak.** `resolve()` gates the promoted path on an
  absolute floor (`face_promoted_min_coherence`, 0.30): the LIVE face must resemble
  the member's anchors, not merely beat the (also-low) other members on margin. This
  makes a junk/blurry confirm HARMLESS (it routes nothing) so it needn't be detached,
  and closes the noise-label leak. Promoted-only members (no anchors) are unaffected.
- **The auditor's detach is not a rejection.** `detach_cluster(reject=False)` for the
  auditor: a low coherence score is statistical, not a human "not me", so it no
  longer poisons the cluster's `rejected` set (which had been suppressing the suggest
  tier and re-queuing cards as unanswerable "who is this?").
- **A confirm clears a prior rejection.** `promote_guest` removes the member from the
  cluster's `rejected` set (a one-time boot migration fixed the 24 legacy rows left
  both promoted-to and rejected-by the same member).
- **Unanswerable single blips aren't queued.** `review_queue` skips a
  seen-exactly-once cluster whose only crop has no locatable face — hidden until it
  recurs (thumb self-heals then). Also: the ledger `reinforced` flag now records an
  ACTUAL fold, not intent (anchored members never fold).

## Footage review + record scope (BUILT 2026-07-04)
The §9.5 review surface is now built end-to-end (was: recorder + index existed, but no
way to browse/play archived clips and every camera recorded).
- **Record scope** = a camera archives footage **iff it declares an RTSP main stream**
  (`Camera.record_url`). `app/camera.py` builds the IP-cam fleet's recorder (codec-copy
  continuous) and gives every MJPEG-only cam — the ESP32-CAM entrance cam + the face-ID
  desk cams on satellites — a hard-off recorder (`mode="off"`). `status().records`
  surfaces this to the dashboard so only recording cams show a Recordings entry point.
- **Review routes** (`app/routes/recordings.py`): `GET /recordings/cameras` (recording
  cams + footage days) and `GET /recordings/{cam}/segments?start=&end=` are
  `require_user` bearer-gated; each segment carries its event markers +
  `GET /recordings/{cam}/clip/{seg_id}?token=` (Range-seekable `FileResponse`, path-
  traversal-guarded, gated by a signed short-TTL token from `app/media_token.py` so a
  `<video>` element with no Authorization header can still play member-only footage).
- **Index reads** added: `index_db.segments_between` / `recording_days` / `segment_by_id`.
- **Dashboard**: Recordings lightbox (day chips → clip list with "who was present" →
  seekable `<video>`), reached from the camera live view when `records` is true.

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
  (README) and record here: ⬜ GO (OV2640 adequate at room distance) · ⬜ NO-GO (use a
  higher-res IP cam for ID rooms — MJPEG-HTTP or RTSP, both supported — ESP32-CAM
  presence-only).
- **#1 GPU contention — voice-TTFA regression target:** _OPEN_ — measure TTFA with a
  real backend on vs idle; record the chosen `VISION_DETECT_FPS` cap + the target
  ("vision must not regress TTFA beyond X ms").
- **#2 ROCm runtime — torch-ROCm vs onnxruntime-ROCm:** _OPEN_ — spike both in a
  throwaway venv (README "M1/M2 ROCm spike"); record the winner + the exact working
  `pip install` line in `requirements.txt`.
