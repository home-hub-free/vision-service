# ESP32-CAM firmware contract (handoff to the `devices` repo session)

CAMERA_VISION_PLAN §3 delegates the firmware to the standalone `devices` repo
(`home-hub-free/devices`, out of scope of the root repo). This file freezes the
**contract** that firmware must honour so it slots into what's already built here:
the hub stores the declare `stream` block (`server` `captureStreamDeclare`) and the
vision-service pulls the stream from the roster. Build firmware against a live roster
today — nothing else is blocked on it.

> **Boundary:** the box side (hub + vision-service + dashboard) is DONE and waits only
> on real frames. The ESP32-CAM does ONE job: serve a reliable MJPEG stream + declare.
> It runs **no neural net** (§3.1) — the box thinks.

## 1. Hardware reality (§3.1)
- AI-Thinker ESP32-CAM (OV2640, 4MB PSRAM, no SIMD/NPU). Do **not** attempt on-device
  detection (~1–2 FPS, useless for ID).
- **Brownout is the #1 field failure.** Require a solid 5V/2A supply + short leads;
  consider `WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0)`. Document for whoever mounts it.
- **OTA:** supported (as of cam fw 20260628.2). The dumb-MJPEG build is ~970 KB, so it
  fits a dual-OTA partition table (`min_spiffs.csv`) and the shared core's pull-based
  OTA lands. **One** USB flash is still needed to lay that partition table down (OTA
  can't repartition); every update after is OTA via `./hub publish --bump camera`.

## 2. Transport — stock HTTP MJPEG (§3.2)
Use the canonical `esp_camera` + `camera_httpd` server. Endpoints (the box's contract):

| Endpoint | Port | Purpose |
|---|---|---|
| `GET /stream`  | 81 | `multipart/x-mixed-replace` MJPEG — the continuous feed the vision-service pulls |
| `GET /capture` | 80/81 | single JPEG snapshot |
| `GET /status`  | 80 | sensor config (res/quality/fps), `FW_VERSION`, uptime |
| `GET /control?var=&val=` | 80 | runtime config (framesize/quality/fps) so the box can tune without reflash |

- **Single consumer.** The cam handles ~1 client. The **vision-service is the only
  thing that opens `/stream`**; the dashboard views via the vision-service
  (`/vision/stream/<id>`), never the cam. State this loudly in the firmware README.
- Sensor config (firmware ships these): `FRAMESIZE_UXGA` (1600×1200 — raised from the
  SVGA "to start" baseline for face pixels at room distance; the box crops faces from
  the full-res frame so resolution → ID accuracy), `jpeg_quality 12`, `fb_count = 2`
  (PSRAM), `grab_mode = CAMERA_GRAB_LATEST`, `xclk 20MHz`, flash LED off. All
  runtime-tunable via `/control` — dial framesize down per zone if bandwidth/GPU bound.

## 3. Registration — declare to the hub (§3.3) — **already accepted box-side**
`POST /device-declare` (`name:"camera"`) like the rest of the fleet, with the stream
block. The hub stores it verbatim and the roster surfaces it (`/get-devices`):

```jsonc
{ "name": "camera", "id": "<stable-device-id>", "ip": "<cam-ip>", "zone": "living-room",
  "fw_version": "<FW_VERSION>",
  "stream": { "proto": "mjpeg-http", "port": 81, "path": "/stream",
              "snapshot": "/capture", "res": "UXGA", "fps": 8 } }
```

- The hub validates + stores this (`captureStreamDeclare`): a stream needs at least
  `path`; `proto`/`port` default to `mjpeg-http`/`81` if omitted; refreshed on every
  declare heartbeat (same lifecycle as `ip`).
- **`zone` is critical** — occupancy is per-zone. DECISION (§3.3): assign zone in the
  dashboard after declare (recommended — units interchangeable). The hub already merges
  a dashboard-set `zone` via `/devices-data-set`; firmware need not hardcode it.

## 4. Reliability (§3.4)
- WiFi auto-reconnect + watchdog reboot on stream stall.
- Resolve the hub via mDNS (`_homehub._tcp`, `HomeHubDevice::resolveHub()`), not a
  hardcoded IP (the `192.168.1.199` migration debt in CLAUDE.md still applies to
  `secrets.h::HOME_SERVER`).
- Re-declare periodically (the hub treats declare as a heartbeat).

## 5. Optional stretch (§3.5) — clearly optional
A cheap on-device frame-difference motion flag (SAD over a threshold) emitted as a hint
could let the vision-service drop idle cameras to low-FPS polling (GPU saving on empty
rooms). Build ONLY if §11.1 GPU contention proves real. Do **not** treat ESP motion as
authoritative presence — the box's detector is the source of truth.
