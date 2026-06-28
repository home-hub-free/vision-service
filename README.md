# vision-service

Local-first camera perception for home-hub. Implements CAMERA_VISION_PLAN.md §4/§9:
the ESP32-CAM is a dumb MJPEG sensor; **all intelligence runs here** (the box). This
service is the single consumer of each camera stream and turns pixels into a small,
debounced **occupancy/identity world-model** that flows to the agent through the
existing ingestion seam (MQTT → Node-RED → memory + agent). The agent never sees
frames — only digested events + a queryable snapshot.

```
ESP32-CAM ──HTTP MJPEG──► vision-service ──► occupancy/identity world-model
 (/stream :81)             decode → YOLO person → ByteTrack         │
                            → SCRFD face → ArcFace embed            ├─► MQTT (homehub/<zone>/<cam>/person|occupancy, meta.identity)
                            → match gallery {household, guest}      ├─► dashboard: annotated stream + WHO is here + enroll + guests
                            → recording + retention + event index   └─► agent: /occupancy /who_is_here /history/* (pull) + edges (push)
```

Port **:8130**, behind nginx at `/vision/`. Parallels the other box services
(memory-service, tts-service, voice-pipeline, speaker-id).

## Run (null/CPU build — M0, no GPU, no torch)
```bash
./setup.sh                 # venv + web/MQTT deps + .env
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8130
# or: sudo cp homehub-vision.service /etc/systemd/system && sudo systemctl enable --now homehub-vision
```
The null build streams, records, prunes, syncs the roster, and serves all the
dashboard plumbing. Identity lights up when a real backend is installed (see below).

## Milestones (each ships standalone — §10)
- **M0** stream + view + record (`VISION_BACKEND=null`) — works today.
- **M1** presence + occupancy: `pip install ultralytics`, `VISION_BACKEND=ultralytics`.
- **M2** household identity: `pip install insightface onnxruntime`, `VISION_FACE_BACKEND=insightface` + Face-ID enroll.
- **M3** guests: auto-clustering is on; review/promote via `/vision/guests`.
- **M4** recording + history: `/vision/history/*` over the event index.

## Endpoints
| Route | Purpose |
|---|---|
| `GET /health` | liveness + backend/camera summary |
| `GET /stream/{cam}` · `/stream/{cam}/raw` · `/snapshot/{cam}` | live view (annotated / raw) — the dashboard views HERE, never the cam |
| `GET /hls/{cam}/live.m3u8` | HLS playback (recorder output) |
| `GET /occupancy` · `/who_is_here?zone=` | the agent's PULL world-model snapshot |
| `GET /history/who-came-by` · `/history/events` · `/history/segment` | answerable history from the event index (never video) |
| `POST /faces/enroll` · `/faces/forget` · `GET /faces/profiles` | Face-ID enrollment (dashboard, bearer-auth) |
| `GET /people` · `GET /faces/thumb/{id}` | every labelled person (household + default-id'd guests) + their captured face |
| `POST /guests/{id}/name` · `/guests/{id}/promote` · `DELETE /guests/{id}` | admin labels a person / links to a member (bearer-auth) |

**Label everyone by default + surface faces (§6):** every detected person is
auto-labelled with a stable id (`guest:N`) + a friendly "Person N" label + a captured
face thumbnail. The dashboard People panel (Settings → Household) shows each face beside
its label so the **admin** (any authenticated member) can name them or link them to a
household member — those write endpoints are bearer-auth gated.

## Contracts (locked — §12)
1. **Declare** `stream` block — stored by the hub (`captureStreamDeclare`), read via `/get-devices`.
2. **Ingestion** — `homehub/<zone>/<cam>/person|occupancy`, `source:"device"`, `meta.identity` (§5.1).
3. **`EventMeta.identity`** — additive hub type (`server/src/clients/ingestion.ts`).
4. **Roster/enroll** — `/get-devices` + `/auth/users` via `HUB_SERVICE_TOKEN`; faces land HERE, not the hub.
5. **Salience** — push edges (`person_entered` …) vs pull (`occupancy`), gated in Node-RED `mqtt-to-agent`.

## Tests
```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/      # pure-logic: occupancy debounce, gallery, mjpeg, retention, static-cams
```

## Validation spike (§2 image-quality go/no-go) — no firmware required

The service is **camera-agnostic**: it pulls any MJPEG/RTSP URL. So you can validate the
*whole box pipeline* (detect → track → face → ID) against a known-good camera **before**
investing in ESP32-CAM firmware — the cheapest de-risk and the answer to DECISIONS #3.

Inject a non-declaring source two ways:

**A — `VISION_STATIC_CAMERAS` escape hatch (no hub write).** A comma-list of
`id@zone@url` (the URL is taken verbatim — http MJPEG or `rtsp://…`):
```bash
# a laptop/USB webcam served as MJPEG, an IP/RTSP cam, or the bare ESP32-CAM's /stream
VISION_STATIC_CAMERAS="lab@sala@http://192.168.1.50:81/stream" \
  .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8130
```
These augment the roster (the roster wins on an id clash) and survive a hub outage.

**B — a real declare (exercises the hub path too).** Hand-build the `stream` block:
```bash
curl -sX POST http://127.0.0.1:8088/device-declare -H 'Content-Type: application/json' -d '{
  "id":"lab-cam","name":"camera","fw_version":"spike",
  "stream":{"proto":"mjpeg-http","port":81,"path":"/stream","snapshot":"/capture","res":"SVGA","fps":10}}'
curl -s http://127.0.0.1:8088/get-devices | jq '.[] | select(.deviceCategory=="camera")'
# the hub reads the IP from the request — declare FROM the cam's host, or assign zone/ip in the dashboard
```

Then with a real backend on (below), **[HUMAN/HW]** stand at realistic room
distance/angle/lighting and check `GET /vision/who_is_here` + the annotated
`GET /vision/stream/<id>`. Record the verdict in **DECISIONS.md #3**.

## M1/M2: enabling the real backends + the ROCm spike

Defaults run the `null`/CPU build (M0). To turn the brain on:

```bash
# M1 person-detect:  pip install ultralytics ; VISION_BACKEND=ultralytics
# M2 face ID:        pip install insightface onnxruntime ; VISION_FACE_BACKEND=insightface
#                    (Face-ID enroll in dashboard Settings → Household)
.venv/bin/pip install opencv-python-headless numpy   # frame decode + annotation
```

**ROCm spike (DECISIONS #2 — the one real engineering risk, mirrors the TTS ROCm
friction).** In a throwaway venv, try both runtimes and keep whichever loads a model
and runs one frame without exploding; record the winner + the working install line in
`requirements.txt`:
- **torch-ROCm** (ultralytics-native): `pip install --index-url
  https://download.pytorch.org/whl/rocm6.2 torch torchvision` → `VISION_DEVICE=cuda`
  (ROCm maps onto the cuda device string).
- **onnxruntime-ROCm** (export YOLO→ONNX): `VISION_ORT_PROVIDERS=ROCMExecutionProvider,CPUExecutionProvider`.
- **CPU fallback is valid** (`VISION_DEVICE=cpu`) — face stages are small; the backend
  falls back to `null` on import failure, so a bad ROCm day degrades to presence-less
  streaming, not a crash.

Then **[HUMAN/HW]** measure voice **TTFA with vision running** vs idle and tune
`VISION_DETECT_FPS` down until within target — record the cap + target in DECISIONS #1.
For the GPU path, uncomment `HIP_VISIBLE_DEVICES=0` + the `render`/`video` groups in
`homehub-vision.service` (mirror the TTS unit).

See **DECISIONS.md** for the open calls (GPU/ROCm, retention numbers, fusion, …),
each wired with a safe default. Biometrics stay on the box; the hub never holds them.
