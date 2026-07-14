---
title: AGENTS.md — vision-service
summary: Read-first orientation for vision-service — first principles (agent never sees frames, one identity space, runnable without ML, biometrics local), layout, conventions.
status: LIVE
owner: vision-service
updated: 2026-06-28
tags: [vision, camera, agents]
---

# AGENTS.md — vision-service

Box-side camera perception. Standalone Python service (FastAPI/uvicorn, :8130), its own
future GitHub repo under `home-hub-free` (co-located in `/opt/home-hub-free` for
convenience, gitignored by the root). Parallels memory-service / tts-service /
voice-pipeline / speaker-id.

## Read first
- `../docs/CAMERA_VISION_PLAN.md` — the design this implements (§4 service, §5 hub
  seam, §8 salience, §9 recording, §11 open decisions, §12 contracts).
- `DECISIONS.md` — every stubbed human decision + where it plugs in.
- The hub's ingestion seam (`../server/src/clients/ingestion.ts`) — we mirror its
  topic scheme + fire-and-forget semantics as an INDEPENDENT producer (§5.2).

## First principles (carry these)
1. **The agent never sees frames.** Pixels stay on the box; only a digested
   occupancy/identity world-model leaves (edges push, snapshot pull).
2. **One identity space.** Face resolves to the SAME `users.id` + `data.user` envelope
   as voice, `via:"face"`. A person known by voice and face is one person, two signals.
3. **Runnable without ML.** `VISION_BACKEND=null` is a first-class mode (M0). Never let
   a missing/broken model or ROCm crash the box — backends fall back to null.
4. **Biometrics local.** Gallery (household + guests) is vision-service sqlite. The hub
   never holds embeddings; it only brokers the roster + identity envelope.
5. **Best-effort telemetry, never control.** MQTT publishes are QoS-0, dropped (not
   buffered) if the broker is down; perception never throws on a publish error.

## Layout
```
app/
  main.py        FastAPI app + lifespan (supervisor + janitor)
  config.py      all env knobs (every §11 decision = a default here)
  supervisor.py  roster sync + per-camera worker lifecycle
  camera.py      per-camera pipeline thread (pull→detect→track→face→resolve→occupancy→emit→record)
  perception.py  Detector/FaceEngine seam: null default + ultralytics/insightface (guarded)
  occupancy.py   per-zone debounce + salient edges + who_is_here snapshot (pure, tested)
  gallery.py     household enroll + guest clustering + promote (sqlite, tested)
  ingest.py      MQTT producer (person/occupancy + meta.identity)
  recorder.py    ffmpeg tee mp4+HLS, modes + pre-roll
  retention.py   age/disk-cap janitor (tested)
  index_db.py    recordings + event index (answerable history)
  hub_client.py  roster + token-auth (read-only hub coupling)
  routes/        streams, occupancy/history, faces (enroll), guests
tests/           pure-logic pytest (no models needed)
```

## Conventions
- Match the speaker-service posture: a runnable `null`/`stub` backend so plumbing + UI
  are built/tested without heavy ML deps; real backend behind an env flag.
- New env knobs go in `config.py` with a docstring citing the plan section, and a row
  in `DECISIONS.md` if they encode a human decision.
- Pure logic (occupancy, gallery match, retention sweep) stays I/O-free + unit-tested.
