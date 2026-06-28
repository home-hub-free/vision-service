"""vision-service entrypoint — FastAPI app + background supervisor/janitor.

Run: `uvicorn app.main:app --host 0.0.0.0 --port 8130` (see homehub-vision.service).
Behind nginx at `/vision/` (see ../nginx). The dashboard hits the stream + occupancy
+ enrollment + guest routes; the agent reads /occupancy + /who_is_here + /history/*.

Ships runnable with VISION_BACKEND=null / VISION_FACE_BACKEND=null (no torch / no
ROCm): streaming, recording, retention, roster-sync, MQTT producer and all dashboard
plumbing work; identity lights up when a real backend is installed (see ../DECISIONS.md).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import cfg
from .ingest import init_ingestion
from .retention import Janitor
from .routes import enroll, guests, occupancy, streams
from .state import index
from .supervisor import Supervisor

_supervisor: Supervisor | None = None
_janitor: Janitor | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _supervisor, _janitor
    init_ingestion()
    _supervisor = Supervisor()
    _supervisor.start()
    _janitor = Janitor(index)
    _janitor.start()
    print(f"[vision] up on :{cfg.port} (backend={cfg.backend}/{cfg.face_backend}, "
          f"rec={cfg.rec_mode_default}, ingestion={cfg.ingestion_enabled})", flush=True)
    try:
        yield
    finally:
        if _supervisor:
            _supervisor.stop()
        if _janitor:
            _janitor.stop()


app = FastAPI(title="home-hub vision-service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.include_router(streams.router)
app.include_router(occupancy.router)
app.include_router(enroll.router)
app.include_router(guests.router)

# HLS playback (§6/§11.5 alternative delivery) — the recorder writes live.m3u8 here.
os.makedirs(cfg.hls_dir, exist_ok=True)
app.mount("/hls", StaticFiles(directory=cfg.hls_dir), name="hls")


@app.get("/health")
def health():
    from .state import workers
    return {
        "ok": True, "service": "vision-service",
        "backend": cfg.backend, "face_backend": cfg.face_backend,
        "cameras": len(workers), "ingestion": cfg.ingestion_enabled,
        "rec_mode": cfg.rec_mode_default,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=cfg.port)
