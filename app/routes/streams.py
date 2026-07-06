"""Live view (§6) — the dashboard streams from HERE, never from the camera.

The ESP32-CAM serves ~1 client reliably (§3.2); the worker is that client, and the
dashboard views the worker's re-served frames (annotated with boxes + names when a
real perception backend is on). A second client hitting the cam directly would stall
the stream — so the dashboard MUST use these endpoints.

DECISION (§6/§11.5 stream delivery): default here is an MJPEG proxy (lowest latency,
the service already has frames). HLS `<video>` (lower CPU, the recorder already emits
`live.m3u8`) is available under /hls/<camId>/live.m3u8 via static mount in main.py.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from ..mjpeg import MJPEG_BOUNDARY
from ..state import privacy, workers

router = APIRouter()


def _worker(cam_id: str):
    w = workers.get(cam_id)
    if w is None:
        raise HTTPException(status_code=404, detail="camera not found / no worker")
    # Privacy mode (app/privacy.py): the worker holds no frames while private, but
    # answer 423 rather than a stalled multipart so a viewer sees WHY it's dark.
    if privacy.is_private(cam_id):
        raise HTTPException(status_code=423, detail="camera is in privacy mode")
    return w


@router.get("/stream/{cam_id}")
def stream(cam_id: str):
    """Annotated MJPEG (boxes + names when available, else raw relay)."""
    w = _worker(cam_id)
    return StreamingResponse(
        w.mjpeg_generator(annotated=True),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
    )


@router.get("/stream/{cam_id}/raw")
def stream_raw(cam_id: str):
    w = _worker(cam_id)
    return StreamingResponse(
        w.mjpeg_generator(annotated=False),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
    )


@router.get("/snapshot/{cam_id}")
def snapshot(cam_id: str):
    w = _worker(cam_id)
    frame = w.latest_annotated or w.latest_raw
    if frame is None:
        raise HTTPException(status_code=503, detail="no frame yet")
    return Response(content=frame, media_type="image/jpeg")
