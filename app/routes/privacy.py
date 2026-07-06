"""Privacy mode routes — the per-camera "stop watching NOW" switch (app/privacy.py).

  GET  /privacy            — every worker's privacy state (open read, house rule)
  POST /privacy/{cam_id}   — {"on": true|false} → flip it; applies IMMEDIATELY

The POST is LAN-open like the rest of the control seam (ptz/imaging): the dashboard
reaches it through the hub's `/camera/:id/privacy` proxy, which is the auth + audit
boundary (WHO covered the cameras is a record worth keeping). Enforcement lives in
the worker — by the time the response returns, the recorder is closed and the
reader is about to disconnect; streams/snapshots answer 423 while private.
"""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from ..state import privacy, workers

router = APIRouter()


@router.get("/privacy")
def privacy_state():
    """Per-camera privacy for every live worker (+ any persisted id whose camera is
    currently off-roster, so a flag can't hide by unplugging the camera)."""
    cams = {cam_id: privacy.is_private(cam_id) for cam_id in workers}
    for cam_id in privacy.all():
        cams.setdefault(cam_id, True)
    return {"cameras": cams}


@router.post("/privacy/{cam_id}")
def set_privacy(cam_id: str, body: dict = Body(...)):
    w = workers.get(cam_id)
    if w is None:
        raise HTTPException(status_code=404, detail="camera not found / no worker")
    on = bool(body.get("on"))
    privacy.set(cam_id, on)
    if on:
        # Tear down NOW from this thread (recorder closed, frames dropped, occupancy
        # withdrawn) — don't wait for the reader loop to notice.
        getattr(w, "pause_for_privacy", lambda: None)()
    # OFF needs no push: the reader loop resumes within ~1s on its own.
    return {"cam_id": cam_id, "zone": getattr(getattr(w, "cam", None), "zone", None),
            "privacy": on}
