"""PTZ control routes (CAMERA_ONVIF_CONTROL_PLAN §2) — pan/tilt + named views.

Open on :8130 like the rest of the service; the HUB proxies these behind
requireAuth/HUB_SERVICE_TOKEN and stamps the audit event (`/camera/:id/ptz/*`) —
one place speaks SOAP (here), one place gates actuation (the hub).

  GET    /ptz/{cam_id}/status          — current pan/tilt + move state
  GET    /ptz/{cam_id}/presets         — named views [{token,name,x,y}]
  POST   /ptz/{cam_id}/goto            — {token} recall a preset
  POST   /ptz/{cam_id}/preset          — {name} save CURRENT aim as a preset
  DELETE /ptz/{cam_id}/preset/{token}  — remove a preset
  POST   /ptz/{cam_id}/move            — {vx,vy,ttl_ms} timed nudge (auto-stops)
  POST   /ptz/{cam_id}/stop            — immediate stop (panic / button release)

Errors: 404 unknown camera · 409 not ONVIF/PTZ-capable (body carries {ptz:false})
· 503 camera unreachable · 502 camera answered with a SOAP fault.
"""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from ..onvif import OnvifClient, OnvifError, get_onvif
from ..state import workers

router = APIRouter(prefix="/ptz")


def zone_of(cam_id: str) -> str:
    """The camera's zone, for the hub's audit emit (it proxies these routes and
    stamps the actuation event; static cameras exist only in THIS roster)."""
    w = workers.get(cam_id)
    return getattr(getattr(w, "cam", None), "zone", "") or ""


def onvif_or_error(cam_id: str, need_ptz: bool = True) -> OnvifClient:
    """Resolve cam_id → live OnvifClient, translating every miss into the right
    HTTP error. The capability check uses the CACHED probe when present and probes
    inline otherwise (first control touch after boot pays it once)."""
    if cam_id not in workers:
        raise HTTPException(status_code=404, detail="camera not found / no worker")
    client = get_onvif(cam_id)
    if client is None:
        raise HTTPException(status_code=409, detail={"ptz": False, "error": "camera is not ONVIF-capable"})
    if need_ptz:
        try:
            caps = client.capabilities_cached() or client.capabilities()
        except OnvifError as e:
            raise HTTPException(status_code=503, detail=f"camera unreachable: {e}") from e
        if not caps.get("ptz"):
            raise HTTPException(status_code=409, detail={"ptz": False, "error": "camera has no PTZ"})
    return client


def run(fn, *args, **kwargs):
    """Run one ONVIF verb, mapping OnvifError → transport 503 / camera-fault 502."""
    try:
        return fn(*args, **kwargs)
    except OnvifError as e:
        raise HTTPException(status_code=502 if e.fault else 503, detail=str(e)) from e


@router.get("/{cam_id}/status")
def status(cam_id: str):
    client = onvif_or_error(cam_id)
    return {"cam_id": cam_id, "zone": zone_of(cam_id), "ptz": True, **run(client.get_status)}


@router.get("/{cam_id}/presets")
def presets(cam_id: str):
    client = onvif_or_error(cam_id)
    return {"cam_id": cam_id, "zone": zone_of(cam_id), "presets": run(client.get_presets)}


@router.post("/{cam_id}/goto")
def goto(cam_id: str, token: str = Body(..., embed=True)):
    client = onvif_or_error(cam_id)
    run(client.goto_preset, token)
    return {"ok": True, "cam_id": cam_id, "zone": zone_of(cam_id), "token": token}


@router.post("/{cam_id}/preset")
def save_preset(cam_id: str, name: str = Body(..., embed=True)):
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="preset needs a non-empty name")
    client = onvif_or_error(cam_id)
    token = run(client.set_preset, name)
    return {"ok": True, "cam_id": cam_id, "zone": zone_of(cam_id), "token": token, "name": name}


@router.delete("/{cam_id}/preset/{token}")
def delete_preset(cam_id: str, token: str):
    client = onvif_or_error(cam_id)
    run(client.remove_preset, token)
    return {"ok": True, "cam_id": cam_id, "zone": zone_of(cam_id), "token": token}


@router.post("/{cam_id}/move")
def move(cam_id: str,
         vx: float = Body(0.0, embed=True),
         vy: float = Body(0.0, embed=True),
         ttl_ms: int = Body(500, embed=True)):
    """Timed nudge: velocities in the camera's -1..1 generic space; the move is
    ALWAYS auto-stopped (ttl clamped ≤ cfg.ptz_max_ttl_s inside move_timed)."""
    client = onvif_or_error(cam_id)
    ttl = run(client.move_timed, vx, vy, max(ttl_ms, 0) / 1000.0)
    return {"ok": True, "cam_id": cam_id, "zone": zone_of(cam_id), "vx": vx, "vy": vy, "ttl_s": ttl}


@router.post("/{cam_id}/stop")
def stop(cam_id: str):
    client = onvif_or_error(cam_id)
    run(client.stop)
    return {"ok": True, "cam_id": cam_id, "zone": zone_of(cam_id)}
