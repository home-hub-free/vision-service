"""Camera-control summary — ONE call that tells a UI what it can draw for a camera.

The dashboard's camera tile needs {is it ONVIF? PTZ? imaging? what presets exist?
where is it aimed? current imaging values} before it renders controls. Rather than
four round-trips (through the hub proxy each), this aggregates them:

  GET /camctl/{cam_id} →
    { cam_id, zone, onvif: null | {ptz,imaging,events},   # null = not an ONVIF cam
      reachable, status?, presets?, imaging? }

Degrades per-capability (plan §1 rule 3): a fixed C110 comes back with
onvif.ptz=false and no presets — the tile simply doesn't draw a D-pad. An
unreachable camera answers `reachable:false` with whatever is cached, 200 — the
tile shows its normal offline state; control taps will surface the 503s.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..onvif import OnvifError, get_onvif
from ..state import workers
from .ptz import zone_of

router = APIRouter()


@router.get("/camctl/{cam_id}")
def camctl(cam_id: str):
    if cam_id not in workers:
        raise HTTPException(status_code=404, detail="camera not found / no worker")
    client = get_onvif(cam_id)
    if client is None:
        return {"cam_id": cam_id, "zone": zone_of(cam_id), "onvif": None, "reachable": False}

    out = {"cam_id": cam_id, "zone": zone_of(cam_id), "onvif": None, "reachable": False}
    try:
        caps = client.capabilities()
    except OnvifError:
        out["onvif"] = client.capabilities_cached()  # may still be None
        return out
    out["onvif"] = caps
    out["reachable"] = True
    # Best-effort detail blocks — a single flaky verb must not 500 the summary.
    if caps.get("ptz"):
        try:
            out["status"] = client.get_status()
            out["presets"] = client.get_presets()
        except OnvifError:
            out["reachable"] = False
    if caps.get("imaging"):
        try:
            out["imaging"] = client.get_imaging()
        except OnvifError:
            out["reachable"] = False
    return out
