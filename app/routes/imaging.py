"""Imaging control routes (CAMERA_ONVIF_CONTROL_PLAN §4) — the sensor-profile knob.

Uses the VIDEO-SOURCE token (`raw_vs1`), never the profile token — the §0 trap.
Values are 0..100 (the MC200's advertised range for all four); `ir_cut`
(ON/OFF/AUTO) is accepted but only applied when the camera actually exposes
IrCutFilter (the MC200's current fw does NOT — verified 2026-07-03; the C110s may).

  GET  /imaging/{cam_id} — current {brightness,saturation,contrast,sharpness[,ir_cut]}
  POST /imaging/{cam_id} — partial update of the same fields (merged, clamped)
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body, HTTPException

from ..onvif import OnvifError
from .ptz import onvif_or_error, run, zone_of

router = APIRouter(prefix="/imaging")


def _imaging_client(cam_id: str):
    client = onvif_or_error(cam_id, need_ptz=False)
    try:
        caps = client.capabilities_cached() or client.capabilities()
    except OnvifError as e:
        raise HTTPException(status_code=503, detail=f"camera unreachable: {e}") from e
    if not caps.get("imaging"):
        raise HTTPException(status_code=409, detail={"imaging": False, "error": "camera has no imaging service"})
    return client


@router.get("/{cam_id}")
def get_imaging(cam_id: str):
    client = _imaging_client(cam_id)
    return {"cam_id": cam_id, "zone": zone_of(cam_id), "imaging": run(client.get_imaging)}


@router.post("/{cam_id}")
def set_imaging(cam_id: str,
                brightness: Optional[float] = Body(None, embed=True),
                saturation: Optional[float] = Body(None, embed=True),
                contrast: Optional[float] = Body(None, embed=True),
                sharpness: Optional[float] = Body(None, embed=True),
                ir_cut: Optional[str] = Body(None, embed=True)):
    updates = {"brightness": brightness, "saturation": saturation,
               "contrast": contrast, "sharpness": sharpness, "ir_cut": ir_cut}
    if all(v is None for v in updates.values()):
        raise HTTPException(status_code=400, detail="no imaging fields to set")
    client = _imaging_client(cam_id)
    return {"ok": True, "cam_id": cam_id, "zone": zone_of(cam_id), "imaging": run(client.set_imaging, updates)}
