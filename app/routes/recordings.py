"""Footage review (§9.5) — browse + play back the archived recordings.

Only the IP cams that actually record (a full-quality RTSP MAIN stream → `record_url`;
camera.py builds their recorder, everything else is `mode="off"`) produce segments, so
this surface lists exactly those cameras. It reads the segment index (`index_db`,
populated by `recorder.py`) and streams the archived mp4 files under `cfg.rec_dir`.

Auth (DECISION — playback is an authenticated dashboard feature): the LIST routes are
bearer-gated with `require_user` (the hub session, same gate as Face-ID enroll). The
CLIP route is fetched by an HTML `<video src>` GET which can't send an Authorization
header, so it's gated by a short-TTL signed token minted by the (bearer-gated) segments
route — see media_token.py.

  GET /recordings/cameras                       — cameras that record + their days   [auth]
  GET /recordings/{cam}/segments?start=&end=    — a day's segments + event markers    [auth]
  GET /recordings/{cam}/clip/{seg_id}?token=    — stream the archived mp4 (Range OK)  [token]
"""
from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse

from ..config import cfg
from ..hub_client import require_user
from ..media_token import sign_clip, verify_clip
from ..state import index, workers

router = APIRouter()


def _recording_cams() -> list:
    """Live workers whose recorder is on (records == True), i.e. the IP-cam fleet."""
    out = []
    for cam_id, w in workers.items():
        st = w.status() if hasattr(w, "status") else {}
        if st.get("records"):
            out.append({"id": cam_id, "name": getattr(getattr(w, "cam", None), "name", None),
                        "zone": st.get("zone")})
    return out


@router.get("/recordings/cameras")
def recording_cameras(authorization: Optional[str] = Header(None)):
    """Every camera that records, with the distinct days it has footage for (the day
    picker). Bearer-gated — footage is only browsable by a signed-in member."""
    require_user(authorization)
    cams = []
    for c in _recording_cams():
        cams.append({**c, "days": index.recording_days(c["id"])})
    return {"cameras": cams}


@router.get("/recordings/{cam_id}/segments")
def segments(cam_id: str, start: float, end: float,
             authorization: Optional[str] = Header(None)):
    """The segments overlapping [start, end] for one camera (a day, usually), each with
    a signed clip URL and the identity/event markers that fall inside it — the timeline
    pins ("David entered") the reviewer scrubs between. Bearer-gated."""
    require_user(authorization)
    segs = index.segments_between(cam_id, start, end)
    zone = next((c["zone"] for c in _recording_cams() if c["id"] == cam_id), None)
    out = []
    for s in segs:
        markers = [
            {"ts": e["ts"], "edge": e["edge"], "identity": e["identity"]}
            for e in index.events_between(s["start"], s["end"] or time.time(), zone)
        ]
        out.append({
            "id": s["id"], "start": s["start"], "end": s["end"], "duration": s["duration"],
            "clip": f"recordings/{cam_id}/clip/{s['id']}?token={sign_clip(s['id'])}",
            "events": markers,
        })
    return {"cam_id": cam_id, "segments": out}


@router.get("/recordings/{cam_id}/clip/{seg_id}")
def clip(cam_id: str, seg_id: int, token: str = Query(...)):
    """Stream one archived segment. Token-gated (see module docstring) — a `<video>` GET
    can't carry a bearer, so the bearer-gated segments route mints this token. FileResponse
    honours Range requests, so the player can seek. Guards path traversal by asserting the
    resolved file lives under cfg.rec_dir."""
    if not verify_clip(seg_id, token):
        raise HTTPException(status_code=403, detail="invalid or expired clip token")
    seg = index.segment_by_id(seg_id)
    if seg is None or seg.get("cam_id") != cam_id:
        raise HTTPException(status_code=404, detail="segment not found")
    path = os.path.realpath(seg["file"])
    rec_root = os.path.realpath(cfg.rec_dir)
    if os.path.commonpath([path, rec_root]) != rec_root:
        raise HTTPException(status_code=403, detail="segment outside recordings root")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="segment file pruned")
    return FileResponse(path, media_type="video/mp4",
                        filename=os.path.basename(path))
