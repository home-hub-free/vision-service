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
  GET /recordings/{cam}/thumb/{seg_id}?token=&t= — a preview frame at offset t        [token]
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse

from ..config import cfg
from ..footage import sync_camera
from ..hub_client import require_user
from ..media_token import sign_clip, verify_clip
from ..state import index, workers

router = APIRouter()

# Segments are immutable once settled and clip URLs are token-stable within a
# 6h bucket (media_token) — tell the browser so: without max-age every hop
# between clips re-downloaded bytes the reviewer had already watched.
CLIP_CACHE = "private, max-age=43200, immutable"
THUMB_CACHE = "private, max-age=86400, immutable"

# Scrub preview frames snap to this grid — hover jitter then reuses the same
# cached jpeg instead of minting one per pixel of pointer movement.
THUMB_STEP_S = 15
# At most this many concurrent ffmpeg extractions; a fast scrub queues briefly
# instead of forking an ffmpeg storm next to the inference threads.
_thumb_gate = threading.Semaphore(2)


def _segment_path(cam_id: str, seg_id: int, token: str) -> str:
    """Shared clip/thumb guard: token → index row → real file under rec_dir."""
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
    return path


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
    picker). Bearer-gated — footage is only browsable by a signed-in member.
    Syncs each camera's index from disk first (footage.py) so a just-finished
    chunk is browsable immediately — the files are the truth, not the DB."""
    require_user(authorization)
    cams = []
    for c in _recording_cams():
        sync_camera(index, c["id"], c["zone"])
        cams.append({**c, "days": index.recording_days(c["id"])})
    return {"cameras": cams}


@router.get("/recordings/{cam_id}/segments")
def segments(cam_id: str, start: float, end: float,
             authorization: Optional[str] = Header(None)):
    """The segments overlapping [start, end] for one camera (a day, usually), each with
    a signed clip URL and the identity/event markers that fall inside it — the timeline
    pins ("David entered") the reviewer scrubs between. Bearer-gated."""
    require_user(authorization)
    zone = next((c["zone"] for c in _recording_cams() if c["id"] == cam_id), None)
    sync_camera(index, cam_id, zone or "_")
    segs = index.segments_between(cam_id, start, end)
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
    path = _segment_path(cam_id, seg_id, token)
    return FileResponse(path, media_type="video/mp4",
                        filename=os.path.basename(path),
                        headers={"Cache-Control": CLIP_CACHE})


THUMB_HEIGHTS = (180, 360)  # bubble-size and full-surface preview-size


@router.get("/recordings/{cam_id}/thumb/{seg_id}")
def thumb(cam_id: str, seg_id: int, token: str = Query(...), t: float = Query(0.0),
          h: int = Query(180)):
    """A preview frame `t` seconds into a segment — the timeline's scrub bubble,
    the player's poster, and the tap-through still overlay. Same token as the
    clip (bound to the segment id). Frames snap to a {THUMB_STEP_S}s grid and are
    extracted once (ffmpeg keyframe seek, ~100ms on a faststart file) into a disk
    cache the janitor ages out alongside the segments themselves. `h` picks the
    rendition height (whitelisted — it's a cache key, not a free transcoder)."""
    path = _segment_path(cam_id, seg_id, token)
    if h not in THUMB_HEIGHTS:
        h = THUMB_HEIGHTS[0]
    at = max(0, int(t // THUMB_STEP_S) * THUMB_STEP_S)
    cache_dir = os.path.join(cfg.thumb_dir, cam_id)
    cached = os.path.join(cache_dir, f"{seg_id}-{at}-{h}.jpg")
    if not os.path.isfile(cached):
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{cached}.{os.getpid()}.tmp"
        with _thumb_gate:
            if not os.path.isfile(cached):  # raced another request for the same frame
                res = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-ss", str(at), "-i", path, "-frames:v", "1",
                     "-vf", f"scale=-2:{h}", "-q:v", "7", "-f", "image2", tmp],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                # -ss past EOF (open-ended last chunk) produces no frame — retry at 0
                # so a thumb always exists for a real segment.
                if (res.returncode != 0 or not os.path.isfile(tmp)
                        or os.path.getsize(tmp) == 0) and at > 0:
                    res = subprocess.run(
                        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                         "-i", path, "-frames:v", "1",
                         "-vf", f"scale=-2:{h}", "-q:v", "7", "-f", "image2", tmp],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                if res.returncode != 0 or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    raise HTTPException(status_code=502, detail="thumbnail extraction failed")
                os.replace(tmp, cached)
    return FileResponse(cached, media_type="image/jpeg",
                        headers={"Cache-Control": THUMB_CACHE})
