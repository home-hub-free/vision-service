"""Face-ID enrollment (§5.3/§6) — the dashboard pushes face samples to the
vision-service (NOT the hub; the hub never holds biometrics — §11.8 decision = here).

Mirrors the voiceprint enroll contract exactly (voice-pipeline speaker_service.py):
the enrolling user is authenticated by their dashboard bearer token (validated against
the hub `/auth/me`), and the embedding is stored keyed to THAT user's `users.id`. Same
UX language as "Voice ID".

  POST /faces/enroll   multipart `image`  (Authorization: Bearer <dashboard token>)
  POST /faces/forget   (Authorization: Bearer <token>)
  GET  /faces/profiles (no vectors — just who's enrolled)
"""
from __future__ import annotations

from typing import Optional

from typing import Any, Dict

from fastapi import APIRouter, Body, File, Header, HTTPException, UploadFile
from fastapi.responses import Response

from ..config import cfg
from ..hub_client import require_user, user_from_token
from ..perception import enroll_embedding, thumbnail_jpeg
from ..state import gallery

router = APIRouter()


@router.get("/faces/profiles")
def profiles():
    return {"profiles": gallery.profiles(), "backend": cfg.face_backend}


@router.get("/faces/thresholds")
def get_thresholds():
    """The recognition thresholds the auto-heal / match / suggest ladder runs on, each
    with its effective value, the code default, and whether it's overridden — so the
    household can see (and tune) how eagerly faces are matched and auto-merged."""
    return {"thresholds": gallery.thresholds()}


@router.post("/faces/thresholds")
def set_thresholds(updates: Dict[str, Any] = Body(..., embed=True),
                   authorization=Header(None)):
    """Adjust thresholds live (persisted in the gallery DB, read by the resolver on the
    next face). A value of null / "default" clears an override back to the code default."""
    require_user(authorization)  # admin-gated
    return {"thresholds": gallery.set_thresholds(updates)}


@router.get("/faces/{user_id}/clusters")
def member_clusters(user_id: str):
    """Every image folded into a household member's face profile (by auto-heal or a
    manual promote), each with its captured thumb + how well it still matches the
    member — the audit trail so a reviewer can catch a wrong auto-heal and detach it."""
    clusters = []
    for c in gallery.member_clusters(user_id):
        clusters.append({**c,
                         "thumb": f"faces/thumb/{c['guest_id']}" if c.get("has_thumb") else None})
    return {"user_id": user_id, "clusters": clusters}


@router.post("/faces/clusters/{guest_id}/detach")
def detach_cluster(guest_id: str, authorization=Header(None)):
    """"That one wasn't me" — un-merge a cluster from the member it was auto-healed into,
    block it from healing back, and send it to the review queue for a fresh decision."""
    require_user(authorization)  # admin-gated
    member = gallery.detach_cluster(guest_id)
    if member is None:
        raise HTTPException(status_code=404, detail="cluster not found or not a member promotion")
    return {"ok": True, "guest_id": guest_id, "detached_from": member}


@router.get("/faces/thumb/{label_id}")
def thumb(label_id: str):
    """The stored face image for a label (`users.id` member or `guest:N` cluster) — the
    face the dashboard shows beside the label. 404 when none captured yet."""
    data = gallery.get_thumb(label_id)
    if not data:
        raise HTTPException(status_code=404, detail="no face captured for this label")
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache"})


@router.post("/faces/enroll")
async def enroll(image: UploadFile = File(...), authorization: Optional[str] = Header(None)):
    user = user_from_token(authorization)  # enroll for the AUTHENTICATED user only
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image upload")
    emb = enroll_embedding(data)
    if emb is None:
        raise HTTPException(status_code=422, detail="no face detected in image")
    # Store a downscaled copy of the enroll image as the member's face thumbnail, so the
    # dashboard People roster shows their face beside their name.
    samples = gallery.enroll(user["id"], user.get("displayName") or user.get("name"),
                             emb, thumb=thumbnail_jpeg(data))
    return {"ok": True, "user_id": user["id"], "samples": samples, "backend": cfg.face_backend}


@router.post("/faces/forget")
def forget(authorization: Optional[str] = Header(None)):
    user = user_from_token(authorization)
    gallery.forget(user["id"])
    return {"ok": True, "user_id": user["id"]}
