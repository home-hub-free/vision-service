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

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from fastapi.responses import Response

from ..config import cfg
from ..hub_client import user_from_token
from ..perception import enroll_embedding, thumbnail_jpeg
from ..state import gallery

router = APIRouter()


@router.get("/faces/profiles")
def profiles():
    return {"profiles": gallery.profiles(), "backend": cfg.face_backend}


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
