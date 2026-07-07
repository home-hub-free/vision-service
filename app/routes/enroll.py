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


@router.get("/faces/health")
def faces_health():
    """The gallery's identity-health tripwire (see app/face_audit.py): live
    member-vs-member confusability + the last scheduled audit report. The one
    number to watch is `member_similarity` — distinct people read ~0.0–0.3; both
    2026-07 pollution incidents sat at 0.45+ for days with nobody looking."""
    import json as _json
    last_raw = gallery.get_kv("face_audit_last")
    return {
        "member_similarity": gallery.member_similarity(),
        "folds_frozen": gallery.folds_frozen,
        "clusters_24h": gallery.clusters_created_since(24.0),
        "smear_alarm_cos": cfg.face_smear_alarm_cos,
        "last_audit": _json.loads(last_raw) if last_raw else None,
    }


@router.post("/faces/audit")
def faces_audit(authorization=Header(None)):
    """Run a full audit pass NOW (smear alarm + promotion coherence + churn) —
    the scheduled auditor's logic, on demand, e.g. right after a cleanup or
    re-enrollment to lift the fold freeze without waiting for the next cycle."""
    require_user(authorization)  # admin-gated
    from ..face_audit import run_audit
    return run_audit(gallery)


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


@router.get("/faces/{owner}/captures")
def list_captures(owner: str, limit: int = 200):
    """The capture ledger for one identity — every archived crop behind their
    recognition decisions (the "soup ingredients"). Newest first; the dashboard's
    photo-collection view renders these with delete controls so a polluted set can
    be cleaned by hand before a rebuild."""
    rows = gallery.captures(owner)
    total = len(rows)
    out = []
    for r in rows[:max(1, min(limit, 1000))]:
        out.append({"id": r["id"], "ts": r["ts"], "kind": r["kind"],
                    "score": r["score"], "reinforced": r["reinforced"],
                    "image": f"faces/captures/{r['id']}/image" if r["path"] else None})
    return {"owner": owner, "total": total, "captures": out}


@router.get("/faces/captures/{capture_id}/image")
def capture_image(capture_id: int):
    data = gallery.capture_image(capture_id)
    if not data:
        raise HTTPException(status_code=404, detail="no image for this capture")
    # Immutable by id — let the browser cache the grid instead of re-pulling crops.
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "private, max-age=86400"})


@router.delete("/faces/captures/{capture_id}")
def delete_capture(capture_id: int, authorization=Header(None)):
    """Remove one photo (row + file) from an identity's ingredient set — the manual
    clean before a rebuild. Deleting never touches the live centroid by itself."""
    require_user(authorization)  # admin-gated
    if not gallery.delete_capture(capture_id):
        raise HTTPException(status_code=404, detail="capture not found")
    return {"ok": True, "id": capture_id}


@router.post("/faces/{user_id}/rebuild")
def rebuild_member(user_id: str, name: Optional[str] = Body(None, embed=True),
                   authorization=Header(None)):
    """Re-do the soup: REPLACE the member's face centroid with the plain mean of
    every capture still archived for them. Delete the wrong photos first — the
    rebuild uses exactly what remains. Same whole-household trust as the review
    flow (any member can fix any profile)."""
    require_user(authorization)  # admin-gated
    samples = gallery.rebuild_member_from_captures(user_id, name=name)
    if samples is None:
        raise HTTPException(status_code=409,
                            detail="no captures archived for this member yet — "
                                   "nothing to rebuild from")
    return {"ok": True, "user_id": user_id, "samples": samples}


@router.get("/faces/thumb/{label_id}")
def thumb(label_id: str):
    """The stored face image for a label (`users.id` member or `guest:N` cluster) — the
    face the dashboard shows beside the label. 404 when none captured yet."""
    data = gallery.get_thumb(label_id)
    if not data:
        raise HTTPException(status_code=404, detail="no face captured for this label")
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache"})


# Why an enrollment photo was refused → what the guided flow tells the user to DO
# about it. Enrollment samples are the anchors every identity decision leans on, so
# refusing junk here (instead of politely folding it in) is the whole ballgame —
# the 2026-07-07 pollution started with profile-view 87px enroll frames.
_ENROLL_REJECTIONS = {
    "no_face": "I can't see a face in that shot — center your face in the oval.",
    "multiple_faces": "I see more than one face — enroll one person at a time.",
    "too_small": "Your face is too small in the frame — move closer to the camera.",
    "off_angle": "Face the camera straight on for this one.",
    "blurry": "That one came out blurry — hold still and try again.",
    "low_confidence": "I can't read your face clearly — try facing the camera in better light.",
}


@router.post("/faces/enroll")
async def enroll(image: UploadFile = File(...), authorization: Optional[str] = Header(None)):
    user = user_from_token(authorization)  # enroll for the AUTHENTICATED user only
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image upload")
    emb, reason = enroll_embedding(data)
    if emb is None:
        raise HTTPException(status_code=422,
                            detail=_ENROLL_REJECTIONS.get(reason or "no_face",
                                                          _ENROLL_REJECTIONS["no_face"]))
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
