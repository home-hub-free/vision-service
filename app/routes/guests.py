"""People roster + admin labeling (§4.3/§6) — "label everyone by default id, surface
with faces, so the admin can label."

Every detected person is auto-labelled with a default id (`guest:N`) + a friendly
"Person N" label + a captured face thumbnail (see gallery). This module surfaces that
roster and lets the **admin** (any authenticated household member — flat household
model) name a person or promote them into a household member's face gallery.

  GET    /people                    — EVERY labelled person (household + guests) + faces
  GET    /people/review             — the "Is this you?" queue (tiers; auto-heals on read)
  GET    /guests?recurring=1        — guest clusters only (back-compat)
  POST   /guests/{id}/promote       — {user_id, name} → into a household member's gallery   [auth]
  POST   /guests/{id}/name          — {name} → name a guest WITHOUT promoting (stays guest) [auth]
  POST   /guests/{id}/merge         — {into} → fold this cluster into a NAMED guest         [auth]
  POST   /guests/{id}/reject        — {user_id} → "not me/them": never suggest that identity [auth]
  DELETE /guests/{id}               — discard a cluster                                      [auth]

The face thumbnails are served by GET /faces/thumb/{id} (see enroll.py).
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body, Header, HTTPException

from ..hub_client import require_user
from ..state import gallery

router = APIRouter()


@router.get("/people")
def people():
    """Every labelled person — household members + every detected guest cluster — each
    with its default label + a `thumb` URL (when a face was captured). The dashboard
    People panel renders this so the admin can put names to faces."""
    out = []
    for p in gallery.people():
        out.append({**p, "thumb": f"faces/thumb/{p['id']}" if p.get("has_thumb") else None})
    return {"people": out}


@router.get("/people/review")
def people_review():
    """The review queue for the dashboard's "Is this you?" card stack. Every
    unpromoted, unnamed cluster bucketed by confidence tier — `suggest` cards carry
    the member the system thinks it is; `unknown` cards go to everyone. Clusters that
    now clear the autoheal threshold are merged into their member as a side effect
    and reported under `healed` (never shown as cards)."""
    result = gallery.review_queue()
    for card in result["queue"]:
        card["thumb"] = f"faces/thumb/{card['guest_id']}" if card.get("has_thumb") else None
    return result


@router.get("/guests")
def guests(recurring: bool = False):
    return {"guests": gallery.guests(recurring_only=recurring)}


@router.post("/guests/{guest_id}/promote")
def promote(guest_id: str, user_id: str = Body(..., embed=True),
            name: Optional[str] = Body(None, embed=True),
            authorization: Optional[str] = Header(None)):
    require_user(authorization)  # admin-gated labeling
    if not gallery.promote_guest(guest_id, user_id, name):
        raise HTTPException(status_code=404, detail="guest not found")
    return {"ok": True, "guest_id": guest_id, "promoted_to": user_id}


@router.post("/guests/{guest_id}/name")
def name_guest(guest_id: str, name: str = Body(..., embed=True),
               authorization: Optional[str] = Header(None)):
    require_user(authorization)  # admin-gated labeling
    if not gallery.name_guest(guest_id, name):
        raise HTTPException(status_code=404, detail="guest not found")
    return {"ok": True, "guest_id": guest_id, "name": name}


@router.post("/guests/{guest_id}/merge")
def merge(guest_id: str, into: str = Body(..., embed=True),
          authorization: Optional[str] = Header(None)):
    """Confirming "yes, that's <named guest>" folds this cluster into them —
    sighting-weighted centroid merge, so the guest keeps getting recognised across
    angles/visits instead of respawning as a new "Person N"."""
    require_user(authorization)
    if gallery.merge_guests(guest_id, into) is None:
        raise HTTPException(status_code=404, detail="guest not found")
    return {"ok": True, "guest_id": guest_id, "merged_into": into}


@router.post("/guests/{guest_id}/reject")
def reject(guest_id: str, user_id: str = Body(..., embed=True),
           authorization: Optional[str] = Header(None)):
    """A member answered "No" to an "Is this you / is this them?" card — the cluster
    is never suggested to (or auto-healed into) that member again."""
    require_user(authorization)
    if not gallery.reject_suggestion(guest_id, user_id):
        raise HTTPException(status_code=404, detail="guest not found")
    return {"ok": True, "guest_id": guest_id, "rejected_user_id": user_id}


@router.delete("/guests/{guest_id}")
def forget_guest(guest_id: str, authorization: Optional[str] = Header(None)):
    require_user(authorization)  # admin-gated
    gallery.forget_guest(guest_id)
    return {"ok": True, "guest_id": guest_id}
