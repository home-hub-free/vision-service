"""Hub room-digest push — the vision-service's feed into the agent's WORLD-MODEL.

Distinct from `ingest.py` (the MQTT producer, which feeds memory + the agent WAKE lane):
this pushes a small per-zone occupancy+identity digest straight to the hub on every
salient change, so the hub can FUSE it (with the satellite mic ambient + PIR presence)
into the `rooms` map the agent reads on `GET /state` (PERCEPTION_TO_AGENT_PLAN §3.1).
The hub is the single aggregator; the gateway stays single-source.

Contract (mirrors the §3 RoomDigest the gateway renders):

    POST /perception
    { "zone": "sala", "count": 2, "occupied": true,
      "people": [ {"id":"u1","name":"Juan","class":"household","confidence":0.82},
                  {"id":null,"name":null,"class":"unknown","confidence":0.0} ] }

Only resolved identity crosses — `{id, name, class, confidence}`, NEVER an embedding
(biometrics stay on the box, CLAUDE.md identity rule). Best-effort exactly like the
MQTT seam: fire-and-forget, never throws into the perception loop, a no-op when the
hub is down or `VISION_HUB_PUSH_ENABLED` is false. A `room_empty` change pushes
`count:0, occupied:false` so a zone clears promptly (the hub TTL-prunes anyway).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

from .actions import Hint, activity_hint
from .config import cfg


# T0 zone-activity states (VISION_CONTEXT_TIERS_PLAN §2) — pure thresholds over the
# dwell/speed the tracker already keeps. No model, ~0 ms.
ACTIVITY_PASSING = "passing"
ACTIVITY_LINGERING = "lingering"
ACTIVITY_SETTLED = "settled"


def _person_activity(dwell_s: float, moving: bool) -> str:
    """"Rushing past" vs "settled in the room" (§2): short dwell OR in motion = passing;
    past the settle bar at low speed = settled; the in-between is lingering."""
    if moving or dwell_s < cfg.activity_pass_dwell_s:
        return ACTIVITY_PASSING
    if dwell_s >= cfg.activity_settle_dwell_s:
        return ACTIVITY_SETTLED
    return ACTIVITY_LINGERING


def zone_activity(snapshot_people: List[dict]) -> Optional[str]:
    """Zone activity = the max-dwell person's state (§2), upgraded with their posture
    when T1 read one ("settled+sitting"). None when nobody carries dwell data (null
    build / no bbox), so the field is simply omitted from the digest."""
    best = None
    for p in snapshot_people:
        if p.get("dwell_s") is None:
            continue
        if best is None or p["dwell_s"] > best["dwell_s"]:
            best = p
    if best is None:
        return None
    act = _person_activity(best["dwell_s"], bool(best.get("moving")))
    posture = best.get("posture")
    return f"{act}+{posture}" if posture else act


# T2a hint hysteresis (per zone): the last fired hint + when it last fired. A hint
# survives brief rule dropouts (a posture flicker, one `moving` snapshot) for
# cfg.hint_hold_s while the zone stays occupied; a new fired hint replaces it
# immediately; an emptied zone clears it.
_hint_hold: Dict[str, Tuple[Hint, float]] = {}


def _held_hint(zone: str, fired: Optional[Hint], occupied: bool,
               now: float) -> Optional[Hint]:
    if not occupied:
        _hint_hold.pop(zone, None)
        return None
    if fired is not None:
        _hint_hold[zone] = (fired, now)
        return fired
    held = _hint_hold.get(zone)
    if held is not None and now - held[1] <= cfg.hint_hold_s:
        return held[0]
    _hint_hold.pop(zone, None)
    return None


def _reset_hint_hold() -> None:
    """Test seam."""
    _hint_hold.clear()


def room_digest_payload(zone: str, snapshot_people: List[dict],
                        hour: Optional[int] = None,
                        now: Optional[float] = None) -> dict:
    """Pure builder: a zone's `tracker.snapshot(zone)` people list → the /perception body.

    `snapshot_people` is the per-zone list `OccupancyTracker.snapshot()` returns (each entry
    is `identity.as_meta()` + track/since). We keep ONLY the resolved-identity fields — id,
    name, class, confidence — plus the T0/T1 activity signals (dwell_s, moving, posture),
    and drop everything biometric/track-internal. Unknowns ride along (class "unknown",
    name null) so the hub/agent can COUNT them without naming them."""
    people = []
    for p in snapshot_people:
        person = {
            "id": p.get("id"),
            "name": p.get("name"),
            "class": p.get("class") or "unknown",
            "confidence": p.get("confidence") or 0.0,
        }
        # T0/T1 fields are additive — the hub tolerates their absence (older producer)
        # and their presence (older hub just ignores unknown fields).
        if p.get("dwell_s") is not None:
            person["dwell_s"] = p["dwell_s"]
            person["moving"] = bool(p.get("moving"))
        if p.get("posture"):
            person["posture"] = p["posture"]
        # SMART_FACE_ID (additive, same pattern): `assumed` = this identity is a
        # position/sensor hypothesis, not a live face read (the snapshot already capped
        # its confidence) → the dashboard hedges the name ("David?"); `pending_left` = a
        # mid-dropout ghost. Forwarded only when set, so older hubs simply don't see them.
        if p.get("assumed"):
            person["assumed"] = True
        if p.get("pending_left"):
            person["pending_left"] = True
        people.append(person)
    body = {
        "zone": zone,
        "count": len(people),
        "occupied": len(people) > 0,
        "people": people,
    }
    activity = zone_activity(snapshot_people)
    if activity:
        body["activity"] = activity
    # T2a (plan §4.2a): context-rule hint over the same signals — additive, hedged
    # downstream ("likely …"/"possibly …"), held through brief dropouts (hysteresis).
    # `hour`/`now` are injectable for tests.
    hint = _held_hint(zone, activity_hint(zone, snapshot_people, hour=hour),
                      occupied=len(people) > 0,
                      now=time.time() if now is None else now)
    if hint:
        body["activity_hint"], body["activity_hint_conf"] = hint
    return body


def _svc_headers() -> dict:
    h = {"content-type": "application/json"}
    if cfg.hub_service_token:
        h["X-Hub-Service-Token"] = cfg.hub_service_token
    return h


def push_room(zone: str, snapshot_people: List[dict], timeout: float = 3.0) -> None:
    """Push the zone's current occupancy+identity digest to the hub. Best-effort — any failure
    (hub down, timeout, bad response) is swallowed; perception must never stall on this."""
    if not cfg.hub_push_enabled or not zone:
        return
    body = json.dumps(room_digest_payload(zone, snapshot_people)).encode()
    req = urllib.request.Request(cfg.hub_url + "/perception", data=body,
                                 headers=_svc_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except (urllib.error.URLError, OSError, ValueError) as e:
        # Hub unreachable / slow — drop, don't buffer (matches the ingestion seam).
        print(f"[vision] hub room-digest push for zone {zone!r} failed: {e}", flush=True)
