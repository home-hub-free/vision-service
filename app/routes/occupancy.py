"""Occupancy + history (§7/§9.4) — the agent's PULL surface (a snapshot, never a wake).

  * GET /occupancy            — full per-zone occupancy world-model snapshot.
  * GET /who_is_here?zone=    — flat "who is in (this) zone" list — the `who_is_here`
                                tool the agent reads when it reasons (§7).
  * GET /history/who-came-by  — "who came by today?" from the event INDEX, never video.
  * GET /history/events       — events between two timestamps (timeline/scrubber, §9.5).
  * GET /history/segment      — the recording segment containing a timestamp (jump-to-clip).
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Query

from ..state import index, tracker, workers

router = APIRouter()


@router.get("/occupancy")
def occupancy():
    return {"zones": tracker.snapshot(), "cameras": [w.status() for w in workers.values()]}


@router.get("/who_is_here")
def who_is_here(zone: Optional[str] = None):
    return {"zone": zone, "people": tracker.who_is_here(zone)}


@router.get("/history/who-came-by")
def who_came_by(since: Optional[float] = Query(None, description="epoch seconds; default last 24h")):
    since_ts = since if since is not None else time.time() - 86400
    return {"since": since_ts, "people": index.who_came_by(since_ts)}


@router.get("/history/events")
def events(start: float, end: float, zone: Optional[str] = None):
    return {"events": index.events_between(start, end, zone)}


@router.get("/history/segment")
def segment(cam_id: str, ts: float):
    seg = index.segment_at(cam_id, ts)
    return {"segment": seg}
