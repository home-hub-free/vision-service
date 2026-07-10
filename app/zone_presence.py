"""Presence-gated perception — don't run GPU vision on rooms nobody is in.

User directive 2026-07-10: gate the heavy vision pipeline on the hub's presence/motion
sensors per zone — a camera whose zone shows no sign of anyone should not be burning
GPU on detection/face-ID. Zones WITHOUT a presence-capable sensor keep today's
always-on behaviour (the hub's GET /presence simply has no row for them).

Safety posture (this gate must never blind the house):
- FAIL OPEN everywhere: hub unreachable, stale poll, missing zone row → process frames.
- SELF-HOLD: vision is itself a presence source and PIRs go blind to still people —
  while this camera recently SAW someone (linger window), the gate stays open even if
  the sensor reads empty, so a person sitting still is never dropped mid-track. The
  gate only closes when the sensor says empty AND vision hasn't seen anyone for
  VISION_PRESENCE_LINGER_S.
- Reopening is instant: the first PIR/mmWave edge flips the zone occupied on the next
  poll (≤ POLL_S) and frames flow again — the camera's own pre-roll and the hub's
  sensor event mean nothing meaningful is missed at a doorway.

Composes with (does not replace) the camera's own motion gate, privacy mode and the
voice-turn GPU yield — all sit on the same gate row in the worker loop.
Kill switch: VISION_PRESENCE_GATE=off.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

from .config import cfg

_POLL_S = float(os.getenv("VISION_PRESENCE_POLL_S", "5"))
_LINGER_S = float(os.getenv("VISION_PRESENCE_LINGER_S", "180"))
_ENABLED = os.getenv("VISION_PRESENCE_GATE", "on").strip().lower() not in ("off", "0", "false", "no")
_STALE_S = max(30.0, _POLL_S * 4)  # poll data older than this → fail open

_lock = threading.Lock()
_zones: dict[str, bool] = {}  # lowercased zone → occupied (only zones WITH sensors appear)
_polled_at = 0.0
_started = False


def _poll_loop() -> None:
    global _polled_at
    while True:
        try:
            with urllib.request.urlopen(cfg.hub_url + "/presence", timeout=4) as res:
                data = json.loads(res.read())
            fresh = {
                str(z.get("zone", "")).strip().lower(): bool(z.get("occupied"))
                for z in data.get("zones", [])
                if str(z.get("zone", "")).strip()
            }
            with _lock:
                _zones.clear()
                _zones.update(fresh)
                _polled_at = time.time()
        except Exception:
            pass  # stale data fails open via _STALE_S — never blind the cameras on a hub blip
        time.sleep(_POLL_S)


def ensure_started() -> None:
    global _started
    if _started or not _ENABLED:
        return
    _started = True
    threading.Thread(target=_poll_loop, name="zone-presence-poll", daemon=True).start()


def allow(zone: str | None, last_person_ts: float) -> bool:
    """Should this camera run its GPU pipeline right now?"""
    if not _ENABLED or not zone:
        return True
    if time.time() - last_person_ts < _LINGER_S:
        return True  # vision recently saw someone here — self-hold regardless of PIR
    with _lock:
        stale = time.time() - _polled_at > _STALE_S
        occupied = _zones.get(zone.strip().lower())
    if stale or occupied is None:
        return True  # no data / no sensor in this zone → always-on behaviour
    return occupied


def status() -> dict:
    with _lock:
        return {
            "enabled": _ENABLED,
            "linger_s": _LINGER_S,
            "zones": dict(_zones),
            "age_s": round(time.time() - _polled_at, 1) if _polled_at else None,
        }
