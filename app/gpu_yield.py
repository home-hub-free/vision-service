"""GPU-yield — voice turns ask vision to stay off the GPU for a few seconds.

The card is shared (LLM + whisper + TTS + this service) and continuous detection
keeps its busy-time at 100%, so every voice stage queues behind vision and runs at
~half speed (measured 2026-07-10: whisper 1.0s alone vs 3.1-3.8s in service; the
LLM 1.8s vs 3.4-4.2s). A voice turn is short and bursty — vision yielding for its
duration costs a few seconds of presence-update lag ONLY while someone is actively
talking to the house, and buys the turn back its uncontended speed.

Design constraints:
- SELF-EXPIRING: a deadline, never a pause/resume pair — a crashed caller cannot
  leave vision stopped. Hard cap per request keeps a buggy caller bounded.
- Extensions monotonic: overlapping requests extend, never shorten.
- The MJPEG relay, recorders (codec-copy ffmpeg) and privacy logic are untouched —
  only the GPU inference pipeline skips frames while yielded. The occupancy ledger
  tolerates the gap by design (leave-confirm is 120s; a ≤15s hole cannot fabricate
  a departure).
"""
from __future__ import annotations

import threading
import time

_MAX_MS = 15_000

_lock = threading.Lock()
_until = 0.0
_requests = 0


def request(ms: int) -> dict:
    """Extend the yield window by up to _MAX_MS from now. Returns status()."""
    global _until, _requests
    ms = max(0, min(int(ms), _MAX_MS))
    with _lock:
        fresh = time.time() >= _until  # entering a new window (vs extending)
        deadline = time.time() + ms / 1000.0
        if deadline > _until:
            _until = deadline
        _requests += 1
    if fresh and ms > 0:
        print(f"[vision] gpu yield {ms}ms (voice turn)", flush=True)
    return status()


def active() -> bool:
    return time.time() < _until


def status() -> dict:
    remaining = max(0.0, _until - time.time())
    return {"active": remaining > 0, "remaining_ms": int(remaining * 1000), "requests": _requests}
