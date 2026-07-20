"""Occupancy reaper — the frame-independent ledger heartbeat (2026-07-19 ghost fix).

The OccupancyTracker's sweep (leave confirmation, assumed-entry expiry, room_empty
derivation) used to run ONLY inside `update()`, i.e. only when some camera delivered
a frame. A stalled reader, an offline camera, privacy mode or a wedged perception
loop therefore froze every pending-left ghost in `/occupancy` — and in the hub's
fused `/state.rooms` — indefinitely (observed live: ~109 immortal ghosts in sala,
guest dwells past 53h).

This thread calls `tracker.reap()` on a short period and gives any resulting edges
the EXACT same treatment the camera worker gives edges from `update()` (camera.py):
MQTT publish (ingest) + event index + hub room-digest push — so a leave confirmed by
the reaper reaches memory and the hub's rooms model just like a frame-driven one.
The reap itself is pure in-memory over the ledger (no I/O), so a 15s period is
negligible; edges are already debounced by the ledger, so this can never spam.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from .config import cfg


class OccupancyReaper(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True, name="vision-occupancy-reaper")
        self._stop = threading.Event()

    def run(self) -> None:
        if cfg.occupancy_reap_s <= 0:
            return  # knob off → frame-driven sweeps only (pre-fix behaviour)
        while not self._stop.wait(cfg.occupancy_reap_s):
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 — the reaper must never crash the box
                print(f"[vision] occupancy reaper error: {e}", flush=True)

    def stop(self) -> None:
        self._stop.set()

    def tick(self, now: Optional[float] = None) -> None:
        # Late imports keep module load light (test posture: occupancy stays pure and
        # unit-testable via tracker.reap(); this thread is just the plumbing around it).
        from . import hub_push, ingest
        from .state import index, tracker

        now = time.time() if now is None else now
        edges = tracker.reap(now)
        if not edges:
            return
        zones = {e.zone for e in edges if e.zone}
        counts = {z: len(tracker.snapshot(z, now=now).get(z, [])) for z in zones}
        for edge in edges:
            ingest.publish_edge(edge, counts.get(edge.zone, 0))
            index.record_event(edge)
        # Zones whose occupancy just changed get a fresh digest push, so the hub's
        # rooms model clears even when no frame will arrive to trigger the worker's push.
        for z in zones:
            hub_push.push_room(z, tracker.snapshot(z, now=now).get(z, []))
