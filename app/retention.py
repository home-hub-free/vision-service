"""Retention janitor (§9.3) — a real janitor, mandatory, never wedges the disk.

Prunes recorded segments by **age** (keep `retention_days`) and/or a **disk cap**
(keep the recordings tree under `disk_cap_gb`, oldest-first). Runs on a timer in a
daemon thread. Recording degrades gracefully: when the cap is hit we drop oldest
files (and their index rows) rather than letting the disk fill.

DECISION (§9.3/§9.6/§11.4): the exact numbers (retention days, disk cap, at-rest
encryption) must come from a MEASURED day of footage — the defaults in config.py are
placeholders. Planning math (verify on real footage): 1080p H.264 low-fps ≈ ~1–4
GB/camera/day continuous; detection-gated is a fraction. At-rest encryption is left
optional (playback is gated behind dashboard auth regardless — §11).
"""
from __future__ import annotations

import os
import threading
import time
from typing import List, Optional, Tuple

from .config import cfg
from .index_db import EventIndex


def _segment_files(root: str) -> List[Tuple[str, float, int]]:
    """(path, mtime, size) for every .mp4 under the recordings tree."""
    out: List[Tuple[str, float, int]] = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if not f.endswith(".mp4"):
                continue
            p = os.path.join(dirpath, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out.append((p, st.st_mtime, st.st_size))
    return out


def sweep(root: str, retention_days: int, disk_cap_gb: float, index: Optional[EventIndex] = None) -> dict:
    """One prune pass. Pure-ish (filesystem side effects only); returns a summary so
    the caller / tests can assert what happened."""
    files = sorted(_segment_files(root), key=lambda x: x[1])  # oldest first
    removed: List[str] = []
    now = time.time()

    # 1) Age prune.
    if retention_days > 0:
        cutoff = now - retention_days * 86400
        survivors = []
        for p, mtime, size in files:
            if mtime < cutoff:
                _rm(p, index)
                removed.append(p)
            else:
                survivors.append((p, mtime, size))
        files = survivors

    # 2) Disk-cap prune (oldest-first until under cap).
    if disk_cap_gb > 0:
        cap = disk_cap_gb * (1024 ** 3)
        total = sum(s for _p, _m, s in files)
        for p, _mtime, size in files:  # already oldest-first
            if total <= cap:
                break
            _rm(p, index)
            removed.append(p)
            total -= size

    return {"removed": len(removed), "files": removed}


def _rm(path: str, index: Optional[EventIndex]) -> None:
    try:
        os.remove(path)
    except OSError:
        return
    if index is not None:
        index.prune_segment(path)


def sweep_thumbs(thumb_root: str, retention_days: int) -> int:
    """Age out the scrub-thumbnail cache with the same horizon as the segments it
    was extracted from (a thumb can't outlive its clip usefully). Returns removed."""
    if retention_days <= 0:
        return 0
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for dirpath, _dirs, files in os.walk(thumb_root):
        for f in files:
            if not f.endswith(".jpg"):
                continue
            p = os.path.join(dirpath, f)
            try:
                if os.stat(p).st_mtime < cutoff:
                    os.remove(p)
                    removed += 1
            except OSError:
                continue
    return removed


class Janitor(threading.Thread):
    def __init__(self, index: EventIndex) -> None:
        super().__init__(daemon=True, name="vision-janitor")
        self.index = index
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(cfg.janitor_interval_s):
            try:
                res = sweep(cfg.rec_dir, cfg.retention_days, cfg.disk_cap_gb, self.index)
                if res["removed"]:
                    print(f"[vision] janitor pruned {res['removed']} segment(s)", flush=True)
                sweep_thumbs(cfg.thumb_dir, cfg.retention_days)
                self._sync_footage()
                self._pregen_thumbs()
            except Exception as e:  # noqa: BLE001 — the janitor must never crash the box
                print(f"[vision] janitor error: {e}", flush=True)

    def _sync_footage(self) -> None:
        """Keep the per-file segment index fresh between dashboard visits (the
        /recordings routes also sync on demand — this just bounds the drift and
        heals the legacy directory-shaped rows without waiting for a visit)."""
        from .footage import sync_camera  # late import — retention stays light for tests
        from .state import workers

        for cam_id, w in list(workers.items()):
            st = w.status() if hasattr(w, "status") else {}
            if st.get("records"):
                sync_camera(self.index, cam_id, st.get("zone") or "_")

    def _pregen_thumbs(self) -> None:
        """Warm the scrub/filmstrip frame cache for freshly settled segments (a
        few per tick — new footage arrives at ~1 segment/5min per camera)."""
        from .state import workers
        from .thumbs import pregenerate_missing

        cams = []
        for cam_id, w in list(workers.items()):
            st = w.status() if hasattr(w, "status") else {}
            if st.get("records"):
                cams.append(cam_id)
        n = pregenerate_missing(self.index, cams)
        if n:
            print(f"[vision] pregenerated thumb sets for {n} segment(s)", flush=True)

    def stop(self) -> None:
        self._stop.set()
