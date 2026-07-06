"""Per-camera privacy mode — the "stop watching this camera NOW" switch.

When a camera is private its worker stops pulling from it entirely: no live view,
no recording, no perception, no occupancy — the frames never reach this box. The
flag is enforced at the WORKER (the single consumer of every camera stream), so
one switch covers every downstream surface at once; the stream/snapshot routes
also refuse with 423 so a viewer gets a clear "privacy" instead of a stalled feed.

Persisted as a tiny JSON file (a set of private cam ids) so a service restart can
never silently resume surveillance the household switched off — the same reason
the flag lives HERE and not on the hub roster: the vision-service must honour it
even when the hub is down.

Toggling is a dashboard action that goes through the hub's `/camera/:id/privacy`
proxy (auth + audit — WHO covered the cameras is a record worth keeping), exactly
like PTZ; the vision route itself stays LAN-open like the rest of the control seam.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Dict, Optional, Set

from .config import cfg


class PrivacyStore:
    """Thread-safe persisted set of private camera ids. Reads are dict-cheap (the
    reader thread consults it per frame); writes persist synchronously — a toggle
    is rare and MUST survive a crash that follows it."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or cfg.privacy_file
        self._lock = threading.Lock()
        self._private: Set[str] = set()
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self._private = {str(c) for c in data.get("private", [])}
        except FileNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001 — a corrupt file must not kill boot
            print(f"[vision] privacy store unreadable ({e}); starting empty", flush=True)

    def _save(self) -> None:
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"private": sorted(self._private)}, f)
        os.replace(tmp, self.path)

    def is_private(self, cam_id: str) -> bool:
        return cam_id in self._private

    def set(self, cam_id: str, on: bool) -> bool:
        """Set one camera's privacy; persists on change. Returns the new state."""
        with self._lock:
            before = set(self._private)
            (self._private.add if on else self._private.discard)(cam_id)
            if self._private != before:
                try:
                    self._save()
                except Exception as e:  # noqa: BLE001 — enforce in-memory regardless
                    print(f"[vision] privacy store save failed: {e}", flush=True)
        return on

    def all(self) -> Dict[str, bool]:
        with self._lock:
            return {cam_id: True for cam_id in self._private}
