"""Supervisor — roster sync + camera-worker lifecycle (§4.1, §11.9 scaling).

Polls the hub roster (`hub_client.fetch_cameras`) every `roster_poll_s` and reconciles
the running CameraWorker set: start a worker for each newly-declared camera, stop one
whose camera vanished or whose stream URL changed. One worker (one daemon thread) per
stream is the per-camera scaling model (§11.9); back-pressure is handled inside the
worker (it relays at full rate but runs perception at `detect_fps`).

Also refreshes the users roster so resolved `users.id`s get a display name.
"""
from __future__ import annotations

import threading
import time

from . import hub_client
from .config import cfg
from .state import workers


class Supervisor(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True, name="vision-supervisor")
        self._stop = threading.Event()
        self.users: dict = {}

    def run(self) -> None:
        # First sync immediately, then on the poll interval.
        while not self._stop.is_set():
            try:
                self._sync()
            except Exception as e:  # noqa: BLE001
                print(f"[vision] supervisor sync error: {e}", flush=True)
            if self._stop.wait(cfg.roster_poll_s):
                break

    def stop(self) -> None:
        self._stop.set()
        for w in list(workers.values()):
            getattr(w, "stop", lambda: None)()

    def _sync(self) -> None:
        from .camera import CameraWorker  # lazy to avoid import cycle at module load

        self.users = hub_client.fetch_users() or self.users
        cams = {c.id: c for c in hub_client.fetch_cameras()}

        # Stop workers whose camera disappeared or changed stream URL.
        for cam_id in list(workers.keys()):
            w = workers[cam_id]
            cam = cams.get(cam_id)
            if cam is None or getattr(w, "cam").stream_url != cam.stream_url:
                getattr(w, "stop", lambda: None)()
                del workers[cam_id]

        # Start workers for new cameras.
        for cam_id, cam in cams.items():
            if cam_id in workers:
                # keep the worker's display zone fresh if reassigned in the dashboard
                getattr(workers[cam_id], "cam").zone = cam.zone
                continue
            w = CameraWorker(cam)
            workers[cam_id] = w
            w.start()
            print(f"[vision] started worker for camera {cam_id} (zone={cam.zone})", flush=True)
