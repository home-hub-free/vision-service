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
from typing import Dict

from . import hub_client
from .config import cfg
from .state import workers


class Supervisor(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True, name="vision-supervisor")
        self._stop = threading.Event()
        self.users: dict = {}
        # ONVIF side-threads (CAMERA_ONVIF_CONTROL_PLAN): one EventPuller per
        # event-capable camera + a per-camera last-clock-sync stamp (§6).
        self._pullers: Dict[str, object] = {}
        self._time_synced: Dict[str, float] = {}

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
        for p in list(self._pullers.values()):
            getattr(p, "stop", lambda: None)()
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
            if cam is None or getattr(w, "cam").stream_url != cam.stream_url \
                    or getattr(w, "cam").record_url != cam.record_url:
                getattr(w, "stop", lambda: None)()
                del workers[cam_id]
                self._drop_onvif(cam_id)

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

        self._sync_onvif()

    # ── ONVIF side-jobs: event pullers (§3) + daily clock push (§6) ────────────
    def _drop_onvif(self, cam_id: str) -> None:
        p = self._pullers.pop(cam_id, None)
        if p is not None:
            getattr(p, "stop", lambda: None)()
        self._time_synced.pop(cam_id, None)

    def _sync_onvif(self) -> None:
        """Reconcile ONVIF side-threads with the live worker set. Capability probes
        run here (the supervisor's own thread, off every hot path) and are cached on
        the client; an unreachable camera backs off inside `capabilities()`, so a
        dead camera costs one short timeout every few minutes, not per sync."""
        if not cfg.onvif_enabled:
            return
        from . import onvif  # late import keeps module load light for tests
        from .onvif_events import EventPuller

        now = time.time()
        for cam_id, w in list(workers.items()):
            client = onvif.get_onvif(cam_id)
            if client is None:
                continue  # not an ONVIF camera (ESP32-CAM MJPEG node)
            try:
                caps = client.capabilities(now)
            except Exception:  # noqa: BLE001 — unreachable/backing off; retry next sync
                continue

            # Motion/tamper puller for event-capable cameras (one per camera).
            puller = self._pullers.get(cam_id)
            if cfg.onvif_events_enabled and caps.get("events") \
                    and (puller is None or not getattr(puller, "is_alive", lambda: False)()):
                p = EventPuller(w, client)
                self._pullers[cam_id] = p
                p.start()

            # Clock push: WAN-blocked cameras can't NTP; drift rots event/OSD
            # timestamps. Cheap (one SOAP call), daily per camera.
            if cfg.onvif_time_sync_h > 0 and \
                    now - self._time_synced.get(cam_id, 0.0) > cfg.onvif_time_sync_h * 3600:
                try:
                    client.set_system_time()
                    self._time_synced[cam_id] = now
                    print(f"[vision] cam {cam_id} clock synced to box time", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[vision] cam {cam_id} clock sync failed: {e}", flush=True)
