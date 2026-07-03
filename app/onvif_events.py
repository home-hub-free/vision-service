"""ONVIF PullPoint event puller — the camera's OWN motion/tamper as box perception.

One daemon thread per event-capable camera (owned by the Supervisor, like the
CameraWorkers), long-polling the camera's PullPoint subscription and turning topic
edges into box signals (docs/CAMERA_ONVIF_CONTROL_PLAN.md §3):

  * `tns1:RuleEngine/CellMotionDetector/Motion` (IsMotion) →
      - `worker.motion_active` / `worker.last_motion_ts` — the opt-in YOLO pre-gate
        (`VISION_DETECT_ON_MOTION`, see camera.py; SMALL_MODELS_PLAN "sense with
        small nets": no full inference on provably-empty scenes);
      - MQTT `homehub/<zone>/<camId>/motion` (provenance `camera_motion`) — the
        memory lane keeps the history; the agent lane drops `motion` as pull-lane
        (like `occupancy` — raw motion must not wake the agent).
  * `.../TamperDetector/Tamper` (IsTamper) → MQTT `tamper` channel — this one DOES
    ride through the agent lane: a covered/moved camera is a high-salience
    security event (SECURITY_HARDENING_PLAN), and memory keeps it too.

Reconnect discipline mirrors rtsp.py: any failure (camera reboot, subscription
expiry, network) → log, exponential backoff, resubscribe. A dead camera or broker
never crashes the thread, and CellMotion stays a coarse "something moved" —
YOLO remains authoritative for *person* (plan §1 rule 5).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from . import ingest
from .config import cfg
from .onvif import OnvifClient, OnvifError, notification_bool

MOTION_TOPIC = "CellMotionDetector/Motion"
TAMPER_TOPIC = "Tamper"


class EventPuller(threading.Thread):
    def __init__(self, worker, client: OnvifClient) -> None:
        cam = getattr(worker, "cam")
        super().__init__(daemon=True, name=f"vision-onvif-ev-{cam.id}")
        self.worker = worker
        self.cam = cam
        self.client = client
        self._stop_evt = threading.Event()
        # Edge state: None = unknown (first report always publishes).
        self.motion: Optional[bool] = None
        self.tamper: Optional[bool] = None

    def stop(self) -> None:
        self._stop_evt.set()

    # ── loop: subscribe → pull → renew, resubscribe on any drop ───────────────
    def run(self) -> None:
        backoff = 2.0
        term_s = max(60, int(cfg.onvif_pull_timeout_s * 4))
        while not self._stop_evt.is_set():
            sub = None
            try:
                sub = self.client.create_pullpoint(term_s=term_s)
                self.worker.events_attached = True
                print(f"[vision] cam {self.cam.id} onvif events subscribed ({sub})", flush=True)
                backoff = 2.0
                while not self._stop_evt.is_set():
                    notes = self.client.pull_messages(
                        sub, timeout_s=cfg.onvif_pull_timeout_s, limit=32)
                    now = time.time()
                    for n in notes:
                        self._dispatch(n, now)
                    self.client.renew(sub, term_s=term_s)
            except OnvifError as e:
                self.worker.events_attached = False  # gate must fall open (always-detect)
                if self._stop_evt.is_set():
                    break
                print(f"[vision] cam {self.cam.id} onvif events dropped: {e}; "
                      f"resubscribe in {backoff:.0f}s", flush=True)
                if self._stop_evt.wait(backoff):
                    break
                backoff = min(60.0, backoff * 2)
            except Exception as e:  # noqa: BLE001 — never die
                self.worker.events_attached = False
                print(f"[vision] cam {self.cam.id} onvif events error: {e!r}", flush=True)
                if self._stop_evt.wait(backoff):
                    break
                backoff = min(60.0, backoff * 2)
        self.worker.events_attached = False
        if sub:
            try:
                self.client.unsubscribe(sub)
            except OnvifError:
                pass

    # ── edge extraction (pure-ish; fixture-tested via dispatch_note) ───────────
    def _dispatch(self, note: dict, now: float) -> None:
        topic = str(note.get("topic") or "")
        data = note.get("data") or {}
        if MOTION_TOPIC in topic:
            val = notification_bool(data, "IsMotion", "State")
            if val is None or val == self.motion:
                return
            self.motion = val
            self.worker.motion_active = val
            if val:
                self.worker.last_motion_ts = now
            ingest.publish_signal(self.cam.zone, self.cam.id, "motion", val,
                                  {"provenance": "camera_motion"})
        elif TAMPER_TOPIC in topic:
            val = notification_bool(data, "IsTamper", "State")
            if val is None or val == self.tamper:
                return
            self.tamper = val
            ingest.publish_signal(self.cam.zone, self.cam.id, "tamper", val,
                                  {"provenance": "camera_tamper"})
            if val:
                print(f"[vision] cam {self.cam.id} TAMPER detected", flush=True)
