"""Ingestion producer — the vision-service's OWN feed into the memory/LLM layer.

Per CAMERA_VISION_PLAN §5.2 the vision-service is its own MQTT producer; it does NOT
route through the hub. It publishes to the SAME topic scheme the hub's ingestion seam
uses — `homehub/<zone>/<camId>/<channel>` — so Node-RED's existing two lanes pick it
up unchanged: `mqtt-to-memory` keeps everything; `mqtt-to-agent` applies the salience
gate (§8). Semantics mirror the hub seam exactly (server/src/clients/ingestion.ts):

  * fire-and-forget QoS 0, gated on a live broker connection (dropped, never buffered,
    if Mosquitto is down — best-effort telemetry, never control traffic);
  * never throws into the perception loop;
  * a no-op when `VISION_INGESTION_ENABLED` is false (isolated bring-up).

Channels (§5.2):
  * `person`    — boolean edge: someone present in the zone (true/false).
  * `occupancy` — count snapshot for the zone (a number channel).
Identity rides in `meta.identity` (§5.1). `source` is always `"device"` — a camera
observation is autonomous, like a sensor report, NOT automation/llm.
"""
from __future__ import annotations

import json
import time
from typing import Optional

from .config import cfg
from .occupancy import (EDGE_ENTERED, EDGE_GUEST_ARRIVED, EDGE_IDENTIFIED,
                        EDGE_LEFT, EDGE_ROOM_EMPTY, Edge)

_client = None
_connected = False


def _ensure_client():
    global _client, _connected
    if not cfg.ingestion_enabled:
        return None
    if _client is not None:
        return _client
    try:
        import paho.mqtt.client as mqtt  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"[vision] paho-mqtt not installed ({e}); ingestion disabled", flush=True)
        return None

    # mqtt://host:port → (host, port)
    url = cfg.mqtt_url.replace("mqtt://", "")
    host, _, port = url.partition(":")
    c = mqtt.Client(client_id=f"vision-service-{int(time.time())}")

    def _on_connect(*_a):
        global _connected
        _connected = True
        print(f"[vision] MQTT connected ({cfg.mqtt_url})", flush=True)

    def _on_disconnect(*_a):
        global _connected
        _connected = False

    c.on_connect = _on_connect
    c.on_disconnect = _on_disconnect
    try:
        c.connect_async(host or "127.0.0.1", int(port or 1883), keepalive=30)
        c.loop_start()
    except Exception as e:  # noqa: BLE001
        print(f"[vision] MQTT connect failed: {e}", flush=True)
        return None
    _client = c
    return c


def init_ingestion() -> None:
    _ensure_client()


# Which occupancy edges carry a boolean `person` edge value (true on arrival/ident,
# false on leave/empty). `occupancy` (a count) is published alongside on every change.
_PERSON_TRUE = {EDGE_ENTERED, EDGE_IDENTIFIED, EDGE_GUEST_ARRIVED}
_PERSON_FALSE = {EDGE_LEFT, EDGE_ROOM_EMPTY}


def _publish(zone: str, cam_id: str, channel: str, value, source: str, meta: Optional[dict]) -> None:
    c = _ensure_client()
    if c is None or not _connected:
        return  # broker not up — drop, don't buffer (matches the hub seam)
    topic = f"homehub/{zone or '_'}/{cam_id or '_'}/{channel}"
    payload = {
        "deviceId": cam_id,
        "zone": zone,
        "ts": _iso(),
        "value": value,
        "unit": "",
        "source": source,
        "channel": channel,
    }
    if meta:
        payload.update(meta)
    try:
        c.publish(topic, json.dumps(payload), qos=0)
    except Exception as e:  # noqa: BLE001 — never break perception on a publish error
        print(f"[vision] publish to {topic} failed: {e}", flush=True)


def publish_edge(edge: Edge, occupancy_count: int) -> None:
    """Map one salient occupancy edge onto the ingestion bus (§5.2). The agent lane's
    salience gate decides which of these actually wake it (§8); memory keeps all."""
    meta = {"identity": edge.identity.as_meta()} if edge.identity.cls != "unknown" else {}
    # The named edge itself, so Node-RED can route on `edge` (person_entered, …).
    meta_edge = {**meta, "edge": edge.edge}
    if edge.edge in _PERSON_TRUE:
        _publish(edge.zone, edge.cam_id, "person", True, "device", meta_edge)
    elif edge.edge in _PERSON_FALSE:
        _publish(edge.zone, edge.cam_id, "person", False, "device", meta_edge)
    # Occupancy count snapshot rides every edge (pull-lane signal; deadbanded downstream).
    _publish(edge.zone, edge.cam_id, "occupancy", occupancy_count, "device", {"edge": edge.edge})


def _iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
