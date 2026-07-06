"""Per-zone occupancy + identity world-model — the digest the agent actually sees.

This is the §8 "classify, don't queue" answer for cameras, and the §4.3 debounce.
Pixels and per-frame detections never leave the box; this module turns a stream of
per-camera track observations into:

  * a **snapshot** (pull): `who_is_here(zone)` — current presence + identity, read
    when the agent reasons, never a wake; and
  * **salient edges** (push): person_entered / person_identified / guest_arrived /
    person_left / room_empty — the wake-worthy events, fired ONCE per arrival with a
    re-arm cooldown so a lingering or flickering person never re-wakes the agent.

It is pure over an injected clock (`now`), with no I/O, so the debounce/hysteresis
is unit-testable without cameras or models (see ../tests/test_occupancy.py). The
camera worker feeds it; the MQTT producer publishes whatever edges it returns.

Mapping to the ingestion contract (§5.2): every edge becomes an MQTT publish on
`homehub/<zone>/<camId>/<channel>` with `source:"device"` (an autonomous
observation, NOT automation/llm) and identity riding in `meta.identity` (§5.1).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import cfg

# Salience classes (§8 / §12.5). PUSH = wakes the agent (edge + cooldown); the raw
# occupancy snapshot is PULL-only (read via who_is_here, never published as a wake).
EDGE_ENTERED = "person_entered"
EDGE_IDENTIFIED = "person_identified"
EDGE_GUEST_ARRIVED = "guest_arrived"
EDGE_LEFT = "person_left"
EDGE_ROOM_EMPTY = "room_empty"
# T1 fall-shaped signal (VISION_CONTEXT_TIERS_PLAN §3): lying, outside a lying-ok zone,
# for longer than the dwell bar. Alert-only (no autonomy), once per lying episode.
EDGE_POSTURE_ALERT = "posture_alert"
PUSH_EDGES = {EDGE_ENTERED, EDGE_IDENTIFIED, EDGE_GUEST_ARRIVED, EDGE_LEFT, EDGE_ROOM_EMPTY,
              EDGE_POSTURE_ALERT}

# EMA weight for the per-track speed estimate (T0). One reading is noisy (bbox jitter
# at 5 fps), so smooth; 0.4 settles in ~3 observations — fast enough to catch someone
# starting to rush, slow enough that one jittery box doesn't flip `moving`.
SPEED_EMA_ALPHA = 0.4


@dataclass
class Identity:
    """Resolves to the SAME shape voice fills (AgentUserContext), via:"face" (§4.4)."""
    id: Optional[str]            # users.id | "guest:<n>" | None
    name: Optional[str]
    cls: str                     # "household" | "guest" | "unknown"
    confidence: float
    via: str = "face"

    def key(self) -> str:
        return self.id or f"unknown:{self.cls}"

    def as_meta(self) -> dict:
        return {"id": self.id, "name": self.name, "class": self.cls,
                "via": self.via, "confidence": round(self.confidence, 3)}


UNKNOWN = Identity(id=None, name=None, cls="unknown", confidence=0.0)


@dataclass
class Observation:
    """One tracked person in one frame (output of the perception pipeline)."""
    track_id: str
    identity: Identity = field(default_factory=lambda: UNKNOWN)
    # T0 (VISION_CONTEXT_TIERS_PLAN §2): the track's bbox in frame pixels + the frame
    # width, so the tracker can keep a camera-agnostic speed (frame-widths/s). Optional:
    # older callers/null builds omit them and dwell/speed simply stays unavailable.
    bbox: Optional[Tuple[int, int, int, int]] = None
    frame_w: int = 0
    # T1 (§3): coarse body state from pose, when the pose engine ran on this frame.
    posture: Optional[str] = None  # "standing" | "sitting" | "lying" | "bent"
    # Whether the source camera is context-capable (Camera.context_capable): satellite/
    # ESP32 cams are face-ID-only — too low-quality for full-body inference — so their
    # tracks carry identity but NO T0/T1/T2a context signals downstream.
    context: bool = True


@dataclass
class _Track:
    key: str
    zone: str
    cam_id: str
    first_seen: float
    last_seen: float
    hits: int = 0
    present: bool = False
    identity: Identity = field(default_factory=lambda: UNKNOWN)
    announced_identity: Optional[str] = None  # identity key we've already pushed
    # T0: EMA of bbox-center displacement, in frame-widths/s (camera-agnostic), plus
    # the last width-normalized center + its timestamp the EMA differentiates against.
    speed: float = 0.0
    norm_cx: Optional[float] = None
    norm_cy: Optional[float] = None
    pos_ts: float = 0.0
    # T1: latest posture read (pose frames only — persists between pose cadence ticks),
    # and the once-per-lying-episode latch for the fall-shaped alert.
    posture: Optional[str] = None
    posture_alerted: bool = False
    # Debounce for posture changes: the candidate posture + when it first disagreed
    # with the committed one (commits after cfg.posture_stable_s of consistency).
    posture_pending: Optional[str] = None
    posture_pending_since: float = 0.0
    # Mirrors Observation.context (constant per camera): False = identity-only track.
    context: bool = True


@dataclass
class Edge:
    edge: str
    zone: str
    cam_id: str
    track_key: str
    identity: Identity
    ts: float


class OccupancyTracker:
    """Holds live per-zone state and turns observations into salient edges.

    Track ids are per-camera, so they're namespaced `"<camId>:<trackId>"`. Identity
    only ever monotonically improves on a track (a higher-confidence read wins, and
    unknown→known is an upgrade); we never downgrade a known person to unknown on a
    frame where the face simply wasn't visible.
    """

    def __init__(self) -> None:
        self._tracks: Dict[str, _Track] = {}
        # Per-zone re-arm memory: identity-key -> ts it last left. A re-entry inside
        # rewake_cooldown_s is treated as a continuation (no new wake), so a person
        # pacing in/out of frame doesn't spam person_entered.
        self._recent_left: Dict[str, Dict[str, float]] = {}
        self._zone_occupied: Dict[str, bool] = {}

    # ── ingest ────────────────────────────────────────────────────────────────
    def update(
        self,
        cam_id: str,
        zone: str,
        observations: List[Observation],
        now: Optional[float] = None,
    ) -> List[Edge]:
        now = time.time() if now is None else now
        zone = zone or "_"
        edges: List[Edge] = []
        seen_keys = set()

        for obs in observations:
            key = f"{cam_id}:{obs.track_id}"
            seen_keys.add(key)
            tr = self._tracks.get(key)
            if tr is None:
                tr = _Track(key=key, zone=zone, cam_id=cam_id, first_seen=now, last_seen=now)
                self._tracks[key] = tr
            tr.last_seen = now
            tr.hits += 1
            tr.context = obs.context
            tr.identity = _better(tr.identity, obs.identity)
            self._update_motion(tr, obs, now)
            if obs.posture is not None:
                self._update_posture(tr, obs.posture, now)
            alert = self._maybe_posture_alert(tr, now)
            if alert is not None:
                edges.append(alert)

            # Cross to "present" after enter_frames consecutive sightings (debounce).
            if not tr.present and tr.hits >= cfg.enter_frames:
                tr.present = True
                # Mark the identity announced REGARDLESS of suppression, so a
                # cooldown-suppressed re-entry also stays quiet on the next frame
                # (otherwise the identify-after-present branch would re-fire it).
                tr.announced_identity = tr.identity.key()
                if not self._suppressed_by_cooldown(zone, tr.identity, now):
                    edges.append(self._mk(EDGE_ENTERED, tr, now))
                    if tr.identity.cls != "unknown":
                        edges.append(self._identify_edge(tr, now))

            # Identity arrived/improved AFTER we already announced presence.
            elif tr.present and tr.identity.cls != "unknown" and tr.announced_identity != tr.identity.key():
                tr.announced_identity = tr.identity.key()
                edges.append(self._identify_edge(tr, now))

        edges += self._expire(cam_id, zone, seen_keys, now)
        self._recompute_zone(zone, now, edges)
        return edges

    # ── leave / empty ───────────────────────────────────────────────────────
    def _expire(self, cam_id: str, zone: str, seen_keys: set, now: float) -> List[Edge]:
        edges: List[Edge] = []
        for key, tr in list(self._tracks.items()):
            if tr.cam_id != cam_id or tr.zone != zone or key in seen_keys:
                continue
            if now - tr.last_seen <= cfg.leave_grace_s:
                continue
            if tr.present:
                edges.append(self._mk(EDGE_LEFT, tr, now))
                self._recent_left.setdefault(zone, {})[tr.identity.key()] = now
            del self._tracks[key]
        return edges

    def _recompute_zone(self, zone: str, now: float, edges: List[Edge]) -> None:
        occupied = any(t.present for t in self._tracks.values() if t.zone == zone)
        was = self._zone_occupied.get(zone, False)
        if was and not occupied:
            edges.append(Edge(EDGE_ROOM_EMPTY, zone, "", "", UNKNOWN, now))
        self._zone_occupied[zone] = occupied

    # ── T0 dwell/speed + T1 posture alert ─────────────────────────────────────
    def _update_motion(self, tr: _Track, obs: Observation, now: float) -> None:
        """EMA the track's speed from bbox-center displacement, normalized by frame
        width (units: frame-widths/s — camera-agnostic). No bbox/width → no-op, so
        the null build and older callers keep working with speed pinned at 0."""
        if obs.bbox is None or obs.frame_w <= 0:
            return
        x1, y1, x2, y2 = obs.bbox
        cx = ((x1 + x2) / 2.0) / obs.frame_w
        cy = ((y1 + y2) / 2.0) / obs.frame_w  # width-normalized both axes: isotropic units
        if tr.norm_cx is not None and tr.norm_cy is not None and now > tr.pos_ts:
            inst = math.hypot(cx - tr.norm_cx, cy - tr.norm_cy) / (now - tr.pos_ts)
            tr.speed = SPEED_EMA_ALPHA * inst + (1.0 - SPEED_EMA_ALPHA) * tr.speed
        tr.norm_cx, tr.norm_cy, tr.pos_ts = cx, cy, now

    def _update_posture(self, tr: _Track, posture: str, now: float) -> None:
        """T1 debounce: a NEW posture must be read consistently for posture_stable_s
        before it replaces the committed one. A partial bbox at frame-exit reads
        "lying" for a few frames — instant commits flapped the digest and dropped a
        T2a hint mid-cooking. The first read commits immediately; a candidate that
        stops being read (person left, posture reverted) simply expires."""
        if tr.posture is None or posture == tr.posture:
            if tr.posture is None:
                tr.posture = posture
            tr.posture_pending = None
            return
        if tr.posture_pending != posture:
            tr.posture_pending, tr.posture_pending_since = posture, now
            return
        if now - tr.posture_pending_since >= cfg.posture_stable_s:
            tr.posture = posture
            tr.posture_pending = None

    def _maybe_posture_alert(self, tr: _Track, now: float) -> Optional[Edge]:
        """Fall-shaped salience (§3): lying + zone not lying-ok + dwell past the bar →
        one alert per lying episode (the latch re-arms when the posture changes)."""
        if tr.posture != "lying":
            tr.posture_alerted = False
            return None
        if tr.posture_alerted or not tr.present:
            return None
        if (now - tr.first_seen) < cfg.lying_alert_dwell_s:
            return None
        ok = {z.strip().lower() for z in cfg.lying_ok_zones.split(",") if z.strip()}
        if tr.zone.lower() in ok:
            return None
        tr.posture_alerted = True
        return self._mk(EDGE_POSTURE_ALERT, tr, now)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _identify_edge(self, tr: _Track, now: float) -> Edge:
        edge = EDGE_GUEST_ARRIVED if tr.identity.cls == "guest" else EDGE_IDENTIFIED
        return self._mk(edge, tr, now)

    def _mk(self, edge: str, tr: _Track, now: float) -> Edge:
        return Edge(edge=edge, zone=tr.zone, cam_id=tr.cam_id, track_key=tr.key,
                    identity=tr.identity, ts=now)

    def _suppressed_by_cooldown(self, zone: str, ident: Identity, now: float) -> bool:
        left_at = self._recent_left.get(zone, {}).get(ident.key())
        return left_at is not None and (now - left_at) < cfg.rewake_cooldown_s

    # ── snapshot (pull surface — who_is_here) ─────────────────────────────────
    def snapshot(self, zone: Optional[str] = None, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        out: Dict[str, list] = {}
        for tr in self._tracks.values():
            if not tr.present:
                continue
            if zone and tr.zone != zone:
                continue
            entry = {
                "track": tr.key,
                "since": tr.first_seen,
                **tr.identity.as_meta(),
            }
            if tr.context:
                # T0 activity signals (§2): how long they've been here + whether they're
                # in motion right now (speed EMA vs the passing bar, camera-agnostic).
                # Identity-only tracks (satellite cams) omit ALL context fields, which is
                # what keeps them out of zone activity + T2a hints downstream.
                entry["dwell_s"] = round(max(0.0, now - tr.first_seen), 1)
                entry["moving"] = tr.speed >= cfg.activity_speed_fws
                if tr.posture:
                    entry["posture"] = tr.posture  # T1 (§3), only once a pose engine read it
            out.setdefault(tr.zone, []).append(entry)
        return out

    def who_is_here(self, zone: Optional[str] = None) -> List[dict]:
        snap = self.snapshot(zone)
        people: List[dict] = []
        for z, occ in snap.items():
            for p in occ:
                people.append({"zone": z, **p})
        return people


def _better(current: Identity, incoming: Identity) -> Identity:
    """Monotonic identity merge: prefer a known person over unknown, then the higher
    confidence. Never demote a known identity to unknown on a face-less frame."""
    if incoming.cls == "unknown":
        return current
    if current.cls == "unknown":
        return incoming
    return incoming if incoming.confidence > current.confidence else current
