"""Per-zone occupancy + identity world-model — the digest the agent actually sees.

This is the §8 "classify, don't queue" answer for cameras, and the §4.3 debounce.
Pixels and per-frame detections never leave the box; this module turns a stream of
per-camera track observations into:

  * a **snapshot** (pull): `who_is_here(zone)` — current presence + identity, read
    when the agent reasons, never a wake; and
  * **salient edges** (push): person_entered / person_identified / guest_arrived /
    person_left / room_empty — the wake-worthy events, fired ONCE per arrival with a
    re-arm cooldown so a lingering or flickering person never re-wakes the agent.

Edges are debounced at the PERSON level via the presence ledger (`_Presence`):
tracks flap (a detector dropout on a seated person kills one and a re-detection
mints another 30–170s later — measured live 2026-07-06), people don't. A vanished
person goes silently pending for `leave_confirm_s` before ONE truthful person_left
(stamped with when they were last seen); returns inside the window heal with zero
edges; unresolved new arrivals hold their entered edge for `identify_settle_s` so
a flap-heal or a short-lived false detection never wakes anyone at all.

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


@dataclass
class _Presence:
    """One PERSON in one zone — the ledger entry the edges are debounced against.

    Tracks are ephemeral (a detector dropout kills one and a re-detection mints a
    fresh id); a person is not. `tracks` is the set of live track keys currently
    supporting this entry (several cameras / a re-formed track all land on the same
    entry via the identity key), and presence survives track churn:
      * lose the last supporting track → `pending_left_since` starts ticking, the
        entry stays in the snapshot (a dropout must not flap the dashboard/agent);
      * a track re-appears inside `leave_confirm_s` → healed, ZERO edges;
      * absence outlives the window → ONE person_left, stamped with `last_seen`
        (when they actually disappeared, not when we gave up waiting).
    `announced` = the entered edge was emitted (or deliberately suppressed by the
    rewake cooldown); an entry that dies unannounced was a blip — total silence.
    """
    zone: str
    key: str                     # identity key ("u1" | "guest:3" | "unknown:<track>")
    identity: Identity
    first_seen: float
    last_seen: float
    tracks: set
    announced: bool = False
    hold_until: float = 0.0      # identify-settle deadline for unresolved entries
    pending_left_since: Optional[float] = None
    last_cam: str = ""
    last_track_key: str = ""
    last_posture: Optional[str] = None
    last_context: bool = True


class OccupancyTracker:
    """Holds live per-zone state and turns observations into salient edges.

    Track ids are per-camera, so they're namespaced `"<camId>:<trackId>"`. Identity
    only ever monotonically improves on a track (a higher-confidence read wins, and
    unknown→known is an upgrade); we never downgrade a known person to unknown on a
    frame where the face simply wasn't visible.

    Edges are emitted from the PRESENCE LEDGER (identity-level), never straight from
    track lifecycle: a detector losing a seated person for 30–170s (measured live
    2026-07-06 — ~700 false enter/leave edges in 6h) re-forms a new track, but the
    ledger just heals the same person. Named people key by their id (so two cameras
    seeing David in one zone = ONE person); unresolved people key per track but
    ADOPT a pending unknown in the zone (a re-detected stranger is presumed to be
    the stranger who just "vanished", so their flaps heal too — while two
    simultaneous strangers still count as two).
    """

    def __init__(self) -> None:
        self._tracks: Dict[str, _Track] = {}
        # The presence ledger: (zone, identity-key) -> _Presence.
        self._presence: Dict[Tuple[str, str], _Presence] = {}
        # Per-zone re-arm memory: identity-key -> ts it last (confirmed) left. A
        # re-entry inside rewake_cooldown_s is a continuation (no new wake).
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

            # Cross to "present" after enter_frames consecutive sightings (debounce),
            # then keep the track registered in the ledger; a resolved/improved
            # identity re-registers under the person's real key.
            if not tr.present and tr.hits >= cfg.enter_frames:
                tr.present = True
                tr.announced_identity = self._register(tr, edges, now)
            elif tr.present:
                want = self._ledger_key(tr)
                if tr.announced_identity != want:
                    continued = self._unregister_for_upgrade(tr)
                    tr.announced_identity = self._register(tr, edges, now, continued)
                else:
                    self._refresh(tr, now)

        self._expire(cam_id, zone, seen_keys, now)
        edges += self._sweep(now)
        return edges

    # ── the presence ledger ───────────────────────────────────────────────────
    def _ledger_key(self, tr: _Track) -> str:
        # Unresolved people can't be matched by identity, so they key per track —
        # adoption (below) is what heals their flaps.
        return tr.identity.key() if tr.identity.id else f"unknown:{tr.key}"

    def _register(self, tr: _Track, edges: List[Edge], now: float,
                  continued: Optional[_Presence] = None) -> str:
        """Attach a present track to its person's ledger entry (creating/healing/
        adopting as needed). `continued` is the entry this track just vacated on an
        identity upgrade — the SAME person under a better key, so its announcement
        and arrival time carry over (an upgrade emits identified, never a second
        entered). Returns the ledger key the track now supports."""
        key = self._ledger_key(tr)
        entry = self._presence.get((tr.zone, key))

        # Adoption: a NEW unresolved track in a zone where an unresolved person is
        # pending-left is presumed to be that same person re-detected.
        if entry is None and not tr.identity.id:
            adopted = self._oldest_pending_unknown(tr.zone)
            if adopted is not None:
                del self._presence[(adopted.zone, adopted.key)]
                adopted.key = key
                self._presence[(tr.zone, key)] = adopted
                entry = adopted

        if entry is not None:
            entry.tracks.add(tr.key)
            entry.pending_left_since = None  # heal — they never really left
            self._touch(entry, tr, now)
            return key

        entry = _Presence(zone=tr.zone, key=key, identity=tr.identity,
                          first_seen=now, last_seen=now, tracks={tr.key},
                          hold_until=(now + cfg.identify_settle_s
                                      if tr.identity.cls == "unknown" else now))
        self._presence[(tr.zone, key)] = entry
        self._touch(entry, tr, now)
        # A named person turning up HERE confirms any pending leave elsewhere at
        # once — the move between rooms should read entered+left, promptly.
        if tr.identity.id:
            edges.extend(self._confirm_elsewhere(key, tr.zone, now))
        if continued is not None and continued.announced:
            # Continuation of an already-announced person: keep their arrival time,
            # skip entered, and announce only what's NEW — the identity.
            entry.announced = True
            entry.first_seen = continued.first_seen
            if entry.identity.cls != "unknown":
                edges.append(self._identify_edge(entry, now))
            return key
        self._try_announce(entry, edges, now)
        return key

    def _unregister_for_upgrade(self, tr: _Track) -> Optional[_Presence]:
        """A present track re-keyed (unknown→named, guest→member, id-switch heal):
        move its support silently. An emptied entry is a CONTINUATION of the same
        person under the new key, so it dies without a left edge — never a phantom
        second person lingering in the count. Returns the vacated entry so the
        re-registration can inherit its announcement + arrival time."""
        old = self._presence.get((tr.zone, tr.announced_identity or ""))
        if old is None:
            return None
        old.tracks.discard(tr.key)
        if not old.tracks:
            del self._presence[(old.zone, old.key)]
            return old
        return None

    def _refresh(self, tr: _Track, now: float) -> None:
        entry = self._presence.get((tr.zone, tr.announced_identity or ""))
        if entry is not None:
            entry.tracks.add(tr.key)
            entry.pending_left_since = None
            self._touch(entry, tr, now)

    def _touch(self, entry: _Presence, tr: _Track, now: float) -> None:
        entry.last_seen = now
        entry.identity = _better(entry.identity, tr.identity)
        entry.last_cam = tr.cam_id
        entry.last_track_key = tr.key
        entry.last_posture = tr.posture
        entry.last_context = tr.context

    def _try_announce(self, entry: _Presence, edges: List[Edge], now: float) -> None:
        if entry.announced or now < entry.hold_until or not entry.tracks:
            return
        entry.announced = True  # set even when suppressed — stay quiet afterwards
        if self._suppressed_by_cooldown(entry.zone, entry.identity, now):
            return
        edges.append(self._mk_presence(EDGE_ENTERED, entry, now))
        if entry.identity.cls != "unknown":
            edges.append(self._identify_edge(entry, now))

    def _oldest_pending_unknown(self, zone: str) -> Optional[_Presence]:
        candidates = [e for e in self._presence.values()
                      if e.zone == zone and e.pending_left_since is not None
                      and not e.identity.id]
        return min(candidates, key=lambda e: e.pending_left_since) if candidates else None

    def _confirm_elsewhere(self, key: str, except_zone: str, now: float) -> List[Edge]:
        edges: List[Edge] = []
        for (zone, k), entry in list(self._presence.items()):
            if k == key and zone != except_zone and entry.pending_left_since is not None:
                edges.extend(self._confirm_left(entry, now))
        return edges

    # ── leave / empty ───────────────────────────────────────────────────────
    def _expire(self, cam_id: str, zone: str, seen_keys: set, now: float) -> None:
        """Tracks that stopped being observed age out after leave_grace_s — but only
        their LEDGER entry reacts (pending-left), never a direct edge."""
        for key, tr in list(self._tracks.items()):
            if tr.cam_id != cam_id or tr.zone != zone or key in seen_keys:
                continue
            if now - tr.last_seen <= cfg.leave_grace_s:
                continue
            entry = self._presence.get((tr.zone, tr.announced_identity or ""))
            if entry is not None:
                entry.tracks.discard(tr.key)
                if not entry.tracks:
                    if entry.announced:
                        entry.pending_left_since = now
                        entry.last_seen = tr.last_seen  # when they actually vanished
                    else:
                        # A blip that died before it settled — total silence.
                        del self._presence[(entry.zone, entry.key)]
            del self._tracks[key]

    def _sweep(self, now: float) -> List[Edge]:
        """Ledger heartbeat, run on every update (any camera): announce entries
        whose identify-settle expired, confirm leaves whose window ran out, and
        derive room_empty from the ledger."""
        edges: List[Edge] = []
        for entry in list(self._presence.values()):
            if not entry.announced and entry.tracks:
                self._try_announce(entry, edges, now)
            if entry.pending_left_since is not None and \
                    now - entry.pending_left_since >= cfg.leave_confirm_s:
                edges.extend(self._confirm_left(entry, now))
        self._recompute_zones(edges, now)
        return edges

    def _confirm_left(self, entry: _Presence, now: float) -> List[Edge]:
        del self._presence[(entry.zone, entry.key)]
        if not entry.announced:
            return []
        self._recent_left.setdefault(entry.zone, {})[entry.identity.key()] = now
        # ts = when they were last SEEN — the truthful departure moment — not the
        # (leave_confirm_s later) moment we stopped waiting for them.
        return [self._mk_presence(EDGE_LEFT, entry, entry.last_seen)]

    def _recompute_zones(self, edges: List[Edge], now: float) -> None:
        occupied_zones = {e.zone for e in self._presence.values() if e.announced}
        for zone, was in list(self._zone_occupied.items()):
            if was and zone not in occupied_zones:
                edges.append(Edge(EDGE_ROOM_EMPTY, zone, "", "", UNKNOWN, now))
        self._zone_occupied = {z: True for z in occupied_zones}

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
    def _identify_edge(self, entry: _Presence, now: float) -> Edge:
        edge = EDGE_GUEST_ARRIVED if entry.identity.cls == "guest" else EDGE_IDENTIFIED
        return self._mk_presence(edge, entry, now)

    def _mk_presence(self, edge: str, entry: _Presence, ts: float) -> Edge:
        return Edge(edge=edge, zone=entry.zone, cam_id=entry.last_cam,
                    track_key=entry.last_track_key, identity=entry.identity, ts=ts)

    def _mk(self, edge: str, tr: _Track, now: float) -> Edge:
        return Edge(edge=edge, zone=tr.zone, cam_id=tr.cam_id, track_key=tr.key,
                    identity=tr.identity, ts=now)

    def _suppressed_by_cooldown(self, zone: str, ident: Identity, now: float) -> bool:
        left_at = self._recent_left.get(zone, {}).get(ident.key())
        return left_at is not None and (now - left_at) < cfg.rewake_cooldown_s

    # ── privacy withdrawal ────────────────────────────────────────────────────
    def drop_camera(self, cam_id: str) -> None:
        """Withdraw one camera's observations SILENTLY (privacy mode): its tracks
        AND the ledger entries only they supported vanish from the snapshot without
        emitting left/room-empty edges — the people may well still be there, the
        camera just stopped looking, so fabricating `person_left` events would
        poison memory. Downstream (the hub's rooms model) goes stale-then-unknown
        via its vision TTL, which is the honest signal. Entries a second camera
        still supports stay."""
        dead = set()
        for key, tr in list(self._tracks.items()):
            if tr.cam_id != cam_id:
                continue
            dead.add(key)
            del self._tracks[key]
        for pkey, entry in list(self._presence.items()):
            supported_here = bool(entry.tracks & dead)
            entry.tracks -= dead
            if supported_here and not entry.tracks:
                del self._presence[pkey]
        self._zone_occupied = {
            z: True for z in {e.zone for e in self._presence.values() if e.announced}}

    # ── snapshot (pull surface — who_is_here) ─────────────────────────────────
    def snapshot(self, zone: Optional[str] = None, now: Optional[float] = None) -> dict:
        """Read from the LEDGER, not the tracks: a person mid-dropout (pending-left,
        no live track) is still shown — presence that flapped Empty↔David on every
        detector hiccup was the dashboard/agent symptom this fixes. Activity fields
        come from the best live supporting track; a ghost keeps its last posture and
        reads as not moving."""
        now = time.time() if now is None else now
        out: Dict[str, list] = {}
        for entry in self._presence.values():
            if zone and entry.zone != zone:
                continue
            tr = self._best_track(entry)
            item = {
                "track": tr.key if tr else entry.last_track_key,
                "since": entry.first_seen,
                **entry.identity.as_meta(),
            }
            context = tr.context if tr else entry.last_context
            if context:
                # T0 activity signals (§2): how long they've been here + whether they're
                # in motion right now (speed EMA vs the passing bar, camera-agnostic).
                # Identity-only tracks (satellite cams) omit ALL context fields, which is
                # what keeps them out of zone activity + T2a hints downstream.
                item["dwell_s"] = round(max(0.0, now - entry.first_seen), 1)
                item["moving"] = bool(tr and tr.speed >= cfg.activity_speed_fws)
                posture = tr.posture if tr else entry.last_posture
                if posture:
                    item["posture"] = posture  # T1 (§3), only once a pose engine read it
            out.setdefault(entry.zone, []).append(item)
        return out

    def _best_track(self, entry: _Presence) -> Optional[_Track]:
        live = [self._tracks[k] for k in entry.tracks if k in self._tracks]
        return max(live, key=lambda t: t.last_seen) if live else None

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
