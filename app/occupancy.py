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
    # Frame height in pixels (SMART_FACE_ID): with bbox width-normalized by frame_w, the
    # y axis runs [0, frame_h/frame_w], so the frame-edge (walked-out) test needs the
    # bottom bound. Optional/default 0 — older callers and null builds omit it and the
    # exit test simply treats the frame as square (harmless — it only affects which
    # dropouts are exempt from adoption).
    frame_h: int = 0
    # T1 (§3): coarse body state from pose, when the pose engine ran on this frame.
    posture: Optional[str] = None  # "standing" | "sitting" | "lying" | "bent"
    # Whether the source camera is context-capable (Camera.context_capable): satellite/
    # ESP32 cams are face-ID-only — too low-quality for full-body inference — so their
    # tracks carry identity but NO T0/T1/T2a context signals downstream.
    context: bool = True
    # SMART_FACE_ID overlap taint: the camera worker sets this when this track's box
    # overlaps another person's past cfg.overlap_taint_iou — the id-swap risk moment. A
    # tainted track can't adopt a named ghost, and an entry supported only by tainted
    # tracks reads `assumed` until a clean confirm. Default False (older callers/null).
    tainted: bool = False


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
    # SMART_FACE_ID: the width-normalized bbox (x1,y1,x2,y2 all ÷ frame_w, isotropic —
    # same space as norm_cx/cy) the ledger remembers as a person's last position for
    # position-anchored adoption; and the y frame-bound (frame_h/frame_w) for the
    # frame-edge exit test. Both stay None/0 until a bbox-carrying obs updates motion.
    norm_bbox: Optional[Tuple[float, float, float, float]] = None
    norm_ymax: float = 0.0
    # Mirrors Observation.tainted for the current frame (overlap id-swap suspicion).
    tainted: bool = False
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
    # ── SMART_FACE_ID: position-anchored identity persistence ──────────────────
    # The last width-normalized bbox a supporting track was seen at (stamped when the
    # last support drops), the anchor a re-detected still person is matched against.
    last_bbox_n: Optional[Tuple[float, float, float, float]] = None
    # True when the last dropout looked like WALKING OUT of frame (exit signature) —
    # exempt from adoption + sensor-hold (the person really left). False = stillness.
    exit_signature: bool = False
    # `assumed` = presence is a POSITION/SENSOR hypothesis, not a live face read: name
    # kept, confidence capped + decaying, last_seen frozen at the last VERIFIED sighting
    # (so person_left stays truthful and a revert is trivial). `assumed_since` anchors
    # the decay; `assumed_from_pending` remembers the pending_left_since adoption cleared
    # (restored verbatim on a mismatch revert); `adopt_block_until` bars re-adoption/hold
    # of this entry after a mismatch disproved it.
    assumed: bool = False
    assumed_since: float = 0.0
    assumed_from_pending: Optional[float] = None
    adopt_block_until: float = 0.0


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
        # SMART_FACE_ID: track-key -> ledger-key an UNRESOLVED track was adopted onto
        # (named ghost adoption). `_ledger_key` consults it so an adopted track keeps
        # supporting the named entry instead of re-minting an unknown one.
        self._adopted: Dict[str, str] = {}
        # Freshest sensor occupancy per zone (mmWave/PIR, injected by the worker):
        # zone -> (occupied, ts). Owns whether a dropout ghost is held past leave_confirm_s.
        self._zone_sensor: Dict[str, Tuple[bool, float]] = {}

    # ── ingest ────────────────────────────────────────────────────────────────
    def update(
        self,
        cam_id: str,
        zone: str,
        observations: List[Observation],
        now: Optional[float] = None,
        zone_occupied: Optional[bool] = None,
    ) -> List[Edge]:
        now = time.time() if now is None else now
        zone = zone or "_"
        # SMART_FACE_ID: remember the freshest sensor occupancy for this zone (kwarg —
        # the tracker stays pure/no-I/O; the worker reads zone_presence and injects it).
        # None = no reading this pass (leaves any prior value to age out via staleness).
        if zone_occupied is not None:
            self._zone_sensor[zone] = (bool(zone_occupied), now)
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
            tr.tainted = obs.tainted
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
                    continued = self._unregister_for_upgrade(tr, now)
                    tr.announced_identity = self._register(tr, edges, now, continued)
                else:
                    self._refresh(tr, now)

        self._expire(cam_id, zone, seen_keys, now)
        edges += self._sweep(now)
        return edges

    # ── the presence ledger ───────────────────────────────────────────────────
    def _ledger_key(self, tr: _Track) -> str:
        # Unresolved people can't be matched by identity, so they key per track —
        # adoption (below) is what heals their flaps. A track adopted onto a NAMED ghost
        # (SMART_FACE_ID) keeps supporting that ghost's key while still unresolved, so
        # update()'s want-vs-announced check doesn't immediately re-mint it as unknown.
        if tr.identity.id:
            return tr.identity.key()
        adopted = self._adopted.get(tr.key)
        return adopted if adopted is not None else f"unknown:{tr.key}"

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

        # Named adoption (SMART_FACE_ID): failing an unknown ghost, a new unresolved
        # track at the remembered POSITION of a named dropout-signature ghost is
        # presumed to be that person, re-detected after a stillness dropout. Provisional
        # (`assumed`), verified for free when a face reads on this same track. Silent —
        # the entry is already announced, so adoption emits ZERO edges. Kill switch off
        # (assume_identity=False) → this whole block no-ops (today's behaviour).
        if entry is None and not tr.identity.id and cfg.assume_identity:
            ghost = self._adoptable_named_ghost(tr, now)
            if ghost is not None:
                self._adopted[tr.key] = ghost.key
                key = ghost.key
                ghost.assumed = True
                ghost.assumed_since = now
                ghost.assumed_from_pending = ghost.pending_left_since
                entry = ghost

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

    def _unregister_for_upgrade(self, tr: _Track, now: float) -> Optional[_Presence]:
        """A present track re-keyed (unknown→named, guest→member, id-switch heal):
        move its support silently. An emptied entry is a CONTINUATION of the same
        person under the new key, so it dies without a left edge — never a phantom
        second person lingering in the count. Returns the vacated entry so the
        re-registration can inherit its announcement + arrival time."""
        old = self._presence.get((tr.zone, tr.announced_identity or ""))
        if old is None:
            return None
        old.tracks.discard(tr.key)
        self._adopted.pop(tr.key, None)
        if old.tracks:
            return None
        # SMART_FACE_ID mismatch: an emptied ASSUMED entry vacated by a track that
        # resolved to a DIFFERENT identity is NOT a continuation — the face just proved
        # the position-adoption wrong. Revert it to exactly the pending-left state
        # adoption cleared (last_seen is still the last VERIFIED sighting, so the
        # eventual person_left is truthful), bar re-adoption/re-hold briefly, and inherit
        # NOTHING (the resolved person is genuinely new → gets their own entered).
        if old.assumed and old.key != self._ledger_key(tr):
            old.assumed = False
            old.assumed_since = 0.0
            old.pending_left_since = (old.assumed_from_pending
                                      if old.assumed_from_pending is not None else old.last_seen)
            old.assumed_from_pending = None
            old.adopt_block_until = now + cfg.adopt_block_s
            return None  # entry stays in _presence as a pending-left ghost, NOT inherited
        del self._presence[(old.zone, old.key)]
        return old

    def _refresh(self, tr: _Track, now: float) -> None:
        entry = self._presence.get((tr.zone, tr.announced_identity or ""))
        if entry is not None:
            entry.tracks.add(tr.key)
            entry.pending_left_since = None
            self._touch(entry, tr, now)

    def _touch(self, entry: _Presence, tr: _Track, now: float) -> None:
        # SMART_FACE_ID: a face that reads the SAME person on a supporting track VERIFIES
        # an assumed entry — clear the assumption (full confidence, live sighting again).
        if entry.assumed and tr.identity.id and entry.identity.id \
                and tr.identity.id == entry.identity.id:
            entry.assumed = False
            entry.assumed_since = 0.0
            entry.assumed_from_pending = None
        # While still assumed, DON'T advance last_seen — keep it at the last VERIFIED
        # sighting so person_left stays truthful and a mismatch revert is trivial. The
        # supporting (adopted, unresolved) track is a hypothesis, not a confirmation.
        if not entry.assumed:
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

    def _adoptable_named_ghost(self, tr: _Track, now: float) -> Optional[_Presence]:
        """The ONE named, dropout-signature ghost on this camera whose remembered
        position `tr` re-detects — or None. Ambiguity is fatal by design: two candidate
        ghosts in range, a tainted new track, OR a second unresolved track also at that
        spot → adopt nobody (guessing between people is exactly the false-ID we avoid).
        A track with no position (null build / face-only cam) can't be anchored either."""
        if tr.tainted or tr.norm_bbox is None:
            return None
        candidates = [
            e for e in self._presence.values()
            if e.zone == tr.zone and e.pending_left_since is not None
            and e.identity.id and not e.exit_signature and not e.assumed
            and e.last_cam == tr.cam_id and now >= e.adopt_block_until
            and e.last_bbox_n is not None
            and self._position_match(e.last_bbox_n, tr.norm_bbox)
        ]
        if len(candidates) != 1:
            return None
        ghost = candidates[0]
        # Two claimants: another unresolved live track also sits at this ghost's spot →
        # we can't tell which one is the returning person, so adopt neither.
        for other in self._tracks.values():
            if other.key == tr.key or other.zone != tr.zone or other.cam_id != tr.cam_id:
                continue
            if other.identity.id or other.norm_bbox is None:
                continue
            if self._position_match(ghost.last_bbox_n, other.norm_bbox):
                return None
        return ghost

    def _position_match(self, a: Tuple[float, float, float, float],
                        b: Tuple[float, float, float, float]) -> bool:
        """Width-normalized bbox `a` (a ghost's last spot) vs `b` (a live track): a match
        is IoU ≥ assume_min_iou OR center distance ≤ assume_radius_fw. Either suffices —
        a re-detected still person's box drifts a little between the drop and re-detect."""
        if _bbox_iou(a, b) >= cfg.assume_min_iou:
            return True
        acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
        bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        return math.hypot(acx - bcx, acy - bcy) <= cfg.assume_radius_fw

    def _exit_signature(self, tr: _Track) -> bool:
        """Did this track WALK OUT of frame (vs drop out to stillness)? Exit = last
        speed ≥ the passing bar AND its last center within edge_exit_margin_fw of a frame
        boundary. Exits are exempt from adoption + sensor-hold (the person really left);
        everything else is a stillness dropout, the case this feature persists across."""
        if tr.speed < cfg.activity_speed_fws or tr.norm_cx is None or tr.norm_cy is None:
            return False
        ymax = tr.norm_ymax if tr.norm_ymax > 0 else 1.0
        dist = min(tr.norm_cx, 1.0 - tr.norm_cx, tr.norm_cy, ymax - tr.norm_cy)
        return dist <= cfg.edge_exit_margin_fw

    def _zone_sensor_state(self, zone: str, now: float) -> Optional[bool]:
        """The freshest injected sensor occupancy for a zone, or None when there is no
        reading or it has gone stale (older than zone_sensor_stale_s) — None means "no
        corroboration", which keeps the sensorless 120s leave behaviour."""
        v = self._zone_sensor.get(zone)
        if v is None:
            return None
        occupied, ts = v
        if now - ts > cfg.zone_sensor_stale_s:
            return None
        return occupied

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
            self._drop_track(tr, now)

    def _drop_track(self, tr: _Track, now: float) -> None:
        """Retire one aged-out track and let its LEDGER entry react (pending-left) —
        never a direct edge. Shared by the per-frame `_expire` and the frame-
        independent `reap` (2026-07-19 ghost fix)."""
        entry = self._presence.get((tr.zone, tr.announced_identity or ""))
        if entry is not None:
            entry.tracks.discard(tr.key)
            if not entry.tracks:
                if entry.announced:
                    entry.pending_left_since = now
                    # SMART_FACE_ID: remember WHERE this last support was and whether
                    # it walked out (exit) or dropped to stillness — the anchor + gate
                    # for position-adoption and the sensor-hold. last_seen stays at the
                    # last VERIFIED sighting for an assumed entry (the dropping track
                    # was itself only a hypothesis) — never advance it to an unverified
                    # track's last frame.
                    if not entry.assumed:
                        entry.last_seen = tr.last_seen  # when they actually vanished
                    if tr.norm_bbox is not None:
                        entry.last_bbox_n = tr.norm_bbox
                    entry.exit_signature = self._exit_signature(tr)
                else:
                    # A blip that died before it settled — total silence.
                    del self._presence[(entry.zone, entry.key)]
        self._adopted.pop(tr.key, None)
        del self._tracks[tr.key]

    def reap(self, now: Optional[float] = None) -> List[Edge]:
        """Time-based ledger heartbeat, INDEPENDENT of frame arrival (2026-07-19 ghost
        fix). `_sweep` used to run only inside `update()` — i.e. only when some camera
        delivered a frame — so a stalled/offline/privacy-paused pipeline left every
        pending-left ghost (and the idle camera's live tracks) frozen in the snapshot
        forever. This is the same sweep `update()` runs, plus a global pass expiring
        tracks NO camera has reported for leave_grace_s (an idle camera never calls
        `_expire` for its own zone). Called by the periodic reaper thread (app/reaper.py);
        emits the same debounced edges update() would, so leaves stay truthful."""
        now = time.time() if now is None else now
        for tr in list(self._tracks.values()):
            if now - tr.last_seen > cfg.leave_grace_s:
                self._drop_track(tr, now)
        return self._sweep(now)

    def _sweep(self, now: float) -> List[Edge]:
        """Ledger heartbeat, run on every update (any camera): announce entries
        whose identify-settle expired, confirm leaves whose window ran out, and
        derive room_empty from the ledger."""
        edges: List[Edge] = []
        for entry in list(self._presence.values()):
            if not entry.announced and entry.tracks:
                self._try_announce(entry, edges, now)
            if entry.pending_left_since is None or \
                    now - entry.pending_left_since < cfg.leave_confirm_s:
                continue
            # Past the leave-confirm window. SMART_FACE_ID sensor-corroborated hold: a
            # HOUSEHOLD, dropout-signature ghost is HELD as an assumed-present entry (no
            # left edge) while the zone's mmWave/PIR still reads occupied — vision lost
            # the track to stillness but the sensor says someone is here. Anything else
            # (an unknown, a guest, an exit-signature walk-out, a just-reverted entry
            # inside its block, or a zone whose sensor reads empty / stale / None)
            # confirms left promptly, exactly as before. Zones with no sensor (None)
            # keep the 120s behaviour.
            #
            # 2026-07-19 ghost fix — two hard bounds this hold was missing:
            #   * household-only (`cls`, not a truthy id): a conf-0.2 `guest:<n>` cluster
            #     is a noise hypothesis, not a verified person — held guests piled up
            #     ~109 immortal ghosts in sala (each flapped guest track mints a FRESH
            #     key, so they never heal into each other, and any one real person kept
            #     the zone sensor reading occupied → held forever);
            #   * a TTL: the hold now expires after assume_max_s of coasting without a
            #     verifying face read — the same window the confidence decay is
            #     documented against (by then we're at the floor: stop believing). The
            #     sensor corroborates "someone is here", never "THIS person is here",
            #     so it must not sustain a name indefinitely.
            if (cfg.assume_identity and entry.identity.cls == "household"
                    and not entry.exit_signature
                    and now >= entry.adopt_block_until
                    and self._zone_sensor_state(entry.zone, now) is True):
                if not entry.assumed:
                    entry.assumed = True
                    entry.assumed_since = now
                if now - entry.assumed_since < cfg.assume_max_s:
                    continue
                # else: fall through — held past the coast window, confirm the leave
                # (truthfully stamped at the last VERIFIED sighting).
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
        # SMART_FACE_ID: the width-normalized bbox (isotropic — same units as the center)
        # the ledger anchors adoption to, plus the y frame-bound for the frame-edge exit
        # test. frame_h absent (older callers) → treat the frame as square (norm_ymax 1.0).
        tr.norm_bbox = (x1 / obs.frame_w, y1 / obs.frame_w, x2 / obs.frame_w, y2 / obs.frame_w)
        if obs.frame_h > 0:
            tr.norm_ymax = obs.frame_h / obs.frame_w

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
            self._adopted.pop(key, None)  # SMART_FACE_ID: drop this cam's adoption links
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
            # SMART_FACE_ID (additive — as_meta() is untouched, it rides MQTT edges): mark
            # a position/sensor-assumed identity so the dashboard hedges the name ("David?")
            # and cap/decay its confidence (a hypothesis, not a read). Mark a mid-dropout
            # ghost pending_left so consumers can treat it as fading rather than solid.
            if entry.assumed:
                item["assumed"] = True
                item["confidence"] = self._assumed_conf(entry, now)
            if entry.pending_left_since is not None:
                item["pending_left"] = True
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

    def _assumed_conf(self, entry: _Presence, now: float) -> float:
        """Confidence for an assumed entry: capped at assume_conf_cap when adopted/held,
        then decayed linearly to assume_conf_floor over assume_max_s of coasting without
        a face read. Never returns above the identity's own confidence, never below floor."""
        base = min(entry.identity.confidence, cfg.assume_conf_cap)
        floor = cfg.assume_conf_floor
        if base <= floor or cfg.assume_max_s <= 0:
            return round(base, 3)
        frac = min(1.0, max(0.0, now - entry.assumed_since) / cfg.assume_max_s)
        return round(base - (base - floor) * frac, 3)

    def who_is_here(self, zone: Optional[str] = None) -> List[dict]:
        snap = self.snapshot(zone)
        people: List[dict] = []
        for z, occ in snap.items():
            for p in occ:
                people.append({"zone": z, **p})
        return people


def _bbox_iou(a: Tuple[float, float, float, float],
              b: Tuple[float, float, float, float]) -> float:
    """IoU of two boxes (SMART_FACE_ID position gate). Kept local to the pure ledger —
    the same formula as perception.bbox_iou, but occupancy stays free of the perception
    module so it remains model-free and unit-testable without pixels."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter or 1.0)


def _better(current: Identity, incoming: Identity) -> Identity:
    """Monotonic identity merge: prefer a known person over unknown, then the higher
    confidence. Never demote a known identity to unknown on a face-less frame."""
    if incoming.cls == "unknown":
        return current
    if current.cls == "unknown":
        return incoming
    return incoming if incoming.confidence > current.confidence else current
