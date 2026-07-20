"""Presence-ledger hysteresis — truthful edges under detector flap.

Measured live 2026-07-06 (yolov8 on seated people): dropouts killed a track and a
re-detection minted a new one 30–170s later — ~700 false enter/leave edges in 6h,
each one an agent wake and a memory lie. These tests pin the ledger semantics:
people flap-heal silently, blips never announce, lefts are confirmed + stamped
with the real disappearance time, and the dashboard snapshot never flickers.
"""
import pytest

from app.config import cfg
from app.occupancy import (EDGE_ENTERED, EDGE_GUEST_ARRIVED, EDGE_IDENTIFIED,
                           EDGE_LEFT, EDGE_ROOM_EMPTY, Identity, Observation,
                           OccupancyTracker)


_ASSUME_KNOBS = ("assume_identity", "assume_min_iou", "assume_radius_fw",
                 "edge_exit_margin_fw", "assume_conf_cap", "assume_conf_floor",
                 "assume_max_s", "adopt_block_s", "zone_sensor_stale_s",
                 "activity_speed_fws")


@pytest.fixture(autouse=True)
def knobs():
    old = (cfg.enter_frames, cfg.leave_grace_s, cfg.rewake_cooldown_s,
           cfg.leave_confirm_s, cfg.identify_settle_s)
    old_assume = {k: getattr(cfg, k) for k in _ASSUME_KNOBS}
    cfg.enter_frames, cfg.leave_grace_s, cfg.rewake_cooldown_s = 1, 5.0, 30.0
    cfg.leave_confirm_s, cfg.identify_settle_s = 120.0, 8.0
    # SMART_FACE_ID knobs pinned to their design defaults so these cases are deterministic
    # regardless of the env the suite runs under.
    cfg.assume_identity = True
    cfg.assume_min_iou, cfg.assume_radius_fw, cfg.edge_exit_margin_fw = 0.3, 0.15, 0.06
    cfg.assume_conf_cap, cfg.assume_conf_floor, cfg.assume_max_s = 0.5, 0.2, 3600.0
    cfg.adopt_block_s, cfg.zone_sensor_stale_s = 60.0, 30.0
    cfg.activity_speed_fws = 0.25
    yield
    (cfg.enter_frames, cfg.leave_grace_s, cfg.rewake_cooldown_s,
     cfg.leave_confirm_s, cfg.identify_settle_s) = old
    for k, v in old_assume.items():
        setattr(cfg, k, v)


def _david():
    return Identity(id="u1", name="David", cls="household", confidence=0.9)


def _ana():
    return Identity(id="u2", name="Ana", cls="household", confidence=0.85)


def _obs(track_id, ident=None, *, cx=0.5, cy=0.5, w=0.2, h=0.4,
         tainted=False, fw=1000, fh=1000):
    """Observation with a bbox centered at (cx, cy) in WIDTH-normalized units (both axes
    ÷ frame_w, matching occupancy's isotropic space) — so position-anchored adoption /
    the frame-edge exit test have real geometry to work with. A square frame (fh==fw)
    puts the y frame-bound at 1.0, so cy is the fraction from top."""
    x1, x2 = int((cx - w / 2) * fw), int((cx + w / 2) * fw)
    y1, y2 = int((cy - h / 2) * fw), int((cy + h / 2) * fw)
    kw = dict(bbox=(x1, y1, x2, y2), frame_w=fw, frame_h=fh, tainted=tainted)
    return Observation(track_id, ident, **kw) if ident else Observation(track_id, **kw)


def _edges(t, *args, **kw):
    return [e.edge for e in t.update(*args, **kw)]


def test_dropout_heals_with_zero_edges_and_snapshot_never_flickers():
    t = OccupancyTracker()
    assert _edges(t, "cam", "sala", [Observation("1", _david())], now=0.0) == \
        [EDGE_ENTERED, EDGE_IDENTIFIED]
    # Track dies (dropout), a NEW track re-detects the same person 40s later.
    assert _edges(t, "cam", "sala", [], now=10.0) == []          # past grace: silent
    assert t.snapshot("sala", now=10.0)["sala"], "ghost stays in the snapshot"
    assert _edges(t, "cam", "sala", [Observation("2", _david())], now=40.0) == []
    assert _edges(t, "cam", "sala", [Observation("2", _david())], now=41.0) == []
    # Dwell is continuous across the dropout — the person never "re-arrived".
    assert t.snapshot("sala", now=41.0)["sala"][0]["since"] == 0.0
    # And no left ever confirms — they came back.
    assert _edges(t, "cam", "sala", [Observation("2", _david())], now=200.0) == []


def test_true_leave_confirms_once_with_truthful_timestamp():
    t = OccupancyTracker()
    t.update("cam", "sala", [Observation("1", _david())], now=0.0)
    t.update("cam", "sala", [Observation("1", _david())], now=50.0)
    assert _edges(t, "cam", "sala", [], now=60.0) == []          # pending, silent
    e = t.update("cam", "sala", [], now=60.0 + cfg.leave_confirm_s + 1)
    assert [x.edge for x in e] == [EDGE_LEFT, EDGE_ROOM_EMPTY]
    left = next(x for x in e if x.edge == EDGE_LEFT)
    assert left.ts == 50.0, "left is stamped when they were last seen"
    assert left.identity.id == "u1"
    assert t.snapshot("sala") == {}


def test_unknown_blip_never_emits_anything():
    """A false detection (chair/shadow) that dies before the settle window."""
    t = OccupancyTracker()
    assert _edges(t, "cam", "sala", [Observation("9")], now=0.0) == []   # held
    assert _edges(t, "cam", "sala", [Observation("9")], now=2.0) == []
    assert _edges(t, "cam", "sala", [], now=9.0) == []                    # died silent
    assert _edges(t, "cam", "sala", [], now=300.0) == []                  # stays silent
    assert t.snapshot("sala") == {}


def test_unknown_resolving_within_settle_announces_once_with_the_name():
    t = OccupancyTracker()
    assert _edges(t, "cam", "sala", [Observation("9")], now=0.0) == []
    e = _edges(t, "cam", "sala", [Observation("9", _david())], now=2.0)
    assert e == [EDGE_ENTERED, EDGE_IDENTIFIED]                  # named from the start
    assert _edges(t, "cam", "sala", [Observation("9", _david())], now=3.0) == []


def test_unknown_that_never_resolves_announces_after_settle_then_flaps_heal():
    t = OccupancyTracker()
    assert _edges(t, "cam", "sala", [Observation("9")], now=0.0) == []
    e = _edges(t, "cam", "sala", [Observation("9")], now=cfg.identify_settle_s + 1)
    assert e == [EDGE_ENTERED]                                   # a real stranger
    # Their track flaps; the re-detected unknown ADOPTS the pending entry: silence.
    assert _edges(t, "cam", "sala", [], now=20.0) == []
    assert _edges(t, "cam", "sala", [Observation("10")], now=45.0) == []
    e = _edges(t, "cam", "sala", [Observation("10")], now=60.0)
    assert e == [], "adopted identity is already announced — settle does not re-fire"
    assert len(t.snapshot("sala", now=60.0)["sala"]) == 1


def test_flap_via_unknown_phase_heals_named_pending():
    """The common couch cycle: David's track dies, the re-detection starts unknown
    and resolves to David seconds later — zero edges end to end."""
    t = OccupancyTracker()
    t.update("cam", "sala", [Observation("1", _david())], now=0.0)
    assert _edges(t, "cam", "sala", [], now=10.0) == []          # David pending
    assert _edges(t, "cam", "sala", [Observation("2")], now=50.0) == []   # unknown, held
    e = _edges(t, "cam", "sala", [Observation("2", _david())], now=52.0)  # resolves
    assert e == [], "heal: no entered, no identified, no left"
    assert _edges(t, "cam", "sala", [Observation("2", _david())], now=250.0) == []


def test_two_simultaneous_strangers_count_as_two():
    t = OccupancyTracker()
    t.update("cam", "sala", [Observation("9"), Observation("10")], now=0.0)
    e = t.update("cam", "sala", [Observation("9"), Observation("10")],
                 now=cfg.identify_settle_s + 1)
    assert [x.edge for x in e] == [EDGE_ENTERED, EDGE_ENTERED]
    assert len(t.snapshot("sala")["sala"]) == 2


def test_same_person_on_two_cameras_is_one_person():
    t = OccupancyTracker()
    t.update("camA", "sala", [Observation("1", _david())], now=0.0)
    e = t.update("camB", "sala", [Observation("7", _david())], now=1.0)
    assert e == [], "second camera adds support, not a second arrival"
    assert len(t.snapshot("sala")["sala"]) == 1
    # camA's track dies — camB still supports the entry: nothing pends, ever.
    t.update("camA", "sala", [], now=10.0)
    e = t.update("camB", "sala", [Observation("7", _david())], now=200.0)
    assert e == []
    assert len(t.snapshot("sala")["sala"]) == 1


def test_cross_zone_move_confirms_the_old_zone_promptly():
    t = OccupancyTracker()
    t.update("cam1", "sala", [Observation("1", _david())], now=0.0)
    t.update("cam1", "sala", [Observation("1", _david())], now=30.0)
    t.update("cam1", "sala", [], now=40.0)                       # sala pending
    e = t.update("cam2", "cocina", [Observation("5", _david())], now=45.0)
    kinds = [x.edge for x in e]
    assert EDGE_ENTERED in kinds and EDGE_IDENTIFIED in kinds
    assert EDGE_LEFT in kinds and EDGE_ROOM_EMPTY in kinds       # sala closed NOW
    left = next(x for x in e if x.edge == EDGE_LEFT)
    assert left.zone == "sala" and left.ts == 30.0               # truthful timestamp
    assert list(t.snapshot(now=46.0).keys()) == ["cocina"]


def test_guest_arrival_still_announces():
    t = OccupancyTracker()
    guest = Identity(id="guest:1", name=None, cls="guest", confidence=0.4)
    e = _edges(t, "cam", "sala", [Observation("9", guest)], now=0.0)
    assert e == [EDGE_ENTERED, EDGE_GUEST_ARRIVED]


def test_privacy_drop_camera_stays_silent_but_keeps_other_camera_support():
    t = OccupancyTracker()
    t.update("camA", "sala", [Observation("1", _david())], now=0.0)
    t.update("camB", "sala", [Observation("7", _david())], now=1.0)
    t.drop_camera("camB")
    assert len(t.snapshot("sala")["sala"]) == 1                  # camA still sees him
    t.drop_camera("camA")
    assert t.snapshot("sala") == {}                              # gone, silently
    assert _edges(t, "camC", "cocina", [], now=500.0) == [], \
        "no deferred left/room_empty ever surfaces for privacy-dropped presence"


# ── SMART_FACE_ID: position-anchored identity persistence ─────────────────────
# A still person whose YOLO track drops out keeps their identity provisionally
# (assumed), verified when a face reads again; overlapping/mismatched reads void the
# assumption; the mmWave sensor — not vision — decides whether the room is still occupied.


def test_stillness_dropout_adopts_named_ghost_then_face_confirms():
    """David sits, his track drops to stillness, a NEW unresolved track re-detects at
    the SAME spot → adopted as (assumed) David: ONE entry, ZERO edges, hedged confidence.
    A face reading David again on that track clears the assumption at full confidence."""
    t = OccupancyTracker()
    assert _edges(t, "cam", "sala", [_obs("1", _david())], now=0.0) == \
        [EDGE_ENTERED, EDGE_IDENTIFIED]
    assert _edges(t, "cam", "sala", [], now=10.0) == []           # dropout: pending-left
    # New unresolved track at the same position, 40s later → adopted SILENTLY.
    assert _edges(t, "cam", "sala", [_obs("2", cx=0.51)], now=40.0) == []
    snap = t.snapshot("sala", now=41.0)["sala"]
    assert len(snap) == 1                                          # NOT David + a stranger
    assert snap[0]["name"] == "David" and snap[0]["assumed"] is True
    assert snap[0]["confidence"] <= cfg.assume_conf_cap           # capped hypothesis
    assert snap[0]["since"] == 0.0                                # dwell continuous
    # A face reads David on that same track → assumption cleared, full confidence back.
    assert _edges(t, "cam", "sala", [_obs("2", _david(), cx=0.51)], now=42.0) == []
    snap = t.snapshot("sala", now=42.0)["sala"]
    assert snap[0].get("assumed") is None
    assert snap[0]["confidence"] == pytest.approx(0.9)


def test_adoption_mismatch_reverts_ghost_and_announces_the_real_person():
    """The adopted track's face resolves to ANA, not David: David's entry reverts to a
    pending-left ghost (assumption dropped, last VERIFIED sighting intact), Ana announces
    as a genuinely new person (no inherited announcement), and David's eventual
    person_left fires ONCE with the truthful timestamp — never a solid double-count."""
    t = OccupancyTracker()
    t.update("cam", "sala", [_obs("1", _david())], now=0.0)
    t.update("cam", "sala", [_obs("1", _david())], now=30.0)
    assert _edges(t, "cam", "sala", [], now=40.0) == []           # David pending-left
    assert _edges(t, "cam", "sala", [_obs("2", cx=0.5)], now=45.0) == []   # adopted
    assert t.snapshot("sala", now=45.0)["sala"][0]["assumed"] is True
    # The face on that track is ANA — the position adoption was wrong.
    e = t.update("cam", "sala", [_obs("2", _ana(), cx=0.5)], now=50.0)
    kinds = [x.edge for x in e]
    assert EDGE_ENTERED in kinds and EDGE_IDENTIFIED in kinds     # Ana is a NEW person
    by_name = {p.get("name"): p for p in t.snapshot("sala", now=50.0)["sala"]}
    assert by_name["Ana"].get("assumed") is None and by_name["Ana"].get("pending_left") is None
    # David is a FADING ghost (pending-left), not a solid second present person.
    assert by_name["David"]["pending_left"] is True and by_name["David"].get("assumed") is None
    solid = [p for p in t.snapshot("sala", now=50.0)["sala"] if not p.get("pending_left")]
    assert len(solid) == 1 and solid[0]["name"] == "Ana"
    # David's person_left confirms once, stamped at his last VERIFIED sighting (30.0).
    e = t.update("cam", "sala", [_obs("2", _ana(), cx=0.5)], now=50.0 + cfg.leave_confirm_s + 1)
    lefts = [x for x in e if x.edge == EDGE_LEFT]
    assert len(lefts) == 1 and lefts[0].identity.id == "u1" and lefts[0].ts == 30.0
    remaining = t.snapshot("sala", now=50.0 + cfg.leave_confirm_s + 1)["sala"]
    assert len(remaining) == 1 and remaining[0]["name"] == "Ana"


def test_exit_signature_ghost_is_never_adopted_or_sensor_held():
    """David WALKS OUT (fast, center at a frame edge) then drops. His ghost carries an
    exit signature → a new track at that spot must NOT adopt him, and even an occupied
    sensor must NOT hold him — 'Ana leaves, David stays' can't resurrect a walk-out."""
    t = OccupancyTracker()
    t.update("cam", "sala", [_obs("1", _david(), cx=0.5)], now=0.0)
    # Fast move to the right edge (speed above the bar, center within the exit margin).
    t.update("cam", "sala", [_obs("1", _david(), cx=0.97)], now=0.5)
    assert _edges(t, "cam", "sala", [], now=10.0) == []           # exit-signature pending-left
    assert _edges(t, "cam", "sala", [_obs("2", cx=0.95)], now=20.0) == []  # new unknown, NO adopt
    snap = t.snapshot("sala", now=20.0)["sala"]
    assert not any(p.get("assumed") for p in snap)                # nobody assumed
    # Sensor reads occupied, yet an EXIT ghost still confirms left at leave_confirm_s.
    e = t.update("cam", "sala", [_obs("2", cx=0.95)],
                 now=10.0 + cfg.leave_confirm_s + 1, zone_occupied=True)
    assert any(x.edge == EDGE_LEFT and x.identity.id == "u1" for x in e)


def test_two_candidate_ghosts_block_adoption():
    """Two named people drop to stillness; a new track between them is in range of BOTH
    → ambiguous, so it adopts neither (guessing between people is the false-ID we avoid)."""
    t = OccupancyTracker()
    t.update("cam", "sala", [_obs("1", _david(), cx=0.4), _obs("2", _ana(), cx=0.6)], now=0.0)
    assert _edges(t, "cam", "sala", [], now=10.0) == []           # both pending-left dropouts
    assert _edges(t, "cam", "sala", [_obs("3", cx=0.5)], now=20.0) == []
    snap = t.snapshot("sala", now=20.0)["sala"]
    assert not any(p.get("assumed") for p in snap)


def test_tainted_track_cannot_adopt():
    """An overlap-tainted new track is barred from ghost adoption even at a perfect
    position match — the id-swap risk case must not crystallize a wrong identity."""
    t = OccupancyTracker()
    t.update("cam", "sala", [_obs("1", _david(), cx=0.5)], now=0.0)
    assert _edges(t, "cam", "sala", [], now=10.0) == []
    assert _edges(t, "cam", "sala", [_obs("2", cx=0.5, tainted=True)], now=20.0) == []
    assert not any(p.get("assumed") for p in t.snapshot("sala", now=20.0)["sala"])


def test_sensor_occupied_holds_named_dropout_past_leave_confirm():
    """The mmWave/PIR says the room is still occupied → a named dropout ghost is HELD as
    an assumed-present entry past leave_confirm_s, no person_left, zero edges."""
    t = OccupancyTracker()
    t.update("cam", "sala", [_obs("1", _david())], now=0.0)
    t.update("cam", "sala", [_obs("1", _david())], now=30.0)
    assert _edges(t, "cam", "sala", [], now=40.0, zone_occupied=True) == []
    e = t.update("cam", "sala", [], now=40.0 + cfg.leave_confirm_s + 5, zone_occupied=True)
    assert [x.edge for x in e] == []                              # held, never left
    snap = t.snapshot("sala", now=40.0 + cfg.leave_confirm_s + 5)["sala"]
    assert len(snap) == 1 and snap[0]["name"] == "David"
    assert snap[0]["assumed"] is True and snap[0]["pending_left"] is True


def test_sensor_hold_releases_left_when_zone_goes_empty():
    """A sensor-held ghost confirms left PROMPTLY (truthful ts) once the sensor flips
    to empty — the hold was contingent on live corroboration."""
    t = OccupancyTracker()
    t.update("cam", "sala", [_obs("1", _david())], now=0.0)
    t.update("cam", "sala", [_obs("1", _david())], now=30.0)
    t.update("cam", "sala", [], now=40.0, zone_occupied=True)
    assert _edges(t, "cam", "sala", [], now=200.0, zone_occupied=True) == []   # held
    assert t.snapshot("sala", now=200.0)["sala"][0]["assumed"] is True
    e = t.update("cam", "sala", [], now=205.0, zone_occupied=False)
    lefts = [x for x in e if x.edge == EDGE_LEFT]
    assert len(lefts) == 1 and lefts[0].ts == 30.0
    assert t.snapshot("sala") == {}


def test_no_sensor_data_keeps_todays_120s_behavior():
    """A zone the hub has no sensor for (zone_occupied always None) keeps the classic
    leave_confirm_s behaviour exactly — no hold, confirm at 120s."""
    t = OccupancyTracker()
    t.update("cam", "sala", [_obs("1", _david())], now=0.0)
    t.update("cam", "sala", [_obs("1", _david())], now=30.0)
    t.update("cam", "sala", [], now=40.0)                         # no zone_occupied, ever
    assert _edges(t, "cam", "sala", [], now=40.0 + cfg.leave_confirm_s - 1) == []
    e = t.update("cam", "sala", [], now=40.0 + cfg.leave_confirm_s + 1)
    lefts = [x for x in e if x.edge == EDGE_LEFT]
    assert len(lefts) == 1 and lefts[0].ts == 30.0


def test_stale_sensor_reading_falls_back_to_120s_leave():
    """A sensor reading that has gone stale (older than zone_sensor_stale_s) is treated
    as no corroboration → the ghost confirms left at leave_confirm_s, not held forever."""
    t = OccupancyTracker()
    t.update("cam", "sala", [_obs("1", _david())], now=0.0)
    t.update("cam", "sala", [_obs("1", _david())], now=30.0)
    t.update("cam", "sala", [], now=40.0, zone_occupied=True)     # last reading at 40.0
    # By 40+121 the reading is ~121s old (>> zone_sensor_stale_s) → confirm left.
    e = t.update("cam", "sala", [], now=40.0 + cfg.leave_confirm_s + 1)
    lefts = [x for x in e if x.edge == EDGE_LEFT]
    assert len(lefts) == 1 and lefts[0].ts == 30.0


def test_kill_switch_off_disables_adoption_and_sensor_hold():
    """assume_identity=False → the whole feature no-ops: no position adoption, no
    sensor-hold, exactly today's 120s-then-left behaviour."""
    cfg.assume_identity = False
    t = OccupancyTracker()
    t.update("cam", "sala", [_obs("1", _david())], now=0.0)
    assert _edges(t, "cam", "sala", [], now=10.0) == []
    assert _edges(t, "cam", "sala", [_obs("2", cx=0.5)], now=20.0) == []
    assert not any(p.get("assumed") for p in t.snapshot("sala", now=20.0)["sala"])
    e = t.update("cam", "sala", [_obs("2", cx=0.5)],
                 now=10.0 + cfg.leave_confirm_s + 1, zone_occupied=True)
    assert any(x.edge == EDGE_LEFT and x.identity.id == "u1" for x in e)

# ── Ghost expiry (2026-07-19 live-defect fix — DATA_INTEGRITY_FOUNDATION Phase 1.5) ─
# Observed live: sala accumulated ~109 immortal ghosts (pending_left+assumed entries,
# guest dwells past 53h) because (a) the sensor-hold had NO lifetime and any one real
# person kept the zone sensor occupied, (b) conf-0.2 `guest:<n>` clusters counted as
# "named" for the hold, and (c) the sweep only ran on frame arrival, so an idle camera
# never reaped anything. These pin the three bounds.


def _guest():
    return Identity(id="guest:7", name=None, cls="guest", confidence=0.2)


def test_sensor_hold_expires_after_assume_max_s_even_if_sensor_stays_occupied():
    """A household ghost is held by the occupied sensor only for assume_max_s of
    unverified coasting; past that window it confirms left (truthful last-seen ts)
    even though the sensor STILL reads occupied — the sensor says 'someone is here',
    never 'David is here', so it must not sustain a name indefinitely."""
    t = OccupancyTracker()
    t.update("cam", "sala", [_obs("1", _david())], now=0.0)
    t.update("cam", "sala", [_obs("1", _david())], now=30.0)
    t.update("cam", "sala", [], now=40.0, zone_occupied=True)      # dropout: pending-left
    hold_at = 40.0 + cfg.leave_confirm_s + 1
    assert _edges(t, "cam", "sala", [], now=hold_at, zone_occupied=True) == []  # held
    snap = t.snapshot("sala", now=hold_at)["sala"]
    assert snap[0]["assumed"] is True and snap[0]["pending_left"] is True
    # Still held mid-window…
    assert _edges(t, "cam", "sala", [],
                  now=hold_at + cfg.assume_max_s - 10, zone_occupied=True) == []
    # …but past assume_max_s of coasting the hold expires — sensor occupied or not.
    e = t.update("cam", "sala", [], now=hold_at + cfg.assume_max_s + 1, zone_occupied=True)
    lefts = [x for x in e if x.edge == EDGE_LEFT]
    assert len(lefts) == 1 and lefts[0].identity.id == "u1" and lefts[0].ts == 30.0
    assert t.snapshot("sala") == {}


def test_guest_ghost_is_never_sensor_held():
    """A low-confidence guest cluster (`guest:<n>`, conf ~0.2) is a noise hypothesis,
    not a verified person: an occupied sensor must NOT hold its ghost — it confirms
    left at leave_confirm_s exactly like the sensorless case. (Live: every flapped
    guest track mints a FRESH guest key, so held guests never healed — they piled up.)"""
    t = OccupancyTracker()
    assert _edges(t, "cam", "sala", [_obs("1", _guest())], now=0.0) == \
        [EDGE_ENTERED, EDGE_GUEST_ARRIVED]
    t.update("cam", "sala", [_obs("1", _guest())], now=30.0)
    t.update("cam", "sala", [], now=40.0, zone_occupied=True)      # dropout: pending-left
    e = t.update("cam", "sala", [], now=40.0 + cfg.leave_confirm_s + 1, zone_occupied=True)
    lefts = [x for x in e if x.edge == EDGE_LEFT]
    assert len(lefts) == 1 and lefts[0].identity.id == "guest:7" and lefts[0].ts == 30.0
    assert not any(p.get("assumed")
                   for z in t.snapshot(now=40.0 + cfg.leave_confirm_s + 1).values() for p in z)
    assert t.snapshot("sala") == {}


def test_reap_confirms_stuck_pending_left_without_any_frames():
    """A ghost already pending-left expires via reap() alone — zero update() calls —
    with the same truthful left + room_empty a frame-driven sweep would emit."""
    t = OccupancyTracker()
    t.update("cam", "sala", [_obs("1", _david())], now=0.0)
    t.update("cam", "sala", [_obs("1", _david())], now=30.0)
    t.update("cam", "sala", [], now=40.0)                          # pending-left at 40
    # Inside the window reap is silent (debounce intact) and the ghost still shows.
    assert [e.edge for e in t.reap(now=40.0 + cfg.leave_confirm_s - 5)] == []
    assert t.snapshot("sala", now=100.0)["sala"][0]["pending_left"] is True
    e = t.reap(now=40.0 + cfg.leave_confirm_s + 1)
    assert [x.edge for x in e] == [EDGE_LEFT, EDGE_ROOM_EMPTY]
    assert next(x for x in e if x.edge == EDGE_LEFT).ts == 30.0
    assert t.snapshot("sala") == {}


def test_reap_expires_tracks_of_an_idle_camera():
    """A camera that stops calling update() entirely (offline/stalled reader) used to
    leave its live tracks — and the ledger entries they support — frozen forever.
    reap() ages those tracks out globally, then runs the normal pending→left path."""
    t = OccupancyTracker()
    t.update("cam", "sala", [_obs("1", _david())], now=0.0)        # …then the camera dies
    edges = t.reap(now=50.0)                                       # past leave_grace_s
    assert [e.edge for e in edges] == []                           # silent: pending, not left
    snap = t.snapshot("sala", now=50.0)["sala"]
    assert snap[0]["name"] == "David" and snap[0]["pending_left"] is True
    e = t.reap(now=50.0 + cfg.leave_confirm_s + 1)
    assert [x.edge for x in e] == [EDGE_LEFT, EDGE_ROOM_EMPTY]
    assert next(x for x in e if x.edge == EDGE_LEFT).ts == 0.0     # last actually seen
    assert t.snapshot("sala") == {}


def test_reap_does_not_regress_flap_healing():
    """The person-level debounce survives the reaper: a reap inside the leave-confirm
    window emits nothing, and a re-detection after that reap still heals with ZERO
    edges and continuous dwell — reap must never turn a dropout into a false leave."""
    t = OccupancyTracker()
    t.update("cam", "sala", [_obs("1", _david())], now=0.0)
    t.update("cam", "sala", [], now=10.0)                          # dropout: pending-left
    assert [e.edge for e in t.reap(now=60.0)] == []                # mid-window: silent
    assert _edges(t, "cam", "sala", [_obs("2", _david())], now=80.0) == []  # heals, silent
    assert t.snapshot("sala", now=81.0)["sala"][0]["since"] == 0.0  # dwell continuous
    assert [e.edge for e in t.reap(now=85.0)] == []                # live track: untouched
    assert t.snapshot("sala", now=85.0)["sala"][0].get("pending_left") is None
