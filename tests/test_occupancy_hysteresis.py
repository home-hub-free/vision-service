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


@pytest.fixture(autouse=True)
def knobs():
    old = (cfg.enter_frames, cfg.leave_grace_s, cfg.rewake_cooldown_s,
           cfg.leave_confirm_s, cfg.identify_settle_s)
    cfg.enter_frames, cfg.leave_grace_s, cfg.rewake_cooldown_s = 1, 5.0, 30.0
    cfg.leave_confirm_s, cfg.identify_settle_s = 120.0, 8.0
    yield
    (cfg.enter_frames, cfg.leave_grace_s, cfg.rewake_cooldown_s,
     cfg.leave_confirm_s, cfg.identify_settle_s) = old


def _david():
    return Identity(id="u1", name="David", cls="household", confidence=0.9)


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