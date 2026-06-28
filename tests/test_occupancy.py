"""Occupancy debounce / salience edges — the §8 push-vs-pull behaviour, no models."""
from app.config import cfg
from app.occupancy import (EDGE_ENTERED, EDGE_GUEST_ARRIVED, EDGE_IDENTIFIED,
                           EDGE_LEFT, EDGE_ROOM_EMPTY, Identity, Observation,
                           OccupancyTracker)


def _household():
    return Identity(id="u1", name="David", cls="household", confidence=0.9)


def test_enter_after_n_frames_then_leave_and_empty():
    cfg.enter_frames, cfg.leave_grace_s, cfg.rewake_cooldown_s = 2, 5, 30
    t = OccupancyTracker()
    assert t.update("cam1", "sala", [Observation("1")], now=100.0) == []      # 1st hit: no edge
    e = t.update("cam1", "sala", [Observation("1")], now=100.2)               # 2nd hit: present
    assert any(x.edge == EDGE_ENTERED for x in e)
    assert t.who_is_here("sala"), "snapshot should show the person present"
    assert t.update("cam1", "sala", [], now=101.0) == []                       # within grace: no leave
    e = t.update("cam1", "sala", [], now=110.0)                                # past grace: leave + empty
    edges = {x.edge for x in e}
    assert EDGE_LEFT in edges and EDGE_ROOM_EMPTY in edges
    assert t.who_is_here("sala") == []


def test_known_identity_emits_identify_with_enter():
    cfg.enter_frames = 1
    t = OccupancyTracker()
    e = t.update("cam", "sala", [Observation("1", _household())], now=1.0)
    edges = {x.edge for x in e}
    assert EDGE_ENTERED in edges and EDGE_IDENTIFIED in edges


def test_guest_arrival_edge():
    cfg.enter_frames = 1
    t = OccupancyTracker()
    guest = Identity(id="guest:1", name=None, cls="guest", confidence=0.4)
    e = t.update("cam", "sala", [Observation("1", guest)], now=1.0)
    assert any(x.edge == EDGE_GUEST_ARRIVED for x in e)


def test_identify_after_unknown_entry():
    cfg.enter_frames = 1
    t = OccupancyTracker()
    t.update("cam", "sala", [Observation("9")], now=1.0)                       # entered unknown
    e = t.update("cam", "sala", [Observation("9", _household())], now=1.2)     # face resolves later
    assert any(x.edge == EDGE_IDENTIFIED for x in e)
    assert not any(x.edge == EDGE_ENTERED for x in e)


def test_rewake_cooldown_suppresses_quick_reentry():
    cfg.enter_frames, cfg.leave_grace_s, cfg.rewake_cooldown_s = 1, 1, 30
    t = OccupancyTracker()
    ident = _household()
    t.update("cam", "sala", [Observation("1", ident)], now=0.0)               # entered
    t.update("cam", "sala", [], now=5.0)                                       # left (5 > grace)
    e = t.update("cam", "sala", [Observation("2", ident)], now=10.0)          # re-enter in cooldown
    assert not any(x.edge in (EDGE_ENTERED, EDGE_IDENTIFIED) for x in e)
    # ...and stays quiet on the following frame too.
    e2 = t.update("cam", "sala", [Observation("2", ident)], now=10.5)
    assert e2 == []
