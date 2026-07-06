"""Occupancy debounce / salience edges — the §8 push-vs-pull behaviour, no models."""
from app.config import cfg
from app.occupancy import (EDGE_ENTERED, EDGE_GUEST_ARRIVED, EDGE_IDENTIFIED,
                           EDGE_LEFT, EDGE_POSTURE_ALERT, EDGE_ROOM_EMPTY,
                           Identity, Observation, OccupancyTracker)


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


# ── T0: dwell + speed (VISION_CONTEXT_TIERS_PLAN §2) ──────────────────────────

def _obs(tid: str, cx: float, w: int = 1000, box: int = 100, **kw) -> Observation:
    """An Observation whose bbox center-x sits at `cx` (pixels) in a `w`-wide frame."""
    half = box // 2
    return Observation(tid, bbox=(int(cx - half), 200, int(cx + half), 200 + box),
                       frame_w=w, **kw)


def test_dwell_and_moving_ride_the_snapshot():
    cfg.enter_frames = 1
    cfg.activity_speed_fws = 0.25
    t = OccupancyTracker()
    # Fast walker: 300 px/frame in a 1000 px frame at 5 fps = 1.5 fw/s.
    t.update("cam", "sala", [_obs("1", 100)], now=0.0)
    t.update("cam", "sala", [_obs("1", 400)], now=0.2)
    t.update("cam", "sala", [_obs("1", 700)], now=0.4)
    p = t.snapshot("sala", now=0.4)["sala"][0]
    assert p["moving"] is True
    assert p["dwell_s"] == 0.4

    # The same person stops: speed EMA decays below the bar.
    now = 0.4
    for _ in range(12):
        now += 0.2
        t.update("cam", "sala", [_obs("1", 700)], now=now)
    p = t.snapshot("sala", now=now)["sala"][0]
    assert p["moving"] is False
    assert p["dwell_s"] == round(now, 1)


def test_no_bbox_means_no_motion_fields_break():
    cfg.enter_frames = 1
    t = OccupancyTracker()
    t.update("cam", "sala", [Observation("1")], now=1.0)  # null build: no bbox
    p = t.snapshot("sala", now=5.0)["sala"][0]
    assert p["dwell_s"] == 4.0
    assert p["moving"] is False


# ── T1: posture + fall-shaped alert (§3) ──────────────────────────────────────

def test_posture_persists_between_pose_frames_and_shows_in_snapshot():
    cfg.enter_frames = 1
    t = OccupancyTracker()
    t.update("cam", "cocina", [Observation("1", posture="standing")], now=0.0)
    t.update("cam", "cocina", [Observation("1")], now=0.2)  # pose skipped this frame
    p = t.snapshot("cocina", now=0.2)["cocina"][0]
    assert p["posture"] == "standing"


def test_posture_alert_fires_once_outside_lying_ok_zones():
    cfg.enter_frames = 1
    cfg.lying_alert_dwell_s = 60.0
    cfg.lying_ok_zones = "bedroom,sala"
    cfg.posture_stable_s = 10.0
    t = OccupancyTracker()
    t.update("cam", "cocina", [Observation("1", posture="lying")], now=0.0)
    e = t.update("cam", "cocina", [Observation("1", posture="lying")], now=30.0)
    assert not any(x.edge == EDGE_POSTURE_ALERT for x in e), "dwell bar not reached yet"
    e = t.update("cam", "cocina", [Observation("1", posture="lying")], now=61.0)
    assert any(x.edge == EDGE_POSTURE_ALERT for x in e)
    e = t.update("cam", "cocina", [Observation("1", posture="lying")], now=62.0)
    assert e == [], "one alert per lying episode"
    # Standing back up (held past the debounce) re-arms; lying long again re-alerts.
    t.update("cam", "cocina", [Observation("1", posture="standing")], now=63.0)
    t.update("cam", "cocina", [Observation("1", posture="standing")], now=74.0)  # commits
    t.update("cam", "cocina", [Observation("1", posture="lying")], now=75.0)
    e = t.update("cam", "cocina", [Observation("1", posture="lying")], now=86.0)  # commits
    assert any(x.edge == EDGE_POSTURE_ALERT for x in e), "dwell already past the bar"


def test_posture_debounce_ignores_frame_exit_flicker():
    """A brief contradictory read (partial bbox at frame-exit says 'lying') must NOT
    replace the committed posture; a sustained change past posture_stable_s must."""
    cfg.enter_frames = 1
    cfg.posture_stable_s = 10.0
    t = OccupancyTracker()
    t.update("cam", "cocina", [Observation("1", posture="standing")], now=0.0)
    # flicker: lying for 3 s, then standing again — committed posture never moves
    t.update("cam", "cocina", [Observation("1", posture="lying")], now=60.0)
    t.update("cam", "cocina", [Observation("1", posture="lying")], now=63.0)
    assert t.snapshot("cocina", now=63.0)["cocina"][0]["posture"] == "standing"
    t.update("cam", "cocina", [Observation("1", posture="standing")], now=64.0)
    # the aborted candidate must not leave residue: a NEW lying spell starts over
    t.update("cam", "cocina", [Observation("1", posture="lying")], now=100.0)
    t.update("cam", "cocina", [Observation("1", posture="lying")], now=109.0)
    assert t.snapshot("cocina", now=109.0)["cocina"][0]["posture"] == "standing"
    # …and commits once it holds past the bar
    t.update("cam", "cocina", [Observation("1", posture="lying")], now=110.5)
    assert t.snapshot("cocina", now=110.5)["cocina"][0]["posture"] == "lying"


def test_no_posture_alert_in_lying_ok_zone():
    cfg.enter_frames = 1
    cfg.lying_alert_dwell_s = 10.0
    cfg.lying_ok_zones = "bedroom,sala"
    t = OccupancyTracker()
    t.update("cam", "sala", [Observation("1", posture="lying")], now=0.0)
    e = t.update("cam", "sala", [Observation("1", posture="lying")], now=100.0)
    assert not any(x.edge == EDGE_POSTURE_ALERT for x in e)
