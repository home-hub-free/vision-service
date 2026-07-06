"""Hub room-digest payload builder — the §3.1 producer push contract (no network)."""
import pytest

from app.hub_push import _reset_hint_hold, room_digest_payload
from app.occupancy import Identity, Observation, OccupancyTracker
from app.config import cfg


@pytest.fixture(autouse=True)
def _clean_hint_hold():
    _reset_hint_hold()
    yield
    _reset_hint_hold()


def _meta(ident: Identity) -> dict:
    # mirror the per-person dict OccupancyTracker.snapshot() emits (identity.as_meta()).
    return {"track": "cam1:1", "since": 1.0, **ident.as_meta()}


def test_payload_keeps_only_resolved_identity_fields_never_embeddings():
    people = [
        _meta(Identity(id="u1", name="Juan", cls="household", confidence=0.82)),
        _meta(Identity(id=None, name=None, cls="unknown", confidence=0.0)),
    ]
    body = room_digest_payload("sala", people)
    assert body["zone"] == "sala"
    assert body["count"] == 2
    assert body["occupied"] is True
    # Exactly the agent-facing fields — no track/since/embedding leakage.
    assert set(body["people"][0].keys()) == {"id", "name", "class", "confidence"}
    assert body["people"][0] == {"id": "u1", "name": "Juan", "class": "household", "confidence": 0.82}
    # Unknown rides along, counted but unnamed.
    assert body["people"][1] == {"id": None, "name": None, "class": "unknown", "confidence": 0.0}


def test_empty_zone_payload_is_unoccupied():
    body = room_digest_payload("cocina", [])
    assert body == {"zone": "cocina", "count": 0, "occupied": False, "people": []}


def test_builds_from_a_real_tracker_snapshot():
    cfg.enter_frames = 1
    t = OccupancyTracker()
    t.update("cam1", "oficina", [Observation("1", Identity(id="u2", name="Ana", cls="household", confidence=0.9))], now=1.0)
    body = room_digest_payload("oficina", t.snapshot("oficina").get("oficina", []))
    assert body["count"] == 1
    assert body["people"][0]["name"] == "Ana"
    assert body["people"][0]["class"] == "household"
    assert "track" not in body["people"][0]


# ── T0/T1: activity classifier + new digest fields (VISION_CONTEXT_TIERS_PLAN §2/§3) ─

def _p(dwell_s=None, moving=False, posture=None, **ident):
    base = _meta(Identity(id=ident.get("id"), name=ident.get("name"),
                          cls=ident.get("cls", "unknown"),
                          confidence=ident.get("confidence", 0.0)))
    if dwell_s is not None:
        base["dwell_s"] = dwell_s
        base["moving"] = moving
    if posture:
        base["posture"] = posture
    return base


def test_activity_classifier_thresholds():
    cfg.activity_pass_dwell_s, cfg.activity_settle_dwell_s, cfg.activity_speed_fws = 20.0, 60.0, 0.25
    from app.hub_push import zone_activity
    assert zone_activity([_p(dwell_s=5.0)]) == "passing"           # short dwell
    assert zone_activity([_p(dwell_s=90.0, moving=True)]) == "passing"  # fast even if long
    assert zone_activity([_p(dwell_s=30.0)]) == "lingering"
    assert zone_activity([_p(dwell_s=120.0)]) == "settled"
    assert zone_activity([_p()]) is None                            # no dwell data (null build)
    assert zone_activity([]) is None


def test_zone_activity_is_max_dwell_person_and_carries_posture():
    from app.hub_push import zone_activity
    people = [_p(dwell_s=5.0, moving=True), _p(dwell_s=120.0, posture="sitting")]
    assert zone_activity(people) == "settled+sitting"


def test_payload_carries_dwell_moving_posture_and_zone_activity():
    cfg.enter_frames = 1
    t = OccupancyTracker()
    t.update("cam1", "cocina",
             [Observation("1", Identity(id="u1", name="David", cls="household", confidence=0.9),
                          bbox=(100, 100, 200, 300), frame_w=1000, posture="standing")],
             now=0.0)
    t.update("cam1", "cocina",
             [Observation("1", bbox=(100, 100, 200, 300), frame_w=1000)], now=90.0)
    body = room_digest_payload("cocina", t.snapshot("cocina", now=90.0).get("cocina", []))
    person = body["people"][0]
    assert person["dwell_s"] == 90.0
    assert person["moving"] is False
    assert person["posture"] == "standing"
    assert body["activity"] == "settled+standing"
    # Identity fields stay exactly the resolved set + the additive activity fields.
    assert "track" not in person and "since" not in person


def test_payload_without_dwell_data_has_no_activity_field():
    body = room_digest_payload("sala", [_p(name="Ana", cls="household", confidence=0.9, id="u2")])
    assert "activity" not in body
    assert "dwell_s" not in body["people"][0]


def test_satellite_track_is_identity_only_no_activity_no_hint():
    """A face-ID-only camera (Observation.context=False — satellite/ESP32 cams) must
    contribute identity but NO T0/T1/T2a signals: no dwell/moving/posture on the person,
    no zone activity, no activity hint — even in prime hint conditions."""
    cfg.enter_frames = 1
    cfg.hints_enabled, cfg.zone_kinds = True, ""
    t = OccupancyTracker()
    t.update("sat1", "cocina",
             [Observation("1", Identity(id="u1", name="David", cls="household", confidence=0.9),
                          bbox=(100, 100, 200, 300), frame_w=1000, posture="standing",
                          context=False)],
             now=0.0)
    t.update("sat1", "cocina", [Observation("1", context=False)], now=90.0)
    body = room_digest_payload("cocina", t.snapshot("cocina", now=90.0).get("cocina", []), hour=7)
    person = body["people"][0]
    assert person["name"] == "David"          # identity still crosses
    assert "dwell_s" not in person and "moving" not in person and "posture" not in person
    assert "activity" not in body
    assert "activity_hint" not in body


# ── T2a: context-rule activity hint on the digest (plan §4.2a) ────────────────

def test_payload_carries_activity_hint_when_a_rule_fires():
    cfg.hints_enabled, cfg.zone_kinds = True, ""
    people = [_p(dwell_s=120.0, posture="standing", name="David", cls="household",
                 confidence=0.9, id="u1")]
    body = room_digest_payload("cocina", people, hour=7)
    assert body["activity"] == "settled+standing"
    assert body["activity_hint"] == "making breakfast or coffee"
    assert body["activity_hint_conf"] == "medium"
    # silent when no rule earns a hint — the fields are simply absent
    body = room_digest_payload("garage", people, hour=7)
    assert "activity_hint" not in body and "activity_hint_conf" not in body


def test_hint_holds_through_brief_dropouts_while_occupied():
    cfg.hints_enabled, cfg.zone_kinds, cfg.hint_hold_s = True, "", 30.0
    settled = [_p(dwell_s=120.0, posture="standing", cls="household", confidence=0.9)]
    body = room_digest_payload("cocina", settled, hour=7, now=1000.0)
    assert body["activity_hint"] == "making breakfast or coffee"
    # posture flickers (kitchen+lying has no rule) → the hint is HELD, not dropped
    flicker = [_p(dwell_s=125.0, posture="lying", cls="household", confidence=0.9)]
    body = room_digest_payload("cocina", flicker, hour=7, now=1005.0)
    assert body["activity_hint"] == "making breakfast or coffee"
    # past the hold window with the rules still silent → the hint finally clears
    body = room_digest_payload("cocina", flicker, hour=7, now=1036.0)
    assert "activity_hint" not in body


def test_hint_replaced_immediately_and_cleared_on_empty_zone():
    cfg.hints_enabled, cfg.zone_kinds, cfg.hint_hold_s = True, "", 30.0
    standing = [_p(dwell_s=120.0, posture="standing", cls="household", confidence=0.9)]
    body = room_digest_payload("cocina", standing, hour=9, now=1000.0)
    assert body["activity_hint"] == "making breakfast or coffee"
    # a DIFFERENT fired hint wins immediately — no debounce on real rule output
    sitting = [_p(dwell_s=150.0, posture="sitting", cls="household", confidence=0.9),
               _p(dwell_s=140.0, posture="sitting", cls="household", confidence=0.9)]
    body = room_digest_payload("cocina", sitting, hour=9, now=1005.0)
    assert body["activity_hint"] == "eating together"
    # an emptied zone clears the hold instantly (no ghost hint on re-entry)
    body = room_digest_payload("cocina", [], hour=9, now=1010.0)
    assert "activity_hint" not in body
    passing = [_p(dwell_s=3.0, posture="standing", cls="household", confidence=0.9)]
    body = room_digest_payload("cocina", passing, hour=9, now=1012.0)
    assert "activity_hint" not in body
