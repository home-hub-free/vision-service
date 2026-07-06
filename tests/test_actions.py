"""T2a context rules — zone-kind resolution + the hint table (plan §4.2a, no network)."""
import pytest

from app.actions import CONF_LOW, CONF_MEDIUM, activity_hint, zone_kind
from app.config import cfg


@pytest.fixture(autouse=True)
def _t2a_defaults():
    cfg.hints_enabled = True
    cfg.zone_kinds = ""
    cfg.activity_pass_dwell_s, cfg.activity_settle_dwell_s = 20.0, 60.0
    yield
    cfg.hints_enabled = True
    cfg.zone_kinds = ""


def _p(dwell_s=None, moving=False, posture=None, cls="household"):
    p = {"id": None, "name": None, "class": cls, "confidence": 0.9}
    if dwell_s is not None:
        p["dwell_s"] = dwell_s
        p["moving"] = moving
    if posture:
        p["posture"] = posture
    return p


# ── zone-kind resolution ──────────────────────────────────────────────────────

def test_zone_kind_synonyms_es_en_accent_insensitive():
    assert zone_kind("cocina") == "kitchen"
    assert zone_kind("Recámara") == "bedroom"
    assert zone_kind("sala-tv") == "living"       # substring match
    assert zone_kind("oficina") == "office"
    assert zone_kind("entrada") == "entrance"
    assert zone_kind("baño") == "bathroom"
    assert zone_kind("garage") is None            # unknown kind
    assert zone_kind("") is None


def test_zone_kind_env_override_wins():
    cfg.zone_kinds = "cueva=office, Sala-TV = living"
    assert zone_kind("cueva") == "office"
    assert zone_kind("sala-tv") == "living"
    cfg.zone_kinds = "cueva=spaceship"            # bogus kind → table fallback
    assert zone_kind("cueva") is None


# ── the rule table ────────────────────────────────────────────────────────────

def test_kitchen_standing_settled_morning_is_the_founding_case():
    assert activity_hint("cocina", [_p(dwell_s=90, posture="standing")], hour=7) == \
        ("making breakfast or coffee", CONF_MEDIUM)
    # bent-at-counter reads the same; off-morning becomes generic food prep
    assert activity_hint("cocina", [_p(dwell_s=90, posture="bent")], hour=14) == \
        ("preparing food", CONF_MEDIUM)
    # T1 dark (no posture): morning zone+dwell still hints, but low
    assert activity_hint("cocina", [_p(dwell_s=90)], hour=7) == \
        ("making breakfast or coffee", CONF_LOW)
    assert activity_hint("cocina", [_p(dwell_s=90)], hour=14) is None


def test_table_sitting_meals_vs_working_confuser():
    two = [_p(dwell_s=120, posture="sitting"), _p(dwell_s=90, posture="sitting")]
    assert activity_hint("comedor", two, hour=14) == ("eating together", CONF_MEDIUM)
    solo = [_p(dwell_s=120, posture="sitting")]
    assert activity_hint("cocina", solo, hour=13) == ("having a meal", CONF_LOW)
    # solo at the table OUTSIDE meal hours = likely working → stay silent (S3 confuser)
    assert activity_hint("comedor", solo, hour=10) is None


def test_living_room_relaxing_and_evening_tv_prior():
    sit = [_p(dwell_s=200, posture="sitting")]
    assert activity_hint("sala", sit, hour=20) == ("relaxing, maybe watching TV", CONF_MEDIUM)
    assert activity_hint("sala", sit, hour=10) == ("relaxing", CONF_LOW)
    assert activity_hint("sala", [_p(dwell_s=200, posture="lying")], hour=15) == \
        ("resting on the couch", CONF_MEDIUM)
    assert activity_hint("sala", [_p(dwell_s=200)], hour=20) == ("relaxing", CONF_LOW)


def test_office_and_bedroom():
    assert activity_hint("oficina", [_p(dwell_s=300, posture="sitting")], hour=11) == \
        ("working at the desk", CONF_MEDIUM)
    assert activity_hint("oficina", [_p(dwell_s=300, posture="standing")], hour=11) is None
    lying = [_p(dwell_s=300, posture="lying")]
    assert activity_hint("recamara", lying, hour=23) == ("sleeping", CONF_MEDIUM)
    assert activity_hint("recamara", lying, hour=3) == ("sleeping", CONF_MEDIUM)
    assert activity_hint("recamara", lying, hour=15) == ("resting", CONF_MEDIUM)


def test_entrance_receiving_fires_on_brief_dwell_with_mixed_classes():
    pair = [_p(dwell_s=8, cls="household"), _p(dwell_s=5, cls="unknown")]
    assert activity_hint("entrada", pair, hour=18) == \
        ("receiving someone at the door", CONF_MEDIUM)
    # two household members at the entrance = not a reception
    both = [_p(dwell_s=8, cls="household"), _p(dwell_s=5, cls="household")]
    assert activity_hint("entrada", both, hour=18) is None


def test_sustained_motion_hints_tidying_low():
    assert activity_hint("sala", [_p(dwell_s=150, moving=True)], hour=11) == \
        ("moving around the room, maybe tidying up", CONF_LOW)
    # brief motion = just passing → silent
    assert activity_hint("sala", [_p(dwell_s=30, moving=True)], hour=11) is None


def test_guards_bathroom_flag_unsettled_and_no_dwell():
    settled = [_p(dwell_s=200, posture="standing")]
    assert activity_hint("baño", settled, hour=7) is None          # privacy, always
    assert activity_hint("cocina", [_p(dwell_s=30, posture="standing")], hour=7) is None  # not settled
    assert activity_hint("cocina", [_p()], hour=7) is None         # null build: no dwell
    assert activity_hint("cocina", [], hour=7) is None
    cfg.hints_enabled = False                                      # §8 kill switch
    assert activity_hint("cocina", settled, hour=7) is None


def test_anchor_is_max_dwell_person():
    # the settled person drives the hint even when someone else is passing through
    people = [_p(dwell_s=5, moving=True), _p(dwell_s=120, posture="standing")]
    assert activity_hint("cocina", people, hour=8) == \
        ("making breakfast or coffee", CONF_MEDIUM)
