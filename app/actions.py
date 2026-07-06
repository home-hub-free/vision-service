"""T2a context rules — zone-kind × dwell × posture × hour → an activity HINT.

The cheapest rung of the action-recognition ladder (VISION_CONTEXT_TIERS_PLAN §4,
candidate 2a): pure code over the signals T0/T1 already produce — no model, ~0 ms,
evaluated only at digest-build time (never per frame). The output is a HINT, not a
fact: a short label ("making breakfast or coffee") plus a confidence tier the
gateway renders hedged ("likely …" for medium, "possibly …" for low). The AGENT
does the final intent interpretation — these rules only move the prior; they never
assert what they can't see. Rules are anchored on the same max-dwell person
`zone_activity` uses, so the hint and the T0 activity word describe the same
subject. Bathroom-kind zones never emit a hint, whatever the signals say.

Rule sources are the §4 scenario rubric (S1 breakfast, S3 eating, S4 relaxing,
S5 guest received, S6 tidying) — the soak scores this table against ~10 real clips
per scenario, and a rule that keeps losing to its confuser graduates to the T2b
pose-sequence classifier instead of growing epicycles here.
"""
from __future__ import annotations

import unicodedata
from datetime import datetime
from typing import List, Optional, Tuple

from .config import cfg

# (label, confidence) — confidence is a tier, not a probability: rules top out at
# "medium" (a static table can't earn more); "low" marks the guessier branches.
Hint = Tuple[str, str]
CONF_LOW = "low"
CONF_MEDIUM = "medium"

# Zone-name synonym table (es/en, accent-insensitive substring match) → zone kind.
# `VISION_ZONE_KINDS` overrides win for names the table can't guess.
_KIND_SYNONYMS = {
    "kitchen": ("cocina", "kitchen"),
    "dining": ("comedor", "dining"),
    "living": ("sala", "living", "estancia", "lounge"),
    "office": ("oficina", "office", "estudio", "despacho", "study"),
    "bedroom": ("recamara", "cuarto", "bedroom", "dormitorio", "habitacion"),
    "entrance": ("entrada", "entrance", "hall", "pasillo", "puerta", "porch"),
    "bathroom": ("bano", "bathroom", "toilet", "aseo"),
}
_KINDS = frozenset(_KIND_SYNONYMS)

# Hour windows (producer-local time — the box and the house share a clock).
_MORNING = range(5, 11)
_EVENING = range(17, 24)
_MEAL_HOURS = frozenset(range(7, 10)) | frozenset(range(12, 16)) | frozenset(range(19, 22))


def _normalize(name: str) -> str:
    """Lowercase + strip accents so "Recámara" matches "recamara"."""
    flat = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return flat.strip().lower()


def zone_kind(zone: str) -> Optional[str]:
    """Resolve a user-assigned zone name to a rule kind. Overrides from
    `VISION_ZONE_KINDS` ("cueva=office,sala-tv=living") win; otherwise the synonym
    table matches by accent-insensitive substring ("sala-tv" contains "sala").
    None = unknown kind → only zone-agnostic rules apply."""
    if not zone:
        return None
    z = _normalize(zone)
    for pair in cfg.zone_kinds.split(","):
        if "=" not in pair:
            continue
        name, kind = pair.split("=", 1)
        if _normalize(name) == z and _normalize(kind) in _KINDS:
            return _normalize(kind)
    for kind, synonyms in _KIND_SYNONYMS.items():
        if any(s in z for s in synonyms):
            return kind
    return None


def _anchor(people: List[dict]) -> Optional[dict]:
    """The max-dwell person — same anchor `hub_push.zone_activity` classifies."""
    best = None
    for p in people:
        if p.get("dwell_s") is None:
            continue
        if best is None or p["dwell_s"] > best["dwell_s"]:
            best = p
    return best


def activity_hint(zone: str, snapshot_people: List[dict],
                  hour: Optional[int] = None) -> Optional[Hint]:
    """The T2a rule table. `hour` is injectable for tests; defaults to local time.
    Returns None whenever no rule earns a hint — silence beats a mushy guess (the
    agent still gets the structured T0/T1 facts either way)."""
    if not cfg.hints_enabled:
        return None
    kind = zone_kind(zone)
    if kind == "bathroom":  # privacy guard: never characterize, whatever the signals
        return None
    anchor = _anchor(snapshot_people)
    if anchor is None:
        return None
    if hour is None:
        hour = datetime.now().hour

    dwell = anchor["dwell_s"]
    moving = bool(anchor.get("moving"))
    posture = anchor.get("posture")
    count = len(snapshot_people)
    classes = {p.get("class") or "unknown" for p in snapshot_people}

    # S5 — a household member with a non-household person at the entrance. Fires on
    # ANY dwell (greetings are brief), before the settled gate below.
    if kind == "entrance" and count >= 2 and "household" in classes and (
            {"guest", "unknown"} & classes):
        return ("receiving someone at the door", CONF_MEDIUM)

    # S6 — sustained motion around a room (settled never fires while moving).
    # Confuser is pacing on a call, hence low.
    if moving and dwell >= 2 * cfg.activity_settle_dwell_s:
        return ("moving around the room, maybe tidying up", CONF_LOW)

    # Everything below needs a settled anchor (the T0 bar: past settle dwell, still).
    if moving or dwell < cfg.activity_settle_dwell_s:
        return None

    if kind == "kitchen":
        if posture in ("standing", "bent"):
            if hour in _MORNING:  # S1, the founding case
                return ("making breakfast or coffee", CONF_MEDIUM)
            return ("preparing food", CONF_MEDIUM)
        if posture == "sitting":
            return _table_hint(count, hour)
        if posture is None and hour in _MORNING:  # T1 dark: zone+dwell+hour only
            return ("making breakfast or coffee", CONF_LOW)
        return None
    if kind == "dining":
        return _table_hint(count, hour) if posture == "sitting" else None
    if kind == "living":
        if posture == "lying":
            return ("resting on the couch", CONF_MEDIUM)
        if posture == "sitting":  # S4; evening raises the TV prior
            if hour in _EVENING:
                return ("relaxing, maybe watching TV", CONF_MEDIUM)
            return ("relaxing", CONF_LOW)
        if posture is None and hour in _EVENING:
            return ("relaxing", CONF_LOW)
        return None
    if kind == "office":
        return ("working at the desk", CONF_MEDIUM) if posture == "sitting" else None
    if kind == "bedroom":
        if posture == "lying":
            night = hour >= 21 or hour < 8
            return ("sleeping", CONF_MEDIUM) if night else ("resting", CONF_MEDIUM)
        return None
    return None


def _table_hint(count: int, hour: int) -> Optional[Hint]:
    """Sitting at a kitchen/dining table (S3). Confuser is working at the table,
    so solo outside meal hours stays silent."""
    if count >= 2:
        return ("eating together", CONF_MEDIUM)
    if hour in _MEAL_HOURS:
        return ("having a meal", CONF_LOW)
    return None
