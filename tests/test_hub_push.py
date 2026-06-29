"""Hub room-digest payload builder — the §3.1 producer push contract (no network)."""
from app.hub_push import room_digest_payload
from app.occupancy import Identity, Observation, OccupancyTracker
from app.config import cfg


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
