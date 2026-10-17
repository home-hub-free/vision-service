"""/people/review + /guests/{id}/reject — the "Is this you?" queue routes.

A temp Gallery is swapped into the routes module (no models, no network); the
hub-session gate (require_user) is stubbed per-test to authed/unauthed.
"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.config import cfg
from app.gallery import Gallery
from app.main import app
from app.routes import guests as guests_routes

client = TestClient(app)


def _vec(seed: float, dim: int = 16):
    v = [0.01] * dim
    v[int(seed) % dim] = 1.0
    return v


def _mix(a_seed: float, b_seed: float, w: float, dim: int = 16):
    va, vb = _vec(a_seed, dim), _vec(b_seed, dim)
    k = (1 - w * w) ** 0.5
    return [w * x + k * y for x, y in zip(va, vb)]


@pytest.fixture()
def gallery(monkeypatch):
    cfg.face_match_threshold = 0.99
    cfg.guest_cluster_threshold = 0.999999
    cfg.face_autoheal_threshold = 0.9
    cfg.face_autoheal_margin = 0.05
    cfg.face_suggest_threshold = 0.4
    # Route tests exercise the tier plumbing; autoheal maturity has its own tests.
    cfg.face_autoheal_min_sightings = 1
    cfg.face_autoheal_min_span_s = 0.0
    cfg.face_autoheal_min_coherence = 0.0
    g = Gallery(os.path.join(tempfile.mkdtemp(), "gallery.db"))
    monkeypatch.setattr(guests_routes, "gallery", g)
    monkeypatch.setattr(guests_routes, "require_user",
                        lambda authorization: {"id": "u1", "displayName": "David"})
    yield g
    cfg.face_autoheal_threshold = 0.5
    cfg.face_autoheal_margin = 0.08
    cfg.face_suggest_threshold = 0.2


def test_review_returns_tiered_queue_with_thumbs(gallery):
    gallery.enroll("u1", "David", _vec(1.0))
    gallery.resolve(_mix(1.0, 2.0, 0.6),                  # suggest tier, face-located
                    thumb=b"CROP", thumb_box=[0.3, 0.2, 0.4, 0.5])
    gallery.resolve(_vec(4.0))                            # unknown tier, no thumb
    res = client.get("/people/review")
    assert res.status_code == 200
    body = res.json()
    assert body["healed"] == []
    by_id = {c["guest_id"]: c for c in body["queue"]}
    assert by_id["guest:1"]["tier"] == "suggest"
    assert by_id["guest:1"]["suggested"]["kind"] == "member"
    assert by_id["guest:1"]["suggested"]["id"] == "u1"
    assert by_id["guest:1"]["thumb"] == "faces/thumb/guest:1"
    assert by_id["guest:1"]["face_box"] == [0.3, 0.2, 0.4, 0.5]
    assert by_id["guest:1"]["no_face"] is False
    assert by_id["guest:2"]["tier"] == "unknown" and by_id["guest:2"]["thumb"] is None
    assert by_id["guest:2"]["face_box"] is None and by_id["guest:2"]["no_face"] is False


def test_review_heals_top_tier_on_read(gallery):
    gallery.enroll("u1", "David", _vec(1.0))
    cfg.face_autoheal_threshold = 0.999                   # too high at resolve time
    gallery.resolve(_mix(1.0, 2.0, 0.95))
    cfg.face_autoheal_threshold = 0.9                     # bar sharpens → heals on read
    body = client.get("/people/review").json()
    assert body["queue"] == []
    assert body["healed"][0]["guest_id"] == "guest:1"
    assert body["healed"][0]["kind"] == "member" and body["healed"][0]["id"] == "u1"


def test_reject_records_not_me_and_requires_auth(gallery, monkeypatch):
    gallery.enroll("u1", "David", _vec(1.0))
    gallery.resolve(_mix(1.0, 2.0, 0.6))
    res = client.post("/guests/guest:1/reject", json={"user_id": "u1"})
    assert res.status_code == 200 and res.json()["rejected_user_id"] == "u1"
    body = client.get("/people/review").json()
    assert body["queue"][0]["tier"] == "unknown"
    assert body["queue"][0]["rejected_user_ids"] == ["u1"]
    # Unknown cluster → 404; unauthed → 401 (hub-session gated like promote/name).
    assert client.post("/guests/guest:404/reject", json={"user_id": "u1"}).status_code == 404

    from fastapi import HTTPException

    def _deny(authorization):
        raise HTTPException(status_code=401, detail="missing bearer token")

    monkeypatch.setattr(guests_routes, "require_user", _deny)
    assert client.post("/guests/guest:1/reject", json={"user_id": "u1"}).status_code == 401


def test_merge_folds_cluster_into_named_guest(gallery):
    gallery.resolve(_vec(2.0))
    gallery.name_guest("guest:1", "Abuela")
    gallery.resolve(_vec(3.0))
    res = client.post("/guests/guest:2/merge", json={"into": "guest:1"})
    assert res.status_code == 200 and res.json()["merged_into"] == "guest:1"
    assert client.post("/guests/guest:2/merge", json={"into": "guest:1"}).status_code == 404


# ── review-queue triage ("Clean up") ─────────────────────────────────────────
def test_triage_lists_unidentifiable_cards_and_applies_only_when_complete(gallery, monkeypatch):
    cfg.face_triage_keep_sightings = 3
    gallery.resolve(_vec(2.0), thumb=b"SHARP", thumb_box=[0.3, 0.2, 0.4, 0.5])   # guest:1 ok
    gallery.resolve(_vec(3.0), thumb=b"BLUR", thumb_box=[0.3, 0.2, 0.4, 0.5])    # guest:2 blurry
    gallery.resolve(_vec(4.0), thumb=b"NOFACE", thumb_box=[0.3, 0.2, 0.4, 0.5])  # guest:3 no face
    gallery.resolve(_vec(5.0))                                                   # guest:4 no photo
    for _ in range(3):                                                           # guest:5 recurring
        gallery.resolve(_vec(6.0), thumb=b"NOFACE", thumb_box=[0.3, 0.2, 0.4, 0.5])
    verdict = {b"SHARP": "ok", b"BLUR": "blurry", b"NOFACE": "no_face"}
    calls = []
    gallery.thumb_assessor = lambda jpeg: calls.append(jpeg) or verdict[jpeg]

    body = client.post("/people/review/triage", json={}).json()
    assert body["complete"] is True and body["applied"] == 0
    assert body["total"] == 5 and body["kept"] == 1 and body["kept_recurring"] == 1
    assert {d["guest_id"]: d["reason"] for d in body["discard"]} == {
        "guest:2": "blurry", "guest:3": "no_face", "guest:4": "no_photo"}
    assert body["reasons"] == {"blurry": 1, "no_face": 1, "no_photo": 1}
    assert sorted(calls) == [b"BLUR", b"NOFACE", b"SHARP"]   # recurring never judged
    # Dry run never deletes; verdicts are cached so the second pass judges nothing.
    assert len(gallery.review_queue()["queue"]) == 5
    calls.clear()
    assert client.post("/people/review/triage", json={}).json()["complete"] is True
    assert calls == []

    # Apply discards exactly the listed cards; the ok + recurring ones survive.
    body = client.post("/people/review/triage", json={"apply": True}).json()
    assert body["applied"] == 3
    left = {c["guest_id"] for c in gallery.review_queue()["queue"]}
    assert left == {"guest:1", "guest:5"}
    cfg.face_triage_keep_sightings = 3


def test_triage_respects_budget_and_never_applies_a_partial_scan(gallery, monkeypatch):
    import types
    from app import gallery as gallery_module
    for i in range(3):
        gallery.resolve(_vec(float(i + 2)), thumb=b"X%d" % i, thumb_box=[0.3, 0.2, 0.4, 0.5])
    ticks = iter([0.0, 0.0, 100.0])                         # budget blows after the 1st judge
    gallery.thumb_assessor = lambda jpeg: "no_face"
    # Patch the module's clock (not time.monotonic globally — the test client's own
    # event loop reads that clock too and would eat the ticks); scoped so the
    # fixture's own patches (auth stub, temp gallery) survive the second call.
    with pytest.MonkeyPatch.context() as clock:
        clock.setattr(gallery_module, "time",
                      types.SimpleNamespace(monotonic=lambda: next(ticks, 100.0)))
        body = client.post("/people/review/triage", json={"apply": True, "budget_s": 5}).json()
    assert body["complete"] is False and body["pending"] == 2 and body["scanned"] == 1
    assert body["applied"] == 0 and len(body["discard"]) == 1
    assert len(gallery.review_queue()["queue"]) == 3           # nothing deleted on a partial pass
    body = client.post("/people/review/triage", json={"apply": True}).json()
    assert body["complete"] is True and body["applied"] == 3


def test_triage_without_engine_judges_nothing_and_requires_auth(gallery, monkeypatch):
    gallery.resolve(_vec(2.0), thumb=b"CROP", thumb_box=[0.3, 0.2, 0.4, 0.5])
    gallery.thumb_assessor = None                               # null build
    body = client.post("/people/review/triage", json={"apply": True}).json()
    assert body["complete"] is False and body["unjudged"] == 1 and body["applied"] == 0
    gallery.thumb_assessor = lambda jpeg: None                  # engine down right now
    assert client.post("/people/review/triage", json={}).json()["unjudged"] == 1
    from fastapi import HTTPException

    def unauthed(authorization):
        raise HTTPException(status_code=401, detail="missing bearer token")
    monkeypatch.setattr(guests_routes, "require_user", unauthed)
    assert client.post("/people/review/triage", json={}).status_code == 401


def test_triage_verdict_resets_when_a_bad_thumb_is_replaced(gallery):
    cfg.guest_cluster_threshold = 0.5                            # let the 2nd sighting cluster
    gallery.resolve(_vec(2.0), thumb=b"BAD", thumb_box=[])     # stored crop: no face
    gallery.thumb_assessor = lambda jpeg: "no_face" if jpeg == b"BAD" else "ok"
    assert gallery.triage_review()["discard"][0]["reason"] == "no_face"
    gallery.resolve(_vec(2.0), thumb=b"GOOD", thumb_box=[0.3, 0.2, 0.4, 0.5])  # self-heal
    body = gallery.triage_review()
    assert body["discard"] == [] and body["kept"] == 1
    cfg.guest_cluster_threshold = 0.999999
