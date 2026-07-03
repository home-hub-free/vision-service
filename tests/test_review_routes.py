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
