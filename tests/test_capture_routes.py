"""/faces/{owner}/captures + delete + rebuild — the dashboard's "re-do the soup" API.

Temp Gallery swapped into the enroll routes; require_user stubbed. Flow under test:
list an identity's archived photos → delete the polluted ones → rebuild replaces the
member centroid from exactly what remains.
"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.config import cfg
from app.gallery import Gallery
from app.main import app
from app.routes import enroll as enroll_routes

client = TestClient(app)

JPEG = b"\xff\xd8CROP\xff\xd9"


def _vec(seed: float, dim: int = 16):
    v = [0.01] * dim
    v[int(seed) % dim] = 1.0
    return v


@pytest.fixture()
def gallery(monkeypatch):
    cfg.face_match_threshold = 0.5
    cfg.face_match_margin = 0.05
    cfg.guest_cluster_threshold = 0.999999
    cfg.captures_enabled = True
    g = Gallery(os.path.join(tempfile.mkdtemp(), "gallery.db"))
    monkeypatch.setattr(enroll_routes, "gallery", g)
    monkeypatch.setattr(enroll_routes, "require_user",
                        lambda authorization: {"id": "u1", "displayName": "David"})
    return g


def test_list_serve_delete_and_rebuild_flow(gallery):
    david, ana = _vec(1.0), _vec(2.0)
    gallery.enroll("u1", "David", david, thumb=JPEG)   # capture: enroll
    gallery.resolve(david, thumb=JPEG)                 # capture: match (reinforces too)
    # Pollution: Ana's face archived under David (simulates the pre-fix reinforce).
    gallery._capture("match", ana, JPEG, resolved_id="u1", score=0.4)

    r = client.get("/faces/u1/captures")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3 and len(body["captures"]) == 3
    newest = body["captures"][0]                       # newest first = the pollution
    assert newest["image"].startswith("faces/captures/")

    img = client.get("/" + newest["image"])
    assert img.status_code == 200 and img.content == JPEG

    r = client.delete(f"/faces/captures/{newest['id']}")
    assert r.status_code == 200
    assert client.get("/" + newest["image"]).status_code == 404   # file+row gone
    assert client.get("/faces/u1/captures").json()["total"] == 2

    # Rebuild from the two remaining (clean, David-only) photos.
    r = client.post("/faces/u1/rebuild", json={"name": "David"})
    assert r.status_code == 200 and r.json()["samples"] == 2
    prof = gallery.profiles()[0]
    assert prof["samples"] == 2 and prof["name"] == "David"
    assert gallery.resolve(david).id == "u1"           # David still matches
    assert gallery.resolve(ana).cls == "guest"         # the salt is out of the pot


def test_rebuild_with_no_captures_is_409_and_deletes_require_auth(gallery, monkeypatch):
    r = client.post("/faces/ghost/rebuild", json={})
    assert r.status_code == 409

    def deny(authorization):
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="no")
    monkeypatch.setattr(enroll_routes, "require_user", deny)
    assert client.delete("/faces/captures/1").status_code == 401
    assert client.post("/faces/u1/rebuild", json={}).status_code == 401
