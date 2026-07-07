"""Gallery audit + smear alarm — the tripwire both 2026-07 incidents lacked.

run_audit measures member-vs-member confusability (freeze silent folds at the
alarm bar, self-clear when healthy), re-scores member promotions against anchors
(detach incoherent ones), and counts cluster churn. GET /faces/health serves it.
"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.config import cfg
from app.face_audit import run_audit
from app.gallery import Gallery
from app.main import app
from app.routes import enroll as enroll_routes

client = TestClient(app)


def _g() -> Gallery:
    cfg.captures_enabled = False
    return Gallery(os.path.join(tempfile.mkdtemp(), "gallery.db"))


def _vec(seed: float, dim: int = 16):
    v = [0.01] * dim
    v[int(seed) % dim] = 1.0
    return v


def _mix(a_seed: float, b_seed: float, w: float, dim: int = 16):
    va, vb = _vec(a_seed, dim), _vec(b_seed, dim)
    k = (1 - w * w) ** 0.5
    return [w * x + k * y for x, y in zip(va, vb)]


def _defaults():
    cfg.face_smear_alarm_cos = 0.45
    cfg.face_audit_detach_below = 0.30
    cfg.face_churn_warn_24h = 30
    cfg.face_match_threshold = 0.99
    cfg.face_match_margin = 0.05
    cfg.guest_cluster_threshold = 0.9
    cfg.face_autoheal_threshold = 0.9
    cfg.face_autoheal_margin = 0.05
    cfg.face_autoheal_min_sightings = 1
    cfg.face_autoheal_min_span_s = 0.0
    cfg.face_autoheal_min_coherence = 0.0


def test_smear_alarm_freezes_folds_and_self_clears():
    _defaults()
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    g.enroll("u2", "Ana", _mix(1.0, 2.0, 0.8))     # confusably close to David (cos 0.8)
    report = run_audit(g)
    assert report["smeared"] and report["folds_frozen"] and g.folds_frozen
    assert report["worst_pair"]["score"] >= 0.45

    # While frozen: a heal-grade cluster does NOT fold, live or on read.
    probe = _mix(1.0, 3.0, 0.95)
    assert g.resolve(probe).cls == "guest"
    assert g.review_queue()["healed"] == []

    # The household fixes Ana's profile (re-enroll distinct) → next audit clears.
    g.forget("u2")
    g.enroll("u2", "Ana", _vec(5.0))
    report = run_audit(g)
    assert not report["smeared"] and not g.folds_frozen
    assert g.resolve(probe).cls == "household"      # folds work again


def test_promotion_audit_detaches_incoherent_clusters():
    _defaults()
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    g.resolve(_mix(1.0, 2.0, 0.95))                 # guest:1 — David-like
    g.resolve(_vec(7.0))                            # guest:2 — nothing like David
    assert g.promote_guest("guest:1", "u1", "David")
    assert g.promote_guest("guest:2", "u1", "David")  # the wrong human answer
    report = run_audit(g)
    detached = {d["guest_id"] for d in report["promotions_detached"]}
    assert detached == {"guest:2"}
    # detached cluster is blocked from re-healing into David and back in review
    conn = g._db()
    try:
        rej, promoted = conn.execute(
            "SELECT rejected_user_ids, promoted_user_id FROM guests WHERE guest_id='guest:2'"
        ).fetchone()
    finally:
        conn.close()
    assert promoted is None and "u1" in rej


def test_promotion_audit_skips_anchorless_members():
    """A promoted-only member (never enrolled) has no ground truth to audit
    against — their promotions must not be judged by a nonexistent anchor set."""
    _defaults()
    g = _g()
    g.resolve(_vec(7.0))
    assert g.promote_guest("guest:1", "u9", "Sam")
    report = run_audit(g)
    assert report["promotions_checked"] == 0
    assert report["promotions_detached"] == []


def test_churn_counts_fresh_clusters():
    _defaults()
    g = _g()
    for seed in (3.0, 7.0, 11.0):
        g.resolve(_vec(seed))
    assert run_audit(g)["clusters_24h"] == 3


def test_health_route_serves_similarity_and_last_report(monkeypatch):
    _defaults()
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    g.enroll("u2", "Ana", _vec(5.0))
    monkeypatch.setattr(enroll_routes, "gallery", g)
    run_audit(g)
    r = client.get("/faces/health")
    assert r.status_code == 200
    body = r.json()
    assert body["folds_frozen"] is False
    assert body["member_similarity"][0]["score"] < 0.45
    assert body["last_audit"]["clusters_24h"] == 0
    assert body["smear_alarm_cos"] == pytest.approx(0.45)
