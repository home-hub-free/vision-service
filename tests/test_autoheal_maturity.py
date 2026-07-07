"""Autoheal maturity — silent folds must be earned; never promote a single frame.

2026-07-07: ~110 mostly single-sighting clusters were auto-promoted into members in
24h — one junk frame scoring 0.62 against a noise-mean centroid became "david" for
good. A cluster is now eligible for SILENT healing only when it has enough
sightings, spread over enough wall-clock time, and its recorded captures agree with
its own centroid. The human review flow stays ungated — a deliberate "yes, that's
me" beats maturity.
"""
import os
import tempfile

from app.config import cfg
from app.gallery import Gallery


def _g(captures: bool = False) -> Gallery:
    cfg.captures_enabled = captures
    return Gallery(os.path.join(tempfile.mkdtemp(), "gallery.db"))


def _vec(seed: float, dim: int = 16):
    v = [0.01] * dim
    v[int(seed) % dim] = 1.0
    return v


def _mix(a_seed: float, b_seed: float, w: float, dim: int = 16):
    va, vb = _vec(a_seed, dim), _vec(b_seed, dim)
    k = (1 - w * w) ** 0.5
    return [w * x + k * y for x, y in zip(va, vb)]


def _tiers(heal=0.9):
    cfg.face_match_threshold = 0.99
    cfg.face_match_margin = 0.05
    cfg.guest_cluster_threshold = 0.5
    cfg.face_autoheal_threshold = heal
    cfg.face_autoheal_margin = 0.05
    cfg.face_suggest_threshold = 0.4
    cfg.face_autoheal_min_sightings = 3
    cfg.face_autoheal_min_span_s = 600.0
    cfg.face_autoheal_min_coherence = 0.45


def _age_cluster(g: Gallery, gid: str, span_s: float = 3600.0):
    """Backdate first_seen so the cluster's sightings span wall-clock time."""
    conn = g._db()
    try:
        conn.execute(
            "UPDATE guests SET first_seen=datetime('now', ?) WHERE guest_id=?",
            (f"-{int(span_s)} seconds", gid))
        conn.commit()
    finally:
        conn.close()


def test_single_frame_never_heals_no_matter_how_well_it_scores():
    _tiers()
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    ident = g.resolve(_mix(1.0, 2.0, 0.95))   # heal-grade score, but ONE sighting
    assert ident.cls == "guest" and ident.id == "guest:1"
    assert g.review_queue()["healed"] == []   # read-time heal refuses too


def test_sightings_without_wallclock_span_do_not_heal():
    """A 20-frame burst is still one look: sightings alone don't mature a cluster."""
    _tiers()
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    for _ in range(4):                        # 4 sightings, all within the same second
        g.resolve(_mix(1.0, 2.0, 0.95))
    assert g.review_queue()["healed"] == []
    _age_cluster(g, "guest:1")                # now the same evidence spans an hour
    out = g.review_queue()
    assert [h["guest_id"] for h in out["healed"]] == ["guest:1"]
    assert out["healed"][0]["id"] == "u1"


def test_mature_cluster_heals_in_the_live_path_too():
    _tiers()
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    for _ in range(3):
        assert g.resolve(_mix(1.0, 2.0, 0.95)).cls == "guest"
    _age_cluster(g, "guest:1")
    ident = g.resolve(_mix(1.0, 2.0, 0.95))   # 4th sighting, now mature → heals live
    assert ident.cls == "household" and ident.id == "u1"


def test_incoherent_cluster_goes_to_review_not_autoheal():
    """A grab-bag cluster (captures disagree with the centroid — different faces
    that happened to chain-cluster) must not silently become somebody."""
    _tiers()
    cfg.guest_cluster_threshold = 0.3         # let dissimilar frames chain into one cluster
    g = _g(captures=True)
    g.enroll("u1", "David", _vec(1.0))
    # Three quite different looks all land in guest:1 (loose cluster threshold),
    # dragging its centroid around — captures record each raw embedding.
    g.resolve(_mix(1.0, 2.0, 0.95), thumb=b"C1")
    g.resolve(_mix(1.0, 3.0, 0.60), thumb=b"C2")
    g.resolve(_mix(1.0, 4.0, 0.60), thumb=b"C3")
    g.resolve(_mix(1.0, 5.0, 0.60), thumb=b"C4")
    _age_cluster(g, "guest:1")
    conn = g._db()
    try:
        n = conn.execute("SELECT COUNT(*) FROM captures WHERE cluster_id='guest:1'").fetchone()[0]
    finally:
        conn.close()
    assert n >= 3                             # coherence check has evidence to judge
    assert g.review_queue()["healed"] == []   # incoherent → stays in the human queue
    assert any(c["guest_id"] == "guest:1" for c in g.review_queue()["queue"])


def test_human_promotion_is_never_maturity_gated():
    """The tinder-flow "yes, that's me" is a deliberate answer — it works on a
    single-sighting cluster (promotion is routing-only; nothing folds)."""
    _tiers()
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    g.resolve(_vec(7.0))                      # brand-new single-sighting cluster
    assert g.promote_guest("guest:1", "u1", "David")
    ident = g.resolve(_vec(7.0))
    assert ident.cls == "household" and ident.id == "u1"
