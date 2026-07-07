"""Anchor-set profiles — enrollment is immutable ground truth.

A member's face profile is their individually-stored, quality-gated enroll
embeddings; matching scores against the top-3 nearest anchors. Nothing at runtime
folds into them (see test_gallery/test_identity_swap for the immutability tests) —
these tests pin the anchor mechanics themselves: scoring, the cap, reset-on-enroll
for legacy centroids, and rebuild replacing the set.
"""
import json
import os
import tempfile

from app.config import cfg
from app.gallery import Gallery


def _g() -> Gallery:
    cfg.captures_enabled = False
    return Gallery(os.path.join(tempfile.mkdtemp(), "gallery.db"))


def _vec(seed: float, dim: int = 16):
    v = [0.01] * dim
    v[int(seed) % dim] = 1.0
    return v


def _variant(seed: float, alt: float, w: float = 0.75, dim: int = 16):
    """A correlated 'other look' of the same face: cos(base, variant) ≈ 1/√(1+w²)
    (~0.8 at w=0.75) — realistic same-person anchor agreement."""
    v = _vec(seed, dim)
    v[int(alt) % dim] = w
    return v


def test_anchor_scoring_matches_a_distinct_look():
    """Two enrolled looks (say frontal + glasses): a probe matching the SECOND look
    scores by its two nearest anchors (top-2 mean) — high — where a single running
    mean would drift toward whatever mixture history happened to fold in."""
    cfg.face_match_threshold = 0.5
    cfg.face_match_margin = 0.05
    g = _g()
    look_a = _vec(1.0)
    look_b = _variant(1.0, 2.0)                    # same face, cos(a,b) ≈ 0.8
    g.enroll("u1", "David", look_a)
    g.enroll("u1", "David", look_b)
    g.enroll("u2", "Ana", _vec(5.0))
    uid, _name, score, margin = g._best_household(look_b)
    assert uid == "u1"
    assert score > 0.85                            # (1.0 + ~0.8) / 2
    assert margin > 0.5                            # Ana is nowhere near


def test_one_rogue_anchor_cannot_impersonate():
    """The reason top-2 (not max): if a wrong-person photo slips past the enroll
    gate into David's anchors, a probe of that person matches ONE anchor perfectly
    but the second-best agreement is noise — the score stays out of match range."""
    cfg.face_match_threshold = 0.5
    cfg.face_match_margin = 0.05
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    g.enroll("u1", "David", _variant(1.0, 2.0))
    g.enroll("u1", "David", _vec(9.0))             # the rogue: actually Ana's face
    ana_probe = _vec(9.0)
    score = g._member_score_one("u1", ana_probe)
    assert score < 0.6                             # (1.0 + ~0.03) / 2 — below a sane bar
    uid, _n, s, _m = g._best_household(ana_probe)
    assert s == score


def test_anchor_cap_keeps_newest():
    cfg.face_anchor_cap = 3
    g = _g()
    for seed in (1.0, 2.0, 3.0, 4.0, 5.0):
        g.enroll("u1", "David", _vec(seed))
    conn = g._db()
    try:
        rows = [json.loads(r[0]) for r in
                conn.execute("SELECT embedding FROM anchors WHERE user_id='u1' ORDER BY id")]
    finally:
        conn.close()
    assert len(rows) == 3
    # oldest (seeds 1, 2) pruned: the newest three one-hot positions remain
    hot = {r.index(max(r)) for r in rows}
    assert hot == {3, 4, 5}
    assert {p["user_id"]: p["samples"] for p in g.profiles()}["u1"] == 3
    cfg.face_anchor_cap = 60


def test_first_gated_enroll_resets_a_legacy_polluted_centroid():
    """A legacy member (running-mean centroid, no anchors — possibly polluted with
    someone else's face) gets their profile REPLACED by clean anchors on the first
    new enroll, not nudged: the pollution doesn't get to keep 99% of the vote."""
    g = _g()
    conn = g._db()
    try:  # a badly polluted legacy centroid with a heavy samples count
        conn.execute("INSERT INTO faces (user_id, name, embedding, samples) VALUES (?,?,?,208)",
                     ("u1", "David", json.dumps(_vec(9.0))))
        conn.commit()
    finally:
        conn.close()
    g.enroll("u1", "David", _vec(1.0))
    assert {p["user_id"]: p["samples"] for p in g.profiles()}["u1"] == 1
    assert g._member_score_one("u1", _vec(1.0)) > 0.95   # the clean look wins outright
    assert g._member_score_one("u1", _vec(9.0)) < 0.2    # the polluted look is gone


def test_forget_clears_anchors_and_rebuild_replaces_them():
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    g.forget("u1")
    conn = g._db()
    try:
        assert conn.execute("SELECT COUNT(*) FROM anchors WHERE user_id='u1'").fetchone()[0] == 0
    finally:
        conn.close()
    # rebuild: the curated set BECOMES the profile (anchors + derived centroid)
    g.enroll("u1", "David", _vec(1.0))
    g.rebuild_member("u1", [_vec(3.0), _variant(3.0, 4.0)])
    conn = g._db()
    try:
        rows = conn.execute("SELECT COUNT(*) FROM anchors WHERE user_id='u1'").fetchone()[0]
    finally:
        conn.close()
    assert rows == 2
    assert g._member_score_one("u1", _vec(3.0)) > 0.85
    assert g._member_score_one("u1", _vec(1.0)) < 0.3    # pre-rebuild anchor discarded
