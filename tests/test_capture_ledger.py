"""Capture ledger + profile rebuild — the identity-pollution insurance.

Centroids are running means, so pollution folded into a member can never be exactly
removed (salt in the pot). The ledger permanently keeps every crop + exact embedding
behind every identity decision, and `rebuild_member` re-makes the profile from a
curated set (same ingredients, less salt).
"""
import os
import tempfile

from app.config import cfg
from app.gallery import Gallery


def _g():
    d = tempfile.mkdtemp()
    return Gallery(os.path.join(d, "gallery.db"))


def _vec(seed: float, dim: int = 16):
    v = [0.01] * dim
    v[int(seed) % dim] = 1.0
    return v


JPEG = b"\xff\xd8CROP\xff\xd9"


def test_resolve_and_enroll_archive_crop_and_embedding_permanently():
    cfg.face_match_threshold = 0.5
    cfg.face_match_margin = 0.05
    cfg.guest_cluster_threshold = 0.999999
    cfg.captures_enabled = True
    g = _g()
    g.enroll("u1", "David", _vec(1.0), thumb=JPEG)
    g.resolve(_vec(1.0), thumb=JPEG)               # household match
    g.resolve(_vec(2.0), thumb=JPEG, thumb_box=[0.1, 0.1, 0.5, 0.5])  # new guest
    rows = g.captures()
    kinds = {r["kind"] for r in rows}
    assert kinds == {"enroll", "match", "cluster"}
    match = next(r for r in rows if r["kind"] == "match")
    assert match["resolved_id"] == "u1" and match["score"] is not None
    # Crops are plain reviewable JPEGs on disk, grouped per identity, next to the DB.
    for r in rows:
        p = os.path.join(g.captures_dir, r["path"])
        assert os.path.isfile(p)
        with open(p, "rb") as fh:
            assert fh.read() == JPEG
    assert g.captures_dir.startswith(os.path.dirname(g.db_path))  # temp DB → temp crops
    # Per-identity filter (what export uses).
    assert {r["kind"] for r in g.captures("u1")} == {"enroll", "match"}


def test_capture_ledger_disabled_or_thumbless_records_nothing():
    cfg.face_match_threshold = 0.5
    cfg.guest_cluster_threshold = 0.999999
    g = _g()
    g.resolve(_vec(3.0))                            # no crop → nothing to review
    cfg.captures_enabled = False
    try:
        g.resolve(_vec(3.0), thumb=JPEG)
    finally:
        cfg.captures_enabled = True
    assert g.captures() == []


def test_rebuild_member_discards_polluted_centroid():
    """A polluted profile (Ana folded into David) must be fully recoverable from
    curated ingredients: rebuild REPLACES the centroid (enroll would only nudge a
    seasoned mean), after which David matches and Ana no longer does."""
    cfg.face_match_threshold = 0.5
    cfg.face_match_margin = 0.05
    cfg.guest_cluster_threshold = 0.999999
    g = _g()
    david, ana = _vec(1.0), _vec(2.0)
    for _ in range(6):
        g.enroll("u1", "David", david, thumb=JPEG)  # seasoned profile...
    for _ in range(6):
        g.enroll("u1", "David", ana)                # ...heavily polluted with Ana
    assert g.resolve(ana).id == "u1"                # the swap bug: Ana reads as David
    samples = g.rebuild_member("u1", [david, david, david])
    assert samples == 3
    prof = g.profiles()[0]
    assert prof["samples"] == 3 and prof["name"] == "David"  # name/thumb kept
    assert prof["has_thumb"]
    assert g.resolve(david).id == "u1"              # David still matches
    assert g.resolve(ana).cls == "guest"            # Ana no longer reads as David
