"""Gallery — household match, guest clustering, promote. Plain vectors, no models."""
import os
import tempfile

from app.config import cfg
from app.gallery import Gallery


def _g():
    d = tempfile.mkdtemp()
    return Gallery(os.path.join(d, "gallery.db"))


def _vec(seed: float, dim: int = 16):
    # Distinct seeds -> ~orthogonal directions (one-hot-ish), same seed -> identical.
    # So an identical embedding cosine-matches and distinct ones don't (cosine ~0).
    v = [0.01] * dim
    v[int(seed) % dim] = 1.0
    return v


def test_household_enroll_and_resolve():
    cfg.face_match_threshold = 0.9
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    ident = g.resolve(_vec(1.0))
    assert ident.cls == "household" and ident.id == "u1" and ident.name == "David"
    assert ident.via == "face" and ident.confidence > 0.7


def test_unknown_becomes_guest_then_recurring():
    cfg.face_match_threshold = 0.99
    cfg.guest_cluster_threshold = 0.999999
    cfg.guest_min_sightings = 3
    g = _g()
    v = _vec(2.0)
    ident = g.resolve(v)
    assert ident.cls == "guest" and ident.id == "guest:1"
    g.resolve(v)
    g.resolve(v)  # 3rd sighting → recurring
    recurring = g.guests(recurring_only=True)
    assert any(x["guest_id"] == "guest:1" and x["recurring"] for x in recurring)


def test_promote_guest_seeds_household_gallery():
    cfg.face_match_threshold = 0.9
    cfg.guest_cluster_threshold = 0.999999
    g = _g()
    v = _vec(3.0)
    g.resolve(v)  # creates guest:1
    assert g.promote_guest("guest:1", "u9", "Sam")
    ident = g.resolve(v)  # now matches the promoted household member
    assert ident.cls == "household" and ident.id == "u9" and ident.name == "Sam"


def test_forget_removes_profile():
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    g.forget("u1")
    assert g.profiles() == []


def test_default_label_and_thumb_stored_for_every_person():
    cfg.face_match_threshold = 0.99
    cfg.guest_cluster_threshold = 0.999999
    g = _g()
    # A detected unknown is labelled by default AND keeps its captured face crop.
    ident = g.resolve(_vec(5.0), thumb=b"\xff\xd8FACEJPEG\xff\xd9")
    assert ident.cls == "guest" and ident.id == "guest:1"
    assert g.default_label("guest:1") == "Person 1"
    assert g.get_thumb("guest:1") == b"\xff\xd8FACEJPEG\xff\xd9"


def test_people_roster_lists_household_and_guests_with_labels():
    cfg.face_match_threshold = 0.99
    cfg.guest_cluster_threshold = 0.999999
    g = _g()
    g.enroll("u1", "David", _vec(1.0), thumb=b"DAVIDJPEG")
    g.resolve(_vec(6.0), thumb=b"GUESTJPEG")  # guest:1, unnamed → "Person 1"
    people = {p["id"]: p for p in g.people()}
    assert people["u1"]["label"] == "David" and people["u1"]["class"] == "household"
    assert people["u1"]["has_thumb"] and people["u1"]["named"]
    assert people["guest:1"]["label"] == "Person 1" and people["guest:1"]["class"] == "guest"
    assert people["guest:1"]["has_thumb"] and not people["guest:1"]["named"]


def test_promoted_guest_drops_from_people_and_carries_face():
    cfg.face_match_threshold = 0.9
    cfg.guest_cluster_threshold = 0.999999
    g = _g()
    g.resolve(_vec(7.0), thumb=b"GUESTFACE")  # guest:1
    g.promote_guest("guest:1", "u9", "Sam")
    ids = {p["id"] for p in g.people()}
    assert "guest:1" not in ids and "u9" in ids       # promoted guest no longer a guest row
    assert g.get_thumb("u9") == b"GUESTFACE"           # face carried into the member profile


def test_online_reinforcement_strengthens_household_on_confident_match():
    cfg.face_match_threshold = 0.3
    cfg.face_reinforce = True
    cfg.face_reinforce_threshold = 0.5
    cfg.face_reinforce_margin = 0.05
    cfg.face_reinforce_cap = 50
    g = _g()
    g.enroll("u1", "David", _vec(1.0))            # samples = 1
    assert g.profiles()[0]["samples"] == 1
    ident = g.resolve(_vec(1.0))                  # confident match, sole member (margin=inf)
    assert ident.cls == "household" and ident.id == "u1"
    assert g.profiles()[0]["samples"] == 2        # ← passive recognition reinforced the centroid
    g.resolve(_vec(1.0))
    assert g.profiles()[0]["samples"] == 3        # keeps self-improving on each confident sighting


def test_reinforcement_skips_ambiguous_match_and_respects_toggle():
    cfg.face_match_threshold = 0.3
    cfg.face_reinforce = True
    cfg.face_reinforce_threshold = 0.5
    cfg.face_reinforce_margin = 0.05
    cfg.face_reinforce_cap = 50
    g = _g()
    # Two members with identical embeddings → a tie (margin 0): a look-alike. Reinforcement
    # must NOT fire — drifting one centroid on a 50/50 call is exactly the failure to avoid.
    g.enroll("u1", "Ann", _vec(1.0))
    g.enroll("u2", "Bea", _vec(1.0))
    g.resolve(_vec(1.0))
    assert all(p["samples"] == 1 for p in g.profiles())   # neither reinforced

    # Kill-switch: disabled → never reinforce, even on a clean sole-member match.
    cfg.face_reinforce = False
    g2 = _g()
    g2.enroll("u1", "David", _vec(1.0))
    g2.resolve(_vec(1.0))
    assert g2.profiles()[0]["samples"] == 1
    cfg.face_reinforce = True   # restore default for any later tests
