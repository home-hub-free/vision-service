"""The two-people-in-one-room identity swap (debugged 2026-07-06).

Three compounding mechanisms, one test file:
  1. face→track assignment: "largest face inside my (overlapping) person box" embeds
     the OTHER person's face → exclusive containment-based assignment.
  2. one member on two tracks at once: per-track resolves are independent → the
     per-frame household dedupe.
  3. wrongly-promoted / coin-flip clusters: answered as (and reinforced) their member
     for whoever walked by → resolve's promoted-path ambiguity gate.
"""
import os
import tempfile

from app.camera import CameraWorker
from app.config import cfg
from app.gallery import Gallery
from app.hub_client import Camera
from app.occupancy import Identity
from app.perception import DetectedTrack, assign_faces_to_tracks


def _track(tid, box):
    return DetectedTrack(track_id=tid, bbox=box)


# ── 1. exclusive face→track assignment ────────────────────────────────────────
def test_face_goes_to_tightest_containing_track_not_largest_in_box():
    """Ana stands in front of David: David's (bigger, overlapping) box contains BOTH
    faces and Ana's is larger — the old rule embedded Ana's face for David's track.
    Containment assignment gives each face to its own (tightest) track."""
    david = _track("d", (0, 0, 400, 600))       # big box, overlaps Ana's
    ana = _track("a", (150, 100, 350, 500))     # tighter box inside David's
    ana_face = ([0.1] * 4, (220, 140, 300, 220))   # inside BOTH boxes, larger
    david_face = ([0.9] * 4, (40, 30, 90, 80))     # inside David's box only, smaller
    assigned = assign_faces_to_tracks([ana_face, david_face], [david, ana])
    assert assigned["a"][1] == (220, 140, 300, 220)
    assert assigned["d"][1] == (40, 30, 90, 80)


def test_one_face_per_track_and_unowned_faces_dropped():
    t = _track("t", (0, 0, 200, 200))
    small = ([0.1] * 4, (10, 10, 30, 30))
    big = ([0.2] * 4, (50, 50, 150, 150))
    outside = ([0.3] * 4, (500, 500, 560, 560))
    assigned = assign_faces_to_tracks([small, big, outside], [t])
    assert list(assigned) == ["t"]
    assert assigned["t"][1] == (50, 50, 150, 150)  # largest face wins WITHIN a track


# ── 2. camera worker: frame-level embed + per-frame household dedupe ─────────
class _FakeFaceEngine:
    backend = "fake"

    def __init__(self, faces):
        self._faces = faces
        self.calls = []

    def faces(self, frame):
        self.calls.append("faces")
        return self._faces

    def embed_face(self, frame, bbox):
        self.calls.append(("embed_face", bbox))
        return None


def _worker():
    return CameraWorker(Camera({"id": "t", "zone": "z", "ip": "1.2.3.4",
                                "stream": {"port": 81, "path": "/s"}}))


def test_embed_tracks_multi_person_uses_one_frame_detection_exclusively():
    w = _worker()
    tracks = [_track("d", (0, 0, 400, 600)), _track("a", (150, 100, 350, 500))]
    w.face = _FakeFaceEngine([([0.1] * 4, (220, 140, 300, 220)),
                              ([0.9] * 4, (40, 30, 90, 80))])
    out = w._embed_tracks(None, tracks, {"d": "resolve", "a": "resolve"})
    assert w.face.calls == ["faces"]           # ONE frame-level pass, no per-crop calls
    assert out["a"][0] == [0.1] * 4 and out["d"][0] == [0.9] * 4


def test_embed_tracks_single_person_keeps_per_crop_path():
    w = _worker()
    tracks = [_track("d", (0, 0, 400, 600))]
    w.face = _FakeFaceEngine([])
    out = w._embed_tracks(None, tracks, {"d": "resolve"})
    assert w.face.calls == [("embed_face", (0, 0, 400, 600))]
    assert out["d"] == (None, None, None)


def test_dedupe_household_keeps_best_confidence_and_resets_the_rest():
    w = _worker()
    david_hi = Identity(id="u1", name="David", cls="household", confidence=0.9)
    david_lo = Identity(id="u1", name="David", cls="household", confidence=0.72)
    ana = Identity(id="u2", name="Ana", cls="household", confidence=0.8)
    idents = {"t1": david_hi, "t2": david_lo, "t3": ana}
    w._idents.update(idents)
    w._ident_ts.update({"t1": 1.0, "t2": 1.0, "t3": 1.0})
    w._dedupe_household(idents)
    assert idents["t1"] is david_hi and idents["t3"] is ana
    assert idents["t2"].cls == "unknown"       # loser resets, re-embeds next frame
    assert "t2" not in w._idents and "t2" not in w._ident_ts
    assert "t1" in w._idents and "t3" in w._idents


# ── 3. gallery: promoted-path ambiguity gate ──────────────────────────────────
def _g():
    d = tempfile.mkdtemp()
    return Gallery(os.path.join(d, "gallery.db"))


def _vec(seed: float, dim: int = 16):
    v = [0.01] * dim
    v[int(seed) % dim] = 1.0
    return v


def _mix(a, b, wa, wb):
    return [wa * x + wb * y for x, y in zip(a, b)]


def test_wrongly_promoted_cluster_stays_anonymous_and_never_reinforces():
    """A cluster of David's face promoted into Ana (a wrong review answer / autoheal)
    must NOT label David as Ana, and must NOT fold David's embedding into Ana's
    centroid (that reinforcement is how the two members' centroids converged to
    cosine 0.459 in prod and started swapping)."""
    cfg.face_match_threshold = 0.75   # the probe (cos ~0.7 vs u1) fails the strict gate
    cfg.face_match_margin = 0.05
    cfg.guest_cluster_threshold = 0.9
    cfg.face_reinforce = True
    cfg.face_reinforce_threshold = 0.5
    cfg.face_reinforce_margin = 0.08
    cfg.face_autoheal_threshold = 2.0  # isolate the promoted path from autoheal
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    for _ in range(8):                 # season Ana's centroid (promote barely moves it)
        g.enroll("u2", "Ana", _vec(2.0))
    probe = _mix(_vec(1.0), _vec(3.0), 0.7, 0.72)  # David-ish look from a far camera
    cfg.face_match_threshold = 2.0     # force the cluster to seed without a member match
    ident = g.resolve(probe)
    assert ident.cls == "guest" and ident.id == "guest:1"
    assert g.promote_guest("guest:1", "u2", "Ana", carry_thumb=False)  # the WRONG answer
    cfg.face_match_threshold = 0.75
    ana_before = {p["user_id"]: p["samples"] for p in g.profiles()}["u2"]
    ident = g.resolve(probe)           # David walks by again
    assert ident.cls == "guest" and ident.name is None   # never asserted as Ana
    ana_after = {p["user_id"]: p["samples"] for p in g.profiles()}["u2"]
    assert ana_after == ana_before     # Ana's centroid untouched by David's face


def test_rightly_promoted_cluster_still_answers_and_reinforces_with_other_members_present():
    """The legit far-camera case must keep working when the household has several
    members: the promoted member wins the live embedding decisively (just not the
    absolute threshold) → answer as them + reinforce."""
    cfg.face_match_threshold = 0.99
    cfg.face_match_margin = 0.05
    cfg.guest_cluster_threshold = 0.9
    cfg.face_reinforce = True
    cfg.face_reinforce_threshold = 0.5
    cfg.face_reinforce_margin = 0.08
    cfg.face_autoheal_threshold = 2.0
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    g.enroll("u2", "Ana", _vec(2.0))
    probe = _mix(_vec(2.0), _vec(3.0), 0.8, 0.55)  # Ana-ish, fails the 0.99 gate
    ident = g.resolve(probe)
    assert ident.cls == "guest" and ident.id == "guest:1"
    assert g.promote_guest("guest:1", "u2", "Ana", carry_thumb=False)  # the RIGHT answer
    before = {p["user_id"]: p["samples"] for p in g.profiles()}["u2"]
    ident = g.resolve(probe)
    assert ident.cls == "household" and ident.id == "u2" and ident.name == "Ana"
    after = {p["user_id"]: p["samples"] for p in g.profiles()}["u2"]
    assert after == before + 1         # convergence reinforcement preserved
