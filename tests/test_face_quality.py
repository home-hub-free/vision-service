"""Identity-grade quality gate — the 2026-07-07 pollution lesson.

Measured on the live capture ledger: ArcFace embeddings of small / blurry /
turned-away faces are noise (one member's own enroll burst self-agreed at cos ~0.2),
and averaging hundreds of noise vectors converged two members' centroids onto the
shared "average junk" direction (cos 0.702) — the swap engine. These tests pin the
gate that keeps such faces OUT of identity decisions: `face_quality_reason` (live,
engine-enforced), `assess_enroll` (strict, actionable reasons), and the enroll
route's 422 mapping the guided flow shows the user.
"""
import io

import pytest
from fastapi.testclient import TestClient

from app.config import cfg
from app.main import app
from app.perception import _InsightFaceEngine, face_quality_reason
from app.routes import enroll as enroll_routes

client = TestClient(app)

np = pytest.importorskip("numpy")


class _Face:
    """Duck-typed insightface Face: det_score / pose (pitch, yaw, roll) / bbox /
    normed_embedding."""

    def __init__(self, det=0.9, pose=(0.0, 0.0, 0.0), bbox=(0, 0, 160, 160), emb=None):
        self.det_score = det
        self.pose = pose
        self.bbox = bbox
        self.normed_embedding = np.array(emb if emb is not None else [1.0, 0.0, 0.0],
                                         dtype=np.float32)


def _sharp_frame(w=320, h=320):
    """High-frequency noise — Laplacian variance far above any sane threshold."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def _flat_frame(w=320, h=320):
    """Uniform gray — Laplacian variance 0 (maximum blur)."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _reset_gates():
    cfg.face_min_det_score = 0.55
    cfg.face_max_yaw = 45.0
    cfg.face_max_pitch = 35.0
    cfg.face_min_sharpness = 30.0
    cfg.enroll_min_px = 110
    cfg.enroll_min_det_score = 0.65
    cfg.enroll_max_yaw = 30.0
    cfg.enroll_max_pitch = 25.0
    cfg.enroll_min_sharpness = 60.0


def test_live_gate_reasons():
    _reset_gates()
    frame = _sharp_frame()
    assert face_quality_reason(_Face(), frame) is None
    assert face_quality_reason(_Face(det=0.4), frame) == "low_confidence"
    assert face_quality_reason(_Face(pose=(0.0, 70.0, 0.0)), frame) == "off_angle"
    assert face_quality_reason(_Face(pose=(-50.0, 0.0, 0.0)), frame) == "off_angle"
    assert face_quality_reason(_Face(), _flat_frame()) == "blurry"


def test_live_gate_skips_missing_pose_and_det():
    """Engines/models that don't supply pose or det_score must not be over-rejected —
    each bar only applies when its signal exists."""
    _reset_gates()
    f = _Face()
    f.pose = None
    del f.det_score
    assert face_quality_reason(f, _sharp_frame()) is None


def test_enroll_gate_is_stricter_than_live():
    """A 40° yaw face passes the live bar (45°) but fails enrollment (30°) — ground
    truth is held to a higher standard than day-to-day sightings."""
    _reset_gates()
    frame = _sharp_frame()
    f = _Face(pose=(0.0, 40.0, 0.0))
    assert face_quality_reason(f, frame) is None
    assert face_quality_reason(f, frame, enroll=True) == "off_angle"


class _FakeApp:
    def __init__(self, faces):
        self._faces = faces

    def get(self, _frame):
        return list(self._faces)


def _engine(faces):
    eng = _InsightFaceEngine.__new__(_InsightFaceEngine)  # skip model loading
    eng._app = _FakeApp(faces)
    return eng


def test_assess_enroll_reasons():
    _reset_gates()
    frame = _sharp_frame()
    assert _engine([]).assess_enroll(frame) == (None, "no_face")
    # two comparable faces → whose profile would this feed? refuse.
    two = [_Face(bbox=(0, 0, 160, 160)), _Face(bbox=(160, 0, 300, 140))]
    assert _engine(two).assess_enroll(frame)[1] == "multiple_faces"
    # a tiny background face does NOT block the main subject
    bg = [_Face(bbox=(0, 0, 160, 160)), _Face(bbox=(300, 0, 330, 30))]
    emb, reason = _engine(bg).assess_enroll(frame)
    assert reason is None and emb is not None
    assert _engine([_Face(bbox=(0, 0, 80, 80))]).assess_enroll(frame)[1] == "too_small"
    assert _engine([_Face(det=0.5)]).assess_enroll(frame)[1] == "low_confidence"
    assert _engine([_Face(pose=(0.0, 35.0, 0.0))]).assess_enroll(frame)[1] == "off_angle"
    assert _engine([_Face()]).assess_enroll(_flat_frame())[1] == "blurry"


def test_enroll_route_maps_reason_to_actionable_422(monkeypatch):
    monkeypatch.setattr(enroll_routes, "user_from_token",
                        lambda authorization: {"id": "u1", "displayName": "David"})
    monkeypatch.setattr(enroll_routes, "enroll_embedding",
                        lambda data: (None, "too_small"))
    r = client.post("/faces/enroll", files={"image": ("f.jpg", io.BytesIO(b"\xff\xd8x"), "image/jpeg")})
    assert r.status_code == 422
    assert "closer" in r.json()["detail"]  # actionable coaching, not a shrug
    monkeypatch.setattr(enroll_routes, "enroll_embedding",
                        lambda data: (None, "multiple_faces"))
    r = client.post("/faces/enroll", files={"image": ("f.jpg", io.BytesIO(b"\xff\xd8x"), "image/jpeg")})
    assert r.status_code == 422 and "one person" in r.json()["detail"]
