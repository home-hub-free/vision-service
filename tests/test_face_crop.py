"""Face-centered review thumbnails — expand_face_box geometry + face_crop_jpeg.

The review card must always SHOW the face: thumbs are cropped around the detected
face bbox (not the full-body track box) and carry the face's normalized position
within the crop so the dashboard can ring exactly the face in question.
"""
import pytest

from app.perception import annotate_face_in_thumb, expand_face_box, face_crop_jpeg


def test_expand_face_box_pads_and_maps_back():
    (cx1, cy1, cx2, cy2), norm = expand_face_box((100, 100, 140, 140), 640, 480)
    assert (cx1, cy1, cx2, cy2) == (76, 76, 164, 180)  # 0.6 pad each side, +0.4 extra below
    fx, fy, fw, fh = norm
    # The normalized box maps back onto the original face bbox through the crop.
    assert abs(cx1 + fx * (cx2 - cx1) - 100) < 1
    assert abs(cy1 + fy * (cy2 - cy1) - 100) < 1
    assert abs(fw * (cx2 - cx1) - 40) < 1
    assert abs(fh * (cy2 - cy1) - 40) < 1


def test_expand_face_box_clamps_to_frame():
    (cx1, cy1, cx2, cy2), norm = expand_face_box((0, 0, 40, 40), 100, 100)
    assert cx1 == 0 and cy1 == 0 and cx2 <= 100 and cy2 <= 100
    assert all(0.0 <= v <= 1.0 for v in norm)


def test_face_crop_jpeg_returns_jpeg_plus_normalized_box():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = face_crop_jpeg(frame, (100, 100, 140, 140))
    assert out is not None
    jpeg, box = out
    assert jpeg[:2] == b"\xff\xd8"  # JPEG SOI marker
    assert len(box) == 4 and all(0.0 <= v <= 1.0 for v in box)


def test_face_crop_jpeg_degenerate_box_is_none():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert face_crop_jpeg(frame, (50, 50, 50, 60)) is None


def test_annotate_face_in_thumb_skips_without_a_real_engine():
    # Null face backend (test default): annotation reports "engine unavailable"
    # (None), NOT "no face" ([]), so the gallery retries once a real engine exists.
    assert annotate_face_in_thumb(b"not-a-jpeg", [0.0] * 8) is None
