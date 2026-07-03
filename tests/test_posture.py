"""T1 posture geometry (VISION_CONTEXT_TIERS_PLAN §3) — pure math, no models.

Synthetic COCO-17 skeletons: only the joints the classifier reads (shoulders, hips,
knees, ankles) get real coordinates; everything else rides at conf 0.
"""
from app.perception import (DetectedTrack, bbox_iou, classify_posture,
                            match_poses_to_tracks)


def _skel(shoulder=None, hip=None, knee=None, ankle=None):
    """17 keypoints, (x, y, conf); pairs are given as one midpoint (both sides equal)."""
    kps = [(0.0, 0.0, 0.0)] * 17
    def put(idx_pair, pt):
        if pt is not None:
            for i in idx_pair:
                kps[i] = (float(pt[0]), float(pt[1]), 0.9)
    put((5, 6), shoulder)
    put((11, 12), hip)
    put((13, 14), knee)
    put((15, 16), ankle)
    return kps


def test_standing_upright_torso_extended_legs():
    kps = _skel(shoulder=(100, 100), hip=(100, 200), knee=(100, 290), ankle=(100, 380))
    assert classify_posture(kps, (60, 40, 140, 400), min_conf=0.3) == "standing"


def test_sitting_side_view_thigh_horizontal():
    kps = _skel(shoulder=(100, 100), hip=(105, 200), knee=(190, 210), ankle=(190, 300))
    assert classify_posture(kps, (60, 40, 220, 320), min_conf=0.3) == "sitting"


def test_sitting_front_view_collapsed_leg_span():
    # Thighs point at the camera: knee reads almost under the hip, ankles barely lower.
    kps = _skel(shoulder=(100, 100), hip=(100, 200), knee=(100, 230), ankle=(100, 260))
    assert classify_posture(kps, (60, 40, 140, 280), min_conf=0.3) == "sitting"


def test_lying_horizontal_torso():
    kps = _skel(shoulder=(100, 200), hip=(220, 210), knee=(300, 215), ankle=(380, 220))
    assert classify_posture(kps, (60, 160, 420, 260), min_conf=0.3) == "lying"


def test_bent_at_counter():
    # Torso tilted ~45° from vertical (leaning over a counter), legs extended.
    kps = _skel(shoulder=(100, 100), hip=(180, 180), knee=(185, 280), ankle=(190, 380))
    assert classify_posture(kps, (60, 60, 240, 400), min_conf=0.3) == "bent"


def test_unreadable_torso_falls_back_to_bbox_shape():
    empty = _skel()
    assert classify_posture(empty, (0, 0, 300, 100), min_conf=0.3) == "lying"  # wide box
    assert classify_posture(empty, (0, 0, 100, 300), min_conf=0.3) is None     # tall box: unknown


def test_match_poses_to_tracks_by_iou():
    t1 = DetectedTrack(track_id="1", bbox=(0, 0, 100, 200))
    t2 = DetectedTrack(track_id="2", bbox=(300, 0, 400, 200))
    kps_a, kps_b = _skel(shoulder=(50, 20)), _skel(shoulder=(350, 20))
    poses = [(kps_a, (5, 5, 95, 195)), (kps_b, (305, 5, 395, 195))]
    m = match_poses_to_tracks(poses, [t1, t2])
    assert m["1"] is kps_a and m["2"] is kps_b
    # A pose overlapping nothing is dropped.
    assert match_poses_to_tracks([(kps_a, (900, 900, 950, 990))], [t1, t2]) == {}


def test_bbox_iou_sanity():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
