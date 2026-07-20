"""Perception backends — person-detect+track and face-detect+embed.

This is where most of CAMERA_VISION_PLAN §11's "things to figure out" live, so the
module is built as a clean SEAM with a runnable null default:

  * `Detector.detect_and_track(frame) -> [DetectedTrack]` — YOLO person + ByteTrack
    (§4.2). NullDetector returns [] → the service still streams/records (M0).
  * `FaceEngine.embed(frame, bbox) -> [float] | None` — SCRFD detect + ArcFace embed
    inside a person box (§4.2). NullFaceEngine returns None → presence without ID (M1).

The real backend (`VISION_BACKEND=ultralytics`, `VISION_FACE_BACKEND=insightface`) is
import-guarded and LAZY: if the heavy deps or ROCm aren't there it logs and falls back
to null, so a bad GPU day degrades to presence-only instead of crashing the box.

OPEN DECISIONS surfaced here (see ../DECISIONS.md):
  - §11.2 ROCm runtime: torch-ROCm (ultralytics' native) vs onnxruntime-ROCm (export
    to ONNX). Selected via VISION_DEVICE / the provider list below; default "cpu" so
    nothing is blocked on ROCm during bring-up.
  - §11.1 GPU contention: detect_fps cap + embed-on-new-track gating live in the worker,
    not here; this module just exposes a cheap-as-possible single-frame call.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .config import cfg

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class DetectedTrack:
    track_id: str
    bbox: BBox
    score: float = 0.0
    face_embedding: Optional[List[float]] = field(default=None)


# ── decode / annotate (cv2 optional — only needed once a real backend is on) ──
def _cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None


def decode_jpeg(buf: bytes):
    """JPEG bytes → BGR ndarray, or None if cv2/numpy aren't installed (null build)."""
    cv2 = _cv2()
    if cv2 is None:
        return None
    import numpy as np  # type: ignore
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_jpeg(frame) -> Optional[bytes]:
    cv2 = _cv2()
    if cv2 is None or frame is None:
        return None
    ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return enc.tobytes() if ok else None


def _resize_max(frame, max_dim: int):
    """Downscale a frame so its longest side is <= max_dim (face thumbnails stay small)."""
    cv2 = _cv2()
    if cv2 is None or frame is None:
        return frame
    h, w = frame.shape[:2]
    scale = max_dim / float(max(h, w)) if max(h, w) > max_dim else 1.0
    if scale < 1.0:
        return cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


def crop_jpeg(frame, bbox: BBox, max_dim: int = 220) -> Optional[bytes]:
    """A small JPEG of the bbox region — the captured face/person thumbnail stored per
    label so the dashboard can show "their face" (CAMERA_VISION_PLAN §6). None in the
    null build (no cv2): identity needs a real backend anyway, so thumbs ride with it."""
    if frame is None:
        return None
    x1, y1, x2, y2 = bbox
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    return encode_jpeg(_resize_max(crop, max_dim))


def expand_face_box(face_box: BBox, frame_w: int, frame_h: int,
                    pad: float = 0.6) -> Tuple[BBox, List[float]]:
    """Grow a face bbox into a review-friendly crop window (a little extra below —
    chin/shoulders read better than forehead+ceiling), clamped to the frame. Returns
    (crop bbox, face box normalized [x,y,w,h] WITHIN that crop) — the normalized box
    is what the dashboard uses to ring the face. Pure math (no cv2) so it's testable
    everywhere."""
    x1, y1, x2, y2 = face_box
    fw, fh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - fw * pad))
    cy1 = max(0, int(y1 - fh * pad))
    cx2 = min(frame_w, int(x2 + fw * pad))
    cy2 = min(frame_h, int(y2 + fh * (pad + 0.4)))
    cw, ch = max(1, cx2 - cx1), max(1, cy2 - cy1)
    norm = [round((x1 - cx1) / cw, 4), round((y1 - cy1) / ch, 4),
            round(fw / cw, 4), round(fh / ch, 4)]
    return (cx1, cy1, cx2, cy2), norm


def face_crop_jpeg(frame, face_box: BBox,
                   max_dim: int = 220) -> Optional[Tuple[bytes, List[float]]]:
    """A face-CENTERED thumbnail (the review card's photo): the detected face bbox
    expanded via expand_face_box, so the person's face is always in frame — never a
    full-body sliver where the face gets crop-cut. Returns (jpeg, normalized face box
    within the crop) or None (no cv2 / degenerate box)."""
    if frame is None:
        return None
    x1, y1, x2, y2 = face_box
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    h, w = frame.shape[:2]
    (cx1, cy1, cx2, cy2), norm = expand_face_box(face_box, w, h)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    jpeg = encode_jpeg(_resize_max(crop, max_dim))
    return (jpeg, norm) if jpeg else None


def thumbnail_jpeg(jpeg: bytes, max_dim: int = 220) -> bytes:
    """Downscale an uploaded enroll image to a thumbnail (passthrough if cv2 absent)."""
    frame = decode_jpeg(jpeg)
    if frame is None:
        return jpeg
    return encode_jpeg(_resize_max(frame, max_dim)) or jpeg


# Overlay palette (BGR) — the dashboard's "Tablero" tokens (_tokens.scss), so the
# annotated view reads as part of the same UI: ink plates, bone text, one calm
# accent per identity tier. State is a flat fill + an indicator dot, never a glow.
_INK = (12, 17, 23)       # #17110c — label plate / keyline under-stroke
_BONE = (214, 228, 236)   # #ece4d6 — label text
_SAGE = (128, 169, 116)   # #74a980 — recognized household member
_STONE = (181, 154, 125)  # #7d9ab5 — "Person N" guest cluster
_MUTED = (129, 145, 156)  # #9c9181 — unresolved detection
_CHIP_ALPHA = 0.86


def _tier_color(name: str) -> Tuple[int, int, int]:
    if name in ("person", "unknown"):
        return _MUTED
    if re.fullmatch(r"Person \d+", name):
        return _STONE
    return _SAGE


def _stroked_rect(cv2, frame, p1, p2, color, thickness: int) -> None:
    """Accent stroke over an ink under-stroke — legible on bright and dark scenes."""
    cv2.rectangle(frame, p1, p2, _INK, thickness + 2, cv2.LINE_AA)
    cv2.rectangle(frame, p1, p2, color, thickness, cv2.LINE_AA)


def _corner_brackets(cv2, frame, bbox: BBox, color, arm: int, thickness: int) -> None:
    import numpy as np  # type: ignore
    x1, y1, x2, y2 = bbox
    pts = [np.array(c, dtype=np.int32) for c in (
        [(x1 + arm, y1), (x1, y1), (x1, y1 + arm)],
        [(x2 - arm, y1), (x2, y1), (x2, y1 + arm)],
        [(x1 + arm, y2), (x1, y2), (x1, y2 - arm)],
        [(x2 - arm, y2), (x2, y2), (x2, y2 - arm)],
    )]
    cv2.polylines(frame, pts, False, _INK, thickness + 2, cv2.LINE_AA)
    cv2.polylines(frame, pts, False, color, thickness, cv2.LINE_AA)


def _rounded_fill(cv2, img, w: int, h: int, r: int, color) -> None:
    cv2.rectangle(img, (r, 0), (w - r - 1, h - 1), color, -1)
    cv2.rectangle(img, (0, r), (w - 1, h - r - 1), color, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def _label_chip(cv2, frame, bbox: BBox, name: str, color, s: float) -> None:
    """The name on a translucent ink plate with a tier indicator dot — sits just
    above the box (falls inside its top edge when clipped by the frame)."""
    fh, fw = frame.shape[:2]
    x1, y1, _, _ = bbox
    font, fscale = cv2.FONT_HERSHEY_SIMPLEX, 0.44 * s
    fthick = max(1, round(s))
    (tw, th), baseline = cv2.getTextSize(name, font, fscale, fthick)
    pad_x, pad_y, gap = round(8 * s), round(5 * s), round(5 * s)
    dot_r = max(2, round(2.6 * s))
    cw = tw + dot_r * 2 + gap + pad_x * 2
    ch = th + baseline + pad_y * 2
    cx = min(max(x1, 0), max(fw - cw, 0))
    cy = y1 - ch - round(6 * s)
    if cy < 0:  # no room above → inside the box, nudged off the keyline
        cy = min(y1 + round(6 * s), max(fh - ch, 0))
        cx = min(max(x1 + round(6 * s), 0), max(fw - cw, 0))
    roi = frame[cy:cy + ch, cx:cx + cw]
    if roi.size == 0 or roi.shape[0] != ch or roi.shape[1] != cw:
        return
    plate = roi.copy()
    _rounded_fill(cv2, plate, cw, ch, min(round(4 * s), ch // 2, cw // 2), _INK)
    frame[cy:cy + ch, cx:cx + cw] = cv2.addWeighted(plate, _CHIP_ALPHA, roi, 1 - _CHIP_ALPHA, 0)
    dot = (cx + pad_x + dot_r, cy + ch // 2)
    cv2.circle(frame, dot, dot_r, color, -1, cv2.LINE_AA)
    cv2.putText(frame, name, (dot[0] + dot_r + gap, cy + pad_y + th),
                font, fscale, _BONE, fthick, cv2.LINE_AA)


def draw_overlay(frame, tracks: List[DetectedTrack], labels: dict) -> bytes:
    """Annotate a frame with the Tablero identity overlay → JPEG bytes for the
    dashboard's annotated view (§6): a keyline + corner brackets around each person
    and an ink label chip that stays readable over any scene. Accent = identity
    tier (sage member / stone-blue guest / muted unresolved). `labels[track_id]` is
    the resolved display string. cv2-only; the null build never calls this (it
    relays raw frames)."""
    cv2 = _cv2()
    if cv2 is None or frame is None:
        return b""
    fh, fw = frame.shape[:2]
    s = min(2.0, max(0.75, min(fw, fh) / 720))
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        name = labels.get(t.track_id, "person")
        color = _tier_color(name)
        _stroked_rect(cv2, frame, (x1, y1), (x2, y2), color, max(1, round(s)))
        arm = min(round(min(max(10 * s, 0.18 * min(x2 - x1, y2 - y1)), 26 * s)),
                  (x2 - x1) // 2, (y2 - y1) // 2)
        if arm > 0:
            _corner_brackets(cv2, frame, t.bbox, color, arm, max(2, round(2 * s)))
        _label_chip(cv2, frame, t.bbox, name, color, s)
    return encode_jpeg(frame) or b""


# ── detectors ────────────────────────────────────────────────────────────────
class NullDetector:
    backend = "null"

    def detect_and_track(self, frame) -> List[DetectedTrack]:
        return []


class _UltralyticsDetector:
    """YOLO person-detect + ByteTrack (ultralytics). Person class only, low input res
    (§4.2 / §11.1). Lazy-loaded; raises on import failure so the factory falls back."""

    backend = "ultralytics"

    def __init__(self) -> None:
        from ultralytics import YOLO  # type: ignore
        model = os.getenv("VISION_YOLO_MODEL", "yolov8n.pt")
        self.device = os.getenv("VISION_DEVICE", "cpu")  # "cpu" | "cuda" (ROCm maps here)
        self.imgsz = int(os.getenv("VISION_IMGSZ", "640"))
        self._yolo = YOLO(model)
        print(f"[vision] ultralytics {model} on {self.device}", flush=True)

    def detect_and_track(self, frame) -> List[DetectedTrack]:
        # persist=True keeps ByteTrack ids stable across frames; classes=[0] = person.
        res = self._yolo.track(frame, persist=True, classes=[0], imgsz=self.imgsz,
                               conf=cfg.person_conf, device=self.device, verbose=False)
        out: List[DetectedTrack] = []
        if not res:
            return out
        boxes = getattr(res[0], "boxes", None)
        if boxes is None or boxes.id is None:
            return out
        for xyxy, tid, conf in zip(boxes.xyxy.tolist(), boxes.id.tolist(), boxes.conf.tolist()):
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            out.append(DetectedTrack(track_id=str(int(tid)), bbox=(x1, y1, x2, y2), score=float(conf)))
        return out


# ── pose → posture (VISION_CONTEXT_TIERS_PLAN §3 — T1) ───────────────────────
# COCO-17 keypoint indices (what yolov8n-pose emits).
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16

Keypoints = List[Tuple[float, float, float]]  # (x, y, conf) in frame pixels

POSTURE_STANDING = "standing"
POSTURE_SITTING = "sitting"
POSTURE_LYING = "lying"
POSTURE_BENT = "bent"


def _kp_mid(kps: Keypoints, i: int, j: int, min_conf: float) -> Optional[Tuple[float, float]]:
    """Midpoint of a left/right keypoint pair, using whichever side(s) are confident."""
    pts = [kps[k] for k in (i, j) if k < len(kps) and kps[k][2] >= min_conf]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def classify_posture(kps: Keypoints, bbox: BBox,
                     min_conf: Optional[float] = None) -> Optional[str]:
    """Keypoints → coarse body state via pure geometry (§3: torso angle + hip/knee
    ratios — no learned head). Pixel coords, y grows downward. Returns None when the
    torso isn't readable (occlusion) and the bbox shape doesn't decide it either.

      lying    — torso far from vertical (or a clearly wider-than-tall box, faceless)
      bent     — torso tilted (bent-at-counter / reaching)
      sitting  — upright torso but thigh folded toward horizontal, or the legs'
                 vertical span collapsed relative to the torso
      standing — upright torso, legs extended
    """
    min_conf = cfg.pose_min_kp_conf if min_conf is None else min_conf
    shoulder = _kp_mid(kps, KP_L_SHOULDER, KP_R_SHOULDER, min_conf)
    hip = _kp_mid(kps, KP_L_HIP, KP_R_HIP, min_conf)
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    if shoulder is None or hip is None:
        # Torso unreadable — a clearly horizontal box is still a lying read.
        return POSTURE_LYING if bw / bh >= 1.5 else None

    # Torso angle from vertical-down: 0° upright, 90° horizontal, >90° inverted.
    dx, dy = hip[0] - shoulder[0], hip[1] - shoulder[1]
    torso_deg = math.degrees(math.atan2(abs(dx), dy))
    if torso_deg > 65:
        return POSTURE_LYING
    if torso_deg > 35:
        return POSTURE_BENT

    knee = _kp_mid(kps, KP_L_KNEE, KP_R_KNEE, min_conf)
    if knee is not None:
        tdx, tdy = knee[0] - hip[0], knee[1] - hip[1]
        thigh_deg = math.degrees(math.atan2(abs(tdx), tdy))
        if thigh_deg > 50:  # thigh folded toward horizontal (side-view sitting)
            return POSTURE_SITTING
    # Front-view sitting: thighs point at the camera (thigh reads near-vertical), but
    # the legs' vertical span collapses relative to the torso.
    ankle = _kp_mid(kps, KP_L_ANKLE, KP_R_ANKLE, min_conf)
    if ankle is not None:
        torso_v = hip[1] - shoulder[1]
        legs_v = ankle[1] - hip[1]
        if torso_v > 0 and legs_v < 1.1 * torso_v:
            return POSTURE_SITTING
    return POSTURE_STANDING


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / float(area_a + area_b - inter or 1)


def match_poses_to_tracks(poses: List[Tuple[Keypoints, BBox]],
                          tracks: List[DetectedTrack],
                          min_iou: float = 0.3) -> dict:
    """track_id → keypoints, by best bbox IoU (pose detections carry no track ids —
    the detect model owns ByteTrack; pose is a separate same-cadence pass). Greedy: each
    pose goes to its best-overlapping track; a track keeps its best-IoU pose."""
    best: dict = {}
    for kps, pbox in poses:
        top_tid, top_iou = None, min_iou
        for t in tracks:
            iou = bbox_iou(pbox, t.bbox)
            if iou > top_iou:
                top_tid, top_iou = t.track_id, iou
        if top_tid is not None and top_iou >= best.get(top_tid, (None, 0.0))[1]:
            best[top_tid] = (kps, top_iou)
    return {tid: kps for tid, (kps, _iou) in best.items()}


def assign_faces_to_tracks(faces: List[Tuple[List[float], BBox]],
                           tracks: List[DetectedTrack]) -> dict:
    """track_id → (embedding, face bbox), mutually exclusive.

    Person boxes OVERLAP whenever two people share a room, so "the largest face
    inside my track box" (the per-crop rule) routinely embeds the OTHER person's
    closer/larger face — the direct mechanism behind A-labelled-as-B swaps. Ownership
    here is containment-based instead: a face belongs to the track whose bbox contains
    the face's center, and when several boxes do, the TIGHTEST (smallest-area) one wins
    — the tighter person box is the body the face is actually attached to. One face per
    track (largest wins — the near/frontal face over a background one); a face whose
    center no track contains is dropped."""
    def _area(b: BBox) -> int:
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    claims: dict = {}  # track_id -> (face_area, emb, fbox)
    for emb, fbox in faces:
        cx, cy = (fbox[0] + fbox[2]) / 2.0, (fbox[1] + fbox[3]) / 2.0
        owner, owner_area = None, 0
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                a = _area(t.bbox)
                if owner is None or a < owner_area:
                    owner, owner_area = t.track_id, a
        if owner is None:
            continue
        fa = _area(fbox)
        if owner not in claims or fa > claims[owner][0]:
            claims[owner] = (fa, emb, fbox)
    return {tid: (emb, fbox) for tid, (_fa, emb, fbox) in claims.items()}


class NullPoseEngine:
    backend = "null"

    def detect(self, frame) -> List[Tuple[Keypoints, BBox]]:
        return []


class _UltralyticsPose:
    """yolov8n-pose on the SAME ultralytics runtime as detect (§3). Plain predict —
    no tracking (ByteTrack ids stay owned by the detect model; poses are matched to
    tracks by IoU). Lazy; raises on import failure so the factory falls back."""

    backend = "ultralytics"

    def __init__(self) -> None:
        from ultralytics import YOLO  # type: ignore
        model = os.getenv("VISION_POSE_MODEL", "yolov8n-pose.pt")
        self.device = os.getenv("VISION_DEVICE", "cpu")
        self.imgsz = int(os.getenv("VISION_IMGSZ", "640"))
        self._yolo = YOLO(model)
        print(f"[vision] ultralytics pose {model} on {self.device}", flush=True)

    def detect(self, frame) -> List[Tuple[Keypoints, BBox]]:
        res = self._yolo.predict(frame, imgsz=self.imgsz, conf=cfg.person_conf,
                                 device=self.device, verbose=False)
        out: List[Tuple[Keypoints, BBox]] = []
        if not res:
            return out
        kobj = getattr(res[0], "keypoints", None)
        boxes = getattr(res[0], "boxes", None)
        if kobj is None or boxes is None or getattr(kobj, "data", None) is None:
            return out
        for kps, xyxy in zip(kobj.data.tolist(), boxes.xyxy.tolist()):
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            out.append(([(float(x), float(y), float(c)) for x, y, c in kps],
                        (x1, y1, x2, y2)))
        return out


def make_pose_engine():
    if cfg.pose_backend == "ultralytics":
        try:
            return _UltralyticsPose()
        except Exception as e:  # noqa: BLE001 — never crash the box on a model issue
            print(f"[vision] ultralytics pose unavailable ({e}); falling back to null pose", flush=True)
    return NullPoseEngine()


# ── identity-grade quality gate (config.py "quality gate" block) ─────────────
def face_sharpness(frame, bbox: BBox) -> Optional[float]:
    """Laplacian variance of the gray face crop — the blur gauge. Downscale-only to
    ≤112px longest side so the number is comparable across face sizes (upscaling a
    small crop would smooth it into an automatic fail, double-counting size, which
    the size floor already owns). None = can't judge (no cv2 / empty crop)."""
    cv2 = _cv2()
    if cv2 is None or frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0
    crop = _resize_max(frame[y1:y2, x1:x2], 112)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_quality_reason(face, frame, *, enroll: bool = False) -> Optional[str]:
    """Why this detected face must ABSTAIN from identity decisions (None = fit).
    ArcFace embeddings of low-confidence / turned-away / blurry faces are noise —
    measured live 2026-07-07: a member's own enroll burst self-agreed at cos ~0.2,
    and noise-mean centroids made two members swap names. Size is deliberately NOT
    checked here for the live path (the camera applies `face_min_px` after its
    high-res rescue); the enroll path checks size in `assess_enroll` (no rescue).
    `face` is an insightface Face (det_score / pose / bbox); pose is skipped
    gracefully when the landmark model didn't supply it."""
    if enroll:
        min_det, max_yaw, max_pitch, min_sharp = (
            cfg.enroll_min_det_score, cfg.enroll_max_yaw,
            cfg.enroll_max_pitch, cfg.enroll_min_sharpness)
    else:
        min_det, max_yaw, max_pitch, min_sharp = (
            cfg.face_min_det_score, cfg.face_max_yaw,
            cfg.face_max_pitch, cfg.face_min_sharpness)
    det = getattr(face, "det_score", None)
    if det is not None and float(det) < min_det:
        return "low_confidence"
    pose = getattr(face, "pose", None)  # (pitch, yaw, roll) degrees, landmark_3d_68
    if pose is not None and len(pose) >= 2:
        pitch, yaw = float(pose[0]), float(pose[1])
        if abs(yaw) > max_yaw or abs(pitch) > max_pitch:
            return "off_angle"
    if min_sharp > 0:
        sharp = face_sharpness(frame, face.bbox)
        if sharp is not None and sharp < min_sharp:
            return "blurry"
    return None


# ── face engines ─────────────────────────────────────────────────────────────
class NullFaceEngine:
    backend = "null"

    def embed(self, frame, bbox: BBox) -> Optional[List[float]]:
        return None

    def embed_face(self, frame, bbox: BBox) -> Optional[Tuple[List[float], BBox]]:
        return None

    def faces(self, frame) -> List[Tuple[List[float], BBox]]:
        return []


class _InsightFaceEngine:
    """SCRFD detect + ArcFace (buffalo_l) embed → 512-d normalised vector (§4.2).
    Lazy; raises on import failure so the factory falls back to null."""

    backend = "insightface"

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis  # type: ignore
        providers = os.getenv("VISION_ORT_PROVIDERS", "CPUExecutionProvider").split(",")
        self._app = FaceAnalysis(name=os.getenv("VISION_FACE_MODEL", "buffalo_l"), providers=providers)
        # det_size / det_thresh are the far-face levers (see config.py): a larger square
        # finds smaller/more-distant faces, a lower threshold keeps weak ones. Size is
        # NOT gated here — the camera owns `face_min_px` (applied after its high-res
        # rescue, so a small substream face can still be saved by the main stream);
        # this engine gates the intrinsic quality bars (det/pose/sharpness) instead.
        det = max(160, cfg.face_det_size)
        self._app.prepare(ctx_id=0 if "CPU" not in providers[0] else -1,
                          det_size=(det, det), det_thresh=cfg.face_det_thresh)
        print(f"[vision] insightface buffalo_l providers={providers} "
              f"det_size={det} det_thresh={cfg.face_det_thresh} "
              f"quality(det≥{cfg.face_min_det_score} yaw≤{cfg.face_max_yaw} "
              f"pitch≤{cfg.face_max_pitch} sharp≥{cfg.face_min_sharpness})", flush=True)

    def embed(self, frame, bbox: BBox) -> Optional[List[float]]:
        hit = self.embed_face(frame, bbox)
        return hit[0] if hit else None

    def embed_face(self, frame, bbox: BBox) -> Optional[Tuple[Optional[List[float]], BBox]]:
        """(embedding, face bbox in FRAME coords) for the largest face inside the
        track box — the face bbox is what the review-card crop centers on. A face
        failing the identity-quality gate answers (None, bbox): FOUND but abstaining
        (no embedding leaves the engine), so the caller can still trigger the
        high-res rescue — det/sharpness genuinely improve on the main-stream frame
        (measured live: det 0.49 sub → 0.77 main on the same face). None means no
        face at all."""
        x1, y1, x2, y2 = bbox
        ox, oy = max(0, x1), max(0, y1)
        crop = frame[oy:y2, ox:x2]
        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        faces = self._app.get(crop)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            return None
        fx1, fy1, fx2, fy2 = (int(v) for v in face.bbox)
        fbox = (ox + fx1, oy + fy1, ox + fx2, oy + fy2)
        if face_quality_reason(face, crop) is not None:
            return None, fbox  # found-but-abstained (blur/angle/confidence)
        return emb.tolist(), fbox

    def faces(self, frame) -> List[Tuple[Optional[List[float]], BBox]]:
        """EVERY face in the frame — identity-grade ones with their embedding,
        quality-abstained ones as (None, bbox) so the multi-person pass can still
        assign them to tracks and trigger the high-res rescue. An abstained face
        can neither claim a member nor seed a cluster (no embedding exists)."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        out: List[Tuple[Optional[List[float]], BBox]] = []
        for face in self._app.get(frame):
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            if face_quality_reason(face, frame) is not None:
                out.append((None, (x1, y1, x2, y2)))
            else:
                out.append((emb.tolist(), (x1, y1, x2, y2)))
        return out

    def assess_enroll(self, frame) -> Tuple[Optional[List[float]], Optional[str]]:
        """(embedding, None) for an enrollment-grade photo, else (None, reason).
        Enrollment is ground truth so the STRICT gate applies, and the reason is
        actionable (the guided flow shows it verbatim): exactly one face of real
        size — a second comparable face means we can't know whose profile this
        feeds (the 2026-07-06 ledger shows three people's enrolls interleaving
        within the same minute); then size / confidence / pose / sharpness."""
        faces = self._app.get(frame)
        if not faces:
            return None, "no_face"
        px = lambda f: max(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1])  # noqa: E731
        faces.sort(key=px, reverse=True)
        main = faces[0]
        main_px = px(main)
        if any(px(f) >= max(40.0, main_px * 0.5) for f in faces[1:]):
            return None, "multiple_faces"
        if main_px < cfg.enroll_min_px:
            return None, "too_small"
        reason = face_quality_reason(main, frame, enroll=True)
        if reason is not None:
            return None, reason
        emb = getattr(main, "normed_embedding", None)
        if emb is None:
            return None, "no_face"
        return emb.tolist(), None


# ── factories (graceful fallback to null) ─────────────────────────────────────
def make_detector():
    if cfg.backend == "ultralytics":
        try:
            return _UltralyticsDetector()
        except Exception as e:  # noqa: BLE001 — never crash the box on a model/ROCm issue
            print(f"[vision] ultralytics unavailable ({e}); falling back to null detector", flush=True)
    return NullDetector()


def make_face_engine():
    if cfg.face_backend == "insightface":
        try:
            return _InsightFaceEngine()
        except Exception as e:  # noqa: BLE001
            print(f"[vision] insightface unavailable ({e}); falling back to null face engine", flush=True)
    return NullFaceEngine()


# ── enrollment (Face ID) ──────────────────────────────────────────────────────
_shared_face_engine = None


def _get_shared_face_engine():
    global _shared_face_engine
    if _shared_face_engine is None:
        _shared_face_engine = make_face_engine()
    return _shared_face_engine


def _stub_embedding(buf: bytes) -> List[float]:
    """Deterministic hash-expanded, L2-normalised vector from the image bytes — same
    posture as the speaker-service `stub` backend. Lets enrollment + the gallery + the
    dashboard Face-ID UI be built/tested with NO face model. NOT facial: identical
    bytes match, nothing else. Never use for real recognition (face_backend=null means
    the live pipeline produces no embeddings, so it never matches against these)."""
    seed = hashlib.sha256(buf).digest()
    vals: List[float] = []
    counter = 0
    while len(vals) < cfg.face_dim:
        block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for j in range(0, len(block), 4):
            if len(vals) >= cfg.face_dim:
                break
            u = struct.unpack(">I", block[j:j + 4])[0]
            vals.append(u / 0xFFFFFFFF * 2.0 - 1.0)
        counter += 1
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def annotate_face_in_thumb(jpeg: bytes, centroid: List[float]) -> Optional[List[float]]:
    """Find THE face a stored guest thumbnail is about. Legacy thumbs are full-person
    (sometimes multi-person) crops, so the review card can't tell which face the
    question refers to — re-detect every face in the thumb, embed each, and pick the
    one closest to the cluster centroid. Returns a normalized [x,y,w,h] box within
    the thumb, [] when the engine ran but found no face (cache "nothing to ring"),
    or None when no real engine/cv2 is available (leave uncached, try again later)."""
    eng = _get_shared_face_engine()
    if getattr(eng, "backend", "null") == "null":
        return None
    frame = decode_jpeg(jpeg)
    if frame is None:
        return None
    hits = [h for h in eng.faces(frame) if h[0] is not None]  # need embeddings to pick WHOSE face
    if not hits:
        return []
    h, w = frame.shape[:2]
    best = max(hits, key=lambda hit: sum(a * b for a, b in zip(hit[0], centroid)))
    x1, y1, x2, y2 = best[1]
    return [round(x1 / w, 4), round(y1 / h, 4),
            round((x2 - x1) / w, 4), round((y2 - y1) / h, 4)]


def enroll_embedding(jpeg: bytes) -> Tuple[Optional[List[float]], Optional[str]]:
    """Image bytes → (embedding, None) when the photo is enrollment-grade, else
    (None, reason). Enrollment samples are the anchors every identity decision
    leans on, so the strict gate runs here (exactly one face, size, confidence,
    pose, sharpness — see `assess_enroll`); the route maps `reason` to an
    actionable 422 the guided flow shows the user. The null build keeps the
    deterministic stub (plumbing tests, no face model installed)."""
    eng = _get_shared_face_engine()
    if getattr(eng, "backend", "null") != "null":
        frame = decode_jpeg(jpeg)
        if frame is None:
            return None, "no_face"
        return eng.assess_enroll(frame)
    return _stub_embedding(jpeg), None
