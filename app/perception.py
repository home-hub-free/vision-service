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


def draw_overlay(frame, tracks: List[DetectedTrack], labels: dict) -> bytes:
    """Annotate a frame with boxes + names → JPEG bytes for the dashboard's annotated
    view (§6). `labels[track_id]` is the resolved display string. cv2-only; the null
    build never calls this (it relays raw frames)."""
    cv2 = _cv2()
    if cv2 is None or frame is None:
        return b""
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        name = labels.get(t.track_id, "person")
        known = name not in ("person", "unknown")
        color = (80, 200, 120) if known else (200, 200, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
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
        # finds smaller/more-distant faces, a lower threshold keeps weak ones. min_px gates
        # out faces too small to embed cleanly (0 = keep everything).
        det = max(160, cfg.face_det_size)
        self._min_px = max(0, cfg.face_min_px)
        self._app.prepare(ctx_id=0 if "CPU" not in providers[0] else -1,
                          det_size=(det, det), det_thresh=cfg.face_det_thresh)
        print(f"[vision] insightface buffalo_l providers={providers} "
              f"det_size={det} det_thresh={cfg.face_det_thresh} min_px={self._min_px}", flush=True)

    def _too_small(self, box) -> bool:
        """True when a detected face bbox is smaller than the min_px gate (longest side)."""
        if self._min_px <= 0:
            return False
        return max(box[2] - box[0], box[3] - box[1]) < self._min_px

    def embed(self, frame, bbox: BBox) -> Optional[List[float]]:
        hit = self.embed_face(frame, bbox)
        return hit[0] if hit else None

    def embed_face(self, frame, bbox: BBox) -> Optional[Tuple[List[float], BBox]]:
        """(embedding, face bbox in FRAME coords) for the largest face inside the
        track box — the face bbox is what the review-card crop centers on."""
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
        if emb is None or self._too_small(face.bbox):
            return None  # no face, or too small/low-detail to embed cleanly
        fx1, fy1, fx2, fy2 = (int(v) for v in face.bbox)
        return emb.tolist(), (ox + fx1, oy + fy1, ox + fx2, oy + fy2)

    def faces(self, frame) -> List[Tuple[List[float], BBox]]:
        """EVERY face in the frame with its embedding — used to re-locate the right
        face inside a stored (legacy full-person) thumbnail."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        out: List[Tuple[List[float], BBox]] = []
        for face in self._app.get(frame):
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            out.append((emb.tolist(), (x1, y1, x2, y2)))
        return out


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
    hits = eng.faces(frame)
    if not hits:
        return []
    h, w = frame.shape[:2]
    best = max(hits, key=lambda hit: sum(a * b for a, b in zip(hit[0], centroid)))
    x1, y1, x2, y2 = best[1]
    return [round(x1 / w, 4), round(y1 / h, 4),
            round((x2 - x1) / w, 4), round((y2 - y1) / h, 4)]


def enroll_embedding(jpeg: bytes) -> Optional[List[float]]:
    """Image bytes → a face embedding to store in the gallery. Uses the real face
    engine (largest face in the frame) when one is installed; otherwise the
    deterministic stub so the plumbing works in the null build."""
    eng = _get_shared_face_engine()
    if getattr(eng, "backend", "null") != "null":
        frame = decode_jpeg(jpeg)
        if frame is not None:
            h, w = frame.shape[:2]
            emb = eng.embed(frame, (0, 0, w, h))
            if emb is not None:
                return emb
        return None  # real engine found no face — let the caller report "no face"
    return _stub_embedding(jpeg)
