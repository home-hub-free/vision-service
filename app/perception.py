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


# ── face engines ─────────────────────────────────────────────────────────────
class NullFaceEngine:
    backend = "null"

    def embed(self, frame, bbox: BBox) -> Optional[List[float]]:
        return None


class _InsightFaceEngine:
    """SCRFD detect + ArcFace (buffalo_l) embed → 512-d normalised vector (§4.2).
    Lazy; raises on import failure so the factory falls back to null."""

    backend = "insightface"

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis  # type: ignore
        providers = os.getenv("VISION_ORT_PROVIDERS", "CPUExecutionProvider").split(",")
        self._app = FaceAnalysis(name=os.getenv("VISION_FACE_MODEL", "buffalo_l"), providers=providers)
        self._app.prepare(ctx_id=0 if "CPU" not in providers[0] else -1, det_size=(640, 640))
        print(f"[vision] insightface buffalo_l providers={providers}", flush=True)

    def embed(self, frame, bbox: BBox) -> Optional[List[float]]:
        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        faces = self._app.get(crop)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = getattr(face, "normed_embedding", None)
        return emb.tolist() if emb is not None else None


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
