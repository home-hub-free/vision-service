"""Recording & storage (§9) — a first-class deliverable, moved off the hub.

Reuses the proven `camera.recorder.ts` ffmpeg recipe (decoded/relayed JPEG frames →
H.264, `tee` into segmented mp4 archive + HLS for low-CPU dashboard playback), now in
the vision-service where the decoded frames + per-zone identity already live (so clips
can be tagged with who was present — the event index does that join, §9.4).

Modes (§9.2, per-camera):
  * "continuous" — always recording (entrances/security).
  * "gated"      — record only while a person is present, with a PRE-ROLL ring buffer
                   so the clip starts ~preroll_seconds BEFORE the trigger.
  * "hybrid"     — gated + pre-roll (the recommended default); set a camera to
                   "continuous" to opt it in.
  * "off"        — no recording.

Encoder (§9.1 / §11.4 DECISION): default CPU `libx264 veryfast`; switch to AMD
`vaapi`/`amf` GPU encode via VISION_REC_ENCODER once measured against GPU contention.
The encoder is a config knob, never a code change.
"""
from __future__ import annotations

import collections
import os
import shutil
import subprocess
import threading
import time
from typing import Deque, Optional, Tuple

from .config import cfg
from .index_db import EventIndex
from .rtsp import is_rtsp


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def rtsp_copy_args(url: str, seg_out: str, hls_out: str,
                   transport: str = "tcp", segment_seconds: int = 300) -> list:
    """ffmpeg argv: pull an RTSP MAIN stream and record it by **codec-copy** (no decode,
    no re-encode → full quality at near-zero CPU) into segmented mp4 + HLS via `tee`. This
    is the dual-stream recording half — detection runs on the cheap substream in camera.py
    (DECISIONS #1). Codec-copy can't pre-roll a ring buffer, so this path is continuous
    (the gated/pre-roll JPEG-pipe path stays for MJPEG cams / when no record_url is set).
    Audio IS recorded: Tapo/Mercusys cams emit pcm_alaw (8kHz mono), which mp4 can't
    hold by copy, so audio alone is transcoded to AAC — negligible CPU next to the
    video copy; the `0:a:0?` optional map keeps audio-less cameras working."""
    tee = (
        f"[f=segment:strftime=1:segment_time={segment_seconds}:reset_timestamps=1]{seg_out}"
        f"|[f=hls:hls_time=2:hls_list_size=20:"
        f"hls_flags=delete_segments+append_list+independent_segments:hls_segment_type=fmp4]{hls_out}"
    )
    return ["ffmpeg", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", transport, "-i", url,
            "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "32k",
            "-f", "tee", tee]


def _encode_args(encoder: str, fps: int) -> list:
    """Map the encoder knob to ffmpeg flags. libx264 is today's recipe; vaapi/amf are
    the GPU-encode decision (§9.1) — wired but unproven, MEASURE before switching."""
    gop = max(2, fps * 2)
    if encoder == "vaapi":
        return ["-vaapi_device", "/dev/dri/renderD128", "-vf", "format=nv12,hwupload",
                "-c:v", "h264_vaapi", "-g", str(gop)]
    if encoder == "amf":
        return ["-c:v", "h264_amf", "-usage", "lowlatency", "-g", str(gop)]
    return ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-g", str(gop), "-keyint_min", str(gop),
            "-sc_threshold", "0"]


class Recorder:
    """One per camera. `write_frame(jpeg)` feeds it; `on_presence(present)` drives the
    gated start/stop. Thread-safe for a single producer (the camera worker)."""

    def __init__(self, cam_id: str, zone: str, index: EventIndex, mode: Optional[str] = None,
                 record_url: Optional[str] = None) -> None:
        self.cam_id = cam_id
        self.zone = zone
        self.index = index
        self.mode = mode or cfg.rec_mode_default
        # Dual-stream: when a full-quality MAIN rtsp:// stream is supplied, record it by
        # codec-copy (continuous) and ignore the fed JPEG frames; the reader/detector runs
        # on the substream. Otherwise: classic JPEG-pipe recording (gated/pre-roll capable).
        self.record_url = record_url
        self._passthrough = is_rtsp(record_url)
        if self._passthrough and self.mode in ("gated", "hybrid"):
            self.mode = "continuous"  # codec-copy can't pre-roll a decoded-frame ring
        self.fps = cfg.rec_fps
        self._proc: Optional[subprocess.Popen] = None
        self._seg_id: Optional[int] = None
        self._lock = threading.Lock()
        # Pre-roll ring buffer (gated): keep ~preroll_seconds of recent frames.
        self._ring: Deque[Tuple[float, bytes]] = collections.deque(maxlen=max(1, int(cfg.preroll_seconds * self.fps)))
        self._last_present = 0.0
        self._tail_s = cfg.preroll_seconds  # keep rolling this long after empty
        self.rec_dir = os.path.join(cfg.rec_dir, cam_id)
        self.hls_dir = os.path.join(cfg.hls_dir, cam_id)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._passthrough:
            if self.mode != "off":
                self._open_passthrough()
            return
        if self.mode == "continuous":
            self._open()

    def _open_passthrough(self) -> None:
        """Open the codec-copy recorder against the MAIN rtsp:// stream (continuous)."""
        with self._lock:
            if self._proc is not None or not ffmpeg_available():
                return
            os.makedirs(self.rec_dir, exist_ok=True)
            os.makedirs(self.hls_dir, exist_ok=True)
            seg_out = os.path.join(self.rec_dir, "%Y%m%d-%H%M%S.mp4")
            hls_out = os.path.join(self.hls_dir, "live.m3u8")
            args = rtsp_copy_args(self.record_url, seg_out, hls_out,
                                  cfg.rtsp_transport, cfg.segment_seconds)
            try:
                # No stdin: ffmpeg pulls the RTSP stream itself (we don't feed frames).
                self._proc = subprocess.Popen(args, stdin=subprocess.DEVNULL,
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:  # noqa: BLE001
                print(f"[vision] recorder({self.cam_id}) rtsp-copy start failed: {e}", flush=True)
                self._proc = None
                return
            self._seg_id = self.index.open_segment(self.cam_id, self.zone, self.rec_dir)

    def stop(self) -> None:
        self._close()

    def clear_ring(self) -> None:
        """Drop the pre-roll ring (privacy pause): frames buffered before the pause
        would be flushed into the NEXT clip's head on resume — after a privacy gap
        they're stale context from another moment, so start the next clip clean."""
        self._ring.clear()

    def _open(self) -> None:
        with self._lock:
            if self._proc is not None or self.mode == "off" or not ffmpeg_available():
                return
            os.makedirs(self.rec_dir, exist_ok=True)
            os.makedirs(self.hls_dir, exist_ok=True)
            seg_out = os.path.join(self.rec_dir, "%Y%m%d-%H%M%S.mp4")
            hls_out = os.path.join(self.hls_dir, "live.m3u8")
            tee = (
                f"[f=segment:strftime=1:segment_time={cfg.segment_seconds}:reset_timestamps=1]{seg_out}"
                f"|[f=hls:hls_time=2:hls_list_size=20:"
                f"hls_flags=delete_segments+append_list+independent_segments:hls_segment_type=fmp4]{hls_out}"
            )
            args = ["ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "mjpeg", "-fflags", "+genpts", "-r", str(self.fps), "-i", "pipe:0",
                    *_encode_args(cfg.rec_encoder, self.fps),
                    "-movflags", "+faststart", "-map", "0:v",
                    "-f", "tee", tee]
            try:
                self._proc = subprocess.Popen(args, stdin=subprocess.PIPE,
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:  # noqa: BLE001
                print(f"[vision] recorder({self.cam_id}) ffmpeg start failed: {e}", flush=True)
                self._proc = None
                return
            self._seg_id = self.index.open_segment(self.cam_id, self.zone, self.rec_dir)
            # Flush the pre-roll ring so the clip starts before the trigger (§9.2).
            for _ts, jpeg in list(self._ring):
                self._feed(jpeg)

    def _close(self) -> None:
        with self._lock:
            if self._proc is None:
                return
            proc, self._proc = self._proc, None
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
            # REAP, don't just signal: without wait() every closed encoder lingers
            # as a zombie for the service's lifetime, and a hung ffmpeg would keep
            # recording after "stop" — privacy mode depends on stop meaning STOPPED,
            # so escalate to SIGKILL if it ignores the TERM (normal exit is <1s).
            try:
                proc.wait(timeout=5.0)
            except Exception:  # noqa: BLE001 — TimeoutExpired or already-gone races
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except Exception:  # noqa: BLE001
                    pass
            if self._seg_id is not None:
                self.index.close_segment(self._seg_id)
                self._seg_id = None

    # ── frame intake ──────────────────────────────────────────────────────────
    def write_frame(self, jpeg: bytes) -> None:
        if self._passthrough:
            return  # ffmpeg pulls the main stream itself; fed frames are unused
        now = time.time()
        self._ring.append((now, jpeg))
        if self._proc is not None:
            self._feed(jpeg)

    def _feed(self, jpeg: bytes) -> None:
        p = self._proc
        if p is None or p.stdin is None:
            return
        try:
            p.stdin.write(jpeg)
        except (BrokenPipeError, ValueError):
            self._close()

    # ── gated driver ──────────────────────────────────────────────────────────
    def on_presence(self, present: bool) -> None:
        """Called from the occupancy loop. In gated/hybrid mode, presence opens the
        recorder (pre-roll flushes in) and a sustained absence closes it after a tail."""
        if self.mode in ("off", "continuous"):
            return
        now = time.time()
        if present:
            self._last_present = now
            if self._proc is None:
                self._open()

    def tick(self) -> None:
        """Periodic: close a gated recording once nobody's been present for a tail."""
        if self.mode in ("off", "continuous") or self._proc is None:
            return
        if time.time() - self._last_present > self._tail_s:
            self._close()
