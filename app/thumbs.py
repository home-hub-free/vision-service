"""Preview-frame pregeneration — the scrub/filmstrip cache is warmed ahead of use.

The /thumb route extracts frames on demand, which is fine for a stray miss but
made *finding* things slow: scrubbing across a day touched dozens of never-seen
frames, each paying a cold ffmpeg spawn+seek serially (~200ms → "every 5-min
interval takes its own seconds"). Instead, as soon as a segment settles, ONE
ffmpeg pass decodes it and emits its whole 15s-grid frame set (~20 jpegs) into
the same disk cache the route serves — far cheaper than 20 individual seeks,
and every later hover/tap lands warm (~2ms).

Runs inside the retention janitor's thread on a per-tick budget so a backlog
never starves the box; `tools/pregen_thumbs.py` bulk-fills history once.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile

from .config import cfg

# Must match the /thumb route's grid + full-surface rendition (recordings.py).
STEP_S = 15
HEIGHT = 360

_FRAME = re.compile(r"f_(\d+)\.jpg$")


def is_pregenerated(cam_id: str, seg_id: int) -> bool:
    """The t=0 frame doubles as the 'set exists' marker — it's always emitted
    first, and a crashed partial run simply regenerates (idempotent)."""
    return os.path.isfile(os.path.join(cfg.thumb_dir, cam_id, f"{seg_id}-0-{HEIGHT}.jpg"))


def pregenerate(cam_id: str, seg_id: int, path: str) -> int:
    """Extract the full 15s-grid frame set for one segment in a single decode
    pass. Returns frames written (0 = ffmpeg failed / unreadable file)."""
    cache_dir = os.path.join(cfg.thumb_dir, cam_id)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f"pregen-{seg_id}-", dir=cache_dir)
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
             "-vf", f"fps=1/{STEP_S},scale=-2:{HEIGHT}", "-q:v", "7",
             os.path.join(tmp, "f_%04d.jpg")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        if res.returncode != 0:
            return 0
        wrote = 0
        for f in sorted(glob.glob(os.path.join(tmp, "f_*.jpg"))):
            m = _FRAME.search(f)
            if not m:
                continue
            t = (int(m.group(1)) - 1) * STEP_S  # ffmpeg numbers from 1; frame 1 = t0
            os.replace(f, os.path.join(cache_dir, f"{seg_id}-{t}-{HEIGHT}.jpg"))
            wrote += 1
        return wrote
    except subprocess.TimeoutExpired:
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def pregenerate_missing(index, cam_ids, budget: int = 6) -> int:
    """Warm at most `budget` segments' frame sets (newest first — that's where
    the reviewer lands). Called from the janitor tick; new segments arrive at
    ~1 per 5 min per camera, so any budget keeps up and the cap only matters
    while chewing a backlog. Returns segments processed."""
    done = 0
    for cam_id in cam_ids:
        if done >= budget:
            break
        for seg in index.recent_segments(cam_id, limit=200):
            if done >= budget:
                break
            if is_pregenerated(cam_id, seg["id"]):
                continue
            if not os.path.isfile(seg["file"]):
                continue
            if pregenerate(cam_id, seg["id"], seg["file"]):
                done += 1
    return done
