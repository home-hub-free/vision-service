"""Per-file footage indexing — the segments table mirrors the actual .mp4 files.

WHY (found live 2026-07-06, first real use of playback): the recorder indexed one
row per RECORDER RUN pointing at the camera's recordings DIRECTORY, while ffmpeg
(`-f segment`, strftime names) writes 5-minute FILES — so the clip route (which
serves one file, and guards with isfile) 404'd on EVERY row, and a SIGKILLed
service left "recording…" rows open forever (an open row overlaps every future
day's window, smearing the timeline). Files ARE the archive — the retention
janitor already prunes per-file — so the index now follows disk:

  * `scan_files` lists a camera's finished mp4s (start from the strftime name,
    end from mtime — codec-copy writes continuously, so last-write ≈ end);
  * `sync_camera` reconciles the table (insert new files, purge the legacy
    directory-shaped rows). Idempotent, cheap (one listdir), run on demand by
    the /recordings routes and periodically by the janitor.

A file younger than SETTLE_S is still being written: segmented mp4 gets its moov
atom only on close, so a growing file is unplayable — skip it until it settles
(the live view covers "now"; the newest chunk appears within a segment length).
"""
from __future__ import annotations

import os
import re
import struct
import time
from typing import List, Optional, Set, Tuple

from .config import cfg
from .index_db import EventIndex

FNAME = re.compile(r"^(\d{8})-(\d{6})\.mp4$")
SETTLE_S = 15.0


def has_moov(path: str) -> bool:
    """Whether the mp4 has a top-level moov atom — i.e. is playable at all. A hard
    death (SIGKILL, crash, power loss) strands the recorder's in-progress chunk
    with NO moov, forever: listing it would put a permanently dead span on the
    timeline (seen live: mc200's July-4 final chunk). Cheap: walks atom headers
    only (a handful of 8-byte reads), never the payload."""
    try:
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            off = 0
            while off + 8 <= size:
                f.seek(off)
                hdr = f.read(8)
                if len(hdr) < 8:
                    return False
                alen, typ = struct.unpack(">I4s", hdr)
                if typ == b"moov":
                    return True
                if alen == 1:  # 64-bit atom length
                    ext = f.read(8)
                    if len(ext) < 8:
                        return False
                    alen = struct.unpack(">Q", ext)[0]
                elif alen == 0:  # "to EOF"
                    alen = size - off
                if alen < 8:
                    return False  # malformed — don't loop forever
                off += alen
    except OSError:
        return False
    return False


def _scan(rec_dir: str, now: Optional[float] = None
          ) -> Tuple[List[Tuple[str, float, float]], Set[str]]:
    """(finished entries, still-settling paths) for one camera's recordings dir.
    Missing dir / unparsable names are skipped, never raised — this runs inside
    request handlers and the janitor. Settled files without a moov atom are
    EXCLUDED (unplayable stranded chunks) so the index purge can drop their rows;
    settling files are reported separately so the purge never races a chunk that
    is only *about* to become playable."""
    entries: List[Tuple[str, float, float]] = []
    pending: Set[str] = set()
    try:
        names = os.listdir(rec_dir)
    except OSError:
        return entries, pending
    now = time.time() if now is None else now
    for name in sorted(names):
        m = FNAME.match(name)
        if not m:
            continue
        path = os.path.join(rec_dir, name)
        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            continue  # raced the janitor's prune
        if now - mtime < SETTLE_S:
            pending.add(path)
            continue  # still growing — no moov atom yet, not playable
        try:
            start = time.mktime(time.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"))
        except (ValueError, OverflowError):
            continue
        if not has_moov(path):
            continue  # stranded by a hard death — permanently unplayable
        entries.append((path, start, max(mtime, start)))
    return entries, pending


def scan_files(rec_dir: str, now: Optional[float] = None) -> List[Tuple[str, float, float]]:
    """(path, start_ts, end_ts) for every finished, playable segment file."""
    return _scan(rec_dir, now)[0]


def sync_camera(index: EventIndex, cam_id: str, zone: str,
                rec_root: Optional[str] = None) -> int:
    """Reconcile one camera's segment rows with its files; returns rows added."""
    rec_dir = os.path.join(rec_root or cfg.rec_dir, cam_id)
    entries, pending = _scan(rec_dir)
    return index.sync_file_segments(cam_id, zone or "_", rec_dir, entries,
                                    protect=pending)
