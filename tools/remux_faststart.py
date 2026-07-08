#!/usr/bin/env python3
"""One-time backlog fixer: move each recorded segment's moov atom to the front.

Segments written before 2026-07-08 have moov at the TAIL (ffmpeg mp4 default), so
the dashboard's <video> had to Range-hunt the index at the end of a ~40MB file
before it could start playing or seek — the "clips take forever to load" bug. New
segments get +faststart from the recorder (recorder.SEG_OPTS); this remuxes the
existing 5-day backlog in place. Codec-copy only — no re-encode, ~1s per file.

Invariants:
  * mtime is PRESERVED (footage.scan_files derives a segment's end_ts from it);
  * replacement is atomic (tmp file + os.replace) — a concurrently served clip
    keeps streaming from the old inode;
  * the youngest file per camera is skipped if still growing (< SETTLE_S old).

Usage: .venv/bin/python tools/remux_faststart.py [--dry-run] [rec_root]
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import time

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "recordings")
SETTLE_S = 60.0


def needs_faststart(path: str) -> bool:
    """True when the top-level moov atom sits AFTER mdat (i.e. not streamable)."""
    order = []
    try:
        with open(path, "rb") as f:
            while len(order) < 8:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                size, typ = struct.unpack(">I4s", hdr)
                if size == 1:
                    size = struct.unpack(">Q", f.read(8))[0]
                    f.seek(size - 16, 1)
                elif size == 0:
                    break
                else:
                    f.seek(size - 8, 1)
                order.append(typ)
    except OSError:
        return False
    if b"moov" not in order or b"mdat" not in order:
        return False  # malformed/still growing — leave it alone
    return order.index(b"moov") > order.index(b"mdat")


def remux(path: str) -> bool:
    st = os.stat(path)
    tmp = path + ".faststart.tmp"
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
         "-map", "0", "-c", "copy", "-movflags", "+faststart", "-f", "mp4", tmp],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode != 0 or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
        err = res.stderr.decode("utf-8", "replace").strip().splitlines()
        if err:
            print(f"[remux] ffmpeg: {err[-1]}", flush=True)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False
    os.utime(tmp, (st.st_atime, st.st_mtime))  # end_ts comes from mtime — keep it
    os.replace(tmp, path)
    return True


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry = "--dry-run" in sys.argv[1:]
    root = args[0] if args else DEFAULT_ROOT
    now = time.time()
    done = skipped = failed = 0
    for cam in sorted(os.listdir(root)):
        cam_dir = os.path.join(root, cam)
        if not os.path.isdir(cam_dir):
            continue
        for name in sorted(os.listdir(cam_dir)):
            if not name.endswith(".mp4"):
                continue
            path = os.path.join(cam_dir, name)
            try:
                if now - os.stat(path).st_mtime < SETTLE_S:
                    continue  # still being written
            except OSError:
                continue  # raced the retention janitor
            if not needs_faststart(path):
                skipped += 1
                continue
            if dry:
                print(f"would remux {path}")
                done += 1
                continue
            if remux(path):
                done += 1
            else:
                failed += 1
                print(f"[remux] FAILED: {path}", flush=True)
        print(f"[remux] {cam}: done so far {done}, already-ok {skipped}, failed {failed}",
              flush=True)
    print(f"[remux] total: remuxed {done}, already-ok {skipped}, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
