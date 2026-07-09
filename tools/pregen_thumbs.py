#!/usr/bin/env python3
"""One-time backfill: pregenerate the scrub/filmstrip frame sets for every
indexed segment (the janitor keeps up with NEW segments; this chews history).

One ffmpeg decode pass per segment (~2-4s) — nice/ionice it:
  nice -n 15 ionice -c3 .venv/bin/python tools/pregen_thumbs.py
"""
from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import cfg  # noqa: E402
from app.thumbs import is_pregenerated, pregenerate  # noqa: E402


def main() -> int:
    db = sqlite3.connect(os.path.join(os.path.dirname(cfg.thumb_dir), "index.db"))
    rows = db.execute(
        "SELECT id, cam_id, file FROM segments WHERE file LIKE '%.mp4' ORDER BY start_ts DESC"
    ).fetchall()
    done = skipped = failed = 0
    for seg_id, cam_id, path in rows:
        if is_pregenerated(cam_id, seg_id):
            skipped += 1
            continue
        if not os.path.isfile(path):
            continue
        if pregenerate(cam_id, seg_id, path):
            done += 1
        else:
            failed += 1
            print(f"[pregen] FAILED seg {seg_id} ({path})", flush=True)
        if done and done % 50 == 0:
            print(f"[pregen] {done} done, {skipped} already warm, {failed} failed",
                  flush=True)
    print(f"[pregen] total: {done} generated, {skipped} already warm, {failed} failed")
    return 1 if failed and not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
