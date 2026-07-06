"""Per-file footage indexing (footage.py) — the segments table mirrors disk.

Found live 2026-07-06 on the playback UI's first real use: recorder-run rows
pointed at the recordings DIRECTORY (clip route 404'd on every row — isfile
guard) and SIGKILLed shutdowns left rows open forever (overlapping every future
day). These tests pin the replacement: rows come from the actual mp4 files.
"""
import os
import time

import pytest

from app.footage import SETTLE_S, scan_files, sync_camera
from app.index_db import EventIndex


def _mp4(rec_dir: str, name: str, mtime: float) -> str:
    path = os.path.join(rec_dir, name)
    with open(path, "wb") as fh:
        fh.write(b"\x00\x00\x00\x18ftypmp42" + b"v" * 32)
    os.utime(path, (mtime, mtime))
    return path


def _name(ts: float) -> str:
    return time.strftime("%Y%m%d-%H%M%S.mp4", time.localtime(ts))


# ── scan_files ────────────────────────────────────────────────────────────────

def test_scan_parses_start_from_name_and_end_from_mtime(tmp_path):
    rec = str(tmp_path)
    start = time.mktime(time.strptime("20260705-101500", "%Y%m%d-%H%M%S"))
    path = _mp4(rec, "20260705-101500.mp4", start + 300)
    entries = scan_files(rec)
    assert entries == [(path, pytest.approx(start), pytest.approx(start + 300))]


def test_scan_skips_growing_garbage_and_missing_dir(tmp_path):
    rec = str(tmp_path)
    now = time.time()
    _mp4(rec, _name(now - 60), now - 5)          # still growing (no moov yet)
    _mp4(rec, "not-a-segment.mp4", now - 3600)   # foreign name
    _mp4(rec, "99999999-999999.mp4", now - 3600) # strftime-shaped but not a date
    with open(os.path.join(rec, "20260705-101500.txt"), "w") as fh:
        fh.write("x")                            # wrong extension
    assert scan_files(rec) == []
    assert scan_files(os.path.join(rec, "nope")) == []  # missing dir → no raise


# ── sync_camera ───────────────────────────────────────────────────────────────

def test_sync_indexes_files_idempotently_and_purges_legacy_dir_rows(tmp_path):
    rec_root = str(tmp_path)
    rec_dir = os.path.join(rec_root, "cam1")
    os.makedirs(rec_dir)
    idx = EventIndex(str(tmp_path / "idx.db"))

    # The legacy shape: a recorder-run row pointing at the DIRECTORY, never closed —
    # including one aimed at an OLD directory from before a camera rename.
    legacy = idx.open_segment("cam1", "sala", rec_dir, start_ts=time.time() - 86400)
    renamed = idx.open_segment("cam1", "sala", os.path.join(rec_root, "cam1-oldname"),
                               start_ts=time.time() - 86400)

    old = time.time() - 7200
    _mp4(rec_dir, _name(old), old + 300)
    _mp4(rec_dir, _name(old + 300), old + 600)
    _mp4(rec_dir, _name(time.time() - 4), time.time() - 1)  # growing → not indexed

    assert sync_camera(idx, "cam1", "sala", rec_root=rec_root) == 2
    assert idx.segment_by_id(legacy) is None   # directory row self-healed away
    assert idx.segment_by_id(renamed) is None  # old-name directory row too

    segs = idx.segments_between("cam1", old - 10, old + 900)
    assert len(segs) == 2
    assert all(os.path.isfile(s["file"]) for s in segs)     # every row is playable
    assert all(s["end"] is not None for s in segs)          # nothing dangles open
    assert segs[0]["duration"] == pytest.approx(300, abs=2)

    # Idempotent: a second sync adds nothing and duplicates nothing.
    assert sync_camera(idx, "cam1", "sala", rec_root=rec_root) == 0
    assert len(idx.segments_between("cam1", old - 10, old + 900)) == 2

    # The settled third file appears once it stops growing.
    grown = os.path.join(rec_dir, sorted(os.listdir(rec_dir))[-1])
    past = time.time() - SETTLE_S - 1
    os.utime(grown, (past, past))
    assert sync_camera(idx, "cam1", "sala", rec_root=rec_root) == 1


def test_sync_scopes_to_its_camera(tmp_path):
    """cam2's rows must survive a cam1 sync (the purge is per-camera)."""
    rec_root = str(tmp_path)
    os.makedirs(os.path.join(rec_root, "cam1"))
    idx = EventIndex(str(tmp_path / "idx.db"))
    keep = idx.open_segment("cam2", "cocina", "/somewhere/else.mp4", start_ts=1.0)
    idx.close_segment(keep, end_ts=2.0)
    sync_camera(idx, "cam1", "sala", rec_root=rec_root)
    assert idx.segment_by_id(keep) is not None
