"""Preview-frame pregeneration (thumbs.py) — one decode pass warms a segment's
whole 15s-grid frame set into the same cache the /thumb route serves."""
import os
import shutil
import subprocess

import pytest

from app.config import cfg
from app.index_db import EventIndex
from app.thumbs import HEIGHT, STEP_S, is_pregenerated, pregenerate, pregenerate_missing

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "thumb_dir", str(tmp_path / "thumbs"))
    clip = str(tmp_path / "20260708-120000.mp4")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=160x120:rate=5:duration=40",
         "-pix_fmt", "yuv420p", clip],
        check=True, timeout=30)
    return {"clip": clip, "thumb_dir": str(tmp_path / "thumbs")}


@needs_ffmpeg
def test_pregenerate_emits_the_grid_and_marks_done(env):
    assert not is_pregenerated("cam1", 7)
    wrote = pregenerate("cam1", 7, env["clip"])
    assert wrote == 3  # 40s clip → frames at t=0, 15, 30
    names = sorted(os.listdir(os.path.join(env["thumb_dir"], "cam1")))
    assert names == [f"7-{t}-{HEIGHT}.jpg" for t in (0, 15, 30)]
    assert is_pregenerated("cam1", 7)
    # Grid + naming match what the /thumb route computes for the same t.
    assert STEP_S == 15


@needs_ffmpeg
def test_pregenerate_missing_respects_budget_and_skips_warm(env, tmp_path):
    idx = EventIndex(str(tmp_path / "idx.db"))
    for i in range(3):
        sid = idx.open_segment("cam1", "sala", env["clip"], start_ts=1000.0 + i)
        idx.close_segment(sid, end_ts=1040.0 + i)
    assert pregenerate_missing(idx, ["cam1"], budget=2) == 2
    # All three rows share one file — but ids differ, so one set is still cold.
    assert pregenerate_missing(idx, ["cam1"], budget=5) == 1
    assert pregenerate_missing(idx, ["cam1"], budget=5) == 0  # everything warm


def test_pregenerate_survives_unreadable_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "thumb_dir", str(tmp_path / "thumbs"))
    bad = tmp_path / "garbage.mp4"
    bad.write_bytes(b"not a video")
    assert pregenerate("cam1", 9, str(bad)) == 0
    assert not is_pregenerated("cam1", 9)
