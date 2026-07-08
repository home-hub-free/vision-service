"""Recorder close semantics — stop must mean STOPPED (privacy depends on it).

Found live 2026-07-06: `_close()` only signalled ffmpeg (terminate) and never
wait()ed — every closed encoder lingered as a zombie, and a hung ffmpeg would
have kept recording after a privacy toggle claimed it stopped.

And the mirror bug, found live 2026-07-08: a passthrough (codec-copy) ffmpeg that
DIED was never noticed — stderr went to DEVNULL and nothing polled it, so the
entrance camera recorded nothing for 4 days. tick() now supervises it.
"""
import subprocess

from app.index_db import EventIndex
from app.recorder import Recorder, rtsp_copy_args


class FakeProc:
    """Stands in for the ffmpeg Popen: records the signal/reap sequence."""

    def __init__(self, hang: bool = False, rc=None) -> None:
        self.hang = hang
        self.rc = rc  # poll() result: None = still running
        self.calls = []
        self.stdin = None

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")

    def poll(self):
        self.calls.append("poll")
        return self.rc

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        # A hung encoder ignores the first (post-TERM) wait; SIGKILL always lands.
        if self.hang and "kill" not in self.calls:
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout)
        return 0


def _recorder(tmp_path) -> Recorder:
    index = EventIndex(str(tmp_path / "idx.db"))
    return Recorder("cam", "zone", index, mode="continuous")


def test_close_reaps_the_encoder(tmp_path):
    rec = _recorder(tmp_path)
    proc = FakeProc()
    rec._proc = proc
    rec.stop()
    assert "terminate" in proc.calls
    assert any(c[0] == "wait" for c in proc.calls if isinstance(c, tuple))
    assert "kill" not in proc.calls          # clean exit → no escalation
    assert rec._proc is None


def test_close_kill_escalates_a_hung_encoder(tmp_path):
    rec = _recorder(tmp_path)
    proc = FakeProc(hang=True)
    rec._proc = proc
    rec.stop()
    assert "terminate" in proc.calls
    assert "kill" in proc.calls              # TERM ignored → SIGKILL + reap
    assert proc.calls[-1] == ("wait", 2.0)
    assert rec._proc is None


# ── passthrough supervision ────────────────────────────────────────────────────

def _passthrough(tmp_path) -> Recorder:
    index = EventIndex(str(tmp_path / "idx.db"))
    rec = Recorder("cam", "zone", index, mode="hybrid",
                   record_url="rtsp://u:p@10.0.0.1:554/stream1")
    rec.rec_dir = str(tmp_path / "rec")
    return rec


def test_supervision_restarts_a_dead_passthrough(tmp_path, monkeypatch):
    rec = _passthrough(tmp_path)
    opens = []
    monkeypatch.setattr(rec, "_open_passthrough", lambda: opens.append(1))
    rec.start()                       # sets _want; "opens" the (stubbed) ffmpeg
    rec._proc = FakeProc(rc=1)        # ...which then dies
    rec.tick()
    assert rec._proc is None          # death noticed + reaped
    assert rec._retry_at > 0          # retry scheduled with backoff
    rec.tick()
    assert len(opens) == 1            # backoff window: no immediate relaunch
    rec._retry_at = 0.0
    rec.tick()
    assert len(opens) == 2            # backoff elapsed → relaunched


def test_supervision_never_resurrects_after_stop(tmp_path, monkeypatch):
    rec = _passthrough(tmp_path)
    opens = []
    monkeypatch.setattr(rec, "_open_passthrough", lambda: opens.append(1))
    rec.start()
    rec._proc = FakeProc(rc=0)
    rec.stop()                        # privacy / shutdown: intent is OFF
    rec._retry_at = 0.0
    rec.tick()
    assert opens == [1]               # only start()'s open — no retry after stop


def test_backoff_resets_after_a_healthy_run(tmp_path, monkeypatch):
    rec = _passthrough(tmp_path)
    monkeypatch.setattr(rec, "_open_passthrough", lambda: None)
    rec.start()
    rec._backoff = 120.0              # ladder was maxed by earlier flapping
    rec._opened_at = 0.0              # ran "forever" → healthy
    rec._proc = FakeProc(rc=255)
    rec.tick()
    assert rec._backoff == 10.0       # reset to RETRY_MIN_S


def test_segments_get_faststart():
    args = rtsp_copy_args("rtsp://x/1", "/r/%Y.mp4", "/h/live.m3u8")
    tee = args[-1]
    seg_slot = tee.split("|")[0]
    assert "segment_format_options=movflags=+faststart" in seg_slot
