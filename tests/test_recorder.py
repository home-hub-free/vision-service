"""Recorder close semantics — stop must mean STOPPED (privacy depends on it).

Found live 2026-07-06: `_close()` only signalled ffmpeg (terminate) and never
wait()ed — every closed encoder lingered as a zombie, and a hung ffmpeg would
have kept recording after a privacy toggle claimed it stopped.
"""
import subprocess

from app.index_db import EventIndex
from app.recorder import Recorder


class FakeProc:
    """Stands in for the ffmpeg Popen: records the signal/reap sequence."""

    def __init__(self, hang: bool = False) -> None:
        self.hang = hang
        self.calls = []
        self.stdin = None

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")

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
