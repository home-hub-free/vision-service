"""RTSP transport helpers — pure logic, no network (mirrors test_static_cameras).

The cv2/ffmpeg streaming path is HW-validated (needs a real camera); these tests pin the
scheme detection, credential redaction, and the codec-copy ffmpeg argv so a typo can't
silently break RTSP cameras.
"""
import pytest

from app.recorder import rtsp_copy_args
from app.rtsp import is_rtsp, iter_rtsp_frames, redact_url


def test_is_rtsp_detects_scheme():
    assert is_rtsp("rtsp://1.2.3.4/live")
    assert is_rtsp("rtsps://1.2.3.4/live")
    assert is_rtsp("RTSP://1.2.3.4/live")  # case-insensitive
    assert not is_rtsp("http://1.2.3.4:81/stream")
    assert not is_rtsp("https://1.2.3.4/stream")
    assert not is_rtsp(None)
    assert not is_rtsp("")


def test_redact_url_hides_credentials():
    out = redact_url("rtsp://user:pass@10.0.0.9:554/Streaming/Channels/101")
    assert "user" not in out and "pass" not in out
    assert "10.0.0.9:554" in out
    assert out.endswith("/Streaming/Channels/101")


def test_redact_url_passthrough_when_no_credentials():
    url = "rtsp://10.0.0.9:554/stream1"
    assert redact_url(url) == url


def test_rtsp_copy_args_is_codec_copy_no_reencode():
    args = rtsp_copy_args("rtsp://u:p@h:554/stream1", "/rec/%Y.mp4", "/hls/live.m3u8",
                          transport="tcp", segment_seconds=300)
    assert args[0] == "ffmpeg"
    # codec-copy = full quality, no decode/re-encode (the dual-stream recording win)
    assert "-c:v" in args and "copy" in args
    assert "-rtsp_transport" in args and "tcp" in args
    assert "rtsp://u:p@h:554/stream1" in args
    # both archive (segment) + playback (hls) sinks via a single tee
    assert "-f" in args and "tee" in args
    joined = " ".join(args)
    assert "/rec/%Y.mp4" in joined and "/hls/live.m3u8" in joined


def test_iter_rtsp_frames_raises_without_opencv():
    try:
        import cv2  # noqa: F401
        pytest.skip("opencv installed; the cv2-missing branch isn't exercised here")
    except Exception:
        pass
    # No cv2 → raise (not silently return), so the worker's reconnect/backoff handles it.
    with pytest.raises(RuntimeError):
        next(iter_rtsp_frames("rtsp://1.2.3.4/live", lambda: False))
