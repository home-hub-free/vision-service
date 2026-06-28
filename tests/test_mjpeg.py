"""MJPEG frame parsing — SOI..EOI extraction from a multipart byte stream."""
from app.mjpeg import iter_jpeg_frames

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"


def _reader(blob: bytes):
    """A read(n) over a fixed blob, like response.read."""
    state = {"i": 0}

    def read(n: int) -> bytes:
        i = state["i"]
        chunk = blob[i:i + n]
        state["i"] = i + len(chunk)
        return chunk

    return read


def test_extracts_two_frames_across_chunk_boundaries():
    f1 = SOI + b"AAAA" + EOI
    f2 = SOI + b"BBBBBB" + EOI
    stream = b"--boundary\r\nContent-Type: image/jpeg\r\n\r\n" + f1 + b"\r\n--boundary\r\n\r\n" + f2 + b"\r\n"
    frames = list(iter_jpeg_frames(_reader(stream), chunk=7))  # tiny chunk → cross boundaries
    assert frames == [f1, f2]


def test_drops_pre_soi_noise_and_flushes_trailing_frame():
    f1 = SOI + b"X" + EOI
    stream = b"garbage-headers-no-soi" + f1
    frames = list(iter_jpeg_frames(_reader(stream), chunk=4))
    assert frames == [f1]
