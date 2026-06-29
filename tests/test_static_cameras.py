"""VISION_STATIC_CAMERAS escape hatch (CAMERA_BRINGUP_PLAN §2) — pure parsing, no net.

The §2 go/no-go validates the whole box pipeline against any MJPEG-over-HTTP source
BEFORE the ESP32-CAM firmware declares. These tests pin the `id@zone@url` parsing + the
roster-augment behaviour so the hatch can't silently break. The parser is deliberately
scheme-agnostic (it preserves an `rtsp://` URL verbatim, forward-compat for a future
RTSP reader), even though the frame reader pulls HTTP MJPEG only today (see app/mjpeg.py).
"""
from app.hub_client import Camera, fetch_cameras, parse_static_cameras


def test_parses_basic_entry():
    cams = parse_static_cameras("lab@sala@http://192.168.1.50:81/stream")
    assert len(cams) == 1
    c = cams[0]
    assert c.id == "lab"
    assert c.zone == "sala"
    assert c.stream_url == "http://192.168.1.50:81/stream"
    assert c.ip == "192.168.1.50"


def test_parses_multiple_and_skips_blanks():
    cams = parse_static_cameras(
        " a@z1@http://h1/s , , b@z2@http://h2:8080/mjpeg ,"
    )
    assert [c.id for c in cams] == ["a", "b"]
    assert cams[1].stream_url == "http://h2:8080/mjpeg"


def test_url_with_at_sign_survives():
    # rtsp credentials contain '@' — split must cap at 3 fields, not break the URL.
    cams = parse_static_cameras("cam@hall@rtsp://user:pass@10.0.0.9:554/Streaming")
    assert len(cams) == 1
    assert cams[0].stream_url == "rtsp://user:pass@10.0.0.9:554/Streaming"
    assert cams[0].zone == "hall"


def test_malformed_entries_are_skipped_not_raised():
    # missing url / missing fields → skipped, never raises into the supervisor poll.
    assert parse_static_cameras("onlyid") == []
    assert parse_static_cameras("id@zone@") == []
    assert parse_static_cameras("@zone@http://h/s") == []
    assert parse_static_cameras("") == []


def test_empty_zone_defaults_to_placeholder():
    cams = parse_static_cameras("c@@http://h/s")
    assert len(cams) == 1
    assert cams[0].zone == "_"


def test_override_takes_precedence_over_builder():
    # An override URL bypasses the ip+port+path builder entirely.
    c = Camera({"id": "x", "ip": "1.2.3.4", "stream": {"path": "/stream", "port": 81}},
               stream_url_override="rtsp://1.2.3.4/live")
    assert c.stream_url == "rtsp://1.2.3.4/live"


def test_dual_stream_two_urls():
    # "<detect-substream> <record-mainstream>": reader/detect on the first, record the 2nd.
    cams = parse_static_cameras(
        "patio@garden@rtsp://u:p@h:554/stream2 rtsp://u:p@h:554/stream1")
    assert len(cams) == 1
    c = cams[0]
    assert c.stream_url == "rtsp://u:p@h:554/stream2"
    assert c.record_url == "rtsp://u:p@h:554/stream1"
    assert c.zone == "garden"


def test_single_url_has_no_record_url():
    cams = parse_static_cameras("lab@sala@http://192.168.1.50:81/stream")
    assert cams[0].record_url is None


def test_record_url_from_override_and_stream_block():
    assert Camera({"id": "x"}, record_url_override="rtsp://h/main").record_url == "rtsp://h/main"
    assert Camera({"id": "y", "stream": {"recordUrl": "rtsp://h/decl"}}).record_url == "rtsp://h/decl"


def test_fetch_cameras_augments_with_static_when_hub_down(monkeypatch):
    from app import hub_client

    # Simulate an unreachable hub: roster fetch raises → only static cams come back.
    def boom(*a, **k):
        raise OSError("hub down")

    monkeypatch.setattr(hub_client, "_get", boom)
    monkeypatch.setattr(hub_client.cfg, "static_cameras",
                        "lab@sala@http://192.168.1.50:81/stream")
    cams = fetch_cameras()
    assert [c.id for c in cams] == ["lab"]


def test_fetch_cameras_roster_wins_on_id_collision(monkeypatch):
    from app import hub_client

    def fake_get(path, headers=None, timeout=4.0):
        return [{"id": "lab", "deviceCategory": "camera", "ip": "10.0.0.2",
                 "zone": "kitchen", "stream": {"path": "/stream", "port": 81}}]

    monkeypatch.setattr(hub_client, "_get", fake_get)
    monkeypatch.setattr(hub_client.cfg, "static_cameras",
                        "lab@sala@http://192.168.1.50:81/stream")
    cams = fetch_cameras()
    assert len(cams) == 1
    # The declared roster camera (kitchen) wins; the static stand-in is dropped.
    assert cams[0].zone == "kitchen"
    assert cams[0].stream_url == "http://10.0.0.2:81/stream"
