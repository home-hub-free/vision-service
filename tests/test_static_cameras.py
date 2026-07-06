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


def test_static_camera_adopts_dashboard_zone_from_hub_record(monkeypatch):
    # The hub knows the camera (we proxy-declared it) but holds no stream block —
    # the static entry adopts the persisted, dashboard-assigned zone while keeping
    # its local URLs. This is what makes an IP cam zone-configurable like any device.
    from app import hub_client

    def fake_get(path, headers=None, timeout=4.0):
        return [{"id": "mc200", "deviceCategory": "camera", "zone": "sala"}]

    declared = []
    monkeypatch.setattr(hub_client, "_get", fake_get)
    monkeypatch.setattr(hub_client, "declare_camera", declared.append)
    monkeypatch.setattr(hub_client.cfg, "static_cameras",
                        "mc200@entrance@rtsp://u:p@h:554/stream2 rtsp://u:p@h:554/stream1")
    cams = fetch_cameras()
    assert len(cams) == 1
    c = cams[0]
    assert c.zone == "sala"  # dashboard zone wins over the .env fallback
    assert c.stream_url == "rtsp://u:p@h:554/stream2"  # URLs stay local
    assert c.record_url == "rtsp://u:p@h:554/stream1"
    assert [d.id for d in declared] == ["mc200"]  # heartbeat re-declare each sync


def test_static_camera_env_zone_is_fallback_when_hub_zone_unset(monkeypatch):
    from app import hub_client

    def fake_get(path, headers=None, timeout=4.0):
        return [{"id": "mc200", "deviceCategory": "camera"}]  # declared, zone not yet assigned

    monkeypatch.setattr(hub_client, "_get", fake_get)
    monkeypatch.setattr(hub_client, "declare_camera", lambda cam: None)
    monkeypatch.setattr(hub_client.cfg, "static_cameras",
                        "mc200@sala@rtsp://u:p@h:554/stream2")
    cams = fetch_cameras()
    assert cams[0].zone == "sala"


def test_static_camera_declared_even_before_hub_knows_it(monkeypatch):
    # First sync: hub has no record yet → still declare (that's what creates the card).
    from app import hub_client

    declared = []
    monkeypatch.setattr(hub_client, "_get", lambda *a, **k: [])
    monkeypatch.setattr(hub_client, "declare_camera", declared.append)
    monkeypatch.setattr(hub_client.cfg, "static_cameras", "cam1@sala@rtsp://u:p@h/s")
    cams = fetch_cameras()
    assert [c.id for c in cams] == ["cam1"]
    assert [d.id for d in declared] == ["cam1"]


def test_static_camera_not_declared_when_hub_down(monkeypatch):
    from app import hub_client

    def boom(*a, **k):
        raise OSError("hub down")

    declared = []
    monkeypatch.setattr(hub_client, "_get", boom)
    monkeypatch.setattr(hub_client, "declare_camera", declared.append)
    monkeypatch.setattr(hub_client.cfg, "static_cameras", "cam1@sala@rtsp://u:p@h/s")
    cams = fetch_cameras()
    assert [c.id for c in cams] == ["cam1"]  # escape hatch still works
    assert declared == []  # but no declare attempt into a dead hub


def test_fetch_cameras_includes_any_device_with_stream_block(monkeypatch):
    from app import hub_client

    # A camera-equipped voice satellite declares `stream` under its own category —
    # eligibility is "declares a pullable stream", not deviceCategory == "camera".
    def fake_get(path, headers=None, timeout=4.0):
        return [
            {"id": "cam1", "deviceCategory": "camera", "ip": "10.0.0.2",
             "zone": "kitchen", "stream": {"path": "/stream", "port": 81}},
            {"id": "sat1", "deviceCategory": "voice-satellite", "ip": "10.0.0.9",
             "zone": "oficina",
             "stream": {"path": "/stream", "port": 81, "snapshot": "/capture",
                        "res": "VGA", "fps": 1}},
            # Camera-less satellite: no stream block → not a vision source.
            {"id": "sat2", "deviceCategory": "voice-satellite", "ip": "10.0.0.10",
             "zone": "sala"},
        ]

    monkeypatch.setattr(hub_client, "_get", fake_get)
    monkeypatch.setattr(hub_client.cfg, "static_cameras", "")
    cams = fetch_cameras()
    assert sorted(c.id for c in cams) == ["cam1", "sat1"]
    sat = next(c for c in cams if c.id == "sat1")
    assert sat.stream_url == "http://10.0.0.9:81/stream"
    assert sat.snapshot_url == "http://10.0.0.9:81/capture"
    assert sat.zone == "oficina"


def test_context_capable_rtsp_yes_mjpeg_satellite_no():
    # Full-body context (T0 speed / T1 posture / T2a hints) only from real IP cams
    # (RTSP detect stream). ESP32 satellite/standalone cams (HTTP-MJPEG) are kept for
    # face ID only — their tracks must stay identity-only.
    rtsp = parse_static_cameras("mc200@sala@rtsp://u:p@1.2.3.4:554/stream2")[0]
    assert rtsp.context_capable is True
    sat = Camera({"id": "6c0057858428", "zone": "oficina", "ip": "1.2.3.5",
                  "stream": {"path": "/stream", "port": 81}})
    assert sat.context_capable is False
    assert Camera({"id": "nostream"}).context_capable is False
