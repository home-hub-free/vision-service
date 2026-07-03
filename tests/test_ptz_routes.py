"""PTZ / imaging / camctl routes — per-capability degrade + error mapping.

Fake workers + fake ONVIF clients injected into state.workers; no lifespan (no
supervisor/janitor threads), no network. Mirrors the plan's acceptance: unknown
camera 404s, a non-ONVIF camera 409s with {ptz:false}, a fixed (no-PTZ) camera
keeps imaging but refuses moves, and the null/offline path never crashes.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.onvif import OnvifError
from app.state import workers

client = TestClient(app)


class FakeOnvif:
    def __init__(self, ptz=True, imaging=True, events=True):
        self.caps = {"ptz": ptz, "imaging": imaging, "events": events}
        self.calls = []

    def capabilities_cached(self):
        return self.caps

    def capabilities(self, now=None):
        return self.caps

    def get_status(self):
        return {"x": 0.1, "y": -0.2, "move_status": "IDLE"}

    def get_presets(self):
        return [{"token": "1", "name": "hub-home", "x": 0.0, "y": -0.5}]

    def goto_preset(self, token):
        self.calls.append(("goto", token))

    def set_preset(self, name):
        self.calls.append(("set_preset", name))
        return "2"

    def remove_preset(self, token):
        self.calls.append(("remove", token))

    def move_timed(self, vx, vy, ttl_s):
        self.calls.append(("move", vx, vy, ttl_s))
        return min(ttl_s, 2.0)

    def stop(self):
        self.calls.append(("stop",))

    def get_imaging(self):
        return {"brightness": 50.0, "contrast": 50.0}

    def set_imaging(self, updates):
        self.calls.append(("set_imaging", updates))
        return {"brightness": 70.0, "contrast": 50.0}


class FakeWorker:
    def __init__(self, cam_id, onvif):
        class Cam:
            id = cam_id
            zone = "entrance"
        self.cam = Cam()
        self._onvif = onvif  # get_onvif() uses this directly (already "built")


@pytest.fixture(autouse=True)
def cams():
    ptz_cam = FakeOnvif()
    fixed_cam = FakeOnvif(ptz=False)  # a C110: events+imaging, no PTZ
    workers.clear()
    workers["mc200"] = FakeWorker("mc200", ptz_cam)
    workers["c110"] = FakeWorker("c110", fixed_cam)
    workers["esp"] = FakeWorker("esp", None)  # MJPEG node — not ONVIF at all
    yield {"ptz": ptz_cam, "fixed": fixed_cam}
    workers.clear()


def test_unknown_camera_404s():
    assert client.get("/ptz/nope/status").status_code == 404
    assert client.get("/camctl/nope").status_code == 404


def test_non_onvif_camera_409s_with_ptz_false():
    r = client.post("/ptz/esp/move", json={"vx": 1, "vy": 0, "ttl_ms": 300})
    assert r.status_code == 409
    assert r.json()["detail"]["ptz"] is False


def test_fixed_camera_refuses_moves_but_keeps_imaging(cams):
    r = client.post("/ptz/c110/move", json={"vx": 0.5, "vy": 0, "ttl_ms": 300})
    assert r.status_code == 409
    r = client.get("/imaging/c110")
    assert r.status_code == 200
    assert r.json()["imaging"]["brightness"] == 50.0


def test_move_and_stop(cams):
    r = client.post("/ptz/mc200/move", json={"vx": 0.5, "vy": -0.5, "ttl_ms": 400})
    assert r.status_code == 200
    assert r.json()["ttl_s"] == pytest.approx(0.4)
    assert cams["ptz"].calls[-1] == ("move", 0.5, -0.5, pytest.approx(0.4))
    client.post("/ptz/mc200/stop")
    assert cams["ptz"].calls[-1] == ("stop",)


def test_preset_lifecycle(cams):
    r = client.get("/ptz/mc200/presets")
    assert r.json()["presets"][0]["name"] == "hub-home"
    r = client.post("/ptz/mc200/preset", json={"name": "door"})
    assert r.status_code == 200 and r.json()["token"] == "2"
    r = client.post("/ptz/mc200/goto", json={"token": "2"})
    assert r.status_code == 200
    r = client.delete("/ptz/mc200/preset/2")
    assert r.status_code == 200
    assert [c[0] for c in cams["ptz"].calls] == ["set_preset", "goto", "remove"]


def test_blank_preset_name_400s():
    assert client.post("/ptz/mc200/preset", json={"name": "  "}).status_code == 400


def test_imaging_set_requires_a_field_and_forwards(cams):
    assert client.post("/imaging/mc200", json={}).status_code == 400
    r = client.post("/imaging/mc200", json={"brightness": 70})
    assert r.status_code == 200
    assert cams["ptz"].calls[-1] == ("set_imaging", {"brightness": 70.0, "saturation": None,
                                                     "contrast": None, "sharpness": None,
                                                     "ir_cut": None})


def test_unreachable_camera_maps_to_503(cams):
    def boom(*a, **k):
        raise OnvifError("unreachable: timeout")
    cams["ptz"].get_status = boom
    assert client.get("/ptz/mc200/status").status_code == 503


def test_camera_fault_maps_to_502(cams):
    def fault(*a, **k):
        raise OnvifError("bad args (InvalidPosition)", fault=True)
    cams["ptz"].goto_preset = fault
    assert client.post("/ptz/mc200/goto", json={"token": "9"}).status_code == 502


def test_camctl_aggregates_for_the_tile(cams):
    r = client.get("/camctl/mc200")
    assert r.status_code == 200
    body = r.json()
    assert body["onvif"] == {"ptz": True, "imaging": True, "events": True}
    assert body["reachable"] is True
    assert body["presets"][0]["token"] == "1"
    assert body["imaging"]["brightness"] == 50.0
    assert body["status"]["move_status"] == "IDLE"


def test_camctl_fixed_camera_has_no_presets_block(cams):
    body = client.get("/camctl/c110").json()
    assert body["onvif"]["ptz"] is False
    assert "presets" not in body and "status" not in body
    assert body["imaging"]["brightness"] == 50.0


def test_camctl_non_onvif_camera_is_null():
    body = client.get("/camctl/esp").json()
    assert body == {"cam_id": "esp", "zone": "entrance", "onvif": None, "reachable": False}
