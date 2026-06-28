"""Hub client — read-only roster access, the only hub coupling the service has.

The vision-service reads two things from the hub, both with the internal-caller
`X-Hub-Service-Token` header (CLAUDE.md "Service token"; same pattern the voice box
uses for /auth/users):

  * `GET /get-devices` → filter `deviceCategory == "camera"` → the camera roster,
    each carrying the `stream` capability block (§3.3) so we can build the MJPEG URL.
  * `GET /auth/users` → the person roster (names) so a resolved `users.id` gets a
    display name in the identity envelope.

It also validates a dashboard bearer token via `GET /auth/me` for Face-ID enrollment
(the enrolling user must be authenticated — same as voiceprint enroll).

If the hub is down the control plane is unaffected and so is this service's media
path; only roster refresh + enrollment auth pause. The hub never calls us.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import List, Optional
from urllib.parse import urlparse

from .config import cfg


def _get(path: str, headers: Optional[dict] = None, timeout: float = 4.0):
    req = urllib.request.Request(cfg.hub_url + path, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _svc_headers() -> dict:
    return {"X-Hub-Service-Token": cfg.hub_service_token} if cfg.hub_service_token else {}


class Camera:
    """A camera as the roster sees it, with the stream URL already built.

    `stream_url_override` / `snapshot_url_override` let a non-declaring source
    (the §2 VISION_STATIC_CAMERAS escape hatch) carry a pre-built, any-scheme URL
    (http MJPEG or rtsp://) that bypasses the ip+port+path builder."""

    def __init__(
        self,
        raw: dict,
        stream_url_override: Optional[str] = None,
        snapshot_url_override: Optional[str] = None,
    ) -> None:
        self.id = str(raw.get("id"))
        self.name = raw.get("name")
        self.zone = raw.get("zone") or "_"
        self.ip = raw.get("ip")
        self.stream = raw.get("stream") or {}
        self.fw_version = raw.get("fwVersion")
        self._stream_url_override = stream_url_override
        self._snapshot_url_override = snapshot_url_override

    @property
    def stream_url(self) -> Optional[str]:
        if self._stream_url_override:
            return self._stream_url_override
        if not self.ip or not self.stream.get("path"):
            return None
        port = self.stream.get("port", 81)
        return f"http://{self.ip}:{port}{self.stream['path']}"

    @property
    def snapshot_url(self) -> Optional[str]:
        if self._snapshot_url_override:
            return self._snapshot_url_override
        snap = self.stream.get("snapshot")
        if not self.ip or not snap:
            return None
        port = self.stream.get("port", 81)
        return f"http://{self.ip}:{port}{snap}"

    def __repr__(self) -> str:
        return f"<Camera {self.id} zone={self.zone} {self.stream_url}>"


def parse_static_cameras(spec: str) -> List[Camera]:
    """Parse VISION_STATIC_CAMERAS into Cameras with a pre-built stream URL.

    Format: a comma-list of `id@zone@url` (CAMERA_BRINGUP_PLAN §2). The URL is taken
    verbatim (split capped at 3 fields, so credentials/paths containing '@' survive),
    so any MJPEG-http or rtsp:// source works. Malformed entries are skipped with a
    log — never raise into the supervisor's hot poll path."""
    cams: List[Camera] = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split("@", 2)
        if len(parts) < 3 or not parts[0].strip() or not parts[2].strip():
            print(f"[vision] ignoring malformed VISION_STATIC_CAMERAS entry {entry!r} "
                  "(want id@zone@url)", flush=True)
            continue
        cam_id, zone, url = parts[0].strip(), parts[1].strip(), parts[2].strip()
        raw = {"id": cam_id, "name": "camera", "zone": zone or "_",
               "ip": urlparse(url).hostname}
        cams.append(Camera(raw, stream_url_override=url))
    return cams


def fetch_cameras() -> List[Camera]:
    cams: List[Camera] = []
    try:
        devices = _get("/get-devices", _svc_headers())
        cams = [Camera(d) for d in devices if d.get("deviceCategory") == "camera"]
        cams = [c for c in cams if c.stream_url]  # only cams we can actually pull
    except (urllib.error.URLError, OSError, ValueError) as e:
        # A hub outage must not disable the §2 escape hatch — fall through to static.
        print(f"[vision] roster fetch failed: {e}", flush=True)
    # §2 escape hatch: augment the roster with non-declaring sources (IP/RTSP/webcam)
    # for the image-quality go/no-go before firmware exists. Roster wins on id clash.
    have = {c.id for c in cams}
    for sc in parse_static_cameras(cfg.static_cameras):
        if sc.id not in have:
            cams.append(sc)
            have.add(sc.id)
    return cams


def fetch_users() -> dict:
    """users.id -> display name, for naming a resolved identity."""
    try:
        users = _get("/auth/users", _svc_headers())
    except (urllib.error.URLError, OSError, ValueError):
        return {}
    out = {}
    for u in users if isinstance(users, list) else users.get("users", []):
        uid = u.get("id")
        if uid:
            out[str(uid)] = u.get("displayName") or u.get("name") or u.get("username")
    return out


def user_from_token(authorization: Optional[str]) -> dict:
    """Validate a dashboard bearer token against the hub; return the user (enroll auth)."""
    from fastapi import HTTPException

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    try:
        data = _get("/auth/me", {"Authorization": authorization}, timeout=3.0)
    except Exception:
        raise HTTPException(status_code=401, detail="token validation failed")
    user = (data or {}).get("user")
    if not user or not user.get("id"):
        raise HTTPException(status_code=401, detail="unauthorized")
    return user


def require_user(authorization: Optional[str]) -> dict:
    """Gate the admin labeling actions (name/promote a person) behind a valid hub
    session — the household model is flat (any logged-in member administers their own
    home; the seed user is the de-facto admin), so an authenticated user IS the admin.
    Forward-compatible: if the hub ever adds a per-user `admin` flag, enforce it here."""
    user = user_from_token(authorization)
    # if not user.get("admin"): raise HTTPException(status_code=403, detail="admin only")
    return user
