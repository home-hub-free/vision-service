"""ONVIF control seam — PTZ / imaging / events / clock for the fleet's RTSP cameras.

Ported from `tools/onvif_cli.py` (the stdlib-only reference client proven against the
Mercusys MC200, 2026-07-02) per docs/CAMERA_ONVIF_CONTROL_PLAN.md. Raw SOAP 1.2 +
WS-Security UsernameToken(PasswordDigest) — deliberately NO onvif-zeep/zeep: one small
module speaks SOAP for the whole box, and every body shape below was verified against
the real unit (fixtures in tests/fixtures/onvif/).

Design rules (plan §1):
  * Credentials come from the camera's existing `rtsp://user:pass@host` stream URL
    (VISION_STATIC_CAMERAS / the declared roster) — never a second secret store.
  * Not every camera is ONVIF (the ESP32-CAMs are MJPEG-only) and not every ONVIF
    camera is PTZ (the C110s are fixed). Everything control-ish gates on
    `capabilities()`, probed once per camera and cached; a transport error is NOT
    cached (camera may be rebooting), a SOAP fault is (the verb truly isn't there).
  * A continuous move must NEVER be left running: `move_timed` clamps the ttl and
    schedules the Stop on a daemon timer (belt) while callers keep /stop (braces).

Token trap (paid for 2026-07-02, don't rediscover): PTZ verbs want the PROFILE token
(`profile_1`); imaging verbs want the VIDEO-SOURCE token (`raw_vs1`). Mixing them 400s.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urlparse

from .config import cfg

SOAP_NS = "http://www.w3.org/2003/05/soap-envelope"
TDS = "http://www.onvif.org/ver10/device/wsdl"
TRT = "http://www.onvif.org/ver10/media/wsdl"
TPTZ = "http://www.onvif.org/ver20/ptz/wsdl"
TIMG = "http://www.onvif.org/ver20/imaging/wsdl"
TEV = "http://www.onvif.org/ver10/events/wsdl"
TT = "http://www.onvif.org/ver10/schema"
WSNT = "http://docs.oasis-open.org/wsn/b-2"
NSDECL = (
    f'xmlns:tptz="{TPTZ}" xmlns:tt="{TT}" xmlns:trt="{TRT}" '
    f'xmlns:timg="{TIMG}" xmlns:tev="{TEV}" xmlns:wsnt="{WSNT}"'
)


class OnvifError(RuntimeError):
    """A control call failed. `fault=True` means the camera answered with a SOAP
    Fault (the verb/feature isn't supported or the args were wrong — a definitive
    no); `fault=False` means transport (camera unreachable/timeout — retryable)."""

    def __init__(self, message: str, fault: bool = False, code: str = "") -> None:
        super().__init__(message)
        self.fault = fault
        self.code = code


# ── tiny regex XML helpers (same approach as the CLI; stdlib-only) ─────────────
def _tags(xml: str, name: str) -> List[str]:
    return re.findall(rf"<[\w:]*{name}[^>]*>(.*?)</[\w:]*{name}>", xml, re.S)


def _tag(xml: str, name: str) -> Optional[str]:
    vals = _tags(xml, name)
    return vals[0] if vals else None


def _text(v: Optional[str]) -> str:
    return re.sub(r"<[^>]+>", " ", v or "").strip()


def _num(v: Optional[str]) -> Optional[float]:
    try:
        return float(_text(v))
    except (TypeError, ValueError):
        return None


def _bool(v: Optional[str]) -> bool:
    return _text(v).lower() in ("true", "1")


class OnvifClient:
    def __init__(self, host: str, user: str, passwd: str,
                 port: Optional[int] = None, timeout: Optional[float] = None) -> None:
        self.host, self.user, self.passwd = host, user, passwd
        self.port = port or cfg.onvif_port
        self.timeout = timeout or cfg.onvif_timeout_s
        base = f"http://{host}:{self.port}/onvif"
        # The MC200 serves every service from one endpoint; per-service paths answer too.
        self.dev = f"{base}/device_service"
        self.media = f"{base}/media_service"
        self.ptz = f"{base}/ptz_service"
        self.img = f"{base}/imaging_service"
        self.events = f"{base}/events_service"

        self._lock = threading.Lock()  # guards the caches + the move auto-stop timer
        self._caps: Optional[Dict[str, bool]] = None
        self._caps_retry_at = 0.0  # backoff for probing an unreachable camera
        self._profile: Optional[str] = None
        self._vsrc: Optional[str] = None
        self._move_timer: Optional[threading.Timer] = None

    # ── transport ────────────────────────────────────────────────────────────
    def _wssec(self) -> str:
        nonce = os.urandom(16)
        created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        digest = base64.b64encode(
            hashlib.sha1(nonce + created.encode() + self.passwd.encode()).digest()
        ).decode()
        n64 = base64.b64encode(nonce).decode()
        return (
            '<s:Header><Security s:mustUnderstand="1" '
            'xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">'
            f"<UsernameToken><Username>{self.user}</Username>"
            '<Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">'
            f"{digest}</Password>"
            '<Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">'
            f"{n64}</Nonce>"
            '<Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">'
            f"{created}</Created></UsernameToken></Security></s:Header>"
        )

    def _post(self, url: str, envelope: bytes, timeout: float) -> str:
        """The raw HTTP POST — the single seam tests monkeypatch."""
        req = urllib.request.Request(
            url, data=envelope,
            headers={"Content-Type": "application/soap+xml; charset=utf-8"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode(errors="replace")

    def call(self, url: str, body: str, auth: bool = True,
             timeout: Optional[float] = None) -> str:
        """POST one SOAP body; return the response XML. Raises OnvifError on any
        transport failure or SOAP Fault (fault=True carries the ter: subcode)."""
        header = self._wssec() if auth else "<s:Header/>"
        env = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<s:Envelope xmlns:s="{SOAP_NS}" {NSDECL}>{header}<s:Body>{body}</s:Body></s:Envelope>'
        )
        try:
            xml = self._post(url, env.encode(), timeout or self.timeout)
        except urllib.error.HTTPError as e:
            # Some stacks put the SOAP Fault on a 4xx/5xx body — surface it as a fault.
            detail = ""
            try:
                detail = e.read().decode(errors="replace")
            except Exception:  # noqa: BLE001
                pass
            if "Fault" in detail:
                raise OnvifError(self._fault_text(detail), fault=True,
                                 code=self._fault_code(detail)) from e
            raise OnvifError(f"HTTP {e.code} from {url}") from e
        except Exception as e:  # noqa: BLE001 — URLError, timeout, socket
            raise OnvifError(f"unreachable: {e}") from e
        if re.search(r"<[\w-]*:?Fault[ >]", xml):
            raise OnvifError(self._fault_text(xml), fault=True, code=self._fault_code(xml))
        return xml

    @staticmethod
    def _fault_code(xml: str) -> str:
        m = re.search(r"ter:(\w+)", xml)
        return m.group(1) if m else ""

    @staticmethod
    def _fault_text(xml: str) -> str:
        reason = _tags(xml, "Text") or _tags(xml, "Reason")
        code = OnvifClient._fault_code(xml)
        txt = _text(reason[0]) if reason else "SOAP fault"
        return f"{txt or 'SOAP fault'}{f' ({code})' if code else ''}"

    # ── media tokens (the profile-vs-video-source trap, cached) ───────────────
    def profile_token(self) -> str:
        with self._lock:
            if self._profile:
                return self._profile
        xml = self.call(self.media, "<trt:GetProfiles/>")
        toks = re.findall(r'<[\w:]*Profiles[^>]*token="([^"]+)"', xml)
        with self._lock:
            self._profile = toks[0] if toks else "profile_1"
            return self._profile

    def snapshot_uri(self) -> Optional[str]:
        """Media GetSnapshotUri for the first (= main/high-res) profile, or None when
        the camera doesn't support it — the C110 answers with a SOAP fault (verified
        live 2026-07-06), so None routes the high-res sampler to its RTSP fallback."""
        try:
            xml = self.call(self.media,
                            f"<trt:GetSnapshotUri><trt:ProfileToken>{self.profile_token()}"
                            "</trt:ProfileToken></trt:GetSnapshotUri>")
        except OnvifError:
            return None
        m = re.search(r"<[\w:]*Uri>([^<]+)</", xml)
        return m.group(1).strip() if m else None

    def video_source_token(self) -> str:
        with self._lock:
            if self._vsrc:
                return self._vsrc
        xml = self.call(self.media, "<trt:GetVideoSources/>")
        toks = re.findall(r'<[\w:]*VideoSources[^>]*token="([^"]+)"', xml)
        with self._lock:
            self._vsrc = toks[0] if toks else "raw_vs1"
            return self._vsrc

    # ── capability probe (plan §1 rule 3 — degrade per-capability) ────────────
    def capabilities(self, now: Optional[float] = None) -> Dict[str, bool]:
        """{"ptz","imaging","events"} — probed once, cached. A SOAP fault caches
        False (the camera answered: not supported); a transport error raises and
        arms a retry backoff so an offline camera doesn't stall every caller."""
        now = now or time.time()
        with self._lock:
            if self._caps is not None:
                return self._caps
            if now < self._caps_retry_at:
                raise OnvifError("camera unreachable (probe backing off)")
        try:
            caps = {
                "ptz": self._probe_ptz(),
                "imaging": self._probe_imaging(),
                "events": self._probe_events(),
            }
        except OnvifError as e:
            if not e.fault:
                with self._lock:
                    self._caps_retry_at = now + 300.0  # camera down — retry in 5 min
            raise
        with self._lock:
            self._caps = caps
        return caps

    def capabilities_cached(self) -> Optional[Dict[str, bool]]:
        """The cached probe result (None = never probed successfully). Never touches
        the network — safe for status()/poll paths."""
        with self._lock:
            return dict(self._caps) if self._caps is not None else None

    def _probe_ptz(self) -> bool:
        try:
            xml = self.call(self.ptz, "<tptz:GetNodes/>")
        except OnvifError as e:
            if e.fault:
                return False
            raise
        return 'PTZNode' in xml and 'token="' in xml

    def _probe_imaging(self) -> bool:
        try:
            self.get_imaging()
        except OnvifError as e:
            if e.fault:
                return False
            raise
        return True

    def _probe_events(self) -> bool:
        try:
            xml = self.call(self.events, "<tev:GetEventProperties/>")
        except OnvifError as e:
            if e.fault:
                return False
            raise
        return "CellMotionDetector" in xml or "TopicSet" in xml

    # ── PTZ (profile token) ────────────────────────────────────────────────────
    def get_status(self) -> Dict[str, object]:
        xml = self.call(self.ptz, f"<tptz:GetStatus><tptz:ProfileToken>{self.profile_token()}"
                                  f"</tptz:ProfileToken></tptz:GetStatus>")
        pos = re.search(r'<[\w:]*PanTilt\s[^>]*x="([^"]+)"[^>]*y="([^"]+)"', xml)
        move = _text(_tag(_tag(xml, "MoveStatus") or "", "PanTilt") or _tag(xml, "MoveStatus"))
        return {
            "x": float(pos.group(1)) if pos else None,
            "y": float(pos.group(2)) if pos else None,
            "move_status": move or None,
        }

    def get_presets(self) -> List[Dict[str, object]]:
        xml = self.call(self.ptz, f"<tptz:GetPresets><tptz:ProfileToken>{self.profile_token()}"
                                  f"</tptz:ProfileToken></tptz:GetPresets>")
        presets: List[Dict[str, object]] = []
        for m in re.finditer(r'<[\w:]*Preset\b[^>]*token="([^"]+)"[^>]*>(.*?)</[\w:]*Preset>', xml, re.S):
            token, inner = m.group(1), m.group(2)
            pos = re.search(r'<[\w:]*PanTilt\s[^>]*x="([^"]+)"[^>]*y="([^"]+)"', inner)
            presets.append({
                "token": token,
                "name": _text(_tag(inner, "Name")) or token,
                "x": float(pos.group(1)) if pos else None,
                "y": float(pos.group(2)) if pos else None,
            })
        return presets

    def set_preset(self, name: str) -> str:
        """Save the CURRENT aim as a named preset; returns the camera's token.
        The MC200 caps at 8 presets — the camera faults past that (surfaced as-is)."""
        xml = self.call(self.ptz, f"<tptz:SetPreset><tptz:ProfileToken>{self.profile_token()}</tptz:ProfileToken>"
                                  f"<tptz:PresetName>{_esc(name)}</tptz:PresetName></tptz:SetPreset>")
        m = re.search(r"PresetToken>([^<]+)<", xml) or re.search(r'PresetToken="([^"]+)"', xml)
        if not m:
            raise OnvifError("SetPreset returned no token")
        return m.group(1)

    def remove_preset(self, token: str) -> None:
        self.call(self.ptz, f"<tptz:RemovePreset><tptz:ProfileToken>{self.profile_token()}</tptz:ProfileToken>"
                            f"<tptz:PresetToken>{_esc(token)}</tptz:PresetToken></tptz:RemovePreset>")

    def goto_preset(self, token: str) -> None:
        self.call(self.ptz, f"<tptz:GotoPreset><tptz:ProfileToken>{self.profile_token()}</tptz:ProfileToken>"
                            f"<tptz:PresetToken>{_esc(token)}</tptz:PresetToken></tptz:GotoPreset>")

    def absolute_move(self, x: float, y: float) -> None:
        x, y = _clamp(x), _clamp(y)
        self.call(self.ptz, f"<tptz:AbsoluteMove><tptz:ProfileToken>{self.profile_token()}</tptz:ProfileToken>"
                            f'<tptz:Position><tt:PanTilt x="{x}" y="{y}"/></tptz:Position></tptz:AbsoluteMove>')

    def continuous_move(self, vx: float, vy: float) -> None:
        vx, vy = _clamp(vx), _clamp(vy)
        self.call(self.ptz, f"<tptz:ContinuousMove><tptz:ProfileToken>{self.profile_token()}</tptz:ProfileToken>"
                            f'<tptz:Velocity><tt:PanTilt x="{vx}" y="{vy}"/></tptz:Velocity></tptz:ContinuousMove>')

    def stop(self) -> None:
        with self._lock:
            if self._move_timer is not None:
                self._move_timer.cancel()
                self._move_timer = None
        self.call(self.ptz, f"<tptz:Stop><tptz:ProfileToken>{self.profile_token()}</tptz:ProfileToken>"
                            f"<tptz:PanTilt>true</tptz:PanTilt></tptz:Stop>")

    def move_timed(self, vx: float, vy: float, ttl_s: float) -> float:
        """ContinuousMove that CANNOT be left running: ttl is clamped to
        (0, cfg.ptz_max_ttl_s] and a daemon timer fires Stop when it expires.
        A new move supersedes (cancels) the previous timer. Returns the ttl used."""
        ttl = max(0.05, min(float(ttl_s), cfg.ptz_max_ttl_s))
        self.continuous_move(vx, vy)
        with self._lock:
            if self._move_timer is not None:
                self._move_timer.cancel()
            t = threading.Timer(ttl, self._auto_stop)
            t.daemon = True
            self._move_timer = t
            t.start()
        return ttl

    def _auto_stop(self) -> None:
        try:
            self.stop()
        except OnvifError as e:
            # Last-ditch: log loudly — a hung continuous move is the one failure
            # mode this seam must never hide.
            print(f"[vision] onvif {self.host}: auto-stop after move FAILED: {e}", flush=True)

    # ── imaging (VIDEO-SOURCE token — not the profile token) ──────────────────
    IMAGING_FIELDS = ("brightness", "saturation", "contrast", "sharpness")
    _IMAGING_TAGS = {  # ours → ONVIF element (emitted in schema order below)
        "brightness": "Brightness",
        "saturation": "ColorSaturation",
        "contrast": "Contrast",
        "sharpness": "Sharpness",
    }

    def get_imaging(self) -> Dict[str, object]:
        xml = self.call(self.img, f"<timg:GetImagingSettings><timg:VideoSourceToken>"
                                  f"{self.video_source_token()}</timg:VideoSourceToken></timg:GetImagingSettings>")
        out: Dict[str, object] = {}
        for key, tag in self._IMAGING_TAGS.items():
            val = _num(_tag(xml, tag))
            if val is not None:
                out[key] = val
        # IrCutFilter (day/night/IR) is in the ONVIF schema but ABSENT on the MC200's
        # current fw (verified 2026-07-03) — carry it only when the camera reports it.
        ir = _tag(xml, "IrCutFilter")
        if ir is not None:
            out["ir_cut"] = _text(ir)
        return out

    def set_imaging(self, updates: Dict[str, object]) -> Dict[str, object]:
        """Merge `updates` into the camera's CURRENT settings and write back the
        full block (ONVIF replaces, not patches). Numbers clamp to 0..100; `ir_cut`
        (ON/OFF/AUTO) is only written when the camera exposes IrCutFilter at all.
        Returns the merged settings that were written."""
        current = self.get_imaging()
        merged = dict(current)
        for k in self.IMAGING_FIELDS:
            if updates.get(k) is not None:
                merged[k] = max(0.0, min(100.0, float(updates[k])))  # type: ignore[arg-type]
        if updates.get("ir_cut") is not None and "ir_cut" in current:
            mode = str(updates["ir_cut"]).upper()
            if mode not in ("ON", "OFF", "AUTO"):
                raise OnvifError(f"invalid ir_cut {mode!r} (want ON/OFF/AUTO)", fault=True)
            merged["ir_cut"] = mode
        # Emit in ONVIF schema order: Brightness, ColorSaturation, Contrast,
        # IrCutFilter, Sharpness — out-of-order elements fault on strict stacks.
        parts: List[str] = []
        for key in ("brightness", "saturation", "contrast"):
            if key in merged:
                parts.append(f"<tt:{self._IMAGING_TAGS[key]}>{merged[key]}</tt:{self._IMAGING_TAGS[key]}>")
        if "ir_cut" in merged:
            parts.append(f"<tt:IrCutFilter>{merged['ir_cut']}</tt:IrCutFilter>")
        if "sharpness" in merged:
            parts.append(f"<tt:Sharpness>{merged['sharpness']}</tt:Sharpness>")
        self.call(self.img, f"<timg:SetImagingSettings><timg:VideoSourceToken>"
                            f"{self.video_source_token()}</timg:VideoSourceToken>"
                            f"<timg:ImagingSettings>{''.join(parts)}</timg:ImagingSettings>"
                            f"</timg:SetImagingSettings>")
        return merged

    # ── device (info / clock / reboot) ─────────────────────────────────────────
    def get_device_info(self) -> Dict[str, str]:
        xml = self.call(self.dev, f'<GetDeviceInformation xmlns="{TDS}"/>')
        return {
            "manufacturer": _text(_tag(xml, "Manufacturer")),
            "model": _text(_tag(xml, "Model")),
            "firmware": _text(_tag(xml, "FirmwareVersion")),
        }

    def set_system_time(self, now: Optional[datetime] = None) -> None:
        """Push the box clock (UTC, Manual mode). WAN-blocked cameras can't NTP —
        without this their OSD/event timestamps drift (plan §6)."""
        now = now or datetime.now(timezone.utc)
        body = (f'<SetSystemDateAndTime xmlns="{TDS}">'
                f"<DateTimeType>Manual</DateTimeType><DaylightSavings>false</DaylightSavings>"
                f"<UTCDateTime>"
                f'<Time xmlns="{TT}"><Hour>{now.hour}</Hour><Minute>{now.minute}</Minute><Second>{now.second}</Second></Time>'
                f'<Date xmlns="{TT}"><Year>{now.year}</Year><Month>{now.month}</Month><Day>{now.day}</Day></Date>'
                f"</UTCDateTime></SetSystemDateAndTime>")
        self.call(self.dev, body)

    def reboot(self) -> str:
        xml = self.call(self.dev, f'<SystemReboot xmlns="{TDS}"/>')
        return _text(_tag(xml, "Message")) or "rebooting"

    # ── events: PullPoint subscription (plan §3) ───────────────────────────────
    def create_pullpoint(self, term_s: int = 120) -> str:
        """Returns the subscription's Address (on the MC200 it lands on a separate
        port, e.g. http://<cam>:1024/event-1024_1024)."""
        xml = self.call(self.events,
                        f"<tev:CreatePullPointSubscription><tev:InitialTerminationTime>PT{int(term_s)}S"
                        f"</tev:InitialTerminationTime></tev:CreatePullPointSubscription>")
        m = re.search(r"<[\w:]*Address[^>]*>\s*([^<\s]+)\s*</[\w:]*Address>", xml)
        if not m:
            raise OnvifError("CreatePullPointSubscription returned no Address")
        return m.group(1)

    def pull_messages(self, sub_addr: str, timeout_s: float = 10.0,
                      limit: int = 32) -> List[Dict[str, object]]:
        """Long-poll the subscription; returns parsed notifications (may be empty).
        The HTTP timeout rides above the pull timeout so the long-poll can complete."""
        xml = self.call(sub_addr,
                        f"<tev:PullMessages><tev:Timeout>PT{int(timeout_s)}S</tev:Timeout>"
                        f"<tev:MessageLimit>{int(limit)}</tev:MessageLimit></tev:PullMessages>",
                        timeout=timeout_s + max(5.0, self.timeout))
        return parse_notifications(xml)

    def renew(self, sub_addr: str, term_s: int = 120) -> None:
        self.call(sub_addr, f"<wsnt:Renew><wsnt:TerminationTime>PT{int(term_s)}S"
                            f"</wsnt:TerminationTime></wsnt:Renew>")

    def unsubscribe(self, sub_addr: str) -> None:
        self.call(sub_addr, "<wsnt:Unsubscribe/>")


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def _esc(v: str) -> str:
    return (str(v).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def parse_notifications(xml: str) -> List[Dict[str, object]]:
    """PullMessagesResponse → [{topic, ts, data:{Name:Value}}]. Pure — fixture-tested.
    Data carries the rule outputs (IsMotion / IsTamper as 'true'/'false' strings)."""
    notes: List[Dict[str, object]] = []
    for block in re.findall(r"<[\w:]*NotificationMessage[^>]*>(.*?)</[\w:]*NotificationMessage>", xml, re.S):
        topic = _text(_tag(block, "Topic"))
        ts = None
        m = re.search(r'UtcTime="([^"]+)"', block)
        if m:
            ts = m.group(1)
        data: Dict[str, str] = {}
        data_block = _tag(block, "Data") or ""
        for im in re.finditer(r'<[\w:]*SimpleItem\b[^>]*/?>', data_block):
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', im.group(0)))
            if "Name" in attrs:
                data[attrs["Name"]] = attrs.get("Value", "")
        notes.append({"topic": topic, "ts": ts, "data": data})
    return notes


def notification_bool(data: Dict[str, str], *names: str) -> Optional[bool]:
    """First present boolean SimpleItem among `names` ('true'/'1' → True)."""
    for n in names:
        if n in data:
            return data[n].strip().lower() in ("true", "1")
    return None


# ── camera plumbing: creds from the existing stream URL, client per worker ─────
def client_for_camera(cam) -> Optional["OnvifClient"]:
    """Build a client from the camera's existing rtsp:// credentials (plan §1 rule 2).
    Returns None for cameras that can't be ONVIF (no rtsp URL / no credentials) —
    e.g. the ESP32-CAM MJPEG nodes."""
    url = getattr(cam, "stream_url", None)
    if not url:
        return None
    try:
        p = urlparse(url)
    except Exception:  # noqa: BLE001
        return None
    if p.scheme not in ("rtsp", "rtsps") or not (p.hostname and p.username and p.password):
        return None
    return OnvifClient(p.hostname, p.username, p.password)


_UNSET = object()


def get_onvif(cam_id: str):
    """The registry hook: lazily attach an OnvifClient to the camera's live worker
    (so caches/timers live exactly as long as the worker — a stream-URL change
    replaces the worker and with it the client). None → no worker, or the camera
    isn't ONVIF-capable."""
    from .state import workers  # late import — state imports nothing from here

    w = workers.get(cam_id)
    if w is None:
        return None
    client = getattr(w, "_onvif", _UNSET)
    if client is _UNSET:
        client = client_for_camera(getattr(w, "cam", None))
        setattr(w, "_onvif", client)
    return client
