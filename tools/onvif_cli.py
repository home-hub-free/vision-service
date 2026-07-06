#!/usr/bin/env python3
"""Dependency-free ONVIF CLI for the fleet cameras (proven against the Mercusys MC200,
2026-07-02). Raw SOAP 1.2 + WS-Security UsernameToken(PasswordDigest) — no onvif-zeep,
no zeep, nothing outside the stdlib. This is the reference implementation for the
in-service control seam (see docs/CAMERA_ONVIF_CONTROL_PLAN.md in the root workspace).

Credentials/host are NOT hardcoded: they're parsed from the first rtsp:// URL in
VISION_STATIC_CAMERAS in ../.env (the same URL the reader uses), or overridden with
--host/--user/--passwd. Run with the venv python (needs nothing, but keeps habits):

  .venv/bin/python tools/onvif_cli.py info
  .venv/bin/python tools/onvif_cli.py services | profiles | ptz-status | presets
  .venv/bin/python tools/onvif_cli.py events-props          # what event topics exist
  .venv/bin/python tools/onvif_cli.py move 0.5 0.0 0.6      # pan right 0.6s (vx vy secs)
  .venv/bin/python tools/onvif_cli.py set-preset NAME
  .venv/bin/python tools/onvif_cli.py goto TOKEN            # e.g. `goto 1` = hub-home aim
  .venv/bin/python tools/onvif_cli.py imaging               # night/IR/brightness state
  .venv/bin/python tools/onvif_cli.py get-time | set-time   # clock sync (WAN-blocked cams drift)
  .venv/bin/python tools/onvif_cli.py reboot
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from urllib.parse import urlparse

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")

SOAP_NS = "http://www.w3.org/2003/05/soap-envelope"
TDS = "http://www.onvif.org/ver10/device/wsdl"
TRT = "http://www.onvif.org/ver10/media/wsdl"
TPTZ = "http://www.onvif.org/ver20/ptz/wsdl"
TIMG = "http://www.onvif.org/ver20/imaging/wsdl"
TEV = "http://www.onvif.org/ver10/events/wsdl"
TT = "http://www.onvif.org/ver10/schema"
NSDECL = f'xmlns:tptz="{TPTZ}" xmlns:tt="{TT}" xmlns:trt="{TRT}" xmlns:timg="{TIMG}" xmlns:tev="{TEV}"'


def creds_from_env():
    """host,user,pass from the first rtsp:// URL in VISION_STATIC_CAMERAS in .env."""
    try:
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line.startswith("VISION_STATIC_CAMERAS="):
                    m = re.search(r"rtsp://([^:]+):([^@]+)@([\d.]+)", line)
                    if m:
                        return m.group(3), m.group(1), m.group(2)
    except OSError:
        pass
    return None, None, None


class Onvif:
    def __init__(self, host: str, user: str, passwd: str, port: int = 2020):
        self.host, self.user, self.passwd = host, user, passwd
        self.base = f"http://{host}:{port}/onvif"
        # MC200 serves every service from one endpoint; per-service paths also answer.
        self.dev = f"{self.base}/device_service"
        self.media = f"{self.base}/media_service"
        self.ptz = f"{self.base}/ptz_service"
        self.img = f"{self.base}/imaging_service"
        self.events = f"{self.base}/events_service"

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

    def call(self, url: str, body: str, auth: bool = True) -> str:
        header = self._wssec() if auth else "<s:Header/>"
        env = (
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<s:Envelope xmlns:s="{SOAP_NS}" {NSDECL}>{header}<s:Body>{body}</s:Body></s:Envelope>'
        )
        req = urllib.request.Request(
            url, data=env.encode(), headers={"Content-Type": "application/soap+xml; charset=utf-8"}
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as r:
                return r.read().decode(errors="replace")
        except urllib.error.HTTPError as e:
            return f"HTTP {e.code}: " + e.read().decode(errors="replace")
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e}"


def tags(xml: str, name: str):
    return re.findall(rf"<[\w:]*{name}[^>]*>(.*?)</[\w:]*{name}>", xml, re.S)


def clean(v: str) -> str:
    return re.sub(r"<[^>]+>", " ", v).strip()


def dump(xml: str, names):
    if xml.startswith(("ERROR", "HTTP")):
        print(xml[:400])
        return
    if "Fault" in xml:
        reason = tags(xml, "Text") or tags(xml, "Reason")
        print("SOAP FAULT:", clean(reason[0])[:200] if reason else xml[:200])
        return
    for n in names:
        vals = [clean(v)[:110] for v in tags(xml, n)]
        vals = [v for v in vals if v]
        if vals:
            print(f"  {n}: {vals[:10]}")


def main() -> int:
    args = sys.argv[1:]
    host = user = passwd = None
    # crude flag parsing so the tool stays stdlib-only
    for flag in ("--host", "--user", "--passwd"):
        if flag in args:
            i = args.index(flag)
            val = args[i + 1]
            del args[i : i + 2]
            if flag == "--host":
                host = val
            elif flag == "--user":
                user = val
            else:
                passwd = val
    eh, eu, ep = creds_from_env()
    host, user, passwd = host or eh, user or eu, passwd or ep
    if not (host and user and passwd):
        print("no credentials: set VISION_STATIC_CAMERAS in .env or pass --host/--user/--passwd")
        return 2
    cam = Onvif(host, user, passwd)
    cmd = args[0] if args else "info"
    profile = "profile_1"

    if cmd == "info":
        dump(cam.call(cam.dev, f'<GetDeviceInformation xmlns="{TDS}"/>'),
             ["Manufacturer", "Model", "FirmwareVersion", "SerialNumber"])
    elif cmd == "services":
        dump(cam.call(cam.dev, f'<GetServices xmlns="{TDS}"><IncludeCapability>false</IncludeCapability></GetServices>'),
             ["Namespace", "XAddr"])
    elif cmd == "profiles":
        xml = cam.call(cam.media, "<trt:GetProfiles/>")
        dump(xml, ["Name"])
        print("  tokens:", re.findall(r'<[\w:]*Profiles[^>]*token="([^"]+)"', xml))
    elif cmd == "video-sources":
        xml = cam.call(cam.media, "<trt:GetVideoSources/>")
        dump(xml, ["Framerate", "Resolution", "Width", "Height"])
        print("  tokens:", re.findall(r'<[\w:]*VideoSources[^>]*token="([^"]+)"', xml))
    elif cmd == "ptz-status":
        dump(cam.call(cam.ptz, f"<tptz:GetStatus><tptz:ProfileToken>{profile}</tptz:ProfileToken></tptz:GetStatus>"),
             ["Position", "MoveStatus", "PanTilt", "x", "y"])
    elif cmd == "presets":
        xml = cam.call(cam.ptz, f"<tptz:GetPresets><tptz:ProfileToken>{profile}</tptz:ProfileToken></tptz:GetPresets>")
        toks = re.findall(r'<[\w:]*Preset\b[^>]*token="([^"]+)"', xml)
        names = tags(xml, "Name")
        print("  presets:", list(zip(toks, [clean(n) for n in names])))
    elif cmd == "set-preset":
        name = args[1] if len(args) > 1 else f"preset-{int(time.time())}"
        xml = cam.call(cam.ptz, f"<tptz:SetPreset><tptz:ProfileToken>{profile}</tptz:ProfileToken>"
                                f"<tptz:PresetName>{name}</tptz:PresetName></tptz:SetPreset>")
        m = re.search(r"PresetToken>([^<]+)<", xml) or re.search(r'PresetToken="([^"]+)"', xml)
        print(f"  set-preset '{name}' -> token={m.group(1) if m else xml[:200]}")
    elif cmd == "goto":
        tok = args[1] if len(args) > 1 else "1"
        xml = cam.call(cam.ptz, f"<tptz:GotoPreset><tptz:ProfileToken>{profile}</tptz:ProfileToken>"
                                f"<tptz:PresetToken>{tok}</tptz:PresetToken></tptz:GotoPreset>")
        print("  goto:", "OK" if "Fault" not in xml and not xml.startswith(("HTTP", "ERROR")) else xml[:200])
    elif cmd == "move":
        vx, vy = float(args[1]), float(args[2])
        secs = float(args[3]) if len(args) > 3 else 0.5
        cam.call(cam.ptz, f"<tptz:ContinuousMove><tptz:ProfileToken>{profile}</tptz:ProfileToken>"
                          f'<tptz:Velocity><tt:PanTilt x="{vx}" y="{vy}"/></tptz:Velocity></tptz:ContinuousMove>')
        time.sleep(secs)
        cam.call(cam.ptz, f"<tptz:Stop><tptz:ProfileToken>{profile}</tptz:ProfileToken>"
                          f"<tptz:PanTilt>true</tptz:PanTilt></tptz:Stop>")
        print(f"  moved vx={vx} vy={vy} for {secs}s, stopped")
    elif cmd == "imaging":
        # find the real VideoSourceToken first (using the profile token here 400s on MC200)
        xml = cam.call(cam.media, "<trt:GetVideoSources/>")
        toks = re.findall(r'<[\w:]*VideoSources[^>]*token="([^"]+)"', xml)
        vst = toks[0] if toks else "vsconf"
        print(f"  video source token: {vst}")
        dump(cam.call(cam.img, f"<timg:GetImagingSettings><timg:VideoSourceToken>{vst}</timg:VideoSourceToken></timg:GetImagingSettings>"),
             ["Brightness", "Contrast", "ColorSaturation", "Sharpness", "IrCutFilter", "Mode"])
    elif cmd == "events-props":
        dump(cam.call(cam.events, "<tev:GetEventProperties/>"),
             ["TopicNamespaceLocation", "MessageContentFilterDialect"])
        xml = cam.call(cam.events, "<tev:GetEventProperties/>")
        # topic tree elements are vendor-y; show raw topic names
        topics = re.findall(r"<(tns1?:[\w/]+|[\w]+:RuleEngine[\w/]*)", xml)
        print("  raw topic-ish tags:", sorted(set(topics))[:20])
        print("  (full XML below for the delegated session)")
        print(xml[:3000])
    elif cmd == "get-time":
        dump(cam.call(cam.dev, f'<GetSystemDateAndTime xmlns="{TDS}"/>', auth=False),
             ["Year", "Month", "Day", "Hour", "Minute", "Second", "DateTimeType"])
    elif cmd == "set-time":
        now = datetime.now(timezone.utc)
        body = (f'<SetSystemDateAndTime xmlns="{TDS}">'
                f"<DateTimeType>Manual</DateTimeType><DaylightSavings>false</DaylightSavings>"
                f"<UTCDateTime>"
                f"<Time xmlns=\"{TT}\"><Hour>{now.hour}</Hour><Minute>{now.minute}</Minute><Second>{now.second}</Second></Time>"
                f"<Date xmlns=\"{TT}\"><Year>{now.year}</Year><Month>{now.month}</Month><Day>{now.day}</Day></Date>"
                f"</UTCDateTime></SetSystemDateAndTime>")
        xml = cam.call(cam.dev, body)
        print("  set-time:", "OK" if "Fault" not in xml and not xml.startswith(("HTTP", "ERROR")) else xml[:300])
    elif cmd == "reboot":
        xml = cam.call(cam.dev, f'<SystemReboot xmlns="{TDS}"/>')
        print("  reboot:", clean(tags(xml, "Message")[0]) if tags(xml, "Message") else xml[:200])
    elif cmd == "snapshot":
        # GetSnapshotUri for each profile + fetch a test JPEG (no-auth, then basic) —
        # the probe behind the on-demand high-res sampler (app/highres.py).
        import base64
        import urllib.request
        xml = cam.call(cam.media, "<trt:GetProfiles/>")
        toks = re.findall(r'<[\w:]*Profiles[^>]*token="([^"]+)"', xml)
        for tok in toks or ["profile_1"]:
            xml = cam.call(cam.media, f"<trt:GetSnapshotUri><trt:ProfileToken>{tok}</trt:ProfileToken></trt:GetSnapshotUri>")
            uris = tags(xml, "Uri")
            uri = clean(uris[0]) if uris else None
            print(f"  profile {tok}: uri={uri}")
            if not uri:
                print(f"    raw: {xml[:400]}")
                continue
            for label, hdr in (("no-auth", None),
                               ("basic", "Basic " + base64.b64encode(
                                   f"{user}:{passwd}".encode()).decode())):
                req = urllib.request.Request(uri)
                if hdr:
                    req.add_header("Authorization", hdr)
                try:
                    with urllib.request.urlopen(req, timeout=5) as r:
                        data = r.read()
                    print(f"    GET {label}: {len(data)} bytes jpeg={data[:2] == bytes([0xFF, 0xD8])}")
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"    GET {label}: {e}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
