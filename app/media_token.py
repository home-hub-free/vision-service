"""Signed clip tokens — member-gate video bytes a `<video>` element can't auth.

Footage listing (`/recordings/*`) is bearer-gated with `require_user` (the hub session,
same as Face-ID enroll). But the archived-clip route is fetched by an HTML `<video src>`
GET, which **cannot carry an `Authorization` header** — so we can't gate the bytes the
same way. Instead the bearer-gated `segments` route mints a short-TTL HMAC token bound to
the segment id and embeds it in the clip URL; the clip route verifies the token from the
query string. Keeps playback seekable (plain GET → `FileResponse` honours `Range`) while
staying member-only: a token is only ever issued to a caller who already proved a session.

The secret is `VISION_MEDIA_SECRET`, falling back to `HUB_SERVICE_TOKEN` (already the
service's shared secret with the hub) so nothing new needs provisioning. Tokens are
opaque `<exp>.<hex-sig>`; the sig is HMAC-SHA256 over `"{seg_id}.{exp}"`.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time

# TTL is generous enough to start playback + scrub a bit; a fresh URL is one relist away.
CLIP_TOKEN_TTL_S = 3600


def _secret() -> bytes:
    return (os.getenv("VISION_MEDIA_SECRET") or os.getenv("HUB_SERVICE_TOKEN") or
            "vision-media-dev-secret").encode()


def _sig(seg_id: int, exp: int) -> str:
    return hmac.new(_secret(), f"{seg_id}.{exp}".encode(), hashlib.sha256).hexdigest()


def sign_clip(seg_id: int, ttl_s: int = CLIP_TOKEN_TTL_S) -> str:
    """Mint a token for a segment, valid for `ttl_s` seconds."""
    exp = int(time.time()) + int(ttl_s)
    return f"{exp}.{_sig(seg_id, exp)}"


def verify_clip(seg_id: int, token: str) -> bool:
    """Constant-time verify a clip token against the segment id + expiry."""
    if not token or "." not in token:
        return False
    exp_str, _, sig = token.partition(".")
    try:
        exp = int(exp_str)
    except ValueError:
        return False
    if exp < time.time():
        return False
    return hmac.compare_digest(sig, _sig(seg_id, exp))
