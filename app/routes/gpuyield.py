"""GPU-yield route — the voice flow's "I need the card for a few seconds" doorbell.

POST /yield {ms?: int}  → current yield status (self-expiring; ≤15s per request;
overlapping requests extend, never shorten). GET /yield reports status. LAN-internal
like the other perception seams; harmless by construction — the worst a caller can
do is delay detection by 15 seconds, and only until the deadline lapses.

Why: the GPU is shared and continuous detection keeps it 100% busy — measured
2026-07-10, voice STT and the LLM ran at ~half speed queueing behind vision. A
voice turn yields vision for its short duration instead (see app/gpu_yield.py).
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .. import gpu_yield

router = APIRouter()


class YieldRequest(BaseModel):
    ms: int = Field(default=10_000, ge=0, le=15_000)


@router.post("/yield")
def request_yield(req: YieldRequest | None = None):
    return gpu_yield.request((req or YieldRequest()).ms)


@router.get("/yield")
def yield_status():
    return gpu_yield.status()
