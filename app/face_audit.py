"""Scheduled gallery audit + smear alarm — the tripwire the 2026-07 incidents lacked.

Both pollution episodes (member-vs-member cos 0.459 on 07-06, 0.702 on 07-07) were
visible for DAYS in one number nobody was watching. The auditor recomputes it on a
schedule and acts:

* SMEAR ALARM — any two member profiles reading confusably alike (max cross-anchor
  cosine ≥ `face_smear_alarm_cos`) freezes every silent fold (autoheal, live and
  read paths) via the gallery's `face_folds_frozen` flag and logs loudly. Human
  review keeps working. The freeze self-clears when a later pass measures healthy
  (post cleanup / re-enroll) — no manual unfreeze to forget about.
* PROMOTION COHERENCE — every AUTO-promotion (autoheal) is re-scored against the
  member's CURRENT anchors; ones that no longer cohere are detached back to the
  review queue (no reject mark — a low score is not a human "not me", so the card can
  be suggested again once it sharpens). HUMAN "yes, it's me" confirms are never
  auto-detached: silently undoing a person's answer and re-queuing the card was a
  loop they could not win (2026-07-08). A drifted human promotion is neutralised at
  resolve time by the coherence floor instead (it stops speaking for the member).
* CHURN — fresh clusters per 24h. A 3-person household creating 150/day means
  embeddings match nobody reliably (the mush signal that precedes pollution).

The last report is persisted in the gallery settings (`face_audit_last`) and served
by GET /faces/health alongside a live similarity read.
"""
from __future__ import annotations

import json
import threading
import time

from .config import cfg
from .gallery import Gallery


def run_audit(gallery: Gallery) -> dict:
    """One full audit pass. Cheap (a few hundred cosines), safe to run on demand."""
    similarity = gallery.member_similarity()
    worst = similarity[0] if similarity else None
    smeared = [p for p in similarity if p["score"] >= cfg.face_smear_alarm_cos]

    was_frozen = gallery.folds_frozen
    if smeared:
        gallery.set_kv("face_folds_frozen", "1")
        if not was_frozen:
            pairs = ", ".join(f"{p['a']}~{p['b']}={p['score']}" for p in smeared)
            print(f"[vision] SMEAR ALARM: member profiles confusably alike ({pairs}) "
                  f"— silent folds FROZEN until a clean audit; review/re-enroll the "
                  f"affected members", flush=True)
    elif was_frozen:
        gallery.set_kv("face_folds_frozen", None)
        print("[vision] smear alarm cleared — member profiles read distinct again; "
              "silent folds re-enabled", flush=True)

    promotions = gallery.audit_promotions(cfg.face_audit_detach_below)
    for d in promotions["detached"]:
        print(f"[vision] audit detached {d['guest_id']} from {d['member']} "
              f"(score {d['score']} < {cfg.face_audit_detach_below}) — back to review",
              flush=True)

    churn = gallery.clusters_created_since(24.0)
    if churn >= max(1, cfg.face_churn_warn_24h):
        print(f"[vision] cluster churn warning: {churn} new clusters in 24h — "
              f"embeddings are matching nobody reliably (check camera focus/light, "
              f"quality-gate knobs)", flush=True)

    report = {
        "ts": int(time.time()),
        "member_similarity": similarity,
        "worst_pair": worst,
        "smeared": smeared,
        "folds_frozen": gallery.folds_frozen,
        "promotions_checked": promotions["checked"],
        "promotions_detached": promotions["detached"],
        "clusters_24h": churn,
    }
    gallery.set_kv("face_audit_last", json.dumps(report))
    return report


class FaceAuditor(threading.Thread):
    """Daemon loop: first pass shortly after boot (a restart shouldn't wait 6h to
    notice a smeared gallery), then every `face_audit_interval_s`."""

    def __init__(self, gallery: Gallery) -> None:
        super().__init__(name="face-auditor", daemon=True)
        self._gallery = gallery
        self._stop = threading.Event()

    def run(self) -> None:
        delay = min(120.0, cfg.face_audit_interval_s)
        while not self._stop.wait(delay):
            delay = cfg.face_audit_interval_s
            try:
                run_audit(self._gallery)
            except Exception as e:  # noqa: BLE001 — the auditor must never die quietly
                print(f"[vision] face audit failed: {e!r}", flush=True)

    def stop(self) -> None:
        self._stop.set()
