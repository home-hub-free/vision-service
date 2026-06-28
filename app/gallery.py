"""Face gallery — household enrollment + guest clustering, all on the box.

Two tiers (§4.3), keyed to ONE identity space (§4 first-principle #4):

  * **Household** — enrolled deliberately via the dashboard "Face ID" control, keyed
    to a hub `users.id` (the SAME id a login / voiceprint resolves to). Running-mean
    centroid per person, exactly like the speaker-service voiceprint store.
  * **Guests** — unmatched embeddings are auto-clustered online: a new unknown either
    joins the nearest guest centroid (within `guest_cluster_threshold`) or seeds a new
    `guest:<n>`. A guest seen `guest_min_sightings`+ times is "recurring" and surfaced
    in the dashboard for naming → promote.

Biometrics never leave the box and the hub never holds them (§4.3, §11.6 decision:
vision-service-local sqlite). Embeddings are L2-normalised float vectors; matching is
cosine similarity (mirrors voiceprint). Plain-Python math (no numpy hard dep) so the
gallery works in the null/stub build, same posture as speaker_service.py.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from typing import List, Optional, Tuple

from .config import cfg
from .occupancy import Identity

_lock = threading.Lock()


def _normalise(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return -1.0
    return sum(x * y for x, y in zip(_normalise(a), _normalise(b)))


def _running_mean(mean: List[float], n: int, emb: List[float]) -> List[float]:
    return [(mean[i] * n + emb[i]) / (n + 1) for i in range(len(emb))]


def _confidence(score: float, threshold: float) -> float:
    """Cosine score → agent 0..1 confidence (same calibration as the voiceprint
    service so face and voice feed the gate on one scale)."""
    span = max(1e-6, 1.0 - threshold)
    return round(min(0.99, 0.7 + 0.29 * (score - threshold) / span), 3)


class Gallery:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or cfg.gallery_db
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init()

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init(self) -> None:
        conn = self._db()
        try:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS faces (
                       user_id    TEXT PRIMARY KEY,
                       name       TEXT,
                       embedding  TEXT NOT NULL,
                       samples    INTEGER NOT NULL,
                       thumb      BLOB,
                       updated_at TEXT NOT NULL DEFAULT (datetime('now')))"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS guests (
                       guest_id   TEXT PRIMARY KEY,
                       name       TEXT,
                       embedding  TEXT NOT NULL,
                       sightings  INTEGER NOT NULL,
                       thumb      BLOB,
                       first_seen TEXT NOT NULL DEFAULT (datetime('now')),
                       last_seen  TEXT NOT NULL DEFAULT (datetime('now')),
                       promoted_user_id TEXT)"""
            )
            # Migrate older DBs that predate the face-thumbnail column.
            self._ensure_column(conn, "faces", "thumb", "BLOB")
            self._ensure_column(conn, "guests", "thumb", "BLOB")
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, col: str, decl: str) -> None:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    @staticmethod
    def default_label(guest_id: str) -> str:
        """The friendly default label for an unnamed cluster — "Person N" derived from
        the stable `guest:N` id, so EVERY detected person carries a human-readable label
        by default (the admin can rename it later)."""
        n = guest_id.split(":")[-1]
        return f"Person {n}" if n.isdigit() else guest_id

    # ── household enrollment (Face ID — §6 / §5.3) ────────────────────────────
    def enroll(self, user_id: str, name: Optional[str], emb: List[float],
               thumb: Optional[bytes] = None) -> int:
        emb = _normalise(emb)
        with _lock:
            conn = self._db()
            try:
                row = conn.execute("SELECT embedding, samples, thumb FROM faces WHERE user_id=?", (user_id,)).fetchone()
                if row:
                    merged = _running_mean(json.loads(row[0]), int(row[1]), emb)
                    samples = int(row[1]) + 1
                    thumb = thumb or row[2]  # keep the existing face image if none supplied
                else:
                    merged, samples = emb, 1
                conn.execute(
                    """INSERT INTO faces (user_id, name, embedding, samples, thumb, updated_at)
                       VALUES (?,?,?,?,?,datetime('now'))
                       ON CONFLICT(user_id) DO UPDATE SET
                         name=excluded.name, embedding=excluded.embedding,
                         samples=excluded.samples, thumb=excluded.thumb,
                         updated_at=excluded.updated_at""",
                    (user_id, name, json.dumps(merged), samples, thumb),
                )
                conn.commit()
                return samples
            finally:
                conn.close()

    def forget(self, user_id: str) -> None:
        with _lock:
            conn = self._db()
            try:
                conn.execute("DELETE FROM faces WHERE user_id=?", (user_id,))
                conn.commit()
            finally:
                conn.close()

    def profiles(self) -> List[dict]:
        conn = self._db()
        try:
            rows = conn.execute(
                "SELECT user_id, name, samples, updated_at, thumb IS NOT NULL FROM faces ORDER BY updated_at DESC"
            ).fetchall()
            return [{"user_id": r[0], "name": r[1], "samples": r[2],
                     "updated_at": r[3], "has_thumb": bool(r[4])} for r in rows]
        finally:
            conn.close()

    def _best_household(self, emb: List[float]) -> Tuple[Optional[str], Optional[str], float]:
        conn = self._db()
        try:
            best = (None, None, -1.0)
            for uid, name, blob in conn.execute("SELECT user_id, name, embedding FROM faces"):
                s = _cosine(emb, json.loads(blob))
                if s > best[2]:
                    best = (uid, name, s)
            return best
        finally:
            conn.close()

    # ── guest clustering (§4.3 / §11.7) ───────────────────────────────────────
    def _cluster_guest(self, emb: List[float], thumb: Optional[bytes] = None) -> Tuple[str, Optional[str], int]:
        """Online cluster an unknown embedding. Returns (guest_id, name, sightings).
        Stores the captured face crop on cluster creation (and backfills it later if the
        first sighting had no crop) so the dashboard can show every person's face."""
        with _lock:
            conn = self._db()
            try:
                best_id, best_blob, best_n, best_s = None, None, 0, -1.0
                for gid, blob, n in conn.execute("SELECT guest_id, embedding, sightings FROM guests"):
                    s = _cosine(emb, json.loads(blob))
                    if s > best_s:
                        best_id, best_blob, best_n, best_s = gid, blob, n, s
                if best_id is not None and best_s >= cfg.guest_cluster_threshold:
                    merged = _running_mean(json.loads(best_blob), best_n, emb)
                    if thumb:  # backfill / refresh the representative face
                        conn.execute(
                            "UPDATE guests SET embedding=?, sightings=?, thumb=COALESCE(thumb, ?), last_seen=datetime('now') WHERE guest_id=?",
                            (json.dumps(merged), best_n + 1, thumb, best_id),
                        )
                    else:
                        conn.execute(
                            "UPDATE guests SET embedding=?, sightings=?, last_seen=datetime('now') WHERE guest_id=?",
                            (json.dumps(merged), best_n + 1, best_id),
                        )
                    conn.commit()
                    name = conn.execute("SELECT name FROM guests WHERE guest_id=?", (best_id,)).fetchone()[0]
                    return best_id, name, best_n + 1
                # New guest cluster — every distinct person gets a default id here.
                seq = conn.execute("SELECT COUNT(*) FROM guests").fetchone()[0] + 1
                gid = f"guest:{seq}"
                conn.execute(
                    "INSERT INTO guests (guest_id, name, embedding, sightings, thumb) VALUES (?,?,?,1,?)",
                    (gid, None, json.dumps(_normalise(emb)), thumb),
                )
                conn.commit()
                return gid, None, 1
            finally:
                conn.close()

    def guests(self, recurring_only: bool = False) -> List[dict]:
        conn = self._db()
        try:
            rows = conn.execute(
                """SELECT guest_id, name, sightings, first_seen, last_seen, promoted_user_id,
                          thumb IS NOT NULL
                   FROM guests ORDER BY last_seen DESC"""
            ).fetchall()
            out = []
            for r in rows:
                if recurring_only and r[2] < cfg.guest_min_sightings:
                    continue
                out.append({
                    "guest_id": r[0],
                    "name": r[1],
                    "label": r[1] or self.default_label(r[0]),  # "Person N" default
                    "sightings": r[2],
                    "first_seen": r[3], "last_seen": r[4],
                    "promoted_user_id": r[5],
                    "recurring": r[2] >= cfg.guest_min_sightings,
                    "has_thumb": bool(r[6]),
                })
            return out
        finally:
            conn.close()

    def people(self) -> List[dict]:
        """Unified roster of EVERY labeled person — household members + every detected
        guest cluster (each with its default "Person N" label + face thumbnail) — so the
        dashboard can show faces and the admin can name/promote them. This is the
        "label everyone by default id, surface with faces" surface."""
        conn = self._db()
        try:
            people: List[dict] = []
            for uid, name, samples, has_thumb in conn.execute(
                "SELECT user_id, name, samples, thumb IS NOT NULL FROM faces ORDER BY updated_at DESC"
            ):
                people.append({
                    "id": uid, "label": name or uid, "name": name, "class": "household",
                    "samples": samples, "has_thumb": bool(has_thumb), "named": name is not None,
                })
            for gid, name, sightings, last_seen, promoted, has_thumb in conn.execute(
                """SELECT guest_id, name, sightings, last_seen, promoted_user_id, thumb IS NOT NULL
                   FROM guests WHERE promoted_user_id IS NULL ORDER BY last_seen DESC"""
            ):
                people.append({
                    "id": gid, "label": name or self.default_label(gid), "name": name,
                    "class": "guest", "sightings": sightings, "last_seen": last_seen,
                    "recurring": sightings >= cfg.guest_min_sightings,
                    "has_thumb": bool(has_thumb), "named": name is not None,
                })
            return people
        finally:
            conn.close()

    def get_thumb(self, label_id: str) -> Optional[bytes]:
        """The stored face crop for a label (a `users.id` household member or a
        `guest:N` cluster), or None. Served by the dashboard as the person's face."""
        conn = self._db()
        try:
            table, col = ("guests", "guest_id") if label_id.startswith("guest:") else ("faces", "user_id")
            row = conn.execute(f"SELECT thumb FROM {table} WHERE {col}=?", (label_id,)).fetchone()
            return row[0] if row and row[0] is not None else None
        finally:
            conn.close()

    def promote_guest(self, guest_id: str, user_id: str, name: Optional[str]) -> bool:
        """Promote a recurring guest cluster into a named household member's gallery:
        its centroid seeds (or merges into) the user's face profile, and the guest row
        is tagged promoted so it stops surfacing for review."""
        with _lock:
            conn = self._db()
            try:
                row = conn.execute("SELECT embedding, thumb FROM guests WHERE guest_id=?", (guest_id,)).fetchone()
                if not row:
                    return False
                conn.execute("UPDATE guests SET promoted_user_id=?, name=? WHERE guest_id=?",
                             (user_id, name, guest_id))
                conn.commit()
            finally:
                conn.close()
        # Seed the member's face profile from the cluster centroid + carry its face image.
        self.enroll(user_id, name, json.loads(row[0]), thumb=row[1])
        return True

    def name_guest(self, guest_id: str, name: str) -> bool:
        """Name a recurring guest WITHOUT promoting to household (stays class:guest)."""
        with _lock:
            conn = self._db()
            try:
                cur = conn.execute("UPDATE guests SET name=? WHERE guest_id=?", (name, guest_id))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def forget_guest(self, guest_id: str) -> None:
        with _lock:
            conn = self._db()
            try:
                conn.execute("DELETE FROM guests WHERE guest_id=?", (guest_id,))
                conn.commit()
            finally:
                conn.close()

    # ── the resolver the pipeline calls per new/unmatched track ──────────────
    def resolve(self, emb: List[float], thumb: Optional[bytes] = None) -> Identity:
        """Embedding → Identity. Household match above threshold wins; otherwise the
        guest pipeline clusters and returns a (possibly named) guest. Every distinct
        person therefore gets a stable default id (`guest:N`); `thumb` (the captured
        face crop) is stored so the dashboard can show their face. This is the one call
        the camera worker makes once per new/unmatched track (§4.1)."""
        uid, name, score = self._best_household(emb)
        if uid is not None and score >= cfg.face_match_threshold:
            return Identity(id=uid, name=name, cls="household",
                            confidence=_confidence(score, cfg.face_match_threshold))
        gid, gname, sightings = self._cluster_guest(emb, thumb)
        # A guest's confidence is modest by design — it's "we've seen this person
        # before", not "this is verified David". The agent stays polite/cautious.
        return Identity(id=gid, name=gname, cls="guest",
                        confidence=min(0.6, 0.3 + 0.05 * sightings))
