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
import re
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
        # Optional hook (wired at app startup to perception.annotate_face_in_thumb;
        # gallery itself stays cv2-free): (thumb_jpeg, centroid) → normalized face
        # box within the thumb, [] for "no face found", None for "engine unavailable".
        # Used to lazily locate the face in LEGACY person-crop thumbs at review time.
        self.thumb_annotator = None
        # Capture-ledger folder: alongside THIS gallery's DB unless overridden, so a
        # temp-DB gallery (tests) never writes crops into the production data dir.
        self.captures_dir = cfg.captures_dir or os.path.join(
            os.path.dirname(self.db_path), "captures")
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
            # "Not me" answers from the review flow — JSON list of users.id this
            # cluster must never be suggested to / auto-healed into again.
            self._ensure_column(conn, "guests", "rejected_user_ids", "TEXT")
            # Where the face sits WITHIN the stored thumb (normalized [x,y,w,h] JSON)
            # so the review card can ring exactly the face the question is about.
            # NULL = not known yet (legacy thumb, annotated lazily); "[]" = the engine
            # looked and found no face (don't re-run).
            self._ensure_column(conn, "guests", "thumb_box", "TEXT")
            # Runtime-adjustable recognition thresholds (the auto-heal/match/suggest
            # knobs). Empty = use the config.py/env defaults; a row overrides it live so
            # the household can tune from Settings without a redeploy. See `thresholds()`.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            # Capture ledger — the permanent archive behind every identity decision
            # (see `_capture`): the crop lands on disk under captures_dir, this row
            # indexes it WITH the exact embedding so a member profile can be rebuilt
            # from curated crops without re-running the face engine.
            conn.execute(
                """CREATE TABLE IF NOT EXISTS captures (
                       id          INTEGER PRIMARY KEY AUTOINCREMENT,
                       ts          TEXT NOT NULL DEFAULT (datetime('now')),
                       kind        TEXT NOT NULL,
                       resolved_id TEXT,
                       cluster_id  TEXT,
                       score       REAL,
                       reinforced  INTEGER NOT NULL DEFAULT 0,
                       embedding   TEXT NOT NULL,
                       path        TEXT,
                       thumb_box   TEXT)"""
            )
            # Anchor set — a member's face profile IS these individually-stored,
            # quality-gated enroll embeddings. Matching scores a live face against
            # the top-k nearest anchors; nothing at runtime ever mutates them (the
            # 2026-07-07 lesson: every "self-improving" fold into a shared running
            # mean — reinforce, promote — was a pollution channel; a running mean of
            # noise is what made two members converge to cos 0.702 and swap names).
            # faces.embedding stays as the derived MEAN of the anchors: a display /
            # legacy-compat cache, not the matching surface.
            conn.execute(
                """CREATE TABLE IF NOT EXISTS anchors (
                       id         INTEGER PRIMARY KEY AUTOINCREMENT,
                       user_id    TEXT NOT NULL,
                       embedding  TEXT NOT NULL,
                       source     TEXT NOT NULL DEFAULT 'enroll',
                       created_at TEXT NOT NULL DEFAULT (datetime('now')))""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_anchors_user ON anchors(user_id)")
            conn.commit()
        finally:
            conn.close()

    # ── runtime-adjustable thresholds (Settings ▸ Face recognition) ───────────
    # Name → config.py default. These are the levers that decide how eagerly a live
    # face matches a member, self-reinforces, clusters as a guest, and auto-heals. The
    # effective value is a `settings` override when present, else the cfg default — so
    # the resolver reads them live and the dashboard can show + edit them.
    _TUNABLES = (
        "face_match_threshold", "face_match_margin",
        "face_reinforce_threshold", "face_reinforce_margin",
        "guest_cluster_threshold", "face_autoheal_threshold", "face_autoheal_margin",
        "face_suggest_threshold",
    )

    def _thr(self, name: str) -> float:
        """Effective value of a tunable threshold: DB override if set, else cfg default."""
        default = float(getattr(cfg, name))
        conn = self._db()
        try:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (name,)).fetchone()
        finally:
            conn.close()
        if not row:
            return default
        try:
            return float(row[0])
        except (ValueError, TypeError):
            return default

    def thresholds(self) -> List[dict]:
        """Every tunable threshold with its effective value, the code default, and
        whether it's currently overridden — the payload the Settings panel renders."""
        conn = self._db()
        try:
            overrides = {k: v for k, v in conn.execute("SELECT key, value FROM settings")}
        finally:
            conn.close()
        out = []
        for name in self._TUNABLES:
            default = round(float(getattr(cfg, name)), 4)
            overridden = name in overrides
            try:
                value = float(overrides[name]) if overridden else default
            except (ValueError, TypeError):
                value, overridden = default, False
            out.append({"key": name, "value": round(value, 4),
                        "default": default, "overridden": overridden})
        return out

    def set_thresholds(self, updates: dict) -> List[dict]:
        """Persist threshold overrides. A value of None (or the string "default") CLEARS
        the override, falling back to the code default. Unknown keys are ignored. Returns
        the fresh `thresholds()` view."""
        with _lock:
            conn = self._db()
            try:
                for key, val in updates.items():
                    if key not in self._TUNABLES:
                        continue
                    if val is None or val == "default":
                        conn.execute("DELETE FROM settings WHERE key=?", (key,))
                        continue
                    conn.execute(
                        "INSERT INTO settings (key, value) VALUES (?,?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, str(float(val))))
                conn.commit()
            finally:
                conn.close()
        return self.thresholds()

    # ── generic settings kv (audit report, fold freeze) ───────────────────────
    def get_kv(self, key: str) -> Optional[str]:
        conn = self._db()
        try:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def set_kv(self, key: str, value: Optional[str]) -> None:
        """Persist (or, with None, clear) a settings row."""
        with _lock:
            conn = self._db()
            try:
                if value is None:
                    conn.execute("DELETE FROM settings WHERE key=?", (key,))
                else:
                    conn.execute(
                        "INSERT INTO settings (key, value) VALUES (?,?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
                conn.commit()
            finally:
                conn.close()

    @property
    def folds_frozen(self) -> bool:
        """True while the smear alarm has silent folds frozen (see app/face_audit.py):
        two member profiles read confusably alike, so autoheal would only deepen the
        cross-contamination. Human review keeps working; the auditor clears the flag
        when a later pass measures healthy again."""
        return self.get_kv("face_folds_frozen") == "1"

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
               thumb: Optional[bytes] = None, source: str = "enroll") -> int:
        """Add one quality-gated sample to the member's ANCHOR SET (the profile is
        the anchors — see _init). The faces-table centroid is recomputed as the plain
        mean of the anchors, so a member with a legacy (possibly polluted) running-
        mean centroid and no anchors gets their profile RESET to clean ground truth
        on their first gated enroll — that's the point, not an accident. Returns the
        anchor count (= samples)."""
        emb = _normalise(emb)
        self._capture("enroll", emb, thumb, resolved_id=user_id)
        with _lock:
            conn = self._db()
            try:
                conn.execute("INSERT INTO anchors (user_id, embedding, source) VALUES (?,?,?)",
                             (user_id, json.dumps(emb), source))
                cap = max(1, cfg.face_anchor_cap)
                conn.execute(
                    """DELETE FROM anchors WHERE user_id=? AND id NOT IN
                       (SELECT id FROM anchors WHERE user_id=? ORDER BY id DESC LIMIT ?)""",
                    (user_id, user_id, cap))
                anchors = [json.loads(r[0]) for r in conn.execute(
                    "SELECT embedding FROM anchors WHERE user_id=?", (user_id,))]
                dim = len(anchors[0])
                merged = _normalise([sum(a[i] for a in anchors) / len(anchors)
                                     for i in range(dim)])
                row = conn.execute("SELECT thumb FROM faces WHERE user_id=?", (user_id,)).fetchone()
                if row:
                    thumb = thumb or row[0]  # keep the existing face image if none supplied
                conn.execute(
                    """INSERT INTO faces (user_id, name, embedding, samples, thumb, updated_at)
                       VALUES (?,?,?,?,?,datetime('now'))
                       ON CONFLICT(user_id) DO UPDATE SET
                         name=excluded.name, embedding=excluded.embedding,
                         samples=excluded.samples, thumb=excluded.thumb,
                         updated_at=excluded.updated_at""",
                    (user_id, name, json.dumps(merged), len(anchors), thumb),
                )
                conn.commit()
                return len(anchors)
            finally:
                conn.close()

    # ── capture ledger (identity-pollution insurance) ─────────────────────────
    def _capture(self, kind: str, emb: List[float], thumb: Optional[bytes],
                 thumb_box: Optional[List[float]] = None,
                 resolved_id: Optional[str] = None, cluster_id: Optional[str] = None,
                 score: Optional[float] = None, reinforced: bool = False) -> None:
        """Permanently archive the face crop + embedding behind an identity decision.

        The member/cluster centroids are running means — once an embedding is folded
        in, its individual contribution is unrecoverable (reinforce folds especially:
        the per-frame embedding used to be discarded on the spot). This ledger keeps
        the raw INGREDIENTS instead: the crop as a plain JPEG on disk (grouped per
        resolved identity, so a human can review a folder of faces), and an index row
        carrying the EXACT embedding — so any member profile can be rebuilt from a
        curated set (tools/rebuild_profile.py) without re-running the face engine.
        Best-effort by design: never throws into resolve/enroll; only sightings that
        carry a crop are recorded (an embedding nobody can review can't be curated)."""
        if not cfg.captures_enabled or thumb is None:
            return
        try:
            owner = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(resolved_id or cluster_id or "unknown"))
            d = os.path.join(self.captures_dir, owner)
            os.makedirs(d, exist_ok=True)
            fname = f"{time.time_ns()}_{kind}.jpg"
            with open(os.path.join(d, fname), "wb") as fh:
                fh.write(thumb)
            with _lock:
                conn = self._db()
                try:
                    conn.execute(
                        """INSERT INTO captures (kind, resolved_id, cluster_id, score,
                                                 reinforced, embedding, path, thumb_box)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (kind, resolved_id, cluster_id,
                         round(float(score), 4) if score is not None else None,
                         1 if reinforced else 0, json.dumps(_normalise(emb)),
                         os.path.join(owner, fname),
                         json.dumps(thumb_box) if thumb_box is not None else None))
                    conn.commit()
                finally:
                    conn.close()
        except Exception as e:  # noqa: BLE001 — the ledger must never break recognition
            print(f"[vision] capture ledger write failed: {e!r}", flush=True)

    def captures(self, resolved_id: Optional[str] = None) -> List[dict]:
        """Ledger rows (newest first), optionally for one identity — the review/audit
        read used by tools/rebuild_profile.py."""
        conn = self._db()
        try:
            q = ("SELECT id, ts, kind, resolved_id, cluster_id, score, reinforced, path "
                 "FROM captures")
            args: tuple = ()
            if resolved_id is not None:
                q += " WHERE resolved_id=?"
                args = (resolved_id,)
            rows = conn.execute(q + " ORDER BY id DESC", args).fetchall()
        finally:
            conn.close()
        return [{"id": r[0], "ts": r[1], "kind": r[2], "resolved_id": r[3],
                 "cluster_id": r[4], "score": r[5], "reinforced": bool(r[6]),
                 "path": r[7]} for r in rows]

    def capture_image(self, capture_id: int) -> Optional[bytes]:
        """The archived crop JPEG for one ledger row, or None (no crop / file gone)."""
        conn = self._db()
        try:
            row = conn.execute("SELECT path FROM captures WHERE id=?", (capture_id,)).fetchone()
        finally:
            conn.close()
        if not row or not row[0]:
            return None
        try:
            with open(os.path.join(self.captures_dir, row[0]), "rb") as fh:
                return fh.read()
        except OSError:
            return None

    def delete_capture(self, capture_id: int) -> bool:
        """Remove one ingredient from the ledger — row AND crop file — so a manual
        clean of a polluted set sticks: a later rebuild uses only what remains."""
        with _lock:
            conn = self._db()
            try:
                row = conn.execute("SELECT path FROM captures WHERE id=?", (capture_id,)).fetchone()
                if not row:
                    return False
                conn.execute("DELETE FROM captures WHERE id=?", (capture_id,))
                conn.commit()
            finally:
                conn.close()
        if row[0]:
            try:
                os.remove(os.path.join(self.captures_dir, row[0]))
            except OSError:
                pass  # file already gone — the row was the source of truth
        return True

    def rebuild_member_from_captures(self, user_id: str,
                                     name: Optional[str] = None) -> Optional[int]:
        """Re-make a member's soup from the ledger: REPLACE their centroid with the
        plain mean of every remaining capture archived for them (the dashboard's
        "rebuild profile" — delete the polluted photos first, then call this).
        Returns the new samples count, or None when no captures exist to build from."""
        conn = self._db()
        try:
            rows = conn.execute("SELECT embedding FROM captures WHERE resolved_id=?",
                                (user_id,)).fetchall()
        finally:
            conn.close()
        if not rows:
            return None
        return self.rebuild_member(user_id, [json.loads(r[0]) for r in rows], name=name)

    def rebuild_member(self, user_id: str, embs: List[List[float]],
                       name: Optional[str] = None, thumb: Optional[bytes] = None) -> int:
        """Re-make the soup: the curated embedding set BECOMES the member's profile —
        it replaces the anchor set (newest-cap kept) and the centroid cache is the
        plain mean. The old anchors/centroid are discarded entirely: rebuild is the
        human saying "exactly these photos are me". name/thumb are kept from the
        existing row unless supplied. Returns the new samples count."""
        if not embs:
            raise ValueError("no embeddings to rebuild from")
        vecs = [_normalise(e) for e in embs][-max(1, cfg.face_anchor_cap):]
        dim = len(vecs[0])
        merged = _normalise([sum(v[i] for v in vecs) / len(vecs) for i in range(dim)])
        with _lock:
            conn = self._db()
            try:
                conn.execute("DELETE FROM anchors WHERE user_id=?", (user_id,))
                conn.executemany(
                    "INSERT INTO anchors (user_id, embedding, source) VALUES (?,?, 'rebuild')",
                    [(user_id, json.dumps(v)) for v in vecs])
                row = conn.execute("SELECT name, thumb FROM faces WHERE user_id=?",
                                   (user_id,)).fetchone()
                keep_name = name if name is not None else (row[0] if row else None)
                keep_thumb = thumb if thumb is not None else (row[1] if row else None)
                conn.execute(
                    """INSERT INTO faces (user_id, name, embedding, samples, thumb, updated_at)
                       VALUES (?,?,?,?,?,datetime('now'))
                       ON CONFLICT(user_id) DO UPDATE SET
                         name=excluded.name, embedding=excluded.embedding,
                         samples=excluded.samples, thumb=excluded.thumb,
                         updated_at=excluded.updated_at""",
                    (user_id, keep_name, json.dumps(merged), len(vecs), keep_thumb))
                conn.commit()
            finally:
                conn.close()
        return len(vecs)

    def forget(self, user_id: str) -> None:
        """Erase a member's face profile — centroid AND anchor set. The capture
        ledger deliberately survives (it's the permanent archive; rebuild can
        resurrect a profile from it)."""
        with _lock:
            conn = self._db()
            try:
                conn.execute("DELETE FROM faces WHERE user_id=?", (user_id,))
                conn.execute("DELETE FROM anchors WHERE user_id=?", (user_id,))
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

    @staticmethod
    def _anchor_score(emb: List[float], anchors: List[List[float]]) -> float:
        """Mean of the top-2 anchor cosines: max alone would let ONE rogue anchor
        impersonate the member (the enroll gate checks quality, not identity — a
        wrong-person photo can still slip in); a full mean would punish legitimate
        look variety (glasses, beard, lighting). Top-2 demands a second honest
        agreement, which same-person gated anchors reliably give (they correlate
        0.55+), while an imposter has to fool two independent enroll shots."""
        sims = sorted((_cosine(emb, a) for a in anchors), reverse=True)
        k = min(2, len(sims))
        return sum(sims[:k]) / k

    def _member_scores(self, emb: List[float],
                       exclude: Optional[set] = None) -> List[Tuple[float, str, Optional[str]]]:
        """Every household member scored against an embedding, best first. Members
        with an anchor set score by top-k anchors (the real matching surface);
        anchor-less legacy members fall back to their centroid cache."""
        conn = self._db()
        try:
            anchors: dict = {}
            for uid, blob in conn.execute("SELECT user_id, embedding FROM anchors"):
                anchors.setdefault(uid, []).append(json.loads(blob))
            scored = []
            for uid, name, blob in conn.execute("SELECT user_id, name, embedding FROM faces"):
                if exclude and uid in exclude:
                    continue
                if uid in anchors:
                    scored.append((self._anchor_score(emb, anchors[uid]), uid, name))
                else:
                    scored.append((_cosine(emb, json.loads(blob)), uid, name))
            scored.sort(key=lambda t: t[0], reverse=True)
            return scored
        finally:
            conn.close()

    def _member_score_one(self, user_id: str, emb: List[float]) -> Optional[float]:
        """This member's (anchor-aware) score for an embedding, or None if they have
        no face profile at all (e.g. promoted-only, never enrolled)."""
        for s, uid, _name in self._member_scores(emb):
            if uid == user_id:
                return s
        return None

    def member_similarity(self) -> List[dict]:
        """Pairwise member-vs-member confusability — THE smear tripwire. Score =
        the max cross-anchor cosine between the two profiles (the impersonation
        risk: it only takes one confusable anchor pair to start swapping names);
        anchor-less legacy members contribute their centroid. Distinct people sit
        ~0.0–0.3; both pollution incidents read 0.45+ for days unwatched."""
        conn = self._db()
        try:
            vecs: dict = {}
            for uid, blob in conn.execute("SELECT user_id, embedding FROM anchors"):
                vecs.setdefault(uid, []).append(json.loads(blob))
            for uid, blob in conn.execute("SELECT user_id, embedding FROM faces"):
                vecs.setdefault(uid, [json.loads(blob)])  # centroid only if no anchors
        finally:
            conn.close()
        ids = sorted(vecs)
        out = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                score = max(_cosine(a, b) for a in vecs[ids[i]] for b in vecs[ids[j]])
                out.append({"a": ids[i], "b": ids[j], "score": round(score, 3)})
        out.sort(key=lambda r: -r["score"])
        return out

    def audit_promotions(self, detach_below: float) -> dict:
        """Re-score every member promotion against the member's CURRENT profile
        (anchors) and detach the ones that no longer cohere — back to the review
        queue with re-heal blocked. Members without anchors are skipped (no ground
        truth to judge against). Returns {checked, detached:[...]}. """
        conn = self._db()
        try:
            anchored = {r[0] for r in conn.execute("SELECT DISTINCT user_id FROM anchors")}
            rows = conn.execute(
                """SELECT guest_id, promoted_user_id, embedding FROM guests
                   WHERE promoted_user_id IS NOT NULL
                     AND promoted_user_id NOT LIKE 'guest:%'""").fetchall()
        finally:
            conn.close()
        checked, detached = 0, []
        for gid, member, blob in rows:
            if member not in anchored:
                continue
            checked += 1
            score = self._member_score_one(member, json.loads(blob))
            if score is not None and score < detach_below:
                if self.detach_cluster(gid):
                    detached.append({"guest_id": gid, "member": member,
                                     "score": round(score, 3)})
        return {"checked": checked, "detached": detached}

    def clusters_created_since(self, hours: float = 24.0) -> int:
        """Cluster churn: fresh guest clusters in the window. A 3-person household
        creating 150/day means embeddings aren't matching ANYONE reliably — the
        mush signal that precedes pollution."""
        conn = self._db()
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM guests WHERE first_seen >= datetime('now', ?)",
                (f"-{int(hours * 3600)} seconds",)).fetchone()[0]
        finally:
            conn.close()

    def _best_household(self, emb: List[float],
                        exclude: Optional[set] = None) -> Tuple[Optional[str], Optional[str], float, float]:
        """Best household match for an embedding. Returns
        (user_id, name, best_score, margin) where `margin` is best_score minus the
        2nd-best member's score — a measure of how UNAMBIGUOUS the match is (large when
        one member clearly wins; small when two members score alike). margin is +inf when
        there's only one enrolled member (nothing to confuse it with). `exclude` drops
        members a reviewer already answered "not me" for (review-tier rejections)."""
        scored = self._member_scores(emb, exclude)
        if not scored:
            return (None, None, -1.0, 0.0)
        best_s, best_uid, best_name = scored[0]
        margin = best_s - scored[1][0] if len(scored) > 1 else float("inf")
        return (best_uid, best_name, best_s, margin)

    def _reinforce_household(self, user_id: str, emb: List[float]) -> None:
        """Online learning for LEGACY (anchor-less) members only: fold a confident,
        unambiguous live embedding into the centroid via capped running-mean. A member
        with an anchor set never reinforces — anchors are immutable ground truth, and
        every runtime fold into a shared mean was a measured pollution channel (the
        2026-07-07 lesson); their centroid row is a derived cache the next enroll
        recomputes, so drifting it would be both risky and pointless."""
        emb = _normalise(emb)
        cap = max(1, cfg.face_reinforce_cap)
        with _lock:
            conn = self._db()
            try:
                anchored = conn.execute("SELECT 1 FROM anchors WHERE user_id=? LIMIT 1",
                                        (user_id,)).fetchone()
                if anchored:
                    return
                row = conn.execute("SELECT embedding, samples FROM faces WHERE user_id=?", (user_id,)).fetchone()
                if not row:
                    return
                cur = int(row[1])
                merged = _running_mean(json.loads(row[0]), min(cur, cap), emb)
                samples = cur + 1 if cur < cap else cur
                conn.execute(
                    "UPDATE faces SET embedding=?, samples=?, updated_at=datetime('now') WHERE user_id=?",
                    (json.dumps(merged), samples, user_id),
                )
                conn.commit()
            finally:
                conn.close()

    def _best_identity(self, emb: List[float], exclude: Optional[set] = None,
                       exclude_guest: Optional[str] = None
                       ) -> Tuple[Optional[str], Optional[str], Optional[str], float, float]:
        """Best match across EVERY known identity — household members AND named guest
        clusters — so a named guest ("Abuela") absorbs re-appearances exactly like a
        member does, instead of every new angle spawning a fresh "Person N". Returns
        (kind, id, name, best_score, margin); kind is "member"/"guest"/None. `exclude`
        holds ids (either kind) a reviewer already rejected for this cluster;
        `exclude_guest` keeps a cluster from matching itself."""
        exclude = exclude or set()
        scored = [(s, "member", uid, name)
                  for s, uid, name in self._member_scores(emb, exclude)]
        conn = self._db()
        try:
            for gid, gname, blob in conn.execute(
                    "SELECT guest_id, name, embedding FROM guests "
                    "WHERE name IS NOT NULL AND promoted_user_id IS NULL"):
                if gid in exclude or gid == exclude_guest:
                    continue
                scored.append((_cosine(emb, json.loads(blob)), "guest", gid, gname))
        finally:
            conn.close()
        if not scored:
            return (None, None, None, -1.0, 0.0)
        scored.sort(key=lambda t: t[0], reverse=True)
        best_s, kind, best_id, best_name = scored[0]
        margin = best_s - scored[1][0] if len(scored) > 1 else float("inf")
        return (kind, best_id, best_name, best_s, margin)

    # ── guest clustering (§4.3 / §11.7) ───────────────────────────────────────
    def _cluster_guest(self, emb: List[float], thumb: Optional[bytes] = None,
                       thumb_box: Optional[List[float]] = None
                       ) -> Tuple[str, Optional[str], int, Optional[str], float]:
        """Online cluster an unknown embedding. Returns (guest_id, name, sightings,
        promoted_user_id, match_score) — promoted_user_id is set when the matched
        cluster was folded into a household member (resolve answers as THEM, not as a
        guest), and match_score is the cosine against the cluster centroid (-1.0 for a
        freshly seeded cluster). Stores the captured face crop on cluster creation (and
        backfills it later if the first sighting had no crop) so the dashboard can show
        every person's face. `thumb_box` is the face's normalized position within that
        crop — it always travels WITH the thumb (only written when this call's thumb is
        the one kept)."""
        box_json = json.dumps(thumb_box) if thumb_box is not None else None
        with _lock:
            conn = self._db()
            try:
                best_id, best_blob, best_n, best_s = None, None, 0, -1.0
                for gid, blob, n in conn.execute("SELECT guest_id, embedding, sightings FROM guests"):
                    s = _cosine(emb, json.loads(blob))
                    if s > best_s:
                        best_id, best_blob, best_n, best_s = gid, blob, n, s
                if best_id is not None and best_s >= self._thr("guest_cluster_threshold"):
                    merged = _running_mean(json.loads(best_blob), best_n, emb)
                    if thumb:
                        # Backfill a missing thumb — and REPLACE a bad one: if the
                        # stored crop has no locatable face (thumb_box '[]', or NULL =
                        # unknown legacy quality) and this sighting's crop is
                        # face-located, the new crop wins. Bad "who is this?" photos
                        # heal themselves the next time the person walks by.
                        conn.execute(
                            """UPDATE guests SET embedding=?, sightings=?,
                                   thumb_box=CASE WHEN thumb IS NULL
                                       OR ((thumb_box IS NULL OR thumb_box='[]') AND ? IS NOT NULL)
                                       THEN ? ELSE thumb_box END,
                                   thumb=CASE WHEN thumb IS NULL
                                       OR ((thumb_box IS NULL OR thumb_box='[]') AND ? IS NOT NULL)
                                       THEN ? ELSE thumb END,
                                   last_seen=datetime('now') WHERE guest_id=?""",
                            (json.dumps(merged), best_n + 1,
                             box_json, box_json, box_json, thumb, best_id),
                        )
                    else:
                        conn.execute(
                            "UPDATE guests SET embedding=?, sightings=?, last_seen=datetime('now') WHERE guest_id=?",
                            (json.dumps(merged), best_n + 1, best_id),
                        )
                    conn.commit()
                    name, promoted = conn.execute(
                        "SELECT name, promoted_user_id FROM guests WHERE guest_id=?", (best_id,)).fetchone()
                    return best_id, name, best_n + 1, promoted, best_s
                # New guest cluster — every distinct person gets a default id here.
                # MAX+1, not COUNT+1: after any deletion COUNT falls below the top id
                # and COUNT+1 collides with an existing row (UNIQUE violation), which
                # killed clustering entirely until the ids realigned.
                row = conn.execute(
                    "SELECT MAX(CAST(substr(guest_id, 7) AS INTEGER)) FROM guests "
                    "WHERE guest_id LIKE 'guest:%'").fetchone()
                seq = (row[0] or 0) + 1
                gid = f"guest:{seq}"
                conn.execute(
                    "INSERT INTO guests (guest_id, name, embedding, sightings, thumb, thumb_box) VALUES (?,?,?,1,?,?)",
                    (gid, None, json.dumps(_normalise(emb)), thumb, box_json if thumb else None),
                )
                conn.commit()
                return gid, None, 1, None, -1.0
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
            seen_members = set()
            for uid, name, samples, has_thumb in conn.execute(
                "SELECT user_id, name, samples, thumb IS NOT NULL FROM faces ORDER BY updated_at DESC"
            ):
                seen_members.add(uid)
                people.append({
                    "id": uid, "label": name or uid, "name": name, "class": "household",
                    "samples": samples, "has_thumb": bool(has_thumb), "named": name is not None,
                })
            # A member the gallery knows ONLY through promoted clusters (promotion no
            # longer seeds a faces row — see promote_guest): still a recognizable
            # household identity, so the roster shows them; samples=0 says "not
            # enrolled yet" honestly.
            for uid, name, has_thumb in conn.execute(
                """SELECT promoted_user_id, MAX(name), MAX(thumb IS NOT NULL)
                   FROM guests WHERE promoted_user_id IS NOT NULL
                     AND promoted_user_id NOT LIKE 'guest:%'
                   GROUP BY promoted_user_id"""
            ):
                if uid in seen_members:
                    continue
                people.append({
                    "id": uid, "label": name or uid, "name": name, "class": "household",
                    "samples": 0, "has_thumb": bool(has_thumb), "named": name is not None,
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
            if row and row[0] is not None:
                return row[0]
            if not label_id.startswith("guest:"):
                # Member without an enroll portrait: lend the face from their most
                # recently seen promoted cluster (promotion no longer copies thumbs
                # into the profile — the cluster keeps owning its crop).
                row = conn.execute(
                    """SELECT thumb FROM guests WHERE promoted_user_id=? AND thumb IS NOT NULL
                       ORDER BY last_seen DESC LIMIT 1""", (label_id,)).fetchone()
                return row[0] if row else None
            return None
        finally:
            conn.close()

    def promote_guest(self, guest_id: str, user_id: str, name: Optional[str],
                      carry_thumb: bool = True) -> bool:
        """Promote a guest cluster into a household member — a ROUTING decision only:
        the cluster is tagged promoted (it stops surfacing for review, and resolve
        answers as that member on a cluster match), but it NEVER folds into the
        member's face profile. It used to (via enroll's running mean, full weight,
        no gate) — and that was the pollution engine of 2026-07-07: ~110 mostly
        single-sighting clusters auto-healed in a day, each dumping a junk embedding
        into a member centroid until two members read cos 0.702 apart and swapped
        names. The member's profile is built from enrollment anchors alone; a
        promoted cluster contributes recognition coverage (far/angled cameras) from
        its OWN centroid. The faces table is never written here at all — a member
        with no enroll portrait borrows a promoted cluster's crop at read time
        (get_thumb), so `carry_thumb` no longer copies anything (kept for caller
        compatibility)."""
        del carry_thumb  # thumb lending moved to get_thumb (read time)
        with _lock:
            conn = self._db()
            try:
                row = conn.execute("SELECT guest_id FROM guests WHERE guest_id=?", (guest_id,)).fetchone()
                if not row:
                    return False
                conn.execute("UPDATE guests SET promoted_user_id=?, name=? WHERE guest_id=?",
                             (user_id, name, guest_id))
                conn.commit()
            finally:
                conn.close()
        return True

    def member_clusters(self, user_id: str) -> List[dict]:
        """Every guest cluster that was folded INTO a household member (by auto-heal or
        a manual promote) — the audit trail behind that member's face profile. Each row
        carries its captured thumb + how well it still matches the member's centroid
        (`score`), so a reviewer can spot an outlier the thresholds got wrong and detach
        it. Ordered worst-match first (the likeliest mistakes float to the top)."""
        conn = self._db()
        try:
            rows = conn.execute(
                """SELECT guest_id, embedding, sightings, thumb, thumb_box,
                          first_seen, last_seen
                   FROM guests WHERE promoted_user_id=? ORDER BY last_seen DESC""",
                (user_id,)).fetchall()
        finally:
            conn.close()
        out = []
        for gid, blob, sightings, thumb, box_raw, first_seen, last_seen in rows:
            emb = json.loads(blob)
            score = self._member_score_one(user_id, emb)
            score = round(score, 3) if score is not None else None
            # Which face the thumb is about (legacy full-person crops hold several) —
            # located up front on capture, or lazily here for old thumbs; so the
            # full-image viewer can ring exactly this cluster's face.
            face_box, no_face = self._face_box_for(gid, thumb, box_raw, emb)
            out.append({
                "guest_id": gid, "sightings": sightings,
                "has_thumb": thumb is not None,
                "first_seen": first_seen, "last_seen": last_seen,
                "score": score,
                "face_box": face_box, "no_face": no_face,
            })
        out.sort(key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else 0.0))
        return out

    def detach_cluster(self, guest_id: str) -> Optional[str]:
        """"This one wasn't me." Reverse a promote/auto-heal of a HOUSEHOLD member:
        clear the promotion + name so the cluster re-enters the review queue, and
        record the member in the cluster's rejected set so it never auto-heals back.
        Nothing to un-merge anymore: promotions stopped folding into the member
        centroid (see promote_guest) — the member profile was never touched.
        Returns the member id it was detached from, or None if the cluster isn't a
        member promotion (missing, or merged into a NAMED guest — handled elsewhere)."""
        with _lock:
            conn = self._db()
            try:
                row = conn.execute(
                    "SELECT embedding, promoted_user_id, rejected_user_ids FROM guests WHERE guest_id=?",
                    (guest_id,)).fetchone()
                if not row or not row[1]:
                    return None
                member = row[1]
                if member.startswith("guest:"):
                    return None  # merged into a NAMED guest, not a member — not our case
                rejected = self._parse_rejected(row[2])
                rejected.add(member)
                conn.execute(
                    """UPDATE guests SET promoted_user_id=NULL, name=NULL, rejected_user_ids=?,
                           last_seen=datetime('now') WHERE guest_id=?""",
                    (json.dumps(sorted(rejected)), guest_id))
                conn.commit()
                return member
            finally:
                conn.close()

    def merge_guests(self, src_id: str, dst_id: str) -> Optional[int]:
        """Fold cluster `src` into (named) guest `dst` — the guest-side twin of
        promote_guest, so confirming "yes, that's Abuela" teaches the system instead
        of leaving a duplicate. Sighting-weighted centroid merge (each cluster counts
        for how often it was seen), sightings sum, and src is tagged promoted-into-dst
        so it drops out of the roster + queue. dst keeps its thumb unless it has none
        (then src's rides over). Returns dst's new sightings, or None if either id is
        missing/already absorbed."""
        if src_id == dst_id:
            return None
        with _lock:
            conn = self._db()
            try:
                src = conn.execute(
                    """SELECT embedding, sightings, thumb, thumb_box FROM guests
                       WHERE guest_id=? AND promoted_user_id IS NULL""", (src_id,)).fetchone()
                dst = conn.execute(
                    """SELECT embedding, sightings FROM guests
                       WHERE guest_id=? AND promoted_user_id IS NULL""", (dst_id,)).fetchone()
                if not src or not dst:
                    return None
                s_emb, s_n = json.loads(src[0]), int(src[1])
                d_emb, d_n = json.loads(dst[0]), int(dst[1])
                total = s_n + d_n
                merged = [(d_emb[i] * d_n + s_emb[i] * s_n) / total for i in range(len(d_emb))]
                conn.execute(
                    """UPDATE guests SET embedding=?, sightings=?,
                           thumb=COALESCE(thumb, ?),
                           thumb_box=CASE WHEN thumb IS NULL THEN ? ELSE thumb_box END,
                           last_seen=datetime('now') WHERE guest_id=?""",
                    (json.dumps(merged), total, src[2], src[3], dst_id),
                )
                conn.execute("UPDATE guests SET promoted_user_id=? WHERE guest_id=?",
                             (dst_id, src_id))
                conn.commit()
                return total
            finally:
                conn.close()

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

    # ── review tiers (self-healing — see config face_autoheal_*/face_suggest_*) ──
    @staticmethod
    def _parse_rejected(raw: Optional[str]) -> set:
        try:
            return set(json.loads(raw)) if raw else set()
        except (ValueError, TypeError):
            return set()

    def reject_suggestion(self, guest_id: str, user_id: str) -> bool:
        """Record a "No, that's not them" answer for a cluster: it is never suggested
        as (or auto-healed into) that identity again — the cluster drops to the
        next-best identity or the everyone-reviews tier. `user_id` is a users.id OR a
        named guest's `guest:N` id (one rejected-set covers both kinds)."""
        with _lock:
            conn = self._db()
            try:
                row = conn.execute("SELECT rejected_user_ids FROM guests WHERE guest_id=?",
                                   (guest_id,)).fetchone()
                if not row:
                    return False
                rejected = self._parse_rejected(row[0])
                rejected.add(user_id)
                conn.execute("UPDATE guests SET rejected_user_ids=? WHERE guest_id=?",
                             (json.dumps(sorted(rejected)), guest_id))
                conn.commit()
                return True
            finally:
                conn.close()

    def _heal_eligible(self, guest_id: str) -> bool:
        """Maturity gate for SILENT auto-folds (the human review flow is never gated —
        a deliberate answer beats maturity): enough sightings, spread over enough
        wall-clock time, and internally coherent. Never promote a single frame —
        2026-07-07: ~110 mostly single-sighting clusters were auto-promoted in 24h,
        each one junk embedding wearing a member's name from then on."""
        conn = self._db()
        try:
            row = conn.execute(
                """SELECT sightings,
                          (julianday(last_seen) - julianday(first_seen)) * 86400.0,
                          embedding
                   FROM guests WHERE guest_id=?""", (guest_id,)).fetchone()
            if not row:
                return False
            sightings, span_s = int(row[0]), float(row[1] or 0.0)
            if sightings < max(1, cfg.face_autoheal_min_sightings):
                return False
            if span_s < cfg.face_autoheal_min_span_s:
                return False
            # Internal coherence: the cluster's recorded captures must agree with its
            # own centroid — a grab-bag of different faces (means of noise can drift
            # anywhere) goes to the human queue instead of silently becoming somebody.
            # Fewer than 3 recorded captures = nothing to judge → don't block (the
            # sightings/span bars above still hold; captures may be disabled).
            if cfg.face_autoheal_min_coherence > 0:
                embs = [json.loads(r[0]) for r in conn.execute(
                    """SELECT embedding FROM captures WHERE cluster_id=?
                       ORDER BY id DESC LIMIT 20""", (guest_id,))]
                if len(embs) >= 3:
                    centroid = json.loads(row[2])
                    coh = sum(_cosine(e, centroid) for e in embs) / len(embs)
                    if coh < cfg.face_autoheal_min_coherence:
                        return False
            return True
        finally:
            conn.close()

    def _maybe_autoheal(self, guest_id: str
                        ) -> Optional[Tuple[str, str, Optional[str], float]]:
        """Top tier of the self-healing ladder: if a MATURE (see _heal_eligible),
        unnamed, unpromoted cluster's centroid now matches a known identity
        decisively (≥ autoheal threshold AND unambiguous margin — same strictness
        posture as reinforce), fold it in silently: household member → promote,
        named guest → merge. Reports (kind, id, name, score). Named clusters are
        deliberate labels and are never auto-merged AWAY; rejected identities are
        never healed into."""
        conn = self._db()
        try:
            row = conn.execute(
                """SELECT embedding, rejected_user_ids, name FROM guests
                   WHERE guest_id=? AND promoted_user_id IS NULL""", (guest_id,)).fetchone()
        finally:
            conn.close()
        if not row or row[2] is not None:
            return None
        if self.folds_frozen:  # smear alarm: folding would deepen contamination
            return None
        if not self._heal_eligible(guest_id):
            return None
        kind, tid, tname, score, margin = self._best_identity(
            json.loads(row[0]), exclude=self._parse_rejected(row[1]),
            exclude_guest=guest_id)
        if (tid is None or score < self._thr("face_autoheal_threshold")
                or margin < self._thr("face_autoheal_margin")):
            return None
        if kind == "member":
            self.promote_guest(guest_id, tid, tname, carry_thumb=False)
        elif self.merge_guests(guest_id, tid) is None:
            return None
        return kind, tid, tname, score

    def review_queue(self) -> dict:
        """The "Is this you?" queue. Scores every unpromoted, unnamed cluster against
        EVERY known identity — household members and named guests — and buckets it:
          * clears the autoheal tier → folded into that identity right here
            (member: promote; named guest: merge), reported under `healed`, never queued;
          * ≥ suggest threshold → queued with a `suggested` identity ("probably them" —
            member cards are addressed to that member; guest cards to everyone);
          * below → queued with `suggested: None` (everyone reviews).
        Runs on read so the backlog re-buckets as centroids sharpen over time."""
        conn = self._db()
        try:
            rows = conn.execute(
                """SELECT guest_id, name, sightings, first_seen, last_seen, embedding,
                          rejected_user_ids, thumb, thumb_box
                   FROM guests WHERE promoted_user_id IS NULL AND name IS NULL
                   ORDER BY last_seen DESC""").fetchall()
        finally:
            conn.close()
        queue, healed = [], []
        frozen = self.folds_frozen  # smear alarm: queue keeps working, heals don't
        for gid, name, sightings, first_seen, last_seen, blob, rejected_raw, thumb, box_raw in rows:
            emb = json.loads(blob)
            rejected = self._parse_rejected(rejected_raw)
            kind, tid, tname, score, margin = self._best_identity(
                emb, exclude=rejected, exclude_guest=gid)
            if (not frozen
                    and tid is not None and score >= self._thr("face_autoheal_threshold")
                    and margin >= self._thr("face_autoheal_margin")
                    and self._heal_eligible(gid)):
                if kind == "member":
                    self.promote_guest(gid, tid, tname, carry_thumb=False)
                else:
                    self.merge_guests(gid, tid)
                healed.append({"guest_id": gid, "kind": kind, "id": tid,
                               "name": tname, "score": round(score, 3)})
                continue
            suggested = None
            if tid is not None and score >= self._thr("face_suggest_threshold"):
                suggested = {"kind": kind, "id": tid, "name": tname,
                             "score": round(score, 3)}
            face_box, no_face = self._face_box_for(gid, thumb, box_raw, emb)
            queue.append({
                "guest_id": gid,
                "label": self.default_label(gid),
                "sightings": sightings,
                "first_seen": first_seen, "last_seen": last_seen,
                "has_thumb": thumb is not None,
                "face_box": face_box,
                # True = the detector LOOKED at this thumb and found no face at all
                # (blurry/cut-off legacy crop) — the card can say so honestly instead
                # of presenting an unanswerable photo. Such thumbs self-replace on
                # the cluster's next face-located sighting.
                "no_face": no_face,
                "tier": "suggest" if suggested else "unknown",
                "suggested": suggested,
                "rejected_user_ids": sorted(rejected),
            })
        return {"queue": queue, "healed": healed}

    def _face_box_for(self, guest_id: str, thumb: Optional[bytes],
                      box_raw: Optional[str],
                      centroid: List[float]) -> Tuple[Optional[List[float]], bool]:
        """(face_box, no_face) for a cluster's thumb — the face's normalized
        [x,y,w,h] within it, or None. New captures store it up front; a LEGACY thumb
        (full-person crop, box NULL) is annotated lazily via the injected
        thumb_annotator — the face closest to the cluster centroid is the one the
        card is about — and the result is cached ([] = looked, none found → no_face
        True) so detection runs at most once per thumb."""
        if thumb is None:
            return None, False
        if box_raw is not None:
            try:
                box = json.loads(box_raw)
            except (ValueError, TypeError):
                return None, False
            return (box or None), box == []
        if self.thumb_annotator is None:
            return None, False
        try:
            box = self.thumb_annotator(thumb, centroid)
        except Exception:  # noqa: BLE001 — annotation is cosmetic, never break review
            return None, False
        if box is None:  # engine unavailable right now — retry on a later read
            return None, False
        with _lock:
            conn = self._db()
            try:
                conn.execute("UPDATE guests SET thumb_box=? WHERE guest_id=?",
                             (json.dumps(box), guest_id))
                conn.commit()
            finally:
                conn.close()
        return (box or None), box == []

    def recheck(self, emb: List[float]) -> Optional[Identity]:
        """Side-effect-free household re-verification for a track's CACHED label.
        The camera worker resolves a face once per track and caches it — but the
        tracker can swap two tracks' ids when people cross paths, leaving each person
        wearing the other's label until the tracks die. Called periodically per live
        household-labelled track: returns the Identity a fresh embedding decisively
        matches (same match+margin gate as resolve), or None when the frame is not
        decisive (bad angle/blur — keep the cached label; NEVER clusters a guest,
        never reinforces, so a poor re-check frame can't disturb the gallery)."""
        match_thr = self._thr("face_match_threshold")
        uid, name, score, margin = self._best_household(emb)
        if (uid is not None and score >= match_thr
                and margin >= self._thr("face_match_margin")):
            return Identity(id=uid, name=name, cls="household",
                            confidence=_confidence(score, match_thr))
        return None

    # ── the resolver the pipeline calls per new/unmatched track ──────────────
    def resolve(self, emb: List[float], thumb: Optional[bytes] = None,
                thumb_box: Optional[List[float]] = None) -> Identity:
        """Embedding → Identity. Household match above threshold wins; otherwise the
        guest pipeline clusters and returns a (possibly named) guest. Every distinct
        person therefore gets a stable default id (`guest:N`); `thumb` (the captured
        face crop) is stored so the dashboard can show their face (`thumb_box` = the
        face's normalized position within it). This is the one call the camera worker
        makes once per new/unmatched track (§4.1)."""
        match_thr = self._thr("face_match_threshold")
        uid, name, score, margin = self._best_household(emb)
        # Margin-gated: a face scoring 0.36-vs-0.34 between two members is AMBIGUOUS,
        # not a match — without this gate a close call gets labelled (and possibly
        # reinforced) as whoever happens to edge ahead, which is how two members'
        # centroids cross-contaminate and swap. Ambiguous faces take the guest path,
        # where the review ladder sorts them out with a human in the loop.
        if (uid is not None and score >= match_thr
                and margin >= self._thr("face_match_margin")):
            # Self-improve on a confident + unambiguous match (gated to prevent drift).
            reinforced = (cfg.face_reinforce
                          and score >= self._thr("face_reinforce_threshold")
                          and margin >= self._thr("face_reinforce_margin"))
            if reinforced:
                self._reinforce_household(uid, emb)
            self._capture("match", emb, thumb, thumb_box, resolved_id=uid,
                          score=score, reinforced=reinforced)
            return Identity(id=uid, name=name, cls="household",
                            confidence=_confidence(score, match_thr))
        gid, gname, sightings, promoted, cscore = self._cluster_guest(emb, thumb, thumb_box)
        # A PROMOTED cluster answers as the household member it was folded into — the
        # promotion ("this cluster IS this member", human-confirmed or auto-healed)
        # outranks the household gallery's strict gate, which a far/angled camera can
        # fail forever. Without this, the member keeps resolving cls="guest" (confidence
        # capped 0.6) from that camera, AND the rich-get-richer trap locks it in: the
        # cluster centroid updates on every sighting while the household centroid only
        # reinforces on confident household matches. So we also (gated) reinforce the
        # member's gallery with the live embedding, letting the household centroid
        # converge on the look this camera actually sees.
        if promoted is not None:
            # The promotion may only speak (and teach) when the LIVE face isn't
            # ambiguous BETWEEN members: the promoted member must beat every other
            # member on this embedding by the match margin. Without this gate a
            # wrongly-promoted or coin-flip cluster (centroid sitting between two
            # members — verified live: clusters scoring 0.404-vs-0.406) stamps its
            # member name on WHOEVER walks by, and the ungated reinforce below then
            # folds person A's embedding into member B's centroid — the very
            # cross-contamination that makes two members' centroids converge and
            # their labels swap whenever they share a room. Note the promoted member
            # does NOT need to clear the absolute match threshold (the whole point of
            # the promoted path is the far/angled camera that never will); it only
            # must not be in a dead heat with someone else.
            own = self._member_score_one(promoted, emb)
            o_uid, _o_name, o_score, _ = self._best_household(emb, exclude={promoted})
            ambiguous = (own is not None and o_uid is not None
                         and own - o_score < self._thr("face_match_margin"))
            if ambiguous:
                # Answer as an anonymous guest sighting — "someone is here", no name —
                # instead of asserting a 50/50 identity. The 20s re-verify upgrades the
                # label as soon as a frame reads decisively.
                self._capture("ambiguous", emb, thumb, thumb_box, cluster_id=gid,
                              score=cscore)
                return Identity(id=gid, name=None, cls="guest",
                                confidence=min(0.6, 0.3 + 0.05 * sightings))
            reinforced = (cfg.face_reinforce
                          and cscore >= self._thr("face_reinforce_threshold")
                          and (own is None or o_uid is None
                               or own - o_score >= self._thr("face_reinforce_margin")))
            if reinforced:
                self._reinforce_household(promoted, emb)
            self._capture("promoted", emb, thumb, thumb_box, resolved_id=promoted,
                          cluster_id=gid, score=cscore, reinforced=reinforced)
            return Identity(id=promoted, name=gname, cls="household",
                            confidence=_confidence(cscore, self._thr("guest_cluster_threshold")))
        # Self-healing top tier: the merged sighting may have pulled the cluster
        # centroid decisively onto a known identity — a household member OR a named
        # guest — fold it in and answer as them instead of a fresh "Person N" (works
        # live, without anyone opening the dashboard review queue).
        if gname is None:
            healed = self._maybe_autoheal(gid)
            if healed:
                hkind, hid, hname, hscore = healed
                self._capture("healed", emb, thumb, thumb_box, resolved_id=hid,
                              cluster_id=gid, score=hscore)
                if hkind == "member":
                    return Identity(id=hid, name=hname, cls="household",
                                    confidence=_confidence(hscore, match_thr))
                return Identity(id=hid, name=hname, cls="guest",
                                confidence=min(0.6, 0.3 + 0.05 * sightings))
        # A guest's confidence is modest by design — it's "we've seen this person
        # before", not "this is verified David". The agent stays polite/cautious.
        self._capture("cluster", emb, thumb, thumb_box, resolved_id=gid,
                      cluster_id=gid, score=cscore)
        return Identity(id=gid, name=gname, cls="guest",
                        confidence=min(0.6, 0.3 + 0.05 * sightings))
