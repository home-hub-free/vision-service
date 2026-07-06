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
        # Optional hook (wired at app startup to perception.annotate_face_in_thumb;
        # gallery itself stays cv2-free): (thumb_jpeg, centroid) → normalized face
        # box within the thumb, [] for "no face found", None for "engine unavailable".
        # Used to lazily locate the face in LEGACY person-crop thumbs at review time.
        self.thumb_annotator = None
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

    def _best_household(self, emb: List[float],
                        exclude: Optional[set] = None) -> Tuple[Optional[str], Optional[str], float, float]:
        """Best household match for an embedding. Returns
        (user_id, name, best_score, margin) where `margin` is best_score minus the
        2nd-best member's score — a measure of how UNAMBIGUOUS the match is (large when
        one member clearly wins; small when two members score alike). margin is +inf when
        there's only one enrolled member (nothing to confuse it with). `exclude` drops
        members a reviewer already answered "not me" for (review-tier rejections)."""
        conn = self._db()
        try:
            scored = []
            for uid, name, blob in conn.execute("SELECT user_id, name, embedding FROM faces"):
                if exclude and uid in exclude:
                    continue
                scored.append((_cosine(emb, json.loads(blob)), uid, name))
            if not scored:
                return (None, None, -1.0, 0.0)
            scored.sort(key=lambda t: t[0], reverse=True)
            best_s, best_uid, best_name = scored[0]
            margin = best_s - scored[1][0] if len(scored) > 1 else float("inf")
            return (best_uid, best_name, best_s, margin)
        finally:
            conn.close()

    def _reinforce_household(self, user_id: str, emb: List[float]) -> None:
        """Online learning: fold a confidently+unambiguously matched LIVE embedding into
        the member's centroid via running-mean, so passive day-to-day recognition keeps
        getting sharper without a manual re-enroll. The caller gates this on score/margin
        (see resolve); here we only bound the influence:
          * the running-mean weight is capped at `face_reinforce_cap`, so once a member is
            well-established each new frame nudges the centroid by ≤ 1/(cap+1) (a gentle
            EMA) — one bad crop can't yank the identity.
          * `samples` stops incrementing at the cap (purely cosmetic; the centroid keeps
            adapting). name + thumb are never touched (we keep the deliberate enroll face)."""
        emb = _normalise(emb)
        cap = max(1, cfg.face_reinforce_cap)
        with _lock:
            conn = self._db()
            try:
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
        conn = self._db()
        try:
            scored = []
            for uid, name, blob in conn.execute("SELECT user_id, name, embedding FROM faces"):
                if uid in exclude:
                    continue
                scored.append((_cosine(emb, json.loads(blob)), "member", uid, name))
            for gid, gname, blob in conn.execute(
                    "SELECT guest_id, name, embedding FROM guests "
                    "WHERE name IS NOT NULL AND promoted_user_id IS NULL"):
                if gid in exclude or gid == exclude_guest:
                    continue
                scored.append((_cosine(emb, json.loads(blob)), "guest", gid, gname))
            if not scored:
                return (None, None, None, -1.0, 0.0)
            scored.sort(key=lambda t: t[0], reverse=True)
            best_s, kind, best_id, best_name = scored[0]
            margin = best_s - scored[1][0] if len(scored) > 1 else float("inf")
            return (kind, best_id, best_name, best_s, margin)
        finally:
            conn.close()

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

    def promote_guest(self, guest_id: str, user_id: str, name: Optional[str],
                      carry_thumb: bool = True) -> bool:
        """Promote a recurring guest cluster into a named household member's gallery:
        its centroid seeds (or merges into) the user's face profile, and the guest row
        is tagged promoted so it stops surfacing for review. `carry_thumb=False` keeps
        the member's deliberate enroll portrait (auto-heal must not swap it for a
        random low-angle crop)."""
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
        self.enroll(user_id, name, json.loads(row[0]), thumb=row[1] if carry_thumb else None)
        return True

    def member_clusters(self, user_id: str) -> List[dict]:
        """Every guest cluster that was folded INTO a household member (by auto-heal or
        a manual promote) — the audit trail behind that member's face profile. Each row
        carries its captured thumb + how well it still matches the member's centroid
        (`score`), so a reviewer can spot an outlier the thresholds got wrong and detach
        it. Ordered worst-match first (the likeliest mistakes float to the top)."""
        member = self._best_member_embedding(user_id)
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
            score = round(_cosine(emb, member), 3) if member else None
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

    def _best_member_embedding(self, user_id: str) -> Optional[List[float]]:
        conn = self._db()
        try:
            row = conn.execute("SELECT embedding FROM faces WHERE user_id=?", (user_id,)).fetchone()
        finally:
            conn.close()
        return json.loads(row[0]) if row else None

    def detach_cluster(self, guest_id: str) -> Optional[str]:
        """"This one wasn't me." Reverse an auto-heal/promote of a HOUSEHOLD member:
          * best-effort un-merge the cluster's contribution from the member centroid
            (weight 1 — the inverse of how promote_guest folded it in via enroll's
            running-mean; a single cluster among many barely moves a seasoned centroid,
            which is intended: one bad fold shouldn't wreck an identity),
          * clear the promotion + name so the cluster re-enters the review queue,
          * record the member in the cluster's rejected set so it never auto-heals back.
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
                face = conn.execute(
                    "SELECT embedding, samples FROM faces WHERE user_id=?", (member,)).fetchone()
                if not face:
                    return None  # promoted into a named guest, not a member — not our case
                c_emb = json.loads(row[0])
                m_emb, m_n = json.loads(face[0]), int(face[1])
                if m_n > 1:
                    # inverse running-mean: recover the centroid before this weight-1 fold.
                    unmerged = _normalise([(m_emb[i] * m_n - c_emb[i]) / (m_n - 1)
                                           for i in range(len(m_emb))])
                    conn.execute(
                        "UPDATE faces SET embedding=?, samples=?, updated_at=datetime('now') WHERE user_id=?",
                        (json.dumps(unmerged), m_n - 1, member))
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

    def _maybe_autoheal(self, guest_id: str
                        ) -> Optional[Tuple[str, str, Optional[str], float]]:
        """Top tier of the self-healing ladder: if an UNNAMED, unpromoted cluster's
        centroid now matches a known identity decisively (≥ autoheal threshold AND
        unambiguous margin — same strictness posture as reinforce), fold it in
        silently: household member → promote, named guest → merge. Reports
        (kind, id, name, score). Named clusters are deliberate labels and are never
        auto-merged AWAY; rejected identities are never healed into."""
        conn = self._db()
        try:
            row = conn.execute(
                """SELECT embedding, rejected_user_ids, name FROM guests
                   WHERE guest_id=? AND promoted_user_id IS NULL""", (guest_id,)).fetchone()
        finally:
            conn.close()
        if not row or row[2] is not None:
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
        for gid, name, sightings, first_seen, last_seen, blob, rejected_raw, thumb, box_raw in rows:
            emb = json.loads(blob)
            rejected = self._parse_rejected(rejected_raw)
            kind, tid, tname, score, margin = self._best_identity(
                emb, exclude=rejected, exclude_guest=gid)
            if (tid is not None and score >= self._thr("face_autoheal_threshold")
                    and margin >= self._thr("face_autoheal_margin")):
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
            if (cfg.face_reinforce and score >= self._thr("face_reinforce_threshold")
                    and margin >= self._thr("face_reinforce_margin")):
                self._reinforce_household(uid, emb)
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
            own_emb = self._best_member_embedding(promoted)
            own = _cosine(emb, own_emb) if own_emb is not None else None
            o_uid, _o_name, o_score, _ = self._best_household(emb, exclude={promoted})
            ambiguous = (own is not None and o_uid is not None
                         and own - o_score < self._thr("face_match_margin"))
            if ambiguous:
                # Answer as an anonymous guest sighting — "someone is here", no name —
                # instead of asserting a 50/50 identity. The 20s re-verify upgrades the
                # label as soon as a frame reads decisively.
                return Identity(id=gid, name=None, cls="guest",
                                confidence=min(0.6, 0.3 + 0.05 * sightings))
            if (cfg.face_reinforce and cscore >= self._thr("face_reinforce_threshold")
                    and (own is None or o_uid is None
                         or own - o_score >= self._thr("face_reinforce_margin"))):
                self._reinforce_household(promoted, emb)
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
                if hkind == "member":
                    return Identity(id=hid, name=hname, cls="household",
                                    confidence=_confidence(hscore, match_thr))
                return Identity(id=hid, name=hname, cls="guest",
                                confidence=min(0.6, 0.3 + 0.05 * sightings))
        # A guest's confidence is modest by design — it's "we've seen this person
        # before", not "this is verified David". The agent stays polite/cautious.
        return Identity(id=gid, name=gname, cls="guest",
                        confidence=min(0.6, 0.3 + 0.05 * sightings))
