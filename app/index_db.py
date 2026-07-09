"""Recordings + event index (§9.4) — the join that makes video answerable history.

A lightweight sqlite index in the vision-service. Two tables on one common clock:

  * `segments` — `{camId, zone, start, end, file}`: each archive mp4 segment.
  * `events`   — every salient occupancy/identity edge (`person_entered`,
    `identified=David`, `guest_arrived`, …) with its timestamp.

This is what lets the agent answer "who came by today?" / "was anyone in the kitchen
at 3pm?" from the INDEX — never by watching frames (§7, §9.4) — and lets the dashboard
jump from a timeline marker straight to the segment that contains it.

DECISION (§9.6/§11.4): whether this index ALSO pushes to memory-service or stays
vision-service-local. Default here: local (the events already reach memory via the
§5.2 MQTT lane; this index just adds the segment pointer). A `to_memory` hook is left
as a clearly-marked stub for the decision.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import List, Optional

from .config import cfg
from .occupancy import Edge

_lock = threading.Lock()


class EventIndex:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or cfg.index_db
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init()

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init(self) -> None:
        conn = self._db()
        try:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS segments (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       cam_id TEXT NOT NULL, zone TEXT,
                       start_ts REAL NOT NULL, end_ts REAL, file TEXT NOT NULL)"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS events (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       cam_id TEXT NOT NULL, zone TEXT, ts REAL NOT NULL,
                       edge TEXT NOT NULL,
                       identity_id TEXT, identity_name TEXT, identity_class TEXT)"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_cam ON segments(cam_id, start_ts)")
            conn.commit()
        finally:
            conn.close()

    # ── writes ────────────────────────────────────────────────────────────────
    def record_event(self, edge: Edge) -> None:
        with _lock:
            conn = self._db()
            try:
                conn.execute(
                    """INSERT INTO events (cam_id, zone, ts, edge, identity_id, identity_name, identity_class)
                       VALUES (?,?,?,?,?,?,?)""",
                    (edge.cam_id, edge.zone, edge.ts, edge.edge,
                     edge.identity.id, edge.identity.name, edge.identity.cls),
                )
                conn.commit()
            finally:
                conn.close()
        self._to_memory(edge)  # DECISION stub (see module docstring / DECISIONS.md)

    def open_segment(self, cam_id: str, zone: str, file: str, start_ts: Optional[float] = None) -> int:
        with _lock:
            conn = self._db()
            try:
                cur = conn.execute(
                    "INSERT INTO segments (cam_id, zone, start_ts, end_ts, file) VALUES (?,?,?,?,?)",
                    (cam_id, zone, start_ts or time.time(), None, file),
                )
                conn.commit()
                return cur.lastrowid
            finally:
                conn.close()

    def close_segment(self, seg_id: int, end_ts: Optional[float] = None) -> None:
        with _lock:
            conn = self._db()
            try:
                conn.execute("UPDATE segments SET end_ts=? WHERE id=?", (end_ts or time.time(), seg_id))
                conn.commit()
            finally:
                conn.close()

    def sync_file_segments(self, cam_id: str, zone: str, rec_dir: str,
                           entries, protect=()) -> int:
        """Reconcile one camera's rows with its on-disk files (footage.py): insert
        a row per new finished mp4, purge the legacy recorder-run rows that
        pointed at the DIRECTORY (unplayable + never-closing — see footage.py),
        and purge rows under THIS rec_dir whose file the scan no longer vouches
        for (deleted out-of-band, or a stranded moov-less chunk footage.py now
        excludes) — `protect` shields still-settling files from that purge.
        One transaction under the module lock so concurrent route calls can't
        double-insert a file. Returns how many rows were added."""
        with _lock:
            conn = self._db()
            try:
                # Legacy rows point at a DIRECTORY — usually this camera's rec_dir,
                # but a renamed camera leaves rows aimed at its OLD dir (seen live:
                # mc200 rows → recordings/mc200-entrance), so purge by shape.
                conn.execute("DELETE FROM segments WHERE cam_id=? AND file NOT LIKE '%.mp4'",
                             (cam_id,))
                keep = {p for p, _s, _e in entries} | set(protect)
                prefix = rec_dir.rstrip(os.sep) + os.sep
                for (f,) in conn.execute(
                        "SELECT file FROM segments WHERE cam_id=? AND file LIKE ?",
                        (cam_id, prefix + "%")).fetchall():
                    if f not in keep:
                        conn.execute("DELETE FROM segments WHERE cam_id=? AND file=?",
                                     (cam_id, f))
                have = {r[0] for r in conn.execute(
                    "SELECT file FROM segments WHERE cam_id=?", (cam_id,))}
                added = 0
                for path, start_ts, end_ts in entries:
                    if path in have:
                        continue
                    conn.execute(
                        "INSERT INTO segments (cam_id, zone, start_ts, end_ts, file) VALUES (?,?,?,?,?)",
                        (cam_id, zone, start_ts, end_ts, path),
                    )
                    added += 1
                conn.commit()
                return added
            finally:
                conn.close()

    def prune_segment(self, file: str) -> None:
        """Drop the index row when the retention janitor deletes a file."""
        with _lock:
            conn = self._db()
            try:
                conn.execute("DELETE FROM segments WHERE file=?", (file,))
                conn.commit()
            finally:
                conn.close()

    # ── reads (agent history surface — §7/§9.4) ──────────────────────────────
    def who_came_by(self, since_ts: float) -> List[dict]:
        conn = self._db()
        try:
            rows = conn.execute(
                """SELECT identity_id, identity_name, identity_class,
                          MIN(ts) AS first_ts, MAX(ts) AS last_ts, COUNT(*) AS n
                   FROM events
                   WHERE ts >= ? AND edge IN ('person_identified','guest_arrived','person_entered')
                   GROUP BY COALESCE(identity_id, identity_class)
                   ORDER BY last_ts DESC""",
                (since_ts,),
            ).fetchall()
            return [{"id": r[0], "name": r[1], "class": r[2],
                     "first_seen": r[3], "last_seen": r[4], "count": r[5]} for r in rows]
        finally:
            conn.close()

    def events_between(self, start_ts: float, end_ts: float, zone: Optional[str] = None) -> List[dict]:
        conn = self._db()
        try:
            q = "SELECT cam_id, zone, ts, edge, identity_id, identity_name, identity_class FROM events WHERE ts BETWEEN ? AND ?"
            args: list = [start_ts, end_ts]
            if zone:
                q += " AND zone=?"
                args.append(zone)
            q += " ORDER BY ts ASC"
            rows = conn.execute(q, args).fetchall()
            return [{"cam_id": r[0], "zone": r[1], "ts": r[2], "edge": r[3],
                     "identity": {"id": r[4], "name": r[5], "class": r[6]}} for r in rows]
        finally:
            conn.close()

    def segments_between(self, cam_id: str, start_ts: float, end_ts: float) -> List[dict]:
        """Every archived segment for a camera overlapping [start_ts, end_ts] — the
        footage-review list (§9.5). A segment overlaps the window when it starts before
        the window ends and ends after the window starts (an open/still-recording segment
        has end_ts NULL → treat it as ongoing = overlaps). `duration` is None while open."""
        conn = self._db()
        try:
            rows = conn.execute(
                """SELECT id, start_ts, end_ts, file FROM segments
                   WHERE cam_id=? AND start_ts <= ?
                     AND (end_ts IS NULL OR end_ts >= ?)
                   ORDER BY start_ts ASC""",
                (cam_id, end_ts, start_ts),
            ).fetchall()
            return [{"id": r[0], "start": r[1], "end": r[2], "file": r[3],
                     "duration": (r[2] - r[1]) if r[2] is not None else None} for r in rows]
        finally:
            conn.close()

    def recording_days(self, cam_id: str) -> List[str]:
        """Distinct LOCAL days (YYYY-MM-DD) that have footage for a camera, newest first —
        the day picker. Grouped in SQLite on the segment start (localtime) so the list is
        cheap even with weeks of 5-min segments."""
        conn = self._db()
        try:
            rows = conn.execute(
                """SELECT DISTINCT date(start_ts, 'unixepoch', 'localtime') AS day
                   FROM segments WHERE cam_id=? ORDER BY day DESC""",
                (cam_id,),
            ).fetchall()
            return [r[0] for r in rows if r[0]]
        finally:
            conn.close()

    def segment_by_id(self, seg_id: int) -> Optional[dict]:
        """Resolve a segment id → its row (for the clip route). Returns None if pruned."""
        conn = self._db()
        try:
            r = conn.execute(
                "SELECT id, cam_id, start_ts, end_ts, file FROM segments WHERE id=?",
                (seg_id,),
            ).fetchone()
            return {"id": r[0], "cam_id": r[1], "start": r[2], "end": r[3], "file": r[4]} if r else None
        finally:
            conn.close()

    def recent_segments(self, cam_id: str, limit: int = 200) -> List[dict]:
        """Newest-first segment rows for one camera (thumb pregeneration order —
        the reviewer lands on recent footage first)."""
        conn = self._db()
        try:
            rows = conn.execute(
                """SELECT id, cam_id, start_ts, end_ts, file FROM segments
                   WHERE cam_id=? AND file LIKE '%.mp4'
                   ORDER BY start_ts DESC LIMIT ?""",
                (cam_id, int(limit)),
            ).fetchall()
            return [{"id": r[0], "cam_id": r[1], "start": r[2], "end": r[3], "file": r[4]}
                    for r in rows]
        finally:
            conn.close()

    def segment_at(self, cam_id: str, ts: float) -> Optional[dict]:
        conn = self._db()
        try:
            r = conn.execute(
                """SELECT file, start_ts, end_ts FROM segments
                   WHERE cam_id=? AND start_ts <= ? AND (end_ts IS NULL OR end_ts >= ?)
                   ORDER BY start_ts DESC LIMIT 1""",
                (cam_id, ts, ts),
            ).fetchone()
            return {"file": r[0], "start_ts": r[1], "end_ts": r[2]} if r else None
        finally:
            conn.close()

    def _to_memory(self, edge: Edge) -> None:
        """DECISION STUB (§9.6/§11.4): also push the indexed event (with its segment
        pointer) to memory-service? Today the edge already reaches memory via the §5.2
        MQTT lane, so this is a no-op. If the segment pointer needs to live in
        memory-service too, POST it here. Left intentionally empty."""
        return
