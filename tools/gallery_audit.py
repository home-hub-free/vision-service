#!/usr/bin/env python3
"""Audit — and optionally detach — wrong/ambiguous member promotions in the gallery.

Why this exists (identity-swap debug, 2026-07-06): guest clusters promoted into the
WRONG household member (or sitting in a dead heat between two members) stamp their
member's name on whoever walks by, and — before the resolve() ambiguity gate shipped —
also reinforced that member's centroid with the other person's embeddings. Verified
live: david-vs-Ana centroid cosine 0.459 (two different people should sit ~0.0–0.25)
and coin-flip clusters like guest:54 (promoted→david, scores 0.404 david / 0.406 Ana).

For every cluster promoted into a household member this scores its centroid against
ALL member centroids and buckets it:

  WRONG      — matches a DIFFERENT member better than the one it was promoted into.
  AMBIGUOUS  — its own member wins, but by less than --margin (dead heat).
  ok         — its own member wins decisively.

Dry-run by default (prints the table, changes nothing). With --apply, WRONG and
AMBIGUOUS clusters are detached via Gallery.detach_cluster (un-merges the weight-1
fold from the member centroid and sends the cluster back to the review queue). The
"rejected" mark detach normally leaves (a human "not me" answer) is stripped unless
--keep-reject, since this tool's verdicts are statistical, not human answers.

Run from the vision-service dir, ideally while the house is quiet (the live service
writes the same sqlite file):

  .venv/bin/python tools/gallery_audit.py                # dry-run report
  .venv/bin/python tools/gallery_audit.py --apply        # detach WRONG + AMBIGUOUS
  .venv/bin/python tools/gallery_audit.py --apply --only-wrong
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.gallery import Gallery, _cosine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=None, help="gallery.db path (default: cfg.gallery_db)")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="own-vs-other margin under which a promotion is AMBIGUOUS (default 0.05)")
    ap.add_argument("--apply", action="store_true", help="detach flagged clusters (default: dry-run)")
    ap.add_argument("--only-wrong", action="store_true", help="with --apply: detach only WRONG, keep AMBIGUOUS")
    ap.add_argument("--keep-reject", action="store_true",
                    help="with --apply: keep the detach's rejected-member mark (blocks re-heal into them)")
    args = ap.parse_args()

    g = Gallery(args.db)
    conn = g._db()
    try:
        members = {uid: (name, json.loads(blob)) for uid, name, blob in
                   conn.execute("SELECT user_id, name, embedding FROM faces")}
        rows = conn.execute(
            """SELECT guest_id, promoted_user_id, sightings, embedding, last_seen
               FROM guests WHERE promoted_user_id IS NOT NULL""").fetchall()
    finally:
        conn.close()

    wrong, ambiguous, ok = [], [], []
    for gid, puid, sightings, blob, last_seen in rows:
        if puid not in members:
            continue  # merged into a named guest — out of scope
        emb = json.loads(blob)
        own = _cosine(emb, members[puid][1])
        others = [(name, _cosine(emb, m)) for uid, (name, m) in members.items() if uid != puid]
        o_name, o_score = max(others, key=lambda t: t[1]) if others else ("-", -math.inf)
        row = (gid, members[puid][0], sightings, own, o_name, o_score, last_seen)
        if o_score > own:
            wrong.append(row)
        elif own - o_score < args.margin:
            ambiguous.append(row)
        else:
            ok.append(row)

    def show(title, bucket):
        print(f"\n{title} ({len(bucket)}):")
        for gid, mname, sightings, own, o_name, o_score, last_seen in sorted(
                bucket, key=lambda r: r[3] - r[5]):
            print(f"  {gid:>10s} -> {mname:<8s} sightings={sightings:<4d} "
                  f"own={own:.3f} best_other={o_name}:{o_score:.3f} last={last_seen}")

    show("WRONG (matches another member better)", wrong)
    show(f"AMBIGUOUS (own wins by < {args.margin})", ambiguous)
    print(f"\nok: {len(ok)} promotions win decisively.")

    to_detach = wrong + ([] if args.only_wrong else ambiguous)
    if not args.apply:
        print(f"\nDry-run: would detach {len(to_detach)} cluster(s). Re-run with --apply.")
        return 0

    detached = 0
    for gid, mname, *_rest in to_detach:
        member = g.detach_cluster(gid)
        if member is None:
            print(f"  detach {gid}: skipped (no longer a member promotion)")
            continue
        detached += 1
        if not args.keep_reject:
            conn = g._db()
            try:
                raw = conn.execute("SELECT rejected_user_ids FROM guests WHERE guest_id=?",
                                   (gid,)).fetchone()
                rejected = g._parse_rejected(raw[0] if raw else None)
                rejected.discard(member)
                conn.execute("UPDATE guests SET rejected_user_ids=? WHERE guest_id=?",
                             (json.dumps(sorted(rejected)), gid))
                conn.commit()
            finally:
                conn.close()
    print(f"\nDetached {detached} cluster(s); they re-enter the review queue "
          f"(next /people/review read re-buckets them under the current thresholds).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
