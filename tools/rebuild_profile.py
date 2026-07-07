#!/usr/bin/env python3
"""Review the capture ledger and rebuild a member's face profile from curated crops.

The gallery's member centroids are running means — pollution folded in (a wrong
promotion, an ungated reinforce) can't be *removed*. But the capture ledger
(gallery.py `_capture`) permanently archives every crop + exact embedding behind
every identity decision, so a profile can always be REBUILT from reviewed
ingredients instead. The workflow:

  1. .venv/bin/python tools/rebuild_profile.py list             # who has captures
  2. .venv/bin/python tools/rebuild_profile.py export <id> DIR  # copy their crops out
  3.   → open DIR in any file manager, DELETE every crop that isn't them
  4. .venv/bin/python tools/rebuild_profile.py rebuild <user_id> --from-dir DIR --apply

`rebuild` replaces the member's centroid with the plain mean of the kept crops'
embeddings. Each kept file is matched back to its ledger row (filenames are unique),
so the EXACT live embedding is reused — no face engine needed; a file with no ledger
row is re-embedded with the real engine (and skipped, loudly, in a null build).
Dry-run by default; --apply writes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.gallery import Gallery  # noqa: E402


def _gallery(db: str | None) -> Gallery:
    return Gallery(db)


def cmd_list(g: Gallery) -> int:
    rows = g.captures()
    if not rows:
        print("ledger is empty (captures start accruing once the service runs this code)")
        return 0
    by_owner: dict = {}
    for r in rows:
        owner = r["resolved_id"] or r["cluster_id"] or "unknown"
        by_owner.setdefault(owner, []).append(r)
    print(f"{len(rows)} captures:")
    for owner, rs in sorted(by_owner.items(), key=lambda kv: -len(kv[1])):
        kinds: dict = {}
        for r in rs:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        kind_s = ", ".join(f"{k}:{n}" for k, n in sorted(kinds.items()))
        print(f"  {owner:>12s}  {len(rs):4d}  ({kind_s})  newest={rs[0]['ts']}")
    print(f"\ncrop files live under: {g.captures_dir}/<identity>/")
    return 0


def cmd_export(g: Gallery, owner: str, out_dir: str) -> int:
    rows = [r for r in g.captures(owner) if r["path"]]
    if not rows:
        print(f"no captures for {owner!r}")
        return 1
    os.makedirs(out_dir, exist_ok=True)
    copied = 0
    for r in rows:
        src = os.path.join(g.captures_dir, r["path"])
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, os.path.basename(r["path"])))
            copied += 1
    print(f"copied {copied}/{len(rows)} crops to {out_dir}")
    print("→ now DELETE every file that is not them, then run:")
    print(f"  tools/rebuild_profile.py rebuild {owner} --from-dir {out_dir} --apply")
    return 0


def _embedding_for(g: Gallery, fname: str) -> list | None:
    """Exact ledger embedding for an exported crop (filenames are ns-unique)."""
    conn = g._db()
    try:
        row = conn.execute("SELECT embedding FROM captures WHERE path LIKE ?",
                           (f"%{fname}",)).fetchone()
    finally:
        conn.close()
    return json.loads(row[0]) if row else None


def cmd_rebuild(g: Gallery, user_id: str, from_dir: str, name: str | None,
                apply: bool) -> int:
    files = sorted(f for f in os.listdir(from_dir) if f.lower().endswith(".jpg"))
    if not files:
        print(f"no .jpg crops in {from_dir}")
        return 1
    embs, reembedded, skipped = [], 0, []
    engine = None
    for fname in files:
        emb = _embedding_for(g, fname)
        if emb is None:  # not from the ledger — fall back to the real face engine
            if engine is None:
                from app.perception import _get_shared_face_engine
                engine = _get_shared_face_engine()
                if getattr(engine, "backend", "null") == "null":
                    print("real face engine unavailable — files without a ledger row "
                          "cannot be embedded and will be skipped")
            if getattr(engine, "backend", "null") != "null":
                from app.perception import enroll_embedding
                with open(os.path.join(from_dir, fname), "rb") as fh:
                    emb, reason = enroll_embedding(fh.read())
                if emb is not None:
                    reembedded += 1
                elif reason:
                    print(f"  {fname}: not enrollment-grade ({reason}) — skipped")
        if emb is None:
            skipped.append(fname)
            continue
        embs.append(emb)
    print(f"{len(embs)} embeddings from {len(files)} crops "
          f"({len(embs) - reembedded} exact from ledger, {reembedded} re-embedded, "
          f"{len(skipped)} skipped{': ' + ', '.join(skipped[:5]) if skipped else ''})")
    if not embs:
        return 1
    if not apply:
        print(f"dry-run: would REPLACE {user_id}'s centroid with the mean of "
              f"{len(embs)} curated embeddings. Re-run with --apply.")
        return 0
    samples = g.rebuild_member(user_id, embs, name=name)
    print(f"rebuilt {user_id}: fresh centroid from {samples} curated samples "
          f"(old centroid discarded). Restart is NOT needed — the gallery reads live.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=None, help="gallery.db path (default: cfg.gallery_db)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="capture counts per identity")
    p_exp = sub.add_parser("export", help="copy one identity's crops to a folder for review")
    p_exp.add_argument("owner")
    p_exp.add_argument("out_dir")
    p_reb = sub.add_parser("rebuild", help="replace a member centroid from curated crops")
    p_reb.add_argument("user_id")
    p_reb.add_argument("--from-dir", required=True)
    p_reb.add_argument("--name", default=None, help="display name (kept from row if omitted)")
    p_reb.add_argument("--apply", action="store_true", help="write (default: dry-run)")
    args = ap.parse_args()
    g = _gallery(args.db)
    if args.cmd == "list":
        return cmd_list(g)
    if args.cmd == "export":
        return cmd_export(g, args.owner, args.out_dir)
    return cmd_rebuild(g, args.user_id, args.from_dir, args.name, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
