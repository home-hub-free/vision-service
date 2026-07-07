"""Gallery — household match, guest clustering, promote. Plain vectors, no models."""
import os
import tempfile

from app.config import cfg
from app.gallery import Gallery


def _g():
    d = tempfile.mkdtemp()
    return Gallery(os.path.join(d, "gallery.db"))


def _vec(seed: float, dim: int = 16):
    # Distinct seeds -> ~orthogonal directions (one-hot-ish), same seed -> identical.
    # So an identical embedding cosine-matches and distinct ones don't (cosine ~0).
    v = [0.01] * dim
    v[int(seed) % dim] = 1.0
    return v


def test_household_enroll_and_resolve():
    cfg.face_match_threshold = 0.9
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    ident = g.resolve(_vec(1.0))
    assert ident.cls == "household" and ident.id == "u1" and ident.name == "David"
    assert ident.via == "face" and ident.confidence > 0.7


def test_unknown_becomes_guest_then_recurring():
    cfg.face_match_threshold = 0.99
    cfg.guest_cluster_threshold = 0.999999
    cfg.guest_min_sightings = 3
    g = _g()
    v = _vec(2.0)
    ident = g.resolve(v)
    assert ident.cls == "guest" and ident.id == "guest:1"
    g.resolve(v)
    g.resolve(v)  # 3rd sighting → recurring
    recurring = g.guests(recurring_only=True)
    assert any(x["guest_id"] == "guest:1" and x["recurring"] for x in recurring)


def test_promote_guest_seeds_household_gallery():
    cfg.face_match_threshold = 0.9
    cfg.guest_cluster_threshold = 0.999999
    g = _g()
    v = _vec(3.0)
    g.resolve(v)  # creates guest:1
    assert g.promote_guest("guest:1", "u9", "Sam")
    ident = g.resolve(v)  # now matches the promoted household member
    assert ident.cls == "household" and ident.id == "u9" and ident.name == "Sam"


def test_promoted_cluster_match_answers_as_household_member():
    """The david-as-guest bug: a live embedding that FAILS the strict household gate
    (far/angled camera) but still matches a PROMOTED cluster must answer as the member
    (cls household, their id), not as a guest. The member's face PROFILE stays
    untouched — promotion is routing, never a fold (the 2026-07-07 pollution lesson:
    folding cluster centroids into members at enroll weight is how two members'
    centroids converged and swapped)."""
    cfg.face_match_threshold = 0.99   # household gate the drifted vector can't clear
    cfg.guest_cluster_threshold = 0.9
    cfg.face_reinforce = True
    cfg.face_reinforce_threshold = 0.9
    g = _g()
    v = _vec(3.0)
    g.resolve(v)  # creates guest:1
    assert g.promote_guest("guest:1", "u9", "Sam")
    drifted = _vec(3.0)
    drifted[4] = 0.33  # cosine ~0.95 to v: fails 0.99 household gate, clears 0.9 cluster
    ident = g.resolve(drifted)
    assert ident.cls == "household" and ident.id == "u9" and ident.name == "Sam"
    assert ident.confidence >= 0.6  # passes the agent's naming gate
    # promotion created NO face profile — the member is recognized via the cluster only
    assert g._best_household(drifted)[0] is None


def test_forget_removes_profile():
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    g.forget("u1")
    assert g.profiles() == []


def test_default_label_and_thumb_stored_for_every_person():
    cfg.face_match_threshold = 0.99
    cfg.guest_cluster_threshold = 0.999999
    g = _g()
    # A detected unknown is labelled by default AND keeps its captured face crop.
    ident = g.resolve(_vec(5.0), thumb=b"\xff\xd8FACEJPEG\xff\xd9")
    assert ident.cls == "guest" and ident.id == "guest:1"
    assert g.default_label("guest:1") == "Person 1"
    assert g.get_thumb("guest:1") == b"\xff\xd8FACEJPEG\xff\xd9"


def test_people_roster_lists_household_and_guests_with_labels():
    cfg.face_match_threshold = 0.99
    cfg.guest_cluster_threshold = 0.999999
    g = _g()
    g.enroll("u1", "David", _vec(1.0), thumb=b"DAVIDJPEG")
    g.resolve(_vec(6.0), thumb=b"GUESTJPEG")  # guest:1, unnamed → "Person 1"
    people = {p["id"]: p for p in g.people()}
    assert people["u1"]["label"] == "David" and people["u1"]["class"] == "household"
    assert people["u1"]["has_thumb"] and people["u1"]["named"]
    assert people["guest:1"]["label"] == "Person 1" and people["guest:1"]["class"] == "guest"
    assert people["guest:1"]["has_thumb"] and not people["guest:1"]["named"]


def test_promoted_guest_drops_from_people_and_carries_face():
    cfg.face_match_threshold = 0.9
    cfg.guest_cluster_threshold = 0.999999
    g = _g()
    g.resolve(_vec(7.0), thumb=b"GUESTFACE")  # guest:1
    g.promote_guest("guest:1", "u9", "Sam")
    ids = {p["id"] for p in g.people()}
    assert "guest:1" not in ids and "u9" in ids       # promoted guest no longer a guest row
    u9 = next(p for p in g.people() if p["id"] == "u9")
    assert u9["class"] == "household" and u9["samples"] == 0   # roster entry, not enrolled
    assert g.get_thumb("u9") == b"GUESTFACE"           # cluster's face LENT at read time
    conn = g._db()
    try:  # promotion never writes the faces table (the fold was the pollution engine)
        assert conn.execute("SELECT COUNT(*) FROM faces WHERE user_id='u9'").fetchone()[0] == 0
    finally:
        conn.close()


def test_online_reinforcement_strengthens_household_on_confident_match():
    cfg.face_match_threshold = 0.3
    cfg.face_reinforce = True
    cfg.face_reinforce_threshold = 0.5
    cfg.face_reinforce_margin = 0.05
    cfg.face_reinforce_cap = 50
    g = _g()
    g.enroll("u1", "David", _vec(1.0))            # samples = 1
    assert g.profiles()[0]["samples"] == 1
    ident = g.resolve(_vec(1.0))                  # confident match, sole member (margin=inf)
    assert ident.cls == "household" and ident.id == "u1"
    assert g.profiles()[0]["samples"] == 2        # ← passive recognition reinforced the centroid
    g.resolve(_vec(1.0))
    assert g.profiles()[0]["samples"] == 3        # keeps self-improving on each confident sighting


def test_reinforcement_skips_ambiguous_match_and_respects_toggle():
    cfg.face_match_threshold = 0.3
    cfg.face_reinforce = True
    cfg.face_reinforce_threshold = 0.5
    cfg.face_reinforce_margin = 0.05
    cfg.face_reinforce_cap = 50
    g = _g()
    # Two members with identical embeddings → a tie (margin 0): a look-alike. Reinforcement
    # must NOT fire — drifting one centroid on a 50/50 call is exactly the failure to avoid.
    g.enroll("u1", "Ann", _vec(1.0))
    g.enroll("u2", "Bea", _vec(1.0))
    g.resolve(_vec(1.0))
    assert all(p["samples"] == 1 for p in g.profiles())   # neither reinforced

    # Kill-switch: disabled → never reinforce, even on a clean sole-member match.
    cfg.face_reinforce = False
    g2 = _g()
    g2.enroll("u1", "David", _vec(1.0))
    g2.resolve(_vec(1.0))
    assert g2.profiles()[0]["samples"] == 1
    cfg.face_reinforce = True   # restore default for any later tests


# ── review tiers (self-healing ladder: autoheal / suggest / unknown) ─────────

def _mix(a_seed: float, b_seed: float, w: float, dim: int = 16):
    # A vector whose cosine to _vec(a_seed) is ≈ w (mixture of two ~orthogonal
    # directions) — crafts "definitely / probably / no idea" similarity levels.
    va, vb = _vec(a_seed, dim), _vec(b_seed, dim)
    k = (1 - w * w) ** 0.5
    return [w * x + k * y for x, y in zip(va, vb)]


def _set_tiers(match=0.99, cluster=0.999999, heal=0.9, margin=0.05, suggest=0.4):
    cfg.face_match_threshold = match
    cfg.guest_cluster_threshold = cluster
    cfg.face_autoheal_threshold = heal
    cfg.face_autoheal_margin = margin
    cfg.face_suggest_threshold = suggest


def test_review_queue_buckets_by_confidence_tier():
    # Backlog scenario: clusters created while the bar was higher (or before the
    # member's centroid sharpened) re-bucket on read — the top tier heals silently.
    _set_tiers(heal=0.999)   # nothing autoheals at resolve time
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    g.resolve(_mix(1.0, 2.0, 0.95))   # guest:1 — will be "definitely David"
    g.resolve(_mix(1.0, 3.0, 0.6))    # guest:2 — "probably David"
    g.resolve(_vec(4.0))              # guest:3 — "no idea"
    cfg.face_autoheal_threshold = 0.9
    out = g.review_queue()
    assert [h["guest_id"] for h in out["healed"]] == ["guest:1"]
    assert out["healed"][0]["kind"] == "member" and out["healed"][0]["id"] == "u1"
    by_id = {c["guest_id"]: c for c in out["queue"]}
    assert set(by_id) == {"guest:2", "guest:3"}
    assert by_id["guest:2"]["tier"] == "suggest"
    assert by_id["guest:2"]["suggested"]["kind"] == "member"
    assert by_id["guest:2"]["suggested"]["id"] == "u1"
    assert by_id["guest:2"]["suggested"]["name"] == "David"
    assert by_id["guest:3"]["tier"] == "unknown" and by_id["guest:3"]["suggested"] is None
    # The healed cluster merged into the member and left the roster.
    assert "guest:1" not in {p["id"] for p in g.people()}
    _set_tiers(heal=0.5, margin=0.08, suggest=0.2)  # restore defaults


def test_resolve_autoheals_cluster_that_drifts_onto_member():
    # Live path: single frames never clear the direct match bar, but merged
    # sightings pull the cluster centroid decisively onto the member → resolve
    # answers household mid-stream, keeping the member's enrolled portrait.
    _set_tiers(cluster=0.5, heal=0.93)
    g = _g()
    g.enroll("u1", "David", _vec(1.0), thumb=b"PORTRAIT")
    first = g.resolve(_mix(1.0, 2.0, 0.9), thumb=b"CROP1")     # below heal bar → guest
    assert first.cls == "guest" and first.id == "guest:1"
    second = g.resolve(_mix(1.0, 3.0, 0.93), thumb=b"CROP2")   # joins cluster, centroid ≥ bar
    assert second.cls == "household" and second.id == "u1" and second.name == "David"
    assert g.get_thumb("u1") == b"PORTRAIT"                    # autoheal never swaps the portrait
    assert g.review_queue()["queue"] == []                     # nothing left to review
    _set_tiers(heal=0.5, margin=0.08, suggest=0.2)


def test_reject_blocks_suggestion_and_autoheal():
    _set_tiers()
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    g.resolve(_mix(1.0, 2.0, 0.6))                 # guest:1 — suggest tier
    assert g.review_queue()["queue"][0]["tier"] == "suggest"
    assert g.reject_suggestion("guest:1", "u1")    # David: "No, that's not me"
    card = g.review_queue()["queue"][0]
    assert card["tier"] == "unknown" and card["suggested"] is None
    assert card["rejected_user_ids"] == ["u1"]
    # Even a decisive score can never auto-merge into a rejected member.
    cfg.face_autoheal_threshold = 0.3
    assert g._maybe_autoheal("guest:1") is None
    assert not g.reject_suggestion("guest:404", "u1")
    _set_tiers(heal=0.5, margin=0.08, suggest=0.2)


def test_named_guest_never_reviewed_or_autohealed():
    _set_tiers()
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    g.resolve(_mix(1.0, 2.0, 0.6))
    g.name_guest("guest:1", "Abuela")              # deliberate label → stays a guest
    assert g.review_queue()["queue"] == []
    cfg.face_autoheal_threshold = 0.3
    assert g._maybe_autoheal("guest:1") is None
    _set_tiers(heal=0.5, margin=0.08, suggest=0.2)


# ── face_box: which face is the card about ───────────────────────────────────

def test_thumb_box_stored_with_new_capture_and_exposed_in_review():
    _set_tiers()
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    g.resolve(_mix(1.0, 2.0, 0.6), thumb=b"FACECROP", thumb_box=[0.3, 0.2, 0.4, 0.5])
    card = g.review_queue()["queue"][0]
    assert card["face_box"] == [0.3, 0.2, 0.4, 0.5]
    _set_tiers(heal=0.5, margin=0.08, suggest=0.2)


def test_thumb_box_travels_with_the_kept_thumb():
    cfg.face_match_threshold = 0.99
    cfg.guest_cluster_threshold = 0.9
    g = _g()
    v = _vec(2.0)
    g.resolve(v)                                                  # cluster w/o thumb
    g.resolve(v, thumb=b"CROP", thumb_box=[0.1, 0.1, 0.5, 0.5])   # backfills the pair
    assert g.review_queue()["queue"][0]["face_box"] == [0.1, 0.1, 0.5, 0.5]
    # A later sighting's crop must NOT retag the kept thumb with a foreign box.
    g.resolve(v, thumb=b"OTHER", thumb_box=[0.9, 0.9, 0.05, 0.05])
    assert g.get_thumb("guest:1") == b"CROP"
    assert g.review_queue()["queue"][0]["face_box"] == [0.1, 0.1, 0.5, 0.5]


def test_legacy_thumb_face_located_lazily_and_cached():
    cfg.face_match_threshold = 0.99
    cfg.guest_cluster_threshold = 0.999999
    g = _g()
    g.resolve(_vec(3.0), thumb=b"LEGACYPERSONCROP")   # stored WITHOUT a box (legacy)
    calls = []

    def annotator(jpeg, centroid):
        calls.append(jpeg)
        return [0.25, 0.1, 0.5, 0.6]

    g.thumb_annotator = annotator
    assert g.review_queue()["queue"][0]["face_box"] == [0.25, 0.1, 0.5, 0.6]
    assert g.review_queue()["queue"][0]["face_box"] == [0.25, 0.1, 0.5, 0.6]
    assert len(calls) == 1                            # cached after the first read
    assert calls[0] == b"LEGACYPERSONCROP"            # ran on the stored bytes


def test_legacy_thumb_annotation_caching_rules():
    cfg.face_match_threshold = 0.99
    cfg.guest_cluster_threshold = 0.999999
    # "No face found" ([]) is cached — never re-run for that thumb.
    g = _g()
    g.resolve(_vec(4.0), thumb=b"NOFACE")
    none_calls = []
    g.thumb_annotator = lambda j, c: none_calls.append(1) or []
    assert g.review_queue()["queue"][0]["face_box"] is None
    g.review_queue()
    assert len(none_calls) == 1
    # "Engine unavailable" (None) is NOT cached — retried on a later read.
    g2 = _g()
    g2.resolve(_vec(5.0), thumb=b"X")
    down_calls = []
    g2.thumb_annotator = lambda j, c: down_calls.append(1)  # → None
    g2.review_queue()
    g2.review_queue()
    assert len(down_calls) == 2
    # No annotator wired (route tests / cold boot): face_box stays None, no crash.
    g3 = _g()
    g3.resolve(_vec(6.0), thumb=b"Y")
    assert g3.review_queue()["queue"][0]["face_box"] is None


def test_bad_thumb_replaced_by_next_face_located_sighting():
    cfg.face_match_threshold = 0.99
    cfg.guest_cluster_threshold = 0.9
    g = _g()
    v = _vec(7.0)
    g.resolve(v, thumb=b"HEADLESS-TORSO")            # legacy crop, box unknown (NULL)
    # The annotator looks and finds nothing → cached '[]' + surfaced as no_face.
    g.thumb_annotator = lambda j, c: []
    card = g.review_queue()["queue"][0]
    assert card["face_box"] is None and card["no_face"] is True
    # Next sighting arrives with a face-located crop → it REPLACES the bad thumb.
    g.resolve(v, thumb=b"PROPER-FACE-CROP", thumb_box=[0.2, 0.1, 0.5, 0.6])
    assert g.get_thumb("guest:1") == b"PROPER-FACE-CROP"
    card = g.review_queue()["queue"][0]
    assert card["face_box"] == [0.2, 0.1, 0.5, 0.6] and card["no_face"] is False
    # A face-located thumb is settled: yet another crop does NOT replace it.
    g.resolve(v, thumb=b"LATER-CROP", thumb_box=[0.9, 0.9, 0.05, 0.05])
    assert g.get_thumb("guest:1") == b"PROPER-FACE-CROP"


def test_unannotated_legacy_thumb_also_upgrades_to_face_located_crop():
    cfg.face_match_threshold = 0.99
    cfg.guest_cluster_threshold = 0.9
    g = _g()
    v = _vec(8.0)
    g.resolve(v, thumb=b"LEGACY")                    # box NULL, never annotated
    g.resolve(v, thumb=b"FACE-CROP", thumb_box=[0.3, 0.3, 0.4, 0.4])
    assert g.get_thumb("guest:1") == b"FACE-CROP"    # guaranteed-face crop wins
    # ...but a box-less crop never displaces an existing thumb.
    g2 = _g()
    g2.resolve(_vec(9.0), thumb=b"FIRST")
    g2.resolve(_vec(9.0), thumb=b"SECOND")           # no box → keep FIRST
    assert g2.get_thumb("guest:1") == b"FIRST"


# ── named guests as first-class identities (persist across re-appearances) ───

def test_named_guest_reappearance_suggested_then_merged():
    _set_tiers()   # heal 0.9, suggest 0.4
    g = _g()
    g.resolve(_vec(2.0), thumb=b"ABUELA-FACE", thumb_box=[0.2, 0.2, 0.5, 0.5])
    g.name_guest("guest:1", "Abuela")
    # Same person from a new angle → a separate cluster, but now the queue
    # recognises her: "Is this Abuela?" instead of a bare "Who is this?".
    g.resolve(_mix(2.0, 3.0, 0.6))
    card = g.review_queue()["queue"][0]
    assert card["guest_id"] == "guest:2" and card["tier"] == "suggest"
    assert card["suggested"]["kind"] == "guest"
    assert card["suggested"]["id"] == "guest:1"
    assert card["suggested"]["name"] == "Abuela"
    # Confirming folds the cluster in: one Abuela, sightings summed, queue empty.
    assert g.merge_guests("guest:2", "guest:1") == 2
    roster = [p for p in g.people() if p["class"] == "guest"]
    assert len(roster) == 1 and roster[0]["label"] == "Abuela"
    assert roster[0]["sightings"] == 2
    assert g.review_queue()["queue"] == []
    _set_tiers(heal=0.5, margin=0.08, suggest=0.2)


def test_named_guest_absorbs_reappearance_live_via_autoheal():
    _set_tiers(heal=0.9)
    g = _g()
    g.resolve(_vec(2.0))
    g.name_guest("guest:1", "Abuela")
    # Cross-angle sighting scores decisively → folded in DURING resolve; the
    # pipeline answers "Abuela", not "Person 2".
    ident = g.resolve(_mix(2.0, 3.0, 0.95))
    assert ident.cls == "guest" and ident.id == "guest:1" and ident.name == "Abuela"
    roster = [p for p in g.people() if p["class"] == "guest"]
    assert len(roster) == 1 and roster[0]["sightings"] == 2
    _set_tiers(heal=0.5, margin=0.08, suggest=0.2)


def test_reject_guest_suggestion_drops_to_unknown():
    _set_tiers()
    g = _g()
    g.resolve(_vec(2.0))
    g.name_guest("guest:1", "Abuela")
    g.resolve(_mix(2.0, 3.0, 0.6))
    assert g.review_queue()["queue"][0]["suggested"]["id"] == "guest:1"
    g.reject_suggestion("guest:2", "guest:1")   # "No, that's not Abuela"
    card = g.review_queue()["queue"][0]
    assert card["tier"] == "unknown" and card["suggested"] is None
    # ...and autoheal can never fold it into her either.
    cfg.face_autoheal_threshold = 0.3
    assert g._maybe_autoheal("guest:2") is None
    _set_tiers(heal=0.5, margin=0.08, suggest=0.2)


def test_member_outranks_guest_when_closer():
    _set_tiers()
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    g.resolve(_vec(2.0))
    g.name_guest("guest:1", "Abuela")
    # Closer to David (0.7) than to Abuela (~0) → member suggestion wins.
    g.resolve(_mix(1.0, 3.0, 0.7))
    card = g.review_queue()["queue"][0]
    assert card["suggested"]["kind"] == "member" and card["suggested"]["id"] == "u1"
    _set_tiers(heal=0.5, margin=0.08, suggest=0.2)


def test_merge_guests_guards():
    g = _g()
    g.resolve(_vec(2.0))
    g.name_guest("guest:1", "Abuela")
    assert g.merge_guests("guest:1", "guest:1") is None      # self
    assert g.merge_guests("guest:404", "guest:1") is None    # missing src
    assert g.merge_guests("guest:1", "guest:404") is None    # missing dst
    # dst keeps its thumb; an absorbed cluster never resurfaces as a merge target.
    g.resolve(_vec(3.0), thumb=b"SRC-FACE", thumb_box=[0.1, 0.1, 0.3, 0.3])
    assert g.merge_guests("guest:2", "guest:1") == 2
    assert g.get_thumb("guest:1") == b"SRC-FACE"             # dst had none → src's rides over
    assert g.merge_guests("guest:2", "guest:1") is None      # already absorbed


def test_threshold_override_roundtrip():
    from app.config import cfg
    g = _g()
    default = float(cfg.face_autoheal_threshold)
    # No override → effective == default.
    assert g._thr("face_autoheal_threshold") == default
    g.set_thresholds({"face_autoheal_threshold": 0.77, "bogus_key": 1.0})
    assert g._thr("face_autoheal_threshold") == 0.77
    view = {t["key"]: t for t in g.thresholds()}
    assert view["face_autoheal_threshold"]["overridden"] is True
    assert view["face_autoheal_threshold"]["value"] == 0.77
    assert "bogus_key" not in view  # unknown keys ignored
    # Clearing restores the default.
    g.set_thresholds({"face_autoheal_threshold": "default"})
    assert g._thr("face_autoheal_threshold") == default
    assert {t["key"]: t for t in g.thresholds()}["face_autoheal_threshold"]["overridden"] is False


def test_member_clusters_and_detach():
    from app.config import cfg
    cfg.face_match_threshold = 0.9
    cfg.guest_cluster_threshold = 0.9
    g = _g()
    g.enroll("u1", "David", _vec(1.0))
    # A guest cluster promoted into the member = an audit-trail entry.
    g._cluster_guest(_vec(1.0), thumb=b"jpegbytes")
    gid = "guest:1"
    assert g.promote_guest(gid, "u1", "David") is True
    clusters = g.member_clusters("u1")
    assert len(clusters) == 1 and clusters[0]["guest_id"] == gid
    assert clusters[0]["score"] is not None and clusters[0]["has_thumb"] is True
    # "That wasn't me": detach un-promotes, blocks re-heal, returns to review.
    member = g.detach_cluster(gid)
    assert member == "u1"
    assert g.member_clusters("u1") == []
    q = g.review_queue()["queue"]
    assert any(c["guest_id"] == gid for c in q)  # back in the review queue
    # And it will never auto-heal back into u1.
    row = next(c for c in q if c["guest_id"] == gid)
    assert "u1" in row["rejected_user_ids"]
    # Detaching a non-promotion is a no-op.
    assert g.detach_cluster("guest:999") is None


def test_guest_ids_never_reused_after_delete():
    # ids come from MAX+1, not COUNT+1 — after a delete COUNT falls below the top
    # id and COUNT+1 collides with a live row (UNIQUE violation → clustering dead).
    cfg.face_match_threshold = 0.99
    cfg.guest_cluster_threshold = 0.999999
    g = _g()
    g.resolve(_vec(1.0))  # guest:1
    g.resolve(_vec(2.0))  # guest:2
    g.forget_guest("guest:1")
    ident = g.resolve(_vec(3.0))  # COUNT+1 would collide with guest:2
    assert ident.cls == "guest" and ident.id == "guest:3"


def _two_close_members(g):
    # u2 sits close to u1 (cos(u1,u2)=0.9) so a u1-perfect probe scores 1.0 vs u1
    # and 0.9 vs u2 — margin 0.1, an ambiguous zone we can gate on.
    e1, e2 = _vec(1.0), _vec(2.0)
    g.enroll("u1", "A", e1)
    g.enroll("u2", "B", [0.9 * a + 0.4359 * b for a, b in zip(e1, e2)])


def test_ambiguous_household_match_takes_guest_path():
    cfg.face_match_threshold = 0.5
    cfg.face_match_margin = 0.2  # demand a decisive win
    cfg.guest_cluster_threshold = 0.999999
    cfg.face_autoheal_threshold = 2.0  # isolate the match gate from autoheal
    g = _g()
    _two_close_members(g)
    ident = g.resolve(_vec(1.0))  # scores ~1.0 vs u1, ~0.9 vs u2 → margin ~0.1 < 0.2
    assert ident.cls == "guest"  # ambiguous → review ladder, never a silent label
    cfg.face_match_margin = 0.05  # decisive enough now
    ident = g.resolve(_vec(1.0))
    assert ident.cls == "household" and ident.id == "u1"


def test_recheck_is_margin_gated_and_side_effect_free():
    cfg.face_match_threshold = 0.5
    cfg.face_match_margin = 0.2
    cfg.face_reinforce = True
    g = _g()
    _two_close_members(g)
    before = {p["user_id"]: p["samples"] for p in g.profiles()}
    # Ambiguous probe → None (keep the cached label), decisively-u2 probe → u2.
    assert g.recheck(_vec(1.0)) is None
    probe_u2 = [0.9 * a + 0.4359 * b for a, b in zip(_vec(1.0), _vec(2.0))]
    # u1 and u2 are 0.9-similar, so even a u2-perfect probe only wins by ~0.1 —
    # still None at margin 0.2, decisive once the bar reflects the roster.
    assert g.recheck(probe_u2) is None
    cfg.face_match_margin = 0.05
    fresh = g.recheck(probe_u2)
    assert fresh is not None and fresh.id == "u2" and fresh.cls == "household"
    # No side effects ever: no guest clusters seeded, no reinforcement folded in.
    assert g.guests() == []
    assert {p["user_id"]: p["samples"] for p in g.profiles()} == before
