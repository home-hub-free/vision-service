"""Retention janitor sweep — age + disk-cap pruning, oldest-first."""
import os
import tempfile
import time

from app.retention import sweep, sweep_thumbs


def _seg(root: str, name: str, age_days: float, size: int) -> str:
    os.makedirs(root, exist_ok=True)
    p = os.path.join(root, name)
    with open(p, "wb") as fh:
        fh.write(b"\0" * size)
    t = time.time() - age_days * 86400
    os.utime(p, (t, t))
    return p


def test_age_prune_removes_only_old_files():
    root = tempfile.mkdtemp()
    old = _seg(root, "old.mp4", age_days=40, size=10)
    fresh = _seg(root, "fresh.mp4", age_days=1, size=10)
    res = sweep(root, retention_days=14, disk_cap_gb=0.0)
    assert res["removed"] == 1
    assert not os.path.exists(old) and os.path.exists(fresh)


def test_thumb_sweep_ages_out_the_scrub_cache():
    root = tempfile.mkdtemp()
    cam = os.path.join(root, "cam1")
    old = _seg(cam, "12-0.jpg", age_days=40, size=10)
    fresh = _seg(cam, "13-15.jpg", age_days=1, size=10)
    other = _seg(cam, "notes.txt", age_days=40, size=10)  # only .jpg is ours to prune
    assert sweep_thumbs(root, retention_days=14) == 1
    assert not os.path.exists(old) and os.path.exists(fresh) and os.path.exists(other)


def test_disk_cap_prunes_oldest_until_under_cap():
    root = tempfile.mkdtemp()
    mb = 1024 * 1024
    a = _seg(root, "a.mp4", age_days=3, size=mb)   # oldest
    b = _seg(root, "b.mp4", age_days=2, size=mb)
    c = _seg(root, "c.mp4", age_days=1, size=mb)   # newest
    # cap at ~1.5 MB → must drop the two oldest, keep the newest.
    res = sweep(root, retention_days=0, disk_cap_gb=1.5 / 1024)
    assert res["removed"] == 2
    assert not os.path.exists(a) and not os.path.exists(b) and os.path.exists(c)
