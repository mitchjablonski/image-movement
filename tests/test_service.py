"""Acceptance tests for the Phase-2 reuse-detection service."""

from __future__ import annotations

import os

from imagemovement.config import DetectorConfig
from imagemovement.data import synthetic_seeds
from imagemovement.perturb import PerturbParams, perturb
from imagemovement.service import ReuseDetectorService


def _cfg(tmp: str) -> DetectorConfig:
    cfg = DetectorConfig()
    cfg.serving.corpus_db = os.path.join(tmp, "corpus.db")
    cfg.serving.blob_dir = os.path.join(tmp, "blobs")
    return cfg


def test_persistence_round_trip(tmp_path):
    cfg = _cfg(str(tmp_path))
    img = synthetic_seeds(1)[0][1]

    svc = ReuseDetectorService(cfg)
    svc.enroll(img, "u", "a")
    svc.close()

    # Reopen from disk: the record survives and is still matchable.
    svc2 = ReuseDetectorService(cfg)
    assert len(svc2.corpus) == 1
    variant = perturb(img, PerturbParams(quality=80, zoom_factor=1.08, dx=4, dy=-3))
    assert len(svc2.check(variant)) == 1
    svc2.close()


def test_cross_user_reuse_alerts(tmp_path):
    cfg = _cfg(str(tmp_path))
    seeds = synthetic_seeds(2)
    svc = ReuseDetectorService(cfg)
    svc.enroll(seeds[0][1], "userA", "a1")

    variant = perturb(seeds[0][1], PerturbParams(quality=75, zoom_factor=1.10, dx=6, dy=-4))
    matches = svc.check(variant)
    assert len(matches) == 1
    assert matches[0].user_id == "userA"
    assert svc.alert(matches).triggered          # default threshold 1

    # A genuinely different image is not a match.
    assert svc.check(seeds[1][1]) == []
    svc.close()


def test_threshold_two_requires_two_distinct_users(tmp_path):
    cfg = _cfg(str(tmp_path))
    cfg.serving.alert.min_distinct_users = 2
    img = synthetic_seeds(1)[0][1]

    svc = ReuseDetectorService(cfg)
    svc.enroll(img, "userA", "a1")
    variant = perturb(img, PerturbParams(quality=80, zoom_factor=1.05, dx=3, dy=2))

    one = svc.check(variant)
    assert len(one) >= 1
    assert not svc.alert(one).triggered          # only one distinct user

    svc.enroll(img, "userB", "b1")               # same image, second user
    two = svc.check(variant)
    assert {m.user_id for m in two} == {"userA", "userB"}
    assert svc.alert(two).triggered              # two distinct users -> alert
    svc.close()
