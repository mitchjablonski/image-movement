"""Reuse-detection service: persistent corpus + 3-gate cascade + cross-user alerting.

Wires the Phase-1 detector to the Phase-2 corpus:
  enroll(image, user_id, attempt_id) -> persist + index a submission
  check(image)                       -> run hash filter -> geometric+photometric
                                        verify against each candidate's pixels
                                        -> confirmed Matches
  alert(matches)                     -> aggregate confirmed matches by distinct
                                        user_id; trigger when the distinct-user
                                        count meets serving.alert.min_distinct_users

Optional retention (serving config): TTL excludes records older than max_age_days
from matching; recency-decay weights each match's contribution to alert severity
by the enrolled record's age. Both off by default (behavior == no-retention).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import imagehash
import numpy as np

from .config import DetectorConfig
from .corpus import CorpusRecord, CorpusStore
from .stage1_hash import make_index
from .stage2_geom import GeoEvidence, GeoVerifier


@dataclass
class Match:
    """A confirmed same-image hit against an enrolled corpus record."""

    record_id: int
    user_id: str
    attempt_id: str
    hash_distance: int
    evidence: GeoEvidence
    enrolled_at: float


@dataclass
class Alert:
    """Outcome of aggregating a query's confirmed matches by distinct user."""

    triggered: bool
    distinct_users: int
    severity: float
    matches: list[Match]


class ReuseDetectorService:
    """Enroll submissions into a persistent corpus and detect image reuse across them."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.corpus = CorpusStore(
            self.config.serving.corpus_db,
            self.config.serving.blob_dir,
            self.config.stage1.hash_size,
        )
        self.verifier = GeoVerifier(self.config.stage2)
        self._records: dict[int, CorpusRecord] = {}
        self.index = make_index(self.config.stage1)
        self._rebuild_index()

    # -- enrollment -------------------------------------------------------
    def _ttl_max_age(self) -> float | None:
        ttl = self.config.serving.ttl
        return ttl.max_age_days if ttl.enabled else None

    def _rebuild_index(self, *, now: float | None = None) -> None:
        """Rebuild the in-memory index + record cache from the persisted corpus."""
        self.index = make_index(self.config.stage1)
        self._records = {}
        for rec in self.corpus.iter_records(max_age_days=self._ttl_max_age(), now=now):
            self.index.add_hash(rec.id, imagehash.hex_to_hash(rec.phash))
            self._records[rec.id] = rec

    def enroll(self, image: np.ndarray, user_id: str, attempt_id: str, *, now: float | None = None) -> CorpusRecord:
        """Persist + index a submitted image."""
        rec = self.corpus.enroll(image, user_id, attempt_id, now=now)
        self.index.add_hash(rec.id, imagehash.hex_to_hash(rec.phash))
        self._records[rec.id] = rec
        return rec

    # -- detection --------------------------------------------------------
    def check(self, image: np.ndarray, *, exclude_record_id: int | None = None) -> list[Match]:
        """Return corpus records confirmed (all 3 gates) to be the same core image."""
        matches: list[Match] = []
        for rec_id, dist in self.index.query(image):                 # stage 1: hash candidates
            if rec_id == exclude_record_id:
                continue
            rec = self._records.get(rec_id)
            if rec is None:
                continue
            ev = self.verifier.verify(image, self.corpus.get_image(rec_id))   # stage 2 + 3
            if self.verifier.is_match(ev):
                matches.append(Match(rec_id, rec.user_id, rec.attempt_id, dist, ev, rec.enrolled_at))
        return sorted(matches, key=lambda m: m.evidence.inliers, reverse=True)

    # -- alerting ---------------------------------------------------------
    def _severity(self, matches: list[Match], now: float) -> float:
        """Weighted distinct-user count (weight 1 unless recency-decay is enabled)."""
        decay = self.config.serving.decay
        per_user: dict[str, float] = {}
        for m in matches:
            weight = 1.0
            if decay.enabled:
                age_days = max(0.0, (now - m.enrolled_at) / 86400.0)
                weight = 0.5 ** (age_days / decay.half_life_days)
            per_user[m.user_id] = max(per_user.get(m.user_id, 0.0), weight)
        return float(sum(per_user.values()))

    def alert(self, matches: list[Match], *, now: float | None = None) -> Alert:
        """Aggregate confirmed matches by distinct user_id into an alert decision."""
        ts = time.time() if now is None else now
        distinct = {m.user_id for m in matches}
        threshold = self.config.serving.alert.min_distinct_users
        return Alert(
            triggered=len(distinct) >= threshold,
            distinct_users=len(distinct),
            severity=self._severity(matches, ts),
            matches=matches,
        )

    # -- convenience ------------------------------------------------------
    def submit(self, image: np.ndarray, user_id: str, attempt_id: str, *, now: float | None = None) -> tuple[Alert, CorpusRecord]:
        """Detect reuse against the existing corpus, then enroll this submission."""
        alert = self.alert(self.check(image), now=now)
        record = self.enroll(image, user_id, attempt_id, now=now)
        return alert, record

    def purge_expired(self, *, now: float | None = None) -> int:
        """Apply the TTL: delete expired records, then rebuild the index. No-op if TTL off."""
        ttl = self.config.serving.ttl
        if not ttl.enabled:
            return 0
        removed = self.corpus.purge_expired(ttl.max_age_days, now=now)
        self._rebuild_index(now=now)
        return removed

    def close(self) -> None:
        self.corpus.close()
