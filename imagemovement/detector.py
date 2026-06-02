"""Two-stage cascade detector: stage-1 hash filter -> stage-2 geometric verify.

Embodies "filter for recall, verify for precision": the perceptual-hash index
nominates candidates and the geometric verifier makes the actual match
decision. Each Match carries the geometric evidence, so a match decision is
explainable (and reviewable by a human).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DetectorConfig
from .stage1_hash import HashIndex
from .stage2_geom import GeoEvidence, GeoVerifier


@dataclass
class Match:
    """A confirmed same-image hit, with the evidence behind the decision."""

    key: str
    hash_distance: int
    evidence: GeoEvidence
    min_inliers: int = 20

    @property
    def confidence(self) -> float:
        """0..1 score from inlier support, saturating at 2x the match threshold."""
        return float(min(1.0, self.evidence.inliers / (2 * self.min_inliers)))


class CascadeDetector:
    """The full detector. Enroll known images, then query a suspect image."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.index = HashIndex(self.config.stage1)
        self.verifier = GeoVerifier(self.config.stage2)
        self._images: dict[str, np.ndarray] = {}

    def add(self, key: str, img_bgr: np.ndarray) -> None:
        """Enroll a known image into the corpus."""
        self.index.add(key, img_bgr)
        self._images[key] = img_bgr

    def query(self, img_bgr: np.ndarray) -> list[Match]:
        """Return corpus images that are the same core image as the query."""
        matches: list[Match] = []
        for key, dist in self.index.query(img_bgr):                 # stage 1
            ev = self.verifier.verify(img_bgr, self._images[key])   # stage 2
            if self.verifier.is_match(ev):
                matches.append(Match(key, dist, ev, self.config.stage2.min_inliers))
        return sorted(matches, key=lambda mt: mt.evidence.inliers, reverse=True)

    def __len__(self) -> int:
        return len(self._images)
