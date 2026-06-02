"""Stage 2 of the cascade: geometric + photometric verification. The decision-maker.

Matches local ORB keypoints between two images and fits a *similarity*
transform (uniform scale + rotation + translation) with RANSAC, then checks that
the aligned images actually share pixels. A genuine copy aligns AND matches
photometrically; a different photo of the same (aligned) subject can align
geometrically but will NOT match photometrically.

Three gates, all required for a match (see is_match + geom_ok):
  1. enough RANSAC inliers (min_inliers),
  2. the fitted transform is near-identity (small zoom, small rotation), and
  3. low photometric residual after alignment (max_residual) -- the gate that
     separates a reused image from a similar photo of the same subject.

Gates 2 and 3 are combined into ``geom_ok`` ("passes all non-inlier gates") so
the evaluation harness, which sweeps the inlier threshold over geom_ok pairs,
picks up the photometric gate with no change.

A similarity model (4 DOF) is deliberately chosen over a full homography
(8 DOF): it matches the expected re-submission edits (zoom + shift) and cannot overfit distinct
images the way a homography can. Small inputs are upscaled before feature
detection (min_feature_dim) using a SHARED factor for both images, so the
estimated scale/rotation -- and the residual comparison -- stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Stage2Config


@dataclass
class GeoEvidence:
    """Explainable output of geometric + photometric verification for one pair."""

    matches: int          # good descriptor matches after Lowe ratio test
    inliers: int          # RANSAC inliers supporting the fitted transform
    inlier_ratio: float
    scale: float          # estimated zoom (1.0 == no zoom)
    rotation_deg: float
    translation: float    # estimated shift magnitude in original pixels
    residual: float       # mean |pixel delta| over the aligned overlap (inf if none)
    geom_ok: bool         # near-identity transform AND residual within bound


class GeoVerifier:
    """Geometric + photometric verifier. A plain engine that consumes Stage2Config."""

    def __init__(self, config: Stage2Config | None = None) -> None:
        self.config = config or Stage2Config()
        self._orb = cv2.ORB_create(nfeatures=self.config.nfeatures)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def _upscale_factor(self, *imgs: np.ndarray) -> float:
        """Shared factor that lifts the smallest side up to min_feature_dim (>= 1.0)."""
        if self.config.min_feature_dim <= 0:
            return 1.0
        smallest = min(min(im.shape[:2]) for im in imgs)
        return max(1.0, self.config.min_feature_dim / smallest)

    def _gray(self, img_bgr: np.ndarray, factor: float) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if factor > 1.0:
            gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        return gray

    def _residual(self, ga: np.ndarray, gb: np.ndarray, m: np.ndarray) -> float:
        """Mean |pixel delta| over the region where warped ``ga`` overlaps ``gb``."""
        h, w = gb.shape[:2]
        warped = cv2.warpAffine(ga, m, (w, h))
        cover = cv2.warpAffine(np.ones(ga.shape[:2], np.uint8), m, (w, h)) > 0
        if int(cover.sum()) < 500:          # too little overlap to judge
            return float("inf")
        diff = np.abs(warped.astype(np.int16) - gb.astype(np.int16))
        return float(diff[cover].mean())

    def verify(self, img_a: np.ndarray, img_b: np.ndarray) -> GeoEvidence:
        cfg = self.config
        none = GeoEvidence(0, 0, 0.0, 1.0, 0.0, 0.0, float("inf"), False)
        factor = self._upscale_factor(img_a, img_b)
        ga, gb = self._gray(img_a, factor), self._gray(img_b, factor)
        ka, da = self._orb.detectAndCompute(ga, None)
        kb, db = self._orb.detectAndCompute(gb, None)
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            return none

        good = []
        for pair in self._matcher.knnMatch(da, db, k=2):
            if len(pair) == 2 and pair[0].distance < cfg.ratio_test * pair[1].distance:
                good.append(pair[0])
        if len(good) < 4:
            return GeoEvidence(len(good), 0, 0.0, 1.0, 0.0, 0.0, float("inf"), False)

        pts_a = np.float32([ka[m.queryIdx].pt for m in good])
        pts_b = np.float32([kb[m.trainIdx].pt for m in good])
        m, mask = cv2.estimateAffinePartial2D(
            pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_thresh * factor,
        )
        if m is None or mask is None:
            return GeoEvidence(len(good), 0, 0.0, 1.0, 0.0, 0.0, float("inf"), False)

        inliers = int(mask.sum())
        # Both images shared one upscale factor, so scale/rotation are unaffected.
        scale = float(np.hypot(m[0, 0], m[1, 0]))
        rotation = float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
        translation = float(np.hypot(m[0, 2], m[1, 2])) / factor   # back to original px
        residual = self._residual(ga, gb, m)
        near_identity = (
            abs(scale - 1.0) <= cfg.max_scale_dev
            and abs(rotation) <= cfg.max_rotation_deg
        )
        geom_ok = near_identity and residual <= cfg.max_residual
        return GeoEvidence(
            matches=len(good),
            inliers=inliers,
            inlier_ratio=inliers / len(good),
            scale=scale,
            rotation_deg=rotation,
            translation=translation,
            residual=residual,
            geom_ok=geom_ok,
        )

    def is_match(self, ev: GeoEvidence) -> bool:
        """Final decision: all non-inlier gates pass AND enough inlier support."""
        return ev.geom_ok and ev.inliers >= self.config.min_inliers
