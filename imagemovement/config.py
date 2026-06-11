"""Central, validated configuration for the detection cascade (pydantic v2).

Per the team decision, everything config-shaped is a validated pydantic model:
the perturbation/test space, the two stage configs, the eval settings, and the
Phase-2 serving/corpus knobs, all nested under DetectorConfig. Validation
happens here, once, at the boundary -- so a bad threshold (from a file, env
var, or CLI) fails loudly instead of silently skewing results. Engine classes
(HashIndex, GeoVerifier, CorpusStore) and computed results (GeoEvidence) stay
plain: they consume config, they aren't config.

DetectorConfig is a BaseSettings, so any field can be overridden from the
environment, e.g. IMOVE_STAGE2__MIN_INLIERS=30 or
IMOVE_SERVING__ALERT__MIN_DISTINCT_USERS=2.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PerturbationSpace(BaseModel):
    """How much perturbation we SIMULATE when generating positive variants.

    This is the *transform band we validate against* -- distinct from what the
    detector tolerates (Stage2Config). DetectorConfig validates that acceptance
    covers this range.

    quality/zoom/shift are each an explicit (min, max) band. Shift is applied
    per axis with a random sign, so shift_min_px=0 means 'no-shift allowed' and
    shift_min_px>0 forces a minimum displacement.
    """

    quality_min: int = Field(60, ge=1, le=100)
    quality_max: int = Field(95, ge=1, le=100)
    zoom_min: float = Field(1.0, gt=0.0)
    zoom_max: float = Field(1.15, gt=0.0)
    shift_min_px: int = Field(0, ge=0)    # min |shift| per axis (0 == no-shift allowed)
    shift_max_px: int = Field(8, ge=0)    # max |shift| per axis
    # 4:2:0 mirrors real camera/photo JPEGs (the realistic default); 4:4:4
    # is near-lossless and useful as a sweep extreme.
    subsampling: Literal["4:2:0", "4:4:4"] = "4:2:0"

    @model_validator(mode="after")
    def _ordered(self) -> "PerturbationSpace":
        if self.quality_min > self.quality_max:
            raise ValueError("quality_min must be <= quality_max")
        if self.zoom_min > self.zoom_max:
            raise ValueError("zoom_min must be <= zoom_max")
        if self.shift_min_px > self.shift_max_px:
            raise ValueError("shift_min_px must be <= shift_max_px")
        return self

    @property
    def max_zoom_dev(self) -> float:
        """Largest |zoom - 1| we simulate (used to sanity-check acceptance)."""
        return max(abs(self.zoom_max - 1.0), abs(self.zoom_min - 1.0))


class Stage1Config(BaseModel):
    """Perceptual-hash filter knobs. Recall-first: max_distance is deliberately loose.

    NOTE: max_distance is interpreted relative to hash_size (a hash_size=8 phash
    is 64 bits). If you raise hash_size, scale max_distance up proportionally.
    """

    hash_size: int = Field(8, ge=4, le=64)
    max_distance: int = Field(32, ge=0)    # loose: the verifier owns precision, so never drop a true copy here
    # Candidate-search backend: 'linear' (pure-Python scan, default) or 'faiss'
    # (IndexBinaryFlat, exact + SIMD-fast). Both return the same candidate set.
    index_backend: Literal["linear", "faiss"] = "linear"


class Stage2Config(BaseModel):
    """Geometric + photometric verifier knobs -- the acceptance bounds that define 'same image'."""

    min_inliers: int = Field(20, ge=1)
    ratio_test: float = Field(0.75, gt=0.0, lt=1.0)
    ransac_thresh: float = Field(4.0, gt=0.0)
    nfeatures: int = Field(2000, ge=1)
    max_scale_dev: float = Field(0.30, ge=0.0)      # |scale - 1| tolerated
    max_rotation_deg: float = Field(8.0, ge=0.0)
    # Photometric gate: max mean |pixel delta| over the overlap AFTER aligning the
    # two images. A true copy is the same pixels post-alignment (residual ~ a few);
    # a distinct-but-similar image pair aligns geometrically but differs in pixels
    # (residual ~tens). 12 is the validated default: true-copy residuals at
    # quality_min=60 stay below this; the 2 borderline genuine distinct-but-similar FPs
    # seen at full scale (residuals 12.7 and 13.8) are rejected.
    max_residual: float = Field(12.0, ge=0.0)
    # Upscale inputs whose smaller side is below this (px) before ORB, so tiny
    # images yield enough keypoints. 0 disables; no-op for larger images.
    min_feature_dim: int = Field(256, ge=0)


class EvalConfig(BaseModel):
    """Validation-harness knobs."""

    variants_per_seed: int = Field(5, ge=1)
    synthetic_seeds: int = Field(12, ge=2)
    rng_seed: int = 0


class AlertConfig(BaseModel):
    """Cross-user alerting: how many DISTINCT user_ids a reused image must match to alert.

    Default 1 = alert on any confirmed match; the distinct-user count is a severity
    signal. Raise to >=2 for higher-confidence-only alerts (joint FP ~1e-8).
    """

    min_distinct_users: int = Field(1, ge=1)


class TtlConfig(BaseModel):
    """Optional retention TTL. When enabled, records older than max_age_days are
    excluded from matching and can be purged. Off by default (lossless)."""

    enabled: bool = False
    max_age_days: float = Field(365.0, gt=0.0)


class DecayConfig(BaseModel):
    """Optional recency-decay weighting of alert severity by enrolled-record age:
    a match's contribution is scaled by 0.5 ** (age_days / half_life_days). Off by default."""

    enabled: bool = False
    half_life_days: float = Field(90.0, gt=0.0)


class ServingConfig(BaseModel):
    """Phase-2 serving/corpus knobs: where the corpus lives, plus alerting + retention."""

    corpus_db: str = "data/corpus/corpus.db"     # SQLite file (rows + hash index source)
    blob_dir: str = "data/corpus/blobs"          # lossless PNG pixels for stage-2 re-derivation
    alert: AlertConfig = Field(default_factory=AlertConfig)
    ttl: TtlConfig = Field(default_factory=TtlConfig)
    decay: DecayConfig = Field(default_factory=DecayConfig)


class DetectorConfig(BaseSettings):
    """Top-level config; every field overridable from env (prefix IMOVE_)."""

    model_config = SettingsConfigDict(env_prefix="IMOVE_", env_nested_delimiter="__")

    stage1: Stage1Config = Field(default_factory=Stage1Config)
    stage2: Stage2Config = Field(default_factory=Stage2Config)
    perturbation: PerturbationSpace = Field(default_factory=PerturbationSpace)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)

    @model_validator(mode="after")
    def _acceptance_covers_perturbation(self) -> "DetectorConfig":
        # The detector must tolerate at least the zoom we simulate, or the eval
        # would manufacture false negatives against its own test data.
        if self.stage2.max_scale_dev < self.perturbation.max_zoom_dev:
            raise ValueError(
                f"stage2.max_scale_dev ({self.stage2.max_scale_dev}) must be >= "
                f"the simulated zoom deviation ({self.perturbation.max_zoom_dev:.3f})"
            )
        return self
