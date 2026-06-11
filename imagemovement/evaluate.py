"""Phase-1 evaluation: does the cascade catch copies without flagging distinct images?

Scores three categories of image pairs, the third being the one that matters
most for reuse detection:

  * POSITIVE      -- a perturbed copy of an enrolled image (should match)
  * NEG_DIFF_ID   -- a different person (should not match)
  * NEG_SAME_ID   -- the SAME person's other genuine photo (should not match)

Sweeping the min-inliers decision threshold yields recall plus a false-positive
rate for each negative class. The recommended operating point is the lowest
threshold with zero false positives on BOTH negative classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DetectorConfig
from .perturb import generate_variants
from .stage1_hash import phash
from .stage2_geom import GeoVerifier

POSITIVE = "pos"
NEG_DIFF_ID = "neg_diff_id"
NEG_SAME_ID = "neg_same_id"


@dataclass
class PairRecord:
    category: str        # POSITIVE | NEG_DIFF_ID | NEG_SAME_ID
    hash_distance: int
    hash_pass: bool
    inliers: int
    geom_ok: bool


@dataclass
class SweepRow:
    threshold: int
    recall: float
    fp_diff_id: float
    fp_same_id: float
    precision: float


@dataclass
class EvalResult:
    n_identities: int
    n_positive: int
    n_neg_diff_id: int
    n_neg_same_id: int
    stage1_recall: float
    stage1_same_id_passthrough: float
    sweep: list[SweepRow]
    operating_point: SweepRow | None


def _decide(verifier, query, enrolled, qhash, ehash, max_distance):
    """Run both stages for one pair; stage 1 prunes before stage 2 runs."""
    dist = qhash - ehash
    if dist > max_distance:
        return dist, False, 0, False
    ev = verifier.verify(query, enrolled)
    return dist, True, ev.inliers, ev.geom_ok


def score_pairs(identities, config, verifier, rng):
    """Score every relevant (query, enrolled) pair across all three categories."""
    hs = config.stage1.hash_size
    maxd = config.stage1.max_distance
    enrolled = {name: imgs[0] for name, imgs in identities}
    ehash = {name: phash(img, hs) for name, img in enrolled.items()}

    records: list[PairRecord] = []
    for name, imgs in identities:
        # Positives + different-identity negatives: perturbed copies of imgs[0].
        for variant, _params in generate_variants(imgs[0], config.eval.variants_per_seed, rng, config.perturbation):
            vhash = phash(variant, hs)
            for other in enrolled:
                dist, hp, inl, gok = _decide(verifier, variant, enrolled[other], vhash, ehash[other], maxd)
                cat = POSITIVE if other == name else NEG_DIFF_ID
                records.append(PairRecord(cat, dist, hp, inl, gok))
        # Same-identity hard negatives: the person's OTHER genuine photos.
        for extra in imgs[1:]:
            xhash = phash(extra, hs)
            dist, hp, inl, gok = _decide(verifier, extra, enrolled[name], xhash, ehash[name], maxd)
            records.append(PairRecord(NEG_SAME_ID, dist, hp, inl, gok))
    return records


def sweep_threshold(records):
    """Sweep the min-inliers decision threshold over scored pairs."""
    pos = [r for r in records if r.category == POSITIVE]
    nd = [r for r in records if r.category == NEG_DIFF_ID]
    ns = [r for r in records if r.category == NEG_SAME_ID]
    max_t = max((r.inliers for r in records), default=0)
    rows: list[SweepRow] = []
    for t in range(max_t + 1):
        tp = sum(1 for r in pos if r.geom_ok and r.inliers >= t)
        fpd = sum(1 for r in nd if r.geom_ok and r.inliers >= t)
        fps = sum(1 for r in ns if r.geom_ok and r.inliers >= t)
        recall = tp / len(pos) if pos else 0.0
        fp_diff = fpd / len(nd) if nd else 0.0
        fp_same = fps / len(ns) if ns else 0.0
        total_fp = fpd + fps
        precision = tp / (tp + total_fp) if (tp + total_fp) else 1.0
        rows.append(SweepRow(t, recall, fp_diff, fp_same, precision))
    return rows


def select_operating_point(sweep):
    """Lowest threshold with zero false positives on BOTH negative classes."""
    safe = [r for r in sweep if r.fp_diff_id == 0.0 and r.fp_same_id == 0.0]
    return max(safe, key=lambda r: r.recall) if safe else None


def evaluate(identities, config: DetectorConfig | None = None) -> EvalResult:
    """Run the full three-category harness and assemble the result."""
    config = config or DetectorConfig()
    rng = np.random.default_rng(config.eval.rng_seed)
    verifier = GeoVerifier(config.stage2)

    records = score_pairs(identities, config, verifier, rng)
    pos = [r for r in records if r.category == POSITIVE]
    ns = [r for r in records if r.category == NEG_SAME_ID]
    nd = [r for r in records if r.category == NEG_DIFF_ID]
    stage1_recall = (sum(r.hash_pass for r in pos) / len(pos)) if pos else 0.0
    stage1_same_id = (sum(r.hash_pass for r in ns) / len(ns)) if ns else 0.0
    sweep = sweep_threshold(records)

    return EvalResult(
        n_identities=len(identities),
        n_positive=len(pos),
        n_neg_diff_id=len(nd),
        n_neg_same_id=len(ns),
        stage1_recall=stage1_recall,
        stage1_same_id_passthrough=stage1_same_id,
        sweep=sweep,
        operating_point=select_operating_point(sweep),
    )


def format_report(result: EvalResult) -> str:
    lines = [
        "=== image-movement Phase-1 evaluation ===",
        f"identities:           {result.n_identities}",
        f"positive pairs:       {result.n_positive}  (true copies)",
        f"diff-identity negs:   {result.n_neg_diff_id}  (different people)",
        f"same-identity negs:   {result.n_neg_same_id}  (same person, different photo -- the hard negative)",
        f"stage-1 recall:       {result.stage1_recall:.1%}  (true copies surviving the hash filter)",
        f"stage-1 same-id leak: {result.stage1_same_id_passthrough:.1%}  (same-person photos the hash alone would pass -> why stage 2 exists)",
        "",
        "  min_inliers    recall   FP(diff-id)  FP(same-id)  precision",
    ]
    stride = max(1, len(result.sweep) // 20)
    for row in result.sweep[::stride]:
        lines.append(
            f"  {row.threshold:>10}   {row.recall:7.1%}   {row.fp_diff_id:10.1%}   {row.fp_same_id:10.1%}   {row.precision:8.1%}"
        )
    lines.append("")
    if result.operating_point:
        op = result.operating_point
        lines.append(
            f"recommended operating point: min_inliers={op.threshold} -> recall={op.recall:.1%}, "
            f"FP(diff-id)={op.fp_diff_id:.1%}, FP(same-id)={op.fp_same_id:.1%}, precision={op.precision:.1%}"
        )
    else:
        lines.append("No zero-FP threshold found -- widen features or tighten geometry bounds.")
    return "\n".join(lines)
