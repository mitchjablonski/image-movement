# image-movement

Detect reuse of the **same core image** across user submissions, even when a
submitter has applied slight deltas (zoom, pixel shift) and the processing pipeline
has re-encoded it (JPEG compression noise). Tuned to flag **only** true copies,
never legitimately different images (including two genuine captures of the same
person).

## Approach — a three-gate cascade

1. **Stage 1, perceptual-hash filter** (`stage1_hash.py`): a fast, permissive
   pHash lookup nominates near-duplicate candidates. Tuned for *recall*.
2. **Stage 2, geometric + photometric verification** (`stage2_geom.py`): ORB
   keypoints + a RANSAC similarity transform confirm the candidate is the same
   image under a near-identity (small zoom/shift) transform, **and** a
   post-alignment photometric residual confirms the pixels actually match. Tuned
   for *precision*; emits explainable evidence (inliers + transform + residual).

> filter for recall, verify for precision

The photometric residual gate is what separates a reused image (same pixels
after alignment) from a *different photo of the same person* (same geometry, but
different pixels) — the case that pure geometric matching cannot. Validated on
full CelebA: 100% recall, 0% different-person FP, ~0.012% genuine same-person FP.

Stage 1 defaults to a pure-Python linear scan; set
`IMOVE_STAGE1__INDEX_BACKEND=faiss` to use a faiss `IndexBinaryFlat` backend
instead — **exact** (returns the identical candidate set, proven by a test) but
SIMD-fast: ~20× faster at 10k images, ~180× at 100k, staying ~1 ms/query while
the linear scan grows O(N).

All tunable thresholds live in one validated place: `config.py`
(`DetectorConfig`). Every field is overridable from the environment, e.g.
`IMOVE_STAGE2__MIN_INLIERS=30` or `IMOVE_SERVING__ALERT__MIN_DISTINCT_USERS=2`.

## Setup

```bash
uv sync
```

## Usage

Compare two images and inspect the evidence:

```bash
uv run python -m imagemovement.cli compare a.jpg b.jpg
```

Run the Phase-1 precision/recall harness. By default it fetches real faces from
LFW (cached after first download); `--synthetic` uses generated shapes instead
(no hard negatives, but zero setup):

```bash
uv run python -m imagemovement.cli eval                 # real LFW faces
uv run python -m imagemovement.cli eval --identities 40  # bigger LFW run
uv run python -m imagemovement.cli eval --synthetic      # shape fallback
```

The harness scores three categories of pairs and reports recall plus a
false-positive rate for each negative class across a swept inlier threshold:

| category | meaning | desired |
|----------|---------|---------|
| positive | a perturbed copy of an enrolled image | match |
| diff-identity negative | a different person | no match |
| **same-identity negative** | the **same person's other genuine photo** | **no match** |

The same-identity negative is the dangerous case (flagging a legitimate
returning user), so its FP rate is reported separately. The recommended
operating point is the lowest inlier threshold with **zero** false positives on
both negative classes.

## Serving — detect reuse across users/attempts

Enroll each submission into a persistent corpus (SQLite rows + lossless image
blobs), then check new submissions against it. Reuse of the same core image
across different users is the signal of interest.

```bash
# enroll submissions
uv run python -m imagemovement.cli enroll alice.jpg --user alice --attempt a1
uv run python -m imagemovement.cli enroll bob.jpg   --user bob   --attempt b1

# check a new submission for reuse against the corpus
uv run python -m imagemovement.cli check suspect.jpg
# -> REUSE DETECTED: 1 match(es) across 1 distinct user(s) -> alert=TRIGGERED ...
```

### Live HTTP demo

Run the API and open the interactive console at `/docs` to upload an image and
run enroll/check from the browser:

```bash
uv run python -m imagemovement.cli serve --port 8000
# POST /enroll (image + user_id/attempt_id), POST /check (image -> matches + alert)
# interactive console: http://127.0.0.1:8000/docs
```

Serving knobs live under `config.py`'s `ServingConfig` (env-overridable):

| knob | env var | default | effect |
|------|---------|---------|--------|
| corpus location | `IMOVE_SERVING__CORPUS_DB` / `__BLOB_DIR` | `data/corpus/...` | where rows + blobs are stored |
| alert threshold | `IMOVE_SERVING__ALERT__MIN_DISTINCT_USERS` | `1` | distinct users a reused image must hit to alert (raise to ≥2 for high-confidence-only) |
| retention TTL | `IMOVE_SERVING__TTL__ENABLED` / `__MAX_AGE_DAYS` | off | exclude/purge records older than the TTL |
| recency decay | `IMOVE_SERVING__DECAY__ENABLED` / `__HALF_LIFE_DAYS` | off | down-weight older records in alert severity |

## Layout

| file | role |
|------|------|
| `config.py` | validated pydantic config (all thresholds + perturbation space) |
| `perturb.py` | perturbation harness (validation fixtures) |
| `stage1_hash.py` | stage-1 perceptual-hash candidate filter |
| `stage2_geom.py` | stage-2 geometric verification (the decision-maker) |
| `detector.py` | the in-memory cascade that wires stage 1 → stage 2 |
| `datasets.py` | LFW loader (real faces grouped by identity) |
| `evaluate.py` | three-category harness + precision/recall metrics |
| `corpus.py` | persistent corpus (SQLite rows + lossless image blobs) |
| `service.py` | reuse-detection service: enroll / check / cross-user alerting |
| `server.py` | FastAPI HTTP transport (enroll/check endpoints + `/docs` console) |
| `cli.py` | `compare`, `eval`, `enroll`, `check`, `serve` entry points |

## Built with deepPairing

This project was designed and built end-to-end in a paired session with
[**deepPairing**](https://github.com/mitchjablonski/deepPairing) — an MCP-based
human-in-the-loop pairing tool where every decision, plan, and code change is
surfaced and reviewed in a companion UI before it lands.
