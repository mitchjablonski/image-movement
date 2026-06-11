# image-movement

Detect reuse of the **same core image** — the same picture appearing more than
once — even when it has been altered by slight deltas (zoom, pixel shift) and
re-encoded (JPEG compression noise). Tuned to flag **only** true copies,
never legitimately different images (including two genuine captures of the same
person).

> 🤝 **Built with [deepPairing](https://github.com/mitchjablonski/deepPairing)** — designed and built end-to-end in a paired session with an MCP-based human-in-the-loop tool, where every decision, plan, and code change is surfaced and reviewed in a companion UI before it lands.

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
full CelebA (shipped `max_residual=12`): 99.95% recall, ~0% different-person FP,
and **no genuine same-person false positives** — only near-duplicate photos remain.

Stage 1 defaults to a pure-Python linear scan; set
`IMOVE_STAGE1__INDEX_BACKEND=faiss` (requires the `faiss` extra) to use a faiss
`IndexBinaryFlat` backend instead — **exact** (returns the identical candidate set, proven by a test) but
SIMD-fast: ~20× faster at 10k images, ~180× at 100k, staying ~1 ms/query while
the linear scan grows O(N).

All tunable thresholds live in one validated place: `config.py`
(`DetectorConfig`). Every field is overridable from the environment, e.g.
`IMOVE_STAGE2__MIN_INLIERS=30` or `IMOVE_SERVING__ALERT__MIN_DISTINCT_USERS=2`.

## Setup

```bash
uv sync --all-extras
```

Core install (`uv sync`) covers `compare` and the in-memory detector. Optional
extras: `eval` (scikit-learn, for the LFW harness), `server` (FastAPI/uvicorn,
for the HTTP API), `faiss` (the fast stage-1 backend).

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

## Validation & datasets

The cascade was stress-tested on real face datasets — deliberately the *hardest*
case, since two different photos of the **same person** (especially face-aligned)
are the most likely to fool a geometry-only matcher. Faces are a stress test, not
the use case: the detector targets reuse of *any* image.

- **CelebA** — full set (8,156 identities; 16,312 true copies; ~16k same-person
  pairs; ~163k different-person pairs), at the shipped default `max_residual=12`:
  **99.95% recall, ~0% different-person false positives (1–2 of 162,910), and
  zero genuine same-person false positives**. The photometric residual gate
  rejects even the borderline *distinct-photo* same-person matches; the only
  same-person matches that survive (22, ~0.1%) are near-duplicate photos —
  effectively the same image. (Tightening the residual gate from 15 to 12 to drop
  those borderline cases cost ~0.05% recall — a deliberate, favourable trade.)
- **LFW** — same pipeline; recall is lower only because the ~94 px thumbnails are
  keypoint-poor, and recovers at higher resolution.

Both datasets are fetched on demand by `imagemovement/datasets.py` and are **not
redistributed** in this repository.

> **Dataset terms** (these apply to the datasets, **not** to this MIT-licensed code):
> **CelebA** is for *non-commercial research use only* — Liu et al.,
> [*Deep Learning Face Attributes in the Wild*](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), ICCV 2015.
> **LFW** is freely usable with attribution — Huang et al.,
> [*Labeled Faces in the Wild*](https://vis-www.cs.umass.edu/lfw/), UMass TR 07-49, 2007.

## Serving — persistent corpus + reuse alerting

Enroll each image into a persistent corpus (SQLite rows + lossless image blobs),
then check new images against it — the same core image appearing more than once
is the signal of interest. Each image can carry an optional source id (e.g. a
user), so an alert can require the image to recur across several distinct sources.

```bash
# enroll images (an optional source id groups matches; here, a user)
uv run python -m imagemovement.cli enroll alice.jpg --user alice --attempt a1
uv run python -m imagemovement.cli enroll bob.jpg   --user bob   --attempt b1

# check a new image for reuse against the corpus
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

## Security & privacy notes

The HTTP server is a **localhost demo**, not a hardened service. It binds
127.0.0.1 by default and deliberately ships without:

- **authentication or rate limiting** — anyone who can reach it can enroll and
  query images;
- **upload limits** — requests are read fully into memory and decoded with
  OpenCV, so cap request size (and decoded dimensions) at a gateway before
  exposing it;
- **source isolation** — match responses intentionally include the matched
  records' `user_id`/`attempt_id`, because cross-source recurrence *is* the
  signal being reported.

Enrolled images are retained **indefinitely** as lossless PNGs (stage-2
verification re-checks actual pixels); enable the TTL
(`IMOVE_SERVING__TTL__ENABLED=1`) to bound retention. Anything beyond a local
demo should sit behind a real gateway (auth, rate limits, body caps) and treat
the corpus directory as sensitive data.

## Layout

| file | role |
|------|------|
| `config.py` | validated pydantic config (all thresholds + perturbation space) |
| `perturb.py` | perturbation harness (validation fixtures) |
| `stage1_hash.py` | stage-1 perceptual-hash candidate filter |
| `stage2_geom.py` | stage-2 geometric verification (the decision-maker) |
| `detector.py` | the in-memory cascade that wires stage 1 → stage 2 |
| `datasets.py` | dataset loaders (LFW + CelebA) for validation |
| `evaluate.py` | three-category harness + precision/recall metrics |
| `corpus.py` | persistent corpus (SQLite rows + lossless image blobs) |
| `service.py` | reuse-detection service: enroll / check / cross-user alerting |
| `server.py` | FastAPI HTTP transport (enroll/check endpoints + `/docs` console) |
| `cli.py` | `compare`, `eval`, `enroll`, `check`, `serve` entry points |
