"""Perturbation harness for reuse-detection validation.

Replays the transformation + capture chain on a seed image so we can generate
labeled near-duplicate ("same core image, slightly moved") positives for
validation. Order matters: geometric manipulation happens first, then the
JPEG re-encode (with realistic chroma subsampling) that the processing pipeline
applies on receipt, so compression noise lands on the already-transformed
pixels -- exactly what the detector will see in production.

The amount of perturbation is read from config.AdversarySpace, the single
place the perturbation space is defined.

Images are OpenCV BGR uint8 ndarrays throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import AdversarySpace


@dataclass(frozen=True)
class PerturbParams:
    """One sampled point in the perturbation space (a value object)."""

    quality: int          # JPEG quality factor (lower = more compression noise)
    zoom_factor: float    # >1 zooms in (crops), <1 zooms out (pads)
    dx: int               # horizontal pixel shift
    dy: int               # vertical pixel shift
    subsampling: str = "4:2:0"   # chroma subsampling used on re-encode


def jpeg_recompress(img: np.ndarray, quality: int, subsampling: str = "4:2:0") -> np.ndarray:
    """Round-trip through JPEG to inject compression noise.

    At high quality the dominant error source is chroma subsampling, not the
    quality factor -- so it is modelled explicitly. '4:2:0' matches real camera
    JPEGs; '4:4:4' disables subsampling and is near-lossless. Either way the
    result is never bit-identical to the source, which is why exact-pixel
    comparison is hopeless.
    """
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    if subsampling == "4:4:4":
        params += [int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                   int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)]
    ok, buf = cv2.imencode(".jpg", img, params)
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def zoom(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale by ``factor`` and return an image of the original size.

    factor > 1 -> zoom in (resize up, center-crop back).
    factor < 1 -> zoom out (resize down, reflect-pad back).
    """
    if factor == 1.0:
        return img.copy()
    h, w = img.shape[:2]
    nh, nw = max(1, round(h * factor)), max(1, round(w * factor))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if factor >= 1.0:
        y0, x0 = (nh - h) // 2, (nw - w) // 2
        return resized[y0:y0 + h, x0:x0 + w]
    pad_y, pad_x = (h - nh) // 2, (w - nw) // 2
    return cv2.copyMakeBorder(
        resized, pad_y, h - nh - pad_y, pad_x, w - nw - pad_x,
        borderType=cv2.BORDER_REFLECT_101,
    )


def translate(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift the image by (dx, dy) pixels, reflecting at the borders."""
    if dx == 0 and dy == 0:
        return img.copy()
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def perturb(img: np.ndarray, params: PerturbParams) -> np.ndarray:
    """Apply the full chain in submission order: zoom -> shift -> JPEG re-encode."""
    out = zoom(img, params.zoom_factor)
    out = translate(out, params.dx, params.dy)
    return jpeg_recompress(out, params.quality, params.subsampling)


def _signed_shift(rng: np.random.Generator, lo: int, hi: int) -> int:
    """Sample a shift magnitude in [lo, hi] and give it a random direction."""
    mag = int(rng.integers(lo, hi + 1))
    return mag if rng.random() < 0.5 else -mag


def random_params(space: AdversarySpace, rng: np.random.Generator) -> PerturbParams:
    """Sample one PerturbParams from the configured perturbation space."""
    return PerturbParams(
        quality=int(rng.integers(space.quality_min, space.quality_max + 1)),
        zoom_factor=float(rng.uniform(space.zoom_min, space.zoom_max)),
        dx=_signed_shift(rng, space.shift_min_px, space.shift_max_px),
        dy=_signed_shift(rng, space.shift_min_px, space.shift_max_px),
        subsampling=space.subsampling,
    )


def generate_variants(
    img: np.ndarray,
    n: int,
    rng: np.random.Generator,
    space: AdversarySpace,
) -> list[tuple[np.ndarray, PerturbParams]]:
    """Produce ``n`` labeled positive variants of ``img`` from the perturbation space."""
    out = []
    for _ in range(n):
        params = random_params(space, rng)
        out.append((perturb(img, params), params))
    return out
