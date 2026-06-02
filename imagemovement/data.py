"""Seed image loading, with a synthetic fallback.

Real validation should run on representative submission images dropped into
``data/seeds/``. When none are present we generate a small set of distinct,
feature-rich synthetic images so the whole pipeline is runnable end-to-end
immediately -- each has sharp shapes (plenty of ORB keypoints) and is visually
distinct from the others (so the negative set is meaningful).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as a BGR uint8 ndarray."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def load_seeds(seeds_dir: str | Path) -> list[tuple[str, np.ndarray]]:
    """Load every image in a directory as (name, image) pairs."""
    d = Path(seeds_dir)
    if not d.is_dir():
        return []
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [(p.stem, load_image(p)) for p in paths]


def synthetic_seeds(n: int, size: tuple[int, int] = (256, 256)) -> list[tuple[str, np.ndarray]]:
    """Generate ``n`` distinct, keypoint-rich synthetic seed images."""
    h, w = size
    seeds: list[tuple[str, np.ndarray]] = []
    for i in range(n):
        # Deterministic per-seed RNG so runs are reproducible.
        r = np.random.default_rng(i + 1)
        # Smooth random color field (upsampled 8x8 noise) gives low-freq texture.
        base = r.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
        # Sharp shapes give strong, localizable corners for ORB to lock onto.
        for _ in range(14):
            color = tuple(int(c) for c in r.integers(0, 256, size=3))
            thickness = int(r.choice([-1, 1, 2, 3]))
            kind = int(r.integers(0, 3))
            a = (int(r.integers(0, w)), int(r.integers(0, h)))
            b = (int(r.integers(0, w)), int(r.integers(0, h)))
            if kind == 0:
                cv2.rectangle(img, a, b, color, thickness)
            elif kind == 1:
                cv2.circle(img, a, int(r.integers(8, 48)), color, thickness)
            else:
                cv2.line(img, a, b, color, max(1, thickness))
        seeds.append((f"synimg_{i:02d}", img))
    return seeds
