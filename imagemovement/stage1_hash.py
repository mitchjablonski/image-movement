"""Stage 1 of the cascade: perceptual-hash candidate filter.

Permissive by design -- tuned for recall, not precision. It nominates a small
set of near-duplicate candidates and explicitly does NOT decide a match; that
is stage 2's job. A loose Hamming threshold means we would rather pass a few
extra candidates to the verifier than ever drop a true copy here.

pHash survives the JPEG compression channel because it quantizes low-frequency
DCT structure and discards the high-frequency content where compression noise
lives.
"""

from __future__ import annotations

import cv2
import imagehash
import numpy as np
from PIL import Image

from .config import Stage1Config


def _to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def phash(img_bgr: np.ndarray, hash_size: int = 8) -> imagehash.ImageHash:
    """DCT-based perceptual hash of a BGR image."""
    return imagehash.phash(_to_pil(img_bgr), hash_size=hash_size)


def hamming(a: imagehash.ImageHash, b: imagehash.ImageHash) -> int:
    """Hamming distance between two perceptual hashes."""
    return a - b


class HashIndex:
    """Recall-first near-duplicate candidate index over perceptual hashes.

    Phase-1 uses a linear scan; the query interface is shaped so a BK-tree or
    Hamming-LSH index can replace it later without touching callers.
    """

    def __init__(self, config: Stage1Config | None = None) -> None:
        self.config = config or Stage1Config()
        self._entries: list[tuple[str, imagehash.ImageHash]] = []

    def add(self, key, img_bgr: np.ndarray) -> None:
        """Index an image under a key (computes its perceptual hash)."""
        self.add_hash(key, phash(img_bgr, self.config.hash_size))

    def add_hash(self, key, image_hash: imagehash.ImageHash) -> None:
        """Index a precomputed perceptual hash (e.g. reconstructed from the corpus)."""
        self._entries.append((key, image_hash))

    def query(self, img_bgr: np.ndarray) -> list[tuple[str, int]]:
        """Return (key, distance) candidates within max_distance, nearest first."""
        h = phash(img_bgr, self.config.hash_size)
        hits = [(key, h - hh) for key, hh in self._entries]
        hits = [(key, d) for key, d in hits if d <= self.config.max_distance]
        return sorted(hits, key=lambda kd: kd[1])

    def __len__(self) -> int:
        return len(self._entries)


def _packed_code(image_hash: imagehash.ImageHash) -> np.ndarray:
    """Pack an ImageHash's bits into a faiss-ready uint8 row vector (1, nbytes)."""
    return np.packbits(image_hash.hash.flatten()).reshape(1, -1)


class FaissBinaryIndex:
    """Exact stage-1 candidate index backed by faiss IndexBinaryFlat.

    Same interface as HashIndex, and returns the IDENTICAL candidate set as the
    linear scan (a Hamming range search), just SIMD-fast and multithreaded. faiss
    range_search is exclusive (distance < radius), so we search at max_distance+1
    to reproduce the linear scan's inclusive (<= max_distance) semantics.
    """

    def __init__(self, config: Stage1Config | None = None) -> None:
        import faiss

        self.config = config or Stage1Config()
        self._bits = self.config.hash_size * self.config.hash_size
        if self._bits % 8 != 0:
            raise ValueError(
                f"hash_size={self.config.hash_size} -> {self._bits} bits, not a multiple of 8 (faiss needs whole bytes)"
            )
        self._index = faiss.IndexBinaryFlat(self._bits)
        self._keys: list = []

    def add_hash(self, key, image_hash: imagehash.ImageHash) -> None:
        self._index.add(_packed_code(image_hash))
        self._keys.append(key)

    def add(self, key, img_bgr: np.ndarray) -> None:
        self.add_hash(key, phash(img_bgr, self.config.hash_size))

    def query(self, img_bgr: np.ndarray) -> list[tuple[object, int]]:
        if self._index.ntotal == 0:
            return []
        code = _packed_code(phash(img_bgr, self.config.hash_size))
        lims, dist, ids = self._index.range_search(code, self.config.max_distance + 1)
        hits = [(self._keys[int(ids[j])], int(dist[j])) for j in range(lims[0], lims[1])]
        return sorted(hits, key=lambda kd: kd[1])

    def __len__(self) -> int:
        return len(self._keys)


def make_index(config: Stage1Config | None = None):
    """Build the configured stage-1 index: linear scan (default) or faiss (exact, faster)."""
    config = config or Stage1Config()
    if getattr(config, "index_backend", "linear") == "faiss":
        return FaissBinaryIndex(config)
    return HashIndex(config)
