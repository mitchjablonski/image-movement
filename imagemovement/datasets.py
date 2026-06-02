"""Real face data for validation: LFW (identity-grouped) and CelebA (higher-res).

LFW (via scikit-learn) carries identity labels -> the same-person hard negative.
CelebA is pulled on demand from the HuggingFace datasets-server at full
resolution (178x218); this mirror's images are served per-row without identity
labels, so each CelebA image is treated as its own identity (validating recall +
different-image precision at higher resolution), while the same-person hard
negative stays covered by LFW.
"""

from __future__ import annotations

import collections
import json
import time
import urllib.request

import cv2
import numpy as np


def _to_bgr_uint8(rgb_float: np.ndarray) -> np.ndarray:
    """LFW images are float32 RGB in [0, 1]; convert to BGR uint8."""
    rgb = np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_lfw(
    min_faces_per_person: int = 5,
    max_identities: int = 25,
    images_per_identity: int = 3,
    resize: float = 1.0,
) -> list[tuple[str, list[np.ndarray]]]:
    """Return [(identity_name, [bgr_image, ...]), ...] for the eval.

    Only identities with at least ``min_faces_per_person`` photos are fetched (so
    a same-identity hard negative exists), then capped to ``images_per_identity``
    photos each and ``max_identities`` identities to keep the run tractable.
    """
    from sklearn.datasets import fetch_lfw_people

    data = fetch_lfw_people(min_faces_per_person=min_faces_per_person, color=True, resize=resize)
    by_id: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    for img, target in zip(data.images, data.target):
        name = str(data.target_names[target])
        if len(by_id[name]) < images_per_identity:
            by_id[name].append(_to_bgr_uint8(img))

    identities = [(name, imgs) for name, imgs in sorted(by_id.items()) if len(imgs) >= 2]
    return identities[:max_identities]


def _http_get(url: str, tries: int = 3) -> bytes:
    """GET with a few retries (the datasets-server can be transiently flaky)."""
    last: Exception | None = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "image-movement/0.1"})
            return urllib.request.urlopen(req, timeout=30).read()
        except Exception as e:  # noqa: BLE001 - retry any transient network error
            last = e
            time.sleep(1.5)
    raise RuntimeError(f"GET failed after {tries} tries: {url}") from last


def load_celeba(
    n_images: int = 30,
    offset: int = 0,
    hf_dataset: str = "nielsr/CelebA-faces",
) -> list[tuple[str, list[np.ndarray]]]:
    """Fetch CelebA faces (full-res 178x218 BGR) from the HF datasets-server.

    Each image is returned as its own single-image identity: this mirror serves
    images without identity labels, so this validates recall + different-image
    precision at higher resolution. The same-person hard negative stays covered
    by load_lfw. Images are pulled on demand (no bulk download).
    """
    base = "https://datasets-server.huggingface.co/rows"
    out: list[tuple[str, list[np.ndarray]]] = []
    fetched = 0
    while fetched < n_images:
        length = min(100, n_images - fetched)
        url = (
            f"{base}?dataset={hf_dataset}&config=default&split=train"
            f"&offset={offset + fetched}&length={length}"
        )
        rows = json.loads(_http_get(url)).get("rows", [])
        if not rows:
            break
        for r in rows:
            src = r["row"]["image"]["src"]
            arr = cv2.imdecode(np.frombuffer(_http_get(src), np.uint8), cv2.IMREAD_COLOR)
            if arr is not None:
                out.append((f"celeba_{offset + fetched:06d}", [arr]))
            fetched += 1
    return out
