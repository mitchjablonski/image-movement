"""SQLite-backed corpus store for the reuse-detection service.

Each enrolled image becomes a SQLite row (submission metadata + perceptual-hash
hex + blob path + timestamp) plus a lossless PNG blob on disk. Stage-2
verification re-derives ORB features and the photometric residual from the
ACTUAL pixels, so the corpus retains the image, not just a fingerprint; PNG is
lossless so the stored reference doesn't re-introduce JPEG noise. SQLite +
filesystem blobs is durable and zero-infra (the resolved Phase-2 storage choice).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .stage1_hash import phash


@dataclass
class CorpusRecord:
    """One enrolled image's metadata (pixels live in the blob at blob_path)."""

    id: int
    user_id: str
    attempt_id: str
    phash: str          # perceptual hash as hex (imagehash.hex_to_hash to reconstruct)
    blob_path: str
    enrolled_at: float  # unix seconds


class CorpusStore:
    """Persistent corpus of enrolled images: SQLite rows + lossless PNG blobs."""

    def __init__(self, db_path: str, blob_dir: str, hash_size: int = 8) -> None:
        self.db_path = Path(db_path)
        self.blob_dir = Path(blob_dir)
        self.hash_size = hash_size
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI runs routes in worker threads, so the
        # connection is shared across threads; mutating ops are serialized by _lock.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                phash      TEXT NOT NULL,
                blob_path  TEXT NOT NULL,
                enrolled_at REAL NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_records_user ON records(user_id)")
        self._conn.commit()

    def enroll(self, image: np.ndarray, user_id: str, attempt_id: str, *, now: float | None = None) -> CorpusRecord:
        """Persist a BGR image + its metadata; returns the stored record (with a stable id)."""
        ts = time.time() if now is None else now
        h = str(phash(image, self.hash_size))
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO records(user_id, attempt_id, phash, blob_path, enrolled_at) VALUES (?,?,?,?,?)",
                (user_id, attempt_id, h, "", ts),
            )
            rec_id = int(cur.lastrowid)
            blob_path = self.blob_dir / f"{rec_id}.png"   # PNG = lossless, faithful pixels for stage 2
            if not cv2.imwrite(str(blob_path), image):
                raise RuntimeError(f"failed to write blob {blob_path}")
            self._conn.execute("UPDATE records SET blob_path=? WHERE id=?", (str(blob_path), rec_id))
            self._conn.commit()
        return CorpusRecord(rec_id, user_id, attempt_id, h, str(blob_path), ts)

    def get_image(self, rec_id: int) -> np.ndarray:
        """Load the original pixels for an enrolled record (for stage-2 verification)."""
        row = self._conn.execute("SELECT blob_path FROM records WHERE id=?", (rec_id,)).fetchone()
        if row is None:
            raise KeyError(rec_id)
        img = cv2.imread(row["blob_path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(row["blob_path"])
        return img

    def iter_records(self, *, max_age_days: float | None = None, now: float | None = None):
        """Yield records, optionally excluding those older than max_age_days (TTL)."""
        if max_age_days is None:
            rows = self._conn.execute("SELECT * FROM records").fetchall()
        else:
            ts = time.time() if now is None else now
            cutoff = ts - max_age_days * 86400.0
            rows = self._conn.execute("SELECT * FROM records WHERE enrolled_at >= ?", (cutoff,)).fetchall()
        for r in rows:
            yield CorpusRecord(r["id"], r["user_id"], r["attempt_id"], r["phash"], r["blob_path"], r["enrolled_at"])

    def purge_expired(self, max_age_days: float, *, now: float | None = None) -> int:
        """Delete records (rows + blobs) older than max_age_days; returns count removed."""
        ts = time.time() if now is None else now
        cutoff = ts - max_age_days * 86400.0
        with self._lock:
            rows = self._conn.execute("SELECT id, blob_path FROM records WHERE enrolled_at < ?", (cutoff,)).fetchall()
            for r in rows:
                Path(r["blob_path"]).unlink(missing_ok=True)
            self._conn.execute("DELETE FROM records WHERE enrolled_at < ?", (cutoff,))
            self._conn.commit()
        return len(rows)

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS n FROM records").fetchone()["n"])

    def close(self) -> None:
        self._conn.close()
