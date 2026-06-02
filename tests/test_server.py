"""Acceptance tests for the FastAPI HTTP transport (headless via TestClient)."""

from __future__ import annotations

import io
import os

import cv2
from fastapi.testclient import TestClient

from imagemovement.config import DetectorConfig
from imagemovement.data import synthetic_seeds
from imagemovement.perturb import PerturbParams, perturb
from imagemovement.server import create_app


def _png_bytes(img) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _client(tmp_path) -> TestClient:
    cfg = DetectorConfig()
    cfg.serving.corpus_db = os.path.join(str(tmp_path), "corpus.db")
    cfg.serving.blob_dir = os.path.join(str(tmp_path), "blobs")
    return TestClient(create_app(cfg))


def test_enroll_then_check_detects_reuse(tmp_path):
    client = _client(tmp_path)
    seeds = synthetic_seeds(2)
    orig, other = seeds[0][1], seeds[1][1]

    r = client.post(
        "/enroll",
        data={"user_id": "alice", "attempt_id": "a1"},
        files={"file": ("orig.png", _png_bytes(orig), "image/png")},
    )
    assert r.status_code == 200
    assert r.json()["corpus_size"] == 1

    # A perturbed copy submitted by someone else -> reuse + alert.
    copy = perturb(orig, PerturbParams(quality=80, zoom_factor=1.10, dx=5, dy=-4))
    r = client.post("/check", files={"file": ("copy.png", _png_bytes(copy), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert len(body["matches"]) == 1
    assert body["matches"][0]["user_id"] == "alice"
    assert body["alert"]["triggered"] is True

    # A genuinely different image -> no reuse.
    r = client.post("/check", files={"file": ("other.png", _png_bytes(other), "image/png")})
    assert r.status_code == 200
    assert r.json()["matches"] == []
    assert r.json()["alert"]["triggered"] is False


def test_submit_detects_reuse_at_intake(tmp_path):
    client = _client(tmp_path)
    orig = synthetic_seeds(1)[0][1]

    # First submission: nothing in the corpus yet -> clean, but enrolled.
    r = client.post(
        "/submit",
        data={"user_id": "alice", "attempt_id": "a1"},
        files={"file": ("orig.png", _png_bytes(orig), "image/png")},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["record_id"] == 1
    assert b["matches"] == []
    assert b["alert"]["triggered"] is False

    # Second submission by another user is a perturbed copy -> reuse flagged AT intake,
    # and still enrolled (corpus grows to 2).
    copy = perturb(orig, PerturbParams(quality=80, zoom_factor=1.10, dx=5, dy=-4))
    r = client.post(
        "/submit",
        data={"user_id": "bob", "attempt_id": "b1"},
        files={"file": ("copy.png", _png_bytes(copy), "image/png")},
    )
    assert r.status_code == 200
    b = r.json()
    assert b["corpus_size"] == 2
    assert len(b["matches"]) == 1
    assert b["matches"][0]["user_id"] == "alice"
    assert b["alert"]["triggered"] is True


def test_bad_upload_is_rejected(tmp_path):
    client = _client(tmp_path)
    r = client.post("/check", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400
