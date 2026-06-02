"""Stage-1 index backends: faiss must return the SAME candidates as the linear scan."""

from __future__ import annotations

import pytest

from imagemovement.config import Stage1Config
from imagemovement.data import synthetic_seeds
from imagemovement.perturb import PerturbParams, perturb
from imagemovement.stage1_hash import HashIndex, make_index

faiss = pytest.importorskip("faiss")


def test_faiss_index_returns_same_candidates_as_linear():
    seeds = synthetic_seeds(16)
    lin = HashIndex(Stage1Config())
    fai = make_index(Stage1Config(index_backend="faiss"))
    assert type(fai).__name__ == "FaissBinaryIndex"
    for name, img in seeds:
        lin.add(name, img)
        fai.add(name, img)

    # Originals (exact, dist 0) + perturbed copies as queries.
    queries = [img for _, img in seeds]
    queries += [perturb(img, PerturbParams(quality=78, zoom_factor=1.08, dx=4, dy=-3)) for _, img in seeds]
    for q in queries:
        lr = lin.query(q)
        fr = fai.query(q)
        assert {k for k, _ in lr} == {k for k, _ in fr}   # same candidate set
        assert dict(lr) == dict(fr)                        # same key -> Hamming distance


def test_make_index_default_is_linear():
    assert isinstance(make_index(Stage1Config()), HashIndex)


def test_faiss_empty_query_returns_nothing():
    fai = make_index(Stage1Config(index_backend="faiss"))
    assert fai.query(synthetic_seeds(1)[0][1]) == []
