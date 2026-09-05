from __future__ import annotations

import builtins

import numpy as np

from norax.brain.hot_path import semantic_router


def test_feature_hashing_does_not_depend_on_process_randomized_hash(monkeypatch) -> None:
    def _unstable_hash(_value):
        raise AssertionError("built-in hash must not determine route embeddings")

    monkeypatch.setattr(builtins, "hash", _unstable_hash)

    first = semantic_router.embed("debug the failing service")
    second = semantic_router.embed("debug the failing service")

    assert np.array_equal(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0)


def test_small_exact_index_returns_real_inner_product_order() -> None:
    index = semantic_router._CosineIndex(3)
    index.add(np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))

    distances, indices = index.search(
        np.asarray([[0.9, 0.1, 0.0]], dtype=np.float32),
        2,
    )

    assert indices.tolist() == [[0, 1]]
    assert np.allclose(distances, [[0.9, 0.1]])
