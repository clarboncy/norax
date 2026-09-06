from __future__ import annotations

import builtins
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_tokenization_hash_signs_and_empty_embedding_are_deterministic() -> None:
    tokens = semantic_router._tokenize("Ab cd-ef")
    assert tokens[:2] == ["ab", "cd-ef"]
    assert {semantic_router._hash_token(token)[1] for token in ("a", "b")} == {
        -1.0,
        1.0,
    }
    assert np.array_equal(
        semantic_router.embed(""),
        np.zeros(semantic_router.EMBEDDING_DIM, dtype=np.float32),
    )


def test_embedding_handles_exact_hash_cancellation(monkeypatch) -> None:
    monkeypatch.setattr(semantic_router, "_tokenize", lambda _text: ["plus", "minus"])
    monkeypatch.setattr(
        semantic_router,
        "_hash_token",
        lambda token: (0, 1.0 if token == "plus" else -1.0),
    )
    assert np.count_nonzero(semantic_router.embed("cancel")) == 0


@pytest.mark.parametrize(
    "vectors",
    [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([[1.0, 0.0]]),
    ],
)
def test_exact_index_rejects_invalid_add_shapes(vectors) -> None:
    with pytest.raises(ValueError, match=r"expected \(\*, 3\)"):
        semantic_router._CosineIndex(3).add(vectors)


@pytest.mark.parametrize(
    "queries",
    [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([[1.0, 0.0]]),
    ],
)
def test_exact_index_rejects_invalid_query_shapes(queries) -> None:
    with pytest.raises(ValueError, match=r"expected \(\*, 3\)"):
        semantic_router._CosineIndex(3).search(queries, 1)


@pytest.mark.parametrize("k", [True, 1.5, 0, -1])
def test_exact_index_rejects_invalid_neighbor_counts(k) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        semantic_router._CosineIndex(3).search(np.zeros((1, 3)), k)


def test_empty_exact_index_returns_sentinel_results() -> None:
    distances, indices = semantic_router._CosineIndex(2).search(np.zeros((2, 2)), 3)
    assert distances.shape == indices.shape == (2, 3)
    assert np.all(np.isneginf(distances))
    assert np.all(indices == -1)


def _route(
    name: semantic_router.TaskClass = "coding",
    *,
    description: str = "write software",
    examples: tuple[str, ...] = ("implement a module",),
    depth: semantic_router.Depth = "deep",
    max_tools: int = 20,
) -> semantic_router.Route:
    return semantic_router.Route(name, description, examples, depth, max_tools)


def test_index_builders_handle_populated_zero_and_empty_corpora() -> None:
    route = _route()
    index, mapping = semantic_router.build_index((route,))
    assert index.ntotal == 1 and mapping == {0: route}

    zero_route = _route(description="", examples=())
    zero_index, zero_mapping = semantic_router.build_index((zero_route,))
    assert zero_index.ntotal == 1 and zero_mapping == {0: zero_route}
    assert np.count_nonzero(zero_index._vectors) == 0

    empty_index, empty_mapping = semantic_router.build_index(())
    assert empty_index.ntotal == 0 and empty_mapping == {}

    example_index, example_mapping = semantic_router.SemanticRouter._build_example_index((route,))
    assert example_index.ntotal == 1 and example_mapping == {0: route}
    no_examples, no_example_mapping = semantic_router.SemanticRouter._build_example_index(
        (zero_route,)
    )
    assert no_examples.ntotal == 0 and no_example_mapping == {}


class _StaticIndex:
    def __init__(self, score: float = 0.0, index: int = -1, *, present: bool = True):
        self.ntotal = int(present)
        self.score = score
        self.index = index

    def search(self, _queries, k):
        assert k == 1
        return np.asarray([[self.score]]), np.asarray([[self.index]])


def _controlled_router(
    *,
    example_score: float = 0.0,
    centroid_score: float = 0.0,
    example_present: bool = True,
    centroid_present: bool = True,
    example_mapped: bool = True,
    centroid_mapped: bool = True,
):
    route = _route()
    router = semantic_router.SemanticRouter(())
    router._example_index = _StaticIndex(example_score, 0, present=example_present)
    router.index = _StaticIndex(centroid_score, 0, present=centroid_present)
    router._example_id_to_route = {0: route} if example_mapped else {}
    router.id_to_route = {0: route} if centroid_mapped else {}
    return router


@pytest.mark.parametrize("message", ["", "   "])
def test_router_treats_empty_and_whitespace_as_trivial(message) -> None:
    result = semantic_router.SemanticRouter(()).route(message)
    assert result.task_class == "trivial_social"
    assert result.max_tool_calls == 0


def test_router_uses_high_confidence_example_match() -> None:
    result = _controlled_router(example_score=0.8, centroid_score=0.7).route("build it")
    assert (result.task_class, result.depth, result.max_tool_calls) == (
        "coding",
        "deep",
        20,
    )
    assert result.confidence == pytest.approx(0.8)
    assert "example_match" in result.reason


def test_router_uses_medium_confidence_centroid_and_caps_budget() -> None:
    result = _controlled_router(example_score=0.4, centroid_score=0.6).route("build it")
    assert (result.task_class, result.depth, result.max_tool_calls) == (
        "coding",
        "normal",
        8,
    )
    assert "centroid_match" in result.reason


def test_router_returns_ambiguous_for_low_confidence() -> None:
    result = _controlled_router(example_score=0.2, centroid_score=0.1).route("something")
    assert (result.task_class, result.depth, result.max_tool_calls) == (
        "ambiguous",
        "normal",
        8,
    )


def test_router_clamps_out_of_range_similarity_scores() -> None:
    high = _controlled_router(example_score=2.0, centroid_score=1.0).route("high")
    low = _controlled_router(example_score=-0.5, centroid_score=-0.8).route("low")
    assert high.confidence == 1.0
    assert low.confidence == 0.0


def test_router_falls_back_when_indexes_have_no_mapped_result() -> None:
    absent = _controlled_router(example_present=False, centroid_present=False).route("x")
    unmapped = _controlled_router(
        example_mapped=False,
        centroid_mapped=False,
    ).route("x")
    assert absent.reason == unmapped.reason == "no_similarity_match"


def test_module_router_is_lazy_reused_and_convenience_function_delegates(monkeypatch) -> None:
    marker_result = semantic_router.SemanticClassification(
        "coding", "deep", 1.0, 250, "coding", "test"
    )
    marker = SimpleNamespace(route=lambda message: (message, marker_result))
    monkeypatch.setattr(semantic_router, "_router", None)
    monkeypatch.setattr(semantic_router, "SemanticRouter", lambda: marker)
    assert semantic_router.get_router() is marker
    assert semantic_router.get_router() is marker
    assert semantic_router.route("delegate") == ("delegate", marker_result)
