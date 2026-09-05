from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

import norax.gateway_client as gateway_module
from norax.gateway_client import SpendGuard, SpendGuardTripped
from norax.memory.retrievers import cross_encoder
from norax.memory.retrievers.cross_encoder import LLMJudgeReranker, rerank
from norax.memory.retrievers.fast import FastContext
from norax.memory.retrievers.multi_signal import MultiSignalRetriever
from norax.memory.store import MemoryStore


def _ollama_response(content: str, *, model: str = "judge") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 20,
            "eval_count": 8,
        },
    )


@pytest.fixture(autouse=True)
def _clear_rerank_state():
    cross_encoder._RERANK_CACHE.clear()
    cross_encoder._circuit_open_until.clear()
    yield
    cross_encoder._RERANK_CACHE.clear()
    cross_encoder._circuit_open_until.clear()


@pytest.mark.asyncio
@respx.mock
async def test_llm_judge_uses_configured_blend_and_structured_gateway() -> None:
    route = respx.post("http://judge/api/chat").mock(
        return_value=_ollama_response('{"scores":[0.0,1.0]}')
    )
    candidates = [("first", 1.0, "rrf"), ("second", 0.5, "rrf")]

    result = await rerank(
        "target",
        candidates,
        model="judge-a",
        blend_weight=1.0,
        base_url="http://judge",
    )

    assert [item for item, _score, _tag in result] == ["second", "first"]
    assert all(tag.endswith("+llm-rerank") for _item, _score, tag in result)
    assert route.call_count == 1
    payload = json.loads(route.calls[0].request.content)
    assert payload["think"] is False
    assert payload["format"]["properties"]["scores"]["minItems"] == 2
    assert payload["format"]["properties"]["scores"]["maxItems"] == 2


@pytest.mark.asyncio
@respx.mock
async def test_incomplete_scores_preserve_fused_ranking_without_fake_tag() -> None:
    respx.post("http://judge/api/chat").mock(return_value=_ollama_response('{"scores":[0.99]}'))
    candidates = [("first", 1.0, "rrf"), ("second", 0.5, "rrf")]

    result = await rerank(
        "target",
        candidates,
        model="judge-a",
        base_url="http://judge",
    )

    assert result == candidates


@pytest.mark.asyncio
@respx.mock
async def test_cache_is_scoped_to_the_selected_judge_model() -> None:
    route = respx.post("http://judge/api/chat").mock(
        side_effect=[
            _ollama_response('{"scores":[1.0,0.0]}', model="judge-a"),
            _ollama_response('{"scores":[0.0,1.0]}', model="judge-b"),
        ]
    )
    candidates = [("first", 1.0, "rrf"), ("second", 1.0, "rrf")]

    first = await rerank(
        "target",
        candidates,
        model="judge-a",
        blend_weight=1.0,
        base_url="http://judge",
    )
    second = await rerank(
        "target",
        candidates,
        model="judge-b",
        blend_weight=1.0,
        base_url="http://judge",
    )

    assert first[0][0] == "first"
    assert second[0][0] == "second"
    assert route.call_count == 2


@pytest.mark.asyncio
async def test_spend_guard_propagates_before_judge_transport(monkeypatch) -> None:
    guard = SpendGuard()
    guard.per_min = 1
    guard.per_hour = 0
    guard.record_and_check()
    monkeypatch.setattr(gateway_module, "_SPEND_GUARD", guard)

    with pytest.raises(SpendGuardTripped):
        await rerank(
            "target",
            [("first", 1.0, "rrf")],
            model="judge-a",
            base_url="http://judge",
        )


@pytest.mark.asyncio
async def test_multisignal_does_not_swallow_spend_guard(tmp_path: Path) -> None:
    semantic = tmp_path / "semantic"
    semantic.mkdir()
    (semantic / "facts.md").write_text("TARGET:relevant memory|W4\n")
    store = MemoryStore(root=tmp_path)
    store.refresh()

    class GuardReranker:
        async def rerank(self, query, candidates, *, top_k=None):
            del query, candidates, top_k
            raise SpendGuardTripped("minute", 1, 1)

    retriever = MultiSignalRetriever(
        keyword=FastContext(store=store),
        reranker=GuardReranker(),  # type: ignore[arg-type]
    )

    with pytest.raises(SpendGuardTripped):
        await retriever.search("relevant target", k=2)


@pytest.mark.asyncio
async def test_disabled_judge_honors_top_k_zero_without_a_model() -> None:
    judge = LLMJudgeReranker(enabled=False)
    assert await judge.rerank("target", [("first", 1.0, "rrf")], top_k=0) == []


def test_enabled_judge_requires_model_and_valid_weight() -> None:
    with pytest.raises(ValueError, match="explicit model"):
        LLMJudgeReranker(model="", enabled=True)
    with pytest.raises(ValueError, match="blend_weight"):
        LLMJudgeReranker(model="judge", blend_weight=float("nan"))
