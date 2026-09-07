from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, cast

import pytest

from norax.dispatch import memory_tool


@pytest.fixture(autouse=True)
def _clear_memory_binding(monkeypatch) -> None:
    monkeypatch.setattr(memory_tool, "_MEMORY_SEARCH_FN", None)


@pytest.mark.asyncio
async def test_memory_search_validates_query_and_bounds_weak_model_k() -> None:
    assert await memory_tool.t_search_memory(query="   ") == {
        "ok": False,
        "error": "query_required",
        "query": "",
    }
    result = await memory_tool.t_search_memory(query=" q " * 20_000, k=cast(Any, "bad"))
    assert result["ok"] is True
    assert len(result["query"]) == memory_tool._MEMORY_QUERY_MAX_CHARS
    assert result["items"] == []
    assert result["note"] == "memory store not initialized"


@pytest.mark.asyncio
async def test_memory_search_awaits_and_normalizes_bounded_results(monkeypatch) -> None:
    class TextNeuron:
        text = "text" * 3_000
        kind = "kind" * 100

        def __str__(self) -> str:
            raise AssertionError("text attribute must avoid eager fallback stringification")

    async def search(query: str, *, k: int):
        assert query == "question"
        assert k == memory_tool._MEMORY_RESULT_MAX_ITEMS
        return [
            (TextNeuron(), 0.87654, "signal" * 1_000),
            (SimpleNamespace(kind="fallback"), "0.25"),
            (SimpleNamespace(text="bad"), math.nan),
            [SimpleNamespace(text="not a tuple"), 1.0],
            ("short",),
        ]

    monkeypatch.setattr(memory_tool, "_MEMORY_SEARCH_FN", search)
    result = await memory_tool.t_search_memory(query="question", k=500)
    assert result["ok"] is True
    assert len(result["items"]) == 2
    assert len(result["items"][0]["text"]) == memory_tool._MEMORY_ITEM_TEXT_MAX_CHARS
    assert len(result["items"][0]["kind"]) == memory_tool._MEMORY_ITEM_KIND_MAX_CHARS
    assert len(result["items"][0]["signals"]) == memory_tool._MEMORY_ITEM_SIGNALS_MAX_CHARS
    assert result["items"][0]["score"] == 0.877
    assert result["items"][1] == {
        "text": "namespace(kind='fallback')",
        "score": 0.25,
        "kind": "fallback",
    }


@pytest.mark.asyncio
async def test_memory_search_skips_broken_rows_and_stops_at_requested_count(monkeypatch) -> None:
    class BrokenNeuron:
        @property
        def text(self):
            raise ValueError("bad neuron")

    def search(_query: str, *, k: int):
        assert k == 2
        return [
            (BrokenNeuron(), 1.0),
            (SimpleNamespace(text="one", kind="fact"), 1.0),
            (SimpleNamespace(text="two", kind="fact"), 0.5),
            (SimpleNamespace(text="three", kind="fact"), 0.1),
        ]

    memory_tool.bind_memory_search(search)
    result = await memory_tool.t_search_memory(query="question", k=2)
    assert [item["text"] for item in result["items"]] == ["one", "two"]
    assert result["truncated"] is True


@pytest.mark.asyncio
async def test_memory_search_enforces_aggregate_character_limit(monkeypatch) -> None:
    def search(_query: str, *, k: int):
        assert k == 50
        return [(SimpleNamespace(text=str(index) * 8_000, kind="fact"), 1.0) for index in range(50)]

    monkeypatch.setattr(memory_tool, "_MEMORY_SEARCH_FN", search)
    result = await memory_tool.t_search_memory(query="question", k=50)
    assert result["truncated"] is True
    assert sum(len(item["text"]) for item in result["items"]) <= (
        memory_tool._MEMORY_RESULT_MAX_CHARS
    )


@pytest.mark.asyncio
async def test_memory_search_caps_malformed_result_scan(monkeypatch) -> None:
    def search(_query: str, *, k: int):
        yield from (["invalid"] for _ in range(memory_tool._MEMORY_RESULT_SCAN_MAX + 1))

    monkeypatch.setattr(memory_tool, "_MEMORY_SEARCH_FN", search)
    result = await memory_tool.t_search_memory(query="question", k=50)
    assert result["items"] == []
    assert result["truncated"] is True


@pytest.mark.asyncio
async def test_memory_search_scrubs_backend_failure(monkeypatch) -> None:
    secret = "sk-" + "x" * 30

    def search(_query: str, *, k: int) -> Any:
        raise RuntimeError(secret + "z" * 1_000)

    monkeypatch.setattr(memory_tool, "_MEMORY_SEARCH_FN", search)
    result = await memory_tool.t_search_memory(query="question", k=0)
    assert result["error"] == "memory_search_failed"
    assert secret not in result["detail"]
    assert "<REDACTED:openai_key>" in result["detail"]
    assert len(result["detail"]) <= 500
