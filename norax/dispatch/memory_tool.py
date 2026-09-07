"""Bounded semantic-memory tool adapter."""

from __future__ import annotations

import inspect
import math
from typing import Any

from ..safety.secrets import redact
from .input_coercion import _coerce_int

_MEMORY_QUERY_MAX_CHARS = 20_000
_MEMORY_ITEM_TEXT_MAX_CHARS = 8_000
_MEMORY_ITEM_SIGNALS_MAX_CHARS = 2_000
_MEMORY_ITEM_KIND_MAX_CHARS = 128
_MEMORY_RESULT_MAX_ITEMS = 50
_MEMORY_RESULT_MAX_CHARS = 32_000
_MEMORY_RESULT_SCAN_MAX = 200

_MEMORY_SEARCH_FN = None


def _memory_diagnostic(value: object) -> str:
    return str(redact(str(value)))[:500]


def bind_memory_search(search_fn) -> None:
    """Bind a live memory search function. Called by Runtime.build()."""
    global _MEMORY_SEARCH_FN
    _MEMORY_SEARCH_FN = search_fn


async def t_search_memory(*, query: str, k: int = 5) -> dict:
    """Search semantic memory without allowing unbounded result materialization."""
    normalized_query = str(query or "").strip()[:_MEMORY_QUERY_MAX_CHARS]
    if not normalized_query:
        return {"ok": False, "error": "query_required", "query": ""}
    parsed_k = _coerce_int(k, default=5) or 5
    result_limit = max(1, min(_MEMORY_RESULT_MAX_ITEMS, parsed_k))
    if _MEMORY_SEARCH_FN is None:
        return {
            "ok": True,
            "query": normalized_query,
            "items": [],
            "note": "memory store not initialized",
        }
    try:
        results = _MEMORY_SEARCH_FN(normalized_query, k=result_limit)
        if inspect.isawaitable(results):
            results = await results
        items: list[dict[str, Any]] = []
        total_chars = 0
        truncated = False
        for row_index, row in enumerate(results):
            if row_index >= _MEMORY_RESULT_SCAN_MAX:
                truncated = True
                break
            if len(items) >= result_limit:
                truncated = True
                break
            if not isinstance(row, tuple) or len(row) < 2:
                continue
            try:
                neuron, raw_score = row[0], row[1]
                score = float(raw_score)
                if not math.isfinite(score):
                    continue
                raw_text = getattr(neuron, "text", None)
                text = str(raw_text if raw_text is not None else neuron)[
                    :_MEMORY_ITEM_TEXT_MAX_CHARS
                ]
                kind = str(getattr(neuron, "kind", ""))[:_MEMORY_ITEM_KIND_MAX_CHARS]
                source = row[2] if len(row) >= 3 else ""
                signals = str(source)[:_MEMORY_ITEM_SIGNALS_MAX_CHARS] if source else ""
            except Exception:
                continue
            item_chars = len(text) + len(kind) + len(signals)
            if items and total_chars + item_chars > _MEMORY_RESULT_MAX_CHARS:
                truncated = True
                break
            item: dict[str, Any] = {
                "text": text,
                "score": round(score, 3),
                "kind": kind,
            }
            if signals:
                item["signals"] = signals
            items.append(item)
            total_chars += item_chars
        response: dict[str, Any] = {
            "ok": True,
            "query": normalized_query,
            "items": items,
        }
        if truncated:
            response["truncated"] = True
        return response
    except Exception as exc:
        return {
            "ok": False,
            "error": "memory_search_failed",
            "detail": _memory_diagnostic(exc),
            "query": normalized_query,
        }
