from __future__ import annotations

import asyncio
from collections import OrderedDict

import pytest

import norax.dispatch.idempotency as idempotency_module
from norax.dispatch import Caller, Dispatcher
from norax.dispatch.idempotency import IdempotencyCache
from norax.dispatch.tools import REGISTRY, ToolSpec


def test_completed_cache_returns_independent_values_and_is_bounded() -> None:
    cache = IdempotencyCache(max_entries=2)
    original = {"ok": True, "nested": {"value": 1}}
    cache.put("one", original, scope="scope-one")
    original["nested"]["value"] = 9

    first = cache.get("one", scope="scope-one")
    assert first == {"ok": True, "nested": {"value": 1}}
    assert first is not None
    first["nested"]["value"] = 8
    assert cache.get("one", scope="scope-one") == {
        "ok": True,
        "nested": {"value": 1},
    }

    cache.put("two", {"ok": True}, scope="scope-two")
    cache.put("three", {"ok": True}, scope="scope-three")
    assert cache.get("one", scope="scope-one") is None


def test_expired_key_can_be_reused_with_a_different_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setattr(idempotency_module.time, "monotonic", lambda: now)
    cache = IdempotencyCache(ttl_seconds=5)
    cache.put("request", {"ok": True}, scope="old")

    now = 106.0
    assert cache.get("request", scope="old") is None
    cache.put("request", {"ok": True, "new": True}, scope="new")

    assert cache.get("request", scope="new") == {"ok": True, "new": True}


def test_lookup_does_not_sweep_unrelated_cache_entries() -> None:
    cache = IdempotencyCache(max_entries=4)
    for index in range(4):
        cache.put(str(index), {"index": index}, scope="same")

    class NoGlobalSweep(OrderedDict):
        def items(self):
            raise AssertionError("lookup attempted a global cache sweep")

    cache._store = NoGlobalSweep(cache._store)
    assert cache.get("3", scope="same") == {"index": 3}


@pytest.mark.asyncio
async def test_dispatch_coalesces_concurrent_exact_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def slow_status() -> dict:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"ok": True, "calls": calls}

    monkeypatch.setitem(
        REGISTRY,
        "status",
        ToolSpec("status", "test status", {}, slow_status),
    )
    dispatcher = Dispatcher()
    caller = Caller(id="owner", tier="owner")

    first_task = asyncio.create_task(
        dispatcher.dispatch(
            tool="status",
            args={},
            caller=caller,
            request_id="concurrent-rid",
        )
    )
    await started.wait()
    second_task = asyncio.create_task(
        dispatcher.dispatch(
            tool="status",
            args={},
            caller=caller,
            request_id="concurrent-rid",
        )
    )
    await asyncio.sleep(0)
    release.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert calls == 1
    assert first.result == second.result == {"ok": True, "calls": 1}
    assert first.risk_tier == "T0"
    assert second.risk_tier == "cached"
