from __future__ import annotations

import asyncio

import pytest

from norax.runtime import watchdog as module


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["", "   ", 4, "x" * 161])
async def test_watchdog_rejects_invalid_names_without_registering(name):
    watchdog = module.Watchdog()
    coroutine = asyncio.sleep(0)
    try:
        with pytest.raises(ValueError, match="name"):
            await watchdog.run(name, coroutine)  # type: ignore[arg-type]
    finally:
        coroutine.close()
    assert watchdog.stats()["active"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [True, "1", 0, -1, float("nan"), float("inf")])
async def test_watchdog_rejects_invalid_timeouts_without_registering(timeout):
    watchdog = module.Watchdog()
    coroutine = asyncio.sleep(0)
    try:
        with pytest.raises(ValueError, match="timeout"):
            await watchdog.run("task", coroutine, timeout=timeout)  # type: ignore[arg-type]
    finally:
        coroutine.close()
    assert watchdog.stats()["active"] == 0


@pytest.mark.asyncio
async def test_watchdog_records_completion_and_cleans_active_entry():
    watchdog = module.Watchdog()
    result = await watchdog.run("  memory sync  ", asyncio.sleep(0, result="done"), timeout=1)
    assert result == "done"
    assert watchdog.stats() == {"active": 0, "total_completions": 1, "total_timeouts": 0}


@pytest.mark.asyncio
async def test_watchdog_timeout_cancels_work_calls_callback_and_preserves_timeout():
    watchdog = module.Watchdog()
    cancelled = asyncio.Event()
    callback = []

    async def hung():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with pytest.raises(TimeoutError):
        await watchdog.run(
            "hung",
            hung(),
            timeout=0.001,
            on_timeout=lambda name, elapsed: callback.append((name, elapsed)),
        )
    assert cancelled.is_set()
    assert callback[0][0] == "hung" and callback[0][1] >= 0
    assert watchdog.stats() == {"active": 0, "total_completions": 0, "total_timeouts": 1}


@pytest.mark.asyncio
async def test_failing_timeout_callback_never_masks_timeout(caplog):
    watchdog = module.Watchdog()

    def fail(_name, _elapsed):
        raise RuntimeError("callback failed")

    with pytest.raises(TimeoutError):
        await watchdog.run("hung", asyncio.sleep(1), timeout=0.001, on_timeout=fail)
    assert "timeout callback failed" in caplog.text


@pytest.mark.asyncio
async def test_timeout_callback_is_optional():
    watchdog = module.Watchdog()
    with pytest.raises(TimeoutError):
        await watchdog.run("hung", asyncio.sleep(1), timeout=0.001)


@pytest.mark.asyncio
async def test_same_named_tasks_have_independent_active_entries():
    watchdog = module.Watchdog()
    release_first = asyncio.Event()
    release_second = asyncio.Event()
    first = asyncio.create_task(watchdog.run("duplicate", release_first.wait(), timeout=10))
    second = asyncio.create_task(watchdog.run("duplicate", release_second.wait(), timeout=10))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    now = module.time.monotonic()
    entries = list(watchdog._entries.values())
    entries[0].started_at = now - 2
    entries[1].started_at = now - 1
    active = watchdog.active_tasks()
    assert len(active) == 2
    assert active == [
        {"name": "duplicate", "elapsed_sec": 2.0, "timeout_sec": 10.0, "remaining_sec": 8.0},
        {"name": "duplicate", "elapsed_sec": 1.0, "timeout_sec": 10.0, "remaining_sec": 9.0},
    ]
    release_first.set()
    await first
    assert watchdog.stats()["active"] == 1
    release_second.set()
    await second
    assert watchdog.stats()["active"] == 0


@pytest.mark.asyncio
async def test_caller_cancellation_cleans_entry_without_counting_completion():
    watchdog = module.Watchdog()
    task = asyncio.create_task(watchdog.run("cancel", asyncio.Event().wait(), timeout=10))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert watchdog.stats() == {"active": 0, "total_completions": 0, "total_timeouts": 0}


def test_global_watchdog_is_lazy_and_reused(monkeypatch):
    monkeypatch.setattr(module, "_watchdog", None)
    first = module.get_watchdog()
    assert module.get_watchdog() is first
