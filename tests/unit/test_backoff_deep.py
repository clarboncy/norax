from __future__ import annotations

import asyncio

import pytest

from norax.runtime import backoff as module


@pytest.mark.parametrize("max_retries", [True, -1, 1.5])
def test_retry_rejects_invalid_retry_count(max_retries):
    async def operation():
        return "unused"

    with pytest.raises(ValueError, match="max_retries"):
        asyncio.run(module.retry_with_backoff(operation, max_retries=max_retries))


def test_backoff_handles_numeric_overflow_and_nonfinite_intermediate(monkeypatch):
    monkeypatch.setattr(module.random, "random", lambda: 0.5)
    overflow = module.ExponentialBackoff(base=1, factor=1e308, max_delay=7, jitter=0)
    overflow._attempt = 2
    assert overflow.next_delay() == 7

    nonfinite = module.ExponentialBackoff(base=1e308, factor=1e308, max_delay=9, jitter=0)
    nonfinite._attempt = 1
    assert nonfinite.next_delay() == 9


def test_retry_first_attempt_success_does_not_log_retry(caplog):
    async def operation():
        return "ok"

    assert asyncio.run(module.retry_with_backoff(operation, max_retries=0)) == "ok"
    assert "retry succeeded" not in caplog.text


def test_retry_exhaustion_raises_original_error():
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("still offline")

    with pytest.raises(RuntimeError, match="still offline"):
        asyncio.run(module.retry_with_backoff(operation, max_retries=0))
    assert attempts == 1


def test_retry_observer_receives_evidence_and_failure_is_contained(monkeypatch, caplog):
    caplog.set_level("DEBUG")
    attempts = 0
    sleeps = []

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient")
        return "recovered"

    def observer(attempt, error, delay):
        assert attempt == 0
        assert str(error) == "transient"
        assert delay == 1
        raise RuntimeError("observer bug")

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    result = asyncio.run(
        module.retry_with_backoff(
            operation,
            max_retries=1,
            jitter=0,
            on_retry=observer,
        )
    )
    assert result == "recovered" and sleeps == [1]
    assert "retry observer failed" in caplog.text
