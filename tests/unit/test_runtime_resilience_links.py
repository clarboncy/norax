"""Executable links for retry, protocol signing, and redaction."""

from __future__ import annotations

import asyncio

import pytest

from norax.observability.retry import with_retry
from norax.remote.protocol import new_token, sign, token_hash, verify
from norax.runtime.backoff import ExponentialBackoff, retry_with_backoff
from norax.safety.secrets import redact


def test_backoff_caps_resets_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("norax.runtime.backoff.random.random", lambda: 0.5)
    backoff = ExponentialBackoff(base=2, factor=3, max_delay=5, jitter=0)
    assert [backoff.next_delay() for _ in range(3)] == [2, 5, 5]
    assert backoff.attempt == 3
    backoff.reset()
    assert backoff.attempt == 0

    sleeps: list[float] = []
    monkeypatch.setattr("norax.runtime.backoff.asyncio.sleep", _record_sleep(sleeps))
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("transient")
        return "ok"

    assert asyncio.run(retry_with_backoff(operation, max_retries=2, jitter=0)) == "ok"
    assert attempts == 3
    assert sleeps == [1.0, 2.0]


def test_backoff_jitter_never_exceeds_cap_or_invents_zero_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("norax.runtime.backoff.random.random", lambda: 1.0)

    assert ExponentialBackoff(base=10, max_delay=5, jitter=1).next_delay() == 5
    assert ExponentialBackoff(base=0, max_delay=0, jitter=0).next_delay() == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base": float("nan")},
        {"factor": 0},
        {"max_delay": -1},
        {"jitter": 1.1},
    ],
)
def test_backoff_rejects_nonfinite_or_invalid_configuration(kwargs) -> None:
    with pytest.raises(ValueError):
        ExponentialBackoff(**kwargs)


def _record_sleep(sleeps: list[float]):
    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    return sleep


def test_retry_does_not_swallow_non_retryable_error() -> None:
    async def operation() -> str:
        raise ValueError("terminal")

    with pytest.raises(ValueError, match="terminal"):
        asyncio.run(retry_with_backoff(operation, retry_on=RuntimeError))


def test_observability_retry_rejects_invalid_delay_without_sleeping() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        return "unused"

    with pytest.raises(ValueError, match="finite non-negative"):
        asyncio.run(with_retry(operation, delays_ms=[float("nan")]))
    assert calls == 0


def test_tool_circuit_recovery_resets_backoff_for_a_new_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from norax.runtime.circuit_breaker import Circuit, CircuitConfig, CircuitState

    monkeypatch.setattr("norax.runtime.circuit_breaker.random.uniform", lambda *_args: 0.0)
    now = 100.0
    monkeypatch.setattr("norax.runtime.circuit_breaker.time.monotonic", lambda: now)
    circuit = Circuit(
        "tool",
        CircuitConfig(
            failure_threshold=1,
            success_threshold=1,
            cooldown_sec=10,
            max_cooldown_sec=60,
        ),
    )

    circuit.record_failure("outage one")
    first_cooldown = circuit.stats()["cooldown_sec"]
    now += first_cooldown
    allowed, _ = circuit.can_execute()
    assert allowed
    circuit.record_success()
    assert circuit.state is CircuitState.CLOSED

    circuit.record_failure("outage two")
    assert circuit.stats()["cooldown_sec"] == first_cooldown


@pytest.mark.parametrize(
    "kwargs",
    [
        {"failure_threshold": 0},
        {"success_threshold": True},
        {"half_open_max_calls": 1.5},
        {"cooldown_sec": float("nan")},
        {"max_cooldown_sec": float("inf")},
        {"cooldown_sec": 2, "max_cooldown_sec": 1},
    ],
)
def test_tool_circuit_rejects_invalid_configuration(kwargs) -> None:
    from norax.runtime.circuit_breaker import CircuitConfig

    with pytest.raises(ValueError):
        CircuitConfig(**kwargs)


def test_abandoned_half_open_tool_probe_releases_slot() -> None:
    from norax.runtime.circuit_breaker import Circuit, CircuitConfig

    circuit = Circuit(
        "tool",
        CircuitConfig(failure_threshold=1, cooldown_sec=0, half_open_max_calls=1),
    )
    circuit.record_failure("outage")
    assert circuit.can_execute()[0] is True
    assert circuit.can_execute()[0] is False

    circuit.abandon_probe()

    assert circuit.can_execute()[0] is True


def test_remote_signatures_are_canonical_and_tamper_evident() -> None:
    token = new_token()
    assert token.startswith("nxn_")
    assert len(token_hash(token)) == 64
    payload = {"node": "n1", "args": {"b": 2, "a": 1}}
    signature = sign(token, payload)
    assert verify(token, {"args": {"a": 1, "b": 2}, "node": "n1"}, signature)
    assert not verify(token, {"node": "n2", "args": {"a": 1, "b": 2}}, signature)
    assert not verify(token + "x", payload, signature)


def test_secret_redaction_preserves_shape_and_nonsecrets() -> None:
    original = {
        "key": "sk-" + "a" * 30,
        "nested": ["safe", ("AKIA" + "A" * 16, 7)],
    }
    cleaned = redact(original)
    assert cleaned == {
        "key": "<REDACTED:openai_key>",
        "nested": ["safe", ("<REDACTED:aws_access_key>", 7)],
    }
    assert original["key"].startswith("sk-")


@pytest.mark.asyncio
async def test_browser_shutdown_closes_every_global_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from norax.dispatch import browser

    closed = []

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close(self) -> None:
            closed.append(self.name)

    class PlaywrightResource:
        async def stop(self) -> None:
            closed.append("playwright")

    monkeypatch.setattr(browser, "_pages", {"one": Resource("page")})
    monkeypatch.setattr(browser, "_page_last_used", {"one": 1.0})
    monkeypatch.setattr(browser, "_context", Resource("context"))
    monkeypatch.setattr(browser, "_browser", Resource("browser"))
    monkeypatch.setattr(browser, "_playwright", PlaywrightResource())

    await browser.shutdown_browser()

    assert closed == ["page", "context", "browser", "playwright"]
    assert browser._pages == {}
    assert browser._page_last_used == {}
    assert browser._context is None
    assert browser._browser is None
    assert browser._playwright is None


def test_checkpoint_status_distinguishes_interrupted_from_complete(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from norax.runtime.checkpoint import has_pending_checkpoint, save_checkpoint

    monkeypatch.setenv("NORAX_CHECKPOINT_DIR", str(tmp_path))
    common = {
        "channel": "chat",
        "turn_id": "7",
        "messages": [{"role": "user", "content": "repair server"}],
        "trace": [{"name": "read", "args": {}, "result": {"ok": True}}],
        "task_state": {"goal": "repair server"},
        "rounds": 1,
        "model": "test",
    }
    save_checkpoint(**common, status="in_progress")
    assert has_pending_checkpoint("chat")
    save_checkpoint(**common, status="complete")
    assert not has_pending_checkpoint("chat")
