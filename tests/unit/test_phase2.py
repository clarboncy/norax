"""Phase 2 — retry + circuit breaker + budget + gateway client."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from norax.dispatch.budget import BudgetEnforcer, BudgetExceeded, Caps
from norax.gateway_client import GatewayClient, GatewayRequest
from norax.observability.circuit import CircuitBreaker, CircuitOpen, CircuitState
from norax.observability.retry import with_retry


@pytest.mark.asyncio
async def test_retry_eventually_succeeds():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("boom")
        return "ok"

    out = await with_retry(
        flaky, delays_ms=(5, 10, 20), jitter_ms=1, retriable=(httpx.ConnectError,)
    )
    assert out == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_retry_exhausts():
    async def always_fail():
        raise httpx.ConnectError("nope")

    with pytest.raises(httpx.ConnectError):
        await with_retry(
            always_fail, delays_ms=(1, 1, 1), jitter_ms=0, retriable=(httpx.ConnectError,)
        )


def test_circuit_trips_and_recovers():
    cb = CircuitBreaker("t", window_seconds=10, failure_threshold=3, open_seconds=0.05)
    assert cb.state == CircuitState.CLOSED
    for _ in range(3):
        cb.before_call()
        cb.on_failure()
    with pytest.raises(CircuitOpen):
        cb.before_call()
    assert cb.state == CircuitState.OPEN

    # Wait for half-open window
    import time as _t

    _t.sleep(0.06)
    assert cb.state == CircuitState.HALF_OPEN
    cb.before_call()
    cb.on_success()
    assert cb.state == CircuitState.CLOSED


def test_half_open_circuit_allows_only_one_concurrent_probe():
    cb = CircuitBreaker("single-probe", failure_threshold=1, open_seconds=0)
    cb.before_call()
    cb.on_failure()
    assert cb.state == CircuitState.HALF_OPEN

    cb.before_call()
    with pytest.raises(CircuitOpen, match="probe is already in flight"):
        cb.before_call()

    cb.on_abandoned()
    cb.before_call()
    cb.on_success()
    assert cb.state == CircuitState.CLOSED


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_seconds": 0},
        {"window_seconds": float("nan")},
        {"failure_threshold": 0},
        {"failure_threshold": True},
        {"open_seconds": -1},
        {"open_seconds": float("inf")},
    ],
)
def test_gateway_circuit_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CircuitBreaker("invalid", **kwargs)


def test_failures_already_in_flight_do_not_extend_open_deadline(monkeypatch):
    now = 100.0
    monkeypatch.setattr("norax.observability.circuit.time.monotonic", lambda: now)
    cb = CircuitBreaker("concurrent", failure_threshold=1, open_seconds=10)

    cb.before_call()
    cb.on_failure()
    now = 105.0
    cb.on_failure()
    now = 110.0

    assert cb.state == CircuitState.HALF_OPEN


def test_budget_caps_per_tier():
    # Use explicit caps since DEFAULT_CAPS are now unlimited for all tiers.
    caps = {
        "owner": Caps(float("inf"), 10**9, 10**6),
        "admin": Caps(float("inf"), 10**9, 10**6),
        "user": Caps(1.0, 500_000, 1_000),
        "guest": Caps(0.10, 50_000, 100),
    }
    be = BudgetEnforcer(caps=caps)
    today = date(2026, 4, 23)
    be.check("u1", "user", today=today)  # ok, empty bucket
    be.record("u1", usd=0.99, input_tokens=100, requests=1, today=today)
    be.check("u1", "user", today=today)  # still under
    be.record("u1", usd=0.02, today=today)  # tip over usd
    # user tier: soft degradation, never hard block
    be.check("u1", "user", today=today)  # should NOT raise
    zone = be.assess("u1", "user", today=today)
    assert zone.zone == "red"  # over cap → red zone (degraded, not killed)
    # guest tier at >100%: DOES hard block
    be.record("g1", usd=0.20, today=today)  # guest cap is $0.10/day
    with pytest.raises(BudgetExceeded):
        be.check("g1", "guest", today=today)


def test_budget_soft_degradation_zones():
    caps = {
        "owner": Caps(float("inf"), 10**9, 10**6),
        "user": Caps(1.0, 500_000, 1_000),
        "guest": Caps(0.10, 50_000, 100),
    }
    be = BudgetEnforcer(caps=caps)
    today = date(2026, 4, 23)
    # Owner: always green
    be.record("owner1", usd=9999, today=today)
    zone = be.assess("owner1", "owner", today=today)
    assert zone.zone == "green"
    # User at 50%: green
    be.record("u3", usd=0.50, today=today)
    zone = be.assess("u3", "user", today=today)
    assert zone.zone == "green"
    # User at 85%: yellow
    be.record("u3", usd=0.35, today=today)
    zone = be.assess("u3", "user", today=today)
    assert zone.zone == "yellow"
    assert zone.should_downgrade_model
    # User at 98%: red
    be.record("u3", usd=0.13, today=today)
    zone = be.assess("u3", "user", today=today)
    assert zone.zone == "red"
    assert zone.read_only
    assert zone.can_use_tools  # still can use tools (read-only)


def test_budget_resets_per_day():
    be = BudgetEnforcer()
    be.record("u2", usd=999, today=date(2026, 4, 22))
    be.check("u2", "user", today=date(2026, 4, 23))  # new day → fresh bucket


def test_default_budget_has_no_hidden_request_or_token_ceiling():
    be = BudgetEnforcer()
    be.record("guest", input_tokens=10**12, requests=10**9)

    zone = be.assess("guest", "guest")

    assert zone.zone == "green"
    assert zone.tool_output_cap is None


def test_budget_rejects_negative_or_nonfinite_accounting():
    be = BudgetEnforcer()
    with pytest.raises(ValueError):
        be.record("u", usd=-1)
    with pytest.raises(ValueError):
        be.record("u", usd=float("nan"))
    with pytest.raises(ValueError):
        be.record("u", usd=True)
    with pytest.raises(ValueError):
        be.record("u", usd="1")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        be.record("u", input_tokens=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        be.record("u", requests=True)


def test_budget_normalizes_integer_usd_to_float():
    be = BudgetEnforcer()
    be.record("u", usd=2)

    assert be.snapshot("u")["usd"] == 2.0


def test_budget_rejects_invalid_caps_at_configuration_time():
    with pytest.raises(ValueError):
        Caps(float("nan"), 0, 0)
    with pytest.raises(ValueError):
        Caps(-1.0, 0, 0)
    with pytest.raises(ValueError):
        Caps(1.0, -1, 0)
    with pytest.raises(ValueError):
        Caps(1.0, 0, True)
    with pytest.raises(ValueError):
        Caps("not-money", 0, 0)  # type: ignore[arg-type]


def test_budget_rejects_invalid_cap_maps_at_configuration_time():
    with pytest.raises(ValueError, match="guest fallback"):
        BudgetEnforcer(caps={"user": Caps(1.0, 10, 10)})
    with pytest.raises(ValueError, match="unknown budget tier"):
        BudgetEnforcer(
            caps={
                "guest": Caps(1.0, 10, 10),
                "stranger": Caps(1.0, 10, 10),  # type: ignore[dict-item]
            }
        )


def test_uncapped_budget_assessment_does_not_allocate_usage_bucket():
    be = BudgetEnforcer()
    zone = be.assess("uncapped", "guest")

    assert zone.zone == "green"
    assert "uncapped" not in be._usage


@pytest.mark.asyncio
@respx.mock
async def test_gateway_client_happy_path():
    route = respx.post("http://stub.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "req-xyz",
                "model": "stub/fake",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello world"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            },
        )
    )
    gc = GatewayClient(base_url="http://stub.test/v1")
    try:
        resp = await gc.chat(
            GatewayRequest(model="stub/fake", messages=[{"role": "user", "content": "hi"}])
        )
        assert resp.content == "hello world"
        assert resp.usage["input_tokens"] == 10
        assert resp.usage["output_tokens"] == 3
        assert route.called
    finally:
        await gc.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_gateway_client_forwards_reasoning_metadata():
    route = respx.post("http://stub.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "r",
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            },
        )
    )
    gc = GatewayClient(base_url="http://stub.test/v1")
    try:
        resp = await gc.chat(
            GatewayRequest(
                model="m",
                messages=[],
                metadata={"reasoning_effort": "xhigh", "reasoning_output": True},
            )
        )
        assert resp.content == "ok"
        payload = route.calls.last.request.content.decode()
        assert '"reasoning_effort":"xhigh"' in payload.replace(" ", "")
        assert '"reasoning":{"summary":"auto"}' in payload.replace(" ", "")
    finally:
        await gc.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_gateway_client_retries_transient():
    respx.post("http://stub.test/v1/chat/completions").mock(
        side_effect=[
            httpx.ConnectError("flaky"),
            httpx.Response(
                200,
                json={
                    "id": "r",
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            ),
        ]
    )
    gc = GatewayClient(base_url="http://stub.test/v1")
    try:
        resp = await gc.chat(GatewayRequest(model="m", messages=[]))
        assert resp.content == "ok"
    finally:
        await gc.aclose()
