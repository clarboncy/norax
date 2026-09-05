"""Completion-probe failover regressions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from norax.gateway_client import GatewayRequest, GatewayResponse
from norax.runtime.core import Runtime


class _Gateway:
    def __init__(self) -> None:
        self.models: list[str] = []
        self.requests: list[GatewayRequest] = []

    async def chat(self, req: GatewayRequest):
        self.models.append(req.model)
        self.requests.append(req)
        if req.model == "primary":
            raise RuntimeError("primary exhausted")
        return GatewayResponse(
            request_id="probe",
            model=req.model,
            content="OK",
            tool_calls=[],
            usage={},
            raw={},
        )

    def route_for(self, model: str) -> tuple[str, str]:
        return f"provider-{model}", f"http://{model}/v1"


@pytest.mark.asyncio
async def test_completion_probe_uses_failover_when_primary_is_unavailable():
    gateway = _Gateway()
    runtime = SimpleNamespace(
        gateway=gateway,
        _effective_model="primary",
        failover_models=["fallback"],
        _completion_probe_mode="completion",
    )

    result = await Runtime._run_completion_probe(runtime)  # type: ignore[arg-type]

    assert gateway.models == ["primary", "fallback"]
    assert result["ok"] is True
    assert result["model"] == "fallback"
    assert result["provider"] == "provider-fallback"
    assert result["fallback"] is True
    assert result["mode"] == "completion"
    assert result["completion_verified"] is True
    assert gateway.requests[0].max_tokens == 32
    assert gateway.requests[0].metadata["reasoning_effort"] == "none"


class _TransportGateway:
    def __init__(self) -> None:
        self.probes: list[str] = []
        self.chat_called = False

    def route_for(self, model: str) -> tuple[str, str]:
        return "local", "http://127.0.0.1:11434/v1"

    async def transport_probe(self, model: str, *, timeout: float):
        self.probes.append(model)
        assert timeout == 5.0
        return {
            "ok": True,
            "provider": "local",
            "kind": "ollama",
            "endpoint": "http://127.0.0.1:11434/api/tags",
            "status_code": 200,
        }

    async def chat(self, req):
        self.chat_called = True
        raise AssertionError("transport mode must not generate tokens")


@pytest.mark.asyncio
async def test_default_completion_probe_is_honest_zero_token_transport_check():
    gateway = _TransportGateway()
    runtime = SimpleNamespace(
        gateway=gateway,
        _effective_model="ollama/model.gguf",
        failover_models=[],
    )

    result = await Runtime._run_completion_probe(runtime)  # type: ignore[arg-type]

    assert gateway.probes == ["ollama/model.gguf"]
    assert gateway.chat_called is False
    assert result["ok"] is True
    assert result["mode"] == "transport"
    assert result["transport_verified"] is True
    assert result["completion_verified"] is False
    assert result["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_completion_probe_rejects_non_exact_model_output():
    class LooseGateway(_Gateway):
        async def chat(self, req: GatewayRequest):
            self.models.append(req.model)
            self.requests.append(req)
            return GatewayResponse(
                request_id="probe",
                model=req.model,
                content="Certainly — OK",
                tool_calls=[],
                usage={},
                raw={},
            )

    gateway = LooseGateway()
    runtime = SimpleNamespace(
        gateway=gateway,
        _effective_model="primary",
        failover_models=[],
        _completion_probe_mode="completion",
    )

    result = await Runtime._run_completion_probe(runtime)  # type: ignore[arg-type]

    assert result["ok"] is False
    assert result["completion_verified"] is False
    assert len(result["failures"]) == 1
    assert "unexpected response chars=" in result["failures"][0]
    assert "Certainly" not in result["failures"][0]


def test_completion_probe_defers_during_and_immediately_after_user_work():
    now = 10_000.0
    runtime = SimpleNamespace(
        _completion_probe_mode="completion",
        _completion_probe_idle_seconds=30,
        _turn_work_count=1,
        _active_turn_tasks={},
        _last_user_turn_time=now - 100,
    )
    assert (
        Runtime._completion_probe_defer_reason(runtime, now=now)  # type: ignore[arg-type]
        == "queued_or_active_turn"
    )

    runtime._turn_work_count = 0
    runtime._active_turn_tasks = {"channel": Mock(done=lambda: False)}
    assert (
        Runtime._completion_probe_defer_reason(runtime, now=now)  # type: ignore[arg-type]
        == "active_turn"
    )

    runtime._active_turn_tasks = {}
    runtime._last_user_turn_time = now - 5
    assert (
        Runtime._completion_probe_defer_reason(runtime, now=now)  # type: ignore[arg-type]
        == "recent_turn"
    )
    runtime._last_user_turn_time = now - 31
    assert Runtime._completion_probe_defer_reason(runtime, now=now) is None  # type: ignore[arg-type]


def test_completion_probe_age_requires_well_formed_timestamp():
    now = datetime.now(UTC)
    assert Runtime._completion_probe_age(None, now=now) == float("inf")
    assert Runtime._completion_probe_age({}, now=now) == float("inf")
    assert Runtime._completion_probe_age(
        {"timestamp": (now - timedelta(seconds=42)).isoformat()}, now=now
    ) == pytest.approx(42.0)
