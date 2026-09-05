"""Ollama: per-model profiles and wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from norax.brain.strong_model_scaffold import (
    build_ollama_exec_guard,
    build_scaffold_prompt,
    is_ollama_agentic_model,
)
from norax.gateway_client import GatewayRequest, GatewayResponse
from norax.gateway_client.ollama_profiles import (
    apply_profile_to_gateway_request,
    build_exec_guard_nudge,
    effective_temperature,
    is_weak_ollama_model,
    resolve_profile,
    tools_for_profile,
    trim_tools,
)
from norax.gateway_client.ollama_wrapper import OllamaGatewayClient
from norax.observability.metrics import Metrics


def test_kimi_k3_profile_is_agentic_and_thinking():
    p = resolve_profile("kimi-k3:cloud")
    assert p.think is True
    assert p.temperature == 1.0
    assert p.role == "executor"
    assert p.tools_enabled is True
    assert p.fallback is None
    assert p.reasoning_efforts == ("low", "medium", "high", "max")


def test_kimi_profile_thinking_temp():
    p = resolve_profile("kimi-k2.7-code:cloud")
    assert p.think is True
    assert p.temperature == 1.0
    assert p.role == "executor"
    assert p.tools_enabled is True
    assert resolve_profile("kimi-anything:cloud").model == "kimi-k2.7-code:cloud"


def test_glm_profile_thinking_default():
    p = resolve_profile("glm-5.1:cloud")
    assert p.temperature == 0.3
    assert p.think is True
    assert p.tools_enabled is True


def test_glm_53_profile_is_coding_executor():
    p = resolve_profile("glm-5.3:cloud")
    assert p.model == "glm-5.3:cloud"
    assert p.temperature == 0.3
    assert p.think is True
    assert p.role == "executor"
    assert p.tools_enabled is True
    assert p.reasoning_efforts == ("low", "medium", "high", "max")
    assert p.clear_thinking is True
    assert resolve_profile("glm-5.3-preview:cloud").model == "glm-5.3:cloud"


def test_build_ollama_chat_body_respects_explicit_think_override():
    from norax.gateway_client.ollama_profiles import build_ollama_chat_body

    body = build_ollama_chat_body({"messages": [], "think": True}, "glm-5.1:cloud")
    assert body["think"] is True

    body = build_ollama_chat_body({"messages": [], "reasoning_effort": "high"}, "glm-5.1:cloud")
    assert body["think"] is True

    body = build_ollama_chat_body({"messages": [], "reasoning_effort": "xhigh"}, "glm-5.3:cloud")
    assert body["reasoning_effort"] == "max"
    assert body["clear_thinking"] is True


def test_qwen_executor_profile():
    p = resolve_profile("qwen3-coder-next:cloud")
    assert p.role == "executor"
    assert p.temperature == 0.2
    assert p.think is False


def test_unknown_local_model_uses_conservative_context_default():
    p = resolve_profile("operator-model:latest")
    assert p.num_ctx == 32768
    assert p.num_predict == 4096


def test_deepseek_planner_profile():
    p = resolve_profile("deepseek-v4-pro:cloud")
    assert p.model == "deepseek-v4-pro:cloud"
    assert p.think is True
    assert p.temperature == 1.0


def test_tool_capable_profiles_keep_complete_manifest():
    tools = [{"type": "function", "function": {"name": f"t{i}"}} for i in range(10)]
    trimmed = tools_for_profile(tools, enabled=True)
    assert trimmed is not None
    assert trimmed == tools


def test_trim_tools_false_strips_for_text_only_profile():
    tools = [{"type": "function", "function": {"name": "read"}}]
    assert tools_for_profile(tools, enabled=False) is None


def test_legacy_numeric_tool_setting_is_enable_disable_not_a_limit():
    tools = [{"type": "function", "function": {"name": f"t{i}"}} for i in range(10)]
    assert trim_tools(tools, 0) is None
    assert trim_tools(tools, 1) == tools


def test_cloud_models_keep_complete_tool_manifest():
    tools = [{"type": "function", "function": {"name": f"t{i}"}} for i in range(25)]

    assert trim_tools(tools, 5, model="kimi-k2.6:cloud") == tools


def test_apply_profile_sets_temperature():
    req = GatewayRequest(
        model="kimi-k2.6:cloud",
        messages=[{"role": "user", "content": "fix bug"}],
        tools=[{"type": "function", "function": {"name": "read"}}],
    )
    out = apply_profile_to_gateway_request(req)
    assert out.temperature == 1.0
    assert out.max_tokens == 8192
    assert out.metadata.get("ollama_profile") == "agentic"


def test_exec_guard_on_math():
    assert build_exec_guard_nudge("What is 17*23+41*19?") is not None
    assert build_exec_guard_nudge("hello") is None


def test_ollama_exec_guard_in_scaffold():
    guard = build_ollama_exec_guard("calculate 17*23")
    assert guard is not None
    scaffold = build_scaffold_prompt(
        "coding", model="kimi-k2.6:cloud", user_prompt="calculate 17*23"
    )
    assert "OLLAMA_EXEC_GUARD" in scaffold


def test_is_ollama_agentic_model():
    assert is_ollama_agentic_model("kimi-k2.6:cloud")
    assert is_ollama_agentic_model("gemma4:31b-cloud") is False


def test_weak_model_lower_tool_temp():
    p = resolve_profile("deepseek-r1:14b")
    assert p.tools_enabled is True
    assert effective_temperature(p, has_tools=True) == 0.05
    assert effective_temperature(p, has_tools=False) == 0.15


def test_is_weak_ollama_model():
    assert is_weak_ollama_model("deepseek-r1:14b")
    assert not is_weak_ollama_model("kimi-k2.7-code:cloud")


@pytest.mark.asyncio
async def test_ollama_gateway_client_wraps_and_records():
    inner = MagicMock()
    inner.base_url = "http://127.0.0.1:11434/v1"
    inner.stream_required = False
    inner.chat_path = "/chat/completions"
    inner.chat = AsyncMock(
        return_value=GatewayResponse(
            request_id="r1",
            model="kimi-k2.7-code:cloud",
            content="ok",
            tool_calls=[],
            usage={"input_tokens": 1, "output_tokens": 2},
            raw={},
        )
    )
    inner.aclose = AsyncMock()

    metrics = Metrics()
    client = OllamaGatewayClient(inner, metrics=metrics)
    req = GatewayRequest(
        model="kimi-k2.7-code:cloud",
        messages=[{"role": "user", "content": "ping"}],
    )
    resp = await client.chat(req)
    assert resp.raw.get("ollama_enhanced") is True
    assert resp.raw.get("profile_role") == "executor"
    inner.chat.assert_awaited_once()
    called_req = inner.chat.await_args[0][0]
    assert called_req.temperature == 1.0


@pytest.mark.asyncio
async def test_ollama_gateway_injects_math_guard():
    inner = MagicMock()
    inner.base_url = "http://127.0.0.1:11434/v1"
    inner.stream_required = False
    inner.chat_path = "/chat/completions"
    inner.chat = AsyncMock(
        return_value=GatewayResponse(
            request_id="r1",
            model="qwen3-coder-next:cloud",
            content="682",
            tool_calls=[],
            usage={"input_tokens": 1, "output_tokens": 1},
            raw={},
        )
    )
    inner.aclose = AsyncMock()

    client = OllamaGatewayClient(inner)
    req = GatewayRequest(
        model="qwen3-coder-next:cloud",
        messages=[{"role": "user", "content": "What is 17*23+41*19?"}],
    )
    await client.chat(req)
    called_req = inner.chat.await_args[0][0]
    assert any("OLLAMA_EXEC_GUARD" in str(m.get("content", "")) for m in called_req.messages)
