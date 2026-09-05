from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from norax.commands import (
    BUILTIN_MODELS,
    model_options_for_discord,
    provider_for_model,
)
from norax.config.loader import load_jsonc
from norax.dispatch.tools import render_tools_for_llm
from norax.gateway_client import (
    GatewayClient,
    GatewayRouter,
    GatewayUpstreamError,
    _parse_openai,
)


def test_ollama_tools_are_trimmed_and_auto_choice_omitted():
    gc = GatewayClient(base_url="http://127.0.0.1:11434/v1")
    tools = render_tools_for_llm(["read"])
    tools[0]["function"]["parameters"]["properties"]["path"]["default"] = "x"
    payload = {
        "model": "llama3.3:latest",
        "messages": [],
        "tools": tools,
        "tool_choice": "auto",
        "metadata": {"x": 1},
    }

    out = gc._sanitize_payload(payload)

    assert "tool_choice" not in out
    assert "metadata" not in out
    assert out["tools"][0]["type"] == "function"
    assert "default" not in out["tools"][0]["function"]["parameters"]["properties"]["path"]


def test_ollama_cloud_models_keep_required_tool_arguments():
    gc = GatewayClient(base_url="http://127.0.0.1:11434/v1")
    tools = render_tools_for_llm(["read"])

    out = gc._format_tools_for_provider(tools, model="kimi-k2.6:cloud")

    assert "path" in out[0]["function"]["parameters"]["required"]


def test_ollama_small_local_models_use_permissive_tool_arguments():
    gc = GatewayClient(base_url="http://127.0.0.1:11434/v1")
    tools = render_tools_for_llm(["read"])

    out = gc._format_tools_for_provider(tools, model="gemma4:12b")

    assert "required" not in out[0]["function"]["parameters"]


def test_current_gpt_model_is_in_selector_and_discord_under_limit():
    mids = [m for m, _ in BUILTIN_MODELS]
    assert "openrouter/openai/gpt-4o-mini" in mids
    opts = model_options_for_discord("openrouter")
    assert len(opts) <= 25
    assert "openrouter/openai/gpt-4o-mini" in [m for m, _ in opts]


def test_ollama_returned_tool_calls_are_dispatcher_safe():
    gc = GatewayClient(base_url="http://127.0.0.1:11434/v1")
    out = gc._normalize_tool_calls_from_provider(
        [{"function": {"name": "read", "arguments": {"path": "README.md"}}}]
    )
    assert out == [
        {
            "function": {"name": "read", "arguments": '{"path": "README.md"}'},
            "type": "function",
            "id": "call_0",
        }
    ]


def test_ollama_tool_names_are_safe_and_short():
    gc = GatewayClient(base_url="http://127.0.0.1:11434/v1")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "remote.exec.with-bad name" * 4,
                "description": "d" * 900,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    out = gc._format_tools_for_provider(tools)
    fn = out[0]["function"]
    assert "-" not in fn["name"] and "." not in fn["name"] and " " not in fn["name"]
    assert len(fn["name"]) <= 64
    assert len(fn["description"]) <= 512


def test_provider_kind_can_be_explicit_even_with_same_url():
    gc = GatewayClient(base_url="http://127.0.0.1:9999/v1", provider_kind="ollama")
    payload = {
        "model": "x",
        "messages": [],
        "metadata": {"a": 1},
        "user": "u",
        "reasoning_effort": "high",
        "tools": render_tools_for_llm(["read"]),
        "tool_choice": "auto",
    }
    out = gc._sanitize_payload(payload)
    assert gc._provider_kind() == "ollama"
    assert "metadata" not in out
    assert "user" not in out
    # Ollama native keeps reasoning_effort long enough to map it to `think`.
    assert out["reasoning_effort"] == "high"
    assert "tool_choice" not in out
    assert out["tools"][0]["function"]["parameters"]["type"] == "object"


def test_chat_path_is_configurable():
    gc = GatewayClient(base_url="http://host/v1", chat_path="responses", provider_kind="openai")
    assert gc.chat_path == "/responses"


def test_anthropic_payload_uses_messages_shape_and_tools():
    gc = GatewayClient(base_url="http://host/v1", chat_path="messages", provider_kind="anthropic")
    payload = {
        "model": "claude-opus-4-6",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hi"},
        ],
        "tools": render_tools_for_llm(["status"]),
    }
    out = gc._to_anthropic_payload(payload)
    assert isinstance(out["system"], list)
    assert out["system"][0]["text"] == "s"
    assert out["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert out["tools"][0]["name"] == "status"
    assert "input_schema" in out["tools"][0]


def test_anthropic_payload_carries_session_id_from_metadata():
    gc = GatewayClient(base_url="http://host/v1", chat_path="messages", provider_kind="anthropic")
    out = gc._to_anthropic_payload(
        {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"norax_request_id": "req_123"},
        }
    )
    assert str(UUID(out["session_id"])) == out["session_id"]
    assert out["metadata"]["norax_request_id"] == "req_123"


def test_anthropic_client_sends_device_id_header():
    gc = GatewayClient(
        base_url="http://host/v1",
        chat_path="messages",
        provider_kind="anthropic",
        api_key="k",
    )
    assert str(UUID(gc._client.headers["x-device-id"])) == gc._client.headers["x-device-id"]
    assert gc._client.headers["authorization"] == "Bearer k"
    # x-api-key may be present depending on SDK version; we only assert authorization header


def test_ollama_models_are_in_selector():
    opts = model_options_for_discord("ollama")
    assert any(model == "kimi-k2.7-code:cloud" for model, _label in opts)
    assert any(model == "glm-5.3:cloud" for model, _label in opts)


def test_openrouter_models_are_in_selector():
    opts = model_options_for_discord("openrouter")
    assert any("openrouter/openai/gpt-4o-mini" in m for m, _ in opts)
    assert provider_for_model("openrouter/openai/gpt-4o-mini") == "openrouter"


def test_ollama_model_routing_matches_runtime_routes():
    assert provider_for_model("kimi-k2.7-code:cloud") == "ollama"
    assert provider_for_model("glm-5.3:cloud") == "ollama"
    assert provider_for_model("openrouter/openai/gpt-4o-mini") == "openrouter"


def test_runtime_router_sends_bare_ollama_tags_to_ollama() -> None:
    class Provider:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

    config_path = Path(__file__).parents[2] / "config" / "runtime.jsonc"
    gateway_config = load_jsonc(config_path)["gateway"]
    configured_routes = [tuple(route) for route in gateway_config["routes"]]
    providers: dict[str, Any] = {
        "ollama": Provider("http://127.0.0.1:11434/v1"),
        "ollama_local": Provider("http://127.0.0.1:11435/v1"),
        "codex_direct": Provider("http://127.0.0.1:4146/v1"),
        "openrouter": Provider("https://openrouter.ai/api/v1"),
        "freetoken": Provider("http://127.0.0.1:1919/v1"),
    }
    router = GatewayRouter(
        providers=providers,
        routes=[("freetoken/*", "freetoken"), *configured_routes],
        default_provider="freetoken",
    )

    assert router.route_for("glm-5.3:cloud")[0] == "ollama"
    assert router.route_for("deepseek-v4-pro:cloud")[0] == "ollama"
    assert router.route_for("ollama/custom-model")[0] == "ollama"
    assert router.route_for("openrouter/openai/gpt-4o-mini")[0] == "openrouter"
    assert router.route_for("freetoken/Ornith-1.5-35B-IQ3_S.gguf")[0] == "freetoken"


def test_ollama_default_models_route_correctly():
    opts = model_options_for_discord("ollama")
    assert any(model == "kimi-k2.7-code:cloud" for model, _label in opts)
    assert provider_for_model("kimi-k2.7-code:cloud") == "ollama"


def test_ollama_models_keep_native_tools_enabled():
    gc = GatewayClient(base_url="http://127.0.0.1:11434/v1", provider_kind="ollama")
    req = __import__("norax.gateway_client", fromlist=["GatewayRequest"]).GatewayRequest(
        model="glm-5.1:cloud",
        messages=[{"role": "user", "content": "Respond ok"}],
        tools=render_tools_for_llm(["read"]),
    )
    assert gc._tools_supported_for_request(req) is True


def test_ollama_tool_capable_models_still_receive_tools():
    gc = GatewayClient(base_url="http://127.0.0.1:11434/v1", provider_kind="ollama")
    req = __import__("norax.gateway_client", fromlist=["GatewayRequest"]).GatewayRequest(
        model="gemma4:31b-cloud",
        messages=[{"role": "user", "content": "Respond ok"}],
        tools=render_tools_for_llm(["read"]),
    )
    assert gc._tools_supported_for_request(req) is True


def test_ollama_native_payload_uses_api_chat_shapes():
    gc = GatewayClient(base_url="http://127.0.0.1:11434/v1", provider_kind="ollama")
    payload = {
        "model": "glm-5.1:cloud",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "x",
                        "type": "function",
                        "function": {"name": "status", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "x", "name": "status", "content": "{}"},
        ],
        "stream": False,
        "tools": render_tools_for_llm(["status"]),
        "max_tokens": 7,
    }
    native = gc._to_ollama_native_payload(payload)
    assert native["options"]["num_predict"] == 7
    assert native["messages"][0]["tool_calls"][0]["function"]["arguments"] == {}
    assert native["messages"][1]["tool_name"] == "status"
    assert "tool_call_id" not in native["messages"][1]
    assert native["tools"][0]["type"] == "function"
    assert native["think"] is True


def test_ollama_native_payload_respects_reasoning_off():
    gc = GatewayClient(base_url="http://127.0.0.1:11434/v1", provider_kind="ollama")
    native = gc._to_ollama_native_payload(
        {
            "model": "deepseek-v4-pro:cloud",
            "messages": [{"role": "user", "content": "x"}],
            "reasoning_effort": "off",
        }
    )
    assert native["think"] is False


def test_qwen_38_native_payload_uses_selected_reasoning_effort():
    gc = GatewayClient(base_url="http://127.0.0.1:11434/v1", provider_kind="ollama")

    medium = gc._to_ollama_native_payload(
        {
            "model": "qwen3.8-27b-fast:latest",
            "messages": [{"role": "user", "content": "fix it"}],
            "reasoning_effort": "medium",
        }
    )
    high = gc._to_ollama_native_payload(
        {
            "model": "qwen3.8-27b-fast:latest",
            "messages": [{"role": "user", "content": "fix it"}],
            "reasoning_effort": "high",
        }
    )

    assert medium["think"] == "medium"
    assert medium["reasoning_effort"] == "medium"
    assert high["think"] == "high"
    assert high["reasoning_effort"] == "high"
    assert medium["options"]["num_ctx"] == 65_536
    assert medium["keep_alive"] == -1
    assert "keep_alive" not in medium["options"]


def test_glm_53_native_payload_preserves_supported_reasoning_controls():
    gc = GatewayClient(base_url="http://127.0.0.1:11434/v1", provider_kind="ollama")
    native = gc._to_ollama_native_payload(
        {
            "model": "glm-5.3:cloud",
            "messages": [{"role": "user", "content": "fix it"}],
            "reasoning_effort": "medium",
        }
    )

    assert native["think"] == "medium"
    assert native["reasoning_effort"] == "medium"
    assert native["clear_thinking"] is True


def test_ollama_native_parse_hides_thinking_by_default():
    gc = GatewayClient(base_url="http://127.0.0.1:11434/v1", provider_kind="ollama")
    resp = gc._parse_ollama_native(
        {
            "model": "deepseek-v4-pro:cloud",
            "message": {"content": "answer", "thinking": "working it out"},
            "prompt_eval_count": 3,
            "eval_count": 4,
        },
        request_id="r1",
        fallback_model="deepseek-v4-pro:cloud",
    )
    assert resp.content == "answer"


def test_ollama_native_parse_exposes_thinking_only_when_requested():
    gc = GatewayClient(base_url="http://127.0.0.1:11434/v1", provider_kind="ollama")
    resp = gc._parse_ollama_native(
        {
            "model": "deepseek-v4-pro:cloud",
            "message": {"content": "answer", "thinking": "working it out"},
        },
        request_id="r1",
        fallback_model="deepseek-v4-pro:cloud",
        expose_reasoning=True,
    )

    assert "<thinking>" in resp.content
    assert "working it out" in resp.content
    assert resp.content.endswith("answer")


def test_openai_parse_removes_inline_reasoning_block():
    resp = _parse_openai(
        {
            "model": "m",
            "choices": [
                {
                    "message": {
                        "content": "<think>private chain</think>Public answer",
                    }
                }
            ],
        },
        request_id="r1",
        fallback_model="m",
    )

    assert resp.content == "Public answer"


def test_openai_reasoning_only_is_not_promoted_to_answer():
    with pytest.raises(GatewayUpstreamError, match="no content"):
        _parse_openai(
            {
                "model": "m",
                "choices": [{"message": {"content": "", "reasoning": "private chain"}}],
            },
            request_id="r1",
            fallback_model="m",
        )


def test_ollama_native_payload_defaults_heavy_num_predict():
    gc = GatewayClient(base_url="http://127.0.0.1:11434/v1", provider_kind="ollama")
    native = gc._to_ollama_native_payload(
        {
            "model": "kimi-k2.6:cloud",
            "messages": [{"role": "user", "content": "heavy task"}],
            "stream": False,
        }
    )
    assert native["options"]["num_predict"] == 8192


def test_list_dir_schema_path_optional():
    schema = render_tools_for_llm(["list_dir"])[0]
    required = schema["function"]["parameters"].get("required", [])
    assert "path" not in required
