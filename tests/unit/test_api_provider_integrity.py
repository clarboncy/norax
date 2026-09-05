from __future__ import annotations

import pytest

from norax.api_provider import NoraxProvider
from norax.gateway_client import GatewayResponse, StreamEvent


@pytest.mark.asyncio
async def test_retired_proxy_models_are_not_catalogued_or_routable() -> None:
    async with NoraxProvider() as provider:
        models = await provider.list_models()
        assert "retired-provider/model" not in models
        with pytest.raises(ValueError, match="ambiguous namespaced model"):
            await provider.chat(
                "retired-provider/model",
                [{"role": "user", "content": "hello"}],
            )


@pytest.mark.asyncio
async def test_provider_directs_streaming_callers_to_real_stream_api() -> None:
    async with NoraxProvider(ollama_base="http://ollama") as provider:
        with pytest.raises(ValueError, match="chat_stream"):
            await provider.chat(
                "model:latest",
                [{"role": "user", "content": "hello"}],
                stream=True,
            )


@pytest.mark.asyncio
async def test_bare_local_and_explicit_codex_models_use_shared_gateway_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with NoraxProvider(ollama_base="http://ollama", codex_direct_base="http://codex") as p:
        requests = []

        async def ollama_chat(request):
            requests.append(("ollama", request))
            return GatewayResponse(
                content="local answer",
                model=request.model,
                usage={"input_tokens": 2, "output_tokens": 3},
                metadata={"finish_reason": "stop"},
            )

        async def codex_chat(request):
            requests.append(("codex", request))
            return GatewayResponse(content="cloud answer", model=request.model)

        monkeypatch.setattr(p._ollama, "chat", ollama_chat)
        monkeypatch.setattr(p._codex_direct, "chat", codex_chat)

        local = await p.chat(
            "llama3",
            [{"role": "user", "content": "use a tool"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "inspect", "parameters": {"type": "object"}},
                }
            ],
        )
        codex = await p.chat(
            "openai-codex/gpt-test",
            [{"role": "user", "content": "answer"}],
        )

        assert local.content == "local answer"
        assert local.routing_info == {"backend": "ollama_direct", "model": "llama3"}
        assert requests[0][1].metadata["allow_text_tool_calls"] is True
        assert codex.content == "cloud answer"
        assert requests[1][1].model == "gpt-test"


@pytest.mark.asyncio
async def test_provider_stream_api_forwards_deltas_and_normalized_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with NoraxProvider(ollama_base="http://ollama") as provider:

        async def stream(_request):
            yield StreamEvent(kind="delta", text="hello")
            yield StreamEvent(
                kind="final",
                response=GatewayResponse(
                    content="hello",
                    model="llama3",
                    metadata={"done_reason": "stop"},
                ),
            )

        monkeypatch.setattr(provider._ollama, "chat_stream", stream)

        events = [
            event
            async for event in provider.chat_stream(
                "llama3",
                [{"role": "user", "content": "hello"}],
            )
        ]

        assert [event.kind for event in events] == ["delta", "final"]
        assert events[0].text == "hello"
        assert events[1].response is not None
        assert events[1].response.content == "hello"


@pytest.mark.asyncio
async def test_provider_validation_is_strict_and_does_not_coerce_controls() -> None:
    with pytest.raises(ValueError, match="timeout"):
        NoraxProvider(timeout=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="credential-free"):
        NoraxProvider(ollama_base="http://user:secret@localhost:11434")

    async with NoraxProvider() as provider:
        with pytest.raises(TypeError, match="model"):
            await provider.chat(123, [])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="temperature"):
            await provider.chat("llama3", [], temperature=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="max_tokens"):
            await provider.chat("llama3", [], max_tokens=0)
        with pytest.raises(TypeError, match="stream"):
            await provider.chat("llama3", [], stream=1)  # type: ignore[arg-type]
