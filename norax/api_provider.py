"""Model-neutral Norax provider built on the production gateway transport.

This is the small integration surface for external agents. Local Ollama,
Ollama cloud, and Codex-compatible proxy models all receive the same response
bounds, tool normalization, circuit accounting, and reasoning redaction as the
main Norax runtime.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from .gateway_client import GatewayClient, GatewayRequest, GatewayResponse

# A curated discovery fallback, not an allowlist. ``chat`` accepts any bounded
# bare/colon-tagged Ollama model and explicit ``ollama/<id>`` model.
ALL_DIRECT_OLLAMA = {
    "qwen3.8-27b-fast:latest",
    "norax-gemma4-12b-agentic:latest",
    "gemma4:12b",
    "kimi-k2.7-code:cloud",
    "glm-5.3:cloud",
    "glm-5.2:cloud",
    "qwen3-coder-next:cloud",
    "qwen3.5:cloud",
    "deepseek-v4-pro:cloud",
    "deepseek-v4-flash:cloud",
    "gemma4:31b-cloud",
    "nemotron-3-super:cloud",
}
ALL_CODEX_MODELS = {
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.5-personal",
    "gpt-5.5-B",
    "openai-codex/gpt-5.6-sol",
    "openai-codex/gpt-5.6-terra",
    "openai-codex/gpt-5.6-luna",
    "openai-codex/gpt-5.5",
}
ALL_MODELS = ALL_DIRECT_OLLAMA | ALL_CODEX_MODELS

_MAX_MODEL_CHARS = 512
_MAX_DISCOVERED_MODELS = 1_000


def _validated_base_url(name: str, value: Any, *, api_suffix: str = "/v1") -> str:
    if not isinstance(value, str) or not value or len(value) > 2_048:
        raise ValueError(f"{name} must be a bounded HTTP(S) URL")
    normalized = value.rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or any(character.isspace() or not character.isprintable() for character in normalized)
    ):
        raise ValueError(f"{name} must be an absolute credential-free HTTP(S) URL")
    try:
        _port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} contains an invalid port") from exc
    if api_suffix and not normalized.endswith(api_suffix):
        normalized = f"{normalized}{api_suffix}"
    return normalized


def _validated_model(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("model must be text")
    model = value.strip()
    if (
        not model
        or len(model) > _MAX_MODEL_CHARS
        or any(character.isspace() or not character.isprintable() for character in model)
    ):
        raise ValueError(f"model must contain 1-{_MAX_MODEL_CHARS} printable characters")
    return model


@dataclass
class ProviderResponse:
    """Normalized buffered response from any configured model transport."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = "stop"
    routing_info: dict[str, Any] = field(default_factory=dict)
    response_ms: float = 0.0


@dataclass
class ProviderStreamEvent:
    """Streaming delta or final normalized provider response."""

    kind: str
    text: str = ""
    tool_calls_partial: list[dict[str, Any]] | None = None
    response: ProviderResponse | None = None


class NoraxProvider:
    """A provider surface that external agents can embed directly."""

    def __init__(
        self,
        *,
        ollama_base: str | None = None,
        codex_direct_base: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(float(timeout))
            or not 0 < float(timeout) <= 3_600
        ):
            raise ValueError("timeout must be finite and between 0 and 3600 seconds")
        ollama_url = _validated_base_url(
            "ollama_base",
            ollama_base or os.environ.get("NORAX_OLLAMA_URL", "http://127.0.0.1:11434"),
        )
        codex_url = _validated_base_url(
            "codex_direct_base",
            codex_direct_base or os.environ.get("NORAX_CODEX_DIRECT_URL", "http://127.0.0.1:4146"),
        )
        self.ollama_base = ollama_url.removesuffix("/v1")
        self.codex_direct_base = codex_url.removesuffix("/v1")
        self.timeout = float(timeout)
        self._ollama = GatewayClient(
            base_url=ollama_url,
            timeout=self.timeout,
            provider_kind="ollama",
        )
        self._codex_direct = GatewayClient(
            base_url=codex_url,
            timeout=self.timeout,
            provider_kind="codex_direct",
        )
        self._closed = False

    async def __aenter__(self) -> NoraxProvider:
        if self._closed:
            raise RuntimeError("provider is closed")
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.gather(self._ollama.aclose(), self._codex_direct.aclose())

    def _route(self, model: str) -> tuple[str, GatewayClient, str]:
        if model.startswith("openai-codex/"):
            return "codex_direct", self._codex_direct, model.removeprefix("openai-codex/")
        if model.startswith("codex_direct/"):
            return "codex_direct", self._codex_direct, model.removeprefix("codex_direct/")
        if model in ALL_CODEX_MODELS:
            return "codex_direct", self._codex_direct, model
        if model.startswith("ollama/"):
            return "ollama_direct", self._ollama, model.removeprefix("ollama/")
        # Bare ids and colon-tagged/namespaced ids are native Ollama model ids.
        # An unknown slash-only prefix is ambiguous and must be made explicit.
        if "/" in model and ":" not in model:
            raise ValueError("ambiguous namespaced model; prefix Ollama ids with 'ollama/'")
        return "ollama_direct", self._ollama, model

    def _request(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, GatewayClient, GatewayRequest]:
        if self._closed:
            raise RuntimeError("provider is closed")
        selected_model = _validated_model(model)
        if not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages):
            raise TypeError("messages must be a list of message objects")
        if tools is not None and (
            not isinstance(tools, list) or any(not isinstance(item, dict) for item in tools)
        ):
            raise TypeError("tools must be a list of tool objects")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, int | float)
            or not math.isfinite(float(temperature))
            or not 0 <= float(temperature) <= 2
        ):
            raise ValueError("temperature must be finite and between 0 and 2")
        if max_tokens is not None and (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 1 <= max_tokens <= 10_000_000
        ):
            raise ValueError("max_tokens must be between 1 and 10000000")
        backend, client, upstream_model = self._route(selected_model)
        request = GatewayRequest(
            model=upstream_model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=float(temperature) if temperature is not None else None,
            metadata={"allow_text_tool_calls": bool(tools)},
        )
        return backend, client, request

    @staticmethod
    def _response(
        response: GatewayResponse,
        *,
        backend: str,
        elapsed_ms: float,
    ) -> ProviderResponse:
        finish_reason = response.metadata.get("finish_reason") or response.metadata.get(
            "done_reason"
        )
        return ProviderResponse(
            content=response.content,
            tool_calls=list(response.tool_calls),
            model=response.model,
            usage=dict(response.usage),
            finish_reason=str(finish_reason or "stop")[:128],
            routing_info={"backend": backend, "model": response.model},
            response_ms=elapsed_ms,
        )

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> ProviderResponse:
        """Return one buffered response; use ``chat_stream`` for streaming."""
        if type(stream) is not bool:
            raise TypeError("stream must be a boolean")
        if stream:
            raise ValueError("use NoraxProvider.chat_stream() for streaming")
        backend, client, request = self._request(
            model,
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        started = time.monotonic()
        response = await client.chat(request)
        return self._response(
            response,
            backend=backend,
            elapsed_ms=(time.monotonic() - started) * 1000,
        )

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ProviderStreamEvent]:
        """Stream normalized text/tool deltas and one final response event."""
        backend, client, request = self._request(
            model,
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        started = time.monotonic()
        async for event in client.chat_stream(request):
            final = None
            if event.response is not None:
                final = self._response(
                    event.response,
                    backend=backend,
                    elapsed_ms=(time.monotonic() - started) * 1000,
                )
            yield ProviderStreamEvent(
                kind=event.kind,
                text=event.text,
                tool_calls_partial=event.tool_calls_partial,
                response=final,
            )

    async def list_models(self) -> list[str]:
        """Return curated route examples; this is not an availability claim."""
        return sorted(ALL_MODELS)

    async def discover_models(self) -> dict[str, dict[str, Any]]:
        """Return bounded live model catalogs with per-backend evidence."""

        async def discover(name: str, client: GatewayClient) -> tuple[str, dict[str, Any]]:
            try:
                raw_models = await client.fetch_model_ids(max_models=_MAX_DISCOVERED_MODELS)
                models = [
                    model
                    for raw in raw_models
                    if isinstance(raw, str)
                    and (model := raw.strip())
                    and len(model) <= _MAX_MODEL_CHARS
                ]
                return name, {"ok": True, "models": models}
            except Exception as exc:  # noqa: BLE001 - evidence returned to caller
                return name, {
                    "ok": False,
                    "models": [],
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }

        pairs = await asyncio.gather(
            discover("ollama", self._ollama),
            discover("codex_direct", self._codex_direct),
        )
        return dict(pairs)

    async def health(self) -> dict[str, bool]:
        """Check both model transports without spending inference tokens."""
        ollama, codex = await asyncio.gather(
            self._ollama.transport_probe(timeout=5.0),
            self._codex_direct.transport_probe(timeout=5.0),
        )
        return {"ollama": ollama.get("ok") is True, "codex_direct": codex.get("ok") is True}


_default_provider: NoraxProvider | None = None


def get_provider(**kwargs: Any) -> NoraxProvider:
    """Return the shared provider, or an explicitly configured independent one."""
    global _default_provider
    if kwargs:
        return NoraxProvider(**kwargs)
    if _default_provider is None or _default_provider._closed:
        _default_provider = NoraxProvider()
    return _default_provider


async def chat(
    model: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> ProviderResponse:
    """Send one buffered request through the shared Norax provider."""
    return await get_provider().chat(model, messages, **kwargs)
