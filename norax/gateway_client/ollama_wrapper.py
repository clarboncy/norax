"""OllamaGatewayClient — per-model wrapper for Norax Ollama path.

Applies per-model profiles (temperature, think, tool limits), exec guard
nudges, observability, and optional fallback before delegating to GatewayClient.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

from .ollama_observability import OllamaObservability
from .ollama_profiles import (
    apply_profile_to_gateway_request,
    build_exec_guard_nudge,
    normalize_model_name,
    resolve_profile,
)

log = logging.getLogger("norax.gateway_client.ollama_wrapper")


class OllamaGatewayClient:
    """GatewayClient duck-type with per-model Ollama best practices."""

    def __init__(
        self,
        inner: Any,
        *,
        metrics: Any | None = None,
    ) -> None:
        self._inner = inner
        self.base_url = inner.base_url
        self.model_prefix = getattr(inner, "model_prefix", None)
        self.stream_required = getattr(inner, "stream_required", False)
        self.chat_path = getattr(inner, "chat_path", "/chat/completions")
        self.provider_kind = "ollama"
        self._obs = OllamaObservability(metrics=metrics)

    @property
    def kind(self) -> str:
        return "ollama"

    def _provider_kind(self) -> str:
        return "ollama"

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def transport_probe(
        self,
        model: str | None = None,
        *,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Delegate the zero-token transport check to the Ollama client."""
        return await self._inner.transport_probe(model, timeout=timeout)

    def _inject_exec_guard(self, req: Any) -> Any:
        """Append exec guard nudge for arithmetic prompts on agentic models."""
        profile = resolve_profile(req.model)
        if profile.role not in {"agentic", "executor"}:
            return req
        user_text = ""
        for m in reversed(req.messages or []):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                user_text = c if isinstance(c, str) else str(c or "")
                break
        nudge = build_exec_guard_nudge(user_text)
        if not nudge:
            return req
        meta = dict(req.metadata or {})
        if meta.get("ollama_exec_guard"):
            return req
        meta["ollama_exec_guard"] = True
        msgs = list(req.messages or [])
        msgs.append({"role": "user", "content": nudge})
        return replace(req, messages=msgs, metadata=meta)

    async def chat(self, req: Any) -> Any:
        from . import GatewayRequest

        if not isinstance(req, GatewayRequest):
            raise TypeError(f"expected GatewayRequest, got {type(req)}")

        req = apply_profile_to_gateway_request(req)
        req = self._inject_exec_guard(req)

        t0 = time.monotonic()
        try:
            resp = await self._inner.chat(req)
        except Exception:
            self._obs.record_call(
                model=normalize_model_name(req.model),
                latency_ms=(time.monotonic() - t0) * 1000,
                status="error",
                profile_role=resolve_profile(req.model).role,
            )
            raise

        latency_ms = (time.monotonic() - t0) * 1000
        profile = resolve_profile(resp.model or req.model)
        tool_count = len(resp.tool_calls or [])
        raw = dict(resp.raw or {})
        raw["ollama_enhanced"] = True
        raw["latency_ms"] = latency_ms
        raw["profile_role"] = profile.role

        self._obs.record_call(
            model=normalize_model_name(resp.model or req.model),
            latency_ms=latency_ms,
            tool_count=tool_count,
            profile_role=profile.role,
        )

        return replace(resp, raw=raw)

    async def chat_stream(self, req: Any):
        from . import GatewayRequest, StreamEvent

        if not isinstance(req, GatewayRequest):
            raise TypeError(f"expected GatewayRequest, got {type(req)}")

        req = apply_profile_to_gateway_request(req)
        req = self._inject_exec_guard(req)

        async for evt in self._inner.chat_stream(req):
            if evt.kind == "final" and evt.response is not None:
                raw = dict(evt.response.raw or {})
                raw["ollama_enhanced"] = True
                raw["profile_role"] = resolve_profile(req.model).role
                yield StreamEvent(kind="final", response=replace(evt.response, raw=raw))
            else:
                yield evt
