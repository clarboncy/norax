#!/usr/bin/env python3
"""Norax Gateway Proxy — provider-neutral OpenAI-compatible reverse proxy.

Forwards requests to one explicitly configured upstream, handling:
  - Streaming SSE (chat/completions with stream:true)
  - Non-streaming chat/completions
  - /v1/models passthrough
  - Auth header forwarding
  - Request/response logging
  - Graceful timeout handling

Usage:
  python -m norax.gateway_proxy              # default port 8899
  python -m norax.gateway_proxy --port 8899

Env:
  NORAX_GATEWAY_UPSTREAM (default: http://127.0.0.1:11434/v1)
  NORAX_GATEWAY_TOKEN  (API key — forwarded as Bearer token)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("norax.gateway_proxy")

UPSTREAM = (os.environ.get("NORAX_GATEWAY_UPSTREAM") or "http://127.0.0.1:11434/v1").rstrip("/")


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and minimum <= value <= maximum else default


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if minimum <= value <= maximum else default


TIMEOUT = _env_float("NORAX_GATEWAY_TIMEOUT", 120.0, minimum=1.0, maximum=3600.0)
PORT = _env_int("GATEWAY_PROXY_PORT", 8899, minimum=1, maximum=65535)
MAX_RETRIES = _env_int("NORAX_GATEWAY_RETRIES", 2, minimum=0, maximum=5)
MAX_REQUEST_BYTES = _env_int(
    "NORAX_GATEWAY_MAX_REQUEST_BYTES", 16_000_000, minimum=1024, maximum=256_000_000
)
MAX_RESPONSE_BYTES = _env_int(
    "NORAX_GATEWAY_MAX_RESPONSE_BYTES", 32_000_000, minimum=1024, maximum=512_000_000
)
MAX_MODEL_CATALOG_ITEMS = _env_int("NORAX_GATEWAY_MAX_MODELS", 2_000, minimum=1, maximum=20_000)

# GPT ~400k tokens; Claude ~1M (char heuristic *4)
_CONTEXT_LIMITS = {"gpt": 1_600_000, "claude": 4_000_000, "default": 1_600_000}

# Single shared httpx client with connection pooling
_client: httpx.AsyncClient | None = None
_stream_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # shutdown
    global _client, _stream_client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None
    if _stream_client and not _stream_client.is_closed:
        await _stream_client.aclose()
        _stream_client = None


app = FastAPI(title="Norax Gateway Proxy", lifespan=lifespan)

# Inbound auth: when NORAX_GATEWAY_INBOUND_KEY is set, all requests must
# include it as a Bearer token. Health endpoints are exempt. When the
# proxy is bound to a non-loopback address, the inbound key is REQUIRED
# — without it anyone who can reach the proxy can consume the upstream
# gateway token budget for free.
_INBOUND_KEY = os.environ.get("NORAX_GATEWAY_INBOUND_KEY", "")
_HEALTH_PATHS = {"/health", "/healthz", "/readyz"}
_BIND_HOST = os.environ.get("NORAX_GATEWAY_BIND", "127.0.0.1")


def _is_loopback_bind(host: str) -> bool:
    normalized = (host or "").strip().lower().strip("[]")
    if normalized == "localhost":
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


if not _is_loopback_bind(_BIND_HOST) and not _INBOUND_KEY:
    raise RuntimeError(
        "NORAX_GATEWAY_INBOUND_KEY must be set when binding to a non-loopback "
        "address. Without it, anyone who can reach the proxy can consume the "
        "upstream gateway token budget."
    )


@app.middleware("http")
async def inbound_auth_middleware(request: Request, call_next):
    """Verify inbound Bearer token when NORAX_GATEWAY_INBOUND_KEY is configured."""
    if _INBOUND_KEY and request.url.path not in _HEALTH_PATHS:
        auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
        token = auth.removeprefix("Bearer ").removeprefix("bearer ").strip()
        if not token or not secrets.compare_digest(token, _INBOUND_KEY):
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid or missing API key", "type": "auth_error"}},
            )
    return await call_next(request)


STREAM_TIMEOUT = _env_float("NORAX_STREAM_TIMEOUT", 300.0, minimum=1.0, maximum=7200.0)
READINESS_CACHE_SECONDS = _env_float(
    "NORAX_GATEWAY_READY_CACHE_SECONDS", 5.0, minimum=0.0, maximum=300.0
)
_readiness_cache: tuple[float, int, dict] | None = None


class _BodyTooLargeError(ValueError):
    pass


class _SSEAggregationError(ValueError):
    """Raised when a successful upstream SSE body cannot be trusted as a result."""


class _ContextTooLargeError(ValueError):
    """Raised when required recent context cannot fit without corrupting semantics."""


async def _read_request_body(request: Request) -> bytes:
    raw_length = request.headers.get("content-length")
    if raw_length:
        try:
            content_length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length header") from exc
        if content_length < 0:
            raise ValueError("invalid Content-Length header")
        if content_length > MAX_REQUEST_BYTES:
            raise _BodyTooLargeError("request body exceeds configured limit")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_REQUEST_BYTES:
            raise _BodyTooLargeError("request body exceeds configured limit")
        chunks.append(chunk)
    return b"".join(chunks)


async def _read_response_body(response: httpx.Response) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            raise _BodyTooLargeError("upstream response exceeds configured limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _sse_error(message: str, error_type: str, status: int) -> bytes:
    payload = {"error": {"message": message, "type": error_type, "status": status}}
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _error_message_from_body(body: bytes) -> str:
    """Extract a bounded, human-readable upstream error without trusting its shape."""
    text = body.decode("utf-8", errors="replace")[:2_000]
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return text or "upstream request failed"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:2_000]
        if isinstance(error, str):
            return error[:2_000]
        if payload.get("message"):
            return str(payload["message"])[:2_000]
    return text or "upstream request failed"


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(TIMEOUT, connect=10.0),
            follow_redirects=True,
            # Bumped from 20/10 → 200/50 per FastAPI/httpx LLM proxy best
            # practices: default pool limits bottleneck at modest concurrency,
            # which we hit during bursty multi-agent runs.
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            http2=True,
        )
    return _client


def _get_stream_client() -> httpx.AsyncClient:
    """Separate client with much longer read timeout for SSE streaming."""
    global _stream_client
    if _stream_client is None or _stream_client.is_closed:
        _stream_client = httpx.AsyncClient(
            timeout=httpx.Timeout(STREAM_TIMEOUT, connect=10.0),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            http2=True,
        )
    return _stream_client


def _normalize_model(model: str) -> str:
    """Normalize harmless surrounding whitespace without rewriting model IDs."""
    return (model or "").strip()


def _model_family(model: str) -> str:
    m = _normalize_model(model).lower()
    if m.startswith(("gpt-", "codex", "o1-", "o3-", "o4-")):
        return "gpt"
    if m == "claude" or m.startswith("claude-"):
        return "claude"
    return "default"


def _estimate_chars(messages: list) -> int:
    total = 0
    for m in messages or []:
        total += 32  # conservative per-message serialization/token overhead
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    total += len(str(part.get("text") or part.get("content") or ""))
        total += len(str(m.get("tool_calls") or ""))
    return total


def _trim_payload_messages(payload: dict) -> bool:
    """Trim old context in linear time without splitting tool exchanges."""
    family = _model_family(str(payload.get("model") or ""))
    limit = _CONTEXT_LIMITS.get(family, _CONTEXT_LIMITS["default"])
    messages = payload.get("messages") or []
    payload_without_messages = {key: value for key, value in payload.items() if key != "messages"}
    fixed_chars = len(
        json.dumps(payload_without_messages, ensure_ascii=False, separators=(",", ":"))
    )
    if fixed_chars + _estimate_chars(messages) <= limit:
        return False

    system_msgs = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    notice = {
        "role": "system",
        "content": "Norax gateway notice: older context was trimmed to fit the model window.",
    }
    required_chars = fixed_chars + _estimate_chars(system_msgs + [notice])
    if required_chars >= limit:
        raise _ContextTooLargeError("system instructions and request metadata exceed model context")

    # An assistant tool request and all immediately following tool results are
    # one atomic context unit. Selecting message-by-message can otherwise leave
    # an orphan result that strict providers reject or, worse, misinterpret.
    units: list[list[dict]] = []
    index = 0
    while index < len(rest):
        message = rest[index]
        unit = [message]
        index += 1
        if message.get("role") == "assistant" and message.get("tool_calls"):
            while index < len(rest) and rest[index].get("role") == "tool":
                unit.append(rest[index])
                index += 1
        units.append(unit)

    selected_reversed: list[list[dict]] = []
    used_chars = required_chars
    for unit in reversed(units):
        unit_chars = _estimate_chars(unit)
        if used_chars + unit_chars > limit:
            break
        selected_reversed.append(unit)
        used_chars += unit_chars
    if rest and not selected_reversed:
        raise _ContextTooLargeError("most recent conversation unit exceeds model context")

    tail = [message for unit in reversed(selected_reversed) for message in unit]
    trimmed = system_msgs + [notice] + tail
    payload["messages"] = trimmed
    return True


def _aggregate_openai_sse(raw: bytes) -> dict:
    """Collect a valid OpenAI SSE stream into one chat.completion JSON.

    This parser is intentionally strict. Returning a plausible empty completion
    after an upstream error or malformed frame would teach callers that a failed
    request succeeded.
    """
    response_id = "chatcmpl-proxy-agg"
    model = "unknown"
    created: int | None = None
    usage: dict | None = None
    choice_states: dict[int, dict] = {}
    valid_events = 0

    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    for block in normalized.split(b"\n\n"):
        if not block.strip():
            continue
        event_type = ""
        data_lines: list[bytes] = []
        for line in block.split(b"\n"):
            if not line or line.startswith(b":"):
                continue
            field, separator, value = line.partition(b":")
            if not separator:
                value = b""
            if value.startswith(b" "):
                value = value[1:]
            if field == b"event":
                event_type = value.decode("utf-8", errors="replace")
            elif field == b"data":
                data_lines.append(value)
        if not data_lines:
            continue
        data = b"\n".join(data_lines)
        if data.strip() == b"[DONE]":
            continue
        try:
            event = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise _SSEAggregationError("upstream returned malformed SSE JSON") from exc
        if not isinstance(event, dict):
            raise _SSEAggregationError("upstream returned a non-object SSE event")
        valid_events += 1

        if event_type == "error" or "error" in event:
            error = event.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or "upstream stream reported an error")
            elif error:
                message = str(error)
            else:
                message = str(event.get("message") or "upstream stream reported an error")
            raise _SSEAggregationError(message[:2_000])

        if event.get("id"):
            response_id = str(event["id"])
        if event.get("model"):
            model = str(event["model"])
        if isinstance(event.get("created"), int):
            created = event["created"]
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]

        choices = event.get("choices") or []
        if not isinstance(choices, list):
            raise _SSEAggregationError("upstream SSE choices field was not a list")
        for position, choice in enumerate(choices):
            if not isinstance(choice, dict):
                raise _SSEAggregationError("upstream SSE contained an invalid choice")
            raw_index = choice.get("index", position)
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise _SSEAggregationError("upstream SSE choice had an invalid index") from exc
            state = choice_states.setdefault(
                index,
                {
                    "role": "assistant",
                    "content": [],
                    "refusal": [],
                    "reasoning_content": [],
                    "tool_calls": {},
                    "finish_reason": None,
                },
            )
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                raise _SSEAggregationError("upstream SSE choice delta was not an object")
            if delta.get("role"):
                state["role"] = str(delta["role"])
            for delta_field in ("content", "refusal", "reasoning_content"):
                delta_value = delta.get(delta_field)
                if delta_value is not None:
                    if not isinstance(delta_value, str):
                        raise _SSEAggregationError(f"upstream SSE {delta_field} delta was not text")
                    state[delta_field].append(delta_value)

            raw_tool_calls = delta.get("tool_calls") or []
            if not isinstance(raw_tool_calls, list):
                raise _SSEAggregationError("upstream SSE tool_calls field was not a list")
            for tool_position, tool_call in enumerate(raw_tool_calls):
                if not isinstance(tool_call, dict):
                    raise _SSEAggregationError("upstream SSE contained an invalid tool call")
                raw_tool_index = tool_call.get("index", tool_position)
                try:
                    tool_index = int(raw_tool_index)
                except (TypeError, ValueError) as exc:
                    raise _SSEAggregationError("upstream tool call had an invalid index") from exc
                entry = state["tool_calls"].setdefault(
                    tool_index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if tool_call.get("id"):
                    entry["id"] = str(tool_call["id"])
                if tool_call.get("type"):
                    entry["type"] = str(tool_call["type"])
                function = tool_call.get("function") or {}
                if not isinstance(function, dict):
                    raise _SSEAggregationError("upstream tool call function was not an object")
                if function.get("name"):
                    entry["function"]["name"] = str(function["name"])
                if function.get("arguments"):
                    entry["function"]["arguments"] += str(function["arguments"])
            if choice.get("finish_reason") is not None:
                state["finish_reason"] = choice["finish_reason"]

    if not valid_events or not choice_states:
        raise _SSEAggregationError("upstream SSE ended without a completion choice")

    aggregated_choices: list[dict] = []
    for choice_index in sorted(choice_states):
        state = choice_states[choice_index]
        tool_calls = [state["tool_calls"][i] for i in sorted(state["tool_calls"])]
        for tool_index, tool_call in enumerate(tool_calls):
            if not tool_call.get("id"):
                tool_call["id"] = f"call_proxy_{choice_index}_{tool_index}"
        content = "".join(state["content"])
        output_message: dict = {"role": state["role"], "content": content or None}
        refusal = "".join(state["refusal"])
        reasoning_content = "".join(state["reasoning_content"])
        if refusal:
            output_message["refusal"] = refusal
        if reasoning_content:
            output_message["reasoning_content"] = reasoning_content
        if tool_calls:
            output_message["tool_calls"] = tool_calls
        finish_reason = state["finish_reason"]
        if finish_reason is None:
            finish_reason = "tool_calls" if tool_calls else "stop"
        aggregated_choices.append(
            {
                "index": choice_index,
                "message": output_message,
                "finish_reason": finish_reason,
            }
        )

    result: dict = {
        "id": response_id,
        "object": "chat.completion",
        "model": model,
        "choices": aggregated_choices,
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    if created is not None:
        result["created"] = created
    return result


def _forward_headers(request: Request) -> dict[str, str]:
    """Forward end-to-end headers without leaking the proxy's own credential."""
    out: dict[str, str] = {}
    skip = {"host", "content-length", "transfer-encoding", "connection"}
    for k, v in request.headers.items():
        if k.lower() not in skip:
            out[k] = v

    incoming_auth = out.get("authorization") or out.get("Authorization") or ""
    if _INBOUND_KEY and incoming_auth:
        candidate = incoming_auth.removeprefix("Bearer ").removeprefix("bearer ").strip()
        if candidate and secrets.compare_digest(candidate, _INBOUND_KEY):
            out.pop("authorization", None)
            out.pop("Authorization", None)
            incoming_auth = ""

    header_api_key = out.get("x-api-key") or out.get("X-Api-Key")
    if not incoming_auth and not header_api_key:
        token = os.environ.get("NORAX_GATEWAY_TOKEN")
        if token:
            out["authorization"] = f"Bearer {token}"
    return out


def _anthropic_json_to_openai(data: dict) -> dict:
    """Anthropic message JSON → OpenAI chat.completion."""
    content_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    for i, block in enumerate(data.get("content") or []):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            content_parts.append(str(block.get("text") or ""))
        elif block.get("type") == "tool_use":
            tool_calls[i] = {
                "id": block.get("id") or f"call_{i}",
                "type": "function",
                "function": {
                    "name": block.get("name") or "tool",
                    "arguments": __import__("json").dumps(block.get("input") or {}),
                },
            }
    content = "".join(content_parts).strip() or None
    tcs = [tool_calls[k] for k in sorted(tool_calls)]
    msg: dict = {"role": "assistant", "content": content}
    if tcs:
        msg["tool_calls"] = tcs
        msg["content"] = None
    return {
        "id": data.get("id") or "chatcmpl-proxy-agg",
        "object": "chat.completion",
        "model": data.get("model") or "unknown",
        "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tcs else "stop"}],
        "usage": {
            "prompt_tokens": int((data.get("usage") or {}).get("input_tokens") or 0),
            "completion_tokens": int((data.get("usage") or {}).get("output_tokens") or 0),
            "total_tokens": 0,
        },
    }


def _aggregate_anthropic_sse(raw: bytes) -> dict:
    """Collect Anthropic SSE events into one message JSON."""
    import json as _json

    text_parts: list[str] = []
    model = "unknown"
    msg_id = "chatcmpl-proxy-agg"
    usage: dict = {}
    for block in raw.split(b"\n\n"):
        if not block.strip():
            continue
        data_line = b""
        for line in block.split(b"\n"):
            if line.startswith(b"data:"):
                data_line = line[5:].strip()
        if not data_line:
            continue
        try:
            evt = _json.loads(data_line)
        except _json.JSONDecodeError:
            continue
        etype = evt.get("type")
        if etype == "message_start":
            message = evt.get("message") or {}
            model = message.get("model") or model
            msg_id = message.get("id") or msg_id
            usage = message.get("usage") or usage
        elif etype == "content_block_delta":
            delta = evt.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                text_parts.append(delta["text"])
        elif etype == "message_delta":
            usage = evt.get("usage") or usage
    return _anthropic_json_to_openai(
        {
            "id": msg_id,
            "model": model,
            "content": ([{"type": "text", "text": "".join(text_parts)}] if text_parts else []),
            "usage": usage,
        }
    )


async def _fetch_upstream_models(
    client: httpx.AsyncClient,
    headers: dict[str, str] | None = None,
) -> list[dict]:
    """Fetch and validate a bounded model list from the configured upstream."""
    async with client.stream("GET", f"{UPSTREAM}/models", headers=headers) as response:
        body = await _read_response_body(response)
        response.raise_for_status()
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("upstream model catalog was not valid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("upstream model catalog did not contain a data list")
    models: list[dict] = []
    seen: set[str] = set()
    for item in payload["data"][:MAX_MODEL_CATALOG_ITEMS]:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if (
            isinstance(model_id, str)
            and 0 < len(model_id) <= 256
            and all(character.isprintable() for character in model_id)
            and model_id not in seen
        ):
            seen.add(model_id)
            models.append(item)
    return models


@app.get("/v1/models")
async def models(request: Request):
    """Return a validated, bounded catalog from the configured upstream."""
    client = _get_client()
    if "group" in request.query_params:
        return JSONResponse(
            {
                "error": {
                    "message": "model groups are not configured on this provider-neutral proxy",
                    "type": "invalid_request",
                }
            },
            status_code=400,
        )
    try:
        upstream_models = await _fetch_upstream_models(client, _forward_headers(request))
    except _BodyTooLargeError:
        return JSONResponse(
            {
                "error": {
                    "message": "upstream model catalog exceeded the configured limit",
                    "type": "upstream_response_too_large",
                }
            },
            status_code=502,
        )
    except Exception as exc:
        log.warning("models.fetch_failed: %s", exc)
        return JSONResponse(
            {"error": {"message": str(exc), "type": "upstream_error"}},
            status_code=502,
        )
    return JSONResponse(content={"object": "list", "data": upstream_models})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Forward chat/completions — handles both streaming and non-streaming."""
    t0 = time.monotonic()
    try:
        body = await _read_request_body(request)
    except _BodyTooLargeError as exc:
        return JSONResponse(
            {"error": {"message": str(exc), "type": "request_too_large"}}, status_code=413
        )
    except ValueError as exc:
        return JSONResponse(
            {"error": {"message": str(exc), "type": "invalid_request"}}, status_code=400
        )

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(
            {"error": {"message": "invalid JSON", "type": "invalid_request"}}, status_code=400
        )
    if not isinstance(payload, dict):
        return JSONResponse(
            {
                "error": {
                    "message": "request JSON must be an object",
                    "type": "invalid_request",
                }
            },
            status_code=400,
        )
    if not isinstance(payload.get("model"), str) or not payload["model"].strip():
        return JSONResponse(
            {"error": {"message": "model must be a non-empty string", "type": "invalid_request"}},
            status_code=400,
        )
    messages = payload.get("messages")
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        return JSONResponse(
            {"error": {"message": "messages must be a list of objects", "type": "invalid_request"}},
            status_code=400,
        )

    raw_model = payload.get("model", "unknown")
    model = _normalize_model(str(raw_model))
    if model != raw_model:
        payload["model"] = model
    headers = _forward_headers(request)
    for key in [key for key in payload if str(key).startswith("_norax_")]:
        payload.pop(key, None)

    client_wants_stream = payload.get("stream", False)
    if not isinstance(client_wants_stream, bool):
        return JSONResponse(
            {"error": {"message": "stream must be a boolean", "type": "invalid_request"}},
            status_code=400,
        )
    try:
        trimmed = _trim_payload_messages(payload)
    except _ContextTooLargeError as exc:
        return JSONResponse(
            {
                "error": {
                    "message": str(exc),
                    "type": "context_length_exceeded",
                }
            },
            status_code=413,
        )
    payload["stream"] = True  # upstream always streams (prevents hang)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    if len(body) > MAX_REQUEST_BYTES:
        return JSONResponse(
            {
                "error": {
                    "message": "normalized request body exceeds configured limit",
                    "type": "request_too_large",
                }
            },
            status_code=413,
        )
    msg_count = len(payload.get("messages", []))

    log.info(
        "chat.request model=%s family=%s msgs=%d client_stream=%s trimmed=%s",
        model,
        _model_family(model),
        msg_count,
        client_wants_stream,
        trimmed,
    )

    if client_wants_stream:
        return await _stream_response(headers, body, model, t0)
    return await _non_stream_response(_get_client(), headers, body, model, t0, aggregate_sse=True)


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Forward Anthropic-compatible Claude Code messages requests."""
    t0 = time.monotonic()
    try:
        body = await _read_request_body(request)
    except _BodyTooLargeError as exc:
        return JSONResponse(
            {"error": {"message": str(exc), "type": "request_too_large"}}, status_code=413
        )
    except ValueError as exc:
        return JSONResponse(
            {"error": {"message": str(exc), "type": "invalid_request"}}, status_code=400
        )

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(
            {"error": {"message": "invalid JSON", "type": "invalid_request"}}, status_code=400
        )
    if not isinstance(payload, dict):
        return JSONResponse(
            {
                "error": {
                    "message": "request JSON must be an object",
                    "type": "invalid_request",
                }
            },
            status_code=400,
        )
    if not isinstance(payload.get("model"), str) or not payload["model"].strip():
        return JSONResponse(
            {"error": {"message": "model must be a non-empty string", "type": "invalid_request"}},
            status_code=400,
        )
    messages = payload.get("messages")
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        return JSONResponse(
            {"error": {"message": "messages must be a list of objects", "type": "invalid_request"}},
            status_code=400,
        )

    raw_model = payload.get("model", "unknown")
    model = _normalize_model(str(raw_model))
    if model != raw_model:
        payload["model"] = model
    headers = _forward_headers(request)
    is_stream = payload.get("stream", False)
    if not isinstance(is_stream, bool):
        return JSONResponse(
            {"error": {"message": "stream must be a boolean", "type": "invalid_request"}},
            status_code=400,
        )
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    if len(body) > MAX_REQUEST_BYTES:
        return JSONResponse(
            {
                "error": {
                    "message": "normalized request body exceeds configured limit",
                    "type": "request_too_large",
                }
            },
            status_code=413,
        )
    msg_count = len(payload.get("messages", []))

    log.info("messages.request model=%s msgs=%d stream=%s", model, msg_count, is_stream)

    if is_stream:
        return await _stream_response(headers, body, model, t0, path="/messages")
    return await _non_stream_response(_get_client(), headers, body, model, t0, path="/messages")


async def _non_stream_response(
    client: httpx.AsyncClient,
    headers: dict,
    body: bytes,
    model: str,
    t0: float,
    path: str = "/chat/completions",
    *,
    aggregate_sse: bool = False,
    anthropic_to_openai: bool = False,
) -> StreamingResponse | JSONResponse:
    """Non-streaming client: stream upstream, optionally aggregate SSE to JSON."""
    import json as _json

    last_error: JSONResponse | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with client.stream(
                "POST",
                f"{UPSTREAM}{path}",
                content=body,
                headers=headers,
            ) as r:
                response_body = await _read_response_body(r)
                elapsed = time.monotonic() - t0
                log.info(
                    "chat.response model=%s status=%d elapsed=%.1fs size=%d attempt=%d",
                    model,
                    r.status_code,
                    elapsed,
                    len(response_body),
                    attempt + 1,
                )

                if r.status_code >= 500 and attempt < MAX_RETRIES:
                    last_error = JSONResponse(
                        content={
                            "error": {
                                "message": response_body.decode("utf-8", errors="replace")[:500]
                            }
                        },
                        status_code=r.status_code,
                    )
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue

                if r.status_code >= 400:
                    log.error(
                        "chat.upstream_error status=%d body=%s", r.status_code, response_body[:500]
                    )
                    try:
                        data = _json.loads(response_body)
                    except _json.JSONDecodeError:
                        data = {
                            "error": {
                                "message": response_body.decode("utf-8", errors="replace")[:500],
                                "type": "upstream_error",
                            }
                        }
                    return JSONResponse(content=data, status_code=r.status_code)

                ctype = (r.headers.get("content-type") or "").lower()
                if aggregate_sse and (
                    "text/event-stream" in ctype
                    or response_body.startswith(b"data:")
                    or response_body.startswith(b"event:")
                ):
                    try:
                        if anthropic_to_openai:
                            data = _aggregate_anthropic_sse(response_body)
                        else:
                            data = _aggregate_openai_sse(response_body)
                    except _SSEAggregationError as exc:
                        log.error("chat.invalid_sse model=%s err=%s", model, exc)
                        return JSONResponse(
                            {
                                "error": {
                                    "message": str(exc),
                                    "type": "invalid_upstream_response",
                                    "status": 502,
                                }
                            },
                            status_code=502,
                        )
                    return JSONResponse(content=data, status_code=200)

                try:
                    data = _json.loads(response_body)
                except (_json.JSONDecodeError, UnicodeDecodeError):
                    return JSONResponse(
                        {
                            "error": {
                                "message": "upstream returned invalid JSON",
                                "type": "invalid_upstream_response",
                                "status": 502,
                            }
                        },
                        status_code=502,
                    )
                if anthropic_to_openai and isinstance(data, dict) and "content" in data:
                    data = _anthropic_json_to_openai(data)
                return JSONResponse(content=data, status_code=r.status_code)
        except _BodyTooLargeError:
            log.error("chat.response_too_large model=%s limit=%d", model, MAX_RESPONSE_BYTES)
            return JSONResponse(
                {
                    "error": {
                        "message": "upstream response exceeded the configured limit",
                        "type": "upstream_response_too_large",
                        "status": 502,
                    }
                },
                status_code=502,
            )
        except httpx.TimeoutException:
            elapsed = time.monotonic() - t0
            if attempt < MAX_RETRIES:
                log.warning(
                    "chat.timeout_retry model=%s elapsed=%.1fs attempt=%d",
                    model,
                    elapsed,
                    attempt + 1,
                )
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            log.error("chat.timeout model=%s elapsed=%.1fs", model, elapsed)
            return JSONResponse(
                {
                    "error": {
                        "message": f"upstream timeout after {elapsed:.0f}s — provider channels may be unavailable",
                        "type": "timeout",
                        "status": 504,
                    }
                },
                status_code=504,
            )
        except Exception as e:
            log.error("chat.error model=%s err=%s", model, e)
            return JSONResponse(
                {"error": {"message": str(e), "type": "proxy_error", "status": 502}},
                status_code=502,
            )
    return last_error or JSONResponse(
        {
            "error": {
                "message": "upstream failed after retries",
                "type": "proxy_error",
                "status": 502,
            }
        },
        status_code=502,
    )


async def _stream_response(
    headers: dict[str, str],
    body: bytes,
    model: str,
    t0: float,
    path: str = "/chat/completions",
) -> StreamingResponse:
    """Forward upstream bytes immediately and never replay after delivery."""

    async def generate():
        stream_client = _get_stream_client()
        current_headers = dict(headers)
        bytes_sent = 0

        for attempt in range(MAX_RETRIES + 1):
            try:
                async with stream_client.stream(
                    "POST",
                    f"{UPSTREAM}{path}",
                    content=body,
                    headers=current_headers,
                ) as response:
                    if response.status_code >= 400:
                        error_body = await _read_response_body(response)
                        log.error(
                            "chat.stream_error status=%d body=%s attempt=%d",
                            response.status_code,
                            error_body[:500],
                            attempt + 1,
                        )
                        if response.status_code >= 500 and attempt < MAX_RETRIES:
                            log.warning(
                                "chat.stream_5xx_retry model=%s status=%d attempt=%d",
                                model,
                                response.status_code,
                                attempt + 1,
                            )
                            await asyncio.sleep(1.5 * (attempt + 1))
                            continue
                        yield _sse_error(
                            _error_message_from_body(error_body),
                            "upstream_error",
                            response.status_code,
                        )
                        return

                    attempt_bytes = 0
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        attempt_bytes += len(chunk)
                        if attempt_bytes > MAX_RESPONSE_BYTES:
                            separator = b"\n\n" if bytes_sent else b""
                            yield separator + _sse_error(
                                "upstream stream exceeded the configured limit",
                                "upstream_response_too_large",
                                502,
                            )
                            return
                        bytes_sent += len(chunk)
                        yield chunk

                    if attempt_bytes == 0:
                        if attempt < MAX_RETRIES:
                            log.warning(
                                "chat.stream_empty_retry model=%s attempt=%d",
                                model,
                                attempt + 1,
                            )
                            await asyncio.sleep(1.5 * (attempt + 1))
                            continue
                        yield _sse_error(
                            "upstream stream ended without data",
                            "invalid_upstream_response",
                            502,
                        )
                        return

                    elapsed = time.monotonic() - t0
                    log.info(
                        "chat.stream_done model=%s elapsed=%.1fs size=%d",
                        model,
                        elapsed,
                        bytes_sent,
                    )
                    return
            except asyncio.CancelledError:
                log.info("chat.stream_cancelled model=%s bytes_sent=%d", model, bytes_sent)
                raise
            except _BodyTooLargeError:
                separator = b"\n\n" if bytes_sent else b""
                yield separator + _sse_error(
                    "upstream response exceeded the configured limit",
                    "upstream_response_too_large",
                    502,
                )
                return
            except httpx.TimeoutException:
                elapsed = time.monotonic() - t0
                if not bytes_sent and attempt < MAX_RETRIES:
                    log.warning(
                        "chat.stream_timeout_retry model=%s elapsed=%.1fs attempt=%d",
                        model,
                        elapsed,
                        attempt + 1,
                    )
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                log.error("chat.stream_timeout model=%s elapsed=%.1fs", model, elapsed)
                separator = b"\n\n" if bytes_sent else b""
                yield separator + _sse_error("upstream timeout", "timeout", 504)
                return
            except httpx.TransportError as exc:
                if not bytes_sent and attempt < MAX_RETRIES:
                    log.warning(
                        "chat.stream_transport_retry model=%s err=%s attempt=%d",
                        model,
                        exc,
                        attempt + 1,
                    )
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                log.error("chat.stream_transport_error model=%s err=%s", model, exc)
                separator = b"\n\n" if bytes_sent else b""
                yield separator + _sse_error(str(exc), "proxy_error", 502)
                return
            except Exception as e:
                log.error("chat.stream_error model=%s err=%s", model, e)
                separator = b"\n\n" if bytes_sent else b""
                yield separator + _sse_error(str(e), "proxy_error", 502)
                return

        # Exhausted all retries
        log.error("chat.stream_exhausted model=%s attempts=%d", model, MAX_RETRIES + 1)
        yield _sse_error("upstream failed after retries", "proxy_error", 502)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/readyz")
async def readyz():
    """Report readiness only after a bounded authenticated upstream probe."""
    global _readiness_cache
    now = time.monotonic()
    if _readiness_cache and now - _readiness_cache[0] <= READINESS_CACHE_SECONDS:
        return JSONResponse(content=_readiness_cache[2], status_code=_readiness_cache[1])

    token = os.environ.get("NORAX_GATEWAY_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else None
    content: dict[str, Any]
    try:
        await _fetch_upstream_models(_get_client(), headers)
        content = {"ready": True}
        status_code = 200
    except Exception as exc:
        log.warning("readyz.upstream_failed: %s", exc)
        content = {
            "ready": False,
            "reason": "upstream model-catalog probe failed",
        }
        status_code = 503
    _readiness_cache = (now, status_code, content)
    return JSONResponse(content=content, status_code=status_code)


def main():
    port = PORT
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        port = int(sys.argv[idx + 1])

    # Default to loopback for security. Override with NORAX_GATEWAY_BIND=0.0.0.0
    # when running behind a reverse proxy or in a container.
    host = os.environ.get("NORAX_GATEWAY_BIND", "127.0.0.1")
    if "--host" in sys.argv:
        idx = sys.argv.index("--host")
        host = sys.argv[idx + 1]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log.info("norax-gateway-proxy starting on %s:%d → %s", host, port, UPSTREAM)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
