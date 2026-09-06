"""Gateway client — OpenAI-compatible chat-completions client.

The gateway is the **only** place Norax talks to LLM providers. Every
generation attempt here:
- carries a diagnostic request id (without pretending providers deduplicate it)
- is circuit-broken and only retried when failure occurred before a request send
- is cost-attributed via the caller (metadata.user)

The client speaks the OpenAI Chat Completions spec
(`POST {base_url}/chat/completions`). The safe standalone default is the
loopback Ollama OpenAI shim (`http://127.0.0.1:11434/v1`); configured routers
may point explicit clients at other compatible providers.

Model routing (which base_url gets which model) is a concern one layer
up (config / `default_model`). This client just forwards.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import math
import mimetypes
import os
import re
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import urlsplit

import httpx
import ulid

_ulid_new_compat = getattr(ulid, "new", None) or (lambda: str(ulid.ULID()))

from ..observability.circuit import CircuitBreaker, CircuitOpen  # noqa: E402
from ..observability.retry import with_retry  # noqa: E402

_REASONING_OPEN_RE = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
_REASONING_CLOSE_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)
_REASONING_BLOCK_RE = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>\s*", re.IGNORECASE | re.DOTALL
)
_REASONING_TAGS = ("<think>", "<thinking>", "</think>", "</thinking>")
_PRIVATE_REASONING_FIELDS = {"reasoning", "reasoning_content", "thinking"}
_SAFE_LLM_RETRY_DELAYS_MS = (250,)
_MAX_OLLAMA_IMAGES = 8
_MAX_OLLAMA_IMAGE_BYTES = 8 * 1024 * 1024
_MAX_OLLAMA_TOTAL_IMAGE_BYTES = 32 * 1024 * 1024
_MAX_IMAGE_URL_LENGTH = 4096
_MAX_CHAT_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_STREAM_RESPONSE_BYTES = 32 * 1024 * 1024
_MAX_STREAM_LINE_BYTES = 4 * 1024 * 1024
_MAX_UPSTREAM_ERROR_BYTES = 64 * 1024
_MAX_REQUEST_MESSAGES = 4_096
_MAX_REQUEST_TOOLS = 512
_MAX_REQUEST_MESSAGES_BYTES = 64 * 1024 * 1024
_MAX_REQUEST_TOOLS_BYTES = 8 * 1024 * 1024
_MAX_REQUEST_METADATA_BYTES = 1024 * 1024
_MAX_REQUEST_TREE_NODES = 500_000
_MAX_REQUEST_TREE_DEPTH = 32
_MAX_MODEL_ID_CHARS = 512
_BASE64_PAYLOAD_RE = re.compile(r"[A-Za-z0-9+/]*={0,2}\Z")
_ROUTER_PROVIDER_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


async def _read_bounded_body(
    response: httpx.Response,
    *,
    limit: int,
    model: str,
) -> bytes:
    """Read a response body without trusting Content-Length or compression."""
    declared = response.headers.get("content-length")
    if declared:
        try:
            declared_size = int(declared)
        except ValueError as exc:
            raise GatewayUpstreamError(
                502, "upstream returned invalid Content-Length", model
            ) from exc
        if declared_size < 0:
            raise GatewayUpstreamError(502, "upstream returned invalid Content-Length", model)
        if declared_size > limit:
            raise GatewayUpstreamError(
                502,
                f"upstream response exceeds {limit} byte limit",
                model,
            )
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > limit:
            raise GatewayUpstreamError(
                502,
                f"upstream response exceeds {limit} byte limit",
                model,
            )
    return bytes(body)


async def _read_error_preview(response: httpx.Response) -> str:
    """Return a bounded error-body preview without buffering hostile payloads."""
    body = bytearray()
    truncated = False
    async for chunk in response.aiter_bytes():
        remaining = _MAX_UPSTREAM_ERROR_BYTES - len(body)
        if remaining <= 0:
            truncated = True
            break
        body.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True
            break
    text = bytes(body).decode("utf-8", errors="replace")
    return f"{text}…[truncated]" if truncated else text


async def _iter_bounded_lines(
    response: httpx.Response,
    *,
    model: str,
) -> AsyncIterator[str]:
    """Decode bounded SSE/JSONL lines without ``aiter_lines`` over-allocation."""
    total = 0
    buffer = bytearray()
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > _MAX_STREAM_RESPONSE_BYTES:
            raise GatewayUpstreamError(
                502,
                f"upstream stream exceeds {_MAX_STREAM_RESPONSE_BYTES} byte limit",
                model,
            )
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                break
            if newline > _MAX_STREAM_LINE_BYTES:
                raise GatewayUpstreamError(
                    502,
                    f"upstream stream line exceeds {_MAX_STREAM_LINE_BYTES} byte limit",
                    model,
                )
            raw_line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            yield raw_line.rstrip(b"\r").decode("utf-8", errors="replace")
        if len(buffer) > _MAX_STREAM_LINE_BYTES:
            raise GatewayUpstreamError(
                502,
                f"upstream stream line exceeds {_MAX_STREAM_LINE_BYTES} byte limit",
                model,
            )
    if buffer:
        yield bytes(buffer).rstrip(b"\r").decode("utf-8", errors="replace")


def strip_reasoning_blocks(content: str) -> str:
    """Remove complete and unterminated inline reasoning blocks."""
    text = str(content or "")
    if not text:
        return text
    text = _REASONING_BLOCK_RE.sub("", text)
    # An unmatched opening tag means everything after it is still private
    # reasoning. Stray closing tags carry no useful answer content.
    text = re.sub(r"<think(?:ing)?>.*\Z", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = _REASONING_CLOSE_RE.sub("", text)
    return text.strip()


def _sanitize_response_raw(value: Any) -> Any:
    """Copy JSON-like provider data without private reasoning payloads."""
    if isinstance(value, dict):
        return {
            key: _sanitize_response_raw(item)
            for key, item in value.items()
            if str(key).lower() not in _PRIVATE_REASONING_FIELDS
        }
    if isinstance(value, list):
        return [_sanitize_response_raw(item) for item in value]
    if isinstance(value, str) and (
        _REASONING_OPEN_RE.search(value) or _REASONING_CLOSE_RE.search(value)
    ):
        return strip_reasoning_blocks(value)
    return value


def _partial_reasoning_tag_suffix(text: str, *, closing_only: bool = False) -> int:
    lower = text.lower()
    tags = _REASONING_TAGS[2:] if closing_only else _REASONING_TAGS
    max_len = min(len(lower), max(len(tag) for tag in tags) - 1)
    for size in range(max_len, 0, -1):
        suffix = lower[-size:]
        if any(tag.startswith(suffix) for tag in tags):
            return size
    return 0


class ReasoningTagFilter:
    """Incrementally suppress inline reasoning, including split XML tags."""

    def __init__(self, *, expose: bool = False) -> None:
        self.expose = expose
        self._inside = False
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        if self.expose:
            return str(chunk or "")
        data = self._buffer + str(chunk or "")
        self._buffer = ""
        output: list[str] = []

        while data:
            if self._inside:
                closing = _REASONING_CLOSE_RE.search(data)
                if closing is None:
                    suffix_len = _partial_reasoning_tag_suffix(data, closing_only=True)
                    self._buffer = data[-suffix_len:] if suffix_len else ""
                    return "".join(output)
                data = data[closing.end() :]
                self._inside = False
                continue

            opening = _REASONING_OPEN_RE.search(data)
            if opening is not None:
                output.append(_REASONING_CLOSE_RE.sub("", data[: opening.start()]))
                data = data[opening.end() :]
                self._inside = True
                continue

            data = _REASONING_CLOSE_RE.sub("", data)
            suffix_len = _partial_reasoning_tag_suffix(data)
            if suffix_len:
                output.append(data[:-suffix_len])
                self._buffer = data[-suffix_len:]
            else:
                output.append(data)
            break

        return "".join(output)

    def finish(self) -> str:
        """Finish a stream, dropping a partial tag or unfinished reasoning."""
        if self.expose:
            buffered, self._buffer = self._buffer, ""
            return buffered
        self._buffer = ""
        return ""


class GatewayUpstreamError(Exception):
    """Raised on HTTP 4xx/5xx from the upstream LLM gateway.

    Carries the parsed upstream error message so the runtime can surface
    a useful user-visible hint (instead of a bare "HTTPStatusError").
    """

    def __init__(self, status: int, upstream_message: str, model: str):
        self.status = status
        self.upstream_message = upstream_message
        self.model = model
        super().__init__(f"gateway {status} [{model}]: {upstream_message}")


class SpendGuardTripped(Exception):
    """Raised when the global LLM call-rate breaker trips.

    Protects paid subscriptions from runaway loops: no matter what the agent
    loop, orchestrator, or sub-agents do, upstream spend is rate-bounded here
    at the single choke point every LLM call passes through.
    """

    def __init__(self, window: str, count: int, limit: int):
        self.window = window
        self.count = count
        self.limit = limit
        super().__init__(
            f"spend guard tripped: {count} LLM calls in the last {window} "
            f"(limit {limit}). Pausing upstream calls to protect subscriptions."
        )


class SpendGuard:
    """Global sliding-window rate breaker for upstream LLM calls.

    Env config:
      NORAX_LLM_MAX_CALLS_PER_MIN   default 0 (disabled)
      NORAX_LLM_MAX_CALLS_PER_HOUR  default 0 (disabled)

    The breaker is operator opt-in. A hidden process-wide default can abort a
    legitimate long-running or parallel task, including calls to local models,
    even when no monetary spend is involved. Runtime round/time limits remain
    the default runaway protection.
    """

    def __init__(self) -> None:
        # Limits are resolved from the environment on each check (see the
        # ``per_min`` / ``per_hour`` properties) so a process-wide singleton
        # created at import time never bakes in a stale/production limit.
        # Tests may pin a limit by assigning the property directly.
        self._per_min_override: int | None = None
        self._per_hour_override: int | None = None
        self._minute_stamps: deque[float] = deque()
        self._hour_stamps: deque[float] = deque()

    @property
    def per_min(self) -> int:
        if self._per_min_override is not None:
            return self._per_min_override
        return self._env_int("NORAX_LLM_MAX_CALLS_PER_MIN", 0)

    @per_min.setter
    def per_min(self, value: int) -> None:
        self._per_min_override = value

    @property
    def per_hour(self) -> int:
        if self._per_hour_override is not None:
            return self._per_hour_override
        return self._env_int("NORAX_LLM_MAX_CALLS_PER_HOUR", 0)

    @per_hour.setter
    def per_hour(self, value: int) -> None:
        self._per_hour_override = value

    @staticmethod
    def _env_int(key: str, default: int) -> int:
        raw = os.environ.get(key, "")
        if raw.strip() in ("0", "-1", "off", "disabled"):
            return 0
        try:
            value = int(raw)
        except (ValueError, TypeError):
            return default
        return value if value > 0 else 0

    def record_and_check(self) -> None:
        """Record one upstream call; raise SpendGuardTripped on breach.

        When per_min or per_hour is 0, that window's check is disabled.
        """
        if self.per_min <= 0 and self.per_hour <= 0:
            return
        now = time.monotonic()
        if self.per_min > 0:
            while self._minute_stamps and now - self._minute_stamps[0] > 60.0:
                self._minute_stamps.popleft()
            if len(self._minute_stamps) >= self.per_min:
                log.error(
                    "spend_guard.tripped window=1m count=%d limit=%d",
                    len(self._minute_stamps),
                    self.per_min,
                )
                raise SpendGuardTripped("minute", len(self._minute_stamps), self.per_min)
        if self.per_hour > 0:
            while self._hour_stamps and now - self._hour_stamps[0] > 3600.0:
                self._hour_stamps.popleft()
            if len(self._hour_stamps) >= self.per_hour:
                log.error(
                    "spend_guard.tripped window=1h count=%d limit=%d",
                    len(self._hour_stamps),
                    self.per_hour,
                )
                raise SpendGuardTripped("hour", len(self._hour_stamps), self.per_hour)
        if self.per_min > 0:
            self._minute_stamps.append(now)
        if self.per_hour > 0:
            self._hour_stamps.append(now)


# Process-wide guard, enforced immediately before every GatewayClient inference
# transport attempt (including direct clients, safe retries, and continuations).
_SPEND_GUARD = SpendGuard()


def _extract_upstream_message(body_text: str) -> str:
    """Best-effort parse of OpenAI-style error envelopes."""
    try:
        obj = json.loads(body_text)
    except Exception:  # noqa: BLE001
        return (body_text or "").strip()[:300]
    err = obj.get("error") if isinstance(obj, dict) else None
    if isinstance(err, dict):
        msg = err.get("message")
        if isinstance(msg, str):
            # Some proxies wrap a stringified JSON inside .error.message
            try:
                inner = json.loads(msg)
                if isinstance(inner, dict):
                    inner_err = inner.get("error", {})
                    if isinstance(inner_err, dict) and inner_err.get("message"):
                        prefix = f"[{inner_err.get('type', '')}]" if inner_err.get("type") else ""
                        return f"{prefix} {inner_err['message']}".strip()[:300]
            except Exception:  # noqa: BLE001
                pass
            # Prepend the error type (e.g. "usage_limit_reached") so callers
            # like _is_usage_limit can match on it.
            etype = err.get("type") or ""
            if etype:
                return f"[{etype}] {msg}".strip()[:300]
            return msg[:300]
    return body_text[:300]


def _error_message_from_obj(obj: Any) -> str | None:
    """Return a provider error message from common JSON envelopes."""
    if not isinstance(obj, dict):
        return None
    err = obj.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or err.get("error") or err.get("type")
        if msg:
            return str(msg)[:300]
    if isinstance(err, str):
        return err[:300]
    msg = obj.get("message")
    if isinstance(msg, str) and obj.get("type") == "error":
        return msg[:300]
    return None


def _status_from_error_envelope(obj: Any) -> int:
    """Extract an HTTP-like status code from a streaming error envelope.

    Some upstream proxies emit errors as SSE data
    events with an ``error`` object.  The error object includes ``code``
    (e.g. ``timeout``) and ``type`` (e.g. ``timeout_error``).  Map
    these to the correct HTTP status so retry/failover logic can distinguish
    timeouts (504) from genuine bad-gateway errors (502).
    """
    if not isinstance(obj, dict):
        return 502
    err = obj.get("error")
    if not isinstance(err, dict):
        return 502
    code = str(err.get("code") or "").lower()
    etype = str(err.get("type") or "").lower()
    if "timeout" in code or "timeout" in etype:
        return 504
    if "network" in code or "network" in etype:
        return 503
    # If the envelope carries an explicit numeric status, use it
    raw_status = err.get("status") or err.get("status_code")
    if isinstance(raw_status, int) and 400 <= raw_status < 600:
        return raw_status
    return 502


def _raise_if_error_envelope(obj: Any, model: str) -> None:
    msg = _error_message_from_obj(obj)
    if msg:
        status = _status_from_error_envelope(obj)
        raise GatewayUpstreamError(status, msg, model)


def _counts_as_upstream_failure(status: int) -> bool:
    """Return whether an HTTP status is evidence of provider unavailability."""
    return status >= 500 or status in (408, 409, 425, 429)


def _ensure_tool_call_id(msg: dict, *, index: int) -> dict:
    """Guarantee a unique non-empty tool_call_id on tool-result messages."""
    if msg.get("role") != "tool":
        return msg
    tid = str(msg.get("tool_call_id") or "").strip()
    if not tid or tid == "tool":
        tid = f"call_sanitized_{index:04d}"
        msg = dict(msg)
        msg["tool_call_id"] = tid
    return msg


def _as_uuid_session_id(value: Any) -> str:
    """Return a Claude Code-compatible UUID session id."""
    if value:
        s = str(value)
        try:
            return str(uuid.UUID(s))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"norax-session:{s}"))
    return str(uuid.uuid4())


log = logging.getLogger("norax.gateway_client")


@dataclass
class GatewayRequest:
    model: str
    messages: list[dict]
    tools: list[dict] | None = None
    tool_choice: str | dict | None = (
        None  # "auto" | "none" | "required" | {"type":"function","function":{"name":...}}
    )
    max_tokens: int | None = None
    temperature: float | None = None
    metadata: dict | None = None


@dataclass
class GatewayResponse:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    request_id: str = ""
    model: str = ""
    usage: dict = field(default_factory=dict)  # {"input_tokens": int, "output_tokens": int}
    raw: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)  # done_reason, continuation info


@dataclass
class StreamEvent:
    """One event in a streaming chat response.

    kind="delta": incremental token text and/or partial tool-call slices.
    kind="final": the assembled GatewayResponse (set in `.response`).
    """

    kind: str
    text: str = ""
    tool_calls_partial: list[dict] | None = None
    response: GatewayResponse | None = None


def _bounded_json_tree(value: Any, *, max_bytes: int, label: str) -> None:
    """Validate a JSON-like tree with allocation, node, and depth ceilings."""
    stack: list[tuple[Any, int, bool]] = [(value, 0, False)]
    active_containers: set[int] = set()
    approximate_bytes = 0
    nodes = 0
    while stack:
        item, depth, leaving = stack.pop()
        if leaving:
            active_containers.discard(id(item))
            continue
        nodes += 1
        if nodes > _MAX_REQUEST_TREE_NODES:
            raise ValueError(f"{label} exceeds its structural node limit")
        if depth > _MAX_REQUEST_TREE_DEPTH:
            raise ValueError(f"{label} exceeds its nesting limit")
        if isinstance(item, str):
            approximate_bytes += len(item.encode("utf-8")) + 2
        elif item is None or isinstance(item, bool):
            approximate_bytes += 5
        elif isinstance(item, int):
            approximate_bytes += len(str(item))
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{label} contains a non-finite number")
            approximate_bytes += 32
        elif isinstance(item, dict | list):
            identity = id(item)
            if identity in active_containers:
                raise ValueError(f"{label} contains a reference cycle")
            active_containers.add(identity)
            stack.append((item, depth, True))
            if isinstance(item, dict):
                if any(not isinstance(key, str) for key in item):
                    raise TypeError(f"{label} object keys must be text")
                approximate_bytes += len(item) * 2
                for key, child in item.items():
                    approximate_bytes += len(key.encode("utf-8")) + 3
                    stack.append((child, depth + 1, False))
            else:
                approximate_bytes += len(item) + 2
                for child in item:
                    stack.append((child, depth + 1, False))
        else:
            raise TypeError(f"{label} contains non-JSON data: {type(item).__name__}")
        if approximate_bytes > max_bytes:
            raise ValueError(f"{label} exceeds its byte limit")


def _validate_gateway_request(req: GatewayRequest) -> None:
    if not isinstance(req, GatewayRequest):
        raise TypeError("gateway request must be a GatewayRequest")
    if (
        not isinstance(req.model, str)
        or not req.model
        or len(req.model) > _MAX_MODEL_ID_CHARS
        or any(character.isspace() or not character.isprintable() for character in req.model)
    ):
        raise ValueError(f"model must contain 1-{_MAX_MODEL_ID_CHARS} printable characters")
    if not isinstance(req.messages, list) or len(req.messages) > _MAX_REQUEST_MESSAGES:
        raise ValueError(f"messages must be a list of at most {_MAX_REQUEST_MESSAGES} entries")
    if any(not isinstance(message, dict) for message in req.messages):
        raise TypeError("messages entries must be objects")
    for message in req.messages:
        role = message.get("role")
        if role is not None and (
            not isinstance(role, str) or not role or len(role) > 64 or not role.isprintable()
        ):
            raise ValueError("message roles must be bounded printable text")
    _bounded_json_tree(
        req.messages,
        max_bytes=_MAX_REQUEST_MESSAGES_BYTES,
        label="messages",
    )

    if req.tools is not None:
        if not isinstance(req.tools, list) or len(req.tools) > _MAX_REQUEST_TOOLS:
            raise ValueError(f"tools must be a list of at most {_MAX_REQUEST_TOOLS} entries")
        if any(not isinstance(tool, dict) for tool in req.tools):
            raise TypeError("tools entries must be objects")
        _bounded_json_tree(req.tools, max_bytes=_MAX_REQUEST_TOOLS_BYTES, label="tools")
    if req.tool_choice is not None:
        if not isinstance(req.tool_choice, str | dict):
            raise TypeError("tool_choice must be text or an object")
        if isinstance(req.tool_choice, str) and (
            not req.tool_choice or len(req.tool_choice) > 128 or not req.tool_choice.isprintable()
        ):
            raise ValueError("tool_choice text is invalid")
        _bounded_json_tree(req.tool_choice, max_bytes=64 * 1024, label="tool_choice")
    if req.max_tokens is not None and (
        isinstance(req.max_tokens, bool)
        or not isinstance(req.max_tokens, int)
        or not 1 <= req.max_tokens <= 10_000_000
    ):
        raise ValueError("max_tokens must be between 1 and 10000000")
    if req.temperature is not None and (
        isinstance(req.temperature, bool)
        or not isinstance(req.temperature, int | float)
        or not math.isfinite(float(req.temperature))
        or not 0 <= float(req.temperature) <= 2
    ):
        raise ValueError("temperature must be finite and between 0 and 2")
    if req.metadata is not None:
        if not isinstance(req.metadata, dict):
            raise TypeError("metadata must be an object")
        _bounded_json_tree(
            req.metadata,
            max_bytes=_MAX_REQUEST_METADATA_BYTES,
            label="metadata",
        )
        for flag in ("reasoning_output", "allow_text_tool_calls"):
            if flag in req.metadata and type(req.metadata[flag]) is not bool:
                raise TypeError(f"metadata.{flag} must be a boolean")
        effort = req.metadata.get("reasoning_effort")
        if effort is not None and (not isinstance(effort, str) or not effort or len(effort) > 32):
            raise ValueError("metadata.reasoning_effort must be bounded text")
        continuation = req.metadata.get("_ollama_continuation_attempt")
        if continuation is not None and (
            isinstance(continuation, bool)
            or not isinstance(continuation, int)
            or not 0 <= continuation <= 16
        ):
            raise ValueError("metadata continuation attempt is invalid")


def get_gateway() -> GatewayClient:
    """Return a one-off client for the explicitly configured gateway.

    ``NORAX_GATEWAY_URL`` remains an explicit override. Without one, auxiliary
    callers use the same direct Ollama endpoint as the shipped runtime config;
    they never fall back to an unrelated remote proxy.
    """
    import os

    base = os.environ.get("NORAX_GATEWAY_URL", "http://127.0.0.1:11434/v1")
    return GatewayClient(base_url=base)


class GatewayClient:
    """OpenAI-compatible chat-completions client.

    `base_url` must be the API root that serves `/chat/completions`
    (typically ending in `/v1`). `api_key` is sent as `Authorization:
    Bearer <key>`; most OpenAI-compatible endpoints accept that.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        *,
        timeout: float = 120.0,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        stream_required: bool = False,
        model_prefix: str | None = None,
        provider_kind: str | None = None,
        chat_path: str = "/chat/completions",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.stream_required = stream_required
        self.provider_kind = provider_kind
        self.chat_path = chat_path if chat_path.startswith("/") else f"/{chat_path}"
        # Prefix on routed model ids to strip before forwarding upstream.
        # Defaults to the provider name (set by router); overridable via
        # config when the routed prefix differs from the provider name.
        self.model_prefix = model_prefix
        headers: dict[str, str] = {}
        if api_key:
            if (provider_kind or "").lower() == "anthropic":
                headers["authorization"] = f"Bearer {api_key}"
                headers["anthropic-version"] = "2023-06-01"
                headers["anthropic-beta"] = (
                    "claude-code-20250219,fine-grained-tool-streaming-2025-05-14"
                )
                headers["user-agent"] = "claude-cli/2.1.76 (external, cli)"
                headers["x-app"] = "cli"
                headers["x-device-id"] = str(uuid.uuid4())
            else:
                headers["authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers=headers,
            limits=httpx.Limits(
                max_connections=64,
                max_keepalive_connections=16,
                keepalive_expiry=30.0,
            ),
            http2=False,
        )
        kind = self._provider_kind()
        if kind == "ollama":
            self._breaker = CircuitBreaker("gateway:ollama", failure_threshold=20, open_seconds=5.0)
        else:
            self._breaker = CircuitBreaker("gateway", failure_threshold=5, open_seconds=30.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _begin_upstream_call(self) -> None:
        """Enter the circuit and account for one real inference attempt.

        The spend guard lives at this transport boundary rather than at the
        router so direct clients, safe retries, and Ollama continuations cannot
        evade it. If the guard rejects a half-open probe, release that circuit
        slot because no network request was sent.
        """
        self._breaker.before_call()
        try:
            _SPEND_GUARD.record_and_check()
        except BaseException:
            self._breaker.on_abandoned()
            raise

    async def transport_probe(
        self,
        model: str | None = None,
        *,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Verify the configured API transport without generating tokens.

        A successful result proves that the model API is reachable and, where
        applicable, that the configured credentials can access its model
        catalog.  It intentionally does *not* claim that an inference was
        completed; callers surface that distinction as transport-only
        readiness evidence.
        """
        del model
        kind = self._provider_kind()
        if kind == "ollama":
            root = self.base_url.removesuffix("/v1").rstrip("/")
            endpoint = f"{root}/api/tags"
        else:
            endpoint = f"{self.base_url.rstrip('/')}/models"

        started = time.monotonic()
        try:
            response = await self._client.get(endpoint, timeout=timeout)
        except asyncio.CancelledError:
            raise
        except httpx.HTTPError as exc:
            return {
                "ok": False,
                "kind": kind,
                "endpoint": endpoint,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }

        # Fall back to OpenAI-style /v1/models if /api/tags is not served
        # (e.g. llama-server with DFlash2 does not expose Ollama's /api/tags).
        if response.status_code == 404 and kind == "ollama":
            endpoint = f"{self.base_url.rstrip('/')}/models"
            try:
                response = await self._client.get(endpoint, timeout=timeout)
            except asyncio.CancelledError:
                raise
            except httpx.HTTPError as exc:
                return {
                    "ok": False,
                    "kind": kind,
                    "endpoint": endpoint,
                    "latency_ms": round((time.monotonic() - started) * 1000, 1),
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }

        result: dict[str, Any] = {
            "ok": 200 <= response.status_code < 300,
            "kind": kind,
            "endpoint": endpoint,
            "status_code": response.status_code,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
        }
        if not result["ok"]:
            result["error"] = f"model catalog returned HTTP {response.status_code}"
        return result

    async def fetch_model_ids(
        self,
        *,
        max_response_bytes: int = 2 * 1024 * 1024,
        max_models: int = 100,
    ) -> list[Any]:
        """Fetch an OpenAI-compatible model catalog with hard allocation caps."""
        if not 1 <= max_response_bytes <= 16 * 1024 * 1024:
            raise ValueError("max_response_bytes must be between 1 byte and 16 MiB")
        if not 1 <= max_models <= 10_000:
            raise ValueError("max_models must be between 1 and 10000")
        raw = bytearray()
        async with self._client.stream("GET", f"{self.base_url}/models") as response:
            response.raise_for_status()
            declared_length = response.headers.get("content-length")
            if declared_length:
                try:
                    parsed_length = int(declared_length)
                except ValueError as exc:
                    raise ValueError("provider returned an invalid Content-Length") from exc
                if parsed_length < 0:
                    raise ValueError("provider returned an invalid Content-Length")
                if parsed_length > max_response_bytes:
                    raise ValueError("provider model catalog exceeds the response-size limit")
            async for chunk in response.aiter_bytes():
                raw.extend(chunk)
                if len(raw) > max_response_bytes:
                    raise ValueError("provider model catalog exceeds the response-size limit")
        try:
            payload = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("provider /models endpoint returned invalid JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("provider /models endpoint must return a data array")
        models: list[Any] = []
        for item in payload["data"]:
            if not isinstance(item, dict) or "id" not in item or item["id"] in (None, ""):
                continue
            models.append(item["id"])
            if len(models) >= max_models:
                break
        return models

    @property
    def kind(self) -> str:
        """Public accessor for the effective provider kind."""
        return self._provider_kind()

    # ---- Provider detection helpers ----

    def _provider_kind(self) -> str:
        """Classify this client into a provider kind for format decisions."""
        if self.provider_kind:
            return self.provider_kind
        url = self.base_url
        if ":11434" in url:
            return "ollama"
        if ":4146" in url or ":4147" in url:
            return "codex_direct"
        return "openai"

    def _is_llama_cpp_endpoint(self) -> bool:
        """True when this client targets a llama.cpp server (llama-server's
        OpenAI-compatible endpoint), where thinking is controlled via
        chat_template_kwargs rather than reasoning_effort."""
        kind = self.provider_kind or ""
        if kind == "llama_cpp":
            return True
        url = self.base_url or ""
        # 11435 is the direct local server; 19136 is Norax's transparent
        # Ollama-compatible relay to that same llama.cpp server.
        try:
            parsed = urlsplit(url)
            return parsed.scheme in {"http", "https"} and parsed.port in {11435, 19136}
        except ValueError:
            return False

    def _apply_llama_cpp_thinking(self, payload: dict[str, Any], effort: object) -> None:
        """Translate Norax effort names to Qwen's llama.cpp template controls."""
        if not self._is_llama_cpp_endpoint() or effort is None:
            return
        normalized = str(effort).strip().lower()
        kwargs = payload.setdefault("chat_template_kwargs", {})
        if normalized in {"off", "none", "minimal", "false", "no"}:
            kwargs["enable_thinking"] = False
        else:
            # Qwen's shipped llama.cpp template supports low, medium, xhigh.
            # Norax exposes high/max/ultra too, so map those to its strongest
            # native level rather than letting the template silently default.
            mapped = {
                "low": "low",
                "medium": "medium",
                "high": "xhigh",
                "xhigh": "xhigh",
                "max": "xhigh",
                "ultra": "xhigh",
            }.get(normalized, "xhigh")
            kwargs["enable_thinking"] = True
            kwargs["reasoning_effort"] = mapped
        payload.pop("reasoning_effort", None)

    def provider_kind_for_model(self, model: str) -> str:
        """Return the request transport kind for a model.

        A single client has one transport; the argument keeps this API
        compatible with GatewayRouter's model-aware implementation.
        """
        del model
        return self._provider_kind()

    # ---- Per-provider payload/tool sanitization ----

    def _format_tools_for_provider(
        self,
        tools: list[dict] | None,
        *,
        model: str = "",
    ) -> list[dict] | None:
        """Normalize tool schemas for the target provider.

        Ollama's OpenAI shim is strict: no optional defaults/metadata,
        JSON-schema object parameters only, and extra wrapper fields can
        trigger "tooling" errors. Keep the standard OpenAI function shape,
        but aggressively trim it.
        """
        if not tools:
            return tools
        kind = self._provider_kind()
        if kind != "ollama":
            return tools

        def clean_schema(obj, *, root: bool = False):
            # Ollama validates function parameters much more narrowly than
            # OpenAI. Keep only the subset it reliably accepts. In practice the
            # killer has been `additionalProperties: false` and fancy schema
            # keys; Ollama surfaces that as "invalid tool call arguments" even
            # on trivial prompts.
            if isinstance(obj, dict):
                banned = {
                    "default",
                    "examples",
                    "title",
                    "$schema",
                    "$defs",
                    "oneOf",
                    "anyOf",
                    "allOf",
                    "nullable",
                    "additionalProperties",
                    "patternProperties",
                    "minLength",
                    "maxLength",
                    "minimum",
                    "maximum",
                    "exclusiveMinimum",
                    "exclusiveMaximum",
                    "format",
                }
                out_d = {k: clean_schema(v) for k, v in obj.items() if k not in banned}
                if root:
                    out_d["type"] = "object"
                    out_d.setdefault("properties", {})
                    if not isinstance(out_d.get("properties"), dict):
                        out_d["properties"] = {}
                return out_d
            if isinstance(obj, list):
                return [clean_schema(v) for v in obj]
            return obj

        from .ollama_profiles import is_weak_ollama_model

        weak_local_model = is_weak_ollama_model(model)
        out: list[dict] = []
        for tool in tools:
            fn = dict(tool.get("function") or {})
            params = clean_schema(
                fn.get("parameters") or {"type": "object", "properties": {}}, root=True
            )
            if not isinstance(params, dict) or params.get("type") != "object":
                params = {"type": "object", "properties": {}}
            params.setdefault("properties", {})
            # Only weak local models receive permissive required fields.
            # Ollama is also a transport for strong cloud models; stripping
            # `required` provider-wide made their otherwise precise native tool
            # calls omit essential arguments.
            if weak_local_model:
                params.pop("required", None)
            name = str(fn.get("name") or tool.get("name") or "tool")
            name = name.replace("-", "_").replace(".", "_").replace(" ", "_")[:64]
            desc = str(fn.get("description") or "")[:512]
            fn = {
                "name": name,
                "description": desc,
                "parameters": params,
            }
            out.append({"type": "function", "function": fn})
        return out

    @staticmethod
    def _normalize_tool_calls_from_provider(tool_calls: list[dict] | None) -> list[dict]:
        """Normalize provider-returned tool calls into OpenAI shape.

        Ollama-compatible servers may return function.arguments as a dict or
        omit id/type. The runtime dispatcher expects stable OpenAI-style calls.
        """
        if not tool_calls:
            return []
        fixed: list[dict] = []
        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            tc = dict(tc)
            fn = tc.get("function")
            if not isinstance(fn, dict):
                fn = {}
            else:
                fn = dict(fn)
            args = fn.get("arguments", "{}")
            if args is None or args == "":
                args = "{}"
            elif not isinstance(args, str):
                args = json.dumps(args, default=str)
            fn["arguments"] = args
            tc["function"] = fn
            tc["type"] = tc.get("type") or "function"
            tc["id"] = tc.get("id") or f"call_{i}"
            fixed.append(tc)
        return fixed

    def _sanitize_payload(self, payload: dict) -> dict:
        """Strip/top-level fields and adapt payloads per provider."""
        kind = self._provider_kind()
        OPENAI_ONLY = {"reasoning_effort", "reasoning"}
        PROXY_STRIP = {"metadata", "user"}
        # Ollama native /api/chat understands reasoning via `think`; keep it
        # until _to_ollama_native_payload() maps it, but strip proxy-only
        # metadata/user fields.
        if kind == "ollama":
            for k in PROXY_STRIP:
                payload.pop(k, None)
        elif kind == "codex_direct":
            # Local Codex proxies on :4146/:4147 use Codex CLI OAuth.
            # They accept standard OpenAI fields but reject temperature
            # with "Unsupported parameter: temperature".
            for k in PROXY_STRIP:
                payload.pop(k, None)
            payload.pop("temperature", None)
        elif kind == "openrouter":
            # OpenRouter is OpenAI-compatible but rejects proxy-only fields.
            # Keep reasoning_effort/reasoning — OpenRouter supports them.
            for k in PROXY_STRIP:
                payload.pop(k, None)
        elif kind == "anthropic":
            for k in OPENAI_ONLY | {"user"}:
                payload.pop(k, None)
        if kind == "ollama":
            if payload.get("tools"):
                payload["tools"] = self._format_tools_for_provider(
                    payload.get("tools"),
                    model=str(payload.get("model") or ""),
                )
                # Ollama works best with tool_choice omitted unless explicitly
                # forcing none/a named tool; auto can produce shim errors.
                if payload.get("tool_choice") == "auto":
                    payload.pop("tool_choice", None)
            else:
                payload.pop("tool_choice", None)
        return payload

    # ---- Per-provider message normalization ----

    @staticmethod
    def _extract_invoke_xml_tool_calls(content: str) -> tuple[list[dict], str]:
        """Extract <invoke> XML-style tool calls (Ornith/Qwen fine-tune format).

        Parses blocks like:
            <invoke name="read">
            <parameter name="path">/foo/bar.py</parameter>
            </invoke>
        Returns (tool_calls, cleaned_content) in OpenAI format.
        """
        if not content or "<invoke" not in content:
            return [], content
        calls: list[dict] = []
        remove_spans: list[tuple[int, int]] = []
        for m in re.finditer(
            r'<invoke\s+name=["\'](?P<name>[\w_]+)["\']>(.*?)</invoke>',
            content,
            re.S,
        ):
            name = m.group("name")
            body = m.group(2)
            args: dict[str, str] = {}
            for pm in re.finditer(
                r'<parameter\s+name=["\'](?P<key>[\w_]+)["\']>(?P<val>.*?)</parameter>',
                body,
                re.S,
            ):
                args[pm.group("key")] = pm.group("val").strip()
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            )
            remove_spans.append((m.start(), m.end()))
        if not calls:
            return [], content
        # Remove the invoke blocks from content
        cleaned_parts: list[str] = []
        pos = 0
        for s, e in sorted(remove_spans):
            cleaned_parts.append(content[pos:s])
            pos = e
        cleaned_parts.append(content[pos:])
        cleaned = "".join(cleaned_parts).strip()
        return calls, cleaned

    @staticmethod
    def _extract_qwen_function_tool_calls(content: str) -> tuple[list[dict], str]:
        """Extract Qwen3-Coder style tool calls from content.

        Parses blocks like:
            <function=read>
            <parameter=path>
            /etc/hostname
            </parameter>
            </function>
        These may be wrapped in  ...  or  ... </tool>.
        Returns (tool_calls, cleaned_content) in OpenAI format.
        """
        if not content or "<function=" not in content:
            return [], content
        calls: list[dict] = []
        remove_spans: list[tuple[int, int]] = []
        # Match <function=name>...</function> blocks
        for m in re.finditer(
            r"<function=(?P<name>[\w_]+)>(?P<body>.*?)</function>",
            content,
            re.S,
        ):
            name = m.group("name")
            body = m.group("body")
            args: dict[str, str] = {}
            for pm in re.finditer(
                r"<parameter=(?P<key>[\w_]+)>(?P<val>.*?)</parameter>",
                body,
                re.S,
            ):
                val = pm.group("val").strip()
                # Strip single leading/trailing newline (qwen3_coder format)
                if val.startswith("\n"):
                    val = val[1:]
                if val.endswith("\n"):
                    val = val[:-1]
                args[pm.group("key")] = val.strip()
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            )
            remove_spans.append((m.start(), m.end()))
        if not calls:
            return [], content
        # Remove the function blocks AND any wrapper tokens (..., </tool>)
        cleaned_parts: list[str] = []
        pos = 0
        for s, e in sorted(remove_spans):
            cleaned_parts.append(content[pos:s])
            pos = e
        cleaned_parts.append(content[pos:])
        cleaned = "".join(cleaned_parts)
        # Strip leftover wrapper tokens
        for wrapper in ("<tool_call>", "</tool_call>", "<tool>", "</tool>"):
            cleaned = cleaned.replace(wrapper, "")
        cleaned = cleaned.strip()
        return calls, cleaned

    def _tools_supported_for_request(self, req: GatewayRequest) -> bool:
        """Provider/model gate for native tool-calling.

        Some OpenAI-compatible providers expose chat but not native tool calls
        for every model. Sending `tools` to those models can 400 even for plain
        text prompts, so known-bad families run text-only.
        """
        return True

    def _text_tool_fallback_names(self, req: GatewayRequest) -> set[str] | None:
        """Return declared names only for routes that require text tool recovery."""
        if not req.tools:
            return None
        metadata = req.metadata or {}
        explicitly_enabled = metadata.get("allow_text_tool_calls") is True
        if not explicitly_enabled:
            return None
        names = {
            str((tool.get("function") or {}).get("name") or "")
            for tool in req.tools
            if isinstance(tool, dict)
        }
        names.discard("")
        return names or None

    def _format_tools_for_anthropic(self, tools: list[dict] | None) -> list[dict] | None:
        if not tools:
            return None
        out: list[dict] = []
        for tool in tools:
            fn = tool.get("function") or {}
            name = str(fn.get("name") or tool.get("name") or "tool")
            out.append(
                {
                    "name": name,
                    "description": str(fn.get("description") or ""),
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
                }
            )
        return out

    def _messages_to_anthropic(self, messages: list[dict]) -> tuple[list[dict] | None, list[dict]]:
        system_parts: list[str] = []
        out: list[dict] = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content") or ""
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                else:
                    system_parts.append(json.dumps(content, default=str))
                continue
            if role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.get("tool_call_id")
                                or m.get("id")
                                or f"call_{len(out):04d}",
                                "content": content
                                if isinstance(content, str)
                                else json.dumps(content, default=str),
                            }
                        ],
                    }
                )
                continue
            if role not in ("user", "assistant"):
                role = "user"
            blocks: list[dict] = []
            if isinstance(content, str):
                if content:
                    blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        blocks.append({"type": "text", "text": str(part.get("text") or "")})
                    elif isinstance(part, dict) and part.get("type") == "image_url":
                        # Claude Code APIs do not accept OpenAI image_url
                        # blocks directly; retain a textual placeholder rather
                        # than sending an invalid content block.
                        blocks.append({"type": "text", "text": "[image omitted]"})
                    else:
                        blocks.append({"type": "text", "text": json.dumps(part, default=str)})
            else:
                blocks.append({"type": "text", "text": json.dumps(content, default=str)})
            if role == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args else {}
                        except Exception:  # noqa: BLE001
                            args = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id") or f"call_{len(blocks)}",
                            "name": fn.get("name") or "tool",
                            "input": args if isinstance(args, dict) else {},
                        }
                    )
            out.append({"role": role, "content": blocks or [{"type": "text", "text": ""}]})
        return ([{"type": "text", "text": p} for p in system_parts] if system_parts else None), out

    def _to_anthropic_payload(self, payload: dict) -> dict:
        system, messages = self._messages_to_anthropic(payload.get("messages") or [])
        out: dict[str, Any] = {
            "model": payload.get("model"),
            "messages": messages,
            "max_tokens": payload.get("max_tokens") or 4096,
        }
        metadata = payload.get("metadata") or {}
        if isinstance(metadata, dict):
            session_id = metadata.get("session_id") or metadata.get("norax_request_id")
            out["session_id"] = _as_uuid_session_id(session_id)
            out["metadata"] = metadata
        else:
            out["session_id"] = _as_uuid_session_id(None)
        if system:
            out["system"] = system
        if payload.get("temperature") is not None:
            out["temperature"] = payload.get("temperature")
        if payload.get("stream"):
            out["stream"] = True
        tools = self._format_tools_for_anthropic(payload.get("tools"))
        if tools:
            out["tools"] = tools
        return out

    def _to_ollama_native_payload(
        self,
        payload: dict,
        *,
        metadata: dict | None = None,
    ) -> dict:
        """Convert OpenAI chat-completions payload to Ollama /api/chat shape.

        Ollama's native tool API uses `options`, `think`, `tool_name` on tool
        result messages, and function.arguments as objects in assistant
        tool_calls. This avoids OpenAI-shim argument validation bugs.
        """
        from .ollama_profiles import (
            profile_options,
            reasoning_effort_for_profile,
            resolve_profile,
            tools_for_profile,
        )

        model = str(payload.get("model") or "")
        profile = resolve_profile(model)
        out = {
            "model": model,
            "messages": self._messages_to_ollama_native(payload.get("messages") or [], model=model),
            "stream": bool(payload.get("stream", False)),
            "think": profile.think,
        }
        if profile.keep_alive is not None:
            out["keep_alive"] = profile.keep_alive
        if profile.clear_thinking is not None:
            out["clear_thinking"] = profile.clear_thinking
        if payload.get("tools"):
            out["tools"] = tools_for_profile(
                self._format_tools_for_provider(payload.get("tools"), model=model),
                enabled=profile.tools_enabled,
            )
        has_tools = bool(out.get("tools"))
        opts = profile_options(profile, has_tools=has_tools)
        if payload.get("temperature") is not None:
            opts["temperature"] = payload.get("temperature")
        if payload.get("max_tokens") is not None:
            opts["num_predict"] = payload.get("max_tokens")
        # Provider sanitization intentionally strips proxy-only metadata before
        # transport. Internal Ollama controls are passed separately so they can
        # shape the native payload without leaking arbitrary metadata upstream.
        meta = metadata if metadata is not None else payload.get("metadata") or {}
        if meta.get("ollama_think") is not None:
            out["think"] = bool(meta["ollama_think"])
        if opts:
            out["options"] = opts
        effort = payload.get("reasoning_effort") or payload.get("reasoning")
        if effort:
            out["think"] = False if effort in ("none", "minimal", "off") else True
        model_effort = reasoning_effort_for_profile(profile, effort)
        if model_effort:
            # Ollama's /api/chat accepts think as: true, false, "low", "medium",
            # "high".  Values like "xhigh"/"max" are rejected (400/500), so fall
            # back to the boolean true (model default thinking level) for any
            # effort string outside the known-good set.
            _OLLAMA_THINK_STRINGS = {"low", "medium", "high"}
            if isinstance(model_effort, str) and model_effort in _OLLAMA_THINK_STRINGS:
                out["think"] = model_effort
            else:
                out["think"] = True
            out["reasoning_effort"] = model_effort
        grammar = meta.get("ollama_grammar") or meta.get("grammar")
        if grammar and meta.get("ollama_use_grammar") is True:
            out["format"] = grammar
        return out

    def _messages_to_ollama_native(self, messages: list[dict], *, model: str = "") -> list[dict]:
        out = []
        for m in messages:
            msg = dict(m)
            content = msg.get("content")
            if isinstance(content, list):
                text_parts: list[str] = []
                images = list(msg.get("images") or [])
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(str(part.get("text") or ""))
                    elif isinstance(part, dict) and part.get("type") == "image_url":
                        image_spec = part.get("image_url")
                        url = (
                            str(image_spec.get("url") or "") if isinstance(image_spec, dict) else ""
                        )
                        _mime, encoded, _size = self._validated_image_data_url(url, model=model)
                        images.append(encoded)
                    else:
                        text_parts.append(json.dumps(part, default=str))
                msg["content"] = "\n".join(part for part in text_parts if part)
                if images:
                    msg["images"] = images
            if msg.get("role") == "tool":
                # Native Ollama expects tool_name, not OpenAI tool_call_id/name.
                msg["tool_name"] = (
                    msg.get("tool_name") or msg.get("name") or msg.get("tool_call_id") or "tool"
                )
                msg.pop("tool_call_id", None)
                msg.pop("name", None)
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                fixed = []
                for i, tc in enumerate(msg.get("tool_calls") or []):
                    if not isinstance(tc, dict):
                        continue
                    fn = dict(tc.get("function") or {})
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args else {}
                        except Exception:
                            args = {}
                    fn["arguments"] = args if isinstance(args, dict) else {}
                    # Native examples omit id and put index under function.
                    fn.setdefault("index", i)
                    fixed.append({"type": "function", "function": fn})
                msg["tool_calls"] = fixed
            out.append(msg)
        return out

    def _parse_ollama_native(
        self,
        data: dict,
        *,
        request_id: str,
        fallback_model: str,
        expose_reasoning: bool = False,
    ) -> GatewayResponse:
        _raise_if_error_envelope(data, fallback_model)
        msg = data.get("message") or {}
        content = msg.get("content") or ""
        thinking = msg.get("thinking") or msg.get("reasoning_content") or ""
        if expose_reasoning and thinking:
            content = f"<thinking>\n{thinking}\n</thinking>\n\n{content}"
        elif isinstance(content, str):
            content = strip_reasoning_blocks(content)
        # Some Ollama-hosted models emit leading blank lines when think=False.
        # Trim only edge whitespace; preserve substantive internal formatting.
        if isinstance(content, str):
            content = content.strip()
        # Capture done_reason for continuation logic (e.g. "length" = budget hit)
        done_reason = data.get("done_reason") or ""
        resp = GatewayResponse(
            request_id=request_id,
            model=data.get("model") or fallback_model,
            content=content,
            tool_calls=self._normalize_tool_calls_from_provider(msg.get("tool_calls") or []),
            usage={
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            },
            raw=data if expose_reasoning else _sanitize_response_raw(data),
        )
        # Stash done_reason on the response so callers can detect truncation.
        resp.metadata["done_reason"] = done_reason
        resp.metadata["ollama_messages"] = (
            data.get("message") or {}
            if expose_reasoning
            else _sanitize_response_raw(data.get("message") or {})
        )
        # Validator: empty content + length = model wanted to say something but
        # ran out of budget. Log a diagnostic so we don't silently swallow it.
        if not resp.content.strip() and not resp.tool_calls and done_reason == "length":
            log.warning(
                "gateway.empty_length model=%s output_tokens=%d — model hit budget with nothing to show",
                resp.model,
                resp.usage.get("output_tokens", 0),
            )
        elif not resp.content.strip() and not resp.tool_calls:
            raise GatewayUpstreamError(
                502, "upstream returned no answer content or tool calls", resp.model
            )
        return resp

    def _parse_anthropic(
        self, data: dict, *, request_id: str, fallback_model: str
    ) -> GatewayResponse:
        _raise_if_error_envelope(data, fallback_model)
        content_parts: list[str] = []
        tool_calls: list[dict] = []
        for i, block in enumerate(data.get("content") or []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                content_parts.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": block.get("name") or "tool",
                            "arguments": json.dumps(block.get("input") or {}, default=str),
                        },
                    }
                )
        usage = data.get("usage") or {}
        resp = GatewayResponse(
            request_id=data.get("id") or request_id,
            model=data.get("model") or fallback_model,
            content="".join(content_parts).strip(),
            tool_calls=self._normalize_tool_calls_from_provider(tool_calls),
            usage={
                "input_tokens": int(usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
            },
            raw=_sanitize_response_raw(data),
        )
        if not resp.content.strip() and not resp.tool_calls:
            raise GatewayUpstreamError(
                502, "upstream returned no content or tool calls", resp.model
            )
        return resp

    async def _normalize_messages_for_provider(
        self, messages: list[dict], *, model: str = ""
    ) -> list[dict]:
        """Fix messages per-provider so tool history never causes 400s.

        Universal:
          - assistant.content must be string (not None)
          - tool_calls.function.arguments must be JSON string
          - tool results must have a unique non-empty tool_call_id
        Ollama:
          - inline images as base64
        """
        kind = self._provider_kind()
        out: list[dict] = []
        _tool_counter = 0
        for m in messages:
            role = m.get("role", "")
            msg = dict(m)
            content = msg.get("content")
            if isinstance(content, list):
                # Provider normalization must never mutate caller-owned history.
                # Image conversion edits these nested objects below.
                msg["content"] = [
                    {
                        **part,
                        **(
                            {"image_url": dict(part["image_url"])}
                            if isinstance(part, dict) and isinstance(part.get("image_url"), dict)
                            else {}
                        ),
                    }
                    if isinstance(part, dict)
                    else part
                    for part in content
                ]
            if role == "assistant":
                if msg.get("content") is None:
                    msg["content"] = ""
                tcs = msg.get("tool_calls")
                if tcs:
                    fixed = []
                    for tc in tcs:
                        tc = dict(tc)
                        fn = tc.get("function")
                        if fn and isinstance(fn, dict):
                            fn = dict(fn)
                            args = fn.get("arguments")
                            if args is not None and not isinstance(args, str):
                                fn["arguments"] = json.dumps(args, default=str)
                            tc["function"] = fn
                        if not tc.get("id"):
                            tc["id"] = f"call_{id(tc)}"
                        fixed.append(tc)
                    msg["tool_calls"] = fixed
            elif role == "tool":
                if msg.get("content") is None:
                    msg["content"] = ""
                elif not isinstance(msg.get("content"), str):
                    msg["content"] = json.dumps(msg["content"], default=str)
                _tool_counter += 1
                msg = _ensure_tool_call_id(msg, index=_tool_counter)
            if kind == "openrouter" and role == "tool":
                msg.pop("name", None)
            out.append(msg)
        if kind in ("openai", "openrouter"):
            out = self._repair_tool_pairing(out)
        if kind == "ollama":
            out = await self._ollama_inline_images(out, model=model)
        return out

    def _repair_tool_pairing(self, messages: list[dict]) -> list[dict]:
        """Make tool history strictly paired.

        Every tool result must be immediately preceded by an assistant whose
        tool_calls contain the tool_call_id. Empty filler assistants are
        dropped; orphan tool results with an id get a synthetic assistant stub.
        """
        repaired: list[dict] = []
        pending_tool_ids: set[str] = set()
        current_assistant_idx: int | None = None

        def _stub(tid: str) -> dict:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tid,
                        "type": "function",
                        "function": {"name": "tool", "arguments": "{}"},
                    }
                ],
            }

        for msg in messages:
            role = msg.get("role")
            if role == "assistant":
                tcs = msg.get("tool_calls") or []
                content = str(msg.get("content") or "")
                if not tcs and not content.strip():
                    continue
                ids = {str(tc.get("id") or "") for tc in tcs if isinstance(tc, dict)}
                pending_tool_ids = {tid for tid in ids if tid}
                repaired.append(msg)
                current_assistant_idx = len(repaired) - 1
                continue

            if role == "tool":
                tid = str(msg.get("tool_call_id") or "")
                if not tid:
                    continue
                if tid not in pending_tool_ids:
                    if current_assistant_idx is not None:
                        assistant = dict(repaired[current_assistant_idx])
                        tcs = list(assistant.get("tool_calls") or [])
                        tcs.append(_stub(tid)["tool_calls"][0])
                        assistant["tool_calls"] = tcs
                        repaired[current_assistant_idx] = assistant
                    else:
                        repaired.append(_stub(tid))
                        current_assistant_idx = len(repaired) - 1
                    pending_tool_ids.add(tid)
                repaired.append(msg)
                pending_tool_ids.discard(tid)
                continue

            pending_tool_ids = set()
            current_assistant_idx = None
            repaired.append(msg)
        return repaired

    @staticmethod
    def _validated_image_data_url(url: str, *, model: str) -> tuple[str, str, int]:
        """Return MIME, base64 payload, and decoded size for a bounded image URL."""
        header, separator, encoded = url.partition(",")
        lower_header = header.lower()
        if (
            not separator
            or not lower_header.startswith("data:image/")
            or ";base64" not in lower_header
            or not encoded
            or len(encoded) % 4
            or _BASE64_PAYLOAD_RE.fullmatch(encoded) is None
        ):
            raise GatewayUpstreamError(400, "invalid base64 image data URL", model)
        padding = len(encoded) - len(encoded.rstrip("="))
        decoded_size = (len(encoded) * 3 // 4) - padding
        if decoded_size > _MAX_OLLAMA_IMAGE_BYTES:
            raise GatewayUpstreamError(
                413,
                f"image exceeds {_MAX_OLLAMA_IMAGE_BYTES} byte limit",
                model,
            )
        mime = header[5:].split(";", 1)[0].strip().lower()
        return mime, encoded, decoded_size

    async def _ollama_inline_images(self, messages: list[dict], *, model: str = "") -> list[dict]:
        """Fetch a bounded set of remote images and inline them for Ollama."""
        import base64

        async def _fetch_image(client: httpx.AsyncClient, url: str) -> tuple[str, int]:
            try:
                async with client.stream("GET", url, headers={"accept": "image/*"}) as response:
                    if response.status_code < 200 or response.status_code >= 300:
                        raise GatewayUpstreamError(
                            422,
                            f"image fetch returned HTTP {response.status_code}",
                            model,
                        )
                    raw_length = response.headers.get("content-length", "")
                    if raw_length.isdigit() and int(raw_length) > _MAX_OLLAMA_IMAGE_BYTES:
                        raise GatewayUpstreamError(
                            413,
                            f"image exceeds {_MAX_OLLAMA_IMAGE_BYTES} byte limit",
                            model,
                        )
                    chunks: list[bytes] = []
                    byte_count = 0
                    async for chunk in response.aiter_bytes():
                        byte_count += len(chunk)
                        if byte_count > _MAX_OLLAMA_IMAGE_BYTES:
                            raise GatewayUpstreamError(
                                413,
                                f"image exceeds {_MAX_OLLAMA_IMAGE_BYTES} byte limit",
                                model,
                            )
                        chunks.append(chunk)
                    content_type = response.headers.get("content-type", "")
            except GatewayUpstreamError:
                raise
            except httpx.HTTPError as exc:
                raise GatewayUpstreamError(
                    422,
                    f"image fetch failed: {type(exc).__name__}",
                    model,
                ) from exc

            mime = content_type.split(";", 1)[0].strip().lower()
            if not mime.startswith("image/"):
                guessed, _encoding = mimetypes.guess_type(url)
                mime = str(guessed or "").lower()
            if not mime.startswith("image/"):
                raise GatewayUpstreamError(415, "image URL did not return image data", model)
            b64 = base64.b64encode(b"".join(chunks)).decode("ascii")
            return f"data:{mime};base64,{b64}", byte_count

        urls_to_fetch: list[tuple[int, int, str]] = []
        image_count = 0
        total_bytes = 0
        for mi, m in enumerate(messages):
            content = m.get("content")
            if isinstance(content, list):
                for pi, part in enumerate(content):
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_count += 1
                        image_spec = part.get("image_url")
                        if not isinstance(image_spec, dict):
                            raise GatewayUpstreamError(400, "invalid image_url block", model)
                        url = str(image_spec.get("url") or "")
                        if not url or len(url) > _MAX_IMAGE_URL_LENGTH:
                            raise GatewayUpstreamError(400, "invalid image URL", model)
                        if url.startswith("data:"):
                            _mime, _encoded, decoded_size = self._validated_image_data_url(
                                url, model=model
                            )
                            total_bytes += decoded_size
                        elif url.startswith(("http://", "https://")):
                            urls_to_fetch.append((mi, pi, url))
                        else:
                            raise GatewayUpstreamError(
                                400, "image URL must use http, https, or data", model
                            )
        if image_count > _MAX_OLLAMA_IMAGES:
            raise GatewayUpstreamError(
                413,
                f"request contains {image_count} images; limit is {_MAX_OLLAMA_IMAGES}",
                model,
            )
        if total_bytes > _MAX_OLLAMA_TOTAL_IMAGE_BYTES:
            raise GatewayUpstreamError(
                413,
                f"images exceed {_MAX_OLLAMA_TOTAL_IMAGE_BYTES} byte aggregate limit",
                model,
            )
        if not urls_to_fetch:
            return messages
        log.debug("gateway.fetching_images count=%d", len(urls_to_fetch))
        connection_count = min(len(urls_to_fetch), _MAX_OLLAMA_IMAGES)
        limits = httpx.Limits(
            max_connections=connection_count,
            max_keepalive_connections=connection_count,
        )
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(15.0, connect=5.0),
            limits=limits,
            follow_redirects=False,
        ) as image_client:
            tasks = [
                asyncio.create_task(_fetch_image(image_client, url)) for _, _, url in urls_to_fetch
            ]
            try:
                fetched = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        total_bytes += sum(size for _data_url, size in fetched)
        if total_bytes > _MAX_OLLAMA_TOTAL_IMAGE_BYTES:
            raise GatewayUpstreamError(
                413,
                f"images exceed {_MAX_OLLAMA_TOTAL_IMAGE_BYTES} byte aggregate limit",
                model,
            )
        for (mi, pi, _), (data_url, _size) in zip(urls_to_fetch, fetched, strict=True):
            messages[mi]["content"][pi] = {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
        return messages

    async def chat(self, req: GatewayRequest) -> GatewayResponse:
        # Explicit stream-only transports are collected into one response.
        if self.stream_required:

            async def _collect_stream() -> GatewayResponse:
                final_resp: GatewayResponse | None = None
                async for evt in self.chat_stream(req):
                    if evt.kind == "final" and evt.response is not None:
                        final_resp = evt.response
                if final_resp is None:
                    raise GatewayUpstreamError(502, "stream ended without final frame", req.model)
                return final_resp

            return await with_retry(
                _collect_stream,
                delays_ms=_SAFE_LLM_RETRY_DELAYS_MS,
                jitter_ms=100,
                retriable=(httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout),
                op_name=f"gateway.chat_stream_collect[{req.model}]",
            )

        _validate_gateway_request(req)

        request_id = str(_ulid_new_compat())
        # Normalize messages for the target provider.
        # Ollama doesn't support image_url format; it needs base64 inline.
        messages = await self._normalize_messages_for_provider(req.messages, model=req.model)

        payload: dict[str, Any] = {
            "model": req.model,
            "messages": messages,
        }
        # Enable llama-server KV prefix cache reuse for local OpenAI-compatible endpoints.
        # This lets the server skip reprocessing the system prompt prefix across turns.
        if self._is_llama_cpp_endpoint():
            payload["cache_prompt"] = True
        if req.tools and self._tools_supported_for_request(req):
            payload["tools"] = self._format_tools_for_provider(req.tools, model=req.model)
            if req.tool_choice is not None:
                payload["tool_choice"] = req.tool_choice
            elif self._provider_kind() != "ollama":
                payload["tool_choice"] = "auto"
        elif req.tool_choice is not None:
            payload["tool_choice"] = req.tool_choice
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.metadata:
            effort = req.metadata.get("reasoning_effort")
            if effort and effort not in ("off", "none"):
                payload["reasoning_effort"] = effort
            elif effort in ("off", "none"):
                payload["reasoning_effort"] = "none"
            if req.metadata.get("reasoning_output"):
                payload["reasoning"] = {"summary": "auto"}
            # llama.cpp reads these controls from chat_template_kwargs.
            self._apply_llama_cpp_thinking(payload, effort)
        # OpenAI lets callers pass a `user` field for abuse tracking;
        # we use `metadata.user` when present. Request id goes in
        # `metadata` which every provider we hit ignores gracefully.
        if req.metadata:
            if "user" in req.metadata:
                payload["user"] = str(req.metadata["user"])
            payload["metadata"] = {**req.metadata, "norax_request_id": request_id}
        else:
            payload["metadata"] = {"norax_request_id": request_id}

        # Strip fields the target provider doesn't understand.
        payload = self._sanitize_payload(payload)
        kind = self._provider_kind()
        if kind == "ollama":
            post_payload = self._to_ollama_native_payload(payload, metadata=req.metadata)
            post_path = "/api/chat"
            post_base = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        elif kind == "anthropic":
            post_payload = self._to_anthropic_payload(payload)
            post_path = self.chat_path
            post_base = self.base_url
        else:
            post_payload = payload
            post_path = self.chat_path
            post_base = self.base_url

        async def _call() -> GatewayResponse:
            self._begin_upstream_call()
            breaker_failure_recorded = False
            try:
                log.debug(
                    "gateway.request model=%s msgs=%d tools=%s",
                    req.model,
                    len(req.messages),
                    len(req.tools) if req.tools else 0,
                )
                async with self._client.stream(
                    "POST",
                    f"{post_base}{post_path}",
                    json=post_payload,
                ) as r:
                    if r.status_code >= 400:
                        body_text = await _read_error_preview(r)
                        body = body_text[:500] if body_text else "(empty)"
                        # Log the payload that caused the error (redacted)
                        msg_summary = [
                            f"{m.get('role', '?')}({len(str(m.get('content', '')))}ch)"
                            for m in req.messages[:10]
                        ]
                        log.error(
                            "gateway.http_%d body=%s payload_msgs=%s model=%s tools=%s",
                            r.status_code,
                            body,
                            msg_summary,
                            req.model,
                            len(req.tools) if req.tools else 0,
                        )
                        upstream = _extract_upstream_message(body_text)
                        # Provider/schema 4xx errors are request/model-specific; do not
                        # poison the shared circuit, especially for Ollama model swaps.
                        if _counts_as_upstream_failure(r.status_code):
                            self._breaker.on_failure()
                        else:
                            self._breaker.on_abandoned()
                        breaker_failure_recorded = True
                        raise GatewayUpstreamError(r.status_code, upstream, req.model)
                    r.raise_for_status()
                    body_bytes = await _read_bounded_body(
                        r,
                        limit=_MAX_CHAT_RESPONSE_BYTES,
                        model=req.model,
                    )
                try:
                    data = json.loads(body_bytes)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise GatewayUpstreamError(
                        502,
                        "upstream returned invalid JSON",
                        req.model,
                    ) from exc
                if not isinstance(data, dict):
                    raise GatewayUpstreamError(
                        502,
                        "upstream JSON response must be an object",
                        req.model,
                    )
                _raise_if_error_envelope(data, req.model)
                if kind == "ollama":
                    response = self._parse_ollama_native(
                        data,
                        request_id=request_id,
                        fallback_model=req.model,
                        expose_reasoning=bool(
                            req.metadata and req.metadata.get("reasoning_output")
                        ),
                    )
                elif kind == "anthropic":
                    response = self._parse_anthropic(
                        data, request_id=request_id, fallback_model=req.model
                    )
                else:
                    response = _parse_openai(
                        data,
                        request_id=request_id,
                        fallback_model=req.model,
                        expose_reasoning=bool(
                            req.metadata and req.metadata.get("reasoning_output")
                        ),
                        text_tool_names=self._text_tool_fallback_names(req),
                    )
                self._breaker.on_success()
                return response
            except asyncio.CancelledError:
                self._breaker.on_abandoned()
                raise
            except GatewayUpstreamError as e:
                if not breaker_failure_recorded and _counts_as_upstream_failure(e.status):
                    self._breaker.on_failure()
                raise
            except (httpx.TransportError, httpx.HTTPStatusError):
                self._breaker.on_failure()
                raise
            except CircuitOpen:
                raise
            except Exception:
                self._breaker.on_failure()
                raise

        response = await with_retry(
            _call,
            delays_ms=_SAFE_LLM_RETRY_DELAYS_MS,
            jitter_ms=100,
            retriable=(httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout),
            op_name=f"gateway.chat[{req.model}]",
        )
        # Continuation sits outside the original call's retry scope. A failure
        # during a continuation must never replay the already-billed first call.
        if kind == "ollama":
            return await self._maybe_continue_ollama(req, response, messages)
        return response

    async def _maybe_continue_ollama(
        self, req: GatewayRequest, resp: GatewayResponse, messages: list[dict]
    ) -> GatewayResponse:
        """Auto-continue when Ollama hits num_predict with done_reason='length'.

        Respects per-profile continuation_attempts. Appends assistant's partial
        response and a continuation prompt, then re-calls chat(). Avoids drift
        by telling the model to finish rather than start over.
        """
        from .ollama_profiles import resolve_profile

        profile = resolve_profile(req.model)
        if profile.continuation_attempts <= 0:
            return resp

        done_reason = resp.metadata.get("done_reason", "")
        if done_reason != "length":
            return resp

        # Don't continue if we got tool calls — the model finished its thought
        if resp.tool_calls:
            return resp

        # Carry the depth in the recursive request. Recording it only after
        # recursion returns never advances the nested call and can recurse
        # indefinitely when every response ends at the token limit.
        request_metadata = dict(req.metadata or {})
        current_attempt = int(request_metadata.get("_ollama_continuation_attempt") or 0)
        attempt = current_attempt + 1
        if attempt > profile.continuation_attempts:
            log.warning(
                "gateway.continuation_exhausted model=%s attempts=%d",
                req.model,
                profile.continuation_attempts,
            )
            return resp

        # Build continuation messages
        cont_messages = list(messages)
        cont_messages.append(
            {
                "role": "assistant",
                "content": resp.content or "",
            }
        )
        cont_messages.append(
            {
                "role": "user",
                "content": profile.continue_prompt
                or "Continue from where you left off. Finish your thought concisely.",
            }
        )

        log.info(
            "gateway.continuation model=%s attempt=%d/%d content_len=%d",
            req.model,
            attempt,
            profile.continuation_attempts,
            len(resp.content),
        )

        # Recursive call with continuation messages
        cont_req = GatewayRequest(
            model=req.model,
            messages=cont_messages,
            tools=req.tools,
            tool_choice=req.tool_choice,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            metadata={**request_metadata, "_ollama_continuation_attempt": attempt},
        )
        cont_resp = await self.chat(cont_req)

        # Merge: append continuation content to original
        merged_content = (resp.content or "") + (cont_resp.content or "")
        cont_resp.content = merged_content
        cont_resp.usage = {
            key: int((resp.usage or {}).get(key) or 0) + int((cont_resp.usage or {}).get(key) or 0)
            for key in ("input_tokens", "output_tokens")
        }
        cont_resp.metadata["continued"] = True
        cont_resp.metadata["continuation_attempt"] = max(
            attempt,
            int(cont_resp.metadata.get("continuation_attempt") or 0),
        )
        cont_resp.metadata["original_done_reason"] = done_reason
        return cont_resp

    async def chat_stream(self, req: GatewayRequest) -> AsyncIterator[StreamEvent]:
        """Stream a chat-completion as Server-Sent Events.

        Yields `StreamEvent` instances. Two kinds:
            kind="delta"   payload={"text": "...", "tool_calls": [...]}
            kind="final"   payload=GatewayResponse  (set in `.response`)

        On any error, raises after the breaker has been notified.
        """
        _validate_gateway_request(req)
        request_id = str(_ulid_new_compat())

        # Normalize messages for the target provider.
        messages = await self._normalize_messages_for_provider(req.messages, model=req.model)

        payload: dict[str, Any] = {
            "model": req.model,
            "messages": messages,
            "stream": True,
        }
        # Enable llama-server KV prefix cache reuse for local OpenAI-compatible endpoints.
        if self._is_llama_cpp_endpoint():
            payload["cache_prompt"] = True
        if req.tools and self._tools_supported_for_request(req):
            payload["tools"] = self._format_tools_for_provider(req.tools, model=req.model)
            if req.tool_choice is not None:
                payload["tool_choice"] = req.tool_choice
            elif self._provider_kind() != "ollama":
                payload["tool_choice"] = "auto"
        elif req.tool_choice is not None:
            payload["tool_choice"] = req.tool_choice
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.metadata:
            effort = req.metadata.get("reasoning_effort")
            if effort and effort not in ("off", "none"):
                payload["reasoning_effort"] = effort
            elif effort in ("off", "none"):
                payload["reasoning_effort"] = "none"
            if req.metadata.get("reasoning_output"):
                payload["reasoning"] = {"summary": "auto"}
            self._apply_llama_cpp_thinking(payload, effort)
        if req.metadata:
            if "user" in req.metadata:
                payload["user"] = str(req.metadata["user"])
            payload["metadata"] = {**req.metadata, "norax_request_id": request_id}
        else:
            payload["metadata"] = {"norax_request_id": request_id}

        # Strip fields the target provider doesn't understand.
        payload = self._sanitize_payload(payload)

        # Accumulators for the final assembled response.
        content_parts: list[str] = []
        ollama_raw_content_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}  # idx → {id,name,arguments}
        reasoning_parts: list[str] = []
        expose_reasoning = bool(req.metadata and req.metadata.get("reasoning_output"))
        reasoning_filter = ReasoningTagFilter(expose=expose_reasoning)
        # If we got a streaming usage, capture it; otherwise use defaults.
        usage: dict = {"input_tokens": 0, "output_tokens": 0}
        upstream_id: str | None = None
        upstream_model: str | None = None
        finish_reason = ""

        kind = self._provider_kind()
        if kind == "ollama":
            post_payload = self._to_ollama_native_payload(payload, metadata=req.metadata)
            post_path = "/api/chat"
            post_base = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        elif kind == "anthropic":
            post_payload = self._to_anthropic_payload(payload)
            post_path = self.chat_path
            post_base = self.base_url
        else:
            post_payload = payload
            post_path = self.chat_path
            post_base = self.base_url
        self._begin_upstream_call()
        breaker_failure_recorded = False
        try:
            async with self._client.stream(
                "POST",
                f"{post_base}{post_path}",
                json=post_payload,
            ) as r:
                if r.status_code >= 400:
                    body_text = await _read_error_preview(r)
                    log.error(
                        "gateway.http_%d body=%s model=%s stream=1",
                        r.status_code,
                        body_text[:500],
                        req.model,
                    )
                    upstream = _extract_upstream_message(body_text)
                    if _counts_as_upstream_failure(r.status_code):
                        self._breaker.on_failure()
                    else:
                        self._breaker.on_abandoned()
                    breaker_failure_recorded = True
                    raise GatewayUpstreamError(r.status_code, upstream, req.model)
                r.raise_for_status()
                async for raw_line in _iter_bounded_lines(r, model=req.model):
                    if not raw_line:
                        continue
                    line = raw_line.strip()

                    # OpenAI-compatible providers stream SSE: `data: {...}`.
                    # Native Ollama `/api/chat` streams plain JSONL: `{...}`.
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                    elif self._provider_kind() == "ollama" and line.startswith("{"):
                        data_str = line
                    else:
                        continue

                    try:
                        evt = json.loads(data_str)
                    except json.JSONDecodeError:
                        log.debug("stream.bad_json: %r", data_str[:200])
                        continue

                    upstream_id = upstream_id or evt.get("id")
                    upstream_model = upstream_model or evt.get("model")
                    _raise_if_error_envelope(evt, req.model)

                    if kind == "anthropic":
                        event_type = evt.get("type")
                        text = ""
                        tcs: list[dict] = []
                        if event_type == "message_start":
                            message = evt.get("message") or {}
                            upstream_id = upstream_id or message.get("id")
                            upstream_model = upstream_model or message.get("model")
                            usage = message.get("usage") or usage
                        elif event_type == "content_block_start":
                            block = evt.get("content_block") or {}
                            if block.get("type") == "tool_use":
                                idx = int(evt.get("index") or 0)
                                slot = tool_calls_acc.setdefault(
                                    idx,
                                    {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                slot["id"] = block.get("id") or slot["id"]
                                slot["function"]["name"] = (
                                    block.get("name") or slot["function"]["name"]
                                )
                        elif event_type == "content_block_delta":
                            idx = int(evt.get("index") or 0)
                            delta = evt.get("delta") or {}
                            if delta.get("type") == "text_delta":
                                text = delta.get("text") or ""
                            elif delta.get("type") == "input_json_delta":
                                slot = tool_calls_acc.setdefault(
                                    idx,
                                    {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                slot["function"]["arguments"] += delta.get("partial_json") or ""
                        elif event_type == "message_delta":
                            delta = evt.get("delta") or {}
                            finish_reason = str(delta.get("stop_reason") or finish_reason)
                            if evt.get("usage"):
                                usage = evt.get("usage") or usage
                    elif kind == "ollama" and "message" in evt:
                        msg = evt.get("message") or {}
                        text = msg.get("content") or ""
                        if expose_reasoning:
                            rtext = msg.get("thinking") or msg.get("reasoning_content") or ""
                            if rtext:
                                reasoning_parts.append(str(rtext))
                        tcs = msg.get("tool_calls") or []
                        if evt.get("done"):
                            finish_reason = str(evt.get("done_reason") or finish_reason)
                            usage = {
                                "input_tokens": evt.get("prompt_eval_count", 0),
                                "output_tokens": evt.get("eval_count", 0),
                            }
                            # Some servers echo the full response in the done
                            # frame; others send all content only in that frame.
                            # Remove only the already-seen prefix so both forms
                            # are handled without duplication or data loss.
                            seen_text = "".join(ollama_raw_content_parts)
                            if seen_text and text == seen_text:
                                text = ""
                            elif seen_text and text.startswith(seen_text):
                                text = text[len(seen_text) :]
                    else:
                        if "usage" in evt and evt["usage"]:
                            usage = evt["usage"]
                        choices = evt.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        finish_reason = str(choice.get("finish_reason") or finish_reason)
                        delta = choice.get("delta") or {}
                        text = delta.get("content") or ""
                        # Always capture reasoning parts so they can be used
                        # as fallback content when the model emits only
                        # reasoning_content and no content (e.g. Qwen3.8
                        # via llama-server with thinking enabled).
                        if kind != "ollama":
                            rtext = delta.get("reasoning") or delta.get("reasoning_content")
                            if rtext:
                                reasoning_parts.append(str(rtext))
                        tcs = delta.get("tool_calls") or []

                    if tcs:
                        for tc in tcs:
                            idx = tc.get("index", 0)
                            fn = tc.get("function") or {}
                            if "index" in fn:
                                idx = fn.get("index", idx)
                            slot = tool_calls_acc.setdefault(
                                idx,
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            if fn.get("name"):
                                slot["function"]["name"] = fn["name"]
                            args = fn.get("arguments")
                            if args:
                                slot["function"]["arguments"] += (
                                    args if isinstance(args, str) else json.dumps(args, default=str)
                                )
                    if kind == "ollama" and text:
                        ollama_raw_content_parts.append(text)
                    text = reasoning_filter.feed(text)
                    if text:
                        content_parts.append(text)
                    if text or tcs:
                        yield StreamEvent(
                            kind="delta",
                            text=text,
                            tool_calls_partial=tcs or None,
                        )
        except (asyncio.CancelledError, GeneratorExit):
            self._breaker.on_abandoned()
            raise
        except GatewayUpstreamError as e:
            if not breaker_failure_recorded and _counts_as_upstream_failure(e.status):
                self._breaker.on_failure()
            raise
        except (httpx.TransportError, httpx.HTTPStatusError):
            self._breaker.on_failure()
            raise
        except CircuitOpen:
            raise
        except Exception:
            self._breaker.on_failure()
            raise

        in_tok = int((usage or {}).get("input_tokens") or (usage or {}).get("prompt_tokens") or 0)
        out_tok = int(
            (usage or {}).get("output_tokens") or (usage or {}).get("completion_tokens") or 0
        )
        tail = reasoning_filter.finish()
        if tail:
            content_parts.append(tail)
            yield StreamEvent(kind="delta", text=tail)
        final_content = "".join(content_parts).strip()
        if expose_reasoning and reasoning_parts:
            reasoning_text = "".join(reasoning_parts).strip()
            if reasoning_text:
                final_content = (
                    f"<thinking>\n{reasoning_text}\n</thinking>\n\n{final_content}"
                ).strip()
        final = GatewayResponse(
            request_id=upstream_id or request_id,
            model=upstream_model or req.model,
            content=final_content,
            tool_calls=self._normalize_tool_calls_from_provider(
                [tool_calls_acc[k] for k in sorted(tool_calls_acc)]
            ),
            usage={"input_tokens": in_tok, "output_tokens": out_tok},
            raw={"streamed": True},
        )
        if finish_reason:
            final.metadata["finish_reason"] = finish_reason
        if not final.content.strip() and not final.tool_calls:
            # Native Ollama models such as Qwen3.8 may emit all reasoning in
            # message.thinking and set content="". Treat that as content so the
            # response isn't discarded as an upstream error.
            if reasoning_parts:
                reasoning_text = "".join(reasoning_parts).strip()
                if reasoning_text:
                    final = replace(final, content=reasoning_text)
            elif final.content == "" and not final.tool_calls:
                self._breaker.on_failure()
                raise GatewayUpstreamError(
                    502,
                    "upstream returned no content or tool calls",
                    final.model or req.model,
                )
        if not final.content.strip() and not final.tool_calls:
            self._breaker.on_failure()
            raise GatewayUpstreamError(
                502,
                "upstream returned no content or tool calls",
                final.model or req.model,
            )
        self._breaker.on_success()
        yield StreamEvent(kind="final", response=final)


def _parse_openai(
    data: dict,
    *,
    request_id: str,
    fallback_model: str,
    expose_reasoning: bool = False,
    text_tool_names: set[str] | None = None,
) -> GatewayResponse:
    """Parse an OpenAI chat-completions response into our envelope.

    Tolerant of minor variations (usage naming across providers).
    """
    _raise_if_error_envelope(data, fallback_model)
    choices = data.get("choices") or []
    if not choices:
        raise GatewayUpstreamError(502, "upstream returned no choices", fallback_model)
    choice = choices[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    if expose_reasoning and reasoning:
        content = f"<thinking>\n{reasoning}\n</thinking>\n\n{content}".strip()
    elif isinstance(content, str):
        content = strip_reasoning_blocks(content)
    tool_calls = GatewayClient._normalize_tool_calls_from_provider(msg.get("tool_calls") or [])

    # Fallback: if the provider returned no structured tool_calls but the
    # content contains <invoke> XML blocks (Ornith/Qwen fine-tune format),
    # extract them so Norax's dispatch loop can execute them.
    if not tool_calls and text_tool_names and content and "<invoke" in content:
        invoke_calls, cleaned = GatewayClient._extract_invoke_xml_tool_calls(content)
        recovered_names = {
            str((call.get("function") or {}).get("name") or "") for call in invoke_calls
        }
        if invoke_calls and not cleaned and recovered_names.issubset(text_tool_names):
            tool_calls = invoke_calls
            content = cleaned

    # Fallback: Qwen3-Coder style <function=name>...</function> blocks
    # (used when the provider's tool-call parser doesn't extract them)
    if not tool_calls and text_tool_names and content and "<function=" in content:
        qwen_calls, cleaned = GatewayClient._extract_qwen_function_tool_calls(content)
        recovered_names = {
            str((call.get("function") or {}).get("name") or "") for call in qwen_calls
        }
        if qwen_calls and not cleaned and recovered_names.issubset(text_tool_names):
            tool_calls = qwen_calls
            content = cleaned

    usage = data.get("usage") or {}
    # OpenAI uses prompt_tokens/completion_tokens; some providers also
    # add input_tokens/output_tokens. Normalize to input/output.
    in_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    out_tok = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)

    resp = GatewayResponse(
        request_id=data.get("id") or request_id,
        model=data.get("model") or fallback_model,
        content=content,
        tool_calls=tool_calls,
        usage={"input_tokens": in_tok, "output_tokens": out_tok},
        raw=data if expose_reasoning else _sanitize_response_raw(data),
    )
    finish_reason = str(choice.get("finish_reason") or "")
    if finish_reason:
        resp.metadata["finish_reason"] = finish_reason
    if not resp.content.strip() and not resp.tool_calls:
        raise GatewayUpstreamError(502, "upstream returned no content or tool calls", resp.model)
    return resp


class GatewayRouter:
    """Routes model requests to the correct GatewayClient by model name.

    Wraps multiple named providers (each a GatewayClient) with a list of
    (glob_pattern, provider_name) routes. First match wins. Falls back to
    the default provider if nothing matches.

    Duck-types GatewayClient: exposes chat(), chat_stream(), aclose(),
    base_url — so the rest of the runtime can use it transparently.
    """

    def __init__(
        self,
        providers: dict[str, GatewayClient],
        routes: list[tuple[str, str]],
        default_provider: str,
    ) -> None:
        if not isinstance(providers, dict) or not providers:
            raise ValueError("gateway router requires at least one provider")
        if any(
            not isinstance(name, str) or not _ROUTER_PROVIDER_NAME_RE.fullmatch(name)
            for name in providers
        ):
            raise ValueError("gateway router contains an invalid provider name")
        if default_provider not in providers:
            raise ValueError("gateway router default provider is not configured")
        validated_routes: list[tuple[str, str]] = []
        if not isinstance(routes, list) or len(routes) > 512:
            raise ValueError("gateway router routes must be a list of at most 512 entries")
        for route in routes:
            if not isinstance(route, (tuple, list)) or len(route) != 2:
                raise ValueError("each gateway route must contain a pattern and provider")
            pattern, provider = route
            if (
                not isinstance(pattern, str)
                or not pattern
                or len(pattern) > 512
                or any(not char.isprintable() or char.isspace() for char in pattern)
            ):
                raise ValueError("gateway route contains an invalid model pattern")
            if provider not in providers:
                raise ValueError(f"gateway route references unknown provider {provider!r}")
            validated_routes.append((pattern, provider))
        # Own snapshots so a caller cannot mutate routing behind lease
        # accounting and strand a client without retirement.
        self._providers = dict(providers)
        self._routes = validated_routes
        self._default = default_provider
        # Provider swaps are rare, while every generation passes through this
        # router.  Keep mutation commits free of ``await`` points and account
        # for active client leases instead of putting a coarse asyncio lock on
        # the request hot path.  Retired clients close only after their final
        # in-flight request/stream releases its lease.
        self._active_clients: dict[int, tuple[Any, int]] = {}
        self._retired_clients: dict[int, Any] = {}
        self._closed = False

    @property
    def base_url(self) -> str:
        return self._providers[self._default].base_url

    @property
    def provider_urls(self) -> dict[str, str]:
        return {name: c.base_url for name, c in self._providers.items()}

    @property
    def default_provider(self) -> str:
        return self._default

    def has_provider(self, name: str) -> bool:
        return name in self._providers

    def _lease_request(self, req: GatewayRequest) -> tuple[str, GatewayClient, GatewayRequest]:
        if self._closed:
            raise RuntimeError("gateway router is closed")
        name, client, req = self._resolve_request(req)
        key = id(client)
        current = self._active_clients.get(key)
        count = current[1] if current is not None else 0
        self._active_clients[key] = (client, count + 1)
        return name, client, req

    def _lease_model(self, model: str) -> tuple[str, GatewayClient]:
        if self._closed:
            raise RuntimeError("gateway router is closed")
        name, client = self._resolve(model)
        key = id(client)
        current = self._active_clients.get(key)
        count = current[1] if current is not None else 0
        self._active_clients[key] = (client, count + 1)
        return name, client

    def _retire_client(self, client: Any) -> Any | None:
        """Retire an unreferenced client, returning it when safe to close."""
        if any(candidate is client for candidate in self._providers.values()):
            return None
        key = id(client)
        if key in self._active_clients:
            self._retired_clients[key] = client
            return None
        self._retired_clients.pop(key, None)
        return client

    @staticmethod
    async def _close_client(client: Any, *, context: str) -> None:
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            # The routing mutation has already committed.  A cleanup failure
            # must be observable, but must not falsely report that the provider
            # update itself failed.
            log.warning("gateway.%s_client_close_failed", context, exc_info=True)

    async def _release_client(self, client: Any) -> None:
        key = id(client)
        current = self._active_clients.get(key)
        close_client = None
        if current is None:
            log.error("gateway.client_lease_underflow client=%s", type(client).__name__)
        elif current[1] > 1:
            self._active_clients[key] = (client, current[1] - 1)
        else:
            self._active_clients.pop(key, None)
            close_client = self._retired_clients.pop(key, None)
        if close_client is not None:
            await self._close_client(close_client, context="retired")

    async def upsert_provider(
        self,
        name: str,
        client: GatewayClient,
        *,
        patterns: list[str] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("gateway router is closed")
        if not isinstance(name, str) or not _ROUTER_PROVIDER_NAME_RE.fullmatch(name):
            raise ValueError("invalid gateway provider name")
        selected_patterns = patterns or [f"{name}/*"]
        if not isinstance(selected_patterns, list) or not 1 <= len(selected_patterns) <= 512:
            raise ValueError("provider patterns must contain 1-512 entries")
        if any(
            not isinstance(pattern, str)
            or not pattern
            or len(pattern) > 512
            or any(not char.isprintable() or char.isspace() for char in pattern)
            for pattern in selected_patterns
        ):
            raise ValueError("provider patterns contain an invalid model pattern")
        old = self._providers.get(name)
        self._providers = {**self._providers, name: client}
        routes = [(pattern, provider) for pattern, provider in self._routes if provider != name]
        dynamic = [(pattern, name) for pattern in selected_patterns]
        self._routes = dynamic + routes
        # A still-active retired client may be reinserted by an overlapping
        # update.  It is live again and must not close on lease release.
        self._retired_clients.pop(id(client), None)
        if old is not None and old is not client:
            close_client = self._retire_client(old)
            if close_client is not None:
                await self._close_client(close_client, context="replaced")

    async def remove_provider(self, name: str) -> bool:
        if self._closed:
            raise RuntimeError("gateway router is closed")
        if name == self._default or name not in self._providers:
            return False
        old = self._providers[name]
        self._providers = {key: value for key, value in self._providers.items() if key != name}
        self._routes = [route for route in self._routes if route[1] != name]
        close_client = self._retire_client(old)
        if close_client is not None:
            await self._close_client(close_client, context="removed")
        return True

    def _resolve(self, model: str) -> tuple[str, GatewayClient]:
        if (
            not isinstance(model, str)
            or not model
            or len(model) > _MAX_MODEL_ID_CHARS
            or any(character.isspace() or not character.isprintable() for character in model)
        ):
            raise ValueError("gateway router model id is invalid")
        for pattern, provider_name in self._routes:
            if fnmatch.fnmatch(model, pattern):
                client = self._providers.get(provider_name)
                if client:
                    log.debug("route %s → %s (pattern %s)", model, provider_name, pattern)
                    return provider_name, client
        log.debug("route %s → %s (default)", model, self._default)
        return self._default, self._providers[self._default]

    def route_for(self, model: str) -> tuple[str, str]:
        """Return the effective provider name and base URL for a model."""
        name, client = self._resolve(model)
        return name, client.base_url

    def provider_kind_for_model(self, model: str) -> str:
        """Return the effective transport kind after model routing."""
        _name, client = self._resolve(model)
        return client._provider_kind()

    async def transport_probe(
        self,
        model: str,
        *,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Probe the transport selected for ``model`` without an inference."""
        name, client = self._lease_model(model)
        try:
            probe = getattr(client, "transport_probe", None)
            if not callable(probe):
                return {
                    "ok": False,
                    "provider": name,
                    "error": "selected gateway client has no transport probe",
                }
            result = await probe(model, timeout=timeout)
            return {**result, "provider": name}
        finally:
            await asyncio.shield(self._release_client(client))

    def _resolve_request(self, req: GatewayRequest) -> tuple[str, GatewayClient, GatewayRequest]:
        name, client = self._resolve(req.model)
        return name, client, req

    def _upstream_req(
        self, req: GatewayRequest, name: str, client: GatewayClient
    ) -> GatewayRequest:
        # Strip `<prefix>/` from the routed model id so the upstream proxy
        # sees the bare model id. Prefix defaults to provider name.
        model = req.model
        prefix = (client.model_prefix or name) + "/"
        if model.startswith(prefix):
            model = model[len(prefix) :]
        # OpenRouter only knows the bare `anthropic/claude-fable-5` id. The
        # Discord selector emits tiered ids (…-thinking-high, …-xhigh, …-fast).
        # Normalize to the bare id and fold the tier into reasoning_effort.
        if name == "openrouter" and "claude-fable-5" in model:
            low = model.lower()
            effort = None
            if "thinking" in low:
                for tier in ("xhigh", "high", "medium", "low", "max"):
                    if low.endswith(tier):
                        effort = "high" if tier in ("xhigh", "max") else tier
                        break
                effort = effort or "high"
            elif low.endswith(("-xhigh", "-high", "-medium", "-low", "-max", "-fast")):
                effort = None  # no-thinking tiers → no reasoning
            model = "anthropic/claude-fable-5"
            md = dict(req.metadata or {})
            if effort:
                md["reasoning_effort"] = effort
            else:
                md.pop("reasoning_effort", None)
            req = replace(req, metadata=md)
        # Codex Direct: strip openai-codex/ prefix and resolve short aliases
        # to the canonical model ids the local Codex proxy expects.
        if name == "codex_direct":
            if model.startswith("openai-codex/"):
                model = model[len("openai-codex/") :]
            _CODEX_ALIASES = {
                "b3": "gpt-5.5",
                "b3-personal": "gpt-5.5-personal",
            }
            model = _CODEX_ALIASES.get(model, model)
        if model != req.model:
            req = replace(req, model=model)
        return req

    async def chat(self, req: GatewayRequest) -> GatewayResponse:
        name, client, req = self._lease_request(req)
        try:
            return await client.chat(self._upstream_req(req, name, client))
        finally:
            await asyncio.shield(self._release_client(client))

    async def chat_stream(self, req: GatewayRequest) -> AsyncIterator[StreamEvent]:
        name, client, req = self._lease_request(req)
        try:
            req = self._upstream_req(req, name, client)
            async for evt in client.chat_stream(req):
                yield evt
        finally:
            await asyncio.shield(self._release_client(client))

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        clients = {
            id(client): client
            for client in (*self._providers.values(), *self._retired_clients.values())
        }
        close_now = []
        for key, client in clients.items():
            if key in self._active_clients:
                # Still has in-flight requests. The runtime drains active
                # turns before calling gateway.aclose(), but a stream may
                # still be settling. Close it anyway — the process is
                # shutting down and leaking connections is worse than
                # interrupting a dying stream.
                self._retired_clients.pop(key, None)
                close_now.append(client)
            else:
                self._retired_clients.pop(key, None)
                close_now.append(client)
        await asyncio.gather(
            *(self._close_client(client, context="shutdown") for client in close_now)
        )
