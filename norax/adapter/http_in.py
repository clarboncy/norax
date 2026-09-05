"""HTTP ingress adapter for synthetic and programmatic sensory events."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import math
import re
import secrets
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any

import ulid
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from uvicorn import Config, Server

from ..envelope import Principal, SensoryInput, ThreadBinding
from ..observability.metrics import Metrics
from ..safety.secrets import redact

log = logging.getLogger("norax.adapter.http")

# Depth of the ingress backlog before the adapter sheds load. Sized well above
# any plausible burst of real owner traffic, but low enough that a runaway
# producer is rejected instead of consuming memory without bound.
_INGRESS_QUEUE_MAX = 256
_MAX_HTTP_REQUEST_BYTES = 1_048_576
_MAX_INGRESS_BODY_CHARS = 262_144
_MAX_THREAD_ID_CHARS = 256
_MAX_DASHBOARD_SOCKETS = 32
_WEBSOCKET_SEND_TIMEOUT_SECONDS = 1.0
_MAX_BURNIN_STATUS_BYTES = 1_048_576
_MAX_PUBLIC_STATUS_TEXT = 500
_MAX_PUBLIC_STATUS_ITEMS = 64
_PUBLIC_PROBE_FIELDS = frozenset(
    {
        "ok",
        "enabled",
        "mode",
        "transport_verified",
        "completion_verified",
        "model",
        "provider",
        "fallback",
        "latency_ms",
        "failures",
        "error",
        "timestamp",
    }
)
_PRIVATE_STATUS_KEY_PARTS = ("token", "secret", "authorization", "api_key", "base_url")
_URL_IN_STATUS_RE = re.compile(r"https?://[^\s,;]+", re.IGNORECASE)


class _PayloadTooLargeError(Exception):
    pass


def _public_status_error(value: object) -> str:
    """Return a bounded diagnostic without publishing credentials or topology."""
    scrubbed = str(redact(str(value)))
    return _URL_IN_STATUS_RE.sub("<redacted-url>", scrubbed)[:_MAX_PUBLIC_STATUS_TEXT]


def _public_status_value(value: object, *, depth: int = 0) -> object:
    """Normalize untrusted probe state into a small JSON-safe public value."""
    if depth >= 4:
        return "<truncated>"
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return str(redact(value))[:_MAX_PUBLIC_STATUS_TEXT]
    if isinstance(value, dict):
        output: dict[str, object] = {}
        for raw_key, item in list(value.items())[:_MAX_PUBLIC_STATUS_ITEMS]:
            key = str(raw_key)[:128]
            normalized_key = key.lower()
            if any(part in normalized_key for part in _PRIVATE_STATUS_KEY_PARTS):
                continue
            if normalized_key in {"endpoint", "url", "evidence"}:
                continue
            if normalized_key in {"error", "last_error"}:
                output[key] = _public_status_error(item)
            else:
                output[key] = _public_status_value(item, depth=depth + 1)
        return output
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            _public_status_value(item, depth=depth + 1)
            for item in list(value)[:_MAX_PUBLIC_STATUS_ITEMS]
        ]
    return str(redact(str(value)))[:_MAX_PUBLIC_STATUS_TEXT]


def _public_completion_probe(probe: dict[str, Any]) -> dict[str, object]:
    """Expose only the stable serving-evidence contract."""
    public = {key: probe[key] for key in _PUBLIC_PROBE_FIELDS if key in probe}
    normalized = _public_status_value(public)
    if not isinstance(normalized, dict):
        return {}
    if "error" in public:
        normalized["error"] = _public_status_error(public["error"])
    failures = public.get("failures")
    if isinstance(failures, (list, tuple)):
        normalized["failures"] = [
            _public_status_error(item) for item in failures[:_MAX_PUBLIC_STATUS_ITEMS]
        ]
    return normalized


class _RequestBodyLimitMiddleware:
    """Reject oversized HTTP bodies while streaming, without buffering a copy."""

    def __init__(self, app: Any, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
            except (TypeError, ValueError):
                await self._reject(send, 400, "invalid_content_length")
                return
            if content_length < 0:
                await self._reject(send, 400, "invalid_content_length")
                return
            if content_length > self.max_bytes:
                await self._reject(send, 413, "request_body_too_large")
                return

        received = 0
        response_started = False

        async def limited_receive() -> dict:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _PayloadTooLargeError
            return message

        async def tracked_send(message: dict) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _PayloadTooLargeError:
            if not response_started:
                await self._reject(send, 413, "request_body_too_large")

    @staticmethod
    async def _reject(send: Any, status: int, error: str) -> None:
        import json

        body = json.dumps({"ok": False, "error": error}, separators=(",", ":")).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class AgentOsChatBridge:
    """Direct owner-chat bridge between the Agent OS dashboard and Norax.

    Registers as an outbound adapter under channel key ``agent_os`` so the
    runtime routes assistant replies back through :meth:`send`, which fans
    them out to every connected dashboard WebSocket. Ingress arrives via the
    ``/ingress/agent_os`` HTTP route (token-authenticated, owner tier).
    """

    def __init__(self) -> None:
        self._sockets: dict[Any, asyncio.Lock] = {}
        # Run state belongs to the runtime, not to any browser connection.
        # A newly attached dashboard can therefore recover the typing state.
        self._working_turns = 0

    # -- websocket registry -------------------------------------------------
    def attach(self, ws: Any) -> bool:
        """Register a dashboard without allowing an unbounded socket fanout."""
        if ws in self._sockets:
            return True
        if len(self._sockets) >= _MAX_DASHBOARD_SOCKETS:
            return False
        self._sockets[ws] = asyncio.Lock()
        return True

    async def send_status_snapshot(self, ws: Any) -> None:
        """Send persisted lifecycle state to one newly connected dashboard."""
        import json

        sent = await self._send_one(
            ws,
            json.dumps(
                {
                    "type": "status",
                    "status": "working" if self._working_turns else "idle",
                }
            ),
        )
        if not sent:
            raise ConnectionError("dashboard websocket did not accept status snapshot")

    def detach(self, ws: Any) -> None:
        self._sockets.pop(ws, None)

    @property
    def connected(self) -> int:
        return len(self._sockets)

    # -- outbound adapter contract ------------------------------------------
    async def send(
        self,
        target: str,
        text: str,
        *,
        reply_to: str | None = None,
        **extra: Any,
    ) -> dict:
        import json

        payload = json.dumps(
            {
                "type": "reply",
                "thread_id": target,
                "text": text,
                "reply_to": reply_to,
                "ts": datetime.now(UTC).isoformat(),
            }
        )
        delivered = await self._broadcast(payload)
        return {
            "ok": delivered > 0,
            "message_id": _new_ulid(),
            "delivered": delivered,
            "error": None if delivered else "no_dashboard_connection",
        }

    async def broadcast_status(self, status: str) -> dict:
        """Persist and publish run lifecycle for reconnect-safe typing state."""
        import json

        if status in {"typing", "working", "processing"}:
            self._working_turns += 1
        elif status == "idle":
            self._working_turns = max(0, self._working_turns - 1)
        effective_status = "working" if self._working_turns else "idle"
        delivered = await self._broadcast(
            json.dumps({"type": "status", "status": effective_status})
        )
        return {"ok": True, "delivered": delivered}

    async def _broadcast(self, payload: str) -> int:
        """Fan a serialized event out and prune disconnected dashboards."""
        sockets = list(self._sockets)
        if not sockets:
            return 0
        results = await asyncio.gather(
            *(self._send_one(ws, payload) for ws in sockets),
            return_exceptions=False,
        )
        for ws, sent in zip(sockets, results, strict=True):
            if not sent:
                self.detach(ws)
        return sum(results)

    async def _send_one(self, ws: Any, payload: str) -> bool:
        """Serialize writes per socket and bound slow-client tail latency."""
        lock = self._sockets.get(ws)
        if lock is None:
            return False

        async def locked_send() -> None:
            async with lock:
                await ws.send_text(payload)

        try:
            await asyncio.wait_for(locked_send(), timeout=_WEBSOCKET_SEND_TIMEOUT_SECONDS)
            return True
        except Exception:  # noqa: BLE001
            return False

    async def broadcast_inbound(
        self,
        *,
        sender_label: str,
        text: str,
        source: str = "discord",
        ts: str | None = None,
    ) -> dict:
        """Push an inbound message (e.g. from Discord) to all dashboard WebSockets.

        This lets the web chat see messages that arrive from other channels
        (Discord DMs, server channels, etc.) in real time.
        """
        import json

        payload = json.dumps(
            {
                "type": "message",
                "source": source,
                "role": "user",
                "sender": sender_label,
                "text": text,
                "ts": ts or datetime.now(UTC).isoformat(),
            }
        )
        delivered = await self._broadcast(payload)
        return {"ok": True, "delivered": delivered}

    async def broadcast_outbound(
        self,
        *,
        text: str,
        sender_label: str = "Norax",
        source: str = "discord",
        ts: str | None = None,
    ) -> dict:
        """Mirror a Norax reply sent on Discord into the dashboard transcript."""
        import json

        payload = json.dumps(
            {
                "type": "message",
                "source": source,
                "role": "assistant",
                "sender": sender_label,
                "text": text,
                "ts": ts or datetime.now(UTC).isoformat(),
            }
        )
        delivered = await self._broadcast(payload)
        return {"ok": True, "delivered": delivered}


def _new_ulid() -> str:
    factory = getattr(ulid, "new", None)
    return str(factory() if factory is not None else ulid.ULID())


def _is_loopback_listener(host: str) -> bool:
    normalized = str(host or "").strip().lower().strip("[]")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _bearer_token(value: str) -> str:
    if not isinstance(value, str):
        return ""
    scheme, separator, token = value.strip().partition(" ")
    if separator and scheme.lower() == "bearer":
        return token.strip()
    return ""


class IngressBody(BaseModel):
    body: str = Field(min_length=1, max_length=_MAX_INGRESS_BODY_CHARS)
    sender_id: str = Field(default="test-user", min_length=1, max_length=256)


class ProviderBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(min_length=1, max_length=2_048)
    api_key: str | None = Field(default=None, max_length=65_536)
    provider_kind: str = Field(default="openai", min_length=1, max_length=32)
    models: list[Annotated[str, Field(min_length=1, max_length=512)]] = Field(
        default_factory=list,
        max_length=100,
    )
    enabled: bool = True


class HttpInAdapter:
    """HTTP ingress with explicit, restart-safe Uvicorn lifecycle ownership."""

    name = "http"

    def __init__(
        self,
        *,
        host: str,
        port: int,
        metrics: Metrics | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.metrics = metrics
        # Bounded: an unbounded queue turns an ingress flood into unbounded
        # memory growth with no signal to the caller. At the cap we shed load
        # explicitly (503) instead of degrading the whole process.
        self._queue: asyncio.Queue[SensoryInput] = asyncio.Queue(maxsize=_INGRESS_QUEUE_MAX)
        self._task: asyncio.Task[None] | None = None
        self._server: Server | None = None
        self._stopping = False
        self.app = FastAPI(title="Norax Ingress")
        self.app.add_middleware(
            _RequestBodyLimitMiddleware,
            max_bytes=_MAX_HTTP_REQUEST_BYTES,
        )
        self._runtime_ref: Any = None
        # Bridge for direct owner chat from the Agent OS dashboard. Wired as an
        # outbound adapter (channel key ``agent_os``) by the runtime during
        # build; also serves the token-authenticated ingress route below.
        self.agent_os_bridge = AgentOsChatBridge()
        self._owner_id: str = ""
        self._owner_label: str = "Owner"
        self._chat_token: str = ""

        if metrics is not None:
            from ..observability.metrics import mount_endpoints

            mount_endpoints(self.app, metrics)

        @self.app.get("/status")
        async def status() -> dict[str, Any]:
            """Lightweight process status used by operators and smoke tests."""
            result: dict[str, Any] = {
                "ok": True,
                "status": "ready" if self._runtime_ref is not None else "starting",
            }
            if self._runtime_ref is not None:
                rt = self._runtime_ref
                raw_caps = rt.capabilities.status()
                caps = _public_status_value(raw_caps)
                if caps:
                    result["capabilities"] = caps
                    failed = rt.capabilities.failed_capabilities()
                    degraded = rt.capabilities.degraded_capabilities()
                    stale = rt.capabilities.stale_capabilities()
                    affected = sorted(set(failed + degraded + stale))
                    if affected:
                        result["degraded"] = affected
                        result["status"] = "degraded"
                result["effective_model"] = getattr(rt, "_effective_model", "?")
                result["effective_provider"] = getattr(rt, "_effective_provider", "?")
                configured_provider = getattr(rt, "_configured_provider", None)
                result["configured_provider"] = configured_provider
                result["provider_selection_reason"] = (
                    "model_route"
                    if configured_provider and configured_provider != result["effective_provider"]
                    else "configured_default"
                )
                from ..version import build_info

                result.update(build_info())
            return result

        @self.app.get("/readyz")
        async def readyz() -> JSONResponse:
            """Readiness check — structured component map for production readiness.

            Checks: capability health, gateway reachability, memory store,
            event log writability, Discord connection state, and the cached
            serving probe. Optional component failures degrade status without
            returning ok=False.
            """
            result: dict[str, Any] = {"ok": True, "components": {}}
            if self._runtime_ref is not None:
                rt = self._runtime_ref
                comps = result["components"]

                # Capability health
                raw_caps = rt.capabilities.status()
                caps = _public_status_value(raw_caps)
                failed = rt.capabilities.failed_capabilities()
                degraded = rt.capabilities.degraded_capabilities()
                stale = rt.capabilities.stale_capabilities()
                blockers = rt.capabilities.readiness_blockers()
                comps["capabilities"] = {
                    "status": caps,
                    "failed": failed or None,
                    "degraded": degraded or None,
                    "stale": stale or None,
                    "readiness_blockers": blockers or None,
                }
                if blockers:
                    result["ok"] = False

                # Gateway route configuration. Reachability comes from the
                # cached serving probe below; /readyz itself performs no
                # outbound I/O and cannot amplify an upstream incident.
                # Provider base URLs are intentionally not exposed — they
                # are internal topology that a public readiness check
                # should not leak.
                gw = getattr(rt, "gateway", None)
                if gw is not None:
                    provider_name = getattr(rt, "_effective_provider", "?")
                    route_error = False
                    route_for = getattr(gw, "route_for", None)
                    if callable(route_for):
                        try:
                            provider_name, _base_url = route_for(
                                getattr(rt, "_effective_model", "")
                            )
                        except Exception as e:
                            comps["gateway"] = {
                                "reachable": False,
                                "error": _public_status_error(
                                    f"route resolution failed: {type(e).__name__}: {e}"
                                ),
                            }
                            result["ok"] = False
                            route_error = True
                    if not route_error:
                        probe = getattr(rt, "_last_probe_result", None)
                        configured_probe_enabled = (
                            probe.get("enabled", True) if probe is not None else True
                        )
                        # Unknown/malformed values fail safe as enabled instead
                        # of letting a truthy string such as "false" bypass a
                        # failed serving probe.
                        probe_enabled = configured_probe_enabled is not False
                        reachable: bool | str = "pending"
                        if probe is not None and not probe_enabled:
                            reachable = "unverified_probe_disabled"
                        elif probe is not None:
                            reachable = probe.get("ok") is True
                        comps["gateway"] = {
                            "reachable": reachable,
                            "provider": provider_name,
                            "evidence_mode": probe.get("mode") if probe else None,
                        }
                else:
                    comps["gateway"] = {"reachable": "no_gateway"}
                    result["ok"] = False

                # Memory store availability
                mem_store = getattr(rt, "_memory_store", None)
                if mem_store is not None:
                    try:
                        count = getattr(mem_store, "canonical_count", None)
                        neurons = count() if callable(count) else len(mem_store.all_canonical())
                        comps["memory"] = {"available": True, "neurons": neurons}
                    except Exception as e:
                        comps["memory"] = {
                            "available": False,
                            "error": _public_status_error(f"{type(e).__name__}: {e}"),
                        }
                        result["ok"] = False
                else:
                    comps["memory"] = {"available": False, "error": "no store"}
                    result["ok"] = False

                # Event log writability
                events = getattr(rt, "events", None)
                if events is not None:
                    try:
                        writable = getattr(events, "_writable", False) is True
                        comps["event_log"] = {
                            "writable": writable,
                            "last_error": _public_status_error(
                                getattr(events, "_last_write_error", "")
                            ),
                        }
                        if not writable:
                            result["ok"] = False
                    except Exception:
                        comps["event_log"] = {"writable": "unknown"}
                        result["ok"] = False
                else:
                    comps["event_log"] = {"writable": "no_log"}
                    result["ok"] = False

                # Discord connection state (optional — degrades, not fails)
                discord_adapter = getattr(rt, "discord", None) or getattr(rt, "_discord", None)
                if discord_adapter is not None:
                    conn_state = getattr(discord_adapter, "connection_state", None)
                    if conn_state is not None:
                        comps["discord"] = _public_status_value(conn_state)
                    else:
                        discord_client = getattr(discord_adapter, "client", None)
                        if discord_client is not None:
                            is_connected = getattr(discord_client, "is_ready", lambda: False)()
                            comps["discord"] = {"connected": is_connected is True}
                        else:
                            comps["discord"] = {"connected": "no_client"}
                else:
                    comps["discord"] = {"connected": "not_configured"}

                # Last cached serving probe. Transport mode is the default and
                # explicitly does not claim a model completion. Strip internal
                # evidence fields that may contain upstream endpoint URLs.
                probe = getattr(rt, "_last_probe_result", None)
                if probe is not None:
                    comps["completion_probe"] = _public_completion_probe(probe)
                    probe_enabled = probe.get("enabled", True) is not False
                    if probe_enabled and probe.get("ok") is not True:
                        result["ok"] = False
                    if probe_enabled:
                        try:
                            probe_at = datetime.fromisoformat(
                                str(probe.get("timestamp", "")).replace("Z", "+00:00")
                            )
                            age = (datetime.now(UTC) - probe_at).total_seconds()
                            comps["completion_probe"]["age_seconds"] = round(age, 1)
                            if age > 900:
                                comps["completion_probe"]["stale"] = True
                                result["ok"] = False
                        except (TypeError, ValueError):
                            comps["completion_probe"]["stale"] = True
                            result["ok"] = False
                else:
                    comps["completion_probe"] = {
                        "ok": False,
                        "error": "no serving probe has completed",
                    }
                    result["ok"] = False

                # Burn-in is release evidence, never a serving dependency.
                # Making its historical assessment affect /readyz creates a
                # circular failure: the monitor samples /readyz, then a failed
                # sample makes every future readiness sample fail permanently.
                try:
                    burnin_path = rt.cfg.state_dir / "burnin" / "status.json"
                    if burnin_path.exists():
                        import json

                        with burnin_path.open("rb") as handle:
                            raw_burnin = handle.read(_MAX_BURNIN_STATUS_BYTES + 1)
                        if len(raw_burnin) > _MAX_BURNIN_STATUS_BYTES:
                            raise ValueError("burn-in status exceeds 1 MiB")
                        burnin = json.loads(raw_burnin.decode("utf-8"))
                        if not isinstance(burnin, dict):
                            raise ValueError("burn-in status root is not an object")
                        assessment = burnin.get("assessment") or {}
                        if not isinstance(assessment, dict):
                            raise ValueError("burn-in assessment is not an object")
                        comps["burnin"] = _public_status_value(assessment)
                    else:
                        comps["burnin"] = {"state": "not_started"}
                except Exception as e:
                    comps["burnin"] = {
                        "state": "unknown",
                        "error": _public_status_error(f"{type(e).__name__}: {e}"),
                    }

            else:
                result["components"]["runtime"] = {"available": False}
                result["ok"] = False

            return JSONResponse(
                content=result,
                status_code=200 if result["ok"] else 503,
            )

        @self.app.post("/ingress/test", response_model=None)
        async def ingress_test(
            payload: IngressBody, request: Request
        ) -> dict[str, str | bool] | JSONResponse:
            # Loopback is the zero-configuration test surface. A public bind
            # must authenticate so this endpoint cannot become a free model-
            # execution relay or local memory/CPU exhaustion primitive.
            if not _is_loopback_listener(self.host) and not self._request_token_ok(request):
                return JSONResponse(
                    status_code=401,
                    content={"ok": False, "error": "unauthorized"},
                )
            # Runtime records ingress after policy/trust classification; counting
            # here would both duplicate that metric and omit required labels.
            env = SensoryInput(
                channel="http",
                source="http-test",
                message_id=_new_ulid(),
                timestamp=datetime.now(UTC),
                sender=Principal(
                    id=payload.sender_id,
                    label=payload.sender_id,
                    trust=False,
                    tier="guest",
                ),
                body=payload.body,
                trusted=False,
                metadata={"source": "http-test"},
            )
            offer_error = self._offer(env)
            if offer_error is not None:
                return JSONResponse(
                    status_code=503,
                    content={"ok": False, "error": offer_error},
                )
            # "received" — not "accepted". The adapter queue has the event,
            # but the runtime turn queue may still reject it under capacity
            # pressure. The runtime sends an explicit reply if that happens.
            return {"ok": True, "status": "received", "message_id": env.message_id}

        # -- direct owner chat (Agent OS dashboard) --------------------------
        @self.app.post("/ingress/agent_os", response_model=None)
        async def ingress_agent_os(request: Request) -> Any:
            """Authenticated owner-chat ingress from the Agent OS dashboard.

            Requires ``Authorization: Bearer <chat_token>`` matching the value
            configured on the adapter. Builds a *trusted, owner-tier* envelope
            so the runtime treats it exactly like an owner Discord DM.
            """
            if not self._request_token_ok(request):
                return JSONResponse(
                    status_code=401,
                    content={"ok": False, "error": "unauthorized"},
                )
            try:
                data = await request.json()
            except Exception:  # noqa: BLE001
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "error": "invalid_json"},
                )
            if not isinstance(data, dict):
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "error": "json_body_must_be_object"},
                )
            raw_body = data.get("body", "")
            if not isinstance(raw_body, str):
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "error": "body_must_be_string"},
                )
            if not raw_body.strip():
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "error": "empty_body"},
                )
            if len(raw_body) > _MAX_INGRESS_BODY_CHARS:
                return JSONResponse(
                    status_code=413,
                    content={"ok": False, "error": "body_too_large"},
                )
            raw_thread_id = data.get("thread_id") or "agent-os-main"
            if not isinstance(raw_thread_id, str):
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "error": "thread_id_must_be_string"},
                )
            thread_id = raw_thread_id.strip()
            if not thread_id or len(thread_id) > _MAX_THREAD_ID_CHARS:
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "error": "invalid_thread_id"},
                )
            owner = self._owner_id or "owner"
            env = SensoryInput(
                channel="chat",
                source="agent_os",
                message_id=_new_ulid(),
                timestamp=datetime.now(UTC),
                sender=Principal(id=owner, label=self._owner_label, trust=True, tier="owner"),
                body=raw_body,
                trusted=True,
                thread_binding=ThreadBinding(
                    channel="agent_os",
                    thread_id=thread_id,
                    kind="custom",
                ),
                metadata={"source": "agent-os"},
            )
            offer_error = self._offer(env)
            if offer_error is not None:
                return JSONResponse(
                    status_code=503,
                    content={"ok": False, "error": offer_error},
                )
            # "received" — the adapter queue accepted the event, but the
            # runtime turn queue may still reject under capacity pressure.
            return {"ok": True, "status": "received", "message_id": env.message_id}

        @self.app.get("/history/agent_os")
        async def history_agent_os(request: Request) -> Any:
            """Return recent rolling-window frames for the Agent OS dashboard."""
            if not self._request_token_ok(request):
                return JSONResponse(
                    status_code=401,
                    content={"ok": False, "error": "unauthorized"},
                )
            if self._runtime_ref is None:
                return {"ok": True, "messages": []}
            rt = self._runtime_ref
            channel_id = request.query_params.get("channel_id", "chat").strip()
            if not channel_id or len(channel_id) > _MAX_THREAD_ID_CHARS:
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "error": "invalid_channel_id"},
                )
            w = rt._get_window(channel_id)
            messages = []
            for frame in w.body[-200:]:
                kind = str(getattr(frame, "kind", ""))
                if kind not in {"user", "assistant"}:
                    continue
                text = getattr(frame, "content", "")
                if isinstance(text, list):
                    text = " ".join(str(t) for t in text)
                messages.append(
                    {
                        "role": kind,
                        "text": str(text)[:16_384],
                        "sender": getattr(frame, "sender", None),
                    }
                )
            return {"ok": True, "messages": messages}

        @self.app.get("/api/providers")
        async def list_providers(request: Request) -> Any:
            if not self._request_token_ok(request):
                return JSONResponse(status_code=401, content={"ok": False, "error": "unauthorized"})
            if self._runtime_ref is None:
                return JSONResponse(
                    status_code=503,
                    content={"ok": False, "error": "runtime_not_ready"},
                )
            return {
                "ok": True,
                "providers": self._runtime_ref._provider_store.list_public(),
                "model_catalog": self._runtime_ref.custom_model_catalog(),
            }

        @self.app.put("/api/providers/{name}")
        async def upsert_provider(name: str, body: ProviderBody, request: Request) -> Any:
            if not self._request_token_ok(request):
                return JSONResponse(status_code=401, content={"ok": False, "error": "unauthorized"})
            if self._runtime_ref is None:
                return JSONResponse(
                    status_code=503,
                    content={"ok": False, "error": "runtime_not_ready"},
                )
            spec = body.model_dump()
            spec["name"] = name
            try:
                provider = await self._runtime_ref.upsert_custom_provider(spec)
            except ValueError as exc:
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "error": "invalid_provider", "detail": str(exc)},
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("provider.upsert_failed name=%s error=%s", name, type(exc).__name__)
                return JSONResponse(
                    status_code=502,
                    content={"ok": False, "error": "provider_probe_failed"},
                )
            return {
                "ok": True,
                "provider": provider,
                "model_catalog": self._runtime_ref.custom_model_catalog(),
            }

        @self.app.delete("/api/providers/{name}")
        async def remove_provider(name: str, request: Request) -> Any:
            if not self._request_token_ok(request):
                return JSONResponse(status_code=401, content={"ok": False, "error": "unauthorized"})
            if self._runtime_ref is None:
                return JSONResponse(
                    status_code=503,
                    content={"ok": False, "error": "runtime_not_ready"},
                )
            try:
                removed = await self._runtime_ref.remove_custom_provider(name)
            except ValueError as exc:
                return JSONResponse(
                    status_code=409,
                    content={"ok": False, "error": "provider_in_use", "detail": str(exc)},
                )
            if not removed:
                return JSONResponse(
                    status_code=404,
                    content={"ok": False, "error": "provider_not_found"},
                )
            return {"ok": True, "removed": name}

        @self.app.websocket("/ws/agent_os")
        async def ws_agent_os(ws: WebSocket) -> None:
            header_token = _bearer_token(ws.headers.get("authorization", ""))
            if not self._token_ok(header_token or ws.query_params.get("token", "")):
                await ws.close(code=4401)
                return
            await ws.accept()
            if not self.agent_os_bridge.attach(ws):
                await ws.close(code=1013, reason="dashboard connection limit reached")
                return
            try:
                await self.agent_os_bridge.send_status_snapshot(ws)
                while True:
                    # Keep the socket open; client pings / sends thread joins.
                    await ws.receive_text()
            except WebSocketDisconnect:
                pass
            except Exception:  # noqa: BLE001
                pass
            finally:
                self.agent_os_bridge.detach(ws)

    def _token_ok(self, candidate: str) -> bool:
        """Constant-time check of the Agent OS shared secret.

        Plain ``!=`` short-circuits on the first differing byte, which leaks
        the secret to an attacker who can time enough requests.
        """
        if not isinstance(candidate, str) or not self._chat_token or not candidate:
            return False
        return secrets.compare_digest(candidate, self._chat_token)

    def _request_token_ok(self, request: Request) -> bool:
        return self._token_ok(_bearer_token(request.headers.get("authorization", "")))

    def _offer(self, env: SensoryInput) -> str | None:
        """Enqueue without blocking; return an explicit rejection reason."""
        if self._stopping:
            return "ingress_stopping"
        try:
            self._queue.put_nowait(env)
            return None
        except asyncio.QueueFull:
            log.warning(
                "ingress queue full (max=%d); rejecting %s/%s",
                _INGRESS_QUEUE_MAX,
                env.channel,
                env.source,
            )
            return "ingress_queue_full"

    def configure_owner_chat(
        self,
        *,
        owner_id: str,
        chat_token: str,
        owner_label: str = "Owner",
    ) -> None:
        """Set the owner identity and shared secret for the Agent OS chat bridge."""
        if not isinstance(chat_token, str) or not chat_token or len(chat_token) > 65_536:
            raise ValueError("chat_token must contain 1-65536 characters")
        self._owner_id = str(owner_id)[:256]
        self._owner_label = str(owner_label)[:256]
        self._chat_token = chat_token

    async def start(self) -> None:
        """Start serving and return only after Uvicorn bound its listener."""
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        self._server = Server(
            Config(
                self.app,
                host=self.host,
                port=self.port,
                log_level="warning",
                ws_max_size=_MAX_HTTP_REQUEST_BYTES,
            )
        )
        self._task = asyncio.create_task(self._server.serve(), name="norax-http-ingress")
        deadline = asyncio.get_running_loop().time() + 5.0
        try:
            while True:
                started = getattr(self._server, "started", False)
                is_started = (
                    started.is_set() if isinstance(started, asyncio.Event) else bool(started)
                )
                if is_started:
                    return
                if self._task.done():
                    await self._task
                    raise RuntimeError("HTTP ingress stopped before becoming ready")
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("HTTP ingress did not become ready within 5 seconds")
                await asyncio.sleep(0.01)
        except BaseException:
            self._stopping = True
            if self._server is not None:
                self._server.should_exit = True
            if self._task is not None and not self._task.done():
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
            self._server = None
            raise

    async def stop(self) -> None:
        """Request graceful shutdown, then cancel only if the server stalls."""
        self._stopping = True
        task, server = self._task, self._server
        if task is None:
            return
        if server is not None:
            server.should_exit = True
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
        except TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        except Exception:  # noqa: BLE001
            log.warning("HTTP ingress task failed during shutdown", exc_info=True)
        finally:
            self._task = None
            self._server = None

    async def events(self) -> AsyncIterator[SensoryInput]:
        while True:
            server_task = self._task
            if server_task is None:
                yield await self._queue.get()
                continue
            queued = asyncio.create_task(self._queue.get())
            try:
                done, _pending = await asyncio.wait(
                    {queued, server_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if queued in done:
                    yield queued.result()
                    continue
                if self._stopping:
                    return
                if server_task.cancelled():
                    raise RuntimeError("HTTP ingress listener was cancelled unexpectedly")
                error = server_task.exception()
                if error is not None:
                    raise RuntimeError("HTTP ingress listener failed") from error
                raise RuntimeError("HTTP ingress listener stopped unexpectedly")
            finally:
                if not queued.done():
                    queued.cancel()
                    await asyncio.gather(queued, return_exceptions=True)
