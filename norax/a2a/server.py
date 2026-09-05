"""A2A server for exposing a Norax runtime to other agents.

The advertised interface implements the non-streaming Agent2Agent 1.0
HTTP+JSON and JSON-RPC operations. Older ``tasks/*`` JSON-RPC method names
remain accepted as an unadvertised compatibility layer.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final
from urllib.parse import urlsplit

log = logging.getLogger("norax.a2a.server")

PROTOCOL_VERSION: Final = "1.0"
MEDIA_TYPE: Final = "application/a2a+json"
MAX_MESSAGE_CHARS: Final = 64_000
MAX_RESULT_CHARS: Final = 256_000
MAX_IDENTIFIER_CHARS: Final = 256
DEFAULT_MAX_TASKS: Final = 1_000
DEFAULT_MAX_CONCURRENT: Final = 4

SUBMITTED: Final = "TASK_STATE_SUBMITTED"
WORKING: Final = "TASK_STATE_WORKING"
COMPLETED: Final = "TASK_STATE_COMPLETED"
FAILED: Final = "TASK_STATE_FAILED"
CANCELED: Final = "TASK_STATE_CANCELED"
REJECTED: Final = "TASK_STATE_REJECTED"
INPUT_REQUIRED: Final = "TASK_STATE_INPUT_REQUIRED"
AUTH_REQUIRED: Final = "TASK_STATE_AUTH_REQUIRED"

TERMINAL_STATES: Final = frozenset({COMPLETED, FAILED, CANCELED, REJECTED})
INTERRUPTED_STATES: Final = frozenset({INPUT_REQUIRED, AUTH_REQUIRED})

_LEGACY_STATUS: Final = {
    SUBMITTED: "submitted",
    WORKING: "working",
    COMPLETED: "completed",
    FAILED: "failed",
    CANCELED: "canceled",
    REJECTED: "rejected",
    INPUT_REQUIRED: "input-required",
    AUTH_REQUIRED: "auth-required",
}


def _utc_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=UTC).isoformat().replace("+00:00", "Z")


def _validated_base_url(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("A2A base_url must be a string")
    value = (value or "http://127.0.0.1:8766").strip().rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("A2A base_url must be an HTTP(S) origin or base path without credentials")
    return value


class A2AError(Exception):
    """A safe protocol error that can be mapped to HTTP and JSON-RPC."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        jsonrpc_code: int = -32602,
        reason: str = "INVALID_ARGUMENT",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.jsonrpc_code = jsonrpc_code
        self.reason = reason

    def problem(self) -> dict[str, Any]:
        return {
            "type": f"https://a2a-protocol.org/errors/{self.reason.lower().replace('_', '-')}",
            "title": self.reason.replace("_", " ").title(),
            "status": self.status_code,
            "detail": str(self),
        }


def _validated_identifier(value: Any, *, label: str, required: bool = False) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise A2AError(f"{label} must be a string", reason="INVALID_ARGUMENT")
    value = value.strip()
    if (required and not value) or len(value) > MAX_IDENTIFIER_CHARS:
        qualifier = "non-empty and " if required else ""
        raise A2AError(
            f"{label} must be {qualifier}at most {MAX_IDENTIFIER_CHARS} characters",
            reason="INVALID_ARGUMENT",
        )
    return value


def _text_from_parts(parts: Any) -> str:
    if not isinstance(parts, list) or not parts:
        raise A2AError("message.parts must be a non-empty array", reason="INVALID_ARGUMENT")
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, str):  # compatibility with the original local client
            text = part
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            text = part["text"]
        else:
            raise A2AError(
                "only text/plain message parts are supported",
                status_code=415,
                jsonrpc_code=-32005,
                reason="CONTENT_TYPE_NOT_SUPPORTED",
            )
        if text:
            text_parts.append(text)
    text = "\n".join(text_parts).strip()
    if not text:
        raise A2AError("message text must not be empty", reason="INVALID_ARGUMENT")
    if len(text) > MAX_MESSAGE_CHARS:
        raise A2AError(
            f"message text exceeds the {MAX_MESSAGE_CHARS} character limit",
            status_code=413,
            reason="PAYLOAD_TOO_LARGE",
        )
    return text


@dataclass(frozen=True)
class AgentSkill:
    """A capability advertised in the public Agent Card."""

    id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    input_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    output_modes: list[str] = field(default_factory=lambda: ["text/plain"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "inputModes": list(self.input_modes),
            "outputModes": list(self.output_modes),
        }


@dataclass(frozen=True)
class AgentCard:
    """Public A2A discovery document."""

    name: str
    description: str
    url: str
    version: str = "0.11.0"
    capabilities: dict[str, Any] = field(
        default_factory=lambda: {
            "streaming": False,
            "pushNotifications": False,
            "extendedAgentCard": False,
        }
    )
    skills: list[AgentSkill] = field(default_factory=list)
    default_input_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    default_output_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    auth_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        interfaces = [
            {
                "url": self.url,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": PROTOCOL_VERSION,
            },
            {
                "url": f"{self.url}/jsonrpc",
                "protocolBinding": "JSONRPC",
                "protocolVersion": PROTOCOL_VERSION,
            },
        ]
        result: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "supportedInterfaces": interfaces,
            "version": self.version,
            "capabilities": dict(self.capabilities),
            "defaultInputModes": list(self.default_input_modes),
            "defaultOutputModes": list(self.default_output_modes),
            "skills": [skill.to_dict() for skill in self.skills],
        }
        if self.auth_required:
            result["securitySchemes"] = {
                "bearer": {
                    "httpAuthSecurityScheme": {
                        "scheme": "bearer",
                    }
                }
            }
            result["securityRequirements"] = [{"schemes": {"bearer": {"list": []}}}]
        return result


@dataclass
class A2ATask:
    """Internal task lifecycle with protocol serializers."""

    id: str
    context_id: str
    status: str = SUBMITTED
    message: str = ""
    result: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def touch(self, status: str) -> None:
        self.status = status
        self.updated_at = time.time()

    def complete(self, result: str) -> None:
        self.result = result
        self.error = ""
        self.artifacts = [
            {
                "artifactId": str(uuid.uuid4()),
                "name": "Norax response",
                "parts": [{"text": result}],
            }
        ]
        self.touch(COMPLETED)

    def fail(self, error: str) -> None:
        self.result = ""
        self.error = error
        self.artifacts = []
        self.touch(FAILED)

    def to_dict(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "state": self.status,
            "timestamp": _utc_timestamp(self.updated_at),
        }
        if self.error:
            status["message"] = {
                "messageId": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.id}:{self.error}")),
                "role": "ROLE_AGENT",
                "parts": [{"text": self.error}],
            }
        result: dict[str, Any] = {
            "id": self.id,
            "contextId": self.context_id,
            "status": status,
        }
        if self.artifacts:
            result["artifacts"] = list(self.artifacts)
        return result

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "contextId": self.context_id,
            "status": _LEGACY_STATUS.get(self.status, "unknown"),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "artifacts": list(self.artifacts),
        }


class _A2AOutboundAdapter:
    """Capture runtime outbound messages and return them to the waiting task."""

    def __init__(self, server: NoraxA2AServer) -> None:
        self._server = server

    async def send(self, target: str, text: str, **_kwargs: Any) -> dict[str, Any]:
        future = self._server._reply_futures.get(str(target))
        chunks = self._server._reply_chunks.get(str(target))
        if future is None or chunks is None:
            return {"ok": False, "error": "unknown_or_finished_a2a_task"}
        cleaned = str(text or "").strip()
        if not cleaned:
            return {"ok": False, "error": "empty_a2a_reply"}
        if sum(len(chunk) for chunk in chunks) + len(cleaned) > MAX_RESULT_CHARS:
            if not future.done():
                future.set_exception(RuntimeError("A2A reply exceeded the size limit"))
            return {"ok": False, "error": "a2a_reply_too_large"}
        chunks.append(cleaned)
        if not future.done():
            future.set_result(cleaned)
        return {"ok": True, "message_id": str(uuid.uuid4())}


class NoraxA2AServer:
    """Bounded, non-streaming A2A facade around a Norax runtime."""

    def __init__(
        self,
        runtime: Any,
        base_url: str = "",
        *,
        auth_token: str | None = None,
        max_tasks: int = DEFAULT_MAX_TASKS,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ) -> None:
        if isinstance(max_tasks, bool) or not isinstance(max_tasks, int) or max_tasks < 1:
            raise ValueError("max_tasks must be a positive integer")
        if (
            isinstance(max_concurrent, bool)
            or not isinstance(max_concurrent, int)
            or max_concurrent < 1
        ):
            raise ValueError("max_concurrent must be a positive integer")
        token = (auth_token or "").strip()
        self.runtime = runtime
        self.base_url = _validated_base_url(base_url)
        self.auth_token = token or None
        self.max_tasks = min(max_tasks, 100_000)
        self.max_concurrent = min(max_concurrent, 128)
        self._tasks: dict[str, A2ATask] = {}
        self._task_futures: dict[str, asyncio.Task[None]] = {}
        self._reply_futures: dict[str, asyncio.Future[str]] = {}
        self._reply_chunks: dict[str, list[str]] = {}
        self._context_locks: dict[str, asyncio.Lock] = {}
        self._slots = asyncio.Semaphore(self.max_concurrent)
        self._closed = False
        self._outbound_registry: Any = None
        self._outbound_adapter: _A2AOutboundAdapter | None = None

        outbound = getattr(runtime, "outbound", None)
        if outbound is not None and callable(getattr(outbound, "register", None)):
            if callable(getattr(outbound, "has", None)) and outbound.has("a2a"):
                raise RuntimeError("runtime already has an A2A outbound adapter")
            adapter = _A2AOutboundAdapter(self)
            outbound.register("a2a", adapter)
            self._outbound_registry = outbound
            self._outbound_adapter = adapter

    def get_agent_card(self) -> AgentCard:
        skills = [
            AgentSkill(
                id="general",
                name="General task assistance",
                description=(
                    "Reason about and execute text-based tasks using capabilities allowed by the "
                    "configured Norax runtime policy."
                ),
                tags=["general", "analysis", "automation"],
            ),
            AgentSkill(
                id="code",
                name="Software engineering",
                description=(
                    "Analyze, write, test, and debug code when repository and tool access are "
                    "permitted by runtime policy."
                ),
                tags=["coding", "development", "testing"],
            ),
            AgentSkill(
                id="research",
                name="Research and synthesis",
                description=(
                    "Research and synthesize information using sources available to the runtime."
                ),
                tags=["research", "analysis"],
            ),
        ]
        return AgentCard(
            name="Norax",
            description="A policy-governed, tool-using agent runtime with durable context.",
            url=self.base_url,
            skills=skills,
            auth_required=self.auth_token is not None,
        )

    def authorize(self, authorization: str | None) -> None:
        if self.auth_token is None:
            return
        scheme, separator, credential = (authorization or "").partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not secrets.compare_digest(credential, self.auth_token)
        ):
            raise A2AError(
                "valid bearer authentication is required",
                status_code=401,
                jsonrpc_code=-32001,
                reason="UNAUTHENTICATED",
            )

    def _store_task(self, task: A2ATask) -> None:
        while len(self._tasks) >= self.max_tasks:
            terminal = [item for item in self._tasks.values() if item.status in TERMINAL_STATES]
            if not terminal:
                raise A2AError(
                    "A2A task capacity is currently exhausted",
                    status_code=503,
                    jsonrpc_code=-32004,
                    reason="RESOURCE_EXHAUSTED",
                )
            oldest = min(terminal, key=lambda item: (item.updated_at, item.id))
            self._tasks.pop(oldest.id, None)
            self._prune_context_lock(oldest.context_id)
        self._tasks[task.id] = task

    def _prune_context_lock(self, context_id: str) -> None:
        if any(task.context_id == context_id for task in self._tasks.values()):
            return
        lock = self._context_locks.get(context_id)
        if lock is not None and not lock.locked():
            self._context_locks.pop(context_id, None)

    @staticmethod
    def _parse_message(params: Any) -> tuple[str, str, str, str]:
        if not isinstance(params, dict):
            raise A2AError("request body must be an object", reason="INVALID_ARGUMENT")
        message = params.get("message")
        if not isinstance(message, dict):
            raise A2AError("message must be an object", reason="INVALID_ARGUMENT")
        role = message.get("role", "ROLE_USER")
        if role not in {"ROLE_USER", "user"}:
            raise A2AError("message.role must be ROLE_USER", reason="INVALID_ARGUMENT")
        text = _text_from_parts(message.get("parts"))
        message_id = _validated_identifier(
            message.get("messageId") or str(uuid.uuid4()),
            label="message.messageId",
            required=True,
        )
        task_id = _validated_identifier(
            message.get("taskId") or params.get("id"),
            label="message.taskId",
        )
        context_id = _validated_identifier(
            message.get("contextId"),
            label="message.contextId",
        )
        return text, message_id, task_id, context_id

    async def send_message(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise A2AError(
                "A2A server is shutting down",
                status_code=503,
                jsonrpc_code=-32004,
                reason="UNAVAILABLE",
            )
        text, message_id, requested_task_id, requested_context_id = self._parse_message(params)
        configuration = params.get("configuration")
        if configuration is None:
            configuration = {}
        elif not isinstance(configuration, dict):
            raise A2AError("configuration must be an object", reason="INVALID_ARGUMENT")
        return_immediately = configuration.get("returnImmediately", False)
        if not isinstance(return_immediately, bool):
            raise A2AError("configuration.returnImmediately must be boolean")

        if requested_task_id:
            task = self._tasks.get(requested_task_id)
            if task is None:
                raise A2AError(
                    f"task not found: {requested_task_id}",
                    status_code=404,
                    jsonrpc_code=-32001,
                    reason="TASK_NOT_FOUND",
                )
            if task.status in TERMINAL_STATES:
                raise A2AError(
                    "messages cannot be added to a terminal task",
                    status_code=409,
                    jsonrpc_code=-32004,
                    reason="UNSUPPORTED_OPERATION",
                )
            if requested_context_id and requested_context_id != task.context_id:
                raise A2AError(
                    "message.contextId does not match the referenced task",
                    status_code=409,
                    reason="INVALID_ARGUMENT",
                )
            if requested_task_id in self._task_futures:
                raise A2AError(
                    "the referenced task is already running",
                    status_code=409,
                    reason="TASK_ALREADY_RUNNING",
                )
            context_id = task.context_id
            task.message = text
            task.error = ""
            task.result = ""
            task.artifacts = []
            task.touch(SUBMITTED)
        else:
            task = A2ATask(
                id=str(uuid.uuid4()),
                context_id=requested_context_id or str(uuid.uuid4()),
                message=text,
            )
            self._store_task(task)
            context_id = task.context_id

        worker = asyncio.create_task(
            self._run_task(task, text, message_id, context_id),
            name=f"a2a-task-{task.id}",
        )
        self._task_futures[task.id] = worker
        if not return_immediately:
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                if worker.cancelled():
                    pass
                else:
                    raise
        return {"task": task.to_dict()}

    async def _run_task(
        self,
        task: A2ATask,
        text: str,
        message_id: str,
        context_id: str,
    ) -> None:
        try:
            async with self._slots:
                if task.status == CANCELED:
                    return
                task.touch(WORKING)
                context_lock = self._context_locks.setdefault(context_id, asyncio.Lock())
                async with context_lock:
                    result = await self._execute_task(text, task.id, context_id, message_id)
                if task.status != CANCELED:
                    task.complete(result)
        except asyncio.CancelledError:
            task.touch(CANCELED)
            raise
        except Exception:  # noqa: BLE001
            log.exception("a2a task failed task_id=%s", task.id)
            task.fail("Task execution failed")
        finally:
            current = self._task_futures.get(task.id)
            if current is asyncio.current_task():
                self._task_futures.pop(task.id, None)

    async def get_task(self, task_id: str) -> dict[str, Any]:
        task_id = _validated_identifier(task_id, label="task id", required=True)
        task = self._tasks.get(task_id)
        if task is None:
            raise A2AError(
                f"task not found: {task_id}",
                status_code=404,
                jsonrpc_code=-32001,
                reason="TASK_NOT_FOUND",
            )
        return {"task": task.to_dict()}

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        task_id = _validated_identifier(task_id, label="task id", required=True)
        task = self._tasks.get(task_id)
        if task is None:
            raise A2AError(
                f"task not found: {task_id}",
                status_code=404,
                jsonrpc_code=-32001,
                reason="TASK_NOT_FOUND",
            )
        if task.status in TERMINAL_STATES:
            raise A2AError(
                "task is no longer cancelable",
                status_code=409,
                jsonrpc_code=-32002,
                reason="TASK_NOT_CANCELABLE",
            )
        task.touch(CANCELED)
        worker = self._task_futures.get(task_id)
        if worker is not None and not worker.done():
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        return {"task": task.to_dict()}

    async def list_tasks(
        self,
        *,
        context_id: str = "",
        status: str = "",
        page_size: int = 50,
        page_token: str = "",
    ) -> dict[str, Any]:
        context_id = _validated_identifier(context_id, label="contextId")
        page_token = _validated_identifier(page_token, label="pageToken")
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or not 1 <= page_size <= 100
        ):
            raise A2AError("pageSize must be an integer between 1 and 100")
        valid_states = TERMINAL_STATES | INTERRUPTED_STATES | {SUBMITTED, WORKING}
        if status and status not in valid_states:
            raise A2AError("status is not a valid A2A task state")

        tasks = sorted(self._tasks.values(), key=lambda item: (-item.updated_at, item.id))
        if context_id:
            tasks = [task for task in tasks if task.context_id == context_id]
        if status:
            tasks = [task for task in tasks if task.status == status]
        start = 0
        if page_token:
            positions = [index for index, task in enumerate(tasks) if task.id == page_token]
            if not positions:
                raise A2AError("pageToken is invalid for this result set")
            start = positions[0] + 1
        page = tasks[start : start + page_size]
        next_token = page[-1].id if start + len(page) < len(tasks) and page else ""
        return {
            "tasks": [task.to_dict() for task in page],
            "totalSize": len(tasks),
            "pageSize": len(page),
            "nextPageToken": next_token,
        }

    async def handle_jsonrpc(self, request: Any) -> dict[str, Any]:
        req_id: Any = request.get("id") if isinstance(request, dict) else None
        try:
            if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
                raise A2AError("invalid JSON-RPC 2.0 request", jsonrpc_code=-32600)
            method = request.get("method")
            params = request.get("params", {})
            if not isinstance(method, str) or not method:
                raise A2AError("JSON-RPC method is required", jsonrpc_code=-32600)
            if not isinstance(params, dict):
                raise A2AError("JSON-RPC params must be an object")

            if method == "SendMessage":
                result = await self.send_message(params)
            elif method == "GetTask":
                result = await self.get_task(params.get("id", ""))
            elif method == "ListTasks":
                result = await self.list_tasks(
                    context_id=params.get("contextId", ""),
                    status=params.get("status", ""),
                    page_size=params.get("pageSize", 50),
                    page_token=params.get("pageToken", ""),
                )
            elif method == "CancelTask":
                result = await self.cancel_task(params.get("id", ""))
            elif method in {"SendStreamingMessage", "SubscribeToTask", "tasks/sendSubscribe"}:
                raise A2AError(
                    "streaming is not supported by this agent",
                    status_code=501,
                    jsonrpc_code=-32004,
                    reason="UNSUPPORTED_OPERATION",
                )
            elif method == "tasks/send":
                result = await self._tasks_send(params)
            elif method == "tasks/get":
                result = await self._tasks_get(params)
            elif method == "tasks/cancel":
                result = await self._tasks_cancel(params)
            else:
                raise A2AError(
                    f"method not found: {method}",
                    status_code=404,
                    jsonrpc_code=-32601,
                    reason="METHOD_NOT_FOUND",
                )
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except asyncio.CancelledError:
            raise
        except A2AError as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": exc.jsonrpc_code,
                    "message": str(exc),
                    "data": {"reason": exc.reason},
                },
            }
        except Exception:  # noqa: BLE001
            log.exception("unhandled A2A JSON-RPC failure")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": "Internal server error"},
            }

    async def _tasks_send(self, params: dict[str, Any]) -> dict[str, Any]:
        """Legacy local method retained for pre-1.0 callers."""
        wrapped = dict(params)
        legacy_task_id = wrapped.pop("id", None)
        message = wrapped.get("message")
        if isinstance(message, dict):
            normalized = dict(message)
            if isinstance(legacy_task_id, str) and legacy_task_id in self._tasks:
                normalized["taskId"] = legacy_task_id
            parts = normalized.get("parts")
            if isinstance(parts, list):
                normalized["parts"] = [
                    {"text": part.get("text", "")}
                    if isinstance(part, dict) and part.get("type") == "text"
                    else part
                    for part in parts
                ]
            wrapped["message"] = normalized
        response = await self.send_message(wrapped)
        task_id = response["task"]["id"]
        return self._tasks[task_id].to_legacy_dict()

    async def _tasks_get(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id = _validated_identifier(params.get("id"), label="task id", required=True)
        await self.get_task(task_id)
        return self._tasks[task_id].to_legacy_dict()

    async def _tasks_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id = _validated_identifier(params.get("id"), label="task id", required=True)
        await self.cancel_task(task_id)
        return self._tasks[task_id].to_legacy_dict()

    async def _execute_task(
        self,
        user_text: str,
        task_id: str,
        context_id: str,
        message_id: str,
    ) -> str:
        if self.runtime is None or not callable(getattr(self.runtime, "_handle_turn", None)):
            raise RuntimeError("Norax runtime is unavailable")
        outbound = getattr(self.runtime, "outbound", None)
        if (
            outbound is None
            or not callable(getattr(outbound, "has", None))
            or not outbound.has("a2a")
        ):
            raise RuntimeError("A2A outbound adapter is unavailable")

        from ..envelope import Principal, SensoryInput, ThreadBinding

        loop = asyncio.get_running_loop()
        reply_future: asyncio.Future[str] = loop.create_future()
        self._reply_futures[task_id] = reply_future
        self._reply_chunks[task_id] = []
        try:
            env = SensoryInput(
                channel="chat",
                source="a2a",
                message_id=message_id,
                timestamp=datetime.now(UTC),
                sender=Principal(
                    id="a2a-remote",
                    label="A2A Remote Agent",
                    trust=False,
                    tier="guest",
                ),
                body=user_text,
                raw={"channel_id": context_id},
                trusted=False,
                thread_binding=ThreadBinding(
                    channel="a2a",
                    thread_id=task_id,
                    kind="custom",
                    session_id=context_id,
                ),
                metadata={"a2a": True, "a2a_task_id": task_id, "a2a_context_id": context_id},
            )
            await self.runtime._handle_turn(env)
            if not reply_future.done():
                raise RuntimeError("runtime completed without an outbound A2A reply")
            reply_future.result()
            chunks = self._reply_chunks[task_id]
            result = "\n\n".join(chunks).strip()
            if not result:
                raise RuntimeError("runtime produced an empty A2A reply")
            return result
        finally:
            future = self._reply_futures.pop(task_id, None)
            if future is not None and not future.done():
                future.cancel()
            self._reply_chunks.pop(task_id, None)

    def task_snapshot(self) -> list[dict[str, Any]]:
        """Return a read-only protocol snapshot for diagnostics and tests."""
        return [task.to_dict() for task in self._tasks.values()]

    async def close(self) -> None:
        """Stop owned workers and detach the runtime outbound adapter."""
        if self._closed:
            return
        self._closed = True
        workers = list(self._task_futures.values())
        for worker in workers:
            if not worker.done():
                worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        for future in self._reply_futures.values():
            if not future.done():
                future.cancel()
        self._reply_futures.clear()
        self._reply_chunks.clear()
        self._task_futures.clear()
        registry = self._outbound_registry
        adapter = self._outbound_adapter
        if registry is not None and adapter is not None:
            unregister = getattr(registry, "unregister", None)
            if callable(unregister):
                unregister("a2a", expected=adapter)
        self._outbound_registry = None
        self._outbound_adapter = None


def create_a2a_app(server: NoraxA2AServer) -> Any:
    """Build the FastAPI transport without starting a listener."""
    from fastapi import Depends, FastAPI, Header, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Norax A2A", version=PROTOCOL_VERSION)
    card = server.get_agent_card().to_dict()
    card_etag = f'"{hashlib.sha256(str(card).encode()).hexdigest()[:24]}"'

    def response(content: Any, *, status_code: int = 200) -> JSONResponse:
        return JSONResponse(content, status_code=status_code, media_type=MEDIA_TYPE)

    def authorized(
        authorization: str | None = Header(default=None),
        a2a_version: str | None = Header(default=None, alias="A2A-Version"),
    ) -> None:
        server.authorize(authorization)
        if a2a_version not in {None, "", PROTOCOL_VERSION}:
            raise A2AError(
                f"A2A protocol version {a2a_version} is not supported",
                status_code=400,
                jsonrpc_code=-32006,
                reason="VERSION_NOT_SUPPORTED",
            )

    @app.exception_handler(A2AError)
    async def _a2a_error(_request: Request, exc: A2AError) -> JSONResponse:
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        return JSONResponse(
            exc.problem(),
            status_code=exc.status_code,
            headers=headers,
            media_type="application/problem+json",
        )

    @app.get("/.well-known/agent-card.json")
    @app.get("/.well-known/agent-card", include_in_schema=False)
    async def _agent_card() -> JSONResponse:
        return JSONResponse(
            card,
            headers={"Cache-Control": "public, max-age=300", "ETag": card_etag},
            media_type=MEDIA_TYPE,
        )

    @app.post("/message:send")
    async def _send_message(
        request: dict[str, Any], _authorized: None = Depends(authorized)
    ) -> JSONResponse:
        return response(await server.send_message(request))

    @app.get("/tasks/{task_id}")
    async def _get_task(task_id: str, _authorized: None = Depends(authorized)) -> JSONResponse:
        return response(await server.get_task(task_id))

    @app.get("/tasks")
    async def _list_tasks(
        contextId: str = "",  # noqa: N803 - protocol spelling
        status: str = "",
        pageSize: int = 50,  # noqa: N803 - protocol spelling
        pageToken: str = "",  # noqa: N803 - protocol spelling
        _authorized: None = Depends(authorized),
    ) -> JSONResponse:
        return response(
            await server.list_tasks(
                context_id=contextId,
                status=status,
                page_size=pageSize,
                page_token=pageToken,
            )
        )

    @app.post("/tasks/{task_id}:cancel")
    async def _cancel_task(task_id: str, _authorized: None = Depends(authorized)) -> JSONResponse:
        return response(await server.cancel_task(task_id))

    @app.post("/message:stream")
    @app.post("/tasks/{task_id}:subscribe")
    async def _unsupported_stream(
        task_id: str = "",
        _authorized: None = Depends(authorized),
    ) -> JSONResponse:
        del task_id
        raise A2AError(
            "streaming is not supported by this agent",
            status_code=501,
            jsonrpc_code=-32004,
            reason="UNSUPPORTED_OPERATION",
        )

    @app.post("/")
    @app.post("/jsonrpc")
    async def _jsonrpc(
        request: dict[str, Any], _authorized: None = Depends(authorized)
    ) -> JSONResponse:
        return response(await server.handle_jsonrpc(request))

    return app
