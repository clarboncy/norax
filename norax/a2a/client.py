"""Client for the Agent2Agent 1.0 HTTP+JSON binding."""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from .server import MAX_IDENTIFIER_CHARS, MAX_MESSAGE_CHARS, MEDIA_TYPE

_STATE_NAMES = {
    "TASK_STATE_SUBMITTED": "submitted",
    "TASK_STATE_WORKING": "working",
    "TASK_STATE_COMPLETED": "completed",
    "TASK_STATE_FAILED": "failed",
    "TASK_STATE_CANCELED": "canceled",
    "TASK_STATE_REJECTED": "rejected",
    "TASK_STATE_INPUT_REQUIRED": "input-required",
    "TASK_STATE_AUTH_REQUIRED": "auth-required",
}
_MAX_SSE_EVENT_CHARS = 512_000


@dataclass(frozen=True)
class A2ATaskResult:
    """Stable client-side view of a task or message response."""

    id: str
    status: str
    result: str
    error: str = ""
    context_id: str = ""


def _validate_base_url(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("base_url must be a string")
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base_url must be an HTTP(S) origin or base path without credentials")
    return normalized


def _validate_identifier(value: str | None, *, label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    value = value.strip()
    if not value or len(value) > MAX_IDENTIFIER_CHARS:
        raise ValueError(f"{label} must be non-empty and at most {MAX_IDENTIFIER_CHARS} characters")
    return value


def _validate_message(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("message must be a string")
    value = value.strip()
    if not value:
        raise ValueError("message must not be empty")
    if len(value) > MAX_MESSAGE_CHARS:
        raise ValueError(f"message exceeds the {MAX_MESSAGE_CHARS} character limit")
    return value


def _parts_text(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    texts = [
        part["text"].strip()
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip()
    ]
    return "\n\n".join(texts)


def _task_result(data: Any, *, fallback_id: str = "") -> A2ATaskResult:
    if not isinstance(data, dict):
        return A2ATaskResult(fallback_id, "error", "", "invalid A2A response body")

    protocol_error = data.get("error")
    if isinstance(protocol_error, dict):
        return A2ATaskResult(
            fallback_id,
            "error",
            "",
            str(protocol_error.get("message") or "A2A request failed"),
        )

    task = data.get("task")
    if not isinstance(task, dict) and isinstance(data.get("result"), dict):
        nested = data["result"]
        task = nested.get("task") if isinstance(nested.get("task"), dict) else nested
    if isinstance(task, dict):
        status_value = task.get("status", "unknown")
        status_message: Any = None
        if isinstance(status_value, dict):
            status_message = status_value.get("message")
            status_value = status_value.get("state", "unknown")
        status = _STATE_NAMES.get(str(status_value), str(status_value).lower())
        artifacts = task.get("artifacts")
        artifact_text: list[str] = []
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if isinstance(artifact, dict):
                    text = _parts_text(artifact.get("parts"))
                    if text:
                        artifact_text.append(text)
        result_text = "\n\n".join(artifact_text) or str(task.get("result") or "")
        error = str(task.get("error") or "")
        if not error and status in {"failed", "rejected"} and isinstance(status_message, dict):
            error = _parts_text(status_message.get("parts"))
        return A2ATaskResult(
            id=str(task.get("id") or fallback_id),
            status=status,
            result=result_text,
            error=error,
            context_id=str(task.get("contextId") or ""),
        )

    message = data.get("message")
    if isinstance(message, dict):
        return A2ATaskResult(
            id=str(message.get("messageId") or fallback_id),
            status="completed",
            result=_parts_text(message.get("parts")),
            context_id=str(message.get("contextId") or ""),
        )

    status_update = data.get("statusUpdate")
    if isinstance(status_update, dict):
        status_value = status_update.get("status", {})
        state = (
            status_value.get("state", "unknown") if isinstance(status_value, dict) else status_value
        )
        return A2ATaskResult(
            id=str(status_update.get("taskId") or fallback_id),
            status=_STATE_NAMES.get(str(state), str(state).lower()),
            result="",
            context_id=str(status_update.get("contextId") or ""),
        )

    artifact_update = data.get("artifactUpdate")
    if isinstance(artifact_update, dict):
        artifact = artifact_update.get("artifact", {})
        text = _parts_text(artifact.get("parts")) if isinstance(artifact, dict) else ""
        return A2ATaskResult(
            id=str(artifact_update.get("taskId") or fallback_id),
            status="working",
            result=text,
            context_id=str(artifact_update.get("contextId") or ""),
        )

    return A2ATaskResult(fallback_id, "error", "", "unrecognized A2A response body")


class A2AClient:
    """Validated async client for A2A HTTP+JSON agents."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        *,
        token: str | None = None,
    ) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
            or timeout > 3_600
        ):
            raise ValueError("timeout must be finite and between 0 and 3600 seconds")
        self.base_url = _validate_base_url(base_url)
        self.timeout = float(timeout)
        self.token = (token or "").strip() or None
        self._client: httpx.AsyncClient | None = None

    def _headers(self, *, accept: str = MEDIA_TYPE) -> dict[str, str]:
        headers = {"Accept": accept, "Content-Type": MEDIA_TYPE, "A2A-Version": "1.0"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def __aenter__(self) -> A2AClient:
        await self._get_client()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_agent_card(self) -> dict[str, Any]:
        """Fetch the standard Agent Card, falling back only for legacy servers."""
        client = await self._get_client()
        response = await client.get(
            f"{self.base_url}/.well-known/agent-card.json",
            headers=self._headers(),
        )
        if response.status_code == 404:
            response = await client.get(
                f"{self.base_url}/.well-known/agent-card",
                headers=self._headers(),
            )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Agent Card response must be a JSON object")
        return data

    async def send_task(
        self,
        message: str,
        *,
        task_id: str | None = None,
        context_id: str | None = None,
        skill_id: str | None = None,
        return_immediately: bool = False,
    ) -> A2ATaskResult:
        """Initiate or continue an A2A task."""
        message = _validate_message(message)
        task_id = _validate_identifier(task_id, label="task_id")
        context_id = _validate_identifier(context_id, label="context_id")
        skill_id = _validate_identifier(skill_id, label="skill_id")
        if not isinstance(return_immediately, bool):
            raise TypeError("return_immediately must be boolean")

        message_object: dict[str, Any] = {
            "messageId": str(uuid.uuid4()),
            "role": "ROLE_USER",
            "parts": [{"text": message}],
        }
        if task_id:
            message_object["taskId"] = task_id
        if context_id:
            message_object["contextId"] = context_id
        if skill_id:
            message_object["metadata"] = {"skillId": skill_id}
        payload = {
            "message": message_object,
            "configuration": {"returnImmediately": return_immediately},
        }

        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/message:send",
            json=payload,
            headers=self._headers(),
        )
        if response.is_error:
            return self._error_result(response, fallback_id=task_id)
        return _task_result(response.json(), fallback_id=task_id)

    async def get_task(self, task_id: str) -> A2ATaskResult:
        task_id = _validate_identifier(task_id, label="task_id")
        client = await self._get_client()
        response = await client.get(
            f"{self.base_url}/tasks/{quote(task_id, safe='')}",
            headers=self._headers(),
        )
        if response.is_error:
            return self._error_result(response, fallback_id=task_id)
        return _task_result(response.json(), fallback_id=task_id)

    async def cancel_task(self, task_id: str) -> A2ATaskResult:
        task_id = _validate_identifier(task_id, label="task_id")
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/tasks/{quote(task_id, safe='')}:cancel",
            headers=self._headers(),
        )
        if response.is_error:
            return self._error_result(response, fallback_id=task_id)
        return _task_result(response.json(), fallback_id=task_id)

    @staticmethod
    def _error_result(response: httpx.Response, *, fallback_id: str = "") -> A2ATaskResult:
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        detail = body.get("detail") if isinstance(body, dict) else None
        error = str(detail or f"A2A request failed with HTTP {response.status_code}")
        return A2ATaskResult(fallback_id, "error", "", error)

    async def send_task_streaming(
        self,
        message: str,
        *,
        task_id: str | None = None,
        context_id: str | None = None,
    ) -> AsyncGenerator[A2ATaskResult, None]:
        """Consume a remote agent's A2A SSE stream when it supports streaming."""
        message = _validate_message(message)
        task_id = _validate_identifier(task_id, label="task_id")
        context_id = _validate_identifier(context_id, label="context_id")
        message_object: dict[str, Any] = {
            "messageId": str(uuid.uuid4()),
            "role": "ROLE_USER",
            "parts": [{"text": message}],
        }
        if task_id:
            message_object["taskId"] = task_id
        if context_id:
            message_object["contextId"] = context_id

        client = await self._get_client()
        async with client.stream(
            "POST",
            f"{self.base_url}/message:stream",
            json={"message": message_object},
            headers=self._headers(accept="text/event-stream"),
        ) as response:
            response.raise_for_status()
            data_lines: list[str] = []
            async for line in response.aiter_lines():
                if not line:
                    if data_lines:
                        yield self._parse_sse_event("\n".join(data_lines), task_id)
                        data_lines.clear()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                    if sum(len(item) for item in data_lines) > _MAX_SSE_EVENT_CHARS:
                        yield A2ATaskResult(
                            task_id,
                            "error",
                            "",
                            "A2A stream event exceeded the size limit",
                        )
                        data_lines.clear()
            if data_lines:
                yield self._parse_sse_event("\n".join(data_lines), task_id)

    @staticmethod
    def _parse_sse_event(payload: str, fallback_id: str = "") -> A2ATaskResult:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return A2ATaskResult(fallback_id, "error", "", "invalid JSON in A2A stream")
        return _task_result(data, fallback_id=fallback_id)
