"""Outbound WebSocket relay for live remote nodes.

Nodes connect outbound to /remote/node with an enrollment token. Control clients
use authenticated HTTP endpoints to enroll nodes, list live nodes, and submit
jobs. This keeps remote PCs behind NAT/firewalls while giving Norax a durable
control plane.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import secrets
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .protocol import read_private_token
from .registry import RemoteRegistry


class EnrollRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    roots: list[str] = Field(default_factory=list)
    capabilities: list[str] | None = None


class JobRequest(BaseModel):
    node_id: str = Field(min_length=1, max_length=120)
    job: dict[str, Any]
    timeout: float = Field(default=60.0, ge=1.0, le=330.0, allow_inf_nan=False)


class PollRequest(BaseModel):
    node_id: str = Field(min_length=1, max_length=120)
    token: str = Field(min_length=1, max_length=4096)
    timeout: float = Field(default=25.0, ge=1.0, le=30.0, allow_inf_nan=False)


class ResultRequest(BaseModel):
    node_id: str = Field(min_length=1, max_length=120)
    token: str = Field(min_length=1, max_length=4096)
    job_id: str = Field(min_length=1, max_length=120)
    result: dict[str, Any]


_MAX_PENDING_JOBS = 1_024
_MAX_PENDING_PER_NODE = 128
_MAX_POLL_WAITERS_PER_NODE = 128
_MAX_JOB_BYTES = 1 * 1024 * 1024
_MAX_RESULT_BYTES = 2 * 1024 * 1024


def _bounded_payload(value: object, *, label: str, max_bytes: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} must be finite JSON data") from exc
    if len(encoded) > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes")
    return value


def _bounded_timeout(value: object, *, default: float, maximum: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout must be a number")
    timeout = float(value)
    if not math.isfinite(timeout):
        raise ValueError("timeout must be finite")
    return min(max(timeout, 1.0), maximum)


class RemoteRelay:
    def __init__(self, registry: RemoteRegistry | None = None) -> None:
        self.registry = registry or RemoteRegistry(
            Path(
                os.environ.get(
                    "NORAX_REMOTE_ROOT",
                    Path(
                        os.environ.get(
                            "NORAX_MEMORY_ROOT", Path.home() / ".local/share/norax/memory"
                        )
                    )
                    / "remote",
                )
            )
        )
        self.nodes: dict[str, WebSocket] = {}
        self.pending: dict[str, tuple[str, asyncio.Future]] = {}
        self.http_queues: dict[str, list[dict[str, Any]]] = {}
        self.poll_waiters: dict[str, list[asyncio.Future]] = {}
        self.lock = asyncio.Lock()

    async def connect_node(self, ws: WebSocket, node_id: str, token: str) -> None:
        node = await asyncio.to_thread(self.registry.authenticate, node_id, token)
        if node is None:
            await ws.close(code=4401)
            return
        await asyncio.to_thread(self.registry.touch, node_id)
        await ws.accept()
        backlog: list[dict[str, Any]] = []
        old: WebSocket | None = None
        async with self.lock:
            old = self.nodes.get(node_id)
            if old is not None:
                self._fail_pending_for_node(node_id, "node_reconnected")
            self.nodes[node_id] = ws
            q = self.http_queues.pop(node_id, [])
            backlog = list(q)
        if old is not None:
            try:
                await old.close(code=4000)
            except Exception:
                pass
        for index, item in enumerate(backlog):
            try:
                await ws.send_json({"type": "job", "job_id": item["job_id"], "job": item["job"]})
            except Exception:
                async with self.lock:
                    self.http_queues.setdefault(node_id, [])[0:0] = backlog[index:]
                break
        try:
            while True:
                msg = await ws.receive_json()
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") == "result":
                    job_id = str(msg.get("job_id", ""))
                    pending = self.pending.get(job_id)
                    if pending is None or pending[0] != node_id:
                        continue
                    _, fut = self.pending.pop(job_id)
                    if not fut.done():
                        raw_result = msg.get("result")
                        try:
                            result = _bounded_payload(
                                raw_result,
                                label="remote result",
                                max_bytes=_MAX_RESULT_BYTES,
                            )
                        except ValueError as exc:
                            result = {
                                "ok": False,
                                "error": "invalid_remote_result",
                                "detail": str(exc),
                            }
                        fut.set_result(result)
                elif msg.get("type") == "hello":
                    await asyncio.to_thread(
                        self.registry.touch,
                        node_id,
                        meta=msg.get("meta") if isinstance(msg.get("meta"), dict) else None,
                    )
        except WebSocketDisconnect:
            pass
        finally:
            async with self.lock:
                if self.nodes.get(node_id) is ws:
                    self.nodes.pop(node_id, None)
                    self._fail_pending_for_node(node_id, "node_disconnected")

    def _fail_pending_for_node(self, node_id: str, error: str) -> None:
        """Resolve jobs tied to a dead transport instead of waiting for timeout."""
        for job_id, (pending_node, fut) in list(self.pending.items()):
            if pending_node != node_id:
                continue
            self.pending.pop(job_id, None)
            if not fut.done():
                fut.set_result({"ok": False, "error": error, "job_id": job_id})

    def _remove_queued_job(self, node_id: str, job_id: str) -> None:
        queue = self.http_queues.get(node_id)
        if queue is None:
            return
        self.http_queues[node_id] = [item for item in queue if item.get("job_id") != job_id]
        if not self.http_queues[node_id]:
            self.http_queues.pop(node_id, None)

    async def run_job(
        self, node_id: str, job: dict[str, Any], timeout: float = 60.0
    ) -> dict[str, Any]:  # noqa: ASYNC109
        try:
            job = _bounded_payload(job, label="remote job", max_bytes=_MAX_JOB_BYTES)
            deadline = _bounded_timeout(timeout, default=60.0, maximum=330.0)
        except ValueError as exc:
            return {"ok": False, "error": "invalid_remote_job", "detail": str(exc)}
        node = await asyncio.to_thread(self.registry.get, node_id)
        if node is None:
            return {"ok": False, "error": f"unknown node: {node_id}"}
        action = job.get("action")
        if not isinstance(action, str) or not action:
            return {"ok": False, "error": "remote job action must be a non-empty string"}
        if action not in node.capabilities and action != "hello":
            return {"ok": False, "error": f"capability not allowed: {action}"}
        async with self.lock:
            if len(self.pending) >= _MAX_PENDING_JOBS:
                return {"ok": False, "error": "remote_job_capacity_reached"}
            node_pending = sum(
                1 for pending_node, _future in self.pending.values() if pending_node == node_id
            )
            if node_pending >= _MAX_PENDING_PER_NODE:
                return {"ok": False, "error": "remote_node_capacity_reached"}
            ws = self.nodes.get(node_id)
            job_id = str(uuid.uuid4())
            fut = asyncio.get_running_loop().create_future()
            self.pending[job_id] = (node_id, fut)
        if ws is None:
            meta = node.meta if isinstance(node.meta, dict) else {}
            uses_poll = str(meta.get("transport", "")).strip().lower() == "https-poll"
            if not uses_poll:
                self.pending.pop(job_id, None)
                return {
                    "ok": False,
                    "error": "node_offline",
                    "detail": "no live websocket; use remote_list_nodes or restart norax-node",
                    "node_id": node_id,
                }
            item = {"job_id": job_id, "job": job}
            async with self.lock:
                waiters = self.poll_waiters.get(node_id, [])
                while waiters and waiters[0].done():
                    waiters.pop(0)
                waiter = waiters.pop(0) if waiters else None
                if waiter is None:
                    queue = self.http_queues.setdefault(node_id, [])
                    if len(queue) >= _MAX_PENDING_PER_NODE:
                        self.pending.pop(job_id, None)
                        return {"ok": False, "error": "remote_node_queue_capacity_reached"}
                    queue.append(item)
                elif not waiter.done():
                    waiter.set_result(item)
        else:
            try:
                await ws.send_json({"type": "job", "job_id": job_id, "job": job})
            except Exception as exc:  # noqa: BLE001
                self.pending.pop(job_id, None)
                return {
                    "ok": False,
                    "error": "node_send_failed",
                    "detail": type(exc).__name__,
                    "job_id": job_id,
                }
        try:
            res = await asyncio.wait_for(fut, timeout=deadline)
        except TimeoutError:
            self.pending.pop(job_id, None)
            async with self.lock:
                self._remove_queued_job(node_id, job_id)
            return {"ok": False, "error": "remote job timeout", "job_id": job_id}
        except asyncio.CancelledError:
            self.pending.pop(job_id, None)
            async with self.lock:
                self._remove_queued_job(node_id, job_id)
            raise
        if not isinstance(res, dict):
            res = {"ok": False, "error": "invalid_remote_result"}
        await asyncio.to_thread(
            self.registry.record_job,
            node_id,
            job | {"job_id": job_id},
            res,
        )
        return res

    def live_set(self) -> set[str]:
        return set(self.nodes)

    def list_live(self) -> list[str]:
        return sorted(set(self.nodes) | set(self.http_queues))

    async def poll_job(self, node_id: str, token: str, timeout: float = 25.0) -> dict[str, Any]:  # noqa: ASYNC109
        try:
            deadline = _bounded_timeout(timeout, default=25.0, maximum=30.0)
        except ValueError as exc:
            return {"ok": False, "error": "invalid_poll_timeout", "detail": str(exc)}
        node = await asyncio.to_thread(self.registry.authenticate, node_id, token)
        if node is None:
            return {"ok": False, "error": "not authenticated"}
        await asyncio.to_thread(
            self.registry.touch,
            node_id,
            meta={"transport": "https-poll"},
        )
        async with self.lock:
            q = self.http_queues.setdefault(node_id, [])
            if q:
                return {"ok": True, "type": "job", **q.pop(0)}
            active_waiters = [
                waiter for waiter in self.poll_waiters.get(node_id, []) if not waiter.done()
            ]
            if len(active_waiters) >= _MAX_POLL_WAITERS_PER_NODE:
                return {"ok": False, "error": "poll_waiter_capacity_reached"}
            fut = asyncio.get_running_loop().create_future()
            active_waiters.append(fut)
            self.poll_waiters[node_id] = active_waiters
        try:
            item = await asyncio.wait_for(fut, timeout=deadline)
            return {"ok": True, "type": "job", **item}
        except TimeoutError:
            return {"ok": True, "type": "noop"}
        finally:
            async with self.lock:
                waiters = self.poll_waiters.get(node_id, [])
                if "fut" in locals() and fut in waiters:
                    waiters.remove(fut)

    async def submit_poll_result(
        self, node_id: str, token: str, job_id: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            result = _bounded_payload(
                result,
                label="remote result",
                max_bytes=_MAX_RESULT_BYTES,
            )
        except ValueError as exc:
            return {"ok": False, "error": "invalid_remote_result", "detail": str(exc)}
        node = await asyncio.to_thread(self.registry.authenticate, node_id, token)
        if node is None:
            return {"ok": False, "error": "not authenticated"}
        await asyncio.to_thread(
            self.registry.touch,
            node_id,
            meta={"transport": "https-poll"},
        )
        pending = self.pending.get(job_id)
        if pending is None:
            return {"ok": False, "error": "unknown_or_expired_job"}
        pending_node, fut = pending
        if pending_node != node_id:
            return {"ok": False, "error": "job_node_mismatch"}
        self.pending.pop(job_id, None)
        if not fut.done():
            fut.set_result(result)
        await asyncio.to_thread(
            self.registry.record_job,
            node_id,
            {"job_id": job_id, "transport": "https-poll"},
            result,
        )
        return {"ok": True}


RELAY = RemoteRelay()


def _control_token() -> str:
    token = os.environ.get("NORAX_REMOTE_CONTROL_TOKEN", "")
    if token:
        return token
    remote_root = Path(
        os.environ.get(
            "NORAX_REMOTE_ROOT",
            Path(os.environ.get("NORAX_MEMORY_ROOT", Path.home() / ".local/share/norax/memory"))
            / "remote",
        )
    )
    token_path = Path(
        os.environ.get("NORAX_REMOTE_CONTROL_TOKEN_FILE", remote_root / "control.token")
    )
    try:
        return read_private_token(token_path)
    except (OSError, UnicodeError, ValueError):
        return ""


def _require_control(auth: str | None) -> None:
    token = _control_token()
    if not token:
        raise HTTPException(503, "remote control token is not configured")
    expected = f"Bearer {token}"
    if not auth or not secrets.compare_digest(auth, expected):
        raise HTTPException(401, "not authenticated")


def _public_enrollment_enabled() -> bool:
    return os.environ.get("NORAX_REMOTE_ALLOW_PUBLIC_ENROLL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _connect_url(node_id: str) -> str:
    base = os.environ.get("NORAX_REMOTE_PUBLIC_URL", "").strip().rstrip("/")
    path = f"/remote/node?node_id={node_id}"
    return f"{base}{path}" if base else path


def app() -> FastAPI:
    api = FastAPI(title="Norax Remote Relay")

    async def _node_ws(ws: WebSocket) -> None:
        node_id = ws.query_params.get("node_id") or ""
        auth = ws.headers.get("authorization") or ""
        token = auth.removeprefix("Bearer ") if auth.startswith("Bearer ") else ""
        # Query-token fallback keeps existing nodes compatible. New clients use
        # the Authorization header so credentials do not land in URL logs.
        token = token or ws.query_params.get("token") or ""
        await RELAY.connect_node(ws, node_id, token)

    async def _health() -> dict[str, Any]:
        live = RELAY.list_live()
        return {"ok": True, "live_nodes": live, "live_count": len(live)}

    async def _nodes(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        await asyncio.to_thread(_require_control, authorization)
        live = RELAY.live_set()
        nodes = await asyncio.to_thread(RELAY.registry.list_nodes)
        return {
            "ok": True,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "name": n.name,
                    "enabled": n.enabled,
                    "live": n.node_id in live,
                    "capabilities": n.capabilities,
                    "roots": n.roots,
                    "last_seen_ms": n.last_seen_ms,
                    "meta": n.meta,
                }
                for n in nodes
            ],
        }

    async def _enroll(
        req: EnrollRequest, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        await asyncio.to_thread(_require_control, authorization)
        node, token = await asyncio.to_thread(
            RELAY.registry.enroll,
            req.name,
            roots=req.roots,
            capabilities=req.capabilities,
        )
        return {
            "ok": True,
            "node_id": node.node_id,
            "name": node.name,
            "token": token,
            "connect_url": _connect_url(node.node_id),
        }

    async def _public_enroll(req: EnrollRequest) -> dict[str, Any]:
        if not _public_enrollment_enabled():
            raise HTTPException(403, "public remote enrollment is disabled")
        node, token = await asyncio.to_thread(
            RELAY.registry.enroll,
            req.name,
            roots=req.roots,
            capabilities=req.capabilities,
        )
        return {
            "ok": True,
            "node_id": node.node_id,
            "name": node.name,
            "token": token,
            "connect_url": _connect_url(node.node_id),
        }

    async def _job(
        req: JobRequest, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        await asyncio.to_thread(_require_control, authorization)
        result = await RELAY.run_job(req.node_id, req.job, timeout=req.timeout)
        return {"ok": result.get("ok") is True, "result": result}

    async def _poll(req: PollRequest) -> dict[str, Any]:
        return await RELAY.poll_job(req.node_id, req.token, req.timeout)

    async def _result(req: ResultRequest) -> dict[str, Any]:
        return await RELAY.submit_poll_result(req.node_id, req.token, req.job_id, req.result)

    api.websocket("/node")(_node_ws)
    api.websocket("/remote/node")(_node_ws)
    api.get("/health")(_health)
    api.get("/remote/health")(_health)
    api.get("/remote/nodes")(_nodes)
    api.post("/remote/enroll")(_enroll)
    api.post("/remote/public/enroll")(_public_enroll)
    api.post("/remote/job")(_job)
    api.post("/remote/node/poll")(_poll)
    api.post("/remote/node/result")(_result)

    return api


def main() -> None:
    import uvicorn

    host = os.environ.get("NORAX_REMOTE_HOST", "127.0.0.1")
    port = int(os.environ.get("NORAX_REMOTE_PORT", "8765"))
    uvicorn.run(app(), host=host, port=port)


if __name__ == "__main__":
    main()
