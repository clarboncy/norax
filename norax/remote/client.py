"""Control-side client for remote nodes.

Supports local subprocess nodes, in-process relay jobs, and authenticated HTTP
submission to the standalone relay service.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .protocol import read_private_token
from .registry import RemoteRegistry

_DEFAULT_RELAY = "http://127.0.0.1:8765/remote"
_MAX_JOB_BYTES = 1 * 1024 * 1024
_MAX_RESULT_BYTES = 2 * 1024 * 1024


def normalize_relay_url(url: str | None) -> str:
    """Ensure relay base ends with /remote (FastAPI mount prefix)."""
    if url is not None and not isinstance(url, str):
        raise TypeError("relay URL must be a string")
    u = (url or "").strip().rstrip("/")
    if not u:
        return _DEFAULT_RELAY
    if len(u) > 4_096:
        raise ValueError("relay URL exceeds 4096 characters")
    parsed = urlsplit(u)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("relay URL must be a credential-free HTTP(S) base URL")
    path = parsed.path.rstrip("/")
    if not path.endswith("/remote"):
        path = f"{path}/remote"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


def _validated_job(job: object) -> tuple[dict[str, Any], bytes]:
    if not isinstance(job, dict):
        raise TypeError("remote job must be an object")
    try:
        payload = json.dumps(
            job,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("remote job must contain finite JSON data") from exc
    if len(payload) > _MAX_JOB_BYTES:
        raise ValueError(f"remote job exceeds {_MAX_JOB_BYTES} bytes")
    return job, payload


def _job_timeout(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("remote job timeout must be a number")
    timeout = float(value)
    if not math.isfinite(timeout):
        raise ValueError("remote job timeout must be finite")
    return min(max(timeout, 1.0), 300.0)


def resolve_relay_url(explicit: str | None = None) -> str:
    """Pick relay HTTP base: explicit arg > env > loopback default."""
    if explicit is not None:
        return normalize_relay_url(explicit) if explicit else ""
    env = os.environ.get("NORAX_REMOTE_RELAY_URL", "").strip()
    if env:
        return normalize_relay_url(env)
    return _DEFAULT_RELAY


class RemoteClient:
    def __init__(
        self,
        registry: RemoteRegistry | None = None,
        *,
        relay_url: str | None = None,
        control_token: str | None = None,
    ) -> None:
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
        self.relay_url = resolve_relay_url(relay_url)
        resolved_token = (
            control_token
            if control_token is not None
            else os.environ.get("NORAX_REMOTE_CONTROL_TOKEN") or self._read_control_token()
        )
        if not isinstance(resolved_token, str):
            raise TypeError("remote control token must be a string")
        self.control_token = resolved_token.strip()
        if self.control_token and (
            len(self.control_token) > 4_096
            or any(character.isspace() for character in self.control_token)
        ):
            raise ValueError("remote control token must be one whitespace-free value")

    @staticmethod
    def _read_control_token() -> str:
        remote_root = Path(
            os.environ.get(
                "NORAX_REMOTE_ROOT",
                Path(os.environ.get("NORAX_MEMORY_ROOT", Path.home() / ".local/share/norax/memory"))
                / "remote",
            )
        )
        path = Path(
            os.environ.get("NORAX_REMOTE_CONTROL_TOKEN_FILE", remote_root / "control.token")
        )
        try:
            return read_private_token(path)
        except (OSError, UnicodeError, ValueError):
            return ""

    async def run_job(self, node_id: str, job: dict[str, Any]) -> dict[str, Any]:
        try:
            job, encoded_job = _validated_job(job)
            timeout = _job_timeout(job.get("timeout"), default=45.0)
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": "invalid_remote_job", "detail": str(exc)}
        node = await asyncio.to_thread(self.registry.get, node_id)
        if node is None:
            return {"ok": False, "error": f"unknown node: {node_id}"}
        action = job.get("action")
        if not isinstance(action, str) or not action:
            return {"ok": False, "error": "remote job action must be a non-empty string"}
        if action not in node.capabilities and action != "hello":
            return {"ok": False, "error": f"capability not allowed: {action}"}

        if node.meta.get("adapter") == "local":
            cmd = [sys.executable, "-m", "norax.remote.node"]
            for root in node.roots:
                cmd += ["--root", root]
            if "exec" in node.capabilities:
                cmd.append("--allow-exec")
            cmd += ["--once", "-"]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(Path.cwd()),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(
                    proc.communicate(encoded_job),
                    timeout=timeout + 5.0,
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return {
                    "ok": False,
                    "error": "node subprocess timed out",
                    "timeout_s": timeout + 5.0,
                }
            if proc.returncode != 0:
                return {
                    "ok": False,
                    "error": "node subprocess failed",
                    "stderr": err.decode(errors="replace")[:4000],
                }
            try:
                if len(out) > _MAX_RESULT_BYTES:
                    raise ValueError("node result exceeds size limit")
                res = json.loads(out.decode())
                if not isinstance(res, dict):
                    raise ValueError("node result must be an object")
            except (UnicodeError, ValueError, json.JSONDecodeError):
                res = {
                    "ok": False,
                    "error": "bad node json",
                    "stdout": out.decode(errors="replace")[:4000],
                }
            await asyncio.to_thread(self.registry.record_job, node_id, job, res)
            return res

        if self.relay_url:
            http_timeout = timeout + 15.0
            headers = (
                {"Authorization": f"Bearer {self.control_token}"} if self.control_token else {}
            )
            async with httpx.AsyncClient(timeout=http_timeout) as c:
                resp = await c.post(
                    f"{self.relay_url}/job",  # base already includes /remote
                    headers=headers,
                    json={"node_id": node_id, "job": job, "timeout": timeout},
                )
                if resp.status_code >= 400:
                    return {
                        "ok": False,
                        "error": f"relay HTTP {resp.status_code}",
                        "body": resp.text[:1000],
                    }
                try:
                    if len(resp.content) > _MAX_RESULT_BYTES:
                        raise ValueError("relay result exceeds size limit")
                    data = resp.json()
                except (ValueError, json.JSONDecodeError):
                    return {"ok": False, "error": "invalid relay JSON result"}
                if not isinstance(data, dict):
                    return {"ok": False, "error": "invalid relay result"}
                result = data.get("result", data)
                return (
                    result
                    if isinstance(result, dict)
                    else {
                        "ok": False,
                        "error": "invalid relay result payload",
                    }
                )

        from .relay import RELAY

        return await RELAY.run_job(node_id, job, timeout=timeout + 5.0)
