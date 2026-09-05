"""Dispatch tool bindings for remote nodes."""

from __future__ import annotations

import shlex
from pathlib import Path

from .client import RemoteClient
from .registry import RemoteRegistry


def _registry() -> RemoteRegistry:
    return RemoteRegistry()


async def t_remote_enroll(*, name: str, root: str | None = None) -> dict:
    reg = _registry()
    info, token = reg.enroll(name, roots=[root] if root else [])
    selected_root = root or str(Path.home())
    quoted_root = shlex.quote(selected_root)
    quoted_token = shlex.quote(token)
    quoted_node = shlex.quote(info.node_id)
    return {
        "ok": True,
        "node_id": info.node_id,
        "name": info.name,
        "token": token,
        "install_hint": f"norax-node --root {quoted_root} --allow-exec",
        "relay_hint": (
            "norax-node --relay ws://<norax-host>:8765/node "
            f"--node-id {quoted_node} --token {quoted_token} "
            f"--root {quoted_root} --allow-exec"
        ),
    }


async def t_remote_list_nodes() -> dict:
    nodes = _registry().list_nodes()
    return {
        "ok": True,
        "nodes": [
            {
                "node_id": n.node_id,
                "name": n.name,
                "enabled": n.enabled,
                "capabilities": n.capabilities,
                "roots": n.roots,
                "last_seen_ms": n.last_seen_ms,
                "meta": n.meta,
            }
            for n in nodes
        ],
    }


_REMOTE_JOB_TIMEOUT = 45.0


async def t_remote_exec(
    *, node_id: str, command: str, cwd: str | None = None, timeout: float = 30.0
) -> dict:  # noqa: ASYNC109
    return await RemoteClient().run_job(
        node_id,
        {"action": "exec", "command": command, "cwd": cwd, "timeout": timeout},
    )


async def t_remote_read(
    *, node_id: str, path: str, limit: int | None = None, offset: int = 0
) -> dict:
    return await RemoteClient().run_job(
        node_id,
        {
            "action": "read",
            "path": path,
            "limit": limit or 20000,
            "offset": offset,
            "timeout": _REMOTE_JOB_TIMEOUT,
        },
    )


async def t_remote_list(*, node_id: str, path: str) -> dict:
    return await RemoteClient().run_job(
        node_id,
        {"action": "list", "path": path, "timeout": _REMOTE_JOB_TIMEOUT},
    )


async def t_remote_write(*, node_id: str, path: str, content: str) -> dict:
    return await RemoteClient().run_job(
        node_id, {"action": "write", "path": path, "content": content}
    )
