import asyncio
import json
import shlex
import sys
from pathlib import Path

import pytest

from norax.remote.client import RemoteClient, normalize_relay_url, resolve_relay_url
from norax.remote.node import NodeWorker, _node_ws_url
from norax.remote.registry import RegistryCorruptError, RemoteRegistry


@pytest.mark.asyncio
async def test_node_worker_file_ops_and_exec(tmp_path: Path):
    w = NodeWorker(roots=[str(tmp_path)], allow_exec=True)
    assert (await w.run({"action": "write", "path": str(tmp_path / "a.txt"), "content": "hi"}))[
        "ok"
    ]
    r = await w.run({"action": "read", "path": str(tmp_path / "a.txt")})
    assert r["content"] == "hi"
    ls = await w.run({"action": "list", "path": str(tmp_path)})
    assert any(i["name"] == "a.txt" for i in ls["items"])
    ex = await w.run({"action": "exec", "command": "printf ok", "cwd": str(tmp_path)})
    assert ex["stdout"] == "ok"


@pytest.mark.asyncio
async def test_node_worker_exec_is_explicit_and_bounded(tmp_path: Path, monkeypatch):
    disabled = NodeWorker(roots=[str(tmp_path)])
    denied = await disabled.run({"action": "exec", "command": "printf should-not-run"})
    assert denied["error"] == "exec_disabled"

    monkeypatch.setenv("NORAX_NODE_TOKEN", "must-not-leak")
    enabled = NodeWorker(roots=[str(tmp_path)], allow_exec=True, max_output=1024)
    hidden = await enabled.run(
        {"action": "exec", "command": 'printf %s "$NORAX_NODE_TOKEN"', "cwd": str(tmp_path)}
    )
    assert hidden["stdout"] == ""
    noisy = await enabled.run(
        {
            "action": "exec",
            "command": f"{shlex.quote(sys.executable)} -c 'print(\"x\" * 5000)'",
            "cwd": str(tmp_path),
        }
    )
    assert len(noisy["stdout"].encode()) <= 1024
    assert noisy["stdout_truncated"] is True

    with pytest.raises(TypeError, match="allow_exec"):
        NodeWorker(roots=[str(tmp_path)], allow_exec="false")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_node_worker_defaults_to_cwd_root_and_blocks_known_destructive_exec(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    worker = NodeWorker(allow_exec=True)
    outside = tmp_path.parent / "outside.txt"

    denied_path = await worker.run({"action": "read", "path": str(outside)})
    denied_command = await worker.run({"action": "exec", "command": "rm -rf ./cache"})

    assert denied_path["ok"] is False
    assert denied_path["type"] == "PermissionError"
    assert denied_command["error"] == "blocked_dangerous_command"


@pytest.mark.asyncio
async def test_node_file_boundary_rejects_symlinks_and_malformed_limits(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    alias = root / "alias.txt"
    alias.symlink_to(outside)
    worker = NodeWorker(roots=[str(root)])

    denied_read = await worker.run({"action": "read", "path": str(alias)})
    denied_write = await worker.run({"action": "write", "path": str(alias), "content": "overwrite"})
    bad_limit = await worker.run({"action": "read", "path": str(root / "missing"), "limit": True})
    bad_content = await worker.run(
        {"action": "write", "path": str(root / "new"), "content": {"not": "text"}}
    )

    assert denied_read["ok"] is False
    assert denied_write["ok"] is False
    assert outside.read_text(encoding="utf-8") == "private"
    assert bad_limit["ok"] is False
    assert bad_content["ok"] is False


def test_node_websocket_url_does_not_embed_credential() -> None:
    url = _node_ws_url("https://relay.example/remote", "worker one")
    assert url == "wss://relay.example/remote/node?node_id=worker+one"
    assert "token" not in url
    assert _node_ws_url("wss://relay.example/remote", "worker") == (
        "wss://relay.example/remote/node?node_id=worker"
    )


@pytest.mark.asyncio
async def test_remote_client_local_adapter(tmp_path: Path, monkeypatch):
    reg = RemoteRegistry(tmp_path / "registry")
    info, _token = reg.enroll("local-test", roots=[str(tmp_path)])
    data = reg._load()
    data["nodes"][info.node_id]["meta"] = {"adapter": "local"}
    reg._save(data)
    client = RemoteClient(reg)
    res = await client.run_job(
        info.node_id, {"action": "exec", "command": "printf remote", "cwd": str(tmp_path)}
    )
    assert res["ok"] is True
    assert res["stdout"] == "remote"


def test_normalize_relay_url_adds_remote_suffix():
    assert normalize_relay_url("http://127.0.0.1:8765") == "http://127.0.0.1:8765/remote"
    assert normalize_relay_url("http://127.0.0.1:8765/remote") == "http://127.0.0.1:8765/remote"
    assert normalize_relay_url("") == "http://127.0.0.1:8765/remote"


def test_resolve_relay_url_explicit_empty_disables_http():
    assert resolve_relay_url("") == ""
    assert resolve_relay_url(None) == "http://127.0.0.1:8765/remote"


def test_registry_corruption_fails_closed(tmp_path: Path) -> None:
    registry = RemoteRegistry(tmp_path / "registry")
    registry.nodes_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(RegistryCorruptError):
        registry.list_nodes()
    with pytest.raises(RegistryCorruptError):
        registry.authenticate("anything", "anything")


def test_registry_rejects_symlink_and_nonboolean_authorization_state(tmp_path: Path) -> None:
    registry = RemoteRegistry(tmp_path / "registry")
    info, _ = registry.enroll("worker")
    payload = registry._load()
    payload["nodes"][info.node_id]["enabled"] = "false"
    registry.nodes_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RegistryCorruptError):
        registry.get(info.node_id)

    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"nodes": {}}), encoding="utf-8")
    registry.nodes_path.unlink()
    registry.nodes_path.symlink_to(outside)
    with pytest.raises(RegistryCorruptError):
        registry.list_nodes()


def test_registry_preserves_empty_capabilities_and_bounds_audit_payload(tmp_path: Path) -> None:
    registry = RemoteRegistry(tmp_path / "registry")
    info, token = registry.enroll("readless", capabilities=[])
    assert info.capabilities == []
    assert registry.authenticate(info.node_id, token) is not None
    with pytest.raises(ValueError, match="unknown remote capabilities"):
        registry.enroll("bad", capabilities=["root_everything"])

    registry.record_job(
        info.node_id,
        {"action": "write", "content": "private body", "token": "node-secret"},
        {"ok": True, "stdout": "x" * 3000},
    )
    audit = json.loads(registry.jobs_path.read_text(encoding="utf-8"))
    assert audit["job"]["token"] == "<REDACTED>"
    assert audit["job"]["content"]["bytes"] == len("private body")
    assert "private body" not in registry.jobs_path.read_text(encoding="utf-8")
    assert len(audit["result"]["stdout"]) < 2100

    cyclic: dict = {"token": "still-secret"}
    cyclic["self"] = cyclic
    registry.record_job(info.node_id, cyclic, {"ok": True, "score": float("nan")})
    rows = registry.jobs_path.read_text(encoding="utf-8").splitlines()
    second = json.loads(rows[-1])
    assert second["job"]["token"] == "<REDACTED>"
    assert second["job"]["self"] == "<CYCLE>"
    assert second["result"]["score"] == "nan"


@pytest.mark.asyncio
async def test_cancelled_poll_job_is_removed_from_queue(tmp_path: Path) -> None:
    from norax.remote.relay import RemoteRelay

    registry = RemoteRegistry(tmp_path / "registry")
    node, _ = registry.enroll("poll-worker")
    data = registry._load()
    data["nodes"][node.node_id]["meta"] = {"transport": "https-poll"}
    registry._save(data)
    relay = RemoteRelay(registry)
    task = asyncio.create_task(relay.run_job(node.node_id, {"action": "read", "path": "x"}))
    for _ in range(100):
        if relay.http_queues.get(node.node_id):
            break
        await asyncio.sleep(0.01)
    assert relay.http_queues[node.node_id]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert node.node_id not in relay.http_queues
    assert relay.pending == {}
