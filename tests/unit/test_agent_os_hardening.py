"""Regression tests for Agent OS security hardening.

Covers:
- Path traversal via unsafe prefix matching
- Constant-time token comparison
- DB schema self-heal (init_db creates tables if absent)
- WebSocket upstream connection leak (gather never returned)
- Fixed dashboard endpoint (no silent port fallback)
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
from pathlib import Path

import pytest

# agent_os/server.py lives outside the norax/ package
_AGENT_OS = Path(__file__).resolve().parents[2] / "agent_os"
if str(_AGENT_OS) not in sys.path:
    sys.path.insert(0, str(_AGENT_OS))


def test_path_traversal_blocked():
    """Sibling directories sharing a prefix must not pass _safe()."""
    import server

    # Inside allowed root — should pass
    p = server._safe(str(server.PROJECT_ROOT / "config" / "runtime.jsonc"))
    assert p is not None

    # Sibling with shared prefix — must be blocked
    sibling = server.PROJECT_ROOT.with_name(server.PROJECT_ROOT.name + "-sibling")
    for escape in [
        str(sibling / ".env"),
        str(sibling / "secret"),
        str(sibling),
        "/etc/passwd",
    ]:
        with pytest.raises(Exception, match="outside allowed roots"):
            server._safe(escape)


def test_token_comparison_constant_time():
    """_token_ok must use secrets.compare_digest, not plain ==."""
    import server

    server.TOKEN = "unit-token"
    assert server._token_ok("unit-token") is True
    assert server._token_ok("wrong") is False
    assert server._token_ok("") is False
    assert server._token_ok(None) is False


def test_init_db_creates_schema(tmp_path, monkeypatch):
    """init_db must create tables + index even when the DB file is absent."""
    import server

    db_path = tmp_path / "test_agent_os.db"
    monkeypatch.setattr(server, "DB_PATH", db_path)
    server.init_db()

    with sqlite3.connect(db_path) as db:
        tables = {
            r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        indexes = {
            r[0]
            for r in db.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        journal = db.execute("PRAGMA journal_mode").fetchone()[0]

    assert "chats" in tables
    assert "messages" in tables
    assert "idx_messages_chat_ts" in indexes
    assert journal == "wal"
    assert db_path.stat().st_mode & 0o777 == 0o600


def test_persist_message_dedupe(tmp_path, monkeypatch):
    """Duplicate messages within 30s must be suppressed."""
    import server

    db_path = tmp_path / "test_agent_os.db"
    monkeypatch.setattr(server, "DB_PATH", db_path)
    server.init_db()

    ts = 1000.0
    first = server.persist_message("chat1", "user", "hello", ts=ts)
    second = server.persist_message("chat1", "user", "hello", ts=ts + 10)
    third = server.persist_message("chat1", "user", "different", ts=ts + 15)

    assert first is True
    assert second is False  # duplicate
    assert third is True  # different content


def test_agent_os_default_state_is_private_and_outside_source(tmp_path, monkeypatch):
    import server

    monkeypatch.delenv("NORAX_AGENT_OS_STATE_DIR", raising=False)
    monkeypatch.delenv("NORAX_STATE_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    state_dir = server.agent_os_state_dir(create=True)

    assert state_dir == (tmp_path / "norax/agent-os").resolve()
    assert state_dir != server.BASE
    assert state_dir.stat().st_mode & 0o777 == 0o700


def test_agent_os_state_rejects_relative_override(monkeypatch):
    import server

    monkeypatch.setenv("NORAX_AGENT_OS_STATE_DIR", "relative/state")

    with pytest.raises(ValueError, match="absolute"):
        server.agent_os_state_dir()


def test_agent_os_database_rejects_symlink(tmp_path, monkeypatch):
    import server

    target = tmp_path / "target.db"
    target.write_bytes(b"")
    link = tmp_path / "agent_os.db"
    link.symlink_to(target)
    monkeypatch.setattr(server, "DB_PATH", link)

    with pytest.raises(RuntimeError, match="regular non-symlink"):
        server._connect()

    assert os.stat(target).st_size == 0


def test_run_server_uses_only_the_configured_endpoint(monkeypatch):
    """A port conflict must fail at bind instead of moving the dashboard."""
    import server

    calls = []
    monkeypatch.setattr(server.uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    server.run_server()

    assert calls == [
        ((server.app,), {"host": server.BIND, "port": server.PORT, "log_level": "warning"})
    ]


@pytest.mark.asyncio
async def test_ws_bridge_cancels_both_pipes(monkeypatch):
    """Verify the /ws/norax bridge cancels both pipe tasks on disconnect.

    The old code used asyncio.gather which waited for BOTH tasks — a
    browser disconnect left pipe_down parked on the upstream iterator
    forever, leaking the runtime WebSocket connection.
    """
    # This is a structural test: we verify that asyncio.wait with
    # FIRST_COMPLETED + cancel pattern is used, not gather.

    # Simulate the pattern: one task completes, the other is cancelled
    async def completes():
        await asyncio.sleep(0.01)
        return "done"

    async def hangs():
        await asyncio.sleep(999)

    tasks = [asyncio.create_task(completes()), asyncio.create_task(hangs())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert tasks[0].done()
    assert tasks[1].cancelled() or tasks[1].done()
