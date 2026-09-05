"""Phase 8 integration — runtime starts cleanly with Discord disabled,
still serves HTTP ingress, emits replies only through registered outbounds.

No live Discord connection is established.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.subprocess_integration


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
async def test_runtime_with_discord_enabled_no_token_starts(tmp_path: Path):
    """Discord enabled=true but no token: runtime still starts, HTTP works,
    no discord adapter registered."""
    port = _free_port()
    work = tmp_path / "work"
    work.mkdir()
    # Create minimal soul files so startup validation passes
    soul_dir = work / "soul"
    soul_dir.mkdir()
    (soul_dir / "SOUL.md").write_text("# Soul\nTest soul.")
    (soul_dir / "IDENTITY.md").write_text("# Identity\nTest identity.")
    (soul_dir / "USER.md").write_text("# User\nTest user.")
    cfg = work / "runtime.jsonc"
    cfg.write_text(
        json.dumps(
            {
                "runtime": {"shutdown_grace_seconds": 5},
                "http": {"bind": f"127.0.0.1:{port}"},
                "owner": {"id": "111"},
                "paths": {
                    "soul": "soul/SOUL.md",
                    "identity": "soul/IDENTITY.md",
                    "user": "soul/USER.md",
                },
                "channels": {
                    "discord": {
                        "enabled": True,
                        "token": None,
                        "groupPolicy": "allowlist",
                        "guilds": {},
                    }
                },
            }
        )
    )

    # A subprocess integration test is a separate deployment. Never inherit
    # production Norax paths, ports, connectors, credentials, or lock files.
    env = {key: value for key, value in os.environ.items() if not key.startswith("NORAX_")}
    env["NORAX_CONFIG"] = str(cfg)
    env["NORAX_PROJECT_ROOT"] = str(work)
    env["NORAX_LOCK_DIR"] = str(work / "run")

    proc = subprocess.Popen(
        [sys.executable, "-m", "norax"],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        deadline = time.time() + 20
        ready = False
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.time() < deadline:
                try:
                    r = await client.get(f"http://127.0.0.1:{port}/status")
                    if r.status_code == 200:
                        ready = True
                        break
                except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
                    await asyncio.sleep(0.1)
            if not ready:
                # Never call read() while the child is alive: a failed startup
                # keeps stdout open and would hang this test indefinitely.
                proc.terminate()
                try:
                    out, _ = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    out, _ = proc.communicate(timeout=5)
                pytest.fail(f"runtime did not become ready: {out.decode(errors='replace')[:2000]}")

            r = await client.post(
                f"http://127.0.0.1:{port}/ingress/test",
                json={"body": "hello phase 8"},
            )
            assert r.status_code == 200
            assert r.json()["ok"] is True

        await asyncio.sleep(0.3)

        events = (work / "state" / "events.jsonl").read_text().splitlines()
        kinds = [json.loads(line)["kind"] for line in events if line.strip()]
        assert "runtime.start" in kinds
        assert "ingress" in kinds
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("runtime did not drain within 10s")
