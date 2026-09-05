"""Phase 1 acceptance smoke test.

1. `python -m norax` starts and serves HTTP
2. POST /ingress/test writes an event to state/events.jsonl
3. SIGTERM drains within grace window
4. Hash chain replays cleanly
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
async def test_runtime_starts_serves_and_drains(tmp_path: Path):
    port = _free_port()
    work = tmp_path / "work"
    work.mkdir()
    # Create minimal soul files so startup validation passes
    soul_dir = work / "soul"
    soul_dir.mkdir()
    (soul_dir / "SOUL.md").write_text("# Soul\nTest soul.")
    (soul_dir / "IDENTITY.md").write_text("# Identity\nTest identity.")
    (soul_dir / "USER.md").write_text("# User\nTest user.")

    cfg_file = work / "runtime.jsonc"
    cfg_file.write_text(
        "{\n"
        '  "runtime": { "shutdown_grace_seconds": 5 },\n'
        f'  "http": {{ "bind": "127.0.0.1:{port}" }},\n'
        '  "paths": {\n'
        '    "soul": "soul/SOUL.md",\n'
        '    "identity": "soul/IDENTITY.md",\n'
        '    "user": "soul/USER.md"\n'
        "  }\n"
        "}\n"
    )

    # A subprocess integration test is a separate deployment. Never inherit
    # production Norax paths, ports, connectors, credentials, or lock files.
    env = {key: value for key, value in os.environ.items() if not key.startswith("NORAX_")}
    env["NORAX_CONFIG"] = str(cfg_file)
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
        # Wait for HTTP ready
        # Coverage-instrumented subprocess startup can be materially slower
        # than a normal service launch on shared CI runners.
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
                proc.terminate()
                try:
                    out, _ = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    out, _ = proc.communicate(timeout=5)
                pytest.fail(f"runtime did not become ready: {out.decode(errors='replace')[:2000]}")

            r = await client.post(
                f"http://127.0.0.1:{port}/ingress/test",
                json={"body": "hello phase 1"},
            )
            assert r.status_code == 200
            assert r.json()["ok"] is True

        # Let the event flush
        await asyncio.sleep(0.3)

        events_path = work / "state" / "events.jsonl"
        assert events_path.exists(), "events.jsonl was not created"
        lines = [line for line in events_path.read_text().splitlines() if line.strip()]
        kinds = [json.loads(line)["kind"] for line in lines]
        assert "runtime.start" in kinds
        assert "ingress" in kinds
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("runtime did not drain within 10s")


def test_hash_chain_replay(tmp_path: Path):
    """Write a few events, then verify the chain script accepts them."""
    from norax.observability.log import EventLog

    log_path = tmp_path / "events.jsonl"
    elog = EventLog(log_path)
    asyncio.run(elog.append("test.a", {"n": 1}))
    asyncio.run(elog.append("test.b", {"n": 2}))
    asyncio.run(elog.append("test.c", {"n": 3}))

    # Verify the chain manually
    import hashlib

    prev = "0" * 64
    for line in log_path.read_text().splitlines():
        rec = json.loads(line)
        recorded = rec.pop("hash")
        assert rec["prev_hash"] == prev
        body = json.dumps(rec, separators=(",", ":"), sort_keys=True)
        expect = hashlib.sha256((prev + body).encode()).hexdigest()
        assert recorded == expect
        prev = recorded
