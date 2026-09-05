#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from norax.remote.client import RemoteClient  # noqa: E402
from norax.remote.registry import RemoteRegistry  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser(
        description="End-to-end healthcheck for a Norax remote relay node."
    )
    ap.add_argument("node_id")
    ap.add_argument("--root", default=None, help="Expected writable root on the remote node.")
    ap.add_argument(
        "--registry",
        default=None,
        help="Registry root; defaults to NORAX_REMOTE_ROOT or memory/remote.",
    )
    args = ap.parse_args()

    reg = RemoteRegistry(args.registry) if args.registry else RemoteRegistry()
    node = reg.get(args.node_id)
    if node is None:
        print(f"FAIL unknown node: {args.node_id}")
        return 2
    root = args.root or (node.roots[0] if node.roots else ".")
    client = RemoteClient(reg)
    checks: list[tuple[str, dict]] = []

    hello = await client.run_job(args.node_id, {"action": "hello", "timeout": 15})
    checks.append(("hello", hello))
    ex = await client.run_job(
        args.node_id,
        {"action": "exec", "command": "printf relay_exec_ok", "cwd": root, "timeout": 15},
    )
    checks.append(("exec", ex))
    ls = await client.run_job(args.node_id, {"action": "list", "path": root, "timeout": 15})
    checks.append(("list", ls))
    test_path = str(Path(root) / ".norax_relay_healthcheck.txt")
    wr = await client.run_job(
        args.node_id,
        {"action": "write", "path": test_path, "content": "relay_write_ok\n", "timeout": 15},
    )
    checks.append(("write", wr))
    rd = await client.run_job(
        args.node_id, {"action": "read", "path": test_path, "limit": 200, "timeout": 15}
    )
    checks.append(("read", rd))
    cleanup = await client.run_job(
        args.node_id,
        {
            "action": "exec",
            "command": f"rm -f -- {shlex.quote(test_path)}",
            "cwd": root,
            "timeout": 15,
        },
    )
    checks.append(("cleanup", cleanup))

    ok = True
    for name, res in checks:
        passed = res.get("ok") is True
        if name == "exec":
            passed = passed and res.get("stdout") == "relay_exec_ok"
        if name == "read":
            passed = passed and "relay_write_ok" in str(res.get("content", ""))
        ok = ok and passed
        print(f"{name}: {'OK' if passed else 'FAIL'} {res}")
    print(f"REMOTE_RELAY_HEALTH={'OK' if ok else 'FAIL'} node_id={args.node_id}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
