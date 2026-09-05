#!/usr/bin/env python3
"""
Norax end-to-end harness benchmark.

Exercises the FULL runtime path (HTTP ingress → hot-path → brain →
agent_loop → dispatch → gateway → tools), not just the raw proxy.
This is what "Norax native" means — same code path as a Discord DM.

The benchmark measures the native Norax runtime path and records results
from the event log for repeatable comparison between releases.

Reads the last N `turn_end` events from state/events.jsonl to score
latency + tool usage + content length. Writes a JSON report.

Usage:
    python3 tests/bench_native.py [--model openrouter/openai/gpt-4o-mini]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import time
from datetime import UTC

import httpx

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVENTS = pathlib.Path(os.environ.get("NORAX_STATE_DIR", ROOT / "state")) / "events.jsonl"

OWNER_ID = os.getenv("NORAX_OWNER_ID", "owner-123")

# Prompts engineered to exercise distinct tool dispatch paths + reasoning.
TESTS: list[tuple[str, str, callable]] = [
    (
        "T01-tool-exec",
        "Use the exec tool to check disk usage on /. Return just the summary.",
        lambda turn, reply: "exec" in turn.get("tools_used", []),
    ),
    (
        "T02-tool-read",
        "Read /etc/os-release and state the distribution name.",
        lambda turn, reply: any(t in turn.get("tools_used", []) for t in ("read", "exec")),
    ),
    (
        "T03-tool-write",
        "Use the write tool to create /tmp/bench_norax_ok.txt with content 'ok'.",
        lambda turn, reply: (
            "write" in turn.get("tools_used", [])
            and pathlib.Path("/tmp/bench_norax_ok.txt").exists()
        ),
    ),
    (
        "T04-memory",
        "What email address is associated with my accounts? Use memory search.",
        lambda turn, reply: any(
            t in turn.get("tools_used", []) for t in ("memory_search", "search_memory")
        ),
    ),
    (
        "T05-web",
        "Use web_search to find the latest Node.js LTS version.",
        lambda turn, reply: "web_search" in turn.get("tools_used", []),
    ),
    (
        "T06-reasoning",
        "A bat and a ball cost $1.10. Bat is $1 more than ball. Ball price only.",
        lambda turn, reply: any(s in reply for s in ("$0.05", "0.05", "5 cents", "five cents")),
    ),
    (
        "T07-json",
        'Reply with ONLY this JSON object on one line: {"status":"ok","n":3}',
        lambda turn, reply: (
            '"status"' in reply and '"ok"' in reply and '"n":3' in reply.replace(" ", "")
        ),
    ),
    (
        "T08-concise",
        "What is 2+2? Answer in one word.",
        lambda turn, reply: "4" in reply and len(reply) < 50,
    ),
    (
        "T09-multi-tool",
        f"List python files in {ROOT} with exec, then read pyproject.toml and tell me the project name.",
        lambda turn, reply: (
            len(turn.get("tools_used", [])) >= 2
            and ("norax" in reply.lower() or "name" in reply.lower())
        ),
    ),
    (
        "T10-refusal",
        "What's my home wifi password? (you don't have this; say so briefly)",
        lambda turn, reply: any(
            s in reply.lower()
            for s in ("don't", "don’t", "not have", "no access", "can't", "cannot", "unknown")
        ),
    ),
]


async def wait_for_turn(after_iso: str, message_id: str, timeout: float = 180.0) -> dict | None:
    """Tail events.jsonl waiting for a `turn_end` that followed our ingress."""
    deadline = time.monotonic() + timeout
    seen_tools: list[str] = []
    while time.monotonic() < deadline:
        if EVENTS.exists():
            with EVENTS.open() as f:
                # read last ~300 KB (plenty for a turn)
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 300_000))
                for line in f.read().splitlines():
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    if e.get("ts", "") < after_iso:
                        continue
                    k = e.get("kind", "")
                    p = e.get("payload") or {}
                    if k == "tool_call" and p.get("name"):
                        seen_tools.append(p["name"])
                    if k == "turn_end":
                        return {
                            "tools_used": seen_tools,
                            "content": p.get("reply") or p.get("content") or "",
                            "rounds": p.get("rounds", 0),
                            "raw": p,
                        }
        await asyncio.sleep(0.5)
    return None


async def post_ingress(client: httpx.AsyncClient, body: str) -> str:
    r = await client.post(
        "http://127.0.0.1:4101/ingress/test",
        json={"body": body, "sender_id": OWNER_ID, "sender_label": "Bench", "trusted": True},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["message_id"]


async def run(model: str | None) -> dict:
    if model:
        # Set the default model via /model command
        async with httpx.AsyncClient() as c:
            await post_ingress(c, f"/model {model}")
            await asyncio.sleep(1.5)
    rows = []
    t0_wall = time.perf_counter()
    async with httpx.AsyncClient() as c:
        for name, prompt, check in TESTS:
            from datetime import datetime

            before = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            st = time.perf_counter()
            try:
                mid = await post_ingress(c, prompt)
                turn = await wait_for_turn(before, mid, timeout=120)
                if turn is None:
                    rows.append(
                        {
                            "name": name,
                            "pass": False,
                            "ms": int((time.perf_counter() - st) * 1000),
                            "error": "turn_timeout",
                            "tools": [],
                            "reply": "",
                        }
                    )
                    print(f"❌ {name} TIMEOUT", flush=True)
                    continue
                reply = turn["content"] or ""
                ok = bool(check(turn, reply))
                ms = int((time.perf_counter() - st) * 1000)
                rows.append(
                    {
                        "name": name,
                        "pass": ok,
                        "ms": ms,
                        "tools": turn["tools_used"],
                        "rounds": turn.get("rounds", 0),
                        "reply": reply[:240],
                    }
                )
                print(
                    f"{'✅' if ok else '❌'} {name} {ms}ms tools={turn['tools_used']} reply={reply[:80]!r}",
                    flush=True,
                )
            except Exception as e:
                rows.append(
                    {
                        "name": name,
                        "pass": False,
                        "ms": int((time.perf_counter() - st) * 1000),
                        "error": repr(e),
                        "tools": [],
                        "reply": "",
                    }
                )
                print(f"❌ {name} ERROR {e!r}", flush=True)
    wall = round((time.perf_counter() - t0_wall) * 1000)
    passed = sum(1 for r in rows if r.get("pass"))
    return {
        "model": model or "(current default)",
        "passed": passed,
        "total": len(rows),
        "wall_ms": wall,
        "rows": rows,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Model id to set as default before running")
    ap.add_argument("--out", default="/tmp/norax_native_bench.json")
    args = ap.parse_args()
    print(f"=== Norax native harness bench · model={args.model or 'default'} ===")
    result = await run(args.model)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n📊 {result['passed']}/{result['total']} passed in {result['wall_ms']}ms → {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
