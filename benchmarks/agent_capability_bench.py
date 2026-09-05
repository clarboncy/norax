"""Isolated live acceptance of model → Norax tools → verified file outcome.

This is a basic execution acceptance check, not a competitive ranking.
Only task-specific file tools are exposed, and all artifacts are temporary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import tempfile
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_BUGGY_SOURCE = """def clamp(value: int, lower: int, upper: int) -> int:
    return min(lower, max(upper, value))
"""
_FIXED_SOURCE = """def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))
"""
_BUGGY_LINE = "return min(lower, max(upper, value))"
_FIXED_LINE = "return max(lower, min(upper, value))"


def confined_tool(function, artifact: Path):
    """Wrap a file tool so a benchmark model can touch only its one artifact."""

    async def call(**kwargs):
        try:
            requested = Path(kwargs.get("path", "")).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            return {"ok": False, "error": "invalid_benchmark_path", "_not_executed": True}
        if requested != artifact:
            return {"ok": False, "error": "outside_benchmark_artifact", "_not_executed": True}
        return await function(**kwargs)

    return call


def _trace_targets_artifact(item: dict, artifact: Path) -> bool:
    try:
        return Path(item.get("args", {}).get("path", "")).expanduser().resolve() == artifact
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


async def execute(args: argparse.Namespace, root: Path) -> dict:
    from norax.brain.agent_loop import run_agent_loop
    from norax.dispatch.tools import REGISTRY
    from norax.gateway_client import GatewayClient

    class EventLog:
        async def append(self, *args, **kwargs):
            pass

    if args.task == "edit-verify":
        artifact = (root / "clamp.py").resolve()
        artifact.write_text(_BUGGY_SOURCE, encoding="utf-8")
        tool_names = ("read", "edit")
        user_prompt = (
            f"The clamp function in {artifact} has a one-line logic bug. "
            "First use read to inspect the current file. Then use edit to replace exactly "
            f"`{_BUGGY_LINE}` with `{_FIXED_LINE}`. Use read again to verify the saved file. "
            "After verification, reply with exactly FIXED and nothing else."
        )
        final_expected = "FIXED"
    else:
        artifact = (root / "proof.txt").resolve()
        nonce = secrets.token_hex(12)
        tool_names = ("write", "read")
        user_prompt = (
            f"Use write to create {artifact} containing exactly {nonce}. "
            "Then use read on that file to verify its content. "
            "After successful verification, reply with only the file content."
        )
        final_expected = nonce
    gateway = GatewayClient(base_url=args.base, provider_kind=args.provider, timeout=60)
    originals = {name: REGISTRY[name] for name in tool_names}
    for name, spec in originals.items():
        REGISTRY[name] = replace(spec, fn=confined_tool(spec.fn, artifact))
    started = time.monotonic()
    try:
        response, trace, rounds, _ = await run_agent_loop(
            gateway=gateway,
            model=args.model,
            system_prompt="Execute the requested task with the available tools. Verify before reporting completion.",
            user_prompt=user_prompt,
            allowed_tools=list(tool_names),
            sender_tier="owner",
            event_log=EventLog(),
            max_rounds=8,
            timeout_seconds=120,
            failover_models=[],
        )
        complete = not (response.raw or {}).get("incomplete", False)
        common = {
            "scope": "isolated_file_tool_acceptance",
            "task": args.task,
            "model": args.model,
            "response_model": response.model,
            "completed": complete,
            "final_verified": response.content.strip() == final_expected,
            "rounds": rounds,
            "seconds": time.monotonic() - started,
            "timestamp": datetime.now(UTC).isoformat(),
            "tool_outcomes": [
                {
                    "tool": row.get("name"),
                    "ok": row.get("result", {}).get("ok"),
                    "error": row.get("result", {}).get("error"),
                }
                for row in trace
            ],
        }
        if args.task == "edit-verify":
            read_before_edit = False
            edit_succeeded = False
            readback_verified = False
            successful_edits = 0
            for item in trace:
                result = item.get("result", {})
                if result.get("ok") is not True or not _trace_targets_artifact(item, artifact):
                    continue
                if item.get("name") == "read" and not edit_succeeded:
                    read_before_edit = str(result.get("content", "")).rstrip("\n") == (
                        _BUGGY_SOURCE.rstrip("\n")
                    )
                elif item.get("name") == "edit" and read_before_edit:
                    call_args = item.get("args", {})
                    if call_args.get("old") == _BUGGY_LINE and call_args.get("new") == _FIXED_LINE:
                        edit_succeeded = True
                        successful_edits += 1
                        readback_verified = False
                elif item.get("name") == "read" and edit_succeeded:
                    readback_verified = str(result.get("content", "")).rstrip("\n") == (
                        _FIXED_SOURCE.rstrip("\n")
                    )
            artifact_ok = (
                artifact.is_file() and artifact.read_text(encoding="utf-8") == _FIXED_SOURCE
            )
            common.update(
                {
                    "passed": read_before_edit
                    and edit_succeeded
                    and successful_edits == 1
                    and readback_verified
                    and artifact_ok
                    and complete
                    and response.content.strip() == final_expected,
                    "initial_read_verified": read_before_edit,
                    "edit_succeeded": edit_succeeded,
                    "successful_edits": successful_edits,
                    "readback_verified": readback_verified,
                    "artifact_verified": artifact_ok,
                }
            )
        else:
            written = False
            verified = False
            for item in trace:
                result = item.get("result", {})
                if result.get("ok") is not True or not _trace_targets_artifact(item, artifact):
                    continue
                if item.get("name") == "write":
                    written = True
                    verified = False
                if item.get("name") == "read" and written:
                    verified = str(result.get("content", "")).strip() == final_expected
            artifact_ok = artifact.is_file() and artifact.read_text() == final_expected
            common.update(
                {
                    "passed": written
                    and verified
                    and artifact_ok
                    and complete
                    and response.content.strip() == final_expected,
                    "write_succeeded": written,
                    "readback_verified": verified,
                    "artifact_verified": artifact_ok,
                }
            )
        return common
    except Exception as exc:
        return {
            "scope": "isolated_file_tool_acceptance",
            "task": args.task,
            "model": args.model,
            "passed": False,
            "error_type": type(exc).__name__,
            "seconds": time.monotonic() - started,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    finally:
        REGISTRY.update(originals)
        await gateway.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=["ollama", "openai"], required=True)
    parser.add_argument(
        "--task",
        choices=["write-read", "edit-verify"],
        default="write-read",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    output_path = args.json_out.expanduser().resolve() if args.json_out else None
    saved = {k: v for k, v in os.environ.items() if k.startswith("NORAX_")}
    try:
        for key in saved:
            del os.environ[key]
        with tempfile.TemporaryDirectory(prefix="norax-model-acceptance-") as temporary:
            root = Path(temporary)
            for key, directory in {
                "PROJECT_ROOT": root,
                "WORKSPACE": root,
                "STATE_DIR": root / "state",
                "MEMORY_ROOT": root / "memory",
                "LOG_DIR": root / "logs",
                "LOCK_DIR": root / "locks",
                "CONFIG_HOME": root / "config",
                "CHECKPOINT_DIR": root / "checkpoints",
            }.items():
                directory.mkdir(exist_ok=True)
                os.environ[f"NORAX_{key}"] = str(directory)
            os.environ["NORAX_AGENT_MAX_OUTPUT_TOKENS"] = "1024"
            result = asyncio.run(execute(args, root))
            if output_path is not None:
                from norax.atomic import atomic_write_text

                atomic_write_text(
                    output_path,
                    json.dumps(result, indent=2) + "\n",
                    mode=0o600,
                )
            print(json.dumps(result, indent=2))
            return 0 if result["passed"] else 1
    finally:
        for key in list(os.environ):
            if key.startswith("NORAX_"):
                del os.environ[key]
        os.environ.update(saved)


if __name__ == "__main__":
    raise SystemExit(main())
