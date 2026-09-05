#!/usr/bin/env python3
"""Live acceptance test for the LLM→tool→verification execution harness."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from norax.__main__ import _load_dotenv
from norax.brain.agent_loop import run_agent_loop
from norax.config.loader import load_config
from norax.dispatch.tools import REGISTRY
from norax.runtime.core import Runtime


class _AcceptanceLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def append(self, kind, payload, attrs=None) -> None:
        self.events.append((kind, payload))


def _confined_file_tool(function, artifact: Path):
    async def call(**kwargs):
        try:
            requested = Path(kwargs.get("path", "")).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            return {"ok": False, "error": "invalid_acceptance_path", "_not_executed": True}
        if requested != artifact:
            return {"ok": False, "error": "outside_acceptance_artifact", "_not_executed": True}
        return await function(**kwargs)

    return call


def _confined_exec_tool(function, *, allowed_commands: set[str], root: Path):
    async def call(**kwargs):
        if str(kwargs.get("command", "")) not in allowed_commands:
            return {"ok": False, "error": "outside_acceptance_command", "_not_executed": True}
        cwd = kwargs.get("cwd")
        if cwd is not None:
            try:
                if Path(cwd).expanduser().resolve() != root:
                    return {
                        "ok": False,
                        "error": "outside_acceptance_workdir",
                        "_not_executed": True,
                    }
            except (OSError, RuntimeError, TypeError, ValueError):
                return {"ok": False, "error": "invalid_acceptance_workdir", "_not_executed": True}
        return await function(**kwargs)

    return call


async def main() -> int:
    _load_dotenv()
    cfg = load_config()
    runtime = Runtime.build(cfg)
    log = _AcceptanceLog()
    try:
        with tempfile.TemporaryDirectory(prefix="norax-agent-acceptance-", dir="/tmp") as raw:
            root = Path(raw)
            artifact = root / "proof.txt"
            missing_command = f"grep -n definitely_missing {artifact}"
            success_command = "printf EXEC_OK"
            originals = {name: REGISTRY[name] for name in ("write", "read", "exec")}
            REGISTRY["write"] = replace(
                originals["write"], fn=_confined_file_tool(originals["write"].fn, artifact)
            )
            REGISTRY["read"] = replace(
                originals["read"], fn=_confined_file_tool(originals["read"].fn, artifact)
            )
            REGISTRY["exec"] = replace(
                originals["exec"],
                fn=_confined_exec_tool(
                    originals["exec"].fn,
                    allowed_commands={missing_command, success_command},
                    root=root,
                ),
            )
            previous_workspace = os.environ.get("NORAX_WORKSPACE")
            os.environ["NORAX_WORKSPACE"] = str(root)
            try:
                response, trace, rounds, _task_state = await run_agent_loop(
                    gateway=runtime.gateway,
                    model=runtime.default_model,
                    system_prompt=(
                        "You are running a deterministic execution-harness acceptance test. "
                        "Follow the requested tool sequence exactly and verify every result."
                    ),
                    user_prompt=(
                        f"Use write to create {artifact} with exactly three lines: FIRST, SECOND, "
                        "and HARNESS_OK. Use read on that same file twice: first offset=0 limit=1, "
                        "then offset=1 limit=1. Use exec to run "
                        f"`{missing_command}`; no matches is the expected success. "
                        f"Then use exec to run `{success_command}`. After every step succeeds, "
                        "reply with exactly AGENT_HARNESS_OK and nothing else."
                    ),
                    allowed_tools=["write", "read", "exec"],
                    sender_tier="owner",
                    event_log=log,
                    max_rounds=12,
                    timeout_seconds=180,
                    failover_models=runtime.failover_models,
                )
            finally:
                if previous_workspace is None:
                    os.environ.pop("NORAX_WORKSPACE", None)
                else:
                    os.environ["NORAX_WORKSPACE"] = previous_workspace
                REGISTRY.update(originals)

            names = [item.get("name") for item in trace]
            exec_ok = any(
                item.get("name") == "exec"
                and item.get("result", {}).get("ok") is True
                and item.get("args", {}).get("command") == success_command
                and item.get("result", {}).get("exit_code") == 0
                and item.get("result", {}).get("stdout") == "EXEC_OK"
                for item in trace
            )
            successful_trace = [item for item in trace if item.get("result", {}).get("ok") is True]
            read_pages = [
                item.get("result", {}).get("content")
                for item in successful_trace
                if item.get("name") == "read"
            ]
            expected_steps = [
                ("write", None),
                ("read", "FIRST"),
                ("read", "SECOND"),
                ("exec", missing_command),
                ("exec", success_command),
            ]
            matched_steps = 0
            for item in successful_trace:
                if matched_steps >= len(expected_steps):
                    break
                expected_name, expected_value = expected_steps[matched_steps]
                if item.get("name") != expected_name:
                    continue
                if expected_name == "read":
                    expected_offset = 0 if expected_value == "FIRST" else 1
                    if (
                        item.get("args", {}).get("offset") != expected_offset
                        or item.get("args", {}).get("limit") != 1
                        or item.get("result", {}).get("content") != expected_value
                    ):
                        continue
                elif expected_name == "exec":
                    if item.get("args", {}).get("command") != expected_value:
                        continue
                matched_steps += 1
            read_sequence_ok = matched_steps >= 3
            tool_sequence_ok = matched_steps == len(expected_steps)
            empty_search_ok = any(
                item.get("name") == "exec"
                and item.get("args", {}).get("command") == missing_command
                and item.get("result", {}).get("ok") is True
                and item.get("result", {}).get("exit_code") == 1
                and item.get("result", {}).get("no_matches") is True
                for item in trace
            )
            artifact_content = artifact.read_text(encoding="utf-8") if artifact.exists() else None
            artifact_lines_ok = artifact_content is not None and artifact_content.splitlines() == [
                "FIRST",
                "SECOND",
                "HARNESS_OK",
            ]
            final_marker_ok = response.content.strip() == "AGENT_HARNESS_OK"
            artifact_write_ok = artifact_lines_ok and any(
                item.get("name") == "write"
                and item.get("result", {}).get("ok") is True
                and Path(item.get("args", {}).get("path", "")).expanduser().resolve() == artifact
                and str(item.get("args", {}).get("content", "")).splitlines()
                == ["FIRST", "SECOND", "HARNESS_OK"]
                for item in trace
            )
            completed = (response.raw or {}).get("incomplete") is not True
            ok = (
                artifact_lines_ok
                and artifact_write_ok
                and read_sequence_ok
                and tool_sequence_ok
                and empty_search_ok
                and exec_ok
                and final_marker_ok
                and completed
            )
            result = {
                "ok": ok,
                "model": response.model,
                "rounds": rounds,
                "tools": names,
                "artifact_created": artifact_content is not None,
                "artifact_lines_ok": artifact_lines_ok,
                "artifact_write_ok": artifact_write_ok,
                "read_pages": read_pages[:4],
                "read_sequence_ok": read_sequence_ok,
                "tool_sequence_ok": tool_sequence_ok,
                "empty_search_ok": empty_search_ok,
                "exec_ok": exec_ok,
                "final_marker_ok": final_marker_ok,
                "completed": completed,
                "final": response.content.strip()[:200],
                "event_count": len(log.events),
            }
            if not ok:
                result["exec_diagnostics"] = [
                    {
                        "command": str(item.get("args", {}).get("command", ""))[:300],
                        "result": item.get("result", {}),
                    }
                    for item in trace
                    if item.get("name") == "exec"
                ]
            print(json.dumps(result, indent=2))
            return 0 if ok else 1
    finally:
        await runtime.gateway.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
