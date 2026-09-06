"""Matched, isolated Norax/Hermes tool-loop trials; not a full-agent ranking.

Both engines get the same task, tools, model endpoint, 250-round/4-hour
budget, and independent artifact checks. Engine-native system prompts remain
in place. Trials run in alternating order, without deployment credentials.
Hermes must be installed separately in its supported editable environment.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

TASKS = ("repair_clamp", "aggregate", "transient_recovery")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUND_LIMIT = 250
DEFAULT_TURN_BUDGET_SECONDS = 14_400
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600
SYSTEM = (
    "Complete the task with the supplied tools. Verify the saved artifact before reporting DONE."
)
DESCRIPTIONS = {
    "bench_read": "Read this task's input artifact. Retry if the result says retryable.",
    "bench_write": "Save the complete output artifact for this task; accepts content as a string.",
    "bench_check": "Run independent checks on the saved output artifact. No arguments.",
}


class Task:
    def __init__(self, root: Path, name: str, seed: int):
        self.root, self.name = root, name
        self.output = root / "output.txt"
        self.trace: list[dict] = []
        rng = random.Random(seed)
        self.nonce = f"receipt-{rng.getrandbits(96):024x}"
        self.rows: list[dict[str, Any]] = [
            {
                "region": rng.choice(["east", "west", "north"]),
                "amount": rng.randrange(1, 100),
                "status": rng.choice(["paid", "paid", "void"]),
            }
            for _ in range(12)
        ]
        self.reads = 0

    @property
    def prompt(self) -> str:
        instructions = {
            "repair_clamp": "Read the Python function. Fix clamp so it returns lower below the interval, upper above it, and value inside it. Save the complete fixed Python function.",
            "aggregate": "Read the JSON rows. Sum amounts for paid rows only, grouped by region. Save only a JSON object mapping each region with paid rows to its integer total.",
            "transient_recovery": "Read the receipt, recovering from a retryable read failure if necessary. Save exactly the receipt string, with no whitespace added.",
        }
        return (
            instructions[self.name]
            + " Use bench_read, bench_write, then bench_check. Only after a passing check, reply exactly DONE."
        )

    def check(self) -> bool:
        if not self.output.exists():
            return False
        content = self.output.read_text()
        if self.name == "transient_recovery":
            return content == self.nonce
        if self.name == "aggregate":
            expected: dict[str, int] = {}
            for row in self.rows:
                if row["status"] == "paid":
                    region = str(row["region"])
                    expected[region] = expected.get(region, 0) + int(row["amount"])
            try:
                actual = json.loads(content)
                return actual == expected and all(type(v) is int for v in actual.values())
            except (ValueError, AttributeError):
                return False
        try:
            tree = ast.parse(content)
            allowed = (
                ast.Module,
                ast.FunctionDef,
                ast.arguments,
                ast.arg,
                ast.Return,
                ast.Call,
                ast.Name,
                ast.Load,
                ast.Constant,
                ast.IfExp,
                ast.Compare,
                ast.Lt,
                ast.Gt,
                ast.LtE,
                ast.GtE,
                ast.If,
            )
            nodes = list(ast.walk(tree))
            if len(nodes) > 100 or any(not isinstance(node, allowed) for node in nodes):
                return False
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                return False
            function = tree.body[0]
            if function.name != "clamp" or function.decorator_list or function.args.defaults:
                return False
            if any(
                isinstance(node, ast.Call)
                and (not isinstance(node.func, ast.Name) or node.func.id not in {"min", "max"})
                for node in nodes
            ):
                return False
            namespace: dict[str, Any] = {"__builtins__": {}, "min": min, "max": max, "int": int}
            exec(compile(tree, "<confined-clamp>", "exec"), namespace)  # noqa: S102
            clamp = namespace["clamp"]
            return all(
                clamp(v, lo, hi) == max(lo, min(hi, v))
                for lo, hi in [(-7, 3), (0, 0), (4, 19)]
                for v in [-20, -7, 0, 3, 7, 19, 100]
            )
        except (ValueError, TypeError, SyntaxError, NameError):
            return False

    def call(self, name: str, args: dict) -> dict:
        if name == "bench_read":
            self.reads += 1
            if self.name == "transient_recovery" and self.reads == 1:
                result = {"ok": False, "error": "temporary_read_failure", "retryable": True}
            else:
                content = {
                    "repair_clamp": "def clamp(value: int, lower: int, upper: int) -> int:\n    return min(lower, max(upper, value))\n",
                    "aggregate": json.dumps(self.rows),
                    "transient_recovery": self.nonce,
                }[self.name]
                result = {"ok": True, "content": content}
        elif name == "bench_write":
            content = args.get("content", "")
            if not isinstance(content, str) or len(content) > 4096:
                result = {"ok": False, "error": "content_must_be_bounded_string"}
            else:
                self.output.write_text(content)
                result = {"ok": True}
        elif name == "bench_check":
            result = {"ok": self.check()}
        else:
            result = {"ok": False, "error": "unknown_tool"}
        self.trace.append({"name": name, "ok": result["ok"]})
        return result


async def norax_trial(task: Task, args) -> str:
    sys.path.insert(0, str(args.norax_source))
    from norax.brain.agent_loop import run_agent_loop
    from norax.dispatch.tools import REGISTRY, ToolSpec
    from norax.gateway_client import GatewayClient

    class Log:
        async def append(self, *args, **kwargs):
            pass

    for name, description in DESCRIPTIONS.items():

        def bind(selected):
            async def call(**kwargs):
                return task.call(selected, kwargs)

            return call

        REGISTRY[name] = ToolSpec(
            name, description, {"content": "str"} if name == "bench_write" else {}, bind(name)
        )
    gateway = GatewayClient(
        base_url=args.base,
        provider_kind="openai",
        timeout=args.request_timeout_seconds,
    )
    try:
        response, _, _, _ = await run_agent_loop(
            gateway=gateway,
            model=args.model,
            system_prompt=SYSTEM,
            user_prompt=task.prompt,
            allowed_tools=list(DESCRIPTIONS),
            sender_tier="owner",
            event_log=Log(),
            max_rounds=args.round_limit,
            timeout_seconds=args.turn_budget_seconds,
            failover_models=[],
        )
        return (
            response.content.strip() if not (response.raw or {}).get("incomplete") else "INCOMPLETE"
        )
    finally:
        await gateway.aclose()


def hermes_trial(task: Task, args) -> str:
    from run_agent import AIAgent
    from tools.registry import registry

    for name, description in DESCRIPTIONS.items():

        def bind(selected):
            def call(arguments, **kwargs):
                return json.dumps(task.call(selected, arguments))

            return call

        properties = {"content": {"type": "string"}} if name == "bench_write" else {}
        registry.register(
            name=name,
            toolset="matched_benchmark",
            description=description,
            schema={
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                },
            },
            handler=bind(name),
        )
    agent = AIAgent(
        model=args.model,
        base_url=args.base,
        api_key="test-key",
        provider="custom",
        api_mode="chat_completions",
        max_iterations=args.round_limit,
        max_tokens=4096,
        run_budget_seconds=args.turn_budget_seconds,
        enabled_toolsets=["matched_benchmark"],
        quiet_mode=True,
        skip_memory=True,
        skip_context_files=True,
        skip_background_review=True,
        ephemeral_system_prompt=SYSTEM,
    )
    names = {tool["function"]["name"] for tool in agent.tools}
    if names != set(DESCRIPTIONS):
        raise RuntimeError(f"unexpected Hermes tool surface: {sorted(names)}")
    result = agent.run_conversation(task.prompt)
    return str(result.get("final_response", "")).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--hermes-python", type=Path)
    parser.add_argument("--norax-source", type=Path, default=ROOT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--engine", choices=["norax", "hermes"])
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--round-limit", type=int, default=DEFAULT_ROUND_LIMIT)
    parser.add_argument("--turn-budget-seconds", type=int, default=DEFAULT_TURN_BUDGET_SECONDS)
    parser.add_argument(
        "--request-timeout-seconds", type=int, default=DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
    args = parser.parse_args()
    if not 1 <= args.round_limit <= 1_000:
        parser.error("--round-limit must be between 1 and 1000")
    if args.turn_budget_seconds <= 0 or args.request_timeout_seconds <= 0:
        parser.error("time budgets must be positive")
    if args.engine:
        task = Task(Path.cwd(), args.task, args.seed)
        start = time.monotonic()
        error = None
        try:
            final = (
                asyncio.run(norax_trial(task, args))
                if args.engine == "norax"
                else hermes_trial(task, args)
            )
        except Exception as exc:  # noqa: BLE001
            final, error = "", type(exc).__name__ + ": " + str(exc)[:300]
        checked = any(row["name"] == "bench_check" and row["ok"] is True for row in task.trace)
        result = {
            "engine": args.engine,
            "task": args.task,
            "seed": args.seed,
            "seconds": round(time.monotonic() - start, 3),
            "trace": task.trace,
            "artifact_verified": task.check(),
            "check_observed": checked,
            "final_verified": final == "DONE",
            "error": error,
            "passed": task.check() and checked and final == "DONE" and error is None,
        }
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return 0 if result["passed"] else 1
    if not args.hermes_python or not 1 <= args.repeats <= 10:
        parser.error("parent run requires --hermes-python and 1-10 repeats")
    results = []
    for repetition in range(args.repeats):
        for name in TASKS:
            for engine in ["norax", "hermes"] if repetition % 2 == 0 else ["hermes", "norax"]:
                with tempfile.TemporaryDirectory(prefix="norax-matched-") as raw:
                    root = Path(raw)
                    receipt = root / "receipt.json"
                    hermes_state = root / "hermes-state"
                    hermes_state.mkdir()
                    # Keep identical direct tool schemas for this matched-loop
                    # comparison, using Hermes' supported configuration switch.
                    (hermes_state / "config.yaml").write_text(
                        'tools:\n  tool_search:\n    enabled: "off"\n'
                    )
                    env = {
                        k: v
                        for k, v in os.environ.items()
                        if k in {"PATH", "LANG", "LC_ALL", "SSL_CERT_FILE"}
                    }
                    env.update(
                        {
                            "HERMES_HOME": str(root / "hermes-state"),
                            "NORAX_WORKSPACE": raw,
                            "NORAX_PROJECT_ROOT": raw,
                            "NORAX_MEMORY_ROOT": str(root / "memory"),
                            "NORAX_STATE_DIR": str(root / "state"),
                            "NORAX_CONFIG_HOME": str(root / "config"),
                            "NORAX_AGENT_MAX_OUTPUT_TOKENS": "4096",
                        }
                    )
                    command = [
                        str(args.hermes_python) if engine == "hermes" else sys.executable,
                        str(Path(__file__).resolve()),
                        "--engine",
                        engine,
                        "--task",
                        name,
                        "--seed",
                        str(args.seed + repetition),
                        "--base",
                        args.base,
                        "--model",
                        args.model,
                        "--norax-source",
                        str(args.norax_source.resolve()),
                        "--json-out",
                        str(receipt),
                        "--round-limit",
                        str(args.round_limit),
                        "--turn-budget-seconds",
                        str(args.turn_budget_seconds),
                        "--request-timeout-seconds",
                        str(args.request_timeout_seconds),
                    ]
                    try:
                        process = subprocess.run(
                            command,
                            cwd=root,
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=args.turn_budget_seconds + 60,
                            check=False,
                        )
                        result = (
                            json.loads(receipt.read_text())
                            if receipt.exists()
                            else {
                                "engine": engine,
                                "task": name,
                                "passed": False,
                                "error": f"child exit {process.returncode}",
                                "diagnostic": process.stderr[-1500:],
                            }
                        )
                    except subprocess.TimeoutExpired:
                        result = {
                            "engine": engine,
                            "task": name,
                            "passed": False,
                            "error": "timeout",
                            "seconds": args.turn_budget_seconds + 60,
                        }
                    results.append(result)
                    print(json.dumps(result), flush=True)
                    summary = {
                        engine_name: {
                            "passed": sum(
                                r["passed"] for r in results if r["engine"] == engine_name
                            ),
                            "trials": sum(r["engine"] == engine_name for r in results),
                            "median_seconds": statistics.median(
                                [
                                    r["seconds"]
                                    for r in results
                                    if r["engine"] == engine_name and "seconds" in r
                                ]
                            )
                            if any(r["engine"] == engine_name and "seconds" in r for r in results)
                            else None,
                        }
                        for engine_name in ("norax", "hermes")
                    }
                    args.json_out.parent.mkdir(parents=True, exist_ok=True)
                    args.json_out.write_text(
                        json.dumps(
                            {
                                "scope": "matched_tool_loop_not_full_agent_ranking",
                                "model": args.model,
                                "repeats": args.repeats,
                                "round_limit": args.round_limit,
                                "turn_budget_seconds": args.turn_budget_seconds,
                                "summary": summary,
                                "trials": results,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
