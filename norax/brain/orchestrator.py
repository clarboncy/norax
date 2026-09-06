"""Dual-model orchestrator: planner plans, Qwen executes tools.

Pattern:
  1. Planner receives task context
  2. Planner decides: which tools to call, in what order, with what args
  3. Qwen (ollama/qwen3-coder-next:cloud) executes each tool call natively
  4. Results feed back to the planner for the next planning iteration
  5. Repeat until the planner produces a final answer (no more tool plans)

The planner produces text-only CoT; Qwen handles native tool execution so we
get strong reasoning without shim pre-execution on the planner path.

Architecture:
  - orchestrator.py — plan/execute loop + plan parsing (this file)
  - strong_model_scaffold.py — task state, planning prompts, model resolution
  - gateway_client/__init__.py — GatewayRouter handles provider routing
  - runtime/core.py — conditional dispatch into orchestrator vs agent_loop

Circuit isolation:
  - Planner calls route through the gateway router (tier fallback built-in)
  - Executor calls go to direct Ollama on :11434
  - Planner failures never bleed into the main agent loop
  - On planner circuit-open, orchestrator falls back to a configured fallback model
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any

import httpx

from ..gateway_client import (
    GatewayRequest,
    GatewayResponse,
    GatewayUpstreamError,
    SpendGuardTripped,
)
from ..observability.circuit import CircuitOpen
from .strong_model_scaffold import (
    build_orchestrator_scaffold,
    build_scaffold_prompt,
    build_task_state,
    resolve_planner_model,
    summarize_mutation_trace,
)

log = logging.getLogger("norax.orchestrator")

# ── Model constants ────────────────────────────────────────────────────────

# Default planner when none specified — Opus 4.8 thinking-high
PLANNER_MODEL = "claude-opus-4-8-thinking-high"

# Fallback planner — used when the primary planner is down or circuit-open
FALLBACK_PLANNER_MODEL = "gpt-5.5"

# Tool-executor model — Qwen Coder Next (reliable tool JSON via direct Ollama)
EXECUTOR_MODEL = "qwen3-coder-next:cloud"
EXECUTOR_PROVIDER = "ollama_direct"

# ── Timeout budgets ────────────────────────────────────────────────────────

PLANNER_TIMEOUT = 120.0  # Composer thinking time
EXECUTOR_TIMEOUT = 30.0  # Per-tool execution time
MAX_ORCH_ROUNDS = 250  # Full-completion default, including runtime auto (0)
HARD_ORCH_ROUND_CAP = 250
_PLANNER_FAILOVER_HTTP_STATUSES = frozenset({401, 404, 408, 409, 425, 429, 500, 502, 503, 504})

# ── Tool result compression ────────────────────────────────────────────────

# Per-tool key-extraction: which fields carry the highest signal?
_RESULT_SIGNAL_KEYS: dict[str, tuple[str, ...]] = {
    "read": ("ok", "path", "total_lines", "content"),
    "list_dir": ("ok", "path", "entries"),
    "exec": ("ok", "exit_code", "stdout", "stderr"),
    "shell": ("ok", "exit_code", "stdout", "stderr", "cwd"),
    "edit": ("ok", "path"),
    "write": ("ok", "path"),
    "write_chunk": ("ok", "path"),
    "search_memory": ("ok", "results"),
    "status": ("ok", "state"),
}
_MAX_RESULT_LENGTH = 24000  # Characters per tool result


class Orchestrator:
    """Hybrid planner/executor loop.

    Usage:
        orch = Orchestrator(gateway_router)
        result = await orch.run(
            system_prompt="...",
            user_prompt="...",
            allowed_tools=["read", "edit", "exec"],
            sender_tier="owner",
        )
    """

    def __init__(self, router: Any) -> None:
        self.router = router
        self._planner_failures: int = 0
        self._planner_fallback_active: bool = False
        self._failed_planner_models: set[str] = set()
        self._turn_usage = {"input_tokens": 0, "output_tokens": 0}

    @staticmethod
    def _is_planner_failover_error(exc: BaseException) -> bool:
        if isinstance(exc, SpendGuardTripped):
            return False
        if isinstance(exc, GatewayUpstreamError):
            return exc.status in _PLANNER_FAILOVER_HTTP_STATUSES
        return isinstance(
            exc,
            (httpx.TransportError, TimeoutError, ConnectionError, OSError, CircuitOpen),
        )

    def _account_response(self, response: GatewayResponse) -> GatewayResponse:
        for key in ("input_tokens", "output_tokens"):
            try:
                value = int((response.usage or {}).get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                self._turn_usage[key] += value
        return response

    async def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        allowed_tools: list[str],
        sender_tier: str,
        event_log: Any | None = None,
        on_delta: Any | None = None,
        max_rounds: int = MAX_ORCH_ROUNDS,
        planner_model: str | None = None,
        prior_messages: list[dict] | None = None,
        timeout_seconds: float | None = None,
    ) -> OrchestratorRunResult:
        """Run the planner within the same wall-clock budget as direct turns."""
        from .agent_loop import DEFAULT_TIMEOUT_SECONDS

        budget = timeout_seconds if timeout_seconds is not None else DEFAULT_TIMEOUT_SECONDS
        if isinstance(budget, bool) or not math.isfinite(budget) or budget <= 0:
            budget = DEFAULT_TIMEOUT_SECONDS
        self._active_trace: list[dict] = []
        self._active_rounds = 0
        timer = asyncio.timeout(min(budget, 86_400))
        try:
            async with timer:
                return await self._run(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    allowed_tools=allowed_tools,
                    sender_tier=sender_tier,
                    event_log=event_log,
                    on_delta=on_delta,
                    max_rounds=max_rounds,
                    planner_model=planner_model,
                    prior_messages=prior_messages,
                )
        except TimeoutError:
            if not timer.expired():
                raise
            return self._finalize_run(
                content="The task reached its wall-clock budget; completion is not verified.",
                trace=self._active_trace,
                rounds=self._active_rounds,
                user_prompt=user_prompt,
                completion_signal="flow_timeout",
            )

    async def _run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        allowed_tools: list[str],
        sender_tier: str,
        event_log: Any | None = None,
        on_delta: Any | None = None,
        max_rounds: int = MAX_ORCH_ROUNDS,
        planner_model: str | None = None,
        prior_messages: list[dict] | None = None,
    ) -> OrchestratorRunResult:
        """Run hybrid orchestration loop.

        The result remains tuple-unpackable as ``(content, trace, rounds)`` for
        compatibility, while carrying explicit completion evidence for callers
        that persist checkpoints or produce training labels.
        """
        trace = self._active_trace
        rounds = 0
        self._planner_model = resolve_planner_model(planner_model or PLANNER_MODEL)
        round_cap = self._resolve_round_cap(max_rounds)
        self._planner_failures = 0
        self._planner_fallback_active = False
        self._failed_planner_models = set()
        self._turn_usage = {"input_tokens": 0, "output_tokens": 0}

        # Build task state scaffolding
        task_state = build_task_state(user_prompt, allowed_tools)
        scaffold = build_scaffold_prompt(
            task_state.task_type,
            model=self._planner_model,
            user_prompt=user_prompt,
        )
        orch_scaffold = build_orchestrator_scaffold(planner_model=self._planner_model)

        # Conversation history for the planner
        composer_messages: list[dict] = [
            {
                "role": "system",
                "content": f"{system_prompt}\n\n{scaffold}\n\n{orch_scaffold}",
            },
        ]
        composer_messages.extend(
            dict(message) for message in (prior_messages or []) if message.get("role") != "system"
        )
        composer_messages.append({"role": "user", "content": user_prompt})

        while rounds < round_cap:
            rounds += 1
            self._active_rounds = rounds

            # ── Phase 1: Planner plans (Opus/Composer → text CoT, no native tools) ──
            plan = await self._call_planner(composer_messages, allowed_tools)
            if plan.done:
                prefix = ""
                if self._planner_fallback_active:
                    prefix = f"[planning fallback: {self._planner_model}] "
                return self._finalize_run(
                    content=prefix + (plan.final or plan.planner_text),
                    trace=trace,
                    rounds=rounds,
                    user_prompt=user_prompt,
                    completion_signal=plan.completion_signal,
                )
            if not plan.steps:
                # Defensive fallback for a malformed PlanResult supplied by a
                # custom planner adapter. The parser itself marks this case.
                return self._finalize_run(
                    content=plan.final or plan.planner_text,
                    trace=trace,
                    rounds=rounds,
                    user_prompt=user_prompt,
                    completion_signal="invalid_plan",
                )

            composer_messages.append(
                {
                    "role": "assistant",
                    "content": plan.planner_text,
                }
            )

            # ── Phase 2: Qwen executes tool plan (native tool-calling) ──
            tool_results = await self._execute_plan(
                plan,
                allowed_tools,
                sender_tier,
                event_log=event_log,
            )
            trace.extend(tool_results)

            # ── Phase 3: Feed results back to Planner ──
            composer_messages.append(
                {
                    "role": "user",
                    "content": self._format_tool_feedback(tool_results),
                }
            )

        # Hit round limit — ask Planner for best-effort final
        final = await self._call_planner_for_final(composer_messages)
        return self._finalize_run(
            content=final,
            trace=trace,
            rounds=rounds,
            user_prompt=user_prompt,
            completion_signal="round_limit",
        )

    # ── Planner calling ─────────────────────────────────────────────────────

    async def _call_planner(
        self,
        messages: list[dict],
        allowed_tools: list[str],
    ) -> PlanResult:
        """Ask the planner what to do next. Returns plan or None if done.

        Handles planner failures with circuit-breaker isolation. Availability
        failures move directly to a distinct configured provider; request,
        permission, and local-code failures remain terminal.
        """

        tool_instructions = self._build_planner_tool_prompt(allowed_tools)
        planner_messages = [dict(message) for message in messages]
        # Inject tool instructions into system message
        if planner_messages and planner_messages[0].get("role") == "system":
            content = planner_messages[0].get("content", "")
            planner_messages[0]["content"] = f"{content}\n\n{tool_instructions}"

        req = GatewayRequest(
            model=self._planner_model,
            messages=planner_messages,
            tools=None,  # Planner gets text-only prompt
            metadata={"orchestrator": True, "phase": "plan"},
        )

        try:
            resp = await self._plan_request(req)
            self._planner_failures = 0
            content = resp.content or ""
            return self._parse_planner_output(content, allowed_tools)
        except SpendGuardTripped:
            raise
        except Exception as exc:
            return await self._handle_planner_failure(exc, messages, allowed_tools)

    async def _plan_request(self, req: GatewayRequest) -> GatewayResponse:
        """Send a planner request with timeout."""
        # Route through the public gateway API so dynamic-provider leases and
        # model-prefix normalization cannot be bypassed by orchestration.
        response = await asyncio.wait_for(self.router.chat(req), timeout=PLANNER_TIMEOUT)
        return self._account_response(response)

    async def _handle_planner_failure(
        self,
        exc: Exception,
        messages: list[dict],
        allowed_tools: list[str],
    ) -> PlanResult:
        """Fail over once per distinct planner without replaying generations."""
        self._planner_failures += 1
        fail_count = self._planner_failures
        log.warning(
            "planner failure #%d model=%s: %s",
            fail_count,
            self._planner_model,
            exc,
        )

        if not self._is_planner_failover_error(exc):
            raise exc

        old_model = self._planner_model
        self._failed_planner_models.add(old_model)
        if FALLBACK_PLANNER_MODEL not in self._failed_planner_models:
            self._planner_fallback_active = True
            self._planner_model = FALLBACK_PLANNER_MODEL
            fallback_tier = "fallback"
        else:
            return PlanResult(
                done=True,
                final=f"⏱️ Planner unavailable after {fail_count} distinct provider failures. "
                "Please retry when a planner is available or switch to /planning direct.",
                steps=[],
                planner_text="",
                completion_signal="planner_unavailable",
            )

        log.warning(
            "planner %s activated: %s → %s after availability failure",
            fallback_tier,
            old_model,
            self._planner_model,
        )
        tool_instructions = self._build_planner_tool_prompt(allowed_tools)
        fallback_messages = [dict(message) for message in messages]
        if fallback_messages and fallback_messages[0].get("role") == "system":
            content = fallback_messages[0].get("content", "")
            fallback_messages[0]["content"] = (
                f"{content}\n\nPLANNER_FALLBACK: Using {self._planner_model} "
                f"because {old_model} is unavailable.\n\n{tool_instructions}"
            )
        req = GatewayRequest(
            model=self._planner_model,
            messages=fallback_messages,
            tools=None,
            metadata={
                "orchestrator": True,
                "phase": "plan",
                "fallback": True,
            },
        )
        try:
            resp = await self._plan_request(req)
            self._planner_failures = 0
            return self._parse_planner_output(resp.content or "", allowed_tools)
        except SpendGuardTripped:
            raise
        except Exception as fallback_exc:
            return await self._handle_planner_failure(
                fallback_exc,
                messages,
                allowed_tools,
            )

    # ── Plan parsing ────────────────────────────────────────────────────────

    def _parse_planner_output(self, content: str, allowed_tools: list[str]) -> PlanResult:
        """Parse planner's plan or detect final answer.

        Handles:
          - DONE: <text> markers
          - PLAN: sections with numbered steps
          - JSON-style tool call blocks (fallback when planner emits JSON)
          - Fallback: treat as final if no structured plan found
        """
        text = content.strip()
        allowed = set(allowed_tools)

        # ── DONE: marker ──
        m_done = re.match(r"(?i)^done\s*:\s*(.+?)(?:\s*\n\s*(?:PLAN|THINK)\s*:|$)", text, re.DOTALL)
        if m_done:
            return PlanResult(
                done=True,
                final=m_done.group(1).strip(),
                steps=[],
                planner_text=text,
                completion_signal="explicit_done",
            )

        # ── PLAN: section (preferred format) ──
        plan_steps = self._extract_plan_steps(text, allowed)
        if plan_steps:
            return PlanResult(
                done=False,
                final="",
                steps=plan_steps,
                planner_text=text,
            )

        # ── JSON tool_call block (fallback when planner emits JSON natively) ──
        json_steps = self._extract_json_tool_calls(text, allowed)
        if json_steps:
            return PlanResult(
                done=False,
                final="",
                steps=json_steps,
                planner_text=text,
            )

        # A PLAN marker with no permitted, parseable steps is not a final
        # answer. Treating the planner's scratch text as completion can turn a
        # misspelled or disallowed tool request into a false success.
        if re.search(r"(?im)^PLAN\s*:\s*$", text):
            return PlanResult(
                done=True,
                final="Planner returned no executable steps; the task was not completed.",
                steps=[],
                planner_text=text,
                completion_signal="invalid_plan",
            )

        # ── No plan found → treat as final answer ──
        return PlanResult(
            done=True,
            final=text,
            steps=[],
            planner_text=text,
            completion_signal="unstructured_final" if text else "empty_response",
        )

    def _extract_plan_steps(self, text: str, allowed: set[str]) -> list[dict]:
        """Extract numbered plan steps from a PLAN: section.

        Supports multiple formats:
          1. read path="foo.py" reason="inspect"
          2. read path=foo.py reason=inspect
          3. exec command="pytest -q" reason="verify"
          4. Multiline JSON-like args (next-line indented)
        """
        steps: list[dict] = []

        # Find PLAN: section
        plan_match = re.search(r"(?im)^PLAN\s*:\s*$", text)
        if not plan_match:
            return []
        plan_start = plan_match.end()
        plan_text = text[plan_start:]

        # Find end of plan section (before ===, DONE:, or next section header)
        end_match = re.search(r"(?im)^(?:===|DONE\s*:|THINK\s*:|SCRATCHPAD)", plan_text)
        if end_match:
            plan_text = plan_text[: end_match.start()]

        # Parse numbered steps
        step_pattern = re.compile(
            r"^\s*(\d+)[.)]\s*(.+?)(?=\s*\d+[.)]\s*|\s*$)", re.MULTILINE | re.DOTALL
        )
        for m in step_pattern.finditer(plan_text):
            step_line = m.group(2).strip()
            if not step_line:
                continue

            # Parse: tool_name [args]
            parts = step_line.split(None, 1)
            if not parts:
                continue
            tool_name = parts[0].strip().lower()
            if tool_name not in allowed:
                continue

            args_str = parts[1] if len(parts) > 1 else ""
            args = self._parse_args(args_str)

            # If no args parsed, treat entire step_line after tool_name as a raw command
            # (handles cases like "exec command='pytest -q'" where Composer uses single quotes)
            if not args and len(parts) > 1:
                args = self._parse_args_loose(args_str, tool_name)

            steps.append(
                {
                    "tool": tool_name,
                    "args": args,
                    "reason": args_str[:200],
                }
            )

        return steps

    def _extract_json_tool_calls(self, text: str, allowed: set[str]) -> list[dict]:
        """Extract JSON tool_call blocks when planner emits native JSON.

        Handles:
          - {"name": "tool", "arguments": {...}}
          - {"function": {"name": "tool", "arguments": "..."}}
          - ```json ... ``` code blocks
        """
        steps: list[dict] = []

        # Try JSON code blocks first
        for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
            block = m.group(1).strip()
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                continue
            step = self._normalize_json_step(parsed, allowed)
            if step:
                steps.append(step)

        if steps:
            return steps

        # Try inline JSON objects
        json_obj_pattern = re.compile(
            r'\{(?:[^{}]|\{[^{}]*\})*"(?:name|function)"\s*:\s*"[^"]+"'
            r"(?:[^{}]|\{[^{}]*\})*\}",
        )
        for m in json_obj_pattern.finditer(text):
            try:
                parsed = json.loads(m.group())
            except json.JSONDecodeError:
                continue
            step = self._normalize_json_step(parsed, allowed)
            if step:
                steps.append(step)

        return steps

    @staticmethod
    def _normalize_json_step(parsed: dict, allowed: set[str]) -> dict | None:
        """Normalize a parsed JSON step into {tool, args}. Returns None if invalid."""
        if "function" in parsed:
            fn = parsed["function"]
            name = fn.get("name", "").strip().lower()
            args_raw = fn.get("arguments", "{}")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = args_raw if isinstance(args_raw, dict) else {}
        elif "name" in parsed:
            name = parsed.get("name", "").strip().lower()
            args = parsed.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
        else:
            return None

        if name in allowed:
            return {"tool": name, "args": args, "reason": "json"}

        return None

    def _parse_args(self, args_str: str) -> dict[str, Any]:
        """Parse key=value pairs from a plan step argument string.

        Supports:
          - key="value with spaces"
          - key='value with spaces'
          - key=value_without_spaces
          - Multiline continuation (value spills to next line)
        """
        args: dict[str, Any] = {}

        # Single/double-quoted values (supports spaces)
        for m in re.finditer(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"', args_str):
            args[m.group(1)] = m.group(2)
        for m in re.finditer(r"(\w+)\s*=\s*'((?:[^'\\]|\\.)*)'", args_str):
            if m.group(1) not in args:
                args[m.group(1)] = m.group(2)

        # Unquoted values (no spaces)
        for m in re.finditer(r'(\w+)\s*=\s*([^\s"\']+)', args_str):
            if m.group(1) not in args:
                val = m.group(2)
                # Strip trailing commas
                val = val.rstrip(",")
                args[m.group(1)] = val

        return args

    @staticmethod
    def _parse_args_loose(args_str: str, tool_name: str) -> dict[str, Any]:
        """Fallback parser for non-standard arg formats.

        Special handling per tool:
          - exec/shell: treat remaining text as the "command" arg
          - read: first token after args is "path"
          - write/edit: first token is "path", rest is "content" / "old"
        """
        args_str = args_str.strip().strip("'\"")
        if tool_name in ("exec", "shell"):
            return {"command": args_str}
        if tool_name == "read":
            return {"path": args_str.split()[0]} if args_str else {}
        if tool_name in ("write", "edit"):
            parts = args_str.split(None, 1)
            if parts:
                return {"path": parts[0]}
        return {}

    # ── Tool feedback formatting ────────────────────────────────────────────

    def _format_tool_feedback(self, tool_results: list[dict]) -> str:
        """Summarize executor output for the planner's next THINK/PLAN pass.

        Compresses large outputs to signal keys only. Highlights errors.
        """
        ok = sum(1 for r in tool_results if r.get("ok") is True)
        fail = len(tool_results) - ok

        # Compress each result to signal keys
        compressed = [self._compress_tool_result(r) for r in tool_results]
        result_json = json.dumps(compressed, default=str, indent=2)
        if len(result_json) > _MAX_RESULT_LENGTH:
            result_json = result_json[: _MAX_RESULT_LENGTH - 50] + "\n… [truncated]"

        lines = [
            f"Tool execution results ({len(tool_results)} steps, {ok} ok, {fail} failed):",
            result_json,
        ]
        if fail:
            failed_names = {r.get("tool", "?") for r in tool_results if r.get("ok") is not True}
            lines.append(
                f"\n⚠️ Failed tools: {', '.join(sorted(failed_names))}. "
                "THINK about the error cause, then PLAN recovery or alternate approach. "
                "Do NOT repeat identical failing steps."
            )
        else:
            lines.append(
                "\nAll tools succeeded. Inspect results, PLAN next steps, "
                "or output DONE: if complete."
            )
        return "\n".join(lines)

    @staticmethod
    def _compress_tool_result(result: dict) -> dict:
        """Keep only signal keys to reduce token spend on planner context."""
        tool_name = result.get("name") or result.get("tool", "")
        signal_keys = _RESULT_SIGNAL_KEYS.get(tool_name)
        if not signal_keys:
            return result

        compressed: dict[str, Any] = {}
        inner = result.get("result", {})
        for key in signal_keys:
            if key in result:
                compressed[key] = result[key]
            elif key in inner:
                compressed[key] = inner[key]

        # Always include error/error message
        for err_key in ("error", "stderr"):
            val = result.get(err_key) or inner.get(err_key)
            if val:
                compressed[err_key] = str(val)[:1000]

        compressed["via"] = result.get("via", "direct")
        return compressed

    # ── Planner tool prompt ─────────────────────────────────────────────────

    def _build_planner_tool_prompt(self, tools: list[str]) -> str:
        """Tell the planner how to request tool calls (text format)."""
        tool_list = "\n".join(f"    - {t}" for t in sorted(tools))
        return (
            f"\n=== AVAILABLE TOOLS (Qwen executes; you plan in text) ===\n"
            f"{tool_list}\n\n"
            "Protocol:\n"
            "  THINK:\n"
            "    (analyze what you know, what's missing, what to verify)\n"
            "  PLAN:\n"
            '    1. tool_name arg1="value" arg2="value" reason="why"\n'
            '    2. tool_name arg="value" reason="why"\n'
            "  Or: DONE: <final answer> when task is complete.\n\n"
            "Rules:\n"
            "  - 1-3 steps per PLAN block\n"
            "  - Use exact paths, exact old/new for edits\n"
            "  - On error: diagnose in THINK, then PLAN recovery (different approach)\n"
            "  - Verify writes with read/exec before DONE\n"
        )

    # ── Plan execution ──────────────────────────────────────────────────────

    async def _execute_plan(
        self,
        plan: PlanResult,
        allowed_tools: list[str],
        sender_tier: str,
        *,
        event_log: Any | None = None,
    ) -> list[dict]:
        """Execute each planned tool — direct dispatch when args are complete, else Qwen."""
        results: list[dict] = []
        allowed = set(allowed_tools)

        for step in plan.steps:
            tool_name = step["tool"]
            if tool_name not in allowed:
                failure = {
                    "ok": False,
                    "error": f"Tool '{tool_name}' not in allowed set",
                }
                results.append(
                    {
                        "name": tool_name,
                        "tool": tool_name,
                        "ok": False,
                        "error": failure["error"],
                        "args": step.get("args") or {},
                        "result": failure,
                        "via": "rejected",
                    }
                )
                continue

            args = step.get("args") or {}
            if args and self._args_complete(tool_name, args):
                result = await self._dispatch_tool(
                    tool_name,
                    args,
                    sender_tier=sender_tier,
                    event_log=event_log,
                )
                results.append(
                    {
                        "name": tool_name,
                        "tool": tool_name,
                        "args": args,
                        "ok": result.get("ok") is True,
                        "result": result,
                        "via": "direct",
                    }
                )
                continue

            # Incomplete args — let Qwen Coder fill in and call natively
            qwen_result = await self._execute_via_qwen(
                step,
                allowed_tools,
                sender_tier,
                event_log=event_log,
            )
            results.append(qwen_result)

        return results

    def _args_complete(self, tool_name: str, args: dict) -> bool:
        """True when planner supplied enough args to dispatch without Qwen."""
        from ..dispatch.tools import REGISTRY

        spec = REGISTRY.get(tool_name)
        if spec is None:
            return False
        schema = spec.schema or {}
        required = [k for k, v in schema.items() if not str(v).endswith("?")]
        return all(args.get(k) not in (None, "") for k in required)

    async def _dispatch_tool(
        self,
        name: str,
        args: dict,
        *,
        sender_tier: str,
        event_log: Any | None = None,
    ) -> dict:
        from .agent_loop import _run_one_tool

        log_ev = event_log or _NullEventLog()
        return await _run_one_tool(name, args, sender_tier=sender_tier, event_log=log_ev)

    async def _execute_via_qwen(
        self,
        step: dict,
        allowed_tools: list[str],
        sender_tier: str,
        *,
        event_log: Any | None = None,
    ) -> dict:
        """Ask Qwen Coder to make one native tool call for this plan step."""
        executor_prompt = self._build_executor_prompt(step)
        tool_name = step["tool"]
        schemas = self._tool_schema(tool_name)
        if not schemas:
            failure = {"ok": False, "error": "unknown_tool_schema"}
            return {
                "name": tool_name,
                "tool": tool_name,
                "args": step.get("args") or {},
                "ok": False,
                "error": failure["error"],
                "result": failure,
                "via": "qwen",
            }

        req = GatewayRequest(
            model=EXECUTOR_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Qwen Coder — a precise tool executor. "
                        "Call exactly one tool with the arguments from the plan."
                    ),
                },
                {"role": "user", "content": executor_prompt},
            ],
            tools=[schemas],
            metadata={"orchestrator": True, "phase": "execute"},
        )

        try:
            resp = self._account_response(
                await asyncio.wait_for(self.router.chat(req), timeout=EXECUTOR_TIMEOUT)
            )
            if len(resp.tool_calls) != 1:
                failure = {
                    "ok": False,
                    "error": "executor_expected_exactly_one_tool_call",
                    "tool_call_count": len(resp.tool_calls),
                }
                return {
                    "name": tool_name,
                    "tool": tool_name,
                    "ok": False,
                    "error": failure["error"],
                    "args": step.get("args") or {},
                    "result": failure,
                    "executor_response": (resp.content or "")[:500],
                    "via": "qwen",
                }

            from ..dispatch.tools import normalize_tool_args, normalize_tool_name

            tc = resp.tool_calls[0]
            fn = tc.get("function", {})
            resolved = normalize_tool_name(fn.get("name", tool_name))
            expected = normalize_tool_name(tool_name)
            if resolved != expected or resolved not in set(allowed_tools):
                failure = {
                    "ok": False,
                    "error": "executor_tool_mismatch",
                    "expected": expected,
                    "received": resolved,
                }
                return {
                    "name": tool_name,
                    "tool": tool_name,
                    "args": step.get("args") or {},
                    "ok": False,
                    "error": failure["error"],
                    "result": failure,
                    "via": "qwen",
                }
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = step.get("args") or {}
            args = normalize_tool_args(resolved, args if isinstance(args, dict) else {})

            result = await self._dispatch_tool(
                resolved,
                args,
                sender_tier=sender_tier,
                event_log=event_log,
            )
            return {
                "name": resolved,
                "tool": resolved,
                "args": args,
                "ok": result.get("ok") is True,
                "result": result,
                "via": "qwen",
            }
        except SpendGuardTripped:
            raise
        except TimeoutError:
            failure = {"ok": False, "error": "executor_timeout"}
            return {
                "name": tool_name,
                "tool": tool_name,
                "args": step.get("args") or {},
                "ok": False,
                "error": failure["error"],
                "result": failure,
                "via": "qwen",
            }
        except Exception as e:
            failure = {"ok": False, "error": str(e)}
            return {
                "name": tool_name,
                "tool": tool_name,
                "args": step.get("args") or {},
                "ok": False,
                "error": failure["error"],
                "result": failure,
                "via": "qwen",
            }

    def _build_executor_prompt(self, step: dict) -> str:
        """Build prompt asking Qwen to execute a specific tool."""
        return (
            f"Execute this tool call precisely:\n\n"
            f"Tool: {step['tool']}\n"
            f"Arguments: {json.dumps(step['args'])}\n\n"
            f"Reason: {step.get('reason', 'none')}\n\n"
            f"Call the tool with these exact arguments."
        )

    def _tool_schema(self, tool_name: str) -> dict | None:
        """Get OpenAI-format schema for a tool."""
        from ..dispatch.tools import render_tools_for_llm

        schemas = render_tools_for_llm([tool_name])
        return schemas[0] if schemas else None

    async def _call_planner_for_final(self, messages: list[dict]) -> str:
        """Force planner to give final answer after max rounds."""
        final_messages = list(messages)
        final_messages.append(
            {
                "role": "user",
                "content": (
                    "Round budget reached. Provide your final answer now based on all tool "
                    "results. Do NOT plan more tools."
                ),
            }
        )
        req = GatewayRequest(
            model=self._planner_model,
            messages=final_messages,
            tools=None,
            metadata={"orchestrator": True, "phase": "final"},
        )
        try:
            resp = await self._plan_request(req)
            return resp.content or ""
        except SpendGuardTripped:
            raise
        except Exception:
            return "⏱️ Final summary unavailable."

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _finalize_run(
        self,
        *,
        content: str,
        trace: list[dict],
        rounds: int,
        user_prompt: str,
        completion_signal: str,
    ) -> OrchestratorRunResult:
        """Create a terminal result from observable completion evidence.

        Planner prose is a completion *signal*, not proof of a successful
        mutation. Local output checks and target-aware mutation verification
        keep outage messages, invalid plans, and unverified writes out of
        completed checkpoints and positive training labels.
        """
        from .output_verifier import IssueSeverity, OutputVerifier

        final_content = str(content or "").strip()
        mutation = summarize_mutation_trace(trace)
        report = OutputVerifier().verify(
            output=final_content,
            user_request=user_prompt,
            tool_trace=trace,
        )
        blocking_categories = sorted(
            {issue.category for issue in report.issues if issue.severity is IssueSeverity.CRITICAL}
        )

        accepted_signal = completion_signal in {"explicit_done", "unstructured_final"}
        mutation_verified = not mutation.attempted or mutation.supports_success_claim
        complete = bool(
            final_content and accepted_signal and not blocking_categories and mutation_verified
        )

        if not final_content:
            status_reason = "empty_response"
        elif not accepted_signal:
            status_reason = completion_signal
        elif blocking_categories:
            status_reason = "output_verification_failed"
        elif not mutation_verified:
            status_reason = "mutation_unverified"
        else:
            status_reason = "verified_terminal_response"

        # Do not deliver a known false-success draft. Round-limit summaries are
        # still useful, but label them as partial instead of implying success.
        if status_reason in {"output_verification_failed", "mutation_unverified"}:
            reasons = ", ".join(blocking_categories) or "post-action verification is missing"
            final_content = (
                "I could not verify a complete result, so I withheld the draft's success "
                f"claim. Blocking evidence: {reasons}."
            )
        elif completion_signal == "round_limit" and final_content:
            final_content = f"Incomplete (orchestration round limit reached):\n\n{final_content}"

        return OrchestratorRunResult(
            content=final_content,
            trace=trace,
            rounds=rounds,
            complete=complete,
            verified_outcome=complete,
            status_reason=status_reason,
            completion_signal=completion_signal,
            planner_model=self._planner_model,
            usage=dict(self._turn_usage),
            mutation_outcome={
                "attempted": mutation.attempted,
                "succeeded": mutation.succeeded,
                "failed": mutation.failed,
                "verified_after_last_mutation": mutation.verified_after_last_mutation,
            },
            blocking_categories=blocking_categories,
        )

    @staticmethod
    def _resolve_round_cap(max_rounds: int) -> int:
        """Map runtime round cap to orchestrator budget."""
        if max_rounds <= 0:
            return MAX_ORCH_ROUNDS
        return min(max_rounds, HARD_ORCH_ROUND_CAP)


# ── Routing helpers ────────────────────────────────────────────────────────


def should_use_orchestrator(planning_mode: str, model: str) -> bool:
    """Return whether the dual-model orchestrator is explicitly enabled.

    `direct` is a hard block: do not silently auto-route planner models
    through the orchestrator. The previous auto-orchestrator behavior made `/planning
    direct` misleading and let planner-side failures surface as user-facing
    CircuitOpen errors before the main turn continued.
    """
    return planning_mode == "orchestrator"


def effective_planning_route(planning_mode: str, model: str) -> dict[str, Any]:
    """Structured routing info for /settings and Discord UI."""
    active = should_use_orchestrator(planning_mode, model)
    planner = resolve_planner_model(model) if active else model
    return {
        "active": active,
        "mode": planning_mode,
        "auto": active and planning_mode != "orchestrator",
        "planner": planner,
        "executor": EXECUTOR_MODEL if active else model,
        "fallback": FALLBACK_PLANNER_MODEL if active else None,
        "tertiary_fallback": None,
        "label": _planning_route_label(planning_mode, model, active, planner),
    }


def _planning_route_label(
    planning_mode: str,
    model: str,
    active: bool,
    planner: str,
) -> str:
    if not active:
        return "Direct — single-model agent loop"
    return f"Orchestrator — {planner} plans, {EXECUTOR_MODEL} executes"


class _NullEventLog:
    async def append(self, *_a: Any, **_kw: Any) -> None:
        return None


@dataclass
class PlanResult:
    done: bool
    final: str
    steps: list[dict]
    planner_text: str = ""
    completion_signal: str = ""


@dataclass(frozen=True)
class OrchestratorRunResult:
    """Structured orchestration result with backward-compatible unpacking."""

    content: str
    trace: list[dict]
    rounds: int
    complete: bool
    verified_outcome: bool
    status_reason: str
    completion_signal: str
    planner_model: str
    usage: dict[str, int]
    mutation_outcome: dict[str, Any]
    blocking_categories: list[str]

    def __iter__(self):
        """Yield the legacy ``(content, trace, rounds)`` tuple fields."""
        yield self.content
        yield self.trace
        yield self.rounds
