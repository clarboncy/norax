"""Multi-agent orchestration — sub-agent spawning, parallel execution, result fusion.

Extends the existing Orchestrator with:
  - Sub-agent spawn: delegate sub-tasks to specialized agents
  - Parallel execution: run multiple sub-agents concurrently
  - Result aggregation: preserve each sub-agent result in one response
  - Specialized roles: research, code, verify, plan

Architecture:
  - SubAgent: wraps a single agent loop with scoped context + tools
  - MultiAgentOrchestrator: spawns SubAgents, collects results, fuses
  - Result fusion: deterministic, attribution-preserving concatenation

This is NOT a replacement for Orchestrator — it's a layer above it.
The parent agent (Orchestrator or agent_loop) calls MultiAgentOrchestrator
when it detects a task that benefits from decomposition.

Usage:
    mao = MultiAgentOrchestrator(router)
    result = await mao.run(
        task="Research MCP protocol and implement a server",
        system_prompt="...",
        allowed_tools=["read", "write", "exec", "web_search"],
        sender_tier="owner",
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..gateway_client import SpendGuardTripped

log = logging.getLogger("norax.multi_agent")
MAX_SUBTASKS = 8
MAX_CONCURRENT_SUBAGENTS = 4
MAX_SUBAGENT_ROUNDS = 32

# ── Sub-agent roles ──────────────────────────────────────────────────────

ROLE_RESEARCH = "research"
ROLE_CODE = "code"
ROLE_VERIFY = "verify"
ROLE_PLAN = "plan"
ROLE_GENERAL = "general"

ROLE_PROMPTS: dict[str, str] = {
    ROLE_RESEARCH: (
        "You are a research specialist. Your job is to gather information, "
        "read documentation, search the web, and synthesize findings. "
        "Focus on thoroughness and accuracy. Always cite sources."
    ),
    ROLE_CODE: (
        "You are a code specialist. Your job is to write, edit, and test code. "
        "Focus on correctness, edge cases, and verification. Always run tests."
    ),
    ROLE_VERIFY: (
        "You are a verification specialist. Your job is to check claims, "
        "verify file contents, run tests, and confirm correctness. "
        "Focus on finding errors. Report pass/fail with evidence."
    ),
    ROLE_PLAN: (
        "You are a planning specialist. Your job is to decompose complex tasks "
        "into steps, identify dependencies, and create execution plans. "
        "Focus on completeness and ordering."
    ),
    ROLE_GENERAL: ("You are a general-purpose agent. Handle the task directly."),
}

ROLE_TOOL_SETS: dict[str, list[str]] = {
    ROLE_RESEARCH: ["read", "list_dir", "web_search", "web_fetch", "search_memory", "repo_explore"],
    ROLE_CODE: [
        "read",
        "write",
        "write_chunk",
        "edit",
        "exec",
        "shell",
        "list_dir",
        "search_memory",
    ],
    ROLE_VERIFY: ["read", "exec", "shell", "list_dir", "search_memory", "status"],
    ROLE_PLAN: ["read", "list_dir", "search_memory", "web_search"],
    ROLE_GENERAL: [],  # gets all allowed_tools
}


@dataclass
class SubTask:
    """A decomposed sub-task for a sub-agent."""

    id: str
    description: str
    role: str = ROLE_GENERAL
    tools: list[str] = field(default_factory=list)
    context: str = ""  # extra context from parent
    max_rounds: int = 50
    depends_on: list[str] = field(default_factory=list)  # subtask IDs this depends on


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution."""

    subtask_id: str
    role: str
    success: bool
    output: str
    tool_calls: list[dict] = field(default_factory=list)
    rounds: int = 0
    elapsed: float = 0.0
    error: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Fused result from multiple sub-agents."""

    summary: str
    sub_results: list[SubAgentResult] = field(default_factory=list)
    total_tool_calls: int = 0
    total_rounds: int = 0
    elapsed: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def verified_outcome(self) -> bool:
        """True only when every requested subtask finished with verified evidence."""
        return bool(self.sub_results) and all(result.success for result in self.sub_results)


class MultiAgentOrchestrator:
    """Spawns sub-agents for parallel task execution.

    Uses the existing Orchestrator or agent_loop for each sub-agent.
    """

    def __init__(self, router: Any, default_model: str | None = None) -> None:
        self.router = router
        self.default_model = default_model

    async def run(
        self,
        *,
        task: str,
        system_prompt: str,
        allowed_tools: list[str],
        sender_tier: str,
        subtasks: list[SubTask] | None = None,
        event_log: Any | None = None,
        max_concurrent: int = 4,
    ) -> FusionResult:
        """Run multi-agent orchestration.

        If subtasks is None, auto-decompose the task.
        """
        t0 = time.time()

        # Auto-decompose if no subtasks provided
        if subtasks is None:
            subtasks = self._auto_decompose(task, allowed_tools)

        if not subtasks:
            # Single-agent fallback
            subtasks = [
                SubTask(
                    id="s0",
                    description=task,
                    role=ROLE_GENERAL,
                    tools=allowed_tools,
                )
            ]

        if len(subtasks) > MAX_SUBTASKS:
            raise ValueError(f"multi-agent task count exceeds limit of {MAX_SUBTASKS}")

        try:
            requested_concurrency = int(max_concurrent)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_concurrent must be an integer") from exc
        concurrency = min(MAX_CONCURRENT_SUBAGENTS, max(1, requested_concurrency))
        log.info("multi_agent: %d subtasks, max_concurrent=%d", len(subtasks), concurrency)

        # Build dependency graph and execute in topological order
        results = await self._run_dag(
            subtasks, system_prompt, allowed_tools, sender_tier, concurrency, event_log
        )

        # Fuse results
        fused = self._fuse(results)
        fused.elapsed = time.time() - t0
        return fused

    async def _run_dag(
        self,
        subtasks: list[SubTask],
        system_prompt: str,
        allowed_tools: list[str],
        sender_tier: str,
        max_concurrent: int,
        event_log: Any | None = None,
    ) -> list[SubAgentResult]:
        """Execute subtasks respecting DAG dependencies.

        Tasks with no unmet dependencies run concurrently (up to max_concurrent).
        Dependent tasks receive predecessor outputs in their context.
        """
        semaphore = asyncio.Semaphore(max(1, max_concurrent))
        results: dict[str, SubAgentResult] = {}
        duplicate_ids = {
            subtask_id
            for subtask_id, count in Counter(st.id for st in subtasks).items()
            if count > 1
        }
        if duplicate_ids:
            return [
                SubAgentResult(
                    subtask_id=st.id,
                    role=st.role,
                    success=False,
                    output="",
                    error=f"duplicate subtask id: {st.id}",
                )
                for st in subtasks
            ]
        remaining = {st.id: st for st in subtasks}

        while remaining:
            # Find all tasks whose dependencies are satisfied
            ready = [
                st for st in remaining.values() if all(dep in results for dep in st.depends_on)
            ]
            if not ready:
                # Deadlock — remaining tasks have unresolvable deps
                log.error("multi_agent: deadlock — unresolvable deps for %s", list(remaining))
                for st in remaining.values():
                    results[st.id] = SubAgentResult(
                        subtask_id=st.id,
                        role=st.role,
                        success=False,
                        output="",
                        error="unresolvable dependency",
                    )
                break

            async def _run_one(st: SubTask) -> SubAgentResult:
                async with semaphore:
                    # Inject predecessor outputs into context
                    enriched_context = st.context
                    for dep_id in st.depends_on:
                        dep_result = results.get(dep_id)
                        if dep_result and dep_result.success and dep_result.output:
                            enriched_context += f"\n\n[From {dep_id} ({dep_result.role})]:\n{dep_result.output[:4000]}"
                        elif dep_result and not dep_result.success:
                            # Predecessor failed — block dependent
                            return SubAgentResult(
                                subtask_id=st.id,
                                role=st.role,
                                success=False,
                                output="",
                                error=f"dependency {dep_id} failed",
                            )
                    st_copy = SubTask(
                        id=st.id,
                        description=st.description,
                        role=st.role,
                        tools=st.tools,
                        context=enriched_context,
                        max_rounds=st.max_rounds,
                        depends_on=st.depends_on,
                    )
                    return await self._run_sub_agent(
                        st_copy,
                        system_prompt,
                        allowed_tools,
                        sender_tier,
                        event_log=event_log,
                    )

            tasks = [asyncio.create_task(_run_one(st)) for st in ready]
            try:
                batch_results = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            for st, res in zip(ready, batch_results, strict=True):
                results[st.id] = res
                del remaining[st.id]

        return [results[st.id] for st in subtasks]

    def _auto_decompose(self, task: str, allowed_tools: list[str]) -> list[SubTask]:
        """Heuristic task decomposition.

        For now, uses a simple heuristic:
        - If task mentions "research" or "investigate" → research subtask
        - If task mentions "implement" or "code" → code subtask
        - If task mentions "verify" or "test" → verify subtask
        - Otherwise → single general subtask

        Future: use the planner model to decompose.
        """
        task_lower = task.lower()
        subtasks: list[SubTask] = []

        # Check for compound tasks (contains "and", "then", "after that")
        has_research = any(w in task_lower for w in ["research", "investigate", "analyze", "study"])
        has_code = any(w in task_lower for w in ["implement", "code", "write", "build", "create"])
        has_verify = any(w in task_lower for w in ["verify", "test", "check", "validate"])

        if has_research and (has_code or has_verify):
            execute_role = ROLE_CODE if has_code else ROLE_VERIFY
            subtasks.append(
                SubTask(
                    id="s_research",
                    description=f"Research phase: {task}",
                    role=ROLE_RESEARCH,
                    tools=ROLE_TOOL_SETS.get(ROLE_RESEARCH, allowed_tools),
                    max_rounds=10,
                )
            )
            subtasks.append(
                SubTask(
                    id="s_execute",
                    description=f"Execution phase (use research findings): {task}",
                    role=execute_role,
                    tools=ROLE_TOOL_SETS.get(execute_role, allowed_tools),
                    context="Use findings from the research phase.",
                    max_rounds=15,
                    depends_on=["s_research"],
                )
            )
        elif has_research:
            subtasks.append(
                SubTask(
                    id="s0",
                    description=task,
                    role=ROLE_RESEARCH,
                    tools=ROLE_TOOL_SETS.get(ROLE_RESEARCH, allowed_tools),
                )
            )
        elif has_code and has_verify:
            subtasks.append(
                SubTask(
                    id="s_code",
                    description=f"Code phase: {task}",
                    role=ROLE_CODE,
                    tools=ROLE_TOOL_SETS.get(ROLE_CODE, allowed_tools),
                    max_rounds=15,
                )
            )
            subtasks.append(
                SubTask(
                    id="s_verify",
                    description=f"Verify the code changes from the code phase: {task}",
                    role=ROLE_VERIFY,
                    tools=ROLE_TOOL_SETS.get(ROLE_VERIFY, allowed_tools),
                    context="Verify the code written in the code phase.",
                    max_rounds=8,
                    depends_on=["s_code"],
                )
            )
        else:
            subtasks.append(
                SubTask(
                    id="s0",
                    description=task,
                    role=ROLE_GENERAL,
                    tools=allowed_tools,
                )
            )

        return subtasks

    async def _run_sub_agent(
        self,
        subtask: SubTask,
        system_prompt: str,
        allowed_tools: list[str],
        sender_tier: str,
        *,
        event_log: Any | None = None,
    ) -> SubAgentResult:
        """Run a single sub-agent using the existing agent loop."""
        t0 = time.time()
        role_prompt = ROLE_PROMPTS.get(subtask.role, ROLE_PROMPTS[ROLE_GENERAL])

        # Merge role tools with allowed tools (intersect)
        if subtask.tools:
            tools = [t for t in subtask.tools if t in allowed_tools]
        else:
            tools = allowed_tools

        # Build sub-agent system prompt
        sub_system = f"{role_prompt}\n\n{system_prompt}"
        if subtask.context:
            sub_system += f"\n\nContext: {subtask.context}"

        # Build user prompt
        user_prompt = subtask.description

        try:
            from .agent_loop import run_agent_loop

            # run_agent_loop needs a gateway (GatewayClient or GatewayRouter)
            # and a concrete model string.  The router passed at construction
            # time duck-types as a GatewayClient.  We resolve a default model
            # from the router's provider config, falling back to the runtime
            # default.
            gateway = self.router
            model = (
                getattr(self, "default_model", None)
                or getattr(gateway, "default_model", None)
                or getattr(gateway, "_effective_model", None)
            )
            if not model:
                raise RuntimeError("multi-agent execution requires a configured model")

            try:
                round_budget = int(subtask.max_rounds)
            except (TypeError, ValueError) as exc:
                raise ValueError("subtask max_rounds must be an integer") from exc
            round_budget = min(MAX_SUBAGENT_ROUNDS, max(1, round_budget))
            resp, trace, rounds, _ts = await run_agent_loop(
                gateway=gateway,
                model=model,
                system_prompt=sub_system,
                user_prompt=user_prompt,
                allowed_tools=tools,
                sender_tier=sender_tier,
                event_log=event_log or _NullEventLog(),
                max_rounds=round_budget,
            )

            content = resp.content if resp is not None else ""
            raw = dict(getattr(resp, "raw", {}) or {}) if resp is not None else {}
            success = bool(content and content.strip() and raw.get("verified_outcome") is True)
            failure_reason = str(
                raw.get("limit_reason")
                or raw.get("incomplete_reason")
                or raw.get("stop_reason")
                or "sub-agent returned an unverified or empty result"
            )

            return SubAgentResult(
                subtask_id=subtask.id,
                role=subtask.role,
                success=success,
                output=content or "",
                tool_calls=trace,
                rounds=rounds,
                elapsed=time.time() - t0,
                error="" if success else failure_reason,
                usage={
                    "input_tokens": int(
                        (getattr(resp, "usage", {}) or {}).get("input_tokens") or 0
                    ),
                    "output_tokens": int(
                        (getattr(resp, "usage", {}) or {}).get("output_tokens") or 0
                    ),
                },
            )
        except SpendGuardTripped:
            raise
        except Exception as exc:
            log.error("sub_agent %s failed: %s", subtask.id, exc)
            return SubAgentResult(
                subtask_id=subtask.id,
                role=subtask.role,
                success=False,
                output="",
                error=str(exc),
                elapsed=time.time() - t0,
            )

    def _fuse(self, results: list[SubAgentResult]) -> FusionResult:
        """Fuse results without inventing a second, unverified synthesis."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        total_tc = sum(len(r.tool_calls) for r in results)
        total_rounds = sum(r.rounds for r in results)
        usage = {
            key: sum(int((result.usage or {}).get(key) or 0) for result in results)
            for key in ("input_tokens", "output_tokens")
        }

        if len(results) == 1 and results[0].success:
            summary = results[0].output
        elif len(successful) == 0:
            summary = "All sub-agents failed:\n" + "\n".join(
                f"  [{r.role}] {r.error}" for r in failed
            )
        else:
            parts = []
            for r in successful:
                header = f"## [{r.role}] {r.subtask_id}"
                parts.append(f"{header}\n{r.output}")
            if failed:
                parts.append(
                    "## Failures\n"
                    + "\n".join(f"  [{r.role}] {r.subtask_id}: {r.error}" for r in failed)
                )
            summary = "\n\n".join(parts)

        return FusionResult(
            summary=summary,
            sub_results=results,
            total_tool_calls=total_tc,
            total_rounds=total_rounds,
            usage=usage,
        )


def should_decompose(task: str) -> bool:
    """Heuristic: should this task be decomposed into sub-agents?"""
    task_lower = task.lower()
    # Multi-phase indicators
    phase_markers = [" and ", " then ", " after that ", " first ", " second ", " finally "]
    role_markers = ["research", "implement", "verify", "test", "build", "analyze"]

    phase_count = sum(1 for m in phase_markers if m in task_lower)
    role_count = sum(1 for m in role_markers if m in task_lower)

    # Decompose if: multiple phases with multiple roles, OR 3+ roles without phases
    return (phase_count >= 1 and role_count >= 2) or role_count >= 3


class _NullEventLog:
    async def append(self, *_args: Any, **_kwargs: Any) -> None:
        return None
