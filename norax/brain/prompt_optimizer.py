"""Prompt candidate evaluation against explicit, objective benchmarks.

Generates prompt variants, tests them against benchmark tasks,
and returns the best measured candidate. It does not deploy prompts.

Architecture:
  - PromptVariant: a candidate prompt with metadata
  - PromptBenchmark: a set of test tasks with expected outcomes
  - PromptOptimizer: generates, tests, and selects best prompts

This is NOT about optimizing the system prompt (SOUL) — it's about
optimizing task-specific prompts (tool instructions, planning prompts,
retrieval prompts, etc.) that the agent generates dynamically.

Usage:
    opt = PromptOptimizer(router=gateway)
    best = await opt.optimize(
        base_prompt="Read the file and summarize it",
        benchmark_tasks=[
            {"input": "read /etc/hostname", "expected_contains": "hostname"},
        ],
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..atomic import append_bounded_text, read_bounded_text

log = logging.getLogger("norax.prompt_optimizer")

_MAX_BASE_PROMPT_CHARS = 64_000
_MAX_TASK_INPUT_CHARS = 16_000
_MAX_ASSERTIONS_PER_FIELD = 64
_MAX_ASSERTION_CHARS = 4_000
_MAX_TOOL_NAME_CHARS = 128
_MAX_BENCHMARK_TOOLS = 64
_MAX_BENCHMARK_TOOLS_BYTES = 256 * 1024
_MAX_MODEL_CHARS = 256
_MAX_HISTORY_BYTES = 64 * 1024 * 1024
_MAX_HISTORY_LINES = 100_000
_MAX_RETAINED_VARIANTS = 10_000
_EVALUATION_TIMEOUT_SECONDS = 30.0
_EVALUATION_MAX_TOKENS = 4_096

# ── Data structures ─────────────────────────────────────────────────────


@dataclass
class PromptVariant:
    """A candidate prompt variant."""

    id: str
    text: str
    score: float = 0.0
    tested_on: int = 0
    failed_on: int = 0
    created_at: float = field(default_factory=time.time)
    parent_id: str = ""
    mutation: str = ""  # what changed from parent


@dataclass
class BenchmarkTask:
    """A benchmark task for evaluating prompts."""

    input: str
    expected_contains: str | list[str] = ""
    expected_not_contains: str | list[str] = ""
    tool_should_be_called: str = ""  # expected tool name
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Result of a prompt optimization run."""

    best_variant: PromptVariant
    all_variants: list[PromptVariant] = field(default_factory=list)
    iterations: int = 0
    improvement: float = 0.0  # best_score - baseline_score
    evaluations_attempted: int = 0
    evaluations_completed: int = 0
    evaluation_failures: int = 0


# ── Mutation strategies ─────────────────────────────────────────────────

MUTATIONS = [
    # Each mutation transforms a prompt variant
    ("clarity", "Rewrite for maximum clarity and directness. Remove ambiguity."),
    ("constraints", "Add explicit constraints: 'Do NOT do X. ALWAYS do Y.'"),
    ("structure", "Restructure with numbered steps or bullet points."),
    ("context", "Add context about why this task matters and what follows."),
    ("simplify", "Simplify language. Shorter sentences. Remove jargon."),
    ("precision", "Make instructions more precise with exact field names and types."),
    ("edge_cases", "Add edge case handling instructions."),
]


def _validate_benchmark_tools(value: Any) -> set[str]:
    if not isinstance(value, list):
        raise TypeError("benchmark tools must be a list")
    if len(value) > _MAX_BENCHMARK_TOOLS:
        raise ValueError("benchmark tools exceeds its entry limit")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("benchmark tools must contain finite JSON data") from exc
    if len(encoded) > _MAX_BENCHMARK_TOOLS_BYTES:
        raise ValueError("benchmark tools exceeds its byte limit")

    names: set[str] = set()
    for tool in value:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("benchmark tools must use function-tool schemas")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError("benchmark function tool is missing its function object")
        name = function.get("name")
        if (
            not isinstance(name, str)
            or not name
            or len(name) > _MAX_TOOL_NAME_CHARS
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]*", name)
        ):
            raise ValueError("benchmark function tool has an invalid name")
        if name in names:
            raise ValueError("benchmark function tool names must be unique")
        names.add(name)
    return names


# ── Prompt Optimizer ─────────────────────────────────────────────────────


class PromptOptimizer:
    """Optimize prompts through mutation and benchmarking.

    Uses the agent's own gateway to test prompt variants.
    """

    def __init__(
        self,
        router: Any | None = None,
        storage_dir: Path | None = None,
    ) -> None:
        self.router = router
        self.storage_dir = storage_dir or Path.home() / ".norax" / "prompt_opt"
        self.storage_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        storage_stat = self.storage_dir.lstat()
        if not stat.S_ISDIR(storage_stat.st_mode):
            raise ValueError("prompt optimizer storage must be a real directory")
        if os.name == "posix":
            os.chmod(self.storage_dir, 0o700)
        self._history: list[PromptVariant] = []

    async def optimize(
        self,
        base_prompt: str,
        benchmark_tasks: list[BenchmarkTask],
        *,
        iterations: int = 3,
        variants_per_iter: int = 4,
        model: str = "",
    ) -> OptimizationResult:
        """Optimize a prompt through mutation and selection.

        Args:
            base_prompt: The initial prompt to optimize
            benchmark_tasks: Tasks to evaluate prompt quality
            iterations: Number of optimization iterations
            variants_per_iter: Variants to generate per iteration
            model: Model to use for testing (empty = default)

        Returns:
            OptimizationResult with the best variant
        """
        if self.router is None:
            raise ValueError("a live gateway/router is required for prompt evaluation")
        if not isinstance(base_prompt, str):
            raise TypeError("base_prompt must be text")
        if not base_prompt.strip():
            raise ValueError("base_prompt must not be empty")
        if len(base_prompt) > _MAX_BASE_PROMPT_CHARS:
            raise ValueError(f"base_prompt exceeds {_MAX_BASE_PROMPT_CHARS} characters")
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or not 1 <= iterations <= 10
        ):
            raise ValueError("iterations must be between 1 and 10")
        if (
            isinstance(variants_per_iter, bool)
            or not isinstance(variants_per_iter, int)
            or not 1 <= variants_per_iter <= 8
        ):
            raise ValueError("variants_per_iter must be between 1 and 8")
        if (
            not isinstance(benchmark_tasks, list)
            or not benchmark_tasks
            or len(benchmark_tasks) > 100
        ):
            raise ValueError("between 1 and 100 benchmark tasks are required")
        if not isinstance(model, str) or len(model) > _MAX_MODEL_CHARS:
            raise ValueError(f"model must be text of at most {_MAX_MODEL_CHARS} characters")
        for task in benchmark_tasks:
            if not isinstance(task, BenchmarkTask):
                raise TypeError("benchmark_tasks must contain BenchmarkTask objects")
            if (
                not isinstance(task.input, str)
                or not task.input.strip()
                or len(task.input) > _MAX_TASK_INPUT_CHARS
            ):
                raise ValueError("every benchmark task needs a non-empty input")
            for field_name in ("expected_contains", "expected_not_contains"):
                value = getattr(task, field_name)
                if not isinstance(value, str | list):
                    raise TypeError(f"{field_name} must be text or a list of text")
                items = value if isinstance(value, list) else [value]
                if len(items) > _MAX_ASSERTIONS_PER_FIELD or not all(
                    isinstance(item, str) and len(item) <= _MAX_ASSERTION_CHARS for item in items
                ):
                    raise ValueError(f"{field_name} exceeds its benchmark limits")
            if (
                not isinstance(task.tool_should_be_called, str)
                or len(task.tool_should_be_called) > _MAX_TOOL_NAME_CHARS
            ):
                raise ValueError("tool_should_be_called must be bounded text")
            tool_names = _validate_benchmark_tools(task.tools)
            if task.tool_should_be_called and task.tool_should_be_called not in tool_names:
                raise ValueError(
                    "tool_should_be_called must name a supplied benchmark function tool"
                )
        if any(not self._task_has_assertions(task) for task in benchmark_tasks):
            raise ValueError("every benchmark task needs at least one objective assertion")

        run_history: list[PromptVariant] = []

        # Create baseline variant
        baseline = PromptVariant(
            id=self._hash(base_prompt),
            text=base_prompt,
        )
        baseline.score = await self._evaluate(baseline, benchmark_tasks, model)
        self._remember(baseline)
        run_history.append(baseline)

        best = baseline
        all_variants = [baseline]
        completed_iterations = 0

        for iteration in range(iterations):
            if best.score >= 1.0:
                break
            log.info(
                "prompt_opt: iteration %d/%d, best_score=%.2f",
                iteration + 1,
                iterations,
                best.score,
            )

            # Generate variants
            variants = await self._generate_variants(best, variants_per_iter)
            if not variants:
                break
            completed_iterations += 1

            # Evaluate variants
            eval_tasks = [self._evaluate(v, benchmark_tasks, model) for v in variants]
            scores = await asyncio.gather(*eval_tasks, return_exceptions=True)

            for v, score in zip(variants, scores, strict=False):
                if isinstance(score, BaseException):
                    v.score = 0.0
                    v.failed_on = len(benchmark_tasks)
                    log.warning("prompt_opt: eval failed: %s", score)
                else:
                    v.score = float(score)
                all_variants.append(v)
                self._remember(v)
                run_history.append(v)

            # Select best
            candidates = [best] + variants
            best = max(candidates, key=lambda v: v.score)

        await asyncio.to_thread(self._save_history, run_history)

        evaluations_completed = sum(variant.tested_on for variant in all_variants)
        evaluation_failures = sum(variant.failed_on for variant in all_variants)

        return OptimizationResult(
            best_variant=best,
            all_variants=all_variants,
            iterations=completed_iterations,
            improvement=best.score - baseline.score,
            evaluations_attempted=evaluations_completed + evaluation_failures,
            evaluations_completed=evaluations_completed,
            evaluation_failures=evaluation_failures,
        )

    async def _generate_variants(
        self,
        parent: PromptVariant,
        count: int,
    ) -> list[PromptVariant]:
        """Generate mutated variants of a prompt."""
        variants: list[PromptVariant] = []

        # Select mutations (cycle through available)
        for i in range(count):
            mut_name, mut_desc = MUTATIONS[i % len(MUTATIONS)]
            mutated_text = self._apply_mutation(parent.text, mut_name, mut_desc)
            if mutated_text == parent.text:
                continue  # Skip if mutation didn't change anything
            if len(mutated_text) > _MAX_BASE_PROMPT_CHARS:
                continue
            variant = PromptVariant(
                id=self._hash(mutated_text),
                text=mutated_text,
                parent_id=parent.id,
                mutation=mut_name,
            )
            variants.append(variant)

        return variants

    def _apply_mutation(self, text: str, mut_name: str, mut_desc: str) -> str:
        """Apply a mutation to a prompt.

        For now, uses simple text transformations.
        Future: use the model to generate mutations.
        """
        if mut_name == "clarity":
            # Remove filler words, make more direct
            fillers = ["please", "kindly", "you might want to", "consider", "perhaps"]
            result = text
            for f in fillers:
                result = result.replace(f, "")
            return result.strip()

        elif mut_name == "constraints":
            return (
                f"{text}\n\nConstraints:\n"
                "  - Follow the requested output format.\n"
                "  - Do not claim an action or fact was verified without evidence."
            )

        elif mut_name == "structure":
            # Add structure
            lines = text.split(". ")
            if len(lines) > 1:
                return "\n".join(f"  {i + 1}. {line.strip()}" for i, line in enumerate(lines))
            return text

        elif mut_name == "context":
            # Add context
            return (
                f"{text}\n\nContext: This task is part of a larger workflow. Accuracy is critical."
            )

        elif mut_name == "simplify":
            # Normalize needless whitespace without deleting instructions.
            return "\n".join(line.strip() for line in text.splitlines() if line.strip())

        elif mut_name == "precision":
            return (
                f"{text}\n\nState exact inputs, outputs, units, and acceptance criteria "
                "that are relevant to this task."
            )

        elif mut_name == "edge_cases":
            return (
                f"{text}\n\nHandle missing or ambiguous inputs explicitly; do not invent "
                "values, successful actions, or unavailable evidence."
            )

        return text

    async def _evaluate(
        self,
        variant: PromptVariant,
        tasks: list[BenchmarkTask],
        model: str,
    ) -> float:
        """Evaluate a prompt variant against benchmark tasks.

        Returns a score in [0.0, 1.0].
        """
        if not tasks or self.router is None:
            raise ValueError("objective tasks and a live router are required")

        scores: list[float] = []
        failures = 0
        for task in tasks:
            score = await self._evaluate_one(variant, task, model)
            if score is None:
                failures += 1
            else:
                scores.append(score)

        variant.tested_on = len(scores)
        variant.failed_on = failures
        if not scores:
            raise RuntimeError("all prompt benchmark evaluations failed")
        return sum(scores) / len(tasks)

    async def _evaluate_one(
        self,
        variant: PromptVariant,
        task: BenchmarkTask,
        model: str,
    ) -> float | None:
        """Evaluate a variant on a single task."""
        if self.router is None:
            raise ValueError("a live gateway/router is required")
        try:
            from ..gateway_client import GatewayRequest

            req = GatewayRequest(
                model=model or "",
                messages=[
                    {"role": "system", "content": variant.text},
                    {"role": "user", "content": task.input},
                ],
                tools=task.tools,
                max_tokens=_EVALUATION_MAX_TOKENS,
                temperature=0.0,
                metadata={
                    "prompt_opt": True,
                    "benchmark_mode": True,
                    "allow_text_tool_calls": bool(task.tools),
                },
            )

            # Always use the public gateway surface.  Reaching into a router's
            # private client bypasses dynamic-provider lifetime accounting.
            resp = await asyncio.wait_for(
                self.router.chat(req),
                timeout=_EVALUATION_TIMEOUT_SECONDS,
            )

            output = (resp.content or "").lower()
            passed = 0
            assertions = 0

            # Check expected contains
            if task.expected_contains:
                expected = task.expected_contains
                if isinstance(expected, str):
                    expected = [expected]
                expected = [item for item in expected if item]
                assertions += len(expected)
                passed += sum(1 for item in expected if item.lower() in output)

            # Check expected not contains
            if task.expected_not_contains:
                not_expected = task.expected_not_contains
                if isinstance(not_expected, str):
                    not_expected = [not_expected]
                not_expected = [item for item in not_expected if item]
                assertions += len(not_expected)
                passed += sum(1 for item in not_expected if item.lower() not in output)

            # Check tool call
            if task.tool_should_be_called:
                assertions += 1
                if resp.tool_calls:
                    for tc in resp.tool_calls:
                        fn = tc.get("function", {})
                        if fn.get("name", "") == task.tool_should_be_called:
                            passed += 1
                            break

            return passed / assertions if assertions else 0.0
        except Exception as exc:
            log.debug("prompt_opt._evaluate_one failed: %s", exc)
            return None

    @staticmethod
    def _task_has_assertions(task: BenchmarkTask) -> bool:
        def populated(value: str | list[str]) -> bool:
            values = [value] if isinstance(value, str) else value
            return any(str(item).strip() for item in values)

        return bool(
            populated(task.expected_contains)
            or populated(task.expected_not_contains)
            or task.tool_should_be_called.strip()
        )

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    def _remember(self, variant: PromptVariant) -> None:
        self._history.append(variant)
        if len(self._history) > _MAX_RETAINED_VARIANTS:
            del self._history[: len(self._history) - _MAX_RETAINED_VARIANTS]

    def _save_history(self, variants: list[PromptVariant]) -> None:
        """Append only this run's bounded evaluation receipts."""
        if not variants:
            return
        path = self.storage_dir / "prompt_history.jsonl"
        payload = "".join(
            json.dumps(
                {
                    "id": variant.id,
                    "text": variant.text[:500],
                    "score": variant.score,
                    "tested_on": variant.tested_on,
                    "failed_on": variant.failed_on,
                    "parent_id": variant.parent_id,
                    "mutation": variant.mutation,
                    "created_at": variant.created_at,
                },
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
            for variant in variants
        )
        append_bounded_text(
            path,
            payload,
            max_bytes=_MAX_HISTORY_BYTES,
            mode=0o600,
        )

    def load_history(self) -> list[dict[str, Any]]:
        """Load bounded optimization receipts, skipping malformed rows."""
        path = self.storage_dir / "prompt_history.jsonl"
        try:
            payload = read_bounded_text(path, max_bytes=_MAX_HISTORY_BYTES)
        except FileNotFoundError:
            return []
        results: list[dict[str, Any]] = []
        for line_no, line in enumerate(payload.splitlines(), start=1):
            if line_no > _MAX_HISTORY_LINES:
                raise ValueError("prompt optimization history exceeds its line limit")
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                results.append(row)
        return results
