"""Background maintenance and explicitly enabled autonomous experiments.

This module closes the gap between "agent that responds to messages" and
"agent that operates autonomously without human intervention."

Five responsibilities:
  1. Checkpoint discovery — surface interrupted tasks after restart
  2. Scratchpad sync — keep hot state current with canonical memory
  3. Self-healing — detect and repair concrete subsystem failures
  4. Optional multi-agent trigger — decompose complex tasks when enabled
  5. Optional prompt optimization and learning experiments

Runs as a background asyncio task alongside the idle sleep loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..atomic import atomic_write_text, path_lock, read_bounded_bytes, read_bounded_text

log = logging.getLogger("norax.autonomy")

_SCRATCHPAD_MAX_BYTES = 16 * 1024 * 1024
_PROMPT_BENCHMARK_MAX_BYTES = 4 * 1024 * 1024
_PROMPT_MARKER_MAX_BYTES = 1 * 1024 * 1024


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean feature flag from the environment.

    Accepts 1/0, true/false, yes/no, on/off (case-insensitive).
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Lazy import for learning loop (avoids circular import at module load)
_learning_loop: tuple[type[Any], type[Any]] | Literal[False] | None = None


def _get_learning_loop():
    global _learning_loop
    if _learning_loop is None:
        try:
            from ..brain.learning_loop import LearningConfig, LearningLoop

            _learning_loop = (LearningLoop, LearningConfig)
        except Exception:
            _learning_loop = False
    return _learning_loop


# ── Autonomy config ──────────────────────────────────────────────────────────


@dataclass
class AutonomyConfig:
    """Configuration for the autonomy engine.

    The engine is opt-in.  Core request handling, memory consolidation, and
    readiness probes do not depend on it.  Enabling a specific experiment also
    requires ``NORAX_AUTONOMY_ENABLED=1`` so no background mutation or model
    traffic starts from a single accidental flag.
    """

    # Master switch for the entire background autonomy engine.
    enabled: bool = field(default_factory=lambda: _env_bool("NORAX_AUTONOMY_ENABLED", False))
    # Checkpoint discovery: check on startup and report pending state. The
    # foreground continuation logic remains responsible for actual resumption.
    checkpoint_resume: bool = True
    checkpoint_max_age_sec: float = 3600.0  # 1 hour
    # Scratchpad sync: keep hot state current
    scratchpad_sync_interval_sec: float = 600.0  # 10 min
    # Self-healing: detect and fix subsystem failures
    self_heal_interval_sec: float = 1800.0  # 30 min
    # Prompt optimization: evolve prompts during idle (makes gateway/LLM calls)
    prompt_opt_enabled: bool = field(
        default_factory=lambda: _env_bool("NORAX_AUTONOMY_PROMPT_OPT", False)
    )
    prompt_opt_interval_sec: float = 3600.0  # 1 hour
    # Historical name retained for config compatibility; this now means the
    # minimum number of explicit objective benchmark cases.
    prompt_opt_min_episodes: int = 10
    # Multi-agent: auto-decompose threshold
    multi_agent_auto: bool = field(
        default_factory=lambda: _env_bool("NORAX_AUTONOMY_MULTI_AGENT", False)
    )
    # Self-directed learning: research + test on staging (LLM + web — expensive)
    learning_enabled: bool = field(
        default_factory=lambda: _env_bool("NORAX_AUTONOMY_LEARNING", False)
    )
    learning_interval_sec: float = 1800.0  # 30 min


# ── Scratchpad sync ──────────────────────────────────────────────────────────


def sync_scratchpad(memory_root: Path) -> dict[str, Any]:
    """Synchronize scratchpad with current system state.

    Updates the measured memory statistics and, only when those statistics
    changed, the timestamp.  It deliberately does not rewrite or deduplicate
    owner content: repeated lines may be intentional and this helper has no
    semantic evidence with which to remove them.

    Returns a dict of changes made.
    """
    scratchpad = memory_root / "scratchpad.md"
    import re

    # Gather measured stats outside the scratchpad file lock so a large memory
    # tree cannot delay a foreground turn that needs to append its outcome.
    sync_line: str | None = None
    sync_change = ""
    try:
        from ..memory.store import MemoryStore

        store = MemoryStore(root=memory_root)
        store.refresh()
        # The generated SYNC line must not change the statistic it reports;
        # excluding it makes repeated synchronization idempotent.
        hot_count = sum(
            1 for neuron in store.hot if not str(getattr(neuron, "text", "")).startswith("SYNC:")
        )
        canonical_count = store.canonical_count()
        sleep_count = len(store.sleep)

        # Try to get entity graph stats
        entity_count = 0
        link_count = 0
        try:
            from ..memory.entity_graph import EntityGraph

            eg = EntityGraph(root=memory_root)
            eg.load()
            entity_count = eg._entity_count
            link_count = eg._link_count
        except Exception as e:
            log.debug("autonomy.entity_graph_stats_failed error=%r", e)

        sync_line = (
            f"SYNC:memory hot={hot_count} canonical={canonical_count} "
            f"sleep={sleep_count} entities={entity_count} links={link_count}"
        )
        sync_change = f"sync_stats: hot={hot_count} canon={canonical_count} ent={entity_count}"
    except Exception as e:
        log.debug("scratchpad_sync.stats_failed: %r", e)

    with path_lock(scratchpad):
        try:
            content = read_bounded_text(
                scratchpad,
                max_bytes=_SCRATCHPAD_MAX_BYTES,
                errors="replace",
            )
        except FileNotFoundError:
            return {"changed": False, "reason": "no scratchpad"}
        original = content
        changes: list[str] = []

        # Replace or insert one measured SYNC line.
        if sync_line is not None:
            if re.search(r"(?m)^SYNC:", content):
                updated = re.sub(r"(?m)^SYNC:.*$", sync_line, content, count=1)
            else:
                lines = content.splitlines()
                if lines:
                    lines.insert(1, sync_line)
                    updated = "\n".join(lines) + "\n"
                else:
                    updated = sync_line + "\n"
            if updated != content:
                content = updated
                changes.append(sync_change)

        if content != original:
            ts = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
            timestamped = re.sub(
                r"(?m)^(SCRATCHPAD;updated=)[^;\n]+",
                f"\\g<1>{ts}",
                content,
                count=1,
            )
            if timestamped != content:
                content = timestamped
                changes.insert(0, "timestamp")
            atomic_write_text(scratchpad, content, mode=0o600)
            log.info("scratchpad_sync: %d changes: %s", len(changes), ", ".join(changes))
            return {"changed": True, "changes": changes}
    return {"changed": False, "changes": []}


# ── RHO self-healing ─────────────────────────────────────────────────────────


def check_subsystem_health(memory_root: Path) -> dict[str, Any]:
    """Check all subsystems for health issues.

    Returns a dict of subsystem → status. Unhealthy subsystems
    get flagged for self-healing.
    """
    issues: list[dict] = []
    healthy: list[str] = []
    observations: list[dict] = []

    # 1. Memory store
    try:
        from ..memory.store import MemoryStore

        store = MemoryStore(root=memory_root)
        store.refresh()
        healthy.append("memory_store")
        if store.canonical_count() == 0:
            observations.append({"subsystem": "memory_store", "state": "ready_empty"})
    except Exception as e:
        issues.append({"subsystem": "memory_store", "issue": str(e), "severity": "critical"})

    # 2. Entity graph
    try:
        from ..memory.entity_graph import EntityGraph

        eg = EntityGraph(root=memory_root)
        eg.load()
        healthy.append("entity_graph")
        if eg._entity_count == 0:
            observations.append({"subsystem": "entity_graph", "state": "ready_empty"})
    except Exception as e:
        issues.append({"subsystem": "entity_graph", "issue": str(e), "severity": "high"})

    # 3. Causal graph
    try:
        from ..memory.causal_graph import CausalGraph

        cg = CausalGraph(root=memory_root)
        cg.load()
        healthy.append("causal_graph")
        if cg._node_count == 0:
            observations.append({"subsystem": "causal_graph", "state": "ready_empty"})
    except Exception as e:
        issues.append({"subsystem": "causal_graph", "issue": str(e), "severity": "medium"})

    # 4. SQLite FTS5 index
    try:
        sqlite_path = memory_root / "index" / "norax_memory.sqlite"
        if not sqlite_path.exists():
            # This is a derived cache and can be initialized lazily. Absence is
            # not evidence that keyword/memory retrieval is broken.
            observations.append({"subsystem": "sqlite_index", "state": "not_initialized"})
        else:
            healthy.append("sqlite_index")
    except Exception as e:
        issues.append({"subsystem": "sqlite_index", "issue": str(e), "severity": "medium"})

    # 5. Episodic buffer. Health checks must not initialize optional state.
    try:
        episodic_dir = memory_root / "episodic"
        if episodic_dir.exists():
            files = list(episodic_dir.glob("episodes-*.jsonl"))
            healthy.append("episodic")
            if not files:
                observations.append({"subsystem": "episodic", "state": "ready_empty"})
        else:
            observations.append({"subsystem": "episodic", "state": "not_initialized"})
    except Exception as e:
        issues.append({"subsystem": "episodic", "issue": str(e), "severity": "medium"})

    # 6. Scratchpad
    try:
        scratchpad = memory_root / "scratchpad.md"
        if not scratchpad.exists():
            issues.append({"subsystem": "scratchpad", "issue": "file missing", "severity": "high"})
        else:
            scratchpad_stat = scratchpad.lstat()
            if not stat.S_ISREG(scratchpad_stat.st_mode):
                issues.append(
                    {
                        "subsystem": "scratchpad",
                        "issue": "not a regular file",
                        "severity": "high",
                    }
                )
            elif scratchpad_stat.st_size == 0:
                issues.append({"subsystem": "scratchpad", "issue": "empty", "severity": "high"})
            elif scratchpad_stat.st_size > _SCRATCHPAD_MAX_BYTES:
                issues.append(
                    {
                        "subsystem": "scratchpad",
                        "issue": "exceeds size limit",
                        "severity": "high",
                    }
                )
            else:
                healthy.append("scratchpad")
    except Exception as e:
        issues.append({"subsystem": "scratchpad", "issue": str(e), "severity": "high"})

    return {
        "healthy": healthy,
        "issues": issues,
        "observations": observations,
        "total_healthy": len(healthy),
        "total_issues": len(issues),
    }


def self_heal(memory_root: Path, issues: list[dict]) -> dict[str, Any]:
    """Attempt to fix subsystem issues autonomously.

    Returns a dict of fixes applied.
    """
    fixes: list[dict] = []
    skipped: list[dict] = []
    failures: list[dict] = []

    for issue in issues:
        subsystem = issue["subsystem"]
        problem = issue["issue"]

        try:
            if subsystem == "scratchpad" and ("missing" in problem or "empty" in problem):
                # Recreate scratchpad from memory store
                from ..memory.store import MemoryStore

                store = MemoryStore(root=memory_root)
                store.refresh()
                ts = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
                lines = [
                    f"SCRATCHPAD;updated={ts};type=hot_memory",
                    "IDENTITY:Norax agent runtime",
                    f"SYNC:memory hot={len(store.hot)} canonical={store.canonical_count()} sleep={len(store.sleep)}",
                    "",
                ]
                atomic_write_text(
                    memory_root / "scratchpad.md",
                    "\n".join(lines),
                    mode=0o600,
                )
                fixes.append({"subsystem": "scratchpad", "fix": "recreated from memory store"})
                log.info("self_heal: recreated scratchpad")

            else:
                skipped.append({"subsystem": subsystem, "reason": "no safe automatic repair"})

        except Exception as e:
            log.warning("self_heal.failed subsystem=%s err=%r", subsystem, e)
            failures.append({"subsystem": subsystem, "error": str(e)})

    return {
        "fixes": fixes,
        "skipped": skipped,
        "failures": failures,
        "total_fixes": len(fixes),
        "total_failures": len(failures),
    }


# ── Checkpoint resume ────────────────────────────────────────────────────────


def resume_pending_checkpoints(memory_root: Path, max_age_sec: float = 3600.0) -> dict[str, Any]:
    """Discover recent pending checkpoints without executing them.

    This doesn't auto-resume (that requires the runtime to inject messages
    into the agent loop). It returns the pending checkpoints so the runtime
    can surface them.
    """
    from ..runtime.checkpoint import list_pending_checkpoints

    pending = [
        {
            "channel": checkpoint.get("channel", ""),
            "turn_id": checkpoint.get("turn_id", ""),
            "timestamp": checkpoint.get("timestamp", 0),
            "rounds": checkpoint.get("rounds", 0),
            "model": checkpoint.get("model", ""),
            "trace_count": len(checkpoint.get("trace", [])),
            "age_sec": time.time() - float(checkpoint.get("timestamp", 0)),
        }
        for checkpoint in list_pending_checkpoints(memory_root, max_age_sec)
    ]
    return {"pending": pending, "count": len(pending)}


# ── Prompt optimization runner ───────────────────────────────────────────────


async def run_idle_prompt_optimization(
    memory_root: Path,
    episodic: Any | None = None,
    gateway: Any | None = None,
    min_episodes: int = 10,
    default_model: str = "",
) -> dict[str, Any]:
    """Evaluate staged prompt candidates against an explicit benchmark file.

    Recent agent behavior is not ground truth, so this path never constructs
    benchmarks from its own tool choices. It also never deploys the winner.
    Operators stage ``state/prompt_opt/benchmark.json`` with ``base_prompt``
    and a ``tasks`` list containing explicit expected/forbidden text or an
    expected tool name.
    """
    del episodic  # Explicit benchmark cases, not self-labelled episodes, are ground truth.
    if gateway is None:
        return {"status": "no_gateway", "optimizations": 0}
    if (
        isinstance(min_episodes, bool)
        or not isinstance(min_episodes, int)
        or not 1 <= min_episodes <= 100
    ):
        return {
            "status": "invalid_config",
            "error": "min_episodes must be an integer between 1 and 100",
            "optimizations": 0,
        }

    storage_dir = memory_root / "state" / "prompt_opt"
    benchmark_path = storage_dir / "benchmark.json"
    if not benchmark_path.is_file():
        return {"status": "no_objective_benchmark", "optimizations": 0}

    try:
        raw_spec = await asyncio.to_thread(
            read_bounded_bytes,
            benchmark_path,
            max_bytes=_PROMPT_BENCHMARK_MAX_BYTES,
        )
        spec = json.loads(raw_spec)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid_benchmark",
            "error": str(exc),
            "optimizations": 0,
        }
    if not isinstance(spec, dict) or not isinstance(spec.get("tasks"), list):
        return {
            "status": "invalid_benchmark",
            "error": "benchmark must be an object with a tasks list",
            "optimizations": 0,
        }
    unknown_spec_fields = set(spec) - {"base_prompt", "tasks", "model"}
    if unknown_spec_fields:
        return {
            "status": "invalid_benchmark",
            "error": f"unsupported benchmark fields: {sorted(unknown_spec_fields)}",
            "optimizations": 0,
        }
    if not min_episodes <= len(spec["tasks"]) <= 100:
        return {
            "status": "insufficient_benchmark",
            "required_tasks": min_episodes,
            "available_tasks": len(spec["tasks"]),
            "optimizations": 0,
        }

    spec_hash = hashlib.sha256(raw_spec).hexdigest()
    if not isinstance(default_model, str) or len(default_model) > 256:
        return {
            "status": "invalid_config",
            "error": "default_model must be a string of at most 256 characters",
            "optimizations": 0,
        }
    if not default_model:
        benchmark_model = spec.get("model", "")
        if not isinstance(benchmark_model, str) or len(benchmark_model) > 256:
            return {
                "status": "invalid_benchmark",
                "error": "model must be a string of at most 256 characters",
                "optimizations": 0,
            }
        default_model = benchmark_model
    if not default_model:
        gateway_model = getattr(gateway, "default_model", "") or ""
        default_model = gateway_model if isinstance(gateway_model, str) else ""
        if not default_model:
            effective_model = getattr(gateway, "_effective_model", "") or ""
            default_model = effective_model if isinstance(effective_model, str) else ""
    if len(default_model) > 256:
        return {
            "status": "invalid_config",
            "error": "resolved model exceeds 256 characters",
            "optimizations": 0,
        }
    marker_path = storage_dir / "last_evaluation.json"
    try:
        marker_text = await asyncio.to_thread(
            read_bounded_text,
            marker_path,
            max_bytes=_PROMPT_MARKER_MAX_BYTES,
        )
        marker = json.loads(marker_text)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        marker = {}
    if (
        isinstance(marker, dict)
        and marker.get("benchmark_sha256") == spec_hash
        and marker.get("model") == default_model
    ):
        return {
            "status": "benchmark_unchanged",
            "benchmark_sha256": spec_hash,
            "optimizations": 0,
        }

    from ..brain.prompt_optimizer import BenchmarkTask, PromptOptimizer

    try:
        benchmark_tasks: list[BenchmarkTask] = []
        for item in spec["tasks"]:
            if not isinstance(item, dict):
                raise TypeError("benchmark tasks must be objects")
            unknown_task_fields = set(item) - {
                "input",
                "expected_contains",
                "expected_not_contains",
                "tool_should_be_called",
                "tools",
            }
            if unknown_task_fields:
                raise ValueError(
                    f"unsupported benchmark task fields: {sorted(unknown_task_fields)}"
                )
            task_input = item.get("input", "")
            expected_contains = item.get("expected_contains", "")
            expected_not_contains = item.get("expected_not_contains", "")
            expected_tool = item.get("tool_should_be_called", "")
            benchmark_tools = item.get("tools", [])
            if not isinstance(task_input, str) or len(task_input) > 16_000:
                raise ValueError("benchmark task input must be at most 16000 characters")
            for field_name, field_value in (
                ("expected_contains", expected_contains),
                ("expected_not_contains", expected_not_contains),
            ):
                values = field_value if isinstance(field_value, list) else [field_value]
                if (
                    not isinstance(field_value, str | list)
                    or len(values) > 64
                    or not all(isinstance(value, str) and len(value) <= 4_000 for value in values)
                ):
                    raise ValueError(f"invalid benchmark {field_name}")
            if not isinstance(expected_tool, str) or len(expected_tool) > 128:
                raise ValueError("invalid benchmark tool_should_be_called")
            if not isinstance(benchmark_tools, list):
                raise ValueError("invalid benchmark tools")
            benchmark_tasks.append(
                BenchmarkTask(
                    input=task_input,
                    expected_contains=expected_contains,
                    expected_not_contains=expected_not_contains,
                    tool_should_be_called=expected_tool,
                    tools=benchmark_tools,
                )
            )
    except (TypeError, ValueError) as exc:
        return {"status": "invalid_benchmark", "error": str(exc), "optimizations": 0}

    raw_base_prompt = spec.get("base_prompt", "")
    if not isinstance(raw_base_prompt, str) or len(raw_base_prompt) > 64_000:
        return {
            "status": "invalid_benchmark",
            "error": "base_prompt must be at most 64000 characters",
            "optimizations": 0,
        }
    base_prompt = raw_base_prompt.strip()
    try:
        opt = PromptOptimizer(router=gateway, storage_dir=storage_dir)
        result = await opt.optimize(
            base_prompt=base_prompt,
            benchmark_tasks=benchmark_tasks,
            iterations=1,
            variants_per_iter=2,
            model=default_model,
        )
        marker_payload = {
            "benchmark_sha256": spec_hash,
            "evaluated_at": time.time(),
            "best_variant_id": result.best_variant.id,
            "best_score": result.best_variant.score,
            "model": default_model,
            "deployed": False,
        }
        await asyncio.to_thread(
            atomic_write_text,
            marker_path,
            json.dumps(marker_payload, allow_nan=False, separators=(",", ":")) + "\n",
            mode=0o600,
        )
        return {
            "status": "evaluated_not_deployed",
            "optimizations": 0,
            "candidates_evaluated": len(result.all_variants),
            "benchmark_sha256": spec_hash,
            "best_variant_id": result.best_variant.id,
            "best_score": result.best_variant.score,
            "improvement": result.improvement,
            "evaluations_attempted": result.evaluations_attempted,
            "evaluations_completed": result.evaluations_completed,
            "evaluation_failures": result.evaluation_failures,
            "deployed": False,
        }
    except Exception as e:
        log.debug("prompt_opt.idle_failed: %r", e)
        return {"status": "error", "error": str(e), "optimizations": 0}


# ── Multi-agent trigger ──────────────────────────────────────────────────────


def should_trigger_multi_agent(task: str, allowed_tools: list[str]) -> bool:
    """Check if a task should be decomposed into sub-agents.

    This is a more sophisticated version of should_decompose that also
    considers tool availability and task complexity.
    """
    from ..brain.multi_agent import should_decompose

    if not should_decompose(task):
        return False

    # Check that we have the tools needed for multi-agent
    research_tools = {"read", "list_dir", "web_search", "web_fetch", "search_memory"}
    code_tools = {"read", "write", "edit", "exec", "shell"}
    verify_tools = {"read", "exec", "shell", "list_dir"}

    available = set(allowed_tools)
    has_research = bool(research_tools & available)
    has_code = bool(code_tools & available)
    has_verify = bool(verify_tools & available)

    # Need at least 2 of 3 capabilities for multi-agent to be useful
    capabilities = sum([has_research, has_code, has_verify])
    return capabilities >= 2


# ── Autonomy engine ──────────────────────────────────────────────────────────


class AutonomyEngine:
    """Background autonomy engine. Runs alongside the idle sleep loop.

    Handles:
    - Periodic scratchpad sync
    - Periodic self-healing
    - Periodic prompt optimization
    - Checkpoint resume on startup
    """

    def __init__(
        self,
        memory_root: Path,
        *,
        config: AutonomyConfig | None = None,
        episodic: Any | None = None,
        gateway: Any | None = None,
        event_log: Any | None = None,
        default_model: str = "",
    ) -> None:
        self.memory_root = memory_root
        self.config = config or AutonomyConfig()
        self.episodic = episodic
        self.gateway = gateway
        self.event_log = event_log
        self.default_model = default_model
        self._last_scratchpad_sync: float = 0.0
        self._last_self_heal: float = 0.0
        # Experimental network/model loops wait a full interval after startup;
        # a zero timestamp would make both fire at the first five-minute tick.
        self._last_prompt_opt: float = time.time()
        self._startup_check_done: bool = False
        self._task: asyncio.Task | None = None
        self._learning_loop: Any | None = None
        self._learning_task: asyncio.Task | None = None
        self._last_learning: float = time.time()

    async def run_startup_check(self) -> dict[str, Any]:
        """Run on startup: check for pending checkpoints, sync scratchpad, self-heal."""
        results: dict[str, Any] = {}

        # 1. Discover pending checkpoints. Foreground continuation logic owns
        # any actual resume so startup never launches an unsolicited task.
        if self.config.checkpoint_resume:
            try:
                cp_result = await asyncio.to_thread(
                    resume_pending_checkpoints,
                    self.memory_root,
                    self.config.checkpoint_max_age_sec,
                )
                results["checkpoints"] = cp_result
                if cp_result["count"] > 0:
                    log.info("autonomy.startup: %d pending checkpoints", cp_result["count"])
                    if self.event_log:
                        await self.event_log.append(
                            "autonomy.checkpoint_pending",
                            {"pending": cp_result["pending"]},
                        )
            except Exception as e:
                log.warning("autonomy.checkpoint_resume.failed: %r", e)

        # 2. Sync scratchpad
        try:
            sync_result = await asyncio.to_thread(sync_scratchpad, self.memory_root)
            results["scratchpad_sync"] = sync_result
            if sync_result["changed"]:
                log.info(
                    "autonomy.startup: scratchpad synced (%d changes)", len(sync_result["changes"])
                )
        except Exception as e:
            log.warning("autonomy.scratchpad_sync.failed: %r", e)

        # 3. Self-heal
        try:
            health = await asyncio.to_thread(check_subsystem_health, self.memory_root)
            results["health"] = health
            if health["total_issues"] > 0:
                log.info("autonomy.startup: %d health issues, self-healing", health["total_issues"])
                heal_result = await asyncio.to_thread(self_heal, self.memory_root, health["issues"])
                results["self_heal"] = heal_result
                if self.event_log:
                    await self.event_log.append(
                        "autonomy.self_heal",
                        {"issues": health["issues"], "fixes": heal_result["fixes"]},
                    )
        except Exception as e:
            log.warning("autonomy.self_heal.failed: %r", e)

        self._startup_check_done = True
        now = time.time()
        self._last_scratchpad_sync = now
        self._last_self_heal = now
        return results

    async def run_forever(self) -> None:
        """Background loop: periodic self-maintenance."""
        if not getattr(self.config, "enabled", True):
            log.info("AutonomyEngine disabled by config; not starting background cycles")
            return

        # Run startup check first
        await self.run_startup_check()

        CHECK_INTERVAL = 300  # 5 min base check
        while True:
            try:
                await asyncio.sleep(CHECK_INTERVAL)
                now = time.time()

                # Scratchpad sync
                if now - self._last_scratchpad_sync > self.config.scratchpad_sync_interval_sec:
                    try:
                        result = await asyncio.to_thread(sync_scratchpad, self.memory_root)
                        if result["changed"]:
                            log.info(
                                "autonomy: scratchpad synced (%d changes)", len(result["changes"])
                            )
                            if self.event_log:
                                await self.event_log.append("autonomy.scratchpad_sync", result)
                    except Exception as e:
                        log.debug("autonomy.scratchpad_sync.error: %r", e)
                    self._last_scratchpad_sync = now

                # Self-healing
                if now - self._last_self_heal > self.config.self_heal_interval_sec:
                    try:
                        health = await asyncio.to_thread(check_subsystem_health, self.memory_root)
                        if health["total_issues"] > 0:
                            heal_result = await asyncio.to_thread(
                                self_heal, self.memory_root, health["issues"]
                            )
                            if heal_result["total_fixes"] > 0:
                                log.info(
                                    "autonomy: self-healed %d issues", heal_result["total_fixes"]
                                )
                                if self.event_log:
                                    await self.event_log.append(
                                        "autonomy.self_heal",
                                        {
                                            "issues": health["issues"],
                                            "fixes": heal_result["fixes"],
                                        },
                                    )
                    except Exception as e:
                        log.debug("autonomy.self_heal.error: %r", e)
                    self._last_self_heal = now

                # Prompt optimization
                if (
                    getattr(self.config, "prompt_opt_enabled", True)
                    and now - self._last_prompt_opt > self.config.prompt_opt_interval_sec
                ):
                    try:
                        opt_result = await run_idle_prompt_optimization(
                            self.memory_root,
                            episodic=self.episodic,
                            gateway=self.gateway,
                            min_episodes=self.config.prompt_opt_min_episodes,
                            default_model=getattr(self, "default_model", "") or "",
                        )
                        if opt_result.get("status") == "evaluated_not_deployed":
                            log.info(
                                "autonomy: prompt candidates evaluated, not deployed (score=%.2f)",
                                opt_result.get("best_score", 0),
                            )
                            if self.event_log:
                                await self.event_log.append("autonomy.prompt_opt", opt_result)
                    except Exception as e:
                        log.debug("autonomy.prompt_opt.error: %r", e)
                    self._last_prompt_opt = now

                # Self-directed learning
                if (
                    self.config.learning_enabled
                    and now - self._last_learning > self.config.learning_interval_sec
                ):
                    try:
                        ll_cls = _get_learning_loop()
                        if ll_cls:
                            LearningLoop, LearningConfig = ll_cls
                            if self._learning_loop is None:
                                self._learning_loop = LearningLoop(
                                    LearningConfig(
                                        enabled=True,
                                        interval_sec=self.config.learning_interval_sec,
                                    ),
                                    gateway=self.gateway,
                                    event_log=self.event_log,
                                )
                            result = await self._learning_loop.run_one_cycle()
                            log.info(
                                "autonomy: learning cycle topic=%s passed=%s",
                                result.topic,
                                result.test_passed,
                            )
                            if self.event_log:
                                await self.event_log.append(
                                    "autonomy.learning",
                                    {
                                        "topic": result.topic,
                                        "test_executed": result.test_executed,
                                        "test_passed": result.test_passed,
                                        "test_verdict": result.test_verdict,
                                        "test_location": result.test_location,
                                        "duration_sec": round(result.duration_sec, 2),
                                        "n_search_results": len(result.search_results),
                                    },
                                )
                    except Exception as e:
                        log.debug("autonomy.learning.error: %r", e)
                    self._last_learning = now

            except asyncio.CancelledError:
                break
            except Exception:
                log.debug("autonomy.loop.error", exc_info=True)
                await asyncio.sleep(60)

    def start(self) -> asyncio.Task:
        """Start the autonomy engine as a background task."""
        self._task = asyncio.create_task(self.run_forever())
        return self._task

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        if self._learning_task is not None:
            self._learning_task.cancel()
            self._learning_task = None
