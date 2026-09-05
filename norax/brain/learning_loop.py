"""Self-directed learning loop — practical stack improvement edition.

Runs as a background task inside AutonomyEngine. Picks a topic from
a rotating curriculum of REAL stack improvements, researches best practices,
tests implementations on the staging clone via SSH, and records
actionable findings to memory.

Execution policy:
  - Test scripts run on a configured staging host by default.
  - Local test execution requires a separate explicit opt-in.
  - Subprocess time and captured output are bounded.
  - One static curriculum experiment runs per cycle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_text, path_lock

log = logging.getLogger("norax.learning")

# Resolve root from environment or fall back to auto-detection
_NORAX_ROOT = Path(os.environ.get("NORAX_ROOT", Path(__file__).resolve().parent.parent.parent))
_MEMORY_ROOT = Path(os.environ.get("NORAX_MEMORY_ROOT", _NORAX_ROOT / "memory")).expanduser()
LEARNING_DIR = _MEMORY_ROOT / "intel"
LEARNING_LOG = LEARNING_DIR / "learning_log.jsonl"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ── Practical curriculum — each topic improves the actual stack ──────────

CURRICULUM: list[dict[str, Any]] = [
    {
        "topic": "memory_retrieval_precision",
        "description": "Improve multi-signal retrieval scoring weights for better recall",
        "search_queries": [
            "hybrid retrieval BM25 dense embedding score weighting 2025 2026",
            "multi-signal memory retrieval fusion RRF reciprocal rank",
            "vector search relevance tuning embedding cosine vs dot product",
        ],
        "test_concept": "Benchmark current retrieval vs RRF-fused retrieval on staging",
        "test_script": "retrieval_benchmark.py",
        "test_code": r"""
import sys, json, time, os, asyncio
from pathlib import Path
NORAX_ROOT = Path(os.environ.get("NORAX_ROOT", "~/norax")).expanduser()
MEMORY_ROOT = Path(os.environ.get("NORAX_MEMORY_ROOT", NORAX_ROOT / "memory")).expanduser()
sys.path.insert(0, str(NORAX_ROOT))
from norax.memory.retrievers.multi_signal import MultiSignalRetriever
from norax.memory.retrievers.fast import FastContext
from norax.memory.store import MemoryStore

store = MemoryStore(root=MEMORY_ROOT)
store.refresh()
fc = FastContext(store)
retriever = MultiSignalRetriever(keyword=fc)

queries = [
    "discord token staging setup",
    "zero division error goal_drift",
    "wallet EVM Base USDC payment",
    "autonomy engine cron jobs",
    "entity graph memory linking",
    "empty response pipeline revision",
    "ollama pipeline routing",
    "revenue engine commerce API",
]

scores = []
times = []
for q in queries:
    start = time.time()
    hits = asyncio.run(retriever.search(q, k=5))
    elapsed = (time.time() - start) * 1000
    top_score = hits[0][1] if hits else 0.0
    top_text = hits[0][0].text[:80] if hits else "NONE"
    scores.append(top_score)
    times.append(elapsed)
    print(f"Q: {q[:40]:40s} | {elapsed:.0f}ms | score={top_score:.3f} | {top_text}")

avg_ms = sum(times) / len(times)
avg_score = sum(scores) / len(scores)
print(f"\nAVG: {avg_ms:.1f}ms, score={avg_score:.3f}")
print(f"VERDICT: {'GOOD' if avg_score > 0.3 else 'NEEDS_TUNING'} (avg_score={avg_score:.3f})")
""",
    },
    {
        "topic": "agent_loop_efficiency",
        "description": "Reduce token waste and round count in agent loops",
        "search_queries": [
            "LLM agent loop token optimization reduce rounds 2025 2026",
            "agent tool call batching parallel execution efficiency",
            "prompt compression context window optimization agent",
        ],
        "test_concept": "Analyze recent episode traces for token waste patterns",
        "test_script": "agent_efficiency.py",
        "test_code": r"""
import sys, json, os, statistics
from pathlib import Path
NORAX_ROOT = Path(os.environ.get("NORAX_ROOT", "~/norax")).expanduser()
MEMORY_ROOT = Path(os.environ.get("NORAX_MEMORY_ROOT", NORAX_ROOT / "memory")).expanduser()
sys.path.insert(0, str(NORAX_ROOT))
from norax.memory.episodic import EpisodicBuffer

eb = EpisodicBuffer(root=MEMORY_ROOT / "episodic")
recent = eb.recent_episodes(days=7, limit=30)

if not recent:
    print("NO_EPISODES: need more interaction data")
    sys.exit(0)

round_counts = []
tool_counts = []
token_estimates = []

for ep in recent:
    rounds = ep.rounds or 0
    tools = len(ep.tool_calls or [])
    # Rough token estimate: 4 chars per token
    input_tokens = len(ep.user_input or "") // 4
    output_tokens = len(ep.response_preview or "") // 4
    round_counts.append(rounds)
    tool_counts.append(tools)
    token_estimates.append(input_tokens + output_tokens)

avg_rounds = statistics.mean(round_counts) if round_counts else 0
avg_tools = statistics.mean(tool_counts) if tool_counts else 0
avg_tokens = statistics.mean(token_estimates) if token_estimates else 0
max_rounds = max(round_counts) if round_counts else 0

print(f"Episodes analyzed: {len(recent)}")
print(f"Avg rounds per turn: {avg_rounds:.1f}")
print(f"Avg tools per turn: {avg_tools:.1f}")
print(f"Avg tokens per turn: {avg_tokens:.0f}")
print(f"Max rounds in a turn: {max_rounds}")
print(f"Rounds > 10: {sum(1 for r in round_counts if r > 10)}/{len(round_counts)}")
print(f"VERDICT: {'EFFICIENT' if avg_rounds < 8 else 'WASTEFUL'} (avg_rounds={avg_rounds:.1f})")
""",
    },
    {
        "topic": "memory_consolidation_quality",
        "description": "Evaluate sleep consolidation output quality and coverage",
        "search_queries": [
            "memory consolidation agent sleep cycle quality 2025 2026",
            "episodic to semantic memory conversion evaluation",
            "agent memory deduplication entity extraction quality",
        ],
        "test_concept": "Audit consolidated memories for duplicates and gaps",
        "test_script": "consolidation_audit.py",
        "test_code": r"""
import sys, os, json, time
from pathlib import Path
from collections import Counter
NORAX_ROOT = Path(os.environ.get("NORAX_ROOT", "~/norax")).expanduser()
MEMORY_ROOT = Path(os.environ.get("NORAX_MEMORY_ROOT", NORAX_ROOT / "memory")).expanduser()
sys.path.insert(0, str(NORAX_ROOT))
from norax.memory.store import MemoryStore

store = MemoryStore(root=MEMORY_ROOT)
store.refresh()

canonical = store.all_canonical()
hot = store.hot
sleep = store.sleep

# Check for near-duplicates (same first 50 chars)
seen_prefixes = {}
duplicates = 0
for mem in canonical:
    prefix = mem.text[:50].strip().lower()
    if prefix in seen_prefixes:
        duplicates += 1
    else:
        seen_prefixes[prefix] = True

# Check kind distribution
kinds = Counter()
for mem in canonical:
    kinds[mem.kind] += 1

# Check for stale memories (no access in 7+ days)
now = time.time()
stale = sum(1 for m in canonical if (now - getattr(m, "last_access", now)) > 7 * 86400)

print(f"Canonical memories: {len(canonical)}")
print(f"Hot memories: {len(hot)}")
print(f"Sleep memories: {len(sleep)}")
print(f"Near-duplicates: {duplicates} ({duplicates/max(len(canonical),1)*100:.1f}%)")
print(f"Stale (>7d no access): {stale} ({stale/max(len(canonical),1)*100:.1f}%)")
print(f"Kind distribution: {dict(kinds)}")
print(f"VERDICT: {'CLEAN' if duplicates < 5 else 'NEEDS_DEDUP'} (duplicates={duplicates})")
""",
    },
    {
        "topic": "gateway_routing_accuracy",
        "description": "Evaluate model routing decisions and fallback behavior",
        "search_queries": [
            "LLM gateway model routing accuracy evaluation 2025 2026",
            "fallback model strategy agent production 2026",
            "model selection routing cost latency tradeoff",
        ],
        "test_concept": "Audit gateway routing logs for misroutes and fallbacks",
        "test_script": "gateway_audit.py",
        "test_code": r"""
import sys, os, re
from pathlib import Path
from collections import Counter

NORAX_ROOT = Path(os.environ.get("NORAX_ROOT", "~/norax")).expanduser()
log_path = NORAX_ROOT / "logs" / "norax-ai.log"
if not log_path.exists():
    print("NO_LOG: gateway log not found")
    sys.exit(0)

# Parse recent gateway routing decisions
routes = Counter()
fallbacks = 0
errors = 0
timeouts = 0
total = 0

with open(log_path) as f:
    for line in f:
        if "gateway" in line.lower() and ("route" in line.lower() or "model" in line.lower()):
            total += 1
            # Extract model name
            m = re.search(r'\[([^\]]+)\]', line)
            if m:
                routes[m.group(1)] += 1
        if "fallback" in line.lower():
            fallbacks += 1
        if "timeout" in line.lower() and "gateway" in line.lower():
            timeouts += 1
        if "502" in line or "503" in line or "500" in line:
            if "gateway" in line.lower() or "upstream" in line.lower():
                errors += 1

print(f"Total gateway log lines: {total}")
print(f"Fallbacks: {fallbacks}")
print(f"Timeouts: {timeouts}")
print(f"5xx errors: {errors}")
print(f"Top routes: {routes.most_common(5)}")
error_rate = (errors + timeouts) / max(total, 1) * 100
print(f"Error rate: {error_rate:.1f}%")
print(f"VERDICT: {'STABLE' if error_rate < 5 else 'UNSTABLE'} (error_rate={error_rate:.1f}%)")
""",
    },
    {
        "topic": "entity_graph_coverage",
        "description": "Audit entity graph for missing entities and broken links",
        "search_queries": [
            "entity graph extraction quality evaluation NER 2025 2026",
            "knowledge graph completeness audit entity linking",
            "entity resolution deduplication graph memory agent",
        ],
        "test_concept": "Check entity graph coverage against canonical memories",
        "test_script": "entity_audit.py",
        "test_code": r"""
import sys, os, json
from pathlib import Path
NORAX_ROOT = Path(os.environ.get("NORAX_ROOT", "~/norax")).expanduser()
MEMORY_ROOT = Path(os.environ.get("NORAX_MEMORY_ROOT", NORAX_ROOT / "memory")).expanduser()
sys.path.insert(0, str(NORAX_ROOT))
from norax.memory.entity_graph import EntityGraph
from norax.memory.store import MemoryStore

store = MemoryStore(root=MEMORY_ROOT)
store.refresh()
canonical = store.all_canonical()

eg = EntityGraph(root=MEMORY_ROOT)
eg.load()

# Count entities per memory (rough: capitalized words, tech terms)
import re
expected_entities = set()
for mem in canonical[:100]:  # Sample first 100
    # Find capitalized multi-word terms, tech terms
    terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', mem.text)
    terms += re.findall(r'\b(?:Python|Discord|Ollama|Base|USDC|ETH|Norax|GitHub|Playwright|FastAPI|PostgreSQL)\b', mem.text)
    expected_entities.update(terms)

actual_entities = set()
# Try different attribute patterns
for attr in ('_entities', 'entities', '_entity_map'):
    val = getattr(eg, attr, None)
    if val and isinstance(val, dict):
        actual_entities = set(val.keys())
        break
# Fallback: try _entity_count
if not actual_entities:
    ec = getattr(eg, '_entity_count', 0)
    print(f"Entity count (from attr): {ec}")

coverage = len(expected_entities & actual_entities) / max(len(expected_entities), 1) * 100
missing = expected_entities - actual_entities

print(f"Expected entities (from 100 memories): {len(expected_entities)}")
print(f"Actual entities in graph: {len(actual_entities)}")
print(f"Coverage: {coverage:.1f}%")
print(f"Missing key entities: {list(missing)[:10]}")
print(f"Entity graph links: {getattr(eg, '_link_count', 'unknown')}")
print(f"VERDICT: {'GOOD' if coverage > 70 else 'NEEDS_EXTRACTION'} (coverage={coverage:.1f}%)")
""",
    },
    {
        "topic": "autonomy_effectiveness",
        "description": "Measure what the autonomy engine actually accomplishes",
        "search_queries": [
            "autonomous agent self-healing monitoring effectiveness 2025 2026",
            "agent idle time productive tasks background maintenance",
            "self-improving agent feedback loop production metrics",
        ],
        "test_concept": "Audit autonomy engine logs for actual work done",
        "test_script": "autonomy_audit.py",
        "test_code": r"""
import sys, os, re
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta

NORAX_ROOT = Path(os.environ.get("NORAX_ROOT", "~/norax")).expanduser()
log_path = NORAX_ROOT / "logs" / "norax-ai.log"
if not log_path.exists():
    print("NO_LOG")
    sys.exit(0)

# Parse autonomy events from last 24h
events = Counter()
fixes = 0
learning_cycles = 0
syncs = 0
prompt_opts = 0

# Read last 5000 lines (roughly 24h)
with open(log_path) as f:
    lines = f.readlines()[-5000:]

for line in lines:
    if "autonomy" in line.lower():
        if "self_heal" in line.lower() and "fix" in line.lower():
            fixes += 1
            events["self_heal"] += 1
        elif "learning" in line.lower():
            learning_cycles += 1
            events["learning"] += 1
        elif "scratchpad" in line.lower() and "sync" in line.lower():
            syncs += 1
            events["scratchpad_sync"] += 1
        elif "prompt_opt" in line.lower():
            prompt_opts += 1
            events["prompt_opt"] += 1

print(f"Autonomy events (last ~24h):")
print(f"  Scratchpad syncs: {syncs}")
print(f"  Self-heal fixes: {fixes}")
print(f"  Learning cycles: {learning_cycles}")
print(f"  Prompt optimizations: {prompt_opts}")
print(f"  Total events: {sum(events.values())}")
productive = fixes + learning_cycles + prompt_opts
print(f"Productive actions (non-sync): {productive}")
print(f"VERDICT: {'ACTIVE' if productive > 3 else 'IDLE'} (productive={productive})")
""",
    },
    {
        "topic": "response_pipeline_quality",
        "description": "Audit response quality, empty responses, and revision failures",
        "search_queries": [
            "LLM response quality monitoring production agent 2025 2026",
            "agent output verification revision pipeline failure modes",
            "response gate empty output prevention agent",
        ],
        "test_concept": "Count empty responses, revision failures, and gate blocks",
        "test_script": "response_audit.py",
        "test_code": r"""
import sys, os, re
from pathlib import Path
from collections import Counter

NORAX_ROOT = Path(os.environ.get("NORAX_ROOT", "~/norax")).expanduser()
log_path = NORAX_ROOT / "logs" / "norax-ai.log"
if not log_path.exists():
    print("NO_LOG")
    sys.exit(0)

with open(log_path) as f:
    lines = f.readlines()[-5000:]

empty_responses = 0
revision_needed = 0
revision_success = 0
revision_empty = 0
gate_blocks = 0
verifier_issues = 0
best_of_n_replacements = 0
stream_retries = 0

for line in lines:
    if "response_gate" in line and "revision_needed" in line:
        revision_needed += 1
    if "response_gate" in line and "revision" in line and "ok" in line.lower():
        revision_success += 1
    if "output_verifier" in line and "issue" in line.lower():
        verifier_issues += 1
    if "best_of_n" in line and "replace" in line.lower():
        best_of_n_replacements += 1
    if "stream_retry" in line:
        stream_retries += 1
    if "no content" in line.lower() or "empty" in line.lower():
        if "response" in line.lower() or "content" in line.lower():
            empty_responses += 1

print(f"Response pipeline audit (last ~5000 log lines):")
print(f"  Revision needed: {revision_needed}")
print(f"  Verifier issues: {verifier_issues}")
print(f"  Best-of-N replacements: {best_of_n_replacements}")
print(f"  Stream retries: {stream_retries}")
print(f"  Empty response mentions: {empty_responses}")
print(f"VERDICT: {'STABLE' if stream_retries < 10 and empty_responses < 5 else 'UNSTABLE'}")
""",
    },
    {
        "topic": "commerce_revenue_readiness",
        "description": "Audit commerce layer readiness and identify revenue opportunities",
        "search_queries": [
            "AI agent API monetization crypto payments 2025 2026",
            "microservice pricing strategy API credits usage-based",
            "autonomous agent revenue generation bounty airdrop",
        ],
        "test_concept": "Verify commerce endpoints are live and accepting payments",
        "test_script": "commerce_audit.py",
        "test_code": r"""
import sys, os, json, subprocess

# Check if admin server is running
commerce_url = os.environ.get("NORAX_COMMERCE_URL", "http://127.0.0.1:8894").rstrip("/")
result = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"{commerce_url}/api/pricing"], capture_output=True, text=True, timeout=5)
pricing_status = result.stdout.strip()

result = subprocess.run(["curl", "-s", f"{commerce_url}/api/wallet"], capture_output=True, text=True, timeout=5)
wallet_data = {}
try:
    wallet_data = json.loads(result.stdout)
except json.JSONDecodeError:
    pass

result = subprocess.run(["curl", "-s", f"{commerce_url}/api/revenue"], capture_output=True, text=True, timeout=5)
revenue_data = {}
try:
    revenue_data = json.loads(result.stdout)
except json.JSONDecodeError:
    pass

print(f"Pricing endpoint: HTTP {pricing_status}")
print(f"Wallet configured: {bool(wallet_data.get('evm_address'))}")
print(f"Revenue earned: ${revenue_data.get('total_earned_usd', 0)}")
print(f"Active sessions: {revenue_data.get('active_api_sessions', 0)}")
print(f"Airdrop campaigns: {revenue_data.get('airdrop_campaigns', {}).get('zero_capital_available', 0)} available")

# Check payment gate module
try:
    from pathlib import Path
    NORAX_ROOT = Path(os.environ.get("NORAX_ROOT", "~/norax")).expanduser()
    sys.path.insert(0, str(NORAX_ROOT))
    from norax.commerce.payment_gate import SessionStore, verify_payment
    print(f"Payment gate module: importable ({SessionStore.__name__}, {verify_payment.__name__})")
except Exception as e:
    print(f"Payment gate module: FAILED ({e})")

print(f"VERDICT: {'READY' if pricing_status == '200' else 'DOWN'} (pricing={pricing_status})")
""",
    },
]


@dataclass
class LearningConfig:
    """Configuration for self-directed learning."""

    enabled: bool = True
    interval_sec: float = 1800.0  # 30 min between learning cycles
    max_search_results: int = 12
    max_test_time_sec: float = 45.0
    max_test_output_lines: int = 80
    staging_host: str = field(
        default_factory=lambda: os.environ.get("NORAX_STAGING_HOST", "").strip()
    )
    staging_path: str = field(
        default_factory=lambda: os.environ.get("NORAX_STAGING_PATH", "~/norax").strip() or "~/norax"
    )
    allow_local_tests: bool = field(
        default_factory=lambda: _env_flag("NORAX_LEARNING_ALLOW_LOCAL_TESTS", False)
    )


@dataclass
class LearningResult:
    """Result of one learning cycle."""

    topic: str
    timestamp: float
    search_results: list[dict] = field(default_factory=list)
    test_executed: bool = False
    test_passed: bool = False
    test_verdict: str = ""
    test_location: str = "not_run"
    test_output: str = ""
    test_error: str = ""
    insights: str = ""
    recommendations: list[str] = field(default_factory=list)
    duration_sec: float = 0.0


class LearningLoop:
    """Self-directed learning engine — practical stack improvement edition.

    Picks topics from a curriculum of real stack improvements, researches
    best practices, tests implementations on staging via SSH, and
    records actionable findings to memory.
    """

    def __init__(
        self,
        config: LearningConfig | None = None,
        *,
        gateway: Any | None = None,
        event_log: Any | None = None,
    ) -> None:
        self.config = config or LearningConfig()
        self.gateway = gateway
        self.event_log = event_log
        self._cycle_idx: int = 0
        self._last_learn: float = 0.0
        self._task: asyncio.Task | None = None
        self._results: list[LearningResult] = []

        # Load progress from log
        self._load_progress()

    def _load_progress(self) -> None:
        """Load learning progress from log file."""
        if LEARNING_LOG.exists():
            try:
                lines = LEARNING_LOG.read_text().strip().split("\n")
                completed_topics: set[str] = set()
                completed_cycles = 0
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("learning_loop: skipping malformed progress record")
                        continue
                    if not isinstance(entry, dict):
                        continue
                    completed_cycles += 1
                    completed_topics.add(str(entry.get("topic", "")))
                self._cycle_idx = completed_cycles
                log.info(
                    "learning_loop: %d completed cycles across %d topics loaded",
                    completed_cycles,
                    len(completed_topics),
                )
            except Exception as e:
                log.debug("learning_loop.load_progress: %r", e)

    def _next_topic(self) -> dict[str, Any]:
        """Get next topic from curriculum, cycling through."""
        idx = self._cycle_idx % len(CURRICULUM)
        self._cycle_idx += 1
        return CURRICULUM[idx]

    async def _search(self, query: str) -> list[dict]:
        """Run web search via gateway."""
        if self.gateway is None:
            return []
        try:
            from norax.dispatch.tools import t_web_search

            results = await t_web_search(query=query, count=self.config.max_search_results)
            return results.get("items", [])
        except Exception as e:
            log.debug("learning_loop.search: %r", e)
            return []

    @staticmethod
    def _bounded_output(output: str, max_lines: int) -> str:
        lines = str(output or "").splitlines()
        max_lines = max(1, int(max_lines))
        if len(lines) <= max_lines:
            return str(output or "")
        return "\n".join(lines[:max_lines]) + "\n... (truncated)"

    @staticmethod
    def _remote_path_expr(path: str) -> str:
        """Quote an operator path while retaining remote-home expansion."""
        value = str(path or "~/norax").strip() or "~/norax"
        if value == "~":
            return '"$HOME"'
        if value.startswith("~/"):
            return '"$HOME"/' + shlex.quote(value[2:])
        return shlex.quote(value)

    def _run_test_on_staging(self, code: str) -> tuple[bool, str, str, str]:
        """Execute one static curriculum script under the configured policy.

        Returns ``(execution_ok, stdout, stderr, location)``. A zero process
        exit only proves that the audit ran; its health verdict is classified
        separately.
        """
        staging_host = self.config.staging_host.strip()
        if not staging_host and not self.config.allow_local_tests:
            return (
                False,
                "",
                "No staging host is configured; local learning tests are disabled. "
                "Set NORAX_STAGING_HOST or explicitly opt in with "
                "NORAX_LEARNING_ALLOW_LOCAL_TESTS=1.",
                "not_run",
            )

        import tempfile

        location = f"ssh:{staging_host}" if staging_host else "local_opt_in"
        try:
            if not staging_host:
                # A temporary script avoids modifying the repository. This is
                # not an OS sandbox, hence the separate local-execution opt-in.
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(code)
                    test_path = f.name
                try:
                    child_env = dict(os.environ)
                    child_env.update(
                        {
                            "NORAX_ROOT": str(_NORAX_ROOT),
                            "NORAX_MEMORY_ROOT": str(_MEMORY_ROOT),
                        }
                    )
                    proc = subprocess.run(
                        [sys.executable, test_path],
                        capture_output=True,
                        text=True,
                        timeout=max(0.1, self.config.max_test_time_sec),
                        cwd=str(_NORAX_ROOT),
                        env=child_env,
                    )
                    return (
                        proc.returncode == 0,
                        self._bounded_output(proc.stdout, self.config.max_test_output_lines),
                        (proc.stderr or "")[-500:],
                        location,
                    )
                finally:
                    Path(test_path).unlink(missing_ok=True)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, prefix="_norax_learn_"
            ) as tf:
                tf.write(code)
                local_test = tf.name
            remote_test = f"/tmp/_norax_learning_{uuid.uuid4().hex}.py"
            try:
                copied = subprocess.run(
                    ["scp", "-q", local_test, f"{staging_host}:{remote_test}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if copied.returncode != 0:
                    return False, "", (copied.stderr or "scp failed")[-500:], location

                root_expr = self._remote_path_expr(self.config.staging_path)
                memory_expr = root_expr + "/memory"
                remote_expr = shlex.quote(remote_test)
                remote_command = (
                    f"cd -- {root_expr} && NORAX_ROOT={root_expr} "
                    f"NORAX_MEMORY_ROOT={memory_expr} python3 {remote_expr} 2>&1; "
                    f"status=$?; rm -f -- {remote_expr}; exit $status"
                )
                proc = subprocess.run(
                    ["ssh", staging_host, remote_command],
                    capture_output=True,
                    text=True,
                    timeout=max(0.1, self.config.max_test_time_sec),
                )
                return (
                    proc.returncode == 0,
                    self._bounded_output(proc.stdout, self.config.max_test_output_lines),
                    (proc.stderr or "")[-500:],
                    location,
                )
            finally:
                Path(local_test).unlink(missing_ok=True)
        except subprocess.TimeoutExpired:
            return False, "", f"Test timed out at {location}", location
        except Exception as e:
            return False, "", f"Test execution error at {location}: {e}", location

    @staticmethod
    def _classify_verdict(output: str) -> tuple[str, bool]:
        """Return the declared audit verdict and whether it denotes health."""
        match = re.search(r"(?m)^VERDICT:\s*([A-Z_]+)\b", output or "")
        if match is None:
            return "INCONCLUSIVE", False
        verdict = match.group(1)
        healthy = verdict in {"GOOD", "EFFICIENT", "CLEAN", "STABLE", "ACTIVE", "READY"}
        return verdict, healthy

    def _extract_insights(
        self,
        search_results: list[dict],
        test_output: str,
        test_passed: bool,
        topic: dict[str, Any],
    ) -> tuple[str, list[str]]:
        """Extract key insights and actionable recommendations."""
        insights = []
        recommendations = []

        # Research insights — deeper extraction
        for r in search_results[:5]:
            title = r.get("title", "")
            snippet = r.get("snippet", "")[:200]
            url = r.get("url", "")
            if title:
                insights.append(f"- [{title}]({url}): {snippet}")

        # Test insights — parse VERDICT line
        for line in test_output.split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith("VERDICT:"):
                insights.append(f"  VERDICT: {line_stripped[8:]}")
            elif any(
                kw in line_stripped.lower()
                for kw in [
                    "avg",
                    "coverage",
                    "error",
                    "rate",
                    "score",
                    "ms",
                    "duplicates",
                    "stale",
                    "missing",
                    "ready",
                    "down",
                    "productive",
                    "efficient",
                    "wasteful",
                    "stable",
                    "unstable",
                ]
            ):
                insights.append(f"  METRIC: {line_stripped}")

        # Generate recommendations based on topic and test results
        topic_name = topic.get("topic", "")
        if "NEEDS_TUNING" in test_output or "WASTEFUL" in test_output or "UNSTABLE" in test_output:
            recommendations.append(
                f"Priority: {topic_name} needs improvement based on test metrics"
            )
        if "NEEDS_DEDUP" in test_output:
            recommendations.append("Run memory deduplication pass on canonical store")
        if "NEEDS_EXTRACTION" in test_output:
            recommendations.append("Rebuild entity graph with improved entity extraction")
        if "IDLE" in test_output:
            recommendations.append(
                "Autonomy engine not productive enough — increase task diversity"
            )
        if "DOWN" in test_output:
            recommendations.append("Commerce server is down — restart admin console")
        if test_passed:
            verdict, _ = self._classify_verdict(test_output)
            recommendations.append(
                f"{topic_name} audit reported {verdict}; no remediation was identified"
            )

        # Add research-based recommendations
        if search_results:
            top_result = search_results[0]
            recommendations.append(
                f"Research suggests: {top_result.get('title', '')} — {top_result.get('snippet', '')[:100]}"
            )

        return "\n".join(insights) if insights else "No insights extracted", recommendations

    def _persist_result(self, result: LearningResult) -> None:
        """Persist learning result to memory."""
        LEARNING_DIR.mkdir(parents=True, exist_ok=True)

        # Append to JSONL log
        entry = {
            "topic": result.topic,
            "timestamp": result.timestamp,
            "test_executed": result.test_executed,
            "test_passed": result.test_passed,
            "test_verdict": result.test_verdict,
            "test_location": result.test_location,
            "duration_sec": round(result.duration_sec, 2),
            "n_search_results": len(result.search_results),
            "insights": result.insights,
            "recommendations": result.recommendations,
            "test_output": result.test_output[:3000],
            "test_error": result.test_error[:500] if result.test_error else "",
        }
        with path_lock(LEARNING_LOG):
            with LEARNING_LOG.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")

        # Also write a readable markdown summary
        md_file = LEARNING_DIR / (
            f"learned_{result.topic}_{int(result.timestamp * 1000)}_{uuid.uuid4().hex[:8]}.md"
        )
        recs_text = (
            "\n".join(f"- {r}" for r in result.recommendations)
            if result.recommendations
            else "None"
        )
        md_content = f"""# Learning: {result.topic}

**Date:** {time.strftime("%Y-%m-%d %H:%M", time.gmtime(result.timestamp))}
**Test executed:** {"✅" if result.test_executed else "❌"}
**Test passed:** {"✅" if result.test_passed else "❌"}
**Verdict:** {result.test_verdict or "INCONCLUSIVE"}
**Duration:** {result.duration_sec:.1f}s
**Tested on:** {result.test_location}

## Research Findings

{result.insights}

## Test Output (staging)

```
{result.test_output}
```

## Test Errors

```
{result.test_error}
```

## Actionable Recommendations

{recs_text}
"""
        atomic_write_text(md_file, md_content)
        log.info("learning_loop: persisted %s -> %s", result.topic, md_file.name)

    async def run_one_cycle(self) -> LearningResult:
        """Run one complete learning cycle."""
        start = time.time()
        topic = self._next_topic()
        result = LearningResult(
            topic=topic["topic"],
            timestamp=time.time(),
        )

        log.info("learning_loop: starting cycle topic=%s", topic["topic"])

        # Phase 1: Research (deeper — all queries, more results)
        for query in topic.get("search_queries", []):
            search_results = await self._search(query)
            result.search_results.extend(search_results)
            await asyncio.sleep(1)  # Rate limit

        # Phase 2: Test on staging via SSH
        test_code = topic.get("test_code", "")
        if test_code:
            execution_ok, stdout, stderr, location = await asyncio.to_thread(
                self._run_test_on_staging,
                test_code,
            )
            verdict, verdict_passed = self._classify_verdict(stdout)
            result.test_executed = execution_ok
            result.test_passed = execution_ok and verdict_passed
            result.test_verdict = verdict
            result.test_location = location
            result.test_output = stdout
            result.test_error = stderr

        # Phase 3: Extract insights and recommendations
        result.insights, result.recommendations = self._extract_insights(
            result.search_results, result.test_output, result.test_passed, topic
        )

        # Phase 4: Persist
        result.duration_sec = time.time() - start
        self._persist_result(result)

        # Phase 5: Record to event log
        if self.event_log:
            try:
                await self.event_log.append(
                    "learning.cycle_complete",
                    {
                        "topic": result.topic,
                        "test_executed": result.test_executed,
                        "test_passed": result.test_passed,
                        "test_verdict": result.test_verdict,
                        "test_location": result.test_location,
                        "duration_sec": round(result.duration_sec, 2),
                        "n_search_results": len(result.search_results),
                        "recommendations": result.recommendations,
                    },
                )
            except Exception as e:
                log.debug("learning_loop.event_record_failed error=%r", e)

        self._results.append(result)
        log.info(
            "learning_loop: cycle complete topic=%s passed=%s duration=%.1fs recs=%d",
            result.topic,
            result.test_passed,
            result.duration_sec,
            len(result.recommendations),
        )
        return result

    async def run_forever(self) -> None:
        """Background loop: run learning cycles at configured interval."""
        log.info("learning_loop: started, interval=%ds", int(self.config.interval_sec))
        while True:
            try:
                await asyncio.sleep(max(1.0, self.config.interval_sec))
                await self.run_one_cycle()
            except asyncio.CancelledError:
                log.info("learning_loop: cancelled")
                break
            except Exception:
                log.debug("learning_loop.error", exc_info=True)
                await asyncio.sleep(60)

    def start(self) -> asyncio.Task:
        """Start the learning loop as a background task."""
        if self._task is not None and not self._task.done():
            return self._task
        self._task = asyncio.create_task(self.run_forever())
        return self._task

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    @property
    def stats(self) -> dict[str, Any]:
        """Return current learning statistics."""
        return {
            "cycles_completed": len(self._results),
            "curriculum_idx": self._cycle_idx,
            "curriculum_size": len(CURRICULUM),
            "topics_covered": list({r.topic for r in self._results}),
            "pass_rate": (
                sum(1 for r in self._results if r.test_passed) / len(self._results)
                if self._results
                else 0.0
            ),
            "total_recommendations": sum(len(r.recommendations) for r in self._results),
        }
