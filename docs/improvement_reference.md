# Norax Improvement Reference

**Date**: 2026-04-24
**Scope**: Legitimate, non-janky upgrades across harness, coding ability, uptime, and memory
**Baseline**: Norax v0.11.0, 163 tests, phases 1–10 green

---

## Where We Stand Now

| Layer | What We Have | Maturity |
|---|---|---|
| **Runtime** | Single-process event loop, systemd `norax-ai.service`, hash-chained `events.jsonl` | ✅ Solid |
| **Ingress** | Discord, HTTP, Cron adapters → ingress bus | ✅ Solid |
| **Brain** | Hot path (amygdala, basal ganglia, thalamus, VTA, arousal), sleep replay, meta-learning | ✅ Novel — ahead of most |
| **Dispatch** | 12 tools, risk gate, loop guard, idempotency, budget, per-tool timeouts | ✅ Solid |
| **Memory** | 11 stores (semantic, procedural, intel, sleep, scratchpad, focus, episodic), Hebbian links, VTA writer | ⚠️ Good foundation, gaps below |
| **Retrieval** | Fast, local, hybrid, attention, external retrievers | ⚠️ Exists but no semantic vector index in prod |
| **Observability** | Prometheus metrics, structured logs, circuit breaker, retry with backoff | ✅ Solid |
| **Safety** | Secret scrubber, rate limits, risk gate per-tier | ✅ Solid |
| **Sleep** | Hippocampal replay: tool patterns, co-activation, flashbulbs, failure extraction | ✅ Novel — ahead of most |
| **Config** | Single `runtime.jsonc`, `.env`, reload on restart | ⚠️ No hot-reload |

---

## Domain 1: Harness & Runtime

### State of the Art (2025–2026 Research)

The industry has converged on **Agent = Model + Harness** (LangChain, HKUST survey, Botlearn). The harness is the moat. Key patterns:

1. **Checkpointing + State Snapshots** — Save execution state at each tool round. Resume after crash without replaying from scratch. (Brightlume, Substack harness dissection)
2. **Event-Sourced State Machines** — Every state transition is an immutable event. Replay from any point. Current state is a fold of events. (We already have hash-chained events.jsonl — this is a strength.)
3. **Thread-per-Conversation** — Multiple concurrent users, each with independent state. No cross-contamination. (Standard in LangGraph, Temporal-based agents)
4. **Graceful Degradation** — If LLM gateway is down, queue inbound messages and process when upstream recovers. Don't lose turns. (SRE for Agentic Systems pattern)
5. **Hot Config Reload** — Reload `runtime.jsonc` without restart. Standard in production daemons.
6. **Structured Tracing** — OpenTelemetry-compatible spans per tool call, per turn, per agent round. Not just logs — actual distributed tracing.

### Our Gaps

| Gap | Priority | Effort | Impact |
|---|---|---|---|
| No checkpoint/resume after crash — we restart from scratch | **P0** | 2–3 days | Eliminates lost work on OOM/kill |
| No hot config reload | P1 | 0.5 day | Easier ops, fewer restarts |
| Single conversation at a time (no thread isolation) | P1 | 3–5 days | Multi-user support |
| No inbound queue on gateway failure | P1 | 1 day | No lost messages during outages |
| No OTel tracing export | P2 | 2 days | Better debugging in staging |
| No structured checkpoint format (just raw events.jsonl replay) | P2 | 1 day | Faster recovery |

### Recommendations

- **R1**: Add turn-level checkpointing. After each tool round completes, snapshot `messages_so_far` + `tools_used` + `pending` to `state/checkpoints/<turn_id>.json`. On restart with pending inbound, check for checkpoint and resume loop instead of starting fresh.
- **R2**: Watch `runtime.jsonc` mtime. On change, call `runtime.reload_config()` without process restart. Validate schema first — reject bad config without crashing.
- **R3**: Build an inbound queue (even a simple SQLite-backed `state/inbound_queue.db`). When gateway returns 5xx or timeout, park the envelope. Retry on next health check.
- **R4**: Export OTel traces from the event log. Our `events.jsonl` already has the data — we just need a span sink that exports to Jaeger/Tempo.

---

## Domain 2: Memory

### State of the Art

The MemGPT/Letta paper defined the modern taxonomy:

| Tier | Analogy | Purpose |
|---|---|---|
| **Core / Working** | RAM | In-context system prompt, active blocks |
| **Archival** | Disk | Long-term facts, compressed history |
| **Recall** | Cache | Recent conversation segments, retrieved on demand |

Letta's key innovation: **the agent self-edits its own memory via tool calls**. It moves data between tiers. This is "LLM as OS".

Other advances:
- **Sleep-time compute** (Letta 2025): Agent reasons offline about recent interactions, consolidates memories, pre-computes likely-needed context before next interaction. We already do this via hippocampal replay.
- **Tiered semantic search**: Not just keyword match — vector similarity with recency+importance decay scoring. (MemGPT, Zep, LangMem)
- **Memory blocks**: Labeled containers that compile into system prompt. Agent can `core_memory_append`, `core_memory_replace`, `archival_memory_insert`, `archival_memory_search`. (Letta)
- **Contrastive memory**: Store failure-success pairs explicitly. On replay, learn "when I did X it failed, when I did Y it worked." (Trial and Error, ACL 2024)
- **Entity-centric memory**: Maintain a knowledge graph of entities and relationships, not just flat documents. (Zep, Graphiti)

### Our Gaps

| Gap | Priority | Effort | Impact |
|---|---|---|---|
| No vector index in production (faiss listed as optional dep) | **P0** | 1 day | Semantic search works properly |
| No agent-self-edit memory tools (no `memory_append`, `memory_replace`) | **P0** | 2 days | Agent manages own memory like Letta |
| No recency × importance decay on retrieval | P1 | 1 day | Better signal-to-noise in recall |
| No entity knowledge graph — just flat files | P1 | 5–7 days | Relational reasoning |
| No contrastive failure-success pairs in sleep replay | P2 | 2 days | Learn from mistakes faster |
| Episodic buffer not compacted aggressively enough | P2 | 1 day | Prevents context bloat over days |

### Recommendations

- **R5**: Ship faiss-cpu in main deps (not optional). Build the vector index on boot from `memory/{semantic,procedural,intel}`. Use it as the primary retrieval path, fallback to keyword match.
- **R6**: Add `memory_append` and `memory_replace` tools. These let the agent write/edit its own memory files during turns. Gated by risk tier (owner can always write, guest restricted). This is the Letta pattern and it's essential for long-lived agents.
- **R7**: Score every Neuron on retrieval as: `weight × recency_decay(age) × relevance(similarity)`. Recency decay = `exp(-age / half_life)` where `half_life` varies by kind (semantic=90d, procedural=30d, intel=7d). This is standard in LangMem/Zep.
- **R8**: In sleep replay, write `(failure_context, success_context)` pairs to `memory/procedural/contrastive.md`. On next retrieval, if a query matches a failure context, the success context is co-retrieved. Simple, powerful.
- **R9**: Add a `memory_compact` step to sleep that summarizes episodes older than 7 days into a few compressed sentences, then archives the raw episodes. Prevents episodic buffer from growing unbounded.

---

## Domain 3: Coding Ability

### State of the Art

- **SWE-bench Verified**: Claude Opus 4.5 hit **80.9%** (first model to beat human engineers). GPT-5.x is competitive. The bar is high.
- **Top coding agents** (2025–2026): Claude Code, Codex CLI, Devin, Cursor Agent. Key patterns they share:
  1. **Multi-file awareness** — They read the entire repo context, not just one file
  2. **Test-driven loops** — Write test → run → fix → repeat until pass
  3. **Sandboxed execution** — Run code in isolated environments, diff output
  4. **Structured output** — JSON diff patches, not free-form text that needs parsing
  5. **Self-verification** — Agent runs its own code, checks output, iterates

### Our Gaps

| Gap | Priority | Effort | Impact |
|---|---|---|---|
| No dedicated code-execution sandbox — `exec` tool runs directly on host | **P0** | 3–4 days | Safety + self-verification |
| No test-run-iterate loop built into dispatch | P1 | 2 days | Fixes stick, fewer false positives |
| No structured diff output for edits — we do string replacement | P1 | 1 day | More reliable multi-hunk edits |
| No repo-level context management — model sees ad-hoc reads | P2 | 2 days | Better architecture decisions |
| No `git` integration (commit, diff, branch) in tool set | P2 | 1 day | Version control for agent changes |

### Recommendations

- **R10**: Add a Docker-based code sandbox. `exec` in the sandbox is isolated; `exec` on host stays gated. Sandbox can run `pytest`, `python`, `node` without risking the host. This is how Claude Code and Codex work.
- **R11**: Add a `test_loop` tool or dispatch pattern: given a test command and max retries, run test → if fail → feed stderr to model → apply fix → re-run. Up to N iterations. This is the core loop of every coding agent.
- **R12**: Replace naive `edit` (first-occurrence string replace) with a diff-based approach: accept `Search/Replace` blocks (old → new) or line-range patches. More reliable for multi-hunk edits.
- **R13**: Add `git_commit`, `git_diff`, `git_log` tools. Low effort, high value. Every change the agent makes should be committable.
- **R14**: On large code tasks, auto-inject a repo map (file tree + key signatures) into context before the first tool call. Similar to Claude Code's repo map. Reduces wandering reads.

---

## Domain 4: 24/7 Uptime & Self-Healing

### State of the Art

From the self-healing agent literature (DEV Community, Recovery-Bench, SRE for Agentic Systems):

1. **Watchdog process** — External health checker pings `/healthz`. If failing, kills + restarts the main process. (We have systemd for this, but it's basic.)
2. **Automatic error classification** — Agent logs errors with severity. Watchdog acts on critical-severity patterns differently from transient ones.
3. **State reconstruction** — On crash, replay events from last checkpoint to rebuild working state. Don't start from zero.
4. **Degraded mode** — If upstream LLM is down, enter degraded mode: queue inbound, serve cached responses or "I'm temporarily offline" messages, auto-recover when gateway returns.
5. **Heartbeat + self-diagnosis** — Agent periodically checks its own subsystems (memory, gateway, tools). If something's wrong, it alerts the owner instead of silently failing.
6. **Circuit breaker patterns** — Per-upstream circuit breaker. If gateway is 5xxing, open the circuit, stop hammering it, try again after cooldown. (We have `circuit.py` — need to verify it's wired.)
7. **Recovery-Bench** (Letta 2025) — A benchmark specifically for error recovery. Agents should be tested on: corrupted state, missing tools, partial output, permission errors, timeout mid-turn.

### Our Gaps

| Gap | Priority | Effort | Impact |
|---|---|---|---|
| No degraded mode on gateway failure | **P0** | 1–2 days | Agent doesn't silently die |
| No self-diagnosis heartbeat | P1 | 1 day | Early warning before owner notices |
| Circuit breaker exists but may not be fully wired | P1 | 0.5 day | Prevents gateway hammering |
| No checkpoint-based state reconstruction | P1 | (depends on R1) | Faster recovery |
| No Recovery-Bench equivalent test suite | P2 | 2–3 days | Confirms resilience |
| No watchdog beyond systemd `Restart=always` | P2 | 1 day | Smarter restart decisions |

### Recommendations

- **R15**: Implement degraded mode. When gateway is unreachable (circuit open), park inbound in queue, send a "temporarily offline" reply to Discord, and keep checking gateway health every 60s. Auto-recover when it's back.
- **R16**: Add a `self_check()` coroutine that runs every 5 minutes: ping gateway, check memory fs is writable, verify `/healthz` responds. On failure, emit a `health_alert` event and message the owner.
- **R17**: Verify `circuit.py` is wired into the gateway client path. If a request 5xxes 3 times in a row, open the circuit for 30s, then half-open (try one request). This is the standard pattern.
- **R18**: Build a Recovery-Bench: a test suite that simulates crash mid-turn, corrupted events.jsonl, missing memory files, gateway timeout, permission errors. Assert agent recovers correctly in each case. This is the real measure of uptime.

---

## Domain 5: Prompt & Context Engineering

### State of the Art

- **Deterministic prompt assembly** (we already do this — strength)
- **Context window management**: Not just "trim oldest messages" — use importance scoring to decide what to keep. (MemGPT, LangChain)
- **Multi-system-prompt blocks**: Compile multiple labeled blocks into the final prompt. Each block can be independently updated. (Letta memory blocks)
- **Prompt compression**: Summarize older turns into dense 1–2 line summaries to save tokens. (Standard in long-context agents)

### Our Gaps

| Gap | Priority | Effort | Impact |
|---|---|---|---|
| No importance-based context trimming — just rolling window | P1 | 1–2 days | Better use of context window |
| No prompt compression for old turns | P1 | 1 day | Saves tokens, fits more history |
| Memory blocks not independently editable by agent | (covered by R6) | — | — |

### Recommendations

- **R19**: Before each LLM call, score messages in the context window by importance (tool calls > text-only messages > short acks). When over budget, drop lowest-importance messages first, replacing them with a 1-line summary.
- **R20**: Implement turn summaries: after every turn, write a compressed summary (≤100 chars) to episodic buffer. On next context assembly, use summaries for turns older than N instead of raw messages.

---

## Priority Roadmap

| Phase | Items | Est. Time | Owner |
|---|---|---|---|
| **Phase A — Stability** | R1 (checkpoints), R15 (degraded mode), R17 (circuit wire), R5 (vector index) | ~5 days | Norax production |
| **Phase B — Memory** | R6 (self-edit tools), R7 (decay scoring), R9 (compaction), R19 (importance trim) | ~5 days | Norax production + staging |
| **Phase C — Coding** | R10 (sandbox), R11 (test loop), R12 (diff edits), R13 (git tools) | ~7 days | Staging first, then prod |
| **Phase D — Resilience** | R16 (self-check), R18 (recovery bench), R2 (hot reload), R4 (OTel), R20 (turn summaries) | ~7 days | Staging first |
| **Phase E — Advanced** | R8 (contrastive pairs), R14 (repo map), R3 (inbound queue), R9 (entity graph) | ~10 days | Staging |

Total estimate: **~34 days of focused work** to go from solid to best-in-class.

---

## What We Already Do Better Than Most

Let's be honest about our strengths:

- **Hippocampal replay** — Most agents don't have offline memory consolidation. Letta has sleep-time compute, but our replay with tool patterns, co-activations, flashbulbs, and failure extraction is more biologically grounded and arguably more robust.
- **Hash-chained event log** — Immutable audit trail. Most agents just log to stdout. We can replay any point in history.
- **Brain architecture** — Amygdala (threat), basal ganglia (habit/loop), thalamus (routing), VTA (reward), arousal (energy). This is novel. Most agents are a flat loop.
- **Risk gate per tier** — Not just "can this tool run" but "can THIS sender at THIS tier run THIS tool." Most agents have a flat allowlist.
- **Deterministic prompt assembly** — No dynamic prompt hacking. Static hash gated. Most agents let prompt drift.

We're not starting from zero. We're starting from a strong, idiosyncratic foundation. The improvements above fill the gaps where industry best practices have caught up or surpassed our current implementation.