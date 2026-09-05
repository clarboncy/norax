# Norax stack improvement review — 2026-04-29

Flow guard active: 30-minute hard stop enforced for this review.

## What web/research confirms

Existing Norax research notes and fresh web checks point to the same priorities:

1. Memory should be first-class, tiered, and editable, not just context stuffing.
2. Vector + graph memory is stronger than vector-only for long-lived agents, especially multi-hop recall.
3. Sleep-time/background consolidation is a quality/latency win.
4. Production agents need checkpoint/resume, explicit loop/time budgets, tracing, and tool-call observability.
5. Tool use improves when tools are retrieved/selected dynamically and results are cached/deduped.

## Current stack strengths

- Event-sourced runtime with hash-chained events.
- Hybrid retrieval classes exist: keyword + embedding RRF.
- Sleep consolidation, episodic buffer, Hebbian links, VTA writer, meta-learning are already present.
- Dispatch has risk gate, idempotency/duplicate guard, per-tool timeouts, read cache, and loop guard.
- Web search has 3-tier cascade: SearXNG -> Ollama search-grounded -> Serper.

## Gaps / fixes to prioritize

### P0 — enforce wall-clock stop everywhere

Issue found: `agent_loop.py` had `DEFAULT_TIMEOUT_SECONDS = 1800`, but deadline handling tried to force one more final LLM call instead of stopping. That can still extend/loop.

Implemented during this review:

- Changed default to `NORAX_FLOW_TIMEOUT_SECONDS` env var, default 1800.
- Deadline now returns a direct user alert `⏱️ 30-minute task limit reached...` and stops without another tool/LLM recovery loop.
- Emits `flow_timeout` event with elapsed, rounds, tool_calls, policy.
- Verified `python3 -m py_compile` passes.

### P0 — turn-level checkpoints

Add `state/checkpoints/<turn_id>.json` after every tool round:

- messages so far
- trace/tool results
- rounds
- started_at/deadline
- active channel/user

On restart, resume or alert instead of replaying blindly.

### P0 — memory self-edit tools

Add explicit `memory_append`, `memory_replace`, `memory_search`, `memory_compact` tools with safe schemas and risk tiers. Current `append_memory` exists, but a dedicated memory API should validate target tier and format.

### P0/P1 — retrieval scoring

Current MemoryStore parses W1-W5 and age, but HybridRetriever RRF does not apply recency/importance decay. Add scoring:

`final = rrf_score * weight * recency_decay(kind)`

Suggested half-lives:

- procedural: 30d
- semantic: 90d
- intel: 7d
- scratch/focus: no decay/hot boost

### P1 — make hybrid retrieval actually visible in planning

Runtime currently keeps plan-time retrieval keyword-only because hot path is sync. Options:

1. Add async `plan_turn_async` so L5 can await HybridRetriever.
2. Prebuild hybrid index and expose sync query path for embeddings.
3. Keep keyword in hot path but increase use of `search_memory` tool when semantic recall is needed.

### P1 — graph memory

Create/populate `knowledge_graph.db` or equivalent with entity triples from sleep replay:

- person/project/tool entities
- relationships: uses, blocked_by, solved_by, prefers, owns, credential_for (redacted)
- co-retrieval: vector results expand by 1-hop graph neighbors

### P1 — contrastive procedural memory

Sleep replay should write failure-success pairs:

`FAIL when: ... -> DO instead: ... |W5`

This directly addresses mistakes like browser/session loops.

### P1 — websearch reliability

DuckDuckGo raw HTML produced bot challenge. Existing registry already prefers SearXNG. Recommended:

- Health-check SearXNG at startup and surface in `status`.
- Add Brave/Tavily/Kagi as optional API fallback if Serper missing.
- Cache search results per query for 1-6h to reduce bot challenges and latency.
- Fetch top result pages only when needed, not during broad search.

### P1 — tool performance / selection

- Add per-tool success metrics and recent failure memory.
- Tool retriever should use both user text and recent failure patterns.
- Promote proven desktop tools (`ydotool`, `grim`, `dmap`, `perception`) into registry instead of ad hoc scripts.
- For browser auth/session tasks: max 2 recovery attempts, then stop and alert.

### P2 — observability

- Export events to OpenTelemetry spans: turn, LLM call, tool call, retrieval, sleep replay.
- Add dashboard panels: tool success rate, loop detections, timeout stops, retrieval source mix, memory writes by VTA route.

### P2 — sandboxed exec / coding loop

- Move risky code execution into container/sandbox.
- Add first-class test-run-fix loop with max iterations and checkpointing.
- Use structured patches for multi-hunk edits.

## Immediate next implementation sequence

1. Restart Norax runtime so new 30-minute stop behavior is live.
2. Add checkpoint/resume around `agent_loop.run_agent_loop` rounds.
3. Add recency/importance scoring in `HybridRetriever`.
4. Add dedicated memory edit tools.
5. Add computer-use/dmap/perception tools to registry.
6. Add websearch provider health status and query cache.
