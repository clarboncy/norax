# Runtime audit follow-up — 2026-09-06

This records the source repairs and regression evidence following the
[previous audit](audit-followup-2026-09-05.md). It is not an external
certification, proof of exhaustive correctness, or a general agent ranking.
No new comparative benchmarks were run during this final audit pass.

## Repairs

- Full-completion defaults are 250 model rounds, a four-hour turn budget,
  and 1,000 tool calls. The configured round limit, minimal-config fallback,
  orchestration, generated subtasks, and settings UI no longer silently use
  the old smaller defaults. Cancellation, authorization, no-progress checks,
  and bounded execution remain enforced; 250 is available capacity, not a
  requirement to waste 250 rounds on a finished task.
- Orchestration now receives prior conversation context and observes a
  wall-clock deadline. Expiry returns an explicitly incomplete result, not
  a fabricated success.
- The persisted conversation window is model-independent. Selecting a
  smaller-context local model no longer permanently shrinks the canonical
  window and discards history needed after switching back to a cloud model.
- Request history is still bounded per model. Selection charges actual JSON
  argument sizes, preserves tool-call/result pairs, strips private reasoning
  variants, and retains recent user/assistant context when a large tool trace
  cannot fit. Grouping frames once avoids rescanning every frame per turn.
- Historical tool arguments are always JSON objects, and tiny text budgets
  cannot accidentally append the entire original message through a zero-index
  tail slice.
- Local llama.cpp requests consistently map reasoning controls and enable
  prefix-cache reuse for direct and relay endpoints. Endpoint detection parses
  the actual URL port; a cloud URL path or query cannot accidentally select
  local-only options. Streaming and nonstreaming paths are covered.
- Native Ollama requests through a dual-protocol relay retain their effort
  field until native serialization. The OpenAI template mapper must not consume
  it first; otherwise `none` becomes default thinking and a small liveness
  request can spend its entire allowance without producing visible output.
- Desktop key validation rejects option-like arguments and malformed
  combinations before either backend runs. An X11 helper printing `--help`
  and exiting zero must not be reported as a successful key press.
- Web fetching uses direct HTTP first and an optional configured rendering
  fallback. Unconfigured fallback, sufficient direct results, and deliberately
  short excerpts add no fallback calls or disclosure checks. External response
  bodies are bounded; HTTP errors, nonliteral success flags, and malformed
  content cannot masquerade as successful fetches. Direct intranet permission
  does not authorize sending private targets to an external scraper.
- Exact-response requests are handled through model instructions, not a
  response-rewriting helper. Regression tests preserve a model's qualifying
  text even after a successful tool call; a completion marker must not hide
  unfinished work.
- Norax's identity and memory architecture remain intact. Public source
  contains neither deployment's learned memories or environment files.
- Memory decay and fact evolution now use bounded, symlink-safe persistence;
  archival is collision-safe and rollback-safe, corrupt records are isolated,
  and private append permissions are enforced. Consolidation preflights
  deduplication atomically so a companion-file problem cannot destroy source
  evidence.
- Vector backends validate finite dimensions consistently, preserve arbitrary
  record identifiers across Qdrant, and actually exercise configured fallback
  order. Learned tool experience preserves evidence counts while removing raw
  arguments, and persisted prediction state cannot double-count on reload.
- Runtime screenshot probes use unique temporary paths, watchdog identities no
  longer collide on duplicate task names, retry observers cannot break retries,
  circuit configurations are not shared mutable state, and public health
  diagnostics are bounded and secret-scrubbed.
- Literal successful capability receipts now recover failed, degraded, or stale
  health state. Capability diagnostics are redacted and bounded before storage
  or logging; this work occurs only during state changes, not in model-token or
  tool-dispatch hot loops.
- Session restore, delivery, and history boundaries now fail closed on corrupt
  shapes while retaining valid state, preserve cancellation semantics, bound
  diagnostics, and avoid unnecessary persistence or retry work. The unused
  duplicate runtime idempotency implementation was removed; the single live
  dispatch implementation is enforced at 100% line and branch coverage.
- Model management now preserves provider credentials transactionally across
  disable/remove failures, clears stale effective-provider state after routing
  errors, bounds discovery and diagnostics, closes discovery clients, and
  coalesces asynchronous persistence without losing the final update. Both
  synchronous and asynchronous failure paths are secret-scrubbed.
- Firecrawl seed discovery uses the documented v2 top-level Map/Crawl schemas,
  bounded streamed responses, one client per crawl, finite polling, and remote
  cancellation on timeout or caller cancellation. Seed discovery is parallel,
  fault-isolated, same-host constrained, de-duplicated, and reuses crawl output
  instead of paying for a second fetch.
- Firecrawl scrape now uses the v2 endpoint too. Connector ownership is isolated
  from the tool registry, with shared finite input coercion and web-safety
  primitives instead of a circular dependency. Compatibility aliases preserve
  the existing tool surface without duplicating implementations. Scrape errors,
  direct-fetch transport errors, and HTTP error bodies are bounded and
  secret-scrubbed before entering model context.
- Local reads reject directories and non-regular streams, contain stat and scan
  failures, cap explicit page requests at 20,000 lines, and retain bounded LRU
  behavior. Directory scans and diagnostics stay finite without adding work to
  the normal small-file path.
- Research persistence and returned connector errors are bounded and
  secret-scrubbed. Research memory is written privately and atomically, cannot
  traverse outside its intel directory, and never erases existing knowledge on
  an empty pass. Externally derived memory, user bodies, and attachment metadata
  are fenced as untrusted content, including protection against forged closing
  delimiters. The in-process topic-lock registry is hard-bounded without
  rejecting overflow work; the process-safe file lock remains authoritative.

## Verification

`scripts/quality_gate.py` passed on the current source:

- 2,255 tests in the branch-coverage selection, plus three isolated subprocess
  acceptance tests; 10 desktop-affecting host tests were intentionally excluded.
- 78.96% statement coverage, 68.06% branch coverage, 76.08% combined coverage;
  all 45 critical-module floors passed. The aggregate release floors remain
  ratcheted to 77% statements, 65% branches, and 74% combined.
- Twenty-six focused production modules now have enforced 100% statement and
  branch coverage, including deep research, prompt assembly, model management,
  session, delivery, history, runtime validation, capability state, health,
  retry, circuit-breaker, watchdog, operations, cognition, Firecrawl, input
  coercion, and outbound web-safety boundaries. The aggregate is not described
  as 100%; uncovered behavior remains audit work.
- Lockfile, systemd units, every shell helper, Ruff lint/format, mypy,
  compilation, stack boundaries, publication hygiene, and secret scanning.
- Real loopback connector transport tests include concurrent MCP calls,
  transport errors, and the A2A boundary. These tests do not require paid
  inference or contact another person's agent.

Before activation, both explicitly selected active memory roots passed lint.
The primary's selected event generations verified 74,007 records and the
secondary's verified 28 records, with zero corrupt records. These are point-in-
time checks; ordinary live operation continues appending events.

The separate development site's restored commerce import and configured chat
router passed seven offline regression tests, including rejection of provider
failures as successful responses, authenticated page aliases, and fail-closed
startup without a configured password. Its unrelated local changes are preserved.

## Deployment and evidence boundaries

Deploy code with a fast-forward source update, never by copying a primary
environment or memory directory to the secondary. Check private configuration
digests before/after, restart sequentially, and verify readiness plus the
deployment's configured connector. The primary passed a real owner-bridge
response check without sending an external Discord message. The secondary's
optional owner bridge is not configured and was not silently enabled; its
running completion probe and Discord connection were verified instead.

Live activation exposed an additional private secondary configuration still
set to 18 rounds. Only that value (now 250) and its relay transport (now the
relay's direct OpenAI-compatible endpoint) were changed; structural comparison
verified the rest of that private configuration was unchanged. The old private
configuration was backed up locally on that deployment.

The development site's public API and admin routes pointed at an unused port.
They now reach the active authenticated server. Public pricing/downloads return
HTTP 200, unauthenticated admin/API requests return 401, and the existing login
successfully opens the admin, hub, and model API through the public hostname.
The site's existing password and cookie secret were moved to a private,
mode-0600 service environment without changing them. The inherited site password
still warrants operator rotation; it was not changed without coordinating
client access. No credentials were added to public source.

The primary's transport-only probe override was corrected to completion mode
for activation. The existing 600-second idle-aware schedule and 32-token,
reasoning-disabled probe are retained. A successful model-catalog request is
not recorded as proof of inference.

A fresh release still needs its own uninterrupted 72-hour reliability window.
Earlier campaigns include interruptions and failed observations; preserve them
when beginning a new campaign. Restarting or passing the source gate cannot
retroactively satisfy that requirement. Published earlier comparison results
remain historical observations of their recorded source/workloads, not a new
evaluation of this final source or proof of superiority across all models.
