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

## Verification

`scripts/quality_gate.py` passed on this source:

- 1,793 tests in the branch-coverage run, plus three isolated subprocess
  acceptance tests; 10 desktop-affecting host tests were intentionally excluded.
- 74.08% statement coverage, 61.33% branch coverage, 70.72% combined coverage;
  all 28 critical-module floors passed without lowering thresholds.
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
