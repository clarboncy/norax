# Audit verification — 2026-09-05

## Disposition

The repository-wide source gate is green and the audited source is deployed to
the Norax user service. The running deployment is completion-verified on its
configured local model, the remote relay is healthy, the operational monitor
passes, local and cloud models have each completed a confined edit-and-verify
agent task, and a clean 72-hour burn-in campaign is collecting evidence.

This is an engineering audit and deployment acceptance result. It is not an
external certification that every behavior is correct or that Norax
outperforms another agent. Comparative superiority requires a matched,
repeatable external evaluation rather than self-authored smoke tests.

## Complete source gate

`scripts/quality_gate.py` verifies all of the following and exits nonzero on
any failure:

- frozen dependency-lock consistency;
- syntax of every shipped systemd unit and Bash helper;
- Ruff lint and formatting across application, tests, tools, scripts,
  benchmarks, and training code;
- mypy across runtime code and every maintenance/benchmark/training helper;
- bytecode compilation;
- publication hygiene and fail-closed secret scanning;
- the stack boundary verifier;
- line, branch, combined, per-module, and critical-module coverage floors;
- subprocess-isolated runtime acceptance tests;
- the explicitly selected live event chain and memory root.

Final measured test scope:

- 1,679 tests in the coverage run;
- three subprocess-integration tests run separately and passed;
- 10 `host_integration` tests intentionally excluded because they can mutate
  the operator's browser, clipboard, pointer, and keyboard session;
- 73.60% statement coverage, 60.91% branch coverage, and 70.25% combined
  coverage;
- all 17 critical-module floors passed without lowering thresholds;
- 8,582 of 8,582 active event records verified with zero corruption after the
  final deployment activation;
- active-memory lint passed;
- `git diff --check` passed.

The locked all-extras dependency export also passes `pip-audit` with no known
vulnerabilities. The lock was upgraded from the vulnerable bundled `pip`
release to 26.2.1. A fresh wheel build installed with dependencies into a clean
virtual environment and loaded packaged defaults, soul data, configuration,
and the packaged computer-use backend successfully. The final wheel contains
183 members with no duplicate names, hidden package residue, unsafe archive
paths, or symlinks. Its SHA-256 is
`84d752e70f561315c0f11ecfb8e5ea9c2772a1a0131647dd3ef2fc3f452801c0`.

## Major correctness and integrity work

The audit covered the production runtime, model routing, tools, memory,
connectors, persistence, operational helpers, packaging, and deployment. Key
fixes include:

- strict provider request/response validation, bounded payloads, honest
  streaming completion, model-route reporting, and failover receipts;
- bounded turn concurrency and queues, cancellation ownership, task-death
  supervision, continuation-state preservation, and explicit failure metrics;
- atomic and locked file writes/edits with permission preservation, symlink
  behavior, UTF-8 byte receipts, transaction serialization, and rollback
  tests;
- atomic memory/index/cache persistence, corruption recovery, incremental
  projection reconciliation, bounded durable logs, and cross-process locking;
- fail-closed A2A, MCP, browser, sandbox, remote-node, payment, commerce, and
  deep-research boundary validation;
- benchmark confinement, exact outcome/readback checks, counter-contamination
  detection, and durable private receipts;
- fail-closed publication, secret, event-chain, and memory helpers instead of
  vacuous success on missing, unreadable, or unselected inputs;
- syntactically valid systemd paths using `%h`, deterministic unit deployment,
  explicit managed-unit drift checks, and exact service/timer health semantics;
- fixed-port startup semantics across the runtime, Agent OS, setup, and docs,
  eliminating silent port selection that could split clients from servers;
- setup that writes `.env` atomically at mode 0600, passes it to both launched
  processes, waits for real health before adding an optional provider, keeps
  provider secrets out of process arguments, and fails if either endpoint is
  unreachable;
- private commerce defaults under the XDG/Norax state root at mode 0700 rather
  than the Python package tree, with publication tests that reject package-local
  commerce state and hidden package files;
- private Agent OS SQLite state under the XDG/Norax state root rather than the
  dashboard source directory, with directory/database modes 0700/0600 and
  regular-file, ownership, hard-link, and symlink validation;
- a bounded systemd-query retry in the operational monitor so a transient user
  bus error does not fabricate an outage, while persistent query failures still
  fail closed with a diagnostic;
- burn-in baselines both systemd's automatic-restart counter and the service
  activation identity, so manual restarts can no longer pass the uninterrupted
  runtime gate; every sample also requires a successful service query and an
  active process;
- capability freshness refreshed from literal-success real tool receipts,
  without synthetic browser launches, desktop captures, or network probes in
  the turn hot path;
- removal of duplicate unwired authorities for graph storage, incremental
  indexing, runtime supervision, memory consolidation/redaction/deduplication,
  and event replay.

`norax_agent_turn_failed_total` is now a real counter. Planning failures,
agent-loop failures, delivery failures, and uncaught turn crashes increment it;
the burn-in gate measures growth from its campaign baseline instead of querying
a nonexistent metric.

## Genuine serving and capability evidence

The periodic serving check no longer treats a model-catalog request as a
completed inference. Production uses completion mode and requires the exact
visible response `OK`. Unexpected text is rejected and logged only as length
plus digest. The probe:

- disables private reasoning for this binary task;
- permits at most 32 output tokens (the observed answer is two tokens);
- runs every 600 seconds;
- waits for 30 seconds of user-idle time;
- yields while turns are queued or active, with bounded evidence staleness so
  readiness cannot remain green forever on old evidence.

The real configured local route passed the pre-deployment probe in 270 ms. The
first probe in the deployed runtime passed in 1,010.2 ms. Interactive browser
and desktop capabilities are genuinely exercised once at startup; lightweight
operational checks continue every five minutes without repeatedly launching a
browser or capturing the user's desktop.

## Local and cloud agent-loop acceptance

`benchmarks/agent_capability_bench.py --task edit-verify` exposes only `read`
and `edit`, confines both to one temporary buggy Python file, disables
failover, and requires this sequence:

1. read and verify the original file;
2. perform one exact edit;
3. read back and verify the saved file;
4. return the exact final marker without an incomplete flag.

| Route | Model | Result | Rounds | Elapsed | Receipt |
| --- | --- | ---: | ---: | ---: | --- |
| Local llama.cpp | `qwen3.8-27b-fast:latest` | pass | 4 | 6.365 s | `benchmarks/runs/model-edit-local-20260905.json` |
| Ollama cloud | `glm-5.3:cloud` | pass | 4 | 5.620 s | `benchmarks/runs/model-edit-cloud-20260905.json` |

Both performed the required read → edit → readback sequence and produced the
exact verified artifact. This establishes working tool use on these two tested
routes; it does not generalize to every installable model.

The hardened legacy harness also exposes only its three required tools, confines
filesystem access and shell commands to a temporary task root, and requires
exact order, readback, exit codes, artifact bytes, and final marker. The local
model passed that five-round live check.

## Deployed runtime acceptance

Only `norax-ai.service` was restarted for the final audited deployment; the
remote relay and unrelated live agent runtime were not restarted. The final
service started at 17:55:51 EDT with PID 1,989,224 and zero automatic restarts.
Readiness reported:

- completion-verified local model route;
- connected Discord ingress with zero reconnects after startup;
- 45,001 memory neurons available;
- writable event log;
- ready browser, computer, sandbox, web-search, remote-relay, tool-manifest,
  and required probe-loop capabilities.

The final exact completion probe passed in 940.1 ms. `scripts/monitor.py --json`
passed the exact unit, timer, endpoint, completion, and relay checks. Repository
and installed managed units pass
`deploy-systemd-units.sh --check` with no drift.

The post-deploy HTTP delivery benchmark passed `direct_fact`, `exact_output`,
and `reasoning_arithmetic` 3/3 at score 1.0. Measurement was uncontaminated:
three expected brain turns, three observed turns, and three gateway requests.
Mean measured brain time was 9.648 seconds. Receipt:
`benchmarks/runs/deployment-final-20260905.json`.

The active process, active `.env`, application source, configuration, scripts,
and installed units contain no references to the retired API provider. All
disabled/inactive legacy provider unit and drop-in artifacts found in the final
sweep were removed from systemd's load path and moved to the desktop trash, so
they remain recoverable. No active service was stopped or restarted during this
cleanup.

The audit also found private commerce residue created by old default paths in
the package source tree. The empty lock was moved to trash and the private
`intel.md` artifact was moved without reading it to
`~/.local/state/norax/audit-quarantine/20260905-package-residue/` at mode 0600.
The quarantine directory is mode 0700. The corrected defaults and publication
gate prevent recurrence.

The disabled Agent OS database was likewise moved without reading it from the
checkout into `~/.local/state/norax/agent-os/agent_os.db`, and two ignored UI
backup artifacts were moved to trash. Removing the obsolete ignore rules makes
future source-tree state visible to publication checks.

## Burn-in state

The old failed campaign was preserved under the private burn-in archive; it was
not deleted or relabeled. The pre-final `post-audit-20260905` campaign collected
305 samples over 18,748.9 seconds with every instantaneous gate passing. It was
then archived rather than blended across the final planned runtime activation.

A short intermediate campaign was also archived when activation-identity
tracking was added to close the manual-restart blind spot. The authoritative
uninterrupted campaign, `final-audit-v2-20260905`, began at 18:03:26 EDT. Its
first sample passed every instantaneous gate:

- HTTP readiness and Discord connectivity;
- exact completion evidence and completion p95 below five seconds;
- clean event chain;
- unchanged systemd activation identity and an active service process;
- no service-restart growth;
- no brain-error growth;
- no agent-turn-failure growth.

Observed readiness and Discord rates are both 100%; the first completion sample
was 940.1 ms.

The corrected timer was live-tested: it fired the collector successfully and
re-armed for the next minute. Burn-in directories are mode 0700 and evidence,
history, status, and archived files are mode 0600. The campaign remains
`collecting` until both the 72-hour duration and minimum-sample gates are met;
the earliest duration boundary is September 8 at approximately 18:03 EDT.
Current state is authoritative in
`~/.local/state/norax/burnin/status.json`.

## Explicit limitations

- The 72-hour burn-in is active but cannot truthfully be marked passed before
  its time boundary.
- The 10 host-input integration tests were not run against the operator's live
  desktop. Startup probes did verify the deployed browser and display backends.
- Local/cloud acceptance samples are deterministic smoke/acceptance checks, not
  SWE-bench, Terminal-Bench, long-horizon, security-red-team, or competitor
  rankings.
- A claim that Norax is “world class” or outperforms competing agents requires
  externally comparable tasks, repeated trials, cost/latency controls, and
  adjudication beyond this repository audit.
