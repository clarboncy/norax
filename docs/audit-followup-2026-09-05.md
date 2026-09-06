# Continued audit and remaining release acceptance

Status: source fixes verified and runtime revision `d23e4902decb` activated
in both deployments on 2026-09-06 UTC. The 72-hour reliability observation
is collecting; it has not passed yet. Passing engineering checks
does not establish comparative superiority or correctness of every possible
configuration.

## Defects reproduced and corrected

Each group below has regression tests that failed against the preceding
implementation before the fix was applied.

| Area | Observed failure | Correction |
| --- | --- | --- |
| Channel ownership | A cancelled older worker detached its replacement's queue, losing an accepted following message. Pre-start cancellation leaked pending capacity; draining still accepted work. | Cleanup is limited to the owning worker, pre-start cancellation releases its queue, and draining rejects new turns. |
| Event persistence | Another writer's append left the rotation anchor stale. Missing/truncated current data could restart a chain or concatenate JSON records. An empty file could produce an empty rotation. | Startup and append share a process lock; rotation reads the actual tail; lost history and unterminated records fail explicitly without rewriting evidence; empty files do not rotate. |
| Response delivery | Metrics exceptions could prevent an already generated answer from being delivered. Empty-response diagnostics exposed raw provider metadata. | Delivery tolerates metrics failure and diagnostic replies omit raw provider data. |
| Idle maintenance | Long user turns could trigger maintenance. No spill files prevented independent pruning and projection synchronization. Skill file work blocked the serving loop. | Active/queued work defers maintenance; spill consolidation no longer gates unrelated maintenance; skill loading/mining/generation use worker threads. |
| Provider transactions | Cancelling a caller during a store write left persisted settings updated while the live route remained unchanged. | The runtime owns and shields complete update/disable/remove transactions from caller cancellation, tracks them for shutdown, and rejects new mutations during draining. Existing rollback and client retirement semantics are preserved. |
| Burn-in evidence | Missing/duplicate timestamps, long collection gaps, stale samples, absent counters, and invalid latency could produce a passing assessment. Overlapping collectors lost samples. | Require ordered observations across the actual 72-hour window, at most 180 seconds between observations, finite counters/latencies and explicit metrics availability. Serialize collection and campaign archival with a process lock. |
| Release gate | An explicitly selected but missing event log was silently skipped. | Always invoke verification for an explicitly selected log; absence fails the gate. |
| Shell validation gate | A single `bash -n` invocation with multiple script paths parsed only the first file, accepting broken later scripts. | Validate each of the nine shell scripts separately. Regression tests demonstrate rejection of an invalid second helper and an invalid setup script without executing either. |

Existing damaged historical logs are preserved. The persistence changes do
not fabricate replacement hashes or erase records to produce a green report.

## Verification

- Full source gate: 1,738 main tests and three subprocess acceptance tests
  passed. Ten interactive host tests remain excluded from unattended runs.
- Coverage: 73.98% statements, 61.18% branches, 70.61% combined. Coverage
  still leaves untested behavior; it is not an exhaustive correctness claim.
- 28 critical modules are gated. Event persistence now has an explicit
  floor; delivery, lifecycle, and cognition floors were raised. Cognition
  coverage increased from 31.28%/21.21% to 59.90%/53.12% line/branch.
- Formatting, lint, type checks, lockfile verification, publication checks,
  secret scanning, systemd/shell syntax, and stack boundary checks passed.
- Concurrent event stress: four writers, 80 records, multiple rotations,
  zero missing or invalid records.
- Event append benchmark: eight alternating batches of 300 appends measured
  median 0.1342 ms before and 0.0903 ms after. This small local benchmark
  establishes no observed regression in that workload, not a general speedup.
- Confined real model task: local `qwen3.8-27b-fast:latest` passed in 6.215 s;
  cloud `glm-5.3:cloud` passed in 4.855 s. Both completed read, exact edit,
  readback, and artifact verification in four rounds. Private receipts are
  retained under `benchmarks/runs/` and excluded from publication.
- Both running agents passed initial operational checks with unchanged process
  identities and zero automatic restarts. A later secondary service activation
  was observed before the source synchronization; no runtime restart commands
  were issued during this follow-up. Both final operational checks passed.
  Selected active memory lint passed.
  The primary deployment's selected event generations verified 73,099/73,099
  records; the secondary deployment's selected active chain verified 3/3.

## Approved activation and live validation

- Both services were restarted sequentially, with primary readiness verified
  before restarting the secondary. Both report runtime build `d23e4902decb`,
  healthy listeners, connected Discord adapters, verified model completions,
  new service invocation identities, and zero automatic restarts.
- Private environment and secondary runtime-settings/configuration checksums
  were unchanged across the restarts. No memories or credentials were copied
  between deployments. Unrelated services were not restarted.
- Each deployment passed the confined execution harness in six model rounds:
  create a three-line artifact, read two distinct pages, verify an expected
  no-match search, run a fixed command, and report the exact completion marker.
  This uses a separate harness process; it is not a live Discord delivery test.
  The first secondary invocation used the shell's default configuration and
  failed to connect. Repeating with that service's actual environment passed.
  Deployment acceptance must select the deployed configuration and environment,
  not assume an SSH shell inherits systemd environment overrides.
- Active memory lint and stack checks passed in both deployments. Selected
  event generations verified 73,136/73,136 primary and 5/5 secondary records,
  with no corruption. Older inactive archives remain untouched.
- Each deployment passed 40 read-only HTTP checks at concurrency four,
  covering readiness/status and unauthorized access rejection for provider
  configuration and owner history. Maximum observed local-listener request
  time was 5.6 ms primary and 78.9 ms secondary. This is a small control-plane
  smoke test, not a generation-throughput or sustained-load benchmark.
- An additional confined cloud `glm-5.3:cloud` write/read task passed exact
  random-content and artifact verification in three rounds, taking 32.504 s.
  Model response time varies; the earlier faster edit task is not representative
  proof of consistently low cloud latency.
- Both corrected campaigns, `activated-d23e490-20260906`, started around
  02:03 UTC on September 6. Earlier evidence was archived intact. Their timers
  are adding observations with explicit metrics, clean event chains, zero
  error deltas, stable service identities, and successful completion probes.
  Earliest possible 72-hour acceptance is around 02:03 UTC on September 9,
  subject to the actual observations and all gates passing.
- The subsequent shell-gate correction changes only validation code and tests;
  no further runtime restart is needed to use it.

## Remaining acceptance work

1. Gather 72 hours of uninterrupted evidence using the corrected collector.
   Older samples lack the new explicit metrics-availability field and cannot
   be retroactively treated as equivalent evidence. Archive prior campaigns
   intact when starting the corrected campaign.
2. Broaden representative tasks across configured providers and connectors,
   including actual end-to-end connector delivery and sustained concurrent use.
   These passing model tasks do not validate every installable local or cloud
   model, every connector, or prolonged concurrent use.
3. Continue focused testing of less-covered turn, provider, cognition, and
   recovery branches. Current coverage percentages explicitly leave gaps.
4. Any comparative performance claim requires a matched, repeatable external
   evaluation. No competitive ranking is asserted by this audit.

The audit remains open until the applicable deployment acceptance evidence is
available. Repository synchronization alone does not close these items.
