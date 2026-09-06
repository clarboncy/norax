# Continued audit and remaining release acceptance

Status: source fixes verified; activation of the updated runtime and the
72-hour reliability observation remain pending. Passing engineering checks
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
| Burn-in evidence | Missing/duplicate timestamps, long collection gaps, stale samples, absent counters, and invalid latency could produce a passing assessment. Overlapping collectors lost samples. | Require ordered observations across the actual 72-hour window, at most 180 seconds between observations, finite counters/latencies and explicit metrics availability. Serialize collection and campaign archival with a process lock. |
| Release gate | An explicitly selected but missing event log was silently skipped. | Always invoke verification for an explicitly selected log; absence fails the gate. |

Existing damaged historical logs are preserved. The persistence changes do
not fabricate replacement hashes or erase records to produce a green report.

## Verification

- Full source gate: 1,732 main tests and three subprocess acceptance tests
  passed. Ten interactive host tests remain excluded from unattended runs.
- Coverage: 73.94% statements, 61.15% branches, 70.57% combined. Coverage
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
- Both running agents passed operational checks with unchanged process
  identities and zero automatic restarts. Selected active memory lint passed.
  The primary deployment's selected event generations verified 73,099/73,099
  records; the secondary deployment's selected active chain verified 3/3.

## Remaining acceptance work

1. Activate the updated runtime in each deployment during an approved restart,
   then verify listeners, delivery, model routes, selected data roots, and
   service identity against the activated revision.
2. Gather 72 hours of uninterrupted evidence using the corrected collector.
   Older samples lack the new explicit metrics-availability field and cannot
   be retroactively treated as equivalent evidence. Archive prior campaigns
   intact when starting the corrected campaign.
3. Broaden representative tasks across configured providers and connectors.
   Two passing model tasks do not validate every installable local or cloud
   model, every connector, or prolonged concurrent use.
4. Continue focused testing of less-covered turn, provider, cognition, and
   recovery branches. Current coverage percentages explicitly leave gaps.
5. Any comparative performance claim requires a matched, repeatable external
   evaluation. No competitive ranking is asserted by this audit.

The audit remains open until the applicable deployment acceptance evidence is
available. Repository synchronization alone does not close these items.
