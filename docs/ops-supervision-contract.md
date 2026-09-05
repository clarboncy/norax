# Norax AI — Operations & Supervision Contract

## Supervision ownership boundary

### systemd owns

| Responsibility | Implementation |
|---|---|
| Process restart | `Restart=always` on `norax-ai.service` |
| Startup ordering | Provider endpoints are probed by `norax-ai.service`; no forwarding proxy is required |
| Rate-limit restart loops | `StartLimitBurst=10`, `StartLimitIntervalSec=120` |
| Graceful shutdown | `TimeoutStopSec=30`, `KillSignal=SIGTERM` |
| Enable on boot | `WantedBy=default.target` + `systemctl --user enable` |

### Norax runtime owns (in-process)

| Responsibility | Implementation |
|---|---|
| Adapter/background-task supervision | `asyncio.create_task` with named tasks in `Runtime.run()` |
| Memory maintenance loop | `_idle_sleep_loop()` task, created in `Runtime.run()` |
| Cache warmup | `_warmup_task`, created in `Runtime.run()` |
| Required task death | Crashes the process (systemd restarts) |
| Optional task death | Degrades capability, logged via `CapabilityRegistry` |
| Task health visibility | `/readyz` endpoint reports failed capabilities |

### Decision model

- **Required tasks** (ingress, event log): if they crash, the process exits and systemd restarts.
- **Optional tasks** (sleep loop, warmup, autonomy): if they crash, the capability is marked as failed in `CapabilityRegistry` and visible in `/readyz`. The runtime continues serving turns.
- **Discord adapter**: optional — if no token configured, runtime starts without it.
- **Model gateway client/router**: in-process. It probes configured provider
  endpoints directly; no forwarding-proxy service is required.

## Sleep / memory maintenance concurrency contract

### Two schedulers

| Scheduler | Trigger | Scope |
|---|---|---|
| In-process idle loop | `Runtime._idle_sleep_loop()` — runs when no turns for N seconds | Consolidation, decay, projection sync |
| `norax-sleep.timer` | systemd hourly at `:17` | Window flush, consolidation, replay |

### Lock boundary

Both schedulers operate on the same `memory/` root. Safe overlap is guaranteed by:

1. **In-process lock**: `MemoryCoordinator` uses an `asyncio.Lock` to serialize projection syncs within the process.
2. **Cross-process lock**: `norax-sleep.service` acquires a file lock via `flock` on `memory/sleep/.lock` before starting. The in-process loop checks the same lock and skips if held.
3. **Idempotent operations**: All memory operations (consolidation, decay, projection sync) are idempotent — re-running them produces the same result.

### Events

- `runtime.start` — includes `config_hash`, `effective_model`, `effective_provider`, `version`
- `sleep.start` / `sleep.end` — emitted by both schedulers
- `sleep.skipped` — emitted when lock is held by the other scheduler

## Capability health

### `/status` endpoint

Returns the effective route and every registered capability with explicit
ready/degraded/failed/stale state. For example:
```json
{
  "ok": true,
  "status": "ready",
  "effective_model": "qwen3.8-27b-fast:latest",
  "effective_provider": "ollama_local",
  "version": "0.11.0",
  "capabilities": {
    "tool_manifest": {
      "enabled": true,
      "state": "ready",
      "required_for_readiness": true,
      "last_error": ""
    }
  },
  "degraded": []
}
```

### `/readyz` endpoint

Returns exact readiness components for capabilities, gateway evidence, memory,
the event log, Discord, completion probe, and burn-in. `ok` is false when a
required capability is failed/stale or another required component is not live.
Transport-only gateway evidence is labeled as such and cannot satisfy the
completion-verified burn-in gate.

## Deployment

```bash
# Deploy systemd units from repository
./ops/systemd/deploy-systemd-units.sh

# Check for drift (CI mode)
./ops/systemd/deploy-systemd-units.sh --check
```

## Managed units

| Unit | Purpose | Enabled |
|---|---|---|
| `norax-ai.service` | Main runtime | yes |
| `norax-remote-relay.service` | Remote relay on :8765 | yes |
| `norax-sleep.timer` | Hourly sleep flush | yes |
| `norax-burnin.timer` | Minute release-evidence sample | yes |
| `norax-agent-os.service` | Optional local dashboard | no |
