# Norax AI — Subsystem Classification

This document classifies every implemented subsystem and its production lifecycle status.

## Classification legend

| Status | Meaning |
|---|---|
| **Production** | Wired into `Runtime.build()`/`run()`, has config, lifecycle, health, tests |
| **Optional** | Wired behind a feature flag defaulting off; capability status in `/readyz` |
| **Standalone** | Supported repository helper with an explicit caller/host contract; not runtime-wired |
| **Compatibility** | Thin legacy interface delegating to a canonical implementation |
| **Experimental** | Measured but non-production helper whose heuristic limits are part of its output contract |
| **Inventory** | Read-only probe; reports observed resources without implying control capability |
| **Host smoke** | Deployment-host validation that may skip absent optional backends |
| **Prototype** | Code exists but not wired; marked unsupported |
| **Dead** | No external consumer; candidate for removal |

## Current classification

| Subsystem | Status | Wired? | Config key | Notes |
|---|---|---|---|---|
| Core runtime (ingress, agent loop, tools) | Production | yes | — | Always on |
| Multi-provider gateway client/router | Production | yes (in-process) | `gateway.*` | Direct provider routes with health evidence |
| Gateway proxy CLI | Standalone | no | environment | Optional compatibility proxy; not part of the active service stack |
| Discord adapter | Production | yes | `channels.discord.enabled` | Starts if token present |
| HTTP ingress adapter | Production | yes | `http.bind` | Always on |
| Cron adapter | Production | yes | `cron.enabled` | Starts if jobs configured |
| Memory store + retrievers | Production | yes | `paths.memory_root` | Always on |
| Sleep consolidation (idle loop) | Production | yes | — | In-process task |
| Sleep flush (timer) | Production | yes (systemd) | — | `norax-sleep.timer` |
| Autonomy engine | Optional | yes | `NORAX_AUTONOMY_ENABLED` | Default off; experiments require their own flags too |
| Graceful degradation | Production | yes | — | Always registered |
| Capability registry | Production | yes | — | Always on |
| Remote relay | Production | yes (systemd) | — | `norax-remote-relay.service` |
| A2A server | Optional | yes | `a2a.enabled` or `connectors.a2a_server` | Default off |
| A2A client | Prototype | no | — | Code exists, not advertised as a connector |
| MCP client | Optional | yes | `mcp.enabled` or `connectors.mcp_client` | Default off; stdio transport |
| MCP server | Prototype | no | — | Code exists, not advertised as a connector |
| Commerce API | Optional | yes | `commerce.enabled` or `connectors.commerce` | Default off |
| Active inference | Optional | yes | `connectors.active_inference` | Default off |
| Idle replay/skill learning | Optional | yes | `cognition.idle_learning` | Default off; positive guidance requires verified outcomes |

## Standalone and packaged helper classification

The repository also contains host-facing helpers. They do not become runtime
capabilities merely by existing in `tools/`; runtime wiring and host dependency
checks are separate concerns.

| Helper surface | Status | Runtime-wired? | Contract |
|---|---|---:|---|
| `norax.dispatch.tools.t_computer_use` + packaged `computer_use.py` | Production | yes | Async child-process boundary; passive capability report; explicit backend failures; packaged in the wheel |
| `computer-use.sh` | Compatibility | no | Thin exit-code-preserving wrapper around the canonical Python backend |
| `action_gate.py` | Standalone | no | Pattern/syntax heuristic, not authentication; HIGH defaults to blocked unless an authenticated host asserts approval; CRITICAL always blocks locally |
| `browser_controller.py` + `bridge.py` | Standalone | no | Local CDP helpers; mutations require an observed focused tab and propagate protocol/readback failures |
| `perception.py` + `dmap.py` | Standalone | no | Host/session-dependent AT-SPI, CDP, and OCR perception; stale coordinate maps are rejected for actions |
| `motor_cortex.py` | Standalone | no | Bounded executor with filesystem confinement and optional before/after UI comparison; backend dispatch is not mislabeled as UI outcome proof |
| `flow_guard.py` + `browser_wrapper.py` | Standalone | no | Cooperative deadline and bounded Playwright wrapper; the guard does not kill arbitrary caller work by itself |
| `cv_fusion.py` + `visual_hierarchy.py` | Experimental | no | Bounded cross-source deduplication and measured screenshot features; numeric scores are labeled heuristics, not calibrated confidence |
| `vision-navigator.py` | Experimental | no | Inference-only CLI over a configured Ollama vision model; model coordinates/descriptions are explicitly unverified; imported automation requires a caller-owned page |
| `desktop-map-cdp.py` | Inventory | no | Measured Chrome-tab and X11-window inventory only; not an accessibility tree |
| `active_brain.py` + `hemispheric.py` | Prototype | no | Legacy rule/prompt hints retained for compatibility; cannot grant tools, establish facts, or enforce model behavior |
| `integration_test.py` | Host smoke | no | Explicitly skips unavailable desktop backends; intended for an interactive deployment host, not generic CI |

Optional desktop dependencies include the active session environment plus some
combination of AT-SPI, Chrome CDP, `ydotool`, `grim`, X11 tools, Tesseract,
OpenCV, Pillow, and Playwright. Capability/status output reports what was
observed; it does not install or silently substitute a different host backend.

## Wiring plan

### Already wired (no action needed)
- Core runtime, gateway, Discord, HTTP, cron, memory, sleep, graceful degradation, remote relay
- Optional runtime wiring: autonomy, MCP client, A2A server, commerce API, active inference

### Prototype (documented as unsupported)

- MCP server, A2A client, `active_brain.py`, and `hemispheric.py`
  remain unwired and are not exposed as production capabilities.

Dead alternate memory stores and the unwired supervisor scaffold were removed.
The canonical SQLite/NumPy indexes, runtime task observation, capability probes,
and systemd lifecycle are the only authorities for those responsibilities.
Legacy memory consolidation/redaction/deduplication scripts and the assertion-
based event replay duplicate were also removed; `norax.sleep`,
`norax.memory.consolidator`, `scripts/memory_lint.py`, and
`norax.verify_event_chain` are the supported maintenance authorities.
