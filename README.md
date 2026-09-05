<div align="center">

# Norax

**A self-hosted AI agent runtime built to preserve state, verify tool work, and resume.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB.svg)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/clarboncy/norax?style=social)](https://github.com/clarboncy/norax)

[Quick Start](#quick-start) · [Why Norax](#why-norax) · [Features](#features) · [Architecture](#architecture) · [Acknowledgments](#acknowledgments)

</div>

---

## The problem

Every AI agent demo works for 5 minutes. Then reality hits:

- The agent **forgets** what it was doing after a few tool calls
- It **loops** — repeating the same failed action or spinning in text without calling tools
- It **lies** — claims success without verifying anything actually worked
- When you restart it, the task is **gone** — no way to resume
- Your memory, personality, and history live on **someone else's server**

Norax addresses these failure modes with durable state, bounded tool loops,
explicit mutation verification, and tamper-evident operational evidence.

## Why Norax

Norax is a self-hosted AI agent runtime built for **long-running, real work**. It runs on your machine, uses configured local or remote models, and stores its runtime memory on your disk. Checkpointed work can be resumed when the underlying state is intact; model output and external side effects remain fallible.

```bash
git clone https://github.com/clarboncy/norax.git
cd norax
bash setup.sh
```

That's it. Norax is live at `http://127.0.0.1:8822` with a full chat dashboard, memory system, and tool loop.

## Features

| Feature | What it means |
|:--------|:-------------|
| **Durable task state** | Checkpointed goals, constraints, facts, blockers, and verification status are designed to survive compaction, model swaps, and restarts. |
| **7-layer memory** | Hot state, semantic facts, procedural skills, episodic records, entity graph, temporal graph, and causal links. Persists across sessions. |
| **Loop resistance** | Hard round ceilings, tool-call caps, duplicate detection, no-progress detection, and wall-clock deadlines bound each agent run. |
| **Verified execution** | Tool receipts plus target-aware readback checks gate mutation success claims. Some external outcomes cannot be proven locally and remain explicitly unverified. |
| **Hash-chained evidence** | Operational events form a verifiable hash chain. This is tamper-evident logging, not immutable storage. |
| **Dynamic model discovery** | Model-list requests query Ollama through a bounded, 30-second cache; newly pulled models do not require a runtime restart. |
| **Multi-provider routing** | Route across Ollama, OpenRouter, and any OpenAI-compatible API. Add providers at runtime without editing config. |
| **Web dashboard** | Chat with Norax in your browser before configuring Discord. Markdown rendering, streaming, stop button, model selector. |
| **Multi-channel** | Discord, HTTP, and cron ingress. One runtime, multiple interfaces. |
| **Owner-controlled safety** | Authenticated sender tiers, risk-gated tools, secret redaction, and filesystem confinement provide operator-controlled boundaries. |
| **MCP support** | Connect Model Context Protocol servers for extended tool access. |
| **Remote nodes** | Optional remote relay for multi-machine setups. |

## How it compares

| Capability | Hosted agents | Coding agents | OpenClaw | **Norax** |
|:-----------|:-------------:|:-------------:|:--------:|:---------:|
| Self-hosted | — | — | ✓ | ✓ |
| Owner controls all data | — | — | ✓ | ✓ |
| Durable task state across restarts | — | Partial | — | ✓ |
| 7-layer memory system | — | — | Partial | ✓ |
| Target-aware mutation verification | — | — | — | ✓ |
| Loop prevention (round/tool/time ceilings) | — | Partial | Partial | ✓ |
| Tamper-evident hash-chained event log | — | — | — | ✓ |
| Dynamic local model discovery | — | — | — | ✓ |
| Add API providers at runtime | — | — | ✓ | ✓ |
| Web dashboard | — | Partial | ✓ | ✓ |
| Multi-channel (Discord, HTTP, cron) | — | — | ✓ | ✓ |
| Fully open source (Apache 2.0) | — | — | ✓ | ✓ |
| No vendor lock-in | — | — | ✓ | ✓ |

## From inspiration to independence

Norax stands on the shoulders of the open-source agent community. The project that most directly shaped our thinking is [OpenClaw](https://github.com/openclaw/openclaw) — Peter Steinberger's open-source personal AI assistant that proved a self-hosted, multi-channel agent could reach hundreds of thousands of users without a walled garden.

OpenClaw's core insight — *your context and skills live on your computer, not someone else's server* — is the foundation Norax builds on. We owe credit to OpenClaw and its community for demonstrating that an open, owner-controlled agent is not just possible but preferable.

**Where Norax diverges:**

| Dimension | OpenClaw | Norax |
|:----------|:---------|:------|
| Language | TypeScript/Node.js | Python 3.12+ |
| Architecture | Gateway + channel plugins | Event-sourced runtime with hot-path brain |
| Memory | Session-based with skills | 7-layer: hot state, semantic, procedural, episodic, entity graph, temporal graph, causal links |
| Context continuity | Session-scoped | Durable task state across compaction, restarts, and model swaps |
| Loop prevention | Tool timeouts | Hard round ceilings, duplicate detection, no-progress detection, wall-clock deadlines |
| Verification | Agent-reported | Tool receipts + target-aware readback gates |
| Evidence | Logs | Tamper-evident hash chain + Prometheus metrics |
| Model routing | Provider configs | Dynamic Ollama discovery + runtime provider API |
| Deployment | npm/pnpm workspace | `uv` + systemd user services |
| License | Other | Apache 2.0 |

Norax is not a fork of OpenClaw. It is a ground-up Python implementation that takes the *philosophy* of owner-controlled agents and pushes it further in three directions:

1. **Reliability over demos.** Every architectural decision optimizes for what happens after the first successful turn — compaction, model failure, tool loops, partial completion, and restart recovery.
2. **Evidence over claims.** Norax does not treat the agent's own success report as proof. Tool receipts and target-aware checks gate mutation claims; outcomes that cannot be observed remain marked unverified.
3. **Memory as infrastructure.** Memory is not a conversation log. It's a 7-layer system with semantic search, procedural skills, episodic records, entity relationships, temporal graphs, causal links, and offline consolidation.

## Quick start

### Guided setup (recommended)

```bash
git clone https://github.com/clarboncy/norax.git
cd norax
bash setup.sh
```

`setup.sh` walks you through owner identity, generates a secure chat token, checks for Ollama, optionally adds API providers and Discord, then starts the runtime and web dashboard.

### Manual setup

```bash
git clone https://github.com/clarboncy/norax.git
cd norax
cp .env.example .env  # Edit .env — set NORAX_AGENT_OS_CHAT_TOKEN and NORAX_OWNER_ID
uv sync --all-extras
uv run --env-file .env python -m norax           # runtime on :4101
uv run --env-file .env python agent_os/server.py  # dashboard on :8822
```

Open `http://127.0.0.1:8822` and paste your chat token. No Discord or external account required.

### Talk to Norax before Discord

The local web dashboard lets you chat with the full agent — memory, tools, personality — in your browser:

1. Set `NORAX_AGENT_OS_CHAT_TOKEN` to any random string in `.env`
2. Start the runtime: `uv run --env-file .env python -m norax`
3. Start the dashboard: `uv run --env-file .env python agent_os/server.py`
4. Open `http://127.0.0.1:8822`

## Architecture

```text
Discord / HTTP / Cron
          │
          ▼
  normalized envelope
          │
          ▼
 brain hot path (L0-L10)
          │
          ├──► prompt assembly
          ├──► memory retrieval
          ├──► policy and risk gates
          │
          ▼
 bounded agent/tool loop
          │
          ├──► task-state checkpoint
          ├──► mutation receipts and readback verification
          ├──► event + metrics evidence
          │
          ▼
      outbound reply
```

```text
norax/
├── adapter/          # Discord, HTTP, and cron ingress
├── brain/            # hot path, agent loop, planning, verification
├── context/          # rolling context and compaction
├── dispatch/         # tools, policy, sandboxing, and risk gates
├── envelope/         # normalized wire shapes
├── gateway_client/   # provider clients, routing, and budgets
├── memory/           # retrieval, indexing, graphs, and consolidation
├── observability/    # structured logs, metrics, and event evidence
├── prompt/           # deterministic prompt assembly
├── runtime/          # lifecycle, checkpoints, supervision, and recovery
├── safety/           # secret handling and safety boundaries
└── soul/             # configurable identity and operating principles
```

## Configuration

Runtime settings live in [`config/runtime.jsonc`](config/runtime.jsonc). Secrets belong in `.env` (gitignored).

| Variable | Purpose |
|:---------|:--------|
| `NORAX_OWNER_ID` | Authoritative owner identity |
| `NORAX_OWNER_LABEL` | Owner display name |
| `NORAX_OWNER_TIMEZONE` | Owner timezone |
| `NORAX_AGENT_OS_CHAT_TOKEN` | Shared secret for the dashboard chat bridge |
| `NORAX_DISCORD_TOKEN` | Discord bot token (optional) |
| `NORAX_DISCORD_ENABLED` | Enables the Discord adapter |
| `NORAX_OLLAMA_URL` | Ollama endpoint (default: `http://127.0.0.1:11434`) |
| `NORAX_OPENROUTER_TOKEN` | OpenRouter API token (optional) |
| `NORAX_MEMORY_ROOT` | Mutable memory directory |
| `NORAX_STATE_DIR` | Events and operational state directory |
| `NORAX_LOG_DIR` | Structured log directory |

See [`.env.example`](.env.example) for a complete template.

### Dynamic model discovery

Norax queries the local Ollama daemon when building the live model list and caches the result for 30 seconds. Pulling a model does not require a runtime restart; refresh the model list after the cache interval.

### Add providers at runtime

```bash
curl --request PUT http://127.0.0.1:4101/api/providers/example \
  --header "Authorization: Bearer $NORAX_AGENT_OS_CHAT_TOKEN" \
  --header "Content-Type: application/json" \
  --data '{
    "base_url": "https://api.example.com/v1",
    "api_key": "replace-me",
    "provider_kind": "openai"
  }'
```

Keys are stored separately under `~/.config/norax/` with mode `0600`. Discovered models appear immediately in Discord `/model`, `/models`, and the dashboard.

## Development

```bash
uv sync --all-extras
uv run python scripts/quality_gate.py
```

The gate checks lockfile consistency, lint/format, the configured typecheck
surface, bytecode compilation, publication/secrets/memory policy, structural
end-to-end wiring, unit coverage, and acceptance tests. Host desktop smoke tests
remain separate because they require a real interactive session and optional
system dependencies.

## Deployment

User-level systemd deployment is documented in [`docs/deploy.md`](docs/deploy.md). Runtime data defaults to:

```text
~/.local/share/norax/memory/
~/.local/state/norax/
~/.local/state/norax/logs/
```

## Security

- Keep HTTP and dashboard on loopback unless token auth is configured
- Treat MCP servers and remote nodes as privileged extensions
- Review tool allowlists and filesystem roots before enabling external access
- Rotate any credential that has appeared in logs, shell history, or git history
- Report vulnerabilities privately via GitHub Security Advisories

## Documentation

- [`architecture/norax_core.md`](architecture/norax_core.md) — architecture and invariants
- [`docs/deploy.md`](docs/deploy.md) — systemd deployment and operations
- [`docs/metrics.md`](docs/metrics.md) — Prometheus metrics
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution and validation workflow
- [`SECURITY.md`](SECURITY.md) — vulnerability reporting

## Acknowledgments

Norax would not exist without the open-source agent community.

- **[OpenClaw](https://github.com/openclaw/openclaw)** — Peter Steinberger and the OpenClaw Foundation proved that a self-hosted, owner-controlled, multi-channel AI agent could reach hundreds of thousands of users. The core philosophy — *your context lives on your computer, not a walled garden* — is the foundation Norax builds on.
- **[soul.md](https://soul.md)** — The concept of a configurable agent identity file originated in the OpenClaw community. Norax adopts and extends this pattern.
- **The agent research community** — Work on long-horizon reliability, context engineering, behavioral state decay, and memory-augmented agents directly shaped Norax's architecture.

Norax is a ground-up implementation, not a fork. It combines durable task state, target-aware verification, layered memory, and tamper-evident event chaining, while treating every model inference and external side effect as fallible.

## License

Copyright 2026 Colby Caron. Licensed under the [Apache License 2.0](LICENSE).
