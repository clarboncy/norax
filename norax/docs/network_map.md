# Norax Network Map — HTTP Handlers

Generated: 2026-05-25

## Ingress Adapters (`adapter/`)

| File | Port | Purpose | Routes |
|------|------|---------|--------|
| `http_in.py` | 4101 | Test ingress endpoint | `GET /status`, `POST /ingress/test` |
| `discord_in.py` | - | Discord bot gateway | WebSocket (discord.py) |
| `cron_in.py` | - | Scheduled task ingress | Internal queue |

## Gateway Layer

| File | Port | Purpose | Routes |
|------|------|---------|--------|
| Ollama | 11434 | Local/cloud model transport | Ollama + OpenAI compatibility APIs |
| `gateway_client/__init__.py` | - | Upstream client | N/A (client-side) |

## Observability

| File | Port | Purpose | Routes |
|------|------|---------|--------|
| `observability/metrics.py` | Dynamic | Prometheus metrics | `GET /metrics` |

## Remote/Relay

| File | Port | Purpose | Routes |
|------|------|---------|--------|
| `remote/relay.py` | TBD | Remote command relay | Internal RPC |
| `remote/client.py` | - | Remote client | N/A (client-side) |

## Dispatch

| File | Purpose |
|------|---------|
| `dispatch/tools.py` | Tool execution dispatcher |

## Summary

```
External → [Discord] [HTTP:4101] [Cron]
              ↓           ↓         ↓
           [Brain Hot-Path] → [Gateway Router]
                                   ↓
                              [Ollama :11434]
                              [Ollama :11434 / configured remote provider]
```
