# Metrics

Prometheus text-exposition at `GET /metrics` (port 4101). Liveness at
`GET /healthz`. All series namespaced `norax_*` with bounded label
cardinality.

## Scrape config

Minimal Prometheus:

```yaml
scrape_configs:
  - job_name: norax
    scrape_interval: 15s
    scrape_timeout: 10s
    static_configs:
      - targets: ['localhost:4101']
        labels:
          host: localhost
          service: norax-ai
```

A ready-to-include fragment lives at [`ops/prometheus.norax.yml`](../ops/prometheus.norax.yml).

Remember: `http.bind` defaults to `127.0.0.1:4101`. To scrape from
another host, bind to the tailnet IP in `config/runtime.jsonc`.

## Series catalog

### Counters

| Metric | Labels | Meaning |
|---|---|---|
| `norax_ingress_messages_total` | `source`, `channel`, `trusted` | inbound envelopes |
| `norax_ingress_dropped_total` | `source`, `reason` | dropped at policy/rate-limit |
| `norax_brain_turns_total` | `decision` | turns by hot-path decision (`emit_reply`/`silent`/`defer`) |
| `norax_brain_errors_total` | `where` | exceptions in brain pipeline |
| `norax_gateway_requests_total` | `model`, `status` | LLM calls |
| `norax_gateway_tokens_in_total` | `model` | input tokens billed |
| `norax_gateway_tokens_out_total` | `model` | output tokens billed |
| `norax_tool_calls_total` | `name`, `ok` | tool invocations |
| `norax_outbound_sends_total` | `channel`, `ok` | replies dispatched |
| `norax_cron_fires_total` | `job` | cron-as-sensory fires |
| `norax_sleep_flush_runs_total` | `dry_run` | sleep-flush invocations |
| `norax_sleep_flush_candidates_total` | `route` | candidate distribution |

### Histograms (seconds)

Buckets: 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120.

| Metric | Labels | Meaning |
|---|---|---|
| `norax_brain_turn_seconds` | — | end-to-end turn latency |
| `norax_gateway_request_seconds` | `model` | LLM call latency |
| `norax_tool_seconds` | `name` | tool execution latency |

### Gauges

| Metric | Meaning |
|---|---|
| `norax_uptime_seconds` | seconds since runtime start |

## Useful PromQL

- p95 turn latency (5m):
  `histogram_quantile(0.95, sum by (le) (rate(norax_brain_turn_seconds_bucket[5m])))`
- replies per minute:
  `rate(norax_outbound_sends_total{ok="true"}[1m]) * 60`
- cost per hour (USD proxy — wire your own token→$ table):
  `sum by (model) (increase(norax_gateway_tokens_out_total[1h]))`
- error rate:
  `sum(rate(norax_brain_errors_total[5m])) / sum(rate(norax_brain_turns_total[5m]))`
- cron health:
  `rate(norax_cron_fires_total[1h])`

## Spans (OTel-compatible)

No OTel SDK dependency. The hash-chained `state/events.jsonl` is the
span sink:

```json
{
  "schema_version": "v1",
  "ts": "2026-04-23T15:35:12.481Z",
  "kind": "reply",
  "trace_id": "01JSYABCX...",
  "span_id": "01JSYABD1...",
  "parent_span_id": "01JSYABCX...",
  "prev_hash": "...",
  "hash": "...",
  "attrs": {
    "gen_ai.system": "gateway",
    "gen_ai.request.model": "ollama/llama3.2:3b",
    "gen_ai.response.model": "ollama/llama3.2:3b",
    "gen_ai.usage.input_tokens": 1523,
    "gen_ai.usage.output_tokens": 312
  },
  "payload": {"content_preview": "...", "usage": {...}}
}
```

Attribute keys follow the OTel `gen_ai.*` semantic convention. To ship
to an external collector, replay `events.jsonl` → OTLP in a small
sidecar (planned for a later phase; not shipped).

## What we deliberately don't expose

- message bodies (in log, scrubbed; **never** in metrics)
- secret values (scrubbed by `norax/safety/secrets.py` before
  event-log append)
- per-user metrics (cardinality risk; use logs for that)
