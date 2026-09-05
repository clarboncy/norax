"""Norax metrics — Prometheus counters/histograms exposed at /metrics.

Design:
  - One `Metrics` instance, owned by `Runtime`, passed to anything that
    needs to record (brain hot_path, dispatch tools, gateway client,
    adapters).
  - All metrics live in a private `CollectorRegistry` so tests can spin
    up an isolated instance without polluting the global default.
  - Exposed via `mount_endpoints(app)` which adds `/metrics` (text
    exposition format) and `/healthz` (liveness JSON) to a FastAPI app.

Naming convention: `norax_<subsystem>_<thing>_<unit>`.
Histograms in seconds. Counters in events/totals.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Default histogram buckets tuned for human-scale conversational latency.
_LAT_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()

        # Inbound
        self.ingress_total = Counter(
            "norax_ingress_messages_total",
            "Inbound messages received from a sensory adapter.",
            ["source", "channel", "trusted"],
            registry=self.registry,
        )
        self.ingress_dropped = Counter(
            "norax_ingress_dropped_total",
            "Inbound messages dropped by gate/policy.",
            ["source", "reason"],
            registry=self.registry,
        )

        # Brain
        self.brain_turns = Counter(
            "norax_brain_turns_total",
            "Brain turns processed.",
            ["decision"],
            registry=self.registry,
        )
        self.brain_turn_seconds = Histogram(
            "norax_brain_turn_seconds",
            "End-to-end brain turn latency (ingress → reply ack).",
            buckets=_LAT_BUCKETS,
            registry=self.registry,
        )
        self.brain_errors = Counter(
            "norax_brain_errors_total",
            "Brain turn errors.",
            ["where"],
            registry=self.registry,
        )
        self.agent_turn_failures = Counter(
            "norax_agent_turn_failed_total",
            "User agent turns that ended without a successfully delivered response.",
            registry=self.registry,
        )
        self.prompt_cache_breaks = Counter(
            "norax_prompt_cache_breaks_total",
            "Static prompt prefix hash changed mid-session (cache invalidation).",
            ["channel"],
            registry=self.registry,
        )
        self.prompt_tools_changes = Counter(
            "norax_prompt_tools_changes_total",
            "Allowed tool set changed mid-session (cache invalidation).",
            ["channel"],
            registry=self.registry,
        )
        self.prompt_cache_breaks_window = Gauge(
            "norax_prompt_cache_breaks_window",
            "Static prompt prefix hash changes in the last 5 minutes (alert if >2).",
            ["channel"],
            registry=self.registry,
        )

        # Gateway / model
        self.gateway_requests = Counter(
            "norax_gateway_requests_total",
            "Requests sent to the LLM gateway.",
            ["model", "status"],
            registry=self.registry,
        )
        self.gateway_latency = Histogram(
            "norax_gateway_request_seconds",
            "Gateway request latency.",
            buckets=_LAT_BUCKETS,
            labelnames=["model"],
            registry=self.registry,
        )
        self.gateway_tokens_in = Counter(
            "norax_gateway_tokens_in_total",
            "Input tokens billed by the gateway.",
            ["model"],
            registry=self.registry,
        )
        self.gateway_tokens_out = Counter(
            "norax_gateway_tokens_out_total",
            "Output tokens billed by the gateway.",
            ["model"],
            registry=self.registry,
        )

        # Ollama enhanced wrapper
        self.ollama_requests = Counter(
            "norax_ollama_requests_total",
            "Ollama gateway requests (enhanced wrapper).",
            ["model", "status"],
            registry=self.registry,
        )
        self.ollama_latency = Histogram(
            "norax_ollama_seconds",
            "Ollama gateway request latency.",
            buckets=_LAT_BUCKETS,
            labelnames=["model"],
            registry=self.registry,
        )
        self.ollama_tools = Counter(
            "norax_ollama_tools_total",
            "Tool calls returned by Ollama models.",
            ["model"],
            registry=self.registry,
        )

        # Tools
        self.tool_calls = Counter(
            "norax_tool_calls_total",
            "Tool invocations.",
            ["name", "ok"],
            registry=self.registry,
        )
        self.tool_latency = Histogram(
            "norax_tool_seconds",
            "Tool execution latency.",
            buckets=_LAT_BUCKETS,
            labelnames=["name"],
            registry=self.registry,
        )

        # Outbound
        self.outbound_sends = Counter(
            "norax_outbound_sends_total",
            "Outbound replies dispatched.",
            ["channel", "ok"],
            registry=self.registry,
        )

        # Cron
        self.cron_fires = Counter(
            "norax_cron_fires_total",
            "Cron job fires.",
            ["job"],
            registry=self.registry,
        )

        # Sleep-flush
        self.sleep_flush_runs = Counter(
            "norax_sleep_flush_runs_total",
            "Sleep-flush invocations.",
            ["dry_run"],
            registry=self.registry,
        )
        self.sleep_flush_candidates = Counter(
            "norax_sleep_flush_candidates_total",
            "Candidates produced by sleep-flush.",
            ["route"],
            registry=self.registry,
        )

        # Process
        self.uptime_seconds = Gauge(
            "norax_uptime_seconds",
            "Time since runtime start.",
            registry=self.registry,
        )
        self._start_ts = time.time()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @contextmanager
    def time_brain_turn(self) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.brain_turn_seconds.observe(time.perf_counter() - t0)

    @contextmanager
    def time_tool(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.tool_latency.labels(name=name).observe(time.perf_counter() - t0)

    @contextmanager
    def time_gateway(self, model: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.gateway_latency.labels(model=model).observe(time.perf_counter() - t0)

    def record_uptime(self) -> None:
        self.uptime_seconds.set(time.time() - self._start_ts)

    # ------------------------------------------------------------------
    # Exposition
    # ------------------------------------------------------------------
    def render(self) -> tuple[bytes, str]:
        """Return (body, content_type) for an HTTP /metrics response."""
        self.record_uptime()
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


def mount_endpoints(app, metrics: Metrics) -> None:
    """Add /metrics and /healthz routes to a FastAPI app.

    Idempotent: re-mounting on the same app is a no-op (FastAPI will
    raise if duplicate; we guard by inspecting routes).
    """
    # Import locally to keep FastAPI optional at module import time.  Do not
    # use it as an endpoint annotation: FastAPI resolves annotations against
    # module globals while building OpenAPI.
    existing = {getattr(r, "path", None) for r in app.routes}

    if "/metrics" not in existing:

        @app.get("/metrics")
        async def metrics_endpoint():  # noqa: D401
            from fastapi import Response

            body, ctype = metrics.render()
            return Response(content=body, media_type=ctype)

    if "/healthz" not in existing:

        @app.get("/healthz")
        async def healthz() -> dict:  # noqa: D401
            metrics.record_uptime()
            return {
                "ok": True,
                "uptime_seconds": round(time.time() - metrics._start_ts, 2),
            }
