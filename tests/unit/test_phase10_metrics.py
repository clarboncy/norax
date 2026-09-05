"""Phase 10 — Metrics + /metrics + /healthz.

Verifies:
  - Metrics class creates an isolated registry.
  - Histograms/counters record correctly.
  - `mount_endpoints` adds /metrics and /healthz to a FastAPI app.
  - Re-mounting is idempotent.
  - Runtime.build wires a Metrics instance onto HttpInAdapter.
  - Live /metrics endpoint returns text exposition format with our
    metric names present.
  - Runtime._handle_turn increments ingress_total and brain_turns.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from norax.envelope import Principal, SensoryInput
from norax.observability.metrics import Metrics, mount_endpoints


class TestMetricsCore:
    def test_isolated_registry(self):
        m1 = Metrics()
        m2 = Metrics()
        assert m1.registry is not m2.registry

    def test_counter_inc(self):
        m = Metrics()
        m.ingress_total.labels(source="discord", channel="chat", trusted="true").inc()
        body, ctype = m.render()
        text = body.decode("utf-8")
        assert "norax_ingress_messages_total" in text
        assert 'source="discord"' in text
        assert "text/plain" in ctype

    def test_histogram_records_brain_turn(self):
        m = Metrics()
        with m.time_brain_turn():
            pass
        body, _ = m.render()
        text = body.decode("utf-8")
        assert "norax_brain_turn_seconds_bucket" in text
        assert "norax_brain_turn_seconds_count" in text

    def test_histogram_records_tool(self):
        m = Metrics()
        with m.time_tool("memory_search"):
            pass
        body, _ = m.render()
        assert b"norax_tool_seconds_bucket" in body

    def test_tokens_counters(self):
        m = Metrics()
        m.gateway_tokens_in.labels(model="x").inc(100)
        m.gateway_tokens_out.labels(model="x").inc(50)
        text = m.render()[0].decode("utf-8")
        assert "norax_gateway_tokens_in_total" in text
        assert 'model="x"' in text

    def test_agent_turn_failure_counter(self):
        m = Metrics()
        m.agent_turn_failures.inc()
        text = m.render()[0].decode("utf-8")
        assert "norax_agent_turn_failed_total 1.0" in text

    def test_uptime_updates(self):
        m = Metrics()
        m.record_uptime()
        text = m.render()[0].decode("utf-8")
        assert "norax_uptime_seconds" in text


class TestHTTPEndpoints:
    def test_mount_endpoints_adds_routes(self):
        app = FastAPI()
        m = Metrics()
        mount_endpoints(app, m)
        paths = {getattr(r, "path", None) for r in app.routes}
        assert "/metrics" in paths
        assert "/healthz" in paths

    def test_remount_is_idempotent(self):
        app = FastAPI()
        m = Metrics()
        mount_endpoints(app, m)
        mount_endpoints(app, m)  # must not raise
        paths = [getattr(r, "path", None) for r in app.routes]
        assert paths.count("/metrics") == 1
        assert paths.count("/healthz") == 1

    def test_metrics_endpoint_response(self):
        app = FastAPI()
        m = Metrics()
        m.ingress_total.labels(source="http", channel="http", trusted="false").inc()
        mount_endpoints(app, m)
        client = TestClient(app)
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]
        assert "norax_ingress_messages_total" in r.text

    def test_healthz_endpoint(self):
        app = FastAPI()
        m = Metrics()
        mount_endpoints(app, m)
        client = TestClient(app)
        r = client.get("/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert "uptime_seconds" in body


class TestHttpAdapterMountsMetrics:
    def test_http_adapter_exposes_metrics_route(self):
        from norax.adapter.http_in import HttpInAdapter

        m = Metrics()
        a = HttpInAdapter(host="127.0.0.1", port=0, metrics=m)
        client = TestClient(a.app)
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "norax_ingress_messages_total" in r.text
        r = client.get("/healthz")
        assert r.status_code == 200

    def test_http_adapter_without_metrics_has_no_routes(self):
        from norax.adapter.http_in import HttpInAdapter

        a = HttpInAdapter(host="127.0.0.1", port=0)
        client = TestClient(a.app)
        r = client.get("/metrics")
        assert r.status_code == 404


class TestRuntimeInstrumentation:
    def test_build_wires_agent_os_bridge_on_runtime(self, tmp_path):
        from norax.config.loader import Config
        from norax.runtime.core import Runtime

        (tmp_path / "state").mkdir()
        (tmp_path / "logs").mkdir()
        cfg = Config(
            raw={
                "http": {"bind": "127.0.0.1:0"},
                "owner": {"id": "1"},
                "agent_os": {"chat_token": "test-secret"},
            },
            project_root=tmp_path,
        )

        rt = Runtime.build(cfg)

        assert rt.agent_os_bridge is not None
        assert rt.agent_os_bridge is rt.outbound.get("agent_os")

    @pytest.mark.asyncio
    async def test_handle_turn_increments_ingress_and_turns(self, tmp_path):
        from norax.config.loader import Config
        from norax.runtime.core import Runtime

        (tmp_path / "state").mkdir()
        (tmp_path / "logs").mkdir()
        cfg = Config(
            raw={"http": {"bind": "127.0.0.1:0"}, "owner": {"id": "1"}},
            project_root=tmp_path,
        )
        rt = Runtime.build(cfg)

        # Stub the brain call — we don't need a real gateway here.
        from norax.brain import hot_path

        class _FakeResp:
            content = "ok"
            model = "fake"
            request_id = "req-1"
            usage = {"input_tokens": 10, "output_tokens": 5}

        class _FakeCtx:
            decision = "silent"
            allowed_tools = []
            focus = SimpleNamespace(summary="")

        class _FakeRendered:
            static_hash = "h"

        async def fake_plan_turn(env, **kwargs):
            return _FakeCtx(), _FakeRendered()

        orig = hot_path.plan_turn
        hot_path.plan_turn = fake_plan_turn
        try:
            env = SensoryInput(
                channel="http",
                source="http",
                message_id="m1",
                timestamp=datetime.now(UTC),
                sender=Principal(id="u", label="U", trust=False, tier="user"),
                body="hi",
                trusted=False,
            )
            await rt._handle_turn(env)
        finally:
            hot_path.plan_turn = orig

        text = rt.metrics.render()[0].decode("utf-8")
        assert "norax_ingress_messages_total" in text
        # labelset from our fake env
        assert 'source="http"' in text
        # brain turn recorded
        assert 'norax_brain_turns_total{decision="silent"}' in text or 'decision="silent"' in text
        # histogram observed at least once
        assert "norax_brain_turn_seconds_count" in text

    @pytest.mark.asyncio
    async def test_brain_error_counter(self, tmp_path):
        from norax.brain import hot_path
        from norax.config.loader import Config
        from norax.runtime.core import Runtime

        (tmp_path / "state").mkdir()
        (tmp_path / "logs").mkdir()
        cfg = Config(
            raw={"http": {"bind": "127.0.0.1:0"}, "owner": {"id": "1"}},
            project_root=tmp_path,
        )
        rt = Runtime.build(cfg)

        async def boom(env, **kwargs):
            raise RuntimeError("boom")

        orig = hot_path.plan_turn
        hot_path.plan_turn = boom
        try:
            env = SensoryInput(
                channel="http",
                source="http",
                message_id="m1",
                timestamp=datetime.now(UTC),
                sender=Principal(id="u", label="U", trust=False, tier="user"),
                body="hi",
                trusted=False,
            )
            await rt._handle_turn(env)
        finally:
            hot_path.plan_turn = orig

        text = rt.metrics.render()[0].decode("utf-8")
        assert "norax_brain_errors_total" in text
        assert 'where="brain"' in text
        assert "norax_agent_turn_failed_total 1.0" in text
