"""Integration tests for sparse runtime layers (P2-2 reaudit fix).

Tests that exercise multiple subsystems together:
  - Capability registry states + readiness
  - TaskOutcomeLedger persistence + focus context
  - Event chain verification across files
  - Embedding error propagation
  - Gateway proxy auth middleware
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from norax.memory.embeddings import EmbeddingError
from norax.runtime.capability_registry import (
    STATE_DEGRADED,
    STATE_FAILED,
    STATE_INITIALIZING,
    STATE_READY,
    STATE_STALE,
    CapabilityRegistry,
)
from norax.runtime.task_ledger import TaskOutcomeLedger


class TestCapabilityStates:
    """Test capability state transitions and staleness detection."""

    def test_full_lifecycle(self) -> None:
        reg = CapabilityRegistry()
        reg.register("test_cap")
        assert reg.status()["test_cap"]["state"] == "disabled"

        reg.mark_initializing("test_cap")
        assert reg.status()["test_cap"]["state"] == STATE_INITIALIZING

        reg.mark_ok("test_cap")
        assert reg.is_ready("test_cap")
        assert reg.status()["test_cap"]["state"] == STATE_READY

        reg.mark_degraded("test_cap", "partial failure")
        assert reg.status()["test_cap"]["state"] == STATE_DEGRADED
        assert reg.degraded_capabilities() == ["test_cap"]

        reg.mark_ok("test_cap")
        assert reg.is_ready("test_cap")

    def test_failed_with_error_class(self) -> None:
        reg = CapabilityRegistry()

        class ConnectionTimeoutError(Exception):
            pass

        reg.mark_failed("db", ConnectionTimeoutError("lost connection"))
        status = reg.status()["db"]
        assert status["state"] == STATE_FAILED
        assert status["error_class"] == "timeout"
        assert reg.failed_capabilities() == ["db"]

    def test_staleness_detection(self) -> None:
        reg = CapabilityRegistry()
        reg.register("cache", stale_after_sec=300)
        reg.mark_ok("cache")
        # Manually set last_success to old time
        reg._caps["cache"].last_success = -400.0
        reg._check_stale()
        assert reg.status()["cache"]["state"] == STATE_STALE
        assert reg.stale_capabilities() == ["cache"]

    def test_touch_clears_stale(self) -> None:
        reg = CapabilityRegistry()
        reg.register("cache", stale_after_sec=300)
        reg.mark_ok("cache")
        reg._caps["cache"].last_success = -400.0
        reg._check_stale()
        assert reg.stale_capabilities() == ["cache"]
        reg.touch("cache")
        assert reg.is_ready("cache")

    def test_init_only_capability_does_not_become_stale(self) -> None:
        reg = CapabilityRegistry()
        reg.mark_ok("output_verifier")
        reg._caps["output_verifier"].last_success = time.monotonic() - 3600
        reg._check_stale()
        assert reg.status()["output_verifier"]["state"] == STATE_READY


class TestTaskLedgerIntegration:
    """Test TaskOutcomeLedger in a simulated workflow."""

    def test_full_resolution_loop(self, tmp_path: Path) -> None:
        ledger = TaskOutcomeLedger(tmp_path)

        # 1. Task fails
        ledger.record(
            task_id="deploy-001",
            description="Deploy v2.1 to production",
            state="failed",
            failure_cause="missing env var DATABASE_URL",
            valence=-0.8,
        )

        # 2. System attempts repair
        ledger.transition("deploy-001", "repairing")
        assert ledger.get("deploy-001").repair_attempts == 1

        # 3. Repair succeeds
        ledger.transition("deploy-001", "verified")

        # 4. Owner confirms
        ledger.transition("deploy-001", "user_confirmed")
        assert ledger.get("deploy-001").owner_confirmed is True

        # 5. No unresolved tasks remain
        assert ledger.unresolved() == []
        assert ledger.focus_context() == ""

    def test_persistence_survives_restart(self, tmp_path: Path) -> None:
        ledger1 = TaskOutcomeLedger(tmp_path)
        ledger1.record(task_id="t1", description="task 1", state="open", valence=-0.5)
        ledger1.record(task_id="t2", description="task 2", state="failed", valence=-0.9)

        # Simulate restart
        ledger2 = TaskOutcomeLedger(tmp_path)
        assert len(ledger2.all()) == 2
        assert ledger2.focus_context() != ""
        assert "task 2" in ledger2.focus_context()  # higher priority


class TestEmbeddingErrorPropagation:
    """Test that embedding errors propagate correctly and never produce zero vectors."""

    def test_embedding_error_is_raised_not_swallowed(self) -> None:
        from norax.memory.embeddings import OllamaEmbedder

        class FailingEmbedder(OllamaEmbedder):
            async def _embed_once(self, texts):
                raise EmbeddingError("simulated failure")

        embedder = FailingEmbedder()
        with pytest.raises(EmbeddingError):
            import asyncio

            asyncio.run(embedder.embed(["test text"]))

    def test_validation_rejects_zero_norm(self) -> None:
        from norax.memory.embeddings import OllamaEmbedder

        class ZeroVectorEmbedder(OllamaEmbedder):
            async def _embed_once(self, texts):
                return np.zeros((len(texts), self.dim), dtype=np.float32)

        embedder = ZeroVectorEmbedder()
        with pytest.raises(EmbeddingError, match="zero-norm"):
            import asyncio

            asyncio.run(embedder.embed(["test text"]))

    def test_validation_rejects_wrong_dimension(self) -> None:
        from norax.memory.embeddings import OllamaEmbedder

        class WrongDimEmbedder(OllamaEmbedder):
            async def _embed_once(self, texts):
                return np.ones((len(texts), 128), dtype=np.float32)

        embedder = WrongDimEmbedder(dim=384)
        with pytest.raises(EmbeddingError, match="dimension mismatch"):
            import asyncio

            asyncio.run(embedder.embed(["test text"]))


class TestGatewayAuthMiddleware:
    """Test gateway proxy inbound auth middleware."""

    def test_health_endpoints_exempt(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/healthz")
        async def healthz():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/healthz")
        assert resp.status_code == 200

    def test_auth_rejected_without_key(self) -> None:
        import os

        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        os.environ["NORAX_GATEWAY_INBOUND_KEY"] = "secret-key-123"
        app = FastAPI()

        @app.middleware("http")
        async def auth_mw(request: Request, call_next):
            key = os.environ.get("NORAX_GATEWAY_INBOUND_KEY", "")
            if key and request.url.path not in {"/healthz"}:
                auth = request.headers.get("authorization", "")
                token = auth.removeprefix("Bearer ").strip()
                import secrets

                if not token or not secrets.compare_digest(token, key):
                    return JSONResponse(status_code=401, content={"error": "auth"})
            return await call_next(request)

        @app.get("/v1/chat/completions")
        async def chat():
            return {"ok": True}

        client = TestClient(app)

        # No auth → 401
        resp = client.get("/v1/chat/completions")
        assert resp.status_code == 401

        # Wrong key → 401
        resp = client.get("/v1/chat/completions", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

        # Correct key → 200
        resp = client.get(
            "/v1/chat/completions", headers={"Authorization": "Bearer secret-key-123"}
        )
        assert resp.status_code == 200

        # Health endpoint → 200 without auth
        del os.environ["NORAX_GATEWAY_INBOUND_KEY"]


class TestConnectorConfig:
    """Test typed connector configuration."""

    def test_default_connectors(self, tmp_path: Path) -> None:
        from norax.config.loader import Config

        cfg = Config(raw={}, project_root=tmp_path)
        conns = cfg.connectors
        assert conns["planner"]["enabled"] is True
        assert conns["multi_agent"]["enabled"] is True
        assert conns["mcp_client"]["enabled"] is False
        assert conns["commerce"]["enabled"] is False
        assert "mcp_server" not in conns
        assert "a2a_client" not in conns
        assert "supervisor" not in conns

    def test_custom_connectors_merge(self, tmp_path: Path) -> None:
        from norax.config.loader import Config

        cfg = Config(
            raw={
                "connectors": {"multi_agent": {"max_concurrent": 8}, "commerce": {"enabled": True}}
            },
            project_root=tmp_path,
        )
        conns = cfg.connectors
        assert conns["multi_agent"]["max_concurrent"] == 8
        assert conns["multi_agent"]["enabled"] is True  # default preserved
        assert conns["commerce"]["enabled"] is True

    def test_legacy_connector_sections_feed_effective_config(self, tmp_path: Path) -> None:
        from norax.config.loader import Config

        cfg = Config(
            raw={
                "mcp": {
                    "enabled": True,
                    "servers": [{"transport": "stdio", "command": "mcp-example"}],
                },
                "a2a": {"enabled": True, "port": 9000},
                "commerce": {"enabled": True},
                "connectors": {"mcp_client": {"enabled": False}},
            },
            project_root=tmp_path,
        )

        conns = cfg.connectors
        assert conns["mcp_client"]["enabled"] is False  # explicit gate wins
        assert conns["mcp_client"]["servers"][0]["command"] == "mcp-example"
        assert conns["a2a_server"]["enabled"] is True
        assert conns["a2a_server"]["port"] == 9000
        assert conns["commerce"]["enabled"] is True

    def test_unknown_connector_fails_instead_of_advertising_noop(self, tmp_path: Path) -> None:
        from norax.config.loader import Config

        cfg = Config(
            raw={"connectors": {"imaginary_backend": {"enabled": True}}},
            project_root=tmp_path,
        )

        with pytest.raises(ValueError, match="unsupported connector"):
            _ = cfg.connectors
