from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import respx
from httpx import ASGITransport, AsyncClient

from norax.adapter.http_in import HttpInAdapter
from norax.config.provider_store import ProviderStore, validate_provider_spec
from norax.gateway_client import GatewayRequest, GatewayRouter
from norax.runtime.core import Runtime


def test_provider_store_separates_public_metadata_and_secret(tmp_path: Path):
    store = ProviderStore(tmp_path)
    saved = store.upsert(
        {
            "name": "example",
            "base_url": "https://api.example.com/v1",
            "provider_kind": "openai",
            "models": ["model-a"],
        },
        "private-key-value",
    )
    assert saved["name"] == "example"
    assert store.list_public()[0]["models"] == ["model-a"]
    assert "api_key" not in store.list_public()[0]
    assert store.list_runtime()[0]["api_key"] == "private-key-value"
    assert "private-key-value" not in store.providers_path.read_text()
    assert "private-key-value" in store.secrets_path.read_text()
    assert store.providers_path.stat().st_mode & 0o777 == 0o600
    assert store.secrets_path.stat().st_mode & 0o777 == 0o600


def test_provider_store_update_without_key_preserves_secret(tmp_path: Path):
    store = ProviderStore(tmp_path)
    base = {
        "name": "example",
        "base_url": "https://api.example.com/v1",
        "provider_kind": "openai",
        "models": ["model-a"],
    }
    store.upsert(base, "private-key-value")
    store.upsert({**base, "models": ["model-b"]})
    assert store.list_runtime()[0]["api_key"] == "private-key-value"
    assert store.list_public()[0]["models"] == ["model-b"]


def test_provider_store_remove_cleans_metadata_and_secret(tmp_path: Path):
    store = ProviderStore(tmp_path)
    store.upsert(
        {
            "name": "example",
            "base_url": "https://api.example.com/v1",
            "models": ["model-a"],
        },
        "private-key-value",
    )
    assert store.remove("example") is True
    assert store.list_public() == []
    assert store.list_runtime() == []


def test_provider_validation_rejects_insecure_remote_http():
    with pytest.raises(ValueError, match="must use HTTPS"):
        validate_provider_spec(
            {"name": "example", "base_url": "http://api.example.com/v1", "models": ["m"]}
        )


def test_provider_validation_parses_enabled_strictly_and_rejects_bad_model_shapes():
    base = {
        "name": "example",
        "base_url": "https://api.example.com/v1",
        "models": ["model-a"],
    }
    assert validate_provider_spec({**base, "enabled": "false"})["enabled"] is False
    with pytest.raises(ValueError, match="models must be a list"):
        validate_provider_spec({**base, "models": "model-a"})
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        validate_provider_spec({**base, "enabled": 1})
    with pytest.raises(ValueError, match="at most 100"):
        validate_provider_spec({**base, "models": [f"model-{i}" for i in range(101)]})


def test_provider_validation_rejects_invalid_ports_and_control_characters():
    with pytest.raises(ValueError, match="invalid port"):
        validate_provider_spec({"name": "example", "base_url": "https://api.example.com:99999/v1"})
    with pytest.raises(ValueError, match="model ids"):
        validate_provider_spec(
            {
                "name": "example",
                "base_url": "https://api.example.com/v1",
                "models": ["bad model"],
            }
        )


def test_provider_store_does_not_overwrite_corrupt_documents(tmp_path: Path):
    store = ProviderStore(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    store.providers_path.write_text("{broken", encoding="utf-8")
    original = store.providers_path.read_bytes()

    with pytest.raises(ValueError, match="cannot safely update"):
        store.upsert(
            {
                "name": "example",
                "base_url": "https://api.example.com/v1",
                "models": ["model-a"],
            }
        )

    assert store.providers_path.read_bytes() == original


def test_provider_store_filters_invalid_persisted_rows(tmp_path: Path):
    store = ProviderStore(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    store.providers_path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "valid": {
                        "base_url": "https://api.example.com/v1",
                        "provider_kind": "openai",
                        "models": ["model-a"],
                        "enabled": "false",
                    },
                    "broken": "not-an-object",
                },
            }
        ),
        encoding="utf-8",
    )

    assert store.list_public() == [
        {
            "name": "valid",
            "base_url": "https://api.example.com/v1",
            "provider_kind": "openai",
            "models": ["model-a"],
            "enabled": False,
        }
    ]


def test_provider_store_serializes_whole_transactions_across_instances(tmp_path: Path):
    first = ProviderStore(tmp_path)
    second = ProviderStore(tmp_path)
    entered_write = threading.Event()
    release_write = threading.Event()
    original_write = first._write

    def delayed_write(path, value):
        if path == first.providers_path and not entered_write.is_set():
            entered_write.set()
            assert release_write.wait(timeout=2)
        return original_write(path, value)

    first._write = delayed_write  # type: ignore[method-assign]
    base = {
        "base_url": "https://api.example.com/v1",
        "provider_kind": "openai",
        "models": ["model-a"],
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(first.upsert, {**base, "name": "first"})
        assert entered_write.wait(timeout=2)
        second_future = pool.submit(second.upsert, {**base, "name": "second"})
        time.sleep(0.05)
        assert second_future.done() is False
        release_write.set()
        first_future.result(timeout=2)
        second_future.result(timeout=2)

    assert [row["name"] for row in first.list_public()] == ["first", "second"]
    assert first.lock_path.stat().st_mode & 0o777 == 0o600


class _FakeRuntime:
    def __init__(self, root: Path) -> None:
        self._provider_store = ProviderStore(root)
        self._providers: dict[str, dict] = {}

    def custom_model_catalog(self):
        return {
            item["name"]: [(f"{item['name']}/{model}", model) for model in item.get("models") or []]
            for item in self._provider_store.list_public()
        }

    async def upsert_custom_provider(self, spec):
        spec = dict(spec)
        spec["models"] = ["model-a", "model-b"]
        api_key = spec.pop("api_key", None)
        clean = validate_provider_spec(spec)
        self._provider_store.upsert(clean, api_key)
        return clean

    async def remove_custom_provider(self, name):
        return self._provider_store.remove(name)


class _FakeGateway:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.model_prefix = None
        self.closed = False

    async def aclose(self):
        self.closed = True


class _Events:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.items: list[tuple[str, dict]] = []

    async def append(self, kind: str, payload: dict):
        if self.fail:
            raise OSError("event log unavailable")
        self.items.append((kind, payload))


def _provider_runtime(tmp_path: Path, gateway, *, events: _Events | None = None) -> Runtime:
    runtime = Runtime.__new__(Runtime)
    runtime._provider_store = ProviderStore(tmp_path)
    runtime._base_provider_names = {"default"}
    runtime._provider_mutation_lock = asyncio.Lock()
    runtime._gateway_timeout_seconds = 12.0
    runtime.gateway = gateway
    runtime.default_model = "default/model"
    runtime.failover_models = []
    runtime.cfg = SimpleNamespace(raw={"vision": {"model": "default/vision"}})
    runtime._vision_config = {"enabled": True, "model": "default/vision"}
    runtime.events = events or _Events()
    runtime.metrics = None
    return runtime


@pytest.mark.asyncio
async def test_gateway_router_hot_swaps_custom_provider():
    default = _FakeGateway("http://default")
    router = GatewayRouter({"default": default}, [], "default")
    custom = _FakeGateway("https://api.example.com/v1")
    await router.upsert_provider("example", custom)
    assert router.route_for("example/model-a") == ("example", "https://api.example.com/v1")
    assert router.route_for("other") == ("default", "http://default")
    assert await router.remove_provider("example") is True
    assert custom.closed is True
    assert router.route_for("example/model-a") == ("default", "http://default")


@pytest.mark.asyncio
async def test_gateway_router_defers_retired_client_close_until_request_finishes():
    class BlockingGateway(_FakeGateway):
        def __init__(self, base_url: str) -> None:
            super().__init__(base_url)
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def chat(self, request):
            self.started.set()
            await self.release.wait()
            return SimpleNamespace(content="ok", model=request.model)

    old = BlockingGateway("http://old")
    router = GatewayRouter(
        {"default": _FakeGateway("http://default"), "example": old},
        [("example/*", "example")],
        "default",
    )
    request = GatewayRequest(model="example/model-a", messages=[])
    task = asyncio.create_task(router.chat(request))
    await old.started.wait()

    replacement = _FakeGateway("http://replacement")
    await router.upsert_provider("example", replacement)
    assert old.closed is False

    old.release.set()
    assert (await task).content == "ok"
    assert old.closed is True
    await router.aclose()


@pytest.mark.asyncio
async def test_gateway_router_cleanup_failure_does_not_falsely_fail_committed_swap():
    class BadCloseGateway(_FakeGateway):
        async def aclose(self):
            raise OSError("close failed")

    router = GatewayRouter(
        {"default": _FakeGateway("http://default"), "example": BadCloseGateway("http://old")},
        [],
        "default",
    )
    replacement = _FakeGateway("http://replacement")

    await router.upsert_provider("example", replacement)

    assert router.route_for("example/model") == ("example", "http://replacement")
    await router.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_runtime_provider_update_is_live_persisted_and_event_failure_is_nonfatal(tmp_path):
    respx.get("https://api.example.com/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"id": "model-a"}, {"id": "model-a"}, {"id": "model-b"}]},
        )
    )
    router = GatewayRouter({"default": _FakeGateway("http://default")}, [], "default")
    runtime = _provider_runtime(tmp_path, router, events=_Events(fail=True))

    result = await runtime.upsert_custom_provider(
        {
            "name": "example",
            "base_url": "https://api.example.com/v1",
            "api_key": "secret",
        }
    )

    assert result["models"] == ["model-a", "model-b"]
    assert runtime._provider_store.list_runtime()[0]["api_key"] == "secret"
    assert router.route_for("example/model-a")[0] == "example"
    await router.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_runtime_provider_revalidates_discovered_model_ids_before_commit(tmp_path):
    respx.get("https://api.example.com/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "bad model"}]})
    )
    router = GatewayRouter({"default": _FakeGateway("http://default")}, [], "default")
    runtime = _provider_runtime(tmp_path, router)
    runtime.thinking_effort = "medium"

    with pytest.raises(ValueError, match="model ids"):
        await runtime.upsert_custom_provider(
            {"name": "example", "base_url": "https://api.example.com/v1"}
        )

    assert runtime._provider_store.list_public() == []
    assert router.has_provider("example") is False
    await router.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_runtime_provider_store_rolls_back_when_live_router_rejects_update(tmp_path):
    class RejectingRouter:
        provider_urls = {"default": "http://default"}
        default_provider = "default"

        def has_provider(self, name):
            return name == "default"

        async def upsert_provider(self, *_args, **_kwargs):
            raise RuntimeError("router closed")

    respx.get("https://api.example.com/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "model-a"}]})
    )
    runtime = _provider_runtime(tmp_path, RejectingRouter())

    with pytest.raises(RuntimeError, match="router closed"):
        await runtime.upsert_custom_provider(
            {"name": "example", "base_url": "https://api.example.com/v1"}
        )

    assert runtime._provider_store.list_public() == []


@pytest.mark.asyncio
async def test_runtime_can_remove_disabled_provider_that_is_not_in_live_router(tmp_path):
    router = GatewayRouter({"default": _FakeGateway("http://default")}, [], "default")
    runtime = _provider_runtime(tmp_path, router)
    runtime._provider_store.upsert(
        {
            "name": "example",
            "base_url": "https://api.example.com/v1",
            "models": ["model-a"],
            "enabled": False,
        },
        "secret",
    )

    assert await runtime.remove_custom_provider("example") is True
    assert runtime._provider_store.list_public() == []
    await router.aclose()


@pytest.mark.asyncio
async def test_runtime_refuses_to_disable_gateway_default_provider(tmp_path):
    router = GatewayRouter({"example": _FakeGateway("https://api.example.com/v1")}, [], "example")
    runtime = _provider_runtime(tmp_path, router)
    runtime._base_provider_names = set()
    runtime.default_model = "unprefixed-model"
    runtime._provider_store.upsert(
        {
            "name": "example",
            "base_url": "https://api.example.com/v1",
            "models": ["model-a"],
        }
    )

    with pytest.raises(ValueError, match="gateway default provider"):
        await runtime.upsert_custom_provider(
            {
                "name": "example",
                "base_url": "https://api.example.com/v1",
                "models": ["model-a"],
                "enabled": False,
            }
        )

    assert runtime._provider_store.list_public()[0]["enabled"] is True
    assert router.has_provider("example") is True
    await router.aclose()


@pytest.mark.asyncio
async def test_provider_api_auth_and_live_catalog(tmp_path: Path):
    adapter = HttpInAdapter(host="127.0.0.1", port=0)
    adapter.configure_owner_chat(owner_id="owner", chat_token="test-secret")
    adapter._runtime_ref = _FakeRuntime(tmp_path)
    headers = {"Authorization": "Bearer test-secret"}
    async with AsyncClient(
        transport=ASGITransport(app=adapter.app), base_url="http://test"
    ) as client:
        denied = await client.get("/api/providers")
        assert denied.status_code == 401
        added = await client.put(
            "/api/providers/example",
            headers=headers,
            json={
                "base_url": "https://api.example.com/v1",
                "api_key": "private-key-value",
            },
        )
        assert added.status_code == 200
        payload = added.json()
        assert payload["model_catalog"]["example"] == [
            ["example/model-a", "model-a"],
            ["example/model-b", "model-b"],
        ]
        assert "private-key-value" not in json.dumps(payload)
        listed = await client.get("/api/providers", headers=headers)
        assert listed.json()["providers"][0]["name"] == "example"
        removed = await client.delete("/api/providers/example", headers=headers)
        assert removed.status_code == 200


@pytest.mark.asyncio
async def test_gateway_router_aclose_closes_retired_clients_with_active_leases():
    """Shutdown must close clients even if they have in-flight leases.

    The runtime drains active turns before calling gateway.aclose(), but
    a stream may still be settling. Leaking connections is worse than
    interrupting a dying stream during process shutdown.
    """

    class BlockingGateway(_FakeGateway):
        def __init__(self, base_url: str) -> None:
            super().__init__(base_url)
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def chat(self, request):
            self.started.set()
            await self.release.wait()
            return SimpleNamespace(content="ok", model=request.model)

    old = BlockingGateway("http://old")
    router = GatewayRouter(
        {"default": _FakeGateway("http://default"), "example": old},
        [("example/*", "example")],
        "default",
    )
    request = GatewayRequest(model="example/model-a", messages=[])
    task = asyncio.create_task(router.chat(request))
    await old.started.wait()

    # Replace while in-flight — old goes to retired.
    replacement = _FakeGateway("http://replacement")
    await router.upsert_provider("example", replacement)
    assert old.closed is False

    # Shutdown while old still has an active lease.
    await router.aclose()
    # Old must be closed by shutdown, not leaked.
    assert old.closed is True
    assert replacement.closed is True

    # Clean up the task.
    old.release.set()
    try:
        await task
    except Exception:
        pass


@pytest.mark.asyncio
async def test_set_default_model_persists_off_event_loop(tmp_path: Path):
    """set_default_model must not block the event loop on fsync.

    The live state is updated synchronously; persistence is scheduled in
    a worker thread. The method returns True immediately.
    """
    router = GatewayRouter({"default": _FakeGateway("http://default")}, [], "default")
    runtime = _provider_runtime(tmp_path, router)
    runtime.thinking_effort = "medium"

    result = runtime.set_default_model("default/new-model")
    assert result is True
    assert runtime.default_model == "default/new-model"
    assert runtime._effective_model == "default/new-model"

    # Wait for the async persistence task to complete.
    await asyncio.sleep(0.1)
    assert runtime._provider_store.runtime_settings().get("default_model") == "default/new-model"
    await router.aclose()


@pytest.mark.asyncio
async def test_default_model_persistence_is_owned_coalesced_and_latest_wins(tmp_path: Path):
    router = GatewayRouter({"default": _FakeGateway("http://default")}, [], "default")
    runtime = _provider_runtime(tmp_path, router)
    runtime.thinking_effort = "medium"
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    class SlowStore:
        def set_runtime_setting(self, _key: str, value: str) -> None:
            calls.append(value)
            if len(calls) == 1:
                started.set()
                assert release.wait(timeout=2)

    runtime._provider_store = SlowStore()
    assert runtime.set_default_model("default/first") is True
    assert await asyncio.to_thread(started.wait, 1) is True
    assert runtime.set_default_model("default/second") is True
    assert runtime.set_default_model("default/latest") is True
    assert len(runtime._maintenance_tasks) == 1

    release.set()
    owner = runtime._default_model_persist_task
    assert owner is not None
    await asyncio.wait_for(owner, timeout=2)

    assert calls == ["default/first", "default/latest"]
    assert runtime._pending_default_model is None
    assert runtime._maintenance_tasks == set()
    await router.aclose()


@pytest.mark.asyncio
async def test_set_default_model_returns_false_without_provider_store(tmp_path: Path):
    """Without a provider store, persistence is impossible but live state still updates."""
    router = GatewayRouter({"default": _FakeGateway("http://default")}, [], "default")
    runtime = _provider_runtime(tmp_path, router)
    runtime._provider_store = None
    runtime.thinking_effort = "medium"

    result = runtime.set_default_model("default/fallback")
    assert result is False
    assert runtime.default_model == "default/fallback"
    await router.aclose()


@pytest.mark.asyncio
async def test_ingress_test_returns_received_not_accepted(tmp_path: Path):
    """The /ingress/test endpoint must say 'received', not 'accepted'.

    The adapter queue accepting the event does not mean the runtime turn
    queue will accept it. Honest semantics prevent callers from assuming
    their turn is guaranteed to run.
    """
    adapter = HttpInAdapter(host="127.0.0.1", port=0)
    adapter.configure_owner_chat(owner_id="owner", chat_token="test-secret")
    adapter._runtime_ref = _FakeRuntime(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=adapter.app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/ingress/test",
            json={"sender_id": "tester", "body": "hello"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["status"] == "received"
        assert body["status"] != "accepted"


@pytest.mark.asyncio
async def test_ingress_agent_os_returns_received_not_accepted(tmp_path: Path):
    """The /ingress/agent_os endpoint must say 'received', not 'accepted'."""
    adapter = HttpInAdapter(host="127.0.0.1", port=0)
    adapter.configure_owner_chat(owner_id="owner", chat_token="test-secret")
    adapter._runtime_ref = _FakeRuntime(tmp_path)
    headers = {"Authorization": "Bearer test-secret"}
    async with AsyncClient(
        transport=ASGITransport(app=adapter.app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/ingress/agent_os",
            headers=headers,
            json={"body": "hello", "thread_id": "t1"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["status"] == "received"
        assert body["status"] != "accepted"


@pytest.mark.asyncio
async def test_status_does_not_leak_config_hash(tmp_path: Path):
    """/status must not expose the internal config hash fingerprint."""
    adapter = HttpInAdapter(host="127.0.0.1", port=0)
    adapter.configure_owner_chat(owner_id="owner", chat_token="test-secret")

    class _StubRuntime:
        _config_hash = "secret-fingerprint-abc123"
        _effective_model = "default/model"
        _effective_provider = "default"
        _configured_provider = "default"

        class capabilities:
            @staticmethod
            def status():
                return None

            @staticmethod
            def failed_capabilities():
                return []

            @staticmethod
            def degraded_capabilities():
                return []

            @staticmethod
            def stale_capabilities():
                return []

    adapter._runtime_ref = _StubRuntime()
    async with AsyncClient(
        transport=ASGITransport(app=adapter.app), base_url="http://test"
    ) as client:
        resp = await client.get("/status")
        assert resp.status_code == 200
        body = resp.json()
        assert "config_hash" not in body
        assert body["effective_model"] == "default/model"
