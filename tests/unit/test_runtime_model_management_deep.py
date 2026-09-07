from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from norax.brain import agent_loop
from norax.runtime import model_management as mm
from norax.runtime.model_management import ModelManagementMixin

_PROVIDER = {
    "name": "example",
    "base_url": "https://api.example.test/v1",
    "provider_kind": "openai",
    "models": ["model-a"],
    "enabled": True,
}


class _Store:
    def __init__(self, *rows: dict[str, Any]) -> None:
        self.rows = {row["name"]: dict(row) for row in rows}
        self.upsert_calls: list[tuple[dict[str, Any], str | None]] = []
        self.remove_calls: list[str] = []
        self.setting_calls: list[tuple[str, str]] = []
        self.upsert_error: BaseException | None = None
        self.remove_error: BaseException | None = None
        self.remove_result: bool | None = None
        self.setting_error: BaseException | None = None

    def list_public(self) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in row.items() if key != "api_key"}
            for row in self.rows.values()
        ]

    def list_runtime(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.rows.values()]

    def upsert(self, spec: dict[str, Any], api_key: str | None = None) -> dict[str, Any]:
        self.upsert_calls.append((dict(spec), api_key))
        if self.upsert_error is not None:
            raise self.upsert_error
        previous = self.rows.get(spec["name"], {})
        row = dict(spec)
        row["api_key"] = previous.get("api_key", "") if api_key is None else api_key
        self.rows[spec["name"]] = row
        return dict(spec)

    def remove(self, name: str) -> bool:
        self.remove_calls.append(name)
        if self.remove_error is not None:
            raise self.remove_error
        if self.remove_result is not None:
            if self.remove_result:
                self.rows.pop(name, None)
            return self.remove_result
        return self.rows.pop(name, None) is not None

    def set_runtime_setting(self, name: str, value: str) -> None:
        self.setting_calls.append((name, value))
        if self.setting_error is not None:
            raise self.setting_error


class _Events:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.items: list[tuple[str, dict[str, Any]]] = []

    async def append(self, kind: str, payload: dict[str, Any]) -> None:
        if self.error is not None:
            raise self.error
        self.items.append((kind, payload))


class _Gateway:
    def __init__(self, *providers: str) -> None:
        self.providers = set(providers)
        self.default_provider = "default"
        self.remove_result = True
        self.remove_error: BaseException | None = None
        self.upsert_error: BaseException | None = None
        self.route_error: BaseException | None = None
        self.removed: list[str] = []
        self.upserts: list[tuple[str, Any, list[str]]] = []

    def has_provider(self, name: str) -> bool:
        return name in self.providers

    async def remove_provider(self, name: str) -> bool:
        self.removed.append(name)
        if self.remove_error is not None:
            raise self.remove_error
        if self.remove_result:
            self.providers.discard(name)
        return self.remove_result

    async def upsert_provider(self, name: str, client: Any, *, patterns: list[str]) -> None:
        self.upserts.append((name, client, patterns))
        if self.upsert_error is not None:
            raise self.upsert_error
        self.providers.add(name)

    def route_for(self, model: str) -> tuple[str, str]:
        if self.route_error is not None:
            raise self.route_error
        return model.partition("/")[0], "https://route.example.test"


class _DiscoveryClient:
    models: list[Any] = ["model-a"]
    discovery_error: BaseException | None = None
    instances: list[_DiscoveryClient] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.fetch_kwargs: dict[str, Any] | None = None
        type(self).instances.append(self)

    async def fetch_model_ids(self, **kwargs: Any) -> list[Any]:
        self.fetch_kwargs = kwargs
        if self.discovery_error is not None:
            raise self.discovery_error
        return list(self.models)

    async def aclose(self) -> None:
        self.closed = True


class _OllamaClient:
    instances: list[_OllamaClient] = []

    def __init__(self, inner: Any, *, metrics: Any) -> None:
        self.inner = inner
        self.metrics = metrics
        self.closed = False
        type(self).instances.append(self)

    async def aclose(self) -> None:
        self.closed = True
        await self.inner.aclose()


def _runtime(
    *,
    store: _Store | None = None,
    gateway: Any | None = None,
    events: _Events | None = None,
) -> ModelManagementMixin:
    runtime = ModelManagementMixin()
    runtime._provider_store = store if store is not None else _Store()
    runtime._base_provider_names = {"default"}
    runtime._provider_mutation_lock = asyncio.Lock()
    runtime._gateway_timeout_seconds = 12.0
    runtime._maintenance_tasks = set()
    runtime._draining = False
    runtime.gateway = gateway if gateway is not None else _Gateway("default")
    runtime.default_model = "default/model"
    runtime._effective_model = runtime.default_model
    runtime._effective_provider = "default"
    runtime.failover_models = []
    runtime._vision_config = {"model": "default/vision"}
    runtime.events = events if events is not None else _Events()
    runtime.metrics = object()
    runtime.thinking_effort = "medium"
    runtime.reasoning_output = False
    runtime._planner_enabled = True
    runtime.planning_mode = "direct"
    runtime.max_tool_rounds = 250
    runtime.memory_depth = "auto"
    runtime.weak_model_boost = "auto"
    runtime.stream_replies = True
    runtime.response_length = "balanced"
    runtime.tool_activity = "normal"
    runtime._default_model_persist_task = None
    runtime._pending_default_model = None
    return runtime


@pytest.fixture(autouse=True)
def _reset_discovery_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _DiscoveryClient.models = ["model-a"]
    _DiscoveryClient.discovery_error = None
    _DiscoveryClient.instances = []
    _OllamaClient.instances = []
    monkeypatch.setattr(mm, "GatewayClient", _DiscoveryClient)


@pytest.mark.asyncio
async def test_provider_mutation_owner_tracks_success_failure_and_cancellation(caplog) -> None:
    runtime = _runtime()
    del runtime._maintenance_tasks

    async def succeed() -> str:
        return "committed"

    assert await runtime._run_provider_mutation(succeed()) == "committed"
    await asyncio.sleep(0)
    assert runtime._maintenance_tasks == set()

    async def fail() -> None:
        raise RuntimeError("transaction failed")

    with caplog.at_level(logging.WARNING, logger="norax.runtime.core"):
        with pytest.raises(RuntimeError, match="transaction failed"):
            await runtime._run_provider_mutation(fail())
        await asyncio.sleep(0)
    assert "provider mutation failed: RuntimeError" in caplog.text
    assert runtime._maintenance_tasks == set()

    async def cancel_self() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await runtime._run_provider_mutation(cancel_self())
    await asyncio.sleep(0)
    assert runtime._maintenance_tasks == set()


def test_custom_catalog_gateway_detection_and_usage_paths() -> None:
    runtime = _runtime(
        store=_Store(
            {**_PROVIDER, "name": "enabled", "models": ["a", "b"]},
            {**_PROVIDER, "name": "empty", "models": []},
            {**_PROVIDER, "name": "disabled", "enabled": False},
        )
    )
    assert runtime.custom_model_catalog() == {
        "enabled": [("enabled/a", "a"), ("enabled/b", "b")],
        "empty": [],
    }
    runtime._provider_store = None
    assert runtime.custom_model_catalog() == {}

    runtime.gateway = _Gateway("present")
    assert runtime._gateway_has_provider("present") is True
    assert runtime._gateway_has_provider("missing") is False
    runtime.gateway = SimpleNamespace(provider_urls={"fallback": "https://example.test"})
    assert runtime._gateway_has_provider("fallback") is True
    runtime.gateway.provider_urls = []
    assert runtime._gateway_has_provider("fallback") is False

    runtime.gateway = _Gateway()
    runtime.default_model = "example/default"
    assert runtime._custom_provider_usage("example") == "default model"
    runtime.default_model = "default/model"
    runtime.failover_models = [None, "example/failover"]
    assert runtime._custom_provider_usage("example") == "failover model"
    runtime.failover_models = []
    runtime._vision_config = {"model": "example/vision"}
    assert runtime._custom_provider_usage("example") == "vision model"
    runtime._vision_config = {"model": None}
    runtime.gateway.default_provider = "example"
    assert runtime._custom_provider_usage("example") == "gateway default provider"
    runtime.gateway.default_provider = "default"
    assert runtime._custom_provider_usage("example") is None


@pytest.mark.asyncio
async def test_provider_event_failure_is_redacted_bounded_and_nonfatal(caplog) -> None:
    secret = "sk-" + "x" * 30
    runtime = _runtime(events=_Events(RuntimeError(secret + "z" * 1_000)))
    with caplog.at_level(logging.WARNING, logger="norax.runtime.core"):
        await runtime._append_provider_event("provider_updated", {"provider": "example"})
    assert secret not in caplog.text
    assert "<REDACTED:openai_key>" in caplog.text
    assert len(caplog.messages[-1]) < 600


@pytest.mark.asyncio
async def test_restore_and_discovery_use_exact_private_state_and_resource_bounds() -> None:
    store = _Store({**_PROVIDER, "api_key": "current"})
    runtime = _runtime(store=store)
    await runtime._restore_custom_provider("example", None)
    assert "example" not in store.rows

    previous = {**_PROVIDER, "models": ["old"], "api_key": "private-key"}
    await runtime._restore_custom_provider("example", previous)
    assert store.rows["example"]["models"] == ["old"]
    assert store.rows["example"]["api_key"] == "private-key"
    assert "api_key" not in store.upsert_calls[-1][0]

    client = _DiscoveryClient(base_url="https://example.test")
    assert await runtime._discover_custom_provider_models(client=client) == ["model-a"]
    assert client.fetch_kwargs == {
        "max_response_bytes": 2 * 1024 * 1024,
        "max_models": 100,
    }


@pytest.mark.asyncio
async def test_upsert_rejects_malformed_reserved_and_foreign_live_providers() -> None:
    runtime = _runtime()
    with pytest.raises(ValueError, match="must be an object"):
        await runtime._upsert_custom_provider([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported provider fields"):
        await runtime._upsert_custom_provider({**_PROVIDER, "unexpected": True})
    with pytest.raises(ValueError, match="reserved"):
        await runtime._upsert_custom_provider({**_PROVIDER, "name": "default"})

    runtime.gateway.providers.add("example")
    with pytest.raises(ValueError, match="active outside"):
        await runtime._upsert_custom_provider(_PROVIDER)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("default", "default model"),
        ("failover", "failover model"),
        ("vision", "vision model"),
    ],
)
async def test_disable_rejects_each_live_model_reference(reference: str, expected: str) -> None:
    previous = {**_PROVIDER, "api_key": "secret"}
    runtime = _runtime(store=_Store(previous), gateway=_Gateway("default", "example"))
    if reference == "default":
        runtime.default_model = "example/model-a"
    elif reference == "failover":
        runtime.failover_models = ["example/model-a"]
    else:
        runtime._vision_config = {"model": "example/model-a"}

    with pytest.raises(ValueError, match=expected):
        await runtime._upsert_custom_provider({**_PROVIDER, "enabled": False})


@pytest.mark.asyncio
async def test_disable_requires_removal_support_and_rolls_back_refusal_or_error() -> None:
    previous = {**_PROVIDER, "api_key": "preserved"}
    store = _Store(previous)
    runtime = _runtime(
        store=store,
        gateway=SimpleNamespace(
            provider_urls={"example": "https://api.example.test"},
            default_provider="default",
        ),
    )
    with pytest.raises(RuntimeError, match="does not support"):
        await runtime._upsert_custom_provider({**_PROVIDER, "enabled": False})
    assert store.rows["example"]["enabled"] is True

    for error in (None, OSError("router exploded")):
        store = _Store(previous)
        gateway = _Gateway("default", "example")
        gateway.remove_result = False
        gateway.remove_error = error
        runtime = _runtime(store=store, gateway=gateway)
        with pytest.raises((RuntimeError, OSError)):
            await runtime._upsert_custom_provider(
                {**_PROVIDER, "api_key": "replacement", "enabled": False}
            )
        assert store.rows["example"]["enabled"] is True
        assert store.rows["example"]["api_key"] == "preserved"


@pytest.mark.asyncio
async def test_disable_non_live_provider_commits_and_preserves_omitted_key() -> None:
    previous = {**_PROVIDER, "api_key": "preserved"}
    store = _Store(previous)
    events = _Events()
    runtime = _runtime(store=store, events=events)

    result = await runtime._upsert_custom_provider({**_PROVIDER, "enabled": False})

    assert result["enabled"] is False
    assert store.rows["example"]["api_key"] == "preserved"
    assert store.upsert_calls[-1][1] is None
    assert events.items == [("provider_disabled", {"provider": "example"})]


@pytest.mark.asyncio
async def test_enabled_provider_requires_dynamic_gateway_and_closes_empty_discovery() -> None:
    runtime = _runtime(gateway=SimpleNamespace(provider_urls={}, default_provider="default"))
    with pytest.raises(RuntimeError, match="multi-provider"):
        await runtime._upsert_custom_provider(_PROVIDER)

    _DiscoveryClient.models = []
    runtime = _runtime()
    with pytest.raises(ValueError, match="returned no models"):
        await runtime._upsert_custom_provider({**_PROVIDER, "models": []})
    assert _DiscoveryClient.instances[-1].closed is True

    _DiscoveryClient.discovery_error = OSError("catalog unavailable")
    with pytest.raises(OSError, match="catalog unavailable"):
        await runtime._upsert_custom_provider(_PROVIDER)
    assert _DiscoveryClient.instances[-1].closed is True


@pytest.mark.asyncio
async def test_enabled_provider_closes_client_on_store_or_gateway_failure_and_restores() -> None:
    store = _Store()
    store.upsert_error = OSError("disk unavailable")
    runtime = _runtime(store=store)
    with pytest.raises(OSError, match="disk unavailable"):
        await runtime._upsert_custom_provider(_PROVIDER)
    assert _DiscoveryClient.instances[-1].closed is True

    previous = {**_PROVIDER, "models": ["old"], "api_key": "preserved"}
    store = _Store(previous)
    gateway = _Gateway("default", "example")
    gateway.upsert_error = RuntimeError("router closed")
    runtime = _runtime(store=store, gateway=gateway)
    with pytest.raises(RuntimeError, match="router closed"):
        await runtime._upsert_custom_provider({**_PROVIDER, "models": ["new"]})
    assert _DiscoveryClient.instances[-1].closed is True
    assert store.rows["example"]["models"] == ["old"]
    assert store.rows["example"]["api_key"] == "preserved"


@pytest.mark.asyncio
async def test_ollama_provider_wraps_client_and_commits_discovered_catalog(monkeypatch) -> None:
    import norax.gateway_client.ollama_wrapper as ollama_wrapper

    monkeypatch.setattr(ollama_wrapper, "OllamaGatewayClient", _OllamaClient)
    _DiscoveryClient.models = ["new-a", "new-b"]
    events = _Events()
    gateway = _Gateway("default")
    runtime = _runtime(gateway=gateway, events=events)

    result = await runtime._upsert_custom_provider(
        {**_PROVIDER, "provider_kind": "ollama", "api_key": "new-secret"}
    )

    assert result["models"] == ["new-a", "new-b"]
    assert _DiscoveryClient.instances[-1].kwargs["api_key"] == "new-secret"
    assert _OllamaClient.instances[-1].inner is _DiscoveryClient.instances[-1]
    assert gateway.upserts[-1][2] == ["example/*"]
    assert events.items == [("provider_updated", {"provider": "example", "models": 2})]


@pytest.mark.asyncio
async def test_remove_rejects_invalid_absent_reserved_used_and_unsupported_live() -> None:
    runtime = _runtime()
    assert await runtime._remove_custom_provider("BAD NAME") is False
    assert await runtime._remove_custom_provider("example") is False

    default = {**_PROVIDER, "name": "default", "api_key": ""}
    runtime = _runtime(store=_Store(default))
    with pytest.raises(ValueError, match="base runtime"):
        await runtime._remove_custom_provider("default")

    previous = {**_PROVIDER, "api_key": "secret"}
    runtime = _runtime(store=_Store(previous))
    runtime.default_model = "example/model-a"
    with pytest.raises(ValueError, match="default model"):
        await runtime._remove_custom_provider("example")

    runtime = _runtime(
        store=_Store(previous),
        gateway=SimpleNamespace(
            provider_urls={"example": "https://api.example.test"},
            default_provider="default",
        ),
    )
    with pytest.raises(RuntimeError, match="does not support"):
        await runtime._remove_custom_provider("example")


@pytest.mark.asyncio
async def test_remove_store_refusal_and_live_failure_are_honest_and_rolled_back() -> None:
    previous = {**_PROVIDER, "api_key": "preserved"}
    store = _Store(previous)
    store.remove_result = False
    runtime = _runtime(store=store)
    assert await runtime._remove_custom_provider("example") is False
    assert "example" in store.rows

    for error in (None, OSError("remove failed")):
        store = _Store(previous)
        gateway = _Gateway("default", "example")
        gateway.remove_result = False
        gateway.remove_error = error
        runtime = _runtime(store=store, gateway=gateway)
        with pytest.raises((RuntimeError, OSError)):
            await runtime._remove_custom_provider("example")
        assert store.rows["example"]["api_key"] == "preserved"


@pytest.mark.asyncio
async def test_remove_non_live_provider_commits_event() -> None:
    store = _Store({**_PROVIDER, "enabled": False, "api_key": "secret"})
    events = _Events()
    runtime = _runtime(store=store, events=events)
    assert await runtime.remove_custom_provider("example") is True
    assert store.rows == {}
    assert events.items == [("provider_removed", {"provider": "example"})]


@pytest.mark.asyncio
async def test_default_model_routing_fallback_and_reasoning_downgrades(caplog) -> None:
    runtime = _runtime()
    runtime._provider_store = None
    runtime.thinking_effort = "ultra"
    assert runtime.set_default_model("openai/plain-model") is False
    assert runtime._effective_provider == "openai"
    assert runtime.thinking_effort == "medium"

    runtime.thinking_effort = "max"
    assert runtime.set_default_model("openai/plain-model") is False
    assert runtime.thinking_effort == "medium"

    runtime.thinking_effort = "ultra"
    assert runtime.set_default_model("openai/gpt-5.6-sol") is False
    assert runtime.thinking_effort == "ultra"
    runtime.thinking_effort = "max"
    assert runtime.set_default_model("ollama/glm-5.3:cloud") is False
    assert runtime.thinking_effort == "max"

    secret = "sk-" + "r" * 30
    runtime.gateway.route_error = RuntimeError(secret)
    runtime.gateway.default_provider = "safe-default"
    with caplog.at_level(logging.WARNING, logger="norax.runtime.core"):
        runtime.set_default_model("provider/model")
    assert runtime._effective_provider == "safe-default"
    assert secret not in caplog.text

    runtime.gateway.default_provider = ""
    runtime.set_default_model("provider/model")
    assert runtime._effective_provider == "unknown"
    runtime.gateway = SimpleNamespace(default_provider="ignored")
    runtime._effective_provider = "unchanged"
    runtime.set_default_model("provider/model")
    assert runtime._effective_provider == "unchanged"


def test_runtime_preference_setters_enforce_capability_and_types(monkeypatch) -> None:
    runtime = _runtime()
    with pytest.raises(ValueError, match="unsupported thinking"):
        runtime.set_thinking_effort("impossible")
    runtime.set_thinking_effort("ultra")
    assert runtime.thinking_effort == "medium"
    runtime.set_thinking_effort("max")
    assert runtime.thinking_effort == "medium"
    runtime.default_model = "openai/gpt-5.6-sol"
    runtime.set_thinking_effort("ultra")
    assert runtime.thinking_effort == "ultra"
    runtime.default_model = "ollama/glm-5.3:cloud"
    runtime.set_thinking_effort("max")
    assert runtime.thinking_effort == "max"

    with pytest.raises(TypeError, match="reasoning_output"):
        runtime.set_reasoning_output(1)  # type: ignore[arg-type]
    runtime.set_reasoning_output(True)
    assert runtime.reasoning_output is True

    with pytest.raises(ValueError, match="unsupported planning"):
        runtime.set_planning_mode("committee")
    runtime._planner_enabled = False
    assert runtime.set_planning_mode("orchestrator") is False
    runtime._planner_enabled = True
    assert runtime.set_planning_mode("orchestrator") is True

    for invalid in (True, 2.5, "250"):
        with pytest.raises(TypeError, match="must be an integer"):
            runtime.set_max_tool_rounds(invalid)  # type: ignore[arg-type]
    for invalid in (-1, agent_loop.HARD_ROUND_CAP + 1):
        with pytest.raises(ValueError, match="must be between"):
            runtime.set_max_tool_rounds(invalid)
    runtime.set_max_tool_rounds(0)
    runtime.set_max_tool_rounds(250)
    assert runtime.max_tool_rounds == 250

    with pytest.raises(ValueError, match="memory depth"):
        runtime.set_memory_depth("infinite")
    runtime.set_memory_depth("deep")
    assert runtime.memory_depth == "deep"
    with pytest.raises(ValueError, match="weak-model boost"):
        runtime.set_weak_model_boost("sometimes")
    runtime.set_weak_model_boost("on")
    assert runtime.weak_model_boost == "on"
    with pytest.raises(TypeError, match="stream_replies"):
        runtime.set_stream_replies(1)  # type: ignore[arg-type]
    runtime.set_stream_replies(False)
    assert runtime.stream_replies is False
    with pytest.raises(ValueError, match="response length"):
        runtime.set_response_length("novel")
    runtime.set_response_length("detailed")
    assert runtime.response_length == "detailed"
    with pytest.raises(ValueError, match="tool activity"):
        runtime.set_tool_activity("constant")
    runtime.set_tool_activity("verbose")
    assert runtime.tool_activity == "verbose"

    runtime.memory_depth = "light"
    assert runtime._memory_k_for_turn("body", "owner") == 3
    runtime.memory_depth = "balanced"
    assert runtime._memory_k_for_turn("body", "owner") == 5
    runtime.memory_depth = "deep"
    assert runtime._memory_k_for_turn("body", "owner") == 7
    runtime.memory_depth = "auto"
    monkeypatch.setattr(mm, "memory_k_for_turn", lambda body, tier: len(body) + len(tier))
    assert runtime._memory_k_for_turn("body", "owner") == 9


def test_sync_default_model_persistence_reports_and_scrubs_failure(caplog) -> None:
    store = _Store()
    runtime = _runtime(store=store)
    assert runtime._persist_default_model("provider/model") is True
    assert store.setting_calls == [("default_model", "provider/model")]

    secret = "sk-" + "s" * 30
    store.setting_error = OSError(secret)
    with caplog.at_level(logging.ERROR, logger="norax.runtime.core"):
        assert runtime._persist_default_model("provider/model") is False
    assert secret not in caplog.text
    assert "<REDACTED:openai_key>" in caplog.text


def test_async_default_model_persistence_falls_back_without_running_loop() -> None:
    store = _Store()
    runtime = _runtime(store=store)
    assert runtime._persist_default_model_async("provider/model") is True
    assert store.setting_calls == [("default_model", "provider/model")]


@pytest.mark.asyncio
async def test_async_default_model_persistence_handles_errors_and_bad_task_container(
    caplog,
) -> None:
    secret = "sk-" + "p" * 30
    store = _Store()
    store.setting_error = OSError(secret)
    runtime = _runtime(store=store)
    runtime._maintenance_tasks = []

    with caplog.at_level(logging.ERROR, logger="norax.runtime.core"):
        assert runtime._persist_default_model_async("provider/model") is True
        owner = runtime._default_model_persist_task
        assert owner is not None
        await owner
        await asyncio.sleep(0)

    assert runtime._maintenance_tasks == set()
    assert runtime._default_model_persist_task is None
    assert secret not in caplog.text
    assert "<REDACTED:openai_key>" in caplog.text


@pytest.mark.asyncio
async def test_async_persistence_callback_keeps_new_owner_and_handles_cancellation() -> None:
    runtime = _runtime()
    assert runtime._persist_default_model_async("provider/model") is True
    first = runtime._default_model_persist_task
    assert first is not None
    replacement = asyncio.create_task(asyncio.sleep(0))
    runtime._default_model_persist_task = replacement
    await first
    await asyncio.sleep(0)
    assert runtime._default_model_persist_task is replacement
    await replacement

    assert runtime._persist_default_model_async("provider/second") is True
    cancelled = runtime._default_model_persist_task
    assert cancelled is not None
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    await asyncio.sleep(0)
    assert runtime._default_model_persist_task is None
    assert runtime._maintenance_tasks == set()


@pytest.mark.asyncio
async def test_async_persistence_owner_reports_unexpected_worker_base_exception(caplog) -> None:
    class FatalPersistence(BaseException):
        pass

    secret = "sk-" + "f" * 30
    store = _Store()
    store.setting_error = FatalPersistence(secret)
    runtime = _runtime(store=store)

    with caplog.at_level(logging.ERROR, logger="norax.runtime.core"):
        assert runtime._persist_default_model_async("provider/model") is True
        owner = runtime._default_model_persist_task
        assert owner is not None
        result = await asyncio.gather(owner, return_exceptions=True)
        await asyncio.sleep(0)

    assert isinstance(result[0], FatalPersistence)
    assert "default-model persistence task crashed" in caplog.text
    assert secret not in caplog.text
    assert "<REDACTED:openai_key>" in caplog.text
    assert runtime._default_model_persist_task is None
    assert runtime._maintenance_tasks == set()
