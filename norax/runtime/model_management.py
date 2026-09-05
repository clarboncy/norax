"""Live model selection, custom-provider mutation, and durable settings."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .. import commands as cmd_mod
from ..brain import agent_loop
from ..brain.policy import memory_k_for_turn
from ..commands import _is_gpt_56_sol, _supports_max_reasoning
from ..gateway_client import GatewayClient
from ._mixin import RuntimeAccessMixin
from .validation import _model_identifier, _provider_identifier

log = logging.getLogger("norax.runtime.core")

_MAX_PROVIDER_DISCOVERY_BYTES = 2 * 1024 * 1024
_MAX_PROVIDER_DISCOVERY_MODELS = 100


class ModelManagementMixin(RuntimeAccessMixin):
    _default_model_persist_task: asyncio.Task[None] | None
    _pending_default_model: str | None
    default_model: str
    max_tool_rounds: int
    memory_depth: str
    planning_mode: str
    reasoning_output: bool
    response_length: str
    stream_replies: bool
    thinking_effort: str
    tool_activity: str
    weak_model_boost: str

    def custom_model_catalog(self) -> dict[str, list[tuple[str, str]]]:
        catalog: dict[str, list[tuple[str, str]]] = {}
        store = getattr(self, "_provider_store", None)
        if store is None:
            return catalog
        for provider in store.list_public():
            if provider.get("enabled") is not True:
                continue
            name = provider["name"]
            catalog[name] = [(f"{name}/{model}", model) for model in provider.get("models") or []]
        return catalog

    def _gateway_has_provider(self, name: str) -> bool:
        check = getattr(self.gateway, "has_provider", None)
        if callable(check):
            return bool(check(name))
        urls = getattr(self.gateway, "provider_urls", {})
        return isinstance(urls, dict) and name in urls

    def _custom_provider_usage(self, name: str) -> str | None:
        prefix = f"{name}/"
        model_references: list[tuple[str, object]] = [("default model", self.default_model)]
        model_references.extend(("failover model", model) for model in self.failover_models)
        model_references.append(("vision model", self._vision_config.get("model")))
        for label, model in model_references:
            if isinstance(model, str) and model.startswith(prefix):
                return label
        if getattr(self.gateway, "default_provider", None) == name:
            return "gateway default provider"
        return None

    async def _append_provider_event(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            await self.events.append(kind, payload)
        except Exception:  # noqa: BLE001
            # The provider transaction is already complete. Event-log health is
            # reported independently and must not turn a successful mutation
            # into a false 502 response that invites a duplicate retry.
            log.warning("%s.event_append_failed", kind, exc_info=True)

    async def _restore_custom_provider(
        self,
        name: str,
        previous: dict[str, Any] | None,
    ) -> None:
        if previous is None:
            await asyncio.to_thread(self._provider_store.remove, name)
            return
        public = {key: value for key, value in previous.items() if key != "api_key"}
        await asyncio.to_thread(
            self._provider_store.upsert,
            public,
            previous.get("api_key", ""),
        )

    @staticmethod
    async def _discover_custom_provider_models(
        *,
        client: GatewayClient,
    ) -> list[Any]:
        return await client.fetch_model_ids(
            max_response_bytes=_MAX_PROVIDER_DISCOVERY_BYTES,
            max_models=_MAX_PROVIDER_DISCOVERY_MODELS,
        )

    async def upsert_custom_provider(self, spec: dict[str, Any]) -> dict[str, Any]:
        from ..config.provider_store import validate_provider_api_key, validate_provider_spec

        if not isinstance(spec, dict):
            raise ValueError("provider specification must be an object")
        unknown = set(spec) - {
            "name",
            "base_url",
            "api_key",
            "provider_kind",
            "models",
            "enabled",
        }
        if unknown:
            raise ValueError(f"unsupported provider fields: {sorted(unknown)}")
        clean = validate_provider_spec(spec)
        supplied_key = spec.get("api_key")
        if supplied_key is not None:
            supplied_key = validate_provider_api_key(supplied_key)

        async with self._provider_mutation_lock:
            existing_rows = await asyncio.to_thread(self._provider_store.list_runtime)
            previous = {row["name"]: row for row in existing_rows}.get(clean["name"])
            if clean["name"] in self._base_provider_names:
                raise ValueError("provider name is reserved by the base runtime configuration")
            if previous is None and self._gateway_has_provider(clean["name"]):
                raise ValueError("provider name is already active outside the provider store")

            api_key = validate_provider_api_key(
                supplied_key if supplied_key is not None else (previous or {}).get("api_key", "")
            )
            if clean["enabled"] is not True:
                usage = self._custom_provider_usage(clean["name"])
                if usage:
                    raise ValueError(f"select a different {usage} before disabling this provider")
                was_live = self._gateway_has_provider(clean["name"])
                remove = getattr(self.gateway, "remove_provider", None)
                if was_live and not callable(remove):
                    raise RuntimeError("active gateway does not support dynamic provider removal")
                await asyncio.to_thread(
                    self._provider_store.upsert,
                    clean,
                    api_key if supplied_key is not None else None,
                )
                try:
                    if was_live and (not callable(remove) or not await remove(clean["name"])):
                        raise RuntimeError("active gateway refused to disable the provider")
                except Exception:
                    await self._restore_custom_provider(clean["name"], previous)
                    raise
                await self._append_provider_event("provider_disabled", {"provider": clean["name"]})
                return clean

            upsert = getattr(self.gateway, "upsert_provider", None)
            if not callable(upsert):
                raise RuntimeError("dynamic providers require multi-provider gateway mode")
            # Construct GatewayClient on the event loop. httpx.AsyncClient
            # initialization is lightweight and must not run in a worker
            # thread — the connection pool is bound to this loop and
            # cross-thread construction can stall during TLS setup.
            base_gateway_client = GatewayClient(
                base_url=clean["base_url"],
                timeout=self._gateway_timeout_seconds,
                api_key=api_key or None,
                provider_kind=clean["provider_kind"],
            )
            try:
                models = await self._discover_custom_provider_models(client=base_gateway_client)
                if models:
                    clean = validate_provider_spec({**clean, "models": models})
                if not clean["models"]:
                    raise ValueError(
                        "provider returned no models; provide an OpenAI-compatible /models endpoint"
                    )
            except Exception:
                await base_gateway_client.aclose()
                raise
            gateway_client: Any = base_gateway_client
            if clean["provider_kind"] == "ollama":
                from ..gateway_client.ollama_wrapper import OllamaGatewayClient

                gateway_client = OllamaGatewayClient(gateway_client, metrics=self.metrics)
            try:
                await asyncio.to_thread(
                    self._provider_store.upsert,
                    clean,
                    api_key if supplied_key is not None else None,
                )
            except Exception:
                await gateway_client.aclose()
                raise
            try:
                await upsert(
                    clean["name"],
                    gateway_client,
                    patterns=[f"{clean['name']}/*"],
                )
            except Exception:
                try:
                    await self._restore_custom_provider(clean["name"], previous)
                finally:
                    await gateway_client.aclose()
                raise
            await self._append_provider_event(
                "provider_updated",
                {"provider": clean["name"], "models": len(clean["models"])},
            )
            return clean

    async def remove_custom_provider(self, name: str) -> bool:
        try:
            clean_name = _provider_identifier(name)
        except ValueError:
            return False
        async with self._provider_mutation_lock:
            existing_rows = await asyncio.to_thread(self._provider_store.list_runtime)
            previous = {row["name"]: row for row in existing_rows}.get(clean_name)
            if previous is None:
                return False
            if clean_name in self._base_provider_names:
                raise ValueError("base runtime providers cannot be removed through this API")
            usage = self._custom_provider_usage(clean_name)
            if usage:
                raise ValueError(f"select a different {usage} before removing this provider")
            was_live = self._gateway_has_provider(clean_name)
            remove = getattr(self.gateway, "remove_provider", None)
            if was_live and not callable(remove):
                raise RuntimeError("active gateway does not support dynamic provider removal")
            removed_from_store = await asyncio.to_thread(self._provider_store.remove, clean_name)
            if not removed_from_store:
                return False
            try:
                if was_live and (not callable(remove) or not await remove(clean_name)):
                    raise RuntimeError("active gateway refused to remove the provider")
            except Exception:
                await self._restore_custom_provider(clean_name, previous)
                raise
            await self._append_provider_event("provider_removed", {"provider": clean_name})
            return True

    def set_default_model(self, model: str) -> bool:
        model = _model_identifier(model)
        self.default_model = model
        self._effective_model = model
        route_for = getattr(self.gateway, "route_for", None)
        if callable(route_for):
            try:
                self._effective_provider = route_for(model)[0]
            except Exception as exc:  # noqa: BLE001
                log.warning("gateway.model_selection_route_failed: %r", exc)
        log.info("default_model set to %s", model)
        # Schedule persistence off the event loop. The fsync inside
        # atomic_write_text can block for milliseconds; doing it inline
        # stalls every ingress stream and active turn on the same loop.
        persisted = self._persist_default_model_async(model)
        # max/ultra thinking is only valid for models that support it; downgrade
        # if the model changes away from a max-capable one.
        if self.thinking_effort == "ultra" and not _is_gpt_56_sol(model):
            self.thinking_effort = "medium"
            log.info("thinking_effort downgraded to medium (model is not GPT-5.6 Sol)")
        elif self.thinking_effort == "max" and not _supports_max_reasoning(model):
            self.thinking_effort = "medium"
            log.info("thinking_effort downgraded to medium (model does not support max)")
        return persisted

    def set_thinking_effort(self, level: str) -> None:
        if level not in cmd_mod.THINK_LABELS:
            raise ValueError(f"unsupported thinking effort: {level!r}")
        if level == "ultra" and not _is_gpt_56_sol(self.default_model):
            log.warning(
                "refusing to set thinking_effort=%s for non-GPT-5.6-Sol model %s",
                level,
                self.default_model,
            )
            return
        if level == "max" and not _supports_max_reasoning(self.default_model):
            log.warning(
                "refusing to set thinking_effort=%s for model %s (requires GPT-5.6 Sol or GLM-5.3)",
                level,
                self.default_model,
            )
            return
        self.thinking_effort = level
        log.info("thinking_effort set to %s", level)

    def set_reasoning_output(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("reasoning_output must be a boolean")
        self.reasoning_output = enabled
        log.info("reasoning_output set to %s", self.reasoning_output)

    def set_planning_mode(self, mode: str) -> bool:
        if mode not in cmd_mod.PLANNING_MODES:
            raise ValueError(f"unsupported planning mode: {mode!r}")
        if mode == "orchestrator" and not self._planner_enabled:
            log.warning("planning_mode=orchestrator rejected: planner connector is disabled")
            return False
        self.planning_mode = mode
        log.info("planning_mode set to %s", mode)
        return True

    def set_max_tool_rounds(self, cap: int) -> None:
        if isinstance(cap, bool) or not isinstance(cap, int):
            raise TypeError("max_tool_rounds must be an integer")
        if not 0 <= cap <= agent_loop.HARD_ROUND_CAP:
            raise ValueError(f"max_tool_rounds must be between 0 and {agent_loop.HARD_ROUND_CAP}")
        self.max_tool_rounds = cap
        log.info("max_tool_rounds set to %s", self.max_tool_rounds)

    def set_memory_depth(self, mode: str) -> None:
        if mode not in cmd_mod.MEMORY_DEPTHS:
            raise ValueError(f"unsupported memory depth: {mode!r}")
        self.memory_depth = mode
        log.info("memory_depth set to %s", mode)

    def set_weak_model_boost(self, mode: str) -> None:
        if mode not in cmd_mod.BOOST_MODES:
            raise ValueError(f"unsupported weak-model boost: {mode!r}")
        self.weak_model_boost = mode
        log.info("weak_model_boost set to %s", mode)

    def set_stream_replies(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("stream_replies must be a boolean")
        self.stream_replies = enabled
        log.info("stream_replies set to %s", self.stream_replies)

    def set_response_length(self, pref: str) -> None:
        if pref not in cmd_mod.LENGTH_PREFS:
            raise ValueError(f"unsupported response length: {pref!r}")
        self.response_length = pref
        log.info("response_length set to %s", pref)

    def set_tool_activity(self, mode: str) -> None:
        if mode not in cmd_mod.ACTIVITY_MODES:
            raise ValueError(f"unsupported tool activity: {mode!r}")
        self.tool_activity = mode
        log.info("tool_activity set to %s", mode)

    def _memory_k_for_turn(self, body: str, sender_tier: str) -> int:
        depth_k = {"light": 3, "balanced": 5, "deep": 7}
        if self.memory_depth in depth_k:
            return depth_k[self.memory_depth]
        return memory_k_for_turn(body, sender_tier)

    def _persist_default_model(self, model: str) -> bool:
        try:
            self._provider_store.set_runtime_setting("default_model", model)
            log.info("persisted default_model=%s to private runtime settings", model)
            return True
        except Exception:  # noqa: BLE001
            log.exception("failed to persist default_model")
            return False

    def _persist_default_model_async(self, model: str) -> bool:
        """Schedule durable persistence off the event loop.

        Returns ``True`` immediately (the live state is already updated);
        the fsync happens in a worker thread so ingress and active turns
        are not stalled.  If the thread fails, the error is logged but
        the live selection remains active — the caller already received
        an honest "active for this process" response.
        """
        store = getattr(self, "_provider_store", None)
        if store is None:
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self._persist_default_model(model)

        self._pending_default_model = model
        current = getattr(self, "_default_model_persist_task", None)
        if current is not None and not current.done():
            return True

        async def _persist_pending() -> None:
            while self._pending_default_model is not None:
                selected = self._pending_default_model
                self._pending_default_model = None
                try:
                    await asyncio.to_thread(
                        store.set_runtime_setting,
                        "default_model",
                        selected,
                    )
                    log.info("persisted default_model=%s to private runtime settings", selected)
                except Exception:  # noqa: BLE001
                    log.exception("failed to persist default_model=%s", selected)

        task = loop.create_task(_persist_pending(), name="persist-default-model")
        self._default_model_persist_task = task
        maintenance_tasks = getattr(self, "_maintenance_tasks", None)
        if not isinstance(maintenance_tasks, set):
            maintenance_tasks = set()
            self._maintenance_tasks = maintenance_tasks
        maintenance_tasks.add(task)

        def _finished(completed: asyncio.Task[None]) -> None:
            self._maintenance_tasks.discard(completed)
            if self._default_model_persist_task is completed:
                self._default_model_persist_task = None
            if completed.cancelled():
                return
            error = completed.exception()
            if error is not None:
                log.error("default-model persistence task crashed: %r", error)

        task.add_done_callback(_finished)
        return True
