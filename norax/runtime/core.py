"""Production runtime composition, construction, and main ingress loop.

Cohesive behavior lives in dependency-directed mixins; ``Runtime`` remains the
stable public API and owns all mutable state initialization.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import commands as cmd_mod
from ..adapter.cron_in import CronAdapter
from ..adapter.discord_in import DiscordInAdapter
from ..adapter.http_in import HttpInAdapter
from ..brain import agent_loop
from ..brain import hot_path as hot_path
from ..brain.hot_path.basal_ganglia import BasalGanglia
from ..brain.hot_path.vta import VTA
from ..config.loader import Config
from ..context.window import RollingWindow
from ..dispatch import tools as dispatch_tools
from ..gateway_client import (
    GatewayClient,
    GatewayRouter,
)
from ..memory.causal_graph import CausalGraph
from ..memory.entity_graph import EntityGraph
from ..memory.episodic import EpisodicBuffer
from ..memory.hebbian import HebbianLearner
from ..memory.injection import ContextInjector
from ..memory.retrievers.fast import FastContext
from ..memory.retrievers.multi_signal import MultiSignalRetriever
from ..memory.store import MemoryStore
from ..memory.temporal_graph import TemporalGraph
from ..memory.user_model import UserModel
from ..observability.log import EventLog
from ..observability.metrics import Metrics
from .capability_registry import CapabilityRegistry
from .cognition import CognitionMixin
from .delivery import DeliveryMixin
from .graceful_degradation import get_degradation_manager
from .health import (
    _deferred_runtime_lifecycle_action as _deferred_runtime_lifecycle_action,
)
from .health import (
    _record_operational_probe as _record_operational_probe,
)
from .health import (
    _refresh_tool_capability_evidence as _refresh_tool_capability_evidence,
)
from .health import (
    _remote_relay_probe_result as _remote_relay_probe_result,
)
from .health import (
    _turn_failure_message as _turn_failure_message,
)
from .history import (
    _bounded_history_text as _bounded_history_text,
)
from .history import (
    _candidate_prior_turn_ids as _candidate_prior_turn_ids,
)
from .history import (
    _checkpoint_status as _checkpoint_status,
)
from .history import (
    _clean_historical_assistant_text as _clean_historical_assistant_text,
)
from .history import (
    _history_budget_for_model as _history_budget_for_model,
)
from .history import (
    _history_turn_limit as _history_turn_limit,
)
from .history import (
    _select_history_tail as _select_history_tail,
)
from .ingress_bus import IngressBus
from .lifecycle import LifecycleMixin
from .memory_coordinator import MemoryCoordinator
from .model_management import ModelManagementMixin
from .operations import OperationalMixin
from .outbound import OutboundRegistry
from .session import SessionMixin
from .turn_pipeline import TurnPipelineMixin
from .validation import (
    _a2a_advertised_url,
    _apply_configured_fleet_startup_fixes,
    _bounded_config_int,
    _bounded_environment_int,
    _flag_enabled,
    _is_loopback_host,
    _model_identifier,
    _model_identifiers,
    _provider_identifier,
    _runtime_choice,
)

log = logging.getLogger("norax.runtime.core")


class Runtime(
    CognitionMixin,
    DeliveryMixin,
    LifecycleMixin,
    ModelManagementMixin,
    OperationalMixin,
    SessionMixin,
    TurnPipelineMixin,
):
    def __init__(
        self,
        *,
        ingress: IngressBus,
        events: EventLog,
        cfg: Config,
        gateway: GatewayClient | GatewayRouter,
        default_model: str = "qwen3.8-27b-fast:latest",
        failover_models: list[str] | None = None,
        outbound: OutboundRegistry | None = None,
        metrics: Metrics | None = None,
        soul: Any | None = None,
    ) -> None:
        self.ingress = ingress
        self.events = events
        self.cfg = cfg
        self.gateway = gateway
        self.default_model = _model_identifier(default_model, label="default model")
        self.failover_models = _model_identifiers(failover_models, label="failover_models")
        self.thinking_effort = _runtime_choice(
            getattr(cfg, "thinking_effort", "medium"),
            set(cmd_mod.THINK_LABELS),
            default="medium",
            label="thinking_effort",
        )
        self.reasoning_output = _flag_enabled(getattr(cfg, "reasoning_output", False))
        self.planning_mode = _runtime_choice(
            getattr(cfg, "planning_mode", "direct"),
            cmd_mod.PLANNING_MODES,
            default="direct",
            label="planning_mode",
        )
        configured_rounds = getattr(cfg, "max_tool_rounds", 0) or 0
        if isinstance(configured_rounds, bool) or not isinstance(configured_rounds, int):
            raise TypeError("max_tool_rounds must be an integer")
        if configured_rounds < 0:
            raise ValueError("max_tool_rounds must be non-negative")
        self.max_tool_rounds = (
            min(configured_rounds, agent_loop.HARD_ROUND_CAP) if configured_rounds else 0
        )
        if self.max_tool_rounds != configured_rounds:
            log.warning(
                "configured max_tool_rounds=%s exceeds the hard safety cap; using %s",
                configured_rounds,
                self.max_tool_rounds,
            )
        self.memory_depth = _runtime_choice(
            getattr(cfg, "memory_depth", "auto"),
            cmd_mod.MEMORY_DEPTHS,
            default="auto",
            label="memory_depth",
        )
        self.weak_model_boost = _runtime_choice(
            getattr(cfg, "weak_model_boost", "auto"),
            cmd_mod.BOOST_MODES,
            default="auto",
            label="weak_model_boost",
        )
        self.stream_replies = _flag_enabled(getattr(cfg, "stream_replies", True), default=True)
        self.response_length = _runtime_choice(
            getattr(cfg, "response_length", "balanced"),
            cmd_mod.LENGTH_PREFS,
            default="balanced",
            label="response_length",
        )
        self.tool_activity = _runtime_choice(
            getattr(cfg, "tool_activity", "normal"),
            cmd_mod.ACTIVITY_MODES,
            default="normal",
            label="tool_activity",
        )
        self.outbound = outbound or OutboundRegistry()
        self.metrics = metrics or Metrics()
        self._soul = soul
        self.capabilities = CapabilityRegistry()
        self._config_hash: str = "unknown"
        self._effective_model: str = "?"
        self._effective_provider: str = "?"
        self._configured_provider: str = "?"
        self._provider_store: Any = None
        self._base_provider_names: set[str] = set()
        self._provider_mutation_lock = asyncio.Lock()
        self._gateway_timeout_seconds = 600.0
        self.discord: Any = None  # set by Runtime.build() if discord is enabled
        self.cron: Any = None  # set by Runtime.build() if cron is enabled
        self.agent_os_bridge: Any = None  # set by Runtime.build() if agent_os chat is wired
        self.reminders: Any = None  # persistent scheduler bound by Runtime.build()
        self.mcp_client: Any = None
        self.a2a_server: Any = None
        self._a2a_uvicorn: Any = None
        self._draining = False
        self._run_started = False
        self._shutdown_complete = False
        self._shutdown_lock: asyncio.Lock | None = None
        self._started_mono = time.monotonic()
        self._started_wall = time.time()
        self._last_user_turn_time: float | None = None
        # Stop signal: per-channel flag to abort active turns
        self._stop_channels: set[str] = set()
        self._active_turn_tasks: dict[str, asyncio.Task[Any]] = {}
        self._turn_slot_releasers: dict[asyncio.Task[Any], Callable[[], None]] = {}
        self._turn_queues: dict[str, deque[Any]] = {}
        self._turn_work_count = 0
        self._max_concurrent_turns = max(
            1,
            min(64, int(getattr(cfg, "max_concurrent_turns", 8) or 8)),
        )
        self._max_pending_turns = max(
            self._max_concurrent_turns,
            min(10_000, int(getattr(cfg, "max_pending_turns", 256) or 256)),
        )
        self._max_pending_per_channel = min(32, self._max_pending_turns)
        self._turn_semaphore = asyncio.Semaphore(self._max_concurrent_turns)
        self._active_typing_handles: dict[asyncio.Task[Any], Any] = {}
        self._active_stream_messages: dict[asyncio.Task[Any], Any] = {}
        self._capability_probe_task: asyncio.Task[None] | None = None
        self._sleep_task: asyncio.Task[None] | None = None
        self._warmup_task: asyncio.Task[None] | None = None
        self._probe_task: asyncio.Task[None] | None = None
        self._autonomy_task: asyncio.Task[Any] | None = None
        self._optional_tasks: list[asyncio.Task[Any]] = []
        self._maintenance_tasks: set[asyncio.Task[Any]] = set()
        self._default_model_persist_task: asyncio.Task[None] | None = None
        self._pending_default_model: str | None = None
        # Per-channel prompt-cache observability (Claude Code SEV pattern).
        self._last_static_hash: dict[str, str] = {}
        self._last_tools_hash: dict[str, str] = {}
        self._cache_break_times: dict[str, list[float]] = {}
        # Per-channel rolling-window: conversation history across turns.
        self._windows: dict[str, RollingWindow] = {}
        # Graceful degradation manager (singleton)
        self._degradation = get_degradation_manager()
        # Memory store + retriever
        mem_root = getattr(cfg, "memory_root", None)
        self._memory_root: Path | None = mem_root if isinstance(mem_root, Path) else None
        # Type annotations (assigned in both branches below)
        self._memory_store: MemoryStore | None = None
        self._memory_coordinator: MemoryCoordinator | None = None
        self._fast_ctx: FastContext | None = None
        self._episodic: EpisodicBuffer | None = None
        self._user_model: UserModel | None = None
        self._skill_learner: Any = None
        self._context_injector: ContextInjector | None = None
        self._causal_graph: CausalGraph | None = None
        self._temporal_graph: TemporalGraph | None = None
        self._graph_update_lock = asyncio.Lock()
        self._hybrid: MultiSignalRetriever | None = None
        self._last_graph_save: float = 0.0
        cognition_cfg = getattr(cfg, "cognition", {})
        if not isinstance(cognition_cfg, dict):
            cognition_cfg = {}
        configured_experiments = cognition_cfg.get("experimental_signals", False)
        self._experimental_cognitive_signals = _flag_enabled(
            configured_experiments
        ) or _flag_enabled(os.environ.get("NORAX_EXPERIMENTAL_COGNITIVE_SIGNALS", ""))
        configured_idle_learning = cognition_cfg.get("idle_learning", False)
        self._idle_learning_enabled = _flag_enabled(configured_idle_learning) or _flag_enabled(
            os.environ.get("NORAX_IDLE_LEARNING_ENABLED", "")
        )
        configured_harness_analysis = cognition_cfg.get("harness_analysis", False)
        self._harness_analysis_enabled = _flag_enabled(
            configured_harness_analysis
        ) or _flag_enabled(os.environ.get("NORAX_HARNESS_ANALYSIS_ENABLED", ""))
        vision_cfg = getattr(cfg, "vision", {})
        self._vision_config = dict(vision_cfg) if isinstance(vision_cfg, dict) else {}
        connectors_cfg = getattr(cfg, "connectors", {})
        if not isinstance(connectors_cfg, dict):
            connectors_cfg = {}
        self._connectors: dict[str, dict[str, Any]] = {
            str(name): dict(value)
            for name, value in connectors_cfg.items()
            if isinstance(value, dict)
        }
        self._planner_enabled = _flag_enabled(
            self._connectors.get("planner", {"enabled": True}).get("enabled", True)
        )
        self._multi_agent_enabled = _flag_enabled(
            self._connectors.get("multi_agent", {"enabled": True}).get("enabled", True)
        )
        try:
            self._multi_agent_max_concurrent = _bounded_config_int(
                self._connectors.get("multi_agent", {}).get("max_concurrent", 4),
                label="multi-agent concurrency",
                minimum=1,
                maximum=32,
            )
        except ValueError as error:
            log.warning("%s; using 4", error)
            self._multi_agent_max_concurrent = 4
        active_inference_cfg = connectors_cfg.get("active_inference", {})
        self._active_inference_enabled = _flag_enabled(active_inference_cfg.get("enabled", False))
        if not self._planner_enabled and self.planning_mode == "orchestrator":
            log.warning("planner connector is disabled; forcing planning_mode=direct")
            self.planning_mode = "direct"
        # Stateful, evidence-backed components are shared across turns.
        # Experimental prompt signals are opt-in and remain off the production
        # hot path until they have grounded provenance and live eval evidence.
        self._output_verifier: Any = None
        self._self_model: Any = None
        self._active_inference: Any = None
        self._metacognitive: Any = None
        self._best_of_n: Any = None
        self._curiosity: Any = None
        self._domain_transfer: Any = None
        self._analogy_engine: Any = None
        if mem_root and isinstance(mem_root, Path):
            self._memory_store = MemoryStore(root=mem_root)
            self._memory_store.refresh()
            self._memory_coordinator = MemoryCoordinator(
                mem_root,
                store=self._memory_store,
            )
            self._fast_ctx = FastContext(store=self._memory_store)
            # Multi-signal retriever: keyword + embedding + entity-link + FTS5 index
            entity_graph = EntityGraph(root=mem_root)
            # Persistent graph instances: shared with the retriever (which loads
            # them once at init) and mutated in-memory post-turn; Sprint E saves
            # debounced. Per-turn load+full-save of the 14MB causal JSON was the
            # hot path cost this replaces.
            causal_graph = CausalGraph(root=mem_root)
            temporal_graph = TemporalGraph(root=mem_root)
            self._causal_graph = causal_graph
            self._temporal_graph = temporal_graph
            sqlite_index = None
            try:
                from ..memory.retrievers.sqlite_index import SQLiteIndexRetriever

                sqlite_index = SQLiteIndexRetriever(memory_root=mem_root)
                # The projection is reconciled in run() before ingress starts.
                # Do not query it here: a fresh workspace has no schema yet.
                log.info("sqlite_index_retriever configured: %s", sqlite_index.db_path)
            except Exception as e:
                log.warning("sqlite_index_retriever.init_failed: %r", e)
            try:
                from ..memory.embeddings import OllamaEmbedder
                from ..memory.retrievers.local import LocalRetriever

                embedder = OllamaEmbedder.from_env()
                local = LocalRetriever(
                    store=self._memory_store,
                    embedder=embedder,
                    degradation_manager=self._degradation,
                )
                from ..memory.retrievers.cross_encoder import LLMJudgeReranker

                reranker_enabled = os.environ.get(
                    "NORAX_RERANK_ENABLED",
                    "false",
                ).strip().lower() in {"1", "true", "yes", "on"}
                reranker_model = os.environ.get("NORAX_RERANK_MODEL", "").strip()
                if reranker_enabled and not reranker_model:
                    log.warning("llm_judge_reranker disabled: NORAX_RERANK_MODEL is required")
                    reranker_enabled = False
                reranker = (
                    LLMJudgeReranker(model=reranker_model, enabled=True)
                    if reranker_enabled
                    else None
                )
                self._hybrid = MultiSignalRetriever(
                    keyword=self._fast_ctx,
                    embedding=local,
                    entity_graph=entity_graph,
                    sqlite_index=sqlite_index,
                    causal_graph=causal_graph,
                    temporal_graph=temporal_graph,
                    reranker=reranker,
                )
                log.info(
                    "multi_signal_retriever ready "
                    "(kw + embed + entity + idx + causal + tmp; llm_judge_rerank=%s)",
                    "enabled" if reranker_enabled else "disabled",
                )
            except Exception as e:
                log.warning("multi_signal_retriever.init_failed: %r (keyword-only)", e)
                self._hybrid = MultiSignalRetriever(
                    keyword=self._fast_ctx,
                    entity_graph=entity_graph,
                    sqlite_index=sqlite_index,
                    causal_graph=causal_graph,
                    temporal_graph=temporal_graph,
                )

            # Sprint F: smart context mixer over keyword + embedding + external
            local_retriever: LocalRetriever | None = None
            embedder_for_external = None
            if self._hybrid is not None:
                try:
                    local_retriever = self._hybrid.embedding
                except Exception:
                    local_retriever = None
            if local_retriever is not None:
                embedder_for_external = local_retriever.embedder
            if embedder_for_external is None:
                try:
                    from ..memory.embeddings import OllamaEmbedder

                    embedder_for_external = OllamaEmbedder.from_env()
                except Exception:
                    embedder_for_external = None
            try:
                from ..memory.retrievers.external import ExternalRetriever

                external = ExternalRetriever(
                    store=self._memory_store,
                    embedder=embedder_for_external,  # type: ignore[arg-type]
                )
                self._context_injector = ContextInjector(
                    fast=self._fast_ctx,
                    local=local_retriever,
                    external=external,
                )
                log.info("context_injector ready")
            except Exception as e:
                log.warning("context_injector.init_failed: %r", e)
            log.info(
                "memory loaded: %d hot, %d canonical, %d sleep",
                len(self._memory_store.hot),
                len(self._memory_store.all_canonical()),
                len(self._memory_store.sleep),
            )
            # Ensure hot identity lines are current (fixes stale Gen5 scratchpad header).
            try:
                from ..memory.hot_inject import refresh_hot_identity

                role = os.environ.get("NORAX_ROLE", "production")
                refresh_hot_identity(mem_root, generation=7, role=role)
                self._memory_store.refresh()
            except Exception:  # noqa: BLE001
                log.debug("hot_identity.refresh_failed", exc_info=True)
            self._turn_count = 0
            # Brain state (persistent across turns)
            state_dir = mem_root / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            self._vta = VTA(state_path=state_dir / "vta.json")
            self._basal_ganglia = BasalGanglia(state_path=state_dir / "basal_ganglia.json")
            self._hebbian = HebbianLearner()
            # Episodic replay and generated skills are experimental learning
            # paths.  Keeping them off by default avoids duplicate per-turn
            # persistence and prevents heuristic patterns from entering live
            # retrieval without an explicit deployment decision.
            if self._idle_learning_enabled or self._experimental_cognitive_signals:
                self._episodic = EpisodicBuffer(root=mem_root / "episodic")
            if self._idle_learning_enabled:
                from ..brain.skill_learner import SkillLearner

                self._skill_learner = SkillLearner(root=mem_root)
            # The user model directly contributes a bounded prompt block.
            self._user_model = UserModel(root=mem_root)
        else:
            self._vta = VTA()
            self._basal_ganglia = BasalGanglia()
            self._hebbian = HebbianLearner()

    @classmethod
    def build(cls, cfg: Config) -> Runtime:
        host, port = cfg.http_bind
        from ..soul import load_soul

        soul = load_soul(
            root=cfg.soul_path.parent,
            soul_path=cfg.soul_path,
            identity_path=cfg.identity_path,
            user_path=cfg.user_path,
        )
        # The dashboard and monitors depend on this exact endpoint. Silent
        # port fallback creates a split-brain deployment that appears started
        # but is unreachable through its configured clients.
        log.info("http binding %s:%d", host, port)

        outbound = OutboundRegistry()
        metrics = Metrics()
        try:
            from ..gateway_client.cursor_observability import get_cursor_observability

            get_cursor_observability(metrics=metrics)
        except ImportError:
            pass
        adapters: list[Any] = [HttpInAdapter(host=host, port=port, metrics=metrics)]

        # Direct owner-chat bridge for the Agent OS dashboard. Registers the
        # bridge as an outbound adapter under channel key ``agent_os`` so the
        # runtime routes assistant replies back to the dashboard websocket, and
        # configures the token-authenticated ingress route.
        http_in = adapters[0]
        chat_token = cfg.agent_os_chat_token
        if chat_token:
            http_in.configure_owner_chat(
                owner_id=cfg.owner_id or "owner",
                owner_label=cfg.owner_label,
                chat_token=chat_token,
            )
            outbound.register("agent_os", http_in.agent_os_bridge)
            log.info("agent_os owner-chat bridge wired")
        else:
            log.info("agent_os chat_token not set; owner-chat bridge disabled")

        disc = None
        dcfg = cfg.discord
        if dcfg.enabled:
            if dcfg.token:
                disc = DiscordInAdapter(config=dcfg, owner_id=cfg.owner_id)
                adapters.append(disc)
                outbound.register("discord", disc)
                log.info("discord adapter wired")
            else:
                log.warning(
                    "discord enabled=true but no token configured; "
                    "adapter NOT started. Set NORAX_DISCORD_TOKEN or "
                    "channels.discord.token to enable."
                )
        # Bind outbound into the motor-tool registry so t_message_send is live.
        dispatch_tools.bind_outbound(outbound)

        ccfg = cfg.cron
        cron = None
        if ccfg.enabled and ccfg.jobs:
            cron = CronAdapter.from_config(ccfg.to_raw(), owner_id=cfg.owner_id)
            if cron.jobs:
                adapters.append(cron)
                log.info("cron adapter wired: %d enabled job(s)", len(cron.jobs))
            else:
                cron = None
                log.warning("cron enabled but no enabled jobs were configured")

        ingress = IngressBus(adapters)
        events = EventLog(cfg.event_log)
        gw_cfg = cfg.gateway
        import os as _os

        from ..config.provider_store import ProviderStore

        provider_store = ProviderStore()
        persisted_settings = provider_store.runtime_settings()
        persisted_model = persisted_settings.get("default_model")
        if persisted_model is not None:
            try:
                gw_cfg["default_model"] = _model_identifier(
                    persisted_model, label="persisted default model"
                )
            except ValueError as error:
                log.warning("ignoring invalid persisted default model: %s", error)
        # Environment is the final deployment authority, including over UI-
        # persisted settings from an earlier run.
        if _os.environ.get("NORAX_DEFAULT_MODEL"):
            gw_cfg["default_model"] = _model_identifier(
                _os.environ["NORAX_DEFAULT_MODEL"], label="NORAX_DEFAULT_MODEL"
            )
        if _os.environ.get("NORAX_DEFAULT_PROVIDER"):
            gw_cfg["default_provider"] = _provider_identifier(
                _os.environ["NORAX_DEFAULT_PROVIDER"], label="NORAX_DEFAULT_PROVIDER"
            )
        custom_providers = [
            row for row in provider_store.list_runtime() if row.get("enabled", True)
        ]
        timeout = gw_cfg["timeout_seconds"]

        configured_providers = gw_cfg.get("providers")
        providers_cfg = dict(configured_providers) if isinstance(configured_providers, dict) else {}
        base_provider_names = set(providers_cfg)
        accepted_custom_providers = []
        for custom in custom_providers:
            if custom["name"] in providers_cfg:
                log.warning("custom provider ignored because name is reserved: %s", custom["name"])
                continue
            accepted_custom_providers.append(custom)
            providers_cfg[custom["name"]] = {
                "base_url": custom["base_url"],
                "provider_kind": custom["provider_kind"],
                "api_key": custom.get("api_key") or None,
            }
        if providers_cfg:
            # Multi-provider mode: build a GatewayRouter.
            clients: dict[str, Any] = {}
            for pname, pcfg in providers_cfg.items():
                if not isinstance(pcfg, dict):
                    continue
                env_name = f"NORAX_{pname.upper().replace('-', '_')}_TOKEN"
                pk = pcfg.get("api_key") or _os.environ.get(env_name)
                provider_kind = str(pcfg.get("provider_kind") or pname).lower()
                if provider_kind == "codex_direct":
                    # Local Codex proxies on :4146/:4147 borrow OAuth from
                    # CODEX_HOME auth.json themselves. Do not attach a
                    # generic gateway bearer; it confuses local wrapping.
                    pk = None
                else:
                    pk = pk or _os.environ.get("NORAX_GATEWAY_TOKEN")
                if provider_kind == "ollama" or pname in ("ollama", "ollama_direct"):
                    from ..gateway_client.ollama_wrapper import OllamaGatewayClient

                    inner = GatewayClient(
                        base_url=pcfg.get("base_url", "http://127.0.0.1:11434/v1"),
                        timeout=timeout,
                        api_key=pk,
                        extra_headers=pcfg.get("extra_headers") or None,
                        stream_required=_flag_enabled(pcfg.get("stream_required", False)),
                        model_prefix=pcfg.get("model_prefix"),
                        provider_kind="ollama",
                        chat_path=pcfg.get("chat_path", "/chat/completions"),
                    )
                    clients[pname] = OllamaGatewayClient(inner, metrics=metrics)
                else:
                    clients[pname] = GatewayClient(
                        base_url=pcfg.get("base_url", "http://127.0.0.1:11434/v1"),
                        timeout=timeout,
                        api_key=pk,
                        extra_headers=pcfg.get("extra_headers") or None,
                        stream_required=_flag_enabled(pcfg.get("stream_required", False)),
                        model_prefix=pcfg.get("model_prefix"),
                        provider_kind=pcfg.get("provider_kind") or pname,
                        chat_path=pcfg.get("chat_path", "/chat/completions"),
                    )
                log.info("gateway.provider %s → %s", pname, pcfg.get("base_url"))
            routes_raw = gw_cfg["routes"]
            routes = [(f"{row['name']}/*", row["name"]) for row in accepted_custom_providers]
            routes.extend((r[0], r[1]) for r in routes_raw)
            default_prov = gw_cfg["default_provider"]
            if default_prov not in clients:
                raw_gateway = cfg.raw.get("gateway")
                explicit_default = (
                    isinstance(raw_gateway, dict) and "default_provider" in raw_gateway
                ) or bool(_os.environ.get("NORAX_DEFAULT_PROVIDER"))
                if explicit_default:
                    raise ValueError(
                        f"default gateway provider {default_prov!r} was not initialized"
                    )
                default_prov = next(iter(clients))
                gw_cfg["default_provider"] = default_prov
            gateway: Any = GatewayRouter(
                providers=clients,
                routes=routes,
                default_provider=default_prov,
            )
            log.info(
                "gateway.router ready: %d providers, %d routes, default=%s",
                len(clients),
                len(routes),
                default_prov,
            )
        else:
            # Legacy single-client mode.
            api_key = gw_cfg.get("api_key") or _os.environ.get("NORAX_GATEWAY_TOKEN")
            gateway = GatewayClient(  # type: ignore[no-redef]
                base_url=gw_cfg.get("base_url", "http://127.0.0.1:11434/v1"),
                timeout=timeout,
                api_key=api_key,
                extra_headers=gw_cfg.get("extra_headers") or None,
                provider_kind=gw_cfg.get("provider_kind", "ollama"),
                chat_path=gw_cfg.get("chat_path", "/chat/completions"),
            )
        rt_instance = cls(
            ingress=ingress,
            events=events,
            cfg=cfg,
            gateway=gateway,
            default_model=gw_cfg.get("default_model", "qwen3.8-27b-fast:latest"),
            outbound=outbound,
            failover_models=gw_cfg.get("failover_models") or [],
            metrics=metrics,
            soul=soul,
        )
        rt_instance._provider_store = provider_store
        rt_instance._base_provider_names = base_provider_names
        rt_instance._gateway_timeout_seconds = timeout
        if chat_token:
            rt_instance.agent_os_bridge = http_in.agent_os_bridge
        from .reminders import ReminderScheduler

        rt_instance.reminders = ReminderScheduler(
            root=cfg.memory_root / "reminders",
            outbound=outbound,
            default_channel="discord",
            default_target=cfg.owner_id,
        )
        dispatch_tools.bind_reminder_scheduler(rt_instance.reminders)
        # Wire runtime reference into HTTP adapter for /status and /readyz
        adapters[0]._runtime_ref = rt_instance
        # Wire live memory search into the search_memory tool
        if rt_instance._hybrid is not None:
            dispatch_tools.bind_memory_search(rt_instance._hybrid.search)
            log.info("search_memory tool wired to hybrid retriever")
        elif hasattr(rt_instance, "_fast_ctx") and rt_instance._fast_ctx is not None:
            dispatch_tools.bind_memory_search(rt_instance._fast_ctx.search)
            log.info("search_memory tool wired to keyword retriever")
        # Wire live status info into the status tool
        import time as _time

        _start_time = _time.time()
        # Compute a config hash for observability — redacts sensitive fields.
        import json as _json

        _cfg_for_hash = {k: v for k, v in cfg.raw.items() if k not in ("owner",)}
        _cfg_hash = hashlib.sha256(
            _json.dumps(_cfg_for_hash, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        rt_instance._config_hash = _cfg_hash
        rt_instance._effective_model = gw_cfg.get("default_model", "?")
        configured_provider = gw_cfg.get("default_provider", "?")
        rt_instance._configured_provider = configured_provider
        rt_instance._effective_provider = configured_provider
        route_for = getattr(gateway, "route_for", None)
        if callable(route_for):
            try:
                rt_instance._effective_provider = route_for(rt_instance._effective_model)[0]
            except Exception as exc:  # noqa: BLE001
                log.warning("gateway.effective_route_failed: %r", exc)

        def _live_status_info() -> dict[str, Any]:
            effective_provider = configured_provider
            live_route_for = getattr(gateway, "route_for", None)
            if callable(live_route_for):
                try:
                    effective_provider = live_route_for(rt_instance.default_model)[0]
                except Exception as exc:  # noqa: BLE001
                    log.warning("gateway.live_status_route_failed: %r", exc)
            rt_instance._effective_model = rt_instance.default_model
            rt_instance._effective_provider = effective_provider
            provider_urls = getattr(gateway, "provider_urls", {})
            providers = sorted(provider_urls) if isinstance(provider_urls, dict) else []
            return {
                "default_model": rt_instance.default_model,
                "default_provider": effective_provider,
                "configured_default_provider": configured_provider,
                "providers": providers,
                "discord_enabled": disc is not None,
                "start_time": _start_time,
                "config_hash": _cfg_hash,
                "version": cmd_mod.NORAX_VERSION,
            }

        dispatch_tools.bind_status_info(_live_status_info)
        # Wire slash-command handler on the discord adapter (if any).
        if disc is not None:
            disc.set_slash_handler(rt_instance._slash_dispatch)
            rt_instance.discord = disc
        if cron is not None:
            rt_instance.cron = cron
        return rt_instance

    async def run(self) -> None:
        if self._run_started:
            raise RuntimeError("Runtime.run() may only be started once")
        if self._shutdown_complete:
            raise RuntimeError("Runtime cannot run after shutdown")
        self._run_started = True
        # Deployment-specific host repair must never mutate a generic install.
        # Operators that own the referenced user services can explicitly opt
        # into the legacy fleet maintenance hook.
        _apply_configured_fleet_startup_fixes()
        # Reconcile canonical files and all derived projections before ingress
        # accepts turns. This prevents stale retrieval after an unclean stop or
        # an out-of-band memory edit while the service was offline.
        try:
            if self._memory_coordinator is not None:
                self._memory_coordinator.canonical_changed("runtime_start")
                await self._memory_coordinator.sync_projections()
        except Exception as e:
            log.warning("memory_projection_sync.startup_failed: %r", e)
        self._mount_commerce_connector()
        await self.ingress.start()
        for capability, adapter in (
            ("discord_ingress", self.discord),
            ("cron_scheduler", self.cron),
        ):
            if adapter is None:
                continue
            self.capabilities.register(capability)
            task = getattr(adapter, "_task", None)
            if not isinstance(task, asyncio.Task):
                self.capabilities.mark_failed(
                    capability,
                    RuntimeError(f"{adapter.name} did not start a supervisor task"),
                )
                continue
            self.capabilities.mark_ok(capability)
            self._track_optional_task(capability, task)
        reminders = getattr(self, "reminders", None)
        if reminders is not None:
            await reminders.start()
            self.capabilities.mark_ok("reminder_scheduler")
            reminder_task = getattr(reminders, "_task", None)
            if isinstance(reminder_task, asyncio.Task):
                self._track_optional_task("reminder_scheduler", reminder_task)
        for capability in (
            "browser_tool",
            "computer_tool",
            "sandbox_tool",
            "web_search_tool",
            "remote_relay",
        ):
            self.capabilities.register(capability, stale_after_sec=900.0)
        self.capabilities.register(
            "tool_manifest",
            stale_after_sec=900.0,
            required_for_readiness=True,
        )
        self.capabilities.register("operational_probe_loop", required_for_readiness=True)
        self.capabilities.mark_initializing("operational_probe_loop")
        self._capability_probe_task = asyncio.create_task(
            self._capability_probe_loop(), name="operational-capability-probe"
        )
        self._observe_service_task("operational_probe_loop", self._capability_probe_task)
        await self.events.append(
            "runtime.start",
            {
                "version": cmd_mod.NORAX_VERSION,
                "config_hash": getattr(self, "_config_hash", "unknown"),
                "effective_model": getattr(self, "_effective_model", "?"),
                "effective_provider": getattr(self, "_effective_provider", "?"),
                "build_commit": cmd_mod.BUILD_COMMIT,
                "build_date": cmd_mod.BUILD_DATE,
            },
        )
        # Sprint B: background sleep consolidation on idle
        self._last_turn_time = time.time()
        self.capabilities.register("idle_memory_maintenance")
        self.capabilities.mark_ok("idle_memory_maintenance")
        self._sleep_task = asyncio.create_task(self._idle_sleep_loop())
        self._observe_service_task("idle_memory_maintenance", self._sleep_task)
        # Pre-warm embedding caches from disk so first query doesn't pay cold-start cost
        self._warmup_task = asyncio.create_task(self._warmup_caches())
        # Synthetic completions consume provider quota, so deployments can
        # disable them independently from serving real user turns.
        self._completion_probe_enabled = os.environ.get(
            "NORAX_COMPLETION_PROBE_ENABLED", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        requested_probe_mode = os.environ.get("NORAX_COMPLETION_PROBE_MODE", "transport")
        self._completion_probe_mode = requested_probe_mode.strip().lower()
        if self._completion_probe_mode not in {"transport", "completion"}:
            log.warning(
                "invalid NORAX_COMPLETION_PROBE_MODE=%r; using transport",
                requested_probe_mode,
            )
            self._completion_probe_mode = "transport"
        # Completion probes are tiny but still consume model capacity. Keep
        # them infrequent, wait for an idle gap, and cap deferral below the
        # readiness freshness window so sustained traffic cannot hide an
        # unhealthy serving path forever.
        self._completion_probe_interval_seconds = _bounded_environment_int(
            "NORAX_COMPLETION_PROBE_INTERVAL_SECONDS",
            default=600,
            minimum=60,
            maximum=840,
        )
        self._completion_probe_idle_seconds = _bounded_environment_int(
            "NORAX_COMPLETION_PROBE_IDLE_SECONDS",
            default=30,
            minimum=0,
            maximum=300,
        )
        self._last_probe_result: dict[str, Any] | None = None
        if self._completion_probe_enabled:
            self.capabilities.register(
                "serving_probe_loop",
                stale_after_sec=900.0,
                required_for_readiness=True,
            )
            self.capabilities.mark_initializing("serving_probe_loop")
            self._probe_task = asyncio.create_task(self._completion_probe_loop())
            self._observe_service_task("serving_probe_loop", self._probe_task)
        else:
            self._probe_task = None
            self._last_probe_result = {
                "ok": True,
                "enabled": False,
                "mode": "disabled",
            }
            log.info("completion_probe disabled by config")
        # Autonomy engine: scratchpad sync, self-healing, prompt optimization, checkpoint resume
        # Register subsystems with graceful degradation manager
        try:
            dm = get_degradation_manager()
            log.info("graceful_degradation: %d subsystems registered", len(dm._subsystems))
        except Exception as e:
            log.warning("graceful_degradation.init_failed: %r", e)
        self.autonomy_config = None
        try:
            from .autonomy import AutonomyConfig, AutonomyEngine

            self.autonomy_config = AutonomyConfig()
            mem_root = getattr(self.cfg, "memory_root", None)
            if mem_root and isinstance(mem_root, Path) and self.autonomy_config.enabled:
                engine = AutonomyEngine(
                    memory_root=mem_root,
                    config=self.autonomy_config,
                    episodic=getattr(self, "_episodic", None),
                    gateway=self.gateway,
                    event_log=self.events,
                    default_model=getattr(self, "_effective_model", "") or "",
                )
                self._autonomy_task = engine.start()
                self.capabilities.register("autonomy_engine")
                self.capabilities.mark_ok("autonomy_engine")
                self._observe_service_task("autonomy_engine", self._autonomy_task)
                log.info("autonomy_engine started")
            elif mem_root and isinstance(mem_root, Path):
                log.info("autonomy_engine disabled by config")
        except Exception as e:
            log.warning("autonomy_engine.start_failed: %r", e)
        # Optional subsystems: wire behind config flags, default off
        # MCP client: discover external tools
        mcp_cfg = self._connectors.get("mcp_client", {})
        if _flag_enabled(mcp_cfg.get("enabled")):
            self.capabilities.register(
                "mcp_client",
                required_for_readiness=_flag_enabled(mcp_cfg.get("required")),
            )
            self.capabilities.mark_initializing("mcp_client")
            try:
                from ..mcp.client import NoraxMCPClient

                self.mcp_client = NoraxMCPClient()
                mcp_client = self.mcp_client
                mcp_client.set_health_handler(self._record_mcp_health)
                connections = []
                servers = mcp_cfg.get("servers") or []
                if not isinstance(servers, list) or not servers:
                    raise ValueError("enabled MCP client has no configured servers")
                for server_cfg in servers:
                    if not isinstance(server_cfg, dict):
                        raise ValueError("each MCP server must be an object")
                    if server_cfg.get("transport") != "stdio":
                        raise ValueError(
                            f"unsupported MCP transport: {server_cfg.get('transport')!r}"
                        )
                    command = str(server_cfg.get("command") or "").strip()
                    if not command:
                        raise ValueError("MCP stdio server requires a command")
                    sname = str(server_cfg.get("name") or "mcp-server")
                    connections.append(
                        mcp_client.connect_stdio(
                            sname,
                            command,
                            *(str(arg) for arg in (server_cfg.get("args") or [])),
                        )
                    )
                await asyncio.wait_for(asyncio.gather(*connections), timeout=30.0)
                self.capabilities.mark_ok("mcp_client")
                for name, owner_task in mcp_client.connection_tasks().items():
                    self._track_mcp_connection_task(mcp_client, name, owner_task)
                self._record_mcp_health(mcp_client.health())
                log.info("mcp_client ready: %d server(s)", len(connections))
            except Exception as e:
                if self.mcp_client is not None:
                    try:
                        await self.mcp_client.disconnect_all()
                    except Exception:  # noqa: BLE001
                        log.warning("mcp_client.startup_cleanup_failed", exc_info=True)
                    self.mcp_client = None
                self.capabilities.mark_failed("mcp_client", e)
                if mcp_cfg.get("required"):
                    raise RuntimeError("required MCP client failed to start") from e
        # A2A server: agent-to-agent HTTP server
        a2a_cfg = self._connectors.get("a2a_server", {})
        if _flag_enabled(a2a_cfg.get("enabled")):
            self.capabilities.register(
                "a2a_server",
                required_for_readiness=_flag_enabled(a2a_cfg.get("required")),
            )
            self.capabilities.mark_initializing("a2a_server")
            a2a_server = None
            a2a_task: asyncio.Task[Any] | None = None
            try:
                import uvicorn

                from ..a2a.server import NoraxA2AServer, create_a2a_app

                a2a_host = str(a2a_cfg.get("host", "127.0.0.1")).strip()
                a2a_port = _bounded_config_int(
                    a2a_cfg.get("port", 8766),
                    label="A2A port",
                    minimum=1,
                    maximum=65_535,
                )
                if not a2a_host:
                    raise ValueError("A2A host must not be empty")
                a2a_token = str(
                    os.environ.get("NORAX_A2A_TOKEN") or a2a_cfg.get("auth_token") or ""
                ).strip()
                if not _is_loopback_host(a2a_host) and not a2a_token:
                    raise ValueError("A2A bearer authentication is required for non-loopback binds")
                max_tasks = _bounded_config_int(
                    a2a_cfg.get("max_tasks", 1_000),
                    label="A2A max_tasks",
                    minimum=1,
                    maximum=100_000,
                )
                max_concurrent = _bounded_config_int(
                    a2a_cfg.get("max_concurrent", 4),
                    label="A2A max_concurrent",
                    minimum=1,
                    maximum=128,
                )
                advertised_url = _a2a_advertised_url(
                    a2a_host,
                    a2a_port,
                    a2a_cfg.get("base_url"),
                )

                a2a_server = NoraxA2AServer(
                    runtime=self,
                    base_url=advertised_url,
                    auth_token=a2a_token or None,
                    max_tasks=max_tasks,
                    max_concurrent=max_concurrent,
                )
                a2a_app = create_a2a_app(a2a_server)
                self.a2a_server = a2a_server
                a2a_config = uvicorn.Config(
                    a2a_app,
                    host=a2a_host,
                    port=a2a_port,
                    log_level="warning",
                )
                a2a_uvicorn = uvicorn.Server(a2a_config)
                self._a2a_uvicorn = a2a_uvicorn
                a2a_task = asyncio.create_task(a2a_uvicorn.serve(), name="a2a-server")
                self._optional_tasks.append(a2a_task)
                await self._await_uvicorn_started(a2a_uvicorn, a2a_task)
                self._track_optional_task("a2a_server", a2a_task)
                self.capabilities.mark_ok("a2a_server")
                log.info("a2a_server started on %s:%d", a2a_host, a2a_port)
            except Exception as e:
                if a2a_task is not None and not a2a_task.done():
                    a2a_task.cancel()
                    await asyncio.gather(a2a_task, return_exceptions=True)
                if a2a_server is not None:
                    await a2a_server.close()
                self.a2a_server = None
                self._a2a_uvicorn = None
                self.capabilities.mark_failed("a2a_server", e)
                if a2a_cfg.get("required"):
                    raise RuntimeError("required A2A server failed to start") from e
        try:
            async for env in self.ingress.stream():
                if self._draining:
                    return
                self._last_turn_time = time.time()
                self._last_user_turn_time = self._last_turn_time
                chan_id = self._channel_id(env)
                body = (env.body or "").strip().lower()

                # Cancellation belongs to the owner-gated command handler.
                # Doing it before authorization let an untrusted /stop abort
                # another user's active turn.
                if body in ("/stop", "!stop"):
                    await self._handle_turn(env)
                    continue

                if not self._queue_turn(chan_id, env):
                    await self._send_command_reply(
                        env,
                        "Norax is at its bounded turn capacity. Try again shortly.",
                        reply_to=env.message_id,
                    )
        except asyncio.CancelledError:
            log.info("runtime loop cancelled")
            for task in (
                self._sleep_task,
                self._probe_task,
                self._warmup_task,
                self._capability_probe_task,
                self._autonomy_task,
            ):
                if task is not None:
                    task.cancel()
            for task in getattr(self, "_optional_tasks", []):
                task.cancel()
            raise
