"""Authenticated turn execution from ingress evidence through persistence."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import subprocess
import time
from typing import Any

from .. import commands as cmd_mod
from ..brain import agent_loop, hot_path
from ..gateway_client import SpendGuardTripped
from ..memory.episodic import Episode
from ._mixin import RuntimeAccessMixin
from .health import (
    _deferred_runtime_lifecycle_action,
    _refresh_tool_capability_evidence,
    _turn_failure_message,
)
from .history import (
    _bounded_history_text,
    _candidate_prior_turn_ids,
    _checkpoint_status,
    _clean_historical_assistant_text,
    _coerce_tool_args_json,
    _history_budget_for_model,
    _is_coding_query,
    _select_history_tail,
)
from .validation import _explicit_result_ok, _flag_enabled

log = logging.getLogger("norax.runtime.core")


class TurnPipelineMixin(RuntimeAccessMixin):
    _last_graph_save: float

    async def _handle_turn(self, env) -> None:
        chan_id = self._channel_id(env)
        owner_task = asyncio.current_task()
        completed = False
        try:
            with self.events.trace_scope() as trace_id:
                if isinstance(env.metadata, dict):
                    env.metadata["trace_id"] = trace_id
                self.metrics.ingress_total.labels(
                    source=env.source,
                    channel=env.channel,
                    trusted=str(env.trusted).lower(),
                ).inc()
                with self.metrics.time_brain_turn():
                    await self._handle_turn_inner(env)
            completed = True
        except asyncio.CancelledError:
            raise
        except Exception:
            self.metrics.agent_turn_failures.inc()
            raise
        finally:
            # Own UI helpers at the turn boundary so cancellation and early
            # returns cannot leak typing keep-alives or partial messages.
            streaming = (
                self._active_stream_messages.pop(owner_task, None)
                if owner_task is not None
                else None
            )
            if not completed and streaming is not None:
                try:
                    await streaming.delete()
                except Exception:  # noqa: BLE001
                    log.debug("stream.cancel_cleanup_failed", exc_info=True)
            typing_keep = (
                self._active_typing_handles.pop(owner_task, None)
                if owner_task is not None
                else None
            )
            if typing_keep is not None:
                try:
                    await typing_keep.stop(timeout=1.5)
                except Exception:  # noqa: BLE001
                    log.debug("typing.turn_cleanup_failed", exc_info=True)
            self._stop_channels.discard(chan_id)
            # Lifecycle is runtime-owned: disconnecting a browser never
            # cancels the turn, and every terminal path clears working.
            if self.agent_os_bridge is not None:
                try:
                    await self.agent_os_bridge.broadcast_status("idle")
                except Exception:
                    log.debug("agent_os idle_failed", exc_info=True)

    async def _handle_turn_inner(self, env) -> None:
        chan_id = self._channel_id(env)
        owner_task = asyncio.current_task()
        if self.agent_os_bridge is not None:
            try:
                await self.agent_os_bridge.broadcast_status("typing")
            except Exception:
                log.debug("agent_os typing_failed", exc_info=True)
        await self.events.append(
            "ingress",
            {
                "channel": env.channel,
                "source": env.source,
                "message_id": env.message_id,
                "sender_id": env.sender.id,
                "sender_tier": env.sender.tier,
                "trusted": env.trusted,
                "body": env.body[:2000],
            },
        )

        # ---- mirror inbound to Agent OS dashboard --------------------
        if (
            self.agent_os_bridge is not None
            and env.source == "discord"
            and env.sender.tier != "system"
        ):
            try:
                await self.agent_os_bridge.broadcast_inbound(
                    sender_label=env.sender.label or env.sender.id,
                    text=env.body or "",
                    source="discord",
                )
            except Exception:  # noqa: BLE001
                log.debug("agent_os mirror_failed", exc_info=True)

        # ---- slash-command interception ------------------------------
        pc = cmd_mod.parse(env.body or "")
        if pc is not None and cmd_mod.is_known(pc):
            await self._handle_command(env, pc)
            return

        # Start Discord typing before any memory or planning work.  Agent OS
        # receives its typing state above; keeping this adjacent prevents the
        # two surfaces from drifting when retrieval is slow or degraded.
        typing_keep = None
        target_channel: str | None = None
        if (
            env.source == "discord"
            and self.discord is not None
            and self.stream_replies
            and (env.raw or {}).get("channel_id")
        ):
            target_channel = str((env.raw or {}).get("channel_id"))
            try:
                typing_keep = await self.discord.start_typing(target_channel)
                if typing_keep is not None and owner_task is not None:
                    self._active_typing_handles[owner_task] = typing_keep
            except Exception:  # noqa: BLE001
                log.debug("typing.start_failed", exc_info=True)

        # Discover memory changes; unchanged files reuse parsed neurons.
        if self._memory_store is not None:
            try:
                await asyncio.to_thread(self._memory_store.refresh)
            except Exception:  # noqa: BLE001
                log.warning(
                    "memory.refresh_failed; continuing without refreshed state", exc_info=True
                )

        retrieval_k = self._memory_k_for_turn(env.body or "", env.sender.tier)

        async def _retrieve(query: str, k: int = 5):
            """Smart context retrieval: ContextInjector first, then hybrid, then keyword."""
            k = min(max(k, retrieval_k), 24)
            if self._context_injector is not None:
                try:
                    task_type = "coding" if _is_coding_query(query) else "general"
                    injected = await self._context_injector.run(
                        query,
                        k_each=min(k, 16),
                        task_type=task_type,
                    )
                    if injected.items:
                        return [(n, score, src) for n, score, src in injected.items]
                except Exception as e:
                    log.warning("context_injector.run.error query=%r: %r", query[:60], e)
            if self._hybrid is not None:
                hits = await asyncio.to_thread(self._hybrid.search_sync, query, k=k)
                return [(n, s, tag) for n, s, tag in hits]
            if self._fast_ctx is not None:
                return await asyncio.to_thread(self._fast_ctx.search, query, k=k)
            return []

        # Inject user model into metadata before plan_turn
        if self._user_model is not None and env.sender.id:
            try:
                user_model_block = self._user_model.render_for_prompt(env.sender.id)
            except Exception:  # noqa: BLE001
                user_model_block = ""
        else:
            user_model_block = ""

        try:
            ctx, rendered = await hot_path.plan_turn(
                env,
                runtime_info={
                    "model": self.default_model,
                    "channel": env.channel,
                    "capabilities": "agentic",
                },
                retrieve=_retrieve,
                memory_root=self._memory_root,
                user_model_block=user_model_block,
                soul=self._soul,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("brain plan failed")
            self.metrics.brain_errors.labels(where="brain").inc()
            self.metrics.agent_turn_failures.inc()
            await self.events.append("error", {"where": "brain.plan", "err": repr(e)})
            await self._send_command_reply(
                env,
                _turn_failure_message(e),
                reply_to=env.message_id,
            )
            return

        self.metrics.brain_turns.labels(decision=ctx.decision).inc()
        ch = env.channel or "default"
        prev_static = self._last_static_hash.get(ch)
        if prev_static and prev_static != rendered.static_hash:
            self.metrics.prompt_cache_breaks.labels(channel=ch).inc()
            now = time.time()
            cache_window = [t for t in self._cache_break_times.get(ch, []) if now - t < 300.0]
            cache_window.append(now)
            self._cache_break_times[ch] = cache_window
            self.metrics.prompt_cache_breaks_window.labels(channel=ch).set(len(cache_window))
            log.info(
                "prompt_cache_break channel=%s old=%s new=%s window_5m=%d",
                ch,
                prev_static,
                rendered.static_hash,
                len(cache_window),
            )
            if len(cache_window) > 2:
                log.warning(
                    "prompt_cache_break_alert channel=%s breaks_in_5m=%d static_hash=%s",
                    ch,
                    len(cache_window),
                    rendered.static_hash,
                )
        self._last_static_hash[ch] = rendered.static_hash
        tools_key = hashlib.sha256(",".join(sorted(ctx.allowed_tools or [])).encode()).hexdigest()[
            :16
        ]
        prev_tools = self._last_tools_hash.get(ch)
        if prev_tools and prev_tools != tools_key:
            self.metrics.prompt_tools_changes.labels(channel=ch).inc()
            log.info(
                "prompt_tools_change channel=%s old=%s new=%s tools=%s",
                ch,
                prev_tools,
                tools_key,
                ctx.allowed_tools,
            )
        self._last_tools_hash[ch] = tools_key
        await self.events.append(
            "brain",
            {
                "decision": ctx.decision,
                "static_hash": rendered.static_hash,
                "tools": ctx.allowed_tools,
                "focus": ctx.focus.summary,
            },
            attrs={
                "gen_ai.system": "gateway",
                "gen_ai.request.model": self.default_model,
            },
        )

        # Streaming setup (Discord only for now).  Typing already started
        # before memory/planning; now prepare the deferred streaming message.
        # If anything fails, we fall back to the classic _emit_reply path.
        # ------------------------------------------------------------------
        streaming_msg = None
        if target_channel is not None:
            try:
                streaming_msg = await self.discord.begin_streaming_message(
                    target_channel,
                    reply_to=env.message_id,
                    defer_initial=True,
                )
                if streaming_msg is not None and owner_task is not None:
                    self._active_stream_messages[owner_task] = streaming_msg
            except Exception:  # noqa: BLE001
                log.debug("stream.begin_failed", exc_info=True)

        if ctx.decision != "emit_reply":
            # Clean up typing before returning early
            if typing_keep is not None:
                try:
                    await typing_keep.stop(timeout=1.5)
                except Exception:  # noqa: BLE001
                    pass
            if streaming_msg is not None:
                try:
                    await streaming_msg.delete()
                except Exception:  # noqa: BLE001
                    pass
            return
        stream_delta_buf: list[str] = []

        _turn_t0 = time.monotonic()

        async def _on_stream_delta(delta: str) -> None:
            stream_delta_buf.append(delta)
            if streaming_msg is not None:
                await streaming_msg.append(delta)

        on_delta = _on_stream_delta if streaming_msg is not None else None

        # Build session history from the rolling window.
        window = self._get_window(chan_id)
        # Record user message in the window.
        turn_id = window.start_turn()
        # For image-only messages (empty body + attachments), store a
        # meaningful placeholder so the window context isn't blank.
        user_text_for_window = env.body or ""
        if not user_text_for_window and (env.attachments or []):
            user_text_for_window = "(image attached)"
        window.add_user(user_text_for_window, turn_id=turn_id)
        # Evict old frames if over budget.
        window.evict_if_needed()
        # Collect prior conversation turns (excluding the current user
        # message). Allocate by the selected model's context capacity: four
        # one-line turns are cheap, while one pasted log can be enormous.
        candidate_prior_turns = _candidate_prior_turn_ids(window.body[:-1], user_text_for_window)
        provider_kind = "unknown"
        provider_kind_for_model = getattr(self.gateway, "provider_kind_for_model", None)
        if callable(provider_kind_for_model):
            try:
                provider_kind = str(provider_kind_for_model(self._effective_model) or "unknown")
            except Exception:  # noqa: BLE001
                log.debug("history.provider_kind_resolution_failed", exc_info=True)
        history_budget = _history_budget_for_model(
            self._effective_model,
            provider_kind=provider_kind,
            action_request=agent_loop.is_action_command(user_text_for_window),
        )
        protected_prior_turns, selected_history_calls, history_text_cap = _select_history_tail(
            window.body[:-1],
            candidate_prior_turns,
            token_budget=history_budget,
        )

        # Build prior messages, grouping same-turn tool_calls into a single
        # assistant message (Claude API requires all parallel tool uses to share
        # one assistant message, otherwise IDs appear duplicate → 400 error).
        #
        # Single-pass design: collect tool_call/tool_result by turn_id, then
        # emit them in body order when the first tool_call for each turn is
        # encountered. User/assistant text frames pass through directly.
        prior: list[dict] = []
        _pending_tool_calls: dict[int, list] = {}  # turn_id -> list of tool_call dicts
        _pending_tool_results: dict[int, list] = {}  # turn_id -> list of tool_result dicts
        _seen_call_ids: set[str] = set()
        _emitted_tool_turns: set[int] = set()

        # Pass 1: collect all tool calls and results grouped by turn_id
        for f in window.body[:-1]:
            if f.turn_id not in protected_prior_turns:
                continue
            if f.kind == "tool_call" and f.call_id:
                if f.call_id not in selected_history_calls:
                    continue
                if f.call_id in _seen_call_ids:
                    continue
                _seen_call_ids.add(f.call_id)
                _pending_tool_calls.setdefault(f.turn_id, []).append(
                    {
                        "id": f.call_id,
                        "type": "function",
                        "function": {
                            "name": f.meta.get("name", "unknown"),
                            "arguments": _coerce_tool_args_json(f.content),
                        },
                    }
                )
            elif f.kind == "tool_result" and f.call_id:
                if f.call_id not in selected_history_calls:
                    continue
                if f.call_id not in _seen_call_ids:
                    continue
                _pending_tool_results.setdefault(f.turn_id, []).append(
                    {
                        "role": "tool",
                        "tool_call_id": f.call_id,
                        "content": _bounded_history_text(f.content, 2_000),
                    }
                )

        # Pass 2: emit messages in body order, flushing tool groups atomically
        for f in window.body[:-1]:
            if f.turn_id not in protected_prior_turns:
                continue
            if f.kind == "user":
                prior.append(
                    {"role": "user", "content": _bounded_history_text(f.content, history_text_cap)}
                )
            elif f.kind == "assistant":
                prior.append(
                    {
                        "role": "assistant",
                        "content": _bounded_history_text(
                            _clean_historical_assistant_text(f.content), history_text_cap
                        ),
                    }
                )
            elif f.kind == "tool_call" and f.call_id:
                if f.call_id not in selected_history_calls:
                    continue
                if f.turn_id in _emitted_tool_turns:
                    continue
                _emitted_tool_turns.add(f.turn_id)
                all_calls = _pending_tool_calls.get(f.turn_id, [])
                if all_calls:
                    prior.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": all_calls,
                        }
                    )
                    for res in _pending_tool_results.get(f.turn_id, []):
                        prior.append(res)
            elif f.kind == "tool_result":
                pass  # already emitted above

        # Build correction gate (L32) — non-LLM fact checker against memory.
        # Uses hybrid retriever (keyword + embedding) if available.
        correction_gate = None
        try:
            from ..context import CorrectionGate

            hybrid_retriever = self._hybrid
            fast_retriever = self._fast_ctx
            if hybrid_retriever is not None:

                async def _gate_retrieve(query, k=5):
                    # Post-draft fact checks use the already-warmed local
                    # keyword/graph/FTS signals. Embedding and reranker calls
                    # add seconds to every response without improving exact
                    # identifier/KV comparisons.
                    return await asyncio.to_thread(hybrid_retriever.search_sync, query, k=k)
            elif fast_retriever is not None:

                async def _gate_retrieve(query, k=5):
                    return fast_retriever.search(query, k=k)

            else:
                raise RuntimeError("no memory retriever is available for correction checks")

            correction_gate = CorrectionGate(retrieve=_gate_retrieve)
            self.capabilities.mark_ok("correction_gate")
        except Exception as e:  # noqa: BLE001
            self.capabilities.mark_failed("correction_gate", e)

        # --- Cognitive amplifiers and zero-call signal engines ---
        self._ensure_cognitive_components()
        output_verifier = self._output_verifier
        self_model = self._self_model
        active_inference = self._active_inference
        metacognitive = self._metacognitive
        best_of_n = self._best_of_n
        curiosity = self._curiosity
        domain_transfer = self._domain_transfer
        analogy_engine = self._analogy_engine

        # Pre-pass: compute the cognitive signals for THIS turn and fold the
        # hints into the system prompt before the loop runs.
        _cog_signal: dict[str, Any] = {}
        _turn_domain = "general"
        try:
            from ..brain.self_model import classify_domain

            _turn_text = env.body or ""
            _turn_domain = classify_domain(_turn_text)
            if curiosity is not None and _turn_text:
                _cur = curiosity.assess(_turn_text, domain=_turn_domain)
                _cog_signal["novelty"] = _cur.novelty
                if _cur.explore_hint:
                    rendered.system = f"{rendered.system}\n\n{_cur.explore_hint}"
                if _cur.novelty >= 0.40:
                    await self.events.append(
                        "curiosity_signal",
                        {
                            "novelty": _cur.novelty,
                            "surprise": _cur.surprise,
                            "new_domain": _cur.is_new_domain,
                            "new_entities": _cur.is_new_entity_mix,
                        },
                    )
            if domain_transfer is not None and _turn_text:
                _dt_hints = domain_transfer.transfer_hints(_turn_domain, _turn_text)
                if _dt_hints:
                    _cog_signal["transfers"] = len(_dt_hints)
                    rendered.system = (
                        f"{rendered.system}\n\n"
                        "TRANSFERRED TACTICS (proven in other domains, apply here):\n"
                        + "\n".join(f"- {h}" for h in _dt_hints)
                    )
            if analogy_engine is not None and _turn_text:
                _analogs = analogy_engine.find_analogies(
                    _turn_text,
                    task_type=_turn_domain,
                    limit=3,
                )
                _block = analogy_engine.render_hints(_analogs)
                if _block:
                    _cog_signal["analogies"] = len(_analogs)
                    rendered.system = f"{rendered.system}\n\n{_block}"
        except Exception as e:  # noqa: BLE001
            log.debug("cognitive_signals.error: %r", e)

        # Run the agent loop: L10 + tool-calls + iteration.
        # Initialise these before the optional multi-agent branch so a false
        # trigger (the normal case) cannot leave the single-agent gate reading
        # an unbound local.
        resp = None
        trace: list[dict[str, Any]] = []
        rounds = 0
        _task_state = None
        _restored_task_state: dict[str, Any] | None = None
        _restored_prior_status: str = "in_progress"

        # Offer the last task_state to build_task_state for crash recovery.
        # That builder restores it only when the new prompt is a confident
        # continuation; unrelated tasks receive a clean state.
        try:
            from ..runtime.checkpoint import load_latest_checkpoint

            _cp = load_latest_checkpoint(chan_id, memory_root=self._memory_root)
            if _cp and _cp.get("task_state"):
                _cp_status = _cp.get("status", "in_progress")
                _cp_task = _cp.get("task_state", {})
                _restored_task_state = _cp_task
                _restored_prior_status = _cp_status
                log.debug(
                    "checkpoint.restore channel=%s status=%s pending=%d completed=%d",
                    chan_id,
                    _cp_status,
                    len(_cp_task.get("pending_steps", [])),
                    len(_cp_task.get("completed_actions", [])),
                )
        except Exception as e:  # noqa: BLE001
            log.debug("checkpoint.restore_failed: %r", e)
        # Optional multi-agent trigger. Heuristic decomposition adds model calls
        # and can change task semantics, so it is explicitly opt-in.
        _multi_agent_started = False
        _multi_agent_execution_error: Exception | None = None
        try:
            if self._multi_agent_enabled and bool(
                getattr(self.autonomy_config, "multi_agent_auto", False)
            ):
                from .autonomy import should_trigger_multi_agent

                should_decompose = should_trigger_multi_agent(
                    env.body or "", list(ctx.allowed_tools or [])
                )
            else:
                should_decompose = False

            if should_decompose:
                from ..brain.multi_agent import MultiAgentOrchestrator
                from ..gateway_client import GatewayResponse

                log.info("multi_agent: auto-decomposing task")
                mao = MultiAgentOrchestrator(self.gateway, default_model=self.default_model)
                _multi_agent_started = True
                ma_result = await mao.run(
                    task=env.body or "",
                    system_prompt=rendered.system,
                    allowed_tools=list(ctx.allowed_tools or []),
                    sender_tier=env.sender.tier,
                    event_log=self.events,
                    max_concurrent=self._multi_agent_max_concurrent,
                )
                await self.events.append(
                    "multi_agent",
                    {
                        "subtasks": len(ma_result.sub_results),
                        "verified_outcome": ma_result.verified_outcome,
                        "failed_subtasks": sum(
                            1 for result in ma_result.sub_results if not result.success
                        ),
                        "total_tool_calls": ma_result.total_tool_calls,
                        "elapsed": round(ma_result.elapsed, 2),
                    },
                )
                resp = GatewayResponse(
                    request_id="",
                    model=self.default_model,
                    content=ma_result.summary,
                    tool_calls=[],
                    usage=dict(ma_result.usage),
                    raw={
                        "multi_agent": True,
                        "subtasks": len(ma_result.sub_results),
                        "verified_outcome": ma_result.verified_outcome,
                        "incomplete": not ma_result.verified_outcome,
                        "status_reason": (
                            "all_subtasks_verified"
                            if ma_result.verified_outcome
                            else "one_or_more_subtasks_unverified"
                        ),
                        "usage_scope": "turn_total",
                    },
                )
                trace = []
                for sr in ma_result.sub_results:
                    for tc in sr.tool_calls or []:
                        trace.append(
                            tc
                            if isinstance(tc, dict)
                            else {"name": str(tc), "result": {"ok": sr.success}}
                        )
                rounds = sum(r.rounds for r in ma_result.sub_results)
        except SpendGuardTripped as e:
            _multi_agent_execution_error = e
        except Exception as e:
            if _multi_agent_started:
                _multi_agent_execution_error = e
            else:
                log.debug("multi_agent.trigger_failed_before_execution: %r", e)

        try:
            if _multi_agent_execution_error is not None:
                raise _multi_agent_execution_error
            from ..brain.orchestrator import Orchestrator
            from ..gateway_client import GatewayResponse

            # A successful multi-agent run is already the execution result.
            # Do not immediately overwrite it with the single-agent path.
            if resp is None and self._planner_enabled and self.planning_mode == "orchestrator":
                from ..brain.orchestrator import EXECUTOR_MODEL
                from ..brain.strong_model_scaffold import resolve_planner_model

                planner = resolve_planner_model(self.default_model)
                log.info(
                    "orchestrator: planner=%s executor=%s",
                    planner,
                    EXECUTOR_MODEL,
                )
                orch = Orchestrator(self.gateway)
                orch_result = await orch.run(
                    system_prompt=rendered.system,
                    user_prompt=rendered.user,
                    allowed_tools=list(ctx.allowed_tools or []),
                    sender_tier=env.sender.tier,
                    event_log=self.events,
                    on_delta=on_delta,
                    max_rounds=self.max_tool_rounds,
                    planner_model=planner,
                    prior_messages=prior,
                )
                content, trace, rounds = orch_result
                resp = GatewayResponse(
                    request_id="",
                    model=self.default_model,
                    content=content or "",
                    tool_calls=[],
                    usage=dict(orch_result.usage),
                    raw={
                        "orchestrator": True,
                        "rounds": rounds,
                        "planner": orch_result.planner_model,
                        "executor": EXECUTOR_MODEL,
                        "session_model": self.default_model,
                        "verified_outcome": orch_result.verified_outcome,
                        "incomplete": not orch_result.complete,
                        "status_reason": orch_result.status_reason,
                        "completion_signal": orch_result.completion_signal,
                        "mutation_outcome": orch_result.mutation_outcome,
                        "blocking_categories": orch_result.blocking_categories,
                        "usage_scope": "turn_total",
                    },
                )
            elif resp is None:
                _tools = list(ctx.allowed_tools or [])
                # Inject MCP tools if the client is connected
                _mcp = getattr(self, "mcp_client", None)
                if _mcp is not None and env.sender.tier == "owner":
                    try:
                        _mcp_tools = await _mcp.aggregate_tools()
                        for _t in _mcp_tools:
                            _tools.append(f"mcp_{_t['name']}")
                    except Exception as e:
                        log.warning("mcp.aggregate_tools_failed: %r", e)
                # Vision: switch to VLM when image attachments are present
                _turn_model = self.default_model
                vision_enabled = _flag_enabled(self._vision_config.get("enabled"))
                if vision_enabled and (env.attachments or []):
                    _vlm = self._vision_config.get("model") or ""
                    if _vlm:
                        _turn_model = _vlm
                        log.info(
                            "vision.model_switch attachments=%d model=%s",
                            len(env.attachments or []),
                            _vlm,
                        )
                        self.capabilities.mark_ok("vision")
                    else:
                        self.capabilities.mark_degraded("vision", "no model configured")
                elif vision_enabled:
                    self.capabilities.mark_ok("vision")
                resp, trace, rounds, _task_state = await agent_loop.run_agent_loop(
                    gateway=self.gateway,  # type: ignore[arg-type]
                    model=_turn_model,
                    system_prompt=rendered.system,
                    user_prompt=rendered.user,
                    allowed_tools=_tools,
                    sender_tier=env.sender.tier,
                    event_log=self.events,
                    on_delta=on_delta,
                    prior_messages=prior,
                    correction_gate=correction_gate,
                    stop_check=lambda: chan_id in self._stop_channels,
                    reasoning_effort=self.thinking_effort,
                    reasoning_output=self.reasoning_output,
                    max_rounds=self.max_tool_rounds,
                    weak_model_boost=self.weak_model_boost,
                    response_length=self.response_length,
                    tool_activity=self.tool_activity,
                    output_verifier=output_verifier,
                    self_model=self_model,
                    active_inference=active_inference,
                    metacognitive=metacognitive,
                    best_of_n=best_of_n,
                    failover_models=self.failover_models,
                    mcp_client=getattr(self, "mcp_client", None),
                    memory_root=self._memory_root,
                    initial_task_state=_restored_task_state,
                    initial_task_status=_restored_prior_status,
                    defer_outcome_recording=True,
                )
        except Exception as e:  # noqa: BLE001
            log.exception("agent.turn_failed")
            self.metrics.brain_errors.labels(where="agent_loop").inc()
            self.metrics.agent_turn_failures.inc()
            await self.events.append(
                "turn_failure", {"error": str(e)[:500], "model": self.default_model}
            )
            await self.events.append("error", {"where": "agent_loop", "err": repr(e)})
            user_msg = _turn_failure_message(e)
            # Clean up typing/placeholder if they were created.
            if typing_keep is not None:
                try:
                    await typing_keep.stop(timeout=1.5)
                except Exception:  # noqa: BLE001
                    pass
            if streaming_msg is not None:
                try:
                    await streaming_msg.finalize(user_msg)
                except Exception:  # noqa: BLE001
                    pass
            elif env.source == "discord" and self.discord is not None:
                # No streaming message was set up — send the failure reply
                # directly to the channel so the user isn't left in silence.
                try:
                    ch_id = (env.raw or {}).get("channel_id")
                    if ch_id:
                        await self.discord.send(str(ch_id), user_msg)
                except Exception:  # noqa: BLE001
                    log.debug("agent.turn_failed_reply_send_failed", exc_info=True)
            else:
                # HTTP, Agent OS, and any future ingress adapters must receive
                # the same terminal failure signal.  Previously this branch
                # only logged the exception, leaving non-Discord callers
                # waiting for a response that could never arrive.
                await self._send_command_reply(env, user_msg, reply_to=env.message_id)
                return
            await self._mirror_discord_reply(env, user_msg)
            return

        # Stop typing as soon as we have a final response — finalize will
        # send/edit the actual content next.
        if typing_keep is not None:
            try:
                await typing_keep.stop(timeout=1.5)
            except Exception:  # noqa: BLE001
                pass

        # Record tool calls + results in the rolling window so future turns
        # have agentic context (what actions were taken and their outcomes).
        if trace:
            # Successful real tool use is stronger freshness evidence than a
            # synthetic heartbeat. This is an in-memory dictionary update and
            # does not add another browser, desktop, network, or sandbox probe.
            _refresh_tool_capability_evidence(self.capabilities, trace)
            for _t_idx, t in enumerate(trace[:20]):  # cap at 20 to preserve more agentic context
                call_id = f"fc_{turn_id}_{_t_idx}"
                import json as _json

                window.add_tool_call(
                    call_id=call_id,
                    content=_json.dumps(t.get("args", {}), default=str)[:800],
                    turn_id=turn_id,
                    name=t["name"],
                )
                result_preview = _json.dumps(t.get("result", {}), default=str)[:1200]
                window.add_tool_result(
                    call_id=call_id,
                    content=result_preview,
                    turn_id=turn_id,
                )

        # Deliver as soon as generation finishes. Window/checkpoint persistence
        # and telemetry happen afterward so local disk latency is never added to
        # the visible response tail.
        if resp is not None:
            try:
                delivery_outcome = await self._deliver_turn_response(
                    env=env,
                    ctx=ctx,
                    resp=resp,
                    trace=trace,
                    rounds=rounds,
                    streaming_msg=streaming_msg,
                    stream_delta_buf=stream_delta_buf,
                    target_channel=target_channel,
                )
            except Exception as error:  # noqa: BLE001
                log.warning("reply.delivery_pipeline_failed error=%r", error, exc_info=True)
                delivery_outcome = {
                    "ok": False,
                    "state": "delivery_pipeline_failed",
                    "attempted": True,
                    "error": f"{type(error).__name__}: {error}"[:500],
                }
            resp.raw = dict(resp.raw or {})
            resp.raw["delivery_succeeded"] = delivery_outcome.get("ok") is True
            resp.raw["delivery_state"] = delivery_outcome.get("state", "unknown")
            if delivery_outcome.get("ok") is not True:
                self.metrics.agent_turn_failures.inc()

            # A bounded delivery attempt is complete. Persistence and learning
            # must not retain scarce global execution capacity or typing state.
            self._release_current_turn_slot()
            if owner_task is not None:
                self._active_stream_messages.pop(owner_task, None)
                delivered_typing = self._active_typing_handles.pop(owner_task, None)
                if delivered_typing is not None:
                    try:
                        await delivered_typing.stop(timeout=1.5)
                    except Exception:  # noqa: BLE001
                        log.debug("typing.delivery_cleanup_failed", exc_info=True)

            # Only an actually delivered assistant message enters conversation
            # history. Suppressed or failed output must not become false shared
            # context on the next turn.
            if delivery_outcome.get("state") == "delivered" and resp.content:
                _clean = resp.content
                try:
                    from ..adapter.reply_tag import parse_reply_tag

                    _clean = parse_reply_tag(_clean).text
                except Exception:  # noqa: BLE001
                    pass
                _clean = _clean_historical_assistant_text(_clean)
                if _clean.strip():
                    window.add_assistant(_clean, turn_id=turn_id)

            # Persist history after delivery without blocking unrelated channel
            # work on serialization or filesystem I/O.
            try:

                def compact_and_persist_window() -> None:
                    window.compact_solved_turns()
                    self._persist_window(chan_id)

                await asyncio.to_thread(compact_and_persist_window)
            except Exception as error:  # noqa: BLE001
                log.warning("window_compaction.failed channel=%s error=%r", chan_id, error)

            # The checkpoint records delivery truth plus a bounded, hashed
            # response preview for operator/restart recovery when routing failed.
            try:
                from ..runtime.checkpoint import save_checkpoint

                _cp_status = (
                    _checkpoint_status(_task_state, resp)
                    if delivery_outcome.get("ok") is True
                    else "delivery_failed"
                )
                await asyncio.to_thread(
                    save_checkpoint,
                    channel=chan_id,
                    turn_id=str(turn_id),
                    messages=prior,
                    trace=trace or [],
                    task_state=_task_state.to_persistable_dict() if _task_state else None,
                    rounds=rounds,
                    model=resp.model or self.default_model,
                    status=_cp_status,
                    response={
                        "content_preview": (resp.content or "")[:2_000],
                        "content_chars": len(resp.content or ""),
                        "content_sha256": hashlib.sha256((resp.content or "").encode()).hexdigest(),
                        "model": resp.model or self.default_model,
                        "request_id": resp.request_id,
                    },
                    delivery=delivery_outcome,
                    memory_root=self._memory_root,
                )
            except Exception as error:  # noqa: BLE001
                log.warning(
                    "checkpoint.save_failed channel=%s turn=%s error=%r",
                    chan_id,
                    turn_id,
                    error,
                )

            try:
                raw_meta = resp.raw if isinstance(resp.raw, dict) else {}
                used_multi_agent = raw_meta.get("multi_agent") is True
                used_orchestrator = raw_meta.get("orchestrator") is True
                path = (
                    "multi_agent"
                    if used_multi_agent
                    else "orchestrator"
                    if used_orchestrator
                    else "agent_loop"
                )
                await self.events.append(
                    "turn_telemetry",
                    {
                        "turn_id": str(turn_id),
                        "channel": chan_id,
                        "path": path,
                        "model": resp.model or self.default_model,
                        "rounds": rounds,
                        "tools_used": [t["name"] for t in (trace or [])[:10]],
                        "multi_agent": used_multi_agent,
                        "planner": raw_meta.get("planner", ""),
                        "executor": raw_meta.get("executor", ""),
                        "content_len": len(resp.content or ""),
                        "delivered": delivery_outcome.get("ok") is True,
                        "delivery_state": delivery_outcome.get("state", "unknown"),
                    },
                )
            except Exception:  # noqa: BLE001
                pass

        # Agent-loop learning and aggregate trajectory persistence are evidence
        # sinks, not acceptance gates. Keep them after delivery so a slow disk
        # cannot hold a completed answer hostage.
        if resp is not None and _task_state is not None:
            try:
                raw_verifier_score = (resp.raw or {}).get("output_verifier_score")
                verifier_score = (
                    float(raw_verifier_score)
                    if isinstance(raw_verifier_score, (int, float))
                    and not isinstance(raw_verifier_score, bool)
                    else None
                )
                await agent_loop.record_agent_outcome(
                    event_log=self.events,
                    task_state=_task_state,
                    trace=trace,
                    final_resp=resp,
                    rounds=rounds,
                    model=resp.model or self.default_model,
                    user_text=env.body or "",
                    memory_root=self._memory_root,
                    self_model=self_model,
                    active_inference=active_inference,
                    metacognitive=metacognitive,
                    verifier_score=verifier_score,
                )
            except Exception:  # noqa: BLE001
                log.warning("agent_outcome.record_failed", exc_info=True)

        # Post-action hooks: brain learns from every interaction.
        if resp is not None:
            self._turn_count = getattr(self, "_turn_count", 0) + 1
            trace = trace or []

            # Phase 2: Trace collector — record turn for observability UI
            try:
                from ..observability.trace_ui import get_trace_collector

                trace_collector = get_trace_collector()
                trace_turn_id = f"turn-{self._turn_count}-{env.message_id[:8]}"
                # Record tool calls
                for t in (trace or [])[:20]:
                    await trace_collector.record_event(
                        turn_id=trace_turn_id,
                        event_type="tool_call",
                        data={
                            "tool": t["name"],
                            "args": t.get("args", {}),
                            "result": t.get("result", {}),
                            "ok": _explicit_result_ok(t.get("result")),
                        },
                    )
                # Record memory hits
                for item in (ctx.memory.items or [])[:10]:
                    if isinstance(item, tuple) and len(item) >= 2:
                        n = item[0]
                        await trace_collector.record_event(
                            turn_id=trace_turn_id,
                            event_type="memory_retrieval",
                            data={
                                "neuron_id": getattr(n, "entity_id", ""),
                                "kind": getattr(n, "kind", ""),
                                "score": float(item[1]) if len(item) > 1 else 0.0,
                            },
                        )
                await trace_collector.end_turn(trace_turn_id, rounds=rounds)
            except Exception as e:  # noqa: BLE001
                log.debug("trace_collector.failed error=%r", e)

            from ..brain.strong_model_scaffold import tool_result_succeeded

            executed_trace = [
                item
                for item in trace
                if isinstance(item.get("result"), dict)
                and item["result"].get("_not_executed") is not True
                and item["result"].get("_cached") is not True
            ]

            def _outcome_key(item: dict[str, Any]) -> str:
                name = str(item.get("name") or "tool")
                raw_args = item.get("args")
                args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
                action = str(args.get("action") or "").strip().lower()
                return f"{name}:{action}" if action else name

            scored_outcomes = [
                (
                    _outcome_key(item),
                    10.0
                    if tool_result_succeeded(
                        str(item.get("name") or ""),
                        item["result"],
                        item.get("args") if isinstance(item.get("args"), dict) else {},
                    )
                    else 0.0,
                )
                for item in executed_trace
            ]
            action_summary = ", ".join(key for key, _ in scored_outcomes[:5])
            score = (
                sum(outcome for _, outcome in scored_outcomes) / len(scored_outcomes)
                if scored_outcomes
                else 0.0
            )
            rpe_result = None
            try:
                if scored_outcomes:
                    # VTA and action history learn only from calls that really
                    # executed. One turn is persisted as one batch.
                    rpe_result = self._vta.record_outcome(action_summary, score)
                    self._basal_ganglia.record_outcomes(scored_outcomes)

                if self._memory_store is not None and rpe_result is not None:
                    from ..memory.vta_writer import build_turn_summary, write_outcome

                    summary = build_turn_summary(
                        user_input=env.body or "",
                        response_preview=resp.content or "",
                        tool_calls=len(trace),
                        rounds=rounds,
                    )
                    thalamic = ctx.metadata.get("thalamic", {})
                    from ..memory.hot_inject import post_turn_hot_maintenance

                    role = os.environ.get("NORAX_ROLE", "production")
                    memory_root = self._memory_store.root
                    focus_summary = ctx.focus.summary or ""
                    user_body = env.body or ""
                    message_type = str(thalamic.get("type", ""))
                    pathway = str(thalamic.get("pathway", ""))
                    turn_count = self._turn_count

                    def persist_hot_turn() -> None:
                        write_outcome(
                            memory_root=memory_root,
                            route=rpe_result.route,
                            text=summary,
                            source="vta",
                        )
                        post_turn_hot_maintenance(
                            memory_root,
                            focus_summary=focus_summary,
                            user_body=user_body,
                            msg_type=message_type,
                            pathway=pathway,
                            turn_count=turn_count,
                            role=role,
                        )

                    # These bounded disk writes occur after reply delivery but
                    # still must not stall unrelated channels on the event loop.
                    await asyncio.to_thread(persist_hot_turn)
                    if self._memory_coordinator is not None:
                        self._memory_coordinator.canonical_changed("post_turn")
                    # Bound index staleness while coalescing writes between syncs.
                    if self._turn_count % 10 == 0:
                        try:
                            if self._memory_coordinator is not None:
                                await self._memory_coordinator.sync_projections()
                        except Exception:
                            log.debug("memory_projection_sync.post_turn_failed", exc_info=True)
                    await self.events.append(
                        "vta_write",
                        {
                            "route": rpe_result.route,
                            "rpe": round(rpe_result.rpe, 2),
                            "reason": rpe_result.reason,
                        },
                    )
            except Exception:  # noqa: BLE001
                log.warning("post_action.failed", exc_info=True)

        # Sprint B: Hebbian strengthening — co-fired neurons get weight boost
        if resp is not None and ctx.memory.items:
            try:
                retrieved_neurons = []
                for item in ctx.memory.items:
                    if isinstance(item, tuple) and len(item) >= 2:
                        n = item[0]
                        if hasattr(n, "entity_id"):
                            retrieved_neurons.append(n)
                if len(retrieved_neurons) >= 2:
                    self._hebbian.record_cofiring(retrieved_neurons)
                    # Flush to disk every 10 turns
                    if self._hebbian.turn_count % 10 == 0:
                        if self._memory_store is not None:
                            self._hebbian.flush(self._memory_store.root)
            except Exception:  # noqa: BLE001
                log.warning("hebbian.failed", exc_info=True)

        # Sprint C: Record episode to episodic buffer
        if resp is not None and self._episodic is not None:
            try:
                import hashlib as _hl

                retrieval_eids = []
                for item in ctx.memory.items or []:
                    if isinstance(item, tuple) and len(item) >= 2:
                        n = item[0]
                        if hasattr(n, "entity_id"):
                            retrieval_eids.append(n.entity_id)

                arousal_data = ctx.metadata.get("arousal")
                arousal_level = (
                    arousal_data.get("level", "normal")
                    if isinstance(arousal_data, dict)
                    else getattr(arousal_data, "level", "normal")
                    if arousal_data
                    else "normal"
                )

                # Get VTA result if we computed it
                vta_rpe = 0.0
                vta_route = "scratchpad"
                if resp is not None and rpe_result is not None:
                    # Re-use the rpe_result from above if available
                    try:
                        vta_rpe = rpe_result.rpe
                        vta_route = rpe_result.route
                    except NameError:
                        pass

                episode_outcomes = [
                    10.0
                    if tool_result_succeeded(
                        str(item.get("name") or ""),
                        item.get("result") if isinstance(item.get("result"), dict) else {},
                        item.get("args") if isinstance(item.get("args"), dict) else {},
                    )
                    else 0.0
                    for item in trace
                    if isinstance(item.get("result"), dict)
                    and item["result"].get("_not_executed") is not True
                    and item["result"].get("_cached") is not True
                ]
                score = sum(episode_outcomes) / len(episode_outcomes) if episode_outcomes else 5.0
                raw_outcome = resp.raw if isinstance(resp.raw, dict) else {}
                verified_episode = raw_outcome.get("verified_outcome") is True
                delivery_episode = raw_outcome.get("delivery_succeeded") is True
                objective_value = raw_outcome.get("objective_outcome_observed")
                objective_episode = (
                    objective_value is True if objective_value is not None else verified_episode
                )

                episode = Episode(
                    user_input=(env.body or "")[:200],
                    user_input_hash=_hl.sha256((env.body or "").encode()).hexdigest()[:12],
                    retrieval_hits=retrieval_eids[:20],
                    tool_calls=[
                        {
                            "name": t["name"],
                            "ok": tool_result_succeeded(
                                str(t.get("name") or ""),
                                (t.get("result") if isinstance(t.get("result"), dict) else {}),
                                t.get("args") if isinstance(t.get("args"), dict) else {},
                            ),
                        }
                        for t in trace[:15]
                    ],
                    response_preview=(resp.content or "")[:200],
                    arousal_level=arousal_level,
                    vta_rpe=vta_rpe,
                    vta_route=vta_route,
                    model=resp.model or self.default_model,
                    rounds=rounds,
                    task_type=getattr(_task_state, "task_type", "") if _task_state else "",
                    outcome_score=score,
                    accepted_outcome=(
                        delivery_episode
                        and (
                            raw_outcome.get("accepted_outcome") is True
                            if raw_outcome.get("accepted_outcome") is not None
                            else verified_episode
                        )
                    ),
                    verified_outcome=verified_episode,
                    objective_outcome_observed=objective_episode,
                    training_eligible=(
                        delivery_episode
                        and (
                            raw_outcome.get("training_eligible") is True
                            if raw_outcome.get("training_eligible") is not None
                            else objective_episode
                        )
                    ),
                )
                self._episodic.record(episode)
            except Exception:  # noqa: BLE001
                log.warning("episodic.record.failed", exc_info=True)

        # Curiosity engine: record this turn's fingerprint so future novelty
        # scoring has this input in its rolling window.
        if resp is not None and curiosity is not None:
            try:
                curiosity.record(env.body or "", domain=_turn_domain)
            except Exception:  # noqa: BLE001
                log.debug("curiosity.record.failed", exc_info=True)

        # Sprint D: record turn to UserModel + trigger SkillLearner
        if resp is not None and self._user_model is not None:
            try:
                tool_names = [t["name"] for t in trace[:20]] if trace else []
                self._user_model.record_turn(
                    user_id=env.sender.id,
                    label=env.sender.label,
                    tier=env.sender.tier,
                    body=env.body or "",
                    response_preview=resp.content or "",
                    tool_calls=tool_names,
                    model=resp.model or self.default_model,
                    command_name=getattr(pc, "name", "") if pc else "",
                )
            except Exception:  # noqa: BLE001
                log.warning("user_model.record.failed", exc_info=True)

        # Sprint E: Feed CausalGraph + TemporalGraph post-turn
        if (
            self._memory_store is not None
            and self._causal_graph is not None
            and self._temporal_graph is not None
        ):
            try:
                async with self._graph_update_lock:
                    causal_graph = self._causal_graph
                    if trace:
                        causal_graph.ingest_trajectory(
                            {
                                "trace": trace,
                                "trajectory_id": turn_id,
                                "timestamp": time.time(),
                            }
                        )
                    # TemporalGraph: record sequence of retrieved neuron IDs
                    temporal_graph = self._temporal_graph
                    if ctx.memory.items:
                        retrieved_ids = []
                        for item in ctx.memory.items[:30]:
                            if isinstance(item, tuple) and len(item) >= 2:
                                n = item[0]
                                if hasattr(n, "entity_id"):
                                    retrieved_ids.append(n.entity_id)
                        session_id = str(
                            (env.raw or {}).get("channel_id") or env.channel or "default"
                        )
                        for nid in retrieved_ids:
                            temporal_graph.record_access(nid, session_id=session_id)
                        if len(retrieved_ids) >= 2:
                            temporal_graph.record_sequence(retrieved_ids, session_id=session_id)
                    # Debounced persist: full JSON rewrites are ~14MB, so save at
                    # most once a minute; shutdown() flushes the tail. Serialize
                    # in worker threads while this async lock prevents a later
                    # post-turn mutation from racing the snapshots.
                    now = time.monotonic()
                    if now - self._last_graph_save >= 60.0:
                        await asyncio.gather(
                            asyncio.to_thread(causal_graph.save),
                            asyncio.to_thread(temporal_graph.save),
                        )
                        self._last_graph_save = now
            except Exception:  # noqa: BLE001
                log.warning("causal_temporal.record.failed", exc_info=True)

        if resp is not None:
            # A tool may request this runtime's own restart/stop. Executing it
            # inside the tool round used to kill the process before Discord
            # finalized long responses, leaving a permanent
            # "[…response continues]" preview. The shell tool now records the
            # intent; execute it only after the complete reply is delivered.
            lifecycle_action = _deferred_runtime_lifecycle_action(trace)
            if lifecycle_action is not None:
                await self.events.append(
                    "runtime_lifecycle_deferred",
                    {
                        "action": lifecycle_action,
                        "after_reply": True,
                        "tool_calls": len(trace),
                    },
                )
                cmd = [
                    "systemctl",
                    "--user",
                    lifecycle_action,
                    "--no-block",
                    "norax-ai.service",
                ]
                log.warning("runtime.lifecycle_after_reply: %s", " ".join(cmd))
                try:
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("runtime.lifecycle_after_reply_failed")
