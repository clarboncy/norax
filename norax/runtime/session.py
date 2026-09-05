"""Per-channel context windows, cancellation, and command dispatch."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import commands as cmd_mod
from ..context.window import RollingWindow
from ._mixin import RuntimeAccessMixin

log = logging.getLogger("norax.runtime.core")


class SessionMixin(RuntimeAccessMixin):
    _turn_work_count: int
    _windows: dict[str, RollingWindow]

    def _window_path(self, channel_id: str) -> Path:
        """Disk location for persisted rolling window."""
        mem_root = getattr(self.cfg, "memory_root", None)
        base = (
            mem_root / "state" / "windows" if isinstance(mem_root, Path) else Path("state/windows")
        )
        # Sanitize channel_id to a filename-safe token.
        if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", channel_id):
            safe = channel_id
        else:
            slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in channel_id)[:64]
            digest = hashlib.sha256(channel_id.encode("utf-8")).hexdigest()[:16]
            safe = f"{slug or 'channel'}-{digest}"
        return base / f"{safe}.json"

    def _get_window(self, channel_id: str) -> RollingWindow:
        if channel_id not in self._windows:
            # Try to load persisted window from disk.
            path = self._window_path(channel_id)
            self._windows[channel_id] = RollingWindow.load(path, budget_tokens=256_000)
            self._windows[channel_id].protect_tail_turns = 8
        # Per-model window budget: profiles with `window_tokens` pin the
        # rolling window to that model's verified max context (e.g.
        # qwen3.8-27b-fast @ 80k) so the window never evicts before the
        # model's context would fill. Other models keep the 256k default.
        try:
            from ..gateway_client.ollama_profiles import resolve_profile

            prof = resolve_profile(getattr(self, "_effective_model", "") or "")
            wt = int(getattr(prof, "window_tokens", 0) or 0)
            self._windows[channel_id].budget_tokens = wt if wt else 256_000
        except Exception:  # noqa: BLE001
            pass
        return self._windows[channel_id]

    def _persist_window(self, channel_id: str) -> None:
        """Save the channel's window to disk. Fails quietly."""
        w = self._windows.get(channel_id)
        if w is None:
            return
        try:
            w.save(self._window_path(channel_id))
        except Exception as e:  # noqa: BLE001
            log.warning("window_persist.failed channel=%s err=%r", channel_id, e)

    def _reset_window(self, channel_id: str) -> bool:
        existed = channel_id in self._windows
        self._windows.pop(channel_id, None)
        # Also delete the persisted file.
        try:
            p = self._window_path(channel_id)
            if p.exists():
                p.unlink()
        except Exception as e:  # noqa: BLE001
            log.warning("window_reset.unlink_failed err=%r", e)
        return existed

    def _dump_window(self, channel_id: str, half: bool = False) -> dict | None:
        """Dump this channel's rolling body context to sleep/.

        The bootstrap/system head is preserved. Full mode clears all body
        frames; half mode clears the older 50% of body frames.
        """
        w = self._get_window(channel_id)
        before_frames = len(w.body)
        before_tokens = w.total_tokens()
        result = w.dump_to_sleep(half=half)
        if result is None:
            return None
        self._persist_window(channel_id)
        return {
            "mode": "half" if half else "full",
            "frames": len(result.evicted),
            "tokens_freed": result.tokens_freed,
            "spill_path": str(result.spill_path) if result.spill_path else None,
            "before_frames": before_frames,
            "remaining_frames": len(w.body),
            "before_tokens": before_tokens,
            "after_tokens": w.total_tokens(),
        }

    def _cancel_channel(self, channel_id: str = "") -> bool:
        """Signal any active turn on this channel to stop."""
        if not channel_id:
            active_tasks = [
                (cid, task) for cid, task in self._active_turn_tasks.items() if not task.done()
            ]
            for cid, task in active_tasks:
                self._stop_channels.add(cid)
                if self._active_turn_tasks.get(cid) is task:
                    self._active_turn_tasks.pop(cid, None)
                queue = self._turn_queues.pop(cid, None)
                if queue:
                    self._turn_work_count = max(0, self._turn_work_count - len(queue))
                    queue.clear()
                task.cancel()
            return bool(active_tasks)
        active_task = self._active_turn_tasks.get(channel_id)
        if active_task is not None and not active_task.done():
            self._stop_channels.add(channel_id)
            if self._active_turn_tasks.get(channel_id) is active_task:
                self._active_turn_tasks.pop(channel_id, None)
            queue = self._turn_queues.pop(channel_id, None)
            if queue:
                self._turn_work_count = max(0, self._turn_work_count - len(queue))
                queue.clear()
            active_task.cancel()
            return True
        if active_task is not None and self._active_turn_tasks.get(channel_id) is active_task:
            self._active_turn_tasks.pop(channel_id, None)
        self._stop_channels.discard(channel_id)
        return False

    def _get_window_stats(self, channel_id: str) -> dict | None:
        """Return context window stats for a channel."""
        w = self._windows.get(channel_id)
        if w is None:
            return None
        total = w.total_tokens()
        budget = w.budget_tokens
        head = w.head_tokens()
        body_count = len(w.body)
        return {
            "current_tokens": total,
            "budget_tokens": budget,
            "head_tokens": head,
            "body_frames": body_count,
            "usage_pct": round(total / max(budget, 1) * 100, 1),
        }

    def _get_brain_stats(self) -> dict[str, Any] | None:
        """Return brain subsystem stats."""
        stats: dict[str, Any] = {}
        if self._memory_store is not None:
            stats["memory"] = {
                "hot": len(self._memory_store.hot),
                "canonical": len(self._memory_store.all_canonical()),
                "sleep": len(self._memory_store.sleep),
            }
        if self._episodic is not None:
            stats["episodic"] = self._episodic.stats()
        if hasattr(self, "_hebbian"):
            stats["hebbian_turns"] = self._hebbian.turn_count
        return stats

    def _build_runtime_handle(self) -> cmd_mod.RuntimeHandle:
        prov_urls = getattr(self.gateway, "provider_urls", None)
        return cmd_mod.RuntimeHandle(
            default_model=self.default_model,
            set_default_model=self.set_default_model,
            started_at=self._started_mono,
            wall_started_at=self._started_wall,
            metrics=self.metrics,
            gateway_base_url=self.gateway.base_url,
            event_log=self.events,
            reset_window_for=self._reset_window,
            dump_window_for=self._dump_window,
            outbound=self.outbound,
            provider_urls=prov_urls,
            gateway=self.gateway,
            get_window_stats=self._get_window_stats,
            get_brain_stats=self._get_brain_stats,
            cancel_stream=self._cancel_channel,
            thinking_effort=self.thinking_effort,
            set_thinking_effort=self.set_thinking_effort,
            reasoning_output=self.reasoning_output,
            set_reasoning_output=self.set_reasoning_output,
            planning_mode=self.planning_mode,
            set_planning_mode=self.set_planning_mode,
            max_tool_rounds=self.max_tool_rounds,
            set_max_tool_rounds=self.set_max_tool_rounds,
            memory_depth=self.memory_depth,
            set_memory_depth=self.set_memory_depth,
            weak_model_boost=self.weak_model_boost,
            set_weak_model_boost=self.set_weak_model_boost,
            stream_replies=self.stream_replies,
            set_stream_replies=self.set_stream_replies,
            response_length=self.response_length,
            set_response_length=self.set_response_length,
            tool_activity=self.tool_activity,
            set_tool_activity=self.set_tool_activity,
            custom_model_catalog=self.custom_model_catalog,
        )

    async def _handle_command(self, env, pc: cmd_mod.ParsedCommand) -> None:
        """Run a slash command and emit its reply via the source channel."""
        rt = self._build_runtime_handle()
        try:
            result = await cmd_mod.handle(pc, env, rt)
        except Exception as e:  # noqa: BLE001
            log.exception("command.%s failed", pc.name)
            self.metrics.brain_errors.labels(where=f"cmd.{pc.name}").inc()
            await self.events.append(
                "cmd.error",
                {"name": pc.name, "err": repr(e), "user": env.sender.id},
            )
            result = cmd_mod.CommandResult(reply=f"Command `/{pc.name}` failed: {e}")

        await self.events.append(
            "cmd",
            {
                "name": pc.name,
                "args": pc.args,
                "user": env.sender.id,
                "tier": env.sender.tier,
                "channel_id": (env.raw or {}).get("channel_id"),
                "reply_preview": (result.reply or "")[:200],
                "post_send": result.post_send is not None,
            },
        )

        # Route the reply via the same outbound path as normal replies.
        delivered = await self._send_command_reply(
            env, result.reply, reply_to=None if pc.name == "status" else env.message_id
        )

        # Post-send effects (shutdown, restart) run after the reply is out.
        if result.post_send is not None and delivered:
            try:
                await result.post_send()
            except Exception:  # noqa: BLE001
                log.exception("command.post_send failed: %s", pc.name)
        elif result.post_send is not None:
            log.error("command.post_send suppressed because reply delivery failed: %s", pc.name)

    async def _slash_dispatch(
        self,
        name: str,
        args: dict,
        principal,
        ctx: dict,
    ) -> dict:
        """Handle a native Discord slash-command invocation.

        Returns a dict with:
            reply: str        — text to send as interaction followup
            post_send: Optional[awaitable] — coroutine to run after reply
        """
        from ..envelope import SensoryInput

        env = SensoryInput(
            channel="chat",
            source="discord",
            message_id=str(ctx.get("interaction_id") or ""),
            timestamp=datetime.now(UTC),
            sender=principal,
            body=f"/{name}",
            raw={
                "channel_id": ctx.get("channel_id"),
                "guild_id": ctx.get("guild_id"),
                "slash_command": True,
            },
            trusted=bool(principal.trust),
            metadata={},
        )

        # Reuse the text-mode ParsedCommand + handle() path.
        pc_args = args.get("args") or []
        # Drop None entries (optional params that weren't supplied).
        pc_args = [a for a in pc_args if a is not None and a != ""]
        pc = cmd_mod.ParsedCommand(name=name, args=list(pc_args), raw=f"/{name}")

        rt_handle = self._build_runtime_handle()
        try:
            result = await cmd_mod.handle(pc, env, rt_handle)
        except Exception as e:  # noqa: BLE001
            log.exception("slash_dispatch.%s failed", name)
            self.metrics.brain_errors.labels(where=f"slash.{name}").inc()
            return {"reply": f"Command `/{name}` failed: {e}", "post_send": None}

        await self.events.append(
            "cmd",
            {
                "name": name,
                "args": pc_args,
                "user": principal.id,
                "tier": principal.tier,
                "source": "slash",
                "channel_id": ctx.get("channel_id"),
                "reply_preview": (result.reply or "")[:200],
                "post_send": result.post_send is not None,
            },
        )
        return {
            "reply": result.reply,
            "post_send": result.post_send,
            "data": result.data,
        }
