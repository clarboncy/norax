"""Outbound delivery, UI streaming, and redacted delivery evidence."""

from __future__ import annotations

import logging
from typing import Any

from ..adapter.reply_tag import parse_reply_tag
from ._mixin import RuntimeAccessMixin
from .validation import _explicit_result_ok

log = logging.getLogger("norax.runtime.core")


class DeliveryMixin(RuntimeAccessMixin):
    async def _append_delivery_event(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """Record post-send telemetry without rewriting delivery truth.

        Once an outbound adapter has accepted a reply, an observability failure
        must not make callers believe delivery failed (or invite a duplicate
        retry).  Event-log health remains independently visible through logs and
        readiness checks.
        """
        try:
            await self.events.append(kind, payload, attrs=attrs)
        except Exception:  # noqa: BLE001
            log.error("%s.delivery_event_append_failed", kind, exc_info=True)

    def _record_outbound_metric(self, *, source: str, delivered: bool) -> None:
        """Keep metrics failures outside the user-visible delivery boundary."""
        try:
            self.metrics.outbound_sends.labels(
                channel=source,
                ok=str(delivered).lower(),
            ).inc()
        except Exception:  # noqa: BLE001
            log.warning("outbound.metric_record_failed source=%s", source, exc_info=True)

    async def _send_command_reply(
        self,
        env,
        text: str,
        *,
        reply_to: str | None = None,
    ) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        target = None
        if env.source == "discord":
            target = (env.raw or {}).get("channel_id")
        if not target and env.thread_binding is not None:
            target = env.thread_binding.thread_id
        if not self.outbound.has(env.source) or not target:
            log.debug("cmd.no_route source=%s target=%s", env.source, target)
            return False
        result = await self.outbound.send(env.source, target, text, reply_to=reply_to)
        delivery_ok = _explicit_result_ok(result)
        self._record_outbound_metric(source=env.source, delivered=delivery_ok)
        await self._append_delivery_event(
            "send",
            {
                "source": env.source,
                "target": target,
                "ok": delivery_ok,
                "result": result,
                "kind": "cmd",
            },
        )
        await self._mirror_discord_reply(env, text)
        return delivery_ok

    async def _mirror_discord_reply(self, env, text: str) -> None:
        """Mirror Discord output to Agent OS without routing Agent OS into Discord."""
        if env.source != "discord" or self.agent_os_bridge is None or not (text or "").strip():
            return
        try:
            await self.agent_os_bridge.broadcast_outbound(text=text.strip())
        except Exception:  # noqa: BLE001
            log.debug("agent_os reply_mirror_failed", exc_info=True)

    async def _deliver_turn_response(
        self,
        *,
        env: Any,
        ctx: Any,
        resp: Any,
        trace: list[dict[str, Any]],
        rounds: int,
        streaming_msg: Any,
        stream_delta_buf: list[str],
        target_channel: str | None,
    ) -> dict[str, Any]:
        """Deliver a completed response and return explicit delivery evidence."""
        usage = resp.usage or {}
        in_tok = int(usage.get("input_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or 0)
        if in_tok:
            self.metrics.gateway_tokens_in.labels(model=resp.model or self.default_model).inc(
                in_tok
            )
        if out_tok:
            self.metrics.gateway_tokens_out.labels(model=resp.model or self.default_model).inc(
                out_tok
            )
        self.metrics.gateway_requests.labels(
            model=resp.model or self.default_model, status="ok"
        ).inc()

        content = (resp.content or "").strip()
        delivery: dict[str, Any]
        if streaming_msg is not None:
            result: dict[str, Any] | None = None
            if content in {"NO_REPLY", "HEARTBEAT_OK"}:
                try:
                    await streaming_msg.delete()
                    delivery = {
                        "ok": True,
                        "state": "intentionally_suppressed",
                        "attempted": False,
                    }
                except Exception:  # noqa: BLE001
                    log.debug("stream.sentinel_delete_failed", exc_info=True)
                    delivery = {
                        "ok": False,
                        "state": "suppression_cleanup_failed",
                        "attempted": True,
                    }
            else:
                if not content:
                    final_text = (
                        f"⚠️ Empty upstream response [{resp.model}] "
                        f"rounds={rounds} tool_calls={len(trace)} "
                        f"streamed_chars={sum(len(x) for x in stream_delta_buf)} "
                        f"usage={resp.usage} raw={str(resp.raw)[:1000]}"
                    )
                    reply_to = env.message_id
                elif content == "." and (resp.model or "").startswith("gemma"):
                    final_text = (
                        f"⚠️ Suspicious single-dot upstream response [{resp.model}] "
                        f"rounds={rounds} tool_calls={len(trace)} "
                        f"streamed_chars={sum(len(x) for x in stream_delta_buf)} "
                        f"usage={resp.usage} raw={str(resp.raw)[:1000]}"
                    )
                    reply_to = env.message_id
                else:
                    parsed = parse_reply_tag(content, current_message_id=env.message_id)
                    final_text = parsed.text
                    reply_to = parsed.reply_to
                try:
                    finalized = await streaming_msg.finalize(final_text)
                    result = (
                        finalized
                        if isinstance(finalized, dict)
                        else {
                            "ok": False,
                            "error": "invalid_stream_finalize_result",
                            "result": finalized,
                        }
                    )
                except Exception:  # noqa: BLE001
                    log.warning("stream.finalize_failed; using direct Discord send", exc_info=True)
                    if target_channel and self.outbound.has(env.source):
                        fallback = await self.outbound.send(
                            env.source,
                            target_channel,
                            final_text,
                            reply_to=reply_to,
                        )
                        result = (
                            fallback
                            if isinstance(fallback, dict)
                            else {
                                "ok": False,
                                "error": "invalid_outbound_result",
                                "result": fallback,
                            }
                        )
                if result is not None:
                    delivery_ok = _explicit_result_ok(result)
                    delivery = {
                        "ok": delivery_ok,
                        "state": "delivered" if delivery_ok else "delivery_failed",
                        "attempted": True,
                        "result": result,
                    }
                    self._record_outbound_metric(source=env.source, delivered=delivery_ok)
                    await self._append_delivery_event(
                        "send",
                        {
                            "source": env.source,
                            "target": target_channel,
                            "ok": delivery_ok,
                            "result": result,
                            "kind": "stream_finalize",
                        },
                    )
                else:
                    delivery = {
                        "ok": False,
                        "state": "unroutable",
                        "attempted": False,
                    }
        else:
            delivery = await self._emit_reply(env, ctx, resp)

        if content and content not in {"NO_REPLY", "HEARTBEAT_OK"}:
            parsed = parse_reply_tag(content, current_message_id=env.message_id)
            await self._mirror_discord_reply(env, parsed.text)

        await self._append_delivery_event(
            "reply",
            {
                "content_preview": (resp.content or "")[:500],
                "usage": resp.usage,
                "rounds": rounds,
                "tool_calls": len(trace),
                "streamed": streaming_msg is not None,
                "delivered": delivery.get("ok") is True,
                "delivery_state": delivery.get("state", "unknown"),
            },
            attrs={
                "gen_ai.response.id": resp.request_id,
                "gen_ai.response.model": resp.model,
                "gen_ai.usage.input_tokens": in_tok,
                "gen_ai.usage.output_tokens": out_tok,
            },
        )
        return delivery

    async def _emit_reply(self, env, ctx, resp) -> dict[str, Any]:
        """Route a brain reply back to the source channel when eligible."""
        if ctx.decision != "emit_reply":
            return {"ok": True, "state": "not_requested", "attempted": False}
        content = (resp.content or "").strip()
        if not content:
            content = (
                f"⚠️ Empty upstream response [{resp.model}] "
                f"usage={resp.usage} raw={str(resp.raw)[:1000]}"
            )
        if content in ("NO_REPLY", "HEARTBEAT_OK"):
            return {
                "ok": True,
                "state": "intentionally_suppressed",
                "attempted": False,
            }

        parsed = parse_reply_tag(content, current_message_id=env.message_id)
        text_out = parsed.text
        reply_to = parsed.reply_to
        if not text_out.strip():
            return {"ok": False, "state": "empty_reply", "attempted": False}

        target = None
        if env.source == "discord":
            target = (env.raw or {}).get("channel_id")
        if not target and env.thread_binding is not None:
            target = env.thread_binding.thread_id

        if not self.outbound.has(env.source):
            log.debug("emit_reply.no_outbound source=%s", env.source)
            return {"ok": False, "state": "no_outbound", "attempted": False}
        if not target:
            log.debug("emit_reply.no_target source=%s msg=%s", env.source, env.message_id)
            return {"ok": False, "state": "no_target", "attempted": False}

        result = await self.outbound.send(env.source, target, text_out, reply_to=reply_to)
        delivery_ok = _explicit_result_ok(result)
        self._record_outbound_metric(source=env.source, delivered=delivery_ok)
        await self._append_delivery_event(
            "send",
            {
                "source": env.source,
                "target": target,
                "ok": delivery_ok,
                "result": result,
            },
        )
        return {
            "ok": delivery_ok,
            "state": "delivered" if delivery_ok else "delivery_failed",
            "attempted": True,
            "result": result,
        }
