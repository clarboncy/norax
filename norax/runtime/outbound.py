"""OutboundRegistry — resolves channel name → adapter with a send() method.

Used by:
  - `norax.runtime.core.Runtime._handle_turn` to dispatch assistant replies
  - `norax.dispatch.tools.t_message_send` (injected via factory)

An outbound adapter is any object exposing:

    async def send(
        self,
        target: str,
        text: str,
        *,
        reply_to: str | None = None,
        react_to: str | None = None,
        emoji: str | None = None,
        files: list[str] | None = None,
        **extra: Any,
    ) -> dict

Contract:
  - Returns a dict {ok: bool, ...}. On success: {"ok": True, "message_id": "..."}.
  - Never raises on provider errors — catches and returns {"ok": False, "error": ...}.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

log = logging.getLogger("norax.runtime.outbound")


class OutboundAdapter(Protocol):
    async def send(
        self,
        target: str,
        text: str,
        *,
        reply_to: str | None = None,
        react_to: str | None = None,
        emoji: str | None = None,
        files: list[str] | None = None,
        **extra: Any,
    ) -> dict: ...


class OutboundRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, OutboundAdapter] = {}

    def register(self, channel: str, adapter: OutboundAdapter) -> None:
        if channel in self._adapters:
            log.warning("outbound.register overwriting channel=%s", channel)
        self._adapters[channel] = adapter

    def unregister(
        self,
        channel: str,
        *,
        expected: OutboundAdapter | None = None,
    ) -> bool:
        """Remove a channel registration, optionally only for one adapter.

        The identity check prevents a shutting-down optional connector from
        deleting a newer registration that replaced it during reconfiguration.
        """
        current = self._adapters.get(channel)
        if current is None or (expected is not None and current is not expected):
            return False
        del self._adapters[channel]
        return True

    def get(self, channel: str) -> OutboundAdapter | None:
        return self._adapters.get(channel)

    def has(self, channel: str) -> bool:
        return channel in self._adapters

    def channels(self) -> list[str]:
        return list(self._adapters)

    async def send(
        self,
        channel: str,
        target: str,
        text: str,
        **kwargs: Any,
    ) -> dict:
        adapter = self._adapters.get(channel)
        if adapter is None:
            log.warning("outbound.send.no_adapter channel=%s", channel)
            return {"ok": False, "error": "channel_not_registered", "channel": channel}
        try:
            return await adapter.send(target, text, **kwargs)
        except Exception as e:  # noqa: BLE001
            log.exception("outbound.send.failed channel=%s target=%s", channel, target)
            return {"ok": False, "error": repr(e), "channel": channel, "target": target}
