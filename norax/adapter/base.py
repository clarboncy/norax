"""Adapter Protocol — every sensory adapter implements this."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from ..envelope import SensoryInput


class Adapter(Protocol):
    name: str

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def events(self) -> AsyncIterator[SensoryInput]: ...
