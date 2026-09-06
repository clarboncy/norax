"""Bounded exponential backoff with jitter.

Usage:
    backoff = ExponentialBackoff(base=1.0, max_delay=60.0, jitter=0.25)
    for attempt in range(max_retries):
        try:
            result = await operation()
            backoff.reset()
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = backoff.next_delay()
            await asyncio.sleep(delay)
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

log = logging.getLogger("norax.runtime.backoff")

T = TypeVar("T")


@dataclass
class ExponentialBackoff:
    """Exponential backoff with jitter.

    delay = min(base * 2^attempt, max_delay) * (1 ± jitter)
    """

    base: float = 1.0
    factor: float = 2.0
    max_delay: float = 60.0
    jitter: float = 0.25  # ±25% jitter
    _attempt: int = 0

    def __post_init__(self) -> None:
        values = (self.base, self.factor, self.max_delay, self.jitter)
        if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
            raise ValueError("backoff parameters must be finite numbers")
        self.base = float(self.base)
        self.factor = float(self.factor)
        self.max_delay = float(self.max_delay)
        self.jitter = float(self.jitter)
        if self.base < 0 or self.factor <= 0 or self.max_delay < 0:
            raise ValueError("base/max_delay must be non-negative and factor must be positive")
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter must be between 0 and 1")

    def next_delay(self) -> float:
        try:
            raw = self.base * (self.factor**self._attempt)
        except OverflowError:
            raw = self.max_delay
        if not math.isfinite(raw):
            raw = self.max_delay
        capped = min(raw, self.max_delay)
        # Symmetric bounded jitter around the capped delay.
        jittered = capped * (1 - self.jitter + random.random() * 2 * self.jitter)
        self._attempt += 1
        return min(self.max_delay, max(0.0, jittered))

    def reset(self) -> None:
        self._attempt = 0

    @property
    def attempt(self) -> int:
        return self._attempt


async def retry_with_backoff[T](
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.25,
    retry_on: type[Exception] | tuple[type[Exception], ...] = Exception,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> T:
    """Retry an async operation with exponential backoff + jitter.

    Args:
        operation: async callable to retry
        max_retries: max number of retry attempts (0 = no retries)
        base: initial delay in seconds
        max_delay: maximum delay cap
        jitter: ±fraction for jitter (0.25 = ±25%)
        retry_on: exception type(s) that should trigger retry
        on_retry: optional callback(attempt, error, delay) called before each retry

    Returns:
        Result of operation on success

    Raises:
        Last exception if all retries exhausted
    """
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    backoff = ExponentialBackoff(base=base, max_delay=max_delay, jitter=jitter)
    attempt = 0
    while True:
        try:
            result = await operation()
            if attempt > 0:
                log.info("retry succeeded attempt=%d", attempt)
            return result
        except retry_on as e:
            if attempt >= max_retries:
                log.warning("retry exhausted attempts=%d error=%s", attempt + 1, str(e)[:100])
                raise
            delay = backoff.next_delay()
            if on_retry:
                try:
                    on_retry(attempt, e, delay)
                except Exception:  # noqa: BLE001
                    log.exception("retry observer failed attempt=%d", attempt)
            log.debug(
                "retry attempt=%d/%d delay=%.2fs error=%s",
                attempt + 1,
                max_retries,
                delay,
                str(e)[:80],
            )
            await asyncio.sleep(delay)
            attempt += 1
