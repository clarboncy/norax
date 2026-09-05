"""Retry with jitter for operations the caller guarantees are idempotent.

The default schedule performs one initial attempt plus retries after roughly
1s, 3s, and 8s (each with up to ±500ms jitter).
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

log = logging.getLogger("norax.observability.retry")

T = TypeVar("T")

DEFAULT_DELAYS_MS = (1000, 3000, 8000)
DEFAULT_JITTER_MS = 500


async def with_retry[T](
    fn: Callable[[], Awaitable[T]],
    *,
    delays_ms: Iterable[int] = DEFAULT_DELAYS_MS,
    jitter_ms: int = DEFAULT_JITTER_MS,
    retriable: tuple[type[BaseException], ...] = (Exception,),
    should_retry: Callable[[BaseException], bool] | None = None,
    op_name: str = "op",
) -> T:
    """Run `fn()` with retries. Caller MUST guarantee idempotency.

    If `should_retry` is provided, it is called with the caught exception;
    returning False causes immediate re-raise without retry.
    """
    try:
        delays = [float(delay) for delay in delays_ms]
        jitter = float(jitter_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("retry delays and jitter must be finite non-negative numbers") from exc
    if (
        any(not math.isfinite(delay) or delay < 0 for delay in delays)
        or not math.isfinite(jitter)
        or jitter < 0
    ):
        raise ValueError("retry delays and jitter must be finite non-negative numbers")
    last_exc: BaseException | None = None
    for attempt in range(len(delays) + 1):
        try:
            return await fn()
        except retriable as e:  # noqa: BLE001
            last_exc = e
            if should_retry and not should_retry(e):
                raise
            if attempt >= len(delays):
                log.warning("retry.exhausted op=%s attempts=%d", op_name, attempt + 1)
                raise
            d = delays[attempt]
            jitter_value = random.uniform(-jitter, jitter)
            wait = max(0.0, (d + jitter_value) / 1000.0)
            log.info(
                "retry.backoff op=%s attempt=%d wait=%.3fs err=%r", op_name, attempt + 1, wait, e
            )
            await asyncio.sleep(wait)
    # unreachable
    assert last_exc is not None
    raise last_exc
