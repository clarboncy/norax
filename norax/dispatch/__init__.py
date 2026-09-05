"""Dispatcher — routes tool calls through RiskGate + Budget + LoopGuard + Idempotency.

Call flow:
  dispatch(name, args, caller) →
    1. risk.check(tool, args, sender_tier) → authorize exact operation
    2. scoped idempotency hit/in-flight duplicate? return same result
    3. budget.assess(caller) → enforce explicit guest/red-zone policy
    4. loop_guard.observe(tool, args)
    5. tool.fn(**args) → result
    6. budget.record(caller)
    7. idempotency.put(scoped request_id, result)
    8. return result
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass

from .budget import BudgetEnforcer
from .budget import BudgetExceeded as BudgetExceeded
from .budget import Tier as BudgetTier
from .idempotency import IdempotencyCache, IdempotencyConflict
from .loop_guard import LoopDetected as LoopDetected
from .loop_guard import LoopGuard
from .risk import check as risk_check
from .tools import REGISTRY, normalize_tool_args, normalize_tool_name

log = logging.getLogger("norax.dispatch")


@dataclass
class Caller:
    id: str
    tier: BudgetTier  # budget tier ("owner"/"user"/...)


@dataclass
class DispatchResult:
    ok: bool
    result: dict
    tool: str
    risk_tier: str
    elapsed_ms: float
    budget_zone: str = "green"


class DispatchError(Exception):
    pass


class Dispatcher:
    def __init__(
        self,
        *,
        budget: BudgetEnforcer | None = None,
        loop_guard: LoopGuard | None = None,
        idempotency: IdempotencyCache | None = None,
    ) -> None:
        self.budget = budget or BudgetEnforcer()
        self.loop_guard = loop_guard or LoopGuard()
        self.idempotency = idempotency or IdempotencyCache()
        self._inflight: dict[str, tuple[str, asyncio.Future[dict]]] = {}

    @staticmethod
    def _request_scope(*, tool: str, args: dict, caller: Caller) -> str:
        payload = json.dumps(
            {
                "caller_id": str(caller.id),
                "caller_tier": str(caller.tier),
                "tool": tool,
                "args": args,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_request_id(request_id: str) -> str:
        value = str(request_id).strip()
        if not value:
            raise DispatchError("request_id must not be empty")
        if len(value) > 256:
            raise DispatchError("request_id exceeds 256 characters")
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise DispatchError("request_id contains control characters")
        return value

    @staticmethod
    def _cached_result(*, tool: str, result: dict) -> DispatchResult:
        return DispatchResult(
            ok=result.get("ok") is True,
            result=result,
            tool=tool,
            risk_tier="cached",
            elapsed_ms=0.0,
            budget_zone="cached",
        )

    async def dispatch(
        self,
        *,
        tool: str,
        args: dict,
        caller: Caller,
        request_id: str | None = None,
    ) -> DispatchResult:
        tool = normalize_tool_name(tool)
        spec = REGISTRY.get(tool)
        if spec is None:
            raise DispatchError(f"unknown tool: {tool}")

        args = normalize_tool_args(tool, args)

        # 1. Authorize the exact operation before consulting cached results.
        decision = risk_check(tool=tool, args=args, sender_tier=caller.tier)
        if not decision.allowed:
            raise DispatchError(
                f"blocked: {decision.reason}"
                + (f" (hit={decision.dangerous_hit!r})" if decision.dangerous_hit else "")
            )

        # 2. Completed and concurrent idempotency are scoped to the exact call.
        normalized_request_id: str | None = None
        request_scope: str | None = None
        if request_id is not None:
            normalized_request_id = self._validate_request_id(request_id)
            request_scope = self._request_scope(tool=tool, args=args, caller=caller)
            try:
                cached = self.idempotency.get(normalized_request_id, scope=request_scope)
            except IdempotencyConflict as exc:
                raise DispatchError(str(exc)) from exc
            if cached is not None:
                log.info("dispatch.idempotency_hit tool=%s", tool)
                return self._cached_result(tool=tool, result=cached)
            in_flight = self._inflight.get(normalized_request_id)
            if in_flight is not None:
                existing_scope, future = in_flight
                if existing_scope != request_scope:
                    raise DispatchError("request_id is in flight for a different operation")
                await asyncio.shield(future)
                cached = self.idempotency.get(normalized_request_id, scope=request_scope)
                if cached is None:
                    raise DispatchError("in-flight operation completed without a cached result")
                return self._cached_result(tool=tool, result=cached)

        # 3. budget
        budget_zone = self.budget.assess(caller.id, caller.tier)
        if budget_zone.zone == "hard_block":
            raise BudgetExceeded(
                caller.id,
                budget_zone.limit_kind or "budget",
                budget_zone.used,
                budget_zone.cap,
            )

        if budget_zone.read_only and decision.tier != "T0":
            raise DispatchError(
                f"blocked: explicit daily budget is in read-only zone ({budget_zone.guidance})"
            )

        # 4. loop guard
        self.loop_guard.observe(tool, args)

        result_future: asyncio.Future[dict] | None = None
        if normalized_request_id is not None and request_scope is not None:
            result_future = asyncio.get_running_loop().create_future()
            self._inflight[normalized_request_id] = (request_scope, result_future)

        # 5. call
        t0 = time.monotonic()
        try:
            try:
                result = await spec.fn(**args)
            except TypeError as exc:
                raise DispatchError(f"bad args for {tool}: {exc}") from exc
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            result_dict = result if isinstance(result, dict) else {"ok": True, "result": result}

            # 6. record budget
            self.budget.record(caller.id, requests=1)

            # 7. idempotency store
            if normalized_request_id is not None and request_scope is not None:
                self.idempotency.put(
                    normalized_request_id,
                    result_dict,
                    scope=request_scope,
                )
                if result_future is not None and not result_future.done():
                    result_future.set_result(result_dict)

            return DispatchResult(
                ok=result_dict.get("ok") is True,
                result=result_dict,
                tool=tool,
                risk_tier=decision.tier,
                elapsed_ms=elapsed_ms,
                budget_zone=budget_zone.zone,
            )
        except BaseException as exc:
            if result_future is not None and not result_future.done():
                if isinstance(exc, asyncio.CancelledError):
                    result_future.cancel()
                else:
                    result_future.set_exception(exc)
                    result_future.exception()
            raise
        finally:
            if normalized_request_id is not None and result_future is not None:
                current = self._inflight.get(normalized_request_id)
                if current is not None and current[1] is result_future:
                    self._inflight.pop(normalized_request_id, None)
