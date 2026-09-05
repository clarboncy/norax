"""Per-caller budget accounting and tool policy.

Design principles (research-backed):
  1. Owner is NEVER budget-limited. Period.
  2. Explicit non-owner caps produce graduated policy signals:
     - GREEN (0-80%):  full capability
     - YELLOW (80-95%): host may trim output or choose a cheaper model
     - RED (95-100%):   Dispatcher permits read-only tools
     - HARD (>100%):    only for guest tier — owner/admin/user never hard-blocked
  3. Budget resets daily at midnight (local).
  4. Throttle state is visible: the agent knows its budget zone and adapts.

Default caps are disabled. ``BudgetExceeded`` is reserved for an explicitly
capped guest at hard block; other tiers receive an inspectable policy zone.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

log = logging.getLogger("norax.dispatch.budget")

Tier = Literal["owner", "admin", "user", "guest"]
Zone = Literal["green", "yellow", "red", "hard_block"]


class BudgetExceeded(Exception):
    """Only raised for guest tier at >100%. Never for owner/admin/user."""

    def __init__(self, caller_id: str, limit_kind: str, used: float, cap: float) -> None:
        super().__init__(
            f"budget.exceeded caller={caller_id} kind={limit_kind} used={used} cap={cap}"
        )
        self.caller_id = caller_id
        self.limit_kind = limit_kind
        self.used = used
        self.cap = cap


@dataclass(frozen=True)
class Caps:
    usd_per_day: float
    input_tokens_per_day: int
    requests_per_day: int

    def __post_init__(self) -> None:
        if isinstance(self.usd_per_day, bool):
            raise ValueError("usd_per_day must be non-negative or positive infinity")
        try:
            usd = float(self.usd_per_day)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("usd_per_day must be non-negative or positive infinity") from exc
        if math.isnan(usd) or usd < 0:
            raise ValueError("usd_per_day must be non-negative or positive infinity")
        object.__setattr__(self, "usd_per_day", usd)
        for name, value in (
            ("input_tokens_per_day", self.input_tokens_per_day),
            ("requests_per_day", self.requests_per_day),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


DEFAULT_CAPS: dict[Tier, Caps] = {
    "owner": Caps(float("inf"), 0, 0),
    "admin": Caps(float("inf"), 0, 0),
    "user": Caps(float("inf"), 0, 0),
    "guest": Caps(float("inf"), 0, 0),
}

# Zone thresholds (fraction of cap consumed)
YELLOW_THRESHOLD = 0.80
RED_THRESHOLD = 0.95


@dataclass
class BudgetZone:
    """Result of assess() — tells the agent how to behave."""

    zone: Zone
    tier: Tier
    utilization: float  # 0.0 - 1.0+, highest across all dimensions
    limit_kind: str | None = None  # which dimension is closest to cap
    used: float = 0.0
    cap: float = 0.0
    guidance: str = ""  # human-readable hint for the agent

    @property
    def can_use_tools(self) -> bool:
        return self.zone != "hard_block"

    @property
    def read_only(self) -> bool:
        return self.zone == "red"

    @property
    def should_downgrade_model(self) -> bool:
        return self.zone in ("yellow", "red")

    @property
    def tool_output_cap(self) -> int | None:
        """Recommended host-side output cap; ``None`` means unrestricted."""
        if self.zone == "green":
            return None
        if self.zone == "yellow":
            return 4000
        if self.zone == "red":
            return 2000
        return 0


@dataclass
class _Usage:
    day: date
    usd: float = 0.0
    input_tokens: int = 0
    requests: int = 0


@dataclass
class BudgetEnforcer:
    caps: dict[Tier, Caps] = field(default_factory=lambda: dict(DEFAULT_CAPS))
    _usage: dict[str, _Usage] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "guest" not in self.caps:
            raise ValueError("budget caps must define the guest fallback tier")
        for tier, caps in self.caps.items():
            if tier not in DEFAULT_CAPS:
                raise ValueError(f"unknown budget tier: {tier}")
            if not isinstance(caps, Caps):
                raise ValueError(f"caps for {tier} must be a Caps instance")

    def _bucket(self, caller_id: str, today: date) -> _Usage:
        u = self._usage.get(caller_id)
        if u is None or u.day != today:
            u = _Usage(day=today)
            self._usage[caller_id] = u
        return u

    def _utilization(self, u: _Usage, caps: Caps) -> tuple[float, str]:
        """Return (max_utilization, limiting_dimension)."""
        ratios = []
        if caps.requests_per_day > 0:
            ratios.append((u.requests / caps.requests_per_day, "requests"))
        if caps.input_tokens_per_day > 0:
            ratios.append((u.input_tokens / caps.input_tokens_per_day, "input_tokens"))
        if caps.usd_per_day > 0 and caps.usd_per_day != float("inf"):
            ratios.append((u.usd / caps.usd_per_day, "usd"))
        if not ratios:
            return 0.0, "none"
        return max(ratios, key=lambda x: x[0])

    def assess(self, caller_id: str, tier: Tier, *, today: date | None = None) -> BudgetZone:
        """Assess budget zone — never raises for owner/admin/user."""
        # Owner is never limited.
        if tier == "owner":
            return BudgetZone(
                zone="green", tier=tier, utilization=0.0, guidance="Owner: unlimited."
            )

        caps = self.caps.get(tier, self.caps["guest"])
        if (
            caps.requests_per_day == 0
            and caps.input_tokens_per_day == 0
            and (caps.usd_per_day == 0 or math.isinf(caps.usd_per_day))
        ):
            return BudgetZone(
                zone="green",
                tier=tier,
                utilization=0.0,
                guidance="No explicit daily cap configured.",
            )

        today = today or date.today()
        u = self._bucket(caller_id, today)
        util, dim = self._utilization(u, caps)

        if util < YELLOW_THRESHOLD:
            return BudgetZone(
                zone="green",
                tier=tier,
                utilization=util,
                limit_kind=dim,
                used=getattr(u, dim, 0),
                cap=getattr(caps, f"{dim}_per_day", 0),
                guidance="Budget healthy.",
            )

        if util < RED_THRESHOLD:
            return BudgetZone(
                zone="yellow",
                tier=tier,
                utilization=util,
                limit_kind=dim,
                used=getattr(u, dim, 0),
                cap=getattr(caps, f"{dim}_per_day", 0),
                guidance=f"Budget at {util:.0%} — trim tool outputs, prefer cheaper models.",
            )

        if util < 1.0 or tier in ("admin", "user"):
            # admin/user: red zone but NEVER hard blocked — they get degraded service
            return BudgetZone(
                zone="red",
                tier=tier,
                utilization=util,
                limit_kind=dim,
                used=getattr(u, dim, 0),
                cap=getattr(caps, f"{dim}_per_day", 0),
                guidance=f"Budget at {util:.0%} — read-only tools, final warning.",
            )

        # Guest tier at >100%: hard block
        return BudgetZone(
            zone="hard_block",
            tier=tier,
            utilization=util,
            limit_kind=dim,
            used=getattr(u, dim, 0),
            cap=getattr(caps, f"{dim}_per_day", 0),
            guidance="Daily budget exhausted. Try again tomorrow.",
        )

    def check(self, caller_id: str, tier: Tier, *, today: date | None = None) -> None:
        """Legacy API — only raises for guest at hard block."""
        zone = self.assess(caller_id, tier, today=today)
        if zone.zone == "hard_block":
            raise BudgetExceeded(caller_id, zone.limit_kind or "budget", zone.used, zone.cap)

    def record(
        self,
        caller_id: str,
        *,
        usd: float = 0.0,
        input_tokens: int = 0,
        requests: int = 1,
        today: date | None = None,
    ) -> None:
        if isinstance(usd, bool) or not isinstance(usd, (int, float)):
            raise ValueError("usd usage must be a finite non-negative number")
        try:
            usd_value = float(usd)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("usd usage must be a finite non-negative number") from exc
        if not math.isfinite(usd_value) or usd_value < 0:
            raise ValueError("usd usage must be a finite non-negative number")
        for name, value in (("input_tokens", input_tokens), ("requests", requests)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} usage must be a non-negative integer")
        today = today or date.today()
        u = self._bucket(caller_id, today)
        u.usd += usd_value
        u.input_tokens += input_tokens
        u.requests += requests

    def snapshot(self, caller_id: str, today: date | None = None) -> dict:
        today = today or date.today()
        u = self._bucket(caller_id, today)
        return {
            "day": u.day.isoformat(),
            "usd": u.usd,
            "input_tokens": u.input_tokens,
            "requests": u.requests,
        }
