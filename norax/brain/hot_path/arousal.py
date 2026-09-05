"""Cheap, stateless salience signal for prompt metadata.

The signal summarizes deterministic message complexity/urgency. It does not
authorize actions, disable reasoning layers, or override the agent loop's real
round/tool ceilings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ArousalLevel = Literal["sleep", "background", "idle", "normal", "focused", "hypervigilant"]

LEVEL_INT: dict[ArousalLevel, int] = {
    "sleep": 1,
    "background": 2,
    "idle": 3,
    "normal": 4,
    "focused": 5,
    "hypervigilant": 6,
}


@dataclass(frozen=True)
class ArousalState:
    level: ArousalLevel
    level_int: int
    reason: str


def assess(
    message: str,
    *,
    complexity: int = 0,
    signal_strength: float = 5.0,
    last_interaction_ago_sec: float | None = None,
) -> ArousalState:
    """Summarize message salience without changing execution capability."""
    del message

    # Boost from signal strength
    level_int = LEVEL_INT["normal"]
    if signal_strength >= 8:
        level_int = min(6, level_int + 2)
    elif signal_strength >= 6:
        level_int = min(6, level_int + 1)
    elif signal_strength <= 2:
        level_int = max(1, level_int - 1)

    # Boost from complexity
    if complexity >= 7:
        level_int = min(6, level_int + 1)

    # Interaction recency is informational only. Returning after a break must
    # not make the same request receive less capable processing.
    recency = (
        f" away_s={max(0.0, float(last_interaction_ago_sec)):.0f}"
        if last_interaction_ago_sec is not None
        else ""
    )

    # Map back to level name
    level: ArousalLevel = {v: k for k, v in LEVEL_INT.items()}[level_int]

    return ArousalState(
        level=level,
        level_int=level_int,
        reason=f"signal={signal_strength:.0f} complexity={complexity}{recency}",
    )
