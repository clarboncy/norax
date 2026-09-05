"""Meta-Learning — self-tuning of learning rates and retrieval weights.

GUARDRAILS (Sprint C safety contract):
  1. ALL parameters are hard-clamped to [min, max] ranges
  2. Every adjustment is logged to meta_learning_audit.jsonl
  3. Maximum adjustment per epoch is capped (no sudden jumps)
  4. Owner can freeze any parameter via the freeze set
  5. If drift exceeds threshold, auto-reverts to defaults and alerts

This module tunes:
  - Hebbian strengthen/decay rates
  - Attention head weights
  - Arousal thresholds
  - VTA expectation learning rate (alpha)

It does NOT tune:
  - Safety gates
  - Trust tiers
  - Owner permissions
  - Memory deletion policies
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_text

log = logging.getLogger("norax.brain.meta_learning")


# ── Hard parameter bounds ──────────────────────────────────────────
# No self-tuned parameter can EVER leave these ranges.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "hebbian_strengthen_delta": (0.01, 0.15),
    "hebbian_decay_delta": (0.005, 0.05),
    "attention_identity_weight": (0.5, 2.0),
    "attention_semantic_weight": (0.5, 2.0),
    "attention_procedural_weight": (0.5, 2.0),
    "attention_temporal_weight": (0.3, 1.5),
    "arousal_signal_boost": (0.5, 3.0),
    "vta_alpha": (0.1, 0.5),
}

# Maximum change per tuning epoch
MAX_STEP = 0.02

# If cumulative drift from defaults exceeds this, revert + alert
DRIFT_ALERT_THRESHOLD = 0.5

DEFAULTS: dict[str, float] = {
    "hebbian_strengthen_delta": 0.06,
    "hebbian_decay_delta": 0.02,
    "attention_identity_weight": 1.0,
    "attention_semantic_weight": 1.2,
    "attention_procedural_weight": 1.0,
    "attention_temporal_weight": 0.8,
    "arousal_signal_boost": 1.0,
    "vta_alpha": 0.3,
}


@dataclass
class TuningEpoch:
    """Record of one tuning pass."""

    timestamp: float
    param: str
    old_value: float
    new_value: float
    reason: str
    metric_before: float
    metric_after: float


@dataclass
class MetaLearner:
    """Self-tuning engine with hard safety bounds."""

    state_path: Path
    audit_path: Path | None = None
    frozen: set[str] = field(default_factory=set)
    _params: dict[str, float] = field(default_factory=dict)
    _epoch_count: int = 0

    def __post_init__(self):
        if self.audit_path is None:
            self.audit_path = self.state_path.parent / "meta_learning_audit.jsonl"
        self._load()

    def _load(self) -> None:
        """Load saved parameters or use defaults."""
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                self._params = data.get("params", dict(DEFAULTS))
                self.frozen = set(data.get("frozen", []))
                self._epoch_count = data.get("epoch_count", 0)
            except Exception:
                self._params = dict(DEFAULTS)
        else:
            self._params = dict(DEFAULTS)
        # Ensure all params exist and are in bounds
        for k, default in DEFAULTS.items():
            if k not in self._params:
                self._params[k] = default
            self._params[k] = self._clamp(k, self._params[k])

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.state_path,
            json.dumps(
                {
                    "params": self._params,
                    "frozen": list(self.frozen),
                    "epoch_count": self._epoch_count,
                },
                indent=2,
            ),
        )

    def _clamp(self, param: str, value: float) -> float:
        """Hard clamp to bounds. This is the safety guarantee."""
        lo, hi = PARAM_BOUNDS.get(param, (0.0, 10.0))
        return max(lo, min(hi, value))

    def _audit(self, epoch: TuningEpoch) -> None:
        """Append to audit log (append-only, never truncated)."""
        if self.audit_path is None:
            return
        try:
            with self.audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(epoch), default=str) + "\n")
        except Exception as e:
            log.warning("meta_learning.audit.error: %r", e)

    def get(self, param: str) -> float:
        """Get current value for a parameter."""
        return self._params.get(param, DEFAULTS.get(param, 1.0))

    def tune(
        self,
        param: str,
        direction: float,
        reason: str,
        metric_before: float = 0.0,
        metric_after: float = 0.0,
    ) -> float | None:
        """Attempt to tune a parameter. Returns new value or None if blocked.

        direction: positive = increase, negative = decrease
        Actual step is clamped to MAX_STEP.
        """
        if param in self.frozen:
            log.info("meta_learning.frozen param=%s", param)
            return None

        if param not in PARAM_BOUNDS:
            log.warning("meta_learning.unknown_param: %s", param)
            return None

        old = self._params[param]
        step = max(-MAX_STEP, min(MAX_STEP, direction))
        new = self._clamp(param, old + step)

        # Drift check FIRST: total distance from defaults (current state)
        # This catches accumulated drift even if the current step is tiny.
        drift = sum(abs((new if k == param else self._params[k]) - DEFAULTS[k]) for k in DEFAULTS)

        if drift > DRIFT_ALERT_THRESHOLD:
            log.warning(
                "meta_learning.DRIFT_ALERT: cumulative drift %.3f > threshold %.3f. "
                "Reverting ALL to defaults.",
                drift,
                DRIFT_ALERT_THRESHOLD,
            )
            self._params = dict(DEFAULTS)
            self._save()
            self._audit(
                TuningEpoch(
                    timestamp=time.time(),
                    param="ALL",
                    old_value=0.0,
                    new_value=0.0,
                    reason=f"DRIFT_REVERT: drift={drift:.3f}",
                    metric_before=metric_before,
                    metric_after=metric_after,
                )
            )
            return None

        # No meaningful change after drift check passed.  Returning the old
        # value used to make callers count a boundary-clamped no-op as a fresh
        # adjustment every idle cycle.
        if abs(new - old) < 0.001:
            return None

        self._params[param] = new
        self._epoch_count += 1

        epoch = TuningEpoch(
            timestamp=time.time(),
            param=param,
            old_value=old,
            new_value=new,
            reason=reason,
            metric_before=metric_before,
            metric_after=metric_after,
        )
        self._audit(epoch)
        self._save()

        log.info(
            "meta_learning.tuned %s: %.4f → %.4f (reason=%s)",
            param,
            old,
            new,
            reason,
        )
        return new

    def freeze(self, param: str) -> None:
        """Owner can freeze any parameter from further tuning."""
        self.frozen.add(param)
        self._save()
        log.info("meta_learning.freeze: %s", param)

    def unfreeze(self, param: str) -> None:
        self.frozen.discard(param)
        self._save()
        log.info("meta_learning.unfreeze: %s", param)

    def reset_to_defaults(self) -> None:
        """Hard reset all parameters to defaults."""
        self._params = dict(DEFAULTS)
        self._save()
        self._audit(
            TuningEpoch(
                timestamp=time.time(),
                param="ALL",
                old_value=0.0,
                new_value=0.0,
                reason="MANUAL_RESET",
                metric_before=0.0,
                metric_after=0.0,
            )
        )
        log.info("meta_learning.reset_to_defaults")

    def status(self) -> dict[str, Any]:
        """Status for /status endpoint."""
        drift = sum(abs(self._params[k] - DEFAULTS[k]) for k in DEFAULTS)
        return {
            "params": dict(self._params),
            "defaults": dict(DEFAULTS),
            "drift": round(drift, 4),
            "drift_limit": DRIFT_ALERT_THRESHOLD,
            "frozen": list(self.frozen),
            "epochs": self._epoch_count,
        }

    def auto_tune_from_episodes(self, episodes: list) -> int:
        """Run one auto-tuning pass based on recent episode outcomes.

        Called during idle replay. Adjusts parameters based on aggregate
        outcome metrics. Returns number of adjustments made.

        Strategy:
          - If average outcome < 6.0 and tool_calls high → lower arousal boost
          - If average outcome > 8.0 → slightly increase strengthen delta
          - If many failures → increase procedural weight
        """
        if len(episodes) < 10:
            return 0

        outcomes = [e.outcome_score for e in episodes if hasattr(e, "outcome_score")]
        if not outcomes:
            return 0

        avg_outcome = sum(outcomes) / len(outcomes)
        adjustments = 0

        # If outcomes are consistently good, strengthen hebbian learning
        if avg_outcome > 8.0:
            r = self.tune(
                "hebbian_strengthen_delta",
                0.005,
                reason=f"high_avg_outcome={avg_outcome:.1f}",
                metric_before=self.get("hebbian_strengthen_delta"),
                metric_after=avg_outcome,
            )
            if r is not None:
                adjustments += 1

        # If outcomes are poor with many tool calls, reduce arousal (less tool spam)
        avg_tools = sum(e.rounds for e in episodes if hasattr(e, "rounds")) / max(len(episodes), 1)
        if avg_outcome < 6.0 and avg_tools > 5:
            r = self.tune(
                "arousal_signal_boost",
                -0.01,
                reason=f"poor_outcome={avg_outcome:.1f}_high_tools={avg_tools:.1f}",
                metric_before=avg_outcome,
                metric_after=avg_tools,
            )
            if r is not None:
                adjustments += 1

        # If many failures, boost procedural retrieval weight
        failure_rate = sum(
            1
            for e in episodes
            for tc in (e.tool_calls if hasattr(e, "tool_calls") else [])
            if tc.get("ok") is not True
        ) / max(sum(len(e.tool_calls) for e in episodes if hasattr(e, "tool_calls")), 1)

        if failure_rate > 0.2:
            r = self.tune(
                "attention_procedural_weight",
                0.01,
                reason=f"high_failure_rate={failure_rate:.2f}",
                metric_before=failure_rate,
                metric_after=self.get("attention_procedural_weight"),
            )
            if r is not None:
                adjustments += 1

        return adjustments
