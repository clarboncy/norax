"""Goal Drift Detection — monitors if agent is still working toward original objective.

Research (Microsoft 2025 whitepaper, Factory AI):
  - "Goal drift" is a core agent failure mode
  - Agents lose coherent access to original objectives by ~60% context mark
  - Without active monitoring, agents silently drift to irrelevant work

Detection signals:
  1. Tool relevance — are recent tool calls related to the original task?
  2. Keyword overlap — does recent work share vocabulary with the task?
  3. Time budget — has the agent spent too long without completing?
  4. Round budget — too many rounds without convergence?
  5. Output relevance — does the draft response address the original question?

On drift detected:
  - Inject a "refocus" system message reminding the agent of the original task
  - If severe, force a plan-refresh (re-run L4..L8)
  - Log the drift event for post-mortem analysis
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger("norax.brain.goal_drift")


class DriftSeverity(Enum):
    NONE = "none"
    MILD = "mild"  # slight drift, inject reminder
    MODERATE = "moderate"  # significant drift, force refocus
    SEVERE = "severe"  # agent has lost the plot, force plan refresh


@dataclass
class DriftResult:
    severity: DriftSeverity
    score: float  # 0.0 = perfect alignment, 1.0 = complete drift
    signals: list[str] = field(default_factory=list)
    recommendation: str = ""
    refocus_message: str = ""


def _extract_keywords(text: str, max_words: int = 20) -> set[str]:
    """Extract meaningful keywords from text."""
    # Remove common stop words
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "about",
        "as",
        "into",
        "like",
        "through",
        "after",
        "over",
        "between",
        "out",
        "against",
        "during",
        "without",
        "before",
        "under",
        "around",
        "among",
        "of",
        "from",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "and",
        "or",
        "but",
        "not",
        "no",
        "yes",
        "if",
        "then",
        "else",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "now",
        "also",
        "get",
        "got",
        "make",
        "made",
        "go",
        "went",
        "one",
        "two",
        "ok",
        "okay",
        "please",
        "thank",
        "thanks",
        "you",
        "your",
        "i",
        "me",
        "my",
        "we",
        "our",
        "us",
        "they",
        "them",
        "their",
        "he",
        "she",
        "his",
        "her",
        "him",
    }
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
    keywords = {w for w in words if w not in stop_words and len(w) >= 3}
    # Sort by length (longer = more specific) and take top N
    return set(sorted(keywords, key=len, reverse=True)[:max_words])


def _keyword_overlap(a: set[str], b: set[str]) -> float:
    """Jaccard-like overlap score. 0.0 = no overlap, 1.0 = identical."""
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union) if union else 0.0


class GoalDriftMonitor:
    """Monitors agent progress for goal drift.

    Call check_drift() after each round with the original task and recent
    tool calls / draft output.
    """

    def __init__(
        self,
        *,
        max_rounds: int = 30,
        max_time_sec: float = 600.0,
        drift_threshold: float = 0.7,
        severe_threshold: float = 0.85,
        grace_rounds: int = 3,
    ) -> None:
        self.max_rounds = max_rounds
        self.max_time_sec = max_time_sec
        self.drift_threshold = drift_threshold
        self.severe_threshold = severe_threshold
        self._grace_rounds = grace_rounds  # no drift signals before this round
        self._task_keywords: set[str] = set()
        self._start_time: float = 0.0
        self._round: int = 0
        self._original_task: str = ""
        self._mild_drift_streak: int = 0
        self._moderate_drift_streak: int = 0

    def set_task(self, task: str) -> None:
        """Set the original task for drift monitoring."""
        self._original_task = task
        self._task_keywords = _extract_keywords(task)
        self._start_time = time.time()
        self._round = 0
        self._mild_drift_streak = 0
        self._moderate_drift_streak = 0
        log.debug(
            "goal_drift: tracking %d keywords for task: %s", len(self._task_keywords), task[:80]
        )

    def check_drift(
        self,
        *,
        round_num: int,
        recent_tool_calls: list[dict],
        draft_output: str = "",
    ) -> DriftResult:
        """Check if the agent has drifted from the original task.

        Returns DriftResult with severity and recommendation.
        """
        if not self._task_keywords or not self._original_task:
            return DriftResult(
                severity=DriftSeverity.NONE,
                score=0.0,
                recommendation="no task set",
            )

        # Guard: skip drift evaluation for tasks with too few extractable
        # keywords. Numeric-heavy or very short prompts (e.g. "What is 2+2?")
        # produce 0-1 keywords because the extractor requires alphabetic
        # words of length >= 3. With no meaningful keyword set, overlap is
        # always ~0.0 and every tool call / output triggers a false drift
        # signal — eventually escalating to a spurious force_abort on a
        # trivial conversational turn.
        if len(self._task_keywords) < 2:
            return DriftResult(
                severity=DriftSeverity.NONE,
                score=0.0,
                recommendation="insufficient_keywords",
            )

        # Grace period: don't evaluate drift on early rounds. The agent is
        # still exploring the problem space (reading files, gathering context)
        # and surface-keyword overlap is naturally low before the agent
        # converges on the relevant parts of the task. Counting these early
        # rounds as drift inflates streaks and causes false force_aborts.
        if round_num <= self._grace_rounds:
            return DriftResult(
                severity=DriftSeverity.NONE,
                score=0.0,
                recommendation="grace_period",
            )

        self._round = round_num
        signals: list[str] = []
        drift_score = 0.0
        signal_count = 0

        # Signal 1: Tool call relevance
        if recent_tool_calls:
            tool_text = " ".join(
                f"{tc.get('name', '')} {tc.get('args', {})}" for tc in recent_tool_calls[-5:]
            )
            tool_keywords = _extract_keywords(tool_text)
            tool_overlap = _keyword_overlap(self._task_keywords, tool_keywords)
            if tool_overlap < 0.1:
                drift_score += 0.3
                signal_count += 1
                signals.append(f"low_tool_relevance (overlap={tool_overlap:.2f})")
            else:
                drift_score += (1.0 - tool_overlap) * 0.15
                signal_count += 1

        # Signal 2: Output relevance — only when there is a draft AND the
        # task is not a trivial single-token-answer (e.g. "4"). Very short
        # outputs on simple tasks routinely have 0 keyword overlap without
        # indicating any drift.
        if draft_output and len(draft_output.strip()) > 12:
            output_keywords = _extract_keywords(draft_output)
            output_overlap = _keyword_overlap(self._task_keywords, output_keywords)
            if output_overlap < 0.05:
                drift_score += 0.35
                signal_count += 1
                signals.append(f"low_output_relevance (overlap={output_overlap:.2f})")
            else:
                drift_score += (1.0 - output_overlap) * 0.15
                signal_count += 1

        # Signal 3: Round budget (skip when max_rounds=0, meaning unlimited)
        if self.max_rounds > 0 and round_num > self.max_rounds:
            excess = (round_num - self.max_rounds) / self.max_rounds
            drift_score += min(0.3, excess * 0.3)
            signal_count += 1
            signals.append(f"round_budget_exceeded (round={round_num}/{self.max_rounds})")

        # Signal 4: Time budget (skip when max_time_sec=0, meaning unlimited)
        elapsed = time.time() - self._start_time
        if self.max_time_sec > 0 and elapsed > self.max_time_sec:
            excess = (elapsed - self.max_time_sec) / self.max_time_sec
            drift_score += min(0.25, excess * 0.25)
            signal_count += 1
            signals.append(f"time_budget_exceeded (elapsed={elapsed:.0f}s/{self.max_time_sec}s)")

        # Normalize
        if signal_count > 0:
            drift_score = min(1.0, drift_score)

        # Determine severity
        if drift_score >= self.severe_threshold:
            severity = DriftSeverity.SEVERE
            recommendation = "force_plan_refresh"
            refocus = (
                f"CRITICAL: You have drifted from the original task. "
                f'Original task: "{self._original_task[:200]}". '
                f"Stop current work and refocus on the original objective."
            )
        elif drift_score >= self.drift_threshold:
            self._moderate_drift_streak += 1
            severity = DriftSeverity.MODERATE
            recommendation = "inject_refocus"
            refocus = (
                f'REMINDER: Your original task is: "{self._original_task[:200]}". '
                f"Make sure your current work is directly addressing this."
            )
        elif drift_score >= 0.4:
            self._mild_drift_streak += 1
            # Escalate to MODERATE after 3 consecutive mild-drift rounds.
            if self._mild_drift_streak >= 3:
                severity = DriftSeverity.MODERATE
                recommendation = "inject_refocus"
                refocus = (
                    f"REMINDER: Your original task is: {self._original_task[:200]}. "
                    f"You have produced {self._mild_drift_streak} consecutive low-progress rounds. "
                    f"Call a tool or provide your final answer now."
                )
                self._mild_drift_streak = 0
            else:
                severity = DriftSeverity.MILD
                recommendation = "monitor"
                refocus = ""
        else:
            severity = DriftSeverity.NONE
            recommendation = "continue"
            refocus = ""
            # Reset all streaks on NONE — the agent made real progress.
            self._mild_drift_streak = 0
            self._moderate_drift_streak = 0

        if severity != DriftSeverity.NONE:
            log.info(
                "goal_drift: severity=%s score=%.2f signals=%s",
                severity.value,
                drift_score,
                signals,
            )

        return DriftResult(
            severity=severity,
            score=round(drift_score, 3),
            signals=signals,
            recommendation=recommendation,
            refocus_message=refocus,
        )

    def reset(self) -> None:
        self._task_keywords = set()
        self._original_task = ""
        self._start_time = 0.0
        self._round = 0
        self._mild_drift_streak = 0
        self._moderate_drift_streak = 0
