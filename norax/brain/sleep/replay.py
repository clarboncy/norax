"""Hippocampal Replay — offline pattern extraction from episodic memory.

During idle periods, replays recent episodes to:
  1. Find recurring verified tool-call patterns → write procedural memories
  2. Count recurring retrieval co-activations for diagnostics
  3. Count high-RPE episodes for diagnostics
  4. Detect failures → write "avoid" procedural memories

This runs in the idle sleep loop (Sprint B), NOT during active turns.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...atomic import atomic_write_text
from ...memory.episodic import Episode, EpisodicBuffer

log = logging.getLogger("norax.brain.sleep.replay")


@dataclass
class ReplayResult:
    """What replay discovered."""

    procedural_patterns: list[str] = field(default_factory=list)
    coactivation_pairs_observed: int = 0
    high_surprise_episodes_seen: int = 0
    failure_patterns: list[str] = field(default_factory=list)


@dataclass
class HippocampalReplay:
    """Replay engine. Stateless — reads episodes, writes memories."""

    episodic: EpisodicBuffer
    memory_root: Path
    min_pattern_count: int = 3  # need 3+ occurrences to be a pattern
    replay_window_days: int = 7  # look back 7 days

    @staticmethod
    def _replace_generated_lines(
        path: Path,
        *,
        header: str,
        prefix: str,
        desired: list[str],
    ) -> bool:
        """Atomically upsert one generated set and compact old duplicates."""
        previous = path.read_text(encoding="utf-8") if path.exists() else ""
        preserved = [
            line
            for line in previous.splitlines()
            if line.strip() and not line.startswith("#") and not line.startswith(prefix)
        ]
        lines = [header, *preserved, *desired]
        updated = "\n".join(lines).rstrip() + "\n"
        if updated == previous:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, updated)
        return True

    def run(self) -> ReplayResult:
        """Execute one replay pass. Returns what was discovered."""
        result = ReplayResult()
        episodes = self.episodic.recent_episodes(
            days=self.replay_window_days,
            limit=500,
        )
        if len(episodes) < 5:
            return result  # not enough data to find patterns

        # 1. Tool-call sequence patterns
        self._extract_tool_patterns(episodes, result)

        # 2. Retrieval co-activation diagnostics.  This helper does not mutate
        # Hebbian state, so the result is deliberately named as an observation.
        result.coactivation_pairs_observed = self._find_coactivations(episodes)

        # 3. High-RPE episode diagnostics
        self._count_high_surprise(episodes, result)

        # 4. Failure patterns
        self._extract_failures(episodes, result)

        return result

    def _extract_tool_patterns(self, episodes: list[Episode], result: ReplayResult) -> None:
        """Find recurring tool sequences across episodes, enriched with outcome context."""
        seq_counter: Counter[str] = Counter()
        seq_outcomes: dict[str, int] = {}  # seq -> success count
        seq_terms: dict[str, Counter[str]] = {}
        for ep in episodes:
            # Tool return codes alone do not prove that the requested outcome
            # was achieved.  Only agent-loop episodes with independent
            # verification can become positive procedural guidance.
            if not ep.training_eligible or not ep.verified_outcome:
                continue
            if not ep.tool_calls:
                continue
            # Build a canonical sequence key
            seq = "→".join(t.get("name", "?") for t in ep.tool_calls[:8])
            seq_counter[seq] += 1
            seq_outcomes[seq] = seq_outcomes.get(seq, 0) + 1
            # Collect task terms for semantic context
            if seq not in seq_terms:
                seq_terms[seq] = Counter()
            for word in str(getattr(ep, "user_input", "")).split():
                word = word.strip(".,!?;:\"'()[]{}").lower()
                if len(word) >= 3:
                    seq_terms[seq][word] += 1

        proc_dir = self.memory_root / "procedural"
        proc_dir.mkdir(parents=True, exist_ok=True)

        desired: list[str] = []
        for seq, count in seq_counter.most_common(10):
            if count < self.min_pattern_count:
                break
            seq_hash = hashlib.sha256(seq.encode()).hexdigest()[:8]
            success_rate = seq_outcomes.get(seq, 0) / count
            # Every counted episode is objectively verified. Frequency controls
            # weight; it does not manufacture verification.
            weight = "W3" if success_rate >= 0.8 and count >= 5 else "W2"
            top_terms = ",".join(t for t, _ in seq_terms[seq].most_common(5))
            desired.append(
                f"PATTERN:id={seq_hash}|tool_seq={seq}|count={count}"
                f"|success_rate={success_rate:.0%}|context={top_terms}|{weight}"
            )

        ts = time.strftime("%Y-%m-%d")
        out = proc_dir / f"replay-patterns-{ts}.md"
        if desired and self._replace_generated_lines(
            out,
            header=f"# hippocampal replay patterns {ts}",
            prefix="PATTERN:",
            desired=desired,
        ):
            result.procedural_patterns.extend(desired)
            log.info("replay: %d tool patterns upserted", len(desired))

    def _find_coactivations(self, episodes: list[Episode]) -> int:
        """Count entity-id pairs that frequently co-occur across episodes."""
        pair_counter: Counter[tuple[str, str]] = Counter()
        for ep in episodes:
            hits = sorted(set(ep.retrieval_hits))
            for i, a in enumerate(hits):
                for b in hits[i + 1 :]:
                    pair_counter[(a, b)] += 1

        # This is observational only; callers must not claim the links changed.
        strong = sum(1 for _, c in pair_counter.items() if c >= self.min_pattern_count)
        return strong

    def _count_high_surprise(self, episodes: list[Episode], result: ReplayResult) -> None:
        """Count high-RPE episodes without claiming a storage verification."""
        for ep in episodes:
            if abs(ep.vta_rpe) < 4:  # only check high-surprise
                continue
            result.high_surprise_episodes_seen += 1

    def _extract_failures(self, episodes: list[Episode], result: ReplayResult) -> None:
        """Find repeated failure patterns with error classification and recovery context.

        Each failure pattern captures:
          - tool name and error type (from stderr/error field)
          - preceding tool in the sequence (for context)
          - task terms from the episode (for semantic matching)
          - recovery: what tool succeeded after the failure, if any
        """
        # Key: (tool_name, error_type) → aggregate stats
        failure_map: dict[tuple[str, str], dict[str, Any]] = {}
        for ep in episodes:
            calls = ep.tool_calls
            task_text = str(getattr(ep, "user_input", ""))
            task_words = [
                w.strip(".,!?;:\"'()[]{}").lower()
                for w in task_text.split()
                if len(w.strip(".,!?;:\"'()[]{}")) >= 3
            ]
            for i, tc in enumerate(calls):
                if tc.get("ok") is True:
                    continue
                tool_name = tc.get("name", "?")
                # Classify error from available fields
                raw_err = (
                    tc.get("error") or tc.get("stderr") or tc.get("error_type") or "tool_error"
                )
                if isinstance(raw_err, dict):
                    raw_err = raw_err.get("code") or raw_err.get("type") or "tool_error"
                err_type = re.sub(r"[^a-z0-9_.-]+", "_", str(raw_err).strip().lower()).strip("_")[
                    :60
                ]
                if not err_type:
                    err_type = "tool_error"
                key = (tool_name, err_type)
                if key not in failure_map:
                    failure_map[key] = {
                        "count": 0,
                        "preceding_tools": Counter(),
                        "task_terms": Counter(),
                        "recovered_with": Counter(),
                    }
                entry = failure_map[key]
                entry["count"] += 1
                # Track what came before the failure
                if i > 0:
                    entry["preceding_tools"][calls[i - 1].get("name", "?")] += 1
                # Track task context
                for word in task_words[:10]:
                    entry["task_terms"][word] += 1
                # Track recovery: what tool succeeded after this failure
                for j in range(i + 1, min(i + 4, len(calls))):
                    if calls[j].get("ok") is True:
                        entry["recovered_with"][calls[j].get("name", "?")] += 1
                        break

        proc_dir = self.memory_root / "procedural"
        proc_dir.mkdir(parents=True, exist_ok=True)

        desired: list[str] = []
        # Sort by count descending, take top 8
        for (tool_name, err_type), stats in sorted(
            failure_map.items(), key=lambda kv: kv[1]["count"], reverse=True
        )[:8]:
            if stats["count"] < self.min_pattern_count:
                break
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", tool_name or "?")
            safe_err = re.sub(r"[^A-Za-z0-9_.-]+", "_", err_type)[:60]
            preceding = ",".join(f"{t}:{c}" for t, c in stats["preceding_tools"].most_common(2))
            context = ",".join(t for t, _ in stats["task_terms"].most_common(4))
            recovery = ",".join(t for t, _ in stats["recovered_with"].most_common(2))
            # W3 if this failure recurs frequently with a known recovery path
            weight = "W3" if stats["count"] >= 5 and stats["recovered_with"] else "W2"
            parts = [
                f"AVOID:tool={safe_name}|error={safe_err}|count={stats['count']}",
            ]
            if preceding:
                parts.append(f"after={preceding}")
            if context:
                parts.append(f"context={context}")
            if recovery:
                parts.append(f"recover_with={recovery}")
            parts.append(weight)
            desired.append("|".join(parts))

        ts = time.strftime("%Y-%m-%d")
        out = proc_dir / f"replay-avoid-{ts}.md"
        if desired and self._replace_generated_lines(
            out,
            header=f"# failure patterns from replay {ts}",
            prefix="AVOID:tool=",
            desired=desired,
        ):
            result.failure_patterns.extend(desired)
            log.info("replay: %d failure patterns upserted", len(desired))
