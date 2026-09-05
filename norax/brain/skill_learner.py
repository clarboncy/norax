"""Skill acquisition — mine repeated tool-call patterns into reusable skills.

Detects tool-call sequences that repeat across 3+ successful trajectories,
extracts them as named skill procedures, and persists them to
memory/procedural/skills/retrievable.{name}.md.

Runs during idle/sleep maintenance — not in the hot path.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .harness_optimizer import HarnessTrajectory

log = logging.getLogger("norax.brain.skill_learner")

# Minimum episode count to trigger skill creation
_MIN_EPISODES = 3
# Minimum tool calls in a pattern to be considered a "skill"
_MIN_TOOL_CALLS = 3
# Maximum tool calls — longer sequences are chunked
_MAX_TOOL_CALLS = 16


@dataclass(slots=True)
class ToolPattern:
    """A repeated sequence of tool calls extracted from trajectories."""

    tools: list[str]  # ordered tool names e.g. ["read","edit","read"]
    count: int
    success_rate: float
    avg_writes: float
    avg_verified: float
    example_goals: list[str] = field(default_factory=list)
    avg_rounds: float = 0.0
    episode_ids: list[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        return "→".join(self.tools)

    @property
    def name(self) -> str:
        """Derive a human-readable skill name from the pattern."""
        if self.tools == ["read", "edit", "read"]:
            return "read-edit-verify"
        if self.tools == ["exec", "read"]:
            return "exec-and-inspect"
        if self.tools == ["write", "read"]:
            return "write-and-verify"
        if self.tools == ["list_dir", "read"]:
            return "explore-and-read"
        if "search_memory" in self.tools and "read" in self.tools:
            return "recall-and-review"
        if "exec" in self.tools and "write" in self.tools:
            return "build-and-test"
        # Fallback: first two tools
        return "-".join(self.tools[:2])


@dataclass(slots=True)
class SkillLearnerResult:
    patterns_found: int
    skills_created: int
    skills_updated: int
    skills: list[str]  # skill names
    details: list[dict[str, Any]] = field(default_factory=list)


class SkillLearner:
    """Mine trajectories for repeated tool patterns → generate skills."""

    def __init__(self, root: Path, *, min_episodes: int = _MIN_EPISODES):
        self.root = Path(root)
        self.skills_dir = self.root / "procedural" / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.min_episodes = min_episodes
        self._known_patterns: dict[str, ToolPattern] = {}
        self._last_run: float = 0.0

    # ------------------------------------------------------------------
    # Pattern mining
    # ------------------------------------------------------------------

    def mine(self, trajectories: list[HarnessTrajectory]) -> list[ToolPattern]:
        """Extract repeated tool-call patterns from successful trajectories."""
        if not trajectories:
            return []

        # Only verified successful trajectories may become prompt-injected
        # procedures. Partial outcomes and unverified writes remain useful for
        # diagnostics, but promoting them would turn correlation into policy.
        usable = [
            t
            for t in trajectories
            if t.training_eligible
            and t.outcome_label == "success"
            and t.outcome_score > 0.50
            and t.failures == 0
            and (not t.writes or t.verified_after_write)
            and len(t.tools) >= _MIN_TOOL_CALLS
        ]
        if not usable:
            return []

        # Sliding window of tool sequences
        pattern_counts: dict[str, dict[str, HarnessTrajectory]] = defaultdict(dict)
        for trajectory_index, t in enumerate(usable):
            tools = t.tools
            for window_size in range(_MIN_TOOL_CALLS, min(len(tools) + 1, _MAX_TOOL_CALLS + 1)):
                for start in range(len(tools) - window_size + 1):
                    seq = tuple(tools[start : start + window_size])
                    if not self._is_trivial(seq):
                        key = "→".join(seq)
                        # Count trajectories, not repeated occurrences inside
                        # one long trajectory.  Otherwise tool-heavy turns can
                        # dominate the learned procedure statistics.
                        episode_key = t.trace_id or f"trajectory:{trajectory_index}"
                        pattern_counts[key][episode_key] = t

        # Filter by minimum episode count
        patterns: list[ToolPattern] = []
        for key, episode_map in pattern_counts.items():
            episodes = list(episode_map.values())
            if len(episodes) < self.min_episodes:
                continue
            tools = key.split("→")
            successes = sum(1 for e in episodes if e.outcome_score >= 0.55)
            writes_list = [e.writes for e in episodes]
            verified_list = [e.verified_after_write for e in episodes]
            rounds_list = [e.rounds for e in episodes]

            patterns.append(
                ToolPattern(
                    tools=tools,
                    count=len(episodes),
                    success_rate=round(successes / max(1, len(episodes)), 3),
                    avg_writes=round(sum(writes_list) / len(writes_list), 1),
                    avg_verified=round(sum(verified_list) / max(1, len(verified_list)), 2),
                    avg_rounds=round(sum(rounds_list) / len(rounds_list), 1),
                    example_goals=[e.goal[:120] for e in episodes[:3] if e.goal],
                    episode_ids=[e.trace_id for e in episodes[:10]],
                )
            )

        # Sort by count * success_rate (best patterns first)
        patterns.sort(key=lambda p: p.count * p.success_rate, reverse=True)
        return patterns

    @staticmethod
    def _is_trivial(tools: tuple[str, ...]) -> bool:
        """Filter out trivial tool sequences."""
        if len(tools) < 2:
            return True
        # All same tool repeated — not a skill
        if len(set(tools)) == 1:
            return True
        return False

    # ------------------------------------------------------------------
    # Skill generation
    # ------------------------------------------------------------------

    def generate(self, patterns: list[ToolPattern]) -> SkillLearnerResult:
        """Generate or update skill files from mined patterns."""
        created = 0
        updated = 0
        skill_names: list[str] = []
        details: list[dict[str, Any]] = []

        selected: list[ToolPattern] = []
        seen_names: set[str] = set()
        for pat in patterns:
            # Several overlapping sequences intentionally share a friendly
            # fallback name.  Only the strongest one may own that file.
            if pat.name in seen_names:
                continue
            seen_names.add(pat.name)
            selected.append(pat)
            if len(selected) >= 10:
                break

        for pat in selected:
            name = pat.name
            path = self.skills_dir / f"retrievable.{name}.md"

            exists = path.exists()
            if exists:
                if self._update_skill(path, pat):
                    updated += 1
            else:
                self._create_skill(path, pat)
                created += 1

            skill_names.append(name)
            details.append(
                {
                    "name": name,
                    "tools": pat.tools,
                    "count": pat.count,
                    "success_rate": pat.success_rate,
                    "created": not exists,
                }
            )
            self._known_patterns[pat.fingerprint] = pat

        self._last_run = time.time()
        return SkillLearnerResult(
            patterns_found=len(patterns),
            skills_created=created,
            skills_updated=updated,
            skills=skill_names,
            details=details,
        )

    def _render_skill(self, pat: ToolPattern, ts: str) -> str:
        """Render a complete, internally consistent learned skill."""
        lines = [
            f"SKILL;id={pat.name};inject=learned;priority={min(5, pat.count)}|W5",
            f"# {pat.name} — auto-learned {ts}",
            "",
            f"Pattern: {' → '.join(pat.tools)}",
            f"Episodes: {pat.count} | Success rate: {pat.success_rate:.0%} | Avg rounds: {pat.avg_rounds}",
            "",
            "## When to use",
            f"This pattern was repeated {pat.count} times with {pat.success_rate:.0%} success.",
            "Trigger when the task requires:",
        ]
        for tool in pat.tools:
            lines.append(f"  - {self._tool_description(tool)}")
        lines.extend(
            [
                "",
                "## Procedure",
            ]
        )
        for i, tool in enumerate(pat.tools, 1):
            lines.append(f"  {i}. **{tool}**: {self._tool_instruction(tool, i, len(pat.tools))}")
        lines.extend(
            [
                "",
                "## Examples",
            ]
        )
        for goal in pat.example_goals[:3]:
            lines.append(f"  - {goal}")
        lines.extend(
            [
                "",
                "## Verification",
                "After completing this pattern, verify the result matches expectations.",
                f"Average write-verify rate: {pat.avg_verified:.0%}",
            ]
        )
        return "\n".join(lines) + "\n"

    def _create_skill(self, path: Path, pat: ToolPattern) -> None:
        """Create a new retrievable skill file."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        path.write_text(self._render_skill(pat, ts), encoding="utf-8")
        log.info("skill_learner.created: %s (%d episodes)", pat.name, pat.count)

    def _update_skill(self, path: Path, pat: ToolPattern) -> bool:
        """Rewrite a changed skill once; leave identical files untouched."""
        try:
            content = path.read_text(encoding="utf-8")
            match = re.search(r"auto-learned (\S+)", content)
            previous_ts = match.group(1) if match else "unknown"
            if self._render_skill(pat, previous_ts) == content:
                return False
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            path.write_text(self._render_skill(pat, ts), encoding="utf-8")
            log.info("skill_learner.updated: %s (%d episodes)", pat.name, pat.count)
            return True
        except Exception as e:
            log.warning("skill_learner.update_failed: %s: %r", pat.name, e)
            return False

    @staticmethod
    def _tool_description(tool: str) -> str:
        desc = {
            "read": "reading file contents",
            "write": "creating new files",
            "edit": "modifying existing files",
            "exec": "running shell commands",
            "list_dir": "exploring directories",
            "search_memory": "recalling stored knowledge",
            "web_search": "searching the web",
            "web_fetch": "fetching web content",
        }
        return desc.get(tool, f"using {tool}")

    @staticmethod
    def _tool_instruction(tool: str, step: int, total: int) -> str:
        instructions = {
            "read": "Read the target file to understand its current state.",
            "write": "Write the complete new content in one call.",
            "edit": "Apply the specific change. Include exact old/new text.",
            "exec": "Run the command. Check exit code and stderr.",
            "list_dir": "List directory contents to find relevant files.",
            "search_memory": "Search memory for relevant context.",
            "web_search": "Search the web for current information.",
            "web_fetch": "Fetch the full content of the target URL.",
        }
        base = instructions.get(tool, f"Execute {tool}.")
        if step == total:
            base += " Verify the result before proceeding."
        return base

    # ------------------------------------------------------------------
    # Trigger detection (for runtime hot-path)
    # ------------------------------------------------------------------

    @staticmethod
    def detect_triggers(body: str) -> list[str]:
        """Given a user message, return matching skill names."""
        if not body:
            return []
        body_lower = body.lower()

        trigger_map = {
            "read-edit-verify": ["fix", "update", "change", "edit", "modify", "patch"],
            "exec-and-inspect": ["run", "execute", "check", "test", "build"],
            "write-and-verify": ["create", "write", "generate", "make", "new file"],
            "explore-and-read": ["look at", "explore", "find", "list", "what files"],
            "recall-and-review": ["remember", "recall", "search memory", "past", "previous"],
            "build-and-test": ["build and test", "compile", "deploy", "ship"],
        }

        matched = []
        for skill_name, triggers in trigger_map.items():
            skill_path = Path("memory/procedural/skills") / f"retrievable.{skill_name}.md"
            if skill_path.exists():
                if any(t in body_lower for t in triggers):
                    matched.append(skill_name)

        return matched

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "patterns_known": len(self._known_patterns),
            "last_run": self._last_run,
            "skills_dir": str(self.skills_dir),
            "existing_skills": len(list(self.skills_dir.glob("retrievable.*.md"))),
        }
