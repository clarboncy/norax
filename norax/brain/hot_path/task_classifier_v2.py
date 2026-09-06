"""Optional deterministic task-complexity classifier.

The production hot path currently uses :mod:`thalamus`; this module is a
side-effect-free diagnostic/experimentation API. Its returned tool counts are
advisory metadata and never authorize tools or override the agent loop's hard
limits.

Tier 1: Deterministic regex (<1μs) — handles obvious patterns.
Tier 2: Stable lexical similarity — catches close paraphrases of route examples.
Merged: High-confidence regex wins; ambiguous/low-confidence falls to Tier 2.

It performs no model or network calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .semantic_router import (
    route as semantic_route,
)

TaskClass = Literal[
    "trivial_social",
    "ack_feedback",
    "simple_question",
    "simple_status",
    "memory_instruction",
    "research",
    "coding",
    "ops_deploy",
    "debug_investigate",
    "danger_sensitive",
    "complex_planning",
    "ambiguous",
]

Depth = Literal["fast", "normal", "deep"]

# Confidence thresholds for Tier merging
REGEX_HIGH_CONFIDENCE = 0.85  # regex result is authoritative
SEMANTIC_OVERRIDE_THRESHOLD = 0.70  # semantic can override ambiguous regex
REGEX_AMBIGUOUS: TaskClass = "ambiguous"


@dataclass(frozen=True)
class TaskClassification:
    task_class: TaskClass
    depth: Depth
    confidence: float
    max_tool_calls: int | None
    reason: str
    tier: str = "regex"  # "regex", "semantic", or "merged"


# ---------------------------------------------------------------------------
# Tier 1: Regex patterns (same as before, proven in production)
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^(hi|hey|hello|yo|sup|gm|gn|good\s+(morning|afternoon|evening|night))[!.\s]*$",
    re.I,
)
_ACK_RE = re.compile(
    r"^(ok|okay|k|kk|cool|nice|good|great|perfect|thanks|thank you|ty|yep|yes|no|done|lol|lmao)[!.\s]*$",
    re.I,
)
_STATUS_RE = re.compile(r"\b(status|health|uptime|running|online|alive|checks?)\b", re.I)

_DANGER_RE = re.compile(
    r"\b(rm\s+-rf|format|dd\s+if=|chmod\s+777|/etc\b|/boot\b|private\s+key|token|secret|wallet|seed\s+phrase)\b",
    re.I,
)
_RESEARCH_RE = re.compile(
    r"\b(websearch|web search|research|sources?|best practices|compare|benchmark|latest|cutting edge|crawl|fetch)\b",
    re.I,
)
_CODE_RE = re.compile(
    r"(```|\b(code|implement|refactor|patch|edit|write\s+(a\s+)?(test|module|file)|pytest|ruff|mypy|function|class|api|bug|fix)\b|\.(py|ts|js|json|md|yaml|yml)\b)",
    re.I,
)
_OPS_RE = re.compile(
    r"\b(deploy|restart|service|systemctl|ssh|rsync|docker|compose|install|upgrade|port|logs?|gateway|online|staging|production)\b",
    re.I,
)
_DEBUG_RE = re.compile(
    r"\b(debug|investigate|trace|broken|crash|failing|failure|error|exception|timeout|permission denied|rate limited)\b",
    re.I,
)
_MEMORY_RE = re.compile(
    r"\b(remember|store|save|memory|forget|clean\s+(his|their|your)?\s*memory|identity|persona)\b",
    re.I,
)
_PLANNING_RE = re.compile(
    r"\b(framework|roadmap|architecture|design|plan|comprehensive|complete|end-to-end|all day|24/7|world class)\b",
    re.I,
)

_SIMPLE_FACT_RE = re.compile(r"^(what|who|when|where|which|why|how|is|are|can|do|does|did)\b", re.I)


def _regex_classify(message: str) -> TaskClassification:
    """Tier 1: Pure regex classification. Returns fast if confident."""
    text = (message or "").strip()
    lower = text.lower()
    words = re.findall(r"[a-z0-9_./:-]+", lower)
    word_count = len(words)
    char_count = len(text)

    if not text:
        return TaskClassification("trivial_social", "fast", 0.99, 0, "empty/noise", "regex")

    # High-risk / high-cost classes win before any fast path.
    if _DANGER_RE.search(text):
        return TaskClassification(
            "danger_sensitive", "deep", 0.95, 20, "sensitive/danger trigger", "regex"
        )
    if _RESEARCH_RE.search(text):
        return TaskClassification("research", "deep", 0.92, 25, "research/web trigger", "regex")
    if _CODE_RE.search(text):
        return TaskClassification("coding", "deep", 0.90, 20, "coding trigger", "regex")
    if _OPS_RE.search(text):
        return TaskClassification("ops_deploy", "deep", 0.90, 20, "ops/deploy trigger", "regex")
    if _DEBUG_RE.search(text):
        return TaskClassification("debug_investigate", "deep", 0.88, 20, "debug trigger", "regex")
    if _MEMORY_RE.search(text):
        return TaskClassification(
            "memory_instruction", "deep", 0.88, 12, "memory/instruction trigger", "regex"
        )
    if _PLANNING_RE.search(text):
        return TaskClassification(
            "complex_planning", "deep", 0.86, 18, "planning/architecture trigger", "regex"
        )

    # Safe fast lanes.
    if word_count <= 4 and _GREETING_RE.match(text):
        return TaskClassification("trivial_social", "fast", 0.98, 1, "short greeting", "regex")
    if word_count <= 5 and _ACK_RE.match(text):
        return TaskClassification("ack_feedback", "fast", 0.97, 1, "short ack/feedback", "regex")

    # Simple status/question lanes.
    if char_count <= 80 and _STATUS_RE.search(text):
        return TaskClassification(
            "simple_status", "normal", 0.82, 4, "short status request", "regex"
        )
    if char_count <= 90 and text.endswith("?") and _SIMPLE_FACT_RE.match(text):
        return TaskClassification(
            "simple_question", "normal", 0.78, 4, "short simple question", "regex"
        )

    # Length/multiple asks implies planning.
    sentence_count = max(1, text.count("?") + text.count("!") + text.count("."))
    if char_count > 220 or sentence_count >= 4:
        return TaskClassification(
            "complex_planning", "deep", 0.75, 18, "long/multi-part message", "regex"
        )

    # No match → ambiguous (Tier 2 will try semantic)
    return TaskClassification(
        REGEX_AMBIGUOUS,
        "normal",
        0.55,
        8,
        "no decisive trigger; conservative normal fallback",
        "regex",
    )


# ---------------------------------------------------------------------------
# Tier 2 + Merge
# ---------------------------------------------------------------------------


def classify_task(message: str) -> TaskClassification:
    """Two-tier classification: regex first, semantic fallback for ambiguous.

    Merge rules:
    1. If regex confidence >= REGEX_HIGH_CONFIDENCE (0.85) and class != ambiguous:
       → Use regex result (Tier 1 authority)
    2. If regex result is "ambiguous" AND semantic confidence >= SEMANTIC_OVERRIDE_THRESHOLD:
       → Override with semantic class (Tier 2 rescue)
    3. If regex result is "ambiguous" AND semantic confidence < SEMANTIC_OVERRIDE_THRESHOLD:
       → Keep "ambiguous" but use semantic depth hint if available
    4. If regex matches but with lower confidence AND semantic agrees:
       → Boost confidence slightly, keep regex class
    5. If regex matches but semantic disagrees with higher confidence:
       → Trust regex for safety-critical classes; otherwise hedge to "normal"

    Safety: danger_sensitive never gets downgraded by semantic router.
    """
    regex_result = _regex_classify(message)

    # Rule 1: High-confidence regex → done
    if (
        regex_result.confidence >= REGEX_HIGH_CONFIDENCE
        and regex_result.task_class != REGEX_AMBIGUOUS
    ):
        return regex_result

    # Try Tier 2 semantic
    try:
        semantic_result = semantic_route(message)
    except Exception:  # noqa: BLE001
        # Semantic router error → fall back to pure regex
        return regex_result

    # Rule 2: Ambiguous regex + high-confidence semantic → semantic wins
    if regex_result.task_class == REGEX_AMBIGUOUS:
        if semantic_result.confidence >= SEMANTIC_OVERRIDE_THRESHOLD:
            return TaskClassification(
                task_class=semantic_result.task_class,
                depth=semantic_result.depth,
                confidence=semantic_result.confidence,
                max_tool_calls=semantic_result.max_tool_calls,
                reason=semantic_result.reason + " [merged: semantic override]",
                tier="merged",
            )
        # Rule 3: Low-confidence semantic → keep ambiguous, use depth hint
        if semantic_result.confidence >= 0.40:
            # Use the semantic class name but cap at normal depth
            return TaskClassification(
                task_class=semantic_result.task_class
                if semantic_result.confidence >= 0.55
                else "ambiguous",
                depth="normal",
                confidence=max(regex_result.confidence, semantic_result.confidence),
                max_tool_calls=semantic_result.max_tool_calls,
                reason=f"{regex_result.reason} + semantic_hint:{semantic_result.task_class}@{semantic_result.confidence:.2f}",
                tier="merged",
            )
        # Very low semantic → pure regex fallback
        return regex_result

    # Rule 4 & 5: Regex matched but with lower confidence
    # If semantic agrees → boost
    if semantic_result.task_class == regex_result.task_class:
        boosted_conf = min(1.0, regex_result.confidence + 0.05)
        return TaskClassification(
            task_class=regex_result.task_class,
            depth=regex_result.depth,
            confidence=boosted_conf,
            max_tool_calls=regex_result.max_tool_calls,
            reason=regex_result.reason + " + semantic_agrees",
            tier="merged",
        )

    # Semantic disagrees with non-ambiguous regex
    # Safety: never downgrade danger_sensitive
    if regex_result.task_class == "danger_sensitive":
        return regex_result

    # Semantic disagrees on non-critical class → hedge depth to normal
    if regex_result.confidence < 0.80:
        return TaskClassification(
            task_class=regex_result.task_class,
            depth="normal",  # hedge
            confidence=regex_result.confidence,
            max_tool_calls=regex_result.max_tool_calls or 500,
            reason=f"{regex_result.reason} + semantic_disagrees:{semantic_result.task_class}@{semantic_result.confidence:.2f}",
            tier="merged",
        )

    # Regex is reasonably confident, semantic is just noise → trust regex
    return regex_result
