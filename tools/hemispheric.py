#!/usr/bin/env python3
"""Unwired heuristic complexity router retained for compatibility.

The scores are rule-based routing hints, not calibrated cognition or factual
evidence. This module never supplies infrastructure facts from hard-coded host
assumptions.
"""

import json
import re
import sys
import time
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════
# COMPLEXITY CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════


class ComplexityClassifier:
    """Classify message complexity to route to System 1 or 2.

    System 1 (fast, ~0ms): pattern match, lookup, simple rules
    System 2 (slow, 100-500ms): planning, multi-step, novel problems
    """

    # Patterns that signal high complexity → System 2
    HIGH_COMPLEXITY = [
        (r"design|architect|build|create|implement|refactor", 0.8, "creative_task"),
        (r"why|explain|reason|analyze|compare|evaluate", 0.7, "deep_reasoning"),
        (r"plan|strategy|approach|roadmap|milestone", 0.75, "planning"),
        (r"debug|fix|diagnose|investigate|troubleshoot", 0.65, "debugging"),
        (r"optimize|improve|enhance|review", 0.60, "optimization"),
        (r"(?:step|phase)\s+\d+|multi.?step|complex", 0.70, "multi_step"),
        (r"tradeoff|vs\.|versus|pros and cons|should I", 0.65, "decision"),
        (r"\band\b.*\band\b.*\band\b", 0.55, "multi_component"),  # 3+ requirements
    ]

    # Patterns that signal low complexity → System 1
    LOW_COMPLEXITY = [
        (r"^(?:what|which|when|where|who)\s+(?:is|are|was|were)\s+", 0.2, "factual_lookup"),
        (r"what\s+port|what\s+(?:is\s+the\s+)?(?:ip|url|address)", 0.1, "port_lookup"),
        (r"^(?:list|show|display|print)\s+", 0.25, "list_request"),
        (r"^(?:check|status|ping|test)\s+", 0.2, "status_check"),
        (r"^(?:yes|no|ok|sure|thanks|hello|hi|hey|yo|sup|howdy)\s*[!.?]*$", 0.05, "short_response"),
        (r"how\s+many|count|how\s+much", 0.2, "counting"),
    ]

    def classify(self, message: str) -> tuple[float, str]:
        """Classify message complexity. Returns (score 0-1, dominant_reason)."""
        msg_lower = message.lower().strip()

        # Question count check FIRST — multiple questions override simplicity
        question_count = message.count("?")
        if question_count >= 3:
            return min(1.0, 0.7), "multi_question"

        # Check for explicit low complexity
        for pattern, score, reason in self.LOW_COMPLEXITY:
            if re.search(pattern, msg_lower, re.I):
                # But still check if there are 2 questions (moderate bump)
                if question_count == 2:
                    return max(score, 0.55), reason
                return score, reason

        # Count high complexity signals
        complexity_score = 0.3  # Base complexity
        dominant_reason = "default"
        max_signal = 0.0

        for pattern, signal, reason in self.HIGH_COMPLEXITY:
            if re.search(pattern, msg_lower, re.I):
                complexity_score = max(complexity_score, signal)
                if signal > max_signal:
                    max_signal = signal
                    dominant_reason = reason

        # Length heuristic
        words = len(message.split())
        if words > 50:
            complexity_score = max(complexity_score, 0.6)
        elif words > 20:
            complexity_score = max(complexity_score, 0.45)
        elif words < 8:
            complexity_score = min(complexity_score, 0.4)

        # Question count heuristic
        question_count = message.count("?")
        if question_count >= 3:
            complexity_score = max(complexity_score, 0.7)
        elif question_count == 2:
            complexity_score = max(complexity_score, 0.55)

        return min(1.0, complexity_score), dominant_reason


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 1 — fast, pattern-matched, left-hemisphere analog
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class System1Result:
    """Result from System 1 fast processing."""

    answer: str | None
    confidence: float
    method: str  # 'production_rules', 'lookup', 'pattern', 'habit'
    source: str
    latency_ms: float


class LeftHemisphere:
    """Fast pattern recognizer for messages that need no explicit plan.

    Fast path — check if we can answer without full LLM:
    1. Production rules (factual lookups)
    2. Pattern matching (common questions)
    3. Cached habits (automatized responses)
    """

    # These patterns classify shape only. They never answer factual questions;
    # the caller still needs its normal model/retrieval path.
    QUICK_ANSWERS = [
        # Greetings / social (just acknowledge — no deep processing needed)
        (r"^(?:hi|hey|hello|yo|sup|hola|howdy|greetings)(?:\s+\w+)?\s*[!.]*$", None, 0.95),
        (r"^how\s+are\s+you", None, 0.90),
        (r"^(?:good\s+)?(?:morning|afternoon|evening|night)", None, 0.90),
        (r"^(?:thanks?|thank\s+you|ty|thx)", None, 0.90),
        (r"^(?:yes|no|ok|sure|yep|nah|nope|gotcha|bet)\s*[!.]*$", None, 0.95),
        # Trivial factual (LLM can answer instantly)
        (r"^what\s+is\s+the\s+(?:capital|population|currency)\s+of\s+", None, 0.85),
        (r"^(?:what|when|where)\s+(?:is|was|are|were)\s+\w+\s*\??$", None, 0.80),
        # Trivial math
        (r"^(?:what\s+is\s+)?\d+\s*[\+\-\*\/\%]\s*\d+", None, 0.80),
        # Simple commands (just execute, no plan needed)
        (r"^(?:restart|start|stop|enable|disable)\s+\w+", None, 0.85),
    ]

    def process(self, message: str) -> System1Result:
        """Try to answer with System 1 fast path."""
        t0 = time.monotonic()
        msg_lower = message.lower()

        # Check quick answers
        for pattern, answer, conf in self.QUICK_ANSWERS:
            if re.search(pattern, msg_lower, re.I):
                # answer=None means "S1 recognizes this as simple, LLM can handle naturally"
                method = "lookup" if answer else "trivial"
                return System1Result(
                    answer=answer,
                    confidence=conf,
                    method=method,
                    source="quick_answers",
                    latency_ms=(time.monotonic() - t0) * 1000,
                )

        # No System 1 answer
        return System1Result(
            answer=None,
            confidence=0.0,
            method="none",
            source="miss",
            latency_ms=(time.monotonic() - t0) * 1000,
        )


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM 2 — slow, deliberate, right-hemisphere analog
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class System2Plan:
    """Plan from System 2 deliberate processing."""

    approach: str  # How to approach this problem
    steps: list[str]  # Decomposed steps
    tools_needed: list[str]
    context_needed: list[str]
    estimated_ms: int | None
    requires_llm: bool


class RightHemisphere:
    """System 2: Holistic, creative, multi-step reasoning.

    Slow path — generate a structured approach:
    1. Decompose the problem
    2. Identify required tools and context
    3. Plan execution sequence
    """

    def plan(self, message: str, complexity_reason: str) -> System2Plan:
        """Generate a structured plan for complex queries."""
        msg_lower = message.lower()

        # Categorize and plan
        if "debug" in msg_lower or "fix" in msg_lower or "error" in msg_lower:
            return self._debug_plan(message)
        elif any(w in msg_lower for w in ["build", "create", "implement", "write"]):
            return self._build_plan(message)
        elif any(w in msg_lower for w in ["design", "architect", "plan", "strategy"]):
            return self._design_plan(message)
        elif any(w in msg_lower for w in ["analyze", "review", "evaluate", "compare"]):
            return self._analysis_plan(message)
        else:
            return self._generic_plan(message)

    def _debug_plan(self, message: str) -> System2Plan:
        return System2Plan(
            approach="Systematic debug: reproduce → isolate → fix → verify",
            steps=[
                "1. Identify the error/symptom",
                "2. Run diagnostics (logs, status)",
                "3. Narrow to root cause",
                "4. Apply fix",
                "5. Verify resolution",
            ],
            tools_needed=["exec", "read"],
            context_needed=["error_logs", "service_status"],
            estimated_ms=None,
            requires_llm=True,
        )

    def _build_plan(self, message: str) -> System2Plan:
        return System2Plan(
            approach="Build: requirements → design → implement → test",
            steps=[
                "1. Clarify requirements",
                "2. Design architecture",
                "3. Implement incrementally",
                "4. Test each component",
                "5. Integration test",
            ],
            tools_needed=["write", "exec", "read"],
            context_needed=["existing_code", "requirements"],
            estimated_ms=None,
            requires_llm=True,
        )

    def _design_plan(self, message: str) -> System2Plan:
        return System2Plan(
            approach="Design: constraints → options → tradeoffs → recommendation",
            steps=[
                "1. Identify constraints and goals",
                "2. Generate 2-3 design options",
                "3. Analyze tradeoffs",
                "4. Recommend best approach",
                "5. Define implementation plan",
            ],
            tools_needed=["read", "web_search"],
            context_needed=["current_state", "requirements"],
            estimated_ms=None,
            requires_llm=True,
        )

    def _analysis_plan(self, message: str) -> System2Plan:
        return System2Plan(
            approach="Analysis: gather data → pattern detection → insight extraction",
            steps=[
                "1. Gather relevant data/context",
                "2. Apply analytical framework",
                "3. Identify patterns and anomalies",
                "4. Extract insights",
                "5. Recommend actions",
            ],
            tools_needed=["read", "exec"],
            context_needed=["data", "metrics"],
            estimated_ms=None,
            requires_llm=True,
        )

    def _generic_plan(self, message: str) -> System2Plan:
        return System2Plan(
            approach="General: context → reasoning → response",
            steps=["1. Gather context", "2. Reason about query", "3. Respond"],
            tools_needed=[],
            context_needed=["relevant_context"],
            estimated_ms=None,
            requires_llm=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# HEMISPHERIC PROCESSOR — routes between S1 and S2
# ═══════════════════════════════════════════════════════════════════════════

# Thresholds
S1_ONLY = 0.30  # Below this: pure System 1
S2_OVERRIDE = 0.70  # Above this: pure System 2
# Between S1_ONLY and S2_OVERRIDE: try S1 first, escalate if confidence < S1_CONFIDENCE_MIN
S1_CONFIDENCE_MIN = 0.75  # S1 must be this confident to avoid System 2


@dataclass
class HemisphericResult:
    """Combined result from hemispheric processing."""

    complexity: float
    complexity_reason: str
    system_used: str  # 'S1', 'S2', 'S1+S2'
    s1_answer: str | None
    s1_confidence: float
    s2_plan: System2Plan | None
    signal: str  # For brain-state injection


class HemisphericProcessor:
    """Routes processing between System 1 (fast) and System 2 (deliberate).

    S1 answers FIRST. If S1 is confident → stop, no LLM needed.
    If S1 misses or is uncertain → S2 generates structured plan.
    Both results available to guide the model's response.
    """

    def __init__(self):
        self.classifier = ComplexityClassifier()
        self.left = LeftHemisphere()
        self.right = RightHemisphere()
        self._s1_hits = 0
        self._s2_hits = 0

    def process(self, message: str) -> HemisphericResult:
        """Route message through appropriate processing pathway."""
        complexity, reason = self.classifier.classify(message)

        s1_answer = None
        s1_confidence = 0.0
        s2_plan = None
        system_used = "S2"  # Default

        if complexity <= S1_ONLY:
            # Pure System 1 path
            s1_result = self.left.process(message)
            s1_answer = s1_result.answer
            s1_confidence = s1_result.confidence
            if s1_answer:
                system_used = "S1"
                self._s1_hits += 1
            elif s1_result.method == "trivial":
                # S1 recognized pattern as simple — no S2 needed, LLM handles naturally
                system_used = "S1"
                s1_confidence = s1_result.confidence
                self._s1_hits += 1
            else:
                # S1 miss on simple query — light S2
                s2_plan = self.right.plan(message, reason)
                system_used = "S1→S2"
                self._s2_hits += 1

        elif complexity >= S2_OVERRIDE:
            # Pure System 2 path — too complex for S1
            s2_plan = self.right.plan(message, reason)
            system_used = "S2"
            self._s2_hits += 1

        else:
            # Middle ground — try S1 first
            s1_result = self.left.process(message)
            s1_answer = s1_result.answer
            s1_confidence = s1_result.confidence

            if s1_answer and s1_confidence >= S1_CONFIDENCE_MIN:
                system_used = "S1"
                self._s1_hits += 1
            elif s1_result.method == "trivial" and s1_confidence >= S1_CONFIDENCE_MIN:
                # Trivial pattern recognized — S1 path
                system_used = "S1"
                self._s1_hits += 1
            else:
                # S1 insufficient — use S2
                s2_plan = self.right.plan(message, reason)
                system_used = "S1→S2" if s1_answer else "S2"
                self._s2_hits += 1

        # Build brain signal
        signal = self._build_signal(
            complexity, reason, system_used, s1_answer, s1_confidence, s2_plan
        )

        return HemisphericResult(
            complexity=complexity,
            complexity_reason=reason,
            system_used=system_used,
            s1_answer=s1_answer,
            s1_confidence=s1_confidence,
            s2_plan=s2_plan,
            signal=signal,
        )

    def _build_signal(
        self,
        complexity: float,
        reason: str,
        system: str,
        s1_answer: str | None,
        s1_conf: float,
        s2_plan: System2Plan | None,
    ) -> str:
        """Build brain-state signal string."""
        parts = [f"HEMISPHERIC:{system}(complexity={complexity:.2f},{reason})"]

        if s1_answer:
            # Truncate long answers
            short_answer = s1_answer[:80] + ("..." if len(s1_answer) > 80 else "")
            parts.append(f"S1_ANSWER:{short_answer}(conf={s1_conf:.2f})")

        if s2_plan:
            parts.append(f"S2_PLAN:{s2_plan.approach[:60]}(steps={len(s2_plan.steps)})")

        return " | ".join(parts)

    def inject_signal(self, message: str) -> str:
        """Generate signal for brain-state."""
        result = self.process(message)
        return result.signal

    def fire(self, message: str, context: str = "") -> str:
        return self.inject_signal(message)

    def get_stats(self) -> dict:
        total = self._s1_hits + self._s2_hits
        return {
            "s1_hits": self._s1_hits,
            "s2_hits": self._s2_hits,
            "s1_rate": self._s1_hits / total if total > 0 else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# MODULE INSTANCE
# ═══════════════════════════════════════════════════════════════════════════

_hemi = HemisphericProcessor()


def process(message: str) -> HemisphericResult:
    return _hemi.process(message)


def inject_signal(message: str) -> str:
    return _hemi.inject_signal(message)


def fire(message: str, context: str = "") -> str:
    return _hemi.fire(message, context)


def get_stats() -> dict:
    return _hemi.get_stats()


if __name__ == "__main__":
    message = " ".join(sys.argv[1:]).strip()
    if not message:
        print(json.dumps({"ok": False, "error": "message is required"}))
        raise SystemExit(2)
    result = HemisphericProcessor().process(message)
    print(
        json.dumps(
            {
                "ok": True,
                "classification_kind": "unwired_rule_based_heuristic",
                "complexity": result.complexity,
                "reason": result.complexity_reason,
                "route": result.system_used,
                "plan": result.s2_plan.__dict__ if result.s2_plan else None,
            },
            indent=2,
        )
    )
