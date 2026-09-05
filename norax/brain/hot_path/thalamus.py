"""L1 Thalamus — Input classification, routing, signal strength.

All sensory input passes through thalamus. It classifies message type,
scores signal strength, and determines routing pathway using local regex and
set operations without model or network calls.

Norax hot-path module.
Norax native: simplified to the classification + signal core. No state file
needed — thalamus is stateless by design (it's a relay, not a store).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

MsgType = Literal[
    "command",
    "question",
    "feedback",
    "directive",
    "social",
    "noise",
    "code",
    "emergency",
    "status",
]


@dataclass(frozen=True)
class ThalamicResult:
    msg_type: MsgType
    signal_strength: float  # 0-10
    pathway: str  # executive, retrieval, reward, procedural_update, social, filter
    complexity: int  # 0-10


# Pattern → (type, priority, pathway)
_PATTERNS: list[tuple[str, MsgType, int, str]] = [
    # Emergency — highest priority
    (r"\b(urgent|emergency|broken|down|crash|critical|fire)\b", "emergency", 10, "executive"),
    # Commands
    (
        r"^(fix|build|create|deploy|install|run|start|stop|check|update|delete|remove|set|change)\b",
        "command",
        5,
        "executive",
    ),
    (
        r"^(do|make|write|edit|add|search|find|look|scan|test|debug|implement|refactor)\b",
        "command",
        5,
        "executive",
    ),
    # Questions
    (r"\?\s*$", "question", 4, "retrieval"),
    (
        r"^(what|where|when|who|why|how|which|is|are|do|does|can|will|should)\b",
        "question",
        4,
        "retrieval",
    ),
    (r"^(tell me|show me|explain|describe|list)\b", "question", 4, "retrieval"),
    # Feedback
    (
        r"\b(good|great|nice|perfect|wrong|bad|correct|exactly|love it|awesome)\b",
        "feedback",
        3,
        "reward",
    ),
    (r"\b(thanks|thank you|appreciated)\b", "feedback", 3, "reward"),
    # Directives
    (
        r"\b(always|never|from now on|going forward|remember that|priority|important)\b",
        "directive",
        6,
        "procedural_update",
    ),
    (r"\b(policy|rule|constraint|requirement)\b", "directive", 6, "procedural_update"),
    # Desktop / system ops that resolve a novel problem — flag for procedural storage
    (
        r"\b(unlock|lock|screen|desktop|session|greeter|display|wayland|cosmic|gdm|sddm|loginctl)\b",
        "directive",
        6,
        "procedural_update",
    ),
    # Social — higher priority than feedback for greetings
    (r"^(hi|hey|hello|yo|sup|gm|gn)\b", "social", 4, "social"),
    (r"^good\s+(morning|night|evening|afternoon)\b", "social", 4, "social"),
    # Code blocks
    (r"```", "code", 4, "executive"),
    # Status requests
    (r"\b(status|health|uptime|state|running)\b.*\?", "status", 3, "retrieval"),
]

# Signal boosters (keyword → boost to signal strength)
_BOOSTERS: dict[str, float] = {
    "urgent": 3,
    "asap": 2,
    "now": 1,
    "broken": 2,
    "down": 2,
    "revenue": 2,
    "money": 1,
    "customer": 1,
    "security": 2,
    "please": 0.5,
    "important": 1,
    "critical": 3,
}

# Complexity signals
_COMPLEXITY_HIGH = {
    "architecture",
    "refactor",
    "migrate",
    "redesign",
    "debug",
    "investigate",
    "analyze",
    "implement",
}
_COMPLEXITY_MED = {"fix", "update", "change", "add", "configure", "set up", "deploy"}
_COMPLEXITY_LOW = {"check", "status", "show", "list", "hello", "hi", "thanks"}


def classify(message: str) -> ThalamicResult:
    """Classify a message — the core thalamic function."""
    msg_lower = message.strip().lower()
    if not msg_lower:
        return ThalamicResult(msg_type="noise", signal_strength=0, pathway="filter", complexity=0)

    # Score each type
    scores: dict[MsgType, int] = {}
    pathways: dict[MsgType, str] = {}
    for pattern, mtype, priority, pathway in _PATTERNS:
        if re.search(pattern, msg_lower):
            scores[mtype] = scores.get(mtype, 0) + priority
            pathways[mtype] = pathway

    if not scores:
        msg_type: MsgType = "question" if "?" in message else "command"
        pathway = "retrieval" if msg_type == "question" else "executive"
    else:
        msg_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        pathway = pathways.get(msg_type, "executive")

    # Signal strength
    signal = 5.0
    for keyword, boost in _BOOSTERS.items():
        if keyword in msg_lower:
            signal += boost
    signal = max(1.0, min(10.0, signal))

    # Complexity
    words = set(msg_lower.split())
    complexity = 0
    if words & _COMPLEXITY_HIGH:
        complexity += 3
    if words & _COMPLEXITY_MED:
        complexity += 1
    if words & _COMPLEXITY_LOW:
        complexity -= 1
    if len(message) > 200:
        complexity += 2
    elif len(message) > 50:
        complexity += 1
    sentences = message.count(".") + message.count("!") + message.count("?")
    if sentences > 3:
        complexity += 2
    complexity = max(0, min(10, complexity))

    return ThalamicResult(
        msg_type=msg_type,
        signal_strength=signal,
        pathway=pathway,
        complexity=complexity,
    )
