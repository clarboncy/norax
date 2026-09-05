"""L14 Amygdala — Emotional scoring, threat detection, valence tagging.

Scores emotional content of messages for:
- Valence: positive ↔ negative (-10 to +10)
- Arousal: calm ↔ intense (1-10)
- Threat level: safe ↔ dangerous (0-10)

Used by the decision gate and post-action hooks to tag memories with
emotional significance (flashbulb events get immediate consolidation).

Norax hot-path module.
Norax native: stateless per-call, returns dataclass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionalScore:
    valence: float  # -10 to +10 (negative ↔ positive)
    arousal: float  # 1-10 (calm ↔ intense)
    threat: float  # 0-10 (safe ↔ dangerous)
    dominant: str  # "positive", "negative", "neutral", "threat"
    tags: tuple[str, ...]  # e.g. ("frustration", "urgency")


# Word → valence score
_POSITIVE: dict[str, float] = {
    "good": 1,
    "great": 2,
    "awesome": 3,
    "perfect": 3,
    "love": 3,
    "thanks": 1,
    "thank": 1,
    "nice": 1,
    "excellent": 2,
    "brilliant": 2,
    "amazing": 3,
    "wonderful": 2,
    "happy": 2,
    "excited": 2,
    "success": 2,
    "working": 1,
    "fixed": 2,
    "solved": 2,
    "done": 1,
    "shipped": 2,
    "profit": 2,
    "revenue": 1,
    "growth": 1,
    "milestone": 2,
}

_NEGATIVE: dict[str, float] = {
    "bad": -1,
    "terrible": -3,
    "awful": -3,
    "hate": -3,
    "broken": -2,
    "fail": -2,
    "failed": -2,
    "error": -1,
    "bug": -1,
    "crash": -2,
    "wrong": -1,
    "stuck": -1,
    "lost": -1,
    "waste": -2,
    "frustrated": -2,
    "angry": -3,
    "furious": -3,
    "annoyed": -1,
    "confused": -1,
    "expensive": -1,
    "slow": -1,
    "down": -1,
    "dead": -2,
    "killed": -2,
}

# Threat patterns → threat score
_THREATS: list[tuple[str, float]] = [
    (r"\b(rm\s+-rf|drop\s+table|delete\s+all|wipe|destroy)\b", 8),
    (r"\b(hack|exploit|inject|overflow|vulnerability)\b", 6),
    (r"\b(urgent|emergency|critical|outage)\b", 5),
    (r"\b(broken|crashed|down|failed)\b", 3),
    (r"\b(fuck|shit|damn|wtf)\b", 2),  # frustration, not threat
]

# Reward patterns → valence boost
_REWARDS: list[tuple[str, float, float]] = [  # pattern, valence_boost, arousal_min
    (r"\b(paid|payment|sale|revenue|earned|profit)\b", 3, 4),
    (r"\b(shipped|deployed|launched|released|live)\b", 2, 3),
    (r"\b(100%|passed|perfect score|benchmark)\b", 3, 4),
    (r"\b(new customer|subscriber|follower)\b", 2, 3),
]


def score(text: str) -> EmotionalScore:
    """Score emotional valence, arousal, and threat from text."""
    text_lower = text.lower()
    valence = 0.0
    arousal = 1.0
    threat = 0.0
    tags: list[str] = []

    # Word-level scoring
    for word, v in _POSITIVE.items():
        if word in text_lower:
            valence += v
            arousal = max(arousal, abs(v))

    for word, v in _NEGATIVE.items():
        if word in text_lower:
            valence += v
            arousal = max(arousal, abs(v))
            if v <= -2:
                tags.append(
                    "frustration"
                    if word in ("frustrated", "angry", "furious", "annoyed")
                    else "negative"
                )

    # Threat detection
    for pattern, t_score in _THREATS:
        if re.search(pattern, text_lower):
            threat = max(threat, t_score)
            if t_score >= 5:
                tags.append("urgency")

    # Reward patterns
    for pattern, v_boost, a_min in _REWARDS:
        if re.search(pattern, text_lower):
            valence += v_boost
            arousal = max(arousal, a_min)
            tags.append("reward")

    # Clamp
    valence = max(-10.0, min(10.0, valence))
    arousal = max(1.0, min(10.0, arousal))
    threat = max(0.0, min(10.0, threat))

    # Dominant emotion
    if threat >= 5:
        dominant = "threat"
    elif valence > 1:
        dominant = "positive"
    elif valence < -1:
        dominant = "negative"
    else:
        dominant = "neutral"

    return EmotionalScore(
        valence=valence,
        arousal=arousal,
        threat=threat,
        dominant=dominant,
        tags=tuple(dict.fromkeys(tags)),  # dedupe preserving order
    )
