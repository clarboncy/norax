"""Context layer — rolling window + correction gate + sleep-flush."""

from .correction import (
    Claim,
    CorrectionGate,
    CorrectionResult,
    extract_claims,
)
from .sleep_flush import (
    Candidate,
    FlushResult,
    SleepFlusher,
    extract_candidates,
)
from .window import (
    CharDiv4Tokenizer,
    EvictionResult,
    Frame,
    FrameKind,
    RollingWindow,
    TokenCounter,
)

__all__ = [
    "Frame",
    "FrameKind",
    "RollingWindow",
    "EvictionResult",
    "CharDiv4Tokenizer",
    "TokenCounter",
    "CorrectionGate",
    "CorrectionResult",
    "Claim",
    "extract_claims",
    "SleepFlusher",
    "FlushResult",
    "Candidate",
    "extract_candidates",
]
