"""Regex-based secret scrubber. Runs on every event payload before write."""

from __future__ import annotations

import re
from typing import Any

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "<REDACTED:anthropic_key>"),
    (re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"), "<REDACTED:openai_key>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<REDACTED:aws_access_key>"),
    (re.compile(r"ghp_[A-Za-z0-9]{36}"), "<REDACTED:github_token>"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{82}"), "<REDACTED:github_pat>"),
    (re.compile(r"xoxb-[A-Za-z0-9-]{20,}"), "<REDACTED:slack_token>"),
    (re.compile(r"(?:[a-zA-Z0-9_-]+\.){2}[a-zA-Z0-9_-]{27,}"), "<REDACTED:jwt_or_discord_token>"),
    (re.compile(r"\b0x[a-fA-F0-9]{64}\b"), "<REDACTED:hex_secret>"),
    (
        re.compile(r"\b(?:[a-z]{3,8}\s+){11,23}[a-z]{3,8}\b", re.IGNORECASE),
        "<REDACTED:possible_seed_phrase>",
    ),
]


def _redact_str(s: str) -> str:
    for pat, repl in _PATTERNS:
        s = pat.sub(repl, s)
    return s


def redact(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_str(value)
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact(v) for v in value)
    return value
