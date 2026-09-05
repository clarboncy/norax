"""Shared remote-node protocol helpers."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from pathlib import Path
from typing import Any

from ..atomic import read_bounded_text

_MAX_TOKEN_BYTES = 8 * 1024
_MAX_TOKEN_CHARS = 4_096


def read_private_token(path: Path | str) -> str:
    """Read one bounded non-symlink credential with owner-only permissions."""
    token = read_bounded_text(
        Path(path).expanduser(),
        max_bytes=_MAX_TOKEN_BYTES,
        require_private=True,
    ).strip()
    if (
        not token
        or len(token) > _MAX_TOKEN_CHARS
        or any(character.isspace() for character in token)
    ):
        raise ValueError("token file must contain one non-empty whitespace-free token")
    return token


def new_token() -> str:
    return "nxn_" + secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def sign(token: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(token.encode(), raw, hashlib.sha256).hexdigest()


def verify(token: str, payload: dict[str, Any], signature: str) -> bool:
    return hmac.compare_digest(sign(token, payload), signature)


def now_ms() -> int:
    return int(time.time() * 1000)
