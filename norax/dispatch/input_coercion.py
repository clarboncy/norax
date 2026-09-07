"""Strict, finite coercion for weak-model tool arguments."""

from __future__ import annotations

import math


def _coerce_int(value: object, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else default
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return int(stripped)
        except ValueError:
            try:
                parsed = float(stripped)
                return int(parsed) if math.isfinite(parsed) and parsed.is_integer() else default
            except (ValueError, OverflowError):
                return default
    return default


def _coerce_float(value: object, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            parsed = float(stripped)
            return parsed if math.isfinite(parsed) else default
        except ValueError:
            return default
    return default
