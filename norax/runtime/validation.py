"""Validated runtime configuration primitives.

Kept independent from Runtime so startup policy is testable without
constructing the message loop.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re

log = logging.getLogger("norax.runtime.core")


def _flag_enabled(value: object, *, default: bool = False) -> bool:
    """Parse an internal feature flag without truthy-string surprises."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _explicit_result_ok(value: object) -> bool:
    """Accept success only from the protocol's literal boolean ``true``."""
    return isinstance(value, dict) and value.get("ok") is True


def _runtime_choice(value: object, choices: set[str], *, default: str, label: str) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in choices:
            return normalized
    log.warning("runtime.invalid_choice key=%s value=%r default=%s", label, value, default)
    return default


def _model_identifier(value: object, *, label: str = "model") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    model = value.strip()
    if not model or len(model) > 512:
        raise ValueError(f"{label} must contain 1-512 characters")
    if any(not character.isprintable() or character.isspace() for character in model):
        raise ValueError(f"{label} must not contain whitespace or control characters")
    return model


def _model_identifiers(value: object, *, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    if len(value) > 32:
        raise ValueError(f"{label} must contain at most 32 models")
    return [_model_identifier(item, label=f"{label} entry") for item in value]


def _provider_identifier(value: object, *, label: str = "provider") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    provider = value.strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", provider):
        raise ValueError(f"{label} contains unsupported characters")
    return provider


def _is_loopback_host(host: str) -> bool:
    """Return whether a listener host is restricted to the local machine."""
    normalized = host.strip().lower().strip("[]")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _a2a_advertised_url(host: str, port: int, configured: object) -> str:
    """Build an honest Agent Card URL and reject ambiguous wildcard binds."""
    if configured is not None and str(configured).strip():
        return str(configured).strip()
    normalized = host.strip().strip("[]")
    if normalized in {"0.0.0.0", "::"}:
        raise ValueError("A2A base_url is required when binding to a wildcard address")
    url_host = f"[{normalized}]" if ":" in normalized else normalized
    return f"http://{url_host}:{port}"


def _bounded_config_int(value: object, *, label: str, minimum: int, maximum: int) -> int:
    """Parse a bounded integer setting without accepting booleans or decimals."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer between {minimum} and {maximum}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().lstrip("+").isdigit():
        parsed = int(value)
    else:
        raise ValueError(f"{label} must be an integer between {minimum} and {maximum}")
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be an integer between {minimum} and {maximum}")
    return parsed


def _bounded_environment_int(
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Read an operator integer while keeping a bad override non-fatal."""
    raw = os.environ.get(name, str(default))
    try:
        return _bounded_config_int(raw, label=name, minimum=minimum, maximum=maximum)
    except ValueError:
        log.warning("runtime.invalid_environment key=%s value=%r default=%d", name, raw, default)
        return default


def _apply_configured_fleet_startup_fixes() -> bool:
    """Run the private fleet hook only after an explicit operator opt-in."""
    if not _flag_enabled(os.environ.get("NORAX_APPLY_FLEET_STARTUP_FIXES", "")):
        return False
    from ..ops.fleet_healthcheck import apply_startup_fixes

    apply_startup_fixes()
    return True
