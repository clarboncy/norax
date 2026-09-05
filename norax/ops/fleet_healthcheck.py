"""Fleet healthcheck maintenance — sync canonical checks at startup."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("norax.ops.fleet_healthcheck")

CHECKS_PATH = Path.home() / ".config/fleet-healthcheck/checks.tsv"
CANONICAL_CHECKS = Path(__file__).resolve().parents[2] / "config/fleet-healthcheck/checks.tsv"


def _sync_canonical_checks() -> dict:
    """Sync the canonical checks.tsv from the repo to the user config dir."""
    changed: list[str] = []
    errors: list[str] = []

    if not CANONICAL_CHECKS.is_file():
        errors.append(f"canonical checks.tsv missing: {CANONICAL_CHECKS}")
        return {"ok": False, "changed": changed, "errors": errors}

    canonical = CANONICAL_CHECKS.read_text(encoding="utf-8")

    if CHECKS_PATH.is_file():
        original = CHECKS_PATH.read_text(encoding="utf-8")
        if original != canonical:
            changed.append("sync checks.tsv from repo canonical")
            CHECKS_PATH.write_text(canonical, encoding="utf-8")
    else:
        changed.append("install checks.tsv from repo canonical")
        CHECKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHECKS_PATH.write_text(canonical, encoding="utf-8")

    return {"ok": not errors, "changed": changed, "errors": errors}


def apply_startup_fixes() -> None:
    """Idempotent one-shot fixes at runtime startup."""
    try:
        result = _sync_canonical_checks()
        if result["changed"]:
            log.info(
                "fleet_healthcheck: %s",
                "; ".join(result["changed"]),
            )
        for err in result["errors"]:
            log.warning("fleet_healthcheck: %s", err)
    except Exception:
        log.exception("fleet_healthcheck startup fix failed")
