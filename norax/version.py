"""Single source of truth for runtime version and build metadata.

Prefers installed package metadata; falls back to hardcoded value when
running from a source checkout without installed metadata.
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

try:
    __version__ = _pkg_version("norax")
except PackageNotFoundError:
    __version__ = "0.11.0"


def _git_metadata() -> tuple[str, str]:
    """Best-effort provenance for source-checkout deployments."""
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        epoch = subprocess.run(
            ["git", "show", "-s", "--format=%ct", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        build_date = datetime.fromtimestamp(float(epoch), UTC).isoformat()
        return commit or "unknown", build_date
    except (OSError, ValueError, subprocess.SubprocessError):
        return "unknown", "unknown"


_GIT_COMMIT, _GIT_DATE = _git_metadata()
# CI/deployment values take precedence; checkout metadata is a safe fallback.
BUILD_COMMIT = os.environ.get("NORAX_BUILD_COMMIT") or _GIT_COMMIT
BUILD_DATE = os.environ.get("NORAX_BUILD_DATE") or _GIT_DATE


def build_info() -> dict[str, str]:
    """Return version + build metadata for /status and event logging."""
    return {
        "version": __version__,
        "build_commit": BUILD_COMMIT,
        "build_date": BUILD_DATE,
    }
