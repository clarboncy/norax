"""CLI wrapper for Norax retrospective harness optimization reports."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from norax.brain.harness_optimizer import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
