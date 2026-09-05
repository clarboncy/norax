#!/usr/bin/env python3
"""Thin wrapper — delegates to the canonical verifier at norax/verify_event_chain.py.

The standalone script previously duplicated the hash-chain verification logic
with a simpler implementation that lacked --all-generations and cross-file
validation. It has been consolidated into ``norax.verify_event_chain`` which
is the single source of truth.

Usage:
  python scripts/verify_event_chain.py [args...]
  python -m norax.verify_event_chain [args...]   # equivalent, preferred
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from norax.verify_event_chain import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
