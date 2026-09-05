#!/usr/bin/env python3
"""Host integration smoke for the standalone perception/action helpers.

The script skips unavailable host backends explicitly. It does not require a
magic element count, a specific theme, a private display, or deleted telemetry
modules in order to claim success.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(os.environ.get("NORAX_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
TOOLS = PROJECT_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    from action_gate import gate
    from motor_cortex import MotorCortex
    from perception import perceive, status

    results: list[dict[str, Any]] = []

    critical = gate("rm -rf /tmp/integration-probe", action_type="shell")
    require(critical["allowed"] is False, "critical action gate did not fail closed")
    results.append({"check": "critical_gate", "ok": True})

    motor = MotorCortex()
    execution = motor.execute_with_verify("exec_command", "printf integration-smoke")
    require(execution.success, execution.verification)
    failure = motor.execute_with_verify("exec_command", "sh -c 'exit 7'")
    require(not failure.success, "non-zero command was reported as successful")
    require(failure.attempts == 1, "unchanged failed command was retried")
    results.append({"check": "motor_execution", "ok": True})

    capabilities = status()
    available_tier = next(
        (tier for tier in ("atspi", "cdp", "ocr") if capabilities.get(tier)), None
    )
    if available_tier:
        state = perceive(tier=available_tier, save=False)
        require(state.tier_used in {available_tier, "none"}, "perception returned the wrong tier")
        require(not state.error, state.error)
        results.append(
            {
                "check": "perception",
                "ok": not bool(state.error),
                "tier": available_tier,
                "elements": len(state.elements),
                "error": state.error or None,
            }
        )
    else:
        results.append({"check": "perception", "ok": None, "skipped": "no backend"})

    if capabilities.get("ocr"):
        snap = subprocess.run(
            [sys.executable, str(TOOLS / "dmap.py"), "snap", "--quiet"],
            capture_output=True,
            text=True,
            timeout=35,
            check=False,
        )
        require(snap.returncode == 0, f"dmap snap failed: {snap.stderr[:300]}")
        read = subprocess.run(
            [sys.executable, str(TOOLS / "dmap.py"), "read", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        require(read.returncode == 0, f"dmap read failed: {read.stderr[:300]}")
        index = json.loads(read.stdout)
        require(isinstance(index, dict) and isinstance(index.get("elements"), list), "bad index")
        results.append({"check": "dmap", "ok": True, "elements": len(index["elements"])})
    else:
        results.append({"check": "dmap", "ok": None, "skipped": "no OCR backend"})

    print(json.dumps({"ok": True, "checks": results}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        raise SystemExit(1) from exc
