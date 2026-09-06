"""Enforce honest line, branch, aggregate, module, and critical-path coverage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Final

DEFAULT_LINE_FLOOR: Final = 68.0
DEFAULT_BRANCH_FLOOR: Final = 55.0
DEFAULT_COMBINED_FLOOR: Final = 64.5
DEFAULT_MODULE_LINE_FLOOR: Final = 12.0
DEFAULT_MIN_STATEMENTS: Final = 20

# These are production control surfaces where an aggregate percentage can hide
# especially costly regressions. Values are line %, branch % respectively.
DEFAULT_CRITICAL_FLOORS: Final[dict[str, tuple[float, float]]] = {
    "norax/a2a/client.py": (97.0, 89.0),
    "norax/a2a/server.py": (87.0, 70.0),
    "norax/brain/agent_loop.py": (64.0, 55.0),
    "norax/brain/output_verifier.py": (88.0, 75.0),
    "norax/brain/weak_model_boost.py": (94.0, 84.0),
    "norax/dispatch/browser.py": (60.0, 62.0),
    "norax/dispatch/budget.py": (92.0, 82.0),
    "norax/dispatch/idempotency.py": (95.0, 83.0),
    "norax/dispatch/risk.py": (91.0, 87.0),
    "norax/gateway_client/__init__.py": (78.0, 67.0),
    "norax/mcp/server.py": (78.0, 72.0),
    "norax/memory/causal_graph.py": (98.0, 93.0),
    "norax/memory/hebbian.py": (90.0, 79.0),
    "norax/memory/index.py": (86.0, 72.0),
    "norax/memory/temporal_graph.py": (97.0, 92.0),
    "norax/memory/user_model.py": (95.0, 83.0),
    "norax/observability/log.py": (80.0, 62.0),
    # Runtime behavior is decomposed by ownership. Keep every production
    # boundary explicit so moving code cannot make the release gate easier.
    "norax/runtime/cognition.py": (55.0, 48.0),
    "norax/runtime/core.py": (70.0, 48.0),
    "norax/runtime/delivery.py": (74.0, 65.0),
    "norax/runtime/health.py": (88.0, 78.0),
    "norax/runtime/history.py": (72.0, 64.0),
    "norax/runtime/lifecycle.py": (74.0, 71.0),
    "norax/runtime/model_management.py": (62.0, 45.0),
    "norax/runtime/operations.py": (38.0, 40.0),
    "norax/runtime/session.py": (58.0, 45.0),
    "norax/runtime/turn_pipeline.py": (48.0, 34.0),
    "norax/runtime/validation.py": (80.0, 72.0),
}


def _percentage(
    container: Any,
    key: str,
    *,
    label: str,
    violations: list[str],
) -> float | None:
    if not isinstance(container, dict):
        violations.append(f"{label} is missing")
        return None
    value = container.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 100.0
    ):
        violations.append(f"{label} is missing or invalid")
        return None
    return float(value)


def _validated_policy(
    *,
    line_floor: float,
    branch_floor: float,
    combined_floor: float,
    module_line_floor: float,
    min_statements: int,
    critical_floors: dict[str, tuple[float, float]],
) -> None:
    for label, value in (
        ("line_floor", line_floor),
        ("branch_floor", branch_floor),
        ("combined_floor", combined_floor),
        ("module_line_floor", module_line_floor),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 100.0
        ):
            raise ValueError(f"{label} must be finite and between 0 and 100")
    if (
        isinstance(min_statements, bool)
        or not isinstance(min_statements, int)
        or min_statements < 1
    ):
        raise ValueError("min_statements must be a positive integer")
    if not isinstance(critical_floors, dict):
        raise ValueError("critical_floors must be a mapping")
    for filename, floors in critical_floors.items():
        if (
            not isinstance(filename, str)
            or not filename
            or not isinstance(floors, (tuple, list))
            or len(floors) != 2
        ):
            raise ValueError("critical_floors must map filenames to (line, branch) pairs")
        _validated_policy(
            line_floor=floors[0],
            branch_floor=floors[1],
            combined_floor=0.0,
            module_line_floor=0.0,
            min_statements=1,
            critical_floors={},
        )


def validate_coverage(
    report: dict[str, Any],
    *,
    line_floor: float = DEFAULT_LINE_FLOOR,
    branch_floor: float = DEFAULT_BRANCH_FLOOR,
    combined_floor: float = DEFAULT_COMBINED_FLOOR,
    module_line_floor: float = DEFAULT_MODULE_LINE_FLOOR,
    min_statements: int = DEFAULT_MIN_STATEMENTS,
    critical_floors: dict[str, tuple[float, float]] | None = None,
) -> list[str]:
    """Return deterministic policy violations from a coverage.py JSON report."""
    if critical_floors is None:
        selected_critical = DEFAULT_CRITICAL_FLOORS
    else:
        if not isinstance(critical_floors, dict):
            raise ValueError("critical_floors must be a mapping")
        selected_critical = dict(critical_floors)
    _validated_policy(
        line_floor=line_floor,
        branch_floor=branch_floor,
        combined_floor=combined_floor,
        module_line_floor=module_line_floor,
        min_statements=min_statements,
        critical_floors=selected_critical,
    )

    violations: list[str] = []
    if not isinstance(report, dict):
        return ["coverage report root must be an object"]
    meta = report.get("meta")
    if not isinstance(meta, dict) or meta.get("branch_coverage") is not True:
        violations.append("coverage report was not collected with branch coverage enabled")

    totals = report.get("totals")
    total_line = _percentage(
        totals,
        "percent_statements_covered",
        label="total line coverage",
        violations=violations,
    )
    total_branch = _percentage(
        totals,
        "percent_branches_covered",
        label="total branch coverage",
        violations=violations,
    )
    total_combined = _percentage(
        totals,
        "percent_covered",
        label="total combined coverage",
        violations=violations,
    )
    if total_line is not None and total_line < line_floor:
        violations.append(f"total line coverage {total_line:.2f}% is below {line_floor:.2f}%")
    if total_branch is not None and total_branch < branch_floor:
        violations.append(f"total branch coverage {total_branch:.2f}% is below {branch_floor:.2f}%")
    if total_combined is not None and total_combined < combined_floor:
        violations.append(
            f"total combined coverage {total_combined:.2f}% is below {combined_floor:.2f}%"
        )

    files = report.get("files")
    if not isinstance(files, dict) or not files:
        violations.append("coverage report contains no files")
        return violations

    for filename, details in sorted(files.items()):
        if not isinstance(filename, str) or not isinstance(details, dict):
            violations.append(f"{filename}: coverage entry is invalid")
            continue
        summary = details.get("summary")
        if not isinstance(summary, dict):
            violations.append(f"{filename}: coverage summary is missing")
            continue
        statements_raw = summary.get("num_statements")
        if (
            isinstance(statements_raw, bool)
            or not isinstance(statements_raw, int)
            or statements_raw < 0
        ):
            violations.append(f"{filename}: statement count is missing or invalid")
            continue
        covered = _percentage(
            summary,
            "percent_statements_covered",
            label=f"{filename}: line coverage",
            violations=violations,
        )
        if covered is not None and statements_raw >= min_statements and covered < module_line_floor:
            violations.append(
                f"{filename}: line coverage {covered:.2f}% is below the "
                f"{module_line_floor:.2f}% module floor ({statements_raw} statements)"
            )

    for filename, (required_line, required_branch) in sorted(selected_critical.items()):
        details = files.get(filename)
        if not isinstance(details, dict):
            violations.append(f"{filename}: critical module is missing from the coverage report")
            continue
        summary = details.get("summary")
        module_line = _percentage(
            summary,
            "percent_statements_covered",
            label=f"{filename}: critical line coverage",
            violations=violations,
        )
        module_branch = _percentage(
            summary,
            "percent_branches_covered",
            label=f"{filename}: critical branch coverage",
            violations=violations,
        )
        if module_line is not None and module_line < required_line:
            violations.append(
                f"{filename}: critical line coverage {module_line:.2f}% is below "
                f"{required_line:.2f}%"
            )
        if module_branch is not None and module_branch < required_branch:
            violations.append(
                f"{filename}: critical branch coverage {module_branch:.2f}% is below "
                f"{required_branch:.2f}%"
            )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--line-floor", type=float, default=DEFAULT_LINE_FLOOR)
    parser.add_argument("--branch-floor", type=float, default=DEFAULT_BRANCH_FLOOR)
    parser.add_argument(
        "--combined-floor",
        "--total-floor",
        dest="combined_floor",
        type=float,
        default=DEFAULT_COMBINED_FLOOR,
    )
    parser.add_argument(
        "--module-line-floor",
        "--module-floor",
        dest="module_line_floor",
        type=float,
        default=DEFAULT_MODULE_LINE_FLOOR,
    )
    parser.add_argument("--min-statements", type=int, default=DEFAULT_MIN_STATEMENTS)
    args = parser.parse_args()

    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        violations = validate_coverage(
            report,
            line_floor=args.line_floor,
            branch_floor=args.branch_floor,
            combined_floor=args.combined_floor,
            module_line_floor=args.module_line_floor,
            min_statements=args.min_statements,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"coverage guard: cannot validate {args.report}: {exc}")
        return 2

    if violations:
        print("coverage guard failed:")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print(
        "coverage guard passed: "
        f"line >= {args.line_floor:.2f}%, "
        f"branch >= {args.branch_floor:.2f}%, "
        f"combined >= {args.combined_floor:.2f}%, "
        f"modules >= {args.module_line_floor:.2f}%, "
        f"{len(DEFAULT_CRITICAL_FLOORS)} critical modules enforced"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
