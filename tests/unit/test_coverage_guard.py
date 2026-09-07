"""Regression tests for the multi-dimensional coverage release policy."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "coverage_guard", Path(__file__).parents[2] / "scripts" / "coverage_guard.py"
)
assert _SPEC and _SPEC.loader
coverage_guard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(coverage_guard)


def _file(
    statements: int,
    line: float,
    branch: float,
    *,
    combined: float | None = None,
) -> dict[str, Any]:
    return {
        "summary": {
            "num_statements": statements,
            "percent_statements_covered": line,
            "percent_branches_covered": branch,
            "percent_covered": combined if combined is not None else min(line, branch),
        }
    }


def _report(
    line: float,
    branch: float,
    combined: float,
    files: dict[str, Any],
    *,
    branch_enabled: bool = True,
) -> dict[str, Any]:
    return {
        "meta": {"branch_coverage": branch_enabled},
        "totals": {
            "percent_statements_covered": line,
            "percent_branches_covered": branch,
            "percent_covered": combined,
        },
        "files": files,
    }


def _without_critical(**kwargs: Any) -> list[str]:
    return coverage_guard.validate_coverage(**kwargs, critical_floors={})


def test_guard_enforces_line_branch_and_combined_totals_independently() -> None:
    violations = _without_critical(
        report=_report(76.99, 64.99, 73.99, {"norax/core.py": _file(100, 80, 80)})
    )
    assert violations == [
        "total line coverage 76.99% is below 77.00%",
        "total branch coverage 64.99% is below 65.00%",
        "total combined coverage 73.99% is below 74.00%",
    ]


def test_guard_rejects_report_without_real_branch_measurement() -> None:
    report = _report(100, 100, 100, {"norax/core.py": _file(100, 100, 100)})
    report["meta"]["branch_coverage"] = False
    assert _without_critical(report=report) == [
        "coverage report was not collected with branch coverage enabled"
    ]
    report.pop("meta")
    assert _without_critical(report=report) == [
        "coverage report was not collected with branch coverage enabled"
    ]


def test_guard_applies_module_line_floor_without_conflating_branch_percentage() -> None:
    report = _report(
        80,
        70,
        75,
        {
            "norax/blind.py": _file(100, 11.99, 90),
            "norax/branchless.py": _file(100, 80, 0),
            "norax/tiny.py": _file(2, 0, 0),
        },
    )
    assert _without_critical(report=report) == [
        "norax/blind.py: line coverage 11.99% is below the 12.00% module floor (100 statements)"
    ]


def test_guard_requires_every_critical_module_and_both_of_its_floors() -> None:
    report = _report(
        80,
        70,
        75,
        {
            "norax/critical.py": _file(100, 79.99, 59.99),
            "norax/ordinary.py": _file(100, 80, 70),
        },
    )
    violations = coverage_guard.validate_coverage(
        report,
        critical_floors={
            "norax/critical.py": (80.0, 60.0),
            "norax/missing.py": (50.0, 40.0),
        },
    )
    assert violations == [
        "norax/critical.py: critical line coverage 79.99% is below 80.00%",
        "norax/critical.py: critical branch coverage 59.99% is below 60.00%",
        "norax/missing.py: critical module is missing from the coverage report",
    ]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 100.01, True, "90"])
def test_guard_rejects_invalid_policy_percentages(bad: Any) -> None:
    report = _report(80, 70, 75, {"norax/core.py": _file(100, 80, 70)})
    with pytest.raises(ValueError, match="line_floor"):
        _without_critical(report=report, line_floor=bad)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_guard_rejects_invalid_minimum_statement_policy(bad: Any) -> None:
    report = _report(80, 70, 75, {"norax/core.py": _file(100, 80, 70)})
    with pytest.raises(ValueError, match="min_statements"):
        _without_critical(report=report, min_statements=bad)


@pytest.mark.parametrize("bad", [1, "bad", {"norax/core.py": 90.0}])
def test_guard_rejects_malformed_critical_floor_policy(bad: Any) -> None:
    report = _report(80, 70, 75, {"norax/core.py": _file(100, 80, 70)})
    with pytest.raises(ValueError, match="critical_floors"):
        coverage_guard.validate_coverage(report, critical_floors=bad)


def test_guard_reports_malformed_metrics_instead_of_crashing() -> None:
    report: dict[str, Any] = {
        "meta": {"branch_coverage": True},
        "totals": {
            "percent_statements_covered": float("nan"),
            "percent_branches_covered": True,
            "percent_covered": "99",
        },
        "files": {
            "norax/bad-count.py": {
                "summary": {
                    "num_statements": True,
                    "percent_statements_covered": 100,
                }
            },
            "norax/no-summary.py": {},
        },
    }
    assert _without_critical(report=report) == [
        "total line coverage is missing or invalid",
        "total branch coverage is missing or invalid",
        "total combined coverage is missing or invalid",
        "norax/bad-count.py: statement count is missing or invalid",
        "norax/no-summary.py: coverage summary is missing",
    ]


def test_guard_rejects_missing_file_data_and_non_object_root() -> None:
    no_files = _report(100, 100, 100, {})
    assert _without_critical(report=no_files) == ["coverage report contains no files"]
    assert _without_critical(report=[]) == ["coverage report root must be an object"]


def test_default_critical_policy_covers_runtime_and_external_protocol_boundaries() -> None:
    critical = coverage_guard.DEFAULT_CRITICAL_FLOORS
    runtime_boundaries = {
        "norax/runtime/backoff.py",
        "norax/runtime/capability_registry.py",
        "norax/runtime/circuit_breaker.py",
        "norax/runtime/cognition.py",
        "norax/runtime/core.py",
        "norax/runtime/delivery.py",
        "norax/runtime/health.py",
        "norax/runtime/history.py",
        "norax/runtime/lifecycle.py",
        "norax/runtime/model_management.py",
        "norax/runtime/operations.py",
        "norax/runtime/session.py",
        "norax/runtime/turn_pipeline.py",
        "norax/runtime/validation.py",
        "norax/runtime/watchdog.py",
    }
    assert runtime_boundaries <= critical.keys()
    assert critical["norax/runtime/core.py"] == (70.0, 48.0)
    assert critical["norax/runtime/capability_registry.py"] == (100.0, 100.0)
    assert critical["norax/runtime/cognition.py"] == (100.0, 100.0)
    assert critical["norax/runtime/delivery.py"] == (100.0, 100.0)
    assert critical["norax/runtime/turn_pipeline.py"] == (48.0, 34.0)
    assert critical["norax/runtime/lifecycle.py"] == (74.0, 71.0)
    assert critical["norax/runtime/operations.py"] == (100.0, 100.0)
    assert critical["norax/runtime/health.py"] == (100.0, 100.0)
    assert critical["norax/runtime/history.py"] == (100.0, 100.0)
    assert critical["norax/runtime/model_management.py"] == (100.0, 100.0)
    assert critical["norax/runtime/session.py"] == (100.0, 100.0)
    assert critical["norax/runtime/validation.py"] == (100.0, 100.0)
    assert critical["norax/observability/log.py"] == (80.0, 62.0)
    assert critical["norax/prompt/assembler.py"] == (100.0, 100.0)
    assert "norax/gateway_client/__init__.py" in critical
    assert "norax/mcp/server.py" in critical
    assert "norax/a2a/server.py" in critical
    assert critical["norax/a2a/client.py"] == (97.0, 89.0)
    assert "norax/dispatch/browser.py" in critical
    assert critical["norax/dispatch/deep_research.py"] == (100.0, 100.0)
    assert critical["norax/dispatch/firecrawl.py"] == (100.0, 100.0)
    assert critical["norax/dispatch/idempotency.py"] == (100.0, 100.0)
    assert critical["norax/dispatch/input_coercion.py"] == (100.0, 100.0)
    assert critical["norax/dispatch/memory_tool.py"] == (100.0, 100.0)
    assert critical["norax/dispatch/web_common.py"] == (100.0, 100.0)
    assert "norax/memory/index.py" in critical
    assert critical["norax/memory/causal_graph.py"] == (98.0, 93.0)
    assert critical["norax/memory/hebbian.py"] == (90.0, 79.0)
    assert critical["norax/memory/temporal_graph.py"] == (97.0, 92.0)
    assert critical["norax/memory/user_model.py"] == (95.0, 83.0)
    newly_closed = {
        "norax/brain/hot_path/semantic_router.py",
        "norax/brain/hot_path/task_classifier_v2.py",
        "norax/brain/prediction_network.py",
        "norax/memory/consolidator.py",
        "norax/memory/decay.py",
        "norax/memory/fact_evolution.py",
        "norax/memory/retrievers/vector_store.py",
        "norax/memory/tool_experience.py",
    }
    assert all(critical[name] == (100.0, 100.0) for name in newly_closed)


def test_cli_passes_a_valid_report_and_supports_legacy_option_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    files = {
        filename: _file(100, line, branch)
        for filename, (line, branch) in coverage_guard.DEFAULT_CRITICAL_FLOORS.items()
    }
    files["norax/ordinary.py"] = _file(100, 80, 70)
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(_report(80, 70, 75, files)), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coverage_guard.py",
            str(path),
            "--total-floor",
            "70",
            "--module-floor",
            "10",
        ],
    )
    assert coverage_guard.main() == 0
    output = capsys.readouterr().out
    assert "line >= 77.00%" in output
    assert "branch >= 65.00%" in output
    assert f"{len(coverage_guard.DEFAULT_CRITICAL_FLOORS)} critical modules enforced" in output


def test_cli_fails_closed_on_invalid_json_and_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "coverage.json"
    path.write_text("{", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["coverage_guard.py", str(path)])
    assert coverage_guard.main() == 2
    assert "cannot validate" in capsys.readouterr().out

    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["coverage_guard.py", str(path), "--line-floor", "nan"],
    )
    assert coverage_guard.main() == 2
    assert "line_floor must be finite" in capsys.readouterr().out
