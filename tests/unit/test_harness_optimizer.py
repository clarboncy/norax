from __future__ import annotations

import json
from pathlib import Path

from norax.brain.harness_optimizer import (
    HarnessTrajectory,
    analyze_harness,
    harness_evidence_score,
    load_trajectories,
    propose_harness_updates,
    select_coreset,
    validate_event_log_hash_chain,
    write_report,
)


def _event(
    goal: str,
    *,
    trace_id: str,
    failures: int = 0,
    writes: int = 0,
    verified: bool = False,
    tools: list[str] | None = None,
) -> dict:
    trace = []
    for i, name in enumerate(tools or ["read", "exec"]):
        ok = i >= failures
        trace.append(
            {
                "name": name,
                "args": {"path": f"file{i}.py"} if name in {"read", "edit"} else {},
                "result": {"ok": ok, "error": "bad_arguments"} if not ok else {"ok": True},
            }
        )
    return {
        "kind": "agent_trajectory",
        "trace_id": trace_id,
        "ts": "2026-06-09T00:00:00Z",
        "payload": {
            "rounds": 3 + failures,
            "tool_calls": len(trace),
            "final_len": 42,
            "task_state": {
                "goal": goal,
                "type": "coding",
                "files_touched": ["x.py"] if writes else [],
                "progress": {
                    "tool_calls": len(trace),
                    "failures": failures,
                    "writes": writes,
                    "verified_after_write": verified,
                },
            },
            "trace": trace,
        },
    }


def test_harness_trajectory_scores_failures_and_missing_verify() -> None:
    tr = HarnessTrajectory.from_event(
        _event(
            "fix bug", trace_id="a", failures=1, writes=1, verified=False, tools=["read", "edit"]
        )
    )
    assert tr is not None
    assert tr.failures == 1
    assert tr.writes == 1
    assert tr.verified_after_write is False
    assert tr.difficulty > 0.4
    assert "bad_arguments" in tr.errors


def test_select_coreset_prefers_difficult_diverse_items() -> None:
    items = [
        HarnessTrajectory.from_event(
            _event("fix python import", trace_id="a", failures=2, tools=["read", "edit"])
        ),
        HarnessTrajectory.from_event(
            _event("fix python import again", trace_id="b", failures=2, tools=["read", "edit"])
        ),
        HarnessTrajectory.from_event(
            _event(
                "research source verification",
                trace_id="c",
                failures=1,
                tools=["web_search", "web_fetch"],
            )
        ),
    ]
    selected = select_coreset([x for x in items if x is not None], k=2)
    assert len(selected) == 2
    assert {s.trace_id for s in selected} <= {"a", "b", "c"}
    assert len({s.fingerprint for s in selected}) == 2


def test_write_report_from_event_log(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    rows = [
        _event(
            "fix bug", trace_id="a", failures=1, writes=1, verified=False, tools=["read", "edit"]
        ),
        _event(
            "check docs",
            trace_id="b",
            failures=0,
            writes=0,
            verified=False,
            tools=["web_search", "web_fetch"],
        ),
    ]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    loaded = load_trajectories(log)
    assert len(loaded) == 2
    report_path = write_report(log, tmp_path / "out", k=2)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["trajectory_count"] == 2
    assert report["diagnosis"]["coreset_size"] == 2
    proposals = propose_harness_updates(report["diagnosis"])
    assert any("agent_trajectory" in p for p in proposals)
    assert (tmp_path / "out" / "rho_report_latest.json").exists()


def test_unassessed_trajectory_is_not_mined_as_training_evidence(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    unassessed = _event("answer a subjective question", trace_id="unknown", tools=[])
    unassessed["payload"].update(
        {
            "outcome": "accepted_unverified",
            "outcome_score": 0.73,
            "training_eligible": False,
        }
    )
    eligible = _event("verify a file", trace_id="checked", tools=["read"])
    log.write_text(
        json.dumps(unassessed) + "\n" + json.dumps(eligible) + "\n",
        encoding="utf-8",
    )

    loaded = load_trajectories(log)

    assert [trajectory.trace_id for trajectory in loaded] == ["checked"]


def test_stringified_tool_results_are_still_mined() -> None:
    row = _event("recover timeout", trace_id="s", failures=0, tools=["exec"])
    row["payload"]["trace"][0]["result"] = json.dumps({"ok": False, "error": "tool_timeout"})
    tr = HarnessTrajectory.from_event(row)
    assert tr is not None
    assert tr.errors == ["tool_timeout"]
    assert tr.failures == 1
    assert tr.outcome_score < 0.55


def test_truthy_non_boolean_evidence_is_not_counted_as_success() -> None:
    row = _event("reject malformed evidence", trace_id="malformed", tools=["read"])
    row["payload"]["trace"][0]["result"] = {"ok": "false"}
    row["payload"]["task_state"]["progress"].update(
        {"failures": 0, "writes": 1, "verified_after_write": "false"}
    )

    trajectory = HarnessTrajectory.from_event(row)

    assert trajectory is not None
    assert trajectory.errors == ["failed"]
    assert trajectory.failures == 1
    assert trajectory.verified_after_write is False

    evidence = harness_evidence_score(
        [trajectory],
        {"verify_compliance_rate": 1.0, "failure_rate": 0.0, "mean_outcome_score": 1.0},
        {"ok": "false", "valid_ratio": 0.0},
    )
    assert evidence["assessment"] == "integrity_failed"


def test_report_contains_readiness_and_integrity_for_unhashed_log(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    rows = [
        _event(
            "verified patch", trace_id="v", writes=1, verified=True, tools=["edit", "read", "exec"]
        )
    ]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    report_path = write_report(log, tmp_path / "out", k=1)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["trajectory_count"] == 1
    assert report["diagnosis"]["verify_compliance_rate"] == 1.0
    assert report["harness_evidence"]["dimensions"]["sample_volume"] == 0.01
    assert report["event_log_integrity"]["ok"] is False


def test_validate_event_log_hash_chain_detects_corruption(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    rec = {
        "schema_version": "v1",
        "ts": "2026-06-09T00:00:00.000Z",
        "kind": "agent_trajectory",
        "trace_id": "t",
        "span_id": "s",
        "parent_span_id": None,
        "prev_hash": "0" * 64,
        "attrs": {},
        "payload": {
            "rounds": 1,
            "tool_calls": 0,
            "final_len": 25,
            "task_state": {"goal": "x", "progress": {}},
            "trace": [],
        },
    }
    import hashlib

    body = json.dumps(rec, separators=(",", ":"), sort_keys=True)
    rec["hash"] = hashlib.sha256((rec["prev_hash"] + body).encode("utf-8")).hexdigest()
    log.write_text(json.dumps(rec, separators=(",", ":")) + "\n", encoding="utf-8")
    assert validate_event_log_hash_chain(log)["ok"] is True
    corrupted = json.loads(log.read_text(encoding="utf-8"))
    corrupted["payload"]["final_len"] = 1
    log.write_text(json.dumps(corrupted, separators=(",", ":")) + "\n", encoding="utf-8")
    check = validate_event_log_hash_chain(log)
    assert check["ok"] is False
    assert check["error"] == "hash_mismatch"


def test_harness_evidence_score_refuses_external_percentile_claim() -> None:
    items = []
    for i in range(8):
        tr = HarnessTrajectory.from_event(
            _event(
                f"verified patch {i}",
                trace_id=str(i),
                writes=1,
                verified=True,
                tools=["read", "edit", "read", "exec"],
            )
        )
        assert tr is not None
        tr.task_type = "coding" if i < 4 else "research"
        tr.fingerprint = f"unique {i}"
        tr.outcome_score = 0.95
        items.append(tr)
    diagnosis = {
        "verify_compliance_rate": 1.0,
        "failure_rate": 0.0,
        "mean_outcome_score": 0.95,
        "coreset_size": len(items),
    }
    score = harness_evidence_score(
        items,
        diagnosis,
        {"ok": True, "valid_ratio": 1.0},
    )
    assert score["assessment"] == "insufficient_live_volume"
    assert score["scope"] == "internal_trajectory_evidence_only"
    assert score["external_percentile_claim"] is False


def test_harness_analysis_writes_reports_but_never_source(tmp_path: Path) -> None:
    event_log = tmp_path / "events.jsonl"
    event_log.write_text(
        json.dumps(
            _event(
                "fix bug",
                trace_id="analysis",
                failures=1,
                writes=1,
                verified=False,
                tools=["read", "edit"],
            )
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "reports"
    before = set(tmp_path.rglob("*"))

    result = analyze_harness(event_log, out_dir)

    assert result["source_mutations"] == 0
    assert Path(result["report_path"]).is_relative_to(out_dir)
    created = set(tmp_path.rglob("*")) - before
    assert created
    assert all(path == out_dir or path.is_relative_to(out_dir) for path in created)
