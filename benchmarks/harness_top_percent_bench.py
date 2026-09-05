from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Make the benchmark directly runnable from a source checkout without requiring
# an editable install or a caller-provided PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from norax.atomic import atomic_write_text  # noqa: E402
from norax.brain.harness_optimizer import write_report  # noqa: E402


def _hash_record(rec: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(rec, separators=(",", ":"), sort_keys=True)
    rec["hash"] = hashlib.sha256((str(rec["prev_hash"]) + body).encode("utf-8")).hexdigest()
    return rec


def _trajectory(
    idx: int,
    *,
    goal: str,
    task_type: str,
    tools: list[str],
    writes: int = 0,
    verified: bool = False,
    errors: list[str] | None = None,
    prev_hash: str,
) -> dict[str, Any]:
    errors = errors or []
    trace = []
    for i, tool in enumerate(tools):
        err = errors[i] if i < len(errors) else ""
        trace.append(
            {
                "name": tool,
                "args": {"path": f"tmp/harness_synthetic_{idx}.py"}
                if tool in {"read", "edit", "write"}
                else {"command": "pytest -q"}
                if tool == "exec"
                else {},
                "result": {"ok": not bool(err), "error": err} if err else {"ok": True},
            }
        )
    rec = {
        "schema_version": "v1",
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "kind": "agent_trajectory",
        "trace_id": f"bench-{idx}",
        "span_id": f"span-{idx}",
        "parent_span_id": None,
        "prev_hash": prev_hash,
        "attrs": {},
        "payload": {
            "model": "bench-model",
            "rounds": max(1, len(tools) // 2),
            "tool_calls": len(tools),
            "final_len": 96,
            "task_state": {
                "goal": goal,
                "type": task_type,
                "files_touched": [f"tmp/harness_synthetic_{idx}.py"] if writes else [],
                "progress": {
                    "tool_calls": len(tools),
                    "failures": len(errors),
                    "writes": writes,
                    "verified_after_write": verified,
                },
            },
            "trace": trace,
        },
    }
    return _hash_record(rec)


def build_synthetic_log(path: Path) -> None:
    prev = "0" * 64
    rows = []
    specs = [
        (
            "patch coding bug with verified tests",
            "coding",
            ["read", "edit", "read", "exec"],
            1,
            True,
            [],
        ),
        (
            "research paper and cite sources",
            "research",
            ["web_search", "web_fetch", "exec"],
            0,
            False,
            [],
        ),
        (
            "recover timeout by narrowing command",
            "ops",
            ["exec", "exec"],
            0,
            False,
            ["tool_timeout"],
        ),
        (
            "detect loop and re-read target",
            "coding",
            ["read", "edit", "read", "exec"],
            1,
            True,
            ["loop_detected"],
        ),
        ("write docs with verification", "writing", ["read", "edit", "read"], 1, True, []),
        (
            "multi-source memory retrieval",
            "general",
            ["search_memory", "read", "exec"],
            0,
            False,
            [],
        ),
        (
            "schema recovery from bad args",
            "ops",
            ["exec", "read", "exec"],
            0,
            False,
            ["bad_arguments"],
        ),
        ("small answer no tools", "general", [], 0, False, []),
    ]
    for i, spec in enumerate(specs):
        rec = _trajectory(
            i,
            goal=spec[0],
            task_type=spec[1],
            tools=spec[2],
            writes=spec[3],
            verified=spec[4],
            errors=spec[5],
            prev_hash=prev,
        )
        rows.append(rec)
        prev = rec["hash"]
    atomic_write_text(path, "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test Norax's internal harness evidence pipeline with synthetic records. "
            "This does not measure an external percentile."
        )
    )
    parser.add_argument("--work-dir", type=Path, default=Path("state/harness_bench"))
    parser.add_argument("--min-score", type=float, default=0.60)
    args = parser.parse_args(argv)
    if not 0.0 <= args.min_score <= 1.0:
        parser.error("--min-score must be between 0 and 1")
    event_log = args.work_dir / "synthetic_events.jsonl"
    out_dir = args.work_dir / "report"
    build_synthetic_log(event_log)
    report_path = write_report(event_log, out_dir, k=8, limit=100)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence = report["harness_evidence"]
    score = float(evidence["score"])
    integrity_ok = bool((report.get("event_log_integrity") or {}).get("ok"))
    scope_ok = evidence.get("scope") == "internal_trajectory_evidence_only"
    claim_ok = evidence.get("external_percentile_claim") is False
    gate_reasons = []
    if score < args.min_score:
        gate_reasons.append("score_below_threshold")
    if not integrity_ok:
        gate_reasons.append("event_log_integrity_failed")
    if not scope_ok:
        gate_reasons.append("unexpected_evidence_scope")
    if not claim_ok:
        gate_reasons.append("external_percentile_claim_present")
    payload = {
        "benchmark_kind": "synthetic_pipeline_smoke",
        "top_percent_claim": False,
        "score": score,
        "min_score": args.min_score,
        "gate_passed": not gate_reasons,
        "gate_reasons": gate_reasons,
        "assessment": evidence["assessment"],
        "scope": evidence["scope"],
        "external_percentile_claim": evidence["external_percentile_claim"],
        "dimensions": evidence["dimensions"],
        "trajectory_count": report["trajectory_count"],
        "integrity": report["event_log_integrity"],
        "diagnosis": {
            k: report["diagnosis"].get(k)
            for k in [
                "mean_difficulty",
                "mean_outcome_score",
                "verify_compliance_rate",
                "failure_rate",
            ]
        },
        "report": str(report_path),
    }
    print(json.dumps(payload, indent=2))
    return 0 if not gate_reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
