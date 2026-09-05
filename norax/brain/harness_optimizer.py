"""Retrospective Harness Optimization utilities for Norax.

This is the production-facing, deterministic half of RHO: persist complete
turn trajectories, mine difficult/diverse failures from the hash-chained event
log, and emit harness-update candidates for review. It never rewrites runtime
source. A proposed change belongs in the normal inspect/patch/test/review path;
trajectory heuristics alone are not sufficient evidence to deploy code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_text

FAILURE_KINDS = {
    "bad_arguments",
    "duplicate_call_blocked",
    "duplicate_mutation_blocked",
    "loop_detected",
    "risk_denied",
    "tool_exception",
    "tool_timeout",
    "unknown_tool",
}
VERIFY_TOOLS = {"read", "list_dir", "exec", "shell"}
WRITE_TOOLS = {"write", "write_chunk", "edit"}


@dataclass(slots=True)
class HarnessTrajectory:
    """Compact task trajectory mined from `agent_trajectory` events."""

    trace_id: str
    ts: str
    goal: str
    task_type: str
    rounds: int
    tool_calls: int
    failures: int
    writes: int
    verified_after_write: bool
    final_len: int
    tools: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    difficulty: float = 0.0
    fingerprint: str = ""
    outcome_score: float = 0.0
    outcome_label: str = ""
    training_eligible: bool = True

    @classmethod
    def from_event(cls, rec: dict[str, Any]) -> HarnessTrajectory | None:
        payload = rec.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}
        if not isinstance(payload, dict):
            payload = {}
        state = payload.get("task_state") or {}
        progress = state.get("progress") or {}
        trace = payload.get("trace") or []
        if not isinstance(trace, list):
            trace = []
        tools: list[str] = []
        errors: list[str] = []
        for item in trace:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            if name:
                tools.append(name)
            result = item.get("result") or {}
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                    result = parsed if isinstance(parsed, dict) else {"raw": result}
                except json.JSONDecodeError:
                    result = {"raw": result}
            if not isinstance(result, dict) or result.get("ok") is not True:
                if not isinstance(result, dict):
                    result = {}
                errors.append(str(result.get("error") or result.get("detail") or "failed"))
        goal = str(state.get("goal") or payload.get("goal") or "").strip()
        if not goal and not trace:
            return None
        obj = cls(
            trace_id=str(rec.get("trace_id") or ""),
            ts=str(rec.get("ts") or ""),
            goal=goal,
            task_type=str(state.get("type") or payload.get("task_type") or "general"),
            rounds=int(payload.get("rounds") or progress.get("tool_rounds") or 0),
            tool_calls=int(payload.get("tool_calls") or progress.get("tool_calls") or len(trace)),
            failures=int(progress.get("failures") or len(errors)),
            writes=int(progress.get("writes") or 0),
            verified_after_write=progress.get("verified_after_write") is True,
            final_len=int(payload.get("final_len") or 0),
            tools=tools,
            errors=errors,
            files_touched=[str(x) for x in (state.get("files_touched") or [])],
        )
        explicit = payload.get("outcome_score")
        if explicit is not None and isinstance(explicit, (int, float)):
            obj.outcome_score = float(explicit)
        else:
            obj.outcome_score = score_outcome(obj)
        obj.outcome_label = str(payload.get("outcome") or "")
        raw_training_eligible = payload.get("training_eligible")
        obj.training_eligible = (
            raw_training_eligible is True
            if raw_training_eligible is not None
            else obj.outcome_label not in {"accepted_unverified", "interrupted"}
        )
        obj.difficulty = score_difficulty(obj)
        obj.fingerprint = fingerprint(obj)
        return obj


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_trajectories(event_log: Path, *, limit: int = 1000) -> list[HarnessTrajectory]:
    out: list[HarnessTrajectory] = []
    for rec in iter_jsonl(event_log):
        if rec.get("kind") != "agent_trajectory":
            continue
        tr = HarnessTrajectory.from_event(rec)
        if tr is not None and tr.training_eligible:
            out.append(tr)
    return out[-limit:]


def score_difficulty(t: HarnessTrajectory) -> float:
    score = 0.15
    score += min(t.rounds, 20) * 0.035
    score += min(t.tool_calls, 40) * 0.018
    score += min(t.failures, 8) * 0.12
    score += 0.20 if t.errors else 0.0
    score += 0.20 if t.writes and not t.verified_after_write else 0.0
    score += 0.10 if t.final_len < 20 and t.goal else 0.0
    score += 0.08 if len(set(t.tools)) >= 4 else 0.0
    return max(0.0, min(score, 1.0))


def score_outcome(t: HarnessTrajectory) -> float:
    score = 0.55
    score += 0.18 if t.tool_calls == 0 and t.final_len >= 20 else 0.0
    score += 0.18 if t.tool_calls > 0 and not t.errors else 0.0
    score += 0.22 if t.writes and t.verified_after_write else 0.0
    score -= min(t.failures, 8) * 0.10
    score -= 0.22 if t.errors else 0.0
    score -= 0.25 if t.writes and not t.verified_after_write else 0.0
    score -= 0.10 if t.final_len < 20 and t.goal else 0.0
    return max(0.0, min(score, 1.0))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_./-]{2,}", text.lower()))


def fingerprint(t: HarnessTrajectory) -> str:
    parts = [t.task_type]
    parts.extend(t.tools[-8:])
    parts.extend(sorted(set(t.errors))[:5])
    parts.extend(Path(p).suffix or p for p in t.files_touched[:6])
    parts.extend(sorted(_tokens(t.goal))[:12])
    return " ".join(parts)


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def select_coreset(
    items: list[HarnessTrajectory], *, k: int = 10, diversity_weight: float = 0.35
) -> list[HarnessTrajectory]:
    if not items or k <= 0:
        return []
    pool = sorted(items, key=lambda x: x.difficulty, reverse=True)
    selected: list[HarnessTrajectory] = []
    while pool and len(selected) < k:
        best_idx = 0
        best_score = -math.inf
        for i, item in enumerate(pool):
            similarity = max(
                (jaccard(item.fingerprint, s.fingerprint) for s in selected), default=0.0
            )
            score = item.difficulty * (1.0 - diversity_weight * similarity)
            if score > best_score:
                best_idx, best_score = i, score
        selected.append(pool.pop(best_idx))
    return selected


def diagnose(coreset: list[HarnessTrajectory]) -> dict[str, Any]:
    tool_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    missing_verify = 0
    loopish = 0
    for t in coreset:
        tool_counts.update(t.tools)
        error_counts.update(t.errors)
        task_counts[t.task_type] += 1
        if t.writes and not t.verified_after_write:
            missing_verify += 1
        if any(
            e in {"duplicate_call_blocked", "duplicate_mutation_blocked", "loop_detected"}
            for e in t.errors
        ):
            loopish += 1

    self_validation: list[str] = []
    self_consistency: list[str] = []
    if error_counts:
        top = ", ".join(f"{k}={v}" for k, v in error_counts.most_common(5))
        self_validation.append(f"Frequent tool failures: {top}.")
    if missing_verify:
        self_validation.append(
            f"{missing_verify}/{len(coreset)} difficult tasks wrote files without verified read/test evidence."
        )
    if loopish:
        self_validation.append(
            f"{loopish}/{len(coreset)} difficult tasks hit duplicate/loop protection."
        )
    if tool_counts:
        spread = ", ".join(f"{k}={v}" for k, v in tool_counts.most_common(8))
        self_consistency.append(f"Action mix across hard tasks: {spread}.")
    if len(task_counts) > 1:
        self_consistency.append(
            "Hard trajectories span multiple task types; harness updates must be general, not task-specific."
        )
    if not self_validation:
        self_validation.append(
            "No repeated failure class dominated the selected coreset; preserve current discipline and improve evidence capture."
        )
    if not self_consistency:
        self_consistency.append(
            "Insufficient trajectory diversity for cross-run consistency; collect more agent_trajectory events."
        )

    mean_outcome = round(sum(t.outcome_score for t in coreset) / max(1, len(coreset)), 3)
    verify_den = sum(1 for t in coreset if t.writes)
    verify_rate = round(
        sum(1 for t in coreset if t.writes and t.verified_after_write) / max(1, verify_den), 3
    )
    failure_rate = round(
        sum(1 for t in coreset if t.failures or t.errors) / max(1, len(coreset)), 3
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "coreset_size": len(coreset),
        "mean_difficulty": round(sum(t.difficulty for t in coreset) / max(1, len(coreset)), 3),
        "mean_outcome_score": mean_outcome,
        "verify_compliance_rate": verify_rate,
        "failure_rate": failure_rate,
        "self_validation": self_validation,
        "self_consistency": self_consistency,
        "top_errors": error_counts.most_common(10),
        "top_tools": tool_counts.most_common(10),
        "tasks": [
            {
                "trace_id": t.trace_id,
                "ts": t.ts,
                "difficulty": round(t.difficulty, 3),
                "outcome_score": round(t.outcome_score, 3),
                "type": t.task_type,
                "rounds": t.rounds,
                "tool_calls": t.tool_calls,
                "failures": t.failures,
                "writes": t.writes,
                "verified_after_write": t.verified_after_write,
                "errors": t.errors[:6],
                "goal": t.goal[:240],
                "fingerprint": t.fingerprint[:500],
            }
            for t in coreset
        ],
    }


def propose_harness_updates(diagnosis: dict[str, Any]) -> list[str]:
    errors = dict(diagnosis.get("top_errors") or [])
    proposals = [
        "Persist every completed turn as an `agent_trajectory` event with TASK_STATE, tool trace, rounds, final length, and verification flags.",
        "Run retrospective coreset selection during sleep/maintenance: prioritize high-difficulty, diverse trajectories over raw recency.",
        "Before promoting a new skill/tool, require self-validation plus self-consistency evidence and a strictly positive pairwise preference score over baseline behavior.",
    ]
    if errors.get("bad_arguments"):
        proposals.append(
            "Strengthen tool-call schema nudges for missing/placeholder arguments and force re-read before retrying repeated bad args."
        )
    if errors.get("tool_timeout"):
        proposals.append(
            "Add timeout-aware recovery: narrow scope, reduce command size, or switch tools after a timeout instead of repeating the same call."
        )
    if (
        errors.get("duplicate_call_blocked")
        or errors.get("duplicate_mutation_blocked")
        or errors.get("loop_detected")
    ):
        proposals.append(
            "Promote loop recovery skill: after duplicate/loop detection, summarize stale assumption, re-read target state, then change tool or finish."
        )
    if any("wrote files without verified" in s for s in diagnosis.get("self_validation") or []):
        proposals.append(
            "Keep VERIFY_GATE target-aware: no success claim after a mutation until relevant readback, tests, or health evidence succeeds."
        )
    return proposals


def validate_event_log_hash_chain(path: Path, *, limit: int | None = None) -> dict[str, Any]:
    from ..verify_event_chain import _verify_single_file

    verified = _verify_single_file(path, limit=limit)
    records_checked = int(verified.get("records", 0))
    valid_records = int(verified.get("valid", 0))
    corrupt_records = int(verified.get("corrupt", 0))
    valid_ratio = round(valid_records / max(1, records_checked), 6)
    out: dict[str, Any] = {
        "ok": verified.get("ok") is True,
        "records_checked": records_checked,
        "valid_records": valid_records,
        "corrupt_records": corrupt_records,
        "valid_ratio": valid_ratio,
        "first_prev_hash": verified.get("first_prev_hash", ""),
        "tail_hash": verified.get("tail_hash", "0" * 64),
    }
    for key in ("error", "line", "expected", "got", "detail"):
        if key in verified:
            out[key] = verified[key]
    if out.get("error") == "file_missing":
        out["error"] = "event_log_missing"
    return out


def harness_evidence_score(
    trajectories: list[HarnessTrajectory], diagnosis: dict[str, Any], integrity: dict[str, Any]
) -> dict[str, Any]:
    """Score internal trajectory evidence without implying an external rank."""
    sample_volume = min(1.0, len(trajectories) / 100.0)
    integrity_score = float(
        integrity.get("valid_ratio", 1.0 if integrity.get("ok") is True else 0.0)
    )
    verification = (
        float(diagnosis.get("verify_compliance_rate") or 0.0)
        if any(t.writes for t in trajectories)
        else 0.8
    )
    recovery = max(0.0, 1.0 - float(diagnosis.get("failure_rate") or 0.0))
    diversity = (
        min(
            1.0,
            len({t.task_type for t in trajectories}) / 4.0
            + len({t.fingerprint for t in trajectories}) / 40.0,
        )
        if trajectories
        else 0.0
    )
    outcome = float(diagnosis.get("mean_outcome_score") or 0.0)
    weights = {
        "sample_volume": 0.18,
        "integrity": 0.17,
        "verification": 0.20,
        "recovery": 0.17,
        "diversity": 0.13,
        "outcome": 0.15,
    }
    dims = {
        "sample_volume": round(sample_volume, 3),
        "integrity": round(integrity_score, 3),
        "verification": round(verification, 3),
        "recovery": round(recovery, 3),
        "diversity": round(diversity, 3),
        "outcome": round(outcome, 3),
    }
    score = round(sum(dims[k] * v for k, v in weights.items()), 3)
    if integrity.get("ok") is not True:
        assessment = "integrity_failed"
    elif len(trajectories) < 50:
        assessment = "insufficient_live_volume"
    elif score >= 0.85:
        assessment = "internal_threshold_met"
    else:
        assessment = "needs_improvement"
    return {
        "score": score,
        "dimensions": dims,
        "target": {
            "internal_score": 0.85,
            "minimum_live_trajectories": 50,
        },
        "assessment": assessment,
        "scope": "internal_trajectory_evidence_only",
        "external_percentile_claim": False,
    }


def write_report(event_log: Path, out_dir: Path, *, k: int = 10, limit: int = 1000) -> Path:
    trajectories = load_trajectories(event_log, limit=limit)
    coreset = select_coreset(trajectories, k=k)
    diagnosis = diagnose(coreset)
    integrity = validate_event_log_hash_chain(event_log)
    report = {
        "source_event_log": str(event_log),
        "event_log_integrity": integrity,
        "trajectory_count": len(trajectories),
        "harness_evidence": harness_evidence_score(trajectories, diagnosis, integrity),
        "diagnosis": diagnosis,
        "harness_update_proposals": propose_harness_updates(diagnosis),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"rho_report_{stamp}.json"
    body = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    atomic_write_text(path, body)
    latest = out_dir / "rho_report_latest.json"
    atomic_write_text(latest, body)
    return path


def analyze_harness(
    event_log: Path,
    out_dir: Path,
    *,
    k: int = 10,
    limit: int = 1000,
) -> dict[str, Any]:
    """Mine trajectories and write a reviewable, non-mutating report.

    Source changes are intentionally outside this maintenance helper. A
    heuristic coreset and a generated string replacement cannot establish
    semantic correctness, test coverage, or safe deployment.
    """
    trajectories = load_trajectories(event_log, limit=limit)
    if not trajectories:
        report_path = write_report(event_log, out_dir, k=k, limit=limit)
        return {
            "status": "no_trajectories",
            "source_mutations": 0,
            "report_path": str(report_path),
        }

    coreset = select_coreset(trajectories, k=k)
    diagnosis = diagnose(coreset)
    integrity = validate_event_log_hash_chain(event_log)
    proposals = propose_harness_updates(diagnosis)
    report_path = write_report(event_log, out_dir, k=k, limit=limit)
    return {
        "status": "analyzed" if integrity.get("ok") is True else "integrity_failed",
        "source_mutations": 0,
        "report_path": str(report_path),
        "proposal_count": len(proposals),
        "integrity": integrity,
        "diagnosis_summary": {
            "coreset_size": diagnosis.get("coreset_size", 0),
            "failure_rate": diagnosis.get("failure_rate", 0),
            "verify_rate": diagnosis.get("verify_compliance_rate", 0),
            "top_errors": diagnosis.get("top_errors", [])[:5],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Norax RHO harness report from event trajectories."
    )
    state_dir = Path(os.environ.get("NORAX_STATE_DIR", "state")).expanduser()
    parser.add_argument("--event-log", type=Path, default=state_dir / "events.jsonl")
    parser.add_argument("--out-dir", type=Path, default=state_dir / "harness_optimizer")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args(argv)
    path = write_report(args.event_log, args.out_dir, k=args.k, limit=args.limit)
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
