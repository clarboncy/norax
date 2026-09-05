#!/usr/bin/env python3
"""Norax live runtime benchmark using measured HTTP turns and counters.

Drives a fixed task suite through the live runtime HTTP ingress
(POST /ingress/agent_os), waits for the brain turn to complete, scrapes
Prometheus /metrics before and after the run, and persists a JSON run
record with per-run deltas:

  - tokens_in / tokens_out (gateway-level, authoritative)
  - brain turn latency (histogram delta)
  - answer quality score (deterministic rubric + optional judge model)

A `diff` command compares matched runs and reports observed deltas. It does
not claim causality or statistical significance from global counters.

Usage:
  python benchmarks/perf_bench.py run --label pre-change
  python benchmarks/perf_bench.py run --label post-change
  python benchmarks/perf_bench.py diff pre-change post-change
  python benchmarks/perf_bench.py list

Notes:
  - Requires the runtime HTTP server (default 127.0.0.1:4101) and a chat
    token (NORAX_AGENT_OS_CHAT_TOKEN or NORAX_RUNTIME_CHAT_TOKEN in .env).
  - Tasks are answered WITHOUT tools where possible so token/latency deltas
    reflect prompt+model behavior, not tool-loop variance. Tool tasks are
    included separately and flagged as noisy.
  - This measures REAL turns on the live system. Background traffic on the
    same runtime will pollute deltas; run during quiet periods. The report
    includes the raw counter deltas so pollution is visible.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from norax.atomic import atomic_write_text  # noqa: E402

RUNS_DIR = _REPO_ROOT / "benchmarks" / "runs"
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")

# ---------------------------------------------------------------------------
# Task suite — fixed, representative, deterministic-rubric-scored where
# possible. Each task is a single owner message; reply captured from history.
# ---------------------------------------------------------------------------
TASKS: list[dict] = [
    {
        "name": "direct_fact",
        "kind": "answer",
        "body": "What is the time complexity of binary search? One sentence.",
        "must_contain": ["log"],
        "max_expected_words": 60,
    },
    {
        "name": "concise_definition",
        "kind": "answer",
        "body": "In one sentence, what is a hash collision?",
        "must_contain": ["key"],
        "max_expected_words": 60,
    },
    {
        "name": "exact_output",
        "kind": "answer",
        "body": "Reply with exactly the word: PONG",
        "expect_exact": "PONG",
    },
    {
        "name": "small_code",
        "kind": "answer",
        "body": (
            "Write a Python function `def is_palindrome(s: str) -> bool` that "
            "ignores case and non-alphanumerics. Code only, no explanation."
        ),
        "must_contain": ["def is_palindrome", "return"],
        "rubric": "code",
    },
    {
        "name": "reasoning_arithmetic",
        "kind": "answer",
        "body": (
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
            "than the ball. How much does the ball cost? Answer with just the "
            "number of cents."
        ),
        "expect_exact": "5",
    },
    {
        "name": "summarize_constraints",
        "kind": "answer",
        "body": (
            "Summarize in exactly 3 bullets: Norax is an event-sourced agent "
            "runtime with hash-chained logs, multi-signal memory retrieval, "
            "and multi-provider model routing with adaptive fallback."
        ),
        "rubric": "three_bullets",
        "must_contain": ["memory", "routing"],
    },
    {
        "name": "refusal_free_tool_plan",
        "kind": "answer",
        "body": (
            "List the exact shell commands (max 3) to show disk usage of "
            "/var/log sorted descending. Commands only."
        ),
        "must_contain": ["du", "sort"],
        "max_expected_words": 80,
    },
    {
        "name": "long_context_recall",
        "kind": "answer",
        "body": "Read the numbered facts, then answer.\n"
        + "\n".join(f"FACT {i:03d}: item_{i} belongs to group_{i % 7}." for i in range(1, 121))
        + "\nQuestion: which group does item_099 belong to? Answer with just the group name.",
        "expect_exact": "group_1",  # 99 % 7 == 1
    },
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _load_env() -> dict[str, str]:
    env = {}
    p = _REPO_ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    env.update(os.environ)
    return env


def _http(
    method: str,
    url: str,
    *,
    token: str | None = None,
    body: dict | None = None,
    timeout: float = 30.0,
):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("content-type", "application/json")
    if token:
        req.add_header("authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def scrape_metrics(base: str) -> dict[str, float]:
    """Scrape Prometheus text format; return flat {metric_name: value} for
    the counters/histograms we care about (summed across labels)."""
    with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=10) as r:
        text = r.read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if not m:
            continue
        name, val = m.group(1), float(m.group(2))
        # Sum across label variants for totals; keep histogram sum/count.
        out[name] = out.get(name, 0.0) + val
    return out


METRIC_KEYS = {
    "tokens_in": "norax_gateway_tokens_in_total",
    "tokens_out": "norax_gateway_tokens_out_total",
    "gateway_requests": "norax_gateway_requests_total",
    "brain_turns": "norax_brain_turns_total",
    "brain_seconds_sum": "norax_brain_turn_seconds_sum",
    "brain_seconds_count": "norax_brain_turn_seconds_count",
}


def snapshot(base: str) -> dict[str, float]:
    raw = scrape_metrics(base)
    snap = {}
    for key, metric in METRIC_KEYS.items():
        if key == "brain_turns":
            snap[key] = raw.get(metric, 0.0)
        else:
            snap[key] = raw.get(metric, 0.0)
    return snap


def delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in METRIC_KEYS}


# ---------------------------------------------------------------------------
# Scoring — deterministic rubric per task
# ---------------------------------------------------------------------------


def score_task(task: dict, reply: str) -> tuple[float, list[str]]:
    """Return diagnostic check coverage and every failed hard constraint."""
    problems: list[str] = []
    if not reply.strip():
        return 0.0, ["empty_reply"]
    low = reply.lower()

    checks = 0

    def require(condition: bool, problem: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            problems.append(problem)

    if "expect_exact" in task:
        require(reply.strip() == task["expect_exact"], "exact_mismatch")
    for s in task.get("must_contain", []):
        require(s.lower() in low, f"missing:{s}")
    for s in task.get("must_not_contain_anywhere", []):
        require(s.lower() not in low, f"forbidden:{s}")
    if task.get("max_expected_words"):
        wc = len(reply.split())
        require(wc <= task["max_expected_words"], f"verbose:{wc}w")
    rubric = task.get("rubric")
    if rubric == "code":
        require("```" in reply or "def " in reply, "no_code")
    elif rubric == "three_bullets":
        bullets = [ln for ln in reply.splitlines() if ln.strip().startswith(("-", "*", "•"))]
        require(len(bullets) == 3, f"bullets={len(bullets)}")

    passed = checks - len(problems)
    return (passed / checks if checks else 1.0), problems


# ---------------------------------------------------------------------------
# Turn driver
# ---------------------------------------------------------------------------


def receive_reply(socket, channel_id: str, timeout_s: float) -> str:
    """Accept only a final reply delivered to this benchmark's unique thread."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            raw = socket.recv(timeout=max(0.001, deadline - time.monotonic()))
        except TimeoutError:
            return ""
        event = json.loads(raw)
        if (
            isinstance(event, dict)
            and event.get("type") == "reply"
            and event.get("thread_id") == channel_id
            and isinstance(event.get("text"), str)
        ):
            return event["text"]
    return ""


def run_task_once(base: str, token: str, task: dict, channel_id: str) -> dict:
    """Connect a real outbound recipient before submitting the request."""
    import secrets

    from websockets.sync.client import connect

    channel_id = f"{channel_id}-{secrets.token_hex(8)}"
    socket_base = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    t0 = time.monotonic()
    with connect(
        socket_base.rstrip("/") + "/ws/agent_os",
        additional_headers={"authorization": f"Bearer {token}"},
        open_timeout=15,
        close_timeout=5,
        max_size=2 * 1024 * 1024,
    ) as socket:
        # The server attaches the recipient before sending this snapshot.
        snapshot = json.loads(socket.recv(timeout=15))
        if not isinstance(snapshot, dict) or snapshot.get("type") != "status":
            raise ValueError("dashboard did not acknowledge attachment")
        response = _http(
            "POST",
            base.rstrip("/") + "/ingress/agent_os",
            token=token,
            body={"body": task["body"], "thread_id": channel_id},
            timeout=20,
        )
        if response.get("ok") is not True:
            return {"task": task["name"], "ok": False, "error": "ingress_rejected", "score": 0.0}
        reply = receive_reply(socket, channel_id, float(task.get("timeout_s", 90)))
    score, problems = score_task(task, reply)
    return {
        "task": task["name"],
        "ok": not problems,
        "score": round(score, 3),
        "problems": problems,
        "reply_chars": len(reply),
        "reply_preview": reply[:200],
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _safe_label(label: str) -> str:
    if label in {".", ".."} or not _LABEL_RE.fullmatch(label):
        raise ValueError(
            "labels must be 1-80 characters using only letters, digits, '.', '_', or '-'"
        )
    return label


def _run_path(label: str) -> Path:
    return RUNS_DIR / f"{_safe_label(label)}.json"


def assess_measurement(
    results: list[dict], metric_delta: dict[str, float], *, expected_turns: int
) -> dict[str, object]:
    """Identify counter pollution or missing turn evidence without hiding quality failures."""
    reasons: list[str] = []
    observed_turns = metric_delta.get("brain_turns", 0.0)
    observed_latency_count = metric_delta.get("brain_seconds_count", 0.0)
    if abs(observed_turns - expected_turns) > 0.01:
        reasons.append(f"brain_turns={observed_turns:g}, expected={expected_turns}")
    if abs(observed_latency_count - expected_turns) > 0.01:
        reasons.append(
            f"brain_latency_samples={observed_latency_count:g}, expected={expected_turns}"
        )
    observed_gateway_requests = metric_delta.get("gateway_requests", 0.0)
    if abs(observed_gateway_requests - expected_turns) > 0.01:
        reasons.append(f"gateway_requests={observed_gateway_requests:g}, expected={expected_turns}")
    incomplete = sum("score" not in result for result in results)
    if incomplete:
        reasons.append(f"incomplete_task_results={incomplete}")
    negative = sorted(key for key, value in metric_delta.items() if value < -0.01)
    if negative:
        reasons.append(f"counters_decreased={','.join(negative)}")
    return {
        "valid": not reasons,
        "expected_brain_turns": expected_turns,
        "observed_brain_turns": observed_turns,
        "observed_gateway_requests": observed_gateway_requests,
        "reasons": reasons,
    }


def cmd_run(args) -> int:
    env = _load_env()
    token = env.get("NORAX_AGENT_OS_CHAT_TOKEN") or env.get("NORAX_RUNTIME_CHAT_TOKEN")
    if not token:
        print(
            "ERROR: no chat token in .env (NORAX_AGENT_OS_CHAT_TOKEN / NORAX_RUNTIME_CHAT_TOKEN)",
            file=sys.stderr,
        )
        return 2
    base = args.base or f"http://127.0.0.1:{env.get('NORAX_HTTP_PORT', '4101')}"

    # Health check
    try:
        scrape_metrics(base)
        readiness = _http("GET", base.rstrip("/") + "/readyz", timeout=15)
        if readiness.get("ok") is not True:
            raise RuntimeError(f"runtime is not ready: {readiness}")
    except Exception as e:
        print(f"ERROR: runtime at {base} is not ready — {e!r}", file=sys.stderr)
        return 2

    readiness_components = readiness.get("components") or {}
    completion_probe = readiness_components.get("completion_probe") or {}
    gateway_status = readiness_components.get("gateway") or {}
    live_model = completion_probe.get("model") or env.get("NORAX_DEFAULT_MODEL", "unknown")
    live_provider = completion_probe.get("provider") or gateway_status.get("provider") or "unknown"

    tasks = TASKS
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        tasks = [t for t in TASKS if t["name"] in wanted]
        if not tasks:
            print(
                f"ERROR: no matching tasks; available: {[t['name'] for t in TASKS]}",
                file=sys.stderr,
            )
            return 2

    try:
        label = _safe_label(args.label or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.repetitions < 1:
        print("ERROR: --repetitions must be positive", file=sys.stderr)
        return 2
    channel_id = f"perf-bench-{label}"

    sample_count = len(tasks) * args.repetitions
    print(
        f"[perf_bench] label={label} base={base} tasks={len(tasks)} repetitions={args.repetitions}",
        flush=True,
    )
    before = snapshot(base)
    started = datetime.now(UTC)

    # Re-snapshot the window before EVERY task inside run_task_once (it calls
    # _fetch_history at send time). After each task completes, sleep so the
    # window settles and the next task's base_idx excludes this task's frames.
    results = []
    sample_index = 0
    for repetition in range(1, args.repetitions + 1):
        for task in tasks:
            sample_index += 1
            print(
                f"  [{sample_index}/{sample_count}] {task['name']} rep={repetition} ...",
                flush=True,
            )
            try:
                result = run_task_once(base, token, task, channel_id)
            except Exception as exc:  # noqa: BLE001
                result = {
                    "task": task["name"],
                    "ok": False,
                    "score": 0.0,
                    "problems": ["turn_driver_failed"],
                    "error_type": type(exc).__name__,
                    "wall_seconds": None,
                }
            result["repetition"] = repetition
            results.append(result)
            status = "PASS" if result.get("ok") else "FAIL"
            print(
                f"      {status} score={result.get('score')} "
                f"wall={result.get('wall_seconds')}s {result.get('problems') or ''}",
                flush=True,
            )
            # Let the shared history and counters settle before the next sample.
            time.sleep(2.0)

    after = snapshot(base)
    d = delta(before, after)
    ended = datetime.now(UTC)
    measurement = assess_measurement(results, d, expected_turns=len(results))

    scores = [r["score"] for r in results if "score" in r]
    passed = sum(1 for result in results if result.get("ok"))
    gate_passed = bool(measurement["valid"]) and passed == len(results)
    record = {
        "schema": "norax.perf_bench.run.v2",
        "label": label,
        "started": started.isoformat(),
        "ended": ended.isoformat(),
        "base": base,
        # Capture the model proven by the live runtime, not merely the .env
        # default. Persisted selector settings can override the environment.
        "model": live_model,
        "provider": live_provider,
        "git_head": _git_head(),
        "task_results": results,
        "metric_delta": d,
        "measurement": measurement,
        "summary": {
            "distinct_tasks": len(tasks),
            "repetitions": args.repetitions,
            "samples": len(results),
            "tasks": len(results),
            "passed": passed,
            "gate_passed": gate_passed,
            "mean_score": round(statistics.mean(scores), 3) if scores else 0.0,
            "tokens_in_delta": d.get("tokens_in", 0.0),
            "tokens_out_delta": d.get("tokens_out", 0.0),
            "gateway_requests_delta": d.get("gateway_requests", 0.0),
            "brain_turns_delta": d.get("brain_turns", 0.0),
            "mean_brain_seconds": (
                round(d["brain_seconds_sum"] / d["brain_seconds_count"], 3)
                if d.get("brain_seconds_count")
                else None
            ),
        },
    }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = _run_path(label)
    atomic_write_text(out, json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"\n[perf_bench] wrote {out}")
    print(json.dumps(record["summary"], indent=2))
    return 0 if gate_passed else 1


def _git_head() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def cmd_list(_args) -> int:
    if not RUNS_DIR.exists():
        print("(no runs)")
        return 0
    for p in sorted(RUNS_DIR.glob("*.json")):
        try:
            r = json.loads(p.read_text())
            s = r.get("summary", {})
            score = float(s.get("mean_score") or 0.0)
            passed = int(s.get("passed") or 0)
            samples = int(s.get("samples") or s.get("tasks") or 0)
            tokens_in = float(s.get("tokens_in_delta") or 0.0)
            tokens_out = float(s.get("tokens_out_delta") or 0.0)
            print(
                f"{r.get('label', p.stem):24} score={score:.3f} "
                f"pass={passed}/{samples} "
                f"tok_in={tokens_in:.0f} tok_out={tokens_out:.0f} "
                f"brain_s={s.get('mean_brain_seconds')} model={r.get('model')}"
            )
        except Exception:
            print(p.name)
    return 0


def _pct_change(new: float | None, old: float | None) -> float | None:
    if old is None or old == 0 or new is None:
        return None
    return round((new - old) / old * 100, 1)


def _sample_map(run: dict) -> tuple[dict[tuple[str, int], dict], bool]:
    samples: dict[tuple[str, int], dict] = {}
    duplicate = False
    for result in run.get("task_results") or []:
        key = (str(result.get("task") or ""), int(result.get("repetition") or 1))
        if key in samples:
            duplicate = True
        samples[key] = result
    return samples, duplicate


def build_diff_report(baseline: dict, candidate: dict) -> dict:
    """Build an honest matched-sample comparison report."""
    sa = baseline.get("summary") or {}
    sb = candidate.get("summary") or {}
    invalid_reasons: list[str] = []
    if baseline.get("schema") != "norax.perf_bench.run.v2":
        invalid_reasons.append("baseline is not a v2 run with validity evidence")
    if candidate.get("schema") != "norax.perf_bench.run.v2":
        invalid_reasons.append("candidate is not a v2 run with validity evidence")
    if not (baseline.get("measurement") or {}).get("valid"):
        invalid_reasons.append("baseline measurement is contaminated or incomplete")
    if not (candidate.get("measurement") or {}).get("valid"):
        invalid_reasons.append("candidate measurement is contaminated or incomplete")

    baseline_samples, baseline_duplicates = _sample_map(baseline)
    candidate_samples, candidate_duplicates = _sample_map(candidate)
    if baseline_duplicates or candidate_duplicates:
        invalid_reasons.append("duplicate task/repetition sample keys")
    if set(baseline_samples) != set(candidate_samples):
        invalid_reasons.append("task/repetition samples do not match")
    sample_count = len(baseline_samples)
    if not sample_count:
        invalid_reasons.append("no matched samples")

    a_mean = float(sa.get("mean_score") or 0.0)
    b_mean = float(sb.get("mean_score") or 0.0)
    a_count = int(sa.get("samples") or sa.get("tasks") or 0)
    b_count = int(sb.get("samples") or sb.get("tasks") or 0)
    if a_count != sample_count or b_count != sample_count:
        invalid_reasons.append("summary sample count does not match task results")

    def per_sample(summary: dict, key: str, count: int) -> float | None:
        return float(summary.get(key) or 0.0) / count if count else None

    a_in = per_sample(sa, "tokens_in_delta", a_count)
    b_in = per_sample(sb, "tokens_in_delta", b_count)
    a_out = per_sample(sa, "tokens_out_delta", a_count)
    b_out = per_sample(sb, "tokens_out_delta", b_count)
    a_latency = (
        float(sa["mean_brain_seconds"]) if sa.get("mean_brain_seconds") is not None else None
    )
    b_latency = (
        float(sb["mean_brain_seconds"]) if sb.get("mean_brain_seconds") is not None else None
    )
    quality_delta = round(b_mean - a_mean, 3)
    token_change = _pct_change(b_in, a_in)
    latency_change = _pct_change(b_latency, a_latency)
    repetitions = min(int(sa.get("repetitions") or 1), int(sb.get("repetitions") or 1))
    distinct_tasks = min(int(sa.get("distinct_tasks") or 0), int(sb.get("distinct_tasks") or 0))
    repeated = repetitions >= 3 and distinct_tasks >= 5 and sample_count >= 15
    confounders: list[str] = []
    if baseline.get("model") != candidate.get("model"):
        confounders.append("model changed")
    if baseline.get("provider") != candidate.get("provider"):
        confounders.append("provider changed")

    task_level = []
    for key in sorted(set(baseline_samples) & set(candidate_samples)):
        before = baseline_samples[key]
        after = candidate_samples[key]
        task_level.append(
            {
                "task": key[0],
                "repetition": key[1],
                "baseline_ok": before.get("ok"),
                "candidate_ok": after.get("ok"),
                "baseline_score": before.get("score"),
                "candidate_score": after.get("score"),
            }
        )

    report = {
        "comparison_valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "evidence_level": "repeated_directional" if repeated else "single_run_directional",
        "limitations": [
            "global runtime counters cannot attribute variance to individual tasks",
            "observed deltas do not establish causality or statistical significance",
        ],
        "confounders": confounders,
        "quality": {
            "baseline_passed": int(sa.get("passed") or 0),
            "candidate_passed": int(sb.get("passed") or 0),
            "samples": sample_count,
            "baseline_mean_score": a_mean,
            "candidate_mean_score": b_mean,
            "delta_score": quality_delta,
        },
        "tokens_per_sample": {
            "baseline_in": round(a_in, 1) if a_in is not None else None,
            "candidate_in": round(b_in, 1) if b_in is not None else None,
            "in_pct_change": token_change,
            "baseline_out": round(a_out, 1) if a_out is not None else None,
            "candidate_out": round(b_out, 1) if b_out is not None else None,
            "out_pct_change": _pct_change(b_out, a_out),
        },
        "latency_seconds": {
            "baseline_mean_brain": a_latency,
            "candidate_mean_brain": b_latency,
            "pct_change": latency_change,
        },
        "task_level": task_level,
    }

    verdicts: list[str] = []
    if invalid_reasons:
        verdicts.append("INVALID COMPARISON: do not interpret performance deltas.")
        report["verdict"] = verdicts
        return report
    if not repeated:
        verdicts.append(
            "INSUFFICIENT REPLICATION: deltas are directional; use at least 3 repetitions "
            "across 5 tasks for repeated evidence."
        )
    if quality_delta > 0.02:
        verdicts.append(f"OBSERVED QUALITY: higher by {quality_delta:+.3f} mean score.")
    elif quality_delta < -0.02:
        verdicts.append(f"OBSERVED QUALITY: lower by {quality_delta:+.3f} mean score.")
    else:
        verdicts.append("OBSERVED QUALITY: within ±0.02 mean score.")
    if token_change is not None:
        verdicts.append(f"OBSERVED TOKENS-IN: {token_change:+.1f}% per matched sample.")
    if latency_change is not None:
        verdicts.append(f"OBSERVED LATENCY: {latency_change:+.1f}% mean brain time.")
    report["verdict"] = verdicts
    return report


def cmd_diff(args) -> int:
    try:
        a_path = _run_path(args.baseline)
        b_path = _run_path(args.candidate)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for path in (a_path, b_path):
        if not path.exists():
            print(f"ERROR: run not found: {path}", file=sys.stderr)
            return 2
    baseline = json.loads(a_path.read_text())
    candidate = json.loads(b_path.read_text())
    report = build_diff_report(baseline, candidate)
    report["baseline"] = args.baseline
    report["candidate"] = args.candidate
    print(json.dumps(report, indent=2))
    return 0 if report["comparison_valid"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Norax live performance benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run the suite against the live runtime")
    p_run.add_argument("--label", help="Run label (default: UTC timestamp)")
    p_run.add_argument(
        "--base", help="Runtime base URL (default: http://127.0.0.1:$NORAX_HTTP_PORT)"
    )
    p_run.add_argument("--tasks", help="Comma-separated task names; default all")
    p_run.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="samples per task; use at least 3 for repeated directional evidence",
    )
    p_run.set_defaults(fn=cmd_run)

    p_list = sub.add_parser("list", help="List recorded runs")
    p_list.set_defaults(fn=cmd_list)

    p_diff = sub.add_parser("diff", help="Compare two runs")
    p_diff.add_argument("baseline")
    p_diff.add_argument("candidate")
    p_diff.set_defaults(fn=cmd_diff)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
