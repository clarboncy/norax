#!/usr/bin/env python3
"""Collect durable 72-hour Norax release-burn-in evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from norax.atomic import append_bounded_text, atomic_write_text, read_bounded_text
from norax.verify_event_chain import _verify_single_file

TARGET_SECONDS = 72 * 60 * 60
MIN_SAMPLES = 200
MAX_HISTORY = 5000
MAX_EVIDENCE_BYTES = 32 * 1024 * 1024
_CAMPAIGN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return round(ordered[index], 1)


def _get_json(url: str, timeout: float = 10.0) -> tuple[int, dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except Exception:  # noqa: BLE001
            return exc.code, {"ok": False, "error": str(exc)}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return 0, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _service_state(unit: str) -> dict[str, Any]:
    properties = ("ActiveState", "SubState", "NRestarts", "Result", "InvocationID", "MainPID")
    command = [
        "systemctl",
        "--user",
        "show",
        unit,
        f"--property={','.join(properties)}",
    ]
    fields: dict[str, Any] = {"unit": unit, "query_ok": False}
    query_error = ""
    for _attempt in range(2):
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            query_error = f"{type(exc).__name__}: {exc}"[:300]
            continue
        parsed: dict[str, Any] = {"unit": unit}
        for line in proc.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                parsed[key] = (
                    int(value) if key in {"NRestarts", "MainPID"} and value.isdigit() else value
                )
        if proc.returncode == 0 and all(key in parsed for key in properties):
            fields = parsed
            fields["query_ok"] = True
            return fields
        query_error = (proc.stderr or f"systemctl exit {proc.returncode}").strip()[:300]
    fields["query_error"] = query_error
    return fields


def evaluate_status(status: dict[str, Any], now_epoch: float | None = None) -> dict[str, Any]:
    now_epoch = now_epoch if now_epoch is not None else time.time()
    started = float(status.get("started_epoch", now_epoch))
    elapsed = max(0.0, now_epoch - started)
    samples = status.get("samples") or []
    latencies = [
        float(sample["completion_latency_ms"])
        for sample in samples
        if isinstance(sample.get("completion_latency_ms"), (int, float))
    ]
    ready_samples = sum(sample.get("ready") is True for sample in samples)
    discord_samples = sum(sample.get("discord_connected") is True for sample in samples)
    total = len(samples)
    ready_rate = ready_samples / total if total else 0.0
    discord_rate = discord_samples / total if total else 0.0
    gates = {
        "duration_72h": elapsed >= TARGET_SECONDS,
        "minimum_samples": total >= MIN_SAMPLES,
        "completion_verified": total > 0
        and all(sample.get("completion_ok") is True for sample in samples),
        "ready_rate_99_5": ready_rate >= 0.995,
        "discord_rate_99_5": discord_rate >= 0.995,
        "completion_p95_under_5s": bool(latencies)
        and (_percentile(latencies, 0.95) or 999999) < 5000,
        "event_chain_clean": total > 0 and all(sample.get("event_chain_ok") for sample in samples),
        "no_restart_growth": total > 0
        and all(
            sample.get("restart_detected", sample.get("restart_delta", 0) != 0) is False
            for sample in samples
        ),
        "service_continuously_active": total > 0
        and all(
            sample.get("service_query_ok") is True
            and sample.get("service_active") is True
            and sample.get("service_invocation_unchanged") is True
            for sample in samples
        ),
        "no_brain_errors": total > 0
        and all(
            sample.get("brain_error_delta", sample.get("brain_errors", 0)) == 0
            for sample in samples
        ),
        "no_agent_turn_failures": total > 0
        and all(
            sample.get("agent_turn_failure_delta", sample.get("agent_turn_failures", 0)) == 0
            for sample in samples
        ),
    }
    return {
        "state": "passed"
        if all(gates.values())
        else "collecting"
        if elapsed < TARGET_SECONDS
        else "failed",
        "elapsed_seconds": round(elapsed, 1),
        "remaining_seconds": round(max(0.0, TARGET_SECONDS - elapsed), 1),
        "sample_count": total,
        "ready_rate": round(ready_rate, 5),
        "discord_rate": round(discord_rate, 5),
        "completion_latency_p50_ms": _percentile(latencies, 0.50),
        "completion_latency_p95_ms": _percentile(latencies, 0.95),
        "gates": gates,
    }


def _metric_value(text: str, name: str) -> float:
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            try:
                total += float(line.rsplit(" ", 1)[-1])
            except ValueError:
                continue
    return total


def _get_metrics(base_url: str) -> str:
    try:
        with urllib.request.urlopen(  # noqa: S310
            base_url.rstrip("/") + "/metrics", timeout=10
        ) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def _archive_current_campaign(burnin_dir: Path, next_campaign: str) -> Path | None:
    """Move the current evidence set aside before beginning a new campaign."""
    if _CAMPAIGN_RE.fullmatch(next_campaign) is None:
        raise ValueError(
            "campaign name must be 1-64 alphanumeric, dot, underscore, or hyphen characters"
        )
    names = ("status.json", "history.json", "evidence.jsonl")
    existing = [burnin_dir / name for name in names if (burnin_dir / name).exists()]
    if not existing:
        return None
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    archive_root = burnin_dir / "archive"
    archive_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    archive_dir = archive_root / f"{timestamp}-before-{next_campaign}"
    archive_dir.mkdir(mode=0o700)
    if os.name == "posix":
        archive_root.chmod(0o700)
        archive_dir.chmod(0o700)
    for source in existing:
        destination = archive_dir / source.name
        source.replace(destination)
        if os.name == "posix":
            destination.chmod(0o600)
    atomic_write_text(
        archive_dir / "archive.json",
        json.dumps(
            {
                "archived_at": _now(),
                "next_campaign": next_campaign,
                "files": [path.name for path in existing],
            },
            indent=2,
        )
        + "\n",
        mode=0o600,
        durable=True,
    )
    return archive_dir


def collect_sample(
    base_url: str,
    event_log: Path,
    baseline_restarts: int,
    baseline_brain_errors: float = 0.0,
    baseline_agent_turn_failures: float = 0.0,
    baseline_service_invocation_id: str = "",
) -> dict[str, Any]:
    status_code, ready = _get_json(base_url.rstrip("/") + "/readyz")
    metrics = _get_metrics(base_url)
    service = _service_state("norax-ai.service")
    restarts = int(service.get("NRestarts", 0) or 0)
    invocation_id = str(service.get("InvocationID", "") or "")
    invocation_unchanged = bool(
        baseline_service_invocation_id
        and invocation_id
        and invocation_id == baseline_service_invocation_id
    )
    restart_delta = max(0, restarts - baseline_restarts)
    restart_detected = restart_delta > 0 or not invocation_unchanged
    components = ready.get("components") or {}
    discord = components.get("discord") or {}
    probe = components.get("completion_probe") or {}
    chain = _verify_single_file(event_log)
    brain_errors = _metric_value(metrics, "norax_brain_errors_total")
    agent_turn_failures = _metric_value(metrics, "norax_agent_turn_failed_total")
    return {
        "timestamp": _now(),
        "ready": status_code == 200 and ready.get("ok") is True,
        "ready_http": status_code,
        "discord_connected": bool(discord.get("connected")),
        "completion_ok": probe.get("ok") is True and probe.get("completion_verified") is True,
        "completion_latency_ms": probe.get("latency_ms")
        if probe.get("completion_verified") is True
        else None,
        "completion_age_seconds": probe.get("age_seconds"),
        "event_chain_ok": chain.get("ok") is True,
        "event_records": chain.get("records", 0),
        "brain_errors": brain_errors,
        "brain_error_delta": max(0.0, brain_errors - baseline_brain_errors),
        "agent_turn_failures": agent_turn_failures,
        "agent_turn_failure_delta": max(0.0, agent_turn_failures - baseline_agent_turn_failures),
        "service_active": service.get("ActiveState") == "active"
        and service.get("SubState") == "running",
        "service_query_ok": service.get("query_ok") is True,
        "service_invocation_id": invocation_id,
        "service_invocation_unchanged": invocation_unchanged,
        "service_main_pid": int(service.get("MainPID", 0) or 0),
        "restart_count": restarts,
        "restart_delta": max(restart_delta, int(not invocation_unchanged)),
        "restart_detected": restart_detected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:4101")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.environ.get("NORAX_STATE_DIR", Path.home() / ".local/state/norax")),
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--start-new-campaign",
        metavar="NAME",
        help="archive the current evidence and start a named campaign",
    )
    args = parser.parse_args()
    burnin_dir = args.state_dir.expanduser().resolve() / "burnin"
    burnin_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        burnin_dir.chmod(0o700)
    status_path = burnin_dir / "status.json"
    history_path = burnin_dir / "history.json"
    evidence_path = burnin_dir / "evidence.jsonl"
    if args.start_new_campaign:
        _archive_current_campaign(burnin_dir, args.start_new_campaign)
    source_path = history_path if history_path.exists() else status_path
    if source_path.exists():
        status = json.loads(read_bounded_text(source_path, max_bytes=16 * 1024 * 1024))
    else:
        metrics = _get_metrics(args.base_url)
        service = _service_state("norax-ai.service")
        status = {
            "version": 3,
            "campaign_id": args.start_new_campaign or "default",
            "started_at": _now(),
            "started_epoch": time.time(),
            "baseline_restarts": int(service.get("NRestarts", 0) or 0),
            "baseline_service_invocation_id": str(service.get("InvocationID", "") or ""),
            "baseline_brain_errors": _metric_value(metrics, "norax_brain_errors_total"),
            "baseline_agent_turn_failures": _metric_value(metrics, "norax_agent_turn_failed_total"),
            "samples": [],
        }
    sample = collect_sample(
        args.base_url,
        args.state_dir / "events.jsonl",
        int(status.get("baseline_restarts", 0)),
        float(status.get("baseline_brain_errors", 0.0)),
        float(status.get("baseline_agent_turn_failures", 0.0)),
        str(status.get("baseline_service_invocation_id", "") or ""),
    )
    status.setdefault("samples", []).append(sample)
    status["samples"] = status["samples"][-MAX_HISTORY:]
    status["updated_at"] = _now()
    status["assessment"] = evaluate_status(status)
    # Keep history available for evaluation without requiring dashboard readers
    # to load thousands of samples. Publish history before its summary.
    atomic_write_text(history_path, json.dumps(status, separators=(",", ":")) + "\n", mode=0o600)
    summary = {key: value for key, value in status.items() if key != "samples"}
    atomic_write_text(status_path, json.dumps(summary, indent=2) + "\n", mode=0o600)
    append_bounded_text(
        evidence_path,
        json.dumps(sample, separators=(",", ":")) + "\n",
        max_bytes=MAX_EVIDENCE_BYTES,
        mode=0o600,
    )
    print(json.dumps({"sample": sample, "assessment": status["assessment"]}, indent=2))
    if args.strict and status["assessment"]["state"] == "failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
