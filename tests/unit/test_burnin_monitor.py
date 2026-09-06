from __future__ import annotations

import importlib.util
import json
import stat
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest


def _module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "burnin_monitor.py"
    spec = importlib.util.spec_from_file_location("burnin_monitor", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _healthy_campaign(module):
    return {
        "started_epoch": 1000.0,
        "samples": [
            {
                "timestamp": datetime.fromtimestamp(1000 + elapsed, UTC).isoformat(),
                "ready": True,
                "discord_connected": True,
                "completion_latency_ms": 1500,
                "completion_ok": True,
                "event_chain_ok": True,
                "restart_detected": False,
                "service_query_ok": True,
                "service_active": True,
                "service_invocation_unchanged": True,
                "metrics_available": True,
                "brain_error_delta": 0,
                "agent_turn_failure_delta": 0,
            }
            for elapsed in range(0, module.TARGET_SECONDS + 1, 60)
        ],
    }


def test_burnin_stays_collecting_before_72_hours():
    module = _module()
    status = {
        "started_epoch": 1000.0,
        "samples": [
            {
                "ready": True,
                "discord_connected": True,
                "completion_latency_ms": 1200,
                "completion_ok": True,
                "event_chain_ok": True,
                "restart_delta": 0,
                "restart_detected": False,
                "service_query_ok": True,
                "service_active": True,
                "service_invocation_unchanged": True,
                "brain_errors": 0,
            }
            for _ in range(module.MIN_SAMPLES)
        ],
    }
    result = module.evaluate_status(status, now_epoch=1000.0 + module.TARGET_SECONDS - 1)
    assert result["state"] == "collecting"
    assert result["gates"]["duration_72h"] is False


def test_burnin_passes_only_when_every_slo_is_satisfied():
    module = _module()
    status = _healthy_campaign(module)
    result = module.evaluate_status(status, now_epoch=1000.0 + module.TARGET_SECONDS)
    assert result["state"] == "passed"
    assert all(result["gates"].values())


@pytest.mark.parametrize("failure", ["missing", "duplicate", "gap", "stale", "future"])
def test_burnin_requires_observations_across_the_actual_window(failure):
    module = _module()
    status = _healthy_campaign(module)
    samples = status["samples"]
    if failure == "missing":
        for sample in samples:
            sample.pop("timestamp")
    elif failure == "duplicate":
        for sample in samples:
            sample["timestamp"] = samples[0]["timestamp"]
    elif failure == "gap":
        del samples[100:200]
    elif failure == "stale":
        del samples[-20:]
    else:
        samples[-1]["timestamp"] = datetime.fromtimestamp(
            1000 + module.TARGET_SECONDS + 60, UTC
        ).isoformat()

    result = module.evaluate_status(status, now_epoch=1000 + module.TARGET_SECONDS)
    assert result["state"] == "failed"


@pytest.mark.parametrize(
    "metric", ["", "norax_agent_turn_failed_total NaN\n", "norax_agent_turn_failed_total -1\n"]
)
def test_missing_or_invalid_metrics_are_not_zero_errors(metric):
    module = _module()
    assert module._metric_value(metric, "norax_agent_turn_failed_total") is None


def test_metrics_outage_cannot_pass_burnin():
    module = _module()
    status = _healthy_campaign(module)
    status["samples"][-1]["metrics_available"] = False
    result = module.evaluate_status(status, now_epoch=1000 + module.TARGET_SECONDS)
    assert result["state"] == "failed"


@pytest.mark.parametrize("value", [None, True, -1, float("nan"), float("inf")])
def test_invalid_latency_cannot_be_ignored_in_passing_campaign(value):
    module = _module()
    status = _healthy_campaign(module)
    status["samples"][-1]["completion_latency_ms"] = value
    result = module.evaluate_status(status, now_epoch=1000 + module.TARGET_SECONDS)
    assert result["gates"]["completion_p95_under_5s"] is False


def test_empty_labelled_counter_requires_declared_counter_family():
    module = _module()
    metric = "norax_brain_errors_total"
    assert module._metric_value("", metric, allow_empty_family=True) is None
    assert module._metric_value(f"# TYPE {metric} counter\n", metric, allow_empty_family=True) == 0
    assert module._metric_value(f'{metric}{{where="brain stage"}} 2 123456\n', metric) == 2


@pytest.mark.parametrize("kind", ["brain_error_delta", "agent_turn_failure_delta"])
def test_counter_reset_is_not_silently_clamped_to_zero(kind):
    module = _module()
    status = _healthy_campaign(module)
    status["samples"][-1][kind] = -1
    result = module.evaluate_status(status, now_epoch=1000 + module.TARGET_SECONDS)
    assert result["state"] == "failed"


def test_burnin_fails_after_window_when_integrity_breaks():
    module = _module()
    samples = [
        {
            "ready": True,
            "discord_connected": True,
            "completion_latency_ms": 1500,
            "completion_ok": True,
            "event_chain_ok": True,
            "restart_delta": 0,
            "restart_detected": False,
            "service_query_ok": True,
            "service_active": True,
            "service_invocation_unchanged": True,
            "brain_errors": 0,
        }
        for _ in range(module.MIN_SAMPLES)
    ]
    samples[-1]["event_chain_ok"] = False
    result = module.evaluate_status(
        {"started_epoch": 1000.0, "samples": samples},
        now_epoch=1000.0 + module.TARGET_SECONDS,
    )
    assert result["state"] == "failed"
    assert result["gates"]["event_chain_clean"] is False


def test_probe_connection_failure_becomes_evidence_instead_of_crashing():
    module = _module()
    with patch.object(
        module.urllib.request,
        "urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        status, payload = module._get_json("http://127.0.0.1:4101/readyz")

    assert status == 0
    assert payload["ok"] is False
    assert "connection refused" in payload["error"]


def test_transport_only_samples_cannot_pass_completion_gate():
    module = _module()
    sample = {
        "ready": True,
        "discord_connected": True,
        "completion_latency_ms": 10,
        "event_chain_ok": True,
        "restart_delta": 0,
        "restart_detected": False,
        "service_query_ok": True,
        "service_active": True,
        "service_invocation_unchanged": True,
        "brain_errors": 0,
        "completion_ok": False,
    }
    result = module.evaluate_status(
        {"started_epoch": 0, "samples": [sample] * module.MIN_SAMPLES},
        now_epoch=module.TARGET_SECONDS,
    )
    assert result["state"] == "failed"
    assert result["gates"]["completion_verified"] is False


def test_status_migration_preserves_samples_in_separate_history(tmp_path, monkeypatch):
    import sys

    module = _module()
    directory = tmp_path / "burnin"
    directory.mkdir()
    original = {"started_epoch": 0, "samples": [{"ready": False}]}
    (directory / "status.json").write_text(json.dumps(original))
    monkeypatch.setattr(module, "collect_sample", lambda *args: {"ready": True})
    monkeypatch.setattr(sys, "argv", ["burnin", "--state-dir", str(tmp_path)])
    assert module.main() == 0
    summary = json.loads((directory / "status.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    assert "samples" not in summary
    assert history["samples"] == [{"ready": False}, {"ready": True}]
    assert summary["assessment"]["sample_count"] == 2


def test_new_campaign_archives_prior_evidence_and_starts_clean(tmp_path, monkeypatch):
    import sys

    module = _module()
    directory = tmp_path / "burnin"
    directory.mkdir()
    for name, content in {
        "status.json": '{"old":true}\n',
        "history.json": '{"samples":[{"old":true}]}\n',
        "evidence.jsonl": '{"old":true}\n',
    }.items():
        (directory / name).write_text(content)
    monkeypatch.setattr(module, "_get_metrics", lambda *args: "")
    monkeypatch.setattr(
        module,
        "_service_state",
        lambda *args: {
            "query_ok": True,
            "NRestarts": 4,
            "ActiveState": "active",
            "SubState": "running",
            "InvocationID": "baseline-invocation",
            "MainPID": 123,
        },
    )
    monkeypatch.setattr(
        module,
        "collect_sample",
        lambda *args: {
            "ready": True,
            "completion_ok": True,
            "discord_connected": True,
            "event_chain_ok": True,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "burnin",
            "--state-dir",
            str(tmp_path),
            "--start-new-campaign",
            "post-audit",
        ],
    )

    assert module.main() == 0

    history = json.loads((directory / "history.json").read_text())
    assert history["version"] == 4
    assert history["campaign_id"] == "post-audit"
    assert len(history["samples"]) == 1
    archived = list((directory / "archive").glob("*-before-post-audit"))
    assert len(archived) == 1
    assert {path.name for path in archived[0].iterdir()} == {
        "archive.json",
        "status.json",
        "history.json",
        "evidence.jsonl",
    }
    assert stat.S_IMODE((directory / "archive").stat().st_mode) == 0o700
    assert stat.S_IMODE(archived[0].stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in archived[0].iterdir())
    assert stat.S_IMODE((directory / "evidence.jsonl").stat().st_mode) == 0o600


def test_new_campaign_name_cannot_escape_archive(tmp_path):
    module = _module()
    directory = tmp_path / "burnin"
    directory.mkdir()
    (directory / "status.json").write_text("{}")

    with pytest.raises(ValueError, match="campaign name"):
        module._archive_current_campaign(directory, "../escape")


def test_overlapping_collectors_preserve_both_samples(tmp_path, monkeypatch):
    import queue
    import sys
    import threading
    from concurrent.futures import ThreadPoolExecutor

    module = _module()
    entered = queue.Queue()
    release = threading.Event()
    monkeypatch.setattr(sys, "argv", ["burnin", "--state-dir", str(tmp_path)])
    monkeypatch.setattr(module, "_get_metrics", lambda *_: "")
    monkeypatch.setattr(module, "_service_state", lambda *_: {})

    def collect(*_args):
        entered.put(True)
        assert release.wait(timeout=3)
        return {"ready": True}

    monkeypatch.setattr(module, "collect_sample", collect)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(module.main)
        entered.get(timeout=2)
        second = executor.submit(module.main)
        try:
            with pytest.raises(queue.Empty):
                entered.get(timeout=0.15)
        finally:
            release.set()
        assert first.result(timeout=3) == 0
        assert second.result(timeout=3) == 0

    history = json.loads((tmp_path / "burnin" / "history.json").read_text())
    assert len(history["samples"]) == 2


def test_new_failures_are_measured_from_campaign_baselines(monkeypatch, tmp_path):
    module = _module()
    monkeypatch.setattr(
        module,
        "_get_json",
        lambda *args: (
            200,
            {
                "ok": True,
                "components": {
                    "discord": {"connected": True},
                    "completion_probe": {
                        "ok": True,
                        "completion_verified": True,
                        "latency_ms": 50,
                    },
                },
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "_get_metrics",
        lambda *args: "norax_brain_errors_total 8\nnorax_agent_turn_failed_total 13\n",
    )
    monkeypatch.setattr(
        module,
        "_service_state",
        lambda *args: {
            "query_ok": True,
            "NRestarts": 5,
            "ActiveState": "active",
            "SubState": "running",
            "InvocationID": "same-invocation",
            "MainPID": 123,
        },
    )
    monkeypatch.setattr(module, "_verify_single_file", lambda *args: {"ok": True})

    sample = module.collect_sample(
        "http://localhost",
        tmp_path / "events.jsonl",
        baseline_restarts=5,
        baseline_brain_errors=7,
        baseline_agent_turn_failures=11,
        baseline_service_invocation_id="same-invocation",
    )

    assert sample["brain_error_delta"] == 1
    assert sample["agent_turn_failure_delta"] == 2


def test_manual_restart_is_detected_from_invocation_identity(monkeypatch, tmp_path):
    module = _module()
    monkeypatch.setattr(
        module,
        "_get_json",
        lambda *args: (
            200,
            {
                "ok": True,
                "components": {
                    "discord": {"connected": True},
                    "completion_probe": {
                        "ok": True,
                        "completion_verified": True,
                        "latency_ms": 50,
                    },
                },
            },
        ),
    )
    monkeypatch.setattr(module, "_get_metrics", lambda *args: "")
    monkeypatch.setattr(
        module,
        "_service_state",
        lambda *args: {
            "query_ok": True,
            "NRestarts": 0,
            "ActiveState": "active",
            "SubState": "running",
            "InvocationID": "new-invocation",
            "MainPID": 456,
        },
    )
    monkeypatch.setattr(module, "_verify_single_file", lambda *args: {"ok": True})

    sample = module.collect_sample(
        "http://localhost",
        tmp_path / "events.jsonl",
        baseline_restarts=0,
        baseline_service_invocation_id="old-invocation",
    )
    assessment = module.evaluate_status(
        {"started_epoch": 0, "samples": [sample] * module.MIN_SAMPLES},
        now_epoch=module.TARGET_SECONDS,
    )

    assert sample["restart_count"] == 0
    assert sample["restart_detected"] is True
    assert sample["service_invocation_unchanged"] is False
    assert assessment["gates"]["no_restart_growth"] is False
    assert assessment["gates"]["service_continuously_active"] is False
