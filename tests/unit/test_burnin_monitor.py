from __future__ import annotations

import importlib.util
import json
import stat
import urllib.error
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
    status = {
        "started_epoch": 1000.0,
        "samples": [
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
        ],
    }
    result = module.evaluate_status(status, now_epoch=1000.0 + module.TARGET_SECONDS)
    assert result["state"] == "passed"
    assert all(result["gates"].values())


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
    assert history["version"] == 3
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
