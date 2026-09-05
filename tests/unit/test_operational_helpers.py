from __future__ import annotations

import io
import subprocess
import sys
from types import SimpleNamespace

from scripts import memory_lint, monitor, publication_check, secret_scan


def test_operational_monitor_requires_active_required_or_enabled_units(monkeypatch):
    state = {
        "required.service": ("disabled", "inactive", "dead"),
        "optional-disabled.service": ("disabled", "inactive", "dead"),
        "optional-static.service": ("static", "inactive", "dead"),
        "optional-enabled.service": ("enabled", "inactive", "dead"),
    }

    def fake_run(command, **_kwargs):
        enabled, active, substate = state[command[3]]
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "LoadState=loaded\n"
                f"ActiveState={active}\n"
                f"SubState={substate}\n"
                f"UnitFileState={enabled}\n"
                "Result=success\n"
                "NRestarts=0\n"
            ),
        )

    monkeypatch.setattr(monitor.subprocess, "run", fake_run)

    assert monitor.systemd_status("required.service", required=True)["ok"] is False
    assert monitor.systemd_status("optional-disabled.service", required=False)["ok"] is True
    assert monitor.systemd_status("optional-static.service", required=False)["ok"] is True
    assert monitor.systemd_status("optional-enabled.service", required=False)["ok"] is False


def test_operational_monitor_retries_transient_systemd_query(monkeypatch):
    calls = 0

    def fake_run(_command, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr="temporary bus failure")
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "LoadState=loaded\n"
                "ActiveState=active\n"
                "SubState=running\n"
                "UnitFileState=enabled\n"
                "Result=success\n"
                "NRestarts=0\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(monitor.subprocess, "run", fake_run)

    result = monitor.systemd_status("required.service", required=True)

    assert calls == 2
    assert result["ok"] is True
    assert "query_error" not in result


def test_operational_monitor_reports_persistent_systemd_query_error(monkeypatch):
    def fake_run(_command, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="user bus unavailable")

    monkeypatch.setattr(monitor.subprocess, "run", fake_run)

    result = monitor.systemd_status("required.service", required=True)

    assert result["ok"] is False
    assert result["query_error"] == "user bus unavailable"


def test_operational_monitor_requires_literal_json_success_and_bounds_response(monkeypatch):
    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        monitor.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(b'{"ok":"true"}'),
    )
    assert monitor.json_health("http://health")["ok"] is False

    monkeypatch.setattr(
        monitor.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(b'{"ok":true}'),
    )
    assert monitor.json_health("http://health")["ok"] is True

    monkeypatch.setattr(
        monitor.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(b"x" * (monitor.MAX_HEALTH_BYTES + 1)),
    )
    oversized = monitor.json_health("http://health")
    assert oversized["ok"] is False
    assert "exceeds" in oversized["error"]


def test_secret_scanner_fails_when_repository_enumeration_fails(monkeypatch, capsys):
    def broken_files():
        raise subprocess.CalledProcessError(9, ["git", "ls-files"])
        yield  # pragma: no cover

    monkeypatch.setattr(secret_scan, "iter_files", broken_files)
    assert secret_scan.main() == 2
    assert "git exit 9" in capsys.readouterr().out


def test_secret_scanner_exempts_only_pure_environment_references():
    env_assignment = "".join(("TO", "KEN", '="${NORAX_', 'TOKEN:-}"\n'))
    assert secret_scan._ENV_REFERENCE_ASSIGNMENT.fullmatch(env_assignment)
    compound_assignment = "".join(
        ("TO", "KEN", '="${NORAX_TOKEN:-}"; PASS', 'WORD="real-secret-value"\n')
    )
    assert not secret_scan._ENV_REFERENCE_ASSIGNMENT.fullmatch(compound_assignment)
    unquoted = next(
        pattern for label, pattern in secret_scan.PATTERNS if label.endswith("unquoted")
    )
    assigned_value = "".join(("to", "ken = ", "abcdefghijklmnopqrstuvwxyz"))
    assert unquoted.search(assigned_value)


def test_memory_lint_refuses_vacuous_default(monkeypatch, capsys):
    monkeypatch.delenv("NORAX_MEMORY_ROOT", raising=False)
    monkeypatch.setattr(sys, "argv", ["memory_lint"])
    assert memory_lint.main() == 2
    assert "is required" in capsys.readouterr().out


def test_publication_check_rejects_runtime_state_inside_package(tmp_path, monkeypatch, capsys):
    residue = tmp_path / "norax/memory/intel.md"
    residue.parent.mkdir(parents=True)
    residue.write_text("private runtime state", encoding="utf-8")
    for required in publication_check.REQUIRED:
        (tmp_path / required).write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(publication_check, "ROOT", tmp_path)
    monkeypatch.setattr(
        publication_check,
        "publication_files",
        lambda: [residue.relative_to(tmp_path)],
    )

    assert publication_check.main() == 1
    assert "private package-tree state" in capsys.readouterr().out


def test_publication_check_rejects_hidden_package_files(tmp_path, monkeypatch, capsys):
    residue = tmp_path / "norax/.unexpected"
    residue.parent.mkdir(parents=True)
    residue.write_text("", encoding="utf-8")
    for required in publication_check.REQUIRED:
        (tmp_path / required).write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(publication_check, "ROOT", tmp_path)
    monkeypatch.setattr(
        publication_check,
        "publication_files",
        lambda: [residue.relative_to(tmp_path)],
    )

    assert publication_check.main() == 1
    assert "hidden package file" in capsys.readouterr().out


def test_publication_check_rejects_unapproved_top_level_tree(tmp_path, monkeypatch, capsys):
    residue = tmp_path / "alternate-source/runtime.py"
    residue.parent.mkdir(parents=True)
    residue.write_text("print('not reviewed')", encoding="utf-8")
    for required in publication_check.REQUIRED:
        (tmp_path / required).write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(publication_check, "ROOT", tmp_path)
    monkeypatch.setattr(
        publication_check,
        "publication_files",
        lambda: [residue.relative_to(tmp_path)],
    )

    assert publication_check.main() == 1
    assert "unapproved top-level publication path" in capsys.readouterr().out


def test_publication_check_rejects_unapproved_binary_archive(tmp_path, monkeypatch, capsys):
    residue = tmp_path / "docs/source-export.zip"
    residue.parent.mkdir(parents=True)
    residue.write_bytes(b"PK\x03\x04not-a-reviewed-source-file")
    for required in publication_check.REQUIRED:
        (tmp_path / required).write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(publication_check, "ROOT", tmp_path)
    monkeypatch.setattr(
        publication_check,
        "publication_files",
        lambda: [residue.relative_to(tmp_path)],
    )

    assert publication_check.main() == 1
    assert "unapproved non-text publication file" in capsys.readouterr().out
