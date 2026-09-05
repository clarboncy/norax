from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "migrate_runtime_data.py"
    spec = importlib.util.spec_from_file_location("migrate_runtime_data", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_tree(root: Path) -> None:
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "fact.md").write_text("durable memory\n", encoding="utf-8")
    (root / "state").mkdir()
    (root / "state" / "runtime.json").write_text("{}\n", encoding="utf-8")
    (root / "state" / "events.jsonl").write_text('{"legacy":true}\n', encoding="utf-8")
    (root / "state" / "events.1.jsonl").write_text('{"older":true}\n', encoding="utf-8")
    (root / "state" / "events.jsonl.anchor").write_text("anchor\n", encoding="utf-8")
    (root / "logs").mkdir()
    (root / "logs" / "norax.log").write_text("legacy log\n", encoding="utf-8")


def test_dry_run_does_not_create_targets(tmp_path: Path):
    module = _module()
    source = tmp_path / "source"
    source.mkdir()
    _source_tree(source)
    target = tmp_path / "external"

    result = module.migrate(
        source_root=source,
        memory_root=target / "memory",
        state_dir=target / "state",
        log_dir=target / "logs",
        execute=False,
    )

    assert result["ok"] is True
    assert result["legacy_event_files"] == 2
    assert not target.exists()


def test_execute_copies_data_and_quarantines_event_history(tmp_path: Path):
    module = _module()
    source = tmp_path / "source"
    source.mkdir()
    _source_tree(source)
    target = tmp_path / "external"
    kwargs = {
        "source_root": source,
        "memory_root": target / "memory",
        "state_dir": target / "state",
        "log_dir": target / "logs",
        "execute": True,
    }

    first = module.migrate(**kwargs)

    assert (target / "memory" / "fact.md").read_text() == "durable memory\n"
    assert (target / "state" / "runtime.json").exists()
    assert not (target / "state" / "events.jsonl").exists()
    assert (target / "logs" / "legacy" / "norax.log").exists()
    legacy_dir = Path(first["migration"]["legacy_event_dir"])
    manifest = json.loads((legacy_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "quarantined_corrupt_history"
    assert {entry["name"] for entry in manifest["files"]} == {
        "events.jsonl",
        "events.1.jsonl",
    }
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])

    second = module.migrate(**kwargs)
    assert second["already_migrated"] is True
    assert second["migration"] == first["migration"]


def test_refuses_targets_inside_source_checkout(tmp_path: Path):
    module = _module()
    source = tmp_path / "source"
    source.mkdir()

    try:
        module.migrate(
            source_root=source,
            memory_root=source / "memory-external",
            state_dir=tmp_path / "state",
            log_dir=tmp_path / "logs",
            execute=False,
        )
    except ValueError as exc:
        assert "outside the source checkout" in str(exc)
    else:
        raise AssertionError("migration accepted a target inside the source checkout")
