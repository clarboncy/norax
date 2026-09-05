#!/usr/bin/env python3
"""Copy live Norax data out of the source checkout and start a clean chain.

The migration is deliberately non-destructive: source memory, logs, and event
generations remain untouched for rollback and forensic retention.  Historical
event files are copied under ``legacy-corrupt/`` and never joined to the new
active chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


def _copy_tree(source: Path, target: Path) -> int:
    copied = 0
    if not source.exists():
        return copied
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if path.is_symlink():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied += 1
    return copied


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_external(source_root: Path, target: Path, label: str) -> None:
    try:
        target.relative_to(source_root)
    except ValueError:
        return
    raise ValueError(f"{label} must be outside the source checkout: {target}")


def migrate(
    *,
    source_root: Path,
    memory_root: Path,
    state_dir: Path,
    log_dir: Path,
    execute: bool,
) -> dict:
    source_root = source_root.resolve()
    memory_root = memory_root.expanduser().resolve()
    state_dir = state_dir.expanduser().resolve()
    log_dir = log_dir.expanduser().resolve()
    for label, target in (
        ("memory root", memory_root),
        ("state directory", state_dir),
        ("log directory", log_dir),
    ):
        _ensure_external(source_root, target, label)

    marker = state_dir / "runtime-data-migration.json"
    event_sources = sorted((source_root / "state").glob("events*.jsonl"))
    plan = {
        "ok": True,
        "execute": execute,
        "source_root": str(source_root),
        "memory_root": str(memory_root),
        "state_dir": str(state_dir),
        "log_dir": str(log_dir),
        "legacy_event_files": len(event_sources),
        "already_migrated": marker.exists(),
    }
    if not execute:
        return plan
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        return {**plan, "already_migrated": True, "migration": previous}
    if (state_dir / "events.jsonl").exists():
        raise FileExistsError(
            f"refusing to replace an active external chain: {state_dir / 'events.jsonl'}"
        )

    memory_root.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    memory_files = _copy_tree(source_root / "memory", memory_root)

    # Copy non-event runtime state directly; event generations are quarantined.
    state_files = 0
    source_state = source_root / "state"
    for path in source_state.rglob("*") if source_state.exists() else []:
        relative = path.relative_to(source_state)
        if relative.parts and relative.parts[0].startswith("events"):
            continue
        destination = state_dir / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file() and not path.is_symlink():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            state_files += 1

    cutover = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    legacy_dir = state_dir / "legacy-corrupt" / cutover
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_records: list[dict[str, object]] = []
    for source in event_sources:
        destination = legacy_dir / source.name
        shutil.copy2(source, destination)
        legacy_records.append(
            {
                "name": source.name,
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    anchor = source_state / "events.jsonl.anchor"
    if anchor.exists():
        shutil.copy2(anchor, legacy_dir / anchor.name)

    legacy_manifest = {
        "status": "quarantined_corrupt_history",
        "reason": "release audit found hash mismatches and continuity breaks",
        "cutover_utc": datetime.now(UTC).isoformat(),
        "source": str(source_state),
        "files": legacy_records,
        "policy": "immutable evidence; never join to the new active chain",
    }
    (legacy_dir / "manifest.json").write_text(
        json.dumps(legacy_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    log_files = _copy_tree(source_root / "logs", log_dir / "legacy")
    migration = {
        "version": 1,
        "completed_utc": datetime.now(UTC).isoformat(),
        "source_root": str(source_root),
        "memory_root": str(memory_root),
        "state_dir": str(state_dir),
        "log_dir": str(log_dir),
        "memory_files_copied": memory_files,
        "state_files_copied": state_files,
        "log_files_copied": log_files,
        "legacy_event_dir": str(legacy_dir),
        "new_chain": str(state_dir / "events.jsonl"),
        "rollback": "remove NORAX_MEMORY_ROOT/NORAX_STATE_DIR/NORAX_LOG_DIR overrides and restart",
    }
    marker.write_text(json.dumps(migration, indent=2) + "\n", encoding="utf-8")
    return {**plan, "migration": migration}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--memory-root", type=Path, default=Path.home() / ".local/share/norax/memory"
    )
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".local/state/norax")
    parser.add_argument("--log-dir", type=Path, default=Path.home() / ".local/state/norax/logs")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        result = migrate(
            source_root=args.source_root,
            memory_root=args.memory_root,
            state_dir=args.state_dir,
            log_dir=args.log_dir,
            execute=args.execute,
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
