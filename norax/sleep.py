"""`python -m norax.sleep` — offline sleep-flush worker.

Per §6 of ongoing.md, sleep-flush is NOT a continuous background
poller inside the runtime. It is a separate, idempotent CLI invoked
by a systemd timer (or `cron`, or a manual run).

Usage:
    python -m norax.sleep                # flush once, exit
    python -m norax.sleep --dry-run      # report what would be flushed
    python -m norax.sleep --memory-root /path/to/memory
    python -m norax.sleep --min-age 60
    python -m norax.sleep --min-importance 0.5

Exit code 0 on success (no spills, or successful flush).
Exit code 1 on unhandled error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .config.loader import load_config
from .context.sleep_flush import SleepFlusher
from .memory.consolidator import MemoryConsolidator

log = logging.getLogger("norax.sleep")


def _resolve_memory_root(cli_arg: str | None, cfg_root: Path) -> Path:
    if cli_arg:
        return Path(cli_arg).expanduser().resolve()
    env = os.environ.get("NORAX_MEMORY_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return cfg_root / "memory"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="norax.sleep",
        description="Norax offline sleep-flush worker (idempotent).",
    )
    parser.add_argument("--memory-root", default=None, help="Override memory root path")
    parser.add_argument(
        "--min-age", type=float, default=None, help="Minimum spill age in seconds (default 300)"
    )
    parser.add_argument(
        "--min-importance",
        type=float,
        default=None,
        help="Minimum candidate importance (default 0.9)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report plan, write nothing")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = load_config()
    memory_root = _resolve_memory_root(args.memory_root, cfg.project_root)
    if not memory_root.exists():
        log.warning("memory_root does not exist: %s", memory_root)
        # It's fine — nothing to flush. Exit 0.
        if args.json:
            print(json.dumps({"ok": True, "skipped": "no_memory_root"}))
        return 0

    flusher_kwargs: dict = {"memory_root": memory_root}
    if args.min_age is not None:
        flusher_kwargs["min_age_sec"] = args.min_age
    if args.min_importance is not None:
        flusher_kwargs["min_importance"] = args.min_importance

    flusher = SleepFlusher(**flusher_kwargs)
    cons_res = None
    try:
        # Serialize with the live runtime's consolidation and projection rebuild.
        from .memory.process_lock import memory_process_lock

        with memory_process_lock(memory_root):
            # The pipeline owns the lock across both phases. Public methods
            # acquire it again on another descriptor and would deadlock.
            res = flusher._flush_unlocked(dry_run=args.dry_run)
            if not args.dry_run:
                consolidator = MemoryConsolidator(
                    memory_root=memory_root,
                    min_age_sec=flusher_kwargs.get("min_age_sec", 300.0),
                )
                cons_res = consolidator._consolidate_unlocked(dry_run=False)
    except Exception:
        log.exception("sleep-flush pipeline crashed")
        return 1

    summary = {
        "ok": True,
        "dry_run": bool(args.dry_run),
        "spills_processed": res.spills_processed,
        "candidates_seen": res.candidates_seen,
        "written": dict(res.written),
        "archived": [str(p) for p in res.archived],
        "memory_root": str(memory_root),
    }
    if cons_res is not None:
        summary["consolidated"] = {
            "files_processed": cons_res.files_processed,
            "facts_written": cons_res.facts_written,
            "procedural_written": cons_res.procedural_written,
            "archived": cons_res.archived,
            "skipped_dup": cons_res.skipped_dup,
        }
    if args.json:
        print(json.dumps(summary))
    else:
        log.info(
            "sleep-flush done spills=%d cands=%d sem=%d proc=%d intel=%d "
            "discard=%d dup=%d archived=%d%s",
            res.spills_processed,
            res.candidates_seen,
            res.written.get("semantic", 0),
            res.written.get("procedural", 0),
            res.written.get("intel", 0),
            res.written.get("discard", 0),
            res.written.get("dup", 0),
            len(res.archived),
            " [dry-run]" if args.dry_run else "",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
