#!/usr/bin/env python3
"""Memory hygiene checks: duplicate archive payloads and oversized semantic blobs."""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_SEMANTIC_BYTES = 512_000
WARN_SEMANTIC_BYTES = 128_000
MAX_LINE_BYTES = 64_000


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-root",
        type=Path,
        default=Path(os.environ["NORAX_MEMORY_ROOT"])
        if os.environ.get("NORAX_MEMORY_ROOT")
        else None,
    )
    args = parser.parse_args()
    if args.memory_root is None:
        print("memory_lint: --memory-root or NORAX_MEMORY_ROOT is required")
        return 2
    memory = args.memory_root.expanduser().resolve()
    if not memory.is_dir():
        print(f"memory_lint: memory root is not a directory: {memory}")
        return 2
    display_root = memory.parent
    rc = 0
    archive = memory / "sleep" / "archive"
    if archive.exists():
        groups: dict[str, list[Path]] = defaultdict(list)
        for p in archive.rglob("*"):
            if p.is_file() and ".deduped" not in p.parts:
                groups[sha(p)].append(p)
        dup_groups = [v for v in groups.values() if len(v) > 1]
        if dup_groups:
            print(
                f"memory_lint: duplicate sleep archive groups={len(dup_groups)} files={sum(len(g) for g in dup_groups)}"
            )
            for group in dup_groups[:10]:
                print("  dup:", ", ".join(str(p.relative_to(display_root)) for p in group[:5]))
            rc = 1
    sem = memory / "semantic"
    if sem.exists():
        for p in sem.rglob("*.md"):
            size = p.stat().st_size
            if size > MAX_SEMANTIC_BYTES:
                print(
                    f"memory_lint: oversized semantic file {p.relative_to(display_root)} {size}B > {MAX_SEMANTIC_BYTES}B"
                )
                rc = 1
            elif size > WARN_SEMANTIC_BYTES:
                print(
                    f"memory_lint: warn large semantic file {p.relative_to(display_root)} {size}B"
                )
            for line_number, line in enumerate(p.open("rb"), 1):
                if len(line) > MAX_LINE_BYTES:
                    print(
                        "memory_lint: oversized semantic line "
                        f"{p.relative_to(display_root)}:{line_number} "
                        f"{len(line)}B > {MAX_LINE_BYTES}B"
                    )
                    rc = 1
    if rc == 0:
        print("memory_lint: ok")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
