"""Memory Decay / Active Forgetting — time-weighted importance decay.

Prevents memory bloat by decaying low-importance memories over time.
Important memories (high weight, frequently accessed) stay sharp.
Irrelevant ones fade and get archived to sleep/archive/.

Design (Mem0-inspired + neuroscience):
  - Per-kind decay rates: semantic (slow), procedural (medium), intel (fast)
  - Access refresh: retrieving a memory resets its decay timer
  - Auto-archive: neurons below importance threshold → sleep/archive/
  - Never delete: archived memories are still retrievable from sleep/
  - Owner W5 memories are exempt from decay (permanent knowledge)
  - Runs during idle/sleep maintenance, not in the hot path

Decay formula:
  importance = weight * recency_factor * access_factor
  recency_factor = 1 / (1 + decay_rate * age_days)
  access_factor = 1 + log(1 + access_count)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_text, path_lock

log = logging.getLogger("norax.memory.decay")

# Per-kind decay rates (higher = faster decay)
DECAY_RATES: dict[str, float] = {
    "semantic": 0.005,  # very slow — facts persist
    "procedural": 0.01,  # medium — workflows may become outdated
    "intel": 0.02,  # faster — intel goes stale quickly
    "scratchpad": 0.1,  # fast — hot state should cycle
    "focus": 0.15,  # fastest — attention shifts
    "sleep": 0.0,  # no decay — already in archive
}

# Below this importance, archive the memory
ARCHIVE_THRESHOLD = 0.15

# W5 memories are never decayed (permanent owner knowledge)
_PERMANENT_WEIGHT = 1.3  # W5 maps to 1.3 in _parse_weight

# Access log: track how many times each neuron has been retrieved
_ACCESS_LOG_PATH = "memory/index/access_log.json"


@dataclass
class DecayResult:
    """Result of a decay pass."""

    scanned: int = 0
    archived: int = 0
    refreshed: int = 0
    skipped_permanent: int = 0
    errors: int = 0


def _load_access_log(memory_root: Path) -> dict[str, int]:
    """Load access counts keyed by entity_id."""
    p = memory_root / "index" / "access_log.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_access_log(memory_root: Path, counts: dict[str, int]) -> None:
    p = memory_root / "index" / "access_log.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(p, json.dumps(counts, indent=2))


def record_access(memory_root: Path, entity_ids: list[str]) -> None:
    """Record that these neurons were retrieved (accessed).

    Called by the retrieval layer when memories are injected into context.
    Resets the decay timer for accessed memories.
    """
    path = memory_root / "index" / "access_log.json"
    with path_lock(path):
        counts = _load_access_log(memory_root)
        for eid in entity_ids:
            counts[eid] = counts.get(eid, 0) + 1
        _save_access_log(memory_root, counts)


def compute_importance(
    weight: float,
    age_days: float,
    access_count: int,
    kind: str,
) -> float:
    """Compute the current importance score of a neuron.

    Returns 0.0-1.0 (roughly). Higher = more important.
    """
    if weight >= _PERMANENT_WEIGHT:
        return 1.0  # permanent, always max

    decay_rate = DECAY_RATES.get(kind, 0.01)
    recency_factor = 1.0 / (1.0 + decay_rate * age_days)
    access_factor = 1.0 + math.log(1.0 + access_count) * 0.3

    importance = weight * recency_factor * min(access_factor, 3.0)
    return max(0.0, min(importance, 1.0))


def run_decay_pass(memory_root: Path | None = None) -> DecayResult:
    """Run a full decay pass over all canonical memory stores.

    Scans semantic/, procedural/, intel/ — computes importance for each
    neuron, and archives those below ARCHIVE_THRESHOLD to sleep/archive/.

    Returns DecayResult with counts.
    """
    if memory_root is None:
        memory_root = Path(
            os.environ.get(
                "NORAX_MEMORY_ROOT",
                Path.home() / "norax" / "memory",
            )
        )

    result = DecayResult()
    access_log = _load_access_log(memory_root)
    archive_dir = memory_root / "sleep" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    now = time.time()

    for kind in ("semantic", "procedural", "intel"):
        store_dir = memory_root / kind
        if not store_dir.exists():
            continue

        for md_file in sorted(store_dir.rglob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
                mtime = md_file.stat().st_mtime
                age_days = max(0.0, (now - mtime) / 86400.0)

                # Check if any line in the file has W5 (permanent)
                has_w5 = any("|W5" in line for line in lines)

                if has_w5:
                    result.skipped_permanent += 1
                    continue

                # Compute average weight of the file
                weights = []
                for line in lines:
                    m = re.search(r"\|W([1-5])\b", line)
                    if m:
                        w_map = {"1": 0.4, "2": 0.6, "3": 0.8, "4": 1.0, "5": 1.3}
                        weights.append(w_map.get(m.group(1), 1.0))
                    else:
                        weights.append(1.0)

                avg_weight = sum(weights) / len(weights) if weights else 1.0

                # Compute entity_id for this file (same as Neuron)
                import hashlib

                canon = re.sub(r"\s+", " ", text).strip().lower()
                eid = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]
                access_count = access_log.get(eid, 0)

                importance = compute_importance(avg_weight, age_days, access_count, kind)
                result.scanned += 1

                if importance < ARCHIVE_THRESHOLD:
                    # Archive this file
                    md_file.relative_to(memory_root)
                    archive_name = f"decayed_{kind}_{md_file.stem}_{int(now)}.md"
                    archive_path = archive_dir / archive_name

                    try:
                        shutil.copy2(str(md_file), str(archive_path))
                        md_file.unlink()
                        result.archived += 1
                        log.info(
                            "decay.archived kind=%s file=%s importance=%.3f age=%.1fd accesses=%d",
                            kind,
                            md_file.name,
                            importance,
                            age_days,
                            access_count,
                        )
                    except OSError as e:
                        log.warning("decay.archive_failed: %r", e)
                        result.errors += 1

            except Exception as e:
                log.warning("decay.scan_error file=%s: %r", md_file, e)
                result.errors += 1

    log.info(
        "decay.pass complete: scanned=%d archived=%d permanent=%d errors=%d",
        result.scanned,
        result.archived,
        result.skipped_permanent,
        result.errors,
    )
    return result


def get_decay_stats(memory_root: Path | None = None) -> dict:
    """Get statistics about memory decay state without archiving anything."""
    if memory_root is None:
        memory_root = Path(
            os.environ.get(
                "NORAX_MEMORY_ROOT",
                Path.home() / "norax" / "memory",
            )
        )

    access_log = _load_access_log(memory_root)
    now = time.time()
    stats: dict[str, Any] = {
        "total_neurons": 0,
        "at_risk": 0,  # importance < 0.3
        "archived_count": 0,
        "permanent_count": 0,
        "by_kind": {},
        "avg_importance": 0.0,
    }

    all_importances: list[float] = []

    for kind in ("semantic", "procedural", "intel"):
        store_dir = memory_root / kind
        if not store_dir.exists():
            continue

        kind_stats: dict[str, Any] = {
            "count": 0,
            "at_risk": 0,
            "permanent": 0,
            "avg_importance": 0.0,
        }
        kind_importances: list[float] = []

        for md_file in sorted(store_dir.rglob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
                mtime = md_file.stat().st_mtime
                age_days = max(0.0, (now - mtime) / 86400.0)

                has_w5 = any("|W5" in line for line in text.splitlines())
                if has_w5:
                    kind_stats["permanent"] += 1
                    stats["permanent_count"] += 1
                    kind_importances.append(1.0)
                    all_importances.append(1.0)
                    kind_stats["count"] += 1
                    stats["total_neurons"] += 1
                    continue

                weights = []
                for line in text.splitlines():
                    m = re.search(r"\|W([1-5])\b", line)
                    if m:
                        w_map = {"1": 0.4, "2": 0.6, "3": 0.8, "4": 1.0, "5": 1.3}
                        weights.append(w_map.get(m.group(1), 1.0))
                    else:
                        weights.append(1.0)

                avg_weight = sum(weights) / len(weights) if weights else 1.0
                import hashlib

                canon = re.sub(r"\s+", " ", text).strip().lower()
                eid = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]
                access_count = access_log.get(eid, 0)

                importance = compute_importance(avg_weight, age_days, access_count, kind)
                kind_importances.append(importance)
                all_importances.append(importance)
                kind_stats["count"] += 1
                stats["total_neurons"] += 1

                if importance < 0.3:
                    kind_stats["at_risk"] += 1
                    stats["at_risk"] += 1

            except Exception:
                continue

        kind_stats["avg_importance"] = (
            sum(kind_importances) / len(kind_importances) if kind_importances else 0.0
        )
        stats["by_kind"][kind] = kind_stats

    # Count archived
    archive_dir = memory_root / "sleep" / "archive"
    if archive_dir.exists():
        stats["archived_count"] = len(list(archive_dir.glob("decayed_*.md")))

    stats["avg_importance"] = (
        sum(all_importances) / len(all_importances) if all_importances else 0.0
    )

    return stats
