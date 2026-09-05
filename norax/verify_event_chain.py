"""Canonical event-chain verifier — validates hash chain across all rotations.

Usage:
  python -m norax.verify_event_chain                    # verify state/events.jsonl
  python -m norax.verify_event_chain --all-generations  # verify all rotated files + current
  python -m norax.verify_event_chain --dir state/        # custom directory
  python -m norax.verify_event_chain --file state/events.jsonl  # single file

When --all-generations is used, the verifier:
  1. Sorts rotated files by timestamp in their filename
  2. Supplies each file's predecessor tail hash as the anchor for the next
  3. Validates the anchor file if present
  4. Reports the global root hash, tail hash, and total record count
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from itertools import islice
from pathlib import Path
from typing import Any

from .atomic import read_bounded_text

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_EVENT_LINE_BYTES = 4 * 1024 * 1024
_MAX_EVENT_FILE_BYTES = 256 * 1024 * 1024
_MAX_ANCHOR_BYTES = 64 * 1024
_MAX_ROTATIONS = 10_000


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and _HASH_RE.fullmatch(value) is not None


def _verify_single_file(
    path: Path,
    *,
    expected_prev_hash: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Verify a single event log file. Returns validation result dict."""
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        return {"ok": False, "file": str(path), "error": "invalid_limit", "records": 0}
    if expected_prev_hash is not None and not _valid_hash(expected_prev_hash):
        return {
            "ok": False,
            "file": str(path),
            "error": "invalid_expected_prev_hash",
            "records": 0,
        }

    records_checked = 0
    valid_records = 0
    corrupt_records = 0
    previous_hash: str | None = expected_prev_hash
    first_prev_hash = ""
    tail_hash = "0" * 64
    first_error: dict[str, Any] | None = None

    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return {"ok": False, "file": str(path), "error": "file_missing", "records": 0}
    except OSError as exc:
        return {
            "ok": False,
            "file": str(path),
            "error": "file_read_error",
            "detail": f"{type(exc).__name__}: {exc}"[:500],
            "records": 0,
        }

    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("event log is not a regular file")
        if file_stat.st_size > _MAX_EVENT_FILE_BYTES:
            return {
                "ok": False,
                "file": str(path),
                "error": "file_too_large",
                "records": 0,
            }
        with os.fdopen(descriptor, "rb") as fh:
            descriptor = -1
            line_no = 0
            while True:
                raw_line = fh.readline(_MAX_EVENT_LINE_BYTES + 1)
                if not raw_line:
                    break
                line_no += 1
                if limit is not None and records_checked >= limit:
                    break
                if len(raw_line) > _MAX_EVENT_LINE_BYTES:
                    while raw_line and not raw_line.endswith(b"\n"):
                        raw_line = fh.readline(_MAX_EVENT_LINE_BYTES + 1)
                    records_checked += 1
                    corrupt_records += 1
                    if first_error is None:
                        first_error = {"line": line_no, "error": "line_too_large"}
                    continue
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    records_checked += 1
                    corrupt_records += 1
                    if first_error is None:
                        first_error = {"line": line_no, "error": "invalid_utf8"}
                    continue
                if not line.strip():
                    continue
                records_checked += 1
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, RecursionError, ValueError) as exc:
                    corrupt_records += 1
                    if first_error is None:
                        first_error = {"line": line_no, "error": f"json_decode:{exc}"}
                    continue

                if not isinstance(rec, dict):
                    corrupt_records += 1
                    if first_error is None:
                        first_error = {"line": line_no, "error": "record_not_object"}
                    continue

                raw_got_hash = rec.get("hash")
                raw_prev_hash = rec.get("prev_hash")
                if not first_prev_hash and isinstance(raw_prev_hash, str):
                    first_prev_hash = raw_prev_hash
                if not _valid_hash(raw_prev_hash):
                    corrupt_records += 1
                    if first_error is None:
                        first_error = {"line": line_no, "error": "invalid_prev_hash"}
                    if _valid_hash(raw_got_hash):
                        previous_hash = str(raw_got_hash)
                        tail_hash = str(raw_got_hash)
                    continue
                if not _valid_hash(raw_got_hash):
                    corrupt_records += 1
                    if first_error is None:
                        first_error = {"line": line_no, "error": "invalid_hash"}
                    continue
                prev_hash = str(raw_prev_hash)
                got_hash = str(raw_got_hash)

                body_rec = dict(rec)
                body_rec.pop("hash", None)
                try:
                    body = json.dumps(body_rec, separators=(",", ":"), sort_keys=True)
                except (RecursionError, TypeError, ValueError) as exc:
                    corrupt_records += 1
                    if first_error is None:
                        first_error = {
                            "line": line_no,
                            "error": f"record_encode:{type(exc).__name__}",
                        }
                    previous_hash = got_hash
                    tail_hash = got_hash
                    continue
                expected = hashlib.sha256((prev_hash + body).encode("utf-8")).hexdigest()
                record_ok = got_hash == expected

                if previous_hash is not None and prev_hash != previous_hash:
                    record_ok = False
                    if first_error is None:
                        first_error = {
                            "line": line_no,
                            "error": "cross_file_prev_hash_mismatch"
                            if records_checked == 1 and expected_prev_hash is not None
                            else "prev_hash_mismatch",
                            "expected": previous_hash,
                            "got": prev_hash,
                        }

                if got_hash != expected and first_error is None:
                    first_error = {"line": line_no, "error": "hash_mismatch"}

                if record_ok:
                    valid_records += 1
                else:
                    corrupt_records += 1

                previous_hash = got_hash
                tail_hash = got_hash
    except (OSError, ValueError) as exc:
        return {
            "ok": False,
            "file": str(path),
            "error": "file_read_error",
            "detail": f"{type(exc).__name__}: {exc}"[:500],
            "records": records_checked,
        }
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    result: dict[str, Any] = {
        "ok": corrupt_records == 0 and records_checked > 0,
        "file": str(path),
        "records": records_checked,
        "valid": valid_records,
        "corrupt": corrupt_records,
        "first_prev_hash": first_prev_hash,
        "tail_hash": tail_hash,
    }
    if first_error:
        result.update(first_error)
    return result


def _parse_rotation_timestamp(filename: str) -> str:
    """Extract the sortable timestamp from an event rotation filename."""
    m = re.fullmatch(
        r"events-(\d{8}T\d{6}(?:\d{6})?Z)(?:-[0-9A-Z]+)?\.jsonl",
        filename,
    )
    if m is None:
        return ""
    timestamp = m.group(1)
    if len(timestamp) == 16:  # Legacy second-resolution writer.
        timestamp = f"{timestamp[:-1]}000000Z"
    return timestamp


def verify_all_generations(state_dir: Path) -> dict[str, Any]:
    """Verify all event log generations in chronological order."""
    # Find all rotated files + current
    rotation_candidates = list(islice(state_dir.glob("events-*.jsonl"), _MAX_ROTATIONS + 1))
    if len(rotation_candidates) > _MAX_ROTATIONS:
        return {
            "ok": False,
            "error": "too_many_rotation_files",
            "dir": str(state_dir),
        }
    invalid_rotation_files = sorted(
        f.name for f in rotation_candidates if not _parse_rotation_timestamp(f.name)
    )
    if invalid_rotation_files:
        return {
            "ok": False,
            "error": "invalid_rotation_filename",
            "dir": str(state_dir),
            "files": invalid_rotation_files,
        }
    rotated = sorted(rotation_candidates, key=lambda f: _parse_rotation_timestamp(f.name))
    current = state_dir / "events.jsonl"
    anchor = state_dir / "events.jsonl.anchor"

    all_files = rotated + ([current] if current.exists() or current.is_symlink() else [])
    if not all_files:
        return {"ok": False, "error": "no_event_files_found", "dir": str(state_dir)}

    # The writer records the latest rotated filename and its tail hash. This is
    # a boundary assertion for the latest rotation/current file, not an anchor
    # for the oldest file in the directory.
    anchor_present = anchor.exists() or anchor.is_symlink()
    anchor_status: dict[str, Any] = {"present": anchor_present, "ok": True}
    anchor_data: dict[str, Any] | None = None
    if anchor_present:
        try:
            raw_anchor = json.loads(read_bounded_text(anchor, max_bytes=_MAX_ANCHOR_BYTES))
            if not isinstance(raw_anchor, dict):
                raise ValueError("anchor must be a JSON object")
            rotated_name = raw_anchor.get("rotated")
            tail_hash = raw_anchor.get("tail_hash")
            if not isinstance(rotated_name, str) or Path(rotated_name).name != rotated_name:
                raise ValueError("anchor rotated filename is invalid")
            if not _valid_hash(tail_hash):
                raise ValueError("anchor tail_hash must be 64 lowercase hex characters")
            anchor_data = {"rotated": rotated_name, "tail_hash": tail_hash}
            anchor_status.update(anchor_data)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            anchor_status = {
                "present": True,
                "ok": False,
                "error": "invalid_anchor",
                "detail": f"{type(exc).__name__}: {exc}"[:500],
            }

    results: list[dict[str, Any]] = []
    total_records = 0
    total_valid = 0
    total_corrupt = 0
    global_ok = True
    global_root = ""
    global_tail = ""

    for i, f in enumerate(all_files):
        prev_for_this = results[-1].get("tail_hash") if results else None
        r = _verify_single_file(f, expected_prev_hash=prev_for_this)
        results.append(r)
        total_records += r.get("records", 0)
        total_valid += r.get("valid", 0)
        total_corrupt += r.get("corrupt", 0)
        if r.get("ok") is not True:
            global_ok = False
        if i == 0:
            global_root = r.get("first_prev_hash", "")
        global_tail = r.get("tail_hash", "")

    if not anchor_status["ok"]:
        global_ok = False
    elif anchor_data is not None:
        if not rotated:
            anchor_status.update(ok=False, error="anchor_without_rotated_file")
            global_ok = False
        else:
            latest_rotated = rotated[-1]
            latest_index = all_files.index(latest_rotated)
            latest_result = results[latest_index]
            if anchor_data["rotated"] != latest_rotated.name:
                anchor_status.update(
                    ok=False,
                    error="anchor_rotation_mismatch",
                    expected_rotated=latest_rotated.name,
                )
                global_ok = False
            elif anchor_data["tail_hash"] != latest_result.get("tail_hash"):
                anchor_status.update(
                    ok=False,
                    error="anchor_tail_hash_mismatch",
                    observed_tail_hash=latest_result.get("tail_hash", ""),
                )
                global_ok = False

    return {
        "ok": global_ok,
        "generations": len(all_files),
        "total_records": total_records,
        "total_valid": total_valid,
        "total_corrupt": total_corrupt,
        "global_root_hash": global_root,
        "global_tail_hash": global_tail,
        "anchor": anchor_status,
        "files": [
            {
                "file": r["file"],
                "ok": r["ok"],
                "records": r.get("records", 0),
                "corrupt": r.get("corrupt", 0),
            }
            for r in results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Norax event chain integrity")
    parser.add_argument(
        "--all-generations",
        action="store_true",
        help="Verify all rotated event files in chronological order",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(os.environ.get("NORAX_STATE_DIR", "state")),
        help="Directory containing event log files (default: state)",
    )
    parser.add_argument(
        "--file", type=Path, default=None, help="Verify a single file instead of the directory"
    )
    args = parser.parse_args()

    if args.file:
        result = _verify_single_file(args.file)
    elif args.all_generations:
        result = verify_all_generations(args.dir)
    else:
        result = _verify_single_file(args.dir / "events.jsonl")

    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    sys.exit(main())
