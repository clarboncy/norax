#!/usr/bin/env python3
"""Measured desktop-window and Chrome-tab inventory.

This is a compatibility inventory helper, not an accessibility-tree mapper.
Use ``perception.py perceive --tier cdp`` for the browser accessibility tree.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def get_cdp_pages() -> tuple[list[dict[str, Any]], str | None]:
    """Return measured Chrome page targets and any probe error."""
    import browser_controller

    pages, error = browser_controller._query_tabs()
    return (
        [
            {
                "id": page.get("id", ""),
                "title": page.get("title", ""),
                "url": page.get("url", ""),
            }
            for page in pages
            if not str(page.get("url", "")).startswith("blob:")
        ],
        error,
    )


def _get_windows_wmctrl(limit: int) -> tuple[list[dict[str, Any]], str | None]:
    """Read EWMH window geometry in one process when wmctrl is available."""
    if shutil.which("wmctrl") is None:
        return [], "wmctrl is unavailable"
    try:
        result = subprocess.run(
            ["wmctrl", "-lG"], capture_output=True, text=True, timeout=4, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return [], result.stderr.strip() or f"wmctrl exited with {result.returncode}"

    windows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines()[:limit]:
        parts = line.split(maxsplit=7)
        if len(parts) < 7:
            continue
        window_id, _desktop, x, y, width, height, _host = parts[:7]
        if not all(value.lstrip("-").isdigit() for value in (x, y, width, height)):
            continue
        title = parts[7].strip() if len(parts) == 8 else ""
        if title and int(width) > 50 and int(height) > 0:
            windows.append(
                {
                    "id": window_id,
                    "name": title[:200],
                    "x": int(x),
                    "y": int(y),
                    "width": int(width),
                    "height": int(height),
                    "source": "wmctrl_ewmh",
                }
            )
    return windows, None


def _get_windows_xdotool(
    limit: int, deadline_seconds: float = 5.0
) -> tuple[list[dict[str, Any]], str | None]:
    """Bounded compatibility fallback for X11 environments without wmctrl."""
    deadline = time.monotonic() + deadline_seconds
    try:
        result = subprocess.run(
            ["xdotool", "search", "--onlyvisible", "--name", ""],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return [], result.stderr.strip() or f"xdotool search exited with {result.returncode}"

    windows: list[dict[str, Any]] = []
    for window_id in result.stdout.splitlines()[:limit]:
        if not window_id.isdigit():
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            name_result = subprocess.run(
                ["xdotool", "getwindowname", window_id],
                capture_output=True,
                text=True,
                timeout=min(0.75, remaining),
                check=False,
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            geometry_result = subprocess.run(
                ["xdotool", "getwindowgeometry", "--shell", window_id],
                capture_output=True,
                text=True,
                timeout=min(0.75, remaining),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if name_result.returncode != 0 or geometry_result.returncode != 0:
            continue
        geometry: dict[str, int] = {}
        for line in geometry_result.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator and value.lstrip("-").isdigit():
                geometry[key] = int(value)
        name = name_result.stdout.strip()
        if name and geometry.get("WIDTH", 0) > 50:
            windows.append(
                {
                    "id": window_id,
                    "name": name[:200],
                    "x": geometry.get("X", 0),
                    "y": geometry.get("Y", 0),
                    "width": geometry.get("WIDTH", 0),
                    "height": geometry.get("HEIGHT", 0),
                    "source": "xdotool_fallback",
                }
            )
    return windows, None


def get_windows(limit: int = 30) -> tuple[list[dict[str, Any]], str | None]:
    """Return measured X11 windows; Wayland-native windows may be absent."""
    windows, wmctrl_error = _get_windows_wmctrl(limit)
    if wmctrl_error is None:
        return windows, None
    windows, xdotool_error = _get_windows_xdotool(limit)
    if xdotool_error is None:
        return windows, None
    return [], f"wmctrl: {wmctrl_error}; xdotool: {xdotool_error}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab", help="Only return a tab with this ID or ID prefix")
    parser.add_argument("--window-limit", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.window_limit <= 100:
        parser.error("--window-limit must be between 1 and 100")

    pages, cdp_error = get_cdp_pages()
    if args.tab:
        pages = [page for page in pages if str(page["id"]).startswith(args.tab)]
        if not pages and cdp_error is None:
            cdp_error = f"tab not found: {args.tab}"
    windows, windows_error = get_windows(args.window_limit)
    ok = cdp_error is None or windows_error is None
    payload = {
        "ok": ok,
        "inventory_kind": "measured_tabs_and_x11_windows",
        "windows": windows,
        "tabs": pages,
        "errors": {
            key: value
            for key, value in {"cdp": cdp_error, "x11_windows": windows_error}.items()
            if value
        },
        "accessibility_tree_command": [
            sys.executable,
            str(TOOLS / "perception.py"),
            "perceive",
            "--tier",
            "cdp",
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
