#!/usr/bin/env python3
"""Verify Norax can drive Chrome via Playwright in available display modes."""

from __future__ import annotations

import os
import subprocess

from playwright.sync_api import sync_playwright


def run(headless: bool) -> None:
    mode = "headless" if headless else "headed"
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=headless, args=["--no-sandbox"])
        page = browser.new_page()
        page.goto("https://example.com", wait_until="domcontentloaded", timeout=30_000)
        title = page.title()
        h1 = page.locator("h1").inner_text()
        browser.close()
    print(f"{mode}: {title} / {h1}")


def headed_display_available() -> bool:
    display = os.environ.get("DISPLAY", "")
    if not display:
        return False
    try:
        probe = subprocess.run(
            ["xdotool", "getdisplaygeometry"],
            capture_output=True,
            timeout=3,
            env=os.environ.copy(),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


if __name__ == "__main__":
    run(True)
    if headed_display_available():
        run(False)
    else:
        print("headed: SKIP (no accessible DISPLAY)")
