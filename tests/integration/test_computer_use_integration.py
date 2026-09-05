#!/usr/bin/env python3
"""
Integration tests for the computer-use bridge pipeline.

Tests multi-command workflows that chain primitives together:
  1. Browser navigation + perception + DOM verification
  2. Tab lifecycle (open → switch → info → close)
  3. Scroll + verify position changed
  4. Clipboard round-trip
  5. Screenshot + map cascade
  6. JS execution + state verification
  7. Desktop actions (click + move + drag)
  8. Scroll direction fix (down/up words)
  9. Full Wikipedia search workflow
 10. Info returns system capabilities
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.host_integration

BRIDGE = Path(__file__).resolve().parents[2] / "tools" / "bridge.py"
PYTHON = sys.executable
TIMEOUT = 30


def _bridge(*args, timeout=TIMEOUT):
    """Run bridge.py with args, return parsed JSON dict."""
    proc = subprocess.run(
        [PYTHON, str(BRIDGE), *args], capture_output=True, text=True, timeout=timeout
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": f"stdout: {proc.stdout[:500]}, stderr: {proc.stderr[:500]}"}


def _bridge_ok(*args, timeout=TIMEOUT):
    """Run bridge, assert ok=True, return data field."""
    r = _bridge(*args, timeout=timeout)
    assert r.get("ok"), f"bridge {args} failed: {r}"
    return r.get("data", r)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Browser navigation + perception + DOM verification
# ─────────────────────────────────────────────────────────────────────────────
def test_navigate_then_perceive_then_js():
    """Navigate to a page, perceive it, verify title via JS."""
    _bridge_ok("navigate", "https://en.wikipedia.org/wiki/Robot")
    _bridge_ok("wait", "2")

    js_result = _bridge_ok("js", "document.title")
    title = js_result.get("value", "")
    assert "Robot" in str(title), f"Expected 'Robot' in title, got: {title}"

    perc = _bridge_ok("perceive", "--tier", "cdp", timeout=60)
    elements = perc.get("elements", [])
    assert len(elements) > 0, f"No elements perceived: {perc}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Tab lifecycle (open → list → switch → info → close)
# ─────────────────────────────────────────────────────────────────────────────
def test_tab_lifecycle():
    """Open a tab, verify it appears in list, switch to it, get info, close it."""
    # tab-list returns data as a list of tab dicts
    initial = _bridge_ok("tab-list")
    initial_tabs = initial if isinstance(initial, list) else initial.get("tabs", [])
    initial_count = len(initial_tabs)

    # Open new tab
    opened = _bridge_ok("tab-open", "https://en.wikipedia.org/wiki/Computer")
    new_tab_id = opened.get("tab_id", "")
    assert new_tab_id, f"No tab_id returned: {opened}"

    _bridge_ok("wait", "2")

    # List should now have one more
    after_open = _bridge_ok("tab-list")
    after_tabs = after_open if isinstance(after_open, list) else after_open.get("tabs", [])
    after_count = len(after_tabs)
    assert after_count == initial_count + 1, f"Tab count {after_count} != {initial_count + 1}"

    # Switch to the new tab (use first 8 chars of ID)
    switched = _bridge_ok("tab-switch", new_tab_id[:8])
    assert switched.get("switched"), f"Switch failed: {switched}"

    # Get tab info — title should contain "Computer"
    info = _bridge_ok("tab-info")
    assert "Computer" in str(info.get("title", "")), f"Title mismatch: {info}"

    # Close the tab
    closed = _bridge_ok("tab-close", new_tab_id[:8])
    assert closed.get("closed"), f"Close failed: {closed}"

    # Verify count back to original
    after_close = _bridge_ok("tab-list")
    final_tabs = after_close if isinstance(after_close, list) else after_close.get("tabs", [])
    final_count = len(final_tabs)
    assert final_count == initial_count, f"Final count {final_count} != {initial_count}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Scroll-page + verify position changed
# ─────────────────────────────────────────────────────────────────────────────
def test_scroll_page_verification():
    """Scroll down, verify scrollY changed, scroll back to top."""
    # Ensure we're on a page with scrollable content
    _bridge_ok("navigate", "https://en.wikipedia.org/wiki/Robot")
    _bridge_ok("wait", "2")

    info_before = _bridge_ok("tab-info")
    scroll_before = info_before.get("scrollY", 0)

    # Scroll down 500px
    _bridge_ok("scroll-page", "down", "500")
    _bridge_ok("wait", "1")

    info_after = _bridge_ok("tab-info")
    scroll_after = info_after.get("scrollY", 0)
    assert scroll_after > scroll_before, f"Scroll didn't increase: {scroll_before} → {scroll_after}"

    # Scroll back to top
    _bridge_ok("scroll-page", "top")
    _bridge_ok("wait", "1")

    info_top = _bridge_ok("tab-info")
    assert info_top.get("scrollY", -1) == 0, f"Didn't return to top: {info_top}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Clipboard round-trip
# ─────────────────────────────────────────────────────────────────────────────
def test_clipboard_roundtrip():
    """Set clipboard, get clipboard, verify content matches."""
    # Skip if no working clipboard (headless / no display)
    probe = subprocess.run(
        ["xclip", "-selection", "clipboard", "-o"],
        capture_output=True,
        text=True,
        timeout=3,
    )
    if probe.returncode != 0:
        pytest.skip("No clipboard available (xclip cannot open display)")
    test_text = f"norax_integration_test_{int(time.time())}"

    set_result = _bridge_ok("clipboard-set", test_text)
    assert set_result.get("ok") is True

    get_result = _bridge_ok("clipboard-get")
    clip_text = get_result.get("text", "")
    assert test_text in str(clip_text), (
        f"Clipboard mismatch: expected '{test_text}', got '{clip_text}'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Screenshot + map cascade
# ─────────────────────────────────────────────────────────────────────────────
def test_screenshot_and_map():
    """Take screenshot, run map on it, verify elements returned."""
    ss = _bridge_ok("screenshot")
    assert "path" in ss, f"No screenshot path: {ss}"
    assert ss.get("ok") is True, f"Screenshot failed: {ss}"

    mapped = _bridge_ok("map", timeout=60)
    elements = mapped.get("elements", [])
    assert len(elements) > 0, f"No elements mapped: {mapped}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: JS execution + state verification
# ─────────────────────────────────────────────────────────────────────────────
def test_js_state_verification():
    """Execute JS that modifies page state, verify it persisted."""
    _bridge_ok("js", "window.__norax_test = 'integration_42'")

    result = _bridge_ok("js", "window.__norax_test")
    value = result.get("value", "")
    assert "integration_42" in str(value), f"JS state not persisted: {value}"

    # Clean up
    _bridge_ok("js", "delete window.__norax_test")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Desktop actions chain (click → move → drag)
# ─────────────────────────────────────────────────────────────────────────────
def test_desktop_action_chain():
    """Chain click, move, and drag — all should succeed."""
    click_r = _bridge_ok("click", "500", "300")
    assert click_r.get("ok") is True

    move_r = _bridge_ok("move", "100", "100")
    assert move_r.get("ok") is True

    drag_r = _bridge_ok("drag", "100", "200", "300", "400")
    assert drag_r.get("ok") is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Scroll direction fix (the bug we just fixed)
# ─────────────────────────────────────────────────────────────────────────────
def test_scroll_direction_words():
    """Scroll with 'down' and 'up' direction words should work after fix."""
    r_down = _bridge("scroll", "down", "3")
    assert r_down.get("ok"), f"scroll down failed: {r_down}"

    r_up = _bridge("scroll", "up", "3")
    assert r_up.get("ok"), f"scroll up failed: {r_up}"

    r_num = _bridge("scroll", "3")
    assert r_num.get("ok"), f"scroll numeric failed: {r_num}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Full workflow — navigate + search + verify
# ─────────────────────────────────────────────────────────────────────────────
def test_wikipedia_search_workflow():
    """Navigate to Wikipedia, type in search, verify results appear."""
    _bridge_ok("navigate", "https://en.wikipedia.org/wiki/Main_Page")
    _bridge_ok("wait", "2")

    info = _bridge_ok("tab-info")
    assert "Wikipedia" in str(info.get("title", "")), f"Not on Wikipedia: {info}"

    # Wikipedia's Vector 2022 search box is a collapsed typeahead (0x0) until
    # the search toggle is clicked — expand it first, as a real user would.
    _bridge_ok("js", "document.querySelector('#p-search a.search-toggle')?.click(); 'expanded'")
    _bridge_ok("wait", "1")

    # Type into search box via CDP
    type_r = _bridge_ok("type", "artificial intelligence")
    assert type_r.get("ok") is True

    # Press Enter to search
    key_r = _bridge_ok("key", "Return")
    assert key_r.get("ok") is True

    _bridge_ok("wait", "3")

    # Verify we navigated to search results or article
    info_after = _bridge_ok("tab-info")
    url_after = info_after.get("url", "")
    title_after = str(info_after.get("title", ""))
    assert "Artificial" in title_after or "search" in url_after.lower(), (
        f"Search didn't navigate: {info_after}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: Info command returns system capabilities
# ─────────────────────────────────────────────────────────────────────────────
def test_info_returns_capabilities():
    """Info should return display, xdotool, scrot, xclip status."""
    info = _bridge_ok("info")
    assert "display" in info or ":99" in str(info), f"No display in info: {info}"
    assert info.get("xdotool") is True or info.get("ydotool") is True, f"No input tool: {info}"
