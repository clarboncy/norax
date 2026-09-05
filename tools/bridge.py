#!/usr/bin/env python3
"""Single JSON CLI for Norax's standalone desktop and CDP helpers.

This adapter intentionally contains no second implementation of CDP or input
semantics. Desktop actions delegate to ``computer_use.py``; browser actions
delegate to ``browser_controller.py``. An observed inner failure is propagated
to the process exit status and top-level ``ok`` field.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import urlsplit

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if not os.environ.get("NORAX_WORKSPACE", "").strip():
    os.environ["NORAX_WORKSPACE"] = str(TOOLS.parent)


def _ok(data: Any = None) -> NoReturn:
    failed = isinstance(data, dict) and (
        ("ok" in data and data.get("ok") is not True)
        or ("success" in data and data.get("success") is not True)
        or bool(data.get("error"))
    )
    envelope: dict[str, Any] = {"ok": not failed, "data": data}
    if failed and isinstance(data, dict) and data.get("error"):
        envelope["error"] = data["error"]
    print(json.dumps(envelope, default=str))
    raise SystemExit(1 if failed else 0)


def _err(message: str) -> NoReturn:
    print(json.dumps({"ok": False, "error": message}, default=str))
    raise SystemExit(1)


def _content_tabs() -> list[dict[str, Any]]:
    import browser_controller as browser

    return [
        tab
        for tab in browser.list_tabs()
        if "omnibox" not in str(tab.get("url", ""))
        and not str(tab.get("url", "")).startswith("chrome://")
    ]


def _select_tab(selector: str | None = None, *, require_focused: bool = False) -> dict[str, Any]:
    import browser_controller as browser

    tabs = _content_tabs()
    if not tabs:
        raise RuntimeError("No content browser tabs found")
    if selector is not None:
        try:
            index = int(selector)
        except ValueError:
            tab = next(
                (
                    candidate
                    for candidate in tabs
                    if str(candidate.get("id", "")).startswith(selector)
                ),
                None,
            )
            if tab is None:
                raise RuntimeError(f"Tab not found: {selector}") from None
            return tab
        if not 0 <= index < len(tabs):
            raise RuntimeError(f"Tab index out of range: {index}")
        return tabs[index]

    # Prefer the page that reports focus. If focus cannot be queried, the
    # first content tab is a deterministic fallback rather than a random tab.
    for tab in tabs:
        ws_url = tab.get("webSocketDebuggerUrl")
        if not isinstance(ws_url, str):
            continue
        client = None
        try:
            client = browser.CDPClient(ws_url)
            if client.eval_value("document.hasFocus()") is True:
                return tab
        except Exception:
            continue
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
    if require_focused:
        raise RuntimeError("No focused content browser tab found")
    return tabs[0]


@contextmanager
def _browser_client(
    selector: str | None = None,
    *,
    require_focused: bool = False,
    selected_tab: dict[str, Any] | None = None,
) -> Iterator[tuple[Any, dict[str, Any]]]:
    import browser_controller as browser

    tab = selected_tab or _select_tab(selector, require_focused=require_focused)
    ws_url = tab.get("webSocketDebuggerUrl")
    if not isinstance(ws_url, str):
        raise RuntimeError("Selected tab has no WebSocket debugger URL")
    client = browser.CDPClient(ws_url)
    try:
        client.enable()
        yield client, tab
    finally:
        client.close()


def _browser_fill(text: str, selected_tab: dict[str, Any] | None = None) -> dict[str, Any]:
    with _browser_client(require_focused=True, selected_tab=selected_tab) as (client, tab):
        result = client.fill_active(text)
    return {**result, "mode": "cdp", "tab_title": tab.get("title", "")}


def _browser_key(key: str, selected_tab: dict[str, Any] | None = None) -> dict[str, Any]:
    with _browser_client(require_focused=True, selected_tab=selected_tab) as (client, tab):
        result = client.press_key(key)
    return {**result, "mode": "cdp", "tab_title": tab.get("title", "")}


def _desktop_command(command: str, args: list[str]) -> dict[str, Any]:
    import computer_use as computer

    if command == "screenshot":
        return computer.screenshot(args[0] if args else "/tmp/norax_screen.png")
    if command in {"click", "doubleclick", "move"}:
        if len(args) < 2:
            raise ValueError(f"{command} requires <x> <y>")
        x, y = int(args[0]), int(args[1])
        if command == "click":
            button_map = {"left": 1, "right": 2, "middle": 3, "1": 1, "2": 2, "3": 3}
            button_name = args[2] if len(args) > 2 else "left"
            if button_name not in button_map:
                raise ValueError(f"unsupported mouse button: {button_name}")
            return computer.click(x, y, button_map[button_name])
        if command == "doubleclick":
            return computer.doubleclick(x, y)
        return computer.move(x, y)
    if command == "drag":
        if len(args) < 4:
            raise ValueError("drag requires <x1> <y1> <x2> <y2>")
        return computer.drag(*(int(value) for value in args[:4]))
    if command == "scroll":
        if not args:
            raise ValueError("scroll requires <amount|up|down> [amount]")
        if args[0] in {"up", "down"}:
            magnitude = abs(int(args[1])) if len(args) > 1 else 3
            amount = magnitude if args[0] == "down" else -magnitude
        else:
            amount = int(args[0])
        return computer.scroll(amount)
    if command == "clipboard-get":
        return computer.clipboard_get()
    if command == "clipboard-set":
        if not args:
            raise ValueError("clipboard-set requires <text>")
        return computer.clipboard_set(" ".join(args))
    raise ValueError(f"Unsupported desktop command: {command}")


def _perceive(args: list[str]) -> dict[str, Any]:
    import perception

    tier: str | None = None
    if args:
        if args[0] == "--tier":
            if len(args) < 2:
                raise ValueError("--tier requires atspi, cdp, or ocr")
            tier = args[1]
        elif args[0].startswith("--tier="):
            tier = args[0].split("=", 1)[1]
        else:
            tier = args[0]
    if tier is not None and tier not in {"atspi", "cdp", "ocr"}:
        raise ValueError(f"unsupported perception tier: {tier}")
    state = perception.perceive(tier=tier)
    result = state.to_dict()
    if state.error:
        result["ok"] = False
    return result


def _map_desktop() -> dict[str, Any]:
    import dmap

    image = dmap.capture()
    if image is None:
        raise RuntimeError("no non-interactive screenshot backend succeeded")
    index = dmap.build_index(image)
    dmap.save_index(index)
    if not index["elements"] and _content_tabs():
        import perception

        state = perception.perceive(tier="cdp", save=False)
        if state.elements:
            elements = state.to_dict()["elements"]
            return {
                "elements": elements,
                "counts": {
                    "total": len(elements),
                    "text": sum(not element.get("actionable", False) for element in elements),
                    "buttons": sum(element.get("actionable", False) for element in elements),
                    "structure": 0,
                },
                "screen": index["screen"],
                "timing": index["timing"],
                "tier_used": "cdp",
            }
    return {
        "elements": index["elements"],
        "counts": index["counts"],
        "screen": index["screen"],
        "timing": index["timing"],
        "tier_used": "cv",
    }


def _act(args: list[str]) -> dict[str, Any]:
    import motor_cortex as motor

    if len(args) < 2:
        raise ValueError("act requires <action_type> <target>")
    action_map = {
        "type": "type_text",
        "key": "key_press",
        "click": "click_button",
        "click_button": "click_button",
        "click_link": "click_link",
        "type_text": "type_text",
        "key_press": "key_press",
        "scroll": "scroll",
        "exec": "exec_command",
        "exec_command": "exec_command",
        "file_write": "file_write",
        "file_read": "file_read",
    }
    action_type = action_map.get(args[0])
    if action_type is None:
        raise ValueError(f"unsupported motor action: {args[0]}")
    target = args[1] if len(args) == 2 else shlex.join(args[1:])
    result = motor.execute(action_type, target)
    return {
        "success": result.success,
        "attempts": result.attempts,
        "verification": result.verification,
        "verification_level": result.verification_level,
        "duration_ms": result.duration_ms,
        "corrections": result.corrections,
    }


def _tab_list() -> list[dict[str, Any]]:
    return [
        {
            "id": tab.get("id", ""),
            "title": tab.get("title", ""),
            "url": tab.get("url", ""),
            "type": tab.get("type", ""),
        }
        for tab in _content_tabs()
    ]


def _tab_info(selector: str | None = None) -> dict[str, Any]:
    with _browser_client(selector) as (client, tab):
        result = client.eval(
            "({url: location.href, title: document.title, scrollY: window.scrollY, "
            "scrollHeight: document.body?.scrollHeight || 0, "
            "viewport: {w: window.innerWidth, h: window.innerHeight}})"
        )
    if "error" in result:
        return {"ok": False, "error": result["error"]}
    value = result.get("value")
    if not isinstance(value, dict):
        return {"ok": False, "error": "tab-info returned no structured value"}
    return {"tab_id": tab.get("id", ""), **value}


def _navigate(url: str) -> dict[str, Any]:
    _validate_navigation_url(url)
    with _browser_client(require_focused=True) as (client, _tab):
        result = client.navigate(url)
    return {
        **result,
        "navigated": result.get("ok") is True,
        "loaded": result.get("readyState") in {"interactive", "complete"},
    }


def _scroll_page(direction: str, raw_amount: str) -> dict[str, Any]:
    if direction not in {"up", "down", "top", "bottom", "up-all", "down-all"}:
        raise ValueError(f"unsupported scroll direction: {direction}")
    amount = int(raw_amount)
    if not 0 <= amount <= 100_000:
        raise ValueError("scroll amount must be between 0 and 100000")
    if direction in {"top", "up-all"}:
        expression = "window.scrollTo(0, 0); window.scrollY"
    elif direction in {"bottom", "down-all"}:
        expression = "window.scrollTo(0, document.body.scrollHeight); window.scrollY"
    else:
        delta = -amount if direction == "up" else amount
        expression = f"window.scrollBy(0, {delta}); window.scrollY"
    with _browser_client(require_focused=True) as (client, _tab):
        result = client.eval(expression)
    if "error" in result:
        return {"ok": False, "error": result["error"]}
    return {
        "scrolled": direction,
        "amount": amount,
        "observed_scroll_y": result.get("value"),
    }


def _validate_navigation_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise ValueError(f"invalid navigation URL: {exc}") from exc
    if url == "about:blank":
        return
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("navigation URL must use http or https (or be about:blank)")


def main() -> None:
    if len(sys.argv) < 2:
        _err("No command. Usage: bridge.py <command> [args...]")
    command = sys.argv[1]
    args = sys.argv[2:]

    try:
        if command in {
            "screenshot",
            "click",
            "doubleclick",
            "move",
            "drag",
            "scroll",
            "clipboard-get",
            "clipboard-set",
        }:
            _ok(_desktop_command(command, args))
        if command == "type":
            if not args:
                _err("type requires <text>")
            text = " ".join(args)
            try:
                focused_tab = _select_tab(require_focused=True)
            except RuntimeError:
                focused_tab = None
            _ok(_browser_fill(text, focused_tab) if focused_tab else _desktop_type(text))
        if command == "key":
            if not args:
                _err("key requires <keyname>")
            try:
                focused_tab = _select_tab(require_focused=True)
            except RuntimeError:
                focused_tab = None
            _ok(_browser_key(args[0], focused_tab) if focused_tab else _desktop_key(args[0]))
        if command == "perceive":
            _ok(_perceive(args))
        if command == "map":
            _ok(_map_desktop())
        if command == "act":
            _ok(_act(args))
        if command in {"cdp_type", "cdp-type"}:
            if not args:
                _err("cdp_type requires <text>")
            _ok(_browser_fill(" ".join(args)))
        if command in {"cdp_key", "cdp-key"}:
            if not args:
                _err("cdp_key requires <keyname>")
            _ok(_browser_key(args[0]))
        if command in {"cdp_submit", "cdp-submit"}:
            with _browser_client(require_focused=True) as (client, tab):
                result = client.eval(
                    "(() => { const el = document.activeElement; const form = el?.closest('form'); "
                    "if (!form) return 'no_active_form'; form.requestSubmit(); return 'request_submitted'; })()"
                )
            submitted = result.get("value") == "request_submitted" and "error" not in result
            _ok(
                {
                    "ok": submitted,
                    "result": result.get("value"),
                    "error": result.get("error") or (None if submitted else "No active form"),
                    "tab_title": tab.get("title", ""),
                }
            )
        if command == "navigate":
            if not args:
                _err("navigate requires <url>")
            _ok(_navigate(args[0]))
        if command == "tab-list":
            _ok(_tab_list())
        if command == "tab-switch":
            if not args:
                _err("tab-switch requires <index|id_prefix>")
            tab = _select_tab(args[0])
            with _browser_client(args[0]) as (client, _selected):
                client.send("Page.bringToFront")
            _ok(
                {
                    "switched": True,
                    "tab_id": tab.get("id", ""),
                    "title": tab.get("title", ""),
                    "url": tab.get("url", ""),
                }
            )
        if command == "tab-open":
            import browser_controller as browser

            new_url = args[0] if args else "about:blank"
            _validate_navigation_url(new_url)
            opened = browser.new_tab(new_url)
            if opened.get("ok") is not True:
                _ok(opened)
            tab = opened.get("tab", {})
            if not isinstance(tab, dict):
                _err("new-tab returned an invalid tab object")
            _ok(
                {
                    "opened": True,
                    "tab_id": tab.get("id", ""),
                    "url": tab.get("url", args[0] if args else "about:blank"),
                }
            )
        if command == "tab-close":
            import browser_controller as browser

            tab = _select_tab(args[0] if args else "0")
            tab_id = tab.get("id")
            if not isinstance(tab_id, str):
                _err("selected tab has no string ID")
            closed = browser.close_tab(tab_id)
            _ok(
                {
                    **closed,
                    "closed": closed.get("ok") is True,
                    "tab_id": tab_id,
                    "title": tab.get("title", ""),
                }
            )
        if command == "tab-info":
            _ok(_tab_info(args[0] if args else None))
        if command == "js":
            if not args:
                _err("js requires <javascript expression>")
            with _browser_client(require_focused=True) as (client, tab):
                result = client.eval(" ".join(args))
            _ok(
                {
                    "ok": "error" not in result,
                    "error": result.get("error"),
                    "value": result.get("value"),
                    "type": result.get("type"),
                    "tab_id": tab.get("id", ""),
                    "tab_title": tab.get("title", ""),
                }
            )
        if command == "wait":
            seconds = float(args[0]) if args else 2.0
            if not 0 <= seconds <= 30:
                raise ValueError("wait must be between 0 and 30 seconds")
            time.sleep(seconds)
            _ok({"waited": seconds})
        if command == "scroll-page":
            _ok(_scroll_page(args[0] if args else "down", args[1] if len(args) > 1 else "600"))
        if command == "info":
            import computer_use as computer

            _ok(computer.info())
        _err(f"Unknown command: {command}")
    except SystemExit:
        raise
    except Exception as exc:
        _err(f"{type(exc).__name__}: {exc}")


def _desktop_type(text: str) -> dict[str, Any]:
    import computer_use as computer

    return computer.type_text(text)


def _desktop_key(key: str) -> dict[str, Any]:
    import computer_use as computer

    return computer.key_press(key)


if __name__ == "__main__":
    main()
