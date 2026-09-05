#!/usr/bin/env python3
"""
browser_controller.py — High-level browser automation via CDP.

Computer use: navigate, read, find elements, click by text,
fill forms, wait for loads, manage tabs — all through Chrome DevTools Protocol.

Usage:
  python3 browser_controller.py navigate <url>
  python3 browser_controller.py read
  python3 browser_controller.py elements
  python3 browser_controller.py click <text>
  python3 browser_controller.py click-xy <x> <y>
  python3 browser_controller.py type <text>
  python3 browser_controller.py fill <selector> <text>
  python3 browser_controller.py key <keyname>
  python3 browser_controller.py scroll <amount>
  python3 browser_controller.py wait <seconds>
  python3 browser_controller.py tabs
  python3 browser_controller.py new-tab <url>
  python3 browser_controller.py switch-tab <index>
  python3 browser_controller.py close-tab
  python3 browser_controller.py url
  python3 browser_controller.py title
  python3 browser_controller.py eval <js>
  python3 browser_controller.py screenshot [path]
  python3 browser_controller.py fill-form <json_file>
  python3 browser_controller.py click-element <selector>
  python3 browser_controller.py wait-for <text>
  python3 browser_controller.py back
  python3 browser_controller.py forward
  python3 browser_controller.py reload
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import quote, urlsplit

websocket: Any
try:
    import websocket as websocket_module

    websocket = websocket_module
except ImportError:
    websocket = None

CDP_HOST = "127.0.0.1"
CDP_PORT = 9222
MSG_ID = 100  # Start high to avoid collision with drained events


def _navigation_error(url: str) -> str | None:
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        return f"invalid URL: {exc}"
    if url == "about:blank":
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return "URL must use http or https (or be about:blank)"
    return None


class CDPClient:
    """Low-level CDP client that ignores events while awaiting its response."""

    def __init__(self, ws_url: str):
        if websocket is None:
            raise RuntimeError("websocket-client is required for browser_controller")
        self.ws = websocket.create_connection(ws_url, timeout=15)
        self._id = MSG_ID
        self._enabled = False

    def _next_id(self):
        self._id += 1
        return self._id

    def enable(self):
        """Enable Runtime and Page domains."""
        self.send("Runtime.enable")
        self.send("Page.enable")
        self._enabled = True

    def send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send CDP command, return result (draining events)."""
        msg_id = self._next_id()
        msg: dict[str, Any] = {"id": msg_id, "method": method}
        if params:
            msg["params"] = params
        self.ws.send(json.dumps(msg))

        # Read until we get our response
        while True:
            raw = self.ws.recv()
            resp = json.loads(raw)
            if not isinstance(resp, dict):
                continue
            if resp.get("id") == msg_id:
                if "error" in resp:
                    raise RuntimeError(f"CDP {method} failed: {resp['error']}")
                return resp
            # Event or other response — ignore

    def eval(self, expression: str, await_promise: bool = False) -> dict[str, Any]:
        """Evaluate JS expression, return result value."""
        params = {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": await_promise,
            "userGesture": True,
        }
        resp = self.send("Runtime.evaluate", params)
        payload = resp.get("result", {})
        if payload.get("exceptionDetails"):
            details = payload["exceptionDetails"]
            return {
                "error": details.get("text", "JavaScript evaluation failed"),
                "exceptionDetails": details,
            }
        result = payload.get("result", {})
        if result.get("type") == "undefined":
            return {"value": None}
        return result

    def eval_value(self, expression: str, await_promise: bool = False):
        """Eval and return just the value."""
        result = self.eval(expression, await_promise)
        if "error" in result:
            raise RuntimeError(f"JavaScript evaluation failed: {result['error']}")
        return result.get("value")

    def navigate(self, url: str, wait: float = 10.0) -> dict:
        """Navigate to URL and poll for DOM readiness up to ``wait`` seconds."""
        invalid = _navigation_error(url)
        if invalid:
            return {"ok": False, "url": url, "error": invalid}
        response = self.send("Page.navigate", {"url": url})
        navigation = response.get("result", {})
        if navigation.get("errorText"):
            return {"ok": False, "url": url, "error": navigation["errorText"]}
        deadline = time.monotonic() + max(0.0, wait)
        ready = None
        while time.monotonic() <= deadline:
            ready = self.eval_value("document.readyState")
            if ready in {"interactive", "complete"}:
                return {"ok": True, "url": url, "readyState": ready}
            time.sleep(0.1)
        return {
            "ok": False,
            "url": url,
            "readyState": ready,
            "error": f"document did not become ready within {wait:g}s",
        }

    def get_text(self) -> str:
        """Get visible page text."""
        return self.eval_value("document.body.innerText") or ""

    def get_title(self) -> str:
        return self.eval_value("document.title") or ""

    def get_url(self) -> str:
        return self.eval_value("window.location.href") or ""

    def get_elements(self) -> list[dict[str, Any]]:
        """Get all interactive elements with positions."""
        js = """(() => {
            const elements = [];
            const selectors = 'a, button, input, select, textarea, [role=button], [role=link], [role=checkbox], [role=tab], [contenteditable], [draggable]';
            const all = document.querySelectorAll(selectors);
            for (const el of all) {
                const rect = el.getBoundingClientRect();
                if (rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.right > 0) {
                    const text = (el.innerText || el.value || el.placeholder || el.getAttribute('aria-label') || el.getAttribute('title') || '').trim();
                    elements.push({
                        tag: el.tagName.toLowerCase(),
                        text: text.slice(0, 120),
                        href: el.href || '',
                        x: Math.round(rect.x + rect.width/2),
                        y: Math.round(rect.y + rect.height/2),
                        w: Math.round(rect.width),
                        h: Math.round(rect.height),
                        type: el.type || '',
                        id: el.id || '',
                        name: el.name || '',
                        className: (el.className || '').toString().slice(0, 80),
                        placeholder: el.placeholder || '',
                        value: el.value || '',
                        checked: el.checked || false,
                        disabled: el.disabled || false,
                        role: el.getAttribute('role') || '',
                        selector: _getSelector(el)
                    });
                }
            }
            function _getSelector(el) {
                if (el.id) return '#' + el.id;
                let path = [];
                let node = el;
                while (node && node.nodeType === 1) {
                    let part = node.tagName.toLowerCase();
                    if (node.id) { part = '#' + node.id; path.unshift(part); break; }
                    let sib = node, nth = 1;
                    while (sib = sib.previousElementSibling) {
                        if (sib.tagName === node.tagName) nth++;
                    }
                    if (nth > 1) part += ':nth-of-type(' + nth + ')';
                    path.unshift(part);
                    node = node.parentElement;
                }
                return path.join(' > ');
            }
            return JSON.stringify(elements);
        })()"""
        result = self.eval(js)
        val = result.get("value")
        if isinstance(val, str) and val:
            decoded = json.loads(val)
            if isinstance(decoded, list):
                return [element for element in decoded if isinstance(element, dict)]
        return []

    def find_element_by_text(self, text: str) -> dict[str, Any] | None:
        """Find element whose text matches (case-insensitive partial match)."""
        if not text.strip():
            return None
        elements = self.get_elements()
        text_lower = text.lower()
        # Exact match first
        for el in elements:
            if el["text"].lower() == text_lower:
                return el
        # Partial match
        for el in elements:
            if text_lower in el["text"].lower():
                return el
        # Match on placeholder
        for el in elements:
            if text_lower in el.get("placeholder", "").lower():
                return el
        # Match on aria-label
        for el in elements:
            if text_lower in el.get("role", "").lower():
                return el
        return None

    def click_by_text(self, text: str) -> dict:
        """Find element by text and click via CDP Input.dispatchMouseEvent."""
        el = self.find_element_by_text(text)
        if not el:
            return {"ok": False, "error": f"No element found matching '{text}'"}
        return self.click_xy(el["x"], el["y"])

    def click_xy(self, x: int, y: int) -> dict:
        """Click at coordinates via CDP."""
        for evt_type in ["mouseMoved", "mousePressed", "mouseReleased"]:
            params = {
                "type": evt_type,
                "x": x,
                "y": y,
                "button": "left",
                "clickCount": 1,
            }
            self.send("Input.dispatchMouseEvent", params)
        return {
            "ok": True,
            "x": x,
            "y": y,
            "verification": "dispatch_accepted",
        }

    def click_selector(self, selector: str) -> dict:
        """Click element by CSS selector via JS."""
        js = f"""(() => {{
            const el = document.querySelector({json.dumps(selector)});
            if (!el) return JSON.stringify({{ok: false, error: 'Element not found'}});
            el.click();
            return JSON.stringify({{ok: true, tag: el.tagName.toLowerCase()}});
        }})()"""
        result = self.eval(js)
        return self._json_eval_result(result)

    def type_text(self, text: str) -> dict:
        """Insert text into the currently focused element."""
        self.send("Input.insertText", {"text": text})
        return {"ok": True, "chars": len(text), "verification": "dispatch_accepted"}

    def fill_active(self, text: str) -> dict[str, Any]:
        """Fill the focused or first visible editable element and read it back."""
        js = f"""(() => {{
            const textTypes = ['text','search','email','url','password','number','tel',''];
            const editable = Array.from(document.querySelectorAll(
              'input, textarea, [contenteditable=true]'
            )).filter(candidate => {{
              if (candidate.type === 'hidden' || candidate.disabled) return false;
              if (candidate.tagName === 'INPUT' && !textTypes.includes(candidate.type)) return false;
              return true;
            }});
            const el = (document.activeElement && editable.includes(document.activeElement))
              ? document.activeElement
              : editable.find(candidate => candidate.offsetParent !== null)
                || editable[0];
            if (!el) return JSON.stringify({{ok: false, error: 'No visible editable element'}});
            el.focus();
            const value = {json.dumps(text)};
            if (el.isContentEditable) {{
              el.textContent = value;
            }} else {{
              const proto = el.tagName === 'TEXTAREA'
                ? window.HTMLTextAreaElement.prototype
                : window.HTMLInputElement.prototype;
              const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
              if (setter) setter.call(el, value); else el.value = value;
            }}
            el.dispatchEvent(new InputEvent('input', {{bubbles: true, inputType: 'insertText', data: value}}));
            el.dispatchEvent(new Event('change', {{bubbles: true}}));
            const observed = el.isContentEditable ? el.textContent : el.value;
            return JSON.stringify({{
              ok: observed === value,
              chars: value.length,
              observed_chars: (observed || '').length,
              tag: el.tagName.toLowerCase(),
              error: observed === value ? null : 'Read-back did not match requested text'
            }});
        }})()"""
        return self._json_eval_result(self.eval(js))

    def fill_field(self, selector: str, value: str) -> dict:
        """Fill an input field by selector and verify its observed value."""
        js = f"""(() => {{
            const el = document.querySelector({json.dumps(selector)});
            if (!el) return JSON.stringify({{ok: false, error: 'Element not found'}});
            if (!(el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement || el.isContentEditable)) {{
              return JSON.stringify({{ok: false, error: 'Element is not editable'}});
            }}
            el.focus();
            const value = {json.dumps(value)};
            if (el.isContentEditable) {{
              el.textContent = value;
            }} else {{
              const proto = el.tagName === 'TEXTAREA'
                ? window.HTMLTextAreaElement.prototype
                : window.HTMLInputElement.prototype;
              const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
              if (setter) setter.call(el, value); else el.value = value;
            }}
            el.dispatchEvent(new InputEvent('input', {{bubbles: true, inputType: 'insertText', data: value}}));
            el.dispatchEvent(new Event('change', {{bubbles: true}}));
            const observed = el.isContentEditable ? el.textContent : el.value;
            return JSON.stringify({{
              ok: observed === value,
              chars: value.length,
              observed_chars: (observed || '').length,
              tag: el.tagName.toLowerCase(),
              error: observed === value ? null : 'Read-back did not match requested text'
            }});
        }})()"""
        result = self.eval(js)
        return self._json_eval_result(result)

    @staticmethod
    def _json_eval_result(result: dict[str, Any]) -> dict[str, Any]:
        if "error" in result:
            return {"ok": False, "error": result["error"]}
        value = result.get("value")
        if not isinstance(value, str):
            return {"ok": False, "error": "JavaScript action returned no structured result"}
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"JavaScript action returned invalid JSON: {exc}"}
        if not isinstance(decoded, dict):
            return {"ok": False, "error": "JavaScript action returned a non-object result"}
        return decoded

    def press_key(self, key: str) -> dict:
        """Press a key via CDP."""
        parts = [part.strip() for part in key.split("+") if part.strip()]
        if not parts:
            return {"ok": False, "error": "key is required"}
        modifier_bits = {"alt": 1, "ctrl": 2, "control": 2, "meta": 4, "shift": 8}
        modifiers = 0
        for modifier in parts[:-1]:
            bit = modifier_bits.get(modifier.lower())
            if bit is None:
                return {"ok": False, "error": f"Unsupported modifier: {modifier}"}
            modifiers |= bit

        aliases = {
            "Return": "Enter",
            "BackSpace": "Backspace",
            "space": "Space",
            "Up": "ArrowUp",
            "Down": "ArrowDown",
            "Left": "ArrowLeft",
            "Right": "ArrowRight",
            "Page_Up": "PageUp",
            "Page_Down": "PageDown",
        }
        main_key = aliases.get(parts[-1], parts[-1])
        key_map: dict[str, dict[str, Any]] = {
            "Enter": {"key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "text": "\r"},
            "Tab": {"key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9},
            "Escape": {"key": "Escape", "code": "Escape", "windowsVirtualKeyCode": 27},
            "Backspace": {"key": "Backspace", "code": "Backspace", "windowsVirtualKeyCode": 8},
            "Delete": {"key": "Delete", "code": "Delete", "windowsVirtualKeyCode": 46},
            "Space": {"key": " ", "code": "Space", "windowsVirtualKeyCode": 32},
            "ArrowDown": {"key": "ArrowDown", "code": "ArrowDown", "windowsVirtualKeyCode": 40},
            "ArrowUp": {"key": "ArrowUp", "code": "ArrowUp", "windowsVirtualKeyCode": 38},
            "ArrowLeft": {"key": "ArrowLeft", "code": "ArrowLeft", "windowsVirtualKeyCode": 37},
            "ArrowRight": {"key": "ArrowRight", "code": "ArrowRight", "windowsVirtualKeyCode": 39},
            "PageUp": {"key": "PageUp", "code": "PageUp", "windowsVirtualKeyCode": 33},
            "PageDown": {"key": "PageDown", "code": "PageDown", "windowsVirtualKeyCode": 34},
            "Home": {"key": "Home", "code": "Home", "windowsVirtualKeyCode": 36},
            "End": {"key": "End", "code": "End", "windowsVirtualKeyCode": 35},
        }
        if len(main_key) == 1:
            key_params: dict[str, Any] = {
                "key": main_key,
                "code": f"Key{main_key.upper()}" if main_key.isalpha() else main_key,
                "windowsVirtualKeyCode": ord(main_key.upper()),
            }
            if modifiers & (1 | 2 | 4) == 0:
                key_params["text"] = main_key
        elif main_key in key_map:
            key_params = key_map[main_key].copy()
        elif main_key.startswith("F") and main_key[1:].isdigit():
            key_params = {"key": main_key, "code": main_key}
        else:
            return {"ok": False, "error": f"Unsupported key: {main_key}"}
        key_params["modifiers"] = modifiers
        for evt_type in ["keyDown", "keyUp"]:
            params: dict[str, Any] = {"type": evt_type}
            params.update(key_params)
            self.send("Input.dispatchKeyEvent", params)
        return {"ok": True, "key": key, "modifiers": modifiers, "verification": "dispatch_accepted"}

    def scroll(self, amount: int) -> dict:
        """Scroll page by amount (positive=down, negative=up)."""
        js = f"window.scrollBy(0, {amount})"
        result = self.eval(js)
        if "error" in result:
            return {"ok": False, "amount": amount, "error": result["error"]}
        return {"ok": True, "amount": amount, "verification": "dispatch_accepted"}

    def wait_for_text(self, text: str, timeout: float = 10.0) -> dict:
        """Wait until text appears on page."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            page_text = self.get_text()
            if text.lower() in page_text.lower():
                return {
                    "ok": True,
                    "found": True,
                    "elapsed": round(time.monotonic() - start, 2),
                }
            time.sleep(0.5)
        return {"ok": False, "found": False, "timeout": timeout}

    def go_back(self) -> dict:
        result = self.eval("history.back()")
        return {
            "ok": "error" not in result,
            "completion_verified": False,
            **({"error": result["error"]} if "error" in result else {}),
        }

    def go_forward(self) -> dict:
        result = self.eval("history.forward()")
        return {
            "ok": "error" not in result,
            "completion_verified": False,
            **({"error": result["error"]} if "error" in result else {}),
        }

    def reload(self) -> dict:
        result = self.eval("location.reload()")
        return {
            "ok": "error" not in result,
            "completion_verified": False,
            **({"error": result["error"]} if "error" in result else {}),
        }

    def screenshot(self, path: str = "/tmp/norax_cdp_screenshot.png") -> dict:
        """Capture screenshot via CDP."""
        resp = self.send("Page.captureScreenshot", {"format": "png"})
        data = resp.get("result", {}).get("data")
        if data:
            import base64

            decoded = base64.b64decode(data, validate=True)
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            candidate = destination.with_name(
                f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
            )
            try:
                candidate.write_bytes(decoded)
                os.replace(candidate, destination)
            finally:
                candidate.unlink(missing_ok=True)
            return {"ok": True, "path": path, "bytes": len(decoded)}
        return {"ok": False, "error": "No screenshot data"}

    def fill_form(self, fields: list[dict[str, Any]]) -> dict:
        """Fill multiple form fields.
        fields: [{"selector": "#email", "value": "test@test.com"}, ...]
        """
        results: list[dict[str, Any]] = []
        for field in fields:
            selector = field.get("selector")
            value = field.get("value")
            if not isinstance(selector, str) or not isinstance(value, str):
                results.append(
                    {
                        "selector": selector,
                        "result": {"ok": False, "error": "selector and value must be strings"},
                    }
                )
                continue
            r = self.fill_field(selector, value)
            results.append({"selector": selector, "result": r})
        return {
            "ok": all(item["result"].get("ok") is True for item in results),
            "fields": results,
        }

    def close(self):
        self.ws.close()


def _query_tabs() -> tuple[list[dict[str, Any]], str | None]:
    try:
        with urllib.request.urlopen(f"http://{CDP_HOST}:{CDP_PORT}/json", timeout=5) as response:
            tabs = json.loads(response.read())
        if not isinstance(tabs, list):
            return [], "CDP tab endpoint returned a non-list"
        return (
            [tab for tab in tabs if isinstance(tab, dict) and tab.get("type") == "page"],
            None,
        )
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def list_tabs() -> list[dict[str, Any]]:
    """List browser tabs; use ``_query_tabs`` when failure detail is required."""
    return _query_tabs()[0]


def get_tab(index: int = 0) -> dict[str, Any] | None:
    """Get tab by index."""
    tabs = list_tabs()
    if not tabs or index < 0:
        return None
    if index >= len(tabs):
        return None
    return tabs[index]


def find_tab_by_url(url_fragment: str) -> dict[str, Any] | None:
    """Find tab whose URL contains fragment."""
    for t in list_tabs():
        if url_fragment.lower() in t.get("url", "").lower():
            return t
    return None


def new_tab(url: str = "about:blank") -> dict[str, Any]:
    """Open a new tab."""
    invalid = _navigation_error(url)
    if invalid:
        return {"ok": False, "error": invalid}
    try:
        endpoint = f"http://{CDP_HOST}:{CDP_PORT}/json/new?{quote(url, safe='')}"
        request = urllib.request.Request(endpoint, method="PUT")
        with urllib.request.urlopen(request, timeout=5) as response:
            decoded = json.loads(response.read())
        if not isinstance(decoded, dict):
            return {"ok": False, "error": "CDP new-tab endpoint returned a non-object"}
        return {"ok": True, "tab": decoded}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def close_tab(tab_id: str) -> dict[str, Any]:
    """Close a tab by ID."""
    if not tab_id:
        return {"ok": False, "error": "tab_id is required"}
    try:
        with urllib.request.urlopen(
            f"http://{CDP_HOST}:{CDP_PORT}/json/close/{quote(tab_id, safe='')}", timeout=5
        ) as response:
            body = response.read().decode("utf-8", errors="replace").strip()
        return {"ok": True, "response": body}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _emit(payload: dict[str, Any]) -> NoReturn:
    """Emit one machine-readable result and make failure visible to the shell."""
    print(json.dumps(payload))
    raise SystemExit(0 if payload.get("ok") is True else 1)


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "No command. See docstring for usage."}))
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]
    client: CDPClient | None = None

    try:
        if cmd == "tabs":
            tabs, error = _query_tabs()
            _emit({"ok": error is None, "count": len(tabs), "tabs": tabs, "error": error})
            return

        if cmd == "new-tab":
            url = args[0] if args else "about:blank"
            result = new_tab(url)
            _emit(result)
            return

        if cmd == "close-tab":
            tabs = list_tabs()
            if not tabs:
                _emit({"ok": False, "error": "No tabs"})
            tab_id_value = args[0] if args else tabs[-1].get("id")
            if not isinstance(tab_id_value, str):
                _emit({"ok": False, "error": "Selected tab has no string ID"})
            tab_id = tab_id_value
            result = close_tab(tab_id)
            _emit(result)
            return

        # All other commands need a CDP connection
        tab_index = 0
        # Check if last arg is --tab <n>
        if "--tab" in args:
            idx = args.index("--tab")
            tab_index = int(args[idx + 1])
            args = args[:idx] + args[idx + 2 :]

        if cmd == "switch-tab":
            tab_index = int(args[0]) if args else 0
            args = args[1:] if args else []

        # Allow finding tab by URL fragment with --find <text>
        tab = None
        if "--find" in args:
            idx = args.index("--find")
            url_frag = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]
            tab = find_tab_by_url(url_frag)

        if not tab:
            tab = get_tab(tab_index)

        if not tab:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "No browser tabs found. Is Chrome running with --remote-debugging-port=9222?",
                    }
                )
            )
            sys.exit(1)

        ws_url = tab.get("webSocketDebuggerUrl")
        if not ws_url:
            print(
                json.dumps(
                    {"ok": False, "error": "No WebSocket URL for tab", "tab": tab.get("url")}
                )
            )
            sys.exit(1)

        client = CDPClient(ws_url)
        client.enable()

        if cmd == "navigate":
            url = args[0]
            result = client.navigate(url)
            _emit({"ok": result.get("ok") is True, "data": result})

        elif cmd == "read":
            text = client.get_text()
            _emit({"ok": True, "text": text[:5000], "length": len(text)})

        elif cmd == "title":
            _emit({"ok": True, "title": client.get_title()})

        elif cmd == "url":
            _emit({"ok": True, "url": client.get_url()})

        elif cmd == "elements":
            elements = client.get_elements()
            _emit({"ok": True, "count": len(elements), "elements": elements})

        elif cmd == "click":
            text = " ".join(args)
            result = client.click_by_text(text)
            _emit({"ok": result.get("ok") is True, "data": result})

        elif cmd == "click-xy":
            x, y = int(args[0]), int(args[1])
            result = client.click_xy(x, y)
            _emit({"ok": result.get("ok") is True, "data": result})

        elif cmd == "click-element":
            selector = args[0]
            result = client.click_selector(selector)
            _emit({"ok": result.get("ok") is True, "data": result})

        elif cmd == "type":
            text = " ".join(args)
            result = client.type_text(text)
            _emit({"ok": result.get("ok") is True, "data": result})

        elif cmd == "fill":
            selector = args[0]
            value = " ".join(args[1:])
            result = client.fill_field(selector, value)
            _emit({"ok": result.get("ok") is True, "data": result})

        elif cmd == "fill-form":
            json_file = args[0]
            fields = json.loads(Path(json_file).read_text(encoding="utf-8"))
            if not isinstance(fields, list) or not all(isinstance(field, dict) for field in fields):
                raise ValueError("fill-form JSON must be a list of field objects")
            result = client.fill_form(fields)
            _emit({"ok": result.get("ok") is True, "data": result})

        elif cmd == "key":
            key = args[0]
            result = client.press_key(key)
            _emit({"ok": result.get("ok") is True, "data": result})

        elif cmd == "scroll":
            amount = int(args[0])
            result = client.scroll(amount)
            _emit({"ok": result.get("ok") is True, "data": result})

        elif cmd == "wait":
            seconds = float(args[0])
            if not 0 <= seconds <= 60:
                raise ValueError("wait must be between 0 and 60 seconds")
            time.sleep(seconds)
            _emit({"ok": True, "waited": seconds})

        elif cmd == "wait-for":
            text = " ".join(args)
            result = client.wait_for_text(text, timeout=15)
            _emit({"ok": result.get("ok") is True, "data": result})

        elif cmd == "eval":
            js = " ".join(args)
            result = client.eval(js)
            _emit({"ok": "error" not in result, "result": result})

        elif cmd == "screenshot":
            path = args[0] if args else "/tmp/norax_cdp_screenshot.png"
            result = client.screenshot(path)
            _emit(result)

        elif cmd == "back":
            result = client.go_back()
            _emit({"ok": result.get("ok") is True, "data": result})

        elif cmd == "forward":
            result = client.go_forward()
            _emit({"ok": result.get("ok") is True, "data": result})

        elif cmd == "reload":
            result = client.reload()
            _emit({"ok": result.get("ok") is True, "data": result})

        elif cmd == "switch-tab":
            client.send("Page.bringToFront")
            _emit(
                {
                    "ok": True,
                    "tab_id": tab.get("id", ""),
                    "url": tab.get("url", ""),
                    "verification": "Page.bringToFront accepted",
                }
            )

        else:
            print(json.dumps({"ok": False, "error": f"Unknown command: {cmd}"}))
            sys.exit(1)

    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
