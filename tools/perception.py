#!/usr/bin/env python3
"""Unified desktop/browser perception helper for Norax.

Converts accessibility, browser, and OCR observations into a bounded element
map. Each tier reports only data it actually observed; unavailable tiers fall
through to the next backend.

Architecture (3-tier cascade, stop at first success):
  Tier 1: AT-SPI accessibility tree (instant, ~200-500 tokens, structured)
  Tier 2: CDP accessibility tree for browser tabs (~300-800 tokens, structured)
  Tier 3: OCR + Computer Vision via dmap.py (~3s, ~1-4KB, fallback)

Output: Unified element map with ref IDs for actions.

Usage:
  python3 perception.py perceive                    # full cascade
  python3 perception.py perceive --tier cdp         # browser only
  python3 perception.py perceive --tier ocr         # OCR only
  python3 perception.py perceive --tier atspi       # AT-SPI only
  python3 perception.py perceive --focus "Discord"  # filter to app
  python3 perception.py click p42                   # click element by ref
  python3 perception.py type p42 "hello world"      # type into element
  python3 perception.py diff                        # what changed since last perceive
  python3 perception.py inject                      # brain-runner signal output
  python3 perception.py status                      # capabilities check

Brain integration:
  from tools.perception import perceive, inject_signal, click, type_text
  state = perceive()                   # returns PerceptionState
  signal = inject_signal(state)        # returns string for brain-runner
  click("p42")                         # click element
  type_text("p42", "hello")            # type into element

"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

TOOLS_DIR = Path(__file__).resolve().parent
_configured_workspace = os.environ.get("NORAX_WORKSPACE", "").strip()
WORKSPACE = Path(_configured_workspace).expanduser() if _configured_workspace else TOOLS_DIR.parent
PERCEPTION_DIR = Path("/tmp/perception")
STATE_FILE = PERCEPTION_DIR / "state.json"
PREV_STATE_FILE = PERCEPTION_DIR / "prev_state.json"
CDP_ENDPOINT = os.environ.get("CDP_ENDPOINT", "http://127.0.0.1:9222")
AT_SPI_BUS_PATH = os.environ.get("AT_SPI_BUS_PATH", f"/run/user/{os.getuid()}/at-spi/bus")
DMAP_SCRIPT = TOOLS_DIR / "dmap.py"
COMPUTER_USE_SCRIPT = TOOLS_DIR / "computer_use.py"

# Limits
MAX_ELEMENTS = 1200  # cap element count (was 800, raised to fit OCR+CDP alongside AT-SPI)
MAX_OUTPUT_TOKENS = 2000  # target output size for brain injection
MAX_DEPTH_ATSPI = 12  # AT-SPI tree depth (Chrome web content at depth 8-9)
MAX_DEPTH_CDP = 8  # CDP AX tree depth
OCR_TIMEOUT_S = 30  # dmap timeout
CDP_TIMEOUT_S = 8  # CDP timeout
ATSPI_TIMEOUT_S = 5  # AT-SPI timeout
DIFF_SIMILARITY = 0.85  # below this = "changed"

# AT-SPI role names (subset)
ATSPI_ROLES = {
    0: "invalid",
    1: "accel_label",
    2: "alert",
    3: "animation",
    4: "arrow",
    5: "calendar",
    6: "canvas",
    7: "check_box",
    8: "check_menu_item",
    9: "color_chooser",
    10: "column_header",
    11: "combo_box",
    22: "dialog",
    25: "filler",
    26: "focus_traversable",
    28: "frame",
    30: "icon",
    32: "label",
    33: "layered_pane",
    34: "list",
    35: "list_item",
    36: "menu",
    37: "menu_bar",
    38: "menu_item",
    46: "page_tab",
    47: "page_tab_list",
    48: "panel",
    51: "progress_bar",
    52: "push_button",
    53: "radio_button",
    54: "radio_menu_item",
    56: "root_pane",
    57: "row_header",
    58: "scroll_bar",
    60: "scroll_pane",
    62: "separator",
    63: "slider",
    64: "spin_button",
    65: "split_pane",
    66: "status_bar",
    68: "table",
    69: "table_cell",
    70: "table_column_header",
    71: "table_row_header",
    73: "text",
    74: "toggle_button",
    75: "tool_bar",
    76: "tool_tip",
    77: "tree",
    78: "tree_table",
    80: "viewport",
    81: "window",
    82: "extended",
    83: "header",
    84: "footer",
    85: "paragraph",
    87: "application",
    88: "autocomplete",
    90: "document_frame",
    95: "embedded",
    96: "entry",
    100: "heading",
    103: "input_method_window",
    108: "link",
    116: "section",
    120: "static",
    124: "title_bar",
    127: "block_quote",
    128: "audio",
    129: "video",
    130: "definition",
    131: "article",
    132: "landmark",
    133: "log",
    134: "marquee",
    135: "math",
    138: "timer",
    139: "description_list",
    140: "description_term",
    141: "description_value",
    143: "notification",
    162: "content_deletion",
    163: "content_insertion",
}

# Interactive roles (worth reporting)
INTERACTIVE_ROLES = {
    # AT-SPI GetRoleName() returns spaces (e.g. "menu item")
    "push button",
    "push_button",
    "check box",
    "check_box",
    "radio button",
    "radio_button",
    "toggle button",
    "toggle_button",
    "combo box",
    "combo_box",
    "entry",
    "text",
    "spin button",
    "spin_button",
    "slider",
    "link",
    "menu item",
    "menu_item",
    "check menu item",
    "check_menu_item",
    "radio menu item",
    "radio_menu_item",
    "list item",
    "list_item",
    "page tab",
    "page_tab",
    "autocomplete",
    "table cell",
    "table_cell",
    "button",
    "menu",
    "tab",
    "checkbox",
    "radiobutton",
    "text field",
    "textfield",
    "password text",
    "password_text",
}

# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════


@dataclass
class Element:
    ref: str  # p0, p1, p2...
    role: str  # button, text_field, label...
    name: str  # display text / label
    value: str = ""  # current value (for inputs)
    state: str = ""  # enabled, focused, checked...
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)  # x,y,w,h
    tier: str = ""  # atspi, cdp, ocr
    app: str = ""  # source application
    actionable: bool = False  # can be clicked/typed


@dataclass
class PerceptionState:
    timestamp: float = 0.0
    tier_used: str = ""  # which tier produced results
    tiers_attempted: list[str] = field(default_factory=list)
    elements: list[Element] = field(default_factory=list)
    apps: list[str] = field(default_factory=list)
    focused_app: str = ""
    focused_element: str = ""
    screen_size: tuple[int, int] = (0, 0)
    token_estimate: int = 0
    capture_ms: int = 0
    error: str = ""
    cv_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["elements"] = [asdict(e) for e in self.elements]
        return d

    @staticmethod
    def _is_meaningful(name: str) -> bool:
        """Filter OCR garbage: single chars, punctuation-only, noise fragments."""
        if not name or len(name.strip()) < 2:
            return False
        stripped = name.strip()
        # Reject if mostly punctuation/symbols (need at least 50% alphanumeric for short strings)
        alpha_count = sum(1 for c in stripped if c.isalnum() or c == " ")
        if len(stripped) < 4 and alpha_count < max(2, len(stripped) * 0.5):
            return False
        if len(stripped) >= 4 and alpha_count < len(stripped) * 0.3:
            return False
        # Reject common OCR noise patterns
        noise = {
            ".",
            "..",
            "...",
            "-",
            "--",
            "---",
            "/",
            "//",
            "=",
            "==",
            ";",
            ":",
            ",",
            ">",
            "<",
            ">>",
            "<<",
            "()",
            "[]",
            "{}",
            "|",
            "||",
            "*",
            "**",
            "~",
            "~~",
            "—",
            "——",
            "———",
            '"',
            "'",
            "``",
            "''",
            '""',
            "a",
            "i",
            "e",
            "o",
            "oe",
            "/a",
            "ai",
            "ee",
            "au",
            "ft",
        }
        if stripped.lower() in noise:
            return False
        return True

    def summary(self, max_elements: int = 80) -> str:
        """Compact text representation for brain injection.

        Strategy: text-first (shows readable content), then meaningful interactive
        elements. Filters OCR noise aggressively. Designed for GPT-4.1 literal
        instruction following — clear, structured, unambiguous.
        """
        lines = []
        lines.append(
            f"PERCEPTION ({self.tier_used}, {len(self.elements)} elements, {self.capture_ms}ms)"
        )

        if self.focused_app:
            lines.append(
                f"  Focus: {self.focused_app}"
                + (f" > {self.focused_element}" if self.focused_element else "")
            )

        if self.apps:
            lines.append(f"  Apps: {', '.join(self.apps[:10])}")

        # Group by app
        by_app: dict[str, list[Element]] = {}
        for e in self.elements:
            by_app.setdefault(e.app or "desktop", []).append(e)

        count = 0
        for app, elems in sorted(by_app.items()):
            if count >= max_elements:
                lines.append(f"  ... +{len(self.elements) - count} more elements")
                break

            # Split into meaningful vs noise
            meaningful_text = [e for e in elems if not e.actionable and self._is_meaningful(e.name)]
            meaningful_interactive = [
                e for e in elems if e.actionable and self._is_meaningful(e.name)
            ]
            noise_count = len(elems) - len(meaningful_text) - len(meaningful_interactive)

            if not meaningful_text and not meaningful_interactive:
                continue

            lines.append(f"  [{app}]")

            # TEXT FIRST — show readable screen content (what the user sees)
            if meaningful_text:
                lines.append("  Visible text:")
                for e in meaningful_text[:30]:
                    lines.append(f'    {e.ref} "{e.name}"')
                    count += 1

            # Then interactive elements with meaningful labels
            if meaningful_interactive:
                lines.append("  Clickable:")
                for e in meaningful_interactive[:20]:
                    parts = [f"    {e.ref} {e.role}"]
                    if e.name:
                        parts.append(f'"{e.name}"')
                    if e.value:
                        parts.append(f"val={e.value}")
                    if e.state:
                        parts.append(f"[{e.state}]")
                    lines.append(" ".join(parts))
                    count += 1

            if noise_count > 5:
                lines.append(f"    ({noise_count} OCR noise elements filtered)")

        # Generate a brief natural-language summary for model comprehension
        all_meaningful = [e.name for e in self.elements if self._is_meaningful(e.name)]
        if all_meaningful:
            # Pick the most informative text elements (longer = more info)
            top = sorted(all_meaningful, key=len, reverse=True)[:8]
            sep = " | "
            lines.append(f"  Screen shows: {sep.join(top)}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# TIER 1: AT-SPI ACCESSIBILITY TREE
# ═══════════════════════════════════════════════════════════════


def _resolve_atspi_bus_path() -> str | None:
    """
    Find a live AT-SPI D-Bus socket.

    The socket path can go stale across session/service restarts (a11y bus
    launcher rotates the socket file, e.g. bus -> bus_1) while the old file
    handle lingers on disk pointing nowhere. Rather than trust one hardcoded
    path, try the configured/default path first, then glob for siblings
    (bus, bus_1, bus_2, ...) newest-mtime-first, and actually test-connect
    each candidate instead of just checking os.path.exists.
    """
    import dbus

    candidates = []
    env_path = os.environ.get("AT_SPI_BUS_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.append(AT_SPI_BUS_PATH)

    bus_dir = os.path.dirname(AT_SPI_BUS_PATH)
    try:
        globbed = sorted(
            Path(bus_dir).glob("bus*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(str(p) for p in globbed)
    except Exception:
        pass

    seen = set()
    for path in candidates:
        if path in seen or not path:
            continue
        seen.add(path)
        if not os.path.exists(path):
            continue
        try:
            bus = dbus.bus.BusConnection(f"unix:path={path}")
            root_obj = bus.get_object("org.a11y.atspi.Registry", "/org/a11y/atspi/accessible/root")
            props = dbus.Interface(root_obj, "org.freedesktop.DBus.Properties")
            props.Get("org.a11y.atspi.Accessible", "ChildCount")
            return path
        except Exception:
            continue
    return None


def _atspi_perceive(focus_app: str | None = None) -> PerceptionState | None:
    """Query AT-SPI2 accessibility tree via D-Bus."""
    try:
        import dbus
    except ImportError:
        return None

    resolved_path = _resolve_atspi_bus_path()
    if not resolved_path:
        return None

    t0 = time.monotonic()
    elements = []
    apps_found = []
    ref_counter = [0]

    try:
        bus = dbus.bus.BusConnection(f"unix:path={resolved_path}")

        # Get registry root
        root_obj = bus.get_object("org.a11y.atspi.Registry", "/org/a11y/atspi/accessible/root")
        root_acc = dbus.Interface(root_obj, "org.a11y.atspi.Accessible")
        root_props = dbus.Interface(root_obj, "org.freedesktop.DBus.Properties")
        child_count = int(root_props.Get("org.a11y.atspi.Accessible", "ChildCount"))

        if child_count == 0:
            return None

        def _safe_get(props, iface, prop, default=None):
            """Get a property safely, returning default on any error."""
            try:
                return props.Get(iface, prop)
            except Exception:
                return default

        def _walk(bus_name, path, depth, app_name):
            if depth > MAX_DEPTH_ATSPI or ref_counter[0] >= MAX_ELEMENTS:
                return
            try:
                obj = bus.get_object(str(bus_name), str(path))
                props = dbus.Interface(obj, "org.freedesktop.DBus.Properties")
                acc = dbus.Interface(obj, "org.a11y.atspi.Accessible")

                name = str(_safe_get(props, "org.a11y.atspi.Accessible", "Name", "") or "")

                # Role: GetRoleName() works, props.Get('Role') throws on some objects
                try:
                    role_name = str(acc.GetRoleName() or "")
                except Exception:
                    try:
                        role_id = int(acc.GetRole())
                        role_name = ATSPI_ROLES.get(role_id, f"role_{role_id}")
                    except Exception:
                        role_name = "unknown"

                n_children = _safe_get(props, "org.a11y.atspi.Accessible", "ChildCount", 0)
                n_children = int(n_children) if n_children is not None else 0

                # Bounds: use Component.GetExtents(0) for screen coords
                bounds = (0, 0, 0, 0)
                try:
                    comp = dbus.Interface(obj, "org.a11y.atspi.Component")
                    ext = comp.GetExtents(0)  # 0 = screen coordinates
                    bounds = (int(ext[0]), int(ext[1]), int(ext[2]), int(ext[3]))
                except Exception:
                    pass

                # Get value if available
                value = ""
                try:
                    value = str(props.Get("org.a11y.atspi.Value", "CurrentValue"))
                except Exception:
                    pass

                if not value:
                    try:
                        text_iface = dbus.Interface(obj, "org.a11y.atspi.Text")
                        char_count = text_iface.GetCharacterCount()
                        if 0 < char_count < 500:
                            value = str(text_iface.GetText(0, char_count))
                    except Exception:
                        pass

                is_interactive = role_name in INTERACTIVE_ROLES
                has_content = bool(name.strip()) or bool(value.strip())
                # Content roles: always include even if name is empty (text is in value)
                CONTENT_ROLES = {
                    "paragraph",
                    "section",
                    "heading",
                    "article",
                    "block quote",
                    "block_quote",
                    "text",
                    "label",
                    "document web",
                    "document_web",
                    "list",
                    "list item",
                    "definition",
                    "description term",
                    "description value",
                }
                is_content = role_name in CONTENT_ROLES

                if has_content or is_interactive or is_content:
                    ref = f"p{ref_counter[0]}"
                    ref_counter[0] += 1
                    elements.append(
                        Element(
                            ref=ref,
                            role=role_name,
                            name=name.strip(),
                            value=value.strip() if value else "",
                            state="",
                            bounds=bounds,
                            tier="atspi",
                            app=app_name,
                            actionable=is_interactive,
                        )
                    )

                # Recurse into children
                for i in range(min(n_children, 50)):
                    try:
                        child_ref = acc.GetChildAtIndex(i)
                        _walk(child_ref[0], child_ref[1], depth + 1, app_name)
                    except Exception:
                        continue
            except Exception:
                return

        # Walk each application
        for i in range(min(child_count, 30)):
            try:
                child_ref = root_acc.GetChildAtIndex(i)
                child_obj = bus.get_object(str(child_ref[0]), str(child_ref[1]))
                child_props = dbus.Interface(child_obj, "org.freedesktop.DBus.Properties")
                app_name = str(child_props.Get("org.a11y.atspi.Accessible", "Name"))

                if focus_app and focus_app.lower() not in app_name.lower():
                    continue

                apps_found.append(app_name)
                _walk(child_ref[0], child_ref[1], 0, app_name)
            except Exception:
                continue

        if not elements:
            return None

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        state = PerceptionState(
            timestamp=time.time(),
            tier_used="atspi",
            tiers_attempted=["atspi"],
            elements=elements,
            apps=apps_found,
            capture_ms=elapsed_ms,
            token_estimate=sum(len(e.name) + len(e.value) + 20 for e in elements) // 4,
        )
        return state

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# TIER 2: CDP ACCESSIBILITY TREE (Chrome/Electron)
# ═══════════════════════════════════════════════════════════════


def _cdp_perceive(
    focus_app: str | None = None, allow_headless: bool = False
) -> PerceptionState | None:
    """Query Chrome DevTools Protocol accessibility tree."""
    import urllib.request

    t0 = time.monotonic()

    try:
        # FIX 5: Reject CDP data from a headless-launched Chrome instance
        # during the AUTO CASCADE only. A background --headless=new Chrome
        # (e.g. spun up for scraping/testing) always reports
        # document.hasFocus()==true for its own tabs, which previously
        # shadowed the real on-screen app and starved the OCR fallback.
        # If the process bound to CDP_ENDPOINT's port was launched with
        # --headless, it is not what the user is looking at — bail out so
        # the cascade falls through to AT-SPI-miss -> OCR.
        # EXCEPTION: when the caller explicitly forces tier="cdp" (or
        # allow_headless=True), honor that — the caller has already
        # decided CDP is the right tier (e.g. our only browser is headless
        # by design, as on this host), so vetoing it here silently starved
        # every forced-cdp perceive call down to zero elements.
        if not allow_headless:
            try:
                port = CDP_ENDPOINT.rsplit(":", 1)[-1]
                ss_out = subprocess.run(
                    ["ss", "-ltnp"], capture_output=True, text=True, timeout=2
                ).stdout
                pid = None
                for line in ss_out.splitlines():
                    if f":{port} " in line or f":{port}\t" in line:
                        m = re.search(r"pid=(\d+)", line)
                        if m:
                            pid = m.group(1)
                            break
                if pid:
                    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "ignore")
                    if "--headless" in cmdline or "ozone-platform=headless" in cmdline:
                        return None
            except Exception:
                pass  # if we can't determine, don't block on this check

        # Get tabs
        tabs_raw = urllib.request.urlopen(f"{CDP_ENDPOINT}/json", timeout=3).read()
        tabs = json.loads(tabs_raw)
        # FIX 4: Filter out Chrome internal UI pages (omnibox, newtab, settings, etc.)
        # These appear as separate "page" targets but are browser chrome, not content.
        CHROME_INTERNAL_PREFIXES = (
            "chrome://omnibox",
            "chrome://newtab",
            "chrome://settings",
            "chrome://extensions",
            "chrome://downloads",
            "chrome://bookmarks",
            "chrome://history",
            "chrome://version",
            "chrome://flags",
            "chrome://device-log",
            "chrome://interstitials",
            "chrome-untrusted://",
            "devtools://",
        )
        real_tabs = [
            t
            for t in tabs
            if t.get("type") == "page"
            and not t.get("url", "").startswith("blob:")
            and not any(t.get("url", "").startswith(p) for p in CHROME_INTERNAL_PREFIXES)
        ]

        if not real_tabs:
            return None

        # Filter if focus_app specified
        if focus_app:
            real_tabs = [
                t
                for t in real_tabs
                if focus_app.lower() in t.get("title", "").lower()
                or focus_app.lower() in t.get("url", "").lower()
            ]
            if not real_tabs:
                return None

        elements = []
        apps_found = []
        ref_counter = [0]

        try:
            import websocket
        except ImportError:
            # Try using screen-map3 as fallback
            return _cdp_via_screenmap3(focus_app)

        # FIX 1: Find the focused tab via document.hasFocus() to avoid stale
        # data from background tabs. CDP /json doesn't expose active state, so
        # we query each tab's websocket. Falls back to first tab if none focused.
        active_tab = None
        for _tab in real_tabs:
            _ws_url = _tab.get("webSocketDebuggerUrl")
            if not _ws_url:
                continue
            try:
                _ws = websocket.create_connection(_ws_url, timeout=3)
                _ws.send(
                    json.dumps(
                        {
                            "id": 1,
                            "method": "Runtime.evaluate",
                            "params": {"expression": "document.hasFocus()", "returnByValue": True},
                        }
                    )
                )
                _resp = json.loads(_ws.recv())
                _ws.close()
                if _resp.get("result", {}).get("result", {}).get("value") is True:
                    active_tab = _tab
                    break
            except Exception:
                continue
        if not active_tab:
            active_tab = real_tabs[0] if real_tabs else None
        if not active_tab:
            return None

        for tab in [active_tab]:
            ws_url = tab.get("webSocketDebuggerUrl")
            if not ws_url:
                continue

            tab_title = tab.get("title", "Unknown")[:50]
            apps_found.append(tab_title)

            try:
                ws = websocket.create_connection(ws_url, timeout=CDP_TIMEOUT_S)

                # Get accessibility tree
                ws.send(
                    json.dumps(
                        {
                            "id": 1,
                            "method": "Accessibility.getFullAXTree",
                            "params": {"depth": MAX_DEPTH_CDP},
                        }
                    )
                )

                resp = json.loads(ws.recv())

                nodes = resp.get("result", {}).get("nodes", [])

                # FIX 3: Resolve bounds via DOM.getBoxModel using backendNodeId
                # directly. The AX tree provides backendDOMNodeId which can be
                # passed straight to getBoxModel — no DOM.getDocument mapping
                # needed (that mapping was broken: DOM nodes return backendNodeId=None).
                box_cache = {}  # keyed by backendDOMNodeId
                msg_id = 3

                # Build AX node map for parent-chain walking (InlineTextBox
                # nodes have no backendDOMNodeId — must inherit from ancestor)
                ax_node_map = {n.get("nodeId", ""): n for n in nodes}

                for node in nodes:
                    if ref_counter[0] >= MAX_ELEMENTS:
                        break

                    role = node.get("role", {}).get("value", "")
                    name_obj = node.get("name", {})
                    name = (
                        name_obj.get("value", "") if isinstance(name_obj, dict) else str(name_obj)
                    )
                    value_obj = node.get("value", {})
                    value = value_obj.get("value", "") if isinstance(value_obj, dict) else ""

                    # Skip noise
                    if (
                        role in ("none", "generic", "InlineTextBox", "LineBreak", "StaticText")
                        and not name.strip()
                    ):
                        continue
                    if role == "StaticText" and len(name) < 2:
                        continue

                    is_interactive = role in (
                        "button",
                        "link",
                        "textbox",
                        "combobox",
                        "checkbox",
                        "radio",
                        "menuitem",
                        "tab",
                        "switch",
                        "searchbox",
                        "spinbutton",
                        "slider",
                        "listitem",
                        "option",
                    )

                    # Get state from properties
                    state_parts = []
                    for prop in node.get("properties", []):
                        pname = prop.get("name", "")
                        pval = prop.get("value", {}).get("value", "")
                        if pname == "focused" and pval:
                            state_parts.append("focused")
                        elif pname == "disabled" and pval:
                            state_parts.append("disabled")
                        elif pname == "checked" and pval:
                            state_parts.append("checked")
                        elif pname == "expanded" and pval:
                            state_parts.append("expanded")

                    if name.strip() or is_interactive:
                        # FIX 3: Resolve real bounds via DOM.getBoxModel
                        # using backendNodeId directly (no DOM mapping needed)
                        bounds = (0, 0, 0, 0)
                        backend_id = node.get("backendDOMNodeId")
                        # FIX 4: InlineTextBox nodes have no backendDOMNodeId.
                        # Walk up the AX parent chain to find the nearest
                        # ancestor with a backendDOMNodeId and inherit it.
                        if not backend_id:
                            cur_pid = node.get("parentId", "")
                            for _ in range(5):
                                parent_node = ax_node_map.get(cur_pid)
                                if not parent_node:
                                    break
                                p_bid = parent_node.get("backendDOMNodeId")
                                if p_bid:
                                    backend_id = p_bid
                                    break
                                cur_pid = parent_node.get("parentId", "")

                        if backend_id and backend_id not in box_cache:
                            try:
                                ws.send(
                                    json.dumps(
                                        {
                                            "id": msg_id,
                                            "method": "DOM.getBoxModel",
                                            "params": {"backendNodeId": backend_id},
                                        }
                                    )
                                )
                                box_resp = json.loads(ws.recv())
                                msg_id += 1
                                content = (
                                    box_resp.get("result", {}).get("model", {}).get("content", [])
                                )
                                if content and len(content) >= 8:
                                    xs = content[0::2]
                                    ys = content[1::2]
                                    bounds = (
                                        int(min(xs)),
                                        int(min(ys)),
                                        int(max(xs) - min(xs)),
                                        int(max(ys) - min(ys)),
                                    )
                                    box_cache[backend_id] = bounds
                                else:
                                    box_cache[backend_id] = (0, 0, 0, 0)
                            except Exception:
                                box_cache[backend_id] = (0, 0, 0, 0)
                        elif backend_id and backend_id in box_cache:
                            bounds = box_cache[backend_id]

                        ref = f"p{ref_counter[0]}"
                        ref_counter[0] += 1
                        elements.append(
                            Element(
                                ref=ref,
                                role=role,
                                name=name.strip()[:100],
                                value=str(value).strip()[:100] if value else "",
                                state=",".join(state_parts),
                                bounds=bounds,
                                tier="cdp",
                                app=tab_title,
                                actionable=is_interactive,
                            )
                        )

                ws.close()
            except Exception:
                continue

        if not elements:
            return None

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        return PerceptionState(
            timestamp=time.time(),
            tier_used="cdp",
            tiers_attempted=["cdp"],
            elements=elements,
            apps=apps_found,
            capture_ms=elapsed_ms,
            token_estimate=sum(len(e.name) + len(e.value) + 20 for e in elements) // 4,
        )
    except Exception:
        return None


def _cdp_via_screenmap3(focus_app: str | None = None) -> PerceptionState | None:
    """Fallback: use screen-map3.py for CDP tree."""
    sm3 = TOOLS_DIR / "screen-map3.py"
    if not sm3.exists():
        return None

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, str(sm3), "--cdp", "--json"],
            capture_output=True,
            text=True,
            timeout=CDP_TIMEOUT_S,
            cwd=str(TOOLS_DIR),
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        elements = []
        ref_counter = 0

        for item in data.get("elements", data.get("nodes", []))[:MAX_ELEMENTS]:
            role = item.get("role", "unknown")
            name = item.get("name", item.get("text", ""))
            is_interactive = role in (
                "button",
                "link",
                "textbox",
                "combobox",
                "checkbox",
                "radio",
                "menuitem",
                "tab",
            )

            if name.strip() or is_interactive:
                elements.append(
                    Element(
                        ref=f"p{ref_counter}",
                        role=role,
                        name=str(name).strip()[:100],
                        value=str(item.get("value", "")).strip()[:100],
                        state="",
                        bounds=tuple(item.get("bounds", [0, 0, 0, 0])[:4]),
                        tier="cdp",
                        app=item.get("app", "Chrome"),
                        actionable=is_interactive,
                    )
                )
                ref_counter += 1

        if not elements:
            return None

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return PerceptionState(
            timestamp=time.time(),
            tier_used="cdp",
            tiers_attempted=["cdp"],
            elements=elements,
            apps=list(set(e.app for e in elements)),
            capture_ms=elapsed_ms,
            token_estimate=sum(len(e.name) + len(e.value) + 20 for e in elements) // 4,
        )
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# TIER 3: OCR + COMPUTER VISION (dmap.py)
# ═══════════════════════════════════════════════════════════════


def _ocr_perceive(focus_app: str | None = None) -> PerceptionState | None:
    """Capture desktop via OCR + CV using dmap.py."""
    if not DMAP_SCRIPT.exists():
        return None

    t0 = time.monotonic()

    try:
        # Run dmap snap + read as JSON
        snapshot = subprocess.run(
            [sys.executable, str(DMAP_SCRIPT), "snap"],
            capture_output=True,
            text=True,
            timeout=OCR_TIMEOUT_S,
            env=os.environ.copy(),
        )
        if snapshot.returncode != 0:
            return None

        # Then read elements
        result2 = subprocess.run(
            [sys.executable, str(DMAP_SCRIPT), "read", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result2.returncode != 0:
            # Try without --json flag
            result2 = subprocess.run(
                [sys.executable, str(DMAP_SCRIPT), "read"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result2.returncode != 0:
                return None

            # Parse text output
            return _parse_dmap_text(result2.stdout, t0, focus_app)

        try:
            data = json.loads(result2.stdout)
        except json.JSONDecodeError:
            return _parse_dmap_text(result2.stdout, t0, focus_app)

        elements = []
        ref_counter = 0

        raw_elements = data if isinstance(data, list) else data.get("elements", [])
        if not isinstance(raw_elements, list):
            return None
        for item in raw_elements:
            if ref_counter >= MAX_ELEMENTS:
                break
            if not isinstance(item, dict):
                continue

            etype_raw = item.get("type", "text")
            text_raw = item.get("text", item.get("label", ""))
            bounds_raw = item.get("bounds", item.get("bbox", [0, 0, 0, 0]))
            etype = etype_raw if isinstance(etype_raw, str) else "text"
            text = text_raw if isinstance(text_raw, str) else ""
            if isinstance(bounds_raw, (list, tuple)) and len(bounds_raw) >= 4:
                try:
                    bounds = (
                        int(bounds_raw[0]),
                        int(bounds_raw[1]),
                        int(bounds_raw[2]),
                        int(bounds_raw[3]),
                    )
                except (TypeError, ValueError):
                    bounds = (0, 0, 0, 0)
            else:
                bounds = (0, 0, 0, 0)

            if not text.strip():
                continue

            # Filter OCR garbage at ingestion: single chars, punctuation noise
            stripped = text.strip()
            if len(stripped) < 2:
                continue
            alpha_count = sum(1 for c in stripped if c.isalnum() or c == " ")
            if alpha_count < len(stripped) * 0.3 and len(stripped) < 5:
                continue

            is_interactive = etype in ("button", "input", "control", "link", "clickable")

            elements.append(
                Element(
                    ref=f"p{ref_counter}",
                    role=etype,
                    name=text.strip()[:100],
                    value="",
                    state="",
                    bounds=bounds,
                    tier="ocr",
                    app=focus_app or "desktop",
                    actionable=is_interactive,
                )
            )
            ref_counter += 1

        if not elements:
            return None

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return PerceptionState(
            timestamp=time.time(),
            tier_used="ocr",
            tiers_attempted=["ocr"],
            elements=elements,
            apps=[focus_app or "desktop"],
            capture_ms=elapsed_ms,
            token_estimate=sum(len(e.name) + 15 for e in elements) // 4,
        )
    except Exception:
        return None


def _parse_dmap_text(text: str, t0: float, focus_app: str | None = None) -> PerceptionState | None:
    """Parse dmap text output into PerceptionState."""
    elements = []
    ref_counter = 0

    for line in text.strip().split("\n"):
        if ref_counter >= MAX_ELEMENTS:
            break
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("="):
            continue

        # dmap format: s42 button "Submit" (100,200,50,30)
        # or: s42 text "Hello World" (100,200,300,20)
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue

        # Current compact format: ``[s42] button (x,y wxh) "Label"``.
        # Older unbracketed refs remain accepted for saved indexes.
        if not re.fullmatch(r"\[?s\d+\]?", parts[0]):
            continue
        etype = parts[1]
        rest = parts[2] if len(parts) > 2 else ""

        # Extract text in quotes
        name = ""
        if '"' in rest:
            try:
                name = rest.split('"')[1]
            except Exception:
                name = rest
        else:
            name = rest.split("(")[0].strip() if "(" in rest else rest

        if not name.strip():
            continue

        bounds = (0, 0, 0, 0)
        bounds_match = re.search(r"\((-?\d+),(-?\d+)\s+(\d+)x(\d+)\)", rest)
        if bounds_match:
            bx, by, bw, bh = (int(value) for value in bounds_match.groups())
            bounds = (bx, by, bw, bh)

        is_interactive = etype in ("button", "key", "icon", "input", "control", "link")

        elements.append(
            Element(
                ref=f"p{ref_counter}",
                role=etype,
                name=name.strip()[:100],
                value="",
                state="",
                bounds=bounds,
                tier="ocr",
                app=focus_app or "desktop",
                actionable=is_interactive,
            )
        )
        ref_counter += 1

    if not elements:
        return None

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return PerceptionState(
        timestamp=time.time(),
        tier_used="ocr",
        tiers_attempted=["ocr"],
        elements=elements,
        apps=[focus_app or "desktop"],
        capture_ms=elapsed_ms,
        token_estimate=sum(len(e.name) + 15 for e in elements) // 4,
    )


# ═══════════════════════════════════════════════════════════════
# UNIFIED PERCEIVE
# ═══════════════════════════════════════════════════════════════


def _persist_state(state: PerceptionState) -> None:
    """Atomically persist current state and rotate the last complete state."""
    PERCEPTION_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_name(f"{STATE_FILE.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(json.dumps(state.to_dict(), default=str), encoding="utf-8")
    try:
        if STATE_FILE.exists():
            os.replace(STATE_FILE, PREV_STATE_FILE)
        os.replace(temporary, STATE_FILE)
    finally:
        temporary.unlink(missing_ok=True)


def perceive(
    tier: str | None = None,
    focus_app: str | None = None,
    save: bool = True,
) -> PerceptionState:
    """
    Main perception entry point. Cascades through tiers.

    Args:
        tier: Force specific tier (atspi, cdp, ocr) or None for cascade
        focus_app: Filter to specific application name
        save: Save state for diff comparisons

    Returns:
        PerceptionState with all detected elements
    """
    tiers_attempted: list[str] = []
    state: PerceptionState | None = None

    if tier is not None:
        tier = tier.strip().lower()
        if tier not in {"atspi", "cdp", "ocr"}:
            tiers_attempted.append(tier)
            state = PerceptionState(
                timestamp=time.time(),
                tier_used="none",
                tiers_attempted=[tier],
                error=f"Unknown perception tier: {tier}",
            )

    if tier and state is None:
        # Forced tier
        tiers_attempted.append(tier)
        if tier == "atspi":
            state = _atspi_perceive(focus_app)
        elif tier == "cdp":
            state = _cdp_perceive(focus_app, allow_headless=True)
        elif tier == "ocr":
            state = _ocr_perceive(focus_app)
    elif state is None:
        # Cascade: AT-SPI → CDP → OCR
        # Try AT-SPI first (cheapest)
        tiers_attempted.append("atspi")
        state = _atspi_perceive(focus_app)

        if not state or len(state.elements) < 3:
            # Try CDP (browser)
            tiers_attempted.append("cdp")
            cdp_state = _cdp_perceive(focus_app)

            if cdp_state:
                if state:
                    # Merge: AT-SPI + CDP
                    ref_offset = len(state.elements)
                    remaining = max(0, MAX_ELEMENTS - ref_offset)
                    for e in cdp_state.elements[:remaining]:
                        e.ref = f"p{ref_offset}"
                        ref_offset += 1
                        state.elements.append(e)
                    state.apps.extend(cdp_state.apps)
                    state.tier_used = "atspi+cdp"
                    state.capture_ms += cdp_state.capture_ms
                else:
                    state = cdp_state

        if not state or len(state.elements) < 3:
            # OCR fallback
            tiers_attempted.append("ocr")
            ocr_state = _ocr_perceive(focus_app)

            if ocr_state:
                if state:
                    ref_offset = len(state.elements)
                    remaining = max(0, MAX_ELEMENTS - ref_offset)
                    for e in ocr_state.elements[:remaining]:
                        e.ref = f"p{ref_offset}"
                        ref_offset += 1
                        state.elements.append(e)
                    state.apps.extend(ocr_state.apps)
                    state.tier_used += "+ocr"
                    state.capture_ms += ocr_state.capture_ms
                else:
                    state = ocr_state

    if not state:
        state = PerceptionState(
            timestamp=time.time(),
            tier_used="none",
            tiers_attempted=tiers_attempted,
            error="No perception data available from any tier",
        )

    state.tiers_attempted = tiers_attempted
    state.elements = state.elements[:MAX_ELEMENTS]
    state.apps = list(dict.fromkeys(state.apps))
    state.token_estimate = sum(len(e.name) + len(e.value) + 20 for e in state.elements) // 4

    # Save for diff
    if save:
        try:
            _persist_state(state)
        except Exception as exc:
            state.cv_meta["state_persistence_error"] = str(exc)

    return state


# ═══════════════════════════════════════════════════════════════
# ACTIONS (Motor Cortex)
# ═══════════════════════════════════════════════════════════════


def _load_state() -> PerceptionState | None:
    """Load last perception state."""
    if not STATE_FILE.exists():
        return None
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        timestamp = data.get("timestamp")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or timestamp <= 0
            or timestamp > time.time() + 300
        ):
            return None
        raw_elements = data.get("elements", [])
        if not isinstance(raw_elements, list):
            return None
        elements: list[Element] = []
        element_fields = {
            "ref",
            "role",
            "name",
            "value",
            "state",
            "bounds",
            "tier",
            "app",
            "actionable",
        }
        for raw in raw_elements[:MAX_ELEMENTS]:
            if not isinstance(raw, dict):
                return None
            payload = {key: value for key, value in raw.items() if key in element_fields}
            bounds = payload.get("bounds", (0, 0, 0, 0))
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
                return None
            try:
                normalized_bounds = tuple(int(value) for value in bounds)
            except (TypeError, ValueError):
                return None
            if normalized_bounds[2] < 0 or normalized_bounds[3] < 0:
                return None
            payload["bounds"] = normalized_bounds
            if "actionable" in payload and not isinstance(payload["actionable"], bool):
                return None
            element = Element(**payload)
            if not all(
                isinstance(value, str)
                for value in (
                    element.ref,
                    element.role,
                    element.name,
                    element.value,
                    element.state,
                    element.tier,
                    element.app,
                )
            ):
                return None
            elements.append(element)

        def string_list(key: str) -> list[str]:
            value = data.get(key, [])
            return (
                [item for item in value if isinstance(item, str)] if isinstance(value, list) else []
            )

        screen_size = data.get("screen_size", (0, 0))
        if not isinstance(screen_size, (list, tuple)) or len(screen_size) != 2:
            screen_size = (0, 0)
        cv_meta = data.get("cv_meta", {})
        return PerceptionState(
            timestamp=float(timestamp),
            tier_used=str(data.get("tier_used", "")),
            tiers_attempted=string_list("tiers_attempted"),
            elements=elements,
            apps=string_list("apps"),
            focused_app=str(data.get("focused_app", "")),
            focused_element=str(data.get("focused_element", "")),
            screen_size=(int(screen_size[0]), int(screen_size[1])),
            token_estimate=int(data.get("token_estimate", 0)),
            capture_ms=int(data.get("capture_ms", 0)),
            error=str(data.get("error", "")),
            cv_meta=cv_meta if isinstance(cv_meta, dict) else {},
        )
    except Exception:
        return None


def _run_computer_action(*args: str, timeout: float = 15.0) -> dict[str, Any]:
    if not COMPUTER_USE_SCRIPT.is_file():
        return {"ok": False, "error": "computer_use backend is unavailable"}
    try:
        result = subprocess.run(
            [sys.executable, str(COMPUTER_USE_SCRIPT), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"computer_use timed out after {exc.timeout}s",
            "outcome_unknown": True,
        }
    except OSError as exc:
        return {"ok": False, "error": f"computer_use failed to start: {exc}"}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        return {"ok": False, "error": f"computer_use returned invalid output: {detail[:500]}"}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "computer_use returned a non-object result"}
    if result.returncode != 0 or payload.get("ok") is not True:
        payload["ok"] = False
        if not payload.get("error"):
            payload["error"] = result.stderr.strip() or "computer_use action failed"
    return payload


def click(ref: str) -> dict[str, Any]:
    """Click an element by ref ID."""
    state = _load_state()
    if state is None:
        return {"ok": False, "error": "No perception state. Run perceive first."}
    age = time.time() - state.timestamp
    if age > 60:
        return {
            "ok": False,
            "error": f"Perception state is stale ({age:.1f}s old). Run perceive again.",
        }
    elem = next((candidate for candidate in state.elements if candidate.ref == ref), None)
    if not elem:
        return {"ok": False, "error": f"Element {ref} not found. Run perceive first."}

    if elem.bounds != (0, 0, 0, 0):
        x = elem.bounds[0] + elem.bounds[2] // 2
        y = elem.bounds[1] + elem.bounds[3] // 2
        result = _run_computer_action("click", str(x), str(y))
        result.update({"action": "click", "ref": ref, "coords": (x, y)})
        if result.get("ok") is True:
            result["verification"] = "input_dispatch_accepted"
        return result

    # A remapped pN reference is not necessarily the original dmap sN ref.
    # Guessing could click an unrelated control, so fail closed.
    return {"ok": False, "error": f"Cannot click {ref} — no coordinates available"}


def type_text(ref: str, text: str) -> dict[str, Any]:
    """Type text into an element."""
    click_result = click(ref)
    if click_result.get("ok") is not True:
        return click_result

    time.sleep(0.2)

    result = _run_computer_action("type", text, "--delay", "30")
    result.update({"action": "type", "ref": ref, "chars": len(text)})
    return result


def key_press(key: str) -> dict[str, Any]:
    """Press a key combination (e.g., 'Return', 'ctrl+c', 'alt+Tab')."""
    if not key:
        return {"ok": False, "error": "key is required"}
    result = _run_computer_action("key", key)
    result.update({"action": "key", "key": key})
    return result


def scroll(direction: str = "down", amount: int = 3) -> dict[str, Any]:
    """Scroll the screen."""
    if direction not in {"up", "down"}:
        return {"ok": False, "error": "direction must be 'up' or 'down'"}
    if not 0 <= amount <= 20:
        return {"ok": False, "error": "amount must be between 0 and 20"}
    signed_amount = amount if direction == "down" else -amount
    result = _run_computer_action("scroll", str(signed_amount))
    result.update({"action": "scroll", "direction": direction, "amount": amount})
    return result


# ═══════════════════════════════════════════════════════════════
# DIFF (Change Detection)
# ═══════════════════════════════════════════════════════════════


def diff() -> dict[str, Any]:
    """Compare current state to previous state."""
    if not STATE_FILE.exists() or not PREV_STATE_FILE.exists():
        return {"error": "Need at least 2 perceive() calls to diff"}

    try:
        curr = json.loads(STATE_FILE.read_text())
        prev = json.loads(PREV_STATE_FILE.read_text())

        curr_names = set(e["name"] for e in curr.get("elements", []) if e.get("name"))
        prev_names = set(e["name"] for e in prev.get("elements", []) if e.get("name"))

        added = curr_names - prev_names
        removed = prev_names - curr_names
        unchanged = curr_names & prev_names

        return {
            "added": sorted(added)[:20],
            "removed": sorted(removed)[:20],
            "unchanged_count": len(unchanged),
            "total_curr": len(curr.get("elements", [])),
            "total_prev": len(prev.get("elements", [])),
            "apps_curr": curr.get("apps", []),
            "apps_prev": prev.get("apps", []),
        }
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# BRAIN INTEGRATION
# ═══════════════════════════════════════════════════════════════


def inject_signal(state: PerceptionState | None = None, max_tokens: int = MAX_OUTPUT_TOKENS) -> str:
    """Generate brain-runner injection signal from perception state."""
    if state is None:
        state = _load_state()

    if not state or not state.elements:
        return ""

    # Only inject if perception is recent (< 30s old)
    age = time.time() - state.timestamp
    if age > 30:
        return ""

    return state.summary(max_elements=80)


def status() -> dict[str, Any]:
    """Passively report installed backends, plus a bounded local CDP probe."""
    capture_backend = shutil.which("grim") or shutil.which("scrot")
    ocr_dependencies = bool(DMAP_SCRIPT.is_file() and shutil.which("tesseract") and capture_backend)
    caps = {
        "status_kind": "dependencies_and_local_cdp_probe",
        "atspi": _resolve_atspi_bus_path() is not None,
        "cdp": False,
        # Kept for compatibility; this means dependencies are installed, not
        # that a compositor capture has been performed successfully.
        "ocr": ocr_dependencies,
        "ocr_dependencies": ocr_dependencies,
        "computer_use": COMPUTER_USE_SCRIPT.is_file(),
        "ydotool": shutil.which("ydotool") is not None,
        "xdotool": shutil.which("xdotool") is not None,
        "grim": shutil.which("grim") is not None,
        "scrot": shutil.which("scrot") is not None,
        "tesseract": shutil.which("tesseract") is not None,
        "display": os.environ.get("DISPLAY", "none"),
        "session_type": os.environ.get("XDG_SESSION_TYPE", "unknown"),
    }

    # Check CDP
    try:
        import urllib.request

        with urllib.request.urlopen(f"{CDP_ENDPOINT}/json/version", timeout=2):
            pass
        caps["cdp"] = True
    except Exception:
        pass

    return caps


# ═══════════════════════════════════════════════════════════════════════════
# FUSED PERCEIVE (AT-SPI + CDP + OCR + CV Fusion)
# ═══════════════════════════════════════════════════════════════════════════


def fused_perceive(focus_app: str | None = None, save: bool = True) -> PerceptionState:
    """
    Full multi-modal perception: AT-SPI + CDP + OCR + bounded CV fusion.

    Runs all tiers, deduplicates compatible cross-source observations using
    ``cv_fusion.py``, and returns those fused elements. If optional fusion
    processing fails, the bounded per-tier observations remain available.
    """
    tiers_attempted: list[str] = []
    atspi_state = None
    cdp_state = None
    ocr_state = None

    tiers_attempted.append("atspi")
    atspi_state = _atspi_perceive(focus_app)

    tiers_attempted.append("cdp")
    cdp_state = _cdp_perceive(focus_app)

    # Always run OCR — it supplements AT-SPI/CDP with visually-detected text
    tiers_attempted.append("ocr")
    ocr_state = _ocr_perceive(focus_app)

    all_elements: list[Element] = []
    ref_counter = 0

    _tier_map = {id(atspi_state): "atspi", id(cdp_state): "cdp", id(ocr_state): "ocr"}
    # OCR and CDP first — they provide unique visual data that AT-SPI can't
    for src_state in [ocr_state, cdp_state, atspi_state]:
        if not src_state:
            continue
        _tier = _tier_map.get(id(src_state), "unknown")
        for e in src_state.elements:
            if ref_counter >= MAX_ELEMENTS:
                break
            e.ref = f"p{ref_counter}"
            e.tier = _tier
            ref_counter += 1
            all_elements.append(e)

    cv_meta: dict[str, Any] = {}
    try:
        if str(TOOLS_DIR) not in sys.path:
            sys.path.insert(0, str(TOOLS_DIR))
        from cv_fusion import fuse as cv_fuse
        from visual_hierarchy import process as vh_process

        # _ocr_perceive already performed the expensive dmap capture/index.
        # Reuse that exact index and screenshot so CV describes the same frame.
        dmap_output = ""
        if ocr_state:
            result = subprocess.run(
                [sys.executable, str(DMAP_SCRIPT), "read", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(TOOLS_DIR),
            )
            if result.returncode == 0:
                dmap_output = result.stdout

        screenshot_path = Path("/tmp/dmap/screen.png")
        visual_percept = None
        if screenshot_path.is_file():
            atspi_text = ""
            if atspi_state and atspi_state.elements:
                atspi_text = " ".join(
                    f"{e.role}:{e.name}" for e in atspi_state.elements[:100] if e.name
                )
            visual_percept = vh_process(
                screenshot_path=str(screenshot_path),
                dmap_output=dmap_output,
                atspi_output=atspi_text,
            )

        profile = cv_fuse(
            atspi_state=atspi_state,
            cdp_state=cdp_state,
            dmap_output=dmap_output,
            visual_percept=visual_percept,
        )
        all_elements = [
            Element(
                ref=f"p{index}",
                role=element.role,
                name=element.name,
                value=element.value,
                state=element.state,
                bounds=element.bounds,
                tier="+".join(element.sources),
                app=element.app,
                actionable=element.actionable,
            )
            for index, element in enumerate(profile.elements[:MAX_ELEMENTS])
        ]
        cv_meta = {
            "app_type": profile.app_type,
            "app_confidence": profile.app_confidence,
            "app_classification_method": profile.app_classification_method,
            "layout": profile.layout_type,
            "edge_density": profile.edge_density,
            "dark_theme": profile.has_dark_theme,
            "contrast": profile.contrast,
            "rectangular_regions": profile.rectangular_regions,
            "text_blocks": profile.text_blocks,
            "button_candidates": profile.button_candidates,
            "fusion_stats": profile.fusion_stats,
        }
    except Exception as e:
        cv_meta = {"cv_error": str(e)}

    if not all_elements:
        return PerceptionState(
            timestamp=time.time(),
            tier_used="none",
            tiers_attempted=tiers_attempted,
            error="No perception data from any tier",
        )

    total_ms = sum(s.capture_ms for s in [atspi_state, cdp_state, ocr_state] if s)
    all_apps = []
    for s in [atspi_state, cdp_state, ocr_state]:
        if s:
            all_apps.extend(s.apps)

    successful_tiers = [
        name
        for name, source_state in (
            ("atspi", atspi_state),
            ("cdp", cdp_state),
            ("ocr", ocr_state),
        )
        if source_state and source_state.elements
    ]
    state = PerceptionState(
        timestamp=time.time(),
        tier_used="+".join(successful_tiers),
        tiers_attempted=tiers_attempted,
        elements=all_elements[:MAX_ELEMENTS],
        apps=list(set(all_apps)),
        capture_ms=total_ms,
        token_estimate=sum(len(e.name) + len(e.value) + 20 for e in all_elements[:MAX_ELEMENTS])
        // 4,
        cv_meta=cv_meta,
    )

    if save:
        try:
            _persist_state(state)
        except Exception as exc:
            state.cv_meta["state_persistence_error"] = str(exc)

    return state


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: perception.py <perceive|fuse|click|type|key|scroll|diff|inject|status> [args...]"
        )
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "perceive":
        tier = None
        focus = None
        for i, arg in enumerate(sys.argv[2:], 2):
            if arg == "--tier" and i + 1 < len(sys.argv):
                tier = sys.argv[i + 1]
            elif arg == "--focus" and i + 1 < len(sys.argv):
                focus = sys.argv[i + 1]

        state = perceive(tier=tier, focus_app=focus)
        print(state.summary())
        print(
            f"\n({state.tier_used}, {len(state.elements)} elements, {state.capture_ms}ms, ~{state.token_estimate} tokens)"
        )

    elif cmd == "click":
        ref = sys.argv[2] if len(sys.argv) > 2 else ""
        result = click(ref)
        print(json.dumps(result))
        if result.get("ok") is not True:
            raise SystemExit(1)

    elif cmd == "type":
        ref = sys.argv[2] if len(sys.argv) > 2 else ""
        text = sys.argv[3] if len(sys.argv) > 3 else ""
        result = type_text(ref, text)
        print(json.dumps(result))
        if result.get("ok") is not True:
            raise SystemExit(1)

    elif cmd == "key":
        key = sys.argv[2] if len(sys.argv) > 2 else "Return"
        result = key_press(key)
        print(json.dumps(result))
        if result.get("ok") is not True:
            raise SystemExit(1)

    elif cmd == "scroll":
        direction = sys.argv[2] if len(sys.argv) > 2 else "down"
        amount = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        result = scroll(direction, amount)
        print(json.dumps(result))
        if result.get("ok") is not True:
            raise SystemExit(1)

    elif cmd == "diff":
        print(json.dumps(diff(), indent=2))

    elif cmd == "inject":
        sig = inject_signal()
        if sig:
            print(sig)
        else:
            print("(no recent perception state)")

    elif cmd == "fuse":
        # Full multi-modal fused perception
        focus = None
        for i, arg in enumerate(sys.argv[2:], 2):
            if arg == "--focus" and i + 1 < len(sys.argv):
                focus = sys.argv[i + 1]
        state = fused_perceive(focus_app=focus)
        print(state.summary())
        if state.cv_meta:
            cv = state.cv_meta
            if "cv_error" in cv:
                print(f"\nCV: error - {cv['cv_error']}")
            else:
                print(
                    f"\nCV: {cv.get('app_type', '?')} (conf={cv.get('app_confidence', 0):.2f}) | layout={cv.get('layout', '?')} | edges={cv.get('edge_density', 0):.2f} | dark={cv.get('dark_theme', False)} | contrast={cv.get('contrast', 0):.2f}"
                )
                print(
                    f"  rects={cv.get('rectangular_regions', 0)} text_blocks={cv.get('text_blocks', 0)} btn_candidates={cv.get('button_candidates', 0)}"
                )
                if cv.get("fusion_stats"):
                    print(f"  Fusion: {cv['fusion_stats']}")
        print(
            f"\n({state.tier_used}, {len(state.elements)} elements, {state.capture_ms}ms, ~{state.token_estimate} tokens)"
        )

    elif cmd == "status":
        print(json.dumps(status(), indent=2))

    elif cmd == "--json":
        # Machine-readable full output
        state = perceive()
        print(json.dumps(state.to_dict(), indent=2, default=str))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
