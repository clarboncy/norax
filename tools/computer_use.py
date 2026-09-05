#!/usr/bin/env python3
"""
computer_use.py — Desktop automation toolkit for Norax Gen5.

Norax desktop automation using ydotool and grim on Wayland.
Hybrid: Wayland-native (ydotool, grim) + X11 fallback (xdotool, scrot)

Actions:
  screenshot [path]              → capture screen to PNG
  click <x> <y> [button]         → mouse click at coordinates
  doubleclick <x> <y>            → double click
  type <text>                    → type text string
  key <keyname>                  → press key (Return, Tab, Escape, BackSpace, etc.)
  move <x> <y>                   → move mouse cursor
  drag <x1> <y1> <x2> <y2>     → drag from point to point
  scroll <amount>                → scroll up (positive) or down (negative)
  clipboard-get                  → get clipboard contents
  clipboard-set <text>           → set clipboard
  info                           → show toolkit status

Usage from Norax:
  from tools.computer_use import screenshot, click, type_text, key_press
  screenshot("/tmp/screen.png")
  click(500, 300)
  type_text("hello world")
  key_press("Return")
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

# Environment
_DEFAULT_RUNTIME_DIR = f"/run/user/{os.getuid()}"
XDG_RUNTIME_DIR = os.environ.get("XDG_RUNTIME_DIR", _DEFAULT_RUNTIME_DIR)
DBUS_SESSION_BUS_ADDRESS = os.environ.get(
    "DBUS_SESSION_BUS_ADDRESS", f"unix:path={_DEFAULT_RUNTIME_DIR}/bus"
)
# Respect the service/session display. A stale Xvfb socket is not proof that
# clients can authenticate to it and previously redirected working :1 sessions
# into a dead :99 display.
DISPLAY = os.environ.get("DISPLAY") or os.environ.get("NORAX_DISPLAY", "")
if not DISPLAY:
    detected_displays = [
        f":{number}" for number in (0, 1, 99) if Path(f"/tmp/.X11-unix/X{number}").exists()
    ]
    if len(detected_displays) == 1:
        DISPLAY = detected_displays[0]
YDOTOOL_SOCKET = os.environ.get("YDOTOOL_SOCKET", "/run/ydotoold/socket")


def _is_wayland_session() -> bool:
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    return session_type == "wayland" or (
        session_type == "" and bool(os.environ.get("WAYLAND_DISPLAY"))
    )


def _is_wayland() -> bool:
    """Return whether a Wayland input backend is plausibly usable.

    This check is deliberately passive.  The previous implementation captured
    the entire screen with ``grim`` before *every* packaged tool call (each call
    runs in a fresh process), adding latency and occasionally hanging during a
    status query.  The requested operation is the genuine bounded probe: its
    exit status is returned to the caller rather than hidden behind a synthetic
    readiness result.
    """
    return _is_wayland_session() and shutil.which("ydotool") is not None


def _run(cmd: list[str], timeout: float = 10.0, input: str | None = None) -> tuple[int, str, str]:
    """Run a subprocess command, return (returncode, stdout, stderr)."""
    env = os.environ.copy()
    env["XDG_RUNTIME_DIR"] = XDG_RUNTIME_DIR
    env["DBUS_SESSION_BUS_ADDRESS"] = DBUS_SESSION_BUS_ADDRESS
    env["DISPLAY"] = DISPLAY
    if os.path.exists(YDOTOOL_SOCKET):
        env["YDOTOOL_SOCKET"] = YDOTOOL_SOCKET
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
            input=input,
            encoding="utf-8",
            errors="replace",
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Timeout after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


# ═══════════════════════════════════════════════════════════════════════════════
# SCREENSHOT
# ═══════════════════════════════════════════════════════════════════════════════


def screenshot(outpath: str = "/tmp/norax-screen.png", mode: str = "auto") -> dict:
    """Capture screenshot. Returns {"ok": bool, "path": str, "size": int, "mode": str}."""
    if mode not in {"auto", "wayland", "x11"}:
        return {
            "ok": False,
            "path": outpath,
            "size": 0,
            "mode": "failed",
            "error": f"unknown mode: {mode}",
        }
    path = Path(outpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    candidate = path.with_name(f".{path.stem}.capture-{os.getpid()}-{time.time_ns()}{suffix}")
    attempts: list[tuple[str, list[str]]] = []
    if mode in {"auto", "wayland"} and (
        mode == "wayland" or (_is_wayland_session() and shutil.which("grim") is not None)
    ):
        attempts.append(("wayland", ["grim", str(candidate)]))
    if mode in {"auto", "x11"}:
        attempts.append(("x11", ["scrot", "-o", str(candidate)]))

    errors: list[str] = []
    try:
        for capture_mode, command in attempts:
            candidate.unlink(missing_ok=True)
            rc, out, err = _run(command, timeout=10)
            if rc == 0 and candidate.is_file() and candidate.stat().st_size > 0:
                size = candidate.stat().st_size
                os.replace(candidate, path)
                return {
                    "ok": True,
                    "path": str(path),
                    "size": size,
                    "mode": capture_mode,
                    "error": None,
                }
            detail = (err or out or f"exited with status {rc}").strip()
            errors.append(f"{capture_mode}: {detail}")
    except OSError as exc:
        errors.append(f"output: {exc}")
    finally:
        candidate.unlink(missing_ok=True)

    if not attempts:
        errors.append(f"no {mode} screenshot backend is available")
    return {
        "ok": False,
        "path": str(path),
        "size": 0,
        "mode": "failed",
        "error": "; ".join(errors),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MOUSE
# ═══════════════════════════════════════════════════════════════════════════════


def click(x: int, y: int, button: int = 1) -> dict:
    """Click at screen coordinates. button: 1=left, 2=right, 3=middle."""
    if button not in {1, 2, 3}:
        return {
            "ok": False,
            "x": x,
            "y": y,
            "button": button,
            "mode": "none",
            "error": f"unsupported mouse button: {button}",
        }
    if _is_wayland():
        # ydotool click codes: 0xC0/1/2 are full left/right/middle clicks.
        btn = {1: "0xC0", 2: "0xC1", 3: "0xC2"}.get(button, "0xC0")
        move_rc, _, move_err = _run(
            ["ydotool", "mousemove", "--absolute", str(x), str(y)], timeout=5
        )
        if move_rc != 0:
            return {
                "ok": False,
                "x": x,
                "y": y,
                "button": button,
                "mode": "wayland",
                "error": move_err,
            }
        time.sleep(0.05)
        rc, out, err = _run(["ydotool", "click", btn], timeout=5)
        return {
            "ok": rc == 0,
            "x": x,
            "y": y,
            "button": button,
            "mode": "wayland",
            "error": err if rc != 0 else None,
        }
    else:
        rc, out, err = _run(
            ["xdotool", "mousemove", str(x), str(y), "click", str(button)], timeout=5
        )
        return {
            "ok": rc == 0,
            "x": x,
            "y": y,
            "button": button,
            "mode": "x11",
            "error": err if rc != 0 else None,
        }


def doubleclick(x: int, y: int) -> dict:
    """Double-click at coordinates."""
    if _is_wayland():
        move_rc, _, move_err = _run(
            ["ydotool", "mousemove", "--absolute", str(x), str(y)], timeout=5
        )
        if move_rc != 0:
            return {
                "ok": False,
                "x": x,
                "y": y,
                "mode": "wayland",
                "error": move_err,
            }
        time.sleep(0.05)
        first_rc, _, first_err = _run(["ydotool", "click", "0xC0"], timeout=5)
        if first_rc != 0:
            return {
                "ok": False,
                "x": x,
                "y": y,
                "mode": "wayland",
                "error": first_err,
            }
        time.sleep(0.08)
        rc, out, err = _run(["ydotool", "click", "0xC0"], timeout=5)
        return {"ok": rc == 0, "x": x, "y": y, "mode": "wayland", "error": err if rc != 0 else None}
    else:
        rc, out, err = _run(
            [
                "xdotool",
                "mousemove",
                str(x),
                str(y),
                "click",
                "--repeat",
                "2",
                "--delay",
                "100",
                "1",
            ],
            timeout=5,
        )
        return {"ok": rc == 0, "x": x, "y": y, "mode": "x11", "error": err if rc != 0 else None}


def move(x: int, y: int) -> dict:
    """Move mouse cursor to coordinates."""
    if _is_wayland():
        rc, out, err = _run(["ydotool", "mousemove", "--absolute", str(x), str(y)], timeout=5)
        return {"ok": rc == 0, "x": x, "y": y, "mode": "wayland", "error": err if rc != 0 else None}
    else:
        rc, out, err = _run(["xdotool", "mousemove", str(x), str(y)], timeout=5)
        return {"ok": rc == 0, "x": x, "y": y, "mode": "x11", "error": err if rc != 0 else None}


def drag(x1: int, y1: int, x2: int, y2: int) -> dict:
    """Drag from (x1,y1) to (x2,y2)."""
    if _is_wayland():
        move_rc, _, move_err = _run(
            ["ydotool", "mousemove", "--absolute", str(x1), str(y1)], timeout=5
        )
        if move_rc != 0:
            return {
                "ok": False,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "mode": "wayland",
                "error": move_err,
            }
        time.sleep(0.05)
        down_rc, _, down_err = _run(["ydotool", "click", "0x40"], timeout=5)
        if down_rc != 0:
            return {
                "ok": False,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "mode": "wayland",
                "error": down_err,
            }
        time.sleep(0.1)
        move_rc, _, move_err = _run(
            ["ydotool", "mousemove", "--absolute", str(x2), str(y2)], timeout=5
        )
        if move_rc != 0:
            # Never leave the primary button held when a partial drag fails.
            _run(["ydotool", "click", "0x80"], timeout=2)
            return {
                "ok": False,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "mode": "wayland",
                "error": move_err,
            }
        time.sleep(0.05)
        rc, out, err = _run(["ydotool", "click", "0x80"], timeout=5)
        return {
            "ok": rc == 0,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "mode": "wayland",
            "error": err if rc != 0 else None,
        }
    else:
        rc, out, err = _run(
            [
                "xdotool",
                "mousemove",
                str(x1),
                str(y1),
                "mousedown",
                "1",
                "mousemove",
                str(x2),
                str(y2),
                "mouseup",
                "1",
            ],
            timeout=5,
        )
        return {
            "ok": rc == 0,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "mode": "x11",
            "error": err if rc != 0 else None,
        }


def scroll(amount: int = 3) -> dict:
    """Scroll: positive=down, negative=up. Amount is number of wheel clicks."""
    if amount == 0:
        return {"ok": True, "amount": 0, "reps": 0, "mode": "none", "error": None}
    reps = max(1, min(abs(amount), 20))  # cap at 20 clicks to avoid timeout
    if _is_wayland():
        btn = "5" if amount > 0 else "4"
        deadline = time.monotonic() + 8.0
        for completed in range(reps):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {
                    "ok": False,
                    "amount": amount,
                    "reps": completed,
                    "mode": "wayland",
                    "error": "scroll deadline exceeded",
                }
            rc, _, err = _run(["ydotool", "click", btn], timeout=min(0.75, remaining))
            if rc != 0:
                return {
                    "ok": False,
                    "amount": amount,
                    "reps": completed,
                    "mode": "wayland",
                    "error": err,
                }
        return {"ok": True, "amount": amount, "reps": reps, "mode": "wayland", "error": None}
    else:
        btn = "5" if amount > 0 else "4"
        rc, out, err = _run(["xdotool", "click", "--repeat", str(reps), btn], timeout=10)
        return {
            "ok": rc == 0,
            "amount": amount,
            "reps": reps,
            "mode": "x11",
            "error": err if rc != 0 else None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# KEYBOARD
# ═══════════════════════════════════════════════════════════════════════════════

# ydotool keycode map (common keys)
YD_KEYMAP = {
    "Return": 28,
    "Enter": 28,
    "Tab": 15,
    "Escape": 1,
    "Esc": 1,
    "BackSpace": 14,
    "Backspace": 14,
    "Delete": 111,
    "Del": 111,
    "space": 57,
    "Space": 57,
    "Up": 103,
    "Down": 108,
    "Left": 105,
    "Right": 106,
    "Page_Up": 104,
    "Page_Down": 109,
    "Home": 102,
    "End": 107,
    "Shift_L": 42,
    "Shift_R": 54,
    "Control_L": 29,
    "Ctrl": 29,
    "ctrl": 29,
    "Control_R": 97,
    "Alt_L": 56,
    "Alt": 56,
    "alt": 56,
    "Alt_R": 100,
    "F1": 59,
    "F2": 60,
    "F3": 61,
    "F4": 62,
    "F5": 63,
    "F6": 64,
    "F7": 65,
    "F8": 66,
    "F9": 67,
    "F10": 68,
    "F11": 87,
    "F12": 88,
    "a": 30,
    "b": 48,
    "c": 46,
    "d": 32,
    "e": 18,
    "f": 33,
    "g": 34,
    "h": 35,
    "i": 23,
    "j": 36,
    "k": 37,
    "l": 38,
    "m": 50,
    "n": 49,
    "o": 24,
    "p": 25,
    "q": 16,
    "r": 19,
    "s": 31,
    "t": 20,
    "u": 22,
    "v": 47,
    "w": 17,
    "x": 45,
    "y": 21,
    "z": 44,
    "0": 11,
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6,
    "6": 7,
    "7": 8,
    "8": 9,
    "9": 10,
}


def _yd_keycode(key: str) -> int | None:
    """Resolve common case variants without weakening key validation."""
    aliases = {
        "control": "Control_L",
        "ctrl": "Control_L",
        "shift": "Shift_L",
        "alt": "Alt_L",
        "enter": "Return",
        "return": "Return",
        "escape": "Escape",
        "esc": "Escape",
        "backspace": "BackSpace",
        "delete": "Delete",
        "del": "Delete",
        "space": "space",
        "pageup": "Page_Up",
        "pagedown": "Page_Down",
    }
    stripped = key.strip()
    canonical = aliases.get(stripped.lower(), stripped)
    if len(canonical) == 1:
        canonical = canonical.lower()
    elif canonical.lower().startswith("f") and canonical[1:].isdigit():
        canonical = canonical.upper()
    return YD_KEYMAP.get(canonical)


def type_text(text: str, delay: int = 12) -> dict:
    """Type text string. delay is ms between keystrokes (xdotool --delay)."""
    if not 0 <= delay <= 1000:
        return {
            "ok": False,
            "chars": 0,
            "mode": "none",
            "error": "delay must be between 0 and 1000 milliseconds",
        }
    if len(text) > 16_384:
        return {
            "ok": False,
            "chars": 0,
            "mode": "none",
            "error": "text exceeds the 16384-character desktop input limit",
        }
    timeout = min(35.0, max(5.0, len(text) * delay / 1000 + 5.0))
    if _is_wayland():
        rc, out, err = _run(
            ["ydotool", "type", "--key-delay", str(delay), "--", text], timeout=timeout
        )
        return {
            "ok": rc == 0,
            "chars": len(text),
            "mode": "wayland",
            "error": err if rc != 0 else None,
        }
    else:
        rc, out, err = _run(
            ["xdotool", "type", "--clearmodifiers", "--delay", str(delay), "--", text],
            timeout=timeout,
        )
        return {"ok": rc == 0, "chars": len(text), "mode": "x11", "error": err if rc != 0 else None}


def key_press(key: str) -> dict:
    """Press a key or key combo. Supports: Return, Tab, Escape, BackSpace, Delete,
    space, Up/Down/Left/Right, ctrl+a, ctrl+c, ctrl+v, ctrl+s, ctrl+z, ctrl+l,
    alt+Tab, alt+F4, and single letters/numbers.
    """
    if not key or len(key) > 128 or not all(c.isalnum() or c in "_+-" for c in key):
        return {
            "ok": False,
            "key": key,
            "mode": "none",
            "error": "key must be a valid key name or '+'-separated key combination",
        }
    if _is_wayland():
        # Handle combos like "ctrl+c"
        if "+" in key:
            parts = key.split("+")
            if len(parts) > 8 or any(not part for part in parts):
                return {
                    "ok": False,
                    "key": key,
                    "mode": "wayland",
                    "error": "invalid key combination",
                }
            codes = []
            for part in parts:
                code = _yd_keycode(part)
                if code is None:
                    return {
                        "ok": False,
                        "key": key,
                        "mode": "wayland",
                        "error": f"Unknown key: {part}",
                    }
                codes.append(code)
            # Press all, release all
            press_seq = (
                " ".join(f"{c}:1" for c in codes)
                + " "
                + " ".join(f"{c}:0" for c in reversed(codes))
            )
            rc, out, err = _run(["ydotool", "key"] + press_seq.split(), timeout=5)
            return {"ok": rc == 0, "key": key, "mode": "wayland", "error": err if rc != 0 else None}
        else:
            code = _yd_keycode(key)
            if code is None:
                return {"ok": False, "key": key, "mode": "wayland", "error": f"Unknown key: {key}"}
            rc, out, err = _run(["ydotool", "key", str(code)], timeout=5)
            return {"ok": rc == 0, "key": key, "mode": "wayland", "error": err if rc != 0 else None}
    else:
        # X11: xdotool handles combos natively
        rc, out, err = _run(["xdotool", "key", "--clearmodifiers", key], timeout=5)
        return {"ok": rc == 0, "key": key, "mode": "x11", "error": err if rc != 0 else None}


# ═══════════════════════════════════════════════════════════════════════════════
# CLIPBOARD
# ═══════════════════════════════════════════════════════════════════════════════


def clipboard_get() -> dict:
    """Get clipboard contents."""
    if _is_wayland_session() and shutil.which("wl-paste"):
        rc, out, err = _run(["wl-paste"], timeout=5)
        if rc == 0:
            limit = 262_144
            return {
                "ok": True,
                "text": out[:limit],
                "truncated": len(out) > limit,
                "mode": "wayland",
                "error": None,
            }
    rc, out, err = _run(["xclip", "-selection", "clipboard", "-o"], timeout=5)
    limit = 262_144
    return {
        "ok": rc == 0,
        "text": out[:limit],
        "truncated": len(out) > limit,
        "mode": "x11",
        "error": err if rc != 0 else None,
    }


def clipboard_set(text: str) -> dict:
    """Set clipboard contents.

    xclip daemonizes — it forks and stays alive to serve clipboard content
    to other X clients. subprocess.run() waits for exit which never comes,
    causing a timeout. Fix: use Popen (fire-and-forget) for xclip, or
    prefer xsel if available (it exits cleanly with -bi).
    """
    if _is_wayland_session() and shutil.which("wl-copy"):
        rc, out, err = _run(["wl-copy"], input=text, timeout=5)
        return {
            "ok": rc == 0,
            "chars": len(text),
            "mode": "wayland",
            "error": err if rc != 0 else None,
        }

    # X11 path — try xsel first (exits cleanly), then xclip via Popen
    if shutil.which("xsel"):
        rc, out, err = _run(["xsel", "--clipboard", "--input"], input=text, timeout=5)
        return {"ok": rc == 0, "chars": len(text), "mode": "x11", "error": err if rc != 0 else None}

    # xclip owns the selection for as long as its process (or daemonized child)
    # remains alive. Killing it after a communicate timeout clears the value,
    # so write stdin and leave the successful selection owner running.
    try:
        env = os.environ.copy()
        env["DISPLAY"] = DISPLAY
        env["XDG_RUNTIME_DIR"] = XDG_RUNTIME_DIR
        proc = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        if proc.stdin is None:
            proc.kill()
            return {
                "ok": False,
                "chars": len(text),
                "mode": "x11",
                "error": "xclip stdin was unavailable",
            }
        proc.stdin.write(text.encode("utf-8"))
        proc.stdin.close()
        time.sleep(0.05)
        return_code = proc.poll()
        if return_code not in {None, 0}:
            return {
                "ok": False,
                "chars": len(text),
                "mode": "x11",
                "error": f"xclip exited with status {return_code}",
            }
        return {
            "ok": True,
            "chars": len(text),
            "mode": "x11",
            "verification": "clipboard selection owner started",
            "error": None,
        }
    except Exception as e:
        return {"ok": False, "chars": len(text), "mode": "x11", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# INFO / STATUS
# ═══════════════════════════════════════════════════════════════════════════════


def info() -> dict:
    """Return passive toolkit status without capturing or changing the desktop."""
    wayland_session = _is_wayland_session()
    has_ydotool = shutil.which("ydotool") is not None
    has_grim = shutil.which("grim") is not None
    status = {
        "ok": True,
        "wayland_session": wayland_session,
        "wayland_input_available": wayland_session and has_ydotool,
        "wayland_capture_available": wayland_session and has_grim,
        "ydotool": has_ydotool,
        "grim": has_grim,
        "xdotool": shutil.which("xdotool") is not None,
        "scrot": shutil.which("scrot") is not None,
        "wl_paste": shutil.which("wl-paste") is not None,
        "wl_copy": shutil.which("wl-copy") is not None,
        "xsel": shutil.which("xsel") is not None,
        "xclip": shutil.which("xclip") is not None,
        "display": DISPLAY,
        "xdg_runtime_dir": XDG_RUNTIME_DIR,
    }
    return status


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entrypoint (for testing)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Norax Computer Use Toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("screenshot", help="Capture screenshot")
    p.add_argument("--output", "-o", default="/tmp/norax-screen.png", help="Output path")
    p.add_argument("--mode", choices=("auto", "wayland", "x11"), default="auto")

    p = sub.add_parser("click", help="Click at coordinates")
    p.add_argument("x", type=int)
    p.add_argument("y", type=int)
    p.add_argument("--button", "-b", type=int, default=1)

    p = sub.add_parser("doubleclick", help="Double-click at coordinates")
    p.add_argument("x", type=int)
    p.add_argument("y", type=int)

    p = sub.add_parser("move", help="Move cursor")
    p.add_argument("x", type=int)
    p.add_argument("y", type=int)

    p = sub.add_parser("drag", help="Drag from point to point")
    p.add_argument("x1", type=int)
    p.add_argument("y1", type=int)
    p.add_argument("x2", type=int)
    p.add_argument("y2", type=int)

    p = sub.add_parser("scroll", help="Scroll")
    p.add_argument("amount", type=int, default=3)

    p = sub.add_parser("type", help="Type text")
    p.add_argument("text", nargs="+")
    p.add_argument("--delay", type=int, default=12)

    p = sub.add_parser("key", help="Press key")
    p.add_argument("keyname")

    p = sub.add_parser("clipboard-get", help="Get clipboard")
    p = sub.add_parser("clipboard-set", help="Set clipboard")
    p.add_argument("text", nargs="+")

    p = sub.add_parser("info", help="Show toolkit info")

    args = parser.parse_args()

    if args.command == "screenshot":
        r = screenshot(args.output, args.mode)
    elif args.command == "click":
        r = click(args.x, args.y, args.button)
    elif args.command == "doubleclick":
        r = doubleclick(args.x, args.y)
    elif args.command == "move":
        r = move(args.x, args.y)
    elif args.command == "drag":
        r = drag(args.x1, args.y1, args.x2, args.y2)
    elif args.command == "scroll":
        r = scroll(args.amount)
    elif args.command == "type":
        r = type_text(" ".join(args.text), args.delay)
    elif args.command == "key":
        r = key_press(args.keyname)
    elif args.command == "clipboard-get":
        r = clipboard_get()
    elif args.command == "clipboard-set":
        r = clipboard_set(" ".join(args.text))
    elif args.command == "info":
        r = info()
    else:
        r = {"ok": False, "error": "Unknown command"}

    print(json.dumps(r, indent=2))
    if r.get("ok") is not True:
        raise SystemExit(1)
