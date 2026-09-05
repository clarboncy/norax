#!/usr/bin/env python3
"""Desktop screenshot indexing and coordinate-action helper.

3-pass detection pipeline:
  Pass 1: Sparse-text OCR (2x CLAHE invert, PSM 11) → text blocks, labels, paragraphs
  Pass 2: Button/widget detection (contour + 4x single-char OCR, PSM 10) → buttons, icons, controls
  Pass 3: Structure detection (edge morphology) → panels, dividers, windows, frames

Agent workflow:
  dmap snap                        # capture + full 3-pass index (~3s)
  dmap find "Calculator"           # fuzzy search: text + buttons + structure
  dmap click s42                   # click element via ydotool
  dmap type s42 "hello"            # click + type
  dmap read                        # full element dump (for context injection)
  dmap region 100,200,500,400      # elements in bounding box
  dmap get s42                     # single element JSON
  dmap nearest 800,400 button      # closest element of type to point
  dmap windows                     # detected window frames only
  dmap buttons                     # all clickable controls
  dmap key enter                   # press keyboard key
  dmap clickxy 800 400             # absolute coordinate click
  dmap drag s10 s20                # drag from one element to another
  dmap scroll up 3                 # scroll wheel
  dmap wait "Save complete" 10     # poll until text appears (max 10s)
  dmap diff                        # compare current screen to last snap

"""

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract

DMAP_DIR = Path("/tmp/dmap")
INDEX_FILE = DMAP_DIR / "index.json"
PREV_INDEX_FILE = DMAP_DIR / "prev_index.json"
SCREENSHOT_FILE = DMAP_DIR / "screen.png"
PREV_SCREENSHOT = DMAP_DIR / "prev_screen.png"
ANNOTATED_FILE = DMAP_DIR / "annotated.png"
MAX_INDEX_ELEMENTS = 2_000
DEFAULT_MAX_BUTTON_OCR = 32


# ═══════════════════════════════════════════════════════════════
# SCREENSHOT CAPTURE
# ═══════════════════════════════════════════════════════════════


def _ensure_session_env():
    """Auto-detect Wayland/X11 session environment when running from SSH/cron.

    Strategy:
    1. If WAYLAND_DISPLAY + DBUS already set -> skip
    2. Probe /proc/*/environ for any process with WAYLAND_DISPLAY
    3. If no process has it (compositor itself), scan XDG_RUNTIME_DIR for wayland-* sockets
    4. Ensure DBUS_SESSION_BUS_ADDRESS is set
    """
    has_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
    has_dbus = bool(os.environ.get("DBUS_SESSION_BUS_ADDRESS"))

    # Respect an explicitly configured display so perception and input operate
    # on the same desktop. Auto-select only when there is exactly one plausible
    # X socket; choosing an arbitrary session can capture and act on different
    # desktops.
    if not os.environ.get("DISPLAY") and os.environ.get("NORAX_DISPLAY"):
        os.environ["DISPLAY"] = os.environ["NORAX_DISPLAY"]
    if not os.environ.get("DISPLAY"):
        detected_displays = [
            f":{number}" for number in (0, 1, 99) if Path(f"/tmp/.X11-unix/X{number}").exists()
        ]
        if len(detected_displays) == 1:
            os.environ["DISPLAY"] = detected_displays[0]

    if has_wayland and has_dbus:
        return

    uid = os.getuid()
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{uid}")

    # Always ensure XDG_RUNTIME_DIR
    if not os.environ.get("XDG_RUNTIME_DIR"):
        os.environ["XDG_RUNTIME_DIR"] = xdg_runtime

    # Strategy A: scan /proc for any process that has WAYLAND_DISPLAY
    if not has_wayland:
        try:
            r = subprocess.run(["pgrep", "-u", str(uid)], capture_output=True, text=True, timeout=2)
            for pid in r.stdout.strip().split()[:100]:
                try:
                    env_data = (
                        open(f"/proc/{pid}/environ", "rb").read().decode("utf-8", errors="ignore")
                    )
                    if "WAYLAND_DISPLAY=" in env_data:
                        env_vars = dict(
                            item.split("=", 1) for item in env_data.split("\0") if "=" in item
                        )
                        wl = env_vars.get("WAYLAND_DISPLAY", "")
                        if wl and os.path.exists(os.path.join(xdg_runtime, wl)):
                            os.environ["WAYLAND_DISPLAY"] = wl
                            for k in [
                                "DISPLAY",
                                "DBUS_SESSION_BUS_ADDRESS",
                                "XDG_SESSION_TYPE",
                                "XDG_CURRENT_DESKTOP",
                            ]:
                                if k in env_vars and not os.environ.get(k):
                                    os.environ[k] = env_vars[k]
                            has_wayland = True
                            break
                except (PermissionError, FileNotFoundError, ValueError):
                    continue
        except Exception:
            pass

    # Strategy B: directly scan for wayland sockets (compositor won't have it in its own env)
    if not has_wayland:
        try:
            for name in sorted(os.listdir(xdg_runtime)):
                if (
                    name.startswith("wayland-")
                    and not name.endswith(".lock")
                    and "-renderD" not in name
                ):
                    sock_path = os.path.join(xdg_runtime, name)
                    if os.path.exists(sock_path) and not os.path.isfile(sock_path):
                        os.environ["WAYLAND_DISPLAY"] = name
                        has_wayland = True
                        break
        except (PermissionError, FileNotFoundError):
            pass

    # Ensure DBUS
    if not has_dbus:
        dbus_path = os.path.join(xdg_runtime, "bus")
        if os.path.exists(dbus_path):
            os.environ["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={dbus_path}"


def capture() -> np.ndarray | None:
    """Capture full desktop. Returns BGR image or None."""
    _ensure_session_env()
    DMAP_DIR.mkdir(parents=True, exist_ok=True)
    candidate = DMAP_DIR / "screen.next.png"
    candidate.unlink(missing_ok=True)

    def commit_candidate() -> np.ndarray | None:
        img = cv2.imread(str(candidate))
        if img is None:
            candidate.unlink(missing_ok=True)
            return None
        if PREV_SCREENSHOT.exists():
            PREV_SCREENSHOT.unlink()
        if SCREENSHOT_FILE.exists():
            SCREENSHOT_FILE.rename(PREV_SCREENSHOT)
        os.replace(candidate, SCREENSHOT_FILE)
        return img

    # Prefer direct, non-interactive capture.  Desktop portals can display a
    # permission dialog, so that fallback is explicitly opt-in.
    display = os.environ.get("DISPLAY", "")
    env = os.environ.copy()
    if display:
        env["DISPLAY"] = display
    capture_tools: list[tuple[str, list[str], dict[str, str]]] = []
    if os.environ.get("WAYLAND_DISPLAY"):
        capture_tools.append(("grim", [], env))
    if display:
        capture_tools.append(("scrot", ["-o"], env))
    # A fallback display may be a different user's/session's desktop. Only use
    # the automation Xvfb when the operator explicitly requests that behavior.
    allow_xvfb_fallback = os.environ.get("NORAX_DMAP_XVFB_FALLBACK", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if allow_xvfb_fallback and display != ":99" and Path("/tmp/.X11-unix/X99").exists():
        env99 = os.environ.copy()
        env99["DISPLAY"] = ":99"
        capture_tools.append(("scrot", ["-o"], env99))
    for tool, args, capture_env in capture_tools:
        try:
            subprocess.run(
                [tool] + args + [str(candidate)],
                capture_output=True,
                timeout=5,
                check=True,
                env=capture_env,
            )
            img = commit_candidate()
            if img is not None:
                return img
        except Exception:
            candidate.unlink(missing_ok=True)

    if os.environ.get("NORAX_DMAP_PORTAL_CAPTURE", "").lower() in {"1", "true", "yes", "on"}:
        try:
            before = set(glob.glob("/tmp/screenshot-*.png"))
            result = subprocess.run(
                [
                    "gdbus",
                    "call",
                    "--session",
                    "--dest",
                    "org.freedesktop.portal.Desktop",
                    "--object-path",
                    "/org/freedesktop/portal/desktop",
                    "--method",
                    "org.freedesktop.portal.Screenshot.Screenshot",
                    "",
                    "{}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                time.sleep(0.4)
                for portal_path in set(glob.glob("/tmp/screenshot-*.png")) - before:
                    img = cv2.imread(portal_path)
                    os.unlink(portal_path)
                    if img is not None and cv2.imwrite(str(candidate), img):
                        committed = commit_candidate()
                        if committed is not None:
                            return committed
        except Exception:
            candidate.unlink(missing_ok=True)
    return None


# ═══════════════════════════════════════════════════════════════
# PASS 1: SPARSE TEXT OCR
# ═══════════════════════════════════════════════════════════════


def pass1_text(gray: np.ndarray, scale: float = 2.0) -> list[dict]:
    """Full-page sparse text: 2x upscale + CLAHE + invert + PSM 11."""
    scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(scaled)
    inverted = cv2.bitwise_not(enhanced)

    data = pytesseract.image_to_data(
        inverted, output_type=pytesseract.Output.DICT, config="--psm 11 --oem 3"
    )

    # Group words → lines
    lines: dict[tuple, list] = {}
    for i in range(len(data["text"])):
        t = data["text"][i].strip()
        c = int(data["conf"][i])
        if not t or c < 35:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(
            {
                "text": t,
                "conf": c,
                "x": int(data["left"][i] / scale),
                "y": int(data["top"][i] / scale),
                "w": int(data["width"][i] / scale),
                "h": int(data["height"][i] / scale),
            }
        )

    elements = []
    for key in sorted(lines):
        words = sorted(lines[key], key=lambda w: w["x"])
        text = " ".join(w["text"] for w in words)
        if len(text.strip()) < 2:
            continue
        x = min(w["x"] for w in words)
        y = min(w["y"] for w in words)
        x2 = max(w["x"] + w["w"] for w in words)
        y2 = max(w["y"] + w["h"] for w in words)
        avg = sum(w["conf"] for w in words) / len(words)
        elements.append(
            {
                "type": "text",
                "bbox": [x, y, x2 - x, y2 - y],
                "text": text[:150],
                "conf": round(avg),
            }
        )
    return elements


# ═══════════════════════════════════════════════════════════════
# PASS 2: BUTTON / WIDGET DETECTION
# ═══════════════════════════════════════════════════════════════


def _ocr_single_button(args: tuple) -> dict | None:
    """OCR a single button ROI. Designed for multiprocessing.Pool."""
    x, y, bw, bh, gray_bytes, gray_shape = args
    gray = np.frombuffer(gray_bytes, dtype=np.uint8).reshape(gray_shape)
    interior = gray[y + 2 : y + bh - 2, x + 2 : x + bw - 2]
    if interior.size == 0:
        return None
    big = cv2.resize(interior, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    inv = cv2.bitwise_not(big)
    text = ""
    for psm in ("10", "7"):
        t = pytesseract.image_to_string(inv, config=f"--psm {psm} --oem 3").strip()
        if t and 1 <= len(t) <= 10:
            text = t
            break
    aspect = bw / max(bh, 1)
    if text:
        etype = "key" if len(text) <= 2 and 0.5 < aspect < 2.0 and bw * bh < 3000 else "button"
    elif bw < 40 and bh < 40 and 0.5 < aspect < 2.0:
        etype = "icon"
    else:
        etype = "control"
    elem: dict[str, Any] = {"type": etype, "bbox": [x, y, bw, bh]}
    if text:
        elem["text"] = text
    return elem


def pass2_buttons(gray: np.ndarray, text_elems: list[dict]) -> list[dict]:
    """Find small controls via contours and bounded sequential per-ROI OCR."""

    h, w = gray.shape

    # Build text occupancy mask
    tmask = np.zeros((h, w), dtype=bool)
    for e in text_elems:
        bx, by, bw, bh = e["bbox"]
        p = 3
        tmask[max(0, by - p) : min(h, by + bh + p), max(0, bx - p) : min(w, bx + bw + p)] = True

    # Multi-threshold edge detection
    edges = cv2.bitwise_or(cv2.Canny(gray, 15, 80), cv2.Canny(gray, 40, 140))
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    seen: set = set()
    candidates: list[tuple] = []

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < 250 or bw < 14 or bh < 12 or bw > 200 or bh > 80:
            continue
        key = (x // 4, y // 4, bw // 4, bh // 4)
        if key in seen:
            continue
        seen.add(key)

        # Skip text-covered
        roi_mask = tmask[max(0, y) : min(h, y + bh), max(0, x) : min(w, x + bw)]
        if roi_mask.size > 0 and roi_mask.sum() / roi_mask.size > 0.5:
            continue

        interior = gray[y + 2 : y + bh - 2, x + 2 : x + bw - 2]
        if interior.size == 0:
            continue
        if float(np.var(interior)) > 2500:
            continue

        # Rectangularity filter: reject organic blobs
        contour_area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        rectangularity = contour_area / area if area > 0 else 0
        straightness = 2 * (bw + bh) / peri if peri > 0 else 0
        if rectangularity < 0.6 or straightness < 0.55:
            continue

        candidates.append((x, y, bw, bh))

    # Per-ROI tesseract processes are expensive. Accessibility/CDP are the
    # preferred structured tiers, so bound this visual fallback rather than
    # allowing a noisy screen to spawn hundreds of OCR subprocesses.
    try:
        max_button_ocr = int(
            os.environ.get("NORAX_DMAP_MAX_BUTTON_OCR", str(DEFAULT_MAX_BUTTON_OCR))
        )
    except ValueError:
        max_button_ocr = DEFAULT_MAX_BUTTON_OCR
    max_button_ocr = max(0, min(max_button_ocr, 200))
    candidates.sort(key=lambda bbox: bbox[2] * bbox[3], reverse=True)
    candidates = candidates[:max_button_ocr]
    gray_bytes = gray.tobytes()
    gray_shape = gray.shape
    args_list = [(x, y, bw, bh, gray_bytes, gray_shape) for x, y, bw, bh in candidates]

    results: list[dict] = []
    if args_list:
        # Sequential OCR — avoids fork/thread deadlocks with tesseract
        results = [r for r in (_ocr_single_button(a) for a in args_list) if r is not None]

    # NMS
    results.sort(key=lambda e: e["bbox"][2] * e["bbox"][3], reverse=True)
    filtered: list[dict] = []
    for elem in results:
        ex, ey, ew, eh = elem["bbox"]
        overlap = False
        for fe in filtered:
            fx, fy, fw, fh = fe["bbox"]
            ix1, iy1 = max(ex, fx), max(ey, fy)
            ix2, iy2 = min(ex + ew, fx + fw), min(ey + eh, fy + fh)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter > 0.35 * min(ew * eh, fw * fh):
                    overlap = True
                    break
        if not overlap:
            filtered.append(elem)

    return filtered


# ═══════════════════════════════════════════════════════════════
# PASS 3: STRUCTURE DETECTION
# ═══════════════════════════════════════════════════════════════


def pass3_structure(gray: np.ndarray, text_elems: list[dict], btn_elems: list[dict]) -> list[dict]:
    """Detect panels, windows, dividers, frames via large-contour analysis."""
    h, w = gray.shape

    # Occupied mask from passes 1+2
    occ = np.zeros((h, w), dtype=bool)
    for e in text_elems + btn_elems:
        bx, by, bw, bh = e["bbox"]
        occ[max(0, by) : min(h, by + bh), max(0, bx) : min(w, bx + bw)] = True

    edges = cv2.Canny(gray, 25, 100)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    elements: list[dict] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < 15000:
            continue
        if bw > w * 0.95 and bh > h * 0.95:
            continue

        # Skip structures whose area is already mostly described by text and
        # controls; emitting them adds noise without new actionable evidence.
        roi = occ[max(0, y) : min(h, y + bh), max(0, x) : min(w, x + bw)]
        if roi.sum() / max(roi.size, 1) > 0.8:
            continue

        # Classify
        if bh < 8 and bw > 80:
            etype = "divider"
        elif bw > 200 and bh > 200:
            # Check if it looks like a window (has title bar = text near top)
            has_title = any(
                e["type"] == "text" and abs(e["bbox"][1] - y) < 40 and x <= e["bbox"][0] <= x + bw
                for e in text_elems
            )
            etype = "window" if has_title else "panel"
        else:
            etype = "panel"

        elements.append(
            {
                "type": etype,
                "bbox": [x, y, bw, bh],
            }
        )

    # NMS
    elements.sort(key=lambda e: e["bbox"][2] * e["bbox"][3], reverse=True)
    filtered: list[dict[str, Any]] = []
    for elem in elements:
        ex, ey, ew, eh = elem["bbox"]
        overlap = False
        for fe in filtered:
            fx, fy, fw, fh = fe["bbox"]
            ix1, iy1 = max(ex, fx), max(ey, fy)
            ix2, iy2 = min(ex + ew, fx + fw), min(ey + eh, fy + fh)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter > 0.5 * min(ew * eh, fw * fh):
                    overlap = True
                    break
        if not overlap:
            filtered.append(elem)
    return filtered


# ═══════════════════════════════════════════════════════════════
# INDEX BUILDER
# ═══════════════════════════════════════════════════════════════


def build_index(img: np.ndarray) -> dict:
    """3-pass pipeline → unified sorted index with refs."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    t0 = time.monotonic()
    text_elems = pass1_text(gray)
    t1 = time.monotonic()
    btn_elems = pass2_buttons(gray, text_elems)
    t2 = time.monotonic()
    struct_elems = pass3_structure(gray, text_elems, btn_elems)
    t3 = time.monotonic()

    all_elems = text_elems + btn_elems + struct_elems
    # Sort: top→bottom in 20px bands, left→right
    all_elems.sort(key=lambda e: (e["bbox"][1] // 20, e["bbox"][0]))
    all_elems = all_elems[:MAX_INDEX_ELEMENTS]
    for i, e in enumerate(all_elems):
        e["ref"] = f"s{i + 1}"

    # Screen hash for diff
    small = cv2.resize(gray, (64, 36))
    screen_hash = hashlib.md5(small.tobytes()).hexdigest()[:12]

    return {
        "screen": [w, h],
        "hash": screen_hash,
        "captured_at": datetime.now(UTC).isoformat(),
        "counts": {
            "total": len(all_elems),
            "text": sum(element["type"] == "text" for element in all_elems),
            "buttons": sum(
                element["type"] in {"button", "key", "icon", "control"} for element in all_elems
            ),
            "structure": sum(
                element["type"] in {"window", "panel", "divider"} for element in all_elems
            ),
        },
        "timing": {
            "pass1_text": round(t1 - t0, 2),
            "pass2_buttons": round(t2 - t1, 2),
            "pass3_structure": round(t3 - t2, 2),
            "total": round(t3 - t0, 2),
        },
        "elements": all_elems,
    }


# ═══════════════════════════════════════════════════════════════
# SEARCH + QUERY
# ═══════════════════════════════════════════════════════════════


def search(
    index: dict, query: str, types: list[str] | None = None, case_insensitive: bool = True
) -> list[dict]:
    """Fuzzy search by text. Supports regex. Optional type filter."""
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        pat = re.compile(query, flags)
    except re.error:
        pat = re.compile(re.escape(query), flags)

    results = []
    for e in index["elements"]:
        if types and e["type"] not in types:
            continue
        text = e.get("text", "")
        if pat.search(text):
            results.append(e)
    return results


def region(
    index: dict, x: int, y: int, w: int, h: int, types: list[str] | None = None
) -> list[dict]:
    """Elements whose center falls within bounding box."""
    results = []
    for e in index["elements"]:
        if types and e["type"] not in types:
            continue
        ex, ey, ew, eh = e["bbox"]
        cx, cy = ex + ew // 2, ey + eh // 2
        if x <= cx <= x + w and y <= cy <= y + h:
            results.append(e)
    return results


def get_elem(index: dict, ref: str) -> dict | None:
    return next((e for e in index["elements"] if e["ref"] == ref), None)


def nearest(index: dict, x: int, y: int, etype: str | None = None) -> dict | None:
    best, best_d = None, float("inf")
    for e in index["elements"]:
        if etype and e["type"] != etype:
            continue
        ex, ey, ew, eh = e["bbox"]
        d = ((ex + ew / 2 - x) ** 2 + (ey + eh / 2 - y) ** 2) ** 0.5
        if d < best_d:
            best, best_d = e, d
    return best


def diff_indexes(old: dict, new: dict) -> dict:
    """Compare two indexes for changes."""
    old_texts = {e.get("text", ""): e for e in old["elements"] if e.get("text")}
    new_texts = {e.get("text", ""): e for e in new["elements"] if e.get("text")}

    added = [new_texts[t] for t in set(new_texts) - set(old_texts)]
    removed = [old_texts[t] for t in set(old_texts) - set(new_texts)]

    return {
        "screen_changed": old.get("hash") != new.get("hash"),
        "old_count": old["counts"]["total"],
        "new_count": new["counts"]["total"],
        "added": len(added),
        "removed": len(removed),
        "added_text": [e.get("text", "")[:60] for e in added[:20]],
        "removed_text": [e.get("text", "")[:60] for e in removed[:20]],
    }


# ═══════════════════════════════════════════════════════════════
# INTERACTION (ydotool)
# ═══════════════════════════════════════════════════════════════


def _ydotool(*args: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Run ydotool once and return its observed status.

    Privilege escalation is never attempted implicitly.  A missing daemon,
    socket permission problem, or command timeout is surfaced to the caller.
    """
    try:
        result = subprocess.run(
            ["ydotool", *args], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    detail = (result.stderr or result.stdout).strip()
    return result.returncode == 0, detail


def click_xy(x: int, y: int, button: str = "left") -> bool:
    button_codes = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
    btn_code = button_codes.get(button)
    if btn_code is None:
        print(f"UNSUPPORTED_BUTTON: {button}", file=sys.stderr)
        return False
    moved, detail = _ydotool("mousemove", "--absolute", str(x), str(y))
    if not moved:
        print(f"MOVE_FAILED: {detail}", file=sys.stderr)
        return False
    time.sleep(0.05)
    clicked, detail = _ydotool("click", btn_code)
    if not clicked:
        print(f"CLICK_FAILED: {detail}", file=sys.stderr)
    return clicked


def click_ref(index: dict, ref: str, button: str = "left") -> bool:
    e = get_elem(index, ref)
    if not e:
        print(f"NOT_FOUND: {ref}", file=sys.stderr)
        return False
    x, y, w, h = e["bbox"]
    cx, cy = x + w // 2, y + h // 2
    if not click_xy(cx, cy, button):
        return False
    label = e.get("text", e["type"])[:30]
    print(f'CLICKED {ref} at ({cx},{cy}) "{label}"', file=sys.stderr)
    return True


def double_click_ref(index: dict, ref: str) -> bool:
    e = get_elem(index, ref)
    if not e:
        return False
    x, y, w, h = e["bbox"]
    cx, cy = x + w // 2, y + h // 2
    moved, _ = _ydotool("mousemove", "--absolute", str(cx), str(cy))
    if not moved:
        return False
    time.sleep(0.03)
    first, _ = _ydotool("click", "0xC0")
    if not first:
        return False
    time.sleep(0.08)
    second, _ = _ydotool("click", "0xC0")
    return second


def type_text(text: str) -> bool:
    if len(text) > 16_384:
        print("TYPE_FAILED: text exceeds 16384 characters", file=sys.stderr)
        return False
    ok, detail = _ydotool("type", "--", text)
    if not ok:
        print(f"TYPE_FAILED: {detail}", file=sys.stderr)
    return ok


def press_key(key: str) -> bool:
    KEY_MAP = {
        "enter": "28:1 28:0",
        "return": "28:1 28:0",
        "tab": "15:1 15:0",
        "escape": "1:1 1:0",
        "esc": "1:1 1:0",
        "backspace": "14:1 14:0",
        "delete": "111:1 111:0",
        "up": "103:1 103:0",
        "down": "108:1 108:0",
        "left": "105:1 105:0",
        "right": "106:1 106:0",
        "space": "57:1 57:0",
        "home": "102:1 102:0",
        "end": "107:1 107:0",
        "super": "125:1 125:0",
        "ctrl+a": "29:1 30:1 30:0 29:0",
        "ctrl+c": "29:1 46:1 46:0 29:0",
        "ctrl+v": "29:1 47:1 47:0 29:0",
        "ctrl+z": "29:1 44:1 44:0 29:0",
        "ctrl+s": "29:1 31:1 31:0 29:0",
        "ctrl+w": "29:1 17:1 17:0 29:0",
        "ctrl+t": "29:1 20:1 20:0 29:0",
        "alt+tab": "56:1 15:1 15:0 56:0",
        "alt+f4": "56:1 62:1 62:0 56:0",
        "f1": "59:1 59:0",
        "f2": "60:1 60:0",
        "f3": "61:1 61:0",
        "f4": "62:1 62:0",
        "f5": "63:1 63:0",
        "f11": "87:1 87:0",
    }
    mapped = KEY_MAP.get(key.lower(), "")
    if mapped:
        ok, detail = _ydotool("key", *mapped.split())
    else:
        # Raw ydotool key arguments are numeric keycodes, not arbitrary names.
        print(f"UNKNOWN_KEY: {key}", file=sys.stderr)
        return False
    if not ok:
        print(f"KEY_FAILED: {detail}", file=sys.stderr)
    return ok


def scroll(direction: str = "down", clicks: int = 3) -> bool:
    if direction not in {"up", "down"}:
        print(f"UNKNOWN_SCROLL_DIRECTION: {direction}", file=sys.stderr)
        return False
    if clicks < 0:
        print("SCROLL_CLICKS_MUST_BE_NONNEGATIVE", file=sys.stderr)
        return False
    button = "4" if direction == "up" else "5"
    deadline = time.monotonic() + 8.0
    for _ in range(min(clicks, 20)):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print("SCROLL_FAILED: scroll deadline exceeded", file=sys.stderr)
            return False
        ok, detail = _ydotool("click", button, timeout=min(0.75, remaining))
        if not ok:
            print(f"SCROLL_FAILED: {detail}", file=sys.stderr)
            return False
    return True


def drag(index: dict, from_ref: str, to_ref: str) -> bool:
    e1, e2 = get_elem(index, from_ref), get_elem(index, to_ref)
    if not e1 or not e2:
        return False
    x1 = e1["bbox"][0] + e1["bbox"][2] // 2
    y1 = e1["bbox"][1] + e1["bbox"][3] // 2
    x2 = e2["bbox"][0] + e2["bbox"][2] // 2
    y2 = e2["bbox"][1] + e2["bbox"][3] // 2
    moved, _ = _ydotool("mousemove", "--absolute", str(x1), str(y1))
    if not moved:
        return False
    time.sleep(0.05)
    # Press left button
    pressed, _ = _ydotool("click", "0x40")  # button down
    if not pressed:
        return False
    time.sleep(0.1)
    moved, _ = _ydotool("mousemove", "--absolute", str(x2), str(y2))
    if not moved:
        _ydotool("click", "0x80")
        return False
    time.sleep(0.05)
    released, _ = _ydotool("click", "0x80")  # button up
    return released


# ═══════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════

COLORS = {
    "text": (0, 255, 0),
    "button": (0, 200, 255),
    "key": (0, 255, 255),
    "icon": (255, 0, 255),
    "control": (255, 200, 0),
    "window": (255, 100, 100),
    "panel": (100, 100, 255),
    "divider": (150, 150, 0),
}


def fmt_compact(elems: list[dict], screen=None, header: str = "") -> str:
    lines = []
    if header:
        lines.append(header)
    if screen:
        lines.append(f"DESKTOP:{screen[0]}x{screen[1]} | {len(elems)} elements")
    lines.append("")
    for e in elems:
        r, t = e["ref"], e["type"]
        x, y, w, h = e["bbox"]
        txt = e.get("text", "")
        c = e.get("conf", 0)
        parts = [f"[{r}]", t, f"({x},{y} {w}x{h})"]
        if txt:
            parts.append(f'"{txt}"')
        if c and c < 55:
            parts.append(f"~{c}%")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def annotate(img: np.ndarray, elems: list[dict], path: str):
    out = img.copy()
    for e in elems:
        x, y, w, h = e["bbox"]
        color = COLORS.get(e["type"], (200, 200, 200))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = e["ref"]
        if e.get("text"):
            label += f" {e['text'][:12]}"
        cv2.putText(
            out,
            label,
            (x + 2, max(y - 4, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(path, out)


# ═══════════════════════════════════════════════════════════════
# FILE I/O
# ═══════════════════════════════════════════════════════════════


def save_index(idx: dict):
    DMAP_DIR.mkdir(parents=True, exist_ok=True)
    temporary = INDEX_FILE.with_name(f"{INDEX_FILE.name}.tmp-{os.getpid()}-{time.time_ns()}")
    previous_temporary = PREV_INDEX_FILE.with_name(
        f"{PREV_INDEX_FILE.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        temporary.write_text(json.dumps(idx, separators=(",", ":")), encoding="utf-8")
        if INDEX_FILE.exists():
            shutil.copyfile(INDEX_FILE, previous_temporary)
            os.replace(previous_temporary, PREV_INDEX_FILE)
        os.replace(temporary, INDEX_FILE)
    finally:
        temporary.unlink(missing_ok=True)
        previous_temporary.unlink(missing_ok=True)


def load_index(*, require_fresh: bool = False) -> dict:
    if not INDEX_FILE.exists():
        print("No index. Run: dmap snap", file=sys.stderr)
        sys.exit(1)
    try:
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Invalid index: {exc}. Run: dmap snap", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, dict) or not isinstance(data.get("elements"), list):
        print("Invalid index structure. Run: dmap snap", file=sys.stderr)
        sys.exit(1)
    if require_fresh:
        captured_at = data.get("captured_at")
        try:
            captured = datetime.fromisoformat(str(captured_at))
            age = (datetime.now(captured.tzinfo) - captured).total_seconds()
            max_age = float(os.environ.get("NORAX_DMAP_MAX_ACTION_AGE_SECONDS", "60"))
            if max_age <= 0:
                raise ValueError("maximum action age must be positive")
        except (TypeError, ValueError) as exc:
            print(f"Invalid index timestamp: {exc}. Run: dmap snap", file=sys.stderr)
            sys.exit(1)
        if age < -300 or age > max_age:
            print(f"Stale index ({age:.1f}s old). Run: dmap snap", file=sys.stderr)
            sys.exit(1)
    return data


def load_prev_index() -> dict | None:
    if PREV_INDEX_FILE.exists():
        with open(PREV_INDEX_FILE) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

HELP = """dmap v2 — Desktop Map (3-pass computer vision)

Capture + Index:
  snap [--annotate]           Capture desktop, run 3-pass detection
  read [--json]               Dump full element map

Search + Query:
  find <query> [--type X]     Fuzzy search text+buttons (regex ok)
  region <x,y,w,h>            Elements in bounding box
  get <ref>                   Single element detail (JSON)
  nearest <x,y> [type]        Closest element to point
  windows                     Detected window frames
  buttons                     All clickable controls (button+key+icon)

Interact:
  click <ref> [right|middle]  Click element center
  dblclick <ref>              Double-click element
  type <ref> <text>           Click + type text
  key <keyname>               Press key (enter/tab/esc/ctrl+c/alt+tab/super...)
  clickxy <x> <y>             Click absolute coordinates
  drag <from_ref> <to_ref>    Drag between elements
  scroll <up|down> [clicks]   Scroll mouse wheel

Monitor:
  diff                        Compare current vs previous snap
  wait <text> [timeout_s]     Poll-snap until text appears
  age                         Time since last snap
  count                       Element type summary"""


def main():
    args = sys.argv[1:]
    if not args:
        print(HELP)
        return
    cmd = args[0]

    # ── snap ──
    if cmd == "snap":
        t0 = time.monotonic()
        img = capture()
        if img is None:
            print("ERROR: no screenshot method", file=sys.stderr)
            sys.exit(1)
        t_cap = time.monotonic() - t0
        eprint(f"Captured ({t_cap:.1f}s) ", end="", flush=True)

        idx = build_index(img)
        save_index(idx)
        c = idx["counts"]
        tm = idx["timing"]
        eprint(
            f"Indexed ({tm['total']:.1f}s): "
            f"{c['text']}t + {c['buttons']}b + {c['structure']}s = {c['total']}"
        )

        if "--annotate" in args:
            annotate(img, idx["elements"], str(ANNOTATED_FILE))
            eprint(f"Annotated: {ANNOTATED_FILE}")

        if "--quiet" not in args and "-q" not in args:
            print(fmt_compact(idx["elements"], idx["screen"]))

    # ── read ──
    elif cmd == "read":
        idx = load_index()
        if "--json" in args:
            print(json.dumps(idx, indent=2))
        else:
            print(fmt_compact(idx["elements"], idx["screen"]))

    # ── find ──
    elif cmd == "find":
        if len(args) < 2:
            print("Usage: dmap find <query> [--type button|text|key|icon]", file=sys.stderr)
            sys.exit(1)
        query = args[1]
        types = None
        if "--type" in args:
            ti = args.index("--type")
            if ti + 1 < len(args):
                types = args[ti + 1].split(",")
        idx = load_index()
        results = search(idx, query, types)
        if results:
            print(fmt_compact(results))
        else:
            print(f'NOT_FOUND: "{query}"')

    # ── region ──
    elif cmd == "region":
        parts = args[1].split(",")
        x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        types = None
        if "--type" in args:
            ti = args.index("--type")
            if ti + 1 < len(args):
                types = args[ti + 1].split(",")
        idx = load_index()
        print(fmt_compact(region(idx, x, y, w, h, types)))

    # ── get ──
    elif cmd == "get":
        idx = load_index()
        e = get_elem(idx, args[1])
        print(json.dumps(e, indent=2) if e else f"NOT_FOUND: {args[1]}")

    # ── nearest ──
    elif cmd == "nearest":
        parts = args[1].split(",")
        x, y = int(parts[0]), int(parts[1])
        etype = args[2] if len(args) > 2 else None
        idx = load_index()
        e = nearest(idx, x, y, etype)
        print(json.dumps(e, indent=2) if e else "NOT_FOUND")

    # ── windows ──
    elif cmd == "windows":
        idx = load_index()
        wins = [e for e in idx["elements"] if e["type"] in ("window", "panel")]
        print(fmt_compact(wins, header="WINDOWS/PANELS:"))

    # ── buttons ──
    elif cmd == "buttons":
        idx = load_index()
        btns = [e for e in idx["elements"] if e["type"] in ("button", "key", "icon", "control")]
        print(fmt_compact(btns, header="CLICKABLE:"))

    # ── click ──
    elif cmd == "click":
        idx = load_index(require_fresh=True)
        button = args[2] if len(args) > 2 else "left"
        if not click_ref(idx, args[1], button):
            sys.exit(1)

    # ── dblclick ──
    elif cmd == "dblclick":
        idx = load_index(require_fresh=True)
        if not double_click_ref(idx, args[1]):
            sys.exit(1)

    # ── type ──
    elif cmd == "type":
        idx = load_index(require_fresh=True)
        if click_ref(idx, args[1]):
            time.sleep(0.12)
            if not type_text(" ".join(args[2:])):
                sys.exit(1)
        else:
            sys.exit(1)

    # ── key ──
    elif cmd == "key":
        if not press_key(args[1]):
            sys.exit(1)

    # ── clickxy ──
    elif cmd == "clickxy":
        if not click_xy(int(args[1]), int(args[2])):
            sys.exit(1)

    # ── drag ──
    elif cmd == "drag":
        idx = load_index(require_fresh=True)
        if not drag(idx, args[1], args[2]):
            sys.exit(1)

    # ── scroll ──
    elif cmd == "scroll":
        direction = args[1] if len(args) > 1 else "down"
        clicks = int(args[2]) if len(args) > 2 else 3
        if not scroll(direction, clicks):
            sys.exit(1)

    # ── diff ──
    elif cmd == "diff":
        idx = load_index()
        prev = load_prev_index()
        if not prev:
            print("No previous snap to compare")
            return
        d = diff_indexes(prev, idx)
        print(
            f"Changed: {d['screen_changed']} | "
            f"{d['old_count']}→{d['new_count']} elements | "
            f"+{d['added']} -{d['removed']}"
        )
        if d["added_text"]:
            print(f"Added: {d['added_text'][:10]}")
        if d["removed_text"]:
            print(f"Removed: {d['removed_text'][:10]}")

    # ── wait ──
    elif cmd == "wait":
        query = args[1]
        timeout = float(args[2]) if len(args) > 2 else 10
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            img = capture()
            if img is None:
                time.sleep(1)
                continue
            idx = build_index(img)
            results = search(idx, query)
            if results:
                save_index(idx)
                print(fmt_compact(results, header=f"FOUND after {time.monotonic() - t0:.1f}s:"))
                return
            time.sleep(1)
        print(f'TIMEOUT: "{query}" not found in {timeout}s')
        sys.exit(1)

    # ── age ──
    elif cmd == "age":
        idx = load_index()
        t = idx.get("captured_at", "")
        if t:
            try:
                ct = datetime.fromisoformat(t)
                age = (datetime.now(ct.tzinfo) - ct).total_seconds()
                print(f"{age:.0f}s ({age / 60:.1f}m) since {t}")
            except Exception:
                print(f"Captured: {t}")

    # ── count ──
    elif cmd == "count":
        idx = load_index()
        c = idx["counts"]
        print(
            f"Total:{c['total']} | text:{c['text']} buttons:{c['buttons']} structure:{c['structure']}"
        )
        tm = idx.get("timing", {})
        if tm:
            print(
                f"Timing: pass1={tm.get('pass1_text', 0)}s pass2={tm.get('pass2_buttons', 0)}s "
                f"pass3={tm.get('pass3_structure', 0)}s total={tm.get('total', 0)}s"
            )

    else:
        print(f"Unknown: {cmd}. Run 'dmap' for help.", file=sys.stderr)
        sys.exit(1)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


if __name__ == "__main__":
    main()
