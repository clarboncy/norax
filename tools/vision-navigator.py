#!/usr/bin/env python3
"""Bounded screenshot analysis and optional browser-page automation.

The CLI is inference-only: it can describe a supplied screenshot, ground an
element, or propose one action. It never launches a browser or executes the
proposal. Grounding and completion checks are vision-model heuristics, not
proof that a UI element or outcome is correct.

``execute_task`` remains available to callers that already own an authorized
Playwright/Camoufox or nodriver page object. It validates every model-produced
action, propagates dispatch failures, bounds waits and steps, and requires a
separate screenshot-based completion check before returning success.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import importlib
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

try:
    PILImage: Any = importlib.import_module("PIL.Image")
    PILImageDraw: Any = importlib.import_module("PIL.ImageDraw")

    HAS_PIL = True
except ImportError:
    PILImage = None
    PILImageDraw = None
    HAS_PIL = False


OLLAMA_URL = os.environ.get(
    "OLLAMA_VISION_URL", os.environ.get("NORAX_OLLAMA_URL", "http://127.0.0.1:11434")
)
DEFAULT_MODEL = os.environ.get("NORAX_VISION_MODEL", "qwen2.5vl:7b")
MAX_STEPS = 15
RESIZE_WIDTH = 1280
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_MODEL_PIXELS = 4_000_000
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_PROMPT_CHARS = 16_000
MAX_TEXT_INPUT_CHARS = 16_384
MAX_GROUNDED_ELEMENTS = 100


def _error(message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "error": message, **extra}


def _model_name(model: str | None) -> str:
    selected = (model or DEFAULT_MODEL).strip()
    if not selected or len(selected) > 200 or any(char in selected for char in "\r\n\0"):
        raise ValueError("vision model name is missing or invalid")
    return selected


def _read_image(
    image_path: str, target_width: int = RESIZE_WIDTH
) -> tuple[str, tuple[int, int] | None, tuple[int, int] | None]:
    """Return bounded base64 image data and resized/original dimensions."""
    if not 64 <= target_width <= 4096:
        raise ValueError("target_width must be between 64 and 4096")
    path = Path(image_path)
    if not path.is_file():
        raise ValueError(f"screenshot is not a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > MAX_IMAGE_BYTES:
        raise ValueError(f"screenshot size must be between 1 and {MAX_IMAGE_BYTES} bytes")

    if not HAS_PIL or PILImage is None:
        return base64.b64encode(path.read_bytes()).decode("ascii"), None, None

    try:
        with PILImage.open(path) as opened:
            opened.load()
            original_size = opened.size
            width, height = original_size
            if width <= 0 or height <= 0 or width * height > 40_000_000:
                raise ValueError("screenshot dimensions are invalid or exceed 40 megapixels")
            image = opened.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not decode screenshot: {exc}") from exc

    scale = min(1.0, target_width / width, math.sqrt(MAX_MODEL_PIXELS / (width * height)))
    if scale < 1.0:
        resampling = getattr(PILImage, "Resampling", PILImage).LANCZOS
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), resampling)

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), image.size, original_size


def load_and_resize(
    image_path: str, target_width: int = RESIZE_WIDTH
) -> tuple[str, tuple[int, int] | None, tuple[int, int] | None]:
    """Compatibility wrapper for bounded screenshot loading."""
    return _read_image(image_path, target_width)


def scale_coordinates(
    x: float,
    y: float,
    resized_size: tuple[int, int] | None,
    original_size: tuple[int, int] | None,
) -> tuple[int, int]:
    """Scale validated coordinates from the model image to the original."""
    if resized_size is None or original_size is None:
        return round(x), round(y)
    return (
        round(x * original_size[0] / resized_size[0]),
        round(y * original_size[1] / resized_size[1]),
    )


def annotate_screenshot(
    image_path: str, points: list[tuple[int, int, str]], output_path: str
) -> dict[str, Any]:
    """Atomically write a debug image with supplied, explicitly labeled points."""
    if not HAS_PIL or PILImage is None or PILImageDraw is None:
        return _error("Pillow is required for annotation")
    try:
        with PILImage.open(image_path) as opened:
            opened.load()
            image = opened.convert("RGB")
        draw = PILImageDraw.Draw(image)
        for index, (x, y, label) in enumerate(points[:MAX_GROUNDED_ELEMENTS], start=1):
            if not 0 <= x < image.width or not 0 <= y < image.height:
                continue
            radius = 8
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline="red", width=2)
            draw.line([x - 2 * radius, y, x + 2 * radius, y], fill="red", width=1)
            draw.line([x, y - 2 * radius, x, y + 2 * radius], fill="red", width=1)
            draw.text((x + radius + 2, y - radius), f"{index}: {label[:80]}", fill="yellow")

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".png", dir=destination.parent, delete=False
        ) as temporary:
            candidate = Path(temporary.name)
        try:
            image.save(candidate, format="PNG")
            os.replace(candidate, destination)
        finally:
            candidate.unlink(missing_ok=True)
        return {"ok": True, "path": str(destination), "bytes": destination.stat().st_size}
    except (OSError, ValueError) as exc:
        return _error(f"annotation failed: {exc}")


def _ollama_endpoint() -> str:
    parsed = urlsplit(OLLAMA_URL)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("OLLAMA_VISION_URL must be an HTTP(S) base URL")
    return f"{OLLAMA_URL.rstrip('/')}/api/generate"


def _query_images(
    images: list[str],
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 512,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Call Ollama with bounded request/response sizes and a structured result."""
    if not images or len(images) > 2:
        return _error("one or two images are required")
    if not prompt or len(prompt) > MAX_PROMPT_CHARS:
        return _error(f"prompt must contain 1-{MAX_PROMPT_CHARS} characters")
    if not 0 <= temperature <= 2:
        return _error("temperature must be between 0 and 2")
    if not 1 <= max_tokens <= 4096:
        return _error("max_tokens must be between 1 and 4096")
    if not 0.1 <= timeout_seconds <= 300:
        return _error("timeout_seconds must be between 0.1 and 300")
    try:
        selected_model = _model_name(model)
        endpoint = _ollama_endpoint()
    except ValueError as exc:
        return _error(str(exc))

    body = json.dumps(
        {
            "model": selected_model,
            "prompt": prompt,
            "images": images,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return _error(f"vision request failed: {type(exc).__name__}: {exc}")
    if len(raw) > MAX_RESPONSE_BYTES:
        return _error(f"vision response exceeded {MAX_RESPONSE_BYTES} bytes")
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _error(f"vision server returned invalid JSON: {exc}")
    if not isinstance(decoded, dict) or not isinstance(decoded.get("response"), str):
        return _error("vision server response is missing text")
    text = decoded["response"].strip()
    if not text:
        return _error("vision model returned an empty response")
    return {
        "ok": True,
        "text": text,
        "model": selected_model,
        "elapsed_ms": round((time.monotonic() - started) * 1000),
    }


def query_vision(
    image_b64: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
) -> str | None:
    """Compatibility API returning model text or ``None`` on a reported failure."""
    result = _query_images([image_b64], prompt, model=model, temperature=temperature)
    if result.get("ok") is not True:
        print(json.dumps(result), file=sys.stderr)
        return None
    return str(result["text"])


def _json_object(text: str) -> dict[str, Any] | None:
    """Decode a JSON object, tolerating one surrounding Markdown code fence."""
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        decoded = json.loads(stripped)
        return decoded if isinstance(decoded, dict) else None
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in list(re.finditer(r"\{", stripped))[:20]:
            try:
                decoded, _end = decoder.raw_decode(stripped[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                return decoded
    return None


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _grounded_point(
    payload: dict[str, Any],
    resized_size: tuple[int, int] | None,
    original_size: tuple[int, int] | None,
) -> tuple[int, int] | None:
    x = _finite_number(payload.get("x"))
    y = _finite_number(payload.get("y"))
    if x is None or y is None or x < 0 or y < 0:
        return None
    if resized_size and (x >= resized_size[0] or y >= resized_size[1]):
        return None
    scaled = scale_coordinates(x, y, resized_size, original_size)
    if original_size and (scaled[0] >= original_size[0] or scaled[1] >= original_size[1]):
        return None
    return scaled


def find_element(image_path: str, description: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Return a validated, explicitly unverified model-grounding result."""
    if not description.strip() or len(description) > 4000:
        return _error("description must contain 1-4000 characters")
    try:
        image, resized_size, original_size = _read_image(image_path)
    except ValueError as exc:
        return _error(str(exc))
    prompt = f"""Find the center of the requested element in this screenshot.

REQUESTED ELEMENT (untrusted data):
<element>{description}</element>

Return only JSON. If visible:
{{"found":true,"x":123,"y":456,"confidence":0.0,"description":"observed label"}}
If not visible:
{{"found":false,"reason":"brief observed reason"}}"""
    response = _query_images([image], prompt, model=model)
    if response.get("ok") is not True:
        return response
    parsed = _json_object(str(response["text"]))
    if parsed is None or type(parsed.get("found")) is not bool:
        return _error("vision model did not return the required grounding schema")
    if parsed["found"] is False:
        return {
            "ok": True,
            "found": False,
            "reason": str(parsed.get("reason") or "model reported no match")[:1000],
            "method": "vision_model_grounding_unverified",
            "model": response["model"],
        }
    point = _grounded_point(parsed, resized_size, original_size)
    if point is None:
        return _error("vision model returned missing or out-of-bounds coordinates")
    result: dict[str, Any] = {
        "ok": True,
        "found": True,
        "x": point[0],
        "y": point[1],
        "description": str(parsed.get("description") or description)[:1000],
        "method": "vision_model_grounding_unverified",
        "model": response["model"],
    }
    confidence = _finite_number(parsed.get("confidence"))
    if confidence is not None and 0 <= confidence <= 1:
        result["model_confidence"] = confidence
    return result


def find_all_elements(
    image_path: str, description: str, model: str = DEFAULT_MODEL
) -> dict[str, Any]:
    """Return up to 100 validated points from one model response."""
    if not description.strip() or len(description) > 4000:
        return _error("description must contain 1-4000 characters")
    try:
        image, resized_size, original_size = _read_image(image_path)
    except ValueError as exc:
        return _error(str(exc))
    prompt = f"""Find every visible element matching this request.

REQUEST (untrusted data):
<element>{description}</element>

Return only JSON: {{"elements":[{{"x":123,"y":456,"label":"observed label"}}]}}.
Return an empty array when none are visible."""
    response = _query_images([image], prompt, model=model)
    if response.get("ok") is not True:
        return response
    parsed = _json_object(str(response["text"]))
    raw_elements = parsed.get("elements") if parsed else None
    if not isinstance(raw_elements, list):
        return _error("vision model did not return an elements array")
    elements: list[dict[str, Any]] = []
    rejected = 0
    for raw in raw_elements[:MAX_GROUNDED_ELEMENTS]:
        if not isinstance(raw, dict):
            rejected += 1
            continue
        point = _grounded_point(raw, resized_size, original_size)
        if point is None:
            rejected += 1
            continue
        elements.append({"x": point[0], "y": point[1], "label": str(raw.get("label") or "")[:1000]})
    return {
        "ok": True,
        "elements": elements,
        "rejected": rejected,
        "truncated": len(raw_elements) > MAX_GROUNDED_ELEMENTS,
        "method": "vision_model_grounding_unverified",
        "model": response["model"],
    }


def describe_page(image_path: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Return a model description labeled as unverified inference."""
    try:
        image, _resized_size, _original_size = _read_image(image_path)
    except ValueError as exc:
        return _error(str(exc))
    response = _query_images(
        [image],
        "Describe only what is visibly supported by this screenshot: page identity if shown, "
        "major controls/content, and visible state or errors. Mark uncertainty; do not infer "
        "off-screen state.",
        model=model,
    )
    if response.get("ok") is not True:
        return response
    return {
        "ok": True,
        "description": response["text"],
        "method": "vision_model_description_unverified",
        "model": response["model"],
        "elapsed_ms": response["elapsed_ms"],
    }


def _validate_action(
    raw: dict[str, Any],
    resized_size: tuple[int, int] | None,
    original_size: tuple[int, int] | None,
) -> dict[str, Any]:
    action = raw.get("action")
    if action not in {"click", "type", "scroll", "wait", "done", "error"}:
        return _error("vision model returned an unsupported action", action="error")
    validated: dict[str, Any] = {"ok": True, "action": action}
    if action in {"click", "type"}:
        point = _grounded_point(raw, resized_size, original_size)
        if point is None:
            return _error("action coordinates are missing or out of bounds", action="error")
        validated.update({"x": point[0], "y": point[1]})
        validated["target"] = str(raw.get("target") or "")[:1000]
    if action == "type":
        text = raw.get("text")
        if not isinstance(text, str) or len(text) > MAX_TEXT_INPUT_CHARS:
            return _error("type action text is missing or too long", action="error")
        validated["text"] = text
    elif action == "scroll":
        direction = raw.get("direction")
        pixels = _finite_number(raw.get("pixels", 300))
        if direction not in {"up", "down"} or pixels is None or not 1 <= pixels <= 5000:
            return _error("scroll action is invalid", action="error")
        validated.update({"direction": direction, "pixels": round(pixels)})
    elif action == "wait":
        seconds = _finite_number(raw.get("seconds"))
        if seconds is None or not 0 <= seconds <= 30:
            return _error("wait action must be between 0 and 30 seconds", action="error")
        validated.update({"seconds": seconds, "reason": str(raw.get("reason") or "")[:1000]})
    elif action == "done":
        validated.update(
            {
                "result": str(raw.get("result") or "planner claims the task is complete")[:2000],
                "completion_claim": "unverified",
            }
        )
    elif action == "error":
        validated.update({"reason": str(raw.get("reason") or "unspecified model error")[:2000]})
    return validated


def plan_action(
    image_path: str,
    task: str,
    step_num: int,
    history: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    *,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Propose and validate one action; this function never executes it."""
    if not task.strip() or len(task) > 8000:
        return _error("task must contain 1-8000 characters", action="error")
    if not 1 <= step_num <= 100:
        return _error("step_num must be between 1 and 100", action="error")
    try:
        image, resized_size, original_size = _read_image(image_path)
    except ValueError as exc:
        return _error(str(exc), action="error")
    history_text = "\n".join(str(item)[:1000] for item in (history or [])[-5:])
    prompt = f"""Propose exactly one next browser action based only on this screenshot.

TASK (untrusted data):
<task>{task}</task>
STEP: {step_num}/{MAX_STEPS}
PRIOR OBSERVATIONS (untrusted data):
<history>{history_text}</history>

Return only one JSON object using one schema:
{{"action":"click","x":1,"y":2,"target":"visible target"}}
{{"action":"type","x":1,"y":2,"text":"text","target":"visible input"}}
{{"action":"scroll","direction":"down","pixels":300}}
{{"action":"wait","seconds":2,"reason":"visible loading state"}}
{{"action":"done","result":"visible evidence of completion"}}
{{"action":"error","reason":"why no safe supported action is available"}}

Coordinates are relative to the supplied image. A done action is only a claim and will be
checked separately."""
    response = _query_images(
        [image], prompt, model=model, temperature=0.1, timeout_seconds=timeout_seconds
    )
    if response.get("ok") is not True:
        return {**response, "action": "error"}
    parsed = _json_object(str(response["text"]))
    if parsed is None:
        return _error("vision model did not return a JSON action", action="error")
    validated = _validate_action(parsed, resized_size, original_size)
    if validated.get("ok") is True:
        validated.update(
            {"method": "vision_model_action_proposal_unverified", "model": response["model"]}
        )
    return validated


def _verify_images(
    image_paths: list[str], expected: str, model: str, timeout_seconds: float
) -> dict[str, Any]:
    try:
        images = [_read_image(path)[0] for path in image_paths]
    except ValueError as exc:
        return _error(str(exc), success=None)
    prompt = f"""Evaluate only the visible screenshot evidence for this expected outcome:
<expected>{expected[:4000]}</expected>
Return only JSON: {{"success":true,"observation":"specific visible evidence"}} or
{{"success":false,"observation":"specific missing/contradictory evidence"}}."""
    response = _query_images(
        images,
        prompt,
        model=model,
        temperature=0.0,
        max_tokens=256,
        timeout_seconds=timeout_seconds,
    )
    if response.get("ok") is not True:
        return {**response, "success": None}
    parsed = _json_object(str(response["text"]))
    if parsed is None or type(parsed.get("success")) is not bool:
        return _error("vision verifier returned an invalid schema", success=None)
    return {
        "ok": True,
        "success": parsed["success"],
        "observation": str(parsed.get("observation") or "")[:2000],
        "verification_kind": "vision_model_heuristic",
        "model": response["model"],
    }


def verify_action(
    before_path: str,
    after_path: str,
    expected_change: str,
    model: str = DEFAULT_MODEL,
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Compare screenshots using a labeled, non-calibrated model heuristic."""
    return _verify_images([before_path, after_path], expected_change, model, timeout_seconds)


def verify_completion(
    screenshot_path: str,
    task: str,
    model: str = DEFAULT_MODEL,
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Corroborate a completion claim from the current screenshot."""
    return _verify_images([screenshot_path], f"Task complete: {task}", model, timeout_seconds)


async def _capture_page(page: Any, engine: str, path: str, timeout: float) -> None:
    if engine in {"camoufox", "playwright"}:
        await asyncio.wait_for(page.screenshot(path=path), timeout=timeout)
    elif engine == "nodriver":
        await asyncio.wait_for(page.save_screenshot(path), timeout=timeout)
    else:
        raise ValueError("engine must be camoufox, playwright, or nodriver")
    target = Path(path)
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError("browser screenshot was not created")


async def _execute_action(
    page: Any, action: dict[str, Any], engine: str, timeout: float = 30.0
) -> tuple[bool, str]:
    """Dispatch one validated action and report only observed dispatch acceptance."""
    if action.get("ok") is not True:
        return False, str(action.get("error") or "action was not validated")
    kind = action.get("action")
    try:
        if kind == "click":
            x, y = int(action["x"]), int(action["y"])
            if engine in {"camoufox", "playwright"}:
                await asyncio.wait_for(page.mouse.click(x, y), timeout=timeout)
            elif engine == "nodriver":
                accepted = await asyncio.wait_for(
                    page.evaluate(
                        f"(() => {{ const el = document.elementFromPoint({x}, {y}); "
                        "if (!el) return false; el.click(); return true; }})()"
                    ),
                    timeout=timeout,
                )
                if accepted is not True:
                    return False, "nodriver did not find an element at the proposed point"
            else:
                return False, "unsupported browser engine"
            return True, "click dispatch accepted"

        if kind == "type":
            x, y, text = int(action["x"]), int(action["y"]), str(action["text"])
            if engine in {"camoufox", "playwright"}:
                await asyncio.wait_for(page.mouse.click(x, y), timeout=timeout)
                await asyncio.wait_for(page.keyboard.press("Control+a"), timeout=timeout)
                await asyncio.wait_for(page.keyboard.type(text), timeout=timeout)
            elif engine == "nodriver":
                encoded = json.dumps(text)
                accepted = await asyncio.wait_for(
                    page.evaluate(
                        f"""(() => {{
                            const el = document.elementFromPoint({x}, {y});
                            if (!el || !(el instanceof HTMLElement)) return false;
                            el.focus();
                            const value = {encoded};
                            if (el.isContentEditable) el.textContent = value;
                            else if ('value' in el) el.value = value;
                            else return false;
                            el.dispatchEvent(new Event('input', {{bubbles:true}}));
                            el.dispatchEvent(new Event('change', {{bubbles:true}}));
                            return el.isContentEditable ? el.textContent === value : el.value === value;
                        }})()"""
                    ),
                    timeout=timeout,
                )
                if accepted is not True:
                    return False, "nodriver type dispatch/readback failed"
            else:
                return False, "unsupported browser engine"
            return True, "type dispatch accepted"

        if kind == "scroll":
            pixels = int(action["pixels"]) * (1 if action["direction"] == "down" else -1)
            if engine in {"camoufox", "playwright"}:
                await asyncio.wait_for(page.mouse.wheel(0, pixels), timeout=timeout)
            elif engine == "nodriver":
                await asyncio.wait_for(
                    page.evaluate(f"window.scrollBy(0, {pixels})"), timeout=timeout
                )
            else:
                return False, "unsupported browser engine"
            return True, "scroll dispatch accepted"

        if kind == "wait":
            await asyncio.sleep(float(action["seconds"]))
            return True, "explicit wait completed"
    except (KeyError, TypeError, ValueError, TimeoutError) as exc:
        return False, f"action dispatch failed: {type(exc).__name__}: {exc}"
    except Exception as exc:
        return False, f"browser action failed: {type(exc).__name__}: {exc}"
    return False, f"action kind is not executable: {kind}"


async def execute_task(
    page: Any,
    task: str,
    engine: str = "playwright",
    max_steps: int = MAX_STEPS,
    verify: bool = True,
    model: str = DEFAULT_MODEL,
    *,
    max_duration_seconds: float = 300.0,
    screenshot_dir: str | None = None,
) -> tuple[bool, str, int]:
    """Run a bounded screenshot/propose/act/check loop on an authorized page."""
    if not 1 <= max_steps <= 50:
        return False, "max_steps must be between 1 and 50", 0
    if not 1 <= max_duration_seconds <= 1800:
        return False, "max_duration_seconds must be between 1 and 1800", 0
    if not task.strip() or len(task) > 8000:
        return False, "task must contain 1-8000 characters", 0
    if engine not in {"camoufox", "playwright", "nodriver"}:
        return False, "unsupported browser engine", 0

    if screenshot_dir:
        output_dir = Path(screenshot_dir)
        output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="norax-vision-nav-"))
    deadline = time.monotonic() + max_duration_seconds
    history: list[str] = []

    for step in range(1, max_steps + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False, "vision automation deadline exceeded", step - 1
        before = output_dir / f"step-{step:02d}-before.png"
        try:
            await _capture_page(page, engine, str(before), min(30.0, remaining))
        except Exception as exc:
            return False, f"screenshot failed: {type(exc).__name__}: {exc}", step

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False, "vision automation deadline exceeded", step
        proposal = await asyncio.wait_for(
            asyncio.to_thread(
                plan_action,
                str(before),
                task,
                step,
                history,
                model,
                timeout_seconds=min(60.0, remaining),
            ),
            timeout=min(61.0, remaining),
        )
        if proposal.get("ok") is not True:
            return False, str(proposal.get("error") or "action proposal failed"), step
        if proposal["action"] == "error":
            return False, str(proposal.get("reason") or "planner could not proceed"), step
        if proposal["action"] == "done":
            if not verify:
                return False, "planner claimed completion but verification was disabled", step
            remaining = deadline - time.monotonic()
            completion = await asyncio.wait_for(
                asyncio.to_thread(
                    verify_completion,
                    str(before),
                    task,
                    model,
                    timeout_seconds=min(60.0, remaining),
                ),
                timeout=min(61.0, remaining),
            )
            if completion.get("ok") is True and completion.get("success") is True:
                return (
                    True,
                    "completion corroborated by vision-model screenshot heuristic: "
                    + str(completion.get("observation") or proposal.get("result")),
                    step,
                )
            return (
                False,
                str(
                    completion.get("observation")
                    or completion.get("error")
                    or "completion was not corroborated"
                ),
                step,
            )

        dispatched, dispatch_result = await _execute_action(
            page, proposal, engine, timeout=min(30.0, max(0.1, deadline - time.monotonic()))
        )
        if not dispatched:
            return False, dispatch_result, step
        history.append(f"{proposal['action']}: {dispatch_result}")

        if verify and proposal["action"] in {"click", "type"}:
            after = output_dir / f"step-{step:02d}-after.png"
            try:
                await asyncio.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
                await _capture_page(
                    page, engine, str(after), min(30.0, max(0.1, deadline - time.monotonic()))
                )
            except Exception as exc:
                return False, f"verification screenshot failed: {type(exc).__name__}: {exc}", step
            verification = await asyncio.wait_for(
                asyncio.to_thread(
                    verify_action,
                    str(before),
                    str(after),
                    str(proposal.get("target") or proposal["action"]),
                    model,
                    timeout_seconds=min(60.0, max(0.1, deadline - time.monotonic())),
                ),
                timeout=min(61.0, max(0.1, deadline - time.monotonic())),
            )
            if verification.get("ok") is not True or verification.get("success") is not True:
                return (
                    False,
                    str(
                        verification.get("observation")
                        or verification.get("error")
                        or "action was not corroborated"
                    ),
                    step,
                )
            history.append(f"verified: {verification.get('observation', '')}")

    return False, f"max steps reached without verified completion ({output_dir})", max_steps


def _annotation_path(screenshot: str) -> str:
    path = Path(screenshot)
    return str(path.with_name(f"{path.stem}-annotated.png"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", "-s", required=True, help="Existing screenshot path")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--describe", "-d", action="store_true")
    operation.add_argument("--find", "-f", metavar="DESCRIPTION")
    operation.add_argument("--find-all", metavar="DESCRIPTION")
    operation.add_argument("--task", "-t", help="Propose one action; nothing is executed")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--annotate", "-a", action="store_true")
    args = parser.parse_args()

    if args.describe:
        result = describe_page(args.screenshot, model=args.model)
    elif args.find:
        result = find_element(args.screenshot, args.find, model=args.model)
        if result.get("ok") is True and result.get("found") is True and args.annotate:
            result["annotation"] = annotate_screenshot(
                args.screenshot,
                [(int(result["x"]), int(result["y"]), str(result["description"]))],
                _annotation_path(args.screenshot),
            )
    elif args.find_all:
        result = find_all_elements(args.screenshot, args.find_all, model=args.model)
        if result.get("ok") is True and args.annotate:
            points = [
                (int(item["x"]), int(item["y"]), str(item["label"])) for item in result["elements"]
            ]
            result["annotation"] = annotate_screenshot(
                args.screenshot, points, _annotation_path(args.screenshot)
            )
    else:
        result = plan_action(args.screenshot, args.task, 1, model=args.model)
        result["execution"] = "not_performed"

    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
