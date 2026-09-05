"""Bounded Playwright-backed browser interaction.

Capabilities:
  - navigate: go to URL, wait for load
  - click: click an element by CSS selector
  - type: type text into an element
  - extract: get text/HTML/attributes from elements
  - screenshot: capture page screenshot (base64 PNG)
  - fill_form: fill multiple fields and optionally submit
  - scroll: scroll the page
  - wait: wait for selector or timeout
  - evaluate: run JS in page context (T2 — owner only)
  - tabs_list: inventory tabs in the existing browser context
  - pdf: save page as PDF

Design:
  - One shared persistent context with a bounded page per session ID
  - Headless by default; headed mode via env NORAX_BROWSER_HEADED=1
  - Optional best-effort stealth via NORAX_BROWSER_STEALTH=1
  - Bounded screenshots, extracts, evaluation results, PDFs, and sessions
  - Mutation results distinguish dispatched actions from observed readback
  - Timeout: 30s default per action, configurable
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import os
import time
from typing import Any
from urllib.parse import urlsplit

log = logging.getLogger("norax.dispatch.browser")

# Lazy-initialized browser resources
_playwright: Any = None
_browser: Any = None
_context: Any = None
_pages: dict[str, Any] = {}  # session_id → Page
_page_last_used: dict[str, float] = {}
_lock = asyncio.Lock()
_page_lock = asyncio.Lock()

_HEADED = os.environ.get("NORAX_BROWSER_HEADED", "0") == "1"
_SCREENSHOT_MAX_BYTES = 200_000  # 200KB cap for base64
_PDF_MAX_BYTES = 250_000
_EXTRACT_MAX_CHARS = 100_000
_EVALUATE_MAX_CHARS = 100_000
_MAX_SESSIONS = 16
_MAX_FIELDS = 100
_ACTIONS = frozenset(
    {
        "navigate",
        "click",
        "type",
        "extract",
        "screenshot",
        "fill_form",
        "scroll",
        "wait",
        "evaluate",
        "pdf",
        "tabs_list",
        "close",
    }
)


def _browser_launch_args() -> list[str]:
    args = ["--disable-dev-shm-usage"]
    if os.environ.get("NORAX_BROWSER_NO_SANDBOX", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        log.warning("browser: Chromium process sandbox explicitly disabled by operator")
        args.append("--no-sandbox")
    if os.environ.get("NORAX_BROWSER_STEALTH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        args.append("--disable-blink-features=AutomationControlled")
    return args


def _compress_png(png_bytes: bytes) -> tuple[bytes, bool]:
    """Downscale a screenshot until it satisfies the serialized byte cap."""
    if len(png_bytes) <= _SCREENSHOT_MAX_BYTES:
        return png_bytes, False
    from PIL import Image

    with Image.open(io.BytesIO(png_bytes)) as opened:
        image = opened.convert("RGB")
    original_size = image.size
    resample = getattr(Image, "Resampling", Image).LANCZOS
    for _ in range(10):
        ratio = min(0.9, (_SCREENSHOT_MAX_BYTES / max(len(png_bytes), 1)) ** 0.5 * 0.95)
        new_size = (
            max(32, int(image.width * ratio)),
            max(32, int(image.height * ratio)),
        )
        if new_size == image.size:
            break
        image = image.resize(new_size, resample)
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()
        if len(png_bytes) <= _SCREENSHOT_MAX_BYTES:
            break
    return png_bytes, image.size != original_size


def _bounded_result(value: Any, *, max_chars: int) -> tuple[Any, int, bool]:
    """Return a JSON-safe bounded representation and its original encoded size."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)
    except (TypeError, ValueError):
        encoded = json.dumps(str(value), ensure_ascii=False, allow_nan=False)
    original_len = len(encoded)
    safe_value = json.loads(encoded)
    if original_len <= max_chars:
        return safe_value, original_len, False
    return {"preview": encoded[:max_chars], "encoding": "json_prefix"}, original_len, True


async def _close_resources_unlocked() -> None:
    global _playwright, _browser, _context
    for page in list(_pages.values()):
        try:
            await page.close()
        except Exception:
            pass
    _pages.clear()
    _page_last_used.clear()
    if _context is not None:
        try:
            await _context.close()
        except Exception:
            pass
        _context = None
    if _browser is not None:
        try:
            await _browser.close()
        except Exception:
            pass
        _browser = None
    if _playwright is not None:
        try:
            await _playwright.stop()
        except Exception:
            pass
        _playwright = None


async def shutdown_browser() -> None:
    async with _lock:
        # Keep session bookkeeping and context teardown atomic with respect to
        # page creation.  Without the page lock, a concurrent _get_page() can
        # publish a page while this function is clearing the same registries.
        async with _page_lock:
            await _close_resources_unlocked()


async def probe_browser_backend(*, timeout: float = 15.0) -> dict[str, Any]:
    """Launch an isolated browser and prove one page can execute.

    The operational probe deliberately does not use Norax's shared browser
    context.  A health check must never create a persistent Chromium process,
    inherit user cookies, or close an in-flight browser session.
    """

    async def _probe() -> dict[str, Any]:
        from playwright.async_api import async_playwright

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True,
                args=_browser_launch_args(),
            )
            try:
                page = await browser.new_page()
                await page.goto(
                    "data:text/html,<title>norax-probe</title><h1>ok</h1>",
                    wait_until="domcontentloaded",
                )
                title = await page.title()
                return {
                    "ok": title == "norax-probe",
                    "evidence": "isolated_launch_and_page_execution",
                }
            finally:
                await browser.close()

    try:
        if isinstance(timeout, bool):
            raise ValueError("boolean timeout")
        parsed_timeout = float(timeout)
        if not math.isfinite(parsed_timeout):
            raise ValueError("non-finite timeout")
        async with asyncio.timeout(min(max(parsed_timeout, 1.0), 60.0)):
            return await _probe()
    except TimeoutError:
        return {"ok": False, "error": "isolated browser probe timed out"}
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": "isolated browser probe failed",
            "detail": str(exc)[:500],
        }


async def _ensure_browser() -> Any:
    """Lazy-init Playwright + browser + context. Reuses across calls."""
    global _playwright, _browser, _context
    async with _lock:
        if _context is not None:
            try:
                _ = _context.pages
                return _context
            except Exception:
                await _close_resources_unlocked()

        from playwright.async_api import async_playwright

        playwright = await async_playwright().start()
        browser = None
        context = None
        try:
            browser = await playwright.chromium.launch(
                headless=not _HEADED,
                args=_browser_launch_args(),
            )
            context_kwargs: dict[str, Any] = {
                "viewport": {"width": 1920, "height": 1080},
            }
            browser_locale = os.environ.get("NORAX_BROWSER_LOCALE", "").strip()
            browser_timezone = os.environ.get("NORAX_BROWSER_TIMEZONE", "").strip()
            if browser_locale:
                context_kwargs["locale"] = browser_locale[:64]
            if browser_timezone:
                context_kwargs["timezone_id"] = browser_timezone[:128]
            context = await browser.new_context(**context_kwargs)
        except BaseException:
            if context is not None:
                try:
                    await context.close()
                except Exception:
                    pass
            if browser is not None:
                try:
                    await browser.close()
                except Exception:
                    pass
            try:
                await playwright.stop()
            except Exception:
                pass
            raise

        _playwright = playwright
        _browser = browser
        _context = context
        return context


async def _get_page(session_id: str = "default") -> Any:
    """Get or create a page for the given session."""
    context = await _ensure_browser()
    async with _page_lock:
        page = _pages.get(session_id)
        if page is not None:
            try:
                if not page.is_closed():
                    _page_last_used[session_id] = time.monotonic()
                    return page
            except Exception:
                pass
            _pages.pop(session_id, None)
            _page_last_used.pop(session_id, None)

        if len(_pages) >= _MAX_SESSIONS:
            oldest_session = min(
                _pages,
                key=lambda key: _page_last_used.get(key, 0.0),
            )
            oldest_page = _pages.pop(oldest_session)
            _page_last_used.pop(oldest_session, None)
            try:
                await oldest_page.close()
            except Exception:
                pass
        page = await context.new_page()
        if os.environ.get("NORAX_BROWSER_STEALTH", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            try:
                from playwright_stealth import stealth_async

                await stealth_async(page)
            except Exception as exc:  # noqa: BLE001
                log.warning("browser: requested stealth could not be applied: %r", exc)
        _pages[session_id] = page
        _page_last_used[session_id] = time.monotonic()

    return page


async def _t_browser_impl(
    *,
    action: str,
    url: str | None = None,
    selector: str | None = None,
    text: str | None = None,
    fields: dict | None = None,
    attribute: str | None = None,
    session_id: str = "default",
    timeout: float = 30.0,
    wait_for: str | None = None,
    wait_seconds: float | None = None,
    extract_type: str = "text",
    scroll_amount: int = 500,
    js: str | None = None,
    submit: bool = False,
) -> dict:
    """Browser automation tool. Actions:

    - navigate: Go to URL. Optional wait_for selector.
    - click: Click element by CSS selector.
    - type: Type text into element by CSS selector.
    - extract: Get text/HTML/attrs from selector (or full page if no selector).
    - screenshot: Capture page screenshot as base64 PNG.
    - fill_form: Fill multiple {selector: value} fields, optionally submit.
    - scroll: Scroll page by scroll_amount pixels.
    - wait: Wait for selector, or wait_seconds (default 1 second).
    - evaluate: Run JavaScript in page context (owner only — T2).
    - pdf: Save page as PDF (returns base64).
    - tabs_list: List open tabs/pages.
    - close: Close the browser session.
    """
    try:
        action = str(action or "").strip().lower()
        if action not in _ACTIONS:
            return {
                "ok": False,
                "error": "unknown_action",
                "action": action[:100],
                "supported_actions": sorted(_ACTIONS),
            }
        try:
            if isinstance(timeout, bool):
                raise ValueError("boolean timeout")
            timeout = float(timeout)
            if not math.isfinite(timeout):
                raise ValueError("non-finite timeout")
            timeout = min(max(timeout, 1.0), 120.0)
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid_timeout"}
        session_id = str(session_id or "default").strip() or "default"
        if len(session_id) > 128 or any(ord(char) < 32 for char in session_id):
            return {"ok": False, "error": "invalid_session_id"}
        if selector is not None and len(str(selector)) > 4096:
            return {"ok": False, "error": "selector_too_long"}
        if wait_for is not None and len(str(wait_for)) > 4096:
            return {"ok": False, "error": "wait_selector_too_long"}
        if attribute is not None and len(str(attribute)) > 256:
            return {"ok": False, "error": "attribute_name_too_long"}
        if not isinstance(submit, bool):
            return {"ok": False, "error": "invalid_submit_flag"}
        try:
            if isinstance(scroll_amount, bool):
                raise ValueError("boolean scroll amount")
            scroll_amount = min(100_000, max(-100_000, int(scroll_amount)))
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid_scroll_amount"}

        if action == "close":
            async with _page_lock:
                page = _pages.pop(session_id, None)
                closed = page is not None
                _page_last_used.pop(session_id, None)
                if page is not None:
                    await page.close()
            return {
                "ok": True,
                "action": "close",
                "session_id": session_id,
                "closed": closed,
            }

        if action == "tabs_list":
            context = _context
            if context is None:
                return {
                    "ok": True,
                    "action": "tabs_list",
                    "tabs": [],
                    "browser_started": False,
                }
            tabs = []
            pages = list(context.pages)[:_MAX_SESSIONS]
            for p in pages:
                tabs.append({"url": str(p.url)[:8192], "title": None})
            title_timeout = False
            try:
                titles = await asyncio.wait_for(
                    asyncio.gather(
                        *(page.title() for page in pages),
                        return_exceptions=True,
                    ),
                    timeout=timeout,
                )
            except TimeoutError:
                titles = []
                title_timeout = True
            for index, title in enumerate(titles):
                if isinstance(title, str):
                    tabs[index]["title"] = title[:1000]
            return {
                "ok": True,
                "action": "tabs_list",
                "tabs": tabs,
                "browser_started": True,
                "truncated": len(context.pages) > len(pages),
                "title_timeout": title_timeout,
            }

        if action == "navigate":
            if not url:
                return {
                    "ok": False,
                    "error": "missing_url",
                    "hint": "Provide url for navigate action",
                }
            if len(url) > 8192:
                return {"ok": False, "error": "url_too_long"}
            scheme = urlsplit(url).scheme.lower()
            if scheme not in {"http", "https", "data", "about"}:
                return {
                    "ok": False,
                    "error": "unsupported_url_scheme",
                    "scheme": scheme,
                }
        if action == "click" and not selector:
            return {"ok": False, "error": "missing_selector"}
        if action == "type" and (not selector or text is None):
            return {"ok": False, "error": "missing_selector_or_text"}
        if action == "type" and text is not None and len(text) > 1_000_000:
            return {"ok": False, "error": "text_too_large"}
        if action == "fill_form" and (not fields or not isinstance(fields, dict)):
            return {
                "ok": False,
                "error": "missing_fields",
                "hint": "Provide fields as {selector: value}",
            }
        if action == "fill_form" and fields is not None:
            if len(fields) > _MAX_FIELDS:
                return {
                    "ok": False,
                    "error": "too_many_fields",
                    "max_fields": _MAX_FIELDS,
                }
            total_field_chars = sum(
                len(str(key)) + len(str(value)) for key, value in fields.items()
            )
            if total_field_chars > 2_000_000:
                return {"ok": False, "error": "form_payload_too_large"}
        if action == "evaluate" and not js:
            return {"ok": False, "error": "missing_js"}
        if action == "evaluate" and js is not None and len(js) > 100_000:
            return {"ok": False, "error": "javascript_too_large"}
        if action == "extract" and extract_type not in {
            "text",
            "html",
            "attribute",
            "all",
        }:
            return {"ok": False, "error": "invalid_extract_type"}
        if action == "extract" and extract_type == "attribute" and not attribute:
            return {"ok": False, "error": "missing_attribute"}
        if action == "wait" and not wait_for:
            try:
                if isinstance(wait_seconds, bool):
                    raise ValueError("boolean wait duration")
                wait_seconds = 1.0 if wait_seconds is None else float(wait_seconds)
                if not math.isfinite(wait_seconds):
                    raise ValueError("non-finite wait duration")
            except (TypeError, ValueError):
                return {"ok": False, "error": "invalid_wait_seconds"}
            wait_seconds = min(max(wait_seconds, 0.0), 30.0)

        page = await _get_page(session_id)

        if action == "navigate":
            t0 = time.perf_counter()
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            if wait_for:
                await page.wait_for_selector(wait_for, timeout=timeout * 1000)
            title = await page.title()
            elapsed = time.perf_counter() - t0
            return {
                "ok": True,
                "action": "navigate",
                "url": page.url[:8192],
                "title": title[:2000],
                "elapsed_ms": int(elapsed * 1000),
                "navigation_observed": True,
            }

        if action == "click":
            before_url = page.url
            await page.wait_for_selector(selector, timeout=timeout * 1000)
            await page.click(selector, timeout=timeout * 1000)
            after_url = page.url
            return {
                "ok": True,
                "action": "click",
                "selector": selector,
                "dispatch_verified": True,
                "url_before": before_url,
                "url_after": after_url,
                "navigation_observed": after_url != before_url,
                "outcome_verified": False,
            }

        if action == "type":
            if selector is None or text is None:
                return {"ok": False, "error": "missing_selector_or_text"}
            await page.wait_for_selector(selector, timeout=timeout * 1000)
            await page.fill(selector, text, timeout=timeout * 1000)
            observed = await page.locator(selector).evaluate(
                "(el) => ('value' in el ? el.value : (el.textContent ?? ''))"
            )
            verified = str(observed) == text
            return {
                "ok": verified,
                "action": "type",
                "selector": selector,
                "chars": len(text),
                "dispatch_verified": True,
                "readback_verified": verified,
                "error": None if verified else "value_readback_mismatch",
            }

        if action == "extract":
            matched_count: int | None = None
            truncated = False
            if selector:
                await page.wait_for_selector(selector, timeout=timeout * 1000)
                if extract_type == "html":
                    content = await page.inner_html(selector)
                elif extract_type == "attribute":
                    content = await page.get_attribute(selector, attribute)
                elif extract_type == "all":
                    locator = page.locator(selector)
                    matched_count = await locator.count()
                    content = await locator.evaluate_all(
                        """
                        (elements, limit) => elements.slice(0, limit).map((el) => ({
                          text: (el.innerText ?? "").slice(0, 1000),
                          html: (el.innerHTML ?? "").slice(0, 4000),
                        }))
                        """,
                        50,
                    )
                    truncated = matched_count > 50
                else:  # text
                    content = await page.inner_text(selector)
            else:
                # Full page extraction
                if extract_type == "html":
                    content = await page.content()
                else:
                    content = await page.inner_text("body")
            if isinstance(content, str):
                original_len = len(content)
                if original_len > _EXTRACT_MAX_CHARS:
                    content = content[:_EXTRACT_MAX_CHARS]
                    truncated = True
                returned_len = len(content)
            else:
                bounded, original_len, bounded_truncated = _bounded_result(
                    content,
                    max_chars=_EXTRACT_MAX_CHARS,
                )
                content = bounded
                returned_len = len(json.dumps(content, ensure_ascii=False, default=str))
                truncated = truncated or bounded_truncated
            return {
                "ok": True,
                "action": "extract",
                "selector": selector,
                "extract_type": extract_type,
                "content": content,
                "content_len": original_len,
                "returned_chars": returned_len,
                "matched_count": matched_count,
                "truncated": truncated,
            }

        if action == "screenshot":
            png_bytes = await page.screenshot(full_page=False, type="png")
            png_bytes, resized = await asyncio.to_thread(_compress_png, png_bytes)
            if len(png_bytes) > _SCREENSHOT_MAX_BYTES:
                return {
                    "ok": False,
                    "error": "screenshot_could_not_meet_size_limit",
                    "image_bytes": len(png_bytes),
                    "limit_bytes": _SCREENSHOT_MAX_BYTES,
                }
            b64 = base64.b64encode(png_bytes).decode("ascii")
            return {
                "ok": True,
                "action": "screenshot",
                "image_base64": b64,
                "image_bytes": len(png_bytes),
                "url": page.url,
                "resized": resized,
                "truncated": False,
            }

        if action == "fill_form":
            if not isinstance(fields, dict) or not fields:
                return {"ok": False, "error": "missing_fields"}
            results: dict[str, dict[str, Any]] = {}
            for raw_sel, val in fields.items():
                sel = str(raw_sel)
                if not sel or len(sel) > 4096:
                    results[sel[:200]] = {
                        "ok": False,
                        "error": "invalid_selector",
                    }
                    continue
                value = str(val)
                if len(value) > 1_000_000:
                    results[sel] = {
                        "ok": False,
                        "error": "value_too_large",
                    }
                    continue
                try:
                    await page.wait_for_selector(sel, timeout=timeout * 1000)
                    await page.fill(sel, value, timeout=timeout * 1000)
                    observed = await page.locator(sel).evaluate(
                        "(el) => ('value' in el ? el.value : (el.textContent ?? ''))"
                    )
                    verified = str(observed) == value
                    results[sel] = {
                        "ok": verified,
                        "dispatch_verified": True,
                        "readback_verified": verified,
                        "error": None if verified else "value_readback_mismatch",
                    }
                except Exception as exc:  # noqa: BLE001
                    results[sel] = {
                        "ok": False,
                        "error": type(exc).__name__,
                        "detail": str(exc)[:500],
                    }
            if submit and selector:
                try:
                    await page.click(selector, timeout=timeout * 1000)
                    results["_submit"] = {
                        "ok": True,
                        "dispatch_verified": True,
                        "outcome_verified": False,
                    }
                except Exception as exc:  # noqa: BLE001
                    results["_submit"] = {
                        "ok": False,
                        "error": type(exc).__name__,
                        "detail": str(exc)[:500],
                    }
            elif submit:
                try:
                    await page.keyboard.press("Enter")
                    results["_submit"] = {
                        "ok": True,
                        "dispatch_verified": True,
                        "outcome_verified": False,
                    }
                except Exception as exc:  # noqa: BLE001
                    results["_submit"] = {
                        "ok": False,
                        "error": type(exc).__name__,
                        "detail": str(exc)[:500],
                    }
            all_verified = bool(results) and all(
                item.get("ok") is True for item in results.values()
            )
            return {
                "ok": all_verified,
                "action": "fill_form",
                "results": results,
                "fields_requested": len(fields),
                "fields_verified": sum(
                    1
                    for key, item in results.items()
                    if key != "_submit" and item.get("readback_verified") is True
                ),
                "submit_requested": submit,
                "error": None if all_verified else "partial_or_unverified_fill",
            }

        if action == "scroll":
            before_y = await page.evaluate("() => window.scrollY")
            await page.mouse.wheel(0, scroll_amount)
            await page.wait_for_timeout(50)
            after_y = await page.evaluate("() => window.scrollY")
            return {
                "ok": True,
                "action": "scroll",
                "amount": scroll_amount,
                "dispatch_verified": True,
                "scroll_y_before": before_y,
                "scroll_y_after": after_y,
                "movement_observed": after_y != before_y,
            }

        if action == "wait":
            if wait_for:
                await page.wait_for_selector(wait_for, timeout=timeout * 1000)
                return {"ok": True, "action": "wait", "selector": wait_for}
            seconds = 1.0 if wait_seconds is None else wait_seconds
            await asyncio.sleep(seconds)
            return {"ok": True, "action": "wait", "seconds": seconds}

        if action == "evaluate":
            if js is None:
                return {"ok": False, "error": "missing_js"}
            result = await page.evaluate(js)
            result, result_chars, truncated = _bounded_result(
                result,
                max_chars=_EVALUATE_MAX_CHARS,
            )
            return {
                "ok": True,
                "action": "evaluate",
                "result": result,
                "result_chars": result_chars,
                "truncated": truncated,
            }

        if action == "pdf":
            pdf_bytes = await page.pdf(format="A4", print_background=True)
            if len(pdf_bytes) > _PDF_MAX_BYTES:
                return {
                    "ok": False,
                    "error": "pdf_too_large",
                    "pdf_bytes": len(pdf_bytes),
                    "limit_bytes": _PDF_MAX_BYTES,
                }
            b64 = base64.b64encode(pdf_bytes).decode("ascii")
            return {
                "ok": True,
                "action": "pdf",
                "pdf_base64": b64,
                "pdf_bytes": len(pdf_bytes),
                "truncated": False,
            }

        return {"ok": False, "error": "internal_unhandled_action", "action": action}

    except Exception as e:
        log.warning("browser.%s error: %r", action, e)
        return {
            "ok": False,
            "error": f"browser_{action}_failed",
            "detail": str(e)[:2000],
        }


async def t_browser(
    *,
    action: str,
    url: str | None = None,
    selector: str | None = None,
    text: str | None = None,
    fields: dict | None = None,
    attribute: str | None = None,
    session_id: str = "default",
    timeout: float = 30.0,
    wait_for: str | None = None,
    wait_seconds: float | None = None,
    extract_type: str = "text",
    scroll_amount: int = 500,
    js: str | None = None,
    submit: bool = False,
) -> dict:
    """Run one browser action under a genuine wall-clock deadline.

    Playwright's operation-specific timeouts do not cover every await (notably
    JavaScript evaluation, compression, and teardown).  The outer deadline
    prevents one malformed page or expression from occupying a dispatcher
    worker indefinitely while adding only one asyncio timeout context per call.
    """
    try:
        if isinstance(timeout, bool):
            raise ValueError("boolean timeout")
        bounded_timeout = float(timeout)
        if not math.isfinite(bounded_timeout):
            raise ValueError("non-finite timeout")
        bounded_timeout = min(max(bounded_timeout, 1.0), 120.0)
    except (TypeError, ValueError):
        return {"ok": False, "error": "invalid_timeout"}

    action_name = str(action or "").strip().lower()
    try:
        async with asyncio.timeout(bounded_timeout):
            return await _t_browser_impl(
                action=action,
                url=url,
                selector=selector,
                text=text,
                fields=fields,
                attribute=attribute,
                session_id=session_id,
                timeout=bounded_timeout,
                wait_for=wait_for,
                wait_seconds=wait_seconds,
                extract_type=extract_type,
                scroll_amount=scroll_amount,
                js=js,
                submit=submit,
            )
    except TimeoutError:
        log.warning(
            "browser.%s exceeded wall-clock timeout %.1fs",
            action_name,
            bounded_timeout,
        )
        return {
            "ok": False,
            "error": f"browser_{action_name or 'action'}_timed_out",
            "timeout_seconds": bounded_timeout,
        }
