#!/usr/bin/env python3
"""
browser_wrapper.py — CDP/Playwright browser wrapper with an enforced deadline.

Any browser automation task using this wrapper auto-enforces the flow timer.
If exceeded: wrapper raises FlowExceededError, MUST be caught by caller.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar
from urllib.parse import urlsplit

from flow_guard import FlowGuard

T = TypeVar("T")


class FlowExceededError(RuntimeError):
    """Raised when the configured flow deadline triggers."""

    pass


class BrowserWrapper:
    """Wrapper around Playwright/Chrome with built-in flow guard."""

    def __init__(self, task_name: str, cdp_port: int = 9333):
        if not 1 <= cdp_port <= 65535:
            raise ValueError("cdp_port must be between 1 and 65535")
        self.guard = FlowGuard(task_name)
        self.cdp_port = cdp_port
        self.page: Any = None
        self.browser: Any = None
        self._context: Any = None
        self._pw: Any = None

    def _check(self) -> None:
        """Internal: verify timer hasn't expired."""
        if self.guard.check():
            elapsed = self.guard.elapsed_seconds()
            raise FlowExceededError(
                f"DEADLINE: Task '{self.guard.task_name}' exceeded "
                f"{self.guard.max_seconds / 60:g} minutes "
                f"(elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s). "
                f"All browser work must stop immediately."
            )

    def _require_page(self) -> Any:
        if self.page is None:
            raise RuntimeError("browser is not connected; call connect() first")
        return self.page

    async def _guarded(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run one async operation within the remaining flow deadline."""
        self._check()
        remaining = self.guard.max_seconds - self.guard.elapsed_seconds()
        try:
            result = await asyncio.wait_for(operation(), timeout=max(0.001, remaining))
        except TimeoutError as exc:
            self.guard.check()
            raise FlowExceededError(
                f"DEADLINE: Task '{self.guard.task_name}' exceeded its flow limit"
            ) from exc
        self._check()
        return result

    async def connect(self) -> Any:
        """Connect to Chrome via CDP."""
        self._check()
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError("Playwright not installed: pip install playwright") from exc

        playwright: Any = async_playwright()
        self._pw = await self._guarded(playwright.start)
        try:
            self.browser = await self._guarded(
                lambda: self._pw.chromium.connect_over_cdp(f"http://127.0.0.1:{self.cdp_port}")
            )
            self._context = (
                self.browser.contexts[0]
                if self.browser.contexts
                else await self._guarded(self.browser.new_context)
            )
            self.page = (
                self._context.pages[0]
                if self._context.pages
                else await self._guarded(self._context.new_page)
            )
        except BaseException:
            # Cleanup must never replace the connection/setup exception.
            try:
                await self.close()
            except BaseException:
                pass
            raise
        self._check()
        return self.page

    async def screenshot(self, path: str = "/tmp/norax-browser.png") -> str:
        """Capture browser screenshot."""
        page = self._require_page()
        await self._guarded(lambda: page.screenshot(path=path, full_page=True))
        return path

    async def goto(self, url: str, wait_until: str = "domcontentloaded") -> None:
        """Navigate to an HTTP(S) URL with a bounded Playwright wait policy."""
        try:
            parsed = urlsplit(url)
        except ValueError as exc:
            raise ValueError(f"invalid URL: {exc}") from exc
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError("browser URL must use http or https")
        if wait_until not in {"commit", "domcontentloaded", "load", "networkidle"}:
            raise ValueError("unsupported wait_until policy")
        page = self._require_page()
        await self._guarded(lambda: page.goto(url, wait_until=wait_until))

    async def click(self, selector: str) -> None:
        """Click element by selector."""
        page = self._require_page()
        await self._guarded(lambda: page.click(selector))

    async def fill(self, selector: str, text: str) -> None:
        """Fill form field."""
        page = self._require_page()
        await self._guarded(lambda: page.fill(selector, text))

    async def evaluate(self, js: str) -> Any:
        """Run JS in page."""
        page = self._require_page()
        return await self._guarded(lambda: page.evaluate(js))

    async def get_text(self, selector: str = "body", max_chars: int = 100_000) -> str:
        """Get text content."""
        if not 1 <= max_chars <= 1_000_000:
            raise ValueError("max_chars must be between 1 and 1000000")
        page = self._require_page()
        el = await self._guarded(lambda: page.query_selector(selector))
        if el:
            text = await self._guarded(el.text_content)
            return (text or "")[:max_chars]
        return ""

    def status(self) -> dict[str, Any]:
        """Current flow status."""
        return self.guard.status()

    def finish(self, success: bool = False) -> None:
        """Mark flow complete."""
        self.guard.finish(success=success)

    async def close(self) -> None:
        """Best-effort bounded cleanup, including partially connected state."""
        first_error: BaseException | None = None
        if self.browser:
            try:
                await asyncio.wait_for(self.browser.close(), timeout=10)
            except BaseException as exc:
                first_error = exc
            finally:
                self.browser = None
                self.page = None
                self._context = None
        if self._pw is not None:
            try:
                await asyncio.wait_for(self._pw.stop(), timeout=10)
            except BaseException as exc:
                first_error = first_error or exc
            finally:
                self._pw = None
        if first_error is not None:
            raise first_error
