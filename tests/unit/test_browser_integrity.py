from __future__ import annotations

import base64
import math
from types import SimpleNamespace

import pytest

from norax.dispatch import browser
from norax.dispatch.risk import check as risk_check


@pytest.mark.asyncio
async def test_invalid_actions_and_inputs_do_not_launch_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def get_page(_session_id: str):
        nonlocal calls
        calls += 1
        raise AssertionError("browser should not launch")

    monkeypatch.setattr(browser, "_get_page", get_page)

    unknown = await browser.t_browser(action="invented")
    missing = await browser.t_browser(action="click")
    unsafe_scheme = await browser.t_browser(action="navigate", url="file:///etc/passwd")

    assert unknown["error"] == "unknown_action"
    assert missing["error"] == "missing_selector"
    assert unsafe_scheme["error"] == "unsupported_url_scheme"
    assert calls == 0


@pytest.mark.asyncio
async def test_tab_inventory_does_not_start_browser_when_none_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(browser, "_context", None)

    result = await browser.t_browser(action="tabs_list")

    assert result == {
        "ok": True,
        "action": "tabs_list",
        "tabs": [],
        "browser_started": False,
    }


@pytest.mark.asyncio
async def test_type_requires_value_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Page:
        async def wait_for_selector(self, *_args, **_kwargs) -> None:
            return None

        async def fill(self, *_args, **_kwargs) -> None:
            return None

        def locator(self, _selector: str):
            async def evaluate(_script: str) -> str:
                return "different"

            return SimpleNamespace(evaluate=evaluate)

    async def get_page(_session_id: str):
        return Page()

    monkeypatch.setattr(browser, "_get_page", get_page)
    result = await browser.t_browser(action="type", selector="#name", text="expected")

    assert result["ok"] is False
    assert result["dispatch_verified"] is True
    assert result["readback_verified"] is False
    assert result["error"] == "value_readback_mismatch"


@pytest.mark.asyncio
async def test_fill_form_reports_partial_failure_as_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Locator:
        def __init__(self, value: str) -> None:
            self.value = value

        async def evaluate(self, _script: str) -> str:
            return self.value

    class Page:
        values: dict[str, str] = {}

        async def wait_for_selector(self, selector: str, **_kwargs) -> None:
            if selector == "#bad":
                raise RuntimeError("missing")

        async def fill(self, selector: str, value: str, **_kwargs) -> None:
            self.values[selector] = value

        def locator(self, selector: str) -> Locator:
            return Locator(self.values[selector])

    async def get_page(_session_id: str):
        return Page()

    monkeypatch.setattr(browser, "_get_page", get_page)
    result = await browser.t_browser(
        action="fill_form",
        fields={"#good": "saved", "#bad": "not-saved"},
    )

    assert result["ok"] is False
    assert result["fields_verified"] == 1
    assert result["results"]["#good"]["readback_verified"] is True
    assert result["results"]["#bad"]["ok"] is False


@pytest.mark.asyncio
async def test_wait_defaults_to_one_second_not_action_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[float] = []

    async def get_page(_session_id: str):
        return object()

    async def fake_sleep(seconds: float) -> None:
        observed.append(seconds)

    monkeypatch.setattr(browser, "_get_page", get_page)
    monkeypatch.setattr(browser.asyncio, "sleep", fake_sleep)
    result = await browser.t_browser(action="wait", timeout=30)

    assert result["ok"] is True
    assert result["seconds"] == 1.0
    assert observed == [1.0]


@pytest.mark.asyncio
async def test_pdf_result_is_rejected_before_huge_base64_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Page:
        async def pdf(self, **_kwargs) -> bytes:
            return b"x" * (browser._PDF_MAX_BYTES + 1)

    async def get_page(_session_id: str):
        return Page()

    monkeypatch.setattr(browser, "_get_page", get_page)
    result = await browser.t_browser(action="pdf")

    assert result["ok"] is False
    assert result["error"] == "pdf_too_large"
    assert "pdf_base64" not in result


def test_browser_launch_keeps_chromium_sandbox_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NORAX_BROWSER_NO_SANDBOX", raising=False)
    assert "--no-sandbox" not in browser._browser_launch_args()


def test_evaluation_result_bound_is_explicit() -> None:
    result, original_chars, truncated = browser._bounded_result("x" * 100, max_chars=20)

    assert truncated is True
    assert original_chars > 20
    assert result["encoding"] == "json_prefix"
    assert len(result["preview"]) == 20


@pytest.mark.asyncio
async def test_every_action_has_an_outer_wall_clock_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def never_returns(**_kwargs):
        await browser.asyncio.Event().wait()

    monkeypatch.setattr(browser, "_t_browser_impl", never_returns)

    result = await browser.t_browser(action="evaluate", js="while (true) {}", timeout=1)

    assert result == {
        "ok": False,
        "error": "browser_evaluate_timed_out",
        "timeout_seconds": 1.0,
    }


@pytest.mark.asyncio
async def test_shutdown_holds_page_registry_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = False

    async def close_resources() -> None:
        nonlocal observed
        observed = browser._page_lock.locked()

    monkeypatch.setattr(browser, "_close_resources_unlocked", close_resources)

    await browser.shutdown_browser()

    assert observed is True


def test_browser_requires_owner_tier() -> None:
    denied = risk_check(
        tool="browser",
        args={"action": "navigate", "url": "http://127.0.0.1"},
        sender_tier="user",
    )
    allowed = risk_check(
        tool="browser",
        args={"action": "evaluate", "js": "document.title"},
        sender_tier="owner",
    )

    assert denied.allowed is False
    assert allowed.allowed is True


class _MockLocator:
    def __init__(self, page, selector: str) -> None:
        self.page = page
        self.selector = selector

    async def evaluate(self, _script: str) -> str:
        return self.page.values.get(self.selector, "")

    async def count(self) -> int:
        return 51

    async def evaluate_all(self, _script: str, limit: int) -> list[dict[str, str]]:
        return [{"text": f"item-{i}", "html": f"<b>{i}</b>"} for i in range(limit)]


class _MockKeyboard:
    def __init__(self) -> None:
        self.keys: list[str] = []

    async def press(self, key: str) -> None:
        self.keys.append(key)


class _MockMouse:
    def __init__(self, page) -> None:
        self.page = page

    async def wheel(self, x: int, y: int) -> None:
        assert x == 0
        self.page.scroll_y += y


class _MockPage:
    def __init__(self) -> None:
        self.url = "about:blank"
        self.values: dict[str, str] = {}
        self.scroll_y = 0
        self.waited: list[str] = []
        self.clicked: list[str] = []
        self.closed = False
        self.keyboard = _MockKeyboard()
        self.mouse = _MockMouse(self)

    async def goto(self, url: str, **_kwargs) -> None:
        self.url = url

    async def title(self) -> str:
        return "Mock Title"

    async def wait_for_selector(self, selector: str, **_kwargs) -> None:
        self.waited.append(selector)

    async def click(self, selector: str, **_kwargs) -> None:
        self.clicked.append(selector)

    async def fill(self, selector: str, value: str, **_kwargs) -> None:
        self.values[selector] = value

    def locator(self, selector: str) -> _MockLocator:
        return _MockLocator(self, selector)

    async def inner_html(self, selector: str) -> str:
        return f"<div>{selector}</div>"

    async def get_attribute(self, selector: str, attribute: str) -> str:
        return f"{selector}:{attribute}"

    async def inner_text(self, selector: str) -> str:
        return f"text:{selector}"

    async def content(self) -> str:
        return "<html>mock</html>"

    async def screenshot(self, **_kwargs) -> bytes:
        return b"mock-png"

    async def evaluate(self, expression: str):
        if "window.scrollY" in expression:
            return self.scroll_y
        return {"answer": 42}

    async def wait_for_timeout(self, milliseconds: int) -> None:
        assert milliseconds == 50

    async def pdf(self, **_kwargs) -> bytes:
        return b"%PDF-mock"

    async def close(self) -> None:
        self.closed = True

    def is_closed(self) -> bool:
        return self.closed


@pytest.mark.asyncio
async def test_mocked_browser_action_contracts_cover_full_dispatch(monkeypatch) -> None:
    page = _MockPage()

    async def get_page(_session_id: str):
        return page

    monkeypatch.setattr(browser, "_get_page", get_page)

    navigated = await browser.t_browser(
        action="navigate",
        url="https://example.test/path",
        wait_for="#ready",
    )
    assert navigated["ok"] is True
    assert navigated["navigation_observed"] is True
    assert navigated["title"] == "Mock Title"

    clicked = await browser.t_browser(action="click", selector="#next")
    assert clicked["dispatch_verified"] is True
    assert clicked["outcome_verified"] is False

    typed = await browser.t_browser(action="type", selector="#name", text="Norax")
    assert typed["ok"] is True
    assert typed["readback_verified"] is True

    html = await browser.t_browser(action="extract", selector="#main", extract_type="html")
    assert html["content"] == "<div>#main</div>"
    attribute = await browser.t_browser(
        action="extract",
        selector="#main",
        extract_type="attribute",
        attribute="data-id",
    )
    assert attribute["content"] == "#main:data-id"
    all_matches = await browser.t_browser(action="extract", selector=".item", extract_type="all")
    assert all_matches["matched_count"] == 51
    assert len(all_matches["content"]) == 50
    assert all_matches["truncated"] is True
    body = await browser.t_browser(action="extract")
    assert body["content"] == "text:body"

    shot = await browser.t_browser(action="screenshot")
    assert base64.b64decode(shot["image_base64"]) == b"mock-png"
    assert shot["truncated"] is False

    filled = await browser.t_browser(
        action="fill_form",
        fields={"#email": "owner@example.test"},
        selector="#submit",
        submit=True,
    )
    assert filled["ok"] is True
    assert filled["fields_verified"] == 1
    assert page.clicked[-1] == "#submit"

    scrolled = await browser.t_browser(action="scroll", scroll_amount=250)
    assert scrolled["movement_observed"] is True
    assert scrolled["scroll_y_after"] == 250

    waited = await browser.t_browser(action="wait", wait_for="#done")
    assert waited == {"ok": True, "action": "wait", "selector": "#done"}

    evaluated = await browser.t_browser(action="evaluate", js="({answer: 42})")
    assert evaluated["result"] == {"answer": 42}
    assert evaluated["truncated"] is False

    pdf = await browser.t_browser(action="pdf")
    assert base64.b64decode(pdf["pdf_base64"]) == b"%PDF-mock"


@pytest.mark.asyncio
async def test_submit_without_selector_uses_keyboard(monkeypatch) -> None:
    page = _MockPage()

    async def get_page(_session_id: str):
        return page

    monkeypatch.setattr(browser, "_get_page", get_page)
    result = await browser.t_browser(
        action="fill_form",
        fields={"#query": "coverage"},
        submit=True,
    )

    assert result["ok"] is True
    assert page.keyboard.keys == ["Enter"]


@pytest.mark.asyncio
async def test_close_and_tab_inventory_manage_existing_resources(monkeypatch) -> None:
    page = _MockPage()
    second = _MockPage()
    second.url = "https://second.test"
    monkeypatch.setattr(browser, "_pages", {"session": page})
    monkeypatch.setattr(browser, "_page_last_used", {"session": 1.0})

    closed = await browser.t_browser(action="close", session_id="session")
    assert closed["closed"] is True
    assert page.closed is True
    closed_again = await browser.t_browser(action="close", session_id="session")
    assert closed_again["closed"] is False

    monkeypatch.setattr(browser, "_context", SimpleNamespace(pages=[page, second]))
    tabs = await browser.t_browser(action="tabs_list")
    assert [item["title"] for item in tabs["tabs"]] == ["Mock Title", "Mock Title"]
    assert tabs["title_timeout"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True, "invalid"])
async def test_nonfinite_or_nonnumeric_durations_never_launch_browser(invalid, monkeypatch) -> None:
    calls = 0

    async def get_page(_session_id: str):
        nonlocal calls
        calls += 1
        return _MockPage()

    monkeypatch.setattr(browser, "_get_page", get_page)
    assert (await browser.t_browser(action="wait", timeout=invalid))["error"] == "invalid_timeout"
    assert (await browser.t_browser(action="wait", wait_seconds=invalid))["error"] == (
        "invalid_wait_seconds"
    )
    assert calls == 0


@pytest.mark.asyncio
async def test_nonfinite_evaluation_result_is_json_safe(monkeypatch) -> None:
    page = _MockPage()

    async def evaluate(_expression: str):
        return {"value": math.nan}

    page.evaluate = evaluate  # type: ignore[method-assign]

    async def get_page(_session_id: str):
        return page

    monkeypatch.setattr(browser, "_get_page", get_page)
    result = await browser.t_browser(action="evaluate", js="NaN")

    assert result["ok"] is True
    assert isinstance(result["result"], str)
    assert "nan" in result["result"].lower()
