"""Discord settings UI helpers."""

from __future__ import annotations

import pytest

from norax.adapter import discord_settings


@pytest.mark.asyncio
async def test_fetch_settings_snapshot_prefers_data_payload():
    async def handler(name, args, _principal, _ctx):
        assert name == "settings"
        return {
            "reply": "ignored",
            "data": {
                "model": "composer/composer-2.5",
                "planning_mode": "direct",
                "thinking_effort": "medium",
                "reasoning_output": False,
            },
        }

    snap = await discord_settings.fetch_settings_snapshot(handler, None, {})
    assert snap["model"] == "composer/composer-2.5"
    assert snap["planning_mode"] == "direct"


@pytest.mark.asyncio
async def test_fetch_settings_snapshot_parses_reply_fallback():
    async def handler(name, _args, _principal, _ctx):
        assert name == "settings"
        return {
            "reply": (
                "**Norax settings**\n"
                "Model: `gpt-5.5`\n"
                "Planning: `orchestrator` — Orchestrator\n"
                "Thinking: `high`\n"
                "Reasoning output: `on`\n"
            )
        }

    snap = await discord_settings.fetch_settings_snapshot(handler, None, {})
    assert snap["model"] == "gpt-5.5"
    assert snap["planning_mode"] == "orchestrator"
    assert snap["thinking_effort"] == "high"
    assert snap["reasoning_output"] is True


@pytest.mark.asyncio
async def test_snapshot_normalization_does_not_treat_false_strings_as_enabled():
    async def handler(_name, _args, _principal, _ctx):
        return {
            "data": {
                "model": 123,
                "planning_mode": "invented",
                "thinking_effort": "impossible",
                "reasoning_output": "false",
                "max_tool_rounds": "not-a-number",
                "memory_depth": "unknown",
                "weak_model_boost": "invalid",
                "stream_replies": "off",
                "response_length": "huge",
                "tool_activity": "noisy",
            }
        }

    snap = await discord_settings.fetch_settings_snapshot(handler, None, {})
    assert snap == {
        "model": "",
        "planning_mode": "direct",
        "thinking_effort": "medium",
        "reasoning_output": False,
        "max_tool_rounds": 0,
        "memory_depth": "auto",
        "weak_model_boost": "auto",
        "stream_replies": False,
        "response_length": "balanced",
        "tool_activity": "normal",
        "model_catalog": {},
    }


def test_settings_message_formats_snapshot():
    main = discord_settings.settings_message(
        {
            "model": "qwen3-coder",
            "planning_mode": "direct",
            "thinking_effort": "low",
            "reasoning_output": True,
            "stream_replies": True,
            "response_length": "balanced",
            "tool_activity": "normal",
        },
        page="main",
    )
    assert "qwen3-coder" in main
    assert "Direct" in main

    advanced = discord_settings.settings_message(
        {
            "model": "qwen3-coder",
            "planning_mode": "direct",
            "thinking_effort": "low",
            "reasoning_output": True,
            "stream_replies": True,
            "response_length": "balanced",
            "tool_activity": "normal",
        },
        page="advanced",
    )
    assert "Low" in advanced
    assert "**Reasoning in replies:** on" in advanced

    output = discord_settings.settings_message(
        {
            "model": "qwen3-coder",
            "planning_mode": "direct",
            "thinking_effort": "low",
            "reasoning_output": True,
            "stream_replies": True,
            "response_length": "balanced",
            "tool_activity": "normal",
        },
        page="output",
    )
    assert "Stream replies:" in output


def _fake_discord():
    from types import SimpleNamespace

    class _ButtonStyle:
        primary = 1
        secondary = 2
        success = 3

    class _UI:
        class Button:
            def __init__(self, *, label=None, style=None, emoji=None, row=0, **_kwargs):
                self.label = label
                self.style = style
                self.emoji = emoji
                self.row = row

        class Select:
            def __init__(
                self,
                *,
                placeholder=None,
                min_values=1,
                max_values=1,
                options=None,
                row=0,
                disabled=False,
                **_kwargs,
            ):
                self.placeholder = placeholder
                self.row = row
                self.options = options or []
                self.disabled = disabled

        class View:
            def __init__(self, *, timeout=600.0):
                self.timeout = timeout
                self.children = []

            def add_item(self, item):
                self.children.append(item)

    class _Embed:
        def __init__(self, **kwargs):
            self.title = kwargs.get("title")
            self.description = kwargs.get("description")
            self.color = kwargs.get("color")
            self.fields = []

        def add_field(self, *, name, value, inline=False):
            self.fields.append((name, value, inline))

        def set_footer(self, *, text):
            self.footer = text

    return SimpleNamespace(
        ButtonStyle=_ButtonStyle,
        ui=_UI,
        Embed=_Embed,
        SelectOption=lambda **kw: SimpleNamespace(**kw),
    )


async def _noop_handler(_name, _args, _principal, _ctx):
    return {"reply": "ok", "data": {}}


@pytest.mark.parametrize("page", ["main", "advanced", "output"])
def test_settings_view_has_done_on_every_page(page):
    snap = {
        "model": "composer/composer-2.5",
        "planning_mode": "direct",
        "thinking_effort": "medium",
        "reasoning_output": False,
        "max_tool_rounds": 0,
        "memory_depth": "auto",
        "weak_model_boost": "auto",
        "stream_replies": True,
        "response_length": "balanced",
        "tool_activity": "normal",
    }
    view = discord_settings.build_settings_view(
        _fake_discord(),
        handler=_noop_handler,
        principal=None,
        ctx={},
        snap=snap,
        is_owner=True,
        initial_page=page,
    )
    assert discord_settings.view_has_done_button(view), f"Done missing on {page}"


def test_settings_view_respects_discord_row_limit():
    """Each page must stay within Discord's 5 action-row cap."""
    snap = {
        "model": "composer/composer-2.5",
        "planning_mode": "direct",
        "thinking_effort": "medium",
        "reasoning_output": False,
        "max_tool_rounds": 0,
        "memory_depth": "auto",
        "weak_model_boost": "auto",
        "stream_replies": True,
        "response_length": "balanced",
        "tool_activity": "normal",
    }
    for page in ("main", "advanced", "output"):
        view = discord_settings.build_settings_view(
            _fake_discord(),
            handler=_noop_handler,
            principal=None,
            ctx={},
            snap=snap,
            is_owner=True,
            initial_page=page,
        )
        rows = {getattr(c, "row", 0) for c in view.children}
        assert max(rows) <= 4, f"{page} exceeds row limit: {rows}"
        assert len(rows) <= 5, f"{page} uses too many rows: {rows}"


def test_model_selector_includes_runtime_provider_catalog():
    catalog = {"example": [("example/model-a", "Model A"), ("example/model-b", "Model B")]}
    _content, view = discord_settings.build_model_selector_view(
        _fake_discord(),
        handler=_noop_handler,
        principal=None,
        ctx={},
        current_model="example/model-a",
        is_owner=True,
        model_catalog=catalog,
    )
    provider_select, model_select = view.children
    assert any(option.value == "example" for option in provider_select.options)
    assert [option.value for option in model_select.options] == [
        "example/model-a",
        "example/model-b",
    ]
