"""Discord UI builders for Norax /settings and model selection."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from ..commands import (
    ACTIVITY_LABELS,
    ACTIVITY_MODES,
    BOOST_LABELS,
    BOOST_MODES,
    LENGTH_LABELS,
    LENGTH_PREFS,
    MEMORY_DEPTH_LABELS,
    MEMORY_DEPTHS,
    PLANNING_LABELS,
    PLANNING_MODES,
    ROUND_CAPS,
    ROUND_LABELS,
    STREAM_LABELS,
    THINK_LABELS,
    _is_gpt_56_sol,
    model_options_for_provider,
    provider_for_model,
    provider_options_for_discord,
)

log = logging.getLogger("norax.discord.settings")

SlashHandler = Callable[
    [str, dict, Any, dict],
    Awaitable[dict | str | None],
]
ModelCatalog = dict[str, list[tuple[str, str]]]


def _snapshot_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _snapshot_choice(value: Any, choices: set[str], *, default: str) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in choices:
            return normalized
    return default


def _snapshot_rounds(value: Any) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError
        rounds = int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0
    return rounds if 0 <= rounds <= 10_000 else 0


def _provider_for_catalog(model: str, catalog: ModelCatalog) -> str:
    for provider, models in catalog.items():
        if any(model_id == model for model_id, _label in models):
            return provider
    prefix = model.partition("/")[0]
    if prefix in catalog:
        return prefix
    return provider_for_model(model) or "ollama"


def _provider_options(catalog: ModelCatalog) -> list[tuple[str, str]]:
    options = list(provider_options_for_discord())
    known = {provider for provider, _label in options}
    options.extend(
        (provider, provider.replace("_", " ").title())
        for provider in catalog
        if provider not in known
    )
    return options[:25]


def _model_options(provider: str, catalog: ModelCatalog) -> list[tuple[str, str]]:
    if provider in catalog:
        return catalog[provider][:25]
    return model_options_for_provider(provider)[:25]


_EMBED_COLOR = 0x5865F2
_PAGE_META = {
    "main": {
        "title": "Norax · Model & Agent",
        "subtitle": "Pick gateway, model, and how Norax plans work.",
        "step": "1 / 3",
    },
    "advanced": {
        "title": "Norax · Performance",
        "subtitle": "Thinking depth, tool budget, and memory retrieval.",
        "step": "2 / 3",
    },
    "output": {
        "title": "Norax · Discord Output",
        "subtitle": "How replies look and feel in this channel.",
        "step": "3 / 3",
    },
}


async def _slash(
    handler: SlashHandler,
    name: str,
    args: list[str],
    principal: Any,
    ctx: dict,
) -> dict:
    result = await handler(name, {"args": args}, principal, ctx)
    if isinstance(result, dict):
        return result
    return {"reply": str(result or "")}


async def fetch_settings_snapshot(
    handler: SlashHandler,
    principal: Any,
    ctx: dict,
) -> dict[str, Any]:
    """Load current settings via the runtime slash handler."""
    result = await _slash(handler, "settings", [], principal, ctx)
    data = result.get("data")
    if isinstance(data, dict):
        return _normalize_snapshot(data)
    reply = result.get("reply") or ""
    snap: dict[str, Any] = {}
    patterns = {
        "model": r"Model:\s*`([^`]+)`",
        "planning_mode": r"Planning:\s*`([^`]+)`",
        "thinking_effort": r"Thinking:\s*`([^`]+)`",
        "reasoning_output": r"Reasoning output:\s*`([^`]+)`",
        "max_tool_rounds": r"Tool rounds:\s*`([^`]+)`",
        "memory_depth": r"Memory depth:\s*`([^`]+)`",
        "weak_model_boost": r"Weak-model boost:\s*`([^`]+)`",
        "stream_replies": r"Stream replies:\s*`([^`]+)`",
        "response_length": r"Response length:\s*`([^`]+)`",
        "tool_activity": r"Tool activity:\s*`([^`]+)`",
    }
    for key, pat in patterns.items():
        m = re.search(pat, reply)
        if not m:
            continue
        val = m.group(1).strip()
        if key == "reasoning_output":
            snap[key] = val.lower() == "on"
        elif key == "max_tool_rounds":
            snap[key] = 0 if val.lower() == "unlimited" else int(val)
        elif key == "stream_replies":
            snap[key] = val.lower() == "on"
        elif key == "planning_mode":
            snap[key] = val.split(" ")[0].split("—")[0].strip()
        else:
            snap[key] = val
    return _normalize_snapshot(snap)


def _normalize_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": raw.get("model", "") if isinstance(raw.get("model", ""), str) else "",
        "planning_mode": _snapshot_choice(
            raw.get("planning_mode"), PLANNING_MODES, default="direct"
        ),
        "thinking_effort": _snapshot_choice(
            raw.get("thinking_effort"), set(THINK_LABELS), default="medium"
        ),
        "reasoning_output": _snapshot_bool(raw.get("reasoning_output"), default=False),
        "max_tool_rounds": _snapshot_rounds(raw.get("max_tool_rounds")),
        "memory_depth": _snapshot_choice(raw.get("memory_depth"), MEMORY_DEPTHS, default="auto"),
        "weak_model_boost": _snapshot_choice(
            raw.get("weak_model_boost"), BOOST_MODES, default="auto"
        ),
        "stream_replies": _snapshot_bool(raw.get("stream_replies"), default=True),
        "response_length": _snapshot_choice(
            raw.get("response_length"), LENGTH_PREFS, default="balanced"
        ),
        "tool_activity": _snapshot_choice(
            raw.get("tool_activity"), ACTIVITY_MODES, default="normal"
        ),
        "model_catalog": raw.get("model_catalog")
        if isinstance(raw.get("model_catalog"), dict)
        else {},
    }


def _rounds_label(snap: dict[str, Any]) -> str:
    rounds = int(snap.get("max_tool_rounds") or 0)
    return "auto" if rounds == 0 else str(rounds)


def settings_message(snap: dict[str, Any], *, note: str = "", page: str = "") -> str:
    """Plain-text fallback when embeds are unavailable."""
    payload = settings_payload(snap, note=note, page=page)
    return payload["content"]


def settings_payload(
    snap: dict[str, Any],
    *,
    note: str = "",
    page: str = "main",
    discord_mod: Any | None = None,
) -> dict[str, Any]:
    """Build Discord message payload (embed + optional content)."""
    meta = _PAGE_META.get(page, _PAGE_META["main"])
    footer = note or "Changes apply immediately · press Done when finished"
    footer = f"{footer} · {meta['step']}"

    if discord_mod is not None and hasattr(discord_mod, "Embed"):
        embed = discord_mod.Embed(
            title=meta["title"],
            description=meta["subtitle"],
            color=_EMBED_COLOR,
        )
        for name, value, inline in _page_fields(snap, page):
            embed.add_field(name=name, value=value, inline=inline)
        embed.set_footer(text=footer)
        return {"embed": embed, "content": None}

    lines = [f"**{meta['title']}**", meta["subtitle"], ""]
    for name, value, _inline in _page_fields(snap, page):
        lines.append(f"**{name}:** {value}")
    lines.append("")
    lines.append(f"_{footer}_")
    return {"content": "\n".join(lines), "embed": None}


def _page_fields(snap: dict[str, Any], page: str) -> list[tuple[str, str, bool]]:
    reasoning = "on" if snap.get("reasoning_output") else "off"
    planning = snap.get("planning_mode", "direct")
    planning_label = snap.get("planning_route") or PLANNING_LABELS.get(planning, planning)
    think = THINK_LABELS.get(
        snap.get("thinking_effort", "medium"), snap.get("thinking_effort", "medium")
    )
    mem = MEMORY_DEPTH_LABELS.get(
        snap.get("memory_depth", "auto"), snap.get("memory_depth", "auto")
    )
    boost = BOOST_LABELS.get(
        snap.get("weak_model_boost", "auto"), snap.get("weak_model_boost", "auto")
    )
    stream = STREAM_LABELS["on" if snap.get("stream_replies", True) else "off"]
    length = LENGTH_LABELS.get(
        snap.get("response_length", "balanced"), snap.get("response_length", "balanced")
    )
    activity = ACTIVITY_LABELS.get(
        snap.get("tool_activity", "normal"), snap.get("tool_activity", "normal")
    )
    provider = _provider_for_catalog(
        snap.get("model") or "",
        snap.get("model_catalog") or {},
    )

    if page == "main":
        return [
            ("Provider", f"`{provider}`", True),
            ("Model", f"`{snap.get('model', '?')}`", True),
            ("Agent mode", planning_label, False),
        ]
    if page == "advanced":
        return [
            ("Thinking", think, True),
            ("Reasoning in replies", reasoning, True),
            ("Tool rounds / turn", _rounds_label(snap), True),
            ("Memory depth", mem, False),
        ]
    if page == "output":
        return [
            ("Weak-model boost", boost, False),
            ("Stream replies", stream, True),
            ("Response length", length, True),
            ("Tool narration", activity, False),
        ]
    return [
        ("Model", f"`{snap.get('model', '?')}`", True),
        ("Planning", planning_label, True),
        ("Thinking", think, True),
        ("Reasoning", reasoning, True),
        ("Tool rounds", _rounds_label(snap), True),
        ("Memory", mem, True),
        ("Boost", boost, True),
        ("Stream", stream, True),
        ("Length", length, True),
        ("Activity", activity, True),
    ]


def build_settings_view(
    discord_mod: Any,
    *,
    handler: SlashHandler,
    principal: Any,
    ctx: dict,
    snap: dict[str, Any],
    is_owner: bool,
    initial_page: str = "main",
) -> Any:
    """Build a three-page Discord settings panel; Done visible on every page."""
    current_model = snap.get("model") or ""
    model_catalog = snap.get("model_catalog") or {}
    current_provider = _provider_for_catalog(current_model, model_catalog)
    selected_provider = {"id": current_provider}
    live_snap = dict(snap)
    status = {"text": ""}

    async def _apply(name: str, args: list[str]) -> str:
        if not is_owner:
            return "Only the owner can change settings."
        result = await _slash(handler, name, args, principal, ctx)
        reply = (result.get("reply") or "").strip()
        if isinstance(result.get("data"), dict):
            live_snap.update(_normalize_snapshot(result["data"]))
        elif name == "model" and args:
            live_snap["model"] = args[0]
        elif name == "planning" and args:
            live_snap["planning_mode"] = args[0]
        elif name == "think" and args:
            live_snap["thinking_effort"] = args[0]
        elif name == "reasoning" and args:
            live_snap["reasoning_output"] = args[0].lower() in {"on", "true", "1", "yes", "show"}
        elif name == "rounds" and args:
            raw = args[0].strip().lower()
            live_snap["max_tool_rounds"] = (
                0 if raw in {"0", "unlimited", "none", "inf"} else int(raw)
            )
        elif name == "memory" and args:
            live_snap["memory_depth"] = args[0]
        elif name == "boost" and args:
            live_snap["weak_model_boost"] = args[0]
        elif name == "stream" and args:
            live_snap["stream_replies"] = args[0].lower() in {"on", "true", "1", "yes"}
        elif name == "length" and args:
            live_snap["response_length"] = args[0]
        elif name == "activity" and args:
            live_snap["tool_activity"] = args[0]
        status["text"] = reply or "Updated."
        return status["text"]

    async def _edit(interaction, *, view, page: str = "main") -> None:
        note = status["text"] or ""
        payload = settings_payload(live_snap, note=note, page=page, discord_mod=discord_mod)
        try:
            await interaction.edit_original_response(
                content=payload.get("content"),
                embed=payload.get("embed"),
                view=view,
            )
        except Exception:  # noqa: BLE001
            log.exception("settings.edit failed")

    async def _swap(interaction, *, view, page: str) -> None:
        note = status["text"] or ""
        payload = settings_payload(live_snap, note=note, page=page, discord_mod=discord_mod)
        try:
            await interaction.response.edit_message(
                content=payload.get("content"),
                embed=payload.get("embed"),
                view=view,
            )
        except Exception:  # noqa: BLE001
            log.exception("settings.swap failed page=%s", page)

    def _add_done(view, row: int = 4) -> None:
        class _DoneButton(discord_mod.ui.Button):
            def __init__(self_b):
                super().__init__(
                    label="Done",
                    style=discord_mod.ButtonStyle.success,
                    emoji="✅",
                    row=row,
                )

            async def callback(self_b, interaction):
                try:
                    await interaction.response.defer()
                except Exception:  # noqa: BLE001
                    pass
                payload = settings_payload(live_snap, discord_mod=discord_mod)
                final_embed = payload.get("embed")
                if final_embed is not None:
                    final_embed.description = (
                        final_embed.description or ""
                    ) + "\n\n✅ **Settings saved.** Panel closed."
                final_content = payload.get("content")
                if final_content:
                    final_content += "\n\n✅ **Settings saved.** Panel closed."
                try:
                    await interaction.edit_original_response(
                        content=final_content,
                        embed=final_embed,
                        view=None,
                    )
                except Exception:  # noqa: BLE001
                    log.exception("settings.done failed")

        view.add_item(_DoneButton())

    def _add_nav(view, *, page: str, row: int = 4) -> None:
        """Navigation row — Done is always appended last."""
        if page == "main":

            class _ForwardBtn(discord_mod.ui.Button):
                def __init__(self_b):
                    super().__init__(
                        label="Performance",
                        style=discord_mod.ButtonStyle.primary,
                        emoji="⚙️",
                        row=row,
                    )

                async def callback(self_b, interaction):
                    await _swap(interaction, view=_advanced_view(), page="advanced")

            view.add_item(_ForwardBtn())

        elif page == "advanced":

            class _BackBtn(discord_mod.ui.Button):
                def __init__(self_b):
                    super().__init__(
                        label="Model",
                        style=discord_mod.ButtonStyle.secondary,
                        emoji="◀️",
                        row=row,
                    )

                async def callback(self_b, interaction):
                    await _swap(
                        interaction,
                        view=_main_view(selected_provider["id"]),
                        page="main",
                    )

            class _OutputBtn(discord_mod.ui.Button):
                def __init__(self_b):
                    super().__init__(
                        label="Output",
                        style=discord_mod.ButtonStyle.primary,
                        emoji="💬",
                        row=row,
                    )

                async def callback(self_b, interaction):
                    await _swap(interaction, view=_output_view(), page="output")

            view.add_item(_BackBtn())
            view.add_item(_OutputBtn())

        elif page == "output":

            class _PerfBackBtn(discord_mod.ui.Button):
                def __init__(self_b):
                    super().__init__(
                        label="Performance",
                        style=discord_mod.ButtonStyle.secondary,
                        emoji="◀️",
                        row=row,
                    )

                async def callback(self_b, interaction):
                    await _swap(interaction, view=_advanced_view(), page="advanced")

            view.add_item(_PerfBackBtn())

        _add_done(view, row=row)

    class _ProviderSelect(discord_mod.ui.Select):
        def __init__(self_s):
            options = [
                discord_mod.SelectOption(
                    label=f"Gateway · {label}"[:100],
                    description=f"Route via {pid}"[:100],
                    value=pid,
                    default=(pid == selected_provider["id"]),
                )
                for pid, label in _provider_options(model_catalog)
            ]
            super().__init__(
                placeholder="① Gateway / provider",
                min_values=1,
                max_values=1,
                options=options,
                row=0,
                disabled=not is_owner,
            )

        async def callback(self_s, interaction):
            selected_provider["id"] = self_s.values[0]
            await _swap(
                interaction,
                view=_main_view(selected_provider["id"]),
                page="main",
            )

    class _ModelSelect(discord_mod.ui.Select):
        def __init__(self_s, provider: str):
            options = [
                discord_mod.SelectOption(
                    label=desc[:100],
                    description=mid[:100],
                    value=mid,
                    default=(mid == live_snap.get("model")),
                )
                for mid, desc in _model_options(provider, model_catalog)
            ]
            if not options:
                options.append(
                    discord_mod.SelectOption(label="No models configured", value="__none__")
                )
            super().__init__(
                placeholder="② Model",
                min_values=1,
                max_values=1,
                options=options[:25],
                row=1,
                disabled=not is_owner,
            )

        async def callback(self_s, interaction):
            selected = self_s.values[0]
            if selected == "__none__":
                await interaction.response.send_message(
                    "No models for that provider.", ephemeral=True
                )
                return
            try:
                await interaction.response.defer()
            except Exception:  # noqa: BLE001
                pass
            await _apply("model", [selected])
            await _edit(interaction, view=_main_view(selected_provider["id"]), page="main")

    class _PlanningSelect(discord_mod.ui.Select):
        def __init__(self_s):
            options = [
                discord_mod.SelectOption(
                    label=PLANNING_LABELS.get(mode, mode)[:100],
                    description=f"mode={mode}"[:100],
                    value=mode,
                    default=(mode == live_snap.get("planning_mode", "direct")),
                )
                for mode in sorted(PLANNING_MODES)
            ]
            super().__init__(
                placeholder="③ Agent mode (planning)",
                min_values=1,
                max_values=1,
                options=options,
                row=2,
                disabled=not is_owner,
            )

        async def callback(self_s, interaction):
            try:
                await interaction.response.defer()
            except Exception:  # noqa: BLE001
                pass
            await _apply("planning", [self_s.values[0]])
            await _edit(interaction, view=_main_view(selected_provider["id"]), page="main")

    class _ThinkSelect(discord_mod.ui.Select):
        def __init__(self_s):
            levels = ["off", "low", "medium", "high", "xhigh"]
            if _is_gpt_56_sol(current_model):
                levels.extend(["max", "ultra"])
            current_effort = live_snap.get("thinking_effort", "medium")
            # Drop sol-only levels from the dropdown if the user somehow still has one selected
            # after switching away from GPT-5.6 Sol; it won't be an option and will fall back to medium.
            options = [
                discord_mod.SelectOption(
                    label=THINK_LABELS.get(lv, lv)[:100],
                    description=f"level={lv}"[:100],
                    value=lv,
                    default=(lv == current_effort),
                )
                for lv in levels
            ]
            super().__init__(
                placeholder="④ Thinking effort",
                min_values=1,
                max_values=1,
                options=options,
                row=0,
                disabled=not is_owner,
            )

        async def callback(self_s, interaction):
            try:
                await interaction.response.defer()
            except Exception:  # noqa: BLE001
                pass
            await _apply("think", [self_s.values[0]])
            await _edit(interaction, view=_advanced_view(), page="advanced")

    class _ReasoningSelect(discord_mod.ui.Select):
        def __init__(self_s):
            current = "on" if live_snap.get("reasoning_output") else "off"
            options = [
                discord_mod.SelectOption(
                    label="Show reasoning in replies",
                    description="Include model thinking blocks in Discord output",
                    value="on",
                    default=(current == "on"),
                ),
                discord_mod.SelectOption(
                    label="Hide reasoning",
                    description="Final answer only — cleaner Discord messages",
                    value="off",
                    default=(current == "off"),
                ),
            ]
            super().__init__(
                placeholder="⑤ Reasoning visibility",
                min_values=1,
                max_values=1,
                options=options,
                row=1,
                disabled=not is_owner,
            )

        async def callback(self_s, interaction):
            try:
                await interaction.response.defer()
            except Exception:  # noqa: BLE001
                pass
            await _apply("reasoning", [self_s.values[0]])
            await _edit(interaction, view=_advanced_view(), page="advanced")

    class _RoundsSelect(discord_mod.ui.Select):
        def __init__(self_s):
            current = int(live_snap.get("max_tool_rounds") or 0)
            options = [
                discord_mod.SelectOption(
                    label=ROUND_LABELS.get(cap, str(cap))[:100],
                    description=(
                        f"{cap} tool rounds per turn"
                        if cap
                        else "Use the bounded 250-round default"
                    ),
                    value=str(cap),
                    default=(cap == current),
                )
                for cap in ROUND_CAPS
            ]
            super().__init__(
                placeholder="⑥ Tool rounds per turn",
                min_values=1,
                max_values=1,
                options=options,
                row=2,
                disabled=not is_owner,
            )

        async def callback(self_s, interaction):
            try:
                await interaction.response.defer()
            except Exception:  # noqa: BLE001
                pass
            val = self_s.values[0]
            arg = "unlimited" if val == "0" else val
            await _apply("rounds", [arg])
            await _edit(interaction, view=_advanced_view(), page="advanced")

    class _MemorySelect(discord_mod.ui.Select):
        def __init__(self_s):
            current = live_snap.get("memory_depth", "auto")
            options = [
                discord_mod.SelectOption(
                    label=MEMORY_DEPTH_LABELS.get(mode, mode)[:100],
                    description=f"mode={mode}"[:100],
                    value=mode,
                    default=(mode == current),
                )
                for mode in sorted(MEMORY_DEPTHS)
            ]
            super().__init__(
                placeholder="⑦ Memory retrieval depth",
                min_values=1,
                max_values=1,
                options=options,
                row=3,
                disabled=not is_owner,
            )

        async def callback(self_s, interaction):
            try:
                await interaction.response.defer()
            except Exception:  # noqa: BLE001
                pass
            await _apply("memory", [self_s.values[0]])
            await _edit(interaction, view=_advanced_view(), page="advanced")

    class _BoostSelect(discord_mod.ui.Select):
        def __init__(self_s):
            current = live_snap.get("weak_model_boost", "auto")
            options = [
                discord_mod.SelectOption(
                    label=BOOST_LABELS.get(mode, mode)[:100],
                    description=f"mode={mode}"[:100],
                    value=mode,
                    default=(mode == current),
                )
                for mode in sorted(BOOST_MODES)
            ]
            super().__init__(
                placeholder="⑧ Weak-model tool boost",
                min_values=1,
                max_values=1,
                options=options,
                row=0,
                disabled=not is_owner,
            )

        async def callback(self_s, interaction):
            try:
                await interaction.response.defer()
            except Exception:  # noqa: BLE001
                pass
            await _apply("boost", [self_s.values[0]])
            await _edit(interaction, view=_output_view(), page="output")

    class _StreamSelect(discord_mod.ui.Select):
        def __init__(self_s):
            current = "on" if live_snap.get("stream_replies", True) else "off"
            options = [
                discord_mod.SelectOption(
                    label=STREAM_LABELS["on"][:100],
                    description="Live token streaming in Discord (typing + edit-in-place)",
                    value="on",
                    default=(current == "on"),
                ),
                discord_mod.SelectOption(
                    label=STREAM_LABELS["off"][:100],
                    description="Single reply after completion — less flicker",
                    value="off",
                    default=(current == "off"),
                ),
            ]
            super().__init__(
                placeholder="⑨ Stream replies in Discord",
                min_values=1,
                max_values=1,
                options=options,
                row=1,
                disabled=not is_owner,
            )

        async def callback(self_s, interaction):
            try:
                await interaction.response.defer()
            except Exception:  # noqa: BLE001
                pass
            await _apply("stream", [self_s.values[0]])
            await _edit(interaction, view=_output_view(), page="output")

    class _LengthSelect(discord_mod.ui.Select):
        def __init__(self_s):
            current = live_snap.get("response_length", "balanced")
            options = [
                discord_mod.SelectOption(
                    label=LENGTH_LABELS.get(mode, mode)[:100],
                    description=f"pref={mode}"[:100],
                    value=mode,
                    default=(mode == current),
                )
                for mode in sorted(LENGTH_PREFS)
            ]
            super().__init__(
                placeholder="⑩ Response length preference",
                min_values=1,
                max_values=1,
                options=options,
                row=2,
                disabled=not is_owner,
            )

        async def callback(self_s, interaction):
            try:
                await interaction.response.defer()
            except Exception:  # noqa: BLE001
                pass
            await _apply("length", [self_s.values[0]])
            await _edit(interaction, view=_output_view(), page="output")

    class _ActivitySelect(discord_mod.ui.Select):
        def __init__(self_s):
            current = live_snap.get("tool_activity", "normal")
            options = [
                discord_mod.SelectOption(
                    label=ACTIVITY_LABELS.get(mode, mode)[:100],
                    description=f"mode={mode}"[:100],
                    value=mode,
                    default=(mode == current),
                )
                for mode in sorted(ACTIVITY_MODES)
            ]
            super().__init__(
                placeholder="⑪ Tool-call narration style",
                min_values=1,
                max_values=1,
                options=options,
                row=3,
                disabled=not is_owner,
            )

        async def callback(self_s, interaction):
            try:
                await interaction.response.defer()
            except Exception:  # noqa: BLE001
                pass
            await _apply("activity", [self_s.values[0]])
            await _edit(interaction, view=_output_view(), page="output")

    def _main_view(provider: str | None = None):
        view = discord_mod.ui.View(timeout=600.0)
        provider = provider or selected_provider["id"]
        view.add_item(_ProviderSelect())
        view.add_item(_ModelSelect(provider))
        view.add_item(_PlanningSelect())
        _add_nav(view, page="main", row=3)
        return view

    def _advanced_view():
        view = discord_mod.ui.View(timeout=600.0)
        view.add_item(_ThinkSelect())
        view.add_item(_ReasoningSelect())
        view.add_item(_RoundsSelect())
        view.add_item(_MemorySelect())
        _add_nav(view, page="advanced", row=4)
        return view

    def _output_view():
        view = discord_mod.ui.View(timeout=600.0)
        view.add_item(_BoostSelect())
        view.add_item(_StreamSelect())
        view.add_item(_LengthSelect())
        view.add_item(_ActivitySelect())
        _add_nav(view, page="output", row=4)
        return view

    pages = {
        "main": lambda: _main_view(current_provider),
        "advanced": _advanced_view,
        "output": _output_view,
    }
    return pages.get(initial_page, pages["main"])()


def build_model_selector_view(
    discord_mod: Any,
    *,
    handler: SlashHandler,
    principal: Any,
    ctx: dict,
    current_model: str,
    is_owner: bool,
    model_catalog: ModelCatalog | None = None,
) -> tuple[str, Any]:
    """Two-row provider + model selector (shared by /model and /models)."""
    catalog: ModelCatalog = model_catalog or {}
    current_provider = _provider_for_catalog(current_model, catalog)
    selected_provider = {"id": current_provider}

    async def _set_model(selected: str) -> str:
        if not is_owner:
            return "Only the owner can change the model."
        result = await _slash(handler, "model", [selected], principal, ctx)
        return (result.get("reply") or "").strip() or f"Model set to `{selected}`."

    class _ProviderSelect(discord_mod.ui.Select):
        def __init__(self_s):
            options = [
                discord_mod.SelectOption(
                    label=label,
                    description=pid[:100],
                    value=pid,
                    default=(pid == selected_provider["id"]),
                )
                for pid, label in _provider_options(catalog)
            ]
            super().__init__(
                placeholder="1) Select provider…",
                min_values=1,
                max_values=1,
                options=options,
                row=0,
            )

        async def callback(self_s, sel_interaction):
            selected_provider["id"] = self_s.values[0]
            view = _ModelView(selected_provider["id"])
            try:
                await sel_interaction.response.edit_message(
                    content=(
                        f"**Current model:** `{current_model}`\n"
                        f"**Provider:** `{selected_provider['id']}`\n"
                        "Now select a model:"
                    ),
                    view=view,
                )
            except Exception:  # noqa: BLE001
                log.exception("provider_select.edit failed")

    class _ModelSelect(discord_mod.ui.Select):
        def __init__(self_s, provider: str):
            options = [
                discord_mod.SelectOption(
                    label=desc[:100],
                    description=mid[:100],
                    value=mid,
                    default=(mid == current_model),
                )
                for mid, desc in _model_options(provider, catalog)
            ]
            if not options:
                options.append(
                    discord_mod.SelectOption(label="No models configured", value="__none__")
                )
            super().__init__(
                placeholder="2) Select model…",
                min_values=1,
                max_values=1,
                options=options[:25],
                row=1,
            )

        async def callback(self_s, sel_interaction):
            selected = self_s.values[0]
            if selected == "__none__":
                await sel_interaction.response.send_message(
                    "No models configured for that provider.", ephemeral=True
                )
                return
            try:
                await sel_interaction.response.defer()
            except Exception:  # noqa: BLE001
                pass
            try:
                text = await _set_model(selected)
            except Exception as exc:  # noqa: BLE001
                text = f"Failed to set model: {exc}"
            try:
                await sel_interaction.edit_original_response(content=text, view=None)
            except Exception:  # noqa: BLE001
                log.exception("model_select.edit failed")

    class _ModelView(discord_mod.ui.View):
        def __init__(self_v, provider: str | None = None):
            super().__init__(timeout=180.0)
            provider = provider or selected_provider["id"]
            self_v.add_item(_ProviderSelect())
            self_v.add_item(_ModelSelect(provider))

    content = (
        f"**Current model:** `{current_model}`\n"
        f"**Provider:** `{current_provider}`\n"
        "Select provider, then model:"
    )
    return content, _ModelView(current_provider)


def view_has_done_button(view: Any) -> bool:
    """True if the view includes a Done navigation button."""
    for child in getattr(view, "children", ()):
        label = getattr(child, "label", None)
        if label == "Done":
            return True
    return False
