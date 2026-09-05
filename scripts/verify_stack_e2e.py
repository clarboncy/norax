#!/usr/bin/env python3
"""End-to-end stack verification — run from repo root."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(name: str, fn) -> bool:
    try:
        fn()
        print(f"  OK  {name}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {name}: {exc}")
        return False


def main() -> int:
    print("=== Norax stack E2E verify ===\n")
    ok = True

    def imports():
        for mod in (
            "norax.dispatch.tools",
            "norax.dispatch.router",
            "norax.adapter.discord_settings",
            "norax.adapter.discord_in",
            "norax.commands",
            "norax.shell.session",
            "norax.shell.manager",
            "norax.memory.store",
            "norax.brain.agent_loop",
            "norax.runtime.core",
        ):
            importlib.import_module(mod)

    ok &= check("core imports", imports)

    def router():
        from norax.dispatch.router import normalize_tool_name

        assert normalize_tool_name("Shell") == "shell"
        assert normalize_tool_name("execute") == "exec"
        assert normalize_tool_name("read_file") == "read"

    ok &= check("tool router aliases", router)

    def registry():
        from norax.dispatch.tools import REGISTRY

        for key in ("exec", "shell", "read", "write", "edit", "web_search"):
            assert key in REGISTRY, f"missing {key}"

    ok &= check("tool registry keys", registry)

    def settings_snapshot():
        from norax.commands import RuntimeHandle, settings_snapshot

        h = RuntimeHandle(
            default_model="test-model",
            set_default_model=lambda _: None,
            started_at=0.0,
            wall_started_at=0.0,
            metrics=None,
            gateway_base_url="http://127.0.0.1:8776/v1",
            event_log=None,
            reset_window_for=lambda _: False,
            outbound=None,
        )
        snap = settings_snapshot(h, ollama_models=[])
        assert snap["model"] == "test-model"
        assert "planning_mode" in snap
        assert "tool_activity" in snap
        assert "planning_route" in snap
        assert "planner_model" in snap
        assert "executor_model" in snap

    ok &= check("settings snapshot", settings_snapshot)

    def settings_views():
        from types import SimpleNamespace

        from norax.adapter import discord_settings

        class Button:
            def __init__(self, *, label=None, style=None, emoji=None, row=0, **_k):
                self.label = label
                self.row = row

        class Select:
            def __init__(self, *, row=0, **_k):
                self.row = row

        class View:
            def __init__(self, *, timeout=600.0):
                self.children = []

            def add_item(self, item):
                self.children.append(item)

        discord_mod = SimpleNamespace(
            ButtonStyle=SimpleNamespace(primary=1, secondary=2, success=3),
            ui=SimpleNamespace(Button=Button, Select=Select, View=View),
            Embed=object,
            SelectOption=lambda **kw: SimpleNamespace(**kw),
        )
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

        async def handler(*_a, **_k):
            return {"reply": "ok"}

        for page in ("main", "advanced", "output"):
            view = discord_settings.build_settings_view(
                discord_mod,
                handler=handler,
                principal=None,
                ctx={},
                snap=snap,
                is_owner=True,
                initial_page=page,
            )
            assert discord_settings.view_has_done_button(view), page
            rows = {getattr(c, "row", 0) for c in view.children}
            assert max(rows) <= 4, f"{page} row overflow: {rows}"

    ok &= check("discord settings views (Done + row limits)", settings_views)

    def sleep_index():
        from norax.memory.store import MemoryStore

        memory_root = Path(os.environ.get("NORAX_MEMORY_ROOT", ROOT / "memory"))
        store = MemoryStore(root=memory_root)
        store.refresh()
        archived = [n for n in store.sleep if "archive" in str(n.path)]
        processed = [n for n in store.sleep if "processed_dumps" in str(n.path)]
        assert not archived, f"archive indexed: {len(archived)}"
        assert not processed, f"processed_dumps indexed: {len(processed)}"

    if os.environ.get("NORAX_VERIFY_MEMORY_ROOT"):
        os.environ["NORAX_MEMORY_ROOT"] = os.environ["NORAX_VERIFY_MEMORY_ROOT"]
        ok &= check("active sleep archive not indexed", sleep_index)
    else:
        print("  INFO active memory check not selected (set NORAX_VERIFY_MEMORY_ROOT)")

    print("\n=== RESULT ===")
    if ok:
        print("All checks passed.")
        return 0
    print("Some checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
