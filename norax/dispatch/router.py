"""Map external IDE tool names to Norax registry keys.

Cursor/Composer and other agentic IDEs emit tool names like Shell, Grep,
run_terminal_cmd. Norax owns execution — this module is the single façade
for translating provider-specific names into our motor tools.
"""

from __future__ import annotations

# External (IDE) name (lowercase) → Norax registry key
EXTERNAL_TOOL_MAP: dict[str, str] = {
    # Stateful shell — IDE session semantics (cwd persists)
    "shell": "shell",
    "bash": "shell",
    "run_terminal_cmd": "shell",
    "run_command": "shell",
    "execute_command": "shell",
    "run": "shell",
    "terminal": "shell",
    "cmd": "shell",
    # One-shot subprocess
    "execute": "exec",
    # Files
    "read_file": "read",
    "write_file": "write",
    "list_directory": "list_dir",
    # Memory / web
    "search": "search_memory",
    "websearch": "web_search",
}

# Back-compat alias used across dispatch + agent_loop
TOOL_ALIASES = EXTERNAL_TOOL_MAP


def normalize_tool_name(name: str) -> str:
    """Map explicit provider aliases to canonical Norax registry keys.

    Tool identity is an execution boundary. Unknown or misspelled names remain
    unknown so the dispatcher can reject them; a similarity guess must never
    turn model output into a different executable operation.
    """
    key = (name or "").strip()
    if not key:
        return key
    lower = key.lower().replace("-", "_")

    # Tier 1: external alias map
    if lower in EXTERNAL_TOOL_MAP:
        return EXTERNAL_TOOL_MAP[lower]
    return lower
