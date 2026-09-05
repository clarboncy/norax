"""Weak-model boosters for explicitly small/local models.

Four active improvements:
  1. Few-shot golden examples injected per task type
  2. Mandatory structured scratchpad before tool calls
  3. Lean, capability-preserving tool schemas
  4. Per-turn tool-result cache for immutable observations

These are injected at different layers:
  - Few-shot: system prompt injection (assembler / scaffold)
  - Scratchpad: system prompt directive
  - Lean schemas: tool registry rendering
  - Cache: agent_loop dispatch layer
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from typing import Any

log = logging.getLogger("norax.brain.weak_model_boost")


# ---------------------------------------------------------------------------
# 1. Few-shot golden examples per task type
# ---------------------------------------------------------------------------

# These are "mythos-style" golden examples — compact, realistic, end-to-end
# demonstrations of correct tool-use cycles. They show the model:
#   a) How to plan before acting
#   b) Correct tool call JSON format
#   c) How to interpret results and decide next steps
#   d) When to stop and give a final answer
#
# Key design principles (from research on few-shot prompting for tool use):
#   - Examples must be COMPLETE cycles (plan → call → result → next)
#   - Use the EXACT tool schemas the model will see
#   - Include at least one error-recovery example
#   - Keep examples SHORT (weak models lose context fast)
#   - Match the task type (coding examples for coding tasks, etc.)

GOLDEN_EXAMPLES: dict[str, str] = {
    "coding": """GOLDEN_EXAMPLES;task_type=coding
Example 1 — Read before edit:
  User: "Fix the typo in config.py"
  Agent thinks: I need to read config.py first to find the typo.
  Agent calls: read(path="config.py")
  Result: {"ok": true, "path": "config.py", "content": "db_hoost = \\"localhost\\"", "total_lines": 1}
  Agent thinks: Found it — "db_hoost" should be "db_host". I'll edit with exact match.
  Agent calls: edit(path="config.py", old="db_hoost", new="db_host")
  Result: {"ok": true, "path": "config.py", "diff_bytes": -1}
  Agent thinks: Edit succeeded. Verify by reading back.
  Agent calls: read(path="config.py")
  Result: {"ok": true, "content": "db_host = \\"localhost\\"", "total_lines": 1}
  Agent answers: Fixed — `db_hoost` → `db_host` in config.py. Verified.

Example 2 — Error recovery:
  Agent calls: edit(path="app.py", old="def main():", new="def main(args):")
  Result: {"ok": false, "error": "old_text_not_found"}
  Agent thinks: My old text didn't match. I need to re-read the file.
  Agent calls: read(path="app.py")
  Result: {"ok": true, "content": "def main( ):", "total_lines": 5}
  Agent thinks: There's a space inside parens. Use exact text.
  Agent calls: edit(path="app.py", old="def main( ):", new="def main(args):")
  Result: {"ok": true}
""",
    "debugging": """GOLDEN_EXAMPLES;task_type=debugging
Example 1 — Diagnose then fix:
  User: "Tests are failing"
  Agent thinks: Run the tests first to see the actual error.
  Agent calls: exec(command="python -m pytest tests/ -x --tb=short", timeout=30)
  Result: {"ok": false, "exit_code": 1, "stderr": "ImportError: cannot import name 'Config' from 'settings'"}
  Agent thinks: Import error — Config was likely renamed. Check settings.py.
  Agent calls: read(path="settings.py")
  Result: {"ok": true, "content": "class AppConfig:\\n    ...", "total_lines": 20}
  Agent thinks: Class is AppConfig not Config. Fix the import.
  Agent calls: exec(command="grep -rn 'from settings import Config' .", timeout=10)
  Result: {"ok": true, "stdout": "./app.py:3:from settings import Config"}
  Agent calls: edit(path="app.py", old="from settings import Config", new="from settings import AppConfig as Config")
  Result: {"ok": true}
  Agent calls: exec(command="python -m pytest tests/ -x --tb=short", timeout=30)
  Result: {"ok": true, "exit_code": 0, "stdout": "5 passed"}
  Agent answers: Fixed. `Config` was renamed to `AppConfig` in settings.py. Updated import in app.py. All 5 tests pass.
""",
    "research": """GOLDEN_EXAMPLES;task_type=research
Example 1 — Search then synthesize:
  User: "What's the best Python web framework for async?"
  Agent thinks: Search for current comparisons.
  Agent calls: web_search(query="best Python async web framework 2025 comparison")
  Result: {"ok": true, "items": [{"title": "FastAPI vs Starlette vs Sanic", "snippet": "FastAPI leads..."}]}
  Agent thinks: Got results. Let me check the top source for details.
  Agent calls: web_fetch(url="https://example.com/comparison")
  Result: {"ok": true, "text": "FastAPI: 4.8/5 stars, best for APIs. Starlette: lower-level..."}
  Agent answers: FastAPI is the consensus pick for async Python — best ecosystem, typing support, and performance. Starlette is better if you need a minimal foundation.
""",
    "general": """GOLDEN_EXAMPLES;task_type=general
Example 1 — Simple tool use:
  User: "What files are in the project?"
  Agent calls: list_dir(path=".")
  Result: {"ok": true, "entries": [{"name": "README.md", "kind": "file"}, {"name": "src", "kind": "dir"}]}
  Agent answers: Two items: README.md and src/ directory.
""",
}


def get_golden_examples(task_type: str) -> str:
    """Return few-shot golden examples for the given task type.

    Falls back to 'general' if no specific examples exist.
    """
    return GOLDEN_EXAMPLES.get(task_type, GOLDEN_EXAMPLES.get("general", ""))


# ---------------------------------------------------------------------------
# 3. Mandatory structured scratchpad before tool calls
# ---------------------------------------------------------------------------

SCRATCHPAD_DIRECTIVE = """SCRATCHPAD_PROTOCOL;weight=W4
Before EVERY tool call, you MUST think through your plan in a brief scratchpad.
Format:
  <think>
  GOAL: what I'm trying to achieve this step
  PLAN: which tool to call and why
  RISK: what could go wrong
  </think>
  [then emit the tool_call]

Rules:
- The <think> block is MANDATORY before any tool_call
- Keep it to 2-3 lines max — don't over-explain
- If you're about to repeat a failed call, your PLAN must explain what changed
- If RISK is "same call that just failed", STOP and try a different approach
"""


def build_scratchpad_directive(*, is_weak_model: bool = True) -> str:
    """Return the scratchpad directive for injection into system prompt.

    Only injected for weak/local models — strong models (Claude, GPT-4+)
    already reason internally and adding this wastes tokens.
    """
    if not is_weak_model:
        return ""
    return SCRATCHPAD_DIRECTIVE


# ---------------------------------------------------------------------------
# 4. Lean tool schemas for weak models
# ---------------------------------------------------------------------------

# Weak models benefit from shorter descriptions and less decorative schema
# metadata, but optional arguments are still real capabilities. Removing
# ``cwd`` or ``timeout`` can make a valid task impossible, so lean schemas
# preserve every property and requirement.

# Short descriptions for weak models
_LEAN_DESCRIPTIONS: dict[str, str] = {
    "read": "Read a file. Use offset/limit for a specific line range.",
    "list_dir": "List files in a directory. Use offset/limit to page large directories.",
    "write": "Write/create a file with content.",
    "write_chunk": "Write a large file in chunks. mode='start' for first chunk, 'append' for rest. Set final=true on last.",
    "edit": "Replace text in a file. `old` must EXACTLY match file content. Use `read` first.",
    "exec": "Run a shell command. Returns stdout/stderr/exit_code.",
    "search_memory": "Search memory for stored facts.",
    "web_search": "Search the web. Returns titles and snippets.",
    "web_fetch": "Fetch a URL and return its text.",
    "append_memory": "Append a line to a memory file.",
    "status": "Return runtime status.",
}


def make_lean_tool_schemas(
    full_schemas: list[dict],
    *,
    is_weak_model: bool = True,
) -> list[dict]:
    """Trim tool schemas for weak models.

    Strong models get the full schema. Weak models get:
    - Every functional parameter and requirement
    - Shorter descriptions
    - Unchanged validation rules and function metadata
    """
    if not is_weak_model:
        return full_schemas

    lean: list[dict] = []
    for schema in full_schemas:
        fn = schema.get("function", {})
        name = fn.get("name", "")

        lean_desc = _LEAN_DESCRIPTIONS.get(name, fn.get("description", ""))
        compact = copy.deepcopy(schema)
        compact.setdefault("function", {})["description"] = lean_desc
        lean.append(compact)

    return lean


# ---------------------------------------------------------------------------
# 5. Per-turn tool result cache (dedup identical calls within a turn)
# ---------------------------------------------------------------------------


class TurnToolCache:
    """In-memory cache for tool results within a single agent turn.

    Prevents weak models from re-calling the same tool with identical args
    multiple times in the same turn. Common failure modes this catches:
      - re-reading the same file 3-4 times
      - re-listing the same directory

    Only caches bounded local filesystem observations. Dynamic status, memory,
    and network reads retain their own freshness/caching policies rather than
    being hidden behind this extra turn cache. Writes/exec/mutations are never
    cached, and the agent loop invalidates all observations after any successful
    mutation.

    Cache keys include the full args so different paths/queries are distinct.
    """

    CACHEABLE_TOOLS = frozenset({"read", "list_dir"})
    MAX_ENTRIES = 64

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}
        self._hits: int = 0
        self._misses: int = 0

    def _key(self, name: str, args: dict) -> str:
        body = json.dumps({"t": name, "a": args}, sort_keys=True, default=str)
        return hashlib.sha256(body.encode()).hexdigest()[:20]

    def get(self, name: str, args: dict) -> dict | None:
        """Return cached result if available, else None."""
        if name not in self.CACHEABLE_TOOLS:
            return None
        key = self._key(name, args)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
            log.debug("turn_cache.hit tool=%s hits=%d", name, self._hits)
            cached = copy.deepcopy(result)
            cached["_cached"] = True
            return cached
        self._misses += 1
        return None

    def put(self, name: str, args: dict, result: dict) -> None:
        """Cache a successful tool result."""
        if name not in self.CACHEABLE_TOOLS:
            return
        if result.get("ok") is not True:
            return  # don't cache failures
        key = self._key(name, args)
        if key not in self._cache and len(self._cache) >= self.MAX_ENTRIES:
            del self._cache[next(iter(self._cache))]
        self._cache[key] = copy.deepcopy(result)

    def invalidate_all(self) -> None:
        """Discard observations after a mutation while preserving diagnostics."""
        self._cache.clear()

    def invalidate_path(self, path: str) -> None:
        """Invalidate cache entries that might reference a modified path.

        Called after write/edit/write_chunk to ensure subsequent reads
        see fresh content.
        """
        to_remove = []
        for key, result in self._cache.items():
            cached_path = result.get("path", "")
            if (
                cached_path
                and path
                and (
                    cached_path == path or cached_path.endswith(path) or path.endswith(cached_path)
                )
            ):
                to_remove.append(key)
        for key in to_remove:
            del self._cache[key]

    @property
    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "cached_entries": len(self._cache),
            "hit_rate": round(self._hits / max(1, self._hits + self._misses), 3),
        }

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0


# ---------------------------------------------------------------------------
# Integration: detect if current model is "weak"
# ---------------------------------------------------------------------------

# Models considered "strong" (don't need boosters)
_STRONG_MODELS = {
    "claude-opus",
    "claude-sonnet",
    "claude-haiku",
    "gpt-5",
    "gpt-5.3",
    "gpt-5.4",
    "gpt-5.5",
    "o1",
    "o3",
    "o4",
    "gemini-2",
    "gemini-pro",
}


def is_weak_model(model_id: str, *, provider_kind: str | None = None) -> bool:
    """Heuristic: is this model likely to benefit from boosters?

    Returns True for explicitly small/local models.
    Returns False for Claude, GPT-4+, Gemini Pro, etc.

    Ollama is a transport, not a capability tier: it serves both small local
    models and frontier cloud models. Use the model-specific Ollama profile
    for every model name we recognize, regardless of which router object the
    caller happens to hold. GatewayRouter deliberately hides the final
    transport, so provider-wide checks are not reliable here.
    """
    if not isinstance(model_id, str):
        raise TypeError("model_id must be text")
    if provider_kind is not None and not isinstance(provider_kind, str):
        raise TypeError("provider_kind must be text or None")
    if not model_id:
        return True
    model_lower = model_id.lower()

    # Hosted/cloud routes must never inherit small-local-model constraints just
    # because their family name contains "qwen", "gemma", or "deepseek".
    if ":cloud" in model_lower or model_lower.startswith(
        (
            "openrouter/",
            "openai/",
            "anthropic/",
        )
    ):
        return False

    # Check against known strong model families
    for strong in _STRONG_MODELS:
        if strong in model_lower:
            return False

    try:
        from ..gateway_client.ollama_profiles import is_weak_ollama_model

        if is_weak_ollama_model(model_id):
            return True
    except Exception:  # noqa: BLE001
        # Fall through to conservative name heuristics if profiles cannot be
        # imported during bootstrap.
        pass

    # Known local families without an explicit size suffix. Keep this list
    # intentionally narrow: unknown or hosted models default to the baseline
    # harness rather than receiving destructive schema/prompt rewrites.
    if model_lower in {"gemma4:12b", "ollama/gemma4:12b"}:
        return True
    if model_lower.startswith(
        (
            "norax-gemma4-12b",
            "xentriom/gemma-4-12b",
            "deepseek-r1:14b",
        )
    ):
        return True

    # FreeToken serves local GGUF models — always treat as weak/local.
    # These are quantized models running on local hardware with limited
    # context budgets. Full 40-tool schemas waste 16K+ chars per turn.
    if "freetoken/" in model_lower or model_lower.endswith(".gguf"):
        return True
    if provider_kind and provider_kind.strip().lower() == "freetoken":
        return True

    return False


# ---------------------------------------------------------------------------
# Master injection: compose all boosters into system prompt additions
# ---------------------------------------------------------------------------


def compose_weak_model_boosters(
    *,
    task_type: str,
    model_id: str,
    tool_schemas: list[dict] | None = None,
    provider_kind: str | None = None,
    force_boost: str = "auto",
) -> dict[str, Any]:
    """Compose all weak-model boosters and return injection points.

    Returns a dict with:
      - "system_additions": str to append to system prompt
      - "lean_schemas": trimmed tool schemas (or original if strong model)
      - "is_weak": whether boosters were applied

    `provider_kind` lets explicit ``on`` mode cover unknown local Ollama models
    without misclassifying hosted ``:cloud`` models.

    `force_boost`:
      - "auto" — heuristic detection (default)
      - "on"   — also boost unknown local Ollama/FreeToken models, but never a
                 recognized strong or hosted/cloud model
      - "off"  — never apply boosters
    """
    if not isinstance(task_type, str) or len(task_type) > 128:
        raise ValueError("task_type must be text of at most 128 characters")
    if not isinstance(force_boost, str):
        raise TypeError("force_boost must be text")
    mode = force_boost.strip().lower()
    if mode not in {"auto", "on", "off"}:
        raise ValueError("force_boost must be auto, on, or off")
    automatic = is_weak_model(model_id, provider_kind=provider_kind)
    if mode == "on":
        model_lower = model_id.lower()
        explicitly_hosted = ":cloud" in model_lower or model_lower.startswith(
            ("openrouter/", "openai/", "anthropic/")
        )
        explicitly_strong = any(strong in model_lower for strong in _STRONG_MODELS)
        local_transport = str(provider_kind or "").strip().lower() in {
            "ollama",
            "freetoken",
        }
        weak = automatic or (local_transport and not explicitly_hosted and not explicitly_strong)
    elif mode == "off":
        weak = False
    else:
        weak = automatic

    system_parts: list[str] = []
    lean_schemas = tool_schemas or []

    if weak:
        # 2. Few-shot golden examples
        examples = get_golden_examples(task_type)
        if examples:
            system_parts.append(examples)

        # 3. Scratchpad directive
        scratchpad = build_scratchpad_directive(is_weak_model=True)
        if scratchpad:
            system_parts.append(scratchpad)

        # 4. Lean tool schemas
        if tool_schemas:
            lean_schemas = make_lean_tool_schemas(tool_schemas, is_weak_model=True)

    return {
        "system_additions": "\n\n".join(system_parts),
        "lean_schemas": lean_schemas,
        "is_weak": weak,
    }
