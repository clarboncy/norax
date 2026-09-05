"""Cheap turn-level policy for context/tool efficiency.

Keeps owner capable while reducing prompt cost: small stable tool/schema set,
adaptive retrieval, and skill hints based on the actual request.
"""

from __future__ import annotations

import re

CORE_OWNER_TOOLS = [
    "read",
    "list_dir",
    "search_memory",
    "status",
    "exec",
    "shell",
    "write",
    "write_chunk",
    "edit",
    "append_memory",
]

REMOTE_TOOLS = [
    "remote_enroll",
    "remote_list_nodes",
    "remote_exec",
    "remote_read",
    "remote_list",
    "remote_write",
]

ALL_OWNER_TOOLS = (
    CORE_OWNER_TOOLS
    + [
        "web_fetch",
        "web_search",
        "deep_research",
        "message_send",
        "gateway_config_patch",
        "schedule_reminder",
        "browser",
        "sandbox_exec",
        "repo_explore",
        "computer_use",
    ]
    + REMOTE_TOOLS
)


def tool_names_for_turn(body: str, sender_tier: str) -> list[str]:
    """Return the complete owner manifest or a least-privilege user subset."""
    del body  # owner capability availability must not depend on prompt wording
    base = ["read", "list_dir", "search_memory", "status"]

    if sender_tier != "owner":
        base += ["web_fetch", "web_search"]
        if sender_tier in {"user", "admin"}:
            base += ["write", "write_chunk", "edit", "append_memory"]
        return list(dict.fromkeys(base))

    # Owner turns expose the complete stable tool set. Tool schemas are prompt-
    # cached, and a full manifest prevents an innocent wording mismatch from
    # making a capability unreachable in the middle of an autonomous task.
    # Non-owner turns remain least-privilege above.
    return list(dict.fromkeys(ALL_OWNER_TOOLS))


def memory_k_for_turn(body: str, sender_tier: str) -> int:
    text = (body or "").lower()
    if len(text) < 80 and not re.search(
        r"\b(memory|remember|procedure|twitter|browser|system|debug|audit|sync|staging|provider|websearch|research)\b",
        text,
    ):
        return 5
    if re.search(
        r"\b(audit|end to end|system|debug|procedure|skill|memory|twitter|provider|staging|sync|research|websearch)\b",
        text,
    ):
        return 16 if sender_tier == "owner" else 8
    return 10 if sender_tier == "owner" else 6


def skill_query_terms(body: str) -> list[str]:
    text = (body or "").lower()
    out: list[str] = []
    mapping = {
        "twitter": ["twitter", "x ", "tweet", "post", "engagement", "reply round"],
        "browser": ["browser", "headed", "headless", "chrome", "site", "login"],
        "code": ["code", "patch", "edit", "test", "implement", "fix"],
        "staging": ["staging", "staging", "sync", "clone"],
        "gateway": ["gateway", "provider", "model", "proxy", "grok", "codex", "ollama"],
        "remote": [
            "remote",
            "computer",
            "laptop",
            "desktop",
            "workstation",
            "node",
            "other machine",
        ],
    }
    for tag, needles in mapping.items():
        if any(n in text for n in needles):
            out.append(tag)
    return out
