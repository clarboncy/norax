"""RiskGate — tier, command-safety, and non-owner path authorization."""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

Tier = Literal["T0", "T1", "T2"]

DANGEROUS_PATTERNS: list[re.Pattern] = [
    re.compile(r"rm\s+-rf\s+/(?!home/|tmp/)"),
    re.compile(r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b", re.I),
    re.compile(r"\bFORMAT\s+[A-Z]:"),
    re.compile(r"\bmkfs\."),
    re.compile(r"\bdd\s+if=/dev/"),
    re.compile(r"\bchmod\s+777\s+/etc\b"),
    re.compile(r"(>|>>)\s*/boot/"),
    re.compile(r"(>|>>)\s*/etc/(?!tmp)"),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:"),
    re.compile(r"curl[^|]+\|\s*(sh|bash)"),
]

_SHELL_TOOLS = {"exec", "shell", "remote_exec", "sandbox_exec"}
_LOCAL_PATH_FIELDS: dict[str, tuple[str, ...]] = {
    "read": ("path",),
    "list_dir": ("path",),
    "repo_explore": ("root",),
    "deep_research": ("out_dir",),
    "write": ("path",),
    "write_chunk": ("path",),
    "edit": ("path",),
    "append_memory": ("path",),
    "message_send": ("files",),
}
_SENSITIVE_PATH_PARTS = frozenset({".git", ".gnupg", ".ssh", "secrets"})
_SENSITIVE_FILENAMES = frozenset(
    {
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "known_hosts",
    }
)
_SENSITIVE_SUFFIXES = (".key", ".p12", ".pfx", ".pem")
_COMMAND_PREFIX = (
    r"(?:(?:sudo|command)(?:\s+-\S+)*\s+(?:--\s+)?|"
    r"env(?:\s+\w+=\S+)*\s+)*"
)
_COMMAND_BOUNDARY = r"(?:^|(?:&&|\|\||[;|\n])\s*)"
_DESTRUCTIVE_COMMAND_PATTERNS: tuple[re.Pattern, ...] = (
    # Block recursive rm regardless of target spelling. Shell state such as
    # `cd /mnt/x` makes a relative target just as destructive as an absolute
    # one, and cannot be safely validated with args["path"].
    re.compile(
        _COMMAND_BOUNDARY
        + _COMMAND_PREFIX
        + r"(?:/[\w./-]+/)?rm\b"
        + r"(?=[^;&|\n]*(?:--recursive\b|-[A-Za-z]*r[A-Za-z]*\b))"
        + r"[^;&|\n]*",
    ),
    # `find -delete` and `find -exec rm` bypass an rm-only matcher.
    re.compile(
        _COMMAND_BOUNDARY
        + _COMMAND_PREFIX
        + r"(?:/[\w./-]+/)?find\b"
        + r"(?=[^;&|\n]*(?:-delete\b|-(?:exec|execdir)\b[^;&|\n]*(?:/[\w./-]+/)?rm\b))"
        + r"[^;&|\n]*",
    ),
    re.compile(r"\bxargs\b[^;&\n]*(?:/[\w./-]+/)?rm\b[^;&\n]*"),
    re.compile(
        _COMMAND_BOUNDARY
        + _COMMAND_PREFIX
        + r"git\s+clean\b[^;&|\n]*(?:-[A-Za-z]*[xfd][A-Za-z]*|--force)\b[^;&|\n]*"
    ),
    re.compile(r"\brsync\b[^;&\n]*\s--delete(?:-before|-during|-delay|-after|-excluded)?\b"),
    re.compile(_COMMAND_BOUNDARY + _COMMAND_PREFIX + r"(?:shred|wipefs)\b[^;&|\n]*"),
    # Lifecycle operations must occur in command position.  Merely searching
    # source or documentation for the word "reboot" is not dangerous.
    re.compile(
        _COMMAND_BOUNDARY
        + _COMMAND_PREFIX
        + r"(?:/[^\s;&|]+/)?(?:shutdown|reboot|poweroff|halt)\b[^;&|\n]*"
    ),
    re.compile(
        _COMMAND_BOUNDARY
        + _COMMAND_PREFIX
        + r"(?:/[^\s;&|]+/)?(?:systemctl|loginctl)\b"
        + r"[^;&|\n]*\b(?:reboot|poweroff|halt)\b[^;&|\n]*"
    ),
)

TOOL_RISK: dict[str, Tier] = {
    "read": "T0",
    "list_dir": "T0",
    "web_fetch": "T0",
    "web_search": "T0",
    "deep_research": "T1",
    "search_memory": "T0",
    "memory_search": "T0",
    "status": "T0",
    "repo_explore": "T0",
    "write": "T1",
    "write_chunk": "T1",
    "edit": "T1",
    "append_memory": "T1",
    "message_send": "T1",
    "schedule_reminder": "T1",
    "exec": "T2",
    "shell": "T2",
    "delete": "T2",
    "gateway_config_patch": "T2",
    "subagent_spawn": "T2",
    "browser": "T2",
    "sandbox_exec": "T2",
}

TIER_ALLOW: dict[str, set[str]] = {
    "guest": {"T0"},
    "user": {"T0", "T1"},
    "admin": {"T0", "T1"},
    "owner": {"T0", "T1", "T2"},
}


@dataclass
class RiskDecision:
    allowed: bool
    tier: Tier
    reason: str
    dangerous_hit: str | None = None


def dangerous_command_hit(command: str) -> str | None:
    """Return the destructive shell fragment, including compound-command bypasses."""
    text = str(command or "")
    for pattern in _DESTRUCTIVE_COMMAND_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0).strip()[:240]
    return None


def _is_inert_literal_inspection(command: str) -> bool:
    """Recognize quoted danger examples passed to simple display/search tools.

    Risk patterns still apply whenever shell control operators or command
    substitutions are present. This narrow exception lets the agent grep or
    print dangerous syntax during audits without treating source text as an
    operation.
    """
    text = str(command or "")
    if "$(" in text or "`" in text or "<(" in text or ">(" in text:
        return False
    try:
        lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return False
    if not tokens or any(token and set(token) <= set(";&|<>") for token in tokens):
        return False
    executable = Path(tokens[0]).name.lower()
    if executable == "command" and len(tokens) > 1:
        executable = Path(tokens[1]).name.lower()
    return executable in {"echo", "grep", "printf", "rg"}


@lru_cache(maxsize=16)
def _non_owner_roots(workspace: str, extras: str, cwd: str) -> tuple[Path, ...]:
    values = [workspace or cwd]
    values.extend(value for value in extras.split(os.pathsep) if value.strip())
    roots: list[Path] = []
    for value in values:
        try:
            root = Path(value).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def _configured_non_owner_roots() -> tuple[Path, ...]:
    return _non_owner_roots(
        os.environ.get("NORAX_WORKSPACE", ""),
        os.environ.get("NORAX_NON_OWNER_ALLOWED_ROOTS", ""),
        str(Path.cwd()),
    )


def _is_sensitive_non_owner_path(path: Path) -> bool:
    if os.environ.get("NORAX_NON_OWNER_ALLOW_SENSITIVE_PATHS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if parts & _SENSITIVE_PATH_PARTS:
        return True
    if name == ".env" or (
        name.startswith(".env.") and not name.endswith((".example", ".sample", ".template"))
    ):
        return True
    return name in _SENSITIVE_FILENAMES or name.endswith(_SENSITIVE_SUFFIXES)


def _local_paths(tool: str, args: dict) -> list[str]:
    values: list[str] = []
    for field in _LOCAL_PATH_FIELDS.get(tool, ()):
        value = args.get(field)
        if value is None and (tool, field) in {("list_dir", "path"), ("repo_explore", "root")}:
            value = "."
        if isinstance(value, str) and value.strip():
            values.append(value)
        elif field == "files" and isinstance(value, (list, tuple)):
            values.extend(item for item in value if isinstance(item, str) and item.strip())
    return values


def _non_owner_path_denial(tool: str, args: dict) -> str | None:
    roots = _configured_non_owner_roots()
    if not roots:
        return "no non-owner filesystem root is configured"
    for raw_path in _local_paths(tool, args):
        try:
            resolved = Path(raw_path).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            return f"invalid local path: {raw_path!r}"
        if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
            return f"path outside non-owner allowed roots: {raw_path!r}"
        if _is_sensitive_non_owner_path(resolved):
            return f"sensitive path requires owner tier: {raw_path!r}"
    return None


def classify(tool: str, args: dict) -> Tier:
    return TOOL_RISK.get(tool, "T2")


def check(*, tool: str, args: dict, sender_tier: str) -> RiskDecision:
    tier = classify(tool, args)
    if tool in _SHELL_TOOLS:
        command = args.get("command") or args.get("cmd") or ""
        hit = dangerous_command_hit(str(command))
        if hit:
            return RiskDecision(
                allowed=False,
                tier=tier,
                reason=f"dangerous pattern: {hit!r}",
                dangerous_hit=hit,
            )

    # These expressions describe executable command syntax. Applying them to
    # ordinary file contents, messages, reminder text, or MCP JSON makes a
    # content censor out of a command safety gate and blocks legitimate work.
    if tool in _SHELL_TOOLS:
        text = str(args.get("command") or args.get("cmd") or "")
        if not _is_inert_literal_inspection(text):
            for p in DANGEROUS_PATTERNS:
                m = p.search(text)
                if m:
                    return RiskDecision(
                        allowed=False,
                        tier=tier,
                        reason=f"dangerous pattern: {m.group(0)!r}",
                        dangerous_hit=m.group(0),
                    )

    allow = TIER_ALLOW.get(sender_tier, {"T0"})
    if tier not in allow:
        return RiskDecision(
            allowed=False,
            tier=tier,
            reason=f"tier {sender_tier} cannot call {tier} tool {tool!r}",
        )
    if sender_tier != "owner":
        path_denial = _non_owner_path_denial(tool, args)
        if path_denial:
            return RiskDecision(allowed=False, tier=tier, reason=path_denial)
    return RiskDecision(allowed=True, tier=tier, reason="ok")
