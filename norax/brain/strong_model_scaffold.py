"""Runtime scaffolding that makes weaker models behave more like strong agents.

The key idea: keep high-signal task state outside the model and inject
small, structured guidance at the right places instead of relying on a
large model to infer workflow every turn.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Any

WRITE_TOOLS = {"write", "write_chunk", "edit"}
READ_TOOLS = {"read", "list_dir"}
DIRECT_READ_TOOLS = READ_TOOLS | {"remote_read", "remote_list"}
EVIDENCE_TOOLS = DIRECT_READ_TOOLS | {
    "remote_list_nodes",
    "web_fetch",
    "web_search",
    "search_memory",
    "memory_search",
    "status",
    "repo_explore",
}
SHELL_TOOLS = {"exec", "shell", "remote_exec", "sandbox_exec"}
FILE_MUTATION_TOOLS = WRITE_TOOLS | {"append_memory", "remote_write", "remote_edit"}
MUTATION_TOOLS = FILE_MUTATION_TOOLS | {
    "deep_research",
    "delete",
    "gateway_config_patch",
    "message_send",
    "schedule_reminder",
    "remote_enroll",
}
_COMPUTER_MUTATION_ACTIONS = frozenset(
    {"click", "doubleclick", "drag", "type_text", "key_press", "clipboard_set"}
)
_COMPUTER_OBSERVATION_ACTIONS = frozenset({"screenshot", "clipboard_get", "info"})
_BROWSER_MUTATION_ACTIONS = frozenset({"click", "type", "fill_form", "evaluate"})
_BROWSER_OBSERVATION_ACTIONS = frozenset({"extract", "screenshot", "tabs_list"})
_MCP_MUTATION_NAME = re.compile(
    r"(?:^|_)(?:add|apply|cancel|create|delete|deploy|disable|edit|enable|"
    r"install|move|patch|post|publish|remove|rename|restart|schedule|send|set|"
    r"start|stop|submit|update|upload|write)(?:_|$)",
    re.I,
)
_AUTHORITATIVE_RECEIPT_TOOLS = frozenset(
    {"gateway_config_patch", "message_send", "schedule_reminder", "remote_enroll"}
)
CODING_TERMS = (
    "implement",
    "fix",
    "bug",
    "patch",
    "code",
    "test",
    "repo",
    "file",
    "edit",
    "write",
    "refactor",
    "pipeline",
    "hook",
    "runtime",
    "verify",
)
UNCERTAINTY_TERMS = (
    "not sure",
    "unclear",
    "ambiguous",
    "maybe",
    "unknown",
    "can't tell",
    "cannot tell",
    "contradict",
)

_BLOCKING_TOOL_ERRORS = frozenset(
    {
        "permission_denied",
        "risk_blocked",
        "tool_not_allowed",
        "tool_not_found",
        "unauthorized",
        "authentication_failed",
    }
)

_MUTATING_COMMAND_PATTERNS = (
    re.compile(
        r"(?:^|(?:&&|\|\||[;|\n])\s*)"
        r"(?:(?:sudo|command)(?:\s+-\S+)*\s+|env(?:\s+\w+=\S+)*\s+)*"
        r"(?:/[\w./-]+/)?"
        r"(?:rm|mv|cp|install|mkdir|rmdir|touch|truncate|chmod|chown|chgrp|ln|tee)\b",
        re.I,
    ),
    re.compile(
        r"(?:^|(?:&&|\|\||[;|\n])\s*)"
        r"(?:(?:sudo|command)(?:\s+-\S+)*\s+)?"
        r"(?:sed\s+(?:-[A-Za-z]*i[A-Za-z]*|--in-place)\b|"
        r"perl\s+-[A-Za-z]*i[A-Za-z]*\b|patch\b)",
        re.I,
    ),
    re.compile(
        r"(?:^|(?:&&|\|\||[;|\n])\s*)"
        r"(?:(?:sudo|command)(?:\s+-\S+)*\s+)?"
        r"(?:apt(?:-get)?|dnf|yum|pacman|zypper|apk|brew|pipx?|uv)\s+"
        r"(?:install|uninstall|remove|upgrade|add|sync)\b",
        re.I,
    ),
    re.compile(
        r"(?:^|(?:&&|\|\||[;|\n])\s*)"
        r"(?:(?:sudo|command)(?:\s+-\S+)*\s+)?"
        r"(?:npm|pnpm|yarn|bun|cargo|go)\s+"
        r"(?:add|install|remove|uninstall|update|upgrade|publish)\b",
        re.I,
    ),
    re.compile(
        r"(?:^|(?:&&|\|\||[;|\n])\s*)"
        r"(?:(?:sudo|command)(?:\s+-\S+)*\s+)?"
        r"systemctl\s+(?:start|stop|restart|reload|daemon-reload|enable|disable|mask|unmask)\b",
        re.I,
    ),
    re.compile(
        r"(?:^|(?:&&|\|\||[;|\n])\s*)"
        r"(?:docker|podman)(?:\s+compose)?\s+"
        r"(?:build|create|kill|pause|pull|push|restart|rm|run|start|stop|update|"
        r"unpause|up|down)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:docker|podman)\s+exec\b[^\n;]*\b"
        r"(?:rm|mv|cp|install|mkdir|rmdir|touch|truncate|chmod|chown|chgrp|ln|tee|"
        r"sed|patch|kill|shutdown|reboot)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:git\s+(?:add|apply|branch|checkout|cherry-pick|clean|commit|config|"
        r"merge|mv|pull|push|rebase|remote|reset|restore|revert|rm|stash|switch|tag)|kill)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:kubectl\s+(?:apply|annotate|autoscale|cordon|create|delete|drain|edit|"
        r"label|patch|replace|scale|set|taint)|"
        r"kubectl\s+rollout\s+(?:pause|restart|resume|undo)|"
        r"terraform\s+(?:apply|destroy|import)|"
        r"service\s+\S+\s+(?:start|stop|restart|reload))\b",
        re.I,
    ),
    re.compile(r"\bcurl\b[^\n]*(?:--request|-X)\s*(?:POST|PUT|PATCH|DELETE)\b", re.I),
    re.compile(r"\bcurl\b[^\n]*(?:--data(?:-[a-z]+)?|-d)\s", re.I),
    re.compile(r"\bfind\b[^;&\n]*(?:-delete\b|-(?:exec|execdir)\b[^;&\n]*\brm\b)", re.I),
    re.compile(r"\bxargs\b[^;&\n]*\brm\b|\brsync\b[^;&\n]*\s--delete", re.I),
    re.compile(
        r"\b(?:python(?:3)?|pypy(?:3)?)\b[^\n;]*(?:"
        r"\.write_(?:text|bytes)\s*\(|\.unlink\s*\(|\.rename\s*\(|"
        r"\.replace\s*\(|\.mkdir\s*\(|\.touch\s*\(|"
        r"\bopen\s*\([^)]*,\s*['\"][^'\"]*[wax+][^'\"]*['\"]|"
        r"\bos\.(?:remove|unlink|rename|replace|mkdir|makedirs|chmod|chown)\s*\(|"
        r"\bshutil\.(?:copy|copy2|copytree|move|rmtree)\s*\()",
        re.I,
    ),
    re.compile(
        r"\bnode\b[^\n;]*(?:fs\.)?(?:writeFile|appendFile|rm|unlink|rename|mkdir|"
        r"copyFile)Sync?\s*\(",
        re.I,
    ),
    re.compile(r"\bcurl\b[^\n;]*(?:--output|-o)\s+(?!/dev/null\b)\S+", re.I),
    re.compile(r"\b(?:wget|unzip\b[^\n;]*\s-d\s+|tar\b[^\n;]*-[^\s]*x)", re.I),
    re.compile(
        r"(?:^|[^<])(?:>>?|[0-9]+>>?)\s*"
        r"(?!(?:&[0-9-]+\b|/dev/(?:null|stdout|stderr)\b))\S+"
    ),
)

_VERIFY_COMMAND_PATTERNS = (
    re.compile(
        r"(?:^|(?:&&|\|\||[;|\n])\s*)"
        r"(?:(?:sudo|command)(?:\s+-\S+)*\s+)?"
        r"(?:ls|stat|test|cat|head|tail|wc|file|du|df|ps|pgrep|grep|rg|find|"
        r"diff|cmp|readlink|realpath|mountpoint|sha(?:1|256|512)sum|md5sum)\b",
        re.I,
    ),
    re.compile(r"\b(?:pytest|unittest|ruff|mypy|pyright|ffprobe|ffmpeg)\b", re.I),
    re.compile(r"\bsystemctl\s+(?:status|is-active|is-enabled|show|list-units)\b", re.I),
    re.compile(r"\b(?:docker|podman)(?:\s+compose)?\s+(?:ps|inspect|logs|images|version)\b", re.I),
    re.compile(
        r"\b(?:docker|podman)\s+exec\b[^\n;]*"
        r"(?:\bmc-monitor\s+status-bedrock\b|\b(?:cat|ls|stat|test|grep|rg|ps)\b)",
        re.I,
    ),
    re.compile(r"\bgit\s+(?:diff|status|show|log)\b", re.I),
    re.compile(r"\bcurl\b", re.I),
)
_VALIDATION_COMMAND_PATTERNS = (
    re.compile(r"\b(?:pytest|unittest|tox|nox|ruff|mypy|pyright)\b", re.I),
    re.compile(r"\b(?:npm|pnpm|yarn)\s+(?:run\s+)?(?:test|check|lint)\b", re.I),
    re.compile(r"\b(?:cargo\s+(?:test|check|clippy)|go\s+test|gradle\w*\s+\S*test)\b", re.I),
    re.compile(r"\b(?:make\s+(?:test|check|lint)|tsc\b[^\n]*--noEmit)\b", re.I),
    re.compile(r"\bpython(?:3)?\s+-m\s+(?:compileall|pytest|unittest)\b", re.I),
)
_OPERATIONAL_VERIFY_PATTERNS = (
    re.compile(r"\bsystemctl\s+(?:status|is-active|is-enabled|show)\b", re.I),
    re.compile(r"\b(?:docker|podman)(?:\s+compose)?\s+(?:ps|inspect|logs)\b", re.I),
    re.compile(r"\b(?:mc-monitor\s+status|curl\b)", re.I),
)
_SEARCH_NO_MATCH_COMMAND = re.compile(
    r"^\s*(?:(?:command|env)\s+)?(?:grep|rg)\b[^;&|\n]*$",
    re.I,
)


def shell_command_is_mutating(command: str) -> bool:
    """Return whether a shell command has an observable state-changing action.

    This is deliberately conservative.  It is used for outcome accounting,
    not as the security boundary (the dispatch risk gate remains authoritative).
    """
    text = str(command or "").strip()
    return bool(text and any(pattern.search(text) for pattern in _MUTATING_COMMAND_PATTERNS))


def tool_call_is_mutating(name: str, args: dict[str, Any] | None = None) -> bool:
    """Classify direct writer tools and state-changing shell commands."""
    normalized = str(name or "").strip().lower()
    data = args or {}
    if normalized in MUTATION_TOOLS:
        return True
    if normalized == "computer_use":
        return str(data.get("action") or "").lower() in _COMPUTER_MUTATION_ACTIONS
    if normalized == "browser":
        return str(data.get("action") or "").lower() in _BROWSER_MUTATION_ACTIONS
    if normalized.startswith("mcp_"):
        return bool(_MCP_MUTATION_NAME.search(normalized[4:]))
    if normalized in SHELL_TOOLS:
        return shell_command_is_mutating(str(data.get("command") or ""))
    return False


def tool_call_has_authoritative_receipt(
    name: str,
    args: dict[str, Any] | None,
    result: dict[str, Any] | None,
) -> bool:
    """Return whether a successful mutation response is durable confirmation.

    File and shell writes still require an independent read/check. API-style
    operations that return a server or scheduler receipt should not be repeated
    merely to manufacture a second tool call.
    """
    normalized = str(name or "").strip().lower()
    if not tool_call_is_mutating(normalized, args) or not tool_result_succeeded(
        normalized, result, args
    ):
        return False
    data = result or {}
    if normalized.startswith("mcp_"):
        return data.get("is_error") is not True
    if normalized == "remote_enroll":
        return bool(data.get("node_id") and data.get("token"))
    if normalized == "gateway_config_patch":
        return bool(data.get("config_path") or data.get("patched_keys"))
    if normalized == "deep_research":
        return bool(data.get("state_path") and data.get("rounds_this_call"))
    return normalized in _AUTHORITATIVE_RECEIPT_TOOLS


def tool_result_succeeded(
    name: str,
    result: dict[str, Any] | None,
    args: dict[str, Any] | None = None,
) -> bool:
    """Return success using both the tool status and shell exit status.

    Some providers have emitted ``ok=true`` for a shell result whose command
    exited non-zero. Such a result must never support a completion claim.
    Read-only grep/rg commands retain their established exit-1/no-match
    semantics when they produced no output or error.
    """
    if not isinstance(result, dict) or result.get("ok") is not True or result.get("error"):
        return False
    normalized = str(name or "").strip().lower()
    if normalized in SHELL_TOOLS:
        code = result.get("exit_code")
        try:
            exit_code = int(code) if code is not None else -1
        except (TypeError, ValueError):
            return False
        if exit_code == 0:
            return True
        command = str((args or {}).get("command") or "")
        return bool(
            exit_code == 1
            and _SEARCH_NO_MATCH_COMMAND.search(command)
            and not str(result.get("stdout") or "").strip()
            and not str(result.get("stderr") or "").strip()
        )
    return True


def tool_call_is_successful_verification(
    name: str,
    args: dict[str, Any] | None,
    result: dict[str, Any] | None,
) -> bool:
    """Return whether a successful, non-mutating call provides check evidence."""
    normalized = str(name or "").strip().lower()
    if tool_call_is_mutating(normalized, args) or not tool_result_succeeded(
        normalized, result, args
    ):
        return False
    if normalized in EVIDENCE_TOOLS or normalized.startswith("mcp_"):
        return True
    if normalized == "computer_use":
        action = str((args or {}).get("action") or "").lower()
        return action in _COMPUTER_OBSERVATION_ACTIONS
    if normalized == "browser":
        action = str((args or {}).get("action") or "").lower()
        return action in _BROWSER_OBSERVATION_ACTIONS
    if normalized not in SHELL_TOOLS:
        return False
    command = str((args or {}).get("command") or "")
    return any(pattern.search(command) for pattern in _VERIFY_COMMAND_PATTERNS)


def _target_path(args: dict[str, Any] | None) -> str:
    data = args or {}
    return str(
        data.get("path")
        or data.get("file")
        or data.get("target")
        or data.get("destination")
        or data.get("dest")
        or ""
    ).strip()


def _paths_equivalent(left: str, right: str) -> bool:
    """Compare absolute/relative spellings without touching the filesystem."""
    a = re.sub(r"/+", "/", left.replace("\\", "/")).rstrip("/")
    b = re.sub(r"/+", "/", right.replace("\\", "/")).rstrip("/")
    if not a or not b:
        return False
    if a == b:
        return True
    # A tool may return an absolute path for a relative mutation argument.
    return (not a.startswith("/") and b.endswith(f"/{a}")) or (
        not b.startswith("/") and a.endswith(f"/{b}")
    )


def _shell_command_targets_path(command: str, target_path: str) -> bool:
    """Return whether shell output syntax explicitly targets a read path.

    Shell mutations do not carry a structured ``path`` argument. Tokenize the
    command and recognize redirection/output operands while ignoring ordinary
    content words. This prevents a command that merely prints ``state.txt``
    from making a later read of that file look like relevant verification.
    """
    if not command or not target_path:
        return False
    try:
        words = shlex.split(command, posix=True)
    except ValueError:
        # A malformed command should already have failed execution. Retain a
        # conservative lexical fallback for trace data from external tools.
        words = command.split()

    expect_output_target = False
    for word in words:
        value = word.strip("'\"(),;|")
        if expect_output_target:
            expect_output_target = False
            if not value.startswith("&") and _paths_equivalent(value, target_path):
                return True
            continue
        if re.fullmatch(r"(?:\d*>>?|&>)", value):
            expect_output_target = True
            continue
        attached = re.fullmatch(r"(?:\d*>>?|&>)(?!&)(.+)", value)
        if attached and _paths_equivalent(attached.group(1), target_path):
            return True
        if value in {"-o", "-O", "--output"}:
            expect_output_target = True
            continue
        if value.startswith("--output=") and _paths_equivalent(value.split("=", 1)[1], target_path):
            return True
    return False


_RESOURCE_STOPWORDS = frozenset(
    {
        "bash",
        "sh",
        "sudo",
        "command",
        "env",
        "exec",
        "shell",
        "read",
        "write",
        "edit",
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "nl",
        "grep",
        "rg",
        "find",
        "ls",
        "stat",
        "test",
        "mkdir",
        "rmdir",
        "touch",
        "rm",
        "mv",
        "cp",
        "install",
        "chmod",
        "chown",
        "ln",
        "tee",
        "sed",
        "patch",
        "docker",
        "podman",
        "compose",
        "systemctl",
        "status",
        "restart",
        "start",
        "stop",
        "inspect",
        "logs",
        "true",
        "false",
    }
)


def _resource_tokens(text: str) -> set[str]:
    """Extract stable resource identifiers from a path or shell command."""
    tokens: set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9_./:@+-]+", str(text or "").lower()):
        value = raw.strip("'\".,:;()[]{}")
        if not value or value.startswith("-") or value in _RESOURCE_STOPWORDS:
            continue
        if value.isdigit() or len(value) < 2:
            continue
        tokens.add(value)
        basename = value.rstrip("/").rsplit("/", 1)[-1]
        if basename and basename not in _RESOURCE_STOPWORDS and len(basename) >= 2:
            tokens.add(basename)
            if "." in basename:
                stem = basename.rsplit(".", 1)[0]
                if len(stem) >= 2 and stem not in _RESOURCE_STOPWORDS:
                    tokens.add(stem)
    return tokens


def tool_call_verifies_mutation(
    mutation_name: str,
    mutation_args: dict[str, Any] | None,
    verifier_name: str,
    verifier_args: dict[str, Any] | None,
    verifier_result: dict[str, Any] | None,
) -> bool:
    """Return whether a check is successful *and relevant* to a mutation.

    Verification used to accept any later read, including reading an unrelated
    file.  This ties direct readback to the same target, shell checks to a
    shared resource, and accepts recognized test/static-analysis commands as
    broad validation evidence.
    """
    if not tool_call_is_successful_verification(verifier_name, verifier_args, verifier_result):
        return False

    m_name = str(mutation_name or "").strip().lower()
    v_name = str(verifier_name or "").strip().lower()
    m_args = mutation_args or {}
    v_args = verifier_args or {}

    if m_name == "computer_use" and v_name == "computer_use":
        mutation_action = str(m_args.get("action") or "").lower()
        verifier_action = str(v_args.get("action") or "").lower()
        if mutation_action == "clipboard_set":
            return verifier_action == "clipboard_get"
        return verifier_action == "screenshot"

    if m_name == "browser" and v_name == "browser":
        mutation_session = str(m_args.get("session_id") or "default")
        verifier_session = str(v_args.get("session_id") or "default")
        return mutation_session == verifier_session

    if v_name in DIRECT_READ_TOOLS:
        mutation_path = _target_path(m_args)
        verifier_path = _target_path(v_args)
        same_target = _paths_equivalent(mutation_path, verifier_path)
        if not same_target and m_name in SHELL_TOOLS:
            same_target = _shell_command_targets_path(
                str(m_args.get("command") or ""), verifier_path
            )
        if not same_target:
            return False
        mutation_node = str(m_args.get("node") or m_args.get("node_id") or "")
        verifier_node = str(v_args.get("node") or v_args.get("node_id") or "")
        return not (mutation_node and verifier_node and mutation_node != verifier_node)

    if v_name not in SHELL_TOOLS:
        return False
    verifier_command = str(v_args.get("command") or "")
    mutation_text = (
        str(m_args.get("command") or "") if m_name in SHELL_TOOLS else _target_path(m_args)
    )
    if any(pattern.search(verifier_command) for pattern in _VALIDATION_COMMAND_PATTERNS):
        # A project test/static-analysis run is broad evidence only when the
        # mutation identifies a source or project-configuration artifact.
        # Otherwise `pytest` after an unrelated mkdir/package/service change
        # would manufacture a verified outcome.
        validation_targets = _resource_tokens(mutation_text)
        valid_suffixes = (
            ".py",
            ".pyi",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".rs",
            ".go",
            ".java",
            ".kt",
            ".c",
            ".cc",
            ".cpp",
            ".h",
            ".hpp",
            ".cs",
            ".rb",
            ".php",
            ".swift",
            ".scala",
            ".sh",
            ".bash",
            ".toml",
            ".json",
            ".jsonc",
            ".yaml",
            ".yml",
            ".ini",
            ".cfg",
        )
        return any(token.endswith(valid_suffixes) for token in validation_targets)

    if _resource_tokens(mutation_text) & _resource_tokens(verifier_command):
        return True

    # A service/container health check is valid evidence after changing its
    # conventional configuration file even when the service name is not
    # encoded in a generic path such as ``compose.yml``.
    mutation_path = _target_path(m_args).lower()
    is_compose_config = "compose" in mutation_path.rsplit("/", 1)[-1]
    return bool(
        is_compose_config
        and any(pattern.search(verifier_command) for pattern in _OPERATIONAL_VERIFY_PATTERNS)
        and re.search(r"\b(?:docker|podman)(?:\s+compose)?\b", verifier_command, re.I)
    )


@dataclass(frozen=True)
class MutationTraceOutcome:
    """Evidence summary for state-changing calls in one tool trace."""

    attempted: int
    succeeded: int
    failed: int
    last_mutation_index: int | None
    last_mutation_succeeded: bool | None
    verified_after_last_mutation: bool

    @property
    def supports_success_claim(self) -> bool:
        return (
            self.attempted > 0
            and self.last_mutation_succeeded is True
            and self.verified_after_last_mutation
        )


def summarize_mutation_trace(tool_trace: list[dict[str, Any]] | None) -> MutationTraceOutcome:
    """Summarize action success and independent post-action verification."""
    attempted = succeeded = failed = 0
    last_index: int | None = None
    last_succeeded: bool | None = None
    trace = tool_trace or []
    for index, entry in enumerate(trace):
        name = str(entry.get("name") or "")
        raw_args = entry.get("args")
        raw_result = entry.get("result")
        args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
        result: dict[str, Any] = raw_result if isinstance(raw_result, dict) else {}
        if result.get("_not_executed") is True:
            continue
        if not tool_call_is_mutating(name, args):
            continue
        attempted += 1
        last_index = index
        last_succeeded = tool_result_succeeded(name, result, args)
        if last_succeeded:
            succeeded += 1
        else:
            failed += 1

    verified = bool(
        last_index is not None
        and last_succeeded is True
        and (
            tool_call_has_authoritative_receipt(
                str(trace[last_index].get("name") or ""),
                trace[last_index].get("args")
                if isinstance(trace[last_index].get("args"), dict)
                else {},
                trace[last_index].get("result")
                if isinstance(trace[last_index].get("result"), dict)
                else {},
            )
            or any(
                tool_call_verifies_mutation(
                    str(trace[last_index].get("name") or ""),
                    trace[last_index].get("args")
                    if isinstance(trace[last_index].get("args"), dict)
                    else {},
                    str(entry.get("name") or ""),
                    entry.get("args") if isinstance(entry.get("args"), dict) else {},
                    entry.get("result") if isinstance(entry.get("result"), dict) else {},
                )
                for entry in trace[last_index + 1 :]
            )
        )
    )
    return MutationTraceOutcome(
        attempted=attempted,
        succeeded=succeeded,
        failed=failed,
        last_mutation_index=last_index,
        last_mutation_succeeded=last_succeeded,
        verified_after_last_mutation=verified,
    )


def tool_failure_is_blocking(name: str, result: dict[str, Any]) -> bool:
    """True only for failures that cannot normally be recovered in-turn."""
    error = str(result.get("error") or result.get("detail") or "").strip().lower()
    if error in _BLOCKING_TOOL_ERRORS:
        return True
    if any(token in error for token in ("permission", "unauthorized", "not_allowed", "risk_")):
        return True
    # A shell exit status, missing path, no match, timeout, or stale edit is an
    # observation to replan around—not a permanent task blocker.
    if name in {"exec", "shell", "remote_exec"} and result.get("exit_code") is not None:
        return False
    return error not in {
        "file_not_found",
        "path_not_found",
        "is_a_directory",
        "old_text_not_found",
        "duplicate_call_blocked",
        "duplicate_mutation_blocked",
        "duplicate_read_blocked",
        "loop_detected",
        "tool_timeout",
        "timeout",
        "bad_arguments",
    }


# Per-tool key-extraction: which fields carry the highest signal?
_HIGH_SIGNAL_KEYS: dict[str, tuple[str, ...]] = {
    "read": ("ok", "path", "total_lines", "content"),
    "list_dir": ("ok", "path", "entries"),
    "exec": ("ok", "exit_code", "stdout", "stderr"),
    "shell": ("ok", "exit_code", "stdout", "stderr", "cwd"),
    "edit": ("ok", "path"),
    "write": ("ok", "path"),
    "write_chunk": ("ok", "path"),
    "search_memory": ("ok", "results"),
    "status": ("ok", "state"),
}

# Next-step suggestions keyed to task state
_NEXT_STEP_ADVICE: dict[str, str] = {
    "verify_write": "Read back the file you just wrote/edited, or run a targeted test/command to confirm it works.",
    "recover_from_errors": "Re-read the file or command output that failed. Identify the specific error. Try a different approach. Do NOT repeat the same failing call.",
    "inspect_repo": "Before editing, read the relevant files and understand the current code. Use list_dir and read first. Check file_summaries in TASK_STATE — if you already read a file this turn, its summary may have what you need without re-reading.",
    "escalate": "You have hit multiple failures or uncertainty. Re-read current state carefully, try a fundamentally different approach, or clearly report the blocker.",
    "continue_or_finalize": "If you have enough evidence to answer, do so now. If not, take one more targeted tool action. Check file_summaries in TASK_STATE before re-reading a file you already read — if the summary has what you need, don't re-read.",
}


@dataclass
class TaskState:
    """Compact per-turn state tracked by runtime, not by model prose."""

    goal: str
    task_type: str = "general"
    constraints: list[str] = field(default_factory=list)
    done_criteria: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    # Pending steps the model still needs to do. Updated as work progresses.
    # This gives the model a clear direction so it doesn't revert to old work.
    pending_steps: list[str] = field(default_factory=list)
    next_step: str = "act"
    tool_rounds: int = 0
    tool_calls: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    writes: int = 0
    mutation_attempts: int = 0
    mutations_succeeded: int = 0
    mutations_failed: int = 0
    last_mutation_succeeded: bool | None = None
    verified_after_write: bool = False
    read_before_write: bool = False
    should_escalate: bool = False
    escalation_reasons: list[str] = field(default_factory=list)
    # Compact per-file read summaries so the model doesn't re-read files it already saw.
    # Maps path → "lines=X; head=<first 3 notable lines>; key=<def/class/symbol names>"
    file_summaries: dict[str, str] = field(default_factory=dict)
    # Completed sub-task tracking — prevents the model from jumping back to
    # already-solved work. Each entry is a compact description of what was done.
    completed_actions: list[str] = field(default_factory=list)
    # Round at which each completed action was recorded, for age-based pruning.
    _completed_action_rounds: list[int] = field(default_factory=list)
    _last_mutation_name: str = ""
    _last_mutation_args: dict[str, Any] = field(default_factory=dict)

    def note_tool(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        self.tool_calls += 1
        is_mutation = tool_call_is_mutating(name, args) and result.get("_not_executed") is not True
        ok = tool_result_succeeded(name, result, args)
        if not ok:
            self.failures += 1
            self.consecutive_failures += 1
            err = str(result.get("error") or result.get("detail") or "tool_failed")[:160]
            if tool_failure_is_blocking(name, result):
                self._add_unique(self.blockers, f"{name}: {err}", limit=6)
        else:
            self.consecutive_failures = 0
            self.blockers = [item for item in self.blockers if not item.startswith(f"{name}:")]
            self.escalation_reasons = [
                reason
                for reason in self.escalation_reasons
                if reason != "three or more tool failures"
            ]
            self.should_escalate = bool(self.escalation_reasons)
        if name in DIRECT_READ_TOOLS and ok:
            self.read_before_write = self.writes == 0 or self.read_before_write
        if is_mutation:
            self.mutation_attempts += 1
            self.last_mutation_succeeded = ok
            self.verified_after_write = False
            self._last_mutation_name = str(name or "")
            self._last_mutation_args = dict(args or {})
            if ok:
                self.mutations_succeeded += 1
                self.writes += 1
            else:
                self.mutations_failed += 1
            path = args.get("path") or args.get("file") or args.get("target")
            if ok and path:
                self._add_unique(self.files_touched, str(path), limit=12)
            if ok and name in FILE_MUTATION_TOOLS and not self.read_before_write:
                self._escalate("write attempted before read/list verification")
            if ok and tool_call_has_authoritative_receipt(name, args, result):
                self.verified_after_write = True
                self._record_completed(name, args, self.tool_rounds)
        if (
            self.writes
            and self.last_mutation_succeeded is True
            and not is_mutation
            and tool_call_verifies_mutation(
                self._last_mutation_name,
                self._last_mutation_args,
                name,
                args,
                result,
            )
        ):
            self.verified_after_write = True
            self._record_completed(
                self._last_mutation_name, self._last_mutation_args, self.tool_rounds
            )
        self._extract_facts(name, result, args)
        # Prune old facts to prevent context bloat — keep only the most recent.
        # Old facts from solved sub-tasks cause the model to re-engage with
        # completed work. Keep last 6 facts only.
        if len(self.facts) > 6:
            self.facts = self.facts[-6:]
        self._update_next_step()

    def _record_completed(self, name: str, args: dict[str, Any], round_num: int) -> None:
        """Record a completed+verified action to prevent revisiting."""
        path = args.get("path") or args.get("file") or args.get("target") or ""
        cmd = args.get("command") or ""
        if name in FILE_MUTATION_TOOLS and path:
            desc = f"{name} {path} (verified round {round_num})"
        elif name in SHELL_TOOLS and cmd:
            # Compact command description — first 80 chars
            desc = f"exec: {str(cmd)[:80]} (verified round {round_num})"
        elif name in READ_TOOLS and path:
            desc = f"verified {name} {path} (round {round_num})"
        else:
            desc = f"{name} verified (round {round_num})"
        # Avoid exact duplicates
        if desc not in self.completed_actions:
            self.completed_actions.append(desc)
            self._completed_action_rounds.append(round_num)
        # Prune: keep only last 8 completed actions
        if len(self.completed_actions) > 8:
            self.completed_actions = self.completed_actions[-8:]
            self._completed_action_rounds = self._completed_action_rounds[-8:]
        # Remove this action from pending_steps if it matches
        self._complete_pending_step(desc)

    def add_pending_step(self, step: str) -> None:
        """Add a pending step the model still needs to do."""
        step = str(step or "").strip()
        if step and step not in self.pending_steps:
            self.pending_steps.append(step)
            # Keep only last 8
            if len(self.pending_steps) > 8:
                self.pending_steps = self.pending_steps[-8:]

    def _complete_pending_step(self, completed_desc: str) -> None:
        """Remove a pending step that matches a completed action."""
        if not self.pending_steps:
            return
        # Match by path/file keyword in the completed description
        path = ""
        for keyword in ("write ", "edit ", "write_chunk ", "read ", "verified "):
            if keyword in completed_desc:
                parts = completed_desc.split(keyword, 1)
                if len(parts) > 1:
                    path = parts[1].split(" (")[0].strip()
                    break
        if path:
            self.pending_steps = [s for s in self.pending_steps if path not in s]

    def to_persistable_dict(self) -> dict[str, Any]:
        """Serialize to a dict for checkpoint persistence."""
        return {
            "goal": self.goal,
            "type": self.task_type,
            "constraints": self.constraints,
            "done_criteria": self.done_criteria,
            "facts": self.facts,
            "files_touched": self.files_touched,
            "file_summaries": dict(self.file_summaries),
            "completed_actions": self.completed_actions,
            "pending_steps": self.pending_steps,
            "blockers": self.blockers,
        }

    def final_check(self, content: str) -> None:
        text = (content or "").lower()
        if self.writes and not self.verified_after_write:
            self._escalate("writes occurred without post-write verification")
        if self.consecutive_failures >= 3:
            self._escalate("three or more tool failures")
        if any(term in text for term in UNCERTAINTY_TERMS):
            self._escalate("final answer contains uncertainty/failure language")

    def context_block(self) -> str:
        advice = _NEXT_STEP_ADVICE.get(self.next_step, "")
        payload = {
            "goal": self.goal[:240],
            "type": self.task_type,
            "constraints": self.constraints[:6],
            "done_criteria": self.done_criteria[:6],
            "facts": self.facts[-6:],
            "files_touched": self.files_touched[-10:],
            "file_summaries": dict(list(self.file_summaries.items())[-12:]),
            "completed_actions": self.completed_actions[-8:],
            "pending_steps": self.pending_steps[-8:],
            "blockers": self.blockers[-6:],
            "next_step": self.next_step,
            "next_step_advice": advice,
            "tool_calls": self.tool_calls,
            "tool_rounds": self.tool_rounds,
            "progress": {
                "tool_rounds": self.tool_rounds,
                "tool_calls": self.tool_calls,
                "failures": self.failures,
                "consecutive_failures": self.consecutive_failures,
                "writes": self.writes,
                "mutation_attempts": self.mutation_attempts,
                "mutations_succeeded": self.mutations_succeeded,
                "mutations_failed": self.mutations_failed,
                "last_mutation_succeeded": self.last_mutation_succeeded,
                "read_before_write": self.read_before_write,
                "verified_after_write": self.verified_after_write,
            },
            "escalate_if_needed": self.should_escalate,
            "escalation_reasons": self.escalation_reasons[-4:],
        }
        directive = (
            "\nCOMPLETED_ACTIONS: Treat completed_actions as evidence of prior work. "
            "Avoid exact redundant calls, but re-inspect with a narrower range or a "
            "different check when new evidence genuinely requires it. "
            "PENDING_STEPS: Work through pending_steps in order. "
            "If pending_steps is empty and you have more work, add steps before acting. "
            "Move forward to the NEXT unresolved step only. "
            "Use completed_actions to decide what evidence is already available."
        )
        return (
            "TASK_STATE\n"
            + json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            + directive
        )

    def _extract_facts(
        self, name: str, result: dict[str, Any], args: dict[str, Any] | None = None
    ) -> None:
        if result.get("ok") is not True:
            return
        if name == "list_dir":
            entries = result.get("entries") or []
            self._add_unique(
                self.facts, f"listed {result.get('path', '?')} ({len(entries)} entries)", limit=6
            )
        elif name == "read":
            path = result.get("path") or "file"
            total = result.get("total_lines")
            content = result.get("content") or ""
            # Build compact summary so model doesn't re-read
            summary_parts = []
            if total is not None:
                summary_parts.append(f"lines={total}")
            # Extract key symbols (class/def/async def/import from...)
            symbols = []
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith(
                    (
                        "class ",
                        "def ",
                        "async def ",
                        "import ",
                        "from ",
                        "export ",
                        "function ",
                        "const ",
                        "interface ",
                        "type ",
                    )
                ):
                    symbols.append(stripped[:80])
                if len(symbols) >= 12:
                    break
            if symbols:
                summary_parts.append("symbols=" + "|".join(symbols[:8]))
            # Store first 5 non-empty lines as head for quick context
            head_lines = [ln.strip()[:80] for ln in content.splitlines() if ln.strip()][:5]
            if head_lines:
                summary_parts.append("head=" + " › ".join(head_lines))
            if summary_parts:
                self.file_summaries[str(path)] = "; ".join(summary_parts)
            fact = f"read {path} ({total} lines)" if total is not None else f"read {path}"
            self._add_unique(self.facts, fact, limit=6)
        elif name in SHELL_TOOLS:
            code = result.get("exit_code")
            self._add_unique(self.facts, f"{name} exit_code={code}", limit=6)
        elif name in FILE_MUTATION_TOOLS:
            self._add_unique(self.facts, f"{name} succeeded", limit=6)
            # Invalidate cached summary for written files
            path = result.get("path") or result.get("file") or result.get("target") or ""
            if path and str(path) in self.file_summaries:
                del self.file_summaries[str(path)]

    def _update_next_step(self) -> None:
        if self.should_escalate:
            self.next_step = "escalate"
        elif self.last_mutation_succeeded is False:
            self.next_step = "recover_from_errors"
        elif self.writes and not self.verified_after_write:
            self.next_step = "verify_write"
        elif self.consecutive_failures >= 3:
            self.next_step = "recover_from_errors"
        elif self.task_type == "coding" and not self.read_before_write:
            self.next_step = "inspect_repo"
        else:
            self.next_step = "continue_or_finalize"

    def _escalate(self, reason: str) -> None:
        self.should_escalate = True
        self._add_unique(self.escalation_reasons, reason, limit=8)

    @staticmethod
    def _add_unique(items: list[str], item: str, *, limit: int) -> None:
        if item and item not in items:
            items.append(item)
        del items[:-limit]


def classify_task(user_prompt: str, allowed_tools: list[str] | None = None) -> str:
    text = (user_prompt or "").lower()
    tools = set(allowed_tools or [])
    if tools & (MUTATION_TOOLS | SHELL_TOOLS) and any(t in text for t in CODING_TERMS):
        return "coding"
    if any(w in text for w in ("search", "research", "compare", "best practices")):
        return "research"
    if any(w in text for w in ("why", "debug", "error", "failed", "traceback")):
        return "debugging"
    return "general"


def build_task_state(
    user_prompt: str,
    allowed_tools: list[str] | None = None,
    restore_from: dict[str, Any] | None = None,
    prior_status: str = "in_progress",
) -> TaskState:
    """Build a TaskState for a new turn.

    Saved operational state is restored only for a confident continuation of
    an unfinished task. Carrying file/action hints into a new request creates
    stale constraints and can make current inspection look redundant.
    """
    # Always build a fresh base state for the current prompt
    task_type = classify_task(user_prompt, allowed_tools)
    constraints = ["answer directly", "verify before claiming success"]
    done_criteria = ["answer directly addresses the user request"]
    if task_type in {"coding", "debugging"}:
        constraints.extend(
            [
                "inspect current files before editing",
                "make minimal patches",
                "run targeted verification after writes",
            ]
        )
        done_criteria.extend(
            [
                "relevant current files inspected before edits",
                "patch is minimal and localized",
                "post-write readback or targeted test/command succeeded",
            ]
        )
    elif task_type == "research":
        done_criteria.extend(
            [
                "claims grounded in checked sources or stated uncertainty",
                "important disagreements or gaps are reported",
            ]
        )
    else:
        done_criteria.append("blockers are reported instead of guessed around")

    ts = TaskState(
        goal=(user_prompt or "").strip(),
        task_type=task_type,
        constraints=constraints,
        done_criteria=done_criteria,
    )

    if restore_from:
        try:
            old_goal_original = str(restore_from.get("goal") or "").strip()
            old_goal = old_goal_original.lower()
            new_prompt = (user_prompt or "").strip().lower()
            _CONTINUE_WORDS = {
                "continue",
                "keep",
                "going",
                "working",
                "resume",
                "pick",
                "carry",
                "ahead",
                "finish",
                "yes",
                "ok",
                "yeah",
                "yep",
                "do",
                "it",
                "that",
                "this",
                "the",
                "now",
                "still",
                "same",
                "thing",
                "what",
                "were",
                "you",
                "doing",
                "on",
                "please",
                "bro",
                "bruh",
                "dude",
                "man",
            }
            new_words = set(re.findall(r"[a-z0-9_+-]+", new_prompt))
            old_words = set(re.findall(r"[a-z0-9_+-]+", old_goal))
            stopwords = _CONTINUE_WORDS | {
                "a",
                "an",
                "and",
                "for",
                "from",
                "in",
                "of",
                "to",
                "with",
            }
            new_significant = {w for w in new_words if len(w) > 2 and w not in stopwords}
            old_significant = {w for w in old_words if len(w) > 2 and w not in stopwords}
            overlap = new_significant & old_significant
            overlap_ratio = len(overlap) / max(1, min(len(new_significant), len(old_significant)))
            explicit_continuation = bool(
                re.match(
                    r"^(?:continue|resume|keep going|carry on|go ahead|finish(?: it)?|"
                    r"pick (?:it )?up|do it|fix that|same thing)\b",
                    new_prompt,
                )
            )
            is_continuation = (
                not new_prompt
                or new_words.issubset(_CONTINUE_WORDS)
                or explicit_continuation
                or (len(overlap) >= 3 and overlap_ratio >= 0.65)
            )

            if is_continuation and old_goal and prior_status != "complete":
                # Reclassify the restored goal so a bare "continue" retains
                # its coding/research verification criteria and model scaffold.
                ts = build_task_state(old_goal_original, allowed_tools)
                for action in restore_from.get("completed_actions") or []:
                    ts.completed_actions.append(str(action))
                for key, value in (restore_from.get("file_summaries") or {}).items():
                    ts.file_summaries[str(key)] = str(value)
                ts.files_touched = list(restore_from.get("files_touched") or [])
                ts.facts = list(restore_from.get("facts") or [])
                ts.pending_steps = list(restore_from.get("pending_steps") or [])
                ts.blockers = list(restore_from.get("blockers") or [])
        except Exception:
            pass  # keep fresh state on any restore error

    return ts


def is_opus_model(model: str) -> bool:
    m = (model or "").lower()
    return (
        "opus-4-8" in m
        or "opus-4.8" in m
        or "opus-4.7" in m
        or "opus-4.6" in m
        or m.startswith("claude-opus-4-8")
        or m.startswith("claude-opus-4-7")
        or m.startswith("claude-4.6-opus")
        or m.startswith("opus-4.8")
        or m in {"opus-4.8", "opus-4.8-thinking", "opus-4.8-max"}
    )


def needs_exec_for_math(text: str) -> bool:
    """True for non-trivial exact arithmetic that benefits from a calculator."""
    from ..gateway_client.ollama_profiles import needs_exec_for_math as _needs_exec

    return _needs_exec(text)


def build_opus_exec_guard(user_prompt: str) -> str | None:
    """Inject exec discipline for Opus on arithmetic prompts."""
    if not needs_exec_for_math(user_prompt):
        return None
    return "\n".join(
        [
            "OPUS_EXEC_GUARD;weight=W5",
            "ARITHMETIC: For this non-trivial exact calculation, run shell/exec first:",
            '  python3 -c "print(<expression>)"',
            "Base the numeric result on the command output.",
        ]
    )


def is_ollama_agentic_model(model: str) -> bool:
    try:
        from ..gateway_client.ollama_profiles import is_ollama_agentic_model as _is

        return _is(model)
    except Exception:  # noqa: BLE001
        m = (model or "").lower()
        return any(
            tag in m
            for tag in (
                "kimi",
                "glm",
                "qwen3-coder",
                "deepseek-v4",
                "qwen3.5",
                "nemotron",
            )
        )


def build_ollama_exec_guard(user_prompt: str) -> str | None:
    """Inject exec discipline for Ollama agentic models on arithmetic prompts."""
    if not needs_exec_for_math(user_prompt):
        return None
    return "\n".join(
        [
            "OLLAMA_EXEC_GUARD;weight=W5",
            "ARITHMETIC: Run python3 -c 'print(<expr>)' via exec/shell before answering.",
            "Base the non-trivial exact result on the command output.",
        ]
    )


def build_scaffold_prompt(task_type: str, *, model: str = "", user_prompt: str = "") -> str:
    base = [
        "STRONG_SMALL_MODEL_SCAFFOLD;weight=W4",
        "STATEFUL: use TASK_STATE as truth for progress; do not rely on memory of prior tool output.",
        "LOOP: plan one concrete next action -> use tool -> inspect result -> update approach.",
        "ERRORS: on stderr/failure, state likely cause, change approach, retry max 3 before escalating.",
        "DONE_CRITERIA: satisfy the task-specific criteria in TASK_STATE before final answer.",
        "FINAL: only claim success after evidence in tool results; include blocker if not verified.",
        "NO_REVISIT: Check completed_actions in TASK_STATE before calling any tool. "
        "Do not repeat a completed call merely for reassurance; re-inspect or revise when "
        "new evidence, a later mutation, or a narrower verification genuinely requires it. "
        "Otherwise move to the next unresolved step or finish.",
        "PENDING_STEPS: Before each action, check pending_steps in TASK_STATE. "
        "Work through them in order. If you think of new sub-tasks, they get added. "
        "When a step is done it auto-removes from pending. "
        "If pending_steps is empty and the task isn't done, plan the next step explicitly.",
        "NO_NARRATION: Do NOT say 'let me check', 'now let me verify', 'let me look at' "
        "before tool calls. Just call the tool. Do NOT output long explanations between "
        "tool calls. Act, inspect the result, act again or finish. "
        "After verification succeeds, give a SHORT final answer and stop.",
        "TOOL_PROTOCOL: When you need a tool, prefer native tool_calls. If the endpoint does not expose them, emit exactly one harness call per action, formatted as a double-bracket JSON object with name and args keys (see example below).",
        'TOOL_PROTOCOL_EXAMPLE: name="<tool_name>" args={...} wrapped in double square brackets, valid JSON on one line.',
        "- Do not wrap the harness call in markdown. Do not narrate before it.",
        "- Multiple tool calls in one turn: emit separate harness calls, each on its own line.",
    ]
    opus_guard = build_opus_exec_guard(user_prompt) if is_opus_model(model) else None
    if opus_guard:
        base.append(opus_guard)
    elif is_ollama_agentic_model(model):
        ollama_guard = build_ollama_exec_guard(user_prompt)
        if ollama_guard:
            base.append(ollama_guard)
    if task_type in {"coding", "debugging"}:
        base.extend(
            [
                "CODING_CHECKLIST: locate relevant files; read before write; patch smallest surface; test/readback after write.",
                "PATCH_DISCIPLINE: avoid broad refactors unless explicitly requested; prefer one coherent edit over many tiny edits.",
                "VERIFY_GATE: after write/edit/write_chunk, call read/list_dir/exec before final answer.",
            ]
        )
    elif task_type == "research":
        base.extend(
            [
                "RESEARCH_CHECKLIST: use primary/reputable sources; compare claims; report uncertainty when evidence is thin.",
            ]
        )
    return "\n".join(base)


def compress_tool_result(
    name: str, result: dict[str, Any], *, max_chars: int = 16000
) -> dict[str, Any]:
    """Return tool feedback with smart signal-preserving compression.

    Explicit large reads receive a bounded 48KB per-result budget. This is
    enough for coherent source/document chunks without allowing one tool call
    to consume the rolling context window.

    Small results pass through unchanged. Large results are trimmed to keep:
      - Status fields (ok, error, exit_code, path, total_lines)
      - The first N and last N lines of stdout/content (head+tail pattern)
      - A middle marker indicating how many lines were elided

    This prevents context bloat from large exec/read outputs while preserving
    the signal the model needs to make decisions. The full result is still in
    the trace for sleep-time consolidation.
    """
    if not isinstance(result, dict):
        return {"ok": True, "result": result}

    if name == "read" and result.get("_large_read"):
        max_chars = max(max_chars, 48_000)

    # Quick path: small results pass through (but still trim large entries lists)
    total_size = len(json.dumps(result, default=str))
    out = dict(result)

    # Trim large entries lists (list_dir with many files) — independent of total_size
    entries = out.get("entries")
    if isinstance(entries, list) and len(entries) > 50:
        out["entries"] = entries[:40] + [f"... ({len(entries) - 40} more entries)"]
        out["_entries_compressed"] = True
        out["_entries_original_count"] = len(entries)

    if total_size <= max_chars:
        return out

    # Trim large text fields with head+tail preservation
    for _field in ("content", "stdout", "stderr"):
        val = out.get(_field)
        if not isinstance(val, str) or len(val) <= max_chars // 3:
            continue
        lines = val.splitlines()
        if len(lines) <= 20:
            # Few lines but very long — truncate each line
            out[_field] = "\n".join(
                line[:200] + ("..." if len(line) > 200 else "") for line in lines
            )
            continue
        # Head + tail with elision marker
        head_n = 15
        tail_n = 10
        head = lines[:head_n]
        tail = lines[-tail_n:]
        elided = len(lines) - head_n - tail_n
        out[_field] = "\n".join(head) + f"\n... ({elided} lines elided) ...\n" + "\n".join(tail)
        out[f"_{_field}_compressed"] = True
        out[f"_{_field}_original_lines"] = len(lines)

    return out


# Non-Opus Cursor sessions fall back to Composer 2.5 for text-only planning.
ORCHESTRATOR_PLANNER_MODEL = "composer/composer-2.5"
# Default Opus tier when session is Opus-family but tier is unspecified/legacy.
ORCHESTRATOR_OPUS_PLANNER = "claude-opus-4-8-thinking-high"


def is_cursor_model(model: str) -> bool:
    """True for Cursor/Composer routed models (text-only planning, no native tools)."""
    m = (model or "").lower()
    return m.startswith(("composer/", "composer-", "cursor/", "cursor-"))


def resolve_planner_model(model: str) -> str:
    """Pick the orchestrator planner model.

    Opus sessions plan on Opus 4.8. Other Cursor variants use Composer 2.5.
    Non-Cursor models plan on themselves.
    """
    m = (model or "").strip()
    if not m:
        return ORCHESTRATOR_OPUS_PLANNER
    bare = m.split("/", 1)[-1] if "/" in m else m
    if is_opus_model(m):
        if bare.startswith(("claude-opus-4-8", "claude-opus-4-7", "claude-4.6-opus")):
            return bare
        return ORCHESTRATOR_OPUS_PLANNER
    if is_cursor_model(m):
        return ORCHESTRATOR_PLANNER_MODEL
    return m


def build_orchestrator_scaffold(*, planner_model: str = "") -> str:
    """Planner-side protocol for Opus/Composer-plan / Qwen-execute mode."""
    planner_label = planner_model or "Opus 4.8 / Composer"
    return "\n".join(
        [
            "ORCHESTRATOR_MODE;weight=W5",
            f"ROLE: You are the planner ({planner_label}). Qwen Coder executes tools natively — you never call tools yourself.",
            "LOOP: THINK (chain-of-thought) → PLAN (1-3 concrete tool steps) → inspect results → repeat or DONE.",
            "PLAN_FORMAT:",
            "  THINK:",
            "  (analyze task, what you know, what to verify next)",
            "  PLAN:",
            '  1. read path="file.py" reason="inspect before edit"',
            '  2. edit path="file.py" old="exact" new="patch" reason="minimal fix"',
            "DONE: output `DONE: <final answer>` when done_criteria met with tool evidence.",
            "DISCIPLINE:",
            "- read/list_dir before write/edit; verify writes with read or exec",
            "- one focused batch per round; replan after failures with a different approach",
            "- be specific: exact paths, exact old/new text for edits",
            "- on tool error: diagnose in THINK, then PLAN recovery steps",
            "- no JSON tool calls — text PLAN only; Qwen handles native execution",
        ]
    )


def _is_cursor_inline_model(model: str) -> bool:
    return is_cursor_model(model)


_ACTION_REQUEST_PREFIX = re.compile(
    r"^(?:(?:please|kindly)[\s,]+|"
    r"(?:can|could|would|will)\s+you\s+|"
    r"(?:i\s+(?:need|want)\s+you\s+to|i(?:'d|\s+would)\s+like\s+you\s+to)\s+|"
    r"(?:go\s+ahead\s+and|let(?:'s|\s+us))\s+)+"
)
_NON_ACTION_PREFIX = re.compile(
    r"^(?:"
    r"how|why|what|when|where|who|which|"
    r"explain|describe|summarize|compare|"
    r"tell\s+me|help\s+me\s+understand|"
    r"(?:can|could|would|should|will|do|does|did|is|are|was|were)\s+(?:i|we|it|this|that|there)\b"
    r")\b"
)
_NEGATED_ACTION_PREFIX = re.compile(r"^(?:do\s+not|don't|never|avoid)\b")
_ACTION_COMMAND_PREFIX = re.compile(
    r"^(?:"
    r"ssh(?:\s+into)?|run|execute|exec|patch|implement|fix|install|deploy|"
    r"start|restart|stop|scan|fetch|merge|port|wire|configure|verify|check|test|"
    r"read|write|edit|inspect|investigate|audit|diagnose|debug|review|"
    r"create|build|add|update|change|remove|delete|move|copy|rename|"
    r"connect|open|send|schedule|get|pull|push|retry|continue|resume|finish|"
    r"try\s+again|do\s+that"
    r")(?:\b|$)"
)


def is_action_command(text: str) -> bool:
    """Return whether *text* directly asks the agent to take an action.

    Match the request's leading intent rather than action words mentioned in a
    question.  This gate drives tool-use retries, so a false positive can waste
    several model rounds while a false negative merely leaves the normal task
    classifier and model free to choose a tool.
    """
    normalized = " ".join((text or "").casefold().strip().split())
    if not normalized:
        return False
    normalized = normalized.lstrip("-–—:;,.! ")
    if _NON_ACTION_PREFIX.match(normalized) or _NEGATED_ACTION_PREFIX.match(normalized):
        return False
    # Prefixes may be stacked ("please, could you ...").  Punctuation between
    # them is normalized here without searching deeper into the sentence.
    while match := _ACTION_REQUEST_PREFIX.match(normalized):
        normalized = normalized[match.end() :].lstrip("-–—:;,.! ")
    return bool(
        normalized
        and not _NON_ACTION_PREFIX.match(normalized)
        and not _NEGATED_ACTION_PREFIX.match(normalized)
        and _ACTION_COMMAND_PREFIX.match(normalized)
    )


def _raw_tool_name(tc: dict[str, Any]) -> str:
    fn = tc.get("function") or {}
    return str(fn.get("name") or tc.get("name") or "").strip().lower()


def nudge_for_dropped_tool_calls(
    raw_tool_calls: list[dict[str, Any]],
    valid_calls: list[dict[str, Any]],
) -> str | None:
    """Return a nudge when the model emitted tool calls that were all dropped."""
    if valid_calls or not raw_tool_calls:
        return None
    names = {_raw_tool_name(tc) for tc in raw_tool_calls}
    if names & {"exec", "shell", "remote_exec"}:
        return (
            'SYSTEM: shell/exec requires a non-empty "command" string with a real shell command. '
            'Emit ONLY JSON, e.g. {"name":"shell","arguments":{"command":"echo ok"}}'
        )
    return (
        "SYSTEM: Tool call(s) were malformed or missing required arguments. "
        'Re-emit valid JSON: {"name":"tool_name","arguments":{...}}'
    )


def enforce_final_verification(
    task: TaskState,
    content: str,
    *,
    model: str = "",
) -> str | None:
    """Return a corrective nudge if final answer needs one more pass."""
    task.final_check(content)
    if task.mutation_attempts and task.last_mutation_succeeded is False:
        return (
            "SYSTEM: The latest state-changing tool call failed. "
            "Do not claim the task is complete. Retry with a corrected action and verify it, "
            "or clearly report the blocker."
        )
    if task.writes and not task.verified_after_write:
        return (
            "SYSTEM: You changed state but have not independently verified the result. "
            "Before final answer, call read/list_dir/exec to confirm the change worked."
        )
    if task.should_escalate and task.failures >= 3:
        return (
            "SYSTEM: Escalation trigger fired: "
            + "; ".join(task.escalation_reasons[-3:])
            + ". Re-read current state, try a different approach, or clearly report blocker."
        )
    # Detect intent-to-continue language in a text-only response — the model
    # narrated its next step but forgot to emit a tool_call. Nudge it to
    # either call the tools or commit to a proper final answer.
    text = (content or "").lower().strip()
    _CONTINUE_SIGNALS = (
        "let me check",
        "i'll check",
        "i will check",
        "let me look",
        "i need to look",
        "i need to inspect",
        "i need to read",
        "i need to check",
        "i need to run",
        "let me read",
        "let me run",
        "let me try",
        "let me verify",
        "let me test",
        "i'll look",
        "i'll read",
        "i'll run",
        "i'll try",
        "i'm going to check",
        "i'm going to read",
        "i'm going to run",
        "i'm checking",
        "i'm looking",
        "i'm investigating",
        "i'm verifying",
        "i'm testing",
        "i'm running",
        "checking now",
        "investigating now",
        "looking into this",
        "about to run",
        "about to check",
        "about to read",
    )
    _CURSOR_HALLUCINATION = (
        "shell rejected",
        "shell was rejected",
        "user rejected",
        "rejected: user",
        "exec rejected",
        "exec blocked",
        "shell blocked",
        "read blocked",
        "read tool is blocked",
        "read tool was blocked",
        "shell is blocked",
        "cannot execute",
        "can't execute",
        "unable to execute",
        "no shell access",
        "shell access is",
        "permission denied",
        "sandbox",
        "trying exec instead",
        "attempting exec",
        "cursor shell",
        "cursor ide",
        "shell tool was",
        "no cursor shell",
    )
    cursor_model = _is_cursor_inline_model(model)
    if any(sig in text for sig in _CURSOR_HALLUCINATION):
        if cursor_model:
            return (
                "SYSTEM: There is no Cursor Shell tool in this runtime; Norax CAN use read and "
                "execute shell via its shell/exec tools, and those tools are available. Retry "
                "with corrected arguments or a different command instead of stopping. "
                'Emit inline JSON such as {"name":"shell","arguments":{"command":"echo ok"}}.'
            )
        return (
            "SYSTEM: Norax runtime CAN use read and execute shell via shell/exec; those tools are available. "
            "Retry with corrected arguments or a different command instead of stopping. "
            'For shell, output JSON such as {"name":"shell","arguments":{"command":"echo ok"}}.'
        )
    # Action commands must not finish text-only on round 1 (common Composer failure mode).
    if cursor_model and task.tool_calls == 0 and is_action_command(task.goal) and len(text) < 400:
        return (
            "SYSTEM: Action command detected but zero tools were called. "
            'Emit ONLY JSON now — e.g. {"name":"remote_exec","arguments":{"node_id":"staging","command":"ls Desktop"}} '
            'or {"name":"shell","arguments":{"command":"echo ok"}} with your real command. No narration.'
        )
    # Continuation-signal detection: if the text contains phrases like
    # "let me check", "i'll run", it's mid-task narration.  But only trigger
    # when the response is short or lacks substantive content — a long answer
    # with code blocks or structured results is a real final answer even if it
    # happens to contain "let me check" in passing.
    _has_substance = (
        len(text) >= 400
        or "```" in text
        or any(sig in text for sig in ("✅", "✗", "result:", "output:", "exit code"))
    )
    if (
        any(sig in text for sig in _CONTINUE_SIGNALS)
        and "let me know" not in text
        and not _has_substance
    ):
        if cursor_model:
            return (
                "SYSTEM: Norax uses inline JSON tool calls, not Cursor IDE Shell. "
                'Emit {"name":"tool_name","arguments":{...}} now — no narration. '
                "If done, provide the actual results."
            )
        return (
            "SYSTEM: Your response looks like an intent to continue, not a final answer. "
            "If there's more work to do, call the tools now. "
            "If you're done, provide the actual results."
        )
    # Long thinking preambles can exceed 400 chars but still be mid-task narration.
    if (
        cursor_model
        and any(sig in text for sig in _CONTINUE_SIGNALS)
        and len(text) < 1200
        and not _has_substance
    ):
        if task.tool_calls == 0 or task.task_type in {"coding", "debugging"}:
            return (
                "SYSTEM: Action task with no tool call detected. "
                'Emit {"name":"shell","arguments":{"command":"echo ok"}} (with your real command) or read/list_dir first.'
            )

    # Do not require magic completion words after a tool call: concise final
    # answers such as "Read: hello world" are valid. Mid-task narration is
    # handled above using explicit continuation signals.
    return None


def repo_map_from_tree(root: str, entries: list[dict[str, Any]], *, limit: int = 60) -> str:
    names = []
    for e in entries[:limit]:
        kind = e.get("kind", "?")
        name = e.get("name", "")
        if name:
            names.append(f"{kind}:{name}")
    return f"REPO_MAP;root={root};entries=" + ",".join(names)
