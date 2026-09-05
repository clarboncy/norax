"""Bounded, resumable human approval records for configured tool calls.

Design:
  - Configurable: always-pause for writes, pause for specific tools/paths
  - Exact tool arguments are bound to each one-shot approval
  - Pending records are bounded, deduplicated, and expire fail-closed
  - The authenticated owner can approve or reject through slash commands
  - Modified arguments are re-run through RiskGate before execution

Integration:
  - agent_loop checks should_interrupt(tool, args) before each tool dispatch
  - An interrupted result includes an ID and is never executed in that turn
  - The owner resolves the ID, then retries/continues the exact operation
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger("norax.runtime.interrupts")


class InterruptAction(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"


class InterruptReason(Enum):
    WRITE_OPERATION = "write_operation"
    DANGEROUS_COMMAND = "dangerous_command"
    SENSITIVE_PATH = "sensitive_path"
    HIGH_COST = "high_cost"
    MANUAL_RULE = "manual_rule"


@dataclass
class InterruptRule:
    """A rule that triggers an interrupt."""

    tools: set[str] = field(default_factory=set)
    path_patterns: list[str] = field(default_factory=list)
    command_patterns: list[str] = field(default_factory=list)
    reason: InterruptReason = InterruptReason.MANUAL_RULE
    description: str = ""
    _path_regexes: tuple[re.Pattern[str], ...] = field(init=False, repr=False)
    _command_regexes: tuple[re.Pattern[str], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.path_patterns) > 32 or len(self.command_patterns) > 32:
            raise ValueError("an interrupt rule may contain at most 32 patterns per kind")
        if any(len(pattern) > 512 for pattern in [*self.path_patterns, *self.command_patterns]):
            raise ValueError("interrupt regex patterns are limited to 512 characters")
        self._path_regexes = tuple(re.compile(pattern) for pattern in self.path_patterns)
        self._command_regexes = tuple(re.compile(pattern) for pattern in self.command_patterns)


@dataclass
class PendingInterrupt:
    """An interrupt waiting for human response."""

    interrupt_id: str
    tool_name: str
    args_digest: str
    reason: InterruptReason
    description: str
    timestamp: float
    resolved: bool = False
    action: InterruptAction | None = None
    modified_args: dict | None = None
    resolved_at: float = 0.0
    resolved_by: str = ""


class InterruptManager:
    """Manages interrupt rules and pending interrupts."""

    def __init__(self, *, max_pending: int = 128, ttl_seconds: float = 86_400.0) -> None:
        self._rules: list[InterruptRule] = []
        self._pending: dict[str, PendingInterrupt] = {}
        self._enabled: bool = True
        self._max_pending = min(4096, max(1, int(max_pending)))
        self._ttl_seconds = min(604_800.0, max(60.0, float(ttl_seconds)))

    def add_rule(self, rule: InterruptRule) -> None:
        self._rules.append(rule)

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def should_interrupt(self, tool_name: str, args: dict) -> tuple[bool, InterruptReason, str]:
        """Check if a tool call should be interrupted.

        Returns (should_interrupt, reason, description).
        """
        if not self._enabled:
            return False, InterruptReason.MANUAL_RULE, ""

        path_text: str | None = None
        command_text: str | None = None
        for rule in self._rules:
            # Check tool match
            if rule.tools and tool_name not in rule.tools:
                continue

            # Check path patterns
            if rule._path_regexes:
                if path_text is None:
                    path_values: list[str] = []
                    for field_name in ("path", "root", "cwd", "files"):
                        value = args.get(field_name)
                        if isinstance(value, str):
                            path_values.append(value[:4096])
                        elif isinstance(value, (list, tuple)):
                            path_values.extend(str(item)[:4096] for item in value[:32])
                    path_text = "\n".join(path_values)[:32_768]
                matched = any(pattern.search(path_text) for pattern in rule._path_regexes)
                if not matched:
                    continue

            # Check command patterns
            if rule._command_regexes:
                if command_text is None:
                    command_text = str(args.get("command", ""))[:100_000]
                matched = any(pattern.search(command_text) for pattern in rule._command_regexes)
                if not matched:
                    continue

            return True, rule.reason, rule.description

        return False, InterruptReason.MANUAL_RULE, ""

    def create_interrupt(
        self, tool_name: str, args: dict, reason: InterruptReason, description: str
    ) -> PendingInterrupt:
        """Create or return an identical unresolved interrupt."""
        self._expire()
        args_digest = self._call_digest(tool_name, args)
        for pending in self._pending.values():
            if (
                not pending.resolved
                and pending.tool_name == tool_name
                and pending.args_digest == args_digest
                and pending.reason is reason
            ):
                return pending
        while len(self._pending) >= self._max_pending:
            oldest = min(self._pending, key=lambda key: self._pending[key].timestamp)
            self._pending.pop(oldest, None)
        interrupt_id = f"interrupt_{secrets.token_urlsafe(18)}"
        pending = PendingInterrupt(
            interrupt_id=interrupt_id,
            tool_name=tool_name,
            args_digest=args_digest,
            reason=reason,
            description=description,
            timestamp=time.time(),
        )
        self._pending[interrupt_id] = pending
        log.info("interrupt.created id=%s tool=%s reason=%s", interrupt_id, tool_name, reason.value)
        return pending

    def resolve_interrupt(
        self,
        interrupt_id: str,
        action: InterruptAction,
        modified_args: dict | None = None,
        *,
        resolved_by: str,
    ) -> bool:
        """Resolve a pending record after host authentication.

        ``resolved_by`` must be the authenticated principal supplied by the
        host adapter, never model-generated tool arguments.
        """
        if not str(resolved_by).strip():
            raise ValueError("resolved_by must identify the authenticated approver")
        self._expire()
        pending = self._pending.get(interrupt_id)
        if pending is None or pending.resolved:
            return False
        if action is InterruptAction.MODIFY and not isinstance(modified_args, dict):
            raise ValueError("modified_args must be a dict for modify resolutions")
        pending.resolved = True
        pending.action = action
        pending.modified_args = dict(modified_args) if modified_args is not None else None
        pending.resolved_at = time.time()
        pending.resolved_by = str(resolved_by)[:256]
        log.info("interrupt.resolved id=%s action=%s", interrupt_id, action.value)
        return True

    def get_pending(self) -> list[PendingInterrupt]:
        """Get all unresolved interrupts."""
        self._expire()
        return [p for p in self._pending.values() if not p.resolved]

    def get_interrupt(self, interrupt_id: str) -> PendingInterrupt | None:
        self._expire()
        return self._pending.get(interrupt_id)

    def clear_resolved(self) -> int:
        """Remove resolved interrupts. Returns count cleared."""
        to_remove = [k for k, v in self._pending.items() if v.resolved]
        for k in to_remove:
            del self._pending[k]
        return len(to_remove)

    def apply_resolution(self, interrupt_id: str) -> dict | None:
        """Inspect a resolved interrupt without consuming its one-shot approval.

        Returns:
          - None if interrupt not found or not resolved
          - {"action": "approve", "args": None} if approved
          - {"action": "reject", "args": None} if rejected
          - {"action": "modify", "args": modified_args} if modified
        """
        pending = self._pending.get(interrupt_id)
        if pending is None or not pending.resolved:
            return None

        if pending.action == InterruptAction.APPROVE:
            return {"action": "approve", "args": None}
        elif pending.action == InterruptAction.REJECT:
            return {"action": "reject", "args": None}
        elif pending.action == InterruptAction.MODIFY:
            return {"action": "modify", "args": dict(pending.modified_args or {})}
        return None

    def consume_resolution(self, tool_name: str, args: dict) -> dict | None:
        """Consume one resolved decision bound to this exact tool call."""
        # Normal autonomous operation has no pending approvals. Avoid
        # serializing and hashing every tool payload on that overwhelmingly
        # common path; exact-call binding is only needed when records exist.
        if not self._pending:
            return None
        self._expire()
        if not self._pending:
            return None
        args_digest = self._call_digest(tool_name, args)
        for interrupt_id, pending in list(self._pending.items()):
            if (
                not pending.resolved
                or pending.tool_name != tool_name
                or pending.args_digest != args_digest
            ):
                continue
            result = self.apply_resolution(interrupt_id)
            self._pending.pop(interrupt_id, None)
            if result is not None and result.get("action") == "approve":
                result["args"] = dict(args)
            return result
        return None

    @staticmethod
    def _call_digest(tool_name: str, args: dict) -> str:
        payload = json.dumps(
            {"tool": tool_name, "args": args},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _expire(self) -> None:
        if not self._pending:
            return
        cutoff = time.time() - self._ttl_seconds
        for interrupt_id in [
            key for key, pending in self._pending.items() if pending.timestamp < cutoff
        ]:
            self._pending.pop(interrupt_id, None)


# Default rules for owner-safe autonomous operation
def create_default_rules() -> list[InterruptRule]:
    """Create default interrupt rules for safe autonomous operation.

    Only genuinely dangerous operations that could cause irreversible data
    loss or security compromise require approval. Routine operations like
    service restarts, docker stop, ollama pull, file edits within the
    workspace, and normal exec/shell commands are allowed autonomously.
    """
    return [
        # Interrupt on writes to critical system paths
        InterruptRule(
            tools={"write", "write_chunk", "edit"},
            path_patterns=[r"/boot/", r"/root/", r"/etc/(?!tmp)"],
            reason=InterruptReason.SENSITIVE_PATH,
            description="Write to critical system path",
        ),
        # Interrupt on genuinely destructive commands — irreversible data loss
        InterruptRule(
            tools={"exec", "shell", "remote_exec"},
            command_patterns=[
                r"\brm\s+-rf\s+/(?!home/|tmp/)",
                r"\bmkfs\b",
                r"\bdd\s+if=/dev/",
                r":\(\)\s*\{\s*:\|:&\s*\};:",
                r"\bDROP\s+(?:TABLE|DATABASE|SCHEMA)\b",
                r"\bFORMAT\s+[A-Z]:",
                r"(?:^|[;&|\n]\s*)(?:sudo\s+)?(?:kill|killall|pkill)\s+.*(?:norax|ollama|python)",
            ],
            reason=InterruptReason.DANGEROUS_COMMAND,
            description="Irreversible destructive command",
        ),
        # Interrupt on remote writes to other machines
        InterruptRule(
            tools={"remote_write"},
            reason=InterruptReason.WRITE_OPERATION,
            description="Remote write to another machine",
        ),
    ]


# Singleton instance
_manager: InterruptManager | None = None


def get_interrupt_manager() -> InterruptManager:
    global _manager
    if _manager is None:
        _manager = InterruptManager()
        for rule in create_default_rules():
            _manager.add_rule(rule)
    return _manager
