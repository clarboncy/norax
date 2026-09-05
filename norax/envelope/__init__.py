"""Wire shapes — pure dataclasses, zero I/O.

Imported by every other package. This module MUST NOT import from
`adapter`, `brain`, `dispatch`, `prompt`, `gateway_client`, `memory`,
`http`, `code`, `observability`, or `runtime`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass(frozen=True)
class Principal:
    id: str
    label: str
    trust: bool
    tier: Literal["owner", "admin", "user", "guest"]


@dataclass(frozen=True)
class ThreadBinding:
    channel: str
    thread_id: str
    kind: Literal["none", "code", "research", "custom"]
    session_id: str | None = None


@dataclass
class SensoryInput:
    channel: Literal["chat", "email", "payment", "schedule", "heartbeat", "node", "http"]
    source: str
    message_id: str
    timestamp: datetime
    sender: Principal
    body: str
    attachments: list[dict] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    trusted: bool = False
    thread_binding: ThreadBinding | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    call_id: str
    name: str
    args: dict
    caller: Principal
    tier: Literal["T0", "T1", "T2", "dynamic"]
    reason: str | None = None
    parent_message_id: str | None = None
    history_window: list[str] = field(default_factory=list)


@dataclass
class ToolResult:
    call_id: str
    ok: bool
    value: Any
    error: str | None = None
    cost: dict = field(default_factory=dict)


@dataclass
class StateBlock:
    valence: float = 0.0
    confidence: float = 1.0
    arousal: float = 1.0


@dataclass
class FocusBlock:
    summary: str = ""
    suggested_skills: list[str] = field(default_factory=list)


@dataclass
class MemoryBlock:
    # Each item is (text, score) or (text, score, kind) where kind is the
    # source category (semantic|procedural|intel|scratchpad|focus|sleep|hot).
    items: list[tuple] = field(default_factory=list)


@dataclass
class SkillsDirectoryBlock:
    entries: list[tuple[str, str, str]] = field(default_factory=list)
    # (name, scope, invocation in {"always","on-request","idle-learned","disabled"})


@dataclass
class BrainContext:
    env: SensoryInput
    state: StateBlock = field(default_factory=StateBlock)
    focus: FocusBlock = field(default_factory=FocusBlock)
    memory: MemoryBlock = field(default_factory=MemoryBlock)
    skills: SkillsDirectoryBlock = field(default_factory=SkillsDirectoryBlock)
    allowed_tools: list[str] = field(default_factory=list)
    runtime_info: dict = field(default_factory=dict)
    decision: Literal["emit_reply", "silent", "defer"] = "silent"
    silent_notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
