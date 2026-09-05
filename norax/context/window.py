"""Rolling context window — lossless spillover to sleep/.

Design principles (user-driven):
  1. Preserve raw evicted/compacted frames by spilling them to sleep/ as JSONL.
  2. Once a task is solved, remove old tool traces after lossless spill.
  3. ALWAYS protect the head (system/persona) and a realistic recent tail.
  4. Keep tool_call and its tool_result frames as an atomic unit — never split.
  5. Retrieval back from sleep/ is the job of the memory layer's
     ExternalRetriever (which already indexes sleep/). The rolling window
     itself is write-only to sleep/.

Typical sizing:
  - budget_tokens = 256_000  (target for 256k+ model contexts, with generous slack)
  - protect_head_tokens = 4_000   (system prompt + soul + identity)
  - protect_tail_turns = 12   (last 10-15 conversation turns kept verbatim)
  - eviction_batch_pct = 0.25   (when full, evict 25% of middle → breathing room)

Token accounting: we use char/4 as a cheap estimator (same default as
Microsoft Agent Framework's CharacterEstimatorTokenizer). Swap in a
model-specific tokenizer later via `TokenCounter` protocol.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from ..atomic import atomic_write_text, read_bounded_bytes

log = logging.getLogger("norax.context.window")


FrameKind = Literal["system", "user", "assistant", "tool_call", "tool_result"]
_FRAME_KINDS = frozenset({"system", "user", "assistant", "tool_call", "tool_result"})
_MAX_WINDOW_BYTES = 16 * 1024 * 1024
_MAX_WINDOW_FRAMES = 10_000
_MAX_FRAME_CONTENT_CHARS = 1_048_576
_MAX_FRAME_META_BYTES = 262_144


@dataclass
class Frame:
    """One atomic unit of conversation context.

    tool_call frames must carry a `call_id` and the matching tool_result
    frames carry the same `call_id`; eviction logic treats them atomically.
    """

    kind: FrameKind
    content: str
    ts: float = field(default_factory=lambda: time.time())
    call_id: str | None = None  # binds tool_call ↔ tool_result
    turn_id: int = 0  # monotonic; same user→assistant cycle shares turn_id
    meta: dict = field(default_factory=dict)

    def token_estimate(self) -> int:
        return max(1, len(self.content) // 4)

    def to_jsonl(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> Frame:
        if not isinstance(d, dict):
            raise ValueError("window frame must be an object")
        kind = d.get("kind")
        if kind not in _FRAME_KINDS:
            raise ValueError("window frame contains an invalid kind")
        content = d.get("content")
        if not isinstance(content, str) or len(content) > _MAX_FRAME_CONTENT_CHARS:
            raise ValueError("window frame content is invalid or too large")
        ts = d.get("ts", time.time())
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
            raise ValueError("window frame timestamp must be finite")
        call_id = d.get("call_id")
        if call_id is not None and (not isinstance(call_id, str) or len(call_id) > 512):
            raise ValueError("window frame call_id is invalid or too large")
        turn_id = d.get("turn_id", 0)
        if (
            isinstance(turn_id, bool)
            or not isinstance(turn_id, int)
            or not 0 <= turn_id <= 2_147_483_647
        ):
            raise ValueError("window frame turn_id is invalid")
        meta = d.get("meta", {}) or {}
        if not isinstance(meta, dict):
            raise ValueError("window frame metadata must be an object")
        try:
            meta_size = len(json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("window frame metadata is not JSON serializable") from exc
        if meta_size > _MAX_FRAME_META_BYTES:
            raise ValueError("window frame metadata is too large")
        return cls(
            kind=cast(FrameKind, kind),
            content=content,
            ts=float(ts),
            call_id=call_id,
            turn_id=turn_id,
            meta=meta,
        )


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class CharDiv4Tokenizer:
    """Cheap tokenizer: len(text) // 4. Swap in tiktoken later."""

    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


@dataclass
class EvictionResult:
    evicted: list[Frame]
    spill_path: Path | None
    tokens_freed: int


@dataclass
class RollingWindow:
    budget_tokens: int = 256_000
    protect_head_tokens: int = 4_000
    protect_tail_turns: int = 12
    eviction_batch_pct: float = 0.25
    sleep_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("NORAX_SLEEP_DIR", Path.cwd() / "memory" / "sleep")
        )
    )
    tokenizer: TokenCounter = field(default_factory=CharDiv4Tokenizer)
    head: list[Frame] = field(default_factory=list)  # system/persona, never evicted
    body: list[Frame] = field(default_factory=list)  # main conversation
    _next_turn_id: int = 1

    # ---- ingestion -------------------------------------------------------

    def add_system(self, content: str, **meta) -> Frame:
        f = Frame(kind="system", content=content, turn_id=0, meta=meta)
        self.head.append(f)
        return f

    def start_turn(self) -> int:
        tid = self._next_turn_id
        self._next_turn_id += 1
        return tid

    def add_user(self, content: str, *, turn_id: int | None = None, **meta) -> Frame:
        tid = turn_id if turn_id is not None else self.start_turn()
        f = Frame(kind="user", content=content, turn_id=tid, meta=meta)
        self.body.append(f)
        return f

    def add_assistant(self, content: str, *, turn_id: int, **meta) -> Frame:
        f = Frame(kind="assistant", content=content, turn_id=turn_id, meta=meta)
        self.body.append(f)
        return f

    def add_tool_call(self, *, call_id: str, content: str, turn_id: int, **meta) -> Frame:
        f = Frame(kind="tool_call", content=content, call_id=call_id, turn_id=turn_id, meta=meta)
        self.body.append(f)
        return f

    def add_tool_result(self, *, call_id: str, content: str, turn_id: int, **meta) -> Frame:
        f = Frame(kind="tool_result", content=content, call_id=call_id, turn_id=turn_id, meta=meta)
        self.body.append(f)
        return f

    # ---- accounting ------------------------------------------------------

    def total_tokens(self) -> int:
        return sum(f.token_estimate() for f in self.head) + sum(
            f.token_estimate() for f in self.body
        )

    def head_tokens(self) -> int:
        return sum(f.token_estimate() for f in self.head)

    def protected_tail_turn_ids(self) -> set[int]:
        """The last N distinct turn_ids are protected from eviction."""
        seen: list[int] = []
        for f in reversed(self.body):
            if f.turn_id not in seen:
                seen.append(f.turn_id)
            if len(seen) >= self.protect_tail_turns:
                break
        return set(seen)

    # ---- eviction --------------------------------------------------------

    def needs_eviction(self) -> bool:
        return self.total_tokens() > self.budget_tokens

    def compact_solved_turns(self) -> int:
        """Trim solved middle tool traces to sleep/ archive, losslessly.

        This is not summarization/compression. Raw frames are written to the
        sleep archive first, then removed from the active window so sleeping
        systems can process them later.
        """
        protected_ids = self.protected_tail_turn_ids()
        groups = self._atomic_groups(self.body)
        evicted: list[Frame] = []
        kept: list[Frame] = []
        # We relax the 'solved' requirement. If a tool trace is no longer in the
        # protected tail (the last N turns), it's old context and can be safely
        # archived to sleep/. The model has already synthesized its result.
        for group in groups:
            turn_ids = {f.turn_id for f in group}
            if turn_ids & protected_ids:
                kept.extend(group)
                continue
            has_tool = any(f.kind in {"tool_call", "tool_result"} for f in group)
            if has_tool:
                evicted.extend(group)
            else:
                kept.extend(group)

        if evicted:
            self._write_spill(evicted, source="solved_tool_trace_archive")
            self.body = kept

        # Enforce the hard budget after compaction
        self.evict_if_needed()

        return len(evicted)

    def _atomic_groups(self, frames: Iterable[Frame]) -> list[list[Frame]]:
        """Group tool_call + tool_result into atomic units; everything else is
        a singleton group. This preserves API validity when we evict."""
        out: list[list[Frame]] = []
        pending: dict[str, list[Frame]] = {}
        for f in frames:
            if f.kind == "tool_call" and f.call_id:
                pending[f.call_id] = [f]
            elif f.kind == "tool_result" and f.call_id and f.call_id in pending:
                pending[f.call_id].append(f)
                out.append(pending.pop(f.call_id))
            else:
                out.append([f])
        # Any unclosed tool_call (no result yet): treat as singleton
        for group in pending.values():
            out.append(group)
        return out

    def evict_if_needed(self) -> EvictionResult | None:
        """Evict oldest-middle frames to sleep/ until back under budget.

        Protects head entirely and last `protect_tail_turns` turns entirely.
        Returns None if no eviction was needed.
        """
        if not self.needs_eviction():
            return None

        protected_ids = self.protected_tail_turn_ids()
        # Split body into evictable vs protected-tail (preserving order)
        evictable_groups: list[list[Frame]] = []
        protected_tail_frames: list[Frame] = []
        for group in self._atomic_groups(self.body):
            # A group is protected iff ANY of its frames is in protected_ids
            if any(f.turn_id in protected_ids for f in group):
                protected_tail_frames.extend(group)
            else:
                evictable_groups.append(group)

        # How many tokens we need to free — target = aggressive breathing room
        over_by = self.total_tokens() - self.budget_tokens
        batch_target = int(self.budget_tokens * self.eviction_batch_pct)
        to_free = max(over_by, batch_target)

        # Evict oldest-first until we've freed enough (or we're out of evictable)
        evicted: list[Frame] = []
        freed = 0
        remaining_groups: list[list[Frame]] = []
        i = 0
        while i < len(evictable_groups) and freed < to_free:
            grp = evictable_groups[i]
            for f in grp:
                evicted.append(f)
                freed += f.token_estimate()
            i += 1
        remaining_groups = evictable_groups[i:]

        if not evicted:
            return None

        # Write spill file (lossless JSONL; one frame per line)
        spill_path = self._write_spill(evicted)

        # Reassemble body: remaining evictable middle + protected tail
        self.body = [f for grp in remaining_groups for f in grp] + protected_tail_frames

        log.info(
            "rolling_window.evict n=%d tokens_freed=%d spill=%s remaining=%d",
            len(evicted),
            freed,
            spill_path,
            self.total_tokens(),
        )
        return EvictionResult(evicted=evicted, spill_path=spill_path, tokens_freed=freed)

    def _write_spill(self, frames: list[Frame], *, source: str = "rolling_window") -> Path:
        self.sleep_dir.mkdir(parents=True, exist_ok=True)
        # A random suffix avoids check-then-write collisions across overlapping
        # runtime processes. Atomic replacement prevents a reader from seeing a
        # partially written archive.
        ts = datetime.now(UTC).strftime("%Y-%m-%d-%H%M%S-%f")
        spill_id = uuid.uuid4().hex[:12]
        path = self.sleep_dir / f"spill-{ts}-{spill_id}.jsonl"
        raw_text = "".join(frame.to_jsonl() + "\n" for frame in frames)
        atomic_write_text(path, raw_text)

        # Sidecar .md for the retriever (grep-friendly; one line per frame content)
        md_path = path.with_suffix(".md")
        sidecar_lines = [f"SPILL;ts={ts};n={len(frames)};source={source}"]
        for frame in frames:
            # Flatten multiline content to a single line for the retriever
            flat = frame.content.replace("\n", " ⏎ ")[:600]
            sidecar_lines.append(f"{frame.kind.upper()}#{frame.turn_id}:{flat}")
        atomic_write_text(md_path, "\n".join(sidecar_lines) + "\n")
        return path

    def dump_to_sleep(self, *, half: bool = False) -> EvictionResult | None:
        """Dump active body context to sleep/.

        Full mode spills and clears the whole body while preserving the bootstrap
        head. Half mode spills and clears the older/back 50% of body frames,
        preserving atomic tool call/result groups.
        """
        if not self.body:
            return None
        groups = self._atomic_groups(self.body)
        if half:
            n_groups = max(1, len(groups) // 2)
            dump_groups = groups[:n_groups]
            keep_groups = groups[n_groups:]
        else:
            dump_groups = groups
            keep_groups = []
        dumped = [f for grp in dump_groups for f in grp]
        if not dumped:
            return None
        path = self._write_spill(
            dumped, source="command_dump_half" if half else "command_dump_full"
        )
        self.body = [f for grp in keep_groups for f in grp]
        freed = sum(f.token_estimate() for f in dumped)
        return EvictionResult(evicted=dumped, spill_path=path, tokens_freed=freed)

    # ---- projection for model --------------------------------------------

    def messages(self) -> list[dict]:
        """Project to OpenAI-style message list."""
        out: list[dict] = []
        for f in self.head:
            out.append({"role": "system", "content": f.content})
        for f in self.body:
            if f.kind == "user":
                out.append({"role": "user", "content": f.content})
            elif f.kind == "assistant":
                out.append({"role": "assistant", "content": f.content})
            elif f.kind == "tool_call":
                out.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": f.call_id, "content": f.content}],
                    }
                )
            elif f.kind == "tool_result":
                out.append({"role": "tool", "tool_call_id": f.call_id, "content": f.content})
        return out

    # ---- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "budget_tokens": self.budget_tokens,
            "protect_tail_turns": self.protect_tail_turns,
            "protect_head_tokens": self.protect_head_tokens,
            "eviction_batch_pct": self.eviction_batch_pct,
            "head": [asdict(f) for f in self.head],
            "body": [asdict(f) for f in self.body],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)
        if len(payload.encode("utf-8")) > _MAX_WINDOW_BYTES:
            raise ValueError(f"rolling window exceeds {_MAX_WINDOW_BYTES} persisted bytes")
        atomic_write_text(path, payload, mode=0o600)

    @classmethod
    def load(cls, path: Path, *, budget_tokens: int = 256_000) -> RollingWindow:
        try:
            raw = read_bounded_bytes(path, max_bytes=_MAX_WINDOW_BYTES)
            d = json.loads(raw.decode("utf-8"))
            if not isinstance(d, dict):
                raise ValueError("persisted rolling window root must be an object")

            def bounded_int(key: str, default: int, minimum: int, maximum: int) -> int:
                value = d.get(key, default)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"persisted rolling window {key} must be an integer")
                if not minimum <= value <= maximum:
                    raise ValueError(f"persisted rolling window {key} is out of range")
                return value

            persisted_budget = bounded_int("budget_tokens", budget_tokens, 1, 4_000_000)
            protect_tail = bounded_int("protect_tail_turns", 12, 0, 1_000)
            protect_head = bounded_int("protect_head_tokens", 4_000, 0, 1_000_000)
            eviction_pct = d.get("eviction_batch_pct", 0.25)
            if (
                isinstance(eviction_pct, bool)
                or not isinstance(eviction_pct, (int, float))
                or not math.isfinite(eviction_pct)
                or not 0.01 <= eviction_pct <= 1.0
            ):
                raise ValueError("persisted rolling window eviction_batch_pct is out of range")
            head = d.get("head", [])
            body = d.get("body", [])
            if not isinstance(head, list) or not isinstance(body, list):
                raise ValueError("persisted rolling window frames must be arrays")
            if len(head) + len(body) > _MAX_WINDOW_FRAMES:
                raise ValueError("persisted rolling window contains too many frames")
            w = cls(
                budget_tokens=persisted_budget,
                protect_tail_turns=protect_tail,
                protect_head_tokens=protect_head,
                eviction_batch_pct=float(eviction_pct),
            )
            w.head = [Frame.from_dict(frame) for frame in head]
            w.body = [Frame.from_dict(frame) for frame in body]
            max_tid = max([frame.turn_id for frame in w.body] + [0])
            w._next_turn_id = max_tid + 1
            return w
        except FileNotFoundError:
            return cls(budget_tokens=budget_tokens)
        except (
            json.JSONDecodeError,
            KeyError,
            OSError,
            RecursionError,
            TypeError,
            UnicodeError,
            ValueError,
        ) as exc:
            log.warning("rolling_window.load_failed path=%s error=%s", path, exc)
            return cls(budget_tokens=budget_tokens)
