"""LoopGuard — detects repeat/ping-pong patterns in tool-call sequences.

Two heuristics:
  1. REPEAT — same (tool, args_hash) called N times in a row.
  2. PING_PONG — (A, B, A, B, ...) cycle of length >= K.

On detection, LoopGuard raises ``LoopDetected``. The agent loop converts that
signal into an explicit non-executed tool result and replans; standalone
``Dispatcher`` callers receive the exception and decide their own recovery.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field


class LoopDetected(Exception):
    def __init__(self, pattern: str, detail: str) -> None:
        super().__init__(f"loop.detected pattern={pattern} detail={detail}")
        self.pattern = pattern
        self.detail = detail


def _fingerprint(tool: str, args: dict) -> str:
    body = json.dumps({"t": tool, "a": args}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()[:12]


@dataclass
class LoopGuard:
    repeat_threshold: int = 3
    pingpong_min_cycles: int = 3
    window: int = 12
    _history: deque[str] = field(init=False)

    def __post_init__(self) -> None:
        self.repeat_threshold = max(2, int(self.repeat_threshold))
        self.pingpong_min_cycles = max(2, int(self.pingpong_min_cycles))
        required = max(self.repeat_threshold, self.pingpong_min_cycles * 2)
        self.window = max(required, int(self.window))
        self._history = deque(maxlen=self.window)

    def observe(self, tool: str, args: dict) -> None:
        fp = _fingerprint(tool, args)
        self._history.append(fp)

        # REPEAT
        if len(self._history) >= self.repeat_threshold:
            tail = list(self._history)[-self.repeat_threshold :]
            if len(set(tail)) == 1:
                raise LoopDetected("repeat", f"fp={fp} x{self.repeat_threshold}")

        # PING_PONG (A,B,A,B,...): cycle of length 2, >= pingpong_min_cycles complete cycles
        need = self.pingpong_min_cycles * 2
        if len(self._history) >= need:
            tail = list(self._history)[-need:]
            a, b = tail[0], tail[1]
            if (
                a != b
                and all(tail[i] == a for i in range(0, need, 2))
                and all(tail[i] == b for i in range(1, need, 2))
            ):
                raise LoopDetected("ping_pong", f"a={a} b={b} cycles={self.pingpong_min_cycles}")

    def reset(self) -> None:
        self._history.clear()
