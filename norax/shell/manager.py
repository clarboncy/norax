"""Per-conversation shell session registry."""

from __future__ import annotations

from .session import ShellSession

_manager: ShellSessionManager | None = None


class ShellSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, ShellSession] = {}

    def get(self, session_id: str) -> ShellSession:
        sid = (session_id or "default").strip() or "default"
        if sid not in self._sessions:
            self._sessions[sid] = ShellSession(session_id=sid)
        return self._sessions[sid]

    def reset(self, session_id: str) -> None:
        sid = (session_id or "default").strip() or "default"
        self._sessions.pop(sid, None)


def get_shell_manager() -> ShellSessionManager:
    global _manager
    if _manager is None:
        _manager = ShellSessionManager()
    return _manager
