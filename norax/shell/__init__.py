"""Norax-owned shell sessions — stateful bash for IDE model tool overrides."""

from .manager import ShellSessionManager, get_shell_manager
from .session import ShellSession

__all__ = ["ShellSession", "ShellSessionManager", "get_shell_manager"]
