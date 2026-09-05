"""Norax remote node control plane.

A node is a user-owned computer running `norax-node`. Nodes connect outbound
with an enrollment token and accept small job envelopes from the runtime.
"""

from .registry import NodeInfo, RemoteRegistry

__all__ = ["RemoteRegistry", "NodeInfo"]
