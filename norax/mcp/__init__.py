"""MCP (Model Context Protocol) integration for Norax.

Exposes Norax's 22+ tools as MCP tools (server) and connects to
external MCP servers (client) for dynamic tool discovery.

Server transports: stdio, streamable HTTP
Client transports: stdio (subprocess), SSE, streamable HTTP
"""

from .client import NoraxMCPClient, discover_tools
from .server import NoraxMCPServer

__all__ = ["NoraxMCPServer", "NoraxMCPClient", "discover_tools"]
