"""A2A (Agent-to-Agent) Protocol — Google's inter-agent communication standard.

Norax A2A implementation:
  - AgentCard: capability advertisement
  - Task lifecycle: submitted → working → input-required → completed → canceled
  - JSON-RPC 2.0 + SSE for streaming
  - Push notifications for long-running tasks

Usage:
    # Server: expose Norax as an A2A agent
    server = NoraxA2AServer(runtime)
    # Client: discover and delegate to remote A2A agents
    client = A2AClient("https://remote-agent.example.com")
    card = await client.get_agent_card()
    result = await client.send_task("analyze this data")
"""

from .client import A2AClient, A2ATaskResult
from .server import A2AError, AgentCard, AgentSkill, NoraxA2AServer, create_a2a_app

__all__ = [
    "NoraxA2AServer",
    "AgentCard",
    "AgentSkill",
    "A2AClient",
    "A2ATaskResult",
    "A2AError",
    "create_a2a_app",
]
