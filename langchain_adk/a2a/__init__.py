"""A2A (Agent-to-Agent) protocol v1.0 support.

Requires the ``a2a`` extra: ``pip install langchain-adk[a2a]``
"""

from langchain_adk.a2a.types import (
    A2AModel,
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Artifact,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageConfiguration,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    data_part,
    file_part,
    text_part,
)


def __getattr__(name: str):
    """Lazy-load A2AServer and events_to_a2a_stream to avoid hard fastapi dependency."""
    if name == "A2AServer":
        from langchain_adk.a2a.server import A2AServer
        return A2AServer
    if name == "events_to_a2a_stream":
        from langchain_adk.a2a.converters import events_to_a2a_stream
        return events_to_a2a_stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "A2AModel",
    "A2AServer",
    "AgentCapabilities",
    "AgentCard",
    "AgentInterface",
    "AgentProvider",
    "AgentSkill",
    "Artifact",
    "JSONRPCError",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "Message",
    "MessageSendParams",
    "Part",
    "Role",
    "SendMessageConfiguration",
    "Task",
    "TaskArtifactUpdateEvent",
    "TaskState",
    "TaskStatus",
    "TaskStatusUpdateEvent",
    "data_part",
    "events_to_a2a_stream",
    "file_part",
    "text_part",
]
