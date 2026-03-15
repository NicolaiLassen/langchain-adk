"""A2A (Agent-to-Agent) protocol support.

Requires the ``a2a`` extra: ``pip install langchain-adk[a2a]``
"""

from langchain_adk.a2a.types import (
    A2AModel,
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
    Artifact,
    DataPart,
    FilePart,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
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
    "AgentCard",
    "AgentCapabilities",
    "AgentSkill",
    "AgentProvider",
    "Artifact",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "JSONRPCError",
    "Message",
    "MessageSendParams",
    "Part",
    "Role",
    "Task",
    "TaskState",
    "TaskStatus",
    "TaskStatusUpdateEvent",
    "TaskArtifactUpdateEvent",
    "TextPart",
    "FilePart",
    "DataPart",
    "events_to_a2a_stream",
]
