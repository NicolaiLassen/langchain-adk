"""A2A wire-format models.

Minimal Pydantic models for the Agent-to-Agent (A2A) protocol.
The server accepts A2ARequest and streams A2AEvent responses.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class A2AMessage(BaseModel):
    """A single message in the A2A protocol.

    Attributes
    ----------
    role : str
        The message role (e.g. "user" or "agent").
    content : str
        The message content.
    metadata : dict[str, Any]
        Optional metadata associated with the message.
    """

    role: str = "user"
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2ARequest(BaseModel):
    """Inbound A2A request sent to an agent endpoint.

    Attributes
    ----------
    id : str
        Unique request identifier.
    message : str
        The user's message or task.
    session_id : str, optional
        Session to resume, or None to start a new session.
    user_id : str
        The user sending the request.
    metadata : dict[str, Any]
        Optional request metadata.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    message: str
    session_id: Optional[str] = None
    user_id: str = "anonymous"
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2AEvent(BaseModel):
    """One streamed event in the A2A response stream.

    Attributes
    ----------
    id : str
        Unique event identifier.
    type : str
        Event type value (mirrors EventType).
    agent_name : str, optional
        Name of the agent that emitted this event.
    content : Any
        Event payload.
    is_final : bool
        True if this is the last event in the stream.
    metadata : dict[str, Any]
        Optional event metadata.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str          # mirrors EventType value
    agent_name: Optional[str] = None
    content: Any = None
    is_final: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
