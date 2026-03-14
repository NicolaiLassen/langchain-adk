"""Agent event models with Content/Part payloads.

Events use a typed hierarchy with ``Content``-based payloads that align
with Google ADK's Content/Part model and the A2A protocol. Each event
carries a ``content: Content`` field with typed parts (TextPart, DataPart,
FilePart) instead of loose string fields.

Convenience properties (``.text``, ``.data``) provide quick access to the
most common content types without iterating over parts manually.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from langchain_adk.events.event_actions import EventActions
from langchain_adk.models.llm_response import LlmResponse
from langchain_adk.models.part import Content


# ---------------------------------------------------------------------------
# EventType
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """All possible event types emitted during agent execution."""

    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Base Event
# ---------------------------------------------------------------------------


class Event(BaseModel):
    """Base event emitted during agent execution.

    Every event carries a ``content: Content`` field with typed parts.
    Subclasses add convenience properties for their specific use case.

    Attributes
    ----------
    id : str
        Unique identifier for this event.
    type : EventType
        The kind of event.
    timestamp : float
        Unix timestamp of when this event was created.
    invocation_id : str
        ID of the agent invocation that produced this event.
    session_id : str, optional
        The session this event belongs to.
    author : str
        Who produced this event: "user" or the agent name.
    agent_name : str, optional
        Name of the agent that emitted this event.
    branch : str
        Dot-separated path showing which agent in the tree produced this.
    partial : bool, optional
        When True, the event is an incomplete streaming chunk.
    content : Content
        The event payload as typed parts (text, data, files).
    actions : EventActions
        Side-effects to apply when this event is committed to the session.
    metadata : dict[str, Any]
        Arbitrary extra metadata.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: EventType
    timestamp: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp()
    )
    invocation_id: str = ""
    session_id: Optional[str] = None
    author: str = ""
    agent_name: Optional[str] = None
    branch: str = ""
    partial: Optional[bool] = None
    content: Content = Field(default_factory=Content)
    actions: EventActions = Field(default_factory=EventActions)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        """Concatenate all text parts in content."""
        return self.content.text

    @property
    def data(self) -> dict[str, Any] | None:
        """Return the first DataPart's data, or None."""
        return self.content.data

    def is_final_response(self) -> bool:
        """Return True if this event represents the agent's final answer."""
        if self.actions.skip_summarization:
            return True
        if isinstance(self, FinalAnswerEvent):
            return True
        if self.partial:
            return False
        return self.type not in (EventType.TOOL_CALL, EventType.TOOL_RESULT)

    @staticmethod
    def new_id() -> str:
        """Generate a new unique event ID."""
        return str(uuid4())


# ---------------------------------------------------------------------------
# Typed event subclasses
# ---------------------------------------------------------------------------


class ThoughtEvent(Event):
    """Emitted when the agent produces an internal thought.

    Content: TextPart with the thought text.
    Access via ``.text`` property.
    """

    type: EventType = EventType.THOUGHT
    scratchpad: str = ""


class ActionEvent(Event):
    """Emitted when the agent decides on an action to take."""

    type: EventType = EventType.ACTION
    action: str = ""
    action_input: str = ""


class ObservationEvent(Event):
    """Emitted when a tool result is observed.

    Content: TextPart with the observation text.
    Access via ``.text`` property.
    """

    type: EventType = EventType.OBSERVATION
    tool_name: str = ""


class FinalAnswerEvent(Event):
    """Emitted when the agent produces its final answer.

    Content parts:
    - TextPart: the answer text
    - DataPart: structured output (when output_schema is set)

    Access via:
    - ``.text`` → concatenated text parts (the answer)
    - ``.data`` → first DataPart's data dict (structured output)
    """

    type: EventType = EventType.FINAL_ANSWER
    scratchpad: str = ""
    llm_response: Optional[LlmResponse] = None


class ToolCallEvent(Event):
    """Emitted immediately before a tool is invoked."""

    type: EventType = EventType.TOOL_CALL
    tool_name: str = ""
    tool_input: Any = None
    llm_response: Optional[LlmResponse] = None


class ToolResultEvent(Event):
    """Emitted after a tool returns a result.

    Content parts:
    - TextPart/DataPart: the tool's output

    The ``error`` field is set when the tool call failed.
    Access via ``.text`` property for text results, ``.data`` for structured.
    """

    type: EventType = EventType.TOOL_RESULT
    tool_name: str = ""
    error: Optional[str] = None


class ErrorEvent(Event):
    """Emitted when an unrecoverable error occurs during agent execution."""

    type: EventType = EventType.ERROR
    message: str = ""
    exception_type: Optional[str] = None
