"""Agent event models.

Represents discrete events emitted during an agent's execution loop:
thoughts, tool calls, observations, and final answers.

Design
------
A typed event hierarchy (ThoughtEvent, ToolCallEvent, etc.) is used rather
than a single unified Event model, because LangChain does not use
Gemini Content/Part objects. The typed hierarchy is more Pythonic and gives
better IDE support.

LlmResponse
-----------
Events that are direct LLM outputs (FinalAnswerEvent, ToolCallEvent) carry
an optional ``llm_response: LlmResponse`` field. This preserves token usage,
model version, and any extra metadata from the model without coupling the
rest of the event model to LangChain types directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from langchain_adk.events.event_actions import EventActions, EventCompaction
from langchain_adk.models.llm_response import LlmResponse


# ---------------------------------------------------------------------------
# EventType
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """All possible event types emitted during agent execution.

    Attributes
    ----------
    AGENT_START : str
        The agent has started processing a request.
    AGENT_END : str
        The agent has finished processing a request.
    THOUGHT : str
        The agent produced an internal thought (ReAct loop).
    ACTION : str
        The agent decided on an action to take.
    OBSERVATION : str
        The result of a tool call was observed.
    FINAL_ANSWER : str
        The agent produced a final answer.
    TOOL_CALL : str
        A tool was invoked.
    TOOL_RESULT : str
        A tool returned a result.
    ERROR : str
        An error occurred during agent execution.
    """

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

    Every event in the SDK extends this class. Agents yield a stream of
    events; sessions store them via append_event(). The type hierarchy
    (ThoughtEvent, ToolCallEvent, etc.) provides structured payloads on
    top of these shared base fields.

    Attributes
    ----------
    id : str
        Unique identifier for this event. Auto-generated.
    type : EventType
        The kind of event.
    timestamp : float
        Unix timestamp of when this event was created.
    invocation_id : str
        ID of the agent invocation that produced this event. Links all
        events from a single run() call together.
    session_id : str, optional
        The session this event belongs to.
    author : str
        Who produced this event: "user" or the agent name. Mirrors ADK.
    agent_name : str, optional
        Name of the agent that emitted this event. Same value as author
        for agent-produced events; kept for explicit attribution.
    branch : str
        Dot-separated path showing which agent in the tree produced this
        event (e.g. "root.planner.coder"). Enables filtering parallel
        agent streams. Mirrors ADK's Event.branch.
    partial : bool, optional
        When True, the event is an incomplete streaming chunk. Consumers
        should buffer partial events and not treat them as final until a
        non-partial event with the same type arrives.
    content : Any
        Primary payload of the event. Type depends on the event subclass.
    actions : EventActions
        Side-effects to apply when this event is committed to the session.
    metadata : dict[str, Any]
        Arbitrary extra metadata. Not interpreted by the SDK.
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
    content: Any = None
    actions: EventActions = Field(default_factory=EventActions)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_final_response(self) -> bool:
        """Return True if this event represents the agent's final answer.

        Mirrors ADK's Event.is_final_response(). An event is considered
        final when:
        - It is a FinalAnswerEvent, or
        - skip_summarization is set (tool result the agent wants to pass
          through directly), or
        - It is not a tool call or tool result, and is not partial.

        Returns
        -------
        bool
            True if this is the last meaningful event from the agent.
        """
        if self.actions.skip_summarization:
            return True
        if isinstance(self, FinalAnswerEvent):
            return True
        if self.partial:
            return False
        return self.type not in (EventType.TOOL_CALL, EventType.TOOL_RESULT)

    @staticmethod
    def new_id() -> str:
        """Generate a new unique event ID.

        Returns
        -------
        str
            A UUID4 string.
        """
        return str(uuid4())


# ---------------------------------------------------------------------------
# Typed event subclasses
# ---------------------------------------------------------------------------


class ThoughtEvent(Event):
    """Emitted when the agent produces an internal thought.

    Used by ReActAgent to surface explicit reasoning steps before taking
    an action.

    Attributes
    ----------
    thought : str
        The agent's reasoning text for this step.
    scratchpad : str
        Accumulated working notes from prior iterations.
    """

    type: EventType = EventType.THOUGHT
    thought: str
    scratchpad: str = ""


class ActionEvent(Event):
    """Emitted when the agent decides on an action to take.

    Attributes
    ----------
    action : str
        The name of the tool or action to invoke.
    action_input : str
        The input to pass to the action.
    """

    type: EventType = EventType.ACTION
    action: str
    action_input: str


class ObservationEvent(Event):
    """Emitted when a tool result is observed.

    Attributes
    ----------
    observation : str
        The result returned by the tool.
    tool_name : str
        The name of the tool that produced this observation.
    """

    type: EventType = EventType.OBSERVATION
    observation: str
    tool_name: str


class FinalAnswerEvent(Event):
    """Emitted when the agent produces its final answer.

    Always has is_final_response() == True. SequentialAgent passes
    this event's answer as input to the next agent in the pipeline.

    Attributes
    ----------
    answer : str
        The final answer text.
    scratchpad : str
        Accumulated working notes from the full reasoning loop.
    llm_response : LlmResponse, optional
        The wrapped model response that produced this answer. Carries
        token usage and model version without coupling consumers to
        LangChain types.
    """

    type: EventType = EventType.FINAL_ANSWER
    answer: str
    scratchpad: str = ""
    llm_response: Optional[LlmResponse] = None
    structured_output: Any = None


class ToolCallEvent(Event):
    """Emitted immediately before a tool is invoked.

    Attributes
    ----------
    tool_name : str
        The name of the tool being called.
    tool_input : Any
        The arguments passed to the tool.
    llm_response : LlmResponse, optional
        The wrapped model response that triggered this tool call. Carries
        token usage and model version for the LLM turn that produced it.
    """

    type: EventType = EventType.TOOL_CALL
    tool_name: str
    tool_input: Any
    llm_response: Optional[LlmResponse] = None


class ToolResultEvent(Event):
    """Emitted after a tool returns a result.

    Attributes
    ----------
    tool_name : str
        The name of the tool that was called.
    result : Any
        The value returned by the tool on success.
    error : str, optional
        Error message if the tool call failed. Mutually exclusive with result.
    """

    type: EventType = EventType.TOOL_RESULT
    tool_name: str
    result: Any = None
    error: Optional[str] = None


class ErrorEvent(Event):
    """Emitted when an unrecoverable error occurs during agent execution.

    Attributes
    ----------
    message : str
        Human-readable description of the error.
    exception_type : str, optional
        Class name of the exception that was raised.
    """

    type: EventType = EventType.ERROR
    message: str
    exception_type: Optional[str] = None
