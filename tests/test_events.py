"""Tests for event models."""

import pytest
from langchain_adk.events.event import (
    Event,
    EventActions,
    EventType,
    ThoughtEvent,
    ActionEvent,
    ObservationEvent,
    FinalAnswerEvent,
    ToolCallEvent,
    ToolResultEvent,
    ErrorEvent,
)


def test_event_actions_defaults():
    actions = EventActions()
    assert actions.state_delta == {}
    assert actions.transfer_to_agent is None
    assert actions.escalate is None
    assert actions.skip_summarization is None


def test_event_actions_escalate():
    actions = EventActions(escalate=True)
    assert actions.escalate is True


def test_event_actions_state_delta():
    actions = EventActions(state_delta={"key": "value"})
    assert actions.state_delta["key"] == "value"


def test_event_base_fields():
    event = Event(type=EventType.AGENT_START, session_id="s1", agent_name="agent")
    assert event.type == EventType.AGENT_START
    assert event.session_id == "s1"
    assert event.agent_name == "agent"
    assert event.id is not None
    assert event.timestamp is not None


def test_final_answer_event():
    event = FinalAnswerEvent(session_id="s1", agent_name="agent", answer="42")
    assert event.type == EventType.FINAL_ANSWER
    assert event.answer == "42"
    assert event.scratchpad == ""


def test_thought_event():
    event = ThoughtEvent(session_id="s1", agent_name="agent", thought="thinking...")
    assert event.type == EventType.THOUGHT
    assert event.thought == "thinking..."


def test_tool_call_event():
    event = ToolCallEvent(
        session_id="s1", agent_name="agent", tool_name="search", tool_input={"q": "test"}
    )
    assert event.type == EventType.TOOL_CALL
    assert event.tool_name == "search"


def test_tool_result_event_with_error():
    event = ToolResultEvent(
        session_id="s1", agent_name="agent", tool_name="search", error="timeout"
    )
    assert event.error == "timeout"
    assert event.result is None


def test_error_event():
    event = ErrorEvent(session_id="s1", agent_name="agent", message="something failed")
    assert event.type == EventType.ERROR
    assert event.message == "something failed"


def test_event_unique_ids():
    e1 = Event(type=EventType.AGENT_START)
    e2 = Event(type=EventType.AGENT_START)
    assert e1.id != e2.id
