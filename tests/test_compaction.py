"""Tests for session event compaction (character-based thresholds)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orxhestra.events.event import Event, EventType
from orxhestra.events.event_actions import EventActions
from orxhestra.events.filters import apply_compaction
from orxhestra.models.part import Content, ToolCallPart, ToolResponsePart
from orxhestra.sessions.compaction import (
    CompactionConfig,
    _estimate_event_chars,
    _events_to_text,
    _find_compaction_boundary,
    compact_session,
)
from orxhestra.sessions.in_memory_session_service import InMemorySessionService
from orxhestra.sessions.session import Session


def _make_event(
    text: str = "hello",
    *,
    event_type: EventType = EventType.AGENT_MESSAGE,
    agent_name: str = "bot",
    session_id: str = "s1",
    timestamp: float | None = None,
    content: Content | None = None,
    actions: EventActions | None = None,
) -> Event:
    """Create a simple event for testing."""
    return Event(
        type=event_type,
        agent_name=agent_name,
        session_id=session_id,
        content=content or Content.from_text(text),
        actions=actions or EventActions(),
        **({"timestamp": timestamp} if timestamp is not None else {}),
    )


def _make_session(
    event_count: int,
    chars_per_event: int = 100,
    session_id: str = "s1",
) -> Session:
    """Create a session with N events, each containing chars_per_event characters."""
    events = [
        _make_event("x" * chars_per_event, timestamp=float(i))
        for i in range(event_count)
    ]
    return Session(id=session_id, app_name="app", user_id="u1", events=events)


def _find_compaction_event(session: Session) -> Event:
    """Return the compaction event from the session."""
    for e in session.events:
        if e.actions.compaction is not None:
            return e
    raise AssertionError("No compaction event found in session")


# --- CompactionConfig ---


def test_compaction_config_defaults() -> None:
    config = CompactionConfig()
    assert config.char_threshold == 100_000
    assert config.retention_chars == 20_000
    assert config.llm is None


def test_compaction_config_custom_values() -> None:
    config = CompactionConfig(char_threshold=50_000, retention_chars=10_000)
    assert config.char_threshold == 50_000
    assert config.retention_chars == 10_000


# --- _estimate_event_chars ---


def test_estimate_event_chars_text() -> None:
    event = _make_event("hello world")
    assert _estimate_event_chars(event) == 11


def test_estimate_event_chars_tool_response() -> None:
    tr = ToolResponsePart(tool_call_id="tc1", tool_name="search", result="found it")
    event = _make_event(
        content=Content(parts=[tr]),
        event_type=EventType.TOOL_RESPONSE,
    )
    assert _estimate_event_chars(event) == len("found it")


# --- _events_to_text ---


def test_events_to_text_with_text_events() -> None:
    events = [_make_event("hello"), _make_event("world")]
    result = _events_to_text(events)
    assert "[agent_message] (bot): hello" in result
    assert "[agent_message] (bot): world" in result


def test_events_to_text_with_tool_call() -> None:
    tc = ToolCallPart(tool_call_id="tc1", tool_name="search", args={"q": "test"})
    event = _make_event(content=Content(parts=[tc]))
    result = _events_to_text([event])
    assert "tool_call: search" in result


def test_events_to_text_with_tool_response() -> None:
    tr = ToolResponsePart(tool_call_id="tc1", tool_name="search", result="found it")
    event = _make_event(
        content=Content(parts=[tr]),
        event_type=EventType.TOOL_RESPONSE,
    )
    result = _events_to_text([event])
    assert "tool_response: search" in result
    assert "found it" in result


def test_events_to_text_truncates_long_text() -> None:
    long_text = "x" * 1000
    events = [_make_event(long_text)]
    result = _events_to_text(events)
    assert len(result) < 1000


def test_events_to_text_empty_list() -> None:
    assert _events_to_text([]) == ""


def test_events_to_text_no_agent_name() -> None:
    event = _make_event("hi", agent_name=None)
    result = _events_to_text([event])
    assert "[agent_message]:" in result
    assert "(None)" not in result


# --- _find_compaction_boundary ---


def test_find_compaction_boundary_no_compactions() -> None:
    events = [_make_event("a"), _make_event("b")]
    assert _find_compaction_boundary(events) == -1.0


def test_find_compaction_boundary_with_compaction() -> None:
    from orxhestra.events.event_actions import EventCompaction

    compaction_event = _make_event(
        "summary",
        actions=EventActions(
            compaction=EventCompaction(
                start_timestamp=0.0,
                end_timestamp=39.0,
                summary="summary",
                event_count=40,
            ),
        ),
    )
    events = [compaction_event, _make_event("after")]
    assert _find_compaction_boundary(events) == 39.0


# --- compact_session (character-based, non-destructive) ---


@pytest.mark.asyncio
async def test_no_compaction_under_threshold() -> None:
    """10 events * 100 chars = 1000 chars, well under 5000 threshold."""
    session = _make_session(10, chars_per_event=100)
    config = CompactionConfig(char_threshold=5000, retention_chars=1000)

    result = await compact_session(session, InMemorySessionService(), config)

    assert result is False
    assert len(session.events) == 10


@pytest.mark.asyncio
async def test_no_compaction_at_exact_threshold() -> None:
    """50 events * 100 chars = 5000 chars, exactly at threshold (not exceeded)."""
    session = _make_session(50, chars_per_event=100)
    config = CompactionConfig(char_threshold=5000, retention_chars=1000)

    result = await compact_session(session, InMemorySessionService(), config)

    assert result is False
    assert len(session.events) == 50


@pytest.mark.asyncio
async def test_compaction_above_threshold_no_llm() -> None:
    """60 events * 100 chars = 6000 chars, above 5000 threshold."""
    session = _make_session(60, chars_per_event=100)
    original_count = len(session.events)
    config = CompactionConfig(char_threshold=5000, retention_chars=1000)

    result = await compact_session(session, InMemorySessionService(), config)

    assert result is True
    # Non-destructive: all original events + 1 appended compaction event
    assert len(session.events) == original_count + 1
    compaction_event = _find_compaction_event(session)
    assert compaction_event.agent_name == "compaction"
    assert compaction_event.actions.compaction.event_count > 0


@pytest.mark.asyncio
async def test_compaction_preserves_all_original_events() -> None:
    session = _make_session(60, chars_per_event=100)
    original_ids = [e.id for e in session.events]
    config = CompactionConfig(char_threshold=5000, retention_chars=1000)

    await compact_session(session, InMemorySessionService(), config)

    current_ids = [e.id for e in session.events if e.actions.compaction is None]
    assert current_ids == original_ids


@pytest.mark.asyncio
async def test_compaction_event_appended_at_end() -> None:
    session = _make_session(60, chars_per_event=100)
    config = CompactionConfig(char_threshold=5000, retention_chars=1000)

    await compact_session(session, InMemorySessionService(), config)

    assert session.events[-1].actions.compaction is not None


@pytest.mark.asyncio
async def test_compaction_event_has_correct_timestamps() -> None:
    session = _make_session(60, chars_per_event=100)
    config = CompactionConfig(char_threshold=5000, retention_chars=1000)

    await compact_session(session, InMemorySessionService(), config)

    compaction = _find_compaction_event(session).actions.compaction
    assert compaction.start_timestamp == 0.0
    # End timestamp is before the retained window
    assert compaction.end_timestamp < 60.0


@pytest.mark.asyncio
async def test_compaction_event_summary_matches_content() -> None:
    session = _make_session(60, chars_per_event=100)
    config = CompactionConfig(char_threshold=5000, retention_chars=1000)

    await compact_session(session, InMemorySessionService(), config)

    compaction_event = _find_compaction_event(session)
    assert compaction_event.text == compaction_event.actions.compaction.summary


@pytest.mark.asyncio
async def test_compaction_with_llm() -> None:
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "This is a summary of the conversation."
    mock_llm.ainvoke.return_value = mock_response

    session = _make_session(60, chars_per_event=100)
    config = CompactionConfig(char_threshold=5000, retention_chars=1000, llm=mock_llm)

    result = await compact_session(session, InMemorySessionService(), config)

    assert result is True
    mock_llm.ainvoke.assert_called_once()
    compaction_event = _find_compaction_event(session)
    assert compaction_event.text == "This is a summary of the conversation."


@pytest.mark.asyncio
async def test_compaction_llm_failure_returns_false() -> None:
    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = RuntimeError("LLM unavailable")

    session = _make_session(60, chars_per_event=100)
    original_count = len(session.events)
    config = CompactionConfig(char_threshold=5000, retention_chars=1000, llm=mock_llm)

    result = await compact_session(session, InMemorySessionService(), config)

    assert result is False
    assert len(session.events) == original_count


@pytest.mark.asyncio
async def test_no_compaction_when_all_fits_in_retention() -> None:
    """All content fits in retention_chars — nothing to compact."""
    session = _make_session(20, chars_per_event=100)
    # 20 * 100 = 2000 chars total, but retention is 5000 — all retained
    config = CompactionConfig(char_threshold=1000, retention_chars=5000)

    result = await compact_session(session, InMemorySessionService(), config)

    assert result is False


@pytest.mark.asyncio
async def test_no_compaction_with_pending_tool_calls() -> None:
    """Events with unresolved tool calls should not be compacted."""
    events = [
        _make_event("x" * 100, timestamp=float(i))
        for i in range(40)
    ]
    tc = ToolCallPart(tool_call_id="pending-tc", tool_name="search", args={})
    events[10] = _make_event(content=Content(parts=[tc]), timestamp=10.0)
    events.extend(
        _make_event("x" * 100, timestamp=float(40 + i))
        for i in range(20)
    )
    session = Session(id="s1", app_name="app", user_id="u1", events=events)
    config = CompactionConfig(char_threshold=5000, retention_chars=1000)

    result = await compact_session(session, InMemorySessionService(), config)

    assert result is False


@pytest.mark.asyncio
async def test_compaction_allows_resolved_tool_calls() -> None:
    """Tool calls with matching responses should be compactable."""
    events = [
        _make_event("x" * 100, timestamp=float(i))
        for i in range(38)
    ]
    tc = ToolCallPart(tool_call_id="resolved-tc", tool_name="search", args={})
    events.append(_make_event(content=Content(parts=[tc]), timestamp=38.0))
    tr = ToolResponsePart(tool_call_id="resolved-tc", tool_name="search", result="ok")
    events.append(
        _make_event(
            content=Content(parts=[tr]),
            event_type=EventType.TOOL_RESPONSE,
            timestamp=39.0,
        )
    )
    events.extend(
        _make_event("x" * 100, timestamp=float(40 + i))
        for i in range(20)
    )
    session = Session(id="s1", app_name="app", user_id="u1", events=events)
    config = CompactionConfig(char_threshold=5000, retention_chars=1000)

    result = await compact_session(session, InMemorySessionService(), config)

    assert result is True
    assert len(session.events) == 61  # 60 originals + 1 compaction


@pytest.mark.asyncio
async def test_compaction_fallback_truncates_at_2000_chars() -> None:
    """Without an LLM, fallback summary is truncated to 2000 chars."""
    events = [
        _make_event("x" * 200, timestamp=float(i))
        for i in range(80)
    ]
    session = Session(id="s1", app_name="app", user_id="u1", events=events)
    config = CompactionConfig(char_threshold=5000, retention_chars=1000)

    await compact_session(session, InMemorySessionService(), config)

    summary = _find_compaction_event(session).actions.compaction.summary
    assert len(summary) <= 2000


@pytest.mark.asyncio
async def test_repeated_compaction_is_idempotent() -> None:
    """Second compaction should be skipped — remaining content under threshold."""
    session = _make_session(60, chars_per_event=100)
    service = InMemorySessionService()
    config = CompactionConfig(char_threshold=5000, retention_chars=1000)

    first = await compact_session(session, service, config)
    assert first is True
    count_after_first = len(session.events)

    second = await compact_session(session, service, config)
    assert second is False
    assert len(session.events) == count_after_first


# --- apply_compaction integration ---


@pytest.mark.asyncio
async def test_apply_compaction_filters_compacted_events() -> None:
    """Verify the view layer hides events covered by the compaction range."""
    session = _make_session(60, chars_per_event=100)
    config = CompactionConfig(char_threshold=5000, retention_chars=1000)

    await compact_session(session, InMemorySessionService(), config)

    filtered = apply_compaction(session.events)

    # Should have compaction summary + retained recent events
    summary_events = [e for e in filtered if "[Compacted summary" in e.text]
    assert len(summary_events) == 1

    # None of the compacted events should be in filtered
    compaction = _find_compaction_event(session).actions.compaction
    for e in filtered:
        if "[Compacted summary" not in e.text:
            assert e.timestamp > compaction.end_timestamp


@pytest.mark.asyncio
async def test_apply_compaction_after_multiple_compactions() -> None:
    """Multiple compaction rounds should produce correct filtered view."""
    events = [
        _make_event("x" * 200, timestamp=float(i))
        for i in range(120)
    ]
    session = Session(id="s1", app_name="app", user_id="u1", events=events)
    service = InMemorySessionService()
    config = CompactionConfig(char_threshold=5000, retention_chars=2000)

    first = await compact_session(session, service, config)
    assert first is True

    # Simulate more events arriving
    for i in range(120, 180):
        session.events.append(_make_event("x" * 200, timestamp=float(i)))

    second = await compact_session(session, service, config)
    assert second is True

    filtered = apply_compaction(session.events)
    summaries = [e for e in filtered if "[Compacted summary" in e.text]
    assert len(summaries) == 2
