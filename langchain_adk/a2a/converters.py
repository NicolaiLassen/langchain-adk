"""Converters between SDK Events and A2A wire format."""

from __future__ import annotations

from langchain_adk.events.event import Event, FinalAnswerEvent
from langchain_adk.a2a.types import A2AEvent


def event_to_a2a(event: Event) -> A2AEvent:
    """Convert an SDK Event to an A2AEvent.

    FinalAnswerEvent sets is_final=True so the client knows the stream is done.

    Parameters
    ----------
    event : Event
        The SDK event to convert.

    Returns
    -------
    A2AEvent
        The wire-format event ready for streaming.
    """
    is_final = isinstance(event, FinalAnswerEvent)

    content: object = event.content
    if isinstance(event, FinalAnswerEvent):
        content = event.answer

    return A2AEvent(
        id=event.id,
        type=event.type.value,
        agent_name=event.agent_name,
        content=content,
        is_final=is_final,
        metadata=event.metadata,
    )
