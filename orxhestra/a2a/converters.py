"""Converters between SDK Events and A2A v1.0 streaming events."""

from __future__ import annotations

from collections.abc import AsyncIterator

from orxhestra.a2a.types import (
    Artifact,
    Part,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from orxhestra.events.event import Event, EventType


async def events_to_a2a_stream(
    events: AsyncIterator[Event],
    *,
    task_id: str,
    context_id: str,
) -> AsyncIterator[TaskStatusUpdateEvent | TaskArtifactUpdateEvent]:
    """Convert SDK event stream into A2A v1.0 streaming events.

    Yields TaskStatusUpdateEvent for status changes and
    TaskArtifactUpdateEvent for content (final answer, tool results).
    """
    async for event in events:
        if event.is_final_response():
            artifact = Artifact(
                parts=[Part(text=event.text, media_type="text/plain")],
                name="answer",
            )
            yield TaskArtifactUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                artifact=artifact,
                last_chunk=True,
            )

        elif event.has_tool_calls:
            tc = event.tool_calls[0]
            yield TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.WORKING,
                    message=None,
                ),
                final=False,
                metadata={"tool_name": tc.tool_name, "tool_input": tc.args},
            )

        elif event.type == EventType.TOOL_RESPONSE:
            responses = event.content.tool_responses
            if responses:
                result_text = responses[0].result or (responses[0].error or "")
                tool_name = responses[0].tool_name
            else:
                result_text = event.text
                tool_name = "unknown"
            artifact = Artifact(
                parts=[Part(text=result_text, media_type="text/plain")],
                name=f"tool_result:{tool_name}",
            )
            yield TaskArtifactUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                artifact=artifact,
                last_chunk=True,
            )
