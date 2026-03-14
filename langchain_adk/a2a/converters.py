"""Converters between SDK Events and A2A streaming events."""

from __future__ import annotations

from typing import AsyncIterator, Union

from langchain_adk.a2a.types import (
    Artifact,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from langchain_adk.events.event import (
    Event,
    FinalAnswerEvent,
    ToolCallEvent,
    ToolResultEvent,
)


async def events_to_a2a_stream(
    events: AsyncIterator[Event],
    *,
    task_id: str,
    context_id: str,
) -> AsyncIterator[Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent]]:
    """Convert SDK event stream into A2A-compliant streaming events.

    Yields TaskStatusUpdateEvent for status changes and
    TaskArtifactUpdateEvent for content (final answer, tool results).
    """
    async for event in events:
        if isinstance(event, FinalAnswerEvent):
            if event.partial:
                continue
            # Emit artifact with the final answer
            artifact = Artifact(
                parts=[TextPart(text=event.text)],
                name="answer",
            )
            yield TaskArtifactUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                artifact=artifact,
                last_chunk=True,
            )

        elif isinstance(event, ToolCallEvent):
            yield TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.working,
                    message=None,
                ),
                final=False,
                metadata={"tool_name": event.tool_name, "tool_input": event.tool_input},
            )

        elif isinstance(event, ToolResultEvent):
            result_text = event.text or (event.error or "")
            artifact = Artifact(
                parts=[TextPart(text=result_text)],
                name=f"tool_result:{event.tool_name}",
            )
            yield TaskArtifactUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                artifact=artifact,
                last_chunk=True,
            )
