"""Session event compaction — LLM-based summarization of old events.

When session history grows beyond a configured threshold, old events
are summarized into a single compaction event using an LLM call.
The summary replaces the original events, keeping the context window
manageable across long conversations.

Approach (sliding window with retention):
  1. Count session events.  If under ``max_events``, do nothing.
  2. Keep the last ``retention_count`` events as raw (never compact).
  3. Summarize the older events using the LLM.
  4. Replace old events with a single compaction event carrying the
     summary.  The LlmAgent's ``_process_compaction`` reads this
     when building the LLM context.

Example::

    config = CompactionConfig(max_events=50, retention_count=20)
    runner = Runner(agent=my_agent, ..., compaction_config=config)

The runner will automatically compact after each invocation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from orxhestra.events.event import Event, EventType
from orxhestra.events.event_actions import EventActions, EventCompaction
from orxhestra.models.part import Content

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from orxhestra.sessions.base_session_service import BaseSessionService
    from orxhestra.sessions.session import Session

logger = logging.getLogger(__name__)


@dataclass
class CompactionConfig:
    """Configuration for automatic session compaction.

    Attributes
    ----------
    max_events : int
        Compact when session has more than this many events.
    retention_count : int
        Always keep the last N events as raw (uncompacted).
    llm : BaseChatModel, optional
        LLM to use for summarization.  If ``None``, the compactor
        will use a simple text-based extraction instead of an LLM call.
    """

    max_events: int = 50
    retention_count: int = 20
    llm: BaseChatModel | None = field(default=None, repr=False)


_SUMMARIZE_PROMPT = """\
Summarize the following conversation events into a concise prose summary.
Preserve key facts, decisions, tool results, and any important data.
Remove redundant or irrelevant information.
Keep the summary under 500 words.

Events:
{events_text}

Summary:"""


def _events_to_text(events: list[Event]) -> str:
    """Convert events to a readable text block for the summarizer."""
    lines: list[str] = []
    for event in events:
        prefix = f"[{event.type.value}]"
        if event.agent_name:
            prefix += f" ({event.agent_name})"

        if event.text:
            lines.append(f"{prefix}: {event.text[:500]}")
        elif event.has_tool_calls:
            for tc in event.tool_calls:
                lines.append(f"{prefix} tool_call: {tc.tool_name}({tc.args})")
        elif event.type == EventType.TOOL_RESPONSE:
            for tr in event.content.tool_responses:
                result = tr.result[:200] if tr.result else ""
                lines.append(f"{prefix} tool_response: {tr.tool_name} → {result}")

    return "\n".join(lines)


async def compact_session(
    session: Session,
    session_service: BaseSessionService,
    config: CompactionConfig,
) -> bool:
    """Compact old session events if the session exceeds the threshold.

    Parameters
    ----------
    session : Session
        The session to potentially compact.
    session_service : BaseSessionService
        Service to persist the compaction event.
    config : CompactionConfig
        Compaction configuration.

    Returns
    -------
    bool
        True if compaction was performed, False otherwise.
    """
    events = session.events
    if len(events) <= config.max_events:
        return False

    # Split: old events to compact vs. recent events to keep raw
    split_idx = len(events) - config.retention_count
    if split_idx <= 0:
        return False

    old_events = events[:split_idx]
    if not old_events:
        return False

    # Skip if there are pending tool calls in the old events
    # (never compact events with unresolved tool calls)
    responded_ids: set[str] = set()
    for e in events:
        if e.type == EventType.TOOL_RESPONSE:
            for tr in e.content.tool_responses:
                if tr.tool_call_id:
                    responded_ids.add(tr.tool_call_id)

    for e in old_events:
        if e.has_tool_calls:
            for tc in e.tool_calls:
                if tc.tool_call_id and tc.tool_call_id not in responded_ids:
                    logger.debug("Skipping compaction: pending tool call %s", tc.tool_call_id)
                    return False

    # Generate summary
    events_text = _events_to_text(old_events)
    if not events_text.strip():
        return False

    if config.llm is not None:
        # LLM-based summarization
        prompt = _SUMMARIZE_PROMPT.format(events_text=events_text)
        try:
            response = await config.llm.ainvoke(prompt)
            summary = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.warning("Compaction LLM call failed: %s", exc)
            return False
    else:
        # Fallback: simple text extraction (no LLM)
        summary = events_text[:2000]

    # Create compaction event
    start_ts = old_events[0].timestamp
    end_ts = old_events[-1].timestamp

    compaction_event = Event(
        type=EventType.AGENT_MESSAGE,
        session_id=session.id,
        agent_name="compaction",
        content=Content.from_text(summary),
        actions=EventActions(
            compaction=EventCompaction(
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                summary=summary,
                event_count=len(old_events),
            ),
        ),
    )

    # Replace old events with compaction event
    session.events = [compaction_event, *events[split_idx:]]

    logger.info(
        "Compacted %d events into summary (%d chars). "
        "Session now has %d events.",
        len(old_events),
        len(summary),
        len(session.events),
    )

    return True
