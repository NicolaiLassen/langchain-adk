"""Textual TUI stream handler for orxhestra agents.

Maps ``runner.astream()`` events to Textual widget updates.
This is the rendering layer — NOT a full app. The app shell
is provided by consumers (e.g. orxhestra-code's ``tui_app.py``).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static

from orxhestra.cli.widgets import (
    ThinkingWidget,
    ToolCallWidget,
    ToolResponseWidget,
    TurnFooter,
)
from orxhestra.events.event import EventType

if TYPE_CHECKING:
    from orxhestra.runner import Runner


def _tool_arg_summary(tool_name: str, args: dict[str, Any]) -> str:
    """Build a one-line summary of tool arguments."""
    if tool_name in ("shell_exec", "shell_exec_background"):
        cmd = args.get("command", "")
        return cmd[:100] + ("..." if len(cmd) > 100 else "")
    if tool_name in ("read_file", "write_file", "edit_file"):
        return args.get("path", "")
    if tool_name == "glob":
        return args.get("pattern", "")
    if tool_name == "grep":
        return f"{args.get('pattern', '')} in {args.get('path', '.')}"
    if tool_name == "web_search":
        return args.get("query", "")[:80]
    if tool_name == "web_fetch":
        return args.get("url", "")[:80]
    return ", ".join(f"{k}={v}" for k, v in list(args.items())[:2])


async def stream_to_widgets(
    runner: Runner,
    session_id: str,
    message: str,
    container: VerticalScroll,
    *,
    on_approval: Callable[[str, dict[str, Any]], Awaitable[str]] | None = None,
    auto_approve: bool = False,
    todo_list: Any | None = None,
) -> dict[str, Any]:
    """Stream an agent response into Textual widgets.

    Parameters
    ----------
    runner : Runner
        The orxhestra runner.
    session_id : str
        Active session ID.
    message : str
        User message to send.
    container : VerticalScroll
        Textual container to mount widgets into.
    on_approval : callable, optional
        Async callback ``(tool_name, args) -> "y"|"n"|"a"`` for approval.
    auto_approve : bool
        Skip approval prompts.
    todo_list : Any, optional
        Todo list to render.

    Returns
    -------
    dict
        Turn metadata: ``elapsed``, ``input_tokens``, ``output_tokens``.
    """
    turn_start = time.monotonic()
    input_tokens = 0
    output_tokens = 0
    md_widget: Markdown | None = None
    md_stream: Any = None
    thinking_active = False
    pending_tool_ids: set[str] = set()
    tool_start = 0.0

    container.anchor()

    async for event in runner.astream(
        user_id="user",
        session_id=session_id,
        new_message=message,
    ):
        # Track token usage.
        if event.llm_response:
            input_tokens += event.llm_response.input_tokens or 0
            output_tokens += event.llm_response.output_tokens or 0

        # Compacting notification.
        if event.metadata.get("compacting"):
            await container.mount(Static("  [dim]Compacting context...[/dim]"))
            continue
        if event.metadata.get("compaction"):
            await container.mount(Static("  [dim]context compacted[/dim]"))
            continue

        # Thinking / reasoning.
        if event.partial and event.type == EventType.AGENT_MESSAGE and event.thinking:
            if not thinking_active:
                thinking_active = True
            await container.mount(ThinkingWidget(event.thinking))
            continue

        # Streaming text.
        if event.partial and event.type == EventType.AGENT_MESSAGE and event.text:
            if thinking_active:
                thinking_active = False
            if md_widget is None:
                md_widget = Markdown("")
                await container.mount(md_widget)
                md_stream = Markdown.get_stream(md_widget)
            await md_stream.write(event.text)
            continue

        # Tool calls.
        if event.has_tool_calls:
            # Stop any active markdown stream.
            if md_stream is not None:
                await md_stream.stop()
                md_stream = None
                md_widget = None

            for tc in event.tool_calls:
                pending_tool_ids.add(tc.tool_call_id)
                summary = _tool_arg_summary(tc.tool_name, tc.args or {})
                await container.mount(ToolCallWidget(tc.tool_name, summary))

            tool_start = time.monotonic()
            continue

        # Tool responses.
        if event.type == EventType.TOOL_RESPONSE:
            tool_call_id = ""
            if event.content.tool_responses:
                tool_call_id = event.content.tool_responses[0].tool_call_id
            pending_tool_ids.discard(tool_call_id)

            # Only show response for last tool in a batch.
            if not pending_tool_ids:
                elapsed = time.monotonic() - tool_start if tool_start else None
                text = (event.text or "")[:200]
                await container.mount(ToolResponseWidget(text, elapsed))
            continue

        # Final response (non-streaming).
        if event.is_final_response():
            if md_stream is not None:
                await md_stream.stop()
                md_stream = None
                md_widget = None
            if not md_widget and event.text:
                md_widget = Markdown(event.text)
                await container.mount(md_widget)
            continue

        # Errors.
        if event.metadata.get("error") and event.text:
            if md_stream is not None:
                await md_stream.stop()
                md_stream = None
                md_widget = None
            await container.mount(
                Static(f"[bold red]Error:[/bold red] {event.text}")
            )
            continue

    # Ensure stream is stopped.
    if md_stream is not None:
        await md_stream.stop()

    elapsed = time.monotonic() - turn_start
    await container.mount(TurnFooter(elapsed, input_tokens, output_tokens))

    return {
        "elapsed": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
