"""Reusable Textual widgets for the orxhestra TUI."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class UserMessage(Static):
    """A user message bubble."""

    DEFAULT_CSS = """
    UserMessage {
        color: $text;
        padding: 0 1;
        margin: 1 0 0 0;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__(f"[bold]> {text}[/bold]")


class ToolCallWidget(Static):
    """Renders a tool call with box-drawing characters."""

    DEFAULT_CSS = """
    ToolCallWidget {
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, tool_name: str, summary: str = "") -> None:
        content = f"  \u250c {tool_name}"
        if summary:
            content += f"\n  \u2502 {summary}"
        super().__init__(content)


class ToolResponseWidget(Static):
    """Renders a tool response with elapsed time."""

    DEFAULT_CSS = """
    ToolResponseWidget {
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, text: str = "", elapsed: float | None = None) -> None:
        elapsed_str = f" ({elapsed:.1f}s)" if elapsed and elapsed >= 0.1 else ""
        if text:
            display = text[:200]
            super().__init__(f"  \u2514 {display}{elapsed_str}")
        else:
            super().__init__(f"  \u2514 done{elapsed_str}")


class ThinkingWidget(Static):
    """Dim italic thinking/reasoning text."""

    DEFAULT_CSS = """
    ThinkingWidget {
        color: $text-muted;
        text-style: italic;
        padding: 0 1;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__(f"  [dim italic]{text}[/dim italic]")


class TurnFooter(Static):
    """Turn summary with elapsed time and token counts."""

    DEFAULT_CSS = """
    TurnFooter {
        color: $text-muted;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(
        self,
        elapsed: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        total = input_tokens + output_tokens
        parts = [f"  \u27e1 {elapsed:.1f}s"]
        if total:
            parts.append(f"{total:,} tokens ({input_tokens:,}\u2191 {output_tokens:,}\u2193)")
        import time

        parts.append(time.strftime("%H:%M"))
        super().__init__(" \u00b7 ".join(parts))


class ApprovalModal(ModalScreen[str]):
    """Modal dialog for tool approval.

    Returns ``"y"``, ``"n"``, or ``"a"`` (approve all).
    """

    DEFAULT_CSS = """
    ApprovalModal {
        align: center middle;
    }
    #approval-dialog {
        width: 60;
        height: auto;
        border: thick $accent;
        padding: 1 2;
    }
    #approval-buttons {
        margin-top: 1;
        height: 3;
    }
    #approval-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(self, tool_name: str, summary: str) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.summary = summary

    def compose(self) -> ComposeResult:
        with Static(id="approval-dialog"):
            yield Static(f"[bold]Allow:[/bold] {self.summary}")
            with Horizontal(id="approval-buttons"):
                yield Button("[y]es", id="y", variant="primary")
                yield Button("[n]o", id="n", variant="error")
                yield Button("approve [a]ll", id="a", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id)

    def key_y(self) -> None:
        self.dismiss("y")

    def key_n(self) -> None:
        self.dismiss("n")

    def key_a(self) -> None:
        self.dismiss("a")
