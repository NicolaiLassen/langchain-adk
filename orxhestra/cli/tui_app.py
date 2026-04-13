"""Base Textual TUI app for orxhestra agents.

Provides a chat-style terminal UI with always-on input,
streaming Markdown output, and tool approval modals.
Used by the ``orx`` CLI directly and extended by ``orx-coder``.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Static
from textual.work import work

from orxhestra.cli.tui import stream_to_widgets
from orxhestra.cli.widgets import ApprovalModal, UserMessage


class OrxTuiApp(App):
    """Base Textual app for orxhestra agent interaction."""

    TITLE = "orx"

    CSS = """
    Screen {
        layout: vertical;
    }
    #messages {
        height: 1fr;
        scrollbar-size: 1 1;
    }
    #user-input {
        dock: bottom;
        height: 3;
        border-top: solid $accent;
    }
    .status-msg {
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Interrupt", show=True),
        Binding("ctrl+d", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_session", "Clear", show=True),
    ]

    AUTO_FOCUS = "#user-input"

    def __init__(
        self,
        runner: Any,
        session_id: str,
        model_name: str = "",
        auto_approve: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.runner = runner
        self.session_id = session_id
        self.model_name = model_name
        self.auto_approve = auto_approve
        self.turn_count = 0
        self._cumulative_input = 0
        self._cumulative_output = 0
        self._agent_running = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="messages")
        yield Input(id="user-input", placeholder=self._prompt_text())
        yield Footer()

    def _prompt_text(self) -> str:
        return "orx> "

    def _update_prompt(self) -> None:
        inp = self.query_one("#user-input", Input)
        inp.placeholder = self._prompt_text()

    # ── Input handling ───────────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()

        if not text:
            return

        if text.startswith("/"):
            await self._handle_slash(text)
            return

        if self._agent_running:
            self._add_status("Agent is still running. Wait or press Ctrl+C.")
            return

        self._run_agent(text)

    # ── Agent execution ──────────────────────────────────────────

    @work
    async def _run_agent(self, message: str) -> None:
        container = self.query_one("#messages", VerticalScroll)
        await container.mount(UserMessage(message))
        container.anchor()

        self._agent_running = True
        try:
            result = await stream_to_widgets(
                self.runner,
                self.session_id,
                message,
                container,
                on_approval=self._handle_approval,
                auto_approve=self.auto_approve,
            )
            self.turn_count += 1
            self._cumulative_input += result.get("input_tokens", 0)
            self._cumulative_output += result.get("output_tokens", 0)
        except Exception as exc:
            await container.mount(
                Static(f"[bold red]Error:[/bold red] {exc}", classes="status-msg")
            )
        finally:
            self._agent_running = False

    # ── Tool approval (override in subclass for permission modes) ─

    async def _handle_approval(self, tool_name: str, tool_args: dict) -> str:
        if self.auto_approve:
            return "y"
        summary = f"{tool_name}: {str(tool_args)[:100]}"
        result = await self.push_screen_wait(ApprovalModal(tool_name, summary))
        if result == "a":
            self.auto_approve = True
            return "y"
        return result

    # ── Slash commands ───────────────────────────────────────────

    async def _handle_slash(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else None

        if cmd in ("/help", "/h"):
            self._add_status(
                "/clear · /cost · /model · /exit · /help"
            )
        elif cmd in ("/exit", "/quit", "/q"):
            self.exit()
        elif cmd in ("/clear", "/reset"):
            self.session_id = str(uuid4())
            container = self.query_one("#messages", VerticalScroll)
            await container.remove_children()
            self._add_status("Session cleared.")
        elif cmd in ("/cost", "/usage"):
            total = self._cumulative_input + self._cumulative_output
            self._add_status(
                f"Session ({self.turn_count} turns): "
                f"{self._cumulative_input:,}\u2191 {self._cumulative_output:,}\u2193 "
                f"= {total:,} tokens"
            )
        else:
            self._add_status(f"Unknown command: {cmd}. Type /help.")

    # ── Actions ──────────────────────────────────────────────────

    def action_interrupt(self) -> None:
        if self._agent_running:
            self._add_status("Interrupted.")
            self._agent_running = False

    def action_clear_session(self) -> None:
        from asyncio import ensure_future

        ensure_future(self._handle_slash("/clear"))

    # ── Helpers ──────────────────────────────────────────────────

    def _add_status(self, text: str) -> None:
        container = self.query_one("#messages", VerticalScroll)
        container.mount(Static(f"  [dim]{text}[/dim]", classes="status-msg"))
        container.anchor()
