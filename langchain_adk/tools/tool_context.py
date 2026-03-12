"""ToolContext - runtime context passed into tool execution.

Gives tools access to the invocation
state and a local ``EventActions`` instance so they can signal escalation,
agent transfer, and state mutations without importing InvocationContext
directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_adk.events.event_actions import EventActions

if TYPE_CHECKING:
    from langchain_adk.context.invocation_context import InvocationContext


class ToolContext:
    """Context object passed to tools during execution.

    Wraps ``InvocationContext`` with a tool-scoped ``EventActions`` that the
    agent inspects after the tool returns to apply side-effects (escalate,
    transfer, state updates).

    Parameters
    ----------
    ctx : InvocationContext
        The parent invocation context for this agent run.

    Attributes
    ----------
    state : dict[str, Any]
        Shared mutable state from the invocation context. Tools may read
        and write this freely; changes are visible to all subsequent agents
        in the same invocation.
    actions : EventActions
        Side-effects the tool wants to signal. Set ``actions.escalate``
        to stop a ``LoopAgent``, or ``actions.transfer_to_agent`` to hand
        off control to another agent.
    agent_name : str
        Name of the agent currently executing this tool.
    session_id : str
        Current session identifier.
    """

    def __init__(self, ctx: InvocationContext) -> None:
        self._ctx = ctx
        self.state: dict[str, Any] = ctx.state
        self.actions: EventActions = EventActions()
        self.agent_name: str = ctx.agent_name
        self.session_id: str = ctx.session_id
