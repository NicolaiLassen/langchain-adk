"""Base agent abstraction.

All agents in the SDK extend BaseAgent. The contract is simple:
given an input and an InvocationContext, yield a stream of Events.

Follows LangChain's Runnable interface:
  - ``astream(input, *, ctx)`` — core abstract method, yields events
  - ``ainvoke(input, *, ctx)`` — runs to completion, returns final answer
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable

# Avoid circular import — RunConfig/StreamingMode used only at runtime
from typing import TYPE_CHECKING

from langchain_adk.agents.tracing import open_trace
from langchain_adk.context.invocation_context import InvocationContext
from langchain_adk.events.event import Event, EventType

if TYPE_CHECKING:
    pass


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Agents are composable async generators following LangChain's Runnable
    interface. Every agent exposes ``astream()`` (yields events) and
    ``ainvoke()`` (returns the final answer).

    Sub-agents are registered via ``sub_agents`` and can be looked up by
    name with ``find_agent()``.

    Attributes
    ----------
    name : str
        Unique name identifying this agent within an agent tree.
    description : str
        Short description used by LLMs for routing decisions.
    sub_agents : list[BaseAgent]
        Child agents registered under this agent.
    parent_agent : BaseAgent, optional
        The parent agent (set automatically on registration).
    before_agent_callback : callable, optional
        Called before astream() starts.
    after_agent_callback : callable, optional
        Called after astream() completes.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self.sub_agents: list[BaseAgent] = []
        self.parent_agent: BaseAgent | None = None
        self.before_agent_callback: Callable[[InvocationContext], Awaitable[None]] | None = None
        self.after_agent_callback: Callable[[InvocationContext], Awaitable[None]] | None = None

    def is_streaming(self, ctx: InvocationContext) -> bool:
        """Check if SSE streaming is enabled for this run.

        When True, agents should emit partial events (``partial=True``)
        with token-level deltas. When False, only complete events.
        """
        from langchain_adk.agents.run_config import StreamingMode

        rc = ctx.run_config
        return rc is not None and rc.streaming_mode == StreamingMode.SSE

    @abstractmethod
    async def astream(
        self,
        input: str,
        *,
        ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Stream all events from the agent.

        This is the core method that subclasses implement. Equivalent to
        LangChain's ``Runnable.astream``.

        Parameters
        ----------
        input : str
            The user message or task description.
        ctx : InvocationContext
            The invocation context for this run.

        Yields
        ------
        Event
            Events emitted during execution.
        """
        ...

    async def ainvoke(
        self,
        input: str,
        *,
        ctx: InvocationContext,
    ) -> Event:
        """Call the agent and return the final answer.

        Runs the agent to completion, discarding partial and intermediate
        events. Equivalent to LangChain's ``Runnable.ainvoke``.

        Parameters
        ----------
        input : str
            The user message or task description.
        ctx : InvocationContext
            The invocation context for this run.

        Returns
        -------
        Event
            The agent's final answer event.

        Raises
        ------
        RuntimeError
            If the agent finishes without producing a final answer.
        """
        last_answer: Event | None = None
        async for event in self.astream(input, ctx=ctx):
            if event.is_final_response():
                last_answer = event
        if last_answer is None:
            raise RuntimeError(f"Agent {self.name!r} produced no final answer")
        return last_answer

    async def _run_with_callbacks(
        self,
        input: str,
        *,
        ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run the agent with before/after callbacks and tracing.

        Internal method used by Runner and orchestration agents. Opens a
        LangChain trace span, fires callbacks, and wraps the event stream
        with AGENT_START / AGENT_END events.

        Parameters
        ----------
        input : str
            The user message or task description.
        ctx : InvocationContext
            The invocation context for this run.

        Yields
        ------
        Event
            Events emitted during execution, wrapped with start/end events.
        """
        # Set up tracing — opens parent span, stores child config on ctx
        lc_config = ctx.run_config.as_langchain_config() if ctx.run_config else {}
        child_config, run_manager = await open_trace(self.name, lc_config, input)
        if child_config:
            ctx.langchain_run_config = child_config

        if self.before_agent_callback:
            await self.before_agent_callback(ctx)

        yield Event(
            type=EventType.AGENT_START,
            session_id=ctx.session_id,
            agent_name=self.name,
        )

        try:
            async for event in self.astream(input, ctx=ctx):
                yield event
        except Exception as exc:
            if run_manager:
                await run_manager.on_chain_error(exc)
            raise

        yield Event(
            type=EventType.AGENT_END,
            session_id=ctx.session_id,
            agent_name=self.name,
        )

        if run_manager:
            await run_manager.on_chain_end({"output": "completed"})

        if self.after_agent_callback:
            await self.after_agent_callback(ctx)

    def register_sub_agent(self, agent: BaseAgent) -> None:
        """Register a child agent under this agent.

        Parameters
        ----------
        agent : BaseAgent
            The child agent to register.
        """
        agent.parent_agent = self
        self.sub_agents.append(agent)

    def find_agent(self, name: str) -> BaseAgent | None:
        """Recursively search the agent tree for an agent by name.

        Parameters
        ----------
        name : str
            The name of the agent to find.

        Returns
        -------
        BaseAgent or None
            The matching agent, or None if not found.
        """
        if self.name == name:
            return self
        for child in self.sub_agents:
            found = child.find_agent(name)
            if found:
                return found
        return None

    @property
    def root_agent(self) -> BaseAgent:
        """Walk up to the root of the agent tree."""
        agent = self
        while agent.parent_agent is not None:
            agent = agent.parent_agent
        return agent

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
