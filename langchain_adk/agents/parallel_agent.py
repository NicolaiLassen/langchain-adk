"""ParallelAgent - run sub-agents concurrently with isolated branches.

Each sub-agent runs in its own branch context so they don't see each
other's events. Events from all agents are merged into a single stream
via an asyncio.Queue.

Uses the pre-3.11 asyncio approach for Python 3.10 compatibility.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.context.invocation_context import InvocationContext
from langchain_adk.events.event import Event


class ParallelAgent(BaseAgent):
    """Runs sub-agents concurrently with branch-isolated contexts.

    All sub-agents start at the same time. Their events are merged into a
    single stream in the order they are produced. Each agent gets a derived
    context with its own branch path, preventing event cross-contamination.

    Attributes
    ----------
    agents : list[BaseAgent]
        Agents to run concurrently.
    """

    def __init__(
        self,
        name: str,
        agents: list[BaseAgent],
        *,
        description: str = "",
    ) -> None:
        super().__init__(name=name, description=description)
        for agent in agents:
            self.register_sub_agent(agent)

    async def astream(
        self,
        input: str,
        *,
        ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run all sub-agents concurrently and merge their event streams.

        Parameters
        ----------
        input : str
            The user message or task passed to every sub-agent.
        ctx : InvocationContext
            The invocation context for this run.

        Yields
        ------
        Event
            Events from all sub-agents interleaved in arrival order.
        """
        if not self.sub_agents:
            return

        sentinel = object()
        queue: asyncio.Queue = asyncio.Queue()

        async def run_agent(agent: BaseAgent) -> None:
            """Run a single agent and enqueue its events."""
            child_ctx = ctx.derive(
                agent_name=agent.name,
                branch_suffix=f"{self.name}.{agent.name}",
            )
            try:
                async for event in agent._run_with_callbacks(input, ctx=child_ctx):
                    resume = asyncio.Event()
                    await queue.put((event, resume))
                    # Wait until the consumer yields this event before continuing
                    await resume.wait()
            finally:
                await queue.put((sentinel, None))

        tasks = [
            asyncio.create_task(run_agent(agent))
            for agent in self.sub_agents
        ]

        sentinel_count = 0
        try:
            while sentinel_count < len(self.sub_agents):
                # Propagate task exceptions eagerly
                for task in tasks:
                    if task.done() and not task.cancelled():
                        task.result()

                item, resume = await queue.get()
                if item is sentinel:
                    sentinel_count += 1
                else:
                    yield item
                    resume.set()  # type: ignore[union-attr]
        finally:
            for task in tasks:
                task.cancel()
