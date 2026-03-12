"""SequentialAgent - run sub-agents one after another.

Each sub-agent receives the previous agent's final answer as its input,
creating a processing pipeline.
"""

from __future__ import annotations

from typing import AsyncIterator

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.context.invocation_context import InvocationContext
from langchain_adk.events.event import Event, FinalAnswerEvent


class SequentialAgent(BaseAgent):
    """Runs a list of sub-agents sequentially, chaining their output.

    The output (FinalAnswerEvent.answer) of each agent becomes the input
    to the next. All events from all agents are yielded upstream.

    If a sub-agent emits an event with `actions.escalate = True`, the
    pipeline stops early and yields no further events.

    Attributes
    ----------
    agents : list[BaseAgent]
        Ordered list of agents to run in sequence.
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

    async def run(
        self,
        input: str,
        *,
        ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run sub-agents in sequence, chaining final answers as input.

        Parameters
        ----------
        input : str
            The initial user message or task description.
        ctx : InvocationContext
            The invocation context for this run.

        Yields
        ------
        Event
            All events from all sub-agents in order.
        """
        if not self.sub_agents:
            return

        current_input = input

        for sub_agent in self.sub_agents:
            child_ctx = ctx.derive(agent_name=sub_agent.name)

            async for event in sub_agent.run_with_callbacks(current_input, ctx=child_ctx):
                yield event

                # Stop pipeline if sub-agent escalates
                if event.actions.escalate:
                    return

                # Chain final answer to next agent's input
                if isinstance(event, FinalAnswerEvent):
                    current_input = event.answer
