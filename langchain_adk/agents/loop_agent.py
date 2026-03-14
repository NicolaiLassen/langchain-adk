"""LoopAgent - repeat sub-agents until escalation or max_iterations.

The loop terminates when:
  - A sub-agent emits an event with actions.escalate = True
  - max_iterations is reached (if set)

The escalate signal is the canonical way for a sub-agent to signal completion.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.context.invocation_context import InvocationContext
from langchain_adk.events.event import ErrorEvent, Event


class LoopAgent(BaseAgent):
    """Runs sub-agents in a loop until a stop condition is met.

    Sub-agents run in order each iteration, exactly like SequentialAgent
    within a single loop. Between iterations the same input is reused
    (unless a sub-agent updates ctx.state which an instruction provider
    can use to build new input).

    Termination:
      - Any event with actions.escalate = True stops the loop.
      - max_iterations reached stops the loop (yields an ErrorEvent).
      - should_continue callback returning False stops the loop.

    Attributes
    ----------
    agents : list[BaseAgent]
        Sub-agents to run each iteration (in order).
    max_iterations : int, optional
        Maximum number of full loop cycles. None = unlimited.
    should_continue : callable, optional
        Optional callable inspecting the last event to decide whether to
        keep looping. Return False to stop.
    """

    def __init__(
        self,
        name: str,
        agents: list[BaseAgent],
        *,
        description: str = "",
        max_iterations: int | None = 10,
        should_continue: Callable[[Event], bool] | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        for agent in agents:
            self.register_sub_agent(agent)
        self.max_iterations = max_iterations
        self.should_continue = should_continue

    async def astream(
        self,
        input: str,
        *,
        ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run sub-agents in a loop until a termination condition is met.

        Parameters
        ----------
        input : str
            The user message or task passed to sub-agents each iteration.
        ctx : InvocationContext
            The invocation context for this run.

        Yields
        ------
        Event
            All events from all sub-agents across all iterations.
        """
        if not self.sub_agents:
            return

        iteration = 0
        last_event: Event | None = None

        while True:
            if self.max_iterations is not None and iteration >= self.max_iterations:
                yield ErrorEvent(
                    session_id=ctx.session_id,
                    agent_name=self.name,
                    message=(
                        f"LoopAgent '{self.name}' reached max_iterations "
                        f"({self.max_iterations}) without escalating."
                    ),
                )
                return

            for sub_agent in self.sub_agents:
                child_ctx = ctx.derive(agent_name=sub_agent.name)

                async for event in sub_agent._run_with_callbacks(input, ctx=child_ctx):
                    yield event
                    last_event = event

                    # Sub-agent signalled it's done - exit loop
                    if event.actions.escalate:
                        return

            # Custom termination check
            if self.should_continue is not None and last_event is not None:
                if not self.should_continue(last_event):
                    return

            iteration += 1
