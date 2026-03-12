"""ReActAgent - structured output Reason+Act loop.

A specialized variant of LlmAgent where the model explicitly outputs its
reasoning at each step. Uses LangChain's with_structured_output() to
force the LLM to produce either:
  - ReasonAndAct  - a thought + tool to call
  - FinalAnswer   - the loop terminates

This is the "thinking" agent for tasks that benefit from explicit
chain-of-thought before each action.
"""

from __future__ import annotations

from typing import AsyncIterator, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.context.invocation_context import InvocationContext
from langchain_adk.events.event import (
    ActionEvent,
    ErrorEvent,
    Event,
    FinalAnswerEvent,
    ObservationEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolResultEvent,
)


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class ReasonAndAct(BaseModel):
    """One reasoning step - think then act.

    Attributes
    ----------
    scratchpad : str
        Running notes accumulated across loop iterations.
    thought : str
        The agent's current reasoning about the problem.
    action : str
        The name of the tool to call.
    action_input : str
        The input to pass to the tool.
    """

    type: Literal["reason_and_act"] = "reason_and_act"
    scratchpad: str
    thought: str
    action: str
    action_input: str


class FinalAnswer(BaseModel):
    """Terminal step - loop ends.

    Attributes
    ----------
    scratchpad : str
        Running notes accumulated across the full loop.
    answer : str
        The final answer to return to the user.
    """

    type: Literal["final_answer"] = "final_answer"
    scratchpad: str
    answer: str


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a reasoning agent. You solve problems step by step using the ReAct pattern.

On each step you MUST output one of:
1. A ReasonAndAct - think about the problem, then call a tool.
2. A FinalAnswer  - when you have enough information to answer.

Available tools:
{tool_descriptions}

Rules:
- Always think before acting.
- Use your scratchpad to accumulate observations and notes across steps.
- Only produce a FinalAnswer when you are confident in the result.
- Never call a tool that is not listed above.
"""


class ReActAgent(BaseAgent):
    """Agent that uses explicit structured reasoning before each action.

    Uses LangChain with_structured_output() to enforce ReasonAndAct |
    FinalAnswer at every step. Ideal for tasks requiring transparent
    step-by-step reasoning.

    Attributes
    ----------
    llm : BaseChatModel
        The LangChain chat model to use.
    tools : list[BaseTool]
        Tools available to the agent.
    max_iterations : int
        Maximum ReAct loop iterations.
    """

    def __init__(
        self,
        name: str,
        llm: BaseChatModel,
        tools: list[BaseTool] | None = None,
        *,
        description: str = "",
        max_iterations: int = 10,
    ) -> None:
        super().__init__(name=name, description=description)
        self._tools = {t.name: t for t in (tools or [])}
        self.max_iterations = max_iterations
        self._llm = llm.with_structured_output(
            schema=ReasonAndAct | FinalAnswer,
            include_raw=False,
        )

    def _system_message(self) -> SystemMessage:
        """Build the system message with available tool descriptions."""
        tool_descriptions = "\n".join(
            f"  - {name}: {tool.description}"
            for name, tool in self._tools.items()
        ) or "  (no tools available)"
        return SystemMessage(
            content=_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)
        )

    async def run(
        self,
        input: str,
        *,
        ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run the ReAct loop with structured reasoning output.

        Parameters
        ----------
        input : str
            The user message or task.
        ctx : InvocationContext
            The invocation context.

        Yields
        ------
        Event
            ThoughtEvent, ActionEvent, ToolCallEvent, ToolResultEvent,
            ObservationEvent, FinalAnswerEvent, or ErrorEvent.
        """
        messages: list[BaseMessage] = [
            self._system_message(),
            HumanMessage(content=input),
        ]

        for _ in range(self.max_iterations):
            try:
                output: ReasonAndAct | FinalAnswer = await self._llm.ainvoke(messages)
            except Exception as exc:
                yield ErrorEvent(
                    session_id=ctx.session_id,
                    agent_name=self.name,
                    message=str(exc),
                    exception_type=type(exc).__name__,
                )
                return

            if isinstance(output, FinalAnswer):
                yield FinalAnswerEvent(
                    session_id=ctx.session_id,
                    agent_name=self.name,
                    answer=output.answer,
                    scratchpad=output.scratchpad,
                )
                return

            # --- ReasonAndAct branch ---
            yield ThoughtEvent(
                session_id=ctx.session_id,
                agent_name=self.name,
                thought=output.thought,
                scratchpad=output.scratchpad,
            )
            yield ActionEvent(
                session_id=ctx.session_id,
                agent_name=self.name,
                action=output.action,
                action_input=output.action_input,
            )

            tool = self._tools.get(output.action)
            if tool is None:
                observation = f"Error: tool '{output.action}' not found."
            else:
                yield ToolCallEvent(
                    session_id=ctx.session_id,
                    agent_name=self.name,
                    tool_name=output.action,
                    tool_input=output.action_input,
                )
                try:
                    result = await tool.ainvoke(output.action_input)
                    observation = str(result)
                    yield ToolResultEvent(
                        session_id=ctx.session_id,
                        agent_name=self.name,
                        tool_name=output.action,
                        result=result,
                    )
                except Exception as exc:
                    observation = f"Error: {exc}"
                    yield ToolResultEvent(
                        session_id=ctx.session_id,
                        agent_name=self.name,
                        tool_name=output.action,
                        error=str(exc),
                    )

            yield ObservationEvent(
                session_id=ctx.session_id,
                agent_name=self.name,
                observation=observation,
                tool_name=output.action,
            )

            messages.append(AIMessage(content=str(output.model_dump())))
            messages.append(HumanMessage(content=f"Observation: {observation}"))

        yield ErrorEvent(
            session_id=ctx.session_id,
            agent_name=self.name,
            message=f"Max iterations ({self.max_iterations}) reached without a final answer.",
        )
