"""Streaming example - token-by-token output from an LlmAgent.

Demonstrates:
  - RunConfig with StreamingMode.SSE
  - Partial FinalAnswerEvent for real-time text streaming
  - Complete events for tool calls / tool results
"""

from __future__ import annotations

import asyncio
import sys

# Swap in any LangChain-supported LLM:
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic

from langchain_core.tools import tool

from langchain_adk.agents.llm_agent import LlmAgent
from langchain_adk.agents.run_config import RunConfig, StreamingMode
from langchain_adk.context.invocation_context import InvocationContext
from langchain_adk.events.event import (
    FinalAnswerEvent,
    ToolCallEvent,
    ToolResultEvent,
)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is partly cloudy and 18°C."


async def main() -> None:
    # --- Replace with a real LLM ---
    # llm = ChatOpenAI(model="gpt-4o-mini")
    # llm = ChatAnthropic(model="claude-3-5-haiku-latest")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    agent = LlmAgent(
        name="StreamingWeatherAgent",
        llm=llm,  # noqa: F821
        tools=[get_weather],
        instructions="You are a helpful weather assistant. Use the get_weather tool.",
    )

    run_config = RunConfig(streaming_mode=StreamingMode.SSE)

    ctx = InvocationContext(
        session_id="streaming-demo",
        agent_name=agent.name,
        run_config=run_config,
    )

    print(f"Running agent (streaming): {agent.name}\n{'=' * 50}")

    async for event in agent.run(
        "What's the weather in Copenhagen and Berlin?",
        ctx=ctx,
    ):
        if isinstance(event, ToolCallEvent):
            print(f"\n[TOOL CALL] {event.tool_name}({event.tool_input})")
        elif isinstance(event, ToolResultEvent):
            print(f"[TOOL RESULT] {event.result or event.error}")
        elif isinstance(event, FinalAnswerEvent):
            if event.partial:
                # Stream tokens as they arrive
                sys.stdout.write(".")
                sys.stdout.flush()
            else:
                # Final complete answer
                print(f"\n\n[ANSWER]\n{event.answer}")


if __name__ == "__main__":
    asyncio.run(main())
