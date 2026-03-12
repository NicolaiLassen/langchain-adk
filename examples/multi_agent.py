"""Multi-agent example - SequentialAgent + AgentTool + skills.

Demonstrates:
  - Two LlmAgents chained via SequentialAgent
  - AgentTool (sub-agent as callable tool)
  - InMemorySkillStore + load_skill tool
  - Prompt catalog (PromptContext + build_system_prompt)
  - Sessions
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from langchain_adk.agents.llm_agent import LlmAgent
from langchain_adk.agents.sequential_agent import SequentialAgent
from langchain_adk.context.invocation_context import InvocationContext
from langchain_adk.events.event import FinalAnswerEvent, ToolCallEvent
from langchain_adk.prompts.catalog import build_system_prompt
from langchain_adk.prompts.context import PromptContext
from langchain_adk.sessions.in_memory_session_service import InMemorySessionService
from langchain_adk.skills.skill import Skill
from langchain_adk.skills.skill_store import InMemorySkillStore
from langchain_adk.skills.load_skill_tool import make_load_skill_tool, make_list_skills_tool
from langchain_adk.tools.agent_tool import AgentTool


async def main() -> None:
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    # --- Skill store ---
    skill_store = InMemorySkillStore([
        Skill(
            name="summarization",
            description="Condense long text into key bullet points.",
            content=(
                "When summarizing, extract the 3-5 most important points. "
                "Use bullet points. Be concise."
            ),
        )
    ])

    # --- Research agent ---
    research_prompt = build_system_prompt(PromptContext(
        agent_name="ResearchAgent",
        goal="Research the given topic and produce a detailed summary.",
        instructions="Search thoroughly. Return your findings as structured text.",
    ))

    research_agent = LlmAgent(
        name="ResearchAgent",
        llm=llm,  # noqa: F821
        instructions=research_prompt,
        tools=[
            make_list_skills_tool(skill_store),
            make_load_skill_tool(skill_store),
        ],
    )

    # --- Writer agent (uses ResearchAgent as a sub-agent tool) ---
    writer_prompt = build_system_prompt(PromptContext(
        agent_name="WriterAgent",
        goal="Write a polished article based on the research provided.",
        instructions="Write clearly. Structure with sections. Use markdown.",
        agents=[{"name": "ResearchAgent", "description": "Provides research on any topic."}],
    ))

    writer_agent = LlmAgent(
        name="WriterAgent",
        llm=llm,  # noqa: F821
        instructions=writer_prompt,
        tools=[AgentTool(research_agent)],
    )

    # --- Sequential pipeline ---
    pipeline = SequentialAgent(
        name="ResearchWriterPipeline",
        agents=[research_agent, writer_agent],
    )

    # --- Sessions ---
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="demo",
        user_id="user-1",
    )

    ctx = InvocationContext(
        session_id=session.id,
        user_id=session.user_id,
        app_name="demo",
        agent_name=pipeline.name,
    )

    print(f"Pipeline: {pipeline.name}\n{'='*40}")

    async for event in pipeline.run("Write an article about LLM agents", ctx=ctx):
        if isinstance(event, ToolCallEvent):
            print(f"[TOOL] {event.agent_name} → {event.tool_name}")
        elif isinstance(event, FinalAnswerEvent):
            print(f"\n[FINAL - {event.agent_name}]\n{event.answer[:500]}...")


if __name__ == "__main__":
    asyncio.run(main())
