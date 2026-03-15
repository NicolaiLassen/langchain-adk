"""Tests for AgentTool - wrapping agents as LangChain tools."""

import pytest

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.agents.context import Context
from langchain_adk.events.event import Event, EventType
from langchain_adk.models.part import Content
from langchain_adk.tools.agent_tool import AgentTool


class EchoAgent(BaseAgent):
    """Agent that echoes input as its final answer."""

    async def astream(self, input, config=None, *, ctx=None):
        ctx = self._ensure_ctx(config, ctx)
        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text(f"echo:{input}"),
        )


class SilentAgent(BaseAgent):
    """Agent that produces no final answer."""

    async def astream(self, input, config=None, *, ctx=None):
        ctx = self._ensure_ctx(config, ctx)
        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text("thinking..."),
            partial=True,
            turn_complete=False,
        )


def test_agent_tool_name_and_description():
    agent = EchoAgent(name="helper", description="Helps with stuff")
    tool = AgentTool(agent)
    assert tool.name == "helper"
    assert tool.description == "Helps with stuff"


def test_agent_tool_default_description():
    agent = EchoAgent(name="helper")
    tool = AgentTool(agent)
    assert "helper" in tool.description


@pytest.mark.asyncio
async def test_agent_tool_returns_final_answer():
    agent = EchoAgent(name="echo")
    tool = AgentTool(agent)
    ctx = Context(session_id="s1", agent_name="parent")
    tool.inject_context(ctx)

    result = await tool.ainvoke({"request": "hello"})
    assert result == "echo:hello"


@pytest.mark.asyncio
async def test_agent_tool_no_final_answer():
    agent = SilentAgent(name="silent")
    tool = AgentTool(agent)
    ctx = Context(session_id="s1", agent_name="parent")
    tool.inject_context(ctx)

    result = await tool.ainvoke({"request": "hello"})
    assert "no final answer" in result


@pytest.mark.asyncio
async def test_agent_tool_raises_without_context():
    agent = EchoAgent(name="echo")
    tool = AgentTool(agent)

    with pytest.raises(RuntimeError, match="no context"):
        await tool.ainvoke({"request": "hello"})


@pytest.mark.asyncio
async def test_agent_tool_derives_context():
    """AgentTool creates a child context with branch attribution."""
    branches = []

    class BranchCapture(BaseAgent):
        async def astream(self, input, config=None, *, ctx=None):
            ctx = self._ensure_ctx(config, ctx)
            branches.append(ctx.branch)
            yield self._emit_event(
                ctx, EventType.AGENT_MESSAGE, content=Content.from_text("done"),
            )

    agent = BranchCapture(name="child")
    tool = AgentTool(agent)
    ctx = Context(session_id="s1", agent_name="parent")
    tool.inject_context(ctx)

    await tool.ainvoke({"request": "test"})
    assert branches == ["child"]


def test_sync_run_raises():
    agent = EchoAgent(name="echo")
    tool = AgentTool(agent)
    with pytest.raises(NotImplementedError):
        tool.invoke({"request": "hello"})
