"""Tests for BaseAgent: ainvoke, astream, is_streaming, find_agent, callbacks."""

import pytest
from typing import AsyncIterator

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.agents.run_config import RunConfig, StreamingMode
from langchain_adk.context.invocation_context import InvocationContext
from langchain_adk.events.event import Event, EventType, FinalAnswerEvent
from langchain_adk.models.part import Content


class StubAgent(BaseAgent):
    """Minimal agent that yields a fixed answer."""

    def __init__(self, name: str = "stub", answer: str = "hello", **kwargs):
        super().__init__(name=name, **kwargs)
        self._answer = answer

    async def astream(self, input: str, *, ctx: InvocationContext) -> AsyncIterator[Event]:
        yield FinalAnswerEvent(
            session_id=ctx.session_id,
            agent_name=self.name,
            content=Content.from_text(self._answer),
        )


class NoAnswerAgent(BaseAgent):
    """Agent that yields no FinalAnswerEvent."""

    async def astream(self, input: str, *, ctx: InvocationContext) -> AsyncIterator[Event]:
        yield Event(type=EventType.AGENT_START, session_id=ctx.session_id, agent_name=self.name)


def _ctx(**kwargs) -> InvocationContext:
    defaults = {"session_id": "test", "agent_name": "stub"}
    defaults.update(kwargs)
    return InvocationContext(**defaults)


@pytest.mark.asyncio
async def test_ainvoke_returns_final_answer():
    agent = StubAgent(answer="42")
    result = await agent.ainvoke("question", ctx=_ctx())
    assert isinstance(result, FinalAnswerEvent)
    assert result.text == "42"


@pytest.mark.asyncio
async def test_ainvoke_raises_on_no_answer():
    agent = NoAnswerAgent(name="empty")
    with pytest.raises(RuntimeError, match="no final answer"):
        await agent.ainvoke("question", ctx=_ctx(agent_name="empty"))


@pytest.mark.asyncio
async def test_astream_yields_all_events():
    agent = StubAgent(answer="streamed")
    events = [e async for e in agent.astream("q", ctx=_ctx())]
    assert len(events) == 1
    assert isinstance(events[0], FinalAnswerEvent)
    assert events[0].text == "streamed"


def test_is_streaming_false_by_default():
    agent = StubAgent()
    ctx = _ctx()
    assert agent.is_streaming(ctx) is False


def test_is_streaming_true_with_sse():
    agent = StubAgent()
    ctx = _ctx()
    ctx.run_config = RunConfig(streaming_mode=StreamingMode.SSE)
    assert agent.is_streaming(ctx) is True


def test_is_streaming_false_with_none_mode():
    agent = StubAgent()
    ctx = _ctx()
    ctx.run_config = RunConfig(streaming_mode=StreamingMode.NONE)
    assert agent.is_streaming(ctx) is False


def test_find_agent_self():
    agent = StubAgent(name="root")
    assert agent.find_agent("root") is agent


def test_find_agent_child():
    parent = StubAgent(name="parent")
    child = StubAgent(name="child")
    parent.register_sub_agent(child)
    assert parent.find_agent("child") is child
    assert child.parent_agent is parent


def test_find_agent_nested():
    root = StubAgent(name="root")
    mid = StubAgent(name="mid")
    leaf = StubAgent(name="leaf")
    root.register_sub_agent(mid)
    mid.register_sub_agent(leaf)
    assert root.find_agent("leaf") is leaf


def test_find_agent_not_found():
    agent = StubAgent(name="root")
    assert agent.find_agent("nonexistent") is None


def test_root_agent():
    root = StubAgent(name="root")
    child = StubAgent(name="child")
    grandchild = StubAgent(name="grandchild")
    root.register_sub_agent(child)
    child.register_sub_agent(grandchild)
    assert grandchild.root_agent is root
    assert child.root_agent is root
    assert root.root_agent is root


@pytest.mark.asyncio
async def test_before_after_callbacks():
    agent = StubAgent(answer="cb")
    calls = []

    async def before(ctx):
        calls.append("before")

    async def after(ctx):
        calls.append("after")

    agent.before_agent_callback = before
    agent.after_agent_callback = after

    events = [e async for e in agent._run_with_callbacks("q", ctx=_ctx())]
    assert "before" in calls
    assert "after" in calls
    # Should have AGENT_START, FinalAnswer, AGENT_END
    types = [e.type for e in events]
    assert EventType.AGENT_START in types
    assert EventType.AGENT_END in types


def test_repr():
    agent = StubAgent(name="test_agent")
    assert "StubAgent" in repr(agent)
    assert "test_agent" in repr(agent)
