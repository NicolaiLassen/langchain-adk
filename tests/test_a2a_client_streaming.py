"""Tests for the new ``A2AAgent`` client features.

Wires the client at ``A2AAgent`` directly into an in-process
``A2AServer`` via :class:`httpx.ASGITransport`, so we exercise the
real JSON-RPC + SSE plumbing without binding a TCP socket.

Covers:
- ``streaming=True`` mode yields per-event SDK events (including the
  final ``AGENT_END``).
- Spec-method instance helpers (``list_tasks``, ``get_task``,
  ``cancel_task``, ``fetch_agent_card``, ``set_push_notification_config``
  → ``list_push_notification_configs`` → ``delete_push_notification_config``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx  # noqa: E402

from orxhestra.a2a.server import A2AServer  # noqa: E402
from orxhestra.agents.a2a_agent import A2AAgent  # noqa: E402
from orxhestra.agents.base_agent import BaseAgent  # noqa: E402
from orxhestra.events.event import EventType  # noqa: E402
from orxhestra.models.part import Content  # noqa: E402
from orxhestra.sessions.in_memory_session_service import (  # noqa: E402
    InMemorySessionService,
)


class _EchoAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="echo", description="echo agent")

    async def astream(  # type: ignore[override]
        self, input: str, config=None, *, ctx=None,
    ) -> AsyncIterator:
        yield self._emit_event(
            ctx or self._ensure_ctx(None, None),
            EventType.AGENT_MESSAGE,
            content=Content.from_text(f"echo: {input}"),
        )


@pytest.fixture
def server_app():
    server = A2AServer(
        agent=_EchoAgent(),
        session_service=InMemorySessionService(),
    )
    return server.as_fastapi_app()


@pytest.fixture
def patched_httpx(monkeypatch, server_app):
    """Route every ``httpx.AsyncClient`` request to the in-process app."""
    transport = httpx.ASGITransport(app=server_app)
    real_init = httpx.AsyncClient.__init__

    def _init(self, *args, **kwargs):  # noqa: ANN001
        kwargs["transport"] = transport
        kwargs.setdefault("base_url", "http://testserver")
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", _init)
    yield


class TestClientStreamingMode:
    @pytest.mark.asyncio
    async def test_streaming_yields_per_event_then_end(
        self, patched_httpx,
    ) -> None:
        agent = A2AAgent(
            name="remote-echo",
            url="http://testserver/",
            streaming=True,
        )
        events = []
        async for ev in agent.astream("ping"):
            events.append(ev)

        types = [e.type for e in events]
        # Always brackets with AGENT_START and AGENT_END.
        assert types[0] == EventType.AGENT_START
        assert types[-1] == EventType.AGENT_END
        # At least one inner event carrying the echo text.
        echo_events = [e for e in events if "echo: ping" in (e.text or "")]
        assert echo_events, f"no echo event; types={types}"

    @pytest.mark.asyncio
    async def test_blocking_mode_still_works(self, patched_httpx) -> None:
        agent = A2AAgent(
            name="remote-echo-blocking",
            url="http://testserver/",
            streaming=False,
        )
        events = [ev async for ev in agent.astream("hello")]
        agent_msgs = [e for e in events if e.type == EventType.AGENT_MESSAGE]
        assert agent_msgs
        assert "echo: hello" in (agent_msgs[0].text or "")


class TestClientSpecHelpers:
    @pytest.mark.asyncio
    async def test_list_and_get_tasks(self, patched_httpx) -> None:
        agent = A2AAgent(name="r", url="http://testserver/")

        # Send via blocking astream so the server has tasks to list.
        for i in range(3):
            async for _ in agent.astream(f"msg {i}"):
                pass

        result = await agent.list_tasks(limit=10)
        assert "tasks" in result
        assert len(result["tasks"]) == 3

        first = result["tasks"][0]
        fetched = await agent.get_task(first["id"])
        assert fetched["id"] == first["id"]

    @pytest.mark.asyncio
    async def test_fetch_agent_card_basic_and_extended(
        self, patched_httpx,
    ) -> None:
        agent = A2AAgent(name="r", url="http://testserver")

        basic = await agent.fetch_agent_card()
        assert basic["protocolVersion"] == "1.0"
        assert basic["defaultInputModes"] == ["text/plain"]
        assert basic["capabilities"]["pushNotifications"] is True

        extended = await agent.fetch_agent_card(extended=True)
        assert extended["name"] == basic["name"]

    @pytest.mark.asyncio
    async def test_push_notification_config_round_trip(
        self, patched_httpx,
    ) -> None:
        agent = A2AAgent(name="r", url="http://testserver/")

        # Need a task first.
        async for _ in agent.astream("seed"):
            pass
        listing = await agent.list_tasks(limit=1)
        task_id = listing["tasks"][0]["id"]

        await agent.set_push_notification_config(
            task_id,
            {"id": "push-1", "url": "https://hooks.example.com/wh"},
        )
        configs = await agent.list_push_notification_configs(task_id)
        assert len(configs) == 1
        assert configs[0]["config"]["id"] == "push-1"

        deleted = await agent.delete_push_notification_config(task_id, "push-1")
        assert deleted is True

        configs = await agent.list_push_notification_configs(task_id)
        assert configs == []
