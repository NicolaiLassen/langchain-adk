"""Tests for the new A2A v1.0 server methods.

Covers:
- ``ListTasks`` (with filtering + pagination)
- ``SubscribeToTask`` (re-attach to a task's stream)
- ``GetExtendedAgentCard``
- The new AgentCard required fields (defaultInputModes, defaultOutputModes,
  protocolVersion, securitySchemes, capabilities.pushNotifications=true).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from orxhestra.a2a.server import A2AServer  # noqa: E402
from orxhestra.a2a.types import AgentSkill, SecurityScheme  # noqa: E402
from orxhestra.agents.base_agent import BaseAgent  # noqa: E402
from orxhestra.events.event import Event, EventType  # noqa: E402
from orxhestra.models.part import Content  # noqa: E402
from orxhestra.sessions.in_memory_session_service import (  # noqa: E402
    InMemorySessionService,
)


class _EchoAgent(BaseAgent):
    """Tiny test agent that echoes the input as a final response."""

    def __init__(self) -> None:
        super().__init__(name="echo", description="echo agent for tests")

    async def astream(  # type: ignore[override]
        self, input: str, config=None, *, ctx=None,
    ) -> AsyncIterator[Event]:
        yield self._emit_event(
            ctx or self._ensure_ctx(None, None),
            EventType.AGENT_MESSAGE,
            content=Content.from_text(f"echo: {input}"),
        )


def _make_server(**kwargs) -> A2AServer:
    return A2AServer(
        agent=_EchoAgent(),
        session_service=InMemorySessionService(),
        **kwargs,
    )


def _rpc(client: TestClient, method: str, params: dict | None = None) -> dict:
    body = {"jsonrpc": "2.0", "id": "t1", "method": method, "params": params or {}}
    resp = client.post("/", json=body)
    # JSON-RPC errors are returned as HTTP 4xx with a valid envelope —
    # don't raise_for_status; let the caller inspect ``error``.
    return resp.json()


def _send_one(client: TestClient, text: str = "hi") -> dict:
    """Helper: post a SendMessage and return the resulting Task dict."""
    out = _rpc(
        client,
        "SendMessage",
        {"message": {"role": "user", "parts": [{"text": text, "mediaType": "text/plain"}]}},
    )
    assert "result" in out, out
    return out["result"]


class TestAgentCardV1:
    def test_card_has_all_required_v1_fields(self) -> None:
        server = _make_server(
            skills=[AgentSkill(id="qa", name="QA", description="qa", tags=["g"])],
            security_schemes={"bearer": SecurityScheme(type="http", scheme="bearer")},
            security_requirements=[{"bearer": []}],
        )
        client = TestClient(server.as_fastapi_app())
        resp = client.get("/.well-known/agent-card.json")
        resp.raise_for_status()
        card = resp.json()

        for field in (
            "name",
            "description",
            "supportedInterfaces",
            "version",
            "protocolVersion",
            "capabilities",
            "skills",
            "defaultInputModes",
            "defaultOutputModes",
        ):
            assert field in card, f"missing required AgentCard field: {field}"

        assert card["protocolVersion"] == "1.0"
        assert card["defaultInputModes"] == ["text/plain"]
        assert card["defaultOutputModes"] == ["text/plain"]
        assert card["capabilities"]["pushNotifications"] is True
        assert card["securitySchemes"] == {
            "bearer": {"type": "http", "scheme": "bearer"},
        }
        assert card["securityRequirements"] == [{"bearer": []}]

    def test_card_alias_path(self) -> None:
        server = _make_server()
        client = TestClient(server.as_fastapi_app())
        resp = client.get("/.well-known/agent.json")
        resp.raise_for_status()
        assert resp.json()["protocolVersion"] == "1.0"


class TestListTasks:
    def test_returns_recent_tasks(self) -> None:
        server = _make_server()
        client = TestClient(server.as_fastapi_app())
        for i in range(3):
            _send_one(client, f"hi {i}")

        out = _rpc(client, "ListTasks", {"limit": 10})
        result = out["result"]
        assert len(result["tasks"]) == 3
        # Newest-first ordering — last sent is first returned.
        assert "echo: hi 2" in result["tasks"][0]["artifacts"][0]["parts"][0]["text"]

    def test_pagination_cursor(self) -> None:
        server = _make_server()
        client = TestClient(server.as_fastapi_app())
        for i in range(5):
            _send_one(client, f"msg {i}")

        page1 = _rpc(client, "ListTasks", {"limit": 2})["result"]
        assert len(page1["tasks"]) == 2
        assert page1["nextCursor"] is not None

        page2 = _rpc(
            client, "ListTasks", {"limit": 2, "cursor": page1["nextCursor"]},
        )["result"]
        assert len(page2["tasks"]) == 2
        assert page2["nextCursor"] is not None

        page3 = _rpc(
            client, "ListTasks", {"limit": 2, "cursor": page2["nextCursor"]},
        )["result"]
        assert len(page3["tasks"]) == 1
        # Last page: nextCursor is None and excluded by ``exclude_none``.
        assert page3.get("nextCursor") is None

    def test_filter_by_context_id(self) -> None:
        server = _make_server()
        client = TestClient(server.as_fastapi_app())
        # Send two messages in one context, one in another.
        out = _rpc(
            client,
            "SendMessage",
            {
                "message": {
                    "role": "user",
                    "parts": [{"text": "a", "mediaType": "text/plain"}],
                    "contextId": "ctx-A",
                },
            },
        )
        ctx_a = out["result"]["contextId"]
        _rpc(
            client,
            "SendMessage",
            {
                "message": {
                    "role": "user",
                    "parts": [{"text": "b", "mediaType": "text/plain"}],
                    "contextId": "ctx-A",
                },
            },
        )
        _rpc(
            client,
            "SendMessage",
            {
                "message": {
                    "role": "user",
                    "parts": [{"text": "c", "mediaType": "text/plain"}],
                    "contextId": "ctx-B",
                },
            },
        )

        out = _rpc(client, "ListTasks", {"contextId": ctx_a})
        assert len(out["result"]["tasks"]) == 2

    def test_via_slug_alias(self) -> None:
        server = _make_server()
        client = TestClient(server.as_fastapi_app())
        _send_one(client)
        out = _rpc(client, "tasks/list", {})
        assert "tasks" in out["result"]


class TestSubscribeToTask:
    def test_resubscribe_returns_completed_status_event(self) -> None:
        server = _make_server()
        client = TestClient(server.as_fastapi_app())
        task = _send_one(client, "ping")
        task_id = task["id"]

        with client.stream(
            "POST",
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "s1",
                "method": "SubscribeToTask",
                "params": {"id": task_id},
            },
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = "".join(resp.iter_text())

        assert "TASK_STATE_COMPLETED" in body
        assert task_id in body

    def test_resubscribe_unknown_task_returns_error(self) -> None:
        server = _make_server()
        client = TestClient(server.as_fastapi_app())
        out = _rpc(client, "SubscribeToTask", {"id": "nonexistent"})
        assert "error" in out
        assert out["error"]["code"] == -32001  # TASK_NOT_FOUND

    def test_via_slug_alias(self) -> None:
        server = _make_server()
        client = TestClient(server.as_fastapi_app())
        task = _send_one(client)
        with client.stream(
            "POST",
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "s1",
                "method": "tasks/resubscribe",
                "params": {"id": task["id"]},
            },
        ) as resp:
            assert resp.status_code == 200


class TestGetExtendedAgentCard:
    def test_default_extended_card_matches_basic(self) -> None:
        server = _make_server()
        client = TestClient(server.as_fastapi_app())
        basic = client.get("/.well-known/agent-card.json").json()
        out = _rpc(client, "GetExtendedAgentCard", {})
        assert out["result"]["name"] == basic["name"]

    def test_extended_card_merges_extras(self) -> None:
        server = _make_server(
            extended_card_extras={"privateSkills": ["secret"], "tier": "premium"},
        )
        client = TestClient(server.as_fastapi_app())
        out = _rpc(client, "GetExtendedAgentCard", {})
        assert out["result"]["privateSkills"] == ["secret"]
        assert out["result"]["tier"] == "premium"
        # Capability should reflect the presence of extras.
        assert out["result"]["capabilities"]["extendedAgentCard"] is True

    def test_via_slug_alias(self) -> None:
        server = _make_server()
        client = TestClient(server.as_fastapi_app())
        out = _rpc(client, "agent/getAuthenticatedExtendedCard", {})
        assert "result" in out
