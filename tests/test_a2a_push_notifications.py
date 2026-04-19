"""Tests for the A2A v1.0 push-notification methods + dispatcher.

Covers:
- The four config CRUD methods (Create / Get / List / Delete) with
  both their PascalCase and slug names.
- ``PushNotificationDispatcher._validate_url`` SSRF rejections.
- End-to-end dispatch: registering a config + completing a task ⇒
  webhook receives at least one POST with a recognisable body.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from orxhestra.a2a.notifications import (  # noqa: E402
    InMemoryPushNotificationStore,
    PushNotificationDispatcher,
)
from orxhestra.a2a.server import A2AServer  # noqa: E402
from orxhestra.a2a.types import (  # noqa: E402
    PushNotificationAuthentication,
    PushNotificationConfig,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
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


def _make_server(**kwargs) -> A2AServer:
    return A2AServer(
        agent=_EchoAgent(),
        session_service=InMemorySessionService(),
        **kwargs,
    )


def _rpc(client: TestClient, method: str, params: dict | None = None) -> dict:
    body = {"jsonrpc": "2.0", "id": "p1", "method": method, "params": params or {}}
    resp = client.post("/", json=body)
    return resp.json()


def _send_one(client: TestClient, text: str = "hi") -> dict:
    out = _rpc(
        client,
        "SendMessage",
        {"message": {"role": "user", "parts": [{"text": text, "mediaType": "text/plain"}]}},
    )
    return out["result"]


class TestPushNotificationConfigCRUD:
    def test_create_get_list_delete_round_trip(self) -> None:
        server = _make_server()
        client = TestClient(server.as_fastapi_app())
        task = _send_one(client)
        task_id = task["id"]

        # Create
        out = _rpc(
            client,
            "CreateTaskPushNotificationConfig",
            {
                "taskId": task_id,
                "config": {
                    "id": "cfg-1",
                    "url": "https://hooks.example.com/webhook",
                    "token": "secret",
                },
            },
        )
        assert out["result"]["taskId"] == task_id
        assert out["result"]["config"]["id"] == "cfg-1"

        # Get (no config_id — returns first)
        out = _rpc(client, "GetTaskPushNotificationConfig", {"taskId": task_id})
        assert out["result"]["config"]["url"] == "https://hooks.example.com/webhook"

        # Get (with config_id)
        out = _rpc(
            client,
            "GetTaskPushNotificationConfig",
            {"taskId": task_id, "configId": "cfg-1"},
        )
        assert out["result"]["config"]["id"] == "cfg-1"

        # List
        out = _rpc(client, "ListTaskPushNotificationConfigs", {"taskId": task_id})
        assert len(out["result"]) == 1
        assert out["result"][0]["config"]["id"] == "cfg-1"

        # Delete
        out = _rpc(
            client,
            "DeleteTaskPushNotificationConfig",
            {"taskId": task_id, "configId": "cfg-1"},
        )
        assert out["result"]["deleted"] is True

        # List → empty
        out = _rpc(client, "ListTaskPushNotificationConfigs", {"taskId": task_id})
        assert out["result"] == []

    def test_create_for_unknown_task_errors(self) -> None:
        server = _make_server()
        client = TestClient(server.as_fastapi_app())
        out = _rpc(
            client,
            "CreateTaskPushNotificationConfig",
            {
                "taskId": "nope",
                "config": {"id": "x", "url": "https://hooks.example.com/x"},
            },
        )
        assert "error" in out
        assert out["error"]["code"] == -32001  # TASK_NOT_FOUND

    def test_get_unknown_returns_error(self) -> None:
        server = _make_server()
        client = TestClient(server.as_fastapi_app())
        out = _rpc(
            client,
            "GetTaskPushNotificationConfig",
            {"taskId": "missing"},
        )
        assert "error" in out
        assert out["error"]["code"] == -32001

    def test_via_slug_aliases(self) -> None:
        server = _make_server()
        client = TestClient(server.as_fastapi_app())
        task = _send_one(client)
        task_id = task["id"]

        out = _rpc(
            client,
            "tasks/pushNotificationConfig/set",
            {
                "taskId": task_id,
                "config": {"id": "s-1", "url": "https://hooks.example.com/s"},
            },
        )
        assert "result" in out

        out = _rpc(
            client, "tasks/pushNotificationConfig/list", {"taskId": task_id},
        )
        assert len(out["result"]) == 1


class TestSSRFGuard:
    def _disp(self) -> PushNotificationDispatcher:
        return PushNotificationDispatcher(InMemoryPushNotificationStore())

    @pytest.mark.parametrize(
        "url",
        [
            "http://example.com/hook",            # plain HTTP
            "https://localhost/hook",             # localhost name
            "https://127.0.0.1/hook",             # loopback IPv4
            "https://[::1]/hook",                 # loopback IPv6
            "https://10.0.0.1/hook",              # RFC1918 private
            "https://192.168.1.10/hook",          # RFC1918 private
            "https://169.254.169.254/latest",     # link-local (AWS metadata!)
        ],
    )
    def test_rejects_dangerous_url(self, url: str) -> None:
        with pytest.raises(ValueError):
            self._disp()._validate_url(url)

    def test_accepts_public_https(self) -> None:
        # Should not raise.
        self._disp()._validate_url("https://hooks.example.com/webhook")

    def test_allow_insecure_skips_checks(self) -> None:
        d = PushNotificationDispatcher(
            InMemoryPushNotificationStore(), allow_insecure_webhooks=True,
        )
        d._validate_url("http://localhost:5000/hook")  # no raise


class TestDispatchE2E:
    @pytest.mark.asyncio
    async def test_dispatch_posts_to_each_registered_url(self) -> None:
        store = InMemoryPushNotificationStore()
        dispatcher = PushNotificationDispatcher(
            store, allow_insecure_webhooks=True,
        )

        await store.set(
            "task-x",
            PushNotificationConfig(
                id="cfg-A",
                url="http://example.test/a",
                authentication=PushNotificationAuthentication(
                    type="bearer", credential="tok-A",
                ),
            ),
        )
        await store.set(
            "task-x",
            PushNotificationConfig(
                id="cfg-B",
                url="http://example.test/b",
                token="tok-B",
            ),
        )

        captured: list[tuple[str, dict, dict]] = []

        class _FakeResponse:
            status_code = 204

        class _FakeClient:
            def __init__(self, *_, **__) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            async def post(self, url, json, headers):
                captured.append((url, json, headers))
                return _FakeResponse()

        import httpx
        original = httpx.AsyncClient
        httpx.AsyncClient = _FakeClient  # type: ignore[assignment]
        try:
            task = Task(
                id="task-x",
                context_id="ctx",
                status=TaskStatus(state=TaskState.WORKING),
            )
            event = TaskStatusUpdateEvent(
                task_id="task-x",
                context_id="ctx",
                status=TaskStatus(state=TaskState.COMPLETED),
                final=True,
            )
            await dispatcher.dispatch(task, event)
            # Background tasks may not be done yet — drain.
            await asyncio.sleep(0.05)
        finally:
            httpx.AsyncClient = original  # type: ignore[assignment]

        urls = sorted(call[0] for call in captured)
        assert urls == ["http://example.test/a", "http://example.test/b"]
        # Each call carried its bearer token.
        auth_headers = {call[0]: call[2].get("Authorization") for call in captured}
        assert auth_headers["http://example.test/a"] == "Bearer tok-A"
        assert auth_headers["http://example.test/b"] == "Bearer tok-B"
        # Body shape includes taskId + event payload.
        sample_body = captured[0][1]
        assert sample_body["taskId"] == "task-x"
        assert sample_body["contextId"] == "ctx"
        assert "event" in sample_body
