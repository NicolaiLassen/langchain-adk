"""A2AAgent — proxy requests to a remote A2A v1.0 server.

Two modes:

* **Blocking** (``streaming=False``, default) — POSTs ``SendMessage``,
  waits for the full :class:`Task`, yields a single ``AGENT_MESSAGE``.
* **Streaming** (``streaming=True``) — POSTs ``SendStreamingMessage``,
  parses the SSE stream, yields per-event updates as the remote agent
  produces them.

Plus client helpers for the rest of the v1.0 method surface:

* :meth:`A2AAgent.list_tasks`
* :meth:`A2AAgent.get_task`
* :meth:`A2AAgent.subscribe_to_task`
* :meth:`A2AAgent.cancel_task`
* :meth:`A2AAgent.set_push_notification_config`
* :meth:`A2AAgent.get_push_notification_config`
* :meth:`A2AAgent.list_push_notification_configs`
* :meth:`A2AAgent.delete_push_notification_config`
* :meth:`A2AAgent.fetch_agent_card`
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.agents.invocation_context import InvocationContext
from orxhestra.events.event import Event, EventType
from orxhestra.models.part import Content

# A2A v1.0 protocol types (lightweight aliases, no external dep).
A2APart = dict[str, Any]
A2AMessage = dict[str, Any]
A2AArtifact = dict[str, Any]
A2AStatus = dict[str, Any]
A2ATask = dict[str, Any]
A2AResponse = dict[str, Any]

_Ed25519PrivateKey = Any
_DidResolver = Any


class A2AAgent(BaseAgent):
    """Agent that delegates to a remote A2A v1.0 server over HTTP.

    Sends a JSON-RPC ``SendMessage`` request per turn and converts
    the server's response stream back into :class:`Event` objects.

    Parameters
    ----------
    name : str
        Local name for this agent in the agent tree.
    url : str
        Base URL of the remote A2A server
        (e.g. ``"http://localhost:9000"``).
    description : str
        Description used for routing decisions.
    signing_key : Ed25519PrivateKey, optional
        When set, outgoing A2A messages are signed with this key and
        the server's :class:`AgentCard` is used to verify responses.
        Requires ``orxhestra[auth]``.
    signing_did : str
        DID matching ``signing_key``.  Attached to each signed
        message.
    require_signed_response : bool
        When ``True``, responses from the remote server that lack a
        valid signature raise :class:`RuntimeError`.  Defaults to
        ``False`` for back-compat with unsigned peers.
    resolver : DidResolver, optional
        Resolver used to verify response signatures.  Defaults to a
        :class:`CompositeResolver` covering ``did:key``.

    See Also
    --------
    BaseAgent : Base class this extends.
    AgentCard : Remote card the server publishes for discovery.
    Message : A2A-wire message type sent on each turn.
    Task : Remote task wrapping the message execution.
    orxhestra.a2a.signing : Signature helpers used on the wire.

    Examples
    --------
    >>> remote = A2AAgent(
    ...     name="remote_researcher",
    ...     url="https://researcher.example.com",
    ...     description="Web research specialist.",
    ... )
    >>> async for event in remote.astream("Summarize arxiv:2024.12345"):
    ...     print(event.text)
    """

    def __init__(
        self,
        name: str,
        url: str,
        description: str = "",
        *,
        streaming: bool = False,
        timeout: float = 120.0,
        signing_key: _Ed25519PrivateKey | None = None,
        signing_did: str = "",
        require_signed_response: bool = False,
        resolver: _DidResolver | None = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            signing_key=signing_key,
            signing_did=signing_did,
        )
        self.url: str = url.rstrip("/")
        self.streaming = streaming
        self.timeout = timeout
        self.require_signed_response = require_signed_response
        self._resolver = resolver

    async def astream(
        self,
        input: str,
        config: RunnableConfig | None = None,
        *,
        ctx: InvocationContext | None = None,
    ) -> AsyncIterator[Event]:
        """Send a message to the remote A2A server and yield events.

        In **blocking** mode (``streaming=False``) the agent posts
        ``SendMessage``, waits for the full task, and yields a single
        ``AGENT_MESSAGE`` with the final answer.

        In **streaming** mode (``streaming=True``) the agent posts
        ``SendStreamingMessage`` and yields per-event updates as the
        remote agent produces them — ``THINKING``, ``TOOL_CALL``,
        ``TOOL_RESPONSE``, ``AGENT_MESSAGE`` — closing with
        ``AGENT_END``.

        Parameters
        ----------
        input
            The user message to forward to the remote server.
        config
            LangChain-compatible config dict (tags, callbacks, etc.).
        ctx
            Invocation context. Auto-created if not provided.
        """
        ctx = self._ensure_ctx(config, ctx)

        yield self._emit_event(ctx, EventType.AGENT_START)

        if self.streaming:
            async for event in self._send_message_streaming(input, ctx):
                yield event
        else:
            response_text: str = await self._send_message(input)
            yield self._emit_event(
                ctx,
                EventType.AGENT_MESSAGE,
                content=Content.from_text(response_text),
            )

        yield self._emit_event(ctx, EventType.AGENT_END)

    async def _send_message(self, text: str) -> str:
        """Send a ``SendMessage`` JSON-RPC request and return the answer.

        When :attr:`signing_key` is set, the outgoing message is
        signed via :func:`orxhestra.a2a.signing.sign_message`.  When
        :attr:`require_signed_response` is true, the returned agent
        message is verified before its text is extracted.

        Parameters
        ----------
        text : str
            Body of the outgoing user message.

        Returns
        -------
        str
            Plain text of the remote agent's reply.

        Raises
        ------
        ImportError
            If ``httpx`` is not installed.
        RuntimeError
            If ``require_signed_response`` is set and the response
            message is unsigned or the signature fails to verify.
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "A2AAgent requires httpx. Install with: pip install httpx"
            ) from None

        from orxhestra.a2a.types import Message as A2AMessageModel
        from orxhestra.a2a.types import Part as A2APartModel
        from orxhestra.a2a.types import Role

        message_model = A2AMessageModel(
            message_id=str(uuid4()),
            role=Role.USER,
            parts=[A2APartModel(text=text, media_type="text/plain")],
        )

        if self.signing_key is not None and self.signing_did:
            from orxhestra.a2a.signing import sign_message as sign_a2a_message

            message_model = sign_a2a_message(
                message_model, self.signing_key, self.signing_did,
            )

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "SendMessage",
            "params": {
                "message": message_model.model_dump(by_alias=True, exclude_none=True),
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                self.url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "A2A-Version": "1.0",
                },
            )
            resp.raise_for_status()
            data: A2AResponse = resp.json()

        if self.require_signed_response:
            await self._verify_response(data)

        return self._extract_answer(data)

    async def _verify_response(self, data: A2AResponse) -> None:
        """Raise ``RuntimeError`` when the response lacks a valid signature.

        Walks the JSON-RPC result looking for the agent's
        ``status.message`` (or direct ``message``) and verifies it via
        :func:`orxhestra.a2a.signing.verify_message`.

        Parameters
        ----------
        data : dict[str, Any]
            Parsed JSON-RPC response body.

        Raises
        ------
        RuntimeError
            If no agent message is present or signature verification
            fails.
        """
        from orxhestra.a2a.signing import verify_message as verify_a2a_message
        from orxhestra.a2a.types import Message as A2AMessageModel

        resolver = self._resolver
        if resolver is None:
            from orxhestra.security.did import DidKeyResolver

            resolver = DidKeyResolver()
            self._resolver = resolver

        result = data.get("result", {}) or {}
        raw_message = result.get("message")
        if raw_message is None:
            status = (result.get("status") or {}) if isinstance(result, dict) else {}
            raw_message = status.get("message")

        if raw_message is None:
            raise RuntimeError(
                "A2AAgent require_signed_response=True but response has no agent message."
            )

        message = A2AMessageModel.model_validate(raw_message)
        if not await verify_a2a_message(message, resolver):
            raise RuntimeError(
                "A2AAgent response signature missing or invalid; "
                "refusing to accept under require_signed_response=True."
            )

    @staticmethod
    def _extract_answer(data: A2AResponse) -> str:
        """Extract the text answer from a ``SendMessage`` response.

        The A2A v1.0 ``SendMessageResponse`` is a oneof: either a
        ``task`` (with artifacts, status, history) or a direct ``message``.
        """
        result: dict[str, Any] = data.get("result", {})

        # Direct message response.
        direct_msg: A2AMessage | None = result.get("message")
        if direct_msg is not None:
            text: str = _extract_text_from_parts(direct_msg.get("parts", []))
            if text:
                return text

        # Task response — check if result itself is a task (has "id" and "status").
        task: A2ATask | None = None
        if "id" in result and "status" in result:
            task = result
        elif "task" in result:
            task = result["task"]

        if task is None:
            return ""

        # 1. Artifacts carry the final output.
        artifacts: list[A2AArtifact] = task.get("artifacts", [])
        for artifact in artifacts:
            text = _extract_text_from_parts(artifact.get("parts", []))
            if text:
                return text

        # 2. Status message.
        status: A2AStatus = task.get("status", {})
        status_msg: A2AMessage | None = status.get("message")
        if status_msg is not None:
            text = _extract_text_from_parts(status_msg.get("parts", []))
            if text:
                return text

        # 3. History — last agent message.
        history: list[A2AMessage] = task.get("history", [])
        for msg in reversed(history):
            if msg.get("role") == "agent":
                text = _extract_text_from_parts(msg.get("parts", []))
                if text:
                    return text

        return ""


def _extract_text_from_parts(parts: list[A2APart]) -> str:
    """Return the first text value from a list of A2A v1.0 parts."""
    for part in parts:
        text: str | None = part.get("text")
        if text:
            return text
    return ""


# ─── Streaming + spec method helpers ────────────────────────────────────────


async def _do_jsonrpc(
    url: str, method: str, params: dict[str, Any], *, timeout: float = 30.0,
) -> A2AResponse:
    """POST a JSON-RPC request and return the parsed envelope.

    Raises :class:`RuntimeError` if the server returns an ``error``
    result (the caller can inspect ``error.code`` / ``error.message``).
    """
    try:
        import httpx
    except ImportError:
        raise ImportError(
            "A2AAgent requires httpx. Install with: pip install httpx",
        ) from None

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": method,
        "params": params,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json", "A2A-Version": "1.0"},
        )
        resp.raise_for_status()
        data: A2AResponse = resp.json()
    if "error" in data and data["error"] is not None:
        err = data["error"]
        raise RuntimeError(
            f"A2A error {err.get('code')}: {err.get('message')}",
        )
    return data


def _sse_iter(stream_resp) -> AsyncIterator[dict[str, Any]]:
    """Iterate JSON-RPC envelopes carried in an SSE response.

    The server formats each envelope as ``data: {...}\\n\\n``.  We
    accumulate ``data:`` lines per event boundary (a blank line),
    parse the JSON, and yield the inner ``result`` dict.
    """
    async def _gen() -> AsyncIterator[dict[str, Any]]:
        buf: list[str] = []
        async for line in stream_resp.aiter_lines():
            if not line:
                if not buf:
                    continue
                joined = "".join(buf)
                buf.clear()
                try:
                    envelope = json.loads(joined)
                except json.JSONDecodeError:
                    continue
                result = envelope.get("result")
                if isinstance(result, dict):
                    yield result
                continue
            if line.startswith("data:"):
                buf.append(line[5:].lstrip())
    return _gen()


def _is_status_event(event: dict[str, Any]) -> bool:
    return "status" in event and "artifact" not in event


def _is_artifact_event(event: dict[str, Any]) -> bool:
    return "artifact" in event


def _convert_a2a_event_to_sdk(
    agent: BaseAgent, ctx: InvocationContext, event: dict[str, Any],
) -> Event | None:
    """Map one A2A streaming event to a single SDK :class:`Event`.

    Returns ``None`` for events that don't translate cleanly (e.g. the
    initial ``working`` status from the server, which is implicit in
    the parent ``AGENT_START``).
    """
    if _is_artifact_event(event):
        artifact = event.get("artifact", {}) or {}
        text = _extract_text_from_parts(artifact.get("parts") or [])
        # Tool-result artifacts are tagged in metadata by the server;
        # everything else is treated as a final answer chunk.
        meta = artifact.get("metadata") or {}
        if meta.get("tool_name"):
            return agent._emit_event(  # noqa: SLF001
                ctx,
                EventType.TOOL_RESPONSE,
                content=Content.from_text(text),
                metadata=meta,
            )
        if text:
            return agent._emit_event(  # noqa: SLF001
                ctx,
                EventType.AGENT_MESSAGE,
                content=Content.from_text(text),
            )
        return None
    if _is_status_event(event):
        status = event.get("status") or {}
        meta = event.get("metadata") or {}
        # Tool-call status with metadata.
        if meta.get("tool_name"):
            return agent._emit_event(  # noqa: SLF001
                ctx,
                EventType.TOOL_CALL,
                metadata=meta,
            )
        # Working / progress updates → THINKING (skip the very first one).
        if status.get("state") == "TASK_STATE_WORKING":
            return None
        return None
    return None


# ─── A2AAgent spec-method instance helpers ──────────────────────────────────
#
# The module-level helpers above are reusable; the instance methods
# wrap them with the agent's URL + timeout.


async def _agent_send_streaming(
    self: A2AAgent, text: str, ctx: InvocationContext,
) -> AsyncIterator[Event]:
    """``SendStreamingMessage`` → SSE → SDK events."""
    try:
        import httpx
    except ImportError:
        raise ImportError(
            "A2AAgent requires httpx. Install with: pip install httpx",
        ) from None

    from orxhestra.a2a.types import Message as A2AMessageModel
    from orxhestra.a2a.types import Part as A2APartModel
    from orxhestra.a2a.types import Role

    message_model = A2AMessageModel(
        message_id=str(uuid4()),
        role=Role.USER,
        parts=[A2APartModel(text=text, media_type="text/plain")],
    )
    if self.signing_key is not None and self.signing_did:
        from orxhestra.a2a.signing import sign_message as sign_a2a_message
        message_model = sign_a2a_message(
            message_model, self.signing_key, self.signing_did,
        )

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": "SendStreamingMessage",
        "params": {
            "message": message_model.model_dump(by_alias=True, exclude_none=True),
        },
    }

    async with httpx.AsyncClient(timeout=self.timeout) as client:
        async with client.stream(
            "POST",
            self.url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "A2A-Version": "1.0",
                "Accept": "text/event-stream",
            },
        ) as resp:
            resp.raise_for_status()
            async for a2a_event in _sse_iter(resp):
                sdk_event = _convert_a2a_event_to_sdk(self, ctx, a2a_event)
                if sdk_event is not None:
                    yield sdk_event


# Bind streaming helper as a method.
A2AAgent._send_message_streaming = _agent_send_streaming  # type: ignore[attr-defined]


async def _list_tasks(
    self: A2AAgent,
    *,
    context_id: str | None = None,
    state: str | None = None,
    limit: int = 50,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Call ``ListTasks`` and return the ``ListTasksResult`` dict."""
    params: dict[str, Any] = {"limit": limit}
    if context_id is not None:
        params["contextId"] = context_id
    if state is not None:
        params["state"] = state
    if cursor is not None:
        params["cursor"] = cursor
    data = await _do_jsonrpc(self.url, "ListTasks", params, timeout=self.timeout)
    return data.get("result", {}) or {}


async def _get_task(
    self: A2AAgent, task_id: str, *, history_length: int | None = None,
) -> A2ATask:
    params: dict[str, Any] = {"id": task_id}
    if history_length is not None:
        params["historyLength"] = history_length
    data = await _do_jsonrpc(self.url, "GetTask", params, timeout=self.timeout)
    return data.get("result", {}) or {}


async def _cancel_task(self: A2AAgent, task_id: str) -> A2ATask:
    data = await _do_jsonrpc(
        self.url, "CancelTask", {"id": task_id}, timeout=self.timeout,
    )
    return data.get("result", {}) or {}


async def _subscribe_to_task(
    self: A2AAgent, task_id: str,
) -> AsyncIterator[dict[str, Any]]:
    """Re-subscribe to an existing task's stream.

    Yields raw A2A streaming events (status / artifact dicts).
    Translation to SDK events is left to the caller — re-subscription
    is typically used outside of a normal ``astream`` flow.
    """
    try:
        import httpx
    except ImportError:
        raise ImportError(
            "A2AAgent requires httpx. Install with: pip install httpx",
        ) from None

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": "SubscribeToTask",
        "params": {"id": task_id},
    }
    async with httpx.AsyncClient(timeout=self.timeout) as client:
        async with client.stream(
            "POST",
            self.url,
            json=payload,
            headers={"Content-Type": "application/json", "A2A-Version": "1.0"},
        ) as resp:
            resp.raise_for_status()
            async for event in _sse_iter(resp):
                yield event


async def _fetch_agent_card(self: A2AAgent, *, extended: bool = False) -> dict[str, Any]:
    """Fetch the well-known agent card, or the extended variant."""
    if extended:
        data = await _do_jsonrpc(
            self.url, "GetExtendedAgentCard", {}, timeout=self.timeout,
        )
        return data.get("result", {}) or {}
    try:
        import httpx
    except ImportError:
        raise ImportError(
            "A2AAgent requires httpx. Install with: pip install httpx",
        ) from None
    async with httpx.AsyncClient(timeout=self.timeout) as client:
        resp = await client.get(f"{self.url}/.well-known/agent-card.json")
        resp.raise_for_status()
        return resp.json()


async def _set_push_notification_config(
    self: A2AAgent, task_id: str, config: dict[str, Any],
) -> dict[str, Any]:
    data = await _do_jsonrpc(
        self.url,
        "CreateTaskPushNotificationConfig",
        {"taskId": task_id, "config": config},
        timeout=self.timeout,
    )
    return data.get("result", {}) or {}


async def _get_push_notification_config(
    self: A2AAgent, task_id: str, *, config_id: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"taskId": task_id}
    if config_id is not None:
        params["configId"] = config_id
    data = await _do_jsonrpc(
        self.url, "GetTaskPushNotificationConfig", params, timeout=self.timeout,
    )
    return data.get("result", {}) or {}


async def _list_push_notification_configs(
    self: A2AAgent, task_id: str,
) -> list[dict[str, Any]]:
    data = await _do_jsonrpc(
        self.url,
        "ListTaskPushNotificationConfigs",
        {"taskId": task_id},
        timeout=self.timeout,
    )
    return data.get("result", []) or []


async def _delete_push_notification_config(
    self: A2AAgent, task_id: str, config_id: str,
) -> bool:
    data = await _do_jsonrpc(
        self.url,
        "DeleteTaskPushNotificationConfig",
        {"taskId": task_id, "configId": config_id},
        timeout=self.timeout,
    )
    result = data.get("result", {}) or {}
    return bool(result.get("deleted"))


# Bind every spec helper as an instance method.
A2AAgent.list_tasks = _list_tasks                                # type: ignore[attr-defined]
A2AAgent.get_task = _get_task                                    # type: ignore[attr-defined]
A2AAgent.cancel_task = _cancel_task                              # type: ignore[attr-defined]
A2AAgent.subscribe_to_task = _subscribe_to_task                  # type: ignore[attr-defined]
A2AAgent.fetch_agent_card = _fetch_agent_card                    # type: ignore[attr-defined]
A2AAgent.set_push_notification_config = _set_push_notification_config              # type: ignore[attr-defined]
A2AAgent.get_push_notification_config = _get_push_notification_config              # type: ignore[attr-defined]
A2AAgent.list_push_notification_configs = _list_push_notification_configs          # type: ignore[attr-defined]
A2AAgent.delete_push_notification_config = _delete_push_notification_config        # type: ignore[attr-defined]
