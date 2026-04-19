"""Spec-compliant A2A v1.0 server over JSON-RPC 2.0.

Exposes any :class:`BaseAgent` as an A2A endpoint with the **complete
v1.0 method set**:

================================================  =========================================
PascalCase                                        JSON-RPC slug
================================================  =========================================
``SendMessage``                                   ``message/send``
``SendStreamingMessage``                          ``message/stream``
``GetTask``                                       ``tasks/get``
``CancelTask``                                    ``tasks/cancel``
``ListTasks``                                     ``tasks/list``
``SubscribeToTask``                               ``tasks/resubscribe``
``GetExtendedAgentCard``                          ``agent/getAuthenticatedExtendedCard``
``CreateTaskPushNotificationConfig``              ``tasks/pushNotificationConfig/set``
``GetTaskPushNotificationConfig``                 ``tasks/pushNotificationConfig/get``
``ListTaskPushNotificationConfigs``               ``tasks/pushNotificationConfig/list``
``DeleteTaskPushNotificationConfig``              ``tasks/pushNotificationConfig/delete``
================================================  =========================================

Plus the discovery endpoint at ``GET /.well-known/agent-card.json``.

Three pluggable backends:

* :class:`~orxhestra.a2a.store.TaskStore` — task persistence.
  Default :class:`~orxhestra.a2a.store.InMemoryTaskStore` (LRU-bounded
  10 000 tasks).  Plug your own (sqlite, redis, postgres) for
  production scale.
* :class:`~orxhestra.a2a.notifications.PushNotificationStore` —
  registry of webhook configs per task.
* :class:`~orxhestra.a2a.notifications.PushNotificationDispatcher`
  — fires HTTP POSTs to registered webhooks on every task lifecycle
  event.  SSRF-guarded by default.

Optionally signs every outgoing agent message with Ed25519 and
verifies incoming signed messages against a
:class:`~orxhestra.security.did.DidResolver`.  Signing is **opt-in**
— when ``signing_key`` is unset the server behaves exactly as it did
before the identity layer existed.

Run with::

    uvicorn my_module:app

See Also
--------
orxhestra.agents.a2a_agent.A2AAgent : Client-side counterpart.
orxhestra.a2a.signing : Message signing / verification helpers.
orxhestra.a2a.store.TaskStore : Task persistence protocol.
orxhestra.a2a.notifications.PushNotificationDispatcher : Webhook delivery.
orxhestra.a2a.types.VerificationMethod : Published on agent cards.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as _exc:
    raise ImportError(
        "A2AServer requires FastAPI. Install with: pip install orxhestra[a2a]"
    ) from _exc

from orxhestra.a2a.converters import events_to_a2a_stream
from orxhestra.a2a.notifications import (
    InMemoryPushNotificationStore,
    PushNotificationDispatcher,
    PushNotificationStore,
)
from orxhestra.a2a.signing import (
    sign_message as sign_a2a_message,
)
from orxhestra.a2a.signing import (
    verify_message as verify_a2a_message,
)
from orxhestra.a2a.store import InMemoryTaskStore, TaskStore
from orxhestra.a2a.types import (
    TERMINAL_STATES,
    A2AErrorCode,
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Artifact,
    DeleteTaskPushNotificationConfigParams,
    GetTaskPushNotificationConfigParams,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    ListTaskPushNotificationConfigsParams,
    ListTasksParams,
    ListTasksResult,
    Message,
    MessageSendParams,
    Part,
    Role,
    SecurityScheme,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    VerificationMethod,
)
from orxhestra.agents.base_agent import BaseAgent
from orxhestra.agents.invocation_context import InvocationContext as Context
from orxhestra.sessions.base_session_service import BaseSessionService

# Local type alias to avoid a runtime dep on cryptography.
_Ed25519PrivateKey = Any
_DidResolver = Any


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Returns
    -------
    str
        Timezone-aware ISO 8601 timestamp — used to stamp
        :class:`TaskStatus` transitions.
    """
    return datetime.now(timezone.utc).isoformat()


class A2AServer:
    """Spec-compliant A2A v1.0 server that adapts a :class:`BaseAgent`.

    Implements:
      - ``SendMessage``            — run agent, return completed Task
      - ``SendStreamingMessage``   — run agent, stream SSE events
      - ``GetTask``                — retrieve task by ID
      - ``CancelTask``             — cancel a running task
      - Agent Card at ``/.well-known/agent-card.json``

    See Also
    --------
    A2AAgent : Client-side counterpart for calling remote servers.
    AgentCard : Discovery manifest served at the well-known URL.
    Task : Task object returned by ``SendMessage``.
    events_to_a2a_stream : Converter from SDK events to A2A events.

    Examples
    --------
    >>> from orxhestra import InMemorySessionService
    >>> from orxhestra.a2a.server import A2AServer
    >>> server = A2AServer(
    ...     agent=my_agent,
    ...     session_service=InMemorySessionService(),
    ...     url="http://localhost:8000",
    ... )
    >>> app = server.app  # FastAPI instance ready for uvicorn
    """

    def __init__(
        self,
        agent: BaseAgent,
        *,
        session_service: BaseSessionService,
        app_name: str = "agent-sdk",
        version: str = "1.0.0",
        url: str = "http://localhost:8000",
        skills: list[AgentSkill] | None = None,
        # Agent-card metadata (optional but recommended for public agents)
        provider: AgentProvider | None = None,
        documentation_url: str | None = None,
        icon_url: str | None = None,
        default_input_modes: list[str] | None = None,
        default_output_modes: list[str] | None = None,
        security_schemes: dict[str, SecurityScheme] | None = None,
        security_requirements: list[dict[str, list[str]]] | None = None,
        extended_card_extras: dict[str, Any] | None = None,
        # Pluggable backends
        task_store: TaskStore | None = None,
        push_notification_store: PushNotificationStore | None = None,
        push_notification_dispatcher: PushNotificationDispatcher | None = None,
        allow_insecure_webhooks: bool = False,
        # Identity / signing
        signing_key: _Ed25519PrivateKey | None = None,
        signer_did: str = "",
        require_signed: bool = False,
        resolver: _DidResolver | None = None,
    ) -> None:
        """Initialize the A2A server.

        Parameters
        ----------
        agent
            The agent to expose via A2A.
        session_service
            Session backend for conversation state.
        app_name
            Application name used in session creation.
        version
            Version string for the Agent Card.
        url
            Base URL where the server is reachable.
        skills
            Skills advertised in the Agent Card.
        provider
            Organization metadata published on the agent card.
        documentation_url, icon_url
            Optional links rendered by clients.
        default_input_modes, default_output_modes
            MIME types used when a skill does not override
            (default ``["text/plain"]``).
        security_schemes
            OpenAPI-style security scheme dict.  This release
            advertises the schemes on the card but does **not**
            enforce them — credential extraction lives in middleware.
        security_requirements
            OpenAPI-style requirement list.
        extended_card_extras
            Extra fields merged into the response of
            ``GetExtendedAgentCard``.  Use to publish authenticated-
            only metadata such as private skill descriptions.
        task_store
            Task persistence backend.  Defaults to
            :class:`InMemoryTaskStore` (10 000-task LRU).
        push_notification_store
            Backing store for push-notification configs.  Defaults to
            :class:`InMemoryPushNotificationStore`.
        push_notification_dispatcher
            Dispatcher that POSTs events to webhooks.  Defaults to a
            new :class:`PushNotificationDispatcher` using the store
            above.
        allow_insecure_webhooks
            Pass-through to the default dispatcher's SSRF guard.  Set
            ``True`` only for local development.
        signing_key
            When set, the server signs every outgoing A2A message and
            publishes its :class:`VerificationMethod` in the agent
            card.  Requires ``orxhestra[auth]``.
        signer_did
            DID matching ``signing_key``.  Required when
            ``signing_key`` is set.
        require_signed
            When ``True``, incoming messages without a valid signature
            are rejected with ``INVALID_REQUEST``.  Defaults to
            ``False`` for back-compat with unsigned peers.
        resolver
            Resolver used to verify incoming signatures.  Defaults to
            a ``did:key`` resolver.
        """
        self.agent = agent
        self.session_service = session_service
        self.app_name = app_name
        self.version = version
        self.url = url
        self.skills = skills or []
        self.provider = provider
        self.documentation_url = documentation_url
        self.icon_url = icon_url
        self.default_input_modes = default_input_modes or ["text/plain"]
        self.default_output_modes = default_output_modes or ["text/plain"]
        self.security_schemes = security_schemes
        self.security_requirements = security_requirements
        self.extended_card_extras = extended_card_extras or {}
        self.signing_key = signing_key
        self.signer_did = signer_did
        self.require_signed = require_signed
        self._resolver = resolver

        # Pluggable backends.
        self._task_store: TaskStore = task_store or InMemoryTaskStore()
        self._push_store: PushNotificationStore = (
            push_notification_store or InMemoryPushNotificationStore()
        )
        self._push_dispatcher: PushNotificationDispatcher = (
            push_notification_dispatcher
            or PushNotificationDispatcher(
                self._push_store,
                allow_insecure_webhooks=allow_insecure_webhooks,
            )
        )


    def _build_agent_card(self) -> AgentCard:
        """Return the :class:`AgentCard` published at the well-known URL.

        When ``signing_key`` is set, derives the matching ``did:key``
        and publishes a :class:`VerificationMethod` that remote peers
        can resolve to verify signed responses.  For ``did:web``
        identities the fragment falls back to ``#key-1`` since the
        spec doesn't mandate a canonical derivation.

        Returns
        -------
        AgentCard
            Fully-populated card ready for
            :meth:`pydantic.BaseModel.model_dump`.
        """
        verification_methods: list[VerificationMethod] | None = None
        controller: str | None = None
        if self.signing_key is not None and self.signer_did:
            import base58

            from orxhestra.security.crypto import (
                did_key_fragment,
                public_key_to_did_key,
                serialize_public_key,
            )

            public_key = self.signing_key.public_key()
            multibase = "z" + base58.b58encode(
                bytes([0xED, 0x01]) + serialize_public_key(public_key),
            ).decode("ascii")
            controller = self.signer_did
            try:
                fragment = did_key_fragment(self.signer_did)
            except ValueError:
                # did:web or other — fall back to fixed fragment.
                fragment = "#key-1"
            # Validate the advertised DID matches the signing key.
            derived_did = public_key_to_did_key(public_key)
            if self.signer_did.startswith("did:key:") and self.signer_did != derived_did:
                controller = derived_did
            verification_methods = [
                VerificationMethod(
                    id=f"{controller}{fragment}",
                    type="Ed25519VerificationKey2020",
                    controller=controller,
                    public_key_multibase=multibase,
                ),
            ]

        return AgentCard(
            name=self.agent.name,
            description=self.agent.description or "An AI agent exposed via A2A.",
            supported_interfaces=[
                AgentInterface(
                    url=self.url,
                    protocol_binding="JSONRPC",
                    protocol_version="1.0",
                ),
            ],
            version=self.version,
            protocol_version="1.0",
            capabilities=AgentCapabilities(
                streaming=True,
                push_notifications=True,
                extended_agent_card=bool(self.extended_card_extras),
            ),
            skills=self.skills,
            default_input_modes=self.default_input_modes,
            default_output_modes=self.default_output_modes,
            security_schemes=self.security_schemes,
            security_requirements=self.security_requirements,
            provider=self.provider,
            documentation_url=self.documentation_url,
            icon_url=self.icon_url,
            controller=controller,
            verification_method=verification_methods,
        )

    def _default_resolver(self):
        """Return the lazily-constructed default :class:`~orxhestra.security.did.DidResolver`.

        When no resolver was supplied at construction, a
        :class:`~orxhestra.security.did.DidKeyResolver` is created on first use and cached
        for subsequent calls.

        Returns
        -------
        DidResolver
        """
        if self._resolver is not None:
            return self._resolver
        from orxhestra.security.did import DidKeyResolver

        self._resolver = DidKeyResolver()
        return self._resolver

    def _maybe_sign(self, message: Message) -> Message:
        """Sign ``message`` when the server has a signing identity configured.

        Parameters
        ----------
        message : Message
            Outgoing message to stamp.

        Returns
        -------
        Message
            ``message`` unchanged when no signing key is configured;
            otherwise a copy carrying a detached Ed25519 signature via
            :func:`orxhestra.a2a.signing.sign_message`.
        """
        if self.signing_key is None or not self.signer_did:
            return message
        return sign_a2a_message(message, self.signing_key, self.signer_did)


    async def _create_task(
        self, message: Message, context_id: str | None = None,
    ) -> Task:
        """Create, store, and return a new :class:`Task` for ``message``.

        Parameters
        ----------
        message
            Initial user message kicking off the task.
        context_id
            Conversation identifier carried on the task.  A fresh
            UUID is generated when omitted.

        Returns
        -------
        Task
            Newly-registered task in ``SUBMITTED`` state.  Persisted
            via :attr:`_task_store` (eviction policy lives in the
            store implementation).
        """
        task = Task(
            id=str(uuid.uuid4()),
            context_id=context_id or str(uuid.uuid4()),
            status=TaskStatus(state=TaskState.SUBMITTED, timestamp=_now_iso()),
            history=[message],
        )
        await self._task_store.put(task)
        return task

    async def _update_task_status(
        self,
        task: Task,
        state: TaskState,
        agent_message: Message | None = None,
    ) -> None:
        """Update ``task.status`` and persist the change.

        Parameters
        ----------
        task
            Task whose status is changing.
        state
            New lifecycle state.
        agent_message
            Latest agent message to attach to the status snapshot.
        """
        task.status = TaskStatus(
            state=state,
            message=agent_message,
            timestamp=_now_iso(),
        )
        await self._task_store.put(task)

    async def _run_agent_for_task(
        self,
        task: Task,
        user_message: str,
    ) -> None:
        """Run the agent, collect the final answer, and sign the response.

        Parameters
        ----------
        task : Task
            Task object to mutate in place with status updates,
            artifacts, and history.
        user_message : str
            Plain text extracted from the incoming user message.

        Notes
        -----
        The constructed agent response message passes through
        :meth:`_maybe_sign` so it inherits the server's signing
        identity when configured.
        """
        session = await self.session_service.create_session(
            app_name=self.app_name,
            user_id="anonymous",
        )
        ctx = Context(
            session_id=session.id,
            user_id="anonymous",
            app_name=self.app_name,
            agent_name=self.agent.name,
            state=dict(session.state),
            session=session,
        )

        await self._update_task_status(task, TaskState.WORKING)

        final_answer = ""
        async for event in self.agent.astream(user_message, ctx=ctx):
            if event.is_final_response():
                final_answer = event.text

        # Build agent response message
        agent_msg = Message(
            role=Role.AGENT,
            parts=[Part(text=final_answer, media_type="text/plain")],
            task_id=task.id,
            context_id=task.context_id,
        )
        agent_msg = self._maybe_sign(agent_msg)

        # Add artifact
        artifact = Artifact(parts=[Part(text=final_answer, media_type="text/plain")])
        task.artifacts = [artifact]

        if task.history is not None:
            task.history.append(agent_msg)

        await self._update_task_status(
            task, TaskState.COMPLETED, agent_message=agent_msg,
        )


    async def _handle_send_message(
        self, params: dict[str, Any], request_id: Any,
    ) -> JSONResponse:
        send_params = MessageSendParams.model_validate(params)
        rejection = await self._verify_incoming(send_params.message, request_id)
        if rejection is not None:
            return rejection
        user_text = _extract_text(send_params.message)
        task = await self._create_task(
            send_params.message,
            context_id=send_params.message.context_id,
        )

        await self._run_agent_for_task(task, user_text)

        return _jsonrpc_success(request_id, task)

    async def _handle_send_streaming_message(
        self, params: dict[str, Any], request_id: Any,
    ) -> StreamingResponse | JSONResponse:
        send_params = MessageSendParams.model_validate(params)
        rejection = await self._verify_incoming(send_params.message, request_id)
        if rejection is not None:
            return rejection
        user_text = _extract_text(send_params.message)
        task = await self._create_task(
            send_params.message,
            context_id=send_params.message.context_id,
        )

        return StreamingResponse(
            self._stream_task(task, user_text, request_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _verify_incoming(
        self, message: Message, request_id: Any,
    ) -> JSONResponse | None:
        """Enforce ``require_signed`` on an incoming message.

        Parameters
        ----------
        message : Message
            Incoming message extracted from the JSON-RPC params.
        request_id : Any
            Correlation id used for error envelopes.

        Returns
        -------
        JSONResponse or None
            ``None`` when the message is acceptable.  A JSON-RPC
            error response (with :data:`A2AErrorCode.INVALID_REQUEST`)
            when ``require_signed`` is set and verification failed.
        """
        if not self.require_signed:
            return None
        if await verify_a2a_message(message, self._default_resolver()):
            return None
        return _jsonrpc_error(
            request_id,
            A2AErrorCode.INVALID_REQUEST,
            "Message signature missing or invalid; server requires signed messages.",
        )

    async def _stream_task(
        self, task: Task, user_text: str, request_id: Any,
    ) -> AsyncIterator[str]:
        """Run the agent and yield SSE lines as JSON-RPC responses.

        Parameters
        ----------
        task : Task
            The task to stream updates for.
        user_text : str
            Plain text extracted from the incoming user message.
        request_id : Any
            Correlation id echoed on each streamed envelope.

        Yields
        ------
        str
            ``data: {...}\\n\\n`` SSE frames containing
            :class:`TaskStatusUpdateEvent` /
            :class:`TaskArtifactUpdateEvent` payloads wrapped in
            JSON-RPC responses.
        """
        session = await self.session_service.create_session(
            app_name=self.app_name,
            user_id="anonymous",
        )
        ctx = Context(
            session_id=session.id,
            user_id="anonymous",
            app_name=self.app_name,
            agent_name=self.agent.name,
            state=dict(session.state),
            session=session,
        )

        await self._update_task_status(task, TaskState.WORKING)

        # Emit initial "working" status
        working_event = TaskStatusUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            status=TaskStatus(state=TaskState.WORKING, timestamp=_now_iso()),
            final=False,
        )
        await self._push_dispatcher.dispatch(task, working_event)
        yield _sse_line(request_id, working_event)

        # Stream agent events, converting to A2A events
        async for a2a_event in events_to_a2a_stream(
            self.agent.astream(user_text, ctx=ctx),
            task_id=task.id,
            context_id=task.context_id,
        ):
            await self._push_dispatcher.dispatch(task, a2a_event)
            yield _sse_line(request_id, a2a_event)

        # Emit final "completed" status
        await self._update_task_status(task, TaskState.COMPLETED)

        completed_event = TaskStatusUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            status=TaskStatus(state=TaskState.COMPLETED, timestamp=_now_iso()),
            final=True,
        )
        await self._push_dispatcher.dispatch(task, completed_event)
        yield _sse_line(request_id, completed_event)

    async def _replay_task_stream(
        self, task: Task, request_id: Any,
    ) -> AsyncIterator[str]:
        """Replay a snapshot of an in-progress / completed task as SSE.

        Used by ``SubscribeToTask`` (tasks/resubscribe).  Emits the
        current status, then any artifacts, then a ``final=True``
        status for terminal tasks.  Live re-attachment to an
        in-progress task's *future* events is out of scope for the
        in-memory store; users running a persistent store can layer
        their own pub/sub on top of :meth:`TaskStore.put`.
        """
        # Current status snapshot
        yield _sse_line(
            request_id,
            TaskStatusUpdateEvent(
                task_id=task.id,
                context_id=task.context_id,
                status=task.status,
                final=task.status.state in TERMINAL_STATES,
            ),
        )
        for artifact in task.artifacts or []:
            yield _sse_line(
                request_id,
                TaskArtifactUpdateEvent(
                    task_id=task.id,
                    context_id=task.context_id,
                    artifact=artifact,
                    last_chunk=True,
                ),
            )

    async def _handle_get_task(
        self, params: dict[str, Any], request_id: Any,
    ) -> JSONResponse:
        query = TaskQueryParams.model_validate(params)
        task = await self._task_store.get(query.id)
        if task is None:
            return _jsonrpc_error(
                request_id,
                A2AErrorCode.TASK_NOT_FOUND,
                f"Task {query.id} not found",
            )
        return _jsonrpc_success(request_id, task)

    async def _handle_cancel_task(
        self, params: dict[str, Any], request_id: Any,
    ) -> JSONResponse:
        task_params = TaskIdParams.model_validate(params)
        task = await self._task_store.get(task_params.id)
        if task is None:
            return _jsonrpc_error(
                request_id,
                A2AErrorCode.TASK_NOT_FOUND,
                f"Task {task_params.id} not found",
            )
        if task.status.state in TERMINAL_STATES:
            return _jsonrpc_error(
                request_id,
                A2AErrorCode.TASK_NOT_CANCELABLE,
                f"Task {task_params.id} is in terminal state {task.status.state.value}",
            )
        await self._update_task_status(task, TaskState.CANCELED)
        return _jsonrpc_success(request_id, task)

    # ─── New v1.0 methods ───────────────────────────────────────────────────

    async def _handle_list_tasks(
        self, params: dict[str, Any], request_id: Any,
    ) -> JSONResponse:
        query = ListTasksParams.model_validate(params)
        tasks, next_cursor = await self._task_store.list(
            context_id=query.context_id,
            state=query.state,
            limit=query.limit,
            cursor=query.cursor,
        )
        return _jsonrpc_success(
            request_id, ListTasksResult(tasks=tasks, next_cursor=next_cursor),
        )

    async def _handle_subscribe_to_task(
        self, params: dict[str, Any], request_id: Any,
    ) -> StreamingResponse | JSONResponse:
        task_params = TaskIdParams.model_validate(params)
        task = await self._task_store.get(task_params.id)
        if task is None:
            return _jsonrpc_error(
                request_id,
                A2AErrorCode.TASK_NOT_FOUND,
                f"Task {task_params.id} not found",
            )
        return StreamingResponse(
            self._replay_task_stream(task, request_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _handle_get_extended_agent_card(
        self, params: dict[str, Any], request_id: Any,
    ) -> JSONResponse:
        # Extended card = base card + caller-supplied extras dict.  The
        # spec leaves the extra-fields shape open; we merge into the
        # serialised card so anything the user passed in
        # ``extended_card_extras`` shows up at the top level.
        card = self._build_agent_card().model_dump(
            by_alias=True, exclude_none=True,
        )
        card.update(self.extended_card_extras)
        return _jsonrpc_success(request_id, card)

    async def _handle_create_push_notification_config(
        self, params: dict[str, Any], request_id: Any,
    ) -> JSONResponse:
        cfg = TaskPushNotificationConfig.model_validate(params)
        if await self._task_store.get(cfg.task_id) is None:
            return _jsonrpc_error(
                request_id,
                A2AErrorCode.TASK_NOT_FOUND,
                f"Task {cfg.task_id} not found",
            )
        await self._push_store.set(cfg.task_id, cfg.config)
        return _jsonrpc_success(request_id, cfg)

    async def _handle_get_push_notification_config(
        self, params: dict[str, Any], request_id: Any,
    ) -> JSONResponse:
        query = GetTaskPushNotificationConfigParams.model_validate(params)
        config = await self._push_store.get(query.task_id, query.config_id)
        if config is None:
            return _jsonrpc_error(
                request_id,
                A2AErrorCode.TASK_NOT_FOUND,
                f"No push-notification config for task {query.task_id}",
            )
        return _jsonrpc_success(
            request_id,
            TaskPushNotificationConfig(task_id=query.task_id, config=config),
        )

    async def _handle_list_push_notification_configs(
        self, params: dict[str, Any], request_id: Any,
    ) -> JSONResponse:
        query = ListTaskPushNotificationConfigsParams.model_validate(params)
        configs = await self._push_store.list(query.task_id)
        return _jsonrpc_success(
            request_id,
            [
                TaskPushNotificationConfig(task_id=query.task_id, config=c)
                for c in configs
            ],
        )

    async def _handle_delete_push_notification_config(
        self, params: dict[str, Any], request_id: Any,
    ) -> JSONResponse:
        query = DeleteTaskPushNotificationConfigParams.model_validate(params)
        deleted = await self._push_store.delete(query.task_id, query.config_id)
        return _jsonrpc_success(request_id, {"deleted": deleted})


    _METHOD_MAP = {
        # Core (v1.0)
        "SendMessage": "_handle_send_message",
        "SendStreamingMessage": "_handle_send_streaming_message",
        "GetTask": "_handle_get_task",
        "CancelTask": "_handle_cancel_task",
        "ListTasks": "_handle_list_tasks",
        "SubscribeToTask": "_handle_subscribe_to_task",
        "GetExtendedAgentCard": "_handle_get_extended_agent_card",
        # Push-notification config CRUD (v1.0)
        "CreateTaskPushNotificationConfig": "_handle_create_push_notification_config",
        "GetTaskPushNotificationConfig": "_handle_get_push_notification_config",
        "ListTaskPushNotificationConfigs": "_handle_list_push_notification_configs",
        "DeleteTaskPushNotificationConfig": "_handle_delete_push_notification_config",
        # JSON-RPC slug aliases (per the v1.0 spec's binding table)
        "message/send": "_handle_send_message",
        "message/stream": "_handle_send_streaming_message",
        "tasks/get": "_handle_get_task",
        "tasks/cancel": "_handle_cancel_task",
        "tasks/list": "_handle_list_tasks",
        "tasks/resubscribe": "_handle_subscribe_to_task",
        "agent/getAuthenticatedExtendedCard": "_handle_get_extended_agent_card",
        "tasks/pushNotificationConfig/set": "_handle_create_push_notification_config",
        "tasks/pushNotificationConfig/get": "_handle_get_push_notification_config",
        "tasks/pushNotificationConfig/list": "_handle_list_push_notification_configs",
        "tasks/pushNotificationConfig/delete": "_handle_delete_push_notification_config",
    }

    async def _dispatch(self, request: Request) -> Any:
        """Parse a JSON-RPC envelope and route to the matching handler.

        Parameters
        ----------
        request : Request
            FastAPI request containing a JSON-RPC 2.0 body.

        Returns
        -------
        Any
            :class:`JSONResponse` or :class:`StreamingResponse`
            depending on the method invoked.  Always a valid JSON-RPC
            response — parse errors and unknown methods are mapped to
            error envelopes rather than raised.
        """
        try:
            body = await request.json()
        except Exception:
            return _jsonrpc_error(None, A2AErrorCode.PARSE_ERROR, "Invalid JSON")

        try:
            rpc = JSONRPCRequest.model_validate(body)
        except Exception:
            return _jsonrpc_error(
                body.get("id"), A2AErrorCode.INVALID_REQUEST, "Invalid JSON-RPC request",
            )

        handler_name = self._METHOD_MAP.get(rpc.method)
        if handler_name is None:
            return _jsonrpc_error(
                rpc.id, A2AErrorCode.METHOD_NOT_FOUND, f"Method {rpc.method!r} not found",
            )

        handler = getattr(self, handler_name)
        return await handler(rpc.params or {}, rpc.id)


    def as_fastapi_app(self) -> FastAPI:
        """Build and return a FastAPI application with A2A v1.0 routes.

        Returns
        -------
        FastAPI
            Configured application with JSON-RPC and Agent Card endpoints.
        """
        app = FastAPI(title=f"{self.agent.name} A2A Server")
        server = self

        @app.get("/.well-known/agent-card.json")
        @app.get("/.well-known/agent.json")
        async def agent_card() -> dict:
            card = server._build_agent_card()
            return card.model_dump(by_alias=True, exclude_none=True)

        @app.post("/")
        async def jsonrpc_endpoint(request: Request) -> Any:
            return await server._dispatch(request)

        @app.get("/")
        async def health() -> dict:
            return {"status": "ok", "agent": server.agent.name}

        return app




def _extract_text(message: Message) -> str:
    """Extract plain text from a Message's parts.

    Parameters
    ----------
    message : Message
        The A2A message to extract text from.

    Returns
    -------
    str
        Concatenated text from all text parts, or empty string.
    """
    texts: list[str] = []
    for part in message.parts:
        if part.text is not None:
            texts.append(part.text)
    return " ".join(texts) if texts else ""


def _jsonrpc_success(request_id: Any, result: Any) -> JSONResponse:
    """Wrap a result in a JSON-RPC 2.0 success response.

    Parameters
    ----------
    request_id : str | int | None
        The JSON-RPC request ID to echo back.
    result : dict
        Payload to include as the ``result`` field.

    Returns
    -------
    JSONResponse
        FastAPI JSON response with the JSON-RPC envelope.
    """
    if hasattr(result, "model_dump"):
        result = result.model_dump(by_alias=True, exclude_none=True)
    resp = JSONRPCResponse(id=request_id, result=result)
    return JSONResponse(resp.model_dump(by_alias=True, exclude_none=True))


def _jsonrpc_error(request_id: Any, code: int, message: str) -> JSONResponse:
    """Wrap an error in a JSON-RPC 2.0 error response.

    Parameters
    ----------
    request_id : str | int | None
        The JSON-RPC request ID to echo back.
    code : int
        Numeric error code (see ``A2AErrorCode``).
    message : str
        Human-readable error description.

    Returns
    -------
    JSONResponse
        FastAPI JSON response with the JSON-RPC error envelope.
    """
    resp = JSONRPCResponse(
        id=request_id or 0,
        error=JSONRPCError(code=code, message=message),
    )
    return JSONResponse(
        resp.model_dump(by_alias=True, exclude_none=True),
        status_code=400 if code != A2AErrorCode.INTERNAL_ERROR else 500,
    )


def _sse_line(request_id: Any, event: Any) -> str:
    """Format an A2A event as a Server-Sent Events data line.

    Parameters
    ----------
    request_id : str | int | None
        The JSON-RPC request ID to echo back.
    event : Event
        A2A event (e.g. ``TaskStatusUpdateEvent``) to serialize.

    Returns
    -------
    str
        SSE-formatted ``data: ...`` line with trailing double newline.
    """
    if hasattr(event, "model_dump"):
        result_data = event.model_dump(by_alias=True, exclude_none=True)
    else:
        result_data = event
    resp = JSONRPCResponse(id=request_id, result=result_data)
    return f"data: {json.dumps(resp.model_dump(by_alias=True, exclude_none=True))}\n\n"
