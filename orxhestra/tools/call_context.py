"""CallContext — high-level context exposed to tools and callbacks.

Wraps the low-level ``InvocationContext`` with a clean API for artifacts,
credentials, memory, and side-effects.  This is what tools receive
during execution.

Architecture::

    InvocationContext (low-level runtime)
        │
        └── CallContext (high-level tool API)
                ├── state         — shared mutable state
                ├── actions       — escalate, transfer, state_delta
                ├── artifacts     — save/load files and blobs
                ├── credentials   — save/load auth tokens
                ├── memory        — search/add long-term memory
                └── confirmation  — request user approval
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from orxhestra.events.event_actions import EventActions

if TYPE_CHECKING:
    from orxhestra.agents.invocation_context import InvocationContext

logger = logging.getLogger(__name__)


class CallContext:
    """Context object passed to tools during execution.

    Wraps ``Context`` with a tool-scoped ``EventActions`` and provides
    a high-level API for artifacts, credentials, and memory services.

    Parameters
    ----------
    ctx : InvocationContext
        The parent invocation context for this agent run.

    Attributes
    ----------
    state : dict[str, Any]
        Shared mutable state from the invocation context.
    actions : EventActions
        Side-effects the tool wants to signal.
    agent_name : str
        Name of the agent currently executing this tool.
    session_id : str
        Current session identifier.
    invocation_id : str
        Current invocation identifier.
    function_call_id : str, optional
        The ID of the current tool call.
    """

    def __init__(self, ctx: InvocationContext) -> None:
        self._ctx = ctx
        self.state: dict[str, Any] = ctx.state
        self.actions: EventActions = EventActions()
        self.agent_name: str = ctx.agent_name
        self.session_id: str = ctx.session_id
        self.invocation_id: str = ctx.invocation_id
        self.function_call_id: str | None = None
        self._confirmation_pending: bool = False

    @property
    def input_content(self) -> str | None:
        """The original input that started this invocation."""
        return self._ctx.input_content

    def end_invocation(self) -> None:
        """Kill switch — force-stop the entire invocation.

        Sets ``ctx.end_invocation = True`` which causes all agents
        (LlmAgent, LoopAgent, SequentialAgent) to stop immediately.
        """
        self._ctx.end_invocation = True

    def request_confirmation(self) -> None:
        """Request user confirmation before proceeding with this tool call.

        When called from a ``before_tool_callback``, the tool execution
        is paused and a confirmation request event is yielded to the
        caller.  The caller must approve or reject before execution
        continues.
        """
        self._confirmation_pending = True

    @property
    def confirmation_pending(self) -> bool:
        """Whether this tool has a pending confirmation request."""
        return self._confirmation_pending

    async def save_artifact(
        self,
        filename: str,
        data: bytes | str,
        *,
        mime_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
    ) -> int | None:
        """Save a file or blob to the artifact store.

        Parameters
        ----------
        filename : str
            Name of the artifact (e.g. ``"report.md"``).
        data : bytes or str
            The artifact content.  Strings are encoded as UTF-8.
        mime_type : str
            MIME type of the artifact.
        metadata : dict, optional
            Additional metadata stored with the artifact.

        Returns
        -------
        int or None
            The version number of the saved artifact, or ``None``
            if no artifact service is configured.
        """
        svc = getattr(self._ctx, "artifact_service", None)
        if svc is None:
            logger.debug("No artifact service configured — save_artifact skipped.")
            return None

        if isinstance(data, str):
            data = data.encode("utf-8")

        return await svc.save_artifact(
            app_name=self._ctx.app_name,
            user_id=self._ctx.user_id,
            session_id=self.session_id,
            filename=filename,
            data=data,
            mime_type=mime_type,
            metadata=metadata or {},
        )

    async def load_artifact(
        self,
        filename: str,
        *,
        version: int | None = None,
    ) -> bytes | None:
        """Load an artifact from the store.

        Parameters
        ----------
        filename : str
            Name of the artifact.
        version : int, optional
            Specific version to load.  ``None`` loads the latest.

        Returns
        -------
        bytes or None
            The artifact content, or ``None`` if not found or no
            artifact service is configured.
        """
        svc = getattr(self._ctx, "artifact_service", None)
        if svc is None:
            return None

        return await svc.load_artifact(
            app_name=self._ctx.app_name,
            user_id=self._ctx.user_id,
            session_id=self.session_id,
            filename=filename,
            version=version,
        )

    async def list_artifacts(self) -> list[str]:
        """List all artifact filenames in the current session.

        Returns
        -------
        list[str]
            Filenames of saved artifacts.
        """
        svc = getattr(self._ctx, "artifact_service", None)
        if svc is None:
            return []

        return await svc.list_artifact_keys(
            app_name=self._ctx.app_name,
            user_id=self._ctx.user_id,
            session_id=self.session_id,
        )

    async def save_credential(
        self,
        key: str,
        value: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a credential (API key, OAuth token, etc.).

        Parameters
        ----------
        key : str
            Credential identifier (e.g. ``"openai_api_key"``).
        value : str
            The credential value.
        metadata : dict, optional
            Additional metadata.
        """
        svc = getattr(self._ctx, "credential_service", None)
        if svc is None:
            logger.debug("No credential service configured — save_credential skipped.")
            return

        await svc.save_credential(
            app_name=self._ctx.app_name,
            user_id=self._ctx.user_id,
            key=key,
            value=value,
            metadata=metadata or {},
        )

    async def load_credential(self, key: str) -> str | None:
        """Load a credential by key.

        Parameters
        ----------
        key : str
            Credential identifier.

        Returns
        -------
        str or None
            The credential value, or ``None`` if not found.
        """
        svc = getattr(self._ctx, "credential_service", None)
        if svc is None:
            return None

        return await svc.load_credential(
            app_name=self._ctx.app_name,
            user_id=self._ctx.user_id,
            key=key,
        )

    async def search_memory(self, query: str) -> list[dict[str, Any]]:
        """Search long-term memory.

        Parameters
        ----------
        query : str
            The search query.

        Returns
        -------
        list[dict]
            Matching memory items.
        """
        svc = self._ctx.memory_service
        if svc is None:
            return []

        return await svc.search(
            app_name=self._ctx.app_name,
            user_id=self._ctx.user_id,
            query=query,
        )

    async def add_memory(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an item to long-term memory.

        Parameters
        ----------
        content : str
            The memory content.
        metadata : dict, optional
            Additional metadata.
        """
        svc = self._ctx.memory_service
        if svc is None:
            logger.debug("No memory service configured — add_memory skipped.")
            return

        await svc.add(
            app_name=self._ctx.app_name,
            user_id=self._ctx.user_id,
            content=content,
            metadata=metadata or {},
        )
