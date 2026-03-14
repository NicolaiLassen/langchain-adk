"""Runner - orchestrates agent execution with session management.

The Runner is the main entry point for running agents. It:

1. Fetches or creates the session via the session service.
2. Builds an ``InvocationContext`` from the session and run config.
3. Delegates to ``agent._run_with_callbacks()``.
4. Persists every event to the session via ``append_event()``.
5. Yields the event stream back to the caller.

Basic usage::

    from langchain_adk.runner import Runner
    from langchain_adk.sessions import InMemorySessionService

    runner = Runner(
        agent=my_agent,
        app_name="my_app",
        session_service=InMemorySessionService(),
    )

    async for event in runner.run_async(
        user_id="user_1",
        session_id="session_1",
        new_message="Hello!",
    ):
        if event.is_final_response():
            print(event.text)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from langchain_adk.agents.run_config import RunConfig
from langchain_adk.context.invocation_context import InvocationContext
from langchain_adk.events.event import Event, EventType
from langchain_adk.models.part import Content
from langchain_adk.sessions.base_session_service import BaseSessionService
from langchain_adk.sessions.session import Session

if TYPE_CHECKING:
    from langchain_adk.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class Runner:
    """Orchestrates a single agent with session persistence.

    Ties together an agent, a session service, and the invocation context.
    All events are automatically persisted to the session so that callers
    only need to consume the event stream.

    Parameters
    ----------
    agent : BaseAgent
        The root agent to run.
    app_name : str
        Application identifier. Used to namespace sessions.
    session_service : BaseSessionService
        Where sessions are stored and retrieved.

    Attributes
    ----------
    agent : BaseAgent
        The root agent.
    app_name : str
        Application name passed to the session service.
    session_service : BaseSessionService
        The backing session store.
    """

    def __init__(
        self,
        agent: BaseAgent,
        *,
        app_name: str,
        session_service: BaseSessionService,
    ) -> None:
        self.agent = agent
        self.app_name = app_name
        self.session_service = session_service

    async def get_or_create_session(
        self,
        *,
        user_id: str,
        session_id: str,
        initial_state: dict | None = None,
    ) -> Session:
        """Fetch an existing session or create a new one.

        Parameters
        ----------
        user_id : str
            The user who owns this session.
        session_id : str
            The session identifier.
        initial_state : dict, optional
            Initial state for the session if it is newly created.

        Returns
        -------
        Session
            The existing or newly created session.
        """
        session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if session is None:
            session = await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id,
                state=initial_state or {},
            )
        return session

    async def run_async(
        self,
        *,
        user_id: str,
        session_id: str,
        new_message: str,
        run_config: RunConfig | None = None,
    ) -> AsyncIterator[Event]:
        """Run the agent and yield its event stream.

        Fetches or creates the session, builds an ``InvocationContext``,
        delegates to the agent, and persists every event.

        Parameters
        ----------
        user_id : str
            The user sending the message.
        session_id : str
            The session to attach this run to.
        new_message : str
            The user's input message for this turn.
        run_config : RunConfig, optional
            Per-run configuration (streaming mode, call limits). Defaults
            to ``RunConfig()`` with non-streaming mode.

        Yields
        ------
        Event
            Every event emitted by the agent and its sub-agents.
        """
        resolved_config = run_config or RunConfig()

        session = await self.get_or_create_session(
            user_id=user_id,
            session_id=session_id,
        )

        ctx = InvocationContext(
            session_id=session.id,
            user_id=user_id,
            app_name=self.app_name,
            agent_name=self.agent.name,
            state=dict(session.state),
            session=session,
            run_config=resolved_config,
        )

        # Persist user message as an event (Google ADK style)
        user_event = Event(
            type=EventType.USER_MESSAGE,
            author="user",
            session_id=session.id,
            invocation_id=ctx.invocation_id,
            content=Content.from_text(new_message),
        )
        await self.session_service.append_event(session, user_event)

        logger.debug(
            "Runner starting: agent=%s session=%s user=%s",
            self.agent.name,
            session_id,
            user_id,
        )

        async for event in self.agent._run_with_callbacks(new_message, ctx=ctx):
            await self.session_service.append_event(session, event)
            yield event
