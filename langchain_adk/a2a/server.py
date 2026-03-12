"""A2AServer - expose any BaseAgent as a FastAPI streaming endpoint.

The server accepts POST /run requests and streams SDK events back as
Server-Sent Events (SSE). Each event is a JSON-encoded A2AEvent.

Examples
--------
>>> from agents.llm_agent import LlmAgent
>>> from a2a.server import A2AServer
>>> from sessions.in_memory_session_service import InMemorySessionService
>>>
>>> agent = LlmAgent("my_agent", llm=llm)
>>> server = A2AServer(agent, session_service=InMemorySessionService())
>>> app = server.as_fastapi_app()
>>>
>>> # Run with: uvicorn main:app
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from langchain_adk.a2a.converters import event_to_a2a
from langchain_adk.a2a.types import A2ARequest, A2AEvent
from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.context.invocation_context import InvocationContext
from langchain_adk.sessions.base_session_service import BaseSessionService


class A2AServer:
    """Adapts a BaseAgent as a FastAPI-based A2A endpoint.

    Attributes
    ----------
    agent : BaseAgent
        The agent to expose.
    session_service : BaseSessionService
        Service for creating/loading sessions.
    app_name : str
        Application name used for session scoping.
    """

    def __init__(
        self,
        agent: BaseAgent,
        *,
        session_service: BaseSessionService,
        app_name: str = "agent-sdk",
    ) -> None:
        self.agent = agent
        self.session_service = session_service
        self.app_name = app_name

    async def handle_request(self, request: A2ARequest) -> AsyncIterator[str]:
        """Run the agent and yield SSE-formatted event strings.

        Parameters
        ----------
        request : A2ARequest
            The inbound A2A request.

        Yields
        ------
        str
            SSE data lines: ``"data: <json>\\n\\n"``
        """
        # Get or create a session
        session_id = request.session_id
        if session_id:
            session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=request.user_id,
                session_id=session_id,
            )
        else:
            session = None

        if session is None:
            session = await self.session_service.create_session(
                app_name=self.app_name,
                user_id=request.user_id,
            )

        ctx = InvocationContext(
            session_id=session.id,
            user_id=request.user_id,
            app_name=self.app_name,
            agent_name=self.agent.name,
            state=dict(session.state),
        )

        async for event in self.agent.run_with_callbacks(request.message, ctx=ctx):
            a2a_event = event_to_a2a(event)
            yield f"data: {a2a_event.model_dump_json()}\n\n"

        # Persist updated state back to session
        await self.session_service.update_session(
            session.id,
            state=ctx.state,
        )

    def as_fastapi_app(self) -> FastAPI:
        """Build and return a FastAPI application.

        Routes:
            GET  /     - Health check
            POST /run  - Run agent, stream SSE events
            GET  /info - Agent metadata

        Returns
        -------
        FastAPI
            A configured FastAPI app ready for uvicorn.
        """
        app = FastAPI(title=f"{self.agent.name} A2A Server")
        server = self

        @app.get("/")
        async def health() -> dict:
            return {"status": "ok", "agent": server.agent.name}

        @app.get("/info")
        async def info() -> dict:
            return {
                "name": server.agent.name,
                "description": server.agent.description,
                "app_name": server.app_name,
            }

        @app.post("/run")
        async def run(request: A2ARequest) -> StreamingResponse:
            return StreamingResponse(
                server.handle_request(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        return app
