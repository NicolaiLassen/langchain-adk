"""Invocation context - propagated through agent call chains.

Carries runtime state for a single agent invocation: who is running,
which session it belongs to, what ephemeral state has accumulated,
and how the run is configured (streaming mode, call limits).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from langchain_adk.sessions.session import Session


class InvocationContext(BaseModel):
    """Runtime context propagated through an agent's execution tree.

    Each agent invocation receives a context. Child agents (sub-agents,
    AgentTool) receive a derived context with an updated branch and
    agent_name - enabling parallel agent isolation and event attribution.

    Attributes
    ----------
    invocation_id : str
        Unique ID for this specific invocation.
    session_id : str
        The session this invocation belongs to.
    user_id : str
        The user who initiated the session.
    app_name : str
        The application running the agent.
    agent_name : str
        The name of the currently executing agent.
    branch : str
        Dot-separated path for nested execution (e.g. "root.child").
        Used to isolate parallel agent event streams.
    state : dict[str, Any]
        Mutable key-value store for cross-agent state within one run.
        Agents may read/write this freely; changes are not persisted
        automatically - the Runner handles persistence.
    session : Session, optional
        The session this invocation belongs to. Provides access to
        conversation history (events) and persisted state for multi-turn.
    run_config : RunConfig, optional
        Per-run configuration (streaming mode, LLM call limits). Set by
        the Runner before the first agent call.
    memory_service : Any, optional
        Optional memory service for long-term recall.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    invocation_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: str = ""
    app_name: str = ""
    agent_name: str
    branch: str = ""
    state: dict[str, Any] = Field(default_factory=dict)
    session: Any | None = None  # Session; Any to avoid circular import at runtime
    run_config: Any | None = None  # RunConfig; Any to avoid circular import at runtime
    memory_service: Any | None = None
    langchain_run_config: dict[str, Any] = Field(default_factory=dict)

    def derive(
        self,
        *,
        agent_name: str,
        branch_suffix: str = "",
    ) -> InvocationContext:
        """Create a child context for a sub-agent invocation.

        The child shares the same session, invocation ID, state reference,
        and run_config, but gets an updated agent_name and branch for
        attribution and isolation.

        Parameters
        ----------
        agent_name : str
            The name of the child agent.
        branch_suffix : str, optional
            Optional suffix appended to the branch path.
            Defaults to agent_name.

        Returns
        -------
        InvocationContext
            A new InvocationContext scoped to the child agent.
        """
        suffix = branch_suffix or agent_name
        new_branch = f"{self.branch}.{suffix}" if self.branch else suffix
        return self.model_copy(
            update={
                "agent_name": agent_name,
                "branch": new_branch,
                "state": self.state,       # shared reference - intentional
                "session": self.session,   # shared reference - intentional
                "run_config": self.run_config,
                "langchain_run_config": self.langchain_run_config,
            }
        )
