"""A2AAgent builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from orxhestra.agents.a2a_agent import A2AAgent
from orxhestra.agents.base_agent import BaseAgent
from orxhestra.composer.errors import ComposerError
from orxhestra.composer.schema import AgentDef, ComposeSpec

if TYPE_CHECKING:
    from orxhestra.composer.builders.agents import Helpers


async def build(
    name: str,
    agent_def: AgentDef,
    spec: ComposeSpec,
    *,
    helpers: Helpers,
) -> BaseAgent:
    """Build an ``A2AAgent`` from a YAML definition.

    Parameters
    ----------
    name : str
        Agent name.
    agent_def : AgentDef
        YAML agent definition.
    spec : ComposeSpec
        Full compose specification.
    helpers : Helpers
        Builder dependencies.

    Returns
    -------
    BaseAgent
        Constructed ``A2AAgent``.
    """
    if not agent_def.url:
        msg = f"A2A agent '{name}' must have a 'url' field"
        raise ComposerError(msg)

    return A2AAgent(
        name=name,
        url=agent_def.url,
        description=agent_def.description,
    )
