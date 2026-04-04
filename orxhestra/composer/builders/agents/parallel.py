"""ParallelAgent builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    """Build a ``ParallelAgent`` from a YAML definition.

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
        Constructed ``ParallelAgent``.

    Raises
    ------
    ComposerError
        If the agent definition has no ``agents`` list.
    """
    from orxhestra.agents.parallel_agent import ParallelAgent

    if not agent_def.agents:
        msg = f"parallel agent '{name}' must have an 'agents' list"
        raise ComposerError(msg)

    sub_agents = [await helpers.build_agent(n) for n in agent_def.agents]
    return ParallelAgent(
        name=name, agents=sub_agents, description=agent_def.description
    )
