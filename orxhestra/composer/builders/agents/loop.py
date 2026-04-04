"""LoopAgent builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    """Build a ``LoopAgent`` from a YAML definition.

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
        Constructed ``LoopAgent``.
    """
    from orxhestra.agents.loop_agent import LoopAgent
    from orxhestra.composer.builders.tools import import_object

    if not agent_def.agents:
        msg = f"loop agent '{name}' must have an 'agents' list"
        raise ComposerError(msg)

    sub_agents = [await helpers.build_agent(n) for n in agent_def.agents]
    kwargs: dict[str, Any] = {
        "name": name,
        "agents": sub_agents,
        "description": agent_def.description,
        "max_iterations": agent_def.max_iterations
        or spec.defaults.max_iterations,
    }
    if agent_def.should_continue:
        kwargs["should_continue"] = import_object(agent_def.should_continue)

    return LoopAgent(**kwargs)
