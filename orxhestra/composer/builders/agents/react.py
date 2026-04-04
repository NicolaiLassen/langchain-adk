"""ReActAgent builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.composer.builders.agents._common import resolve_llm_kwargs
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
    """Build a ``ReActAgent`` from a YAML definition.

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
        Constructed ``ReActAgent``.
    """
    from orxhestra.agents.react_agent import ReActAgent

    kwargs = await resolve_llm_kwargs(name, agent_def, spec, helpers=helpers)
    return ReActAgent(**kwargs)
