"""A2AAgent builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_adk.agents.a2a_agent import A2AAgent
from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.composer.errors import ComposerError
from langchain_adk.composer.schema import AgentDef, ComposeSpec

if TYPE_CHECKING:
    from langchain_adk.composer.builders.agents import Helpers


async def build(
    name: str,
    agent_def: AgentDef,
    spec: ComposeSpec,
    *,
    helpers: Helpers,
) -> BaseAgent:
    """Build an ``A2AAgent`` from a YAML definition."""
    if not agent_def.url:
        msg = f"A2A agent '{name}' must have a 'url' field"
        raise ComposerError(msg)

    return A2AAgent(
        name=name,
        url=agent_def.url,
        description=agent_def.description,
    )
