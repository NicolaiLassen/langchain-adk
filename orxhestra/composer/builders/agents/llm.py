"""LlmAgent builder."""

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
    """Build an ``LlmAgent`` from a YAML definition."""
    from orxhestra.agents.llm_agent import LlmAgent

    kwargs = await resolve_llm_kwargs(name, agent_def, spec, helpers=helpers)
    return LlmAgent(**kwargs)
