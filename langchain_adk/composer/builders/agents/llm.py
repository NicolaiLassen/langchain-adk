"""LlmAgent builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.composer.builders.agents._common import resolve_llm_kwargs
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
    """Build an ``LlmAgent`` from a YAML definition."""
    from langchain_adk.agents.llm_agent import LlmAgent

    kwargs = await resolve_llm_kwargs(name, agent_def, spec, helpers=helpers)
    return LlmAgent(**kwargs)
