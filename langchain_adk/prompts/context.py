"""PromptContext - typed model for building system prompts."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class PromptContext(BaseModel):
    """Typed context for building an agent's system prompt.

    Pass this to build_system_prompt() to get a fully rendered prompt
    string. All fields are optional except agent_name - the catalog will
    only include non-empty sections.

    Attributes
    ----------
    agent_name : str
        Name of the agent (required).
    current_date : str
        Today's date, injected into the prompt.
    goal : str
        High-level goal or objective for this agent.
    instructions : str
        Core agent instructions / behaviour rules.
    skills : list[dict[str, Any]]
        Available skills shown in the prompt roster.
        Each entry: {"name": str, "description": str}
    agents : list[dict[str, Any]]
        Available sub-agents shown in the prompt roster.
        Each entry: {"name": str, "description": str}
    tasks : list[dict[str, Any]]
        Current tasks the agent must address.
        Each entry: {"tag": str, "title": str, "description": str}
    workflow_instructions : str
        Step-by-step workflow guidance text.
    extra_sections : list[str]
        Additional freeform text blocks appended at the end.
    """

    agent_name: str
    current_date: str = Field(default_factory=lambda: date.today().isoformat())
    goal: str = ""
    instructions: str = ""
    skills: list[dict[str, Any]] = Field(default_factory=list)
    agents: list[dict[str, Any]] = Field(default_factory=list)
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    workflow_instructions: str = ""
    extra_sections: list[str] = Field(default_factory=list)
