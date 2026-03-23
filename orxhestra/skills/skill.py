"""Skill model - a named knowledge block injectable into agent prompts."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Skill(BaseModel):
    """A named knowledge block that extends an agent's capabilities.

    Skills can be injected into an agent's system prompt statically at
    construction time, or loaded dynamically at runtime via the
    `load_skill` tool.

    Attributes
    ----------
    id : str
        Unique skill identifier.
    name : str
        Short identifier used by the LLM to reference this skill.
    description : str
        One-line summary shown in the agent's skill roster.
    content : str
        The full instruction text injected when this skill loads.
    metadata : dict[str, Any]
        Optional tags/version/source info.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
