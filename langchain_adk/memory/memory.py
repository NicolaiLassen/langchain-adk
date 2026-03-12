"""Memory model."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Memory(BaseModel):
    """Represents one memory.

    Attributes
    ----------
    content : str
        The main content of the memory.
    metadata : dict[str, Any]
        Optional metadata associated with the memory.
    id : str, optional
        The unique identifier of the memory.
    author : str, optional
        The author of the memory.
    timestamp : str, optional
        The timestamp when the original content of this memory happened.
        This string will be forwarded to LLM. Preferred format is ISO 8601.
    """

    content: str
    """The main content of the memory."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Optional metadata associated with the memory."""

    id: Optional[str] = None
    """The unique identifier of the memory."""

    author: Optional[str] = None
    """The author of the memory."""

    timestamp: Optional[str] = None
    """The timestamp when the original content of this memory happened.

    This string will be forwarded to LLM. Preferred format is ISO 8601 format.
    """
