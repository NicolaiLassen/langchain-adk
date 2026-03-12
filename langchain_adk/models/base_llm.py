"""BaseLlm - abstract base class for LangChain model wrappers.

ADK defines a ``BaseLlm`` that every model backend (Gemini, Anthropic, LiteLLM)
implements via ``generate_content_async()``. We mirror this for LangChain:
``BaseLlm`` wraps any ``BaseChatModel`` and exposes a single async generator
that yields ``LlmResponse`` objects.

This decouples agents from LangChain internals. Agents only interact with
``LlmRequest`` and ``LlmResponse`` - the model backend is swappable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

from pydantic import BaseModel, ConfigDict

from langchain_adk.models.llm_request import LlmRequest
from langchain_adk.models.llm_response import LlmResponse


class BaseLlm(ABC, BaseModel):
    """Abstract base class for all LLM wrappers.

    Subclasses implement ``generate_async()`` to call a specific LangChain
    chat model and yield ``LlmResponse`` objects.  Agents call this instead
    of touching LangChain APIs directly.

    Parameters
    ----------
    model : str
        The model identifier string (e.g. ``"gpt-4o"``, ``"claude-3-5-sonnet"``).

    Attributes
    ----------
    model : str
        The model identifier used for this instance.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str

    @classmethod
    def supported_models(cls) -> list[str]:
        """Return regex patterns for model names this backend handles.

        Used by ``LlmRegistry`` to route model name strings to the right
        backend class.

        Returns
        -------
        list[str]
            List of regex patterns.  Empty list means not auto-registered.
        """
        return []

    @abstractmethod
    async def generate_async(
        self,
        request: LlmRequest,
        *,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        """Call the model and yield one or more response chunks.

        Parameters
        ----------
        request : LlmRequest
            The fully built request containing messages, tools, and config.
        stream : bool, optional
            When True, yield partial chunks as they arrive. The final chunk
            always has ``partial=False``.

        Yields
        ------
        LlmResponse
            One response per model chunk (streaming) or a single complete
            response (non-streaming).
        """
        ...

    def __repr__(self) -> str:
        """Return a short string representation.

        Returns
        -------
        str
        """
        return f"{self.__class__.__name__}(model={self.model!r})"
