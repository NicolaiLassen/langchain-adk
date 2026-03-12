"""LlmRegistry - route model name strings to the right BaseLlm backend.

Each ``BaseLlm`` subclass declares which model
name patterns it supports via ``supported_models()`` (a list of regexes).
``LlmRegistry.new_llm()`` finds the first matching backend and instantiates it.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_adk.models.base_llm import BaseLlm

logger = logging.getLogger("langchain_adk.models.registry")

_registry: dict[str, type[BaseLlm]] = {}
"""Map of model-name regex -> BaseLlm subclass."""


class LlmRegistry:
    """Routes model name strings to registered BaseLlm backends.

    Backends register themselves by calling ``LlmRegistry.register()`` with
    a regex that matches model names they support. ``new_llm()`` searches the
    registry and instantiates the first match.

    Examples
    --------
    Register a backend::

        LlmRegistry.register(r"gpt-.*", OpenAiLlm)

    Instantiate by name::

        llm = LlmRegistry.new_llm("gpt-4o")
    """

    @staticmethod
    def register(pattern: str, llm_cls: type[BaseLlm]) -> None:
        """Register a backend for a model name pattern.

        Parameters
        ----------
        pattern : str
            Regex matched against the full model name string.
        llm_cls : type[BaseLlm]
            The backend class to instantiate when pattern matches.
        """
        _registry[pattern] = llm_cls
        logger.debug("Registered LLM backend %s for pattern %r", llm_cls.__name__, pattern)

    @staticmethod
    def resolve(model: str) -> type[BaseLlm]:
        """Find the backend class for a model name.

        Parameters
        ----------
        model : str
            The model name to look up (e.g. ``"gpt-4o"``).

        Returns
        -------
        type[BaseLlm]
            The first registered backend whose pattern matches ``model``.

        Raises
        ------
        ValueError
            If no registered backend matches the model name.
        """
        for pattern, cls in _registry.items():
            if re.fullmatch(pattern, model):
                return cls
        raise ValueError(
            f"No registered LLM backend for model {model!r}. "
            f"Registered patterns: {list(_registry)}"
        )

    @staticmethod
    def new_llm(model: str) -> BaseLlm:
        """Instantiate the matching backend for a model name.

        Parameters
        ----------
        model : str
            The model name to instantiate (e.g. ``"gpt-4o"``).

        Returns
        -------
        BaseLlm
            A ready-to-use LLM backend instance.
        """
        cls = LlmRegistry.resolve(model)
        return cls(model=model)
