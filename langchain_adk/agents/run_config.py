"""RunConfig - per-run configuration for agent execution.

Mirrors LangChain's ``RunnableConfig`` fields — carries streaming mode,
call limits, plus all standard LangChain config keys (callbacks, tags,
metadata, run_name, etc.) in one flat object.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StreamingMode(Enum):
    """Streaming behaviour for agent execution.

    Attributes
    ----------
    NONE : str
        Non-streaming mode. The runner yields one complete event per
        model turn. Suitable for CLI tools, batch processing, and any
        context where incremental output is not needed.
    SSE : str
        Server-Sent Events streaming mode. The runner yields partial
        ``Event`` objects (``partial=True``) as the model
        generates text, then a final complete event (``partial=False``).
        Suitable for web UIs and chat applications with typewriter effects.
    """

    NONE = "none"
    SSE = "sse"


class RunConfig(BaseModel):
    """Configuration for a single agent run.

    Mirrors LangChain's ``RunnableConfig`` fields. All standard LangChain
    config keys (``callbacks``, ``tags``, ``metadata``, ``run_name``, etc.)
    are first-class fields, plus ADK-specific fields like ``streaming_mode``
    and ``max_llm_calls``.

    Pass an instance to ``Runner.run_async()`` or store it on
    ``InvocationContext.run_config``.

    Attributes
    ----------
    streaming_mode : StreamingMode
        Whether to stream partial text events. Defaults to NONE.
    max_llm_calls : int
        Maximum total LLM calls allowed in this run. Set to 0 for no limit.
    callbacks : list[Any], optional
        LangChain callback handlers (e.g. Langfuse, LangSmith). Passed to
        every ``ainvoke()`` / ``astream()`` call for automatic tracing.
    tags : list[str], optional
        Tags propagated to all LangChain calls for filtering in tracing UIs.
    metadata : dict[str, Any], optional
        Metadata propagated to all LangChain calls.
    run_name : str, optional
        Name for the top-level trace span. Defaults to the agent name.
    max_concurrency : int, optional
        Max concurrent LangChain calls (passed through to RunnableConfig).
    configurable : dict[str, Any], optional
        Extra configurable values (passed through to RunnableConfig).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # ADK-specific
    streaming_mode: StreamingMode = StreamingMode.NONE
    max_llm_calls: int = Field(default=500, ge=0)

    # LangChain RunnableConfig fields
    callbacks: list[Any] | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    run_name: str | None = None
    max_concurrency: int | None = None
    configurable: dict[str, Any] | None = None

    def as_langchain_config(self) -> dict[str, Any]:
        """Build a LangChain ``RunnableConfig`` dict from this run config.

        Only includes keys that are set (non-None), so LangChain's defaults
        apply for omitted fields.

        Returns
        -------
        dict[str, Any]
            A dict suitable for LangChain's ``config`` parameter.
        """
        config: dict[str, Any] = {}
        for key in ("callbacks", "tags", "metadata", "run_name", "max_concurrency", "configurable"):
            value = getattr(self, key)
            if value is not None:
                config[key] = value
        return config
