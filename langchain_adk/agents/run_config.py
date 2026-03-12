"""RunConfig - per-run configuration for agent execution.

Controls streaming mode and per-run limits passed into the Runner and
down through InvocationContext.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

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
        ``FinalAnswerEvent`` objects (``partial=True``) as the model
        generates text, then a final complete event (``partial=False``).
        Suitable for web UIs and chat applications with typewriter effects.
    """

    NONE = "none"
    SSE = "sse"


class RunConfig(BaseModel):
    """Configuration for a single agent run.

    Pass an instance to ``Runner.run_async()`` or store it on
    ``InvocationContext.run_config`` to control streaming and call limits
    for the entire invocation.

    Attributes
    ----------
    streaming_mode : StreamingMode
        Whether to stream partial text events. Defaults to NONE.
    max_llm_calls : int
        Maximum total LLM calls allowed in this run. The runner raises an
        error if the agent exceeds this limit. Set to 0 for no limit.
    """

    model_config = ConfigDict(extra="forbid")

    streaming_mode: StreamingMode = StreamingMode.NONE
    max_llm_calls: int = Field(default=500, ge=0)
