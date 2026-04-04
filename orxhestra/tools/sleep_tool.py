"""SleepTool — delay execution for a specified duration."""

from __future__ import annotations

import asyncio

from langchain_core.tools import BaseTool, StructuredTool

# Maximum sleep duration to prevent accidental long hangs.
_MAX_SLEEP_SECONDS: int = 300


def make_sleep_tool(max_seconds: int = _MAX_SLEEP_SECONDS) -> BaseTool:
    """Create a tool that pauses execution for N seconds.

    Parameters
    ----------
    max_seconds : int
        Maximum allowed sleep duration. Default 300 (5 minutes).

    Returns
    -------
    BaseTool
        A ``sleep`` structured tool.
    """

    async def sleep(seconds: float) -> str:
        """Pause execution for the specified number of seconds.

        Args:
            seconds: Duration to sleep (max 300 seconds).
        """
        clamped: float = max(0, min(seconds, max_seconds))
        await asyncio.sleep(clamped)
        return f"Slept for {clamped:.1f} seconds."

    return StructuredTool.from_function(
        coroutine=sleep,
        name="sleep",
        description=(
            "Pause execution for a specified number of seconds. "
            "Use when you need to wait before retrying an operation "
            "or polling for a result. Max 300 seconds."
        ),
    )
