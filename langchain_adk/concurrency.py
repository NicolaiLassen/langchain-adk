"""Concurrency utilities for streaming event-driven agent execution.

Provides ``gather_with_event_queue``, a reusable pattern for running
multiple coroutines concurrently while draining an event queue in real
time.  This is the core mechanism that lets nested agents and
long-running tools surface progress while they are still executing.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine, Sequence
from contextlib import suppress
from typing import Any, TypeVar

_T = TypeVar("_T")


async def gather_with_event_queue(
    coros: Sequence[Coroutine[Any, Any, _T]],
    event_queue: asyncio.Queue,
) -> AsyncIterator[_T | Any]:
    """Run coroutines concurrently while draining an event queue in real time.

    Coroutines may push items into *event_queue* during execution (e.g. an
    ``AgentTool`` pushing child-agent events).  Those items are yielded
    immediately — they are **not** buffered until the coroutines finish.

    After all coroutines complete, their results are yielded **in the
    original order** so the caller can process them deterministically.

    The caller can distinguish queued events from task results by type::

        async for item in gather_with_event_queue(coros, queue):
            if isinstance(item, Event):
                yield item          # intermediate event from child
            else:
                handle_result(item)  # coroutine return value

    Parameters
    ----------
    coros : Sequence[Coroutine]
        Coroutines to run concurrently.
    event_queue : asyncio.Queue
        Queue that coroutines may push events into during execution.

    Yields
    ------
    Queue items (in real time) followed by coroutine results (in original order).
    """
    if not coros:
        return

    tasks = {asyncio.create_task(c): i for i, c in enumerate(coros)}
    results: list[_T | None] = [None] * len(tasks)
    queue_task: asyncio.Task | None = asyncio.create_task(event_queue.get())

    try:
        while tasks:
            wait_set: list[asyncio.Task] = list(tasks)
            if queue_task is not None:
                wait_set.append(queue_task)

            done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

            # Yield queued events first — these are real-time progress.
            handled_queue_task = None
            if queue_task is not None and queue_task in done:
                handled_queue_task = queue_task
                yield queue_task.result()
                queue_task = asyncio.create_task(event_queue.get())

            for task in done:
                if task is handled_queue_task:
                    continue
                idx = tasks.pop(task)
                results[idx] = task.result()
    finally:
        if queue_task is not None:
            queue_task.cancel()
            with suppress(asyncio.CancelledError):
                await queue_task

    # Drain any events pushed after the last task completed.
    while not event_queue.empty():
        yield event_queue.get_nowait()

    # Yield task results in original order.
    for result in results:
        yield result
