"""Task persistence for the A2A server.

The store is an indirection between the server's task lifecycle and
the underlying storage.  Default implementation is in-memory with an
LRU eviction limit (mirroring the previous behaviour of the inline
``dict`` on :class:`A2AServer`).  Adopters can supply their own
implementation — sqlite, redis, postgres — by satisfying the
:class:`TaskStore` protocol.

The store is what powers ``ListTasks`` (new in A2A v1.0): the spec
requires paginated, filterable task history, which a plain dict
cannot provide ergonomically.

See Also
--------
:class:`orxhestra.a2a.types.Task`
:class:`orxhestra.a2a.types.TaskState`
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections import OrderedDict
from typing import Protocol, runtime_checkable

from orxhestra.a2a.types import Task, TaskState


@runtime_checkable
class TaskStore(Protocol):
    """Async storage protocol for A2A :class:`~orxhestra.a2a.types.Task` objects.

    Every method is awaitable so the protocol works equally well for
    in-memory, file-backed, and remote-database implementations.

    Implementations must be safe to call concurrently from multiple
    coroutines on the same event loop — the server's SSE streaming
    path interleaves reads and writes.
    """

    async def get(self, task_id: str) -> Task | None:
        """Return the stored task, or ``None`` if not found."""

    async def put(self, task: Task) -> None:
        """Insert or replace a task by its ``id``."""

    async def list(
        self,
        *,
        context_id: str | None = None,
        state: TaskState | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> tuple[list[Task], str | None]:
        """Return a page of tasks plus a continuation cursor.

        Parameters
        ----------
        context_id
            If set, only return tasks belonging to this conversation.
        state
            If set, only return tasks currently in this lifecycle state.
        limit
            Maximum number of tasks to return.  ``50`` by default; the
            spec allows servers to cap this at their discretion.
        cursor
            Opaque pagination token returned by a prior call.

        Returns
        -------
        tuple[list[Task], str | None]
            ``(tasks, next_cursor)`` where ``next_cursor`` is ``None``
            once the caller has paged through every matching task.
        """

    async def delete(self, task_id: str) -> bool:
        """Remove a task.  Returns ``True`` if it existed."""


class InMemoryTaskStore:
    """Default :class:`TaskStore` — backed by an :class:`~collections.OrderedDict`.

    Bounded to ``max_tasks`` entries; on overflow the oldest task is
    evicted (LRU semantics: every successful :meth:`get` and
    :meth:`put` moves the entry to the end).  No persistence — all
    state lost on process restart.

    Parameters
    ----------
    max_tasks
        Cap on the number of tasks held simultaneously.  Defaults to
        ``10_000``, matching the previous in-memory behaviour.
    """

    def __init__(self, max_tasks: int = 10_000) -> None:
        self._tasks: OrderedDict[str, Task] = OrderedDict()
        self._max_tasks = max_tasks
        self._lock = asyncio.Lock()

    async def get(self, task_id: str) -> Task | None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                self._tasks.move_to_end(task_id)
            return task

    async def put(self, task: Task) -> None:
        async with self._lock:
            self._tasks[task.id] = task
            self._tasks.move_to_end(task.id)
            while len(self._tasks) > self._max_tasks:
                self._tasks.popitem(last=False)

    async def list(
        self,
        *,
        context_id: str | None = None,
        state: TaskState | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> tuple[list[Task], str | None]:
        async with self._lock:
            # Newest-first ordering — the OrderedDict tracks insertion +
            # touch order, so reversed() gives us LRU-newest-first.
            all_tasks = list(reversed(self._tasks.values()))

        # Apply filters before pagination so the cursor lines up with
        # the filtered subsequence.
        if context_id is not None:
            all_tasks = [t for t in all_tasks if t.context_id == context_id]
        if state is not None:
            all_tasks = [t for t in all_tasks if t.status.state == state]

        start = _decode_cursor(cursor)
        end = start + max(1, limit)
        page = all_tasks[start:end]
        next_cursor = _encode_cursor(end) if end < len(all_tasks) else None
        return page, next_cursor

    async def delete(self, task_id: str) -> bool:
        async with self._lock:
            return self._tasks.pop(task_id, None) is not None


def _encode_cursor(offset: int) -> str:
    """Opaque-but-trivial pagination token."""
    raw = json.dumps({"o": offset}).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str | None) -> int:
    if cursor is None:
        return 0
    try:
        # Restore the padding that ``rstrip("=")`` removed.
        padded = cursor + "=" * (-len(cursor) % 4)
        return int(json.loads(base64.urlsafe_b64decode(padded))["o"])
    except Exception:
        return 0
