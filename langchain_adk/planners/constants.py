"""Task constants - actions, statuses, and state keys."""

from __future__ import annotations


class TaskAction:
    """Canonical actions for the manage_tasks tool."""

    INITIALIZE = "initialize"
    LIST = "list"
    COMPLETE = "complete"
    CREATE = "create"
    UPDATE = "update"
    REMOVE = "remove"

    _ALL = frozenset({"initialize", "list", "complete", "create", "update", "remove"})


_ACTION_ALIASES: dict[str, str] = {
    "init": TaskAction.INITIALIZE,
    "start": TaskAction.INITIALIZE,
    "view": TaskAction.LIST,
    "show": TaskAction.LIST,
    "done": TaskAction.COMPLETE,
    "delete": TaskAction.REMOVE,
}


def normalize_action(raw: str) -> str | None:
    """Resolve a raw action string to its canonical TaskAction value.

    Parameters
    ----------
    raw : str
        The raw action string from the LLM or caller.

    Returns
    -------
    str or None
        The canonical action name, or None if unrecognised.
    """
    key = raw.strip().lower()
    if key in TaskAction._ALL:
        return key
    return _ACTION_ALIASES.get(key)


class TaskStatus:
    """Valid task statuses."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

    _ALL = frozenset({"pending", "in_progress", "completed", "blocked"})


class StateKey:
    """Keys used in InvocationContext.state."""

    TASK_BOARD = "task_board"
    PENDING_QUESTION = "pending_question"
    ACTIVE_AGENT = "active_agent"
