"""NotFoundError - raised when a requested resource does not exist."""

from __future__ import annotations


class NotFoundError(Exception):
    """Raised when a requested entity cannot be found.

    Parameters
    ----------
    message : str, optional
        Human-readable description of what was not found.
    """

    def __init__(self, message: str = "The requested item was not found.") -> None:
        self.message = message
        super().__init__(self.message)
