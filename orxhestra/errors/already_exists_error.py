"""AlreadyExistsError - raised when a resource already exists."""

from __future__ import annotations


class AlreadyExistsError(Exception):
    """Raised when attempting to create a resource that already exists.

    Parameters
    ----------
    message : str, optional
        Human-readable description of the conflict.
    """

    def __init__(self, message: str = "The resource already exists.") -> None:
        self.message = message
        super().__init__(self.message)
