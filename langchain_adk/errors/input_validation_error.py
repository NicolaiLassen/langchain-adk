"""InputValidationError - raised when user input fails validation."""

from __future__ import annotations


class InputValidationError(ValueError):
    """Raised when user-supplied input does not meet validation requirements.

    Extends ``ValueError`` so it can be caught by callers expecting standard
    Python validation errors.

    Parameters
    ----------
    message : str, optional
        Human-readable description of why the input is invalid.
    """

    def __init__(self, message: str = "Invalid input.") -> None:
        self.message = message
        super().__init__(self.message)
