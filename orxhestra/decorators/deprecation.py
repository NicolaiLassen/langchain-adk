"""Deprecation decorators for classes, functions, and parameters.

Usage::

    from orxhestra.decorators import deprecated, deprecated_param

    @deprecated("0.1.0", removal="0.2.0", alternative="NewClass")
    class OldClass:
        ...

    @deprecated("0.1.0", alternative="new_function()")
    def old_function():
        ...

    @deprecated_param("old_name", since="0.1.0", alternative="new_name")
    def my_func(new_name: str = "", old_name: str = ""):
        ...
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class OrxhestraDeprecationWarning(DeprecationWarning):
    """Custom deprecation warning for the orxhestra framework.

    Subclasses ``DeprecationWarning`` so users can filter orxhestra
    deprecations independently from other libraries::

        import warnings
        warnings.filterwarnings("ignore", category=OrxhestraDeprecationWarning)
    """


def _format_message(
    name: str,
    since: str,
    *,
    removal: str | None = None,
    alternative: str | None = None,
) -> str:
    """Build a standardised deprecation message.

    Parameters
    ----------
    name : str
        Name of the deprecated item.
    since : str
        Version when the item was deprecated.
    removal : str, optional
        Version when the item will be removed.
    alternative : str, optional
        Replacement to suggest to the user.

    Returns
    -------
    str
        Formatted deprecation message.
    """
    msg: str = f"{name} is deprecated since orxhestra {since}"
    if removal:
        msg += f" and will be removed in {removal}"
    msg += "."
    if alternative:
        msg += f" Use {alternative} instead."
    return msg


def deprecated(
    since: str,
    *,
    removal: str | None = None,
    alternative: str | None = None,
    stacklevel: int = 2,
) -> Callable[[F], F]:
    """Mark a class or function as deprecated.

    Emits ``OrxhestraDeprecationWarning`` on first use (instantiation
    for classes, call for functions).

    Parameters
    ----------
    since : str
        Version when the item was deprecated (e.g. ``"0.1.0"``).
    removal : str, optional
        Version when the item will be removed.
    alternative : str, optional
        Name of the replacement to suggest.
    stacklevel : int
        Stack level for the warning (default 2 points to the caller).

    Returns
    -------
    callable
        A decorator that wraps the target with a deprecation warning.

    Examples
    --------
    >>> @deprecated("0.1.0", removal="0.2.0", alternative="NewAgent")
    ... class OldAgent:
    ...     pass
    >>> obj = OldAgent()  # emits OrxhestraDeprecationWarning
    """
    def decorator(obj: F) -> F:
        name: str = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
        msg: str = _format_message(name, since, removal=removal, alternative=alternative)

        if isinstance(obj, type):
            # Class — warn on __init__
            original_init = obj.__init__

            @functools.wraps(original_init)
            def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(msg, OrxhestraDeprecationWarning, stacklevel=stacklevel)
                original_init(self, *args, **kwargs)

            obj.__init__ = new_init  # type: ignore[assignment]
            obj.__doc__ = (obj.__doc__ or "") + f"\n\n.. deprecated:: {since}\n   {msg}\n"
            return obj  # type: ignore[return-value]

        # Function / method — warn on call
        @functools.wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, OrxhestraDeprecationWarning, stacklevel=stacklevel)
            return obj(*args, **kwargs)

        wrapper.__doc__ = (obj.__doc__ or "") + f"\n\n.. deprecated:: {since}\n   {msg}\n"
        return wrapper  # type: ignore[return-value]

    return decorator


def deprecated_param(
    param_name: str,
    *,
    since: str,
    removal: str | None = None,
    alternative: str | None = None,
    stacklevel: int = 2,
) -> Callable[[F], F]:
    """Warn when a deprecated keyword argument is passed.

    The decorated function still accepts the old parameter so that
    existing code keeps working until the removal version.

    Parameters
    ----------
    param_name : str
        Name of the deprecated keyword argument.
    since : str
        Version when the parameter was deprecated.
    removal : str, optional
        Version when the parameter will be removed.
    alternative : str, optional
        Name of the replacement parameter.
    stacklevel : int
        Stack level for the warning (default 2 points to the caller).

    Returns
    -------
    callable
        A decorator that wraps the function with a parameter check.

    Examples
    --------
    >>> @deprecated_param("old_name", since="0.1.0", alternative="new_name")
    ... def greet(new_name: str = "", old_name: str = "") -> str:
    ...     name = new_name or old_name
    ...     return f"Hello, {name}"
    >>> greet(old_name="world")  # emits OrxhestraDeprecationWarning
    'Hello, world'
    """
    def decorator(fn: F) -> F:
        msg: str = _format_message(
            f"Parameter '{param_name}' of {fn.__qualname__}",
            since,
            removal=removal,
            alternative=f"'{alternative}'" if alternative else None,
        )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if param_name in kwargs and kwargs[param_name] is not None:
                warnings.warn(msg, OrxhestraDeprecationWarning, stacklevel=stacklevel)
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
