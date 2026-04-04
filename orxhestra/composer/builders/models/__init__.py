"""Model provider registry and factory.

Built-in providers (``openai``, ``anthropic``, ``google``) are registered
lazily on first use.  Custom providers can be added via
:func:`register` or by passing a dotted import path as the provider name.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from orxhestra.composer.errors import ComposerError

_REGISTRY: dict[str, type[BaseChatModel]] = {}

# Lazy-import specs: provider name -> (module, class_name)
_LAZY_PROVIDERS: dict[str, tuple[str, str]] = {
    "openai": ("langchain_openai", "ChatOpenAI"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic"),
    "google": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
}


def register(name: str, cls: type[BaseChatModel]) -> None:
    """Register a custom model provider class.

    Parameters
    ----------
    name : str
        Provider name used in YAML ``provider:`` field.
    cls : type[BaseChatModel]
        Chat model class to register.

    Example::

        register("my_provider", MyCustomChatModel)
    """
    _REGISTRY[name] = cls


def _resolve_provider(provider: str) -> type[BaseChatModel]:
    """Look up a provider class, lazy-importing built-ins as needed.

    Parameters
    ----------
    provider : str
        Registered name or dotted import path.

    Returns
    -------
    type[BaseChatModel]
        The resolved chat model class.

    Raises
    ------
    ComposerError
        If the provider cannot be imported.
    """
    if provider in _REGISTRY:
        return _REGISTRY[provider]

    if provider in _LAZY_PROVIDERS:
        module_path, class_name = _LAZY_PROVIDERS[provider]
        import importlib

        module = importlib.import_module(module_path)
        cls: type[BaseChatModel] = getattr(module, class_name)
        _REGISTRY[provider] = cls
        return cls

    # Treat as a dotted import path to a custom class.
    from orxhestra.composer.builders.tools import import_object

    cls = import_object(provider)
    _REGISTRY[provider] = cls
    return cls




def create(provider: str, name: str, **kwargs: Any) -> BaseChatModel:
    """Create a ``BaseChatModel`` from a provider name and model name.

    Parameters
    ----------
    provider : str
        A registered name (``"openai"``, ``"anthropic"``, ``"google"``), or
        a fully-qualified class import path.
    name : str
        Model name passed to the provider constructor.
    **kwargs : Any
        Extra keyword arguments forwarded to the model constructor.

    Returns
    -------
    BaseChatModel
        Instantiated chat model.
    """
    try:
        cls = _resolve_provider(provider)
    except Exception as exc:
        msg = f"Failed to load model provider '{provider}': {exc}"
        raise ComposerError(msg) from exc
    return cls(model=name, **kwargs)
