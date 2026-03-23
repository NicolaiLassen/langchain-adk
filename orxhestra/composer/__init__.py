"""Declarative YAML-based agent composition.

Registries for extending the composer:

- ``register_builder`` — add custom agent types
- ``register_provider`` — add custom model providers
- ``register_builtin_tool`` — add custom builtin tools
"""

from orxhestra.composer.builders.agents import register as register_builder
from orxhestra.composer.builders.models import register as register_provider
from orxhestra.composer.builders.tools import register_builtin as register_builtin_tool
from orxhestra.composer.composer import Composer
from orxhestra.composer.errors import CircularReferenceError, ComposerError

__all__ = [
    "Composer",
    "ComposerError",
    "CircularReferenceError",
    "register_builder",
    "register_provider",
    "register_builtin_tool",
]
