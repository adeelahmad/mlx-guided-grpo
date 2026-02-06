"""
Type Component Metaclass - Auto-Registration via Metaprogramming
=================================================================

Metaclass that automatically registers type components (reward, loader,
generator) when subclasses define a ``type_name`` class attribute.

Design Patterns:
- Metaclass Registry: Classes self-register at definition time
- Lazy Instantiation: Classes stored, instantiated on demand

Usage:
    # Defining a class auto-registers it:
    class MCQReward(BaseReward):
        type_name = "mcq"
        ...

    # Later, flush into coordinator:
    auto_register_pending(coordinator, tokenizer)
"""

from __future__ import annotations

import logging
from abc import ABCMeta
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .coordinator import TypeCoordinator
    from transformers import PreTrainedTokenizer

__all__ = [
    "TypeComponentMeta",
    "auto_register_pending",
    "get_pending_types",
]

logger = logging.getLogger(__name__)


# =============================================================================
# PENDING REGISTRY (module-level, filled by metaclass)
# =============================================================================

# Structure: {type_name: {"reward": cls, "loader": cls, "generator": cls}}
_PENDING_REGISTRY: dict[str, dict[str, type]] = {}


# =============================================================================
# COMPONENT KIND DETECTION
# =============================================================================

_BASE_CLASS_KINDS: dict[str, str] = {
    "BaseReward": "reward",
    "BaseDatasetLoader": "loader",
    "BaseRolloutGenerator": "generator",
}


def _detect_component_kind(bases: tuple[type, ...]) -> str | None:
    """Detect component kind from base classes.

    Walks the MRO to find which base class category this is.
    """
    for base in bases:
        for cls in base.__mro__:
            name = cls.__name__
            if name in _BASE_CLASS_KINDS:
                return _BASE_CLASS_KINDS[name]
    return None


# =============================================================================
# METACLASS
# =============================================================================

class TypeComponentMeta(ABCMeta):
    """Metaclass that auto-registers type components into a pending registry.

    When a concrete class (non-abstract, has ``type_name``) is created,
    it is stored in the pending registry keyed by type_name and component kind.

    The pending registry is flushed into a TypeCoordinator via
    ``auto_register_pending()``.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> TypeComponentMeta:
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip abstract base classes
        type_name = namespace.get("type_name")
        if not type_name:
            return cls

        # Skip if type_name is inherited (only register explicit declarations)
        if "type_name" not in namespace:
            return cls

        # Detect component kind
        kind = _detect_component_kind(bases)
        if kind is None:
            logger.warning(
                "Class %s has type_name='%s' but unknown base class. Skipping.",
                name, type_name,
            )
            return cls

        # Register
        if type_name not in _PENDING_REGISTRY:
            _PENDING_REGISTRY[type_name] = {}

        _PENDING_REGISTRY[type_name][kind] = cls
        logger.debug(
            "Auto-registered %s as %s for type '%s'", name, kind, type_name
        )

        return cls


# =============================================================================
# FLUSH PENDING INTO COORDINATOR
# =============================================================================

def auto_register_pending(
    coordinator: TypeCoordinator,
    tokenizer: PreTrainedTokenizer,
    event_bus: Any = None,
) -> list[str]:
    """Instantiate pending classes and register with coordinator.

    Args:
        coordinator: TypeCoordinator to register into
        tokenizer: Tokenizer for dataset loaders
        event_bus: Optional EventBus to pass to components

    Returns:
        List of type names that were registered
    """
    registered = []

    for type_name, components in _PENDING_REGISTRY.items():
        reward_cls = components.get("reward")
        loader_cls = components.get("loader")
        generator_cls = components.get("generator")

        if not all([reward_cls, loader_cls, generator_cls]):
            missing = [
                k for k in ("reward", "loader", "generator")
                if k not in components
            ]
            logger.warning(
                "Type '%s' missing components: %s. Skipping.",
                type_name, missing,
            )
            continue

        try:
            # Instantiate components
            kwargs = {"event_bus": event_bus} if event_bus else {}
            reward = reward_cls(**kwargs)
            loader = loader_cls(tokenizer=tokenizer, **kwargs)
            generator = generator_cls(**kwargs)

            # Register
            is_default = type_name == "general_qna"
            coordinator.register_type(
                type_name=type_name,
                reward=reward,
                loader=loader,
                generator=generator,
                metadata={"auto_registered": True},
                set_as_default=is_default,
            )
            registered.append(type_name)

        except Exception:
            logger.exception(
                "Failed to auto-register type '%s'", type_name
            )

    return registered


def get_pending_types() -> dict[str, list[str]]:
    """Inspect pending registrations (for debugging).

    Returns:
        Dict mapping type names to list of registered component kinds
    """
    return {
        type_name: list(components.keys())
        for type_name, components in _PENDING_REGISTRY.items()
    }
