"""
Reward Function Registry
========================

Central registry for reward functions with metadata,
versioning, and discovery support.

Features:
    - Decorator-based registration
    - Version tracking
    - Dependency management
    - Runtime discovery
"""

import functools
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# Type for reward functions
RewardFn = Callable[..., Union[List[float], float]]
F = TypeVar("F", bound=RewardFn)


@dataclass
class RewardFunctionMetadata:
    """Metadata for a registered reward function."""

    name: str
    description: str
    version: str
    level: str  # Which level this belongs to
    requires_answer: bool = True  # Whether it needs expected answer
    requires_prompt: bool = False  # Whether it needs the prompt
    is_batch: bool = True  # Whether it processes batches
    is_component: bool = False  # Whether it's a sub-component
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    author: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "level": self.level,
            "requires_answer": self.requires_answer,
            "requires_prompt": self.requires_prompt,
            "is_batch": self.is_batch,
            "is_component": self.is_component,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "author": self.author,
        }


class RewardFunctionRegistry:
    """
    Central registry for all reward functions.

    Singleton pattern ensures global consistency.
    Thread-safe for concurrent access.
    """

    _instance: Optional["RewardFunctionRegistry"] = None
    _lock = None

    def __new__(cls):
        if cls._instance is None:
            import threading

            cls._lock = threading.Lock()
            cls._instance = super().__new__(cls)
            cls._instance._functions: Dict[str, RewardFn] = {}
            cls._instance._metadata: Dict[str, RewardFunctionMetadata] = {}
            cls._instance._initialized = True
            logger.debug("RewardFunctionRegistry initialized")
        return cls._instance

    def register(
        self,
        name: str,
        fn: RewardFn,
        metadata: RewardFunctionMetadata,
    ) -> None:
        """
        Register a reward function.

        Args:
            name: Unique function name
            fn: The reward function
            metadata: Function metadata
        """
        with self._lock:
            if name in self._functions:
                logger.warning(f"Overwriting existing reward function: {name}")

            self._functions[name] = fn
            self._metadata[name] = metadata
            logger.debug(f"Registered reward function: {name} (v{metadata.version})")

    def get(self, name: str) -> Optional[RewardFn]:
        """Get a registered function by name."""
        return self._functions.get(name)

    def get_metadata(self, name: str) -> Optional[RewardFunctionMetadata]:
        """Get metadata for a registered function."""
        return self._metadata.get(name)

    def list_functions(self, level: Optional[str] = None) -> List[str]:
        """
        List registered function names.

        Args:
            level: Filter by level (optional)
        """
        if level is None:
            return list(self._functions.keys())

        return [name for name, meta in self._metadata.items() if meta.level == level]

    def list_by_tag(self, tag: str) -> List[str]:
        """List functions with a specific tag."""
        return [name for name, meta in self._metadata.items() if tag in meta.tags]

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all registered functions."""
        return {name: meta.to_dict() for name, meta in self._metadata.items()}

    def unregister(self, name: str) -> bool:
        """Unregister a function. Returns True if found."""
        with self._lock:
            if name in self._functions:
                del self._functions[name]
                del self._metadata[name]
                logger.debug(f"Unregistered reward function: {name}")
                return True
            return False

    def clear(self) -> None:
        """Clear all registrations. Useful for testing."""
        with self._lock:
            self._functions.clear()
            self._metadata.clear()
            logger.debug("Cleared all reward function registrations")


# Global registry instance
_registry = RewardFunctionRegistry()


def register_reward_function(
    name: str,
    description: str = "",
    version: str = "1.0.0",
    level: str = "component",
    requires_answer: bool = True,
    requires_prompt: bool = False,
    is_batch: bool = True,
    is_component: bool = False,
    dependencies: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    author: str = "",
) -> Callable[[F], F]:
    """
    Decorator to register a reward function.

    Usage:
        @register_reward_function(
            "my_reward",
            description="My custom reward function",
            version="1.0.0",
            level="correctness"
        )
        def my_reward(prompts, completions, answer):
            ...

    Args:
        name: Unique function name
        description: Human-readable description
        version: Semantic version string
        level: Reward level (foundation/correctness/quality/polish/aggregate)
        requires_answer: Whether function needs expected answers
        requires_prompt: Whether function needs prompts
        is_batch: Whether function processes batches
        is_component: Whether this is a sub-component
        dependencies: Names of required other functions
        tags: Searchable tags
        author: Function author

    Returns:
        Decorator that registers the function
    """

    def decorator(fn: F) -> F:
        metadata = RewardFunctionMetadata(
            name=name,
            description=description or fn.__doc__ or "",
            version=version,
            level=level,
            requires_answer=requires_answer,
            requires_prompt=requires_prompt,
            is_batch=is_batch,
            is_component=is_component,
            dependencies=dependencies or [],
            tags=tags or [],
            author=author,
        )

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        _registry.register(name, wrapper, metadata)

        # Attach metadata to function for introspection
        wrapper._reward_metadata = metadata

        return wrapper

    return decorator


def get_reward_function(name: str) -> Optional[RewardFn]:
    """
    Get a registered reward function by name.

    Args:
        name: Function name

    Returns:
        The function or None if not found
    """
    return _registry.get(name)


def list_reward_functions(level: Optional[str] = None) -> List[str]:
    """
    List registered reward function names.

    Args:
        level: Filter by level (optional)

    Returns:
        List of function names
    """
    return _registry.list_functions(level)


def get_reward_function_metadata(name: str) -> Optional[RewardFunctionMetadata]:
    """Get metadata for a reward function."""
    return _registry.get_metadata(name)


def get_all_reward_metadata() -> Dict[str, Dict[str, Any]]:
    """Get metadata for all registered functions."""
    return _registry.get_all_metadata()


def call_reward_function(
    name: str, prompts: List[str], completions: List[str], answer: List[str], **kwargs
) -> List[float]:
    """
    Call a registered reward function by name.

    Args:
        name: Function name
        prompts: List of prompts
        completions: List of completions
        answer: List of expected answers
        **kwargs: Additional arguments

    Returns:
        List of reward scores

    Raises:
        ValueError: If function not found
    """
    fn = get_reward_function(name)
    if fn is None:
        raise ValueError(f"Unknown reward function: {name}")

    metadata = get_reward_function_metadata(name)

    # Build arguments based on metadata
    if metadata and not metadata.requires_answer:
        return fn(prompts, completions, **kwargs)

    return fn(prompts, completions, answer, **kwargs)
