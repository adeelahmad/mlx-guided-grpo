"""
Event Bus - Lightweight Pub/Sub for Type System Lifecycle
==========================================================

Central event dispatcher for cross-cutting concerns (logging, metrics,
debugging) without coupling components.

Design Patterns:
- Observer Pattern: Decouple event producers from consumers
- Singleton-friendly: One bus per coordinator

Usage:
    bus = EventBus()
    bus.subscribe(REWARD_COMPUTED, lambda e: print(e.data["score"]))
    bus.publish(Event(REWARD_COMPUTED, {"score": 0.85}))
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "Event",
    "EventBus",
    # Event name constants
    "SAMPLE_VALIDATED",
    "SAMPLE_LOADED",
    "REWARD_COMPUTED",
    "REWARD_INVALID",
    "GENERATION_STARTED",
    "GENERATION_COMPLETED",
    "PHASE_COMPLETED",
    "CURRICULUM_APPLIED",
    "TYPE_REGISTERED",
]

logger = logging.getLogger(__name__)


# =============================================================================
# EVENT NAMES
# =============================================================================

SAMPLE_VALIDATED = "sample.validated"
SAMPLE_LOADED = "sample.loaded"
REWARD_COMPUTED = "reward.computed"
REWARD_INVALID = "reward.invalid"
GENERATION_STARTED = "generation.started"
GENERATION_COMPLETED = "generation.completed"
PHASE_COMPLETED = "generation.phase_completed"
CURRICULUM_APPLIED = "curriculum.applied"
TYPE_REGISTERED = "type.registered"


# =============================================================================
# EVENT DATACLASS
# =============================================================================

@dataclass(frozen=True)
class Event:
    """Immutable event payload.

    Attributes:
        name: Event name (use constants above)
        data: Event-specific payload
        timestamp: Monotonic timestamp (auto-set)
    """
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)


# =============================================================================
# EVENT BUS
# =============================================================================

EventHandler = Callable[[Event], None]


class EventBus:
    """Lightweight publish/subscribe event dispatcher.

    Thread-safe for single-writer scenarios (typical in training loops).
    Handlers are called synchronously in subscription order.
    """

    __slots__ = ("_subscribers",)

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """Subscribe a handler to an event.

        Args:
            event_name: Event to listen for
            handler: Callable receiving Event instance
        """
        if handler not in self._subscribers[event_name]:
            self._subscribers[event_name].append(handler)

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        """Remove a handler from an event.

        Args:
            event_name: Event to stop listening
            handler: Handler to remove
        """
        handlers = self._subscribers.get(event_name, [])
        if handler in handlers:
            handlers.remove(handler)

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.

        Handlers are called synchronously. Exceptions in handlers are
        logged but do not propagate (fault isolation).

        Args:
            event: Event to publish
        """
        for handler in self._subscribers.get(event.name, []):
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Error in event handler for '%s'", event.name
                )

    def clear(self) -> None:
        """Remove all subscribers."""
        self._subscribers.clear()

    def subscriber_count(self, event_name: str) -> int:
        """Number of handlers for an event."""
        return len(self._subscribers.get(event_name, []))
