"""
Lightweight event bus for decoupled graph change notifications.

Follows publisher-subscriber pattern for real-time updates without coupling.

Design Principles:
- Publisher-subscriber pattern (decoupled)
- Supports both sync and async handlers
- Non-blocking (async handlers scheduled via create_task)
- Singleton for global access
- Type-safe events via msgspec

Architecture:
    Graph DB / Tools → EventBus → [WebSocket, Logger, Future Webhooks]

Usage:
    # Publisher (in graph_db.py or tools.py)
    from infrastructure.event_bus import get_event_bus, GraphEvent, EventType
    import time

    event_bus = get_event_bus()
    event_bus.publish(GraphEvent(
        type=EventType.NODE_CREATED,
        payload={
            "node_id": "...",
            "node_type": "REQ",
            "metadata": {...}
        },
        timestamp=time.time(),
        source="agent_tools"
    ))

    # Subscriber (in routes.py WebSocket handler)
    async def on_node_created(event: GraphEvent):
        await broadcast_delta(event.payload)

    event_bus.subscribe_async(EventType.NODE_CREATED, on_node_created)
"""
from typing import Callable, List, Dict, Any, Optional
from enum import Enum
import msgspec
import asyncio
from collections import defaultdict
import logging


logger = logging.getLogger("paragon.event_bus")


class EventType(str, Enum):
    """Types of events published by the graph layer."""
    NODE_CREATED = "node_created"
    NODE_UPDATED = "node_updated"
    NODE_DELETED = "node_deleted"
    EDGE_CREATED = "edge_created"
    EDGE_DELETED = "edge_deleted"
    BATCH_MUTATION = "batch_mutation"
    ORCHESTRATOR_ERROR = "orchestrator_error"
    PHASE_CHANGED = "phase_changed"
    # Notification events
    NOTIFICATION_CREATED = "notification_created"
    CROSS_TAB_NOTIFICATION = "cross_tab_notification"
    # Research feedback events
    RESEARCH_FEEDBACK_RECEIVED = "research_feedback_received"
    RESEARCH_TASK_COMPLETED = "research_task_completed"
    MESSAGE_CREATED = "message_created"
    DIALOGUE_TURN_ADDED = "dialogue_turn_added"


class GraphEvent(msgspec.Struct, kw_only=True):
    """
    Event emitted when graph changes.

    Attributes:
        type: Type of event (NODE_CREATED, EDGE_CREATED, etc.)
        payload: Event-specific data (node_id, node_type, etc.)
        timestamp: Unix timestamp when event occurred
        source: Source of event ("agent_tools", "api", "orchestrator")
    """
    type: EventType
    payload: Dict[str, Any]
    timestamp: float
    source: str


class EventBus:
    """
    Singleton event bus for graph change notifications.

    This class provides a lightweight pub/sub mechanism for broadcasting
    graph mutations to interested subscribers without tight coupling.

    Thread Safety:
        NOT thread-safe. Use external locking if needed for concurrent access.
        However, asyncio.create_task is used for async handlers, making it
        safe for async/await contexts.

    Performance:
        - O(1) event publishing
        - O(n) notification per event type (where n = subscriber count)
        - Non-blocking for async handlers (fire-and-forget)
    """

    def __init__(self):
        """Initialize empty subscriber lists."""
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._async_subscribers: Dict[EventType, List[Callable]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: Callable[[GraphEvent], None]):
        """
        Subscribe to events with a synchronous handler.

        Args:
            event_type: Type of event to listen for
            handler: Callable that takes GraphEvent as argument

        Example:
            def on_node_created(event: GraphEvent):
                print(f"Node created: {event.payload['node_id']}")

            event_bus.subscribe(EventType.NODE_CREATED, on_node_created)
        """
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)
            logger.debug(f"Subscribed sync handler to {event_type.value}")

    def subscribe_async(self, event_type: EventType, handler: Callable[[GraphEvent], Any]):
        """
        Subscribe to events with an async handler.

        Args:
            event_type: Type of event to listen for
            handler: Async callable that takes GraphEvent as argument

        Example:
            async def on_node_created(event: GraphEvent):
                await broadcast_to_websocket(event.payload)

            event_bus.subscribe_async(EventType.NODE_CREATED, on_node_created)
        """
        if handler not in self._async_subscribers[event_type]:
            self._async_subscribers[event_type].append(handler)
            logger.debug(f"Subscribed async handler to {event_type.value}")

    def publish(self, event: GraphEvent):
        """
        Publish an event to all subscribers.

        This method runs synchronously but schedules async handlers via
        asyncio.create_task for non-blocking execution.

        Args:
            event: GraphEvent to publish

        Note:
            - Sync handlers run immediately (blocking)
            - Async handlers are scheduled and run in the background
            - Exceptions in handlers are logged but don't propagate
        """
        logger.debug(
            f"Publishing {event.type.value} from {event.source} "
            f"(payload keys: {list(event.payload.keys())})"
        )

        # Run sync handlers immediately
        for handler in self._subscribers[event.type]:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Error in sync handler for {event.type.value}: {e}",
                    exc_info=True
                )

        # Schedule async handlers (non-blocking)
        for handler in self._async_subscribers[event.type]:
            try:
                # Check if event loop is running
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(handler(event))
                except RuntimeError:
                    # No event loop running - log warning
                    logger.warning(
                        f"Cannot schedule async handler for {event.type.value}: "
                        "no event loop running"
                    )
            except Exception as e:
                logger.error(
                    f"Error scheduling async handler for {event.type.value}: {e}",
                    exc_info=True
                )

    def unsubscribe(self, event_type: EventType, handler: Callable):
        """
        Unsubscribe from events.

        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler to remove (must be same instance)

        Example:
            event_bus.unsubscribe(EventType.NODE_CREATED, on_node_created)
        """
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            logger.debug(f"Unsubscribed sync handler from {event_type.value}")

        if handler in self._async_subscribers[event_type]:
            self._async_subscribers[event_type].remove(handler)
            logger.debug(f"Unsubscribed async handler from {event_type.value}")

    def clear_subscribers(self, event_type: EventType = None):
        """
        Clear all subscribers for an event type (or all types).

        Args:
            event_type: Event type to clear (None = all types)

        Warning:
            This is primarily for testing. Use with caution in production.
        """
        if event_type is None:
            self._subscribers.clear()
            self._async_subscribers.clear()
            logger.info("Cleared all event subscribers")
        else:
            self._subscribers[event_type].clear()
            self._async_subscribers[event_type].clear()
            logger.info(f"Cleared subscribers for {event_type.value}")

    def subscriber_count(self, event_type: EventType = None) -> int:
        """
        Get count of subscribers for an event type.

        Args:
            event_type: Event type to count (None = all types)

        Returns:
            Total number of subscribers (sync + async)
        """
        if event_type is None:
            total = sum(len(handlers) for handlers in self._subscribers.values())
            total += sum(len(handlers) for handlers in self._async_subscribers.values())
            return total
        else:
            return (
                len(self._subscribers[event_type]) +
                len(self._async_subscribers[event_type])
            )


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_event_bus: EventBus = None


def get_event_bus() -> EventBus:
    """
    Get the global event bus instance (singleton).

    Returns:
        EventBus singleton instance

    Example:
        from infrastructure.event_bus import get_event_bus, EventType, GraphEvent
        import time

        bus = get_event_bus()
        bus.publish(GraphEvent(
            type=EventType.NODE_CREATED,
            payload={"node_id": "abc123"},
            timestamp=time.time(),
            source="api"
        ))
    """
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
        logger.info("Initialized global event bus")
    return _event_bus


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def publish_node_created(node_id: str, node_type: str, metadata: Dict[str, Any] = None, source: str = "unknown"):
    """
    Convenience function to publish NODE_CREATED event.

    Args:
        node_id: ID of created node
        node_type: Type of node (REQ, CODE, TEST, etc.)
        metadata: Additional metadata about the node
        source: Source of the event
    """
    import time
    bus = get_event_bus()
    bus.publish(GraphEvent(
        type=EventType.NODE_CREATED,
        payload={
            "node_id": node_id,
            "node_type": node_type,
            "metadata": metadata or {},
        },
        timestamp=time.time(),
        source=source
    ))


def publish_edge_created(source_id: str, target_id: str, edge_type: str, source: str = "unknown"):
    """
    Convenience function to publish EDGE_CREATED event.

    Args:
        source_id: Source node ID
        target_id: Target node ID
        edge_type: Type of edge (DEPENDS_ON, IMPLEMENTS, etc.)
        source: Source of the event
    """
    import time
    bus = get_event_bus()
    bus.publish(GraphEvent(
        type=EventType.EDGE_CREATED,
        payload={
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": edge_type,
        },
        timestamp=time.time(),
        source=source
    ))


def publish_orchestrator_error(error_message: str, phase: str, error_type: str = "Exception", source: str = "orchestrator"):
    """
    Convenience function to publish ORCHESTRATOR_ERROR event.

    Args:
        error_message: Human-readable error message
        phase: Orchestrator phase where error occurred (DIALECTIC, RESEARCH, etc.)
        error_type: Exception type name
        source: Source of the event
    """
    import time
    bus = get_event_bus()
    bus.publish(GraphEvent(
        type=EventType.ORCHESTRATOR_ERROR,
        payload={
            "error_message": error_message,
            "phase": phase,
            "error_type": error_type,
        },
        timestamp=time.time(),
        source=source
    ))


def publish_phase_changed(phase: str, session_id: str = "", source: str = "orchestrator"):
    """
    Convenience function to publish PHASE_CHANGED event.

    Args:
        phase: New phase name (DIALECTIC, RESEARCH, PLAN, BUILD, TEST)
        session_id: Session identifier
        source: Source of the event
    """
    import time
    bus = get_event_bus()
    bus.publish(GraphEvent(
        type=EventType.PHASE_CHANGED,
        payload={
            "phase": phase,
            "session_id": session_id,
        },
        timestamp=time.time(),
        source=source
    ))


def publish_notification(
    notification_type: str,
    message: str,
    target_tabs: List[str],
    urgency: str = "info",
    related_node_id: Optional[str] = None,
    metadata: Dict[str, Any] = None,
    source: str = "system"
):
    """
    Convenience function to publish NOTIFICATION_CREATED event.

    Args:
        notification_type: Type of notification (spec_updated, research_complete, etc.)
        message: Human-readable notification message
        target_tabs: List of UI tabs to show notification in
        urgency: Urgency level (info, warning, critical)
        related_node_id: Optional node ID this notification relates to
        metadata: Additional metadata
        source: Source of the event
    """
    import time
    bus = get_event_bus()
    bus.publish(GraphEvent(
        type=EventType.NOTIFICATION_CREATED,
        payload={
            "notification_type": notification_type,
            "message": message,
            "target_tabs": target_tabs,
            "urgency": urgency,
            "related_node_id": related_node_id,
            "metadata": metadata or {},
        },
        timestamp=time.time(),
        source=source
    ))


def publish_dialogue_turn(
    node_id: str,
    turn_number: int,
    agent: str,
    turn_type: str,
    content: str,
    metadata: Dict[str, Any] = None,
    source: str = "orchestrator"
):
    """
    Convenience function to publish DIALOGUE_TURN_ADDED event.

    Args:
        node_id: Node ID associated with the dialogue
        turn_number: Turn number in the dialogue
        agent: Agent name (system, user, assistant)
        turn_type: Type of turn (question, answer, feedback)
        content: Content of the dialogue turn
        metadata: Additional metadata
        source: Source of the event
    """
    import time
    bus = get_event_bus()
    bus.publish(GraphEvent(
        type=EventType.DIALOGUE_TURN_ADDED,
        payload={
            "node_id": node_id,
            "turn_number": turn_number,
            "agent": agent,
            "type": turn_type,
            "content": content,
            "metadata": metadata or {},
        },
        timestamp=time.time(),
        source=source
    ))
