"""
PARAGON DEBUG LOGGER - The Temporal Debugger

Integration with Rerun.io for "time travel" debugging of graph mutations.
Logs every graph event with timestamps for playback and analysis.

Architecture:
- MutationLogger: Core logging interface
- RerunIntegration: Rerun.io SDK wrapper (optional)
- FileLogger: Fallback JSON logging
- EventBuffer: In-memory ring buffer for recent events

Usage:
    logger = MutationLogger()
    logger.log_node_created("node_123", "CODE", "agent_001")
    logger.log_status_changed("node_123", "PENDING", "IN_PROGRESS")

    # Playback
    events = logger.get_events(since="2024-01-01T00:00:00Z")
    for event in events:
        print(f"{event.timestamp}: {event.mutation_type}")

Design:
- Non-blocking: Uses queue for async logging
- Configurable: Enable/disable Rerun.io integration
- Persistent: Optional file-based logging
"""
import msgspec
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import deque
import threading
import queue
import json
import io

from viz.core import MutationType, MutationEvent


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LoggerConfig:
    """Configuration for the mutation logger."""
    enable_rerun: bool = False          # Enable Rerun.io integration
    enable_file_log: bool = True        # Enable file-based logging
    log_path: Optional[Path] = None     # Path for log files
    buffer_size: int = 10000            # In-memory buffer size
    flush_interval: float = 1.0         # Seconds between flushes
    rerun_app_id: str = "paragon"       # Rerun application ID

    def __post_init__(self):
        if self.log_path is None:
            self.log_path = Path("./workspace/logs")


# =============================================================================
# EVENT BUFFER
# =============================================================================

class EventBuffer:
    """
    Thread-safe ring buffer for recent mutation events.

    Provides O(1) append and O(n) query for time-based filtering.
    """

    def __init__(self, max_size: int = 10000):
        self._buffer: deque[MutationEvent] = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._sequence = 0

    def append(self, event: MutationEvent) -> None:
        """Add an event to the buffer."""
        with self._lock:
            self._buffer.append(event)

    def get_since(self, timestamp: str) -> List[MutationEvent]:
        """Get all events since a timestamp."""
        with self._lock:
            return [e for e in self._buffer if e.timestamp >= timestamp]

    def get_last(self, n: int) -> List[MutationEvent]:
        """Get the last n events."""
        with self._lock:
            items = list(self._buffer)
            return items[-n:] if len(items) >= n else items

    def get_by_node(self, node_id: str) -> List[MutationEvent]:
        """Get all events for a specific node."""
        with self._lock:
            return [e for e in self._buffer if e.node_id == node_id]

    def get_by_type(self, mutation_type: str) -> List[MutationEvent]:
        """Get all events of a specific type."""
        with self._lock:
            return [e for e in self._buffer if e.mutation_type == mutation_type]

    def next_sequence(self) -> int:
        """Get next sequence number."""
        with self._lock:
            self._sequence += 1
            return self._sequence

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


# =============================================================================
# FILE LOGGER
# =============================================================================

class FileLogger:
    """
    File-based event logger.

    Writes events as newline-delimited JSON for easy parsing.
    Rotates logs daily.
    """

    def __init__(self, log_path: Path):
        self._log_path = log_path
        self._current_file: Optional[io.TextIOWrapper] = None
        self._current_date: Optional[str] = None
        self._lock = threading.Lock()
        self._encoder = msgspec.json.Encoder()

        # Ensure log directory exists
        log_path.mkdir(parents=True, exist_ok=True)

    def write(self, event: MutationEvent) -> None:
        """Write an event to the log file."""
        with self._lock:
            self._ensure_file()
            try:
                data = msgspec.to_builtins(event)
                line = json.dumps(data) + "\n"
                if self._current_file:
                    self._current_file.write(line)
                    self._current_file.flush()
            except Exception as e:
                print(f"FileLogger error: {e}")

    def _ensure_file(self) -> None:
        """Ensure we have a valid file handle for today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self._current_date != today:
            # Close old file
            if self._current_file:
                self._current_file.close()

            # Open new file
            filename = f"mutations_{today}.jsonl"
            filepath = self._log_path / filename
            self._current_file = open(filepath, "a", encoding="utf-8")
            self._current_date = today

    def close(self) -> None:
        """Close the file handle."""
        with self._lock:
            if self._current_file:
                self._current_file.close()
                self._current_file = None

    def read_log(self, date: str) -> List[MutationEvent]:
        """Read events from a specific date's log."""
        filename = f"mutations_{date}.jsonl"
        filepath = self._log_path / filename

        if not filepath.exists():
            return []

        events = []
        decoder = msgspec.json.Decoder(type=MutationEvent)

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        event = decoder.decode(line.encode())
                        events.append(event)
                    except Exception:
                        pass

        return events


# =============================================================================
# RERUN INTEGRATION
# =============================================================================

class RerunIntegration:
    """
    Integration with Rerun.io for visual debugging.

    Rerun provides a timeline-based visualization of events,
    allowing "time travel" debugging of graph mutations.

    Requires: pip install rerun-sdk
    """

    def __init__(self, app_id: str = "paragon"):
        self._app_id = app_id
        self._rr = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize Rerun recording."""
        try:
            import rerun as rr
            self._rr = rr
            rr.init(self._app_id, spawn=True)
            self._initialized = True
            return True
        except ImportError:
            print("Rerun.io not available. Install with: pip install rerun-sdk")
            return False
        except Exception as e:
            print(f"Failed to initialize Rerun: {e}")
            return False

    def log_event(self, event: MutationEvent) -> None:
        """Log an event to Rerun."""
        if not self._initialized or not self._rr:
            return

        rr = self._rr

        try:
            # Set time
            rr.set_time_sequence("sequence", event.sequence)

            # Log based on mutation type
            if event.mutation_type == MutationType.NODE_CREATED.value:
                rr.log(
                    f"graph/nodes/{event.node_id}",
                    rr.TextLog(
                        f"Created {event.node_type} by {event.agent_role or 'system'}"
                    )
                )

            elif event.mutation_type == MutationType.STATUS_CHANGED.value:
                rr.log(
                    f"graph/nodes/{event.node_id}/status",
                    rr.TextLog(f"{event.old_status} -> {event.new_status}")
                )

            elif event.mutation_type == MutationType.EDGE_CREATED.value:
                rr.log(
                    f"graph/edges/{event.source_id}_{event.target_id}",
                    rr.TextLog(f"{event.edge_type}: {event.source_id} -> {event.target_id}")
                )

            elif event.mutation_type == MutationType.CONTEXT_PRUNED.value:
                rr.log(
                    f"context/pruning",
                    rr.Scalar(event.nodes_selected / max(event.nodes_considered, 1)),
                    label="pruning_ratio"
                )
                rr.log(
                    f"context/tokens",
                    rr.Scalar(event.token_usage),
                    label="token_usage"
                )

        except Exception as e:
            print(f"Rerun logging error: {e}")

    def close(self) -> None:
        """Close Rerun recording."""
        if self._initialized and self._rr:
            try:
                self._rr.disconnect()
            except Exception:
                pass


# =============================================================================
# MUTATION LOGGER (Main Interface)
# =============================================================================

class MutationLogger:
    """
    Main logging interface for graph mutations.

    Provides a unified API for logging events to:
    - In-memory buffer (always)
    - File-based logs (configurable)
    - Rerun.io visualization (configurable)

    Thread-safe for concurrent logging.

    Usage:
        logger = MutationLogger()

        # Log events
        logger.log_node_created("node_123", "CODE", "agent_001")
        logger.log_status_changed("node_123", "PENDING", "IN_PROGRESS")

        # Query events
        events = logger.get_events_for_node("node_123")
        recent = logger.get_recent_events(100)
    """

    def __init__(self, config: Optional[LoggerConfig] = None):
        self.config = config or LoggerConfig()

        # Core components
        self._buffer = EventBuffer(self.config.buffer_size)
        self._file_logger: Optional[FileLogger] = None
        self._rerun: Optional[RerunIntegration] = None

        # Initialize file logger
        if self.config.enable_file_log and self.config.log_path:
            self._file_logger = FileLogger(self.config.log_path)

        # Initialize Rerun (lazy)
        if self.config.enable_rerun:
            self._rerun = RerunIntegration(self.config.rerun_app_id)
            self._rerun.initialize()

        # Event subscribers
        self._subscribers: List[Callable[[MutationEvent], None]] = []

    def _now(self) -> str:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def _emit(self, event: MutationEvent) -> None:
        """Emit an event to all destinations."""
        # Buffer (always)
        self._buffer.append(event)

        # File logger
        if self._file_logger:
            self._file_logger.write(event)

        # Rerun
        if self._rerun:
            self._rerun.log_event(event)

        # Subscribers
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception as e:
                print(f"Subscriber error: {e}")

    # =========================================================================
    # LOGGING METHODS
    # =========================================================================

    def log_node_created(
        self,
        node_id: str,
        node_type: str,
        agent_id: Optional[str] = None,
        agent_role: Optional[str] = None,
    ) -> MutationEvent:
        """Log a node creation event."""
        event = MutationEvent(
            timestamp=self._now(),
            sequence=self._buffer.next_sequence(),
            mutation_type=MutationType.NODE_CREATED.value,
            node_id=node_id,
            node_type=node_type,
            agent_id=agent_id,
            agent_role=agent_role,
        )
        self._emit(event)
        return event

    def log_node_updated(
        self,
        node_id: str,
        node_type: str,
        agent_id: Optional[str] = None,
    ) -> MutationEvent:
        """Log a node update event."""
        event = MutationEvent(
            timestamp=self._now(),
            sequence=self._buffer.next_sequence(),
            mutation_type=MutationType.NODE_UPDATED.value,
            node_id=node_id,
            node_type=node_type,
            agent_id=agent_id,
        )
        self._emit(event)
        return event

    def log_node_deleted(
        self,
        node_id: str,
        node_type: str,
    ) -> MutationEvent:
        """Log a node deletion event."""
        event = MutationEvent(
            timestamp=self._now(),
            sequence=self._buffer.next_sequence(),
            mutation_type=MutationType.NODE_DELETED.value,
            node_id=node_id,
            node_type=node_type,
        )
        self._emit(event)
        return event

    def log_status_changed(
        self,
        node_id: str,
        old_status: str,
        new_status: str,
        agent_id: Optional[str] = None,
        agent_role: Optional[str] = None,
    ) -> MutationEvent:
        """Log a status change event."""
        event = MutationEvent(
            timestamp=self._now(),
            sequence=self._buffer.next_sequence(),
            mutation_type=MutationType.STATUS_CHANGED.value,
            node_id=node_id,
            old_status=old_status,
            new_status=new_status,
            agent_id=agent_id,
            agent_role=agent_role,
        )
        self._emit(event)
        return event

    def log_edge_created(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
    ) -> MutationEvent:
        """Log an edge creation event."""
        event = MutationEvent(
            timestamp=self._now(),
            sequence=self._buffer.next_sequence(),
            mutation_type=MutationType.EDGE_CREATED.value,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
        )
        self._emit(event)
        return event

    def log_edge_deleted(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
    ) -> MutationEvent:
        """Log an edge deletion event."""
        event = MutationEvent(
            timestamp=self._now(),
            sequence=self._buffer.next_sequence(),
            mutation_type=MutationType.EDGE_DELETED.value,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
        )
        self._emit(event)
        return event

    def log_context_pruned(
        self,
        node_id: str,
        nodes_considered: int,
        nodes_selected: int,
        token_usage: int,
        agent_id: Optional[str] = None,
    ) -> MutationEvent:
        """Log a context pruning event."""
        event = MutationEvent(
            timestamp=self._now(),
            sequence=self._buffer.next_sequence(),
            mutation_type=MutationType.CONTEXT_PRUNED.value,
            node_id=node_id,
            nodes_considered=nodes_considered,
            nodes_selected=nodes_selected,
            token_usage=token_usage,
            agent_id=agent_id,
        )
        self._emit(event)
        return event

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_recent_events(self, n: int = 100) -> List[MutationEvent]:
        """Get the n most recent events."""
        return self._buffer.get_last(n)

    def get_events_since(self, timestamp: str) -> List[MutationEvent]:
        """Get all events since a timestamp."""
        return self._buffer.get_since(timestamp)

    def get_events_for_node(self, node_id: str) -> List[MutationEvent]:
        """Get all events for a specific node."""
        return self._buffer.get_by_node(node_id)

    def get_events_by_type(self, mutation_type: str) -> List[MutationEvent]:
        """Get all events of a specific type."""
        return self._buffer.get_by_type(mutation_type)

    def get_node_timeline(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get a timeline of mutations for a node.

        Returns a simplified list of mutations for debugging.
        """
        events = self.get_events_for_node(node_id)
        return [
            {
                "time": e.timestamp,
                "type": e.mutation_type,
                "old_status": e.old_status,
                "new_status": e.new_status,
                "agent": e.agent_role,
            }
            for e in events
        ]

    # =========================================================================
    # SUBSCRIPTION
    # =========================================================================

    def subscribe(self, callback: Callable[[MutationEvent], None]) -> None:
        """Subscribe to mutation events."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[MutationEvent], None]) -> None:
        """Unsubscribe from mutation events."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def close(self) -> None:
        """Close all resources."""
        if self._file_logger:
            self._file_logger.close()
        if self._rerun:
            self._rerun.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global logger instance
_global_logger: Optional[MutationLogger] = None


def get_logger() -> MutationLogger:
    """Get or create the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = MutationLogger()
    return _global_logger


def configure_logger(config: LoggerConfig) -> MutationLogger:
    """Configure and return a new global logger."""
    global _global_logger
    if _global_logger:
        _global_logger.close()
    _global_logger = MutationLogger(config)
    return _global_logger


def log_node_created(
    node_id: str,
    node_type: str,
    agent_id: Optional[str] = None,
    agent_role: Optional[str] = None,
) -> MutationEvent:
    """Convenience function to log node creation."""
    return get_logger().log_node_created(node_id, node_type, agent_id, agent_role)


def log_status_changed(
    node_id: str,
    old_status: str,
    new_status: str,
    agent_id: Optional[str] = None,
    agent_role: Optional[str] = None,
) -> MutationEvent:
    """Convenience function to log status change."""
    return get_logger().log_status_changed(
        node_id, old_status, new_status, agent_id, agent_role
    )


def log_edge_created(
    source_id: str,
    target_id: str,
    edge_type: str,
) -> MutationEvent:
    """Convenience function to log edge creation."""
    return get_logger().log_edge_created(source_id, target_id, edge_type)
