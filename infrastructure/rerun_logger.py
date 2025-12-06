"""
PARAGON TIME MACHINE - Visual Flight Recorder using Rerun SDK

This module provides temporal debugging and visualization using Rerun.io's SDK.
It records graph topology, code changes, and agent reasoning chains to .rrd files
that can be played back timeline-style.

Architecture:
- Stream 1 (Topology): Nodes as 3D points + Edges as lines
- Stream 2 (Content): Code diffs as TextDocument streams
- Stream 3 (Reasoning): LLM thought chains as console logs

Key Features:
- Zero-copy graph topology logging (send raw positions, let Rerun handle layout)
- Timeline scrubbing for debugging graph mutations
- Agent reasoning trace visualization
- Session-based .rrd file output for replay

Design Principles:
1. NO LAYOUT COMPUTATION: Send node positions as-is, use Rerun's ForceDistance client-side
2. msgspec.Struct for data classes (NO Pydantic)
3. Graceful degradation if Rerun unavailable
4. Config-driven paths and settings

Usage:
    logger = RerunLogger(session_id="my_session")
    logger.log_node("node_123", "CODE", "def foo(): pass", {"author": "agent_1"})
    logger.log_edge("node_123", "node_456", "DEPENDS_ON")
    logger.log_thought("agent_1", "Analyzing requirements...")
    path = logger.save()  # Returns Path to .rrd file
"""
import msgspec
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime, timezone
import warnings
import uuid

# Conditional import for graceful fallback
RERUN_AVAILABLE = False
try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    rr = None


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

def load_observability_config() -> Dict[str, Any]:
    """
    Load observability configuration from paragon.toml.

    Returns:
        Dict with keys: log_dir, session_pattern, rerun_enabled
    """
    try:
        import tomllib
        config_path = Path(__file__).parent.parent / "config" / "paragon.toml"

        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        return config.get("observability", {
            "log_dir": "./data/sessions",
            "session_pattern": "{timestamp}_{session_id}.rrd",
            "rerun_enabled": True,
        })
    except Exception as e:
        warnings.warn(f"Failed to load config, using defaults: {e}")
        return {
            "log_dir": "./data/sessions",
            "session_pattern": "{timestamp}_{session_id}.rrd",
            "rerun_enabled": True,
        }


# =============================================================================
# COLOR MAPPING (Consistent with viz/core.py)
# =============================================================================

# Node colors by type (RGB tuples for Rerun)
NODE_COLORS_RGB: Dict[str, Tuple[int, int, int]] = {
    "REQ": (230, 57, 70),           # Red - requirements
    "SPEC": (244, 162, 97),         # Orange - specifications
    "CODE": (42, 157, 143),         # Teal - implementation
    "TEST": (38, 70, 83),           # Dark blue - verification
    "TEST_SUITE": (155, 89, 182),   # Purple - test suites
    "RESEARCH": (168, 218, 220),    # Light blue - research
    "DOC": (69, 123, 157),          # Medium blue - documentation
    "CLARIFICATION": (255, 193, 7), # Amber - needs input
    "ESCALATION": (220, 53, 69),    # Red - failures
    "PLAN": (156, 39, 176),         # Purple - planning
}

DEFAULT_COLOR_RGB = (108, 117, 125)  # Gray


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_node_color(node_type: str) -> Tuple[int, int, int]:
    """Get RGB color for a node type."""
    return NODE_COLORS_RGB.get(node_type, DEFAULT_COLOR_RGB)


# =============================================================================
# RERUN LOGGER
# =============================================================================

class RerunLogger:
    """
    Visual flight recorder using Rerun SDK.

    Records graph topology, code diffs, and reasoning traces to .rrd files
    for timeline-style replay and debugging.

    Attributes:
        session_id: Unique identifier for this recording session
        recording_path: Path to the .rrd file being written
        _initialized: Whether Rerun recording is active
        _sequence: Monotonic sequence counter for timeline ordering
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize the Rerun logger.

        Args:
            session_id: Unique session identifier. If None, generates UUID.
        """
        self.session_id = session_id or self._generate_session_id()
        self.config = load_observability_config()
        self.recording_path: Optional[Path] = None
        self._initialized = False
        self._sequence = 0
        self._node_positions: Dict[str, Tuple[float, float, float]] = {}

        # Initialize if enabled and available
        if self.config.get("rerun_enabled", True):
            self._initialize()

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return uuid.uuid4().hex[:8]

    def _initialize(self) -> bool:
        """
        Initialize Rerun recording.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if not RERUN_AVAILABLE:
            warnings.warn(
                "Rerun SDK not available. Install with: pip install rerun-sdk\n"
                "Continuing without visual recording."
            )
            return False

        try:
            # Create session directory
            log_dir = Path(self.config["log_dir"])
            log_dir.mkdir(parents=True, exist_ok=True)

            # Generate recording path
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            pattern = self.config["session_pattern"]
            filename = pattern.format(timestamp=timestamp, session_id=self.session_id)
            self.recording_path = log_dir / filename

            # Initialize Rerun recording
            rr.init(f"paragon_{self.session_id}", spawn=False)
            rr.save(str(self.recording_path))

            # Log blueprint configuration
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

            self._initialized = True
            return True

        except Exception as e:
            warnings.warn(f"Failed to initialize Rerun: {e}")
            self._initialized = False
            return False

    def _next_sequence(self) -> int:
        """Get next sequence number for timeline ordering."""
        self._sequence += 1
        return self._sequence

    def log_node(
        self,
        node_id: str,
        node_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        position: Optional[Tuple[float, float, float]] = None
    ) -> None:
        """
        Log a node to the topology stream.

        Args:
            node_id: Unique node identifier
            node_type: Node type (REQ, CODE, SPEC, etc.)
            content: Node content (for tooltip/inspection)
            metadata: Additional metadata dict
            position: Optional 3D position (x, y, z). If None, uses simple layout.
        """
        if not self._initialized:
            return

        try:
            # Set timeline
            seq = self._next_sequence()
            rr.set_time_sequence("event", seq)

            # Determine position
            if position is None:
                # Simple grid layout - Rerun's client-side force layout will improve this
                # We hash the node_id to get deterministic but pseudo-random positions
                hash_val = hash(node_id)
                x = (hash_val % 100) * 2.0
                y = ((hash_val // 100) % 100) * 2.0
                z = ((hash_val // 10000) % 100) * 2.0
                position = (x, y, z)

            # Cache position for edge logging
            self._node_positions[node_id] = position

            # Get color based on type
            color = get_node_color(node_type)

            # Log node as 3D point
            entity_path = f"topology/nodes/{node_id}"
            rr.log(
                entity_path,
                rr.Points3D(
                    positions=[position],
                    colors=[color],
                    radii=[0.5],
                    labels=[node_id],
                )
            )

            # Log content as annotation (for inspection)
            preview = content[:200] + "..." if len(content) > 200 else content
            rr.log(
                f"{entity_path}/content",
                rr.TextDocument(
                    preview,
                    media_type=rr.MediaType.TEXT
                )
            )

            # Log metadata if provided
            if metadata:
                meta_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
                rr.log(
                    f"{entity_path}/metadata",
                    rr.TextDocument(meta_str, media_type=rr.MediaType.TEXT)
                )

        except Exception as e:
            warnings.warn(f"Failed to log node {node_id}: {e}")

    def log_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an edge to the topology stream.

        Args:
            source_id: Source node identifier
            target_id: Target node identifier
            edge_type: Edge type (DEPENDS_ON, IMPLEMENTS, etc.)
            metadata: Additional metadata dict
        """
        if not self._initialized:
            return

        try:
            # Set timeline
            seq = self._next_sequence()
            rr.set_time_sequence("event", seq)

            # Get node positions (or use default if not logged yet)
            source_pos = self._node_positions.get(source_id, (0.0, 0.0, 0.0))
            target_pos = self._node_positions.get(target_id, (10.0, 10.0, 10.0))

            # Log edge as LineStrips3D
            entity_path = f"topology/edges/{source_id}_to_{target_id}"
            rr.log(
                entity_path,
                rr.LineStrips3D(
                    strips=[[source_pos, target_pos]],
                    colors=[(150, 150, 150)],  # Gray edges
                    radii=[0.1]
                )
            )

            # Log edge type as text annotation
            mid_pos = tuple((s + t) / 2 for s, t in zip(source_pos, target_pos))
            rr.log(
                f"{entity_path}/type",
                rr.TextLog(
                    edge_type,
                    level=rr.TextLogLevel.INFO
                )
            )

            # Log metadata if provided
            if metadata:
                meta_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
                rr.log(
                    f"{entity_path}/metadata",
                    rr.TextDocument(meta_str, media_type=rr.MediaType.TEXT)
                )

        except Exception as e:
            warnings.warn(f"Failed to log edge {source_id}->{target_id}: {e}")

    def log_code_diff(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
        context: Optional[str] = None
    ) -> None:
        """
        Log a code diff to the content stream.

        Args:
            file_path: Path to the file being modified
            old_content: Original content
            new_content: New content
            context: Optional context (e.g., commit message, reason)
        """
        if not self._initialized:
            return

        try:
            # Set timeline
            seq = self._next_sequence()
            rr.set_time_sequence("event", seq)

            # Create a simple diff representation
            diff_text = f"File: {file_path}\n"
            if context:
                diff_text += f"Context: {context}\n"
            diff_text += "\n--- OLD ---\n"
            diff_text += old_content
            diff_text += "\n\n--- NEW ---\n"
            diff_text += new_content

            # Log as TextDocument
            entity_path = f"content/diffs/{file_path.replace('/', '_')}"
            rr.log(
                entity_path,
                rr.TextDocument(
                    diff_text,
                    media_type=rr.MediaType.TEXT
                )
            )

        except Exception as e:
            warnings.warn(f"Failed to log code diff for {file_path}: {e}")

    def log_thought(
        self,
        agent_id: str,
        thought: str,
        timestamp: Optional[float] = None,
        level: str = "INFO"
    ) -> None:
        """
        Log an agent thought/reasoning step to the reasoning stream.

        Args:
            agent_id: Identifier of the agent
            thought: The thought/reasoning text
            timestamp: Optional Unix timestamp (default: now)
            level: Log level (INFO, DEBUG, WARNING, ERROR)
        """
        if not self._initialized:
            return

        try:
            # Set timeline
            seq = self._next_sequence()
            rr.set_time_sequence("event", seq)

            if timestamp:
                rr.set_time_seconds("wall_time", timestamp)

            # Map level string to Rerun TextLogLevel
            level_map = {
                "DEBUG": rr.TextLogLevel.DEBUG,
                "INFO": rr.TextLogLevel.INFO,
                "WARNING": rr.TextLogLevel.WARN,
                "ERROR": rr.TextLogLevel.ERROR,
            }
            rr_level = level_map.get(level.upper(), rr.TextLogLevel.INFO)

            # Log thought
            entity_path = f"reasoning/{agent_id}"
            rr.log(
                entity_path,
                rr.TextLog(
                    thought,
                    level=rr_level
                )
            )

        except Exception as e:
            warnings.warn(f"Failed to log thought for {agent_id}: {e}")

    def log_graph_snapshot(
        self,
        nodes: list,
        edges: list,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a complete graph snapshot (batch operation).

        Args:
            nodes: List of node dicts with keys: id, type, content, (optional) position
            edges: List of edge dicts with keys: source_id, target_id, type
            metadata: Optional metadata about the snapshot
        """
        if not self._initialized:
            return

        try:
            # Log all nodes
            for node in nodes:
                self.log_node(
                    node_id=node["id"],
                    node_type=node["type"],
                    content=node.get("content", ""),
                    metadata=node.get("metadata"),
                    position=node.get("position")
                )

            # Log all edges
            for edge in edges:
                self.log_edge(
                    source_id=edge["source_id"],
                    target_id=edge["target_id"],
                    edge_type=edge["type"],
                    metadata=edge.get("metadata")
                )

            # Log snapshot metadata
            if metadata:
                seq = self._next_sequence()
                rr.set_time_sequence("event", seq)
                meta_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
                rr.log(
                    "snapshot/metadata",
                    rr.TextDocument(meta_str, media_type=rr.MediaType.TEXT)
                )

        except Exception as e:
            warnings.warn(f"Failed to log graph snapshot: {e}")

    def save(self) -> Optional[Path]:
        """
        Finalize the recording and return path to .rrd file.

        Returns:
            Path to the .rrd file if successful, None otherwise
        """
        if not self._initialized:
            return None

        try:
            # Flush any pending data
            rr.save(str(self.recording_path))
            return self.recording_path

        except Exception as e:
            warnings.warn(f"Failed to save recording: {e}")
            return None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure recording is saved."""
        self.save()


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_logger(session_id: Optional[str] = None) -> RerunLogger:
    """
    Factory function to create a RerunLogger instance.

    Args:
        session_id: Optional session identifier

    Returns:
        RerunLogger instance (may be in degraded mode if Rerun unavailable)
    """
    return RerunLogger(session_id=session_id)


# =============================================================================
# EXAMPLE USAGE (for testing)
# =============================================================================

if __name__ == "__main__":
    # Example: Log a simple graph
    with create_logger(session_id="test") as logger:
        # Log nodes
        logger.log_node(
            "req_1",
            "REQ",
            "Build a data pipeline",
            metadata={"priority": "high"},
            position=(0.0, 0.0, 0.0)
        )
        logger.log_node(
            "spec_1",
            "SPEC",
            "Design ETL architecture",
            position=(10.0, 0.0, 0.0)
        )
        logger.log_node(
            "code_1",
            "CODE",
            "def extract(): pass",
            position=(20.0, 0.0, 0.0)
        )

        # Log edges
        logger.log_edge("spec_1", "req_1", "TRACES_TO")
        logger.log_edge("code_1", "spec_1", "IMPLEMENTS")

        # Log a thought
        logger.log_thought(
            "architect_agent",
            "Analyzing requirement: need to determine data sources..."
        )

        # Log a code diff
        logger.log_code_diff(
            "pipeline/extract.py",
            "# TODO: implement",
            "def extract():\n    return fetch_data()",
            context="Initial implementation"
        )

        print(f"Recording saved to: {logger.recording_path}")
