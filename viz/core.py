"""
PARAGON VISUALIZATION CORE - The Graph Renderer's Data Model

This module provides the data structures and serialization for Paragon's
visualization layer. It bridges ParagonDB's internal representation with
Cosmograph's expected format via Apache Arrow IPC.

Architecture:
- VizNode/VizEdge: Lightweight visualization-focused representations
- GraphSnapshot: Full graph state for initial render
- GraphDelta: Incremental updates for real-time WebSocket streaming
- MutationEvent: Individual graph mutation for temporal debugging

Performance:
- Uses polars for zero-copy Arrow IPC serialization
- Batch updates minimize WebSocket messages
- Color/position calculations done server-side to reduce client load
"""
import msgspec
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io

import polars as pl

from core.ontology import NodeType, NodeStatus, EdgeType


# =============================================================================
# COLOR PALETTES (Consistent across views)
# =============================================================================

# Node colors by type (Cosmograph-compatible hex)
NODE_COLORS: Dict[str, str] = {
    NodeType.REQ.value: "#E63946",      # Red - requirements are the source
    NodeType.SPEC.value: "#F4A261",     # Orange - specifications
    NodeType.CODE.value: "#2A9D8F",     # Teal - implementation
    NodeType.TEST.value: "#264653",     # Dark blue - verification
    NodeType.RESEARCH.value: "#A8DADC", # Light blue - research artifacts
    NodeType.DOC.value: "#457B9D",      # Medium blue - documentation
    "default": "#6C757D",               # Gray - unknown types
}

# Node colors by status (for status-based coloring mode)
STATUS_COLORS: Dict[str, str] = {
    NodeStatus.PENDING.value: "#FFC107",    # Amber - waiting
    NodeStatus.PROCESSING.value: "#17A2B8", # Cyan - processing
    NodeStatus.TESTING.value: "#9B59B6",    # Purple - testing
    NodeStatus.TESTED.value: "#3498DB",     # Blue - tested
    NodeStatus.VERIFIED.value: "#28A745",   # Green - success
    NodeStatus.FAILED.value: "#DC3545",     # Red - failure
    NodeStatus.BLOCKED.value: "#6C757D",    # Gray - blocked
    "default": "#6C757D",
}

# Edge colors by type
EDGE_COLORS: Dict[str, str] = {
    EdgeType.DEPENDS_ON.value: "#ADB5BD",      # Light gray
    EdgeType.IMPLEMENTS.value: "#2A9D8F",      # Teal
    EdgeType.TRACES_TO.value: "#E63946",       # Red
    EdgeType.VERIFIES.value: "#264653",        # Dark blue
    EdgeType.TESTS.value: "#9B59B6",           # Purple
    EdgeType.DEFINES.value: "#F4A261",         # Orange
    EdgeType.CONTAINS.value: "#457B9D",        # Blue
    EdgeType.REFERENCES.value: "#3498DB",      # Light blue
    EdgeType.INHERITS.value: "#E74C3C",        # Red
    EdgeType.FEEDBACK.value: "#E83E8C",        # Pink
    EdgeType.RESEARCH_FOR.value: "#A8DADC",    # Light cyan
    "default": "#6C757D",
}


# =============================================================================
# MUTATION TYPES (For event logging)
# =============================================================================

class MutationType(str, Enum):
    """Types of graph mutations for event tracking."""
    NODE_CREATED = "NODE_CREATED"
    NODE_UPDATED = "NODE_UPDATED"
    NODE_DELETED = "NODE_DELETED"
    EDGE_CREATED = "EDGE_CREATED"
    EDGE_DELETED = "EDGE_DELETED"
    STATUS_CHANGED = "STATUS_CHANGED"
    CONTEXT_PRUNED = "CONTEXT_PRUNED"
    BATCH_UPDATE = "BATCH_UPDATE"


# =============================================================================
# VISUALIZATION DATA STRUCTURES
# =============================================================================

class VizNode(msgspec.Struct, kw_only=True):
    """
    Lightweight node representation for visualization.

    Contains only the fields needed for rendering, plus computed
    visualization properties (color, size, position hints).
    """
    id: str
    type: str
    status: str
    label: str                          # Short display label
    color: str                          # Hex color code
    size: float = 1.0                   # Relative size (1.0 = normal)
    x: Optional[float] = None           # X position (optional, can be computed by renderer)
    y: Optional[float] = None           # Y position

    # Metadata for tooltips
    created_by: str = "system"
    created_at: str = ""
    teleology_status: str = "unchecked"

    # Layout hints
    layer: int = 0                      # Topological layer (for hierarchical layout)
    is_root: bool = False               # True if no predecessors
    is_leaf: bool = False               # True if no successors

    @classmethod
    def from_node_data(
        cls,
        node,  # NodeData
        color_mode: str = "type",       # "type" or "status"
        layer: int = 0,
        is_root: bool = False,
        is_leaf: bool = False,
    ) -> "VizNode":
        """Create VizNode from NodeData."""
        # Determine color based on mode
        if color_mode == "status":
            color = STATUS_COLORS.get(node.status, STATUS_COLORS["default"])
        else:
            color = NODE_COLORS.get(node.type, NODE_COLORS["default"])

        # Create short label
        label = f"{node.type[:4]}:{node.id[:8]}"
        if hasattr(node, 'data') and node.data.get('name'):
            label = node.data['name'][:20]

        return cls(
            id=node.id,
            type=node.type,
            status=node.status,
            label=label,
            color=color,
            created_by=node.created_by,
            created_at=node.created_at,
            teleology_status=getattr(node, 'teleology_status', 'unchecked'),
            layer=layer,
            is_root=is_root,
            is_leaf=is_leaf,
        )


class VizEdge(msgspec.Struct, kw_only=True):
    """
    Lightweight edge representation for visualization.
    """
    source: str                         # Source node ID
    target: str                         # Target node ID
    type: str                           # Edge type
    color: str                          # Hex color code
    weight: float = 1.0                 # Line thickness multiplier

    @classmethod
    def from_edge_data(cls, edge) -> "VizEdge":
        """Create VizEdge from EdgeData."""
        color = EDGE_COLORS.get(edge.type, EDGE_COLORS["default"])

        return cls(
            source=edge.source_id,
            target=edge.target_id,
            type=edge.type,
            color=color,
            weight=edge.weight,
        )


class GraphSnapshot(msgspec.Struct, kw_only=True):
    """
    Complete graph state for initial render.

    Sent on WebSocket connection or page load.
    Uses polars serialization for efficient transfer.
    """
    timestamp: str
    node_count: int
    edge_count: int
    nodes: List[VizNode]
    edges: List[VizEdge]

    # Graph metrics for display
    layer_count: int = 0
    has_cycle: bool = False
    root_count: int = 0
    leaf_count: int = 0

    # Comparison metadata (for dev view)
    version: str = "current"
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "nodes": [msgspec.to_builtins(n) for n in self.nodes],
            "edges": [msgspec.to_builtins(e) for e in self.edges],
            "layer_count": self.layer_count,
            "has_cycle": self.has_cycle,
            "root_count": self.root_count,
            "leaf_count": self.leaf_count,
            "version": self.version,
            "label": self.label,
        }


class GraphDelta(msgspec.Struct, kw_only=True):
    """
    Incremental graph update for real-time streaming.

    Sent via WebSocket when graph changes.
    Much smaller than full snapshot for efficiency.
    """
    timestamp: str
    sequence: int                       # Monotonic sequence number

    # Changes
    nodes_added: List[VizNode] = msgspec.field(default_factory=list)
    nodes_updated: List[VizNode] = msgspec.field(default_factory=list)
    nodes_removed: List[str] = msgspec.field(default_factory=list)  # IDs only
    edges_added: List[VizEdge] = msgspec.field(default_factory=list)
    edges_removed: List[Tuple[str, str]] = msgspec.field(default_factory=list)  # (source, target) pairs

    def is_empty(self) -> bool:
        """Check if delta contains any changes."""
        return (
            not self.nodes_added and
            not self.nodes_updated and
            not self.nodes_removed and
            not self.edges_added and
            not self.edges_removed
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "nodes_added": [msgspec.to_builtins(n) for n in self.nodes_added],
            "nodes_updated": [msgspec.to_builtins(n) for n in self.nodes_updated],
            "nodes_removed": self.nodes_removed,
            "edges_added": [msgspec.to_builtins(e) for e in self.edges_added],
            "edges_removed": self.edges_removed,
        }


class MutationEvent(msgspec.Struct, kw_only=True):
    """
    Individual mutation event for temporal debugging.

    Logged to Rerun.io for "time travel" debugging.
    """
    timestamp: str
    sequence: int
    mutation_type: str                  # MutationType value
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    old_status: Optional[str] = None
    new_status: Optional[str] = None
    agent_id: Optional[str] = None
    agent_role: Optional[str] = None

    # For context pruning events
    nodes_considered: int = 0
    nodes_selected: int = 0
    token_usage: int = 0

    # Source/target for edges
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    edge_type: Optional[str] = None


# =============================================================================
# VISUALIZATION GRAPH (Aggregated State)
# =============================================================================

class VizGraph:
    """
    Visualization-focused graph wrapper.

    Maintains visualization state and generates snapshots/deltas.
    Designed for real-time updates via WebSocket.
    """

    def __init__(self):
        self._nodes: Dict[str, VizNode] = {}
        self._edges: Dict[Tuple[str, str], VizEdge] = {}
        self._sequence: int = 0
        self._layer_map: Dict[str, int] = {}

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def add_node(self, node: VizNode) -> None:
        """Add or update a node."""
        self._nodes[node.id] = node

    def remove_node(self, node_id: str) -> Optional[VizNode]:
        """Remove a node and its incident edges."""
        node = self._nodes.pop(node_id, None)
        # Remove incident edges
        edges_to_remove = [
            key for key in self._edges
            if key[0] == node_id or key[1] == node_id
        ]
        for key in edges_to_remove:
            del self._edges[key]
        return node

    def add_edge(self, edge: VizEdge) -> None:
        """Add or update an edge."""
        self._edges[(edge.source, edge.target)] = edge

    def remove_edge(self, source: str, target: str) -> Optional[VizEdge]:
        """Remove an edge."""
        return self._edges.pop((source, target), None)

    def get_snapshot(
        self,
        version: str = "current",
        label: str = "",
    ) -> GraphSnapshot:
        """Generate a complete snapshot of current state."""
        # Count roots and leaves
        sources = {e.source for e in self._edges.values()}
        targets = {e.target for e in self._edges.values()}

        root_count = sum(1 for n in self._nodes.values() if n.is_root)
        leaf_count = sum(1 for n in self._nodes.values() if n.is_leaf)

        # Compute layer count
        layer_count = max((n.layer for n in self._nodes.values()), default=0) + 1

        return GraphSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            node_count=self.node_count,
            edge_count=self.edge_count,
            nodes=list(self._nodes.values()),
            edges=list(self._edges.values()),
            layer_count=layer_count,
            has_cycle=False,  # Assume valid DAG
            root_count=root_count,
            leaf_count=leaf_count,
            version=version,
            label=label,
        )

    def create_delta(
        self,
        added_nodes: List[VizNode] = None,
        updated_nodes: List[VizNode] = None,
        removed_node_ids: List[str] = None,
        added_edges: List[VizEdge] = None,
        removed_edges: List[Tuple[str, str]] = None,
    ) -> GraphDelta:
        """Create a delta for incremental update."""
        self._sequence += 1

        return GraphDelta(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sequence=self._sequence,
            nodes_added=added_nodes or [],
            nodes_updated=updated_nodes or [],
            nodes_removed=removed_node_ids or [],
            edges_added=added_edges or [],
            edges_removed=removed_edges or [],
        )


# =============================================================================
# ARROW IPC SERIALIZATION
# =============================================================================

def serialize_to_arrow(snapshot: GraphSnapshot) -> Tuple[bytes, bytes]:
    """
    Serialize GraphSnapshot to Apache Arrow IPC format.

    Returns:
        Tuple of (nodes_arrow_bytes, edges_arrow_bytes)

    This is the format expected by Cosmograph for efficient rendering
    of large graphs (100K+ nodes).
    """
    # Nodes DataFrame
    nodes_df = pl.DataFrame({
        "id": [n.id for n in snapshot.nodes],
        "type": [n.type for n in snapshot.nodes],
        "status": [n.status for n in snapshot.nodes],
        "label": [n.label for n in snapshot.nodes],
        "color": [n.color for n in snapshot.nodes],
        "size": [n.size for n in snapshot.nodes],
        "x": [n.x for n in snapshot.nodes],
        "y": [n.y for n in snapshot.nodes],
        "layer": [n.layer for n in snapshot.nodes],
        "is_root": [n.is_root for n in snapshot.nodes],
        "is_leaf": [n.is_leaf for n in snapshot.nodes],
    })

    # Edges DataFrame
    edges_df = pl.DataFrame({
        "source": [e.source for e in snapshot.edges],
        "target": [e.target for e in snapshot.edges],
        "type": [e.type for e in snapshot.edges],
        "color": [e.color for e in snapshot.edges],
        "weight": [e.weight for e in snapshot.edges],
    })

    # Serialize to Arrow IPC
    nodes_buffer = io.BytesIO()
    edges_buffer = io.BytesIO()

    nodes_df.write_ipc(nodes_buffer)
    edges_df.write_ipc(edges_buffer)

    return nodes_buffer.getvalue(), edges_buffer.getvalue()


def create_snapshot_from_db(
    db,  # ParagonDB
    color_mode: str = "type",
    version: str = "current",
    label: str = "",
) -> GraphSnapshot:
    """
    Create a GraphSnapshot from a ParagonDB instance.

    This is the main entry point for visualization.

    Args:
        db: ParagonDB instance
        color_mode: "type" or "status" for node coloring
        version: Version identifier for comparison view
        label: Human-readable label

    Returns:
        GraphSnapshot ready for rendering
    """
    # Get all nodes
    all_nodes = db.get_all_nodes()
    all_edges = db.get_all_edges()

    # Compute layers for layout hints
    try:
        waves = db.get_waves()
        layer_map = {}
        for layer_idx, wave in enumerate(waves):
            for node in wave:
                layer_map[node.id] = layer_idx
    except Exception:
        layer_map = {}
        waves = []

    # Find roots and leaves
    root_ids = {n.id for n in db.get_root_nodes()}
    leaf_ids = {n.id for n in db.get_leaf_nodes()}

    # Convert to VizNodes
    viz_nodes = []
    for node in all_nodes:
        viz_node = VizNode.from_node_data(
            node,
            color_mode=color_mode,
            layer=layer_map.get(node.id, 0),
            is_root=node.id in root_ids,
            is_leaf=node.id in leaf_ids,
        )
        viz_nodes.append(viz_node)

    # Convert to VizEdges
    viz_edges = [VizEdge.from_edge_data(e) for e in all_edges]

    return GraphSnapshot(
        timestamp=datetime.now(timezone.utc).isoformat(),
        node_count=len(viz_nodes),
        edge_count=len(viz_edges),
        nodes=viz_nodes,
        edges=viz_edges,
        layer_count=len(waves),
        has_cycle=db.has_cycle(),
        root_count=len(root_ids),
        leaf_count=len(leaf_ids),
        version=version,
        label=label,
    )


# =============================================================================
# COMPARISON UTILITIES (For Development View)
# =============================================================================

@dataclass
class GraphComparison:
    """
    Comparison between two graph snapshots.

    Used for regression testing in the Development View.
    """
    baseline: GraphSnapshot
    treatment: GraphSnapshot

    # Delta metrics
    node_count_delta: int = 0
    edge_count_delta: int = 0

    # Type breakdowns
    type_deltas: Dict[str, int] = field(default_factory=dict)
    status_deltas: Dict[str, int] = field(default_factory=dict)
    edge_type_deltas: Dict[str, int] = field(default_factory=dict)

    # Structural changes
    nodes_only_in_baseline: List[str] = field(default_factory=list)
    nodes_only_in_treatment: List[str] = field(default_factory=list)
    nodes_in_both: List[str] = field(default_factory=list)

    def __post_init__(self):
        self._compute_deltas()

    def _compute_deltas(self):
        """Compute comparison metrics."""
        self.node_count_delta = self.treatment.node_count - self.baseline.node_count
        self.edge_count_delta = self.treatment.edge_count - self.baseline.edge_count

        # Node ID sets
        baseline_ids = {n.id for n in self.baseline.nodes}
        treatment_ids = {n.id for n in self.treatment.nodes}

        self.nodes_only_in_baseline = list(baseline_ids - treatment_ids)
        self.nodes_only_in_treatment = list(treatment_ids - baseline_ids)
        self.nodes_in_both = list(baseline_ids & treatment_ids)

        # Type counts
        baseline_types = {}
        for n in self.baseline.nodes:
            baseline_types[n.type] = baseline_types.get(n.type, 0) + 1

        treatment_types = {}
        for n in self.treatment.nodes:
            treatment_types[n.type] = treatment_types.get(n.type, 0) + 1

        all_types = set(baseline_types.keys()) | set(treatment_types.keys())
        for t in all_types:
            self.type_deltas[t] = treatment_types.get(t, 0) - baseline_types.get(t, 0)

        # Status counts
        baseline_statuses = {}
        for n in self.baseline.nodes:
            baseline_statuses[n.status] = baseline_statuses.get(n.status, 0) + 1

        treatment_statuses = {}
        for n in self.treatment.nodes:
            treatment_statuses[n.status] = treatment_statuses.get(n.status, 0) + 1

        all_statuses = set(baseline_statuses.keys()) | set(treatment_statuses.keys())
        for s in all_statuses:
            self.status_deltas[s] = treatment_statuses.get(s, 0) - baseline_statuses.get(s, 0)

        # Edge type counts
        baseline_edge_types = {}
        for e in self.baseline.edges:
            baseline_edge_types[e.type] = baseline_edge_types.get(e.type, 0) + 1

        treatment_edge_types = {}
        for e in self.treatment.edges:
            treatment_edge_types[e.type] = treatment_edge_types.get(e.type, 0) + 1

        all_edge_types = set(baseline_edge_types.keys()) | set(treatment_edge_types.keys())
        for et in all_edge_types:
            self.edge_type_deltas[et] = treatment_edge_types.get(et, 0) - baseline_edge_types.get(et, 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "baseline": self.baseline.to_dict(),
            "treatment": self.treatment.to_dict(),
            "node_count_delta": self.node_count_delta,
            "edge_count_delta": self.edge_count_delta,
            "type_deltas": self.type_deltas,
            "status_deltas": self.status_deltas,
            "edge_type_deltas": self.edge_type_deltas,
            "nodes_only_in_baseline": self.nodes_only_in_baseline,
            "nodes_only_in_treatment": self.nodes_only_in_treatment,
            "nodes_in_both_count": len(self.nodes_in_both),
        }


def compare_snapshots(
    baseline: GraphSnapshot,
    treatment: GraphSnapshot,
) -> GraphComparison:
    """
    Compare two graph snapshots.

    Used for regression testing in the Development View.

    Args:
        baseline: The reference snapshot (left side)
        treatment: The new snapshot (right side)

    Returns:
        GraphComparison with computed metrics
    """
    return GraphComparison(baseline=baseline, treatment=treatment)
