"""
PARAGON GRAPH DATABASE - The Rust-Accelerated Brain

This is the most critical file in the system. It bridges Python's UUID strings
with rustworkx's integer indices, enabling:
- O(1) node/edge lookup by business ID
- Rust-native graph algorithms (waves, descendants, topological sort)
- Batch operations that cross Python/Rust boundary once, not N times

Architecture (The Bridge Pattern):
  Python Layer (Business Logic)
  - Uses string UUIDs: "abc123", "def456"
  - Calls: db.add_node(data), db.get_descendants("abc123")

  Bridge Layer (This File)
  - _node_map: Dict[str, int]  (UUID -> Index)
  - _inv_map: Dict[int, str]   (Index -> UUID)

  Rust Layer (rustworkx.PyDiGraph)
  - Uses integer indices: 0, 1, 2, ...
  - Vectorized ops: rx.layers(), rx.descendants()

Performance Characteristics:
- Node lookup: O(1) via _node_map
- Wave computation: O(V+E) via rx.layers (vs O(V^2) Python recursion)
- Batch insert: Single Rust call for N nodes
- Memory: ~60% of NetworkX for same graph
"""
import rustworkx as rx
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator
from pathlib import Path
import polars as pl

from core.schemas import NodeData, EdgeData, NodePayload, EdgePayload
from core.ontology import (
    NodeType, EdgeType, NodeStatus,
    TopologyConstraint, EdgeConstraint, ConstraintMode,
    StructuralTrigger, EdgePattern, StatusPattern,
    TOPOLOGY_CONSTRAINTS, STRUCTURAL_TRIGGERS,
    get_constraint, get_required_edges, validate_status_for_type,
)

# Infrastructure integrations (graceful degradation if not available)
try:
    from infrastructure.logger import get_logger as get_mutation_logger
    from infrastructure.metrics import get_collector as get_metrics_collector
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

# Event bus for real-time notifications (graceful degradation)
try:
    from infrastructure.event_bus import get_event_bus, GraphEvent, EventType
    import time as _event_time
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class GraphError(Exception):
    """Base exception for graph operations."""
    pass


class NodeNotFoundError(GraphError):
    """Raised when a node UUID is not in the graph."""
    def __init__(self, node_id: str):
        self.node_id = node_id
        super().__init__(f"Node not found: {node_id}")


class EdgeNotFoundError(GraphError):
    """Raised when an edge is not in the graph."""
    def __init__(self, source_id: str, target_id: str):
        self.source_id = source_id
        self.target_id = target_id
        super().__init__(f"Edge not found: {source_id} -> {target_id}")


class GraphInvariantError(GraphError):
    """Raised when a graph invariant is violated (cycles in DAG, etc.)."""
    pass


class DuplicateNodeError(GraphError):
    """Raised when attempting to add a node with existing ID."""
    def __init__(self, node_id: str):
        self.node_id = node_id
        super().__init__(f"Node already exists: {node_id}")


class TopologyViolationError(GraphError):
    """Raised when a topology constraint is violated."""
    def __init__(self, message: str, constraint: Optional[EdgeConstraint] = None):
        self.constraint = constraint
        super().__init__(message)


# =============================================================================
# PARAGON DATABASE (The Graph Engine)
# =============================================================================

class ParagonDB:
    """
    In-memory graph database backed by rustworkx.

    This class provides a business-friendly interface over rustworkx's
    integer-indexed PyDiGraph. All public methods accept/return string UUIDs;
    the translation to/from integer indices is handled internally.

    Usage:
        db = ParagonDB()

        # Add nodes
        node = NodeData.create(type="CODE", content="def hello(): pass")
        db.add_node(node)

        # Add edges
        edge = EdgeData.depends_on(node1.id, node2.id)
        db.add_edge(edge)

        # Compute execution waves
        waves = db.get_waves()  # [[layer0_nodes], [layer1_nodes], ...]

    Thread Safety:
        NOT thread-safe. Use external locking if needed for concurrent access.
    """

    def __init__(self, multigraph: bool = False):
        """
        Initialize an empty graph database.

        Args:
            multigraph: If True, allow multiple edges between same nodes.
                       Default False for DAG execution graphs.
        """
        # Core storage: Rust-native directed graph
        self._graph: rx.PyDiGraph = rx.PyDiGraph(multigraph=multigraph)

        # The Bridge: bidirectional UUID <-> index mapping
        self._node_map: Dict[str, int] = {}   # UUID -> rustworkx index
        self._inv_map: Dict[int, str] = {}    # rustworkx index -> UUID

        # Edge tracking for fast lookup
        self._edge_map: Dict[Tuple[str, str], int] = {}  # (src_id, tgt_id) -> edge_idx

        # Event bus for real-time notifications
        self._event_bus = get_event_bus() if EVENT_BUS_AVAILABLE else None

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return self._graph.num_nodes()

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return self._graph.num_edges()

    @property
    def is_empty(self) -> bool:
        """True if graph has no nodes."""
        return self.node_count == 0

    # =========================================================================
    # NODE OPERATIONS
    # =========================================================================

    def add_node(self, data: NodeData, allow_duplicate: bool = False) -> int:
        """
        Add a node to the graph.

        Args:
            data: NodeData payload for the node
            allow_duplicate: If True, return existing index for duplicate IDs.
                           If False, raise DuplicateNodeError.

        Returns:
            The rustworkx index of the node (existing or new)

        Raises:
            DuplicateNodeError: If node ID exists and allow_duplicate=False
        """
        node_id = data.id

        # Check for existing node
        if node_id in self._node_map:
            if allow_duplicate:
                return self._node_map[node_id]
            raise DuplicateNodeError(node_id)

        # Add to rustworkx graph (crosses Python/Rust boundary once)
        idx = self._graph.add_node(data)

        # Update both maps
        self._node_map[node_id] = idx
        self._inv_map[idx] = node_id

        # Log mutation event (if observability available)
        if OBSERVABILITY_AVAILABLE:
            try:
                logger = get_mutation_logger()
                logger.log_node_created(
                    node_id=node_id,
                    node_type=data.node_type.value,
                    agent_id=data.payload.created_by if hasattr(data.payload, 'created_by') else None,
                )

                # Record metrics
                metrics = get_metrics_collector()
                metrics.record_node_created(
                    node_id=node_id,
                    node_type=data.node_type.value,
                    created_by=data.payload.created_by if hasattr(data.payload, 'created_by') else "system",
                )
            except Exception:
                pass  # Don't let observability break graph operations

        # Publish event for real-time notifications
        if self._event_bus:
            try:
                self._event_bus.publish(GraphEvent(
                    type=EventType.NODE_CREATED,
                    payload={
                        "node_id": node_id,
                        "node_type": data.node_type.value,
                        "status": data.status.value if hasattr(data.status, 'value') else data.status,
                        "content": data.content[:100] if data.content else "",  # Truncate for performance
                        "created_by": data.payload.created_by if hasattr(data.payload, 'created_by') else "system",
                    },
                    timestamp=_event_time.time(),
                    source="graph_db"
                ))
            except Exception:
                pass  # Don't let event bus break graph operations

        return idx

    def add_nodes_batch(self, nodes: List[NodeData]) -> List[int]:
        """
        Add multiple nodes in a single Rust call.

        Performance Critical: This method crosses the Python/Rust boundary
        once for N nodes, vs N times for individual add_node calls.

        Args:
            nodes: List of NodeData payloads

        Returns:
            List of rustworkx indices (in same order as input)

        Note:
            Duplicate nodes are silently skipped (existing index returned).
        """
        if not nodes:
            return []

        # Separate new nodes from existing
        new_nodes: List[NodeData] = []
        new_indices: List[int] = []
        result_indices: List[int] = []
        node_positions: List[int] = []  # Track positions of new nodes

        for i, node in enumerate(nodes):
            if node.id in self._node_map:
                # Already exists, use existing index
                result_indices.append(self._node_map[node.id])
            else:
                # Mark for batch insert
                new_nodes.append(node)
                node_positions.append(i)
                result_indices.append(-1)  # Placeholder

        if new_nodes:
            # Single Rust call for all new nodes
            indices = self._graph.add_nodes_from(new_nodes)

            # Update maps and fill in result indices
            for j, (node, idx) in enumerate(zip(new_nodes, indices)):
                self._node_map[node.id] = idx
                self._inv_map[idx] = node.id
                result_indices[node_positions[j]] = idx

        return result_indices

    def get_node(self, node_id: str) -> NodeData:
        """
        Retrieve a node by its UUID.

        Args:
            node_id: The business UUID of the node

        Returns:
            The NodeData payload

        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        if node_id not in self._node_map:
            raise NodeNotFoundError(node_id)

        idx = self._node_map[node_id]
        return self._graph[idx]

    def get_node_by_index(self, idx: int) -> NodeData:
        """
        Retrieve a node by its rustworkx index.

        Primarily for internal use and debugging.
        """
        if idx not in self._inv_map:
            raise GraphError(f"Invalid index: {idx}")
        return self._graph[idx]

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""
        return node_id in self._node_map

    def update_node(self, node_id: str, data: NodeData) -> None:
        """
        Replace a node's data in-place.

        Args:
            node_id: The node to update
            data: New NodeData (must have same ID)

        Raises:
            NodeNotFoundError: If node doesn't exist
            ValueError: If data.id doesn't match node_id
        """
        if data.id != node_id:
            raise ValueError(f"Node ID mismatch: {node_id} vs {data.id}")

        if node_id not in self._node_map:
            raise NodeNotFoundError(node_id)

        idx = self._node_map[node_id]
        self._graph[idx] = data

        # Publish event for real-time notifications
        if self._event_bus:
            try:
                self._event_bus.publish(GraphEvent(
                    type=EventType.NODE_UPDATED,
                    payload={
                        "node_id": node_id,
                        "node_type": data.node_type.value,
                        "status": data.status.value if hasattr(data.status, 'value') else data.status,
                        "content": data.content[:100] if data.content else "",
                    },
                    timestamp=_event_time.time(),
                    source="graph_db"
                ))
            except Exception:
                pass  # Don't let event bus break graph operations

    def remove_node(self, node_id: str) -> NodeData:
        """
        Remove a node and all its edges.

        Args:
            node_id: The node to remove

        Returns:
            The removed NodeData

        Raises:
            NodeNotFoundError: If node doesn't exist

        Warning:
            This operation invalidates indices. After removal, the _inv_map
            may have gaps. Use with caution in hot paths.
        """
        if node_id not in self._node_map:
            raise NodeNotFoundError(node_id)

        idx = self._node_map[node_id]
        data = self._graph[idx]

        # Remove edges first (from edge_map)
        edges_to_remove = [
            key for key in self._edge_map
            if key[0] == node_id or key[1] == node_id
        ]
        for key in edges_to_remove:
            del self._edge_map[key]

        # Remove from graph (also removes incident edges)
        self._graph.remove_node(idx)

        # Clean up maps
        del self._node_map[node_id]
        del self._inv_map[idx]

        # Publish event for real-time notifications
        if self._event_bus:
            try:
                self._event_bus.publish(GraphEvent(
                    type=EventType.NODE_DELETED,
                    payload={
                        "node_id": node_id,
                        "node_type": data.node_type.value,
                    },
                    timestamp=_event_time.time(),
                    source="graph_db"
                ))
            except Exception:
                pass  # Don't let event bus break graph operations

        return data

    def iter_nodes(self) -> Iterator[NodeData]:
        """Iterate over all nodes."""
        return iter(self._graph.nodes())

    def get_all_nodes(self) -> List[NodeData]:
        """Get all nodes as a list."""
        return list(self._graph.nodes())

    def get_nodes_by_type(self, node_type: str) -> List[NodeData]:
        """Get all nodes of a specific type."""
        return [n for n in self._graph.nodes() if n.type == node_type]

    def get_nodes_by_status(self, status: str) -> List[NodeData]:
        """Get all nodes with a specific status."""
        return [n for n in self._graph.nodes() if n.status == status]

    # =========================================================================
    # EDGE OPERATIONS
    # =========================================================================

    def add_edge(self, edge: EdgeData, check_cycle: bool = True) -> int:
        """
        Add an edge between two nodes.

        Graph-Native Invariant: The graph MUST remain a DAG.
        This is enforced at write time, not as a query.

        Args:
            edge: EdgeData with source_id, target_id, and type
            check_cycle: If True (default), prevent edges that would create cycles.
                        Set to False only for batch operations that check afterwards.

        Returns:
            The rustworkx edge index

        Raises:
            NodeNotFoundError: If source or target node doesn't exist
            GraphInvariantError: If adding this edge would create a cycle
        """
        source_id = edge.source_id
        target_id = edge.target_id

        if source_id not in self._node_map:
            raise NodeNotFoundError(source_id)
        if target_id not in self._node_map:
            raise NodeNotFoundError(target_id)

        src_idx = self._node_map[source_id]
        tgt_idx = self._node_map[target_id]

        # Wave 2 Refactor: Enforce DAG invariant at write time
        # If there's a path from target to source, adding source→target creates cycle
        if check_cycle and source_id != target_id:
            # Check if target can reach source (would create cycle)
            try:
                # rx.has_path returns True if there's a path from first to second arg
                if rx.has_path(self._graph, tgt_idx, src_idx):
                    raise GraphInvariantError(
                        f"Cannot add edge {source_id} -> {target_id}: "
                        f"would create cycle (path exists from target to source)"
                    )
            except Exception as e:
                if isinstance(e, GraphInvariantError):
                    raise
                # If has_path fails for other reasons, continue (edge case)
                pass

        # Self-loops are always cycles
        if source_id == target_id:
            raise GraphInvariantError(
                f"Cannot add self-loop edge {source_id} -> {target_id}"
            )

        # Add to graph
        edge_idx = self._graph.add_edge(src_idx, tgt_idx, edge)

        # Track in edge map
        self._edge_map[(source_id, target_id)] = edge_idx

        # Publish event for real-time notifications
        if self._event_bus:
            try:
                self._event_bus.publish(GraphEvent(
                    type=EventType.EDGE_CREATED,
                    payload={
                        "source_id": source_id,
                        "target_id": target_id,
                        "edge_type": edge.type.value if hasattr(edge.type, 'value') else edge.type,
                        "weight": edge.weight,
                    },
                    timestamp=_event_time.time(),
                    source="graph_db"
                ))
            except Exception:
                pass  # Don't let event bus break graph operations

        return edge_idx

    def add_edges_batch(self, edges: List[EdgeData]) -> List[int]:
        """
        Add multiple edges in batch.

        Args:
            edges: List of EdgeData to add

        Returns:
            List of edge indices

        Raises:
            NodeNotFoundError: If any source or target doesn't exist
        """
        if not edges:
            return []

        # Convert to tuples for rustworkx
        edge_tuples = []
        for edge in edges:
            if edge.source_id not in self._node_map:
                raise NodeNotFoundError(edge.source_id)
            if edge.target_id not in self._node_map:
                raise NodeNotFoundError(edge.target_id)

            src_idx = self._node_map[edge.source_id]
            tgt_idx = self._node_map[edge.target_id]
            edge_tuples.append((src_idx, tgt_idx, edge))

        # Single Rust call
        indices = self._graph.add_edges_from(edge_tuples)

        # Update edge map
        for edge, idx in zip(edges, indices):
            self._edge_map[(edge.source_id, edge.target_id)] = idx

        return list(indices)

    def get_edge(self, source_id: str, target_id: str) -> EdgeData:
        """
        Get edge data between two nodes.

        Raises:
            EdgeNotFoundError: If edge doesn't exist
        """
        key = (source_id, target_id)
        if key not in self._edge_map:
            raise EdgeNotFoundError(source_id, target_id)

        edge_idx = self._edge_map[key]
        return self._graph.get_edge_data_by_index(edge_idx)

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Check if an edge exists."""
        return (source_id, target_id) in self._edge_map

    def remove_edge(self, source_id: str, target_id: str) -> EdgeData:
        """
        Remove an edge.

        Returns:
            The removed EdgeData

        Raises:
            EdgeNotFoundError: If edge doesn't exist
        """
        key = (source_id, target_id)
        if key not in self._edge_map:
            raise EdgeNotFoundError(source_id, target_id)

        src_idx = self._node_map[source_id]
        tgt_idx = self._node_map[target_id]

        # Get data before removal
        edge_data = self._graph.get_edge_data(src_idx, tgt_idx)

        # Remove from graph
        self._graph.remove_edge(src_idx, tgt_idx)

        # Clean up map
        del self._edge_map[key]

        # Publish event for real-time notifications
        if self._event_bus:
            try:
                self._event_bus.publish(GraphEvent(
                    type=EventType.EDGE_DELETED,
                    payload={
                        "source_id": source_id,
                        "target_id": target_id,
                        "edge_type": edge_data.type.value if hasattr(edge_data.type, 'value') else edge_data.type,
                    },
                    timestamp=_event_time.time(),
                    source="graph_db"
                ))
            except Exception:
                pass  # Don't let event bus break graph operations

        return edge_data

    def get_all_edges(self) -> List[EdgeData]:
        """Get all edges as a list."""
        edges = []
        for i in range(self._graph.num_edges()):
            try:
                edge_data = self._graph.get_edge_data_by_index(i)
                if edge_data is not None:
                    edges.append(edge_data)
            except (IndexError, ValueError):
                # Edge index might be invalid if edges were removed
                continue
        return edges

    def get_incoming_edges(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get all edges pointing TO a node.

        Returns:
            List of dicts with {source, target, type, ...} for each edge
        """
        if node_id not in self._node_map:
            return []

        idx = self._node_map[node_id]
        result = []

        for pred_idx in self._graph.predecessor_indices(idx):
            edge_data = self._graph.get_edge_data(pred_idx, idx)
            if edge_data:
                pred_id = self._inv_map.get(pred_idx, "")
                result.append({
                    "source": pred_id,
                    "target": node_id,
                    "type": edge_data.type,
                    "weight": edge_data.weight,
                })

        return result

    def get_outgoing_edges(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get all edges pointing FROM a node.

        Returns:
            List of dicts with {source, target, type, ...} for each edge
        """
        if node_id not in self._node_map:
            return []

        idx = self._node_map[node_id]
        result = []

        for succ_idx in self._graph.successor_indices(idx):
            edge_data = self._graph.get_edge_data(idx, succ_idx)
            if edge_data:
                succ_id = self._inv_map.get(succ_idx, "")
                result.append({
                    "source": node_id,
                    "target": succ_id,
                    "type": edge_data.type,
                    "weight": edge_data.weight,
                })

        return result

    # =========================================================================
    # GRAPH TRAVERSAL (Rust-Accelerated)
    # =========================================================================

    def get_waves(self, first_layer: Optional[List[str]] = None) -> List[List[NodeData]]:
        """
        Compute execution waves (layers) using Rust-native algorithm.

        This replaces the Python-recursive "Wavefront" logic with rx.layers(),
        achieving O(V+E) complexity vs O(V^2) for recursive Python.

        Args:
            first_layer: Optional list of node IDs to start from.
                        If None, starts from all root nodes (no predecessors).

        Returns:
            List of waves, each wave is a list of NodeData that can
            execute in parallel within that wave.

        Raises:
            GraphInvariantError: If the graph contains cycles

        Example:
            waves = db.get_waves()
            for wave_idx, wave in enumerate(waves):
                # Execute all nodes in this wave in parallel
                parallel_execute(wave)
        """
        if self.is_empty:
            return []

        # Determine starting nodes
        if first_layer is not None:
            start_indices = [self._node_map[nid] for nid in first_layer
                           if nid in self._node_map]
        else:
            # Find root nodes (no predecessors)
            start_indices = [
                idx for idx in self._graph.node_indices()
                if self._graph.in_degree(idx) == 0
            ]

        if not start_indices:
            # No roots found - might be a cycle or disconnected
            # Try with all nodes
            start_indices = list(self._graph.node_indices())

        try:
            # Rust-native layer computation
            # rx.layers returns list of lists of node DATA (not indices)
            layers = rx.layers(self._graph, start_indices)
            return [list(layer) for layer in layers]

        except rx.DAGHasCycle:
            raise GraphInvariantError("Cycle detected in DAG execution flow")

    def get_descendants(self, node_id: str) -> List[NodeData]:
        """
        Get all descendants (transitive successors) of a node.

        Uses Rust-native rx.descendants() for O(V+E) traversal.

        Args:
            node_id: The starting node

        Returns:
            List of all descendant NodeData (not including the starting node)

        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        if node_id not in self._node_map:
            raise NodeNotFoundError(node_id)

        idx = self._node_map[node_id]
        desc_indices = rx.descendants(self._graph, idx)

        return [self._graph[i] for i in desc_indices]

    def get_ancestors(self, node_id: str) -> List[NodeData]:
        """
        Get all ancestors (transitive predecessors) of a node.

        Args:
            node_id: The starting node

        Returns:
            List of all ancestor NodeData
        """
        if node_id not in self._node_map:
            raise NodeNotFoundError(node_id)

        idx = self._node_map[node_id]
        anc_indices = rx.ancestors(self._graph, idx)

        return [self._graph[i] for i in anc_indices]

    def get_successors(self, node_id: str) -> List[NodeData]:
        """Get immediate successors (children) of a node."""
        if node_id not in self._node_map:
            raise NodeNotFoundError(node_id)

        idx = self._node_map[node_id]
        return [self._graph[i] for i in self._graph.successor_indices(idx)]

    def get_predecessors(self, node_id: str) -> List[NodeData]:
        """Get immediate predecessors (parents) of a node."""
        if node_id not in self._node_map:
            raise NodeNotFoundError(node_id)

        idx = self._node_map[node_id]
        return [self._graph[i] for i in self._graph.predecessor_indices(idx)]

    def get_root_nodes(self) -> List[NodeData]:
        """Get all nodes with no predecessors (entry points)."""
        return [
            self._graph[idx]
            for idx in self._graph.node_indices()
            if self._graph.in_degree(idx) == 0
        ]

    def get_leaf_nodes(self) -> List[NodeData]:
        """Get all nodes with no successors (terminal nodes)."""
        return [
            self._graph[idx]
            for idx in self._graph.node_indices()
            if self._graph.out_degree(idx) == 0
        ]

    def topological_sort(self) -> List[NodeData]:
        """
        Return nodes in topological order.

        Raises:
            GraphInvariantError: If graph has cycles
        """
        try:
            sorted_indices = rx.topological_sort(self._graph)
            return [self._graph[idx] for idx in sorted_indices]
        except rx.DAGHasCycle:
            raise GraphInvariantError("Cannot topologically sort: graph has cycles")

    def has_cycle(self) -> bool:
        """Check if the graph contains any cycles."""
        return not rx.is_directed_acyclic_graph(self._graph)

    def get_dominators(self, node_id: str) -> List[NodeData]:
        """
        Get the immediate dominators of a node.

        A node D dominates node N if every path from a root to N goes through D.
        This is the "Economic Topology" - the minimal set of ancestors that
        MUST be understood to understand node N.

        For context window optimization:
        - get_ancestors() returns ALL upstream nodes (can explode on deep graphs)
        - get_dominators() returns only the ESSENTIAL control points

        Uses Rust-native rx.immediate_dominators() for O(V+E) computation.

        Args:
            node_id: The target node

        Returns:
            List of dominator NodeData (nodes that gate understanding of target)

        Raises:
            NodeNotFoundError: If node doesn't exist
        """
        if node_id not in self._node_map:
            raise NodeNotFoundError(node_id)

        target_idx = self._node_map[node_id]

        # For dominator analysis, we need to compute from roots
        # rx.immediate_dominators computes dominators for the entire graph
        # from a given start node
        root_nodes = [
            idx for idx in self._graph.node_indices()
            if self._graph.in_degree(idx) == 0
        ]

        if not root_nodes:
            return []

        # Collect dominators from all roots
        dominator_indices: Set[int] = set()
        for root_idx in root_nodes:
            try:
                # immediate_dominators returns Dict[node_idx, dominator_idx]
                dom_map = rx.immediate_dominators(self._graph, root_idx)
                # Walk the dominator chain for our target
                current = target_idx
                while current in dom_map:
                    dom_idx = dom_map[current]
                    if dom_idx == current:
                        break  # Self-dominator (root)
                    dominator_indices.add(dom_idx)
                    current = dom_idx
            except Exception:
                # Some roots may not reach our target
                continue

        # Exclude the target node itself
        dominator_indices.discard(target_idx)

        return [self._graph[i] for i in dominator_indices]

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def find_nodes(
        self,
        type: Optional[str] = None,
        status: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> List[NodeData]:
        """
        Find nodes matching criteria.

        Args:
            type: Filter by node type
            status: Filter by status
            created_by: Filter by creator

        Returns:
            List of matching NodeData
        """
        results = []
        for node in self._graph.nodes():
            if type is not None and node.type != type:
                continue
            if status is not None and node.status != status:
                continue
            if created_by is not None and node.created_by != created_by:
                continue
            results.append(node)
        return results

    def find_processable_nodes(self) -> List[NodeData]:
        """Find all nodes that are ready to be processed."""
        return [n for n in self._graph.nodes() if n.is_processable()]

    def find_pending_nodes(self) -> List[NodeData]:
        """Find all nodes with PENDING status."""
        return self.get_nodes_by_status(NodeStatus.PENDING.value)

    # =========================================================================
    # MESSAGE-TO-NODE MAPPING (Dialogue-to-Graph Correspondence)
    # =========================================================================

    def link_message_to_node(
        self,
        message_id: str,
        node_id: str,
        edge_type: str = EdgeType.REFERENCES.value,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Create a link from a message/dialogue turn to a node.

        This establishes the dialogue-to-graph correspondence, allowing UI
        to show which messages reference which nodes and vice versa.

        Args:
            message_id: ID of the MESSAGE node or dialogue turn ID
            node_id: ID of the node being referenced
            edge_type: Type of edge (REFERENCES or DEFINES_DIALOGUE)
            metadata: Optional metadata for the edge (e.g., turn context)

        Returns:
            Edge index

        Raises:
            NodeNotFoundError: If either node doesn't exist

        Example:
            # When a dialogue turn references a node
            db.link_message_to_node("turn_123", "node_456", EdgeType.REFERENCES.value)

            # When a dialogue turn creates/defines a node
            db.link_message_to_node("turn_123", "node_456", EdgeType.DEFINES_DIALOGUE.value)
        """
        edge_metadata = metadata or {}
        edge = EdgeData.create(
            source_id=message_id,
            target_id=node_id,
            type=edge_type,
            metadata=edge_metadata,
        )
        return self.add_edge(edge, check_cycle=False)  # Messages may not be in DAG

    def get_messages_for_node(self, node_id: str) -> List[NodeData]:
        """
        Get all MESSAGE nodes that reference a given node.

        This answers: "Which dialogue turns/messages mention this node?"

        Args:
            node_id: The node to query

        Returns:
            List of MESSAGE nodes with REFERENCES or DEFINES_DIALOGUE edges to this node

        Example:
            messages = db.get_messages_for_node("code_node_123")
            for msg in messages:
                print(f"Turn {msg.data.get('turn_number')}: {msg.content}")
        """
        if node_id not in self._node_map:
            return []

        idx = self._node_map[node_id]
        message_nodes = []

        # Find all predecessors connected via REFERENCES or DEFINES_DIALOGUE
        for pred_idx in self._graph.predecessor_indices(idx):
            edge_data = self._graph.get_edge_data(pred_idx, idx)
            if edge_data and edge_data.type in (
                EdgeType.REFERENCES.value,
                EdgeType.DEFINES_DIALOGUE.value,
            ):
                pred_node = self._graph[pred_idx]
                # Only include MESSAGE nodes
                if pred_node.type == NodeType.MESSAGE.value:
                    message_nodes.append(pred_node)

        return message_nodes

    def get_nodes_from_message(self, message_id: str) -> List[NodeData]:
        """
        Get all nodes referenced by a message/dialogue turn.

        This answers: "Which nodes were discussed in this dialogue turn?"

        Args:
            message_id: The MESSAGE node ID or dialogue turn ID

        Returns:
            List of nodes referenced by this message

        Example:
            nodes = db.get_nodes_from_message("turn_5")
            for node in nodes:
                print(f"Referenced {node.type}: {node.id}")
        """
        if message_id not in self._node_map:
            return []

        idx = self._node_map[message_id]
        referenced_nodes = []

        # Find all successors connected via REFERENCES or DEFINES_DIALOGUE
        for succ_idx in self._graph.successor_indices(idx):
            edge_data = self._graph.get_edge_data(idx, succ_idx)
            if edge_data and edge_data.type in (
                EdgeType.REFERENCES.value,
                EdgeType.DEFINES_DIALOGUE.value,
            ):
                succ_node = self._graph[succ_idx]
                referenced_nodes.append(succ_node)

        return referenced_nodes

    def update_node_dialogue_metadata(
        self,
        node_id: str,
        dialogue_turn_id: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
        definition_turn: Optional[str] = None,
        referenced_in_turns: Optional[List[str]] = None,
        hover_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update the dialogue-related metadata in a node's data dictionary.

        This is a convenience method for updating the structured fields
        documented in NodeData.data.

        Args:
            node_id: Node to update
            dialogue_turn_id: ID of the turn that defined this node
            message_ids: List of message IDs referencing this node
            definition_turn: Turn ID when first defined
            referenced_in_turns: List of turn IDs that mention this node
            hover_metadata: UI hover display metadata

        Raises:
            NodeNotFoundError: If node doesn't exist

        Example:
            db.update_node_dialogue_metadata(
                node_id="spec_123",
                definition_turn="turn_3",
                referenced_in_turns=["turn_3", "turn_5", "turn_7"],
                hover_metadata={
                    "phase": "planning",
                    "created_by": "ARCHITECT",
                    "status": "VERIFIED",
                    "key_findings": ["Must handle edge case X"]
                }
            )
        """
        if node_id not in self._node_map:
            raise NodeNotFoundError(node_id)

        node = self.get_node(node_id)

        # Update only the provided fields
        if dialogue_turn_id is not None:
            node.data["dialogue_turn_id"] = dialogue_turn_id

        if message_ids is not None:
            node.data["message_ids"] = message_ids

        if definition_turn is not None:
            node.data["definition_turn"] = definition_turn

        if referenced_in_turns is not None:
            # Merge with existing if present
            existing = node.data.get("referenced_in_turns", [])
            # Use set to deduplicate, then convert back to list
            merged = list(set(existing + referenced_in_turns))
            node.data["referenced_in_turns"] = merged

        if hover_metadata is not None:
            # Merge with existing hover_metadata if present
            existing_hover = node.data.get("hover_metadata", {})
            existing_hover.update(hover_metadata)
            node.data["hover_metadata"] = existing_hover

        # Touch the node to update timestamps
        node.touch()

        # Update in graph
        self.update_node(node_id, node)

    def get_node_hover_metadata(self, node_id: str) -> Dict[str, Any]:
        """
        Get the hover metadata for a node, computing it if not already present.

        This is useful for UI tooltips/hovers, showing context about a node
        without requiring the full node data.

        Args:
            node_id: Node to get hover data for

        Returns:
            Dictionary with hover metadata, or empty dict if node not found

        Example:
            hover = db.get_node_hover_metadata("code_123")
            # Returns: {
            #   "phase": "testing",
            #   "created_by": "BUILDER",
            #   "status": "TESTED",
            #   "teleology_status": "justified",
            #   "related_nodes": ["spec_456", "test_789"]
            # }
        """
        if node_id not in self._node_map:
            return {}

        node = self.get_node(node_id)

        # Return existing hover metadata if present
        if "hover_metadata" in node.data:
            return node.data["hover_metadata"]

        # Otherwise, compute it on the fly
        related_node_ids = []

        # Get immediate neighbors
        for pred in self.get_predecessors(node_id):
            related_node_ids.append(pred.id)
        for succ in self.get_successors(node_id):
            related_node_ids.append(succ.id)

        hover_data = {
            "phase": self._infer_phase_from_node(node),
            "created_by": node.created_by,
            "created_at": node.created_at,
            "status": node.status,
            "teleology_status": node.teleology_status,
            "related_nodes": related_node_ids[:10],  # Limit to first 10
        }

        return hover_data

    def _infer_phase_from_node(self, node: NodeData) -> str:
        """
        Infer the current phase from a node's type and status.

        Internal helper for get_node_hover_metadata.
        """
        node_type = node.type

        # Map node types to phases
        if node_type == NodeType.REQ.value:
            return "requirements"
        elif node_type == NodeType.RESEARCH.value:
            return "research"
        elif node_type in (NodeType.SPEC.value, NodeType.PLAN.value):
            return "planning"
        elif node_type == NodeType.CODE.value:
            if node.status in (NodeStatus.TESTING.value, NodeStatus.TESTED.value):
                return "testing"
            return "building"
        elif node_type in (NodeType.TEST.value, NodeType.TEST_SUITE.value):
            return "testing"
        elif node_type == NodeType.DOC.value:
            return "documentation"
        else:
            return "unknown"

    # =========================================================================
    # SEMANTIC SIMILARITY (Hybrid Context Assembly - Fuzzy Layer)
    # =========================================================================

    def find_similar_nodes(
        self,
        query: str,
        threshold: float = 0.6,
        limit: int = 5,
        node_type: Optional[str] = None,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[NodeData, float]]:
        """
        Find nodes semantically similar to a query text.

        This is the FUZZY layer of hybrid context assembly.
        Use in conjunction with graph edges (compiler layer) for best results.

        Architecture:
        - Query: Compute embedding for query text
        - Search: O(n) cosine similarity scan over node embeddings
        - Filter: Apply type filter and exclusion set
        - Rank: Return top-k by similarity score

        Args:
            query: Text to find similar nodes for
            threshold: Minimum cosine similarity (0.6 recommended)
            limit: Maximum nodes to return
            node_type: Optional filter by node type
            exclude_ids: Node IDs to exclude from results

        Returns:
            List of (NodeData, similarity_score) tuples, sorted by score

        Note:
            Returns empty list if embeddings unavailable or no matches found.
            This graceful degradation allows compiler-only fallback.
        """
        try:
            from core.embeddings import compute_embedding, cosine_similarity, is_available
        except ImportError:
            return []

        if not is_available():
            return []

        # Compute query embedding
        query_embedding = compute_embedding(query)
        if query_embedding is None:
            return []

        exclude_ids = exclude_ids or set()
        results = []

        for node in self._graph.nodes():
            # Apply filters
            if node.id in exclude_ids:
                continue
            if node_type and node.type != node_type:
                continue
            if node.embedding is None:
                continue

            # Compute similarity
            score = cosine_similarity(query_embedding, node.embedding)
            if score >= threshold:
                results.append((node, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def find_similar_to_node(
        self,
        node_id: str,
        threshold: float = 0.6,
        limit: int = 5,
        node_type: Optional[str] = None,
    ) -> List[Tuple[NodeData, float]]:
        """
        Find nodes similar to an existing node by its embedding.

        Useful for finding related implementations, tests, or patterns.

        Args:
            node_id: Node to find similar nodes for
            threshold: Minimum similarity score
            limit: Maximum results
            node_type: Optional type filter

        Returns:
            List of (NodeData, score) tuples (excludes the query node)
        """
        if node_id not in self._node_map:
            return []

        node = self.get_node(node_id)
        if node.embedding is None:
            # Fallback: use content text
            return self.find_similar_nodes(
                query=node.content,
                threshold=threshold,
                limit=limit,
                node_type=node_type,
                exclude_ids={node_id},
            )

        # Use existing embedding for faster lookup
        try:
            from core.embeddings import cosine_similarity
        except ImportError:
            return []

        exclude_ids = {node_id}
        results = []

        for other in self._graph.nodes():
            if other.id in exclude_ids:
                continue
            if node_type and other.type != node_type:
                continue
            if other.embedding is None:
                continue

            score = cosine_similarity(node.embedding, other.embedding)
            if score >= threshold:
                results.append((other, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def update_node_embedding(self, node_id: str) -> bool:
        """
        Compute and store embedding for a node.

        Called automatically during node creation if embeddings available.
        Can also be called manually to update stale embeddings.

        Args:
            node_id: Node to update

        Returns:
            True if embedding was computed, False otherwise
        """
        if node_id not in self._node_map:
            return False

        try:
            from core.embeddings import compute_embedding
        except ImportError:
            return False

        node = self.get_node(node_id)
        embedding = compute_embedding(node.content)
        if embedding is not None:
            node.embedding = embedding
            return True
        return False

    def update_all_embeddings(self, batch_size: int = 100) -> int:
        """
        Compute embeddings for all nodes that don't have one.

        Useful for initial population or migration.

        Args:
            batch_size: Process nodes in batches for efficiency

        Returns:
            Number of embeddings computed
        """
        try:
            from core.embeddings import compute_embeddings_batch
        except ImportError:
            return 0

        nodes_needing_embeddings = [
            n for n in self._graph.nodes()
            if n.embedding is None and n.content
        ]

        if not nodes_needing_embeddings:
            return 0

        count = 0
        for i in range(0, len(nodes_needing_embeddings), batch_size):
            batch = nodes_needing_embeddings[i:i + batch_size]
            texts = [n.content for n in batch]
            embeddings = compute_embeddings_batch(texts)

            for node, embedding in zip(batch, embeddings):
                if embedding is not None:
                    node.embedding = embedding
                    count += 1

        return count

    # =========================================================================
    # TOPOLOGY VALIDATION (Graph-Native Constraints)
    # =========================================================================

    def validate_topology(
        self,
        node_id: str,
        mode: str = "hard",
    ) -> List[str]:
        """
        Validate a node against its topology constraints.

        This is the core of the Graph-Native design. Instead of checking
        transitions against a matrix, we check that the graph SHAPE is valid.

        Args:
            node_id: Node to validate
            mode: "hard" (only hard constraints), "soft" (all), or "all"

        Returns:
            List of violation messages (empty = valid)

        Example:
            violations = db.validate_topology("node123", mode="hard")
            if violations:
                raise TopologyViolationError(violations[0])
        """
        if node_id not in self._node_map:
            return [f"Node not found: {node_id}"]

        node = self.get_node(node_id)
        violations = []

        # Get constraints for this node type
        edge_constraints = get_required_edges(node.type, mode)

        for constraint in edge_constraints:
            violation = self._check_edge_constraint(node_id, node, constraint)
            if violation:
                violations.append(violation)

        # Check status is valid for this node type
        if not validate_status_for_type(node.type, node.status):
            violations.append(
                f"Invalid status '{node.status}' for node type '{node.type}'"
            )

        return violations

    def _check_edge_constraint(
        self,
        node_id: str,
        node: NodeData,
        constraint: EdgeConstraint,
    ) -> Optional[str]:
        """
        Check a single edge constraint against a node.

        Returns violation message or None if satisfied.
        """
        idx = self._node_map[node_id]

        # Count matching edges
        count = 0

        if constraint.direction in ("incoming", "any"):
            for pred_idx in self._graph.predecessor_indices(idx):
                # Check if there's an edge of the right type
                edge_data = self._graph.get_edge_data(pred_idx, idx)
                if edge_data and edge_data.type == constraint.edge_type:
                    # Check target node type if specified
                    if constraint.target_node_type:
                        pred_node = self._graph[pred_idx]
                        if pred_node.type != constraint.target_node_type:
                            continue
                    count += 1

        if constraint.direction in ("outgoing", "any"):
            for succ_idx in self._graph.successor_indices(idx):
                edge_data = self._graph.get_edge_data(idx, succ_idx)
                if edge_data and edge_data.type == constraint.edge_type:
                    if constraint.target_node_type:
                        succ_node = self._graph[succ_idx]
                        if succ_node.type != constraint.target_node_type:
                            continue
                    count += 1

        # Check min count
        if count < constraint.min_count:
            return (
                f"Node '{node_id}' ({node.type}) requires at least "
                f"{constraint.min_count} {constraint.direction} edge(s) of type "
                f"'{constraint.edge_type}', but has {count}"
            )

        # Check max count
        if constraint.max_count is not None and count > constraint.max_count:
            return (
                f"Node '{node_id}' ({node.type}) allows at most "
                f"{constraint.max_count} {constraint.direction} edge(s) of type "
                f"'{constraint.edge_type}', but has {count}"
            )

        return None

    def validate_edge_topology(
        self,
        edge: EdgeData,
    ) -> List[str]:
        """
        Validate an edge against topology constraints BEFORE adding it.

        This enables pre-flight checks for hard constraints.

        Args:
            edge: Edge to validate

        Returns:
            List of violation messages (empty = valid)
        """
        violations = []

        # Both nodes must exist
        if edge.source_id not in self._node_map:
            violations.append(f"Source node not found: {edge.source_id}")
            return violations
        if edge.target_id not in self._node_map:
            violations.append(f"Target node not found: {edge.target_id}")
            return violations

        source_node = self.get_node(edge.source_id)
        target_node = self.get_node(edge.target_id)

        # Check hard constraints on source (outgoing edge)
        source_constraints = get_required_edges(source_node.type, mode="hard")
        for constraint in source_constraints:
            if constraint.direction in ("outgoing", "any"):
                if constraint.edge_type == edge.type:
                    if constraint.target_node_type:
                        if target_node.type != constraint.target_node_type:
                            violations.append(
                                f"Edge type '{edge.type}' from {source_node.type} "
                                f"must target {constraint.target_node_type}, "
                                f"not {target_node.type}"
                            )

        # Check hard constraints on target (incoming edge)
        target_constraints = get_required_edges(target_node.type, mode="hard")
        for constraint in target_constraints:
            if constraint.direction in ("incoming", "any"):
                if constraint.edge_type == edge.type:
                    if constraint.target_node_type:
                        if source_node.type != constraint.target_node_type:
                            violations.append(
                                f"Edge type '{edge.type}' to {target_node.type} "
                                f"must come from {constraint.target_node_type}, "
                                f"not {source_node.type}"
                            )

        return violations

    # =========================================================================
    # STRUCTURAL TRIGGER MATCHING (Graph-Native Agent Dispatch)
    # =========================================================================

    def match_triggers(
        self,
        node_id: str,
    ) -> List[StructuralTrigger]:
        """
        Find all structural triggers that match a node's current state.

        This replaces the old "condition evaluation" pattern with pure
        graph topology queries.

        Args:
            node_id: Node to check

        Returns:
            List of matching StructuralTrigger objects

        Example:
            triggers = db.match_triggers("code_node_123")
            for trigger in triggers:
                dispatch_agent(trigger.agent_role, node_id)
        """
        if node_id not in self._node_map:
            return []

        node = self.get_node(node_id)
        matching = []

        # Get triggers for this node type
        for trigger in STRUCTURAL_TRIGGERS.values():
            if trigger.target_node_type != node.type:
                continue

            if self._matches_trigger(node_id, node, trigger):
                matching.append(trigger)

        return matching

    def _matches_trigger(
        self,
        node_id: str,
        node: NodeData,
        trigger: StructuralTrigger,
    ) -> bool:
        """Check if a node matches a structural trigger."""
        idx = self._node_map[node_id]

        # Check status patterns
        for sp in trigger.status_patterns:
            if sp.is_not:
                if node.status == sp.status:
                    return False
            else:
                if node.status != sp.status:
                    return False

        # Check required edges
        for ep in trigger.required_edges:
            if not self._has_edge_pattern(idx, node, ep):
                return False

        # Check forbidden edges
        for ep in trigger.forbidden_edges:
            if self._has_edge_pattern(idx, node, ep):
                return False

        # Check predecessor status requirements
        if trigger.all_predecessors_status:
            predecessors = self.get_predecessors(node_id)
            if predecessors:  # Only check if there are predecessors
                for pred in predecessors:
                    if pred.status != trigger.all_predecessors_status:
                        return False

        if trigger.any_predecessor_status:
            predecessors = self.get_predecessors(node_id)
            if predecessors:
                has_match = any(
                    pred.status == trigger.any_predecessor_status
                    for pred in predecessors
                )
                if not has_match:
                    return False

        return True

    def _has_edge_pattern(
        self,
        node_idx: int,
        node: NodeData,
        pattern: EdgePattern,
    ) -> bool:
        """Check if a node has an edge matching the pattern."""
        if pattern.direction in ("incoming", "any"):
            for pred_idx in self._graph.predecessor_indices(node_idx):
                edge_data = self._graph.get_edge_data(pred_idx, node_idx)
                if edge_data and edge_data.type == pattern.edge_type:
                    pred_node = self._graph[pred_idx]
                    if pattern.target_node_type and pred_node.type != pattern.target_node_type:
                        continue
                    if pattern.target_node_status and pred_node.status != pattern.target_node_status:
                        continue
                    return True

        if pattern.direction in ("outgoing", "any"):
            for succ_idx in self._graph.successor_indices(node_idx):
                edge_data = self._graph.get_edge_data(node_idx, succ_idx)
                if edge_data and edge_data.type == pattern.edge_type:
                    succ_node = self._graph[succ_idx]
                    if pattern.target_node_type and succ_node.type != pattern.target_node_type:
                        continue
                    if pattern.target_node_status and succ_node.status != pattern.target_node_status:
                        continue
                    return True

        return False

    def find_triggerable_nodes(self) -> Dict[str, List[StructuralTrigger]]:
        """
        Find all nodes that have matching triggers.

        Returns:
            Dict mapping node_id -> list of matching triggers
        """
        result = {}
        for node in self._graph.nodes():
            triggers = self.match_triggers(node.id)
            if triggers:
                result[node.id] = triggers
        return result

    def is_blocked_by_dependencies(self, node_id: str) -> bool:
        """
        Check if a node is blocked by unverified predecessors.

        This is the Graph-Native replacement for explicit BLOCKED status.
        A node is structurally blocked if any predecessor is not VERIFIED.

        Args:
            node_id: Node to check

        Returns:
            True if any predecessor is not VERIFIED
        """
        if node_id not in self._node_map:
            return False

        predecessors = self.get_predecessors(node_id)
        for pred in predecessors:
            if pred.status != NodeStatus.VERIFIED.value:
                return True
        return False

    def is_blocked(self, node_id: str) -> bool:
        """
        Check if a node is blocked (explicit OR structural).

        Combines:
        1. Explicit BLOCKED status (human override)
        2. Structural blocking (unverified predecessors)

        Args:
            node_id: Node to check

        Returns:
            True if node is blocked for any reason
        """
        if node_id not in self._node_map:
            return False

        node = self.get_node(node_id)

        # Explicit block takes precedence
        if node.status == NodeStatus.BLOCKED.value:
            return True

        # Check structural blocking
        return self.is_blocked_by_dependencies(node_id)

    def get_ready_nodes(self) -> List[NodeData]:
        """
        Find all nodes that are ready to be processed.

        A node is ready if:
        1. Status is PENDING
        2. Not blocked (explicit or structural)
        3. Has at least one matching trigger

        Returns:
            List of ready nodes
        """
        ready = []
        for node in self._graph.nodes():
            if node.status != NodeStatus.PENDING.value:
                continue
            if self.is_blocked(node.id):
                continue
            triggers = self.match_triggers(node.id)
            if triggers:
                ready.append(node)
        return ready

    # =========================================================================
    # DIALOGUE-TO-GRAPH HIGHLIGHTING (Interactive Visualization)
    # =========================================================================

    def get_related_nodes(
        self,
        node_id: str,
        mode: Literal["exact", "related", "dependent"] = "exact"
    ) -> List[str]:
        """
        Get nodes related to a given node for highlighting purposes.

        This method supports click-to-highlight features in the UI where
        clicking a message/node highlights related nodes in the graph.

        Args:
            node_id: The starting node ID
            mode: Highlighting mode:
                - "exact": Just the node itself
                - "related": Node + immediate dependencies (DEPENDS_ON, IMPLEMENTS edges)
                - "dependent": Node + all nodes that depend on it (reverse dependencies)

        Returns:
            List of node IDs to highlight

        Raises:
            NodeNotFoundError: If node doesn't exist

        Example:
            # Highlight a node and its dependencies
            related = db.get_related_nodes("spec_123", mode="related")
            for node_id in related:
                highlight_in_ui(node_id)
        """
        if node_id not in self._node_map:
            raise NodeNotFoundError(node_id)

        if mode == "exact":
            return [node_id]

        result_ids: Set[str] = {node_id}
        idx = self._node_map[node_id]

        if mode == "related":
            # Get immediate dependencies via DEPENDS_ON and IMPLEMENTS edges
            for pred_idx in self._graph.predecessor_indices(idx):
                edge_data = self._graph.get_edge_data(pred_idx, idx)
                if edge_data and edge_data.type in (EdgeType.DEPENDS_ON, EdgeType.IMPLEMENTS):
                    pred_id = self._inv_map[pred_idx]
                    result_ids.add(pred_id)

            # Also include direct successors via same edge types
            for succ_idx in self._graph.successor_indices(idx):
                edge_data = self._graph.get_edge_data(idx, succ_idx)
                if edge_data and edge_data.type in (EdgeType.DEPENDS_ON, EdgeType.IMPLEMENTS):
                    succ_id = self._inv_map[succ_idx]
                    result_ids.add(succ_id)

        elif mode == "dependent":
            # Get all nodes that depend on this node (reverse dependencies)
            # This includes all descendants in the dependency graph
            desc_indices = rx.descendants(self._graph, idx)
            for desc_idx in desc_indices:
                result_ids.add(self._inv_map[desc_idx])

            # Also include immediate dependents
            for succ_idx in self._graph.successor_indices(idx):
                result_ids.add(self._inv_map[succ_idx])

        return sorted(result_ids)

    def get_reverse_connections(self, node_id: str) -> Dict[str, Any]:
        """
        Find all MESSAGE/DIALOGUE nodes and edges that reference this node.

        This enables hover-to-show-references features where hovering a node
        displays which messages reference it and what edges point to it.

        Args:
            node_id: The node to find connections for

        Returns:
            Dict with:
                - referenced_in_dialogue: List of DIALOGUE/MESSAGE node IDs
                - referenced_in_messages: List of MESSAGE nodes that reference this node
                - incoming_edges: List of edge info dicts with source, target, type
                - outgoing_edges: List of edge info dicts
                - definition_location: Optional file/location where node was defined
                - last_modified_by: Who last modified this node
                - last_modified_at: When it was last modified

        Raises:
            NodeNotFoundError: If node doesn't exist

        Example:
            connections = db.get_reverse_connections("code_abc")
            for msg_id in connections["referenced_in_messages"]:
                show_message_reference(msg_id)
        """
        if node_id not in self._node_map:
            raise NodeNotFoundError(node_id)

        node = self.get_node(node_id)
        idx = self._node_map[node_id]

        # Find MESSAGE and THREAD nodes that reference this node via REFERENCES edges
        referenced_in_dialogue = []
        referenced_in_messages = []

        for other_idx in self._graph.node_indices():
            other_node = self._graph[other_idx]

            # Check if this is a MESSAGE or THREAD node
            if other_node.type in (NodeType.MESSAGE.value, NodeType.THREAD.value):
                # Check if there's a REFERENCES edge from the message to our node
                if self._graph.has_edge(other_idx, idx):
                    edge_data = self._graph.get_edge_data(other_idx, idx)
                    if edge_data and edge_data.type == EdgeType.REFERENCES:
                        other_id = self._inv_map[other_idx]
                        referenced_in_dialogue.append(other_id)
                        if other_node.type == NodeType.MESSAGE.value:
                            referenced_in_messages.append(other_id)

        # Get all incoming edges
        incoming_edges = []
        for pred_idx in self._graph.predecessor_indices(idx):
            edge_data = self._graph.get_edge_data(pred_idx, idx)
            if edge_data:
                pred_id = self._inv_map[pred_idx]
                pred_node = self._graph[pred_idx]
                incoming_edges.append({
                    "source": pred_id,
                    "target": node_id,
                    "type": edge_data.type.value if hasattr(edge_data.type, 'value') else edge_data.type,
                    "weight": edge_data.weight,
                    "source_node_type": pred_node.type,
                })

        # Get all outgoing edges
        outgoing_edges = []
        for succ_idx in self._graph.successor_indices(idx):
            edge_data = self._graph.get_edge_data(idx, succ_idx)
            if edge_data:
                succ_id = self._inv_map[succ_idx]
                succ_node = self._graph[succ_idx]
                outgoing_edges.append({
                    "source": node_id,
                    "target": succ_id,
                    "type": edge_data.type.value if hasattr(edge_data.type, 'value') else edge_data.type,
                    "weight": edge_data.weight,
                    "target_node_type": succ_node.type,
                })

        # Extract metadata from node
        definition_location = None
        if hasattr(node, 'metadata') and node.metadata:
            extra = node.metadata.extra if hasattr(node.metadata, 'extra') else {}
            definition_location = extra.get('file_path') or extra.get('location')

        last_modified_by = node.created_by if hasattr(node, 'created_by') else "unknown"
        last_modified_at = node.created_at if hasattr(node, 'created_at') else ""

        return {
            "node_id": node_id,
            "referenced_in_dialogue": referenced_in_dialogue,
            "referenced_in_messages": referenced_in_messages,
            "incoming_edges": incoming_edges,
            "outgoing_edges": outgoing_edges,
            "definition_location": definition_location,
            "last_modified_by": last_modified_by,
            "last_modified_at": last_modified_at,
        }

    def get_nodes_for_message(self, message_id: str) -> List[str]:
        """
        Get all nodes referenced by a MESSAGE or DIALOGUE_TURN node.

        This uses REFERENCES edges created by the message-to-node mapping system.

        Args:
            message_id: The MESSAGE or THREAD node ID

        Returns:
            List of node IDs referenced by this message

        Raises:
            NodeNotFoundError: If message node doesn't exist

        Example:
            # User clicks a message in the dialogue panel
            nodes = db.get_nodes_for_message("msg_123")
            highlight_nodes_in_graph(nodes)
        """
        if message_id not in self._node_map:
            raise NodeNotFoundError(message_id)

        message_node = self.get_node(message_id)

        # Verify this is actually a MESSAGE or THREAD node
        if message_node.type not in (NodeType.MESSAGE.value, NodeType.THREAD.value):
            # Not a message node, return empty list
            return []

        idx = self._node_map[message_id]
        referenced_nodes = []

        # Find all outgoing REFERENCES edges from this message
        for succ_idx in self._graph.successor_indices(idx):
            edge_data = self._graph.get_edge_data(idx, succ_idx)
            if edge_data and edge_data.type == EdgeType.REFERENCES:
                succ_id = self._inv_map[succ_idx]
                referenced_nodes.append(succ_id)

        return referenced_nodes

    # =========================================================================
    # PERSISTENCE (Polars-Compatible)
    # =========================================================================

    def to_polars_nodes(self) -> pl.DataFrame:
        """
        Export nodes to a Polars DataFrame.

        Useful for analytics, persistence, and Cosmograph visualization.
        """
        if self.is_empty:
            return pl.DataFrame({
                "id": [],
                "type": [],
                "status": [],
                "content": [],
                "created_by": [],
                "created_at": [],
                "version": [],
            })

        nodes = list(self._graph.nodes())
        return pl.DataFrame({
            "id": [n.id for n in nodes],
            "type": [n.type for n in nodes],
            "status": [n.status for n in nodes],
            "content": [n.content for n in nodes],
            "created_by": [n.created_by for n in nodes],
            "created_at": [n.created_at for n in nodes],
            "version": [n.version for n in nodes],
        })

    def to_polars_edges(self) -> pl.DataFrame:
        """Export edges to a Polars DataFrame."""
        if self.edge_count == 0:
            return pl.DataFrame({
                "source_id": [],
                "target_id": [],
                "type": [],
                "weight": [],
            })

        edges = self.get_all_edges()
        return pl.DataFrame({
            "source_id": [e.source_id for e in edges],
            "target_id": [e.target_id for e in edges],
            "type": [e.type for e in edges],
            "weight": [e.weight for e in edges],
        })

    def save_parquet(self, path: Path) -> None:
        """
        Save graph state to parquet files.

        Creates two files:
        - {path}_nodes.parquet
        - {path}_edges.parquet

        Args:
            path: Base path (without extension)
        """
        path = Path(path)
        nodes_path = path.with_suffix(".nodes.parquet")
        edges_path = path.with_suffix(".edges.parquet")

        self.to_polars_nodes().write_parquet(nodes_path)
        self.to_polars_edges().write_parquet(edges_path)

    def save_arrow(self, path: Path) -> None:
        """
        Save graph state to Arrow IPC files.

        Faster than parquet for inter-process communication.
        """
        path = Path(path)
        nodes_path = path.with_suffix(".nodes.arrow")
        edges_path = path.with_suffix(".edges.arrow")

        self.to_polars_nodes().write_ipc(nodes_path)
        self.to_polars_edges().write_ipc(edges_path)

    # =========================================================================
    # LEGACY SUPPORT
    # =========================================================================

    def export_networkx(self):
        """
        Export to NetworkX graph for legacy compatibility.

        WARNING: Only use for debugging or legacy visualization.
        Never use NetworkX for compute - it's 10-100x slower than rustworkx.

        Returns:
            networkx.DiGraph
        """
        return rx.networkx_converter(self._graph)

    # =========================================================================
    # INTERNAL UTILITIES
    # =========================================================================

    def _get_index(self, node_id: str) -> int:
        """Internal: get rustworkx index for a node ID."""
        if node_id not in self._node_map:
            raise NodeNotFoundError(node_id)
        return self._node_map[node_id]

    def _get_id(self, idx: int) -> str:
        """Internal: get node ID for a rustworkx index."""
        if idx not in self._inv_map:
            raise GraphError(f"Invalid index: {idx}")
        return self._inv_map[idx]

    def __len__(self) -> int:
        """Return number of nodes."""
        return self.node_count

    def __contains__(self, node_id: str) -> bool:
        """Check if node exists."""
        return node_id in self._node_map

    def __repr__(self) -> str:
        return f"ParagonDB(nodes={self.node_count}, edges={self.edge_count})"


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_empty_db() -> ParagonDB:
    """Create an empty ParagonDB instance."""
    return ParagonDB()


def create_db_from_nodes(nodes: List[NodeData], edges: Optional[List[EdgeData]] = None) -> ParagonDB:
    """
    Create a ParagonDB pre-populated with nodes and edges.

    Args:
        nodes: Initial nodes to add
        edges: Optional edges to add after nodes

    Returns:
        Populated ParagonDB
    """
    db = ParagonDB()
    db.add_nodes_batch(nodes)
    if edges:
        db.add_edges_batch(edges)
    return db
