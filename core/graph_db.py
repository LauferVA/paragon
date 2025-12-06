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

    def add_edge(self, edge: EdgeData) -> int:
        """
        Add an edge between two nodes.

        Args:
            edge: EdgeData with source_id, target_id, and type

        Returns:
            The rustworkx edge index

        Raises:
            NodeNotFoundError: If source or target node doesn't exist
        """
        source_id = edge.source_id
        target_id = edge.target_id

        if source_id not in self._node_map:
            raise NodeNotFoundError(source_id)
        if target_id not in self._node_map:
            raise NodeNotFoundError(target_id)

        src_idx = self._node_map[source_id]
        tgt_idx = self._node_map[target_id]

        # Add to graph
        edge_idx = self._graph.add_edge(src_idx, tgt_idx, edge)

        # Track in edge map
        self._edge_map[(source_id, target_id)] = edge_idx

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
