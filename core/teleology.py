"""
PARAGON TELEOLOGY - The Golden Thread

Every node must justify its existence by tracing back to a requirement.
Code that exists "for its own sake" is hallucinated scope.

The teleological check answers: "Why does this node exist?"
If the answer isn't "because a requirement demanded it", the node is suspect.

Algorithm:
1. Identify all REQ nodes (the "why" roots)
2. Reverse BFS from REQ nodes following TRACES_TO/IMPLEMENTS/DEPENDS_ON edges
3. Any node NOT visited is "Unjustified" - it has no teleological chain

This is the "Golden Thread" that connects every artifact back to intent.

Layer 8: The Philosophy Engine
"""
import rustworkx as rx
from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core.schemas import NodeData
from core.ontology import NodeType, EdgeType


# =============================================================================
# TELEOLOGY STATUS
# =============================================================================

class TeleologyStatus(Enum):
    """The teleological status of a node."""
    JUSTIFIED = "justified"        # Has clear lineage to REQ
    UNJUSTIFIED = "unjustified"    # No lineage - hallucinated scope
    ROOT = "root"                  # Is itself a REQ (the source of justification)
    ORPHANED = "orphaned"          # Disconnected from main graph entirely
    UNCHECKED = "unchecked"        # Not yet evaluated


@dataclass
class TeleologyResult:
    """Result of teleology validation for a single node."""
    node_id: str
    status: TeleologyStatus
    lineage_depth: int = 0         # Distance from nearest REQ
    nearest_req: Optional[str] = None  # ID of the REQ this traces to
    path_to_req: List[str] = None  # Node IDs in path to REQ

    def __post_init__(self):
        if self.path_to_req is None:
            self.path_to_req = []


@dataclass
class TeleologyReport:
    """Complete teleology validation report."""
    total_nodes: int
    justified_count: int
    unjustified_count: int
    root_count: int
    orphaned_count: int
    unjustified_nodes: List[str]   # Node IDs that are unjustified
    node_results: Dict[str, TeleologyResult]

    @property
    def is_valid(self) -> bool:
        """True if no unjustified or orphaned nodes exist."""
        return self.unjustified_count == 0 and self.orphaned_count == 0

    @property
    def justification_rate(self) -> float:
        """Percentage of non-root nodes that are justified."""
        non_root = self.total_nodes - self.root_count
        if non_root == 0:
            return 1.0
        return self.justified_count / non_root


# =============================================================================
# TELEOLOGY VALIDATOR
# =============================================================================

class TeleologyValidator:
    """
    Validates that all nodes have teleological justification.

    Uses reverse BFS from REQ nodes to find all nodes that have
    a clear chain of causation back to requirements.

    Performance: O(V + E) using rustworkx traversal.
    """

    def __init__(
        self,
        graph: rx.PyDiGraph,
        node_map: Dict[str, int],
        inv_map: Dict[int, str],
    ):
        """
        Initialize the validator.

        Args:
            graph: The rustworkx PyDiGraph
            node_map: UUID -> rustworkx index mapping
            inv_map: rustworkx index -> UUID mapping
        """
        self.graph = graph
        self.node_map = node_map
        self.inv_map = inv_map

    def validate(self) -> TeleologyReport:
        """
        Run full teleology validation.

        Returns:
            TeleologyReport with all results
        """
        if self.graph.num_nodes() == 0:
            return TeleologyReport(
                total_nodes=0,
                justified_count=0,
                unjustified_count=0,
                root_count=0,
                orphaned_count=0,
                unjustified_nodes=[],
                node_results={}
            )

        # Step 1: Find all REQ nodes (the roots of justification)
        req_indices = self._find_req_nodes()

        # Step 2: Run reverse BFS from all REQ nodes
        justified_indices, depth_map, nearest_req_map = self._reverse_bfs_from_roots(req_indices)

        # Step 3: Classify all nodes
        results = {}
        justified_count = 0
        unjustified_count = 0
        root_count = len(req_indices)
        orphaned_count = 0
        unjustified_nodes = []

        for idx in self.graph.node_indices():
            node_id = self.inv_map.get(idx, str(idx))
            node = self.graph[idx]

            if idx in req_indices:
                # This is a root (REQ node)
                results[node_id] = TeleologyResult(
                    node_id=node_id,
                    status=TeleologyStatus.ROOT,
                    lineage_depth=0,
                    nearest_req=node_id
                )
            elif idx in justified_indices:
                # Has lineage to a REQ
                justified_count += 1
                depth = depth_map.get(idx, 0)
                nearest_req_idx = nearest_req_map.get(idx)
                nearest_req_id = self.inv_map.get(nearest_req_idx, None) if nearest_req_idx else None

                results[node_id] = TeleologyResult(
                    node_id=node_id,
                    status=TeleologyStatus.JUSTIFIED,
                    lineage_depth=depth,
                    nearest_req=nearest_req_id,
                    path_to_req=self._find_path_to_req(idx, req_indices)
                )
            elif self._is_orphaned(idx):
                # Completely disconnected
                orphaned_count += 1
                results[node_id] = TeleologyResult(
                    node_id=node_id,
                    status=TeleologyStatus.ORPHANED
                )
                unjustified_nodes.append(node_id)
            else:
                # Connected but no path to REQ - hallucinated scope
                unjustified_count += 1
                results[node_id] = TeleologyResult(
                    node_id=node_id,
                    status=TeleologyStatus.UNJUSTIFIED
                )
                unjustified_nodes.append(node_id)

        return TeleologyReport(
            total_nodes=self.graph.num_nodes(),
            justified_count=justified_count,
            unjustified_count=unjustified_count,
            root_count=root_count,
            orphaned_count=orphaned_count,
            unjustified_nodes=unjustified_nodes,
            node_results=results
        )

    def check_node(self, node_id: str) -> TeleologyResult:
        """
        Check teleology for a single node.

        Args:
            node_id: The node UUID to check

        Returns:
            TeleologyResult for this node
        """
        if node_id not in self.node_map:
            return TeleologyResult(
                node_id=node_id,
                status=TeleologyStatus.UNCHECKED
            )

        idx = self.node_map[node_id]
        node = self.graph[idx]

        # Is this a REQ node?
        if hasattr(node, 'type') and node.type == NodeType.REQ.value:
            return TeleologyResult(
                node_id=node_id,
                status=TeleologyStatus.ROOT,
                lineage_depth=0,
                nearest_req=node_id
            )

        # Find path to any REQ node
        req_indices = self._find_req_nodes()
        path = self._find_path_to_req(idx, req_indices)

        if path:
            return TeleologyResult(
                node_id=node_id,
                status=TeleologyStatus.JUSTIFIED,
                lineage_depth=len(path),
                nearest_req=path[-1] if path else None,
                path_to_req=path
            )

        # Check if orphaned
        if self._is_orphaned(idx):
            return TeleologyResult(
                node_id=node_id,
                status=TeleologyStatus.ORPHANED
            )

        return TeleologyResult(
            node_id=node_id,
            status=TeleologyStatus.UNJUSTIFIED
        )

    def _find_req_nodes(self) -> Set[int]:
        """Find all REQ node indices."""
        req_indices = set()
        for idx in self.graph.node_indices():
            node = self.graph[idx]
            if hasattr(node, 'type') and node.type == NodeType.REQ.value:
                req_indices.add(idx)
        return req_indices

    def _reverse_bfs_from_roots(
        self,
        root_indices: Set[int]
    ) -> Tuple[Set[int], Dict[int, int], Dict[int, int]]:
        """
        BFS from REQ nodes to find all justified nodes.

        In Paragon's edge model:
        - CODE --IMPLEMENTS--> SPEC (code implements spec)
        - CODE --DEPENDS_ON--> CODE (code depends on code)
        - SPEC --TRACES_TO--> REQ (spec traces to req)

        From REQ, we traverse:
        - Predecessors: nodes that point TO REQ (like SPEC --TRACES_TO--> REQ)
        - Successors: nodes that REQ points to

        A node is justified if it has ANY path to/from a REQ node.

        Returns:
            (justified_indices, depth_map, nearest_req_map)
        """
        justified = set()
        depth_map = {}
        nearest_req_map = {}

        # BFS queue: (node_idx, depth, source_req_idx)
        queue = []
        for req_idx in root_indices:
            queue.append((req_idx, 0, req_idx))
            depth_map[req_idx] = 0
            nearest_req_map[req_idx] = req_idx

        visited = set(root_indices)

        while queue:
            current_idx, depth, source_req = queue.pop(0)

            # Traverse to predecessors (nodes that point TO this one)
            for pred_idx in self.graph.predecessor_indices(current_idx):
                if pred_idx not in visited:
                    visited.add(pred_idx)
                    justified.add(pred_idx)
                    depth_map[pred_idx] = depth + 1
                    nearest_req_map[pred_idx] = source_req
                    queue.append((pred_idx, depth + 1, source_req))

            # Traverse to successors (nodes this points TO)
            for succ_idx in self.graph.successor_indices(current_idx):
                if succ_idx not in visited:
                    visited.add(succ_idx)
                    justified.add(succ_idx)
                    depth_map[succ_idx] = depth + 1
                    nearest_req_map[succ_idx] = source_req
                    queue.append((succ_idx, depth + 1, source_req))

        return justified, depth_map, nearest_req_map

    def _find_path_to_req(self, start_idx: int, req_indices: Set[int]) -> List[str]:
        """Find the path from a node to any REQ node."""
        if start_idx in req_indices:
            return [self.inv_map.get(start_idx, str(start_idx))]

        # BFS to find shortest path to any REQ
        visited = {start_idx}
        queue = [(start_idx, [self.inv_map.get(start_idx, str(start_idx))])]

        while queue:
            current, path = queue.pop(0)

            # Check successors
            for succ_idx in self.graph.successor_indices(current):
                if succ_idx in req_indices:
                    return path + [self.inv_map.get(succ_idx, str(succ_idx))]
                if succ_idx not in visited:
                    visited.add(succ_idx)
                    queue.append((succ_idx, path + [self.inv_map.get(succ_idx, str(succ_idx))]))

            # Check predecessors
            for pred_idx in self.graph.predecessor_indices(current):
                if pred_idx in req_indices:
                    return path + [self.inv_map.get(pred_idx, str(pred_idx))]
                if pred_idx not in visited:
                    visited.add(pred_idx)
                    queue.append((pred_idx, path + [self.inv_map.get(pred_idx, str(pred_idx))]))

        return []  # No path found

    def _is_orphaned(self, idx: int) -> bool:
        """Check if a node is completely disconnected."""
        return (
            self.graph.in_degree(idx) == 0 and
            self.graph.out_degree(idx) == 0
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_teleology(
    graph: rx.PyDiGraph,
    node_map: Dict[str, int],
    inv_map: Dict[int, str]
) -> TeleologyReport:
    """
    Convenience function to validate teleology.

    Args:
        graph: The rustworkx PyDiGraph
        node_map: UUID -> index mapping
        inv_map: index -> UUID mapping

    Returns:
        TeleologyReport
    """
    validator = TeleologyValidator(graph, node_map, inv_map)
    return validator.validate()


def find_unjustified_nodes(
    graph: rx.PyDiGraph,
    node_map: Dict[str, int],
    inv_map: Dict[int, str]
) -> List[str]:
    """
    Quick check to find unjustified nodes.

    Returns:
        List of node IDs that are unjustified
    """
    report = validate_teleology(graph, node_map, inv_map)
    return report.unjustified_nodes


def has_teleological_integrity(
    graph: rx.PyDiGraph,
    node_map: Dict[str, int],
    inv_map: Dict[int, str]
) -> bool:
    """
    Quick boolean check for teleological integrity.

    Returns:
        True if all nodes are justified
    """
    report = validate_teleology(graph, node_map, inv_map)
    return report.is_valid
