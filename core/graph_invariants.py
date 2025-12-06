"""
PARAGON GRAPH INVARIANTS - The Mathematical Superego

This module enforces the physics of the graph. If the topology is invalid,
the code is rejected BEFORE it touches the database.

Invariants Implemented:
1. Handshaking Lemma: sum(in_degree) == sum(out_degree) == |E|
2. Balis Degree: Every source must reach every sink (no orphaned subgraphs)
3. DAG Acyclicity: No cycles allowed in execution graphs
4. Stratification: Topological layers must respect type ordering

Design Philosophy:
- These are MATHEMATICAL constraints, not business rules
- Violations are errors, not warnings
- Checks are O(V+E) using rustworkx primitives

Layer 8: The Physics Engine
"""
import rustworkx as rx
from typing import List, Set, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from core.schemas import NodeData


# =============================================================================
# INVARIANT RESULTS
# =============================================================================

class InvariantSeverity(Enum):
    """Severity levels for invariant violations."""
    ERROR = "error"      # Must be fixed before proceeding
    WARNING = "warning"  # Should be investigated
    INFO = "info"        # For metrics/diagnostics


@dataclass
class InvariantViolation:
    """A specific invariant violation."""
    invariant: str           # Name of the invariant
    severity: InvariantSeverity
    message: str
    nodes_involved: List[str] = None  # Node IDs involved
    edges_involved: List[Tuple[str, str]] = None  # (source, target) pairs

    def __post_init__(self):
        if self.nodes_involved is None:
            self.nodes_involved = []
        if self.edges_involved is None:
            self.edges_involved = []


@dataclass
class InvariantReport:
    """Complete invariant validation report."""
    valid: bool
    violations: List[InvariantViolation]
    metrics: Dict[str, Any]

    @property
    def errors(self) -> List[InvariantViolation]:
        return [v for v in self.violations if v.severity == InvariantSeverity.ERROR]

    @property
    def warnings(self) -> List[InvariantViolation]:
        return [v for v in self.violations if v.severity == InvariantSeverity.WARNING]


# =============================================================================
# GRAPH INVARIANTS (Rustworkx-Native)
# =============================================================================

class GraphInvariants:
    """
    Graph-theoretic invariant validators using rustworkx.

    All methods are static and work directly on rx.PyDiGraph instances.
    The ParagonDB wraps these for business-friendly access.

    Performance: All operations are O(V+E) or better.
    """

    @staticmethod
    def validate_handshaking_lemma(graph: rx.PyDiGraph) -> Tuple[bool, Optional[InvariantViolation]]:
        """
        Handshaking Lemma: sum(in_degree) == sum(out_degree) == |E|

        For directed graphs, total in-degree must equal total out-degree,
        and both must equal the number of edges.

        This is a fundamental graph theory invariant that catches
        corrupted edge state.

        Returns:
            (is_valid, violation or None)
        """
        num_edges = graph.num_edges()
        node_indices = list(graph.node_indices())

        total_in = sum(graph.in_degree(idx) for idx in node_indices)
        total_out = sum(graph.out_degree(idx) for idx in node_indices)

        if total_in != total_out:
            return False, InvariantViolation(
                invariant="handshaking_lemma",
                severity=InvariantSeverity.ERROR,
                message=f"sum(in_degree)={total_in} != sum(out_degree)={total_out}"
            )

        if total_in != num_edges:
            return False, InvariantViolation(
                invariant="handshaking_lemma",
                severity=InvariantSeverity.ERROR,
                message=f"sum(degrees)={total_in} != |E|={num_edges}"
            )

        return True, None

    @staticmethod
    def validate_dag_acyclicity(graph: rx.PyDiGraph) -> Tuple[bool, Optional[InvariantViolation]]:
        """
        DAG Topological Invariant: The graph must be acyclic.

        Uses rustworkx's is_directed_acyclic_graph for O(V+E) check.

        Returns:
            (is_valid, violation or None)
        """
        if rx.is_directed_acyclic_graph(graph):
            return True, None

        # Graph has cycles - find one for the error message
        # We'll use topological_sort which raises DAGHasCycle
        try:
            rx.topological_sort(graph)
            return True, None  # Shouldn't reach here
        except rx.DAGHasCycle:
            # Find a cycle using DFS-based cycle detection
            cycle_nodes = GraphInvariants._find_cycle_nodes(graph)
            return False, InvariantViolation(
                invariant="dag_acyclicity",
                severity=InvariantSeverity.ERROR,
                message=f"Cycle detected involving {len(cycle_nodes)} nodes",
                nodes_involved=cycle_nodes
            )

    @staticmethod
    def _find_cycle_nodes(graph: rx.PyDiGraph) -> List[str]:
        """Find nodes involved in a cycle (for error reporting)."""
        # Simple DFS-based cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {idx: WHITE for idx in graph.node_indices()}
        cycle_nodes = []

        def dfs(node):
            color[node] = GRAY
            for succ in graph.successor_indices(node):
                if color[succ] == GRAY:
                    # Found cycle
                    node_data = graph[node]
                    if hasattr(node_data, 'id'):
                        cycle_nodes.append(node_data.id)
                    return True
                if color[succ] == WHITE:
                    if dfs(succ):
                        node_data = graph[node]
                        if hasattr(node_data, 'id'):
                            cycle_nodes.append(node_data.id)
                        return True
            color[node] = BLACK
            return False

        for idx in graph.node_indices():
            if color[idx] == WHITE:
                if dfs(idx):
                    break

        return cycle_nodes[:10]  # Limit to 10 nodes for readability

    @staticmethod
    def validate_balis_degree(
        graph: rx.PyDiGraph,
        inv_map: Optional[Dict[int, str]] = None
    ) -> Tuple[bool, Optional[InvariantViolation], List[Tuple[str, str]]]:
        """
        Balis Degree Invariant: Every source must reach every sink.

        This ensures no orphaned subgraphs exist - all code traces back
        to requirements and forward to deliverables.

        For Paragon: Sources are REQ nodes, Sinks are CODE/TEST nodes.

        Args:
            graph: The rustworkx PyDiGraph
            inv_map: Optional index->ID mapping for error messages

        Returns:
            (is_valid, violation or None, unreachable_pairs)
        """
        node_indices = list(graph.node_indices())

        if not node_indices:
            return True, None, []

        # Find sources (in_degree == 0) and sinks (out_degree == 0)
        sources = [idx for idx in node_indices if graph.in_degree(idx) == 0]
        sinks = [idx for idx in node_indices if graph.out_degree(idx) == 0]

        if not sources or not sinks:
            return True, None, []

        # Check reachability from each source to each sink
        unreachable = []
        for source_idx in sources:
            # Get all descendants of this source
            reachable = set(rx.descendants(graph, source_idx))
            reachable.add(source_idx)

            for sink_idx in sinks:
                if sink_idx not in reachable and source_idx != sink_idx:
                    # Map back to IDs if possible
                    source_id = inv_map.get(source_idx, str(source_idx)) if inv_map else str(source_idx)
                    sink_id = inv_map.get(sink_idx, str(sink_idx)) if inv_map else str(sink_idx)
                    unreachable.append((source_id, sink_id))

        if unreachable:
            return False, InvariantViolation(
                invariant="balis_degree",
                severity=InvariantSeverity.WARNING,  # Warning, not error - may be intentional
                message=f"{len(unreachable)} source-sink pair(s) unreachable",
                edges_involved=unreachable[:10]  # Limit for readability
            ), unreachable

        return True, None, []

    @staticmethod
    def validate_stratification(
        graph: rx.PyDiGraph,
        type_order: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[InvariantViolation]]:
        """
        Stratification Invariant: Topological layers must respect type ordering.

        For Paragon, the expected order is:
        REQ -> SPEC -> CODE -> TEST -> (verified)

        Uses rx.topological_generations for O(V+E) layer computation.

        Args:
            graph: The rustworkx PyDiGraph
            type_order: Expected type ordering (first = earliest layer)

        Returns:
            (is_valid, violation or None)
        """
        if type_order is None:
            # Default Paragon type ordering
            type_order = ["REQ", "RESEARCH", "PLAN", "SPEC", "CODE", "TEST"]

        if graph.num_nodes() == 0:
            return True, None

        type_to_rank = {t: i for i, t in enumerate(type_order)}

        try:
            # Get topological generations (layers)
            generations = rx.topological_generations(graph)

            violations = []
            for gen_idx, generation in enumerate(generations):
                for node_idx in generation:
                    node = graph[node_idx]
                    if not hasattr(node, 'type'):
                        continue

                    node_type = node.type
                    expected_rank = type_to_rank.get(node_type, -1)

                    if expected_rank == -1:
                        continue  # Unknown type, skip

                    # Check predecessors - they should have lower or equal rank
                    for pred_idx in graph.predecessor_indices(node_idx):
                        pred_node = graph[pred_idx]
                        if not hasattr(pred_node, 'type'):
                            continue
                        pred_type = pred_node.type
                        pred_rank = type_to_rank.get(pred_type, -1)

                        if pred_rank > expected_rank:
                            node_id = node.id if hasattr(node, 'id') else str(node_idx)
                            pred_id = pred_node.id if hasattr(pred_node, 'id') else str(pred_idx)
                            violations.append(f"{pred_type}({pred_id}) -> {node_type}({node_id})")

            if violations:
                return False, InvariantViolation(
                    invariant="stratification",
                    severity=InvariantSeverity.WARNING,
                    message=f"Type ordering violations: {violations[:5]}"
                )

            return True, None

        except rx.DAGHasCycle:
            return False, InvariantViolation(
                invariant="stratification",
                severity=InvariantSeverity.ERROR,
                message="Cannot check stratification: graph has cycles"
            )

    @staticmethod
    def compute_cyclomatic_complexity(graph: rx.PyDiGraph) -> int:
        """
        Cyclomatic Complexity (Betti number b₁): |E| - |V| + connected_components

        For a DAG, this measures the number of independent execution paths.
        Higher values indicate more complex control flow.

        Returns:
            The cyclomatic complexity value
        """
        num_edges = graph.num_edges()
        num_nodes = graph.num_nodes()
        num_components = rx.number_weakly_connected_components(graph)

        return num_edges - num_nodes + num_components

    @staticmethod
    def get_articulation_points(graph: rx.PyDiGraph) -> Set[int]:
        """
        Find articulation points (cut vertices) in the graph.

        Articulation points are nodes whose removal would disconnect the graph.
        These are CRITICAL nodes that need redundancy or special handling.

        Returns:
            Set of node indices that are articulation points
        """
        # rustworkx doesn't have direct articulation_points
        # We compute it by checking connectivity after hypothetical removal
        articulation = set()

        if graph.num_nodes() <= 2:
            return articulation

        # Get baseline component count
        baseline_components = rx.number_weakly_connected_components(graph)

        for idx in graph.node_indices():
            # Create a view without this node
            remaining = [i for i in graph.node_indices() if i != idx]
            if not remaining:
                continue

            # Build subgraph
            subgraph = graph.subgraph(remaining)
            new_components = rx.number_weakly_connected_components(subgraph)

            if new_components > baseline_components:
                articulation.add(idx)

        return articulation

    @staticmethod
    def get_bridge_edges(
        graph: rx.PyDiGraph,
        inv_map: Optional[Dict[int, str]] = None
    ) -> Set[Tuple[str, str]]:
        """
        Find bridge edges (cut edges) in the graph.

        Bridge edges are edges whose removal would disconnect the graph.
        These represent CRITICAL dependencies.

        Returns:
            Set of (source_id, target_id) tuples representing bridge edges
        """
        bridges = set()

        if graph.num_edges() == 0:
            return bridges

        baseline_components = rx.number_weakly_connected_components(graph)

        # Check each edge
        edge_list = list(graph.edge_index_map().items())
        for (src_idx, tgt_idx), edge_idx in edge_list:
            # Get edge data to preserve it
            edge_data = graph.get_edge_data_by_index(edge_idx)

            # Remove edge
            graph.remove_edge(src_idx, tgt_idx)

            new_components = rx.number_weakly_connected_components(graph)

            # Restore edge
            graph.add_edge(src_idx, tgt_idx, edge_data)

            if new_components > baseline_components:
                src_id = inv_map.get(src_idx, str(src_idx)) if inv_map else str(src_idx)
                tgt_id = inv_map.get(tgt_idx, str(tgt_idx)) if inv_map else str(tgt_idx)
                bridges.add((src_id, tgt_id))

        return bridges

    @staticmethod
    def validate_all(
        graph: rx.PyDiGraph,
        inv_map: Optional[Dict[int, str]] = None,
        type_order: Optional[List[str]] = None,
        raise_on_error: bool = False
    ) -> InvariantReport:
        """
        Run all invariant validations and return a comprehensive report.

        Args:
            graph: The rustworkx PyDiGraph to validate
            inv_map: Optional index->ID mapping for error messages
            type_order: Optional type ordering for stratification check
            raise_on_error: If True, raise GraphInvariantError on first ERROR

        Returns:
            InvariantReport with all results and metrics
        """
        violations = []
        metrics = {}

        # 1. Handshaking Lemma
        valid, violation = GraphInvariants.validate_handshaking_lemma(graph)
        if violation:
            violations.append(violation)
            if raise_on_error and violation.severity == InvariantSeverity.ERROR:
                from core.graph_db import GraphInvariantError
                raise GraphInvariantError(violation.message)

        # 2. DAG Acyclicity
        valid, violation = GraphInvariants.validate_dag_acyclicity(graph)
        if violation:
            violations.append(violation)
            if raise_on_error and violation.severity == InvariantSeverity.ERROR:
                from core.graph_db import GraphInvariantError
                raise GraphInvariantError(violation.message)

        # 3. Balis Degree
        valid, violation, unreachable = GraphInvariants.validate_balis_degree(graph, inv_map)
        if violation:
            violations.append(violation)
        metrics['unreachable_pairs'] = len(unreachable)

        # 4. Stratification
        valid, violation = GraphInvariants.validate_stratification(graph, type_order)
        if violation:
            violations.append(violation)

        # Compute metrics
        metrics['node_count'] = graph.num_nodes()
        metrics['edge_count'] = graph.num_edges()
        metrics['cyclomatic_complexity'] = GraphInvariants.compute_cyclomatic_complexity(graph)
        metrics['weakly_connected_components'] = rx.number_weakly_connected_components(graph)

        # Articulation points (expensive, so optional)
        if graph.num_nodes() < 1000:  # Only for reasonable sizes
            articulation = GraphInvariants.get_articulation_points(graph)
            metrics['articulation_point_count'] = len(articulation)

        # Overall validity (no ERROR-level violations)
        is_valid = all(v.severity != InvariantSeverity.ERROR for v in violations)

        return InvariantReport(
            valid=is_valid,
            violations=violations,
            metrics=metrics
        )


# =============================================================================
# INCREMENTAL VALIDATORS (For Pre-Insert Checks)
# =============================================================================

class IncrementalValidator:
    """
    Validators for checking invariants BEFORE mutations.

    These are cheaper than full validation because they only check
    the affected local neighborhood.
    """

    @staticmethod
    def would_create_cycle(
        graph: rx.PyDiGraph,
        source_idx: int,
        target_idx: int
    ) -> bool:
        """
        Check if adding edge source->target would create a cycle.

        Uses O(V+E) reachability check: if target can already reach source,
        adding source->target creates a cycle.

        Args:
            graph: The current graph
            source_idx: Source node index
            target_idx: Target node index

        Returns:
            True if the edge would create a cycle
        """
        # If target can reach source, adding source->target creates cycle
        ancestors_of_source = rx.ancestors(graph, source_idx)
        return target_idx in ancestors_of_source

    @staticmethod
    def would_violate_stratification(
        graph: rx.PyDiGraph,
        source_idx: int,
        target_idx: int,
        type_order: Optional[List[str]] = None
    ) -> bool:
        """
        Check if adding edge would violate type ordering.

        Args:
            graph: The current graph
            source_idx: Source node index
            target_idx: Target node index
            type_order: Expected type ordering

        Returns:
            True if the edge would violate stratification
        """
        if type_order is None:
            type_order = ["REQ", "RESEARCH", "PLAN", "SPEC", "CODE", "TEST"]

        type_to_rank = {t: i for i, t in enumerate(type_order)}

        source_node = graph[source_idx]
        target_node = graph[target_idx]

        if not hasattr(source_node, 'type') or not hasattr(target_node, 'type'):
            return False

        source_rank = type_to_rank.get(source_node.type, -1)
        target_rank = type_to_rank.get(target_node.type, -1)

        if source_rank == -1 or target_rank == -1:
            return False

        # Source should have lower or equal rank than target
        return source_rank > target_rank


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_graph(graph: rx.PyDiGraph, **kwargs) -> InvariantReport:
    """Convenience function to validate a graph."""
    return GraphInvariants.validate_all(graph, **kwargs)


def is_valid_dag(graph: rx.PyDiGraph) -> bool:
    """Quick check if graph is a valid DAG."""
    return rx.is_directed_acyclic_graph(graph)


def get_graph_metrics(graph: rx.PyDiGraph) -> Dict[str, Any]:
    """Get basic graph metrics without full validation."""
    return {
        'node_count': graph.num_nodes(),
        'edge_count': graph.num_edges(),
        'cyclomatic_complexity': GraphInvariants.compute_cyclomatic_complexity(graph),
        'is_dag': rx.is_directed_acyclic_graph(graph),
        'weakly_connected_components': rx.number_weakly_connected_components(graph),
    }
