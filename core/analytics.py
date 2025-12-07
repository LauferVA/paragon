"""
PARAGON ANALYTICS - Topological Metrics as Insight

Wave 5 Refactor: Graph theory concepts applied to code understanding

This module provides read-only analytics over the graph structure.
These metrics help answer questions like:
- Which nodes are most critical? (centrality)
- How modular is the architecture? (clustering)
- Where are the weak points? (articulation points)
- How deep are dependency chains? (diameter)

All functions are read-only queries - they observe but don't modify.
Zero collision risk with other operations.

Reference: Topological Metrics as Insight (from design discussion)
"""
import rustworkx as rx
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from core.schemas import NodeData


# =============================================================================
# DATA CLASSES FOR RESULTS
# =============================================================================

@dataclass
class CentralityReport:
    """Report on node centrality metrics."""
    node_id: str
    node_type: str
    in_degree: int
    out_degree: int
    total_degree: int
    betweenness: float
    is_hotspot: bool  # Unusually high degree


@dataclass
class GraphHealthReport:
    """Overall health metrics for the graph."""
    total_nodes: int
    total_edges: int
    density: float
    max_depth: int
    wave_count: int
    orphan_count: int
    hotspot_count: int
    articulation_point_count: int
    is_dag: bool


# =============================================================================
# CENTRALITY METRICS (Which nodes are critical?)
# =============================================================================

def compute_degree_centrality(db) -> List[CentralityReport]:
    """
    Compute degree centrality for all nodes.

    High in-degree: Many things depend on this (fragile point)
    High out-degree: This depends on many things (integration risk)

    Args:
        db: ParagonDB instance

    Returns:
        List of CentralityReport sorted by total degree (descending)
    """
    reports = []
    graph = db._graph

    # Calculate mean degree for hotspot detection
    degrees = [graph.in_degree(i) + graph.out_degree(i)
               for i in graph.node_indices()]
    mean_degree = sum(degrees) / len(degrees) if degrees else 0
    std_degree = (sum((d - mean_degree) ** 2 for d in degrees) / len(degrees)) ** 0.5 if degrees else 0
    hotspot_threshold = mean_degree + 2 * std_degree

    for idx in graph.node_indices():
        node = graph[idx]
        node_id = db._inv_map.get(idx, str(idx))

        in_deg = graph.in_degree(idx)
        out_deg = graph.out_degree(idx)
        total_deg = in_deg + out_deg

        reports.append(CentralityReport(
            node_id=node_id,
            node_type=node.type if hasattr(node, 'type') else 'unknown',
            in_degree=in_deg,
            out_degree=out_deg,
            total_degree=total_deg,
            betweenness=0.0,  # Computed separately if needed
            is_hotspot=total_deg > hotspot_threshold,
        ))

    # Sort by total degree descending
    reports.sort(key=lambda r: r.total_degree, reverse=True)
    return reports


def find_hotspots(db, threshold_percentile: float = 95) -> List[str]:
    """
    Find nodes with unusually high degree (architectural hotspots).

    These are nodes that many other nodes depend on - changes to them
    have high impact.

    Args:
        db: ParagonDB instance
        threshold_percentile: Percentile above which a node is a hotspot

    Returns:
        List of hotspot node IDs
    """
    centrality = compute_degree_centrality(db)
    if not centrality:
        return []

    # Find threshold
    degrees = sorted([r.total_degree for r in centrality])
    threshold_idx = int(len(degrees) * threshold_percentile / 100)
    threshold = degrees[threshold_idx] if threshold_idx < len(degrees) else 0

    return [r.node_id for r in centrality if r.total_degree >= threshold]


def compute_betweenness_centrality(db) -> Dict[str, float]:
    """
    Compute betweenness centrality for all nodes.

    Nodes with high betweenness lie on many shortest paths between
    other nodes - they are critical infrastructure.

    Args:
        db: ParagonDB instance

    Returns:
        Dict mapping node_id to betweenness score
    """
    graph = db._graph

    # Use rustworkx betweenness centrality
    try:
        betweenness = rx.betweenness_centrality(graph)
    except Exception:
        # Fallback if not available
        return {}

    result = {}
    for idx, score in betweenness.items():
        node_id = db._inv_map.get(idx, str(idx))
        result[node_id] = score

    return result


# =============================================================================
# STRUCTURAL VULNERABILITY (Where are the weak points?)
# =============================================================================

def find_articulation_points(db) -> List[str]:
    """
    Find articulation points (cut vertices) in the graph.

    These are nodes whose removal would disconnect the graph.
    They are single points of failure.

    Note: For DAGs, we treat the graph as undirected for this analysis.

    Args:
        db: ParagonDB instance

    Returns:
        List of articulation point node IDs
    """
    graph = db._graph

    # Convert to undirected for articulation point analysis
    try:
        # Create undirected copy
        undirected = rx.PyGraph()
        node_map = {}

        for idx in graph.node_indices():
            new_idx = undirected.add_node(graph[idx])
            node_map[idx] = new_idx

        for edge_idx in graph.edge_indices():
            src, tgt = graph.get_edge_endpoints_by_index(edge_idx)
            try:
                undirected.add_edge(node_map[src], node_map[tgt], None)
            except Exception:
                pass  # Edge might already exist

        # Find articulation points
        articulation = rx.articulation_points(undirected)

        # Map back to original node IDs
        result = []
        inv_node_map = {v: k for k, v in node_map.items()}
        for new_idx in articulation:
            orig_idx = inv_node_map.get(new_idx)
            if orig_idx is not None:
                node_id = db._inv_map.get(orig_idx, str(orig_idx))
                result.append(node_id)

        return result

    except Exception:
        return []


def find_orphan_nodes(db) -> List[str]:
    """
    Find nodes with no connections (orphans).

    Orphan nodes are either dead code or missing integration.

    Args:
        db: ParagonDB instance

    Returns:
        List of orphan node IDs
    """
    graph = db._graph
    orphans = []

    for idx in graph.node_indices():
        if graph.in_degree(idx) == 0 and graph.out_degree(idx) == 0:
            node_id = db._inv_map.get(idx, str(idx))
            orphans.append(node_id)

    return orphans


def find_dead_code_candidates(db) -> List[str]:
    """
    Find nodes that nothing depends on (except entry points).

    These are leaf nodes that might be dead code if they're not
    explicitly entry points (REQ nodes).

    Args:
        db: ParagonDB instance

    Returns:
        List of potential dead code node IDs
    """
    from core.ontology import NodeType

    graph = db._graph
    candidates = []

    for idx in graph.node_indices():
        node = graph[idx]

        # Skip REQ nodes (entry points) and orphans
        if hasattr(node, 'type') and node.type == NodeType.REQ.value:
            continue

        # Nodes with no incoming edges and some outgoing (not orphans)
        if graph.in_degree(idx) == 0 and graph.out_degree(idx) > 0:
            node_id = db._inv_map.get(idx, str(idx))
            candidates.append(node_id)

    return candidates


# =============================================================================
# GRAPH STRUCTURE METRICS (Overall health)
# =============================================================================

def compute_graph_density(db) -> float:
    """
    Compute graph density (actual edges / possible edges).

    Very dense: tight coupling (everything depends on everything)
    Very sparse: disconnected (modules not integrated)

    Args:
        db: ParagonDB instance

    Returns:
        Density ratio (0.0 to 1.0)
    """
    graph = db._graph
    n = graph.num_nodes()

    if n <= 1:
        return 0.0

    # For directed graph: max edges = n * (n - 1)
    max_edges = n * (n - 1)
    actual_edges = graph.num_edges()

    return actual_edges / max_edges if max_edges > 0 else 0.0


def compute_max_depth(db) -> int:
    """
    Compute the maximum depth of the dependency graph.

    This is the longest path from any root to any leaf.
    Deep graphs = long dependency chains = slow propagation.

    Args:
        db: ParagonDB instance

    Returns:
        Maximum depth (0 if empty)
    """
    graph = db._graph

    if graph.num_nodes() == 0:
        return 0

    try:
        # Use dag_longest_path_length if available
        return rx.dag_longest_path_length(graph)
    except Exception:
        # Fallback: compute manually
        try:
            topo_order = rx.topological_sort(graph)
            depths = {}

            for idx in topo_order:
                preds = list(graph.predecessor_indices(idx))
                if not preds:
                    depths[idx] = 0
                else:
                    depths[idx] = max(depths.get(p, 0) for p in preds) + 1

            return max(depths.values()) if depths else 0
        except Exception:
            return 0


def compute_wave_count(db) -> int:
    """
    Compute the number of execution waves (parallel layers).

    Fewer waves = more parallelism potential.
    Many waves = sequential dependency chains.

    Args:
        db: ParagonDB instance

    Returns:
        Number of waves
    """
    try:
        waves = db.get_waves()
        return len(waves)
    except Exception:
        return 0


def get_graph_health_report(db) -> GraphHealthReport:
    """
    Generate a comprehensive health report for the graph.

    Args:
        db: ParagonDB instance

    Returns:
        GraphHealthReport with all metrics
    """
    graph = db._graph

    total_nodes = graph.num_nodes()
    total_edges = graph.num_edges()
    density = compute_graph_density(db)
    max_depth = compute_max_depth(db)
    wave_count = compute_wave_count(db)
    orphans = find_orphan_nodes(db)
    hotspots = find_hotspots(db)
    articulation = find_articulation_points(db)
    is_dag = rx.is_directed_acyclic_graph(graph)

    return GraphHealthReport(
        total_nodes=total_nodes,
        total_edges=total_edges,
        density=density,
        max_depth=max_depth,
        wave_count=wave_count,
        orphan_count=len(orphans),
        hotspot_count=len(hotspots),
        articulation_point_count=len(articulation),
        is_dag=is_dag,
    )


# =============================================================================
# COMPONENT ANALYSIS (Modularity)
# =============================================================================

def find_connected_components(db) -> List[List[str]]:
    """
    Find connected components in the graph.

    Multiple components = separate subsystems.
    One component = everything is connected.

    Args:
        db: ParagonDB instance

    Returns:
        List of components (each is a list of node IDs)
    """
    graph = db._graph

    try:
        # Weakly connected components for directed graph
        components = rx.weakly_connected_components(graph)

        result = []
        for component in components:
            node_ids = [db._inv_map.get(idx, str(idx)) for idx in component]
            result.append(node_ids)

        return result

    except Exception:
        return []


def find_strongly_connected_components(db) -> List[List[str]]:
    """
    Find strongly connected components (mutual reachability).

    In a DAG, SCCs should all be single nodes.
    Multi-node SCCs indicate cycles (which violate DAG property).

    Args:
        db: ParagonDB instance

    Returns:
        List of SCCs with more than one node (potential cycles)
    """
    graph = db._graph

    try:
        sccs = rx.strongly_connected_components(graph)

        result = []
        for scc in sccs:
            if len(scc) > 1:  # Only multi-node SCCs (cycles)
                node_ids = [db._inv_map.get(idx, str(idx)) for idx in scc]
                result.append(node_ids)

        return result

    except Exception:
        return []


# =============================================================================
# TYPE-BASED ANALYTICS
# =============================================================================

def count_nodes_by_type(db) -> Dict[str, int]:
    """
    Count nodes grouped by type.

    Args:
        db: ParagonDB instance

    Returns:
        Dict mapping node type to count
    """
    graph = db._graph
    counts: Dict[str, int] = {}

    for idx in graph.node_indices():
        node = graph[idx]
        node_type = node.type if hasattr(node, 'type') else 'unknown'
        counts[node_type] = counts.get(node_type, 0) + 1

    return counts


def count_nodes_by_status(db) -> Dict[str, int]:
    """
    Count nodes grouped by status.

    Args:
        db: ParagonDB instance

    Returns:
        Dict mapping status to count
    """
    graph = db._graph
    counts: Dict[str, int] = {}

    for idx in graph.node_indices():
        node = graph[idx]
        status = node.status if hasattr(node, 'status') else 'unknown'
        counts[status] = counts.get(status, 0) + 1

    return counts


def get_type_dependency_matrix(db) -> Dict[str, Dict[str, int]]:
    """
    Compute how many edges exist between each pair of node types.

    Useful for understanding architectural patterns:
    - How many CODE -> SPEC edges?
    - How many TEST_SUITE -> CODE edges?

    Args:
        db: ParagonDB instance

    Returns:
        Nested dict: result[source_type][target_type] = edge_count
    """
    graph = db._graph
    matrix: Dict[str, Dict[str, int]] = {}

    for edge_idx in graph.edge_indices():
        try:
            src_idx, tgt_idx = graph.get_edge_endpoints_by_index(edge_idx)
            src_node = graph[src_idx]
            tgt_node = graph[tgt_idx]

            src_type = src_node.type if hasattr(src_node, 'type') else 'unknown'
            tgt_type = tgt_node.type if hasattr(tgt_node, 'type') else 'unknown'

            if src_type not in matrix:
                matrix[src_type] = {}
            matrix[src_type][tgt_type] = matrix[src_type].get(tgt_type, 0) + 1

        except Exception:
            continue

    return matrix


# =============================================================================
# REFACTORING DETECTION (via Alignment)
# =============================================================================

def detect_graph_changes(
    db_old,
    db_new,
    min_score: float = 0.3,
) -> Dict[str, any]:
    """
    Detect refactoring/structural changes between two graph states.

    Uses the alignment module to compare graphs and identify:
    - Renamed nodes (matched but name changed)
    - Added nodes (in new but not old)
    - Removed nodes (in old but not new)
    - Unchanged nodes (matched with same name)

    Args:
        db_old: ParagonDB instance with old state
        db_new: ParagonDB instance with new state
        min_score: Minimum matching score to consider (0.0-1.0)

    Returns:
        Dict with:
        - renamed: List of (old_id, new_id) tuples
        - added: List of new_id
        - removed: List of old_id
        - unchanged: List of (old_id, new_id) tuples
        - similarity_score: Overall similarity (0.0-1.0)
    """
    try:
        from core.alignment import detect_refactoring

        # Extract nodes and edges from both databases
        old_nodes = list(db_old.query_nodes())
        old_edges = list(db_old.query_edges())
        new_nodes = list(db_new.query_nodes())
        new_edges = list(db_new.query_edges())

        return detect_refactoring(old_nodes, old_edges, new_nodes, new_edges)

    except ImportError:
        return {
            "renamed": [],
            "added": [],
            "removed": [],
            "unchanged": [],
            "similarity_score": 0.0,
            "error": "alignment module not available",
        }
    except Exception as e:
        return {
            "renamed": [],
            "added": [],
            "removed": [],
            "unchanged": [],
            "similarity_score": 0.0,
            "error": str(e),
        }


def compare_graph_snapshots(
    nodes1: List,
    edges1: List,
    nodes2: List,
    edges2: List,
) -> float:
    """
    Compute similarity between two graph snapshots.

    Args:
        nodes1, edges1: First graph
        nodes2, edges2: Second graph

    Returns:
        Similarity score (0.0-1.0)
    """
    try:
        from core.alignment import compute_similarity
        return compute_similarity(nodes1, edges1, nodes2, edges2)
    except ImportError:
        return 0.0
