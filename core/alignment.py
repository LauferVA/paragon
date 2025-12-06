"""
PARAGON ALIGNMENT - Graph Matching via Pygmtools

High-performance graph alignment for comparing code structures.

Use Cases:
1. CODE SIMILARITY: Compare two code graphs to detect clones/refactoring
2. TRACEABILITY: Match REQ nodes to SPEC nodes to CODE nodes
3. EVOLUTION: Track how a codebase changes between commits
4. DEPENDENCY MAPPING: Align import graphs between projects

Why Pygmtools over manual numpy:
1. OPTIMIZED: C/CUDA implementations for RRWM, Hungarian, etc.
2. BATCHED: Process multiple graph pairs simultaneously
3. EXTENSIBLE: Multiple algorithms (classic, neural, multi-graph)
4. STANDARDIZED: Well-tested QAP solvers

Architecture:
- GraphAligner: Main entry point for aligning ParagonDB graphs
- FeatureExtractor: Converts nodes/edges to feature vectors
- AffinityBuilder: Constructs affinity matrices for matching
- Output is (matching_matrix, alignment_score, node_mapping)

API Note (pygmtools 0.5.x):
    K = pygm.utils.build_aff_mat(node_feat1, edge_feat1, conn1, ...)
    X = pygm.rrwm(K, n1, n2)  # Soft matching matrix
    X_hard = pygm.hungarian(X, n1, n2)  # Discrete matching
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

import pygmtools as pygm

from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, EdgeType


# =============================================================================
# CONFIGURATION
# =============================================================================

# Set numpy backend (supports numpy, torch, jax, paddle, etc.)
pygm.set_backend('numpy')


class MatchingAlgorithm(str, Enum):
    """Available matching algorithms."""
    RRWM = "rrwm"           # Reweighted Random Walk Matching (fast, approximate)
    IPFP = "ipfp"           # Integer Projected Fixed Point (more accurate)
    SM = "sm"               # Spectral Matching (classic)
    HUNGARIAN = "hungarian"  # Hungarian algorithm (linear assignment only)


@dataclass
class AlignmentResult:
    """Result of graph alignment."""
    soft_matching: np.ndarray       # (n1 x n2) soft assignment matrix
    hard_matching: np.ndarray       # (n1 x n2) discrete assignment matrix
    node_mapping: Dict[str, str]    # source_id -> target_id mapping
    score: float                    # Overall alignment score (0-1)
    unmapped_source: List[str]      # Source nodes with no match
    unmapped_target: List[str]      # Target nodes with no match


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class FeatureExtractor:
    """
    Extracts numerical features from graph nodes and edges.

    Features are used to compute node/edge affinities for graph matching.
    The feature dimension must be consistent across graphs being compared.
    """

    # Default feature dimensions
    NODE_FEATURE_DIM = 32
    EDGE_FEATURE_DIM = 16  # Must be > len(EdgeType) + 1 for weight

    def __init__(self):
        """Initialize feature extractor with type encodings."""
        # One-hot encoding indices for node types
        self._node_type_idx = {nt.value: i for i, nt in enumerate(NodeType)}
        self._edge_type_idx = {et.value: i for i, et in enumerate(EdgeType)}

        self._node_type_count = len(NodeType)
        self._edge_type_count = len(EdgeType)

    def extract_node_features(
        self,
        nodes: List[NodeData],
        node_id_to_idx: Dict[str, int]
    ) -> np.ndarray:
        """
        Extract feature vectors for nodes.

        Args:
            nodes: List of NodeData
            node_id_to_idx: Mapping from node ID to array index

        Returns:
            (n x NODE_FEATURE_DIM) feature matrix
        """
        n = len(nodes)
        features = np.zeros((n, self.NODE_FEATURE_DIM), dtype=np.float32)

        for node in nodes:
            idx = node_id_to_idx.get(node.id)
            if idx is None:
                continue

            # Feature 1: Node type one-hot (dim 0 to node_type_count)
            type_idx = self._node_type_idx.get(node.type, 0)
            features[idx, type_idx] = 1.0

            # Feature 2: Content length (normalized, dim node_type_count)
            content_len = len(node.content) if node.content else 0
            features[idx, self._node_type_count] = min(content_len / 10000.0, 1.0)

            # Feature 3: Structural features from data dict
            data = node.data or {}

            # Line span (if available)
            start_line = data.get("start_line", 0)
            end_line = data.get("end_line", 0)
            line_span = end_line - start_line if end_line > start_line else 0
            features[idx, self._node_type_count + 1] = min(line_span / 500.0, 1.0)

            # Has parent (for methods)
            features[idx, self._node_type_count + 2] = 1.0 if data.get("parent") else 0.0

            # Kind encoding (secondary type info)
            kind = data.get("kind", "")
            kind_hash = hash(kind) % 16  # Simple hash to bucket
            features[idx, self._node_type_count + 3 + kind_hash] = 1.0

        return features

    def extract_edge_features(
        self,
        edges: List[EdgeData],
        node_id_to_idx: Dict[str, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract edge features and connectivity.

        Args:
            edges: List of EdgeData
            node_id_to_idx: Mapping from node ID to array index

        Returns:
            Tuple of:
            - (e x EDGE_FEATURE_DIM) feature matrix
            - (e x 2) connectivity matrix (source_idx, target_idx)
        """
        # Filter edges to only those with valid nodes
        valid_edges = [
            e for e in edges
            if e.source_id in node_id_to_idx and e.target_id in node_id_to_idx
        ]

        e = len(valid_edges)
        if e == 0:
            return np.zeros((0, self.EDGE_FEATURE_DIM), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

        features = np.zeros((e, self.EDGE_FEATURE_DIM), dtype=np.float32)
        connectivity = np.zeros((e, 2), dtype=np.int32)

        for i, edge in enumerate(valid_edges):
            # Connectivity
            connectivity[i, 0] = node_id_to_idx[edge.source_id]
            connectivity[i, 1] = node_id_to_idx[edge.target_id]

            # Feature 1: Edge type one-hot
            type_idx = self._edge_type_idx.get(edge.type, 0)
            features[i, type_idx] = 1.0

            # Feature 2: Edge weight
            features[i, self._edge_type_count] = edge.weight

        return features, connectivity


# =============================================================================
# GRAPH ALIGNER
# =============================================================================

class GraphAligner:
    """
    High-level graph alignment using pygmtools.

    Supports multiple matching algorithms and handles the conversion
    between ParagonDB structures and pygmtools' numpy arrays.

    Usage:
        aligner = GraphAligner()

        # Align two sets of nodes/edges
        result = aligner.align(nodes1, edges1, nodes2, edges2)

        # Get mapping
        for src_id, tgt_id in result.node_mapping.items():
            print(f"{src_id} -> {tgt_id}")
    """

    def __init__(
        self,
        algorithm: MatchingAlgorithm = MatchingAlgorithm.RRWM,
        node_aff_fn: Optional[Callable] = None,
        edge_aff_fn: Optional[Callable] = None,
    ):
        """
        Initialize aligner.

        Args:
            algorithm: Which matching algorithm to use
            node_aff_fn: Custom node affinity function (default: inner product)
            edge_aff_fn: Custom edge affinity function (default: inner product)
        """
        self.algorithm = algorithm
        self.node_aff_fn = node_aff_fn
        self.edge_aff_fn = edge_aff_fn
        self.feature_extractor = FeatureExtractor()

    def align(
        self,
        nodes1: List[NodeData],
        edges1: List[EdgeData],
        nodes2: List[NodeData],
        edges2: List[EdgeData],
        matching_threshold: float = 0.5,
    ) -> AlignmentResult:
        """
        Align two graphs and return the matching.

        Args:
            nodes1: Nodes from graph 1
            edges1: Edges from graph 1
            nodes2: Nodes from graph 2
            edges2: Edges from graph 2
            matching_threshold: Minimum score to consider a match valid

        Returns:
            AlignmentResult with matching matrices and node mapping
        """
        n1, n2 = len(nodes1), len(nodes2)

        if n1 == 0 or n2 == 0:
            return AlignmentResult(
                soft_matching=np.zeros((n1, n2)),
                hard_matching=np.zeros((n1, n2)),
                node_mapping={},
                score=0.0,
                unmapped_source=[n.id for n in nodes1],
                unmapped_target=[n.id for n in nodes2],
            )

        # Build node ID to index mappings
        id_to_idx1 = {n.id: i for i, n in enumerate(nodes1)}
        id_to_idx2 = {n.id: i for i, n in enumerate(nodes2)}

        # Extract features
        node_feat1 = self.feature_extractor.extract_node_features(nodes1, id_to_idx1)
        node_feat2 = self.feature_extractor.extract_node_features(nodes2, id_to_idx2)

        edge_feat1, conn1 = self.feature_extractor.extract_edge_features(edges1, id_to_idx1)
        edge_feat2, conn2 = self.feature_extractor.extract_edge_features(edges2, id_to_idx2)

        # Handle empty edge cases
        if edge_feat1.shape[0] == 0:
            edge_feat1 = np.zeros((0, FeatureExtractor.EDGE_FEATURE_DIM), dtype=np.float32)
            conn1 = np.zeros((0, 2), dtype=np.int32)
        if edge_feat2.shape[0] == 0:
            edge_feat2 = np.zeros((0, FeatureExtractor.EDGE_FEATURE_DIM), dtype=np.float32)
            conn2 = np.zeros((0, 2), dtype=np.int32)

        # Build affinity matrix
        K = pygm.utils.build_aff_mat(
            node_feat1, edge_feat1, conn1,
            node_feat2, edge_feat2, conn2,
            n1=n1, n2=n2,
            node_aff_fn=self.node_aff_fn,
            edge_aff_fn=self.edge_aff_fn,
        )

        # Solve matching problem
        X_soft = self._solve(K, n1, n2)

        # Discretize to hard assignment
        X_hard = pygm.hungarian(X_soft, n1, n2)

        # Build node mapping
        node_mapping = self._build_mapping(
            X_hard, nodes1, nodes2, matching_threshold
        )

        # Calculate score
        score = self._calculate_score(X_soft, X_hard)

        # Find unmapped nodes
        mapped_sources = set(node_mapping.keys())
        mapped_targets = set(node_mapping.values())
        unmapped_source = [n.id for n in nodes1 if n.id not in mapped_sources]
        unmapped_target = [n.id for n in nodes2 if n.id not in mapped_targets]

        return AlignmentResult(
            soft_matching=X_soft,
            hard_matching=X_hard,
            node_mapping=node_mapping,
            score=score,
            unmapped_source=unmapped_source,
            unmapped_target=unmapped_target,
        )

    def _solve(self, K: np.ndarray, n1: int, n2: int) -> np.ndarray:
        """Run the selected matching algorithm."""
        if self.algorithm == MatchingAlgorithm.RRWM:
            return pygm.rrwm(K, n1, n2)
        elif self.algorithm == MatchingAlgorithm.IPFP:
            return pygm.ipfp(K, n1, n2)
        elif self.algorithm == MatchingAlgorithm.SM:
            return pygm.sm(K, n1, n2)
        elif self.algorithm == MatchingAlgorithm.HUNGARIAN:
            # Hungarian needs doubly-stochastic initialization
            X_init = np.ones((n1, n2)) / (n1 * n2)
            return pygm.hungarian(X_init, n1, n2)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _build_mapping(
        self,
        X_hard: np.ndarray,
        nodes1: List[NodeData],
        nodes2: List[NodeData],
        threshold: float,
    ) -> Dict[str, str]:
        """Convert hard matching matrix to node ID mapping."""
        mapping = {}

        # Find assignments from X_hard (should be permutation matrix)
        assignments = np.argmax(X_hard, axis=1)

        for i, j in enumerate(assignments):
            if i < len(nodes1) and j < len(nodes2):
                # Check if this assignment is strong enough
                if X_hard[i, j] >= threshold:
                    mapping[nodes1[i].id] = nodes2[j].id

        return mapping

    def _calculate_score(self, X_soft: np.ndarray, X_hard: np.ndarray) -> float:
        """Calculate overall alignment score."""
        if X_hard.size == 0:
            return 0.0

        # Score is the sum of soft matching values at hard assignment positions
        # normalized by the number of assignments
        n_assigned = np.sum(X_hard)
        if n_assigned == 0:
            return 0.0

        score = np.sum(X_soft * X_hard) / n_assigned
        return float(score)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def align_graphs(
    nodes1: List[NodeData],
    edges1: List[EdgeData],
    nodes2: List[NodeData],
    edges2: List[EdgeData],
    algorithm: str = "rrwm",
) -> AlignmentResult:
    """
    Align two graphs and return matching.

    Convenience wrapper around GraphAligner.

    Args:
        nodes1, edges1: First graph
        nodes2, edges2: Second graph
        algorithm: "rrwm", "ipfp", "sm", or "hungarian"

    Returns:
        AlignmentResult with node mapping and scores
    """
    algo = MatchingAlgorithm(algorithm)
    aligner = GraphAligner(algorithm=algo)
    return aligner.align(nodes1, edges1, nodes2, edges2)


def compute_similarity(
    nodes1: List[NodeData],
    edges1: List[EdgeData],
    nodes2: List[NodeData],
    edges2: List[EdgeData],
) -> float:
    """
    Compute similarity score between two graphs.

    Returns a value between 0 (no similarity) and 1 (identical).
    """
    result = align_graphs(nodes1, edges1, nodes2, edges2)
    return result.score


def find_matches(
    source_nodes: List[NodeData],
    source_edges: List[EdgeData],
    target_nodes: List[NodeData],
    target_edges: List[EdgeData],
    min_score: float = 0.3,
) -> List[Tuple[str, str, float]]:
    """
    Find matching nodes between two graphs.

    Args:
        source_nodes, source_edges: Source graph
        target_nodes, target_edges: Target graph
        min_score: Minimum matching score to include

    Returns:
        List of (source_id, target_id, score) tuples
    """
    aligner = GraphAligner()
    result = aligner.align(source_nodes, source_edges, target_nodes, target_edges)

    matches = []
    for src_id, tgt_id in result.node_mapping.items():
        # Get indices
        src_idx = next((i for i, n in enumerate(source_nodes) if n.id == src_id), None)
        tgt_idx = next((i for i, n in enumerate(target_nodes) if n.id == tgt_id), None)

        if src_idx is not None and tgt_idx is not None:
            score = float(result.soft_matching[src_idx, tgt_idx])
            if score >= min_score:
                matches.append((src_id, tgt_id, score))

    return sorted(matches, key=lambda x: x[2], reverse=True)


def detect_refactoring(
    old_nodes: List[NodeData],
    old_edges: List[EdgeData],
    new_nodes: List[NodeData],
    new_edges: List[EdgeData],
) -> Dict[str, any]:
    """
    Detect refactoring changes between two versions of a codebase.

    Returns:
        Dict with:
        - renamed: List of (old_id, new_id) for renamed items
        - added: List of new_id for added items
        - removed: List of old_id for removed items
        - unchanged: List of (old_id, new_id) for unchanged items
    """
    result = align_graphs(old_nodes, old_edges, new_nodes, new_edges)

    # Categorize changes based on matching scores
    renamed = []
    unchanged = []

    for old_id, new_id in result.node_mapping.items():
        old_node = next((n for n in old_nodes if n.id == old_id), None)
        new_node = next((n for n in new_nodes if n.id == new_id), None)

        if old_node and new_node:
            # Check if names changed
            old_name = old_node.data.get("name", "")
            new_name = new_node.data.get("name", "")

            if old_name != new_name:
                renamed.append((old_id, new_id))
            else:
                unchanged.append((old_id, new_id))

    return {
        "renamed": renamed,
        "added": result.unmapped_target,
        "removed": result.unmapped_source,
        "unchanged": unchanged,
        "similarity_score": result.score,
    }
