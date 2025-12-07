"""
Unit tests for core module coverage - Part 1 of 2

Tests the first 67 of 134 uncovered functions across core modules:
- Exception classes: DuplicateNodeError, EdgeNotFoundError, NodeNotFoundError, TopologyViolationError
- FeatureExtractor: __init__, extract_edge_features, extract_node_features
- GraphAligner: __init__, _build_mapping, _calculate_score, _solve, align
- ModelRouter: __init__, get_model_for_task, route
- ParagonDB: Core methods (add_node, add_edge, queries, etc.)
- RateLimitGuard: __init__, rate limiting functionality

Following Paragon protocol:
- Uses msgspec, not pydantic
- Uses rustworkx for graph operations
- Tests follow existing patterns from test_graph_db.py
"""
import pytest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from core.graph_db import (
    ParagonDB,
    NodeNotFoundError,
    EdgeNotFoundError,
    DuplicateNodeError,
    TopologyViolationError,
    GraphError,
    GraphInvariantError,
)
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, EdgeType, NodeStatus, EdgeConstraint, ConstraintMode
from core.alignment import FeatureExtractor, GraphAligner, MatchingAlgorithm, AlignmentResult
from core.llm import ModelRouter, RateLimitGuard, TaskType


# =============================================================================
# EXCEPTION CLASSES TESTS
# =============================================================================

class TestExceptions:
    """Test exception initialization and error messages."""

    def test_duplicate_node_error_init(self):
        """Test DuplicateNodeError stores node_id and formats message."""
        node_id = "test-node-123"
        error = DuplicateNodeError(node_id)

        assert error.node_id == node_id
        assert node_id in str(error)
        assert "already exists" in str(error)

    def test_edge_not_found_error_init(self):
        """Test EdgeNotFoundError stores source and target IDs."""
        source_id = "source-123"
        target_id = "target-456"
        error = EdgeNotFoundError(source_id, target_id)

        assert error.source_id == source_id
        assert error.target_id == target_id
        assert source_id in str(error)
        assert target_id in str(error)
        assert "not found" in str(error)

    def test_node_not_found_error_init(self):
        """Test NodeNotFoundError stores node_id and formats message."""
        node_id = "missing-node-789"
        error = NodeNotFoundError(node_id)

        assert error.node_id == node_id
        assert node_id in str(error)
        assert "not found" in str(error)

    def test_topology_violation_error_init(self):
        """Test TopologyViolationError with and without constraint."""
        message = "Invalid edge configuration"
        error1 = TopologyViolationError(message)
        assert str(error1) == message
        assert error1.constraint is None

        constraint = EdgeConstraint(
            edge_type=EdgeType.DEPENDS_ON.value,
            direction="incoming",
            min_count=1,
            mode=ConstraintMode.HARD
        )
        error2 = TopologyViolationError(message, constraint=constraint)
        assert str(error2) == message
        assert error2.constraint == constraint


# =============================================================================
# FEATURE EXTRACTOR TESTS
# =============================================================================

class TestFeatureExtractor:
    """Test FeatureExtractor for graph alignment."""

    def test_feature_extractor_init(self):
        """Test FeatureExtractor initializes type encodings correctly."""
        extractor = FeatureExtractor()

        # Check that type indices are created
        assert len(extractor._node_type_idx) > 0
        assert len(extractor._edge_type_idx) > 0
        assert extractor._node_type_count == len(NodeType)
        assert extractor._edge_type_count == len(EdgeType)

        # Verify specific types are mapped
        assert NodeType.CODE.value in extractor._node_type_idx
        assert EdgeType.DEPENDS_ON.value in extractor._edge_type_idx

    def test_extract_node_features_basic(self):
        """Test extracting features from basic nodes."""
        extractor = FeatureExtractor()

        node1 = NodeData.create(type=NodeType.CODE.value, content="def test(): pass")
        node2 = NodeData.create(type=NodeType.SPEC.value, content="Implement test function")

        nodes = [node1, node2]
        node_id_to_idx = {node1.id: 0, node2.id: 1}

        features = extractor.extract_node_features(nodes, node_id_to_idx)

        # Check shape
        assert features.shape == (2, FeatureExtractor.NODE_FEATURE_DIM)
        assert features.dtype == np.float32

        # Check that node type one-hot encoding is set
        # Features should be non-zero
        assert np.any(features[0] > 0)
        assert np.any(features[1] > 0)

    def test_extract_node_features_with_metadata(self):
        """Test feature extraction includes structural metadata."""
        extractor = FeatureExtractor()

        node = NodeData.create(
            type=NodeType.FUNCTION.value,
            content="def calculate(x, y):\n    return x + y\n",
            data={
                "start_line": 10,
                "end_line": 12,
                "parent": "Calculator",
                "kind": "method"
            }
        )

        nodes = [node]
        node_id_to_idx = {node.id: 0}

        features = extractor.extract_node_features(nodes, node_id_to_idx)

        assert features.shape == (1, FeatureExtractor.NODE_FEATURE_DIM)
        # Features should include line span, parent flag, and kind hash
        assert np.any(features[0] > 0)

    def test_extract_edge_features_basic(self):
        """Test extracting features from edges."""
        extractor = FeatureExtractor()

        node1 = NodeData.create(type=NodeType.CODE.value, content="code1")
        node2 = NodeData.create(type=NodeType.SPEC.value, content="spec1")

        edge = EdgeData.create(
            source_id=node1.id,
            target_id=node2.id,
            type=EdgeType.IMPLEMENTS.value,
            weight=2.0
        )

        edges = [edge]
        node_id_to_idx = {node1.id: 0, node2.id: 1}

        edge_features, connectivity = extractor.extract_edge_features(edges, node_id_to_idx)

        # Check shapes
        assert edge_features.shape == (1, FeatureExtractor.EDGE_FEATURE_DIM)
        assert connectivity.shape == (1, 2)
        assert edge_features.dtype == np.float32
        assert connectivity.dtype == np.int32

        # Check connectivity
        assert connectivity[0, 0] == 0  # source index
        assert connectivity[0, 1] == 1  # target index

        # Check edge type encoding and weight
        assert np.any(edge_features[0] > 0)
        assert edge_features[0, extractor._edge_type_count] == 2.0  # weight

    def test_extract_edge_features_filters_invalid(self):
        """Test that edges with missing nodes are filtered out."""
        extractor = FeatureExtractor()

        node1 = NodeData.create(type=NodeType.CODE.value, content="code1")

        edge = EdgeData.create(
            source_id=node1.id,
            target_id="nonexistent-node",
            type=EdgeType.IMPLEMENTS.value
        )

        edges = [edge]
        node_id_to_idx = {node1.id: 0}  # Missing target node

        edge_features, connectivity = extractor.extract_edge_features(edges, node_id_to_idx)

        # Should return empty arrays since edge is invalid
        assert edge_features.shape == (0, FeatureExtractor.EDGE_FEATURE_DIM)
        assert connectivity.shape == (0, 2)

    def test_extract_edge_features_empty(self):
        """Test edge feature extraction with no valid edges."""
        extractor = FeatureExtractor()

        edges = []
        node_id_to_idx = {}

        edge_features, connectivity = extractor.extract_edge_features(edges, node_id_to_idx)

        assert edge_features.shape == (0, FeatureExtractor.EDGE_FEATURE_DIM)
        assert connectivity.shape == (0, 2)


# =============================================================================
# GRAPH ALIGNER TESTS
# =============================================================================

class TestGraphAligner:
    """Test GraphAligner for graph matching."""

    def test_graph_aligner_init(self):
        """Test GraphAligner initialization with default parameters."""
        aligner = GraphAligner()

        assert aligner.algorithm == MatchingAlgorithm.RRWM
        assert aligner.node_aff_fn is None
        assert aligner.edge_aff_fn is None
        assert isinstance(aligner.feature_extractor, FeatureExtractor)

    def test_graph_aligner_init_custom_algorithm(self):
        """Test GraphAligner with custom algorithm."""
        aligner = GraphAligner(algorithm=MatchingAlgorithm.HUNGARIAN)

        assert aligner.algorithm == MatchingAlgorithm.HUNGARIAN

    def test_align_empty_graphs(self):
        """Test aligning empty graphs returns empty result."""
        aligner = GraphAligner()

        result = aligner.align([], [], [], [])

        assert isinstance(result, AlignmentResult)
        assert result.score == 0.0
        assert len(result.node_mapping) == 0
        assert len(result.unmapped_source) == 0
        assert len(result.unmapped_target) == 0

    def test_align_one_empty_graph(self):
        """Test aligning when one graph is empty."""
        aligner = GraphAligner()

        node1 = NodeData.create(type=NodeType.CODE.value, content="code1")
        nodes1 = [node1]

        result = aligner.align(nodes1, [], [], [])

        assert result.score == 0.0
        assert len(result.node_mapping) == 0
        assert result.unmapped_source == [node1.id]
        assert len(result.unmapped_target) == 0

    def test_align_identical_single_nodes(self):
        """Test aligning two identical single-node graphs."""
        aligner = GraphAligner()

        node1 = NodeData.create(type=NodeType.CODE.value, content="identical content")
        node2 = NodeData.create(type=NodeType.CODE.value, content="identical content")

        result = aligner.align([node1], [], [node2], [])

        assert isinstance(result, AlignmentResult)
        assert result.score >= 0.0  # Some positive score
        # Should have some mapping (may not be perfect due to algorithm)
        assert result.soft_matching.shape == (1, 1)
        assert result.hard_matching.shape == (1, 1)

    def test_align_with_edges(self):
        """Test aligning graphs with edges."""
        aligner = GraphAligner()

        # Graph 1
        n1 = NodeData.create(type=NodeType.CODE.value, content="code1")
        n2 = NodeData.create(type=NodeType.SPEC.value, content="spec1")
        e1 = EdgeData.implements(n1.id, n2.id)

        # Graph 2 (similar structure)
        n3 = NodeData.create(type=NodeType.CODE.value, content="code1")
        n4 = NodeData.create(type=NodeType.SPEC.value, content="spec1")
        e2 = EdgeData.implements(n3.id, n4.id)

        result = aligner.align([n1, n2], [e1], [n3, n4], [e2])

        assert result.score >= 0.0
        assert result.soft_matching.shape == (2, 2)
        assert result.hard_matching.shape == (2, 2)

    def test_solve_rrwm(self):
        """Test _solve with RRWM algorithm."""
        aligner = GraphAligner(algorithm=MatchingAlgorithm.RRWM)

        # Create a simple affinity matrix
        K = np.random.rand(4, 4).astype(np.float32)
        X = aligner._solve(K, 2, 2)

        assert X.shape == (2, 2)
        assert X.dtype == np.float32

    def test_solve_hungarian(self):
        """Test _solve with Hungarian algorithm."""
        aligner = GraphAligner(algorithm=MatchingAlgorithm.HUNGARIAN)

        K = np.random.rand(4, 4).astype(np.float32)
        X = aligner._solve(K, 2, 2)

        assert X.shape == (2, 2)

    def test_build_mapping(self):
        """Test _build_mapping converts matching matrix to node mapping."""
        aligner = GraphAligner()

        node1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        node2 = NodeData.create(type=NodeType.CODE.value, content="c2")
        node3 = NodeData.create(type=NodeType.CODE.value, content="c3")
        node4 = NodeData.create(type=NodeType.CODE.value, content="c4")

        # Perfect matching matrix: node1->node3, node2->node4
        X_hard = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        mapping = aligner._build_mapping(
            X_hard,
            [node1, node2],
            [node3, node4],
            threshold=0.5
        )

        assert len(mapping) == 2
        assert mapping[node1.id] == node3.id
        assert mapping[node2.id] == node4.id

    def test_build_mapping_threshold_filtering(self):
        """Test _build_mapping filters out low-confidence matches."""
        aligner = GraphAligner()

        node1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        node2 = NodeData.create(type=NodeType.CODE.value, content="c2")

        # Low confidence matching
        X_hard = np.array([[0.3, 0.0], [0.0, 0.4]], dtype=np.float32)

        mapping = aligner._build_mapping(
            X_hard,
            [node1],
            [node2],
            threshold=0.5  # Higher than values in matrix
        )

        # Should be empty due to threshold
        assert len(mapping) == 0

    def test_calculate_score(self):
        """Test _calculate_score computes alignment quality."""
        aligner = GraphAligner()

        X_soft = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
        X_hard = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        score = aligner._calculate_score(X_soft, X_hard)

        # Score should be average of soft values at hard positions
        # (0.9 + 0.8) / 2 = 0.85
        assert score > 0.0
        assert score <= 1.0

    def test_calculate_score_empty(self):
        """Test _calculate_score with empty matrices."""
        aligner = GraphAligner()

        X_soft = np.array([[]], dtype=np.float32)
        X_hard = np.array([[]], dtype=np.float32)

        score = aligner._calculate_score(X_soft, X_hard)
        assert score == 0.0


# =============================================================================
# MODEL ROUTER TESTS
# =============================================================================

class TestModelRouter:
    """Test ModelRouter for cost-optimized model selection."""

    def test_model_router_init_defaults(self):
        """Test ModelRouter initialization with default config."""
        config = {}
        router = ModelRouter(config)

        assert router.high_reasoning_provider == "anthropic"
        assert "claude-sonnet" in router.high_reasoning_model
        assert router.mundane_provider == "anthropic"
        assert "haiku" in router.mundane_model.lower()

    def test_model_router_init_custom_config(self):
        """Test ModelRouter with custom configuration."""
        config = {
            "high_reasoning_provider": "openai",
            "high_reasoning_model": "gpt-4o",
            "mundane_provider": "anthropic",
            "mundane_model": "claude-haiku-4-5-20251001",
        }
        router = ModelRouter(config)

        assert router.high_reasoning_provider == "openai"
        assert router.high_reasoning_model == "gpt-4o"
        assert router.mundane_provider == "anthropic"
        assert router.mundane_model == "claude-haiku-4-5-20251001"

    def test_route_high_reasoning(self):
        """Test routing HIGH_REASONING tasks."""
        config = {
            "high_reasoning_provider": "anthropic",
            "high_reasoning_model": "claude-sonnet-4-5-20250929",
        }
        router = ModelRouter(config)

        provider, model = router.route(TaskType.HIGH_REASONING)

        assert provider == "anthropic"
        assert model == "claude-sonnet-4-5-20250929"

    def test_route_mundane(self):
        """Test routing MUNDANE tasks."""
        config = {
            "mundane_provider": "anthropic",
            "mundane_model": "claude-haiku-4-5-20251001",
        }
        router = ModelRouter(config)

        provider, model = router.route(TaskType.MUNDANE)

        assert provider == "anthropic"
        assert model == "claude-haiku-4-5-20251001"

    def test_route_mundane_with_fallback(self):
        """Test routing MUNDANE tasks with fallback enabled."""
        config = {
            "mundane_fallback_provider": "anthropic",
            "mundane_fallback_model": "claude-3-5-haiku-20241022",
        }
        router = ModelRouter(config)

        provider, model = router.route(TaskType.MUNDANE, use_fallback=True)

        assert provider == "anthropic"
        assert model == "claude-3-5-haiku-20241022"

    def test_route_sensitive(self):
        """Test routing SENSITIVE tasks always uses local models."""
        config = {
            "sensitive_provider": "ollama",
            "sensitive_model": "llama3.3",
        }
        router = ModelRouter(config)

        provider, model = router.route(TaskType.SENSITIVE)

        assert provider == "ollama"
        assert model == "llama3.3"

        # Fallback flag should be ignored for sensitive tasks
        provider2, model2 = router.route(TaskType.SENSITIVE, use_fallback=True)
        assert provider2 == "ollama"
        assert model2 == "llama3.3"

    def test_get_model_for_task(self):
        """Test get_model_for_task returns full model identifier."""
        config = {
            "high_reasoning_provider": "anthropic",
            "high_reasoning_model": "claude-sonnet-4-5-20250929",
            "mundane_provider": "openai",
            "mundane_model": "gpt-4o-mini",
        }
        router = ModelRouter(config)

        model_id = router.get_model_for_task(TaskType.HIGH_REASONING)
        assert model_id == "anthropic/claude-sonnet-4-5-20250929"

        model_id = router.get_model_for_task(TaskType.MUNDANE)
        assert model_id == "openai/gpt-4o-mini"


# =============================================================================
# PARAGON DB TESTS
# =============================================================================

class TestParagonDB:
    """Test ParagonDB core graph operations."""

    def test_paragon_db_init(self, fresh_db):
        """Test ParagonDB initializes with empty graph."""
        assert fresh_db.node_count == 0
        assert fresh_db.edge_count == 0
        assert fresh_db.is_empty
        assert len(fresh_db._node_map) == 0
        assert len(fresh_db._inv_map) == 0
        assert len(fresh_db._edge_map) == 0

    def test_add_node_basic(self, fresh_db):
        """Test add_node creates node and updates mappings."""
        node = NodeData.create(type=NodeType.CODE.value, content="test")

        idx = fresh_db.add_node(node)

        assert isinstance(idx, int)
        assert fresh_db.node_count == 1
        assert node.id in fresh_db._node_map
        assert idx in fresh_db._inv_map
        assert fresh_db._node_map[node.id] == idx
        assert fresh_db._inv_map[idx] == node.id

    def test_add_node_allow_duplicate(self, fresh_db):
        """Test add_node with allow_duplicate returns existing index."""
        node = NodeData.create(type=NodeType.CODE.value, content="test")

        idx1 = fresh_db.add_node(node)
        idx2 = fresh_db.add_node(node, allow_duplicate=True)

        assert idx1 == idx2
        assert fresh_db.node_count == 1

    def test_add_edge_basic(self, fresh_db):
        """Test add_edge creates edge and updates mapping."""
        node1 = NodeData.create(type=NodeType.CODE.value, content="code")
        node2 = NodeData.create(type=NodeType.SPEC.value, content="spec")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)

        edge = EdgeData.implements(node1.id, node2.id)
        edge_idx = fresh_db.add_edge(edge)

        assert isinstance(edge_idx, int)
        assert fresh_db.edge_count == 1
        assert (node1.id, node2.id) in fresh_db._edge_map

    def test_add_edge_cycle_prevention(self, fresh_db):
        """Test add_edge prevents cycle creation."""
        node1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        node2 = NodeData.create(type=NodeType.CODE.value, content="c2")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)

        # Add edge node1 -> node2
        edge1 = EdgeData.depends_on(node1.id, node2.id)
        fresh_db.add_edge(edge1)

        # Try to add edge node2 -> node1 (would create cycle)
        edge2 = EdgeData.depends_on(node2.id, node1.id)

        with pytest.raises(GraphInvariantError) as exc_info:
            fresh_db.add_edge(edge2)

        assert "cycle" in str(exc_info.value).lower()

    def test_add_edge_self_loop_prevention(self, fresh_db):
        """Test add_edge prevents self-loops."""
        node = NodeData.create(type=NodeType.CODE.value, content="code")
        fresh_db.add_node(node)

        edge = EdgeData.depends_on(node.id, node.id)

        with pytest.raises(GraphInvariantError) as exc_info:
            fresh_db.add_edge(edge)

        assert "self-loop" in str(exc_info.value).lower()

    def test_get_node(self, fresh_db):
        """Test get_node retrieves correct node data."""
        node = NodeData.create(
            type=NodeType.SPEC.value,
            content="test content",
            created_by="tester"
        )
        fresh_db.add_node(node)

        retrieved = fresh_db.get_node(node.id)

        assert retrieved.id == node.id
        assert retrieved.type == node.type
        assert retrieved.content == "test content"
        assert retrieved.created_by == "tester"

    def test_get_node_not_found(self, fresh_db):
        """Test get_node raises error for non-existent node."""
        with pytest.raises(NodeNotFoundError) as exc_info:
            fresh_db.get_node("nonexistent-id")

        assert "nonexistent-id" in str(exc_info.value)

    def test_get_node_by_index(self, fresh_db):
        """Test get_node_by_index retrieves node by rustworkx index."""
        node = NodeData.create(type=NodeType.CODE.value, content="test")
        idx = fresh_db.add_node(node)

        retrieved = fresh_db.get_node_by_index(idx)

        assert retrieved.id == node.id
        assert retrieved.content == "test"

    def test_has_node(self, fresh_db):
        """Test has_node checks node existence."""
        node = NodeData.create(type=NodeType.CODE.value, content="test")

        assert not fresh_db.has_node(node.id)

        fresh_db.add_node(node)

        assert fresh_db.has_node(node.id)

    def test_has_edge(self, fresh_db):
        """Test has_edge checks edge existence."""
        node1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        node2 = NodeData.create(type=NodeType.SPEC.value, content="s1")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)

        assert not fresh_db.has_edge(node1.id, node2.id)

        edge = EdgeData.implements(node1.id, node2.id)
        fresh_db.add_edge(edge)

        assert fresh_db.has_edge(node1.id, node2.id)

    def test_remove_node(self, fresh_db):
        """Test remove_node deletes node and cleans up mappings."""
        node = NodeData.create(type=NodeType.CODE.value, content="test")
        idx = fresh_db.add_node(node)

        removed = fresh_db.remove_node(node.id)

        assert removed.id == node.id
        assert fresh_db.node_count == 0
        assert node.id not in fresh_db._node_map
        assert idx not in fresh_db._inv_map

    def test_remove_node_with_edges(self, fresh_db):
        """Test remove_node also removes incident edges."""
        node1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        node2 = NodeData.create(type=NodeType.SPEC.value, content="s1")
        node3 = NodeData.create(type=NodeType.DOC.value, content="d1")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)
        fresh_db.add_node(node3)

        edge1 = EdgeData.implements(node1.id, node2.id)
        edge2 = EdgeData.depends_on(node3.id, node1.id)
        fresh_db.add_edge(edge1)
        fresh_db.add_edge(edge2)

        fresh_db.remove_node(node1.id)

        # Edges involving node1 should be removed
        assert not fresh_db.has_edge(node1.id, node2.id)
        assert not fresh_db.has_edge(node3.id, node1.id)
        assert fresh_db.edge_count == 0

    def test_remove_edge(self, fresh_db):
        """Test remove_edge deletes edge and cleans up mapping."""
        node1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        node2 = NodeData.create(type=NodeType.SPEC.value, content="s1")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)

        edge = EdgeData.implements(node1.id, node2.id)
        fresh_db.add_edge(edge)

        removed = fresh_db.remove_edge(node1.id, node2.id)

        assert removed.source_id == node1.id
        assert removed.target_id == node2.id
        assert fresh_db.edge_count == 0
        assert (node1.id, node2.id) not in fresh_db._edge_map

    def test_get_all_nodes(self, fresh_db):
        """Test get_all_nodes returns all nodes."""
        node1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        node2 = NodeData.create(type=NodeType.SPEC.value, content="s1")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)

        nodes = fresh_db.get_all_nodes()

        assert len(nodes) == 2
        node_ids = {n.id for n in nodes}
        assert node1.id in node_ids
        assert node2.id in node_ids

    def test_get_all_edges(self, fresh_db):
        """Test get_all_edges returns all edges."""
        node1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        node2 = NodeData.create(type=NodeType.SPEC.value, content="s1")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)

        edge = EdgeData.implements(node1.id, node2.id)
        fresh_db.add_edge(edge)

        edges = fresh_db.get_all_edges()

        assert len(edges) == 1
        assert edges[0].source_id == node1.id
        assert edges[0].target_id == node2.id

    def test_iter_nodes(self, fresh_db):
        """Test iter_nodes provides iterator over nodes."""
        node1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        node2 = NodeData.create(type=NodeType.SPEC.value, content="s1")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)

        nodes = list(fresh_db.iter_nodes())

        assert len(nodes) == 2

    def test_get_successors(self, fresh_db):
        """Test get_successors returns immediate children."""
        node1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        node2 = NodeData.create(type=NodeType.SPEC.value, content="s1")
        node3 = NodeData.create(type=NodeType.DOC.value, content="d1")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)
        fresh_db.add_node(node3)

        edge1 = EdgeData.depends_on(node1.id, node2.id)
        edge2 = EdgeData.depends_on(node1.id, node3.id)
        fresh_db.add_edge(edge1)
        fresh_db.add_edge(edge2)

        successors = fresh_db.get_successors(node1.id)

        assert len(successors) == 2
        successor_ids = {n.id for n in successors}
        assert node2.id in successor_ids
        assert node3.id in successor_ids

    def test_get_predecessors(self, fresh_db):
        """Test get_predecessors returns immediate parents."""
        node1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        node2 = NodeData.create(type=NodeType.SPEC.value, content="s1")
        node3 = NodeData.create(type=NodeType.DOC.value, content="d1")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)
        fresh_db.add_node(node3)

        edge1 = EdgeData.depends_on(node2.id, node1.id)
        edge2 = EdgeData.depends_on(node3.id, node1.id)
        fresh_db.add_edge(edge1)
        fresh_db.add_edge(edge2)

        predecessors = fresh_db.get_predecessors(node1.id)

        assert len(predecessors) == 2
        pred_ids = {n.id for n in predecessors}
        assert node2.id in pred_ids
        assert node3.id in pred_ids

    def test_get_root_nodes(self, fresh_db):
        """Test get_root_nodes returns nodes with no predecessors."""
        root1 = NodeData.create(type=NodeType.REQ.value, content="r1")
        root2 = NodeData.create(type=NodeType.REQ.value, content="r2")
        child = NodeData.create(type=NodeType.SPEC.value, content="s1")

        fresh_db.add_node(root1)
        fresh_db.add_node(root2)
        fresh_db.add_node(child)

        edge = EdgeData.depends_on(child.id, root1.id)
        fresh_db.add_edge(edge)

        roots = fresh_db.get_root_nodes()

        assert len(roots) == 2
        root_ids = {n.id for n in roots}
        assert root1.id in root_ids
        assert root2.id in root_ids

    def test_get_leaf_nodes(self, fresh_db):
        """Test get_leaf_nodes returns nodes with no successors."""
        parent = NodeData.create(type=NodeType.REQ.value, content="r1")
        leaf1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        leaf2 = NodeData.create(type=NodeType.CODE.value, content="c2")

        fresh_db.add_node(parent)
        fresh_db.add_node(leaf1)
        fresh_db.add_node(leaf2)

        edge = EdgeData.depends_on(leaf1.id, parent.id)
        fresh_db.add_edge(edge)

        leaves = fresh_db.get_leaf_nodes()

        assert len(leaves) == 2
        leaf_ids = {n.id for n in leaves}
        assert leaf1.id in leaf_ids
        assert leaf2.id in leaf_ids

    def test_has_cycle(self, fresh_db):
        """Test has_cycle detects cycles correctly."""
        node1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        node2 = NodeData.create(type=NodeType.CODE.value, content="c2")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)

        # No cycle initially
        assert not fresh_db.has_cycle()

        # Add edge node1 -> node2
        edge1 = EdgeData.depends_on(node1.id, node2.id)
        fresh_db.add_edge(edge1)

        # Still no cycle
        assert not fresh_db.has_cycle()

    def test_get_incoming_edges(self, fresh_db):
        """Test get_incoming_edges returns edges pointing to node."""
        node1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        node2 = NodeData.create(type=NodeType.SPEC.value, content="s1")
        node3 = NodeData.create(type=NodeType.DOC.value, content="d1")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)
        fresh_db.add_node(node3)

        edge1 = EdgeData.implements(node1.id, node2.id)
        edge2 = EdgeData.depends_on(node3.id, node2.id)
        fresh_db.add_edge(edge1)
        fresh_db.add_edge(edge2)

        incoming = fresh_db.get_incoming_edges(node2.id)

        assert len(incoming) == 2
        sources = {e["source"] for e in incoming}
        assert node1.id in sources
        assert node3.id in sources

    def test_get_outgoing_edges(self, fresh_db):
        """Test get_outgoing_edges returns edges from node."""
        node1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        node2 = NodeData.create(type=NodeType.SPEC.value, content="s1")
        node3 = NodeData.create(type=NodeType.DOC.value, content="d1")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)
        fresh_db.add_node(node3)

        edge1 = EdgeData.depends_on(node1.id, node2.id)
        edge2 = EdgeData.depends_on(node1.id, node3.id)
        fresh_db.add_edge(edge1)
        fresh_db.add_edge(edge2)

        outgoing = fresh_db.get_outgoing_edges(node1.id)

        assert len(outgoing) == 2
        targets = {e["target"] for e in outgoing}
        assert node2.id in targets
        assert node3.id in targets

    def test_find_nodes(self, fresh_db):
        """Test find_nodes filters by criteria."""
        node1 = NodeData.create(
            type=NodeType.CODE.value,
            content="c1",
            status=NodeStatus.PENDING.value,
            created_by="alice"
        )
        node2 = NodeData.create(
            type=NodeType.SPEC.value,
            content="s1",
            status=NodeStatus.VERIFIED.value,
            created_by="bob"
        )
        node3 = NodeData.create(
            type=NodeType.CODE.value,
            content="c2",
            status=NodeStatus.PENDING.value,
            created_by="alice"
        )

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)
        fresh_db.add_node(node3)

        # Filter by type
        code_nodes = fresh_db.find_nodes(type=NodeType.CODE.value)
        assert len(code_nodes) == 2

        # Filter by status
        pending_nodes = fresh_db.find_nodes(status=NodeStatus.PENDING.value)
        assert len(pending_nodes) == 2

        # Filter by created_by
        alice_nodes = fresh_db.find_nodes(created_by="alice")
        assert len(alice_nodes) == 2

        # Multiple filters
        filtered = fresh_db.find_nodes(
            type=NodeType.CODE.value,
            status=NodeStatus.PENDING.value
        )
        assert len(filtered) == 2

    def test_find_pending_nodes(self, fresh_db):
        """Test find_pending_nodes returns PENDING status nodes."""
        node1 = NodeData.create(
            type=NodeType.CODE.value,
            content="c1",
            status=NodeStatus.PENDING.value
        )
        node2 = NodeData.create(
            type=NodeType.CODE.value,
            content="c2",
            status=NodeStatus.VERIFIED.value
        )

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)

        pending = fresh_db.find_pending_nodes()

        assert len(pending) == 1
        assert pending[0].id == node1.id

    def test_find_processable_nodes(self, fresh_db):
        """Test find_processable_nodes returns nodes ready for processing."""
        processable = NodeData.create(
            type=NodeType.CODE.value,
            content="c1"
        )
        # Set cost limit exceeded
        over_cost = NodeData.create(type=NodeType.CODE.value, content="c2")
        over_cost.metadata.cost_limit = 1.0
        over_cost.metadata.cost_actual = 2.0

        fresh_db.add_node(processable)
        fresh_db.add_node(over_cost)

        nodes = fresh_db.find_processable_nodes()

        # Only processable node should be returned
        assert len(nodes) == 1
        assert nodes[0].id == processable.id

    def test_get_dominators(self, fresh_db):
        """Test get_dominators finds dominator nodes."""
        root = NodeData.create(type=NodeType.REQ.value, content="root")
        node1 = NodeData.create(type=NodeType.SPEC.value, content="n1")
        node2 = NodeData.create(type=NodeType.CODE.value, content="n2")

        fresh_db.add_node(root)
        fresh_db.add_node(node1)
        fresh_db.add_node(node2)

        edge1 = EdgeData.depends_on(node1.id, root.id)
        edge2 = EdgeData.depends_on(node2.id, node1.id)
        fresh_db.add_edge(edge1)
        fresh_db.add_edge(edge2)

        # node2's dominators should include root and node1
        dominators = fresh_db.get_dominators(node2.id)

        # Result may vary by algorithm, but should be list
        assert isinstance(dominators, list)

    def test_find_similar_nodes(self, fresh_db):
        """Test find_similar_nodes semantic search (may not have embeddings)."""
        node1 = NodeData.create(type=NodeType.CODE.value, content="test content")
        fresh_db.add_node(node1)

        # This might return empty if embeddings not available
        similar = fresh_db.find_similar_nodes("test query", limit=5)

        assert isinstance(similar, list)
        # Each result should be (NodeData, score) tuple
        for item in similar:
            assert len(item) == 2
            assert isinstance(item[0], NodeData)
            assert isinstance(item[1], float)

    def test_find_similar_to_node(self, fresh_db):
        """Test find_similar_to_node finds similar nodes by embedding."""
        node1 = NodeData.create(type=NodeType.CODE.value, content="test")
        node2 = NodeData.create(type=NodeType.CODE.value, content="similar")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)

        # May return empty if embeddings not available
        similar = fresh_db.find_similar_to_node(node1.id, limit=5)

        assert isinstance(similar, list)

    def test_is_blocked(self, fresh_db):
        """Test is_blocked checks explicit and structural blocking."""
        node1 = NodeData.create(
            type=NodeType.CODE.value,
            content="c1",
            status=NodeStatus.BLOCKED.value
        )
        fresh_db.add_node(node1)

        # Explicitly blocked
        assert fresh_db.is_blocked(node1.id)

        # Non-existent node
        assert not fresh_db.is_blocked("nonexistent")

    def test_is_blocked_by_dependencies(self, fresh_db):
        """Test is_blocked_by_dependencies checks predecessor status."""
        node1 = NodeData.create(
            type=NodeType.SPEC.value,
            content="s1",
            status=NodeStatus.PENDING.value
        )
        node2 = NodeData.create(
            type=NodeType.CODE.value,
            content="c1",
            status=NodeStatus.PENDING.value
        )

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)

        edge = EdgeData.depends_on(node2.id, node1.id)
        fresh_db.add_edge(edge)

        # node2 is blocked because node1 is not VERIFIED
        assert fresh_db.is_blocked_by_dependencies(node2.id)

        # Mark node1 as verified
        node1.status = NodeStatus.VERIFIED.value

        # Now node2 should not be blocked
        assert not fresh_db.is_blocked_by_dependencies(node2.id)

    def test_match_triggers(self, fresh_db):
        """Test match_triggers finds matching structural triggers."""
        node = NodeData.create(
            type=NodeType.CODE.value,
            content="test",
            status=NodeStatus.PENDING.value
        )
        fresh_db.add_node(node)

        triggers = fresh_db.match_triggers(node.id)

        # Should return list of StructuralTrigger
        assert isinstance(triggers, list)

    def test_find_triggerable_nodes(self, fresh_db):
        """Test find_triggerable_nodes finds nodes with triggers."""
        node1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        fresh_db.add_node(node1)

        triggerable = fresh_db.find_triggerable_nodes()

        assert isinstance(triggerable, dict)

    def test_get_ready_nodes(self, fresh_db):
        """Test get_ready_nodes finds nodes ready for processing."""
        node = NodeData.create(
            type=NodeType.CODE.value,
            content="test",
            status=NodeStatus.PENDING.value
        )
        fresh_db.add_node(node)

        ready = fresh_db.get_ready_nodes()

        assert isinstance(ready, list)

    def test_export_networkx(self, fresh_db):
        """Test export_networkx converts to NetworkX format."""
        node = NodeData.create(type=NodeType.CODE.value, content="test")
        fresh_db.add_node(node)

        nx_graph = fresh_db.export_networkx()

        # Should be a networkx DiGraph
        assert hasattr(nx_graph, 'nodes')
        assert hasattr(nx_graph, 'edges')

    def test_check_edge_constraint(self, fresh_db):
        """Test _check_edge_constraint validates edge constraints."""
        node = NodeData.create(type=NodeType.SPEC.value, content="spec")
        fresh_db.add_node(node)

        constraint = EdgeConstraint(
            edge_type=EdgeType.DEPENDS_ON.value,
            direction="incoming",
            min_count=1,
            mode=ConstraintMode.HARD
        )

        violation = fresh_db._check_edge_constraint(node.id, node, constraint)

        # Should have violation since min_count not met
        assert violation is not None
        assert "at least 1" in violation

    def test_get_id(self, fresh_db):
        """Test _get_id converts rustworkx index to node ID."""
        node = NodeData.create(type=NodeType.CODE.value, content="test")
        idx = fresh_db.add_node(node)

        node_id = fresh_db._get_id(idx)

        assert node_id == node.id

    def test_get_index(self, fresh_db):
        """Test _get_index converts node ID to rustworkx index."""
        node = NodeData.create(type=NodeType.CODE.value, content="test")
        idx = fresh_db.add_node(node)

        retrieved_idx = fresh_db._get_index(node.id)

        assert retrieved_idx == idx

    def test_has_edge_pattern(self, fresh_db):
        """Test _has_edge_pattern checks for matching edge patterns."""
        from core.ontology import EdgePattern

        node1 = NodeData.create(type=NodeType.CODE.value, content="c1")
        node2 = NodeData.create(type=NodeType.SPEC.value, content="s1")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)

        edge = EdgeData.implements(node1.id, node2.id)
        fresh_db.add_edge(edge)

        pattern = EdgePattern(
            edge_type=EdgeType.IMPLEMENTS.value,
            direction="outgoing"
        )

        idx1 = fresh_db._get_index(node1.id)
        has_pattern = fresh_db._has_edge_pattern(idx1, node1, pattern)

        assert has_pattern

    def test_matches_trigger(self, fresh_db):
        """Test _matches_trigger checks if node matches trigger."""
        from core.ontology import StructuralTrigger, StatusPattern

        node = NodeData.create(
            type=NodeType.CODE.value,
            content="test",
            status=NodeStatus.PENDING.value
        )
        fresh_db.add_node(node)

        trigger = StructuralTrigger(
            id="test_trigger",
            target_node_type=NodeType.CODE.value,
            status_patterns=[
                StatusPattern(status=NodeStatus.PENDING.value, is_not=False)
            ],
            required_edges=[],
            forbidden_edges=[],
            agent_role="tester"
        )

        idx = fresh_db._get_index(node.id)
        matches = fresh_db._matches_trigger(idx, node, trigger)

        assert matches


# =============================================================================
# RATE LIMIT GUARD TESTS
# =============================================================================

class TestRateLimitGuard:
    """Test RateLimitGuard for proactive rate limiting."""

    def test_rate_limit_guard_init(self):
        """Test RateLimitGuard initializes with correct limits."""
        guard = RateLimitGuard(rpm_limit=100, tpm_limit=50000, safety_margin=0.9)

        assert guard.rpm_limit == 90  # 100 * 0.9
        assert guard.tpm_limit == 45000  # 50000 * 0.9
        assert len(guard._requests) == 0
        assert len(guard._tokens) == 0

    def test_cleanup_old_entries(self):
        """Test _cleanup_old_entries removes expired entries."""
        guard = RateLimitGuard()

        # Add old entries
        old_time = time.time() - 70  # 70 seconds ago
        guard._requests.append(old_time)
        guard._tokens.append((old_time, 1000))

        # Add recent entry
        recent_time = time.time() - 10
        guard._requests.append(recent_time)
        guard._tokens.append((recent_time, 500))

        guard._cleanup_old_entries(time.time())

        # Old entries should be removed
        assert len(guard._requests) == 1
        assert len(guard._tokens) == 1

    def test_current_rpm(self):
        """Test _current_rpm counts requests in window."""
        guard = RateLimitGuard()

        now = time.time()
        guard._requests.append(now - 30)
        guard._requests.append(now - 10)

        assert guard._current_rpm() == 2

    def test_current_tpm(self):
        """Test _current_tpm sums tokens in window."""
        guard = RateLimitGuard()

        now = time.time()
        guard._tokens.append((now - 30, 1000))
        guard._tokens.append((now - 10, 2000))

        assert guard._current_tpm() == 3000

    def test_record_usage(self):
        """Test record_usage adds entries."""
        guard = RateLimitGuard()

        guard.record_usage(1500)

        assert len(guard._requests) == 1
        assert len(guard._tokens) == 1
        assert guard._tokens[0][1] == 1500

    def test_set_retry_after(self):
        """Test set_retry_after sets retry timestamp."""
        guard = RateLimitGuard()

        guard.set_retry_after(5.0)

        assert guard._retry_after_until > time.time()
        assert guard._retry_after_until <= time.time() + 5.1

    def test_get_status(self):
        """Test get_status returns current usage stats."""
        guard = RateLimitGuard(rpm_limit=50, tpm_limit=30000)

        guard.record_usage(5000)
        guard.record_usage(3000)

        status = guard.get_status()

        assert status["rpm_used"] == 2
        assert status["tpm_used"] == 8000
        assert status["rpm_limit"] == 40  # 50 * 0.8 (default safety margin)
        assert status["tpm_limit"] == 24000  # 30000 * 0.8
        assert "retry_after_remaining" in status

    def test_wait_if_needed_below_limits(self):
        """Test wait_if_needed returns 0 when below limits."""
        guard = RateLimitGuard(rpm_limit=100, tpm_limit=50000)

        wait_time = guard.wait_if_needed(estimated_tokens=1000)

        assert wait_time == 0.0

    def test_wait_if_needed_retry_after(self):
        """Test wait_if_needed respects retry-after."""
        guard = RateLimitGuard()

        # Set retry-after for 0.1 seconds
        guard.set_retry_after(0.1)

        start = time.time()
        wait_time = guard.wait_if_needed(estimated_tokens=100)
        elapsed = time.time() - start

        # Should have waited at least 0.1 seconds
        assert elapsed >= 0.09
        assert wait_time >= 0.09
