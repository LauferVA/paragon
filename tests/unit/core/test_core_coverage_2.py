"""
Unit tests for core module - Part 2 (second half of uncovered functions).

Covers:
- ResourceGuard: __init__, _monitor_loop, get_current_usage, get_signal, start, stop, wait_for_resources
- StructuredLLM: __init__, _build_system_prompt, _clean_response, _estimate_tokens, _get_schema_prompt, _record_diagnostic, generate_with_history
- TeleologyValidator: __init__, _find_path_to_req, _find_req_nodes, _is_orphaned, _reverse_bfs_from_roots, check_node, validate
- ParagonDB: save_arrow, save_parquet, to_polars_edges, to_polars_nodes, topological_sort, update_all_embeddings, update_node, update_node_embedding, validate_edge_topology, validate_topology
- Standalone functions from embeddings, analytics, alignment, ontology
"""

import pytest
import tempfile
import time
import threading
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
import msgspec
import rustworkx as rx
import polars as pl
import numpy as np

from core.resource_guard import (
    ResourceGuard, ResourceSignal, get_resource_guard,
    init_resource_guard, shutdown_resource_guard
)
from core.llm import (
    StructuredLLM, RateLimitGuard, ModelRouter, TaskType,
    get_llm, set_llm, reset_llm, get_rate_limit_guard, reset_rate_limit_guard,
    get_learning_manager
)
from core.teleology import (
    TeleologyValidator, TeleologyStatus, TeleologyResult,
    validate_teleology, find_unjustified_nodes, has_teleological_integrity
)
from core.graph_db import ParagonDB, create_empty_db, create_db_from_nodes
from core.schemas import NodeData, EdgeData
from core.ontology import (
    NodeType, EdgeType, NodeStatus, get_constraint, get_required_edges,
    validate_status_for_type, get_triggers_for_node_type
)

# Import analytics functions
from core.analytics import (
    compute_degree_centrality, find_hotspots, compute_betweenness_centrality,
    find_articulation_points, find_orphan_nodes, find_dead_code_candidates,
    compute_graph_density, compute_max_depth, compute_wave_count,
    get_graph_health_report, find_connected_components,
    find_strongly_connected_components, count_nodes_by_type,
    count_nodes_by_status, get_type_dependency_matrix,
    detect_graph_changes, compare_graph_snapshots
)

# Import embedding functions
from core.embeddings import (
    compute_embedding, compute_embeddings_batch, cosine_similarity,
    find_similar_embeddings, embedding_dimension, normalize_embedding,
    text_to_embedding_key, compute_embeddings_for_nodes,
    update_node_embedding, is_available, _get_model
)

# Import alignment functions
from core.alignment import (
    align_graphs, compute_similarity, find_matches, detect_refactoring
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def resource_config():
    """Test configuration for ResourceGuard."""
    return {
        "ram_threshold_percent": 90,
        "cpu_threshold_percent": 95,
        "sustained_duration_seconds": 1,  # Short for testing
        "poll_interval_seconds": 0.1,  # Fast polling for tests
    }


@pytest.fixture
def sample_db():
    """Create a sample database for testing."""
    db = ParagonDB()

    # Create some test nodes
    req = NodeData.create(type=NodeType.REQ.value, content="Test requirement")
    spec = NodeData.create(type=NodeType.SPEC.value, content="Test spec")
    code = NodeData.create(type=NodeType.CODE.value, content="def test(): pass")

    db.add_node(req)
    db.add_node(spec)
    db.add_node(code)

    # Create edges
    db.add_edge(EdgeData.traces_to(spec.id, req.id))
    db.add_edge(EdgeData.implements(code.id, spec.id))

    return db


@pytest.fixture
def simple_schema():
    """Simple msgspec schema for testing."""
    class TestOutput(msgspec.Struct):
        name: str
        value: int
    return TestOutput


# =============================================================================
# RESOURCEGUARD TESTS
# =============================================================================

class TestResourceGuard:
    """Tests for ResourceGuard class."""

    def test_init(self, resource_config):
        """Test ResourceGuard initialization."""
        guard = ResourceGuard(resource_config)

        assert guard.ram_threshold == 90
        assert guard.cpu_threshold == 95
        assert guard.sustained_duration == 1
        assert guard.poll_interval == 0.1
        assert guard._signal == ResourceSignal.OK
        assert guard._monitor_thread is None
        assert guard._violation_start_time is None

    def test_start_stop(self, resource_config):
        """Test starting and stopping the monitor thread."""
        guard = ResourceGuard(resource_config)

        guard.start()
        assert guard._monitor_thread is not None
        assert guard._monitor_thread.is_alive()

        guard.stop()
        time.sleep(0.2)
        assert not guard._monitor_thread.is_alive()

    def test_get_signal(self, resource_config):
        """Test getting the current signal."""
        guard = ResourceGuard(resource_config)
        assert guard.get_signal() == ResourceSignal.OK

    def test_get_current_usage(self, resource_config):
        """Test getting current resource usage."""
        guard = ResourceGuard(resource_config)
        usage = guard.get_current_usage()

        assert "ram_percent" in usage
        assert "cpu_percent" in usage
        assert usage["ram_percent"] >= 0
        assert usage["cpu_percent"] >= 0

    def test_wait_for_resources_immediate(self, resource_config):
        """Test wait_for_resources when resources are available."""
        guard = ResourceGuard(resource_config)

        # Should return immediately with True
        result = guard.wait_for_resources(timeout=1)
        assert result is True

    def test_wait_for_resources_timeout(self, resource_config):
        """Test wait_for_resources with timeout."""
        guard = ResourceGuard(resource_config)
        guard._signal = ResourceSignal.PAUSE
        guard._resume_event.clear()

        # Should timeout
        start = time.time()
        result = guard.wait_for_resources(timeout=0.2)
        elapsed = time.time() - start

        assert result is False
        assert elapsed >= 0.2

    @patch('core.resource_guard.psutil.virtual_memory')
    @patch('core.resource_guard.psutil.cpu_percent')
    def test_monitor_loop_pause_trigger(self, mock_cpu, mock_mem, resource_config):
        """Test that monitor loop triggers PAUSE signal."""
        # Mock high resource usage
        mock_mem.return_value = Mock(percent=95.0)
        mock_cpu.return_value = 98.0

        guard = ResourceGuard(resource_config)
        guard.start()

        # Wait for sustained duration + buffer
        time.sleep(1.5)

        signal = guard.get_signal()
        guard.stop()

        # Should have triggered PAUSE
        assert signal == ResourceSignal.PAUSE

    @patch('core.resource_guard.psutil.virtual_memory')
    @patch('core.resource_guard.psutil.cpu_percent')
    def test_monitor_loop_recovery(self, mock_cpu, mock_mem, resource_config):
        """Test that monitor loop recovers from PAUSE."""
        # Start with high usage
        mock_mem.return_value = Mock(percent=95.0)
        mock_cpu.return_value = 98.0

        guard = ResourceGuard(resource_config)
        guard.start()
        time.sleep(1.5)

        # Now drop to normal
        mock_mem.return_value = Mock(percent=50.0)
        mock_cpu.return_value = 20.0
        time.sleep(0.5)

        signal = guard.get_signal()
        guard.stop()

        # Should have recovered
        assert signal == ResourceSignal.OK


def test_global_resource_guard_singleton(resource_config):
    """Test global resource guard singleton functions."""
    # Clean slate
    shutdown_resource_guard()

    # Should be None initially
    assert get_resource_guard() is None

    # Initialize
    guard1 = init_resource_guard(resource_config)
    assert guard1 is not None

    # Get should return same instance
    guard2 = get_resource_guard()
    assert guard2 is guard1

    # Cleanup
    shutdown_resource_guard()
    assert get_resource_guard() is None


# =============================================================================
# STRUCTUREDLLM TESTS
# =============================================================================

class TestStructuredLLM:
    """Tests for StructuredLLM class."""

    def test_init(self):
        """Test StructuredLLM initialization."""
        llm = StructuredLLM(model="test-model", temperature=0.5, max_tokens=1000)

        assert llm.model == "test-model"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 1000

    def test_get_schema_prompt(self, simple_schema):
        """Test schema prompt generation."""
        llm = StructuredLLM()
        prompt = llm._get_schema_prompt(simple_schema)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Should contain field names
        assert "name" in prompt or "value" in prompt

    def test_build_system_prompt(self, simple_schema):
        """Test building system prompt with schema."""
        llm = StructuredLLM()
        base = "You are a helpful assistant."
        full_prompt = llm._build_system_prompt(base, simple_schema)

        assert base in full_prompt
        assert "OUTPUT CONTRACT" in full_prompt
        assert "JSON" in full_prompt

    def test_clean_response(self):
        """Test response cleaning."""
        llm = StructuredLLM()

        # Test markdown removal
        assert llm._clean_response("```json\n{}\n```") == "{}"
        assert llm._clean_response("```\n{}\n```") == "{}"
        assert llm._clean_response("  {}  ") == "{}"

    def test_estimate_tokens(self):
        """Test token estimation."""
        llm = StructuredLLM()

        system = "System prompt"
        user = "User prompt"
        tokens = llm._estimate_tokens(system, user)

        assert tokens > 0
        assert isinstance(tokens, int)

    @patch('core.llm.get_diagnostics')
    def test_record_diagnostic(self, mock_get_diagnostics):
        """Test diagnostic recording."""
        mock_dx = Mock()
        mock_get_diagnostics.return_value = mock_dx

        llm = StructuredLLM()
        llm._record_diagnostic(
            schema_name="TestSchema",
            start_time=time.time(),
            input_tokens=100,
            output_tokens=50,
            success=True,
            truncated=False,
            error=None
        )

        # Should have called diagnostics
        mock_dx.record_llm_call_simple.assert_called_once()

    @patch('core.llm.litellm.completion')
    def test_generate_with_history(self, mock_completion, simple_schema):
        """Test generation with message history."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"name": "test", "value": 42}'))]
        mock_response.usage = None
        mock_completion.return_value = mock_response

        llm = StructuredLLM()
        messages = [
            {"role": "user", "content": "Generate test data"}
        ]

        result = llm.generate_with_history(messages, simple_schema)

        assert result.name == "test"
        assert result.value == 42


class TestRateLimitGuard:
    """Tests for RateLimitGuard class."""

    def test_init(self):
        """Test RateLimitGuard initialization."""
        guard = RateLimitGuard(rpm_limit=50, tpm_limit=30000, safety_margin=0.8)

        assert guard.rpm_limit == 40  # 50 * 0.8
        assert guard.tpm_limit == 24000  # 30000 * 0.8

    def test_record_usage(self):
        """Test recording API usage."""
        guard = RateLimitGuard()
        guard.record_usage(1000)

        status = guard.get_status()
        assert status["rpm_used"] == 1
        assert status["tpm_used"] == 1000

    def test_wait_if_needed_no_wait(self):
        """Test wait_if_needed when under limits."""
        guard = RateLimitGuard(rpm_limit=50, tpm_limit=30000)

        wait_time = guard.wait_if_needed(estimated_tokens=1000)
        assert wait_time == 0.0

    def test_wait_if_needed_rpm_limit(self):
        """Test wait_if_needed when RPM limit approached."""
        guard = RateLimitGuard(rpm_limit=2, tpm_limit=30000)

        # Fill up requests
        for _ in range(2):
            guard.record_usage(100)

        # Should wait briefly (we can't test full 60s in unit test)
        # Just verify it doesn't crash
        wait_time = guard.wait_if_needed(estimated_tokens=100)
        assert wait_time >= 0


class TestModelRouter:
    """Tests for ModelRouter class."""

    def test_init(self):
        """Test ModelRouter initialization."""
        config = {
            "high_reasoning_model": "claude-sonnet-4-5",
            "mundane_model": "claude-haiku-4-5",
        }
        router = ModelRouter(config)

        assert router.high_reasoning_model == "claude-sonnet-4-5"
        assert router.mundane_model == "claude-haiku-4-5"

    def test_route_high_reasoning(self):
        """Test routing high reasoning tasks."""
        router = ModelRouter({})
        provider, model = router.route(TaskType.HIGH_REASONING)

        assert provider == "anthropic"
        assert "sonnet" in model.lower() or "4-5" in model

    def test_route_mundane(self):
        """Test routing mundane tasks."""
        router = ModelRouter({})
        provider, model = router.route(TaskType.MUNDANE)

        assert provider in ["anthropic", "ollama"]

    def test_get_model_for_task(self):
        """Test getting full model identifier."""
        router = ModelRouter({})
        model_str = router.get_model_for_task(TaskType.HIGH_REASONING)

        assert "/" in model_str or model_str.startswith("claude")


def test_global_llm_singleton():
    """Test global LLM singleton functions."""
    reset_llm()

    # Get should create instance
    llm1 = get_llm()
    assert llm1 is not None

    # Get again should return same
    llm2 = get_llm()
    assert llm2 is llm1

    # Set custom
    custom = StructuredLLM(model="custom")
    set_llm(custom)
    assert get_llm() is custom

    # Reset
    reset_llm()


def test_rate_limit_guard_singleton():
    """Test rate limit guard singleton."""
    reset_rate_limit_guard()

    guard1 = get_rate_limit_guard()
    assert guard1 is not None

    guard2 = get_rate_limit_guard()
    assert guard2 is guard1


def test_get_learning_manager():
    """Test learning manager accessor."""
    # Should not crash even if not available
    manager = get_learning_manager()
    # manager may be None if learning not installed


# =============================================================================
# TELEOLOGYVALIDATOR TESTS
# =============================================================================

class TestTeleologyValidator:
    """Tests for TeleologyValidator class."""

    def test_init(self, sample_db):
        """Test TeleologyValidator initialization."""
        validator = TeleologyValidator(
            sample_db._graph,
            sample_db._node_map,
            sample_db._inv_map
        )

        assert validator.graph is sample_db._graph
        assert validator.node_map is sample_db._node_map
        assert validator.inv_map is sample_db._inv_map

    def test_find_req_nodes(self, sample_db):
        """Test finding REQ nodes."""
        validator = TeleologyValidator(
            sample_db._graph,
            sample_db._node_map,
            sample_db._inv_map
        )

        req_indices = validator._find_req_nodes()
        assert len(req_indices) == 1  # One REQ node in sample

    def test_is_orphaned(self, sample_db):
        """Test orphan detection."""
        validator = TeleologyValidator(
            sample_db._graph,
            sample_db._node_map,
            sample_db._inv_map
        )

        # Add an orphan node
        orphan = NodeData.create(type=NodeType.CODE.value, content="orphan")
        sample_db.add_node(orphan)
        orphan_idx = sample_db._node_map[orphan.id]

        assert validator._is_orphaned(orphan_idx) is True

    def test_reverse_bfs_from_roots(self, sample_db):
        """Test reverse BFS from REQ nodes."""
        validator = TeleologyValidator(
            sample_db._graph,
            sample_db._node_map,
            sample_db._inv_map
        )

        req_indices = validator._find_req_nodes()
        justified, depth_map, nearest_req = validator._reverse_bfs_from_roots(req_indices)

        # All connected nodes should be justified
        assert len(justified) >= 2  # At least SPEC and CODE

    def test_find_path_to_req(self, sample_db):
        """Test finding path to REQ."""
        validator = TeleologyValidator(
            sample_db._graph,
            sample_db._node_map,
            sample_db._inv_map
        )

        # Get CODE node
        code_nodes = sample_db.get_nodes_by_type(NodeType.CODE.value)
        code_idx = sample_db._node_map[code_nodes[0].id]

        req_indices = validator._find_req_nodes()
        path = validator._find_path_to_req(code_idx, req_indices)

        assert len(path) > 0  # Should find a path

    def test_validate(self, sample_db):
        """Test full validation."""
        validator = TeleologyValidator(
            sample_db._graph,
            sample_db._node_map,
            sample_db._inv_map
        )

        report = validator.validate()

        assert report.total_nodes == 3
        assert report.root_count == 1
        assert report.justified_count >= 2
        assert report.unjustified_count == 0

    def test_check_node(self, sample_db):
        """Test checking single node."""
        validator = TeleologyValidator(
            sample_db._graph,
            sample_db._node_map,
            sample_db._inv_map
        )

        code_nodes = sample_db.get_nodes_by_type(NodeType.CODE.value)
        result = validator.check_node(code_nodes[0].id)

        assert result.status == TeleologyStatus.JUSTIFIED
        assert result.nearest_req is not None


def test_validate_teleology_convenience(sample_db):
    """Test validate_teleology convenience function."""
    report = validate_teleology(
        sample_db._graph,
        sample_db._node_map,
        sample_db._inv_map
    )

    assert report.total_nodes > 0
    assert isinstance(report.is_valid, bool)


def test_find_unjustified_nodes_convenience(sample_db):
    """Test find_unjustified_nodes convenience function."""
    unjustified = find_unjustified_nodes(
        sample_db._graph,
        sample_db._node_map,
        sample_db._inv_map
    )

    assert isinstance(unjustified, list)


def test_has_teleological_integrity_convenience(sample_db):
    """Test has_teleological_integrity convenience function."""
    result = has_teleological_integrity(
        sample_db._graph,
        sample_db._node_map,
        sample_db._inv_map
    )

    assert isinstance(result, bool)


# =============================================================================
# PARAGONDB ADDITIONAL TESTS
# =============================================================================

class TestParagonDBPersistence:
    """Tests for ParagonDB persistence methods."""

    def test_to_polars_nodes(self, sample_db):
        """Test converting nodes to Polars DataFrame."""
        df = sample_db.to_polars_nodes()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert "id" in df.columns
        assert "type" in df.columns
        assert "status" in df.columns

    def test_to_polars_edges(self, sample_db):
        """Test converting edges to Polars DataFrame."""
        df = sample_db.to_polars_edges()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert "source_id" in df.columns
        assert "target_id" in df.columns
        assert "type" in df.columns

    def test_save_parquet(self, sample_db):
        """Test saving to Parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test"
            sample_db.save_parquet(path)

            # Check files exist
            assert (Path(tmpdir) / "test.nodes.parquet").exists()
            assert (Path(tmpdir) / "test.edges.parquet").exists()

    def test_save_arrow(self, sample_db):
        """Test saving to Arrow IPC."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test"
            sample_db.save_arrow(path)

            # Check files exist
            assert (Path(tmpdir) / "test.nodes.arrow").exists()
            assert (Path(tmpdir) / "test.edges.arrow").exists()


class TestParagonDBTopology:
    """Tests for ParagonDB topology validation."""

    def test_topological_sort(self, sample_db):
        """Test topological sort."""
        sorted_nodes = sample_db.topological_sort()

        assert len(sorted_nodes) == 3
        # REQ should come before SPEC, SPEC before CODE (generally)

    def test_update_node(self, sample_db):
        """Test updating a node."""
        nodes = sample_db.get_all_nodes()
        node = nodes[0]

        # Update content
        updated = NodeData(
            id=node.id,
            type=node.type,
            status=node.status,
            content="Updated content",
            created_by=node.created_by,
            created_at=node.created_at,
            version=node.version + 1
        )

        sample_db.update_node(node.id, updated)

        retrieved = sample_db.get_node(node.id)
        assert retrieved.content == "Updated content"

    def test_update_node_embedding(self, sample_db):
        """Test updating node embedding."""
        nodes = sample_db.get_all_nodes()
        node_id = nodes[0].id

        # This may return False if embeddings not available
        result = sample_db.update_node_embedding(node_id)
        assert isinstance(result, bool)

    def test_update_all_embeddings(self, sample_db):
        """Test batch embedding update."""
        count = sample_db.update_all_embeddings(batch_size=10)
        assert isinstance(count, int)
        assert count >= 0

    def test_validate_topology(self, sample_db):
        """Test topology validation."""
        nodes = sample_db.get_all_nodes()

        for node in nodes:
            violations = sample_db.validate_topology(node.id, mode="soft")
            assert isinstance(violations, list)

    def test_validate_edge_topology(self, sample_db):
        """Test edge topology validation."""
        # Create a valid edge
        nodes = sample_db.get_all_nodes()
        if len(nodes) >= 2:
            edge = EdgeData.depends_on(nodes[0].id, nodes[1].id)
            violations = sample_db.validate_edge_topology(edge)
            assert isinstance(violations, list)


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

def test_create_empty_db():
    """Test creating empty database."""
    db = create_empty_db()
    assert db.node_count == 0
    assert db.edge_count == 0


def test_create_db_from_nodes():
    """Test creating database from nodes."""
    nodes = [
        NodeData.create(type=NodeType.REQ.value, content="req1"),
        NodeData.create(type=NodeType.SPEC.value, content="spec1"),
    ]
    edges = [
        EdgeData.traces_to(nodes[1].id, nodes[0].id)
    ]

    db = create_db_from_nodes(nodes, edges)

    assert db.node_count == 2
    assert db.edge_count == 1


# =============================================================================
# ANALYTICS TESTS
# =============================================================================

class TestAnalytics:
    """Tests for analytics functions."""

    def test_compute_degree_centrality(self, sample_db):
        """Test degree centrality computation."""
        reports = compute_degree_centrality(sample_db)

        assert len(reports) == 3
        assert all(hasattr(r, 'total_degree') for r in reports)

    def test_find_hotspots(self, sample_db):
        """Test finding hotspot nodes."""
        hotspots = find_hotspots(sample_db, threshold_percentile=50)

        assert isinstance(hotspots, list)

    def test_compute_betweenness_centrality(self, sample_db):
        """Test betweenness centrality."""
        result = compute_betweenness_centrality(sample_db)

        assert isinstance(result, dict)

    def test_find_articulation_points(self, sample_db):
        """Test finding articulation points."""
        points = find_articulation_points(sample_db)

        assert isinstance(points, list)

    def test_find_orphan_nodes(self, sample_db):
        """Test finding orphan nodes."""
        orphans = find_orphan_nodes(sample_db)

        assert isinstance(orphans, list)

    def test_find_dead_code_candidates(self, sample_db):
        """Test finding dead code candidates."""
        candidates = find_dead_code_candidates(sample_db)

        assert isinstance(candidates, list)

    def test_compute_graph_density(self, sample_db):
        """Test graph density computation."""
        density = compute_graph_density(sample_db)

        assert 0.0 <= density <= 1.0

    def test_compute_max_depth(self, sample_db):
        """Test max depth computation."""
        depth = compute_max_depth(sample_db)

        assert depth >= 0

    def test_compute_wave_count(self, sample_db):
        """Test wave count computation."""
        count = compute_wave_count(sample_db)

        assert count >= 0

    def test_get_graph_health_report(self, sample_db):
        """Test health report generation."""
        report = get_graph_health_report(sample_db)

        assert report.total_nodes == 3
        assert report.total_edges == 2
        assert report.is_dag is True

    def test_find_connected_components(self, sample_db):
        """Test finding connected components."""
        components = find_connected_components(sample_db)

        assert isinstance(components, list)
        assert len(components) >= 1

    def test_find_strongly_connected_components(self, sample_db):
        """Test finding strongly connected components."""
        sccs = find_strongly_connected_components(sample_db)

        assert isinstance(sccs, list)

    def test_count_nodes_by_type(self, sample_db):
        """Test counting nodes by type."""
        counts = count_nodes_by_type(sample_db)

        assert isinstance(counts, dict)
        assert NodeType.REQ.value in counts

    def test_count_nodes_by_status(self, sample_db):
        """Test counting nodes by status."""
        counts = count_nodes_by_status(sample_db)

        assert isinstance(counts, dict)

    def test_get_type_dependency_matrix(self, sample_db):
        """Test type dependency matrix."""
        matrix = get_type_dependency_matrix(sample_db)

        assert isinstance(matrix, dict)


def test_compare_graph_snapshots():
    """Test graph snapshot comparison."""
    nodes1 = [NodeData.create(type=NodeType.REQ.value, content="test")]
    edges1 = []
    nodes2 = [NodeData.create(type=NodeType.REQ.value, content="test")]
    edges2 = []

    score = compare_graph_snapshots(nodes1, edges1, nodes2, edges2)

    assert 0.0 <= score <= 1.0


# =============================================================================
# EMBEDDINGS TESTS
# =============================================================================

class TestEmbeddings:
    """Tests for embedding functions."""

    def test_is_available(self):
        """Test checking if embeddings are available."""
        result = is_available()
        assert isinstance(result, bool)

    def test_get_model(self):
        """Test lazy model loading."""
        model = _get_model()
        # May be None if not installed
        assert model is None or hasattr(model, 'encode')

    def test_compute_embedding(self):
        """Test computing single embedding."""
        emb = compute_embedding("test text")

        if emb is not None:
            assert isinstance(emb, list)
            assert len(emb) == 384

    def test_compute_embeddings_batch(self):
        """Test batch embedding computation."""
        texts = ["text1", "text2", "text3"]
        embs = compute_embeddings_batch(texts)

        assert len(embs) == 3

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]

        assert cosine_similarity(vec1, vec2) == pytest.approx(1.0)
        assert cosine_similarity(vec1, vec3) == pytest.approx(0.0)

    def test_cosine_similarity_edge_cases(self):
        """Test cosine similarity edge cases."""
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0  # Different lengths
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0  # Zero vectors

    def test_find_similar_embeddings(self):
        """Test finding similar embeddings."""
        query = [1.0, 0.0, 0.0]
        candidates = [
            ("id1", [1.0, 0.0, 0.0]),
            ("id2", [0.9, 0.1, 0.0]),
            ("id3", [0.0, 1.0, 0.0]),
        ]

        results = find_similar_embeddings(query, candidates, threshold=0.8, limit=2)

        assert len(results) <= 2
        if results:
            assert results[0][1] >= 0.8  # Score above threshold

    def test_embedding_dimension(self):
        """Test embedding dimension."""
        assert embedding_dimension() == 384

    def test_normalize_embedding(self):
        """Test embedding normalization."""
        vec = [3.0, 4.0]
        normalized = normalize_embedding(vec)

        # Should be unit length
        length = sum(x*x for x in normalized) ** 0.5
        assert length == pytest.approx(1.0)

    def test_normalize_embedding_edge_cases(self):
        """Test normalization edge cases."""
        assert normalize_embedding([]) == []
        assert normalize_embedding([0.0, 0.0]) == [0.0, 0.0]

    def test_text_to_embedding_key(self):
        """Test embedding key generation."""
        key1 = text_to_embedding_key("test text")
        key2 = text_to_embedding_key("test text")
        key3 = text_to_embedding_key("different")

        assert key1 == key2  # Same text = same key
        assert key1 != key3  # Different text = different key
        assert len(key1) == 16  # Hash length

    def test_compute_embeddings_for_nodes(self):
        """Test batch embedding for node dicts."""
        nodes = [
            {"id": "n1", "content": "text1"},
            {"id": "n2", "content": "text2"},
        ]

        result = compute_embeddings_for_nodes(nodes, content_field="content")

        assert isinstance(result, dict)

    def test_update_node_embedding_function(self):
        """Test node embedding update function."""
        node = NodeData.create(type=NodeType.CODE.value, content="test code")

        result = update_node_embedding(node)
        assert isinstance(result, bool)


# =============================================================================
# ALIGNMENT TESTS
# =============================================================================

class TestAlignment:
    """Tests for alignment functions."""

    def test_align_graphs_basic(self):
        """Test basic graph alignment."""
        nodes1 = [NodeData.create(type=NodeType.REQ.value, content="test1")]
        edges1 = []
        nodes2 = [NodeData.create(type=NodeType.REQ.value, content="test1")]
        edges2 = []

        result = align_graphs(nodes1, edges1, nodes2, edges2, algorithm="rrwm")

        assert hasattr(result, 'soft_matching')
        assert hasattr(result, 'hard_matching')
        assert hasattr(result, 'node_mapping')
        assert hasattr(result, 'score')

    def test_align_graphs_empty(self):
        """Test alignment with empty graphs."""
        result = align_graphs([], [], [], [])

        assert result.score == 0.0
        assert len(result.node_mapping) == 0

    def test_compute_similarity_alignment(self):
        """Test similarity computation."""
        nodes1 = [NodeData.create(type=NodeType.REQ.value, content="test")]
        edges1 = []
        nodes2 = [NodeData.create(type=NodeType.REQ.value, content="test")]
        edges2 = []

        score = compute_similarity(nodes1, edges1, nodes2, edges2)

        assert 0.0 <= score <= 1.0

    def test_find_matches_alignment(self):
        """Test finding matches between graphs."""
        nodes1 = [NodeData.create(type=NodeType.REQ.value, content="test")]
        edges1 = []
        nodes2 = [NodeData.create(type=NodeType.REQ.value, content="test")]
        edges2 = []

        matches = find_matches(nodes1, edges1, nodes2, edges2, min_score=0.3)

        assert isinstance(matches, list)

    def test_detect_refactoring_alignment(self):
        """Test refactoring detection."""
        old_nodes = [NodeData.create(type=NodeType.CODE.value, content="old")]
        old_edges = []
        new_nodes = [
            NodeData.create(type=NodeType.CODE.value, content="old"),
            NodeData.create(type=NodeType.CODE.value, content="new"),
        ]
        new_edges = []

        result = detect_refactoring(old_nodes, old_edges, new_nodes, new_edges)

        assert "renamed" in result
        assert "added" in result
        assert "removed" in result
        assert "unchanged" in result
        assert "similarity_score" in result


# =============================================================================
# ONTOLOGY TESTS
# =============================================================================

class TestOntologyFunctions:
    """Tests for ontology utility functions."""

    def test_get_constraint(self):
        """Test getting constraint for node type."""
        constraint = get_constraint(NodeType.REQ.value)

        assert constraint is not None
        assert constraint.node_type == NodeType.REQ.value

    def test_get_constraint_invalid(self):
        """Test getting constraint for invalid type."""
        constraint = get_constraint("INVALID_TYPE")

        assert constraint is None

    def test_get_required_edges_all(self):
        """Test getting all edge constraints."""
        edges = get_required_edges(NodeType.CODE.value, mode="all")

        assert isinstance(edges, list)

    def test_get_required_edges_hard(self):
        """Test getting hard constraints only."""
        edges = get_required_edges(NodeType.CODE.value, mode="hard")

        assert isinstance(edges, list)
        assert all(ec.mode == "hard" for ec in edges)

    def test_get_required_edges_soft(self):
        """Test getting soft constraints only."""
        edges = get_required_edges(NodeType.CODE.value, mode="soft")

        assert isinstance(edges, list)
        assert all(ec.mode == "soft" for ec in edges)

    def test_validate_status_for_type(self):
        """Test status validation."""
        # Valid status
        assert validate_status_for_type(
            NodeType.CODE.value,
            NodeStatus.PENDING.value
        ) is True

        # Invalid status (if constraints exist)
        # Note: Some types may allow all statuses
        result = validate_status_for_type(
            NodeType.CODE.value,
            "INVALID_STATUS"
        )
        assert isinstance(result, bool)

    def test_get_triggers_for_node_type(self):
        """Test getting triggers for node type."""
        triggers = get_triggers_for_node_type(NodeType.CODE.value)

        assert isinstance(triggers, list)
        assert all(t.target_node_type == NodeType.CODE.value for t in triggers)


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
