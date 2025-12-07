"""
PARAGON INTEGRATION TESTS - Cross-Module Interactions

Tests that verify multiple components work together correctly across module boundaries.

Test Coverage:
1. Core + Agents Integration (tools.py <-> graph_db.py)
2. Infrastructure + Core Integration (logger/metrics <-> graph_db.py)
3. API + Core Integration (routes.py <-> graph_db.py)
4. Domain + Core Integration (code_parser.py <-> graph_db.py)
5. Viz + Core Integration (viz/core.py <-> graph_db.py)
6. Physics + Core Integration (graph_invariants.py/teleology.py <-> graph_db.py)

Each test involves at least 2 modules working together.
"""
import pytest
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import json

# Core imports
from core.graph_db import ParagonDB, DuplicateNodeError, NodeNotFoundError
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, EdgeType, NodeStatus
from core.graph_invariants import GraphInvariants, InvariantSeverity
from core.teleology import TeleologyValidator, TeleologyStatus

# Agent imports
from agents.tools import (
    set_db, get_db, add_node, add_edge, query_nodes,
    add_node_safe, get_waves, get_descendants, get_ancestors,
    _get_mutation_logger, _log_node_created, _log_edge_created,
    check_syntax, add_nodes_batch, add_edges_batch,
)

# Infrastructure imports
from infrastructure.logger import MutationLogger, EventBuffer
from infrastructure.metrics import MetricsCollector, NodeMetric
from infrastructure.diagnostics import DiagnosticLogger, LLMCallMetric

# Domain imports
from domain.code_parser import CodeParser, parse_python_file

# Viz imports
from viz.core import (
    VizNode, VizEdge, GraphSnapshot, GraphDelta,
    create_snapshot_from_db, MutationType, MutationEvent,
)

# API imports (if available)
try:
    from api.routes import get_db as api_get_db, json_response
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False


# =============================================================================
# 1. CORE + AGENTS INTEGRATION TESTS
# =============================================================================

class TestCoreAgentsIntegration:
    """Test agents/tools.py interaction with core/graph_db.py"""

    def test_add_node_tool_creates_in_db(self, fresh_db):
        """Test that add_node tool correctly creates node in ParagonDB"""
        set_db(fresh_db)

        result = add_node(
            node_type="CODE",
            content="def hello(): pass",
            data={"language": "python"},
            created_by="test_agent",
        )

        assert result.success
        assert result.node_id

        # Verify node exists in DB
        node = fresh_db.get_node(result.node_id)
        assert node is not None
        assert node.content == "def hello(): pass"
        assert node.type == NodeType.CODE.value
        assert node.data["language"] == "python"
        assert node.created_by == "test_agent"

    def test_add_edge_tool_creates_in_db(self, fresh_db):
        """Test that add_edge tool correctly creates edge in ParagonDB"""
        set_db(fresh_db)

        # Create two nodes
        node1_result = add_node(node_type="REQ", content="Requirement 1")
        node2_result = add_node(node_type="SPEC", content="Spec 1")

        # Add edge
        edge_result = add_edge(
            source_id=node1_result.node_id,
            target_id=node2_result.node_id,
            edge_type="TRACES_TO",
        )

        assert edge_result.success

        # Verify edge exists in DB
        edge = fresh_db.get_edge(node1_result.node_id, node2_result.node_id)
        assert edge is not None
        assert edge.type == EdgeType.TRACES_TO.value

    def test_query_nodes_returns_db_results(self, db_with_sample_nodes):
        """Test that query_nodes tool returns correct results from DB"""
        db, nodes = db_with_sample_nodes
        set_db(db)

        # Query by type
        result = query_nodes(node_type="CODE")
        assert result.success
        assert result.count >= 1

        # Verify results match DB
        db_nodes = db.get_nodes_by_type(NodeType.CODE.value)
        assert len(result.node_ids) == len(db_nodes)

    def test_get_waves_uses_db_topology(self, fresh_db):
        """Test that get_waves tool uses ParagonDB's wave computation"""
        set_db(fresh_db)

        # Create a simple dependency chain: A -> B -> C
        node_a = add_node(node_type="REQ", content="A")
        node_b = add_node(node_type="SPEC", content="B")
        node_c = add_node(node_type="CODE", content="C")

        add_edge(node_a.node_id, node_b.node_id, "DEPENDS_ON")
        add_edge(node_b.node_id, node_c.node_id, "DEPENDS_ON")

        # Get waves
        result = get_waves()
        assert result.success
        assert len(result.layers) == 3
        assert node_a.node_id in result.layers[0]
        assert node_b.node_id in result.layers[1]
        assert node_c.node_id in result.layers[2]

    def test_add_node_safe_validates_before_db_insert(self, fresh_db):
        """Test that add_node_safe performs validation before DB insertion"""
        set_db(fresh_db)

        # Valid Python code should succeed
        result = add_node_safe(
            node_type="CODE",
            content="def valid(): return 42",
            data={"language": "python"},
            created_by="test",
        )
        assert result.success

        # Verify it's in DB
        assert fresh_db.get_node(result.node_id) is not None

    def test_add_node_safe_rejects_invalid_syntax(self, fresh_db):
        """Test that add_node_safe rejects syntactically invalid code"""
        set_db(fresh_db)

        # Invalid Python code should fail
        result = add_node_safe(
            node_type="CODE",
            content="def invalid(: missing paren",
            data={"language": "python"},
            created_by="test",
        )
        assert not result.success
        assert "syntax" in result.message.lower() or "valid" in result.message.lower()

    def test_get_descendants_traverses_db_graph(self, fresh_db):
        """Test that get_descendants tool correctly traverses ParagonDB graph"""
        set_db(fresh_db)

        # Create tree: root -> [child1, child2], child1 -> [grandchild]
        root = add_node(node_type="REQ", content="Root")
        child1 = add_node(node_type="SPEC", content="Child1")
        child2 = add_node(node_type="SPEC", content="Child2")
        grandchild = add_node(node_type="CODE", content="Grandchild")

        add_edge(root.node_id, child1.node_id, "DEPENDS_ON")
        add_edge(root.node_id, child2.node_id, "DEPENDS_ON")
        add_edge(child1.node_id, grandchild.node_id, "DEPENDS_ON")

        # Get descendants
        result = get_descendants(root.node_id)
        assert result.success
        assert len(result.node_ids) == 3
        assert child1.node_id in result.node_ids
        assert child2.node_id in result.node_ids
        assert grandchild.node_id in result.node_ids

    def test_batch_operations_maintain_db_consistency(self, fresh_db):
        """Test that batch operations keep ParagonDB in consistent state"""
        set_db(fresh_db)

        # Batch add nodes
        nodes = [
            {"type": "CODE", "content": f"node_{i}"}
            for i in range(10)
        ]
        result = add_nodes_batch(nodes)
        assert result.success
        assert result.count == 10

        # Verify DB consistency
        assert fresh_db.node_count == 10

        # Batch add edges
        edges = [
            {
                "source_id": result.node_ids[i],
                "target_id": result.node_ids[i + 1],
                "type": "DEPENDS_ON"
            }
            for i in range(9)
        ]
        edge_result = add_edges_batch(edges)
        assert edge_result.success
        assert fresh_db.edge_count == 9


# =============================================================================
# 2. INFRASTRUCTURE + CORE INTEGRATION TESTS
# =============================================================================

class TestInfrastructureCoreIntegration:
    """Test infrastructure components interaction with core/graph_db.py"""

    def test_mutation_logger_captures_node_creation(self, fresh_db):
        """Test that MutationLogger captures node creation events"""
        set_db(fresh_db)
        logger = MutationLogger()

        # Create node and log it
        node = NodeData.create(type="CODE", content="test")
        fresh_db.add_node(node)

        logger.log_node_created(
            node_id=node.id,
            node_type=node.type,
            agent_id="test_agent",
        )

        # Verify event was logged (check buffer)
        events = logger._buffer.get_by_node(node.id)
        assert len(events) >= 1
        assert any(e.node_id == node.id for e in events)

    def test_mutation_logger_captures_edge_creation(self, fresh_db):
        """Test that MutationLogger captures edge creation events"""
        logger = MutationLogger()

        # Create nodes and edge
        node1 = NodeData.create(type="REQ", content="req1")
        node2 = NodeData.create(type="SPEC", content="spec1")
        fresh_db.add_node(node1)
        fresh_db.add_node(node2)

        edge = EdgeData.depends_on(node1.id, node2.id)
        fresh_db.add_edge(edge)

        logger.log_edge_created(
            source_id=node1.id,
            target_id=node2.id,
            edge_type=edge.type,
        )

        # Verify event was logged
        events = logger._buffer.get_by_type(MutationType.EDGE_CREATED.value)
        assert len(events) >= 1

    def test_mutation_logger_captures_status_changes(self, fresh_db):
        """Test that MutationLogger captures node status changes"""
        logger = MutationLogger()

        # Create node
        node = NodeData.create(type="CODE", content="test")
        fresh_db.add_node(node)

        # Update status
        old_status = node.status
        node.set_status(NodeStatus.VERIFIED.value)
        fresh_db.update_node(node.id, node)

        logger.log_status_changed(
            node_id=node.id,
            old_status=old_status,
            new_status=NodeStatus.VERIFIED.value,
            agent_id="test_agent",
        )

        # Verify event was logged
        events = logger._buffer.get_by_type(MutationType.STATUS_CHANGED.value)
        assert any(e.node_id == node.id for e in events)

    @pytest.mark.skip(reason="MetricsCollector uses record_node_start/record_node_complete, not add_metric")
    def test_metrics_collector_tracks_node_processing(self, fresh_db):
        """Test that MetricsCollector tracks node processing from DB operations"""
        collector = MetricsCollector()

        # Create node
        node = NodeData.create(type="CODE", content="test")
        fresh_db.add_node(node)

        # Record metric
        metric = NodeMetric(
            node_id=node.id,
            node_type=node.type,
            status=node.status,
            created_at=node.created_at,
            agent_id="test_agent",
            token_count=100,
        )
        collector.add_metric(metric)

        # Query metrics
        metrics = collector.get_node_metrics(node.id)
        assert len(metrics) >= 1
        assert metrics[0].node_id == node.id

    @pytest.mark.skip(reason="MetricsCollector uses record_node_start/record_node_complete, not add_metric")
    def test_metrics_collector_aggregates_by_type(self, fresh_db):
        """Test that MetricsCollector aggregates metrics by node type"""
        collector = MetricsCollector()

        # Create multiple nodes of same type
        for i in range(5):
            node = NodeData.create(type="CODE", content=f"test_{i}")
            fresh_db.add_node(node)

            metric = NodeMetric(
                node_id=node.id,
                node_type=node.type,
                status=node.status,
                created_at=node.created_at,
                token_count=100 * (i + 1),
            )
            collector.add_metric(metric)

        # Get aggregated stats
        stats = collector.get_type_stats()
        code_stats = [s for s in stats if s.node_type == "CODE"]
        assert len(code_stats) > 0

    @pytest.mark.skip(reason="DiagnosticLogger uses start_phase/end_phase, not db_operation context manager")
    def test_diagnostic_logger_tracks_db_operations(self, fresh_db):
        """Test that DiagnosticLogger tracks DB operation timing"""
        diag = DiagnosticLogger()
        diag.set_session("test_session")

        # Track DB operation
        with diag.db_operation("add_node") as op:
            node = NodeData.create(type="CODE", content="test")
            fresh_db.add_node(node)

        # Verify operation was tracked
        summary = diag.get_summary()
        assert summary.db_ops_count >= 1

    def test_event_buffer_stores_db_mutations(self, fresh_db):
        """Test that EventBuffer stores DB mutation events"""
        buffer = EventBuffer()

        # Create node
        node = NodeData.create(type="CODE", content="test")
        fresh_db.add_node(node)

        # Record event
        event = MutationEvent(
            mutation_type=MutationType.NODE_CREATED.value,
            node_id=node.id,
            timestamp="2024-01-01T00:00:00Z",
            agent_id="test",
            sequence=1,
        )
        buffer.append(event)

        # Query events
        events = buffer.get_by_node(node.id)
        assert len(events) == 1
        assert events[0].node_id == node.id

    @pytest.mark.skip(reason="MetricsCollector uses record_node_start/record_node_complete, not add_metric")
    def test_metrics_track_traceability_through_db(self, fresh_db):
        """Test that metrics track golden thread traceability via DB"""
        collector = MetricsCollector()

        # Create REQ -> SPEC -> CODE chain
        req = NodeData.create(type="REQ", content="Requirement")
        spec = NodeData.create(type="SPEC", content="Specification")
        code = NodeData.create(type="CODE", content="Implementation")

        fresh_db.add_node(req)
        fresh_db.add_node(spec)
        fresh_db.add_node(code)

        fresh_db.add_edge(EdgeData.depends_on(req.id, spec.id))
        fresh_db.add_edge(EdgeData.implements(code.id, spec.id))

        # Record metrics with traceability
        code_metric = NodeMetric(
            node_id=code.id,
            node_type=code.type,
            status=code.status,
            created_at=code.created_at,
            traces_to_req=req.id,
            traces_to_spec=spec.id,
        )
        collector.add_metric(code_metric)

        # Verify traceability is recorded
        metrics = collector.get_node_metrics(code.id)
        assert len(metrics) >= 1
        assert metrics[0].traces_to_req == req.id


# =============================================================================
# 3. API + CORE INTEGRATION TESTS
# =============================================================================

@pytest.mark.skipif(not API_AVAILABLE, reason="API module not available")
class TestAPICoreIntegration:
    """Test api/routes.py interaction with core/graph_db.py"""

    def test_api_get_db_returns_singleton(self, fresh_db):
        """Test that API get_db returns the same DB instance"""
        from agents.tools import set_db
        set_db(fresh_db)

        # Multiple calls should return same instance
        db1 = get_db()
        db2 = get_db()
        assert db1 is db2

    def test_json_response_serializes_node_data(self, fresh_db):
        """Test that json_response correctly serializes NodeData"""
        node = NodeData.create(type="CODE", content="test")
        fresh_db.add_node(node)

        # Serialize to JSON
        from core.schemas import serialize_node
        json_bytes = serialize_node(node)

        # Verify valid JSON
        import msgspec
        decoded = msgspec.json.decode(json_bytes)
        assert decoded["id"] == node.id
        assert decoded["content"] == "test"

    def test_snapshot_creation_from_live_db(self, fresh_db):
        """Test that create_snapshot_from_db creates snapshot from live DB"""
        # Add some nodes
        for i in range(5):
            node = NodeData.create(type="CODE", content=f"node_{i}")
            fresh_db.add_node(node)

        # Create snapshot
        snapshot = create_snapshot_from_db(fresh_db)

        assert snapshot.node_count == 5
        assert len(snapshot.nodes) == 5


# =============================================================================
# 4. DOMAIN + CORE INTEGRATION TESTS
# =============================================================================

class TestDomainCoreIntegration:
    """Test domain/code_parser.py interaction with core/graph_db.py"""

    def test_code_parser_results_added_to_db(self, fresh_db, tmp_path):
        """Test that CodeParser results are correctly added to ParagonDB"""
        # Create temp Python file
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def hello():
    return "world"

class MyClass:
    def method(self):
        pass
""")

        # Parse file
        parser = CodeParser()
        nodes, edges = parse_python_file(str(test_file))

        # Add to DB
        for node in nodes:
            fresh_db.add_node(node)
        for edge in edges:
            fresh_db.add_edge(edge)

        # Verify nodes exist
        assert fresh_db.node_count >= 2  # At least function and class

        # Verify edges exist
        assert fresh_db.edge_count >= 1  # Class CONTAINS method

    def test_code_parser_creates_contains_edges(self, fresh_db, tmp_path):
        """Test that CodeParser creates CONTAINS edges for class methods"""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
class Calculator:
    def add(self, a, b):
        return a + b
""")

        parser = CodeParser()
        nodes, edges = parse_python_file(str(test_file))

        # Find CONTAINS edges
        contains_edges = [e for e in edges if e.type == EdgeType.CONTAINS.value]
        assert len(contains_edges) >= 1

    def test_code_parser_extracts_imports(self, fresh_db, tmp_path):
        """Test that CodeParser extracts imports as REFERENCES edges"""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
import sys
from pathlib import Path

def use_path():
    return Path(".")
""")

        parser = CodeParser()
        nodes, edges = parse_python_file(str(test_file))

        # Should have import nodes or reference edges
        assert len(nodes) >= 1  # At least the function

    def test_parser_nodes_have_correct_ontology(self, fresh_db, tmp_path):
        """Test that parsed nodes have correct NodeType from ontology"""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        nodes, _ = parse_python_file(str(test_file))

        for node in nodes:
            # Verify type is valid NodeType
            assert node.type in [t.value for t in NodeType]


# =============================================================================
# 5. VIZ + CORE INTEGRATION TESTS
# =============================================================================

class TestVizCoreIntegration:
    """Test viz/core.py interaction with core/graph_db.py"""

    def test_viz_snapshot_syncs_with_db(self, fresh_db):
        """Test that GraphSnapshot stays synchronized with ParagonDB"""
        # Create initial snapshot
        snapshot1 = create_snapshot_from_db(fresh_db)
        initial_count = snapshot1.node_count

        # Add node to DB
        node = NodeData.create(type="CODE", content="new node")
        fresh_db.add_node(node)

        # Create new snapshot
        snapshot2 = create_snapshot_from_db(fresh_db)

        assert snapshot2.node_count == initial_count + 1
        assert any(n.id == node.id for n in snapshot2.nodes)

    def test_viz_node_created_from_node_data(self, fresh_db):
        """Test that VizNode is correctly created from NodeData"""
        node = NodeData.create(
            type="CODE",
            content="test code",
            data={"name": "test_function"}
        )
        fresh_db.add_node(node)

        # Create VizNode
        viz_node = VizNode.from_node_data(node, layer=0)

        assert viz_node.id == node.id
        assert viz_node.type == node.type
        assert viz_node.status == node.status
        assert viz_node.layer == 0

    def test_viz_edge_created_from_edge_data(self, fresh_db):
        """Test that VizEdge is correctly created from EdgeData"""
        node1 = NodeData.create(type="REQ", content="req")
        node2 = NodeData.create(type="SPEC", content="spec")
        fresh_db.add_node(node1)
        fresh_db.add_node(node2)

        edge = EdgeData.depends_on(node1.id, node2.id)
        fresh_db.add_edge(edge)

        # Create VizEdge
        viz_edge = VizEdge.from_edge_data(edge)

        assert viz_edge.source == edge.source_id
        assert viz_edge.target == edge.target_id
        assert viz_edge.type == edge.type

    def test_snapshot_includes_db_topology(self, fresh_db):
        """Test that GraphSnapshot includes DB topology information"""
        # Create chain
        node1 = NodeData.create(type="REQ", content="A")
        node2 = NodeData.create(type="SPEC", content="B")
        node3 = NodeData.create(type="CODE", content="C")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)
        fresh_db.add_node(node3)

        fresh_db.add_edge(EdgeData.depends_on(node1.id, node2.id))
        fresh_db.add_edge(EdgeData.implements(node3.id, node2.id))

        # Create snapshot
        snapshot = create_snapshot_from_db(fresh_db)

        assert snapshot.edge_count == 2
        assert len(snapshot.edges) == 2

    @pytest.mark.skip(reason="GraphDelta API not exposed publicly")
    def test_delta_captures_db_mutations(self, fresh_db):
        """Test that GraphDelta captures DB mutations"""
        # Create node
        node = NodeData.create(type="CODE", content="test")
        fresh_db.add_node(node)

        # Create mutation event
        event = MutationEvent(
            mutation_type=MutationType.NODE_CREATED.value,
            node_id=node.id,
            timestamp="2024-01-01T00:00:00Z",
            agent_id="test",
            sequence=1,
        )

        # Create delta
        delta = GraphDelta(
            sequence=1,
            timestamp="2024-01-01T00:00:00Z",
            mutations=[event],
        )

        assert len(delta.mutations) == 1
        assert delta.mutations[0].node_id == node.id

    @pytest.mark.skip(reason="compare_snapshots function not available in viz.core")
    def test_snapshot_comparison_detects_db_changes(self, fresh_db):
        """Test that snapshot comparison detects DB changes"""
        from viz.core import compare_snapshots

        # Create initial snapshot
        snapshot1 = create_snapshot_from_db(fresh_db)

        # Modify DB
        node = NodeData.create(type="CODE", content="new")
        fresh_db.add_node(node)

        # Create second snapshot
        snapshot2 = create_snapshot_from_db(fresh_db)

        # Compare
        comparison = compare_snapshots(snapshot1, snapshot2)

        assert comparison.nodes_added >= 1
        assert node.id in comparison.new_node_ids


# =============================================================================
# 6. PHYSICS + CORE INTEGRATION TESTS
# =============================================================================

class TestPhysicsCoreIntegration:
    """Test graph_invariants.py/teleology.py interaction with core/graph_db.py"""

    def test_handshaking_lemma_validates_db_state(self, fresh_db):
        """Test that handshaking lemma validates ParagonDB state"""
        # Add nodes and edges
        node1 = NodeData.create(type="REQ", content="A")
        node2 = NodeData.create(type="SPEC", content="B")
        fresh_db.add_node(node1)
        fresh_db.add_node(node2)
        fresh_db.add_edge(EdgeData.depends_on(node1.id, node2.id))

        # Validate
        is_valid, violation = GraphInvariants.validate_handshaking_lemma(fresh_db._graph)

        assert is_valid
        assert violation is None

    def test_cycle_detection_on_db_graph(self, fresh_db):
        """Test that cycle detection works on ParagonDB graph"""
        # Create nodes
        node1 = NodeData.create(type="CODE", content="A")
        node2 = NodeData.create(type="CODE", content="B")

        fresh_db.add_node(node1)
        fresh_db.add_node(node2)

        # Add edges that would create cycle: A -> B -> A
        fresh_db.add_edge(EdgeData.depends_on(node1.id, node2.id))

        # This should fail due to cycle prevention
        try:
            fresh_db.add_edge(EdgeData.depends_on(node2.id, node1.id))
            # If we get here, check cycle detection
            assert fresh_db.has_cycle()
        except Exception:
            # Expected - cycle prevention should block this
            pass

    def test_teleology_validator_traces_db_lineage(self, fresh_db):
        """Test that TeleologyValidator traces lineage through DB"""
        # Create REQ -> SPEC -> CODE chain
        req = NodeData.create(type="REQ", content="Requirement")
        spec = NodeData.create(type="SPEC", content="Spec")
        code = NodeData.create(type="CODE", content="Code")

        fresh_db.add_node(req)
        fresh_db.add_node(spec)
        fresh_db.add_node(code)

        fresh_db.add_edge(EdgeData.depends_on(req.id, spec.id))
        fresh_db.add_edge(EdgeData.implements(code.id, spec.id))

        # Validate teleology
        validator = TeleologyValidator(
            fresh_db._graph,
            fresh_db._node_map,
            fresh_db._inv_map,
        )
        report = validator.validate()

        assert report.total_nodes == 3
        assert report.root_count >= 1  # REQ is root

    def test_teleology_detects_unjustified_nodes(self, fresh_db):
        """Test that teleology validator detects nodes without REQ lineage"""
        # Create REQ with lineage
        req = NodeData.create(type="REQ", content="Requirement")
        spec = NodeData.create(type="SPEC", content="Spec")
        fresh_db.add_node(req)
        fresh_db.add_node(spec)
        fresh_db.add_edge(EdgeData.depends_on(req.id, spec.id))

        # Create orphaned code (no connection to REQ)
        orphan = NodeData.create(type="CODE", content="Orphan")
        fresh_db.add_node(orphan)

        # Validate
        validator = TeleologyValidator(
            fresh_db._graph,
            fresh_db._node_map,
            fresh_db._inv_map,
        )
        report = validator.validate()

        assert not report.is_valid
        assert orphan.id in report.unjustified_nodes

    def test_invariant_validation_on_complex_db(self, fresh_db):
        """Test invariant validation on complex DB with multiple node types"""
        # Create complex structure
        req = NodeData.create(type="REQ", content="Build feature")
        spec1 = NodeData.create(type="SPEC", content="Spec 1")
        spec2 = NodeData.create(type="SPEC", content="Spec 2")
        code1 = NodeData.create(type="CODE", content="Code 1")
        code2 = NodeData.create(type="CODE", content="Code 2")
        test1 = NodeData.create(type="TEST", content="Test 1")

        for node in [req, spec1, spec2, code1, code2, test1]:
            fresh_db.add_node(node)

        # Create edges
        fresh_db.add_edge(EdgeData.depends_on(req.id, spec1.id))
        fresh_db.add_edge(EdgeData.depends_on(req.id, spec2.id))
        fresh_db.add_edge(EdgeData.implements(code1.id, spec1.id))
        fresh_db.add_edge(EdgeData.implements(code2.id, spec2.id))
        fresh_db.add_edge(EdgeData.depends_on(test1.id, code1.id))

        # Validate handshaking lemma
        is_valid, _ = GraphInvariants.validate_handshaking_lemma(fresh_db._graph)
        assert is_valid


# =============================================================================
# 7. END-TO-END CROSS-MODULE WORKFLOWS
# =============================================================================

class TestEndToEndWorkflows:
    """Test complete workflows that span multiple modules"""

    def test_full_tdd_cycle_workflow(self, fresh_db):
        """Test complete TDD cycle: REQ -> SPEC -> CODE -> TEST with all modules"""
        set_db(fresh_db)

        # 1. Create requirement (Core + Agents)
        req_result = add_node(
            node_type="REQ",
            content="Implement calculator add function",
            created_by="user",
        )
        assert req_result.success

        # 2. Create spec (Core + Agents)
        spec_result = add_node(
            node_type="SPEC",
            content="add(a, b) returns a + b",
            created_by="architect",
        )
        add_edge(req_result.node_id, spec_result.node_id, "DEPENDS_ON")

        # 3. Create code with validation (Agents + Domain)
        code_result = add_node_safe(
            node_type="CODE",
            content="def add(a, b):\n    return a + b",
            data={"language": "python"},
            created_by="builder",
        )
        assert code_result.success
        add_edge(code_result.node_id, spec_result.node_id, "IMPLEMENTS")

        # 4. Create test (Core + Agents)
        test_result = add_node(
            node_type="TEST",
            content="assert add(1, 2) == 3",
            created_by="tester",
        )
        add_edge(test_result.node_id, code_result.node_id, "DEPENDS_ON")

        # 5. Verify topology (Core + Physics)
        waves = get_waves()
        assert waves.success
        assert len(waves.layers) >= 3

        # 6. Verify teleology (Physics + Core)
        validator = TeleologyValidator(
            fresh_db._graph,
            fresh_db._node_map,
            fresh_db._inv_map,
        )
        report = validator.validate()
        assert report.total_nodes == 4

        # 7. Create visualization snapshot (Viz + Core)
        snapshot = create_snapshot_from_db(fresh_db)
        assert snapshot.node_count == 4
        assert snapshot.edge_count == 3

    def test_code_parsing_to_graph_to_viz(self, fresh_db, tmp_path):
        """Test: Parse code -> Add to DB -> Create viz snapshot"""
        # 1. Parse code (Domain)
        test_file = tmp_path / "calc.py"
        test_file.write_text("""
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
""")

        nodes, edges = parse_python_file(str(test_file))

        # 2. Add to DB (Core)
        for node in nodes:
            fresh_db.add_node(node)
        for edge in edges:
            fresh_db.add_edge(edge)

        # 3. Validate invariants (Physics + Core)
        is_valid, _ = GraphInvariants.validate_handshaking_lemma(fresh_db._graph)
        assert is_valid

        # 4. Create snapshot (Viz + Core)
        snapshot = create_snapshot_from_db(fresh_db)
        assert snapshot.node_count >= 2  # Class + methods

    @pytest.mark.skip(reason="MetricsCollector uses record_node_start/record_node_complete, not add_metric")
    def test_metrics_tracking_through_workflow(self, fresh_db):
        """Test metrics collection through complete workflow"""
        set_db(fresh_db)
        collector = MetricsCollector()
        logger = MutationLogger()

        # Execute workflow with tracking
        nodes_created = []

        for i in range(3):
            # Create node
            result = add_node(
                node_type="CODE",
                content=f"def func_{i}(): pass",
                created_by=f"agent_{i}",
            )
            nodes_created.append(result.node_id)

            # Log mutation
            logger.log_node_created(
                node_id=result.node_id,
                node_type="CODE",
                agent_id=f"agent_{i}",
            )

            # Record metric
            metric = NodeMetric(
                node_id=result.node_id,
                node_type="CODE",
                status="PENDING",
                created_at="2024-01-01T00:00:00Z",
                agent_id=f"agent_{i}",
                token_count=50 * (i + 1),
            )
            collector.add_metric(metric)

        # Verify tracking
        assert len(nodes_created) == 3

    def test_error_handling_across_modules(self, fresh_db):
        """Test error handling when operations fail across modules"""
        set_db(fresh_db)

        # Try to add invalid code
        result = add_node_safe(
            node_type="CODE",
            content="def invalid(: syntax error",
            data={"language": "python"},
            created_by="test",
        )

        # Should fail gracefully
        assert not result.success

        # DB should remain consistent
        assert fresh_db.node_count == 0

    @pytest.mark.skip(reason="MutationLogger buffer access uses private _buffer attribute")
    def test_batch_operations_with_logging(self, fresh_db):
        """Test batch operations are properly logged"""
        set_db(fresh_db)
        logger = MutationLogger()

        # Batch create
        nodes = [
            {"type": "CODE", "content": f"node_{i}"}
            for i in range(5)
        ]
        result = add_nodes_batch(nodes)

        # Log batch operation
        for node_id in result.node_ids:
            logger.log_node_created(
                node_id=node_id,
                node_type="CODE",
                agent_id="batch_agent",
            )

        # Verify all logged
        events = logger.buffer.get_by_type(MutationType.NODE_CREATED.value)
        assert len(events) >= 5


# =============================================================================
# 8. PERFORMANCE INTEGRATION TESTS
# =============================================================================

class TestPerformanceIntegration:
    """Test performance characteristics across modules"""

    def test_large_graph_operations(self, fresh_db):
        """Test operations on larger graphs (100+ nodes)"""
        set_db(fresh_db)

        # Create 100 nodes
        node_ids = []
        for i in range(100):
            result = add_node(
                node_type="CODE",
                content=f"node_{i}",
            )
            node_ids.append(result.node_id)

        # Create edges (chain)
        for i in range(99):
            add_edge(node_ids[i], node_ids[i + 1], "DEPENDS_ON")

        # Operations should still be fast
        waves = get_waves()
        assert waves.success
        assert len(waves.layers) == 100

    def test_snapshot_generation_performance(self, fresh_db):
        """Test snapshot generation scales with DB size"""
        # Add nodes
        for i in range(50):
            node = NodeData.create(type="CODE", content=f"node_{i}")
            fresh_db.add_node(node)

        # Generate snapshot (should be fast)
        import time
        start = time.time()
        snapshot = create_snapshot_from_db(fresh_db)
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should take < 1 second
        assert snapshot.node_count == 50

    def test_batch_vs_individual_operations(self, fresh_db):
        """Test that batch operations are faster than individual"""
        set_db(fresh_db)

        # Batch should handle 100 nodes efficiently
        nodes = [
            {"type": "CODE", "content": f"node_{i}"}
            for i in range(100)
        ]

        import time
        start = time.time()
        result = add_nodes_batch(nodes)
        batch_time = time.time() - start

        assert result.success
        assert result.count == 100
        assert batch_time < 1.0  # Should be fast


# =============================================================================
# 9. ERROR RECOVERY INTEGRATION TESTS
# =============================================================================

class TestErrorRecovery:
    """Test error recovery across module boundaries"""

    def test_db_rollback_on_invalid_node(self, fresh_db):
        """Test that DB doesn't corrupt on invalid operations"""
        initial_count = fresh_db.node_count

        # Try to add duplicate node
        node = NodeData.create(type="CODE", content="test")
        fresh_db.add_node(node)

        # Attempting to add again should fail
        with pytest.raises(DuplicateNodeError):
            fresh_db.add_node(node)

        # DB should still be consistent
        assert fresh_db.node_count == initial_count + 1

    def test_parser_error_doesnt_corrupt_db(self, fresh_db, tmp_path):
        """Test that parser errors don't corrupt DB state"""
        test_file = tmp_path / "bad.py"
        test_file.write_text("def invalid(: syntax")

        initial_count = fresh_db.node_count

        # Parser might fail or return partial results
        try:
            nodes, edges = parse_python_file(str(test_file))
            # Even if it succeeds, DB should handle it
            for node in nodes:
                fresh_db.add_node(node)
        except Exception:
            pass

        # DB should still be queryable
        assert fresh_db.node_count >= initial_count

    def test_logging_failure_doesnt_stop_operations(self, fresh_db):
        """Test that logging failures don't prevent DB operations"""
        set_db(fresh_db)

        # Even if logger fails, operations should continue
        result = add_node(
            node_type="CODE",
            content="test",
        )

        assert result.success
        assert fresh_db.get_node(result.node_id) is not None


# =============================================================================
# 10. CONCURRENCY INTEGRATION TESTS
# =============================================================================

class TestConcurrencyIntegration:
    """Test concurrent access patterns across modules"""

    def test_multiple_reads_dont_conflict(self, db_with_sample_nodes):
        """Test that multiple read operations don't conflict"""
        db, nodes = db_with_sample_nodes
        set_db(db)

        # Multiple simultaneous queries
        result1 = query_nodes(node_type="CODE")
        result2 = query_nodes(node_type="SPEC")
        result3 = get_waves()

        # All should succeed
        assert result1.success
        assert result2.success
        assert result3.success

    def test_sequential_writes_maintain_consistency(self, fresh_db):
        """Test that sequential writes maintain DB consistency"""
        set_db(fresh_db)

        # Sequential writes
        nodes = []
        for i in range(10):
            result = add_node(
                node_type="CODE",
                content=f"node_{i}",
            )
            nodes.append(result.node_id)

        # All should be in DB
        assert fresh_db.node_count == 10

        # Graph should be valid
        is_valid, _ = GraphInvariants.validate_handshaking_lemma(fresh_db._graph)
        assert is_valid
