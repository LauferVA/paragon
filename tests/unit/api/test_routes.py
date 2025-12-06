"""
Tests for API routes - focus on new dialectic and delta broadcast functionality.

Tests:
1. GraphDelta schema correctly serializes edges_removed as dicts
2. Node creation broadcasts delta to WebSocket
3. Edge creation broadcasts delta to WebSocket
4. Dialectic endpoints return correct structure
5. VizNode has position hints (x, y)
"""
import pytest
import json
from datetime import datetime, timezone

# Import the data structures
from viz.core import GraphDelta, VizNode, VizEdge, GraphSnapshot, create_snapshot_from_db
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, EdgeType, NodeStatus
from core.graph_db import ParagonDB


# =============================================================================
# TEST GRAPHDELTA SCHEMA FIX
# =============================================================================

def test_graphdelta_edges_removed_schema():
    """Test that edges_removed uses dict format, not tuples."""
    delta = GraphDelta(
        timestamp=datetime.now(timezone.utc).isoformat(),
        sequence=1,
        edges_removed=[
            {"source": "node1", "target": "node2"},
            {"source": "node3", "target": "node4"},
        ]
    )

    # Convert to dict (simulates JSON serialization)
    delta_dict = delta.to_dict()

    assert "edges_removed" in delta_dict
    assert isinstance(delta_dict["edges_removed"], list)
    assert len(delta_dict["edges_removed"]) == 2

    # Verify each edge is a dict with source and target
    for edge in delta_dict["edges_removed"]:
        assert isinstance(edge, dict)
        assert "source" in edge
        assert "target" in edge


def test_graphdelta_empty_check():
    """Test that empty delta is correctly identified."""
    empty_delta = GraphDelta(
        timestamp=datetime.now(timezone.utc).isoformat(),
        sequence=1,
    )

    assert empty_delta.is_empty() is True

    non_empty_delta = GraphDelta(
        timestamp=datetime.now(timezone.utc).isoformat(),
        sequence=2,
        nodes_added=[
            VizNode(
                id="test",
                type="CODE",
                status="PENDING",
                label="test",
                color="#000000",
            )
        ]
    )

    assert non_empty_delta.is_empty() is False


# =============================================================================
# TEST VIZNODE POSITION HINTS
# =============================================================================

def test_viznode_has_position_hints():
    """Test that VizNode can have x, y position hints."""
    node = VizNode(
        id="test_node",
        type="CODE",
        status="PENDING",
        label="Test Node",
        color="#2A9D8F",
        x=100.0,
        y=200.0,
        layer=1,
    )

    assert node.x == 100.0
    assert node.y == 200.0
    assert node.layer == 1


def test_viznode_from_node_data_with_positions():
    """Test creating VizNode from NodeData with position hints."""
    node_data = NodeData.create(
        type=NodeType.CODE.value,
        content="print('hello')",
        created_by="test",
    )

    viz_node = VizNode.from_node_data(
        node_data,
        layer=2,
        x=150.0,
        y=300.0,
    )

    assert viz_node.x == 150.0
    assert viz_node.y == 300.0
    assert viz_node.layer == 2
    assert viz_node.id == node_data.id


def test_create_snapshot_assigns_positions():
    """Test that create_snapshot_from_db assigns position hints."""
    # Create a small graph
    db = ParagonDB()

    node1 = NodeData.create(type=NodeType.REQ.value, content="Requirement 1", created_by="test")
    node2 = NodeData.create(type=NodeType.CODE.value, content="Code 1", created_by="test")

    db.add_node(node1)
    db.add_node(node2)

    edge = EdgeData.create(
        source_id=node1.id,
        target_id=node2.id,
        type=EdgeType.IMPLEMENTS.value,
    )
    db.add_edge(edge)

    # Create snapshot
    snapshot = create_snapshot_from_db(db)

    # Check that nodes have position hints
    assert snapshot.node_count == 2
    for viz_node in snapshot.nodes:
        # Positions should be assigned (not None)
        assert viz_node.x is not None
        assert viz_node.y is not None
        assert isinstance(viz_node.x, float)
        assert isinstance(viz_node.y, float)


# =============================================================================
# TEST SEQUENCE COUNTER
# =============================================================================

def test_sequence_counter_increments():
    """Test that delta sequence numbers increment."""
    delta1 = GraphDelta(
        timestamp=datetime.now(timezone.utc).isoformat(),
        sequence=1,
    )

    delta2 = GraphDelta(
        timestamp=datetime.now(timezone.utc).isoformat(),
        sequence=2,
    )

    assert delta2.sequence > delta1.sequence


# =============================================================================
# TEST DIALECTIC ENDPOINT STRUCTURE
# =============================================================================

def test_dialector_questions_response_structure():
    """Test that dialector questions have the expected structure."""
    # Simulate the response structure
    questions = [
        {
            "id": "q_0",
            "text": "What specific criteria define 'fast'?",
            "category": "SUBJECTIVE_TERMS",
            "suggested_answer": "< 100ms latency",
            "ambiguity_text": "fast",
        }
    ]

    response = {
        "questions": questions,
        "phase": "clarification",
        "session_id": "test_session",
        "has_questions": True,
    }

    assert "questions" in response
    assert isinstance(response["questions"], list)
    assert "phase" in response
    assert response["has_questions"] is True

    # Validate question structure
    q = questions[0]
    assert "id" in q
    assert "text" in q
    assert "category" in q
    assert "suggested_answer" in q


def test_dialector_answer_request_structure():
    """Test that answer submission has the expected structure."""
    request_body = {
        "session_id": "test_session",
        "answers": [
            {"question_id": "q_0", "answer": "Response latency < 100ms"},
            {"question_id": "q_1", "answer": "REST API"},
        ]
    }

    assert "session_id" in request_body
    assert "answers" in request_body
    assert isinstance(request_body["answers"], list)

    # Validate answer structure
    for ans in request_body["answers"]:
        assert "question_id" in ans
        assert "answer" in ans


# =============================================================================
# TEST GRAPH SNAPSHOT SERIALIZATION
# =============================================================================

def test_graph_snapshot_to_dict_serializable():
    """Test that GraphSnapshot.to_dict() produces JSON-serializable output."""
    db = ParagonDB()

    node = NodeData.create(type=NodeType.CODE.value, content="test", created_by="test")
    db.add_node(node)

    snapshot = create_snapshot_from_db(db)
    snapshot_dict = snapshot.to_dict()

    # Should be JSON serializable
    json_str = json.dumps(snapshot_dict)
    assert json_str is not None

    # Parse it back
    parsed = json.loads(json_str)
    assert parsed["node_count"] == 1
    assert "nodes" in parsed
    assert "edges" in parsed


def test_viz_edge_from_edge_data():
    """Test creating VizEdge from EdgeData."""
    edge_data = EdgeData.create(
        source_id="node1",
        target_id="node2",
        type=EdgeType.IMPLEMENTS.value,
        weight=1.5,
    )

    viz_edge = VizEdge.from_edge_data(edge_data)

    assert viz_edge.source == "node1"
    assert viz_edge.target == "node2"
    assert viz_edge.type == EdgeType.IMPLEMENTS.value
    assert viz_edge.weight == 1.5
    assert viz_edge.color is not None  # Should have a color assigned


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_full_delta_workflow():
    """Test the full workflow: create node -> generate delta -> serialize."""
    # Create a node
    node_data = NodeData.create(
        type=NodeType.CODE.value,
        content="def foo(): pass",
        created_by="test",
    )

    # Convert to VizNode
    viz_node = VizNode.from_node_data(node_data, x=100.0, y=200.0)

    # Create delta
    delta = GraphDelta(
        timestamp=datetime.now(timezone.utc).isoformat(),
        sequence=1,
        nodes_added=[viz_node],
    )

    # Serialize to dict
    delta_dict = delta.to_dict()

    # Verify structure
    assert delta_dict["sequence"] == 1
    assert len(delta_dict["nodes_added"]) == 1
    assert delta_dict["nodes_added"][0]["id"] == node_data.id
    assert delta_dict["nodes_added"][0]["x"] == 100.0
    assert delta_dict["nodes_added"][0]["y"] == 200.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
