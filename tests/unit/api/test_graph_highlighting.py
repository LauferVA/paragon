"""
UNIT TESTS: Graph Highlighting API

Tests for the dialogue-to-graph highlighting endpoints:
- POST /api/graph/highlight
- GET /api/nodes/{node_id}/reverse-connections
- GET /api/nodes/{node_id}/messages

Tests both the API endpoints and the underlying ParagonDB methods.
"""
import pytest
from starlette.testclient import TestClient

from core.graph_db import ParagonDB
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, NodeStatus, EdgeType
from api.routes import create_app


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def db():
    """Create a fresh test database with sample data."""
    db = ParagonDB()

    # Create some sample nodes
    req_node = NodeData.create(
        type=NodeType.REQ.value,
        content="Implement user authentication",
        created_by="user",
    )

    spec_node = NodeData.create(
        type=NodeType.SPEC.value,
        content="Add login endpoint with JWT",
        created_by="ARCHITECT",
    )

    code_node = NodeData.create(
        type=NodeType.CODE.value,
        content="def login(username, password): ...",
        created_by="BUILDER",
    )

    test_node = NodeData.create(
        type=NodeType.TEST_SUITE.value,
        content="test_login_success, test_login_failure",
        created_by="TESTER",
    )

    message_node = NodeData.create(
        type=NodeType.MESSAGE.value,
        content="Created login specification based on requirements",
        created_by="ARCHITECT",
    )

    # Add nodes to graph
    db.add_node(req_node)
    db.add_node(spec_node)
    db.add_node(code_node)
    db.add_node(test_node)
    db.add_node(message_node)

    # Add edges to create a dependency chain
    db.add_edge(EdgeData.create(
        source_id=spec_node.id,
        target_id=req_node.id,
        type=EdgeType.TRACES_TO.value,
    ))

    db.add_edge(EdgeData.create(
        source_id=code_node.id,
        target_id=spec_node.id,
        type=EdgeType.IMPLEMENTS.value,
    ))

    db.add_edge(EdgeData.create(
        source_id=test_node.id,
        target_id=code_node.id,
        type=EdgeType.TESTS.value,
    ))

    # Link message to spec node
    db.add_edge(EdgeData.create(
        source_id=message_node.id,
        target_id=spec_node.id,
        type=EdgeType.REFERENCES.value,
    ))

    return db, {
        "req": req_node,
        "spec": spec_node,
        "code": code_node,
        "test": test_node,
        "message": message_node,
    }


@pytest.fixture
def client(db):
    """Create a test client for the API."""
    # Inject the test database into the app
    from agents import tools
    db_instance, _ = db
    tools._db = db_instance

    app = create_app()
    return TestClient(app)


# =============================================================================
# PARAGONDB METHOD TESTS
# =============================================================================

def test_get_related_nodes_exact(db):
    """Test get_related_nodes with exact mode."""
    db_instance, nodes = db

    related = db_instance.get_related_nodes(nodes["spec"].id, mode="exact")

    assert len(related) == 1
    assert related[0] == nodes["spec"].id


def test_get_related_nodes_related(db):
    """Test get_related_nodes with related mode."""
    db_instance, nodes = db

    related = db_instance.get_related_nodes(nodes["spec"].id, mode="related")

    # Should include spec + req (TRACES_TO) + code (IMPLEMENTS)
    assert len(related) >= 2
    assert nodes["spec"].id in related
    assert nodes["req"].id in related  # Predecessor via TRACES_TO
    assert nodes["code"].id in related  # Successor via IMPLEMENTS


def test_get_related_nodes_dependent(db):
    """Test get_related_nodes with dependent mode."""
    db_instance, nodes = db

    related = db_instance.get_related_nodes(nodes["spec"].id, mode="dependent")

    # Should include spec + all descendants (code, test)
    assert len(related) >= 2
    assert nodes["spec"].id in related
    assert nodes["code"].id in related
    assert nodes["test"].id in related


def test_get_reverse_connections(db):
    """Test get_reverse_connections method."""
    db_instance, nodes = db

    connections = db_instance.get_reverse_connections(nodes["spec"].id)

    # Verify structure
    assert "node_id" in connections
    assert connections["node_id"] == nodes["spec"].id

    assert "referenced_in_dialogue" in connections
    assert "referenced_in_messages" in connections
    assert "incoming_edges" in connections
    assert "outgoing_edges" in connections

    # Should have message node referencing it
    assert len(connections["referenced_in_messages"]) >= 1
    assert nodes["message"].id in connections["referenced_in_messages"]

    # Should have incoming edge from message
    assert len(connections["incoming_edges"]) >= 1


def test_get_nodes_for_message(db):
    """Test get_nodes_for_message method."""
    db_instance, nodes = db

    referenced = db_instance.get_nodes_for_message(nodes["message"].id)

    # Message should reference spec node
    assert len(referenced) >= 1
    assert nodes["spec"].id in referenced


def test_get_nodes_for_message_non_message_node(db):
    """Test get_nodes_for_message with non-message node."""
    db_instance, nodes = db

    # Passing a non-MESSAGE node should return empty list
    referenced = db_instance.get_nodes_for_message(nodes["code"].id)
    assert len(referenced) == 0


# =============================================================================
# API ENDPOINT TESTS
# =============================================================================

def test_graph_highlight_node_exact(client, db):
    """Test POST /api/graph/highlight with node type and exact mode."""
    _, nodes = db

    response = client.post("/api/graph/highlight", json={
        "highlight_type": "node",
        "source_id": nodes["spec"].id,
        "highlight_mode": "exact",
    })

    assert response.status_code == 200
    data = response.json()

    assert "nodes_to_highlight" in data
    assert "edges_to_highlight" in data
    assert "context" in data

    # Exact mode should only return the node itself
    assert len(data["nodes_to_highlight"]) == 1
    assert nodes["spec"].id in data["nodes_to_highlight"]


def test_graph_highlight_node_related(client, db):
    """Test POST /api/graph/highlight with node type and related mode."""
    _, nodes = db

    response = client.post("/api/graph/highlight", json={
        "highlight_type": "node",
        "source_id": nodes["spec"].id,
        "highlight_mode": "related",
    })

    assert response.status_code == 200
    data = response.json()

    # Related mode should include connected nodes
    assert len(data["nodes_to_highlight"]) >= 2
    assert nodes["spec"].id in data["nodes_to_highlight"]

    # Should include edges between highlighted nodes
    assert len(data["edges_to_highlight"]) >= 0


def test_graph_highlight_message(client, db):
    """Test POST /api/graph/highlight with message type."""
    _, nodes = db

    response = client.post("/api/graph/highlight", json={
        "highlight_type": "message",
        "source_id": nodes["message"].id,
        "highlight_mode": "related",
    })

    assert response.status_code == 200
    data = response.json()

    # Should highlight nodes referenced by the message
    assert len(data["nodes_to_highlight"]) >= 1
    assert "Message" in data["context"]


def test_graph_highlight_edge(client, db):
    """Test POST /api/graph/highlight with edge type."""
    _, nodes = db

    edge_id = f"{nodes['code'].id}:{nodes['spec'].id}"

    response = client.post("/api/graph/highlight", json={
        "highlight_type": "edge",
        "source_id": edge_id,
        "highlight_mode": "exact",
    })

    assert response.status_code == 200
    data = response.json()

    # Should highlight both endpoints
    assert len(data["nodes_to_highlight"]) >= 2
    assert nodes["code"].id in data["nodes_to_highlight"]
    assert nodes["spec"].id in data["nodes_to_highlight"]

    # Should include the edge itself
    assert len(data["edges_to_highlight"]) >= 1


def test_graph_highlight_invalid_type(client, db):
    """Test POST /api/graph/highlight with invalid highlight_type."""
    _, nodes = db

    response = client.post("/api/graph/highlight", json={
        "highlight_type": "invalid",
        "source_id": nodes["spec"].id,
        "highlight_mode": "exact",
    })

    assert response.status_code == 400
    assert "error" in response.json()


def test_graph_highlight_missing_source_id(client):
    """Test POST /api/graph/highlight without source_id."""
    response = client.post("/api/graph/highlight", json={
        "highlight_type": "node",
        "highlight_mode": "exact",
    })

    assert response.status_code == 400
    assert "error" in response.json()


def test_graph_highlight_nonexistent_node(client):
    """Test POST /api/graph/highlight with nonexistent node."""
    response = client.post("/api/graph/highlight", json={
        "highlight_type": "node",
        "source_id": "nonexistent_id",
        "highlight_mode": "exact",
    })

    assert response.status_code == 404
    assert "error" in response.json()


def test_get_node_reverse_connections_endpoint(client, db):
    """Test GET /api/nodes/{node_id}/reverse-connections."""
    _, nodes = db

    response = client.get(f"/api/nodes/{nodes['spec'].id}/reverse-connections")

    assert response.status_code == 200
    data = response.json()

    assert "node_id" in data
    assert data["node_id"] == nodes["spec"].id

    assert "referenced_in_messages" in data
    assert "incoming_edges" in data
    assert "outgoing_edges" in data

    assert "last_modified_by" in data
    assert "last_modified_at" in data


def test_get_node_reverse_connections_nonexistent(client):
    """Test GET /api/nodes/{node_id}/reverse-connections with nonexistent node."""
    response = client.get("/api/nodes/nonexistent_id/reverse-connections")

    assert response.status_code == 404
    assert "error" in response.json()


def test_get_node_messages_endpoint(client, db):
    """Test GET /api/nodes/{node_id}/messages."""
    _, nodes = db

    response = client.get(f"/api/nodes/{nodes['spec'].id}/messages")

    assert response.status_code == 200
    data = response.json()

    assert "node_id" in data
    assert data["node_id"] == nodes["spec"].id

    assert "messages" in data
    assert "count" in data

    # Should have at least the message node we created
    assert data["count"] >= 1
    assert len(data["messages"]) >= 1

    # Verify message structure
    msg = data["messages"][0]
    assert "message_id" in msg
    assert "content" in msg
    assert "created_at" in msg
    assert "created_by" in msg


def test_get_node_messages_nonexistent(client):
    """Test GET /api/nodes/{node_id}/messages with nonexistent node."""
    response = client.get("/api/nodes/nonexistent_id/messages")

    assert response.status_code == 404
    assert "error" in response.json()


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_get_related_nodes_isolated_node(db):
    """Test get_related_nodes with an isolated node (no edges)."""
    db_instance, _ = db

    isolated = NodeData.create(
        type=NodeType.SPEC.value,
        content="Isolated spec",
        created_by="test",
    )
    db_instance.add_node(isolated)

    related = db_instance.get_related_nodes(isolated.id, mode="related")

    # Isolated node should only return itself
    assert len(related) == 1
    assert related[0] == isolated.id


def test_graph_highlight_edge_invalid_format(client, db):
    """Test POST /api/graph/highlight with edge in wrong format."""
    response = client.post("/api/graph/highlight", json={
        "highlight_type": "edge",
        "source_id": "invalid_format_no_colon",
        "highlight_mode": "exact",
    })

    assert response.status_code == 400
    assert "error" in response.json()


def test_get_reverse_connections_no_messages(db):
    """Test get_reverse_connections for node with no message references."""
    db_instance, nodes = db

    connections = db_instance.get_reverse_connections(nodes["test"].id)

    # Test node has no message references
    assert len(connections["referenced_in_messages"]) == 0

    # But should still have edges
    assert len(connections["incoming_edges"]) >= 0
    assert len(connections["outgoing_edges"]) >= 0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

def test_get_related_nodes_performance():
    """Test get_related_nodes performance on larger graph."""
    db = ParagonDB()

    # Create a chain of 100 nodes
    nodes = []
    for i in range(100):
        node = NodeData.create(
            type=NodeType.SPEC.value,
            content=f"Spec {i}",
            created_by="test",
        )
        db.add_node(node)
        nodes.append(node)

    # Connect them in a chain
    for i in range(len(nodes) - 1):
        db.add_edge(EdgeData.create(
            source_id=nodes[i+1].id,
            target_id=nodes[i].id,
            type=EdgeType.DEPENDS_ON.value,
        ))

    # Test performance of getting related nodes for middle node
    import time
    start = time.time()
    related = db.get_related_nodes(nodes[50].id, mode="related")
    elapsed = time.time() - start

    # Should complete in under 100ms for 100-node graph
    assert elapsed < 0.1

    # Should include immediate neighbors
    assert len(related) >= 1
    assert nodes[50].id in related


def test_get_reverse_connections_performance():
    """Test get_reverse_connections performance on larger graph."""
    db = ParagonDB()

    # Create target node
    target = NodeData.create(
        type=NodeType.CODE.value,
        content="Target code",
        created_by="test",
    )
    db.add_node(target)

    # Create 50 message nodes referencing it
    for i in range(50):
        msg = NodeData.create(
            type=NodeType.MESSAGE.value,
            content=f"Message {i}",
            created_by="test",
        )
        db.add_node(msg)
        db.add_edge(EdgeData.create(
            source_id=msg.id,
            target_id=target.id,
            type=EdgeType.REFERENCES.value,
        ))

    import time
    start = time.time()
    connections = db.get_reverse_connections(target.id)
    elapsed = time.time() - start

    # Should complete in under 100ms for 50 message nodes
    assert elapsed < 0.1

    # Should find all 50 messages
    assert len(connections["referenced_in_messages"]) == 50
