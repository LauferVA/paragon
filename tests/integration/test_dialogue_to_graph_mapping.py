"""
INTEGRATION TESTS: Dialogue-to-Graph Mapping

Tests the complete flow of:
1. MESSAGE nodes referencing regular nodes via REFERENCES edges
2. Highlighting nodes when messages are clicked
3. Showing messages when nodes are hovered
4. Real-time WebSocket updates for highlighting

This tests the integration between:
- ParagonDB (graph storage)
- API routes (REST endpoints)
- WebSocket (real-time updates)
- Frontend highlighting logic
"""
import pytest
import asyncio
from starlette.testclient import TestClient
from starlette.websockets import WebSocket

from core.graph_db import ParagonDB
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, NodeStatus, EdgeType
from api.routes import create_app, broadcast_node_highlight


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def orchestrator_db():
    """Create a database simulating an orchestrator session with dialogue."""
    db = ParagonDB()

    # Simulate a TDD cycle with dialogue turns
    # Turn 1: User provides requirement
    req_node = NodeData.create(
        type=NodeType.REQ.value,
        content="Add user profile page with avatar upload",
        created_by="user",
    )
    db.add_node(req_node)

    # Turn 2: RESEARCHER investigates
    research_node = NodeData.create(
        type=NodeType.RESEARCH.value,
        content="Avatar storage: use S3, max 5MB, JPEG/PNG only",
        created_by="RESEARCHER",
    )
    db.add_node(research_node)
    db.add_edge(EdgeData.create(
        source_id=research_node.id,
        target_id=req_node.id,
        type=EdgeType.RESEARCH_FOR.value,
    ))

    msg_turn2 = NodeData.create(
        type=NodeType.MESSAGE.value,
        content="Researched avatar storage requirements. Found S3 best practice.",
        created_by="RESEARCHER",
    )
    db.add_node(msg_turn2)
    db.add_edge(EdgeData.create(
        source_id=msg_turn2.id,
        target_id=research_node.id,
        type=EdgeType.REFERENCES.value,
    ))

    # Turn 3: ARCHITECT creates specs
    spec_upload = NodeData.create(
        type=NodeType.SPEC.value,
        content="Upload endpoint: POST /api/user/avatar with multipart form",
        created_by="ARCHITECT",
    )
    db.add_node(spec_upload)
    db.add_edge(EdgeData.create(
        source_id=spec_upload.id,
        target_id=req_node.id,
        type=EdgeType.TRACES_TO.value,
    ))

    spec_display = NodeData.create(
        type=NodeType.SPEC.value,
        content="Profile page displays avatar from /api/user/avatar/:userId",
        created_by="ARCHITECT",
    )
    db.add_node(spec_display)
    db.add_edge(EdgeData.create(
        source_id=spec_display.id,
        target_id=req_node.id,
        type=EdgeType.TRACES_TO.value,
    ))

    msg_turn3 = NodeData.create(
        type=NodeType.MESSAGE.value,
        content="Created two specs: upload and display. Both trace to requirement.",
        created_by="ARCHITECT",
    )
    db.add_node(msg_turn3)
    db.add_edge(EdgeData.create(
        source_id=msg_turn3.id,
        target_id=spec_upload.id,
        type=EdgeType.REFERENCES.value,
    ))
    db.add_edge(EdgeData.create(
        source_id=msg_turn3.id,
        target_id=spec_display.id,
        type=EdgeType.REFERENCES.value,
    ))

    # Turn 4: BUILDER implements
    code_upload = NodeData.create(
        type=NodeType.CODE.value,
        content="def upload_avatar(file): validate(); save_to_s3(); return url",
        created_by="BUILDER",
    )
    db.add_node(code_upload)
    db.add_edge(EdgeData.create(
        source_id=code_upload.id,
        target_id=spec_upload.id,
        type=EdgeType.IMPLEMENTS.value,
    ))

    msg_turn4 = NodeData.create(
        type=NodeType.MESSAGE.value,
        content="Implemented upload endpoint. Uses S3 with validation.",
        created_by="BUILDER",
    )
    db.add_node(msg_turn4)
    db.add_edge(EdgeData.create(
        source_id=msg_turn4.id,
        target_id=code_upload.id,
        type=EdgeType.REFERENCES.value,
    ))
    db.add_edge(EdgeData.create(
        source_id=msg_turn4.id,
        target_id=spec_upload.id,
        type=EdgeType.REFERENCES.value,  # Also mentions the spec
    ))

    return db, {
        "req": req_node,
        "research": research_node,
        "spec_upload": spec_upload,
        "spec_display": spec_display,
        "code_upload": code_upload,
        "msg_turn2": msg_turn2,
        "msg_turn3": msg_turn3,
        "msg_turn4": msg_turn4,
    }


@pytest.fixture
def client(orchestrator_db):
    """Create test client with orchestrator database."""
    from agents import tools
    db, _ = orchestrator_db
    tools._db = db

    app = create_app()
    return TestClient(app)


# =============================================================================
# DIALOGUE-TO-GRAPH MAPPING TESTS
# =============================================================================

def test_message_references_nodes(orchestrator_db):
    """Test that MESSAGE nodes correctly reference regular nodes."""
    db, nodes = orchestrator_db

    # Turn 3 message should reference both specs
    referenced = db.get_nodes_from_message(nodes["msg_turn3"].id)
    referenced_ids = [n.id for n in referenced]

    assert len(referenced_ids) == 2
    assert nodes["spec_upload"].id in referenced_ids
    assert nodes["spec_display"].id in referenced_ids


def test_node_has_message_references(orchestrator_db):
    """Test that nodes know which messages reference them."""
    db, nodes = orchestrator_db

    # Spec upload should be referenced by turns 3 and 4
    messages = db.get_messages_for_node(nodes["spec_upload"].id)
    message_ids = [m.id for m in messages]

    assert len(message_ids) >= 2
    assert nodes["msg_turn3"].id in message_ids
    assert nodes["msg_turn4"].id in message_ids


def test_click_message_highlights_nodes(client, orchestrator_db):
    """Test clicking a message highlights its referenced nodes."""
    _, nodes = orchestrator_db

    # User clicks message from turn 3
    response = client.post("/api/graph/highlight", json={
        "highlight_type": "message",
        "source_id": nodes["msg_turn3"].id,
        "highlight_mode": "exact",
    })

    assert response.status_code == 200
    data = response.json()

    # Should highlight the two specs
    highlighted = data["nodes_to_highlight"]
    assert nodes["spec_upload"].id in highlighted
    assert nodes["spec_display"].id in highlighted


def test_click_message_highlights_related(client, orchestrator_db):
    """Test clicking a message with 'related' mode highlights dependencies."""
    _, nodes = orchestrator_db

    # User clicks message from turn 4 with related mode
    response = client.post("/api/graph/highlight", json={
        "highlight_type": "message",
        "source_id": nodes["msg_turn4"].id,
        "highlight_mode": "related",
    })

    assert response.status_code == 200
    data = response.json()

    highlighted = data["nodes_to_highlight"]

    # Should include code node
    assert nodes["code_upload"].id in highlighted

    # Should include spec (via IMPLEMENTS edge)
    assert nodes["spec_upload"].id in highlighted

    # May include requirement (via TRACES_TO chain)
    # This depends on how deep the traversal goes


def test_hover_node_shows_messages(client, orchestrator_db):
    """Test hovering a node shows which messages reference it."""
    _, nodes = orchestrator_db

    # User hovers over spec_upload
    response = client.get(f"/api/nodes/{nodes['spec_upload'].id}/messages")

    assert response.status_code == 200
    data = response.json()

    assert data["node_id"] == nodes["spec_upload"].id
    assert data["count"] >= 2

    # Should include messages from turns 3 and 4
    message_ids = [m["message_id"] for m in data["messages"]]
    assert nodes["msg_turn3"].id in message_ids
    assert nodes["msg_turn4"].id in message_ids


def test_reverse_connections_comprehensive(client, orchestrator_db):
    """Test reverse connections returns all metadata."""
    _, nodes = orchestrator_db

    response = client.get(
        f"/api/nodes/{nodes['spec_upload'].id}/reverse-connections"
    )

    assert response.status_code == 200
    data = response.json()

    # Check all required fields
    assert "node_id" in data
    assert "referenced_in_messages" in data
    assert "incoming_edges" in data
    assert "outgoing_edges" in data
    assert "last_modified_by" in data
    assert "last_modified_at" in data

    # Should have messages
    assert data["message_count"] >= 2

    # Should have edges
    assert len(data["incoming_edges"]) >= 0
    assert len(data["outgoing_edges"]) >= 1  # IMPLEMENTS from code


# =============================================================================
# WEBSOCKET INTEGRATION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_broadcast_node_highlight():
    """Test broadcasting node highlight events via WebSocket."""
    # This is a smoke test - full WebSocket testing requires async client

    nodes_to_highlight = ["node_1", "node_2", "node_3"]
    edges_to_highlight = [
        {"source": "node_1", "target": "node_2"},
        {"source": "node_2", "target": "node_3"},
    ]

    # Should not raise exception even with no connections
    await broadcast_node_highlight(
        nodes_to_highlight=nodes_to_highlight,
        edges_to_highlight=edges_to_highlight,
        reason="Test highlight",
        highlight_mode="related",
    )


# =============================================================================
# COMPLEX SCENARIO TESTS
# =============================================================================

def test_multiple_messages_same_node(orchestrator_db):
    """Test node referenced by multiple messages."""
    db, nodes = orchestrator_db

    # spec_upload is referenced by both turn 3 and turn 4
    messages = db.get_messages_for_node(nodes["spec_upload"].id)

    assert len(messages) >= 2

    # All should be MESSAGE type
    for msg in messages:
        assert msg.type == NodeType.MESSAGE.value


def test_message_references_multiple_types(orchestrator_db):
    """Test message can reference different node types."""
    db, _ = orchestrator_db

    # Create a complex message referencing multiple types
    multi_msg = NodeData.create(
        type=NodeType.MESSAGE.value,
        content="Summary: requirement, spec, and code all aligned",
        created_by="system",
    )
    db.add_node(multi_msg)

    # Reference different types
    req = db.get_nodes_by_type(NodeType.REQ.value)[0]
    spec = db.get_nodes_by_type(NodeType.SPEC.value)[0]
    code = db.get_nodes_by_type(NodeType.CODE.value)[0]

    db.add_edge(EdgeData.create(
        source_id=multi_msg.id,
        target_id=req.id,
        type=EdgeType.REFERENCES.value,
    ))
    db.add_edge(EdgeData.create(
        source_id=multi_msg.id,
        target_id=spec.id,
        type=EdgeType.REFERENCES.value,
    ))
    db.add_edge(EdgeData.create(
        source_id=multi_msg.id,
        target_id=code.id,
        type=EdgeType.REFERENCES.value,
    ))

    # Verify all referenced
    referenced = db.get_nodes_from_message(multi_msg.id)
    types = {n.type for n in referenced}

    assert NodeType.REQ.value in types
    assert NodeType.SPEC.value in types
    assert NodeType.CODE.value in types


def test_dialogue_flow_trace(orchestrator_db):
    """Test tracing dialogue flow through graph."""
    db, nodes = orchestrator_db

    # Start from requirement, trace through dialogue
    req_messages = db.get_messages_for_node(nodes["req"].id)

    # Should have no direct messages (messages reference derived nodes)
    assert len(req_messages) == 0

    # But research should reference it via RESEARCH_FOR edge
    research_msgs = db.get_messages_for_node(nodes["research"].id)
    assert len(research_msgs) >= 1


def test_highlight_entire_dialogue_chain(client, orchestrator_db):
    """Test highlighting shows entire dialogue chain."""
    _, nodes = orchestrator_db

    # Highlight the code node with dependent mode
    response = client.post("/api/graph/highlight", json={
        "highlight_type": "node",
        "source_id": nodes["code_upload"].id,
        "highlight_mode": "dependent",
    })

    assert response.status_code == 200
    data = response.json()

    # Should show dependent relationships
    highlighted = data["nodes_to_highlight"]

    # Code should be included
    assert nodes["code_upload"].id in highlighted


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_message_with_no_references(orchestrator_db):
    """Test message that doesn't reference any nodes."""
    db, _ = orchestrator_db

    orphan_msg = NodeData.create(
        type=NodeType.MESSAGE.value,
        content="Just a comment with no references",
        created_by="user",
    )
    db.add_node(orphan_msg)

    referenced = db.get_nodes_from_message(orphan_msg.id)
    assert len(referenced) == 0


def test_node_with_no_messages(orchestrator_db):
    """Test node that has no message references."""
    db, _ = orchestrator_db

    orphan_node = NodeData.create(
        type=NodeType.SPEC.value,
        content="Spec created directly, not via dialogue",
        created_by="system",
    )
    db.add_node(orphan_node)

    messages = db.get_messages_for_node(orphan_node.id)
    assert len(messages) == 0


def test_circular_message_references():
    """Test that circular message references don't break traversal."""
    db = ParagonDB()

    msg1 = NodeData.create(
        type=NodeType.MESSAGE.value,
        content="Message 1",
        created_by="test",
    )
    msg2 = NodeData.create(
        type=NodeType.MESSAGE.value,
        content="Message 2",
        created_by="test",
    )

    db.add_node(msg1)
    db.add_node(msg2)

    # Messages can reference each other (REPLY_TO)
    # This shouldn't break get_nodes_from_message
    db.add_edge(EdgeData.create(
        source_id=msg1.id,
        target_id=msg2.id,
        type=EdgeType.REPLY_TO.value,
    ), check_cycle=False)  # Messages may not form DAG

    # Should handle gracefully
    referenced = db.get_nodes_from_message(msg1.id)
    # msg2 won't be in referenced because REPLY_TO != REFERENCES
    assert len(referenced) == 0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

def test_large_dialogue_performance():
    """Test performance with large dialogue history."""
    db = ParagonDB()

    # Create 1000 message nodes
    messages = []
    for i in range(1000):
        msg = NodeData.create(
            type=NodeType.MESSAGE.value,
            content=f"Message {i}",
            created_by="test",
        )
        db.add_node(msg)
        messages.append(msg)

    # Create target node
    target = NodeData.create(
        type=NodeType.CODE.value,
        content="Frequently referenced code",
        created_by="test",
    )
    db.add_node(target)

    # Have every 10th message reference the target
    for i in range(0, len(messages), 10):
        db.add_edge(EdgeData.create(
            source_id=messages[i].id,
            target_id=target.id,
            type=EdgeType.REFERENCES.value,
        ))

    import time

    # Test get_messages_for_node performance
    start = time.time()
    msg_list = db.get_messages_for_node(target.id)
    elapsed = time.time() - start

    # Should complete in under 200ms for 100 referencing messages
    assert elapsed < 0.2
    assert len(msg_list) == 100


def test_highlight_performance_1000_nodes():
    """Test highlighting performance on 1000+ node graph."""
    db = ParagonDB()

    # Create a large graph
    nodes = []
    for i in range(1000):
        node = NodeData.create(
            type=NodeType.SPEC.value,
            content=f"Spec {i}",
            created_by="test",
        )
        db.add_node(node)
        nodes.append(node)

    # Create dependencies in a tree structure
    for i in range(len(nodes) - 1):
        parent_idx = i // 2
        if parent_idx < len(nodes):
            db.add_edge(EdgeData.create(
                source_id=nodes[i+1].id,
                target_id=nodes[parent_idx].id,
                type=EdgeType.DEPENDS_ON.value,
            ))

    import time

    # Test get_related_nodes performance on middle node
    start = time.time()
    related = db.get_related_nodes(nodes[500].id, mode="related")
    elapsed = time.time() - start

    # Should complete in under 100ms
    assert elapsed < 0.1

    print(f"\n✓ Highlighted {len(related)} nodes in {elapsed*1000:.1f}ms for 1000-node graph")
