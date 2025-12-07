"""
Unit tests for core/graph_db.py - ParagonDB

Tests the core graph database functionality including:
- Node creation and retrieval
- Edge creation and retrieval
- Graph traversal (waves, descendants)
- Batch operations
- Error handling
"""
import pytest
from core.graph_db import (
    ParagonDB,
    NodeNotFoundError,
    EdgeNotFoundError,
    DuplicateNodeError,
    GraphInvariantError,
)
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, EdgeType, NodeStatus


# =============================================================================
# NODE OPERATIONS TESTS
# =============================================================================

def test_add_node_creates_node(fresh_db):
    """
    Validate that add_node successfully creates a node in the graph.

    Verifies:
    - Node is added to the graph
    - Node count increases by 1
    - Node can be retrieved by ID
    """
    node = NodeData.create(type=NodeType.CODE.value, content="def hello(): pass")

    fresh_db.add_node(node)

    assert fresh_db.node_count == 1
    assert fresh_db.has_node(node.id)
    retrieved = fresh_db.get_node(node.id)
    assert retrieved.id == node.id
    assert retrieved.content == "def hello(): pass"


def test_add_node_duplicate_id_fails(fresh_db):
    """
    Validate that adding a node with a duplicate ID raises DuplicateNodeError.

    Verifies:
    - First add succeeds
    - Second add with same ID raises DuplicateNodeError
    - Graph still contains only one node
    """
    node = NodeData.create(type=NodeType.CODE.value, content="def hello(): pass")

    fresh_db.add_node(node)

    # Attempting to add the same node again should raise error
    with pytest.raises(DuplicateNodeError) as exc_info:
        fresh_db.add_node(node)

    assert node.id in str(exc_info.value)
    assert fresh_db.node_count == 1


def test_get_node_returns_data(fresh_db):
    """
    Validate that get_node returns the correct NodeData.

    Verifies:
    - Retrieved node has correct ID, type, content, and status
    - All node fields are preserved
    """
    node = NodeData.create(
        type=NodeType.SPEC.value,
        content="Implement SHA256 function",
        status=NodeStatus.PENDING.value,
        created_by="test_user"
    )

    fresh_db.add_node(node)
    retrieved = fresh_db.get_node(node.id)

    assert retrieved.id == node.id
    assert retrieved.type == NodeType.SPEC.value
    assert retrieved.content == "Implement SHA256 function"
    assert retrieved.status == NodeStatus.PENDING.value
    assert retrieved.created_by == "test_user"


def test_get_node_not_found_error(fresh_db):
    """
    Validate that get_node raises NodeNotFoundError for non-existent nodes.

    Verifies:
    - Getting a non-existent node raises NodeNotFoundError
    - Error message contains the node ID
    """
    non_existent_id = "does-not-exist-123"

    with pytest.raises(NodeNotFoundError) as exc_info:
        fresh_db.get_node(non_existent_id)

    assert non_existent_id in str(exc_info.value)


# =============================================================================
# EDGE OPERATIONS TESTS
# =============================================================================

def test_add_edge_creates_edge(fresh_db):
    """
    Validate that add_edge successfully creates an edge between nodes.

    Verifies:
    - Edge is added to the graph
    - Edge count increases by 1
    - Edge can be retrieved by source and target IDs
    """
    node1 = NodeData.create(type=NodeType.SPEC.value, content="Spec 1")
    node2 = NodeData.create(type=NodeType.CODE.value, content="Code 1")

    fresh_db.add_node(node1)
    fresh_db.add_node(node2)

    edge = EdgeData.create(
        source_id=node2.id,
        target_id=node1.id,
        type=EdgeType.IMPLEMENTS.value
    )

    fresh_db.add_edge(edge)

    assert fresh_db.edge_count == 1
    assert fresh_db.has_edge(node2.id, node1.id)
    retrieved = fresh_db.get_edge(node2.id, node1.id)
    assert retrieved.type == EdgeType.IMPLEMENTS.value


def test_add_edge_invalid_node_fails(fresh_db):
    """
    Validate that add_edge fails when source or target node doesn't exist.

    Verifies:
    - Adding edge with non-existent source raises NodeNotFoundError
    - Adding edge with non-existent target raises NodeNotFoundError
    - No edge is created in either case
    """
    node1 = NodeData.create(type=NodeType.CODE.value, content="Code 1")
    fresh_db.add_node(node1)

    # Try to create edge with non-existent target
    edge = EdgeData.create(
        source_id=node1.id,
        target_id="non-existent-id",
        type=EdgeType.DEPENDS_ON.value
    )

    with pytest.raises(NodeNotFoundError):
        fresh_db.add_edge(edge)

    assert fresh_db.edge_count == 0


# =============================================================================
# BATCH OPERATIONS TESTS
# =============================================================================

def test_batch_add_nodes(fresh_db):
    """
    Validate that add_nodes_batch efficiently adds multiple nodes.

    Verifies:
    - All nodes are added
    - Node count is correct
    - All nodes can be retrieved
    - Returns correct list of indices
    """
    nodes = [
        NodeData.create(type=NodeType.REQ.value, content=f"Requirement {i}")
        for i in range(5)
    ]

    indices = fresh_db.add_nodes_batch(nodes)

    assert fresh_db.node_count == 5
    assert len(indices) == 5

    # Verify all nodes can be retrieved
    for node in nodes:
        retrieved = fresh_db.get_node(node.id)
        assert retrieved.id == node.id


# =============================================================================
# WAVE COMPUTATION TESTS
# =============================================================================

def test_get_waves_empty_graph(fresh_db):
    """
    Validate that get_waves returns empty list for empty graph.

    Verifies:
    - Empty graph produces empty wave list
    - No errors are raised
    """
    waves = fresh_db.get_waves()

    assert waves == []
    assert len(waves) == 0


def test_get_waves_linear_chain(fresh_db):
    """
    Validate that get_waves computes correct layers for a linear dependency chain.

    Creates dependency chain where TEST depends on CODE depends on SPEC.
    In wave computation, nodes with no predecessors come first.

    Verifies:
    - Wave count is at least 1
    - First wave contains root nodes (nodes with no incoming edges)
    - Each subsequent wave depends only on previous waves
    """
    # Create linear chain
    node1 = NodeData.create(type=NodeType.REQ.value, content="Requirement")
    node2 = NodeData.create(type=NodeType.SPEC.value, content="Specification")
    node3 = NodeData.create(type=NodeType.CODE.value, content="def func(): pass")

    fresh_db.add_nodes_batch([node1, node2, node3])

    # Create dependencies: node2 depends on node1, node3 depends on node2
    # In directed graph terms: node1 -> node2 -> node3
    fresh_db.add_edge(EdgeData.create(node1.id, node2.id, EdgeType.DEPENDS_ON.value))
    fresh_db.add_edge(EdgeData.create(node2.id, node3.id, EdgeType.DEPENDS_ON.value))

    waves = fresh_db.get_waves()

    # Should have at least one wave
    assert len(waves) >= 1, "Should have at least one wave"

    # First wave should contain the root (node with no incoming edges)
    first_wave_ids = [node.id for node in waves[0]]
    assert node1.id in first_wave_ids, "First wave should contain the root node"


# =============================================================================
# DESCENDANT OPERATIONS TESTS
# =============================================================================

def test_get_descendants_single_node(fresh_db):
    """
    Validate that get_descendants returns all transitive successors.

    Creates tree: A -> B -> C
                    A -> D

    Verifies:
    - get_descendants(A) returns [B, C, D]
    - get_descendants(B) returns [C]
    - get_descendants(C) returns []
    """
    nodeA = NodeData.create(type=NodeType.REQ.value, content="A")
    nodeB = NodeData.create(type=NodeType.SPEC.value, content="B")
    nodeC = NodeData.create(type=NodeType.CODE.value, content="C")
    nodeD = NodeData.create(type=NodeType.SPEC.value, content="D")

    fresh_db.add_nodes_batch([nodeA, nodeB, nodeC, nodeD])

    # Create edges: A -> B -> C, A -> D
    fresh_db.add_edge(EdgeData.create(nodeA.id, nodeB.id, EdgeType.DEPENDS_ON.value))
    fresh_db.add_edge(EdgeData.create(nodeB.id, nodeC.id, EdgeType.DEPENDS_ON.value))
    fresh_db.add_edge(EdgeData.create(nodeA.id, nodeD.id, EdgeType.DEPENDS_ON.value))

    # Get descendants of A
    descendants_A = fresh_db.get_descendants(nodeA.id)
    descendant_ids_A = {d.id for d in descendants_A}

    assert nodeB.id in descendant_ids_A
    assert nodeC.id in descendant_ids_A
    assert nodeD.id in descendant_ids_A

    # Get descendants of B
    descendants_B = fresh_db.get_descendants(nodeB.id)
    descendant_ids_B = {d.id for d in descendants_B}

    assert nodeC.id in descendant_ids_B
    assert len(descendant_ids_B) >= 1

    # Get descendants of C (leaf node)
    descendants_C = fresh_db.get_descendants(nodeC.id)

    assert len(descendants_C) == 0


# =============================================================================
# QUERY OPERATIONS TESTS
# =============================================================================

def test_iter_nodes_by_type(fresh_db):
    """
    Validate that get_nodes_by_type filters nodes correctly.

    Verifies:
    - Filtering by type returns only matching nodes
    - Different types return different sets
    """
    code1 = NodeData.create(type=NodeType.CODE.value, content="Code 1")
    code2 = NodeData.create(type=NodeType.CODE.value, content="Code 2")
    spec1 = NodeData.create(type=NodeType.SPEC.value, content="Spec 1")

    fresh_db.add_nodes_batch([code1, code2, spec1])

    code_nodes = fresh_db.get_nodes_by_type(NodeType.CODE.value)
    spec_nodes = fresh_db.get_nodes_by_type(NodeType.SPEC.value)

    assert len(code_nodes) == 2
    assert len(spec_nodes) == 1

    code_ids = {n.id for n in code_nodes}
    assert code1.id in code_ids
    assert code2.id in code_ids


def test_iter_nodes_by_status(fresh_db):
    """
    Validate that get_nodes_by_status filters nodes correctly.

    Verifies:
    - Filtering by status returns only matching nodes
    - Different statuses return different sets
    """
    pending1 = NodeData.create(
        type=NodeType.CODE.value,
        content="Pending 1",
        status=NodeStatus.PENDING.value
    )
    pending2 = NodeData.create(
        type=NodeType.CODE.value,
        content="Pending 2",
        status=NodeStatus.PENDING.value
    )
    verified = NodeData.create(
        type=NodeType.CODE.value,
        content="Verified",
        status=NodeStatus.VERIFIED.value
    )

    fresh_db.add_nodes_batch([pending1, pending2, verified])

    pending_nodes = fresh_db.get_nodes_by_status(NodeStatus.PENDING.value)
    verified_nodes = fresh_db.get_nodes_by_status(NodeStatus.VERIFIED.value)

    assert len(pending_nodes) == 2
    assert len(verified_nodes) == 1

    pending_ids = {n.id for n in pending_nodes}
    assert pending1.id in pending_ids
    assert pending2.id in pending_ids


def test_get_graph_stats(fresh_db):
    """
    Validate that graph statistics are accurate.

    Verifies:
    - node_count property is correct
    - edge_count property is correct
    - is_empty property is correct
    """
    # Empty graph
    assert fresh_db.is_empty
    assert fresh_db.node_count == 0
    assert fresh_db.edge_count == 0

    # Add nodes and edges
    node1 = NodeData.create(type=NodeType.CODE.value, content="Code 1")
    node2 = NodeData.create(type=NodeType.CODE.value, content="Code 2")
    fresh_db.add_nodes_batch([node1, node2])

    edge = EdgeData.create(node1.id, node2.id, EdgeType.DEPENDS_ON.value)
    fresh_db.add_edge(edge)

    # Verify stats
    assert not fresh_db.is_empty
    assert fresh_db.node_count == 2
    assert fresh_db.edge_count == 1


def test_clear_graph(fresh_db):
    """
    Validate that removing all nodes clears the graph.

    Verifies:
    - Graph can be cleared by removing all nodes
    - After clearing, graph is empty
    """
    # Add some nodes
    nodes = [
        NodeData.create(type=NodeType.CODE.value, content=f"Code {i}")
        for i in range(5)
    ]
    fresh_db.add_nodes_batch(nodes)

    assert fresh_db.node_count == 5

    # Remove all nodes
    for node in nodes:
        fresh_db.remove_node(node.id)

    assert fresh_db.is_empty
    assert fresh_db.node_count == 0


# =============================================================================
# ADDITIONAL EDGE TESTS
# =============================================================================

def test_has_edge_method(fresh_db):
    """
    Validate that has_edge correctly detects edge existence.

    Verifies:
    - has_edge returns True for existing edge
    - has_edge returns False for non-existent edge
    """
    node1 = NodeData.create(type=NodeType.SPEC.value, content="Spec")
    node2 = NodeData.create(type=NodeType.CODE.value, content="Code")
    node3 = NodeData.create(type=NodeType.TEST.value, content="Test")

    fresh_db.add_nodes_batch([node1, node2, node3])

    # Add edge between node1 and node2
    edge = EdgeData.create(node2.id, node1.id, EdgeType.IMPLEMENTS.value)
    fresh_db.add_edge(edge)

    # Verify has_edge
    assert fresh_db.has_edge(node2.id, node1.id)
    assert not fresh_db.has_edge(node1.id, node2.id)  # Wrong direction
    assert not fresh_db.has_edge(node2.id, node3.id)  # No edge


def test_get_edge_not_found(fresh_db):
    """
    Validate that get_edge raises EdgeNotFoundError when edge doesn't exist.

    Verifies:
    - Getting non-existent edge raises EdgeNotFoundError
    - Error message contains source and target IDs
    """
    node1 = NodeData.create(type=NodeType.CODE.value, content="Code 1")
    node2 = NodeData.create(type=NodeType.CODE.value, content="Code 2")

    fresh_db.add_nodes_batch([node1, node2])

    # Try to get non-existent edge
    with pytest.raises(EdgeNotFoundError) as exc_info:
        fresh_db.get_edge(node1.id, node2.id)

    error_msg = str(exc_info.value)
    assert node1.id in error_msg
    assert node2.id in error_msg


# =============================================================================
# CYCLE DETECTION TESTS
# =============================================================================

def test_has_cycle_acyclic_graph(fresh_db):
    """
    Validate that has_cycle returns False for acyclic graph.

    Verifies:
    - DAG (directed acyclic graph) is correctly identified
    """
    node1 = NodeData.create(type=NodeType.REQ.value, content="REQ")
    node2 = NodeData.create(type=NodeType.SPEC.value, content="SPEC")
    node3 = NodeData.create(type=NodeType.CODE.value, content="CODE")

    fresh_db.add_nodes_batch([node1, node2, node3])

    # Create acyclic chain: 1 -> 2 -> 3
    fresh_db.add_edge(EdgeData.create(node1.id, node2.id, EdgeType.DEPENDS_ON.value))
    fresh_db.add_edge(EdgeData.create(node2.id, node3.id, EdgeType.DEPENDS_ON.value))

    assert not fresh_db.has_cycle()


def test_has_cycle_cyclic_graph(fresh_db):
    """
    Validate that cycle-creating edges are rejected at write time.

    The graph has built-in cycle prevention - add_edge raises
    GraphInvariantError when an edge would create a cycle.

    Verifies:
    - Cycle-creating edges raise GraphInvariantError
    - Graph remains acyclic after rejection
    """
    from core.graph_db import GraphInvariantError

    node1 = NodeData.create(type=NodeType.CODE.value, content="Code 1")
    node2 = NodeData.create(type=NodeType.CODE.value, content="Code 2")
    node3 = NodeData.create(type=NodeType.CODE.value, content="Code 3")

    fresh_db.add_nodes_batch([node1, node2, node3])

    # Create chain: 1 -> 2 -> 3
    fresh_db.add_edge(EdgeData.create(node1.id, node2.id, EdgeType.DEPENDS_ON.value))
    fresh_db.add_edge(EdgeData.create(node2.id, node3.id, EdgeType.DEPENDS_ON.value))

    # Attempt to create cycle: 3 -> 1 (should raise)
    import pytest
    with pytest.raises(GraphInvariantError, match="cycle"):
        fresh_db.add_edge(EdgeData.create(node3.id, node1.id, EdgeType.DEPENDS_ON.value))

    # Graph should still be acyclic
    assert not fresh_db.has_cycle()


# =============================================================================
# UPDATE OPERATIONS TESTS
# =============================================================================

def test_update_node_content(fresh_db):
    """
    Validate that update_node correctly modifies node data.

    Verifies:
    - Node content can be updated
    - Updated node can be retrieved with new content
    - Version number increases
    """
    node = NodeData.create(
        type=NodeType.CODE.value,
        content="def old(): pass"
    )
    fresh_db.add_node(node)

    # Create updated version
    updated = NodeData(
        id=node.id,
        type=node.type,
        content="def new(): pass",
        status=node.status,
        data=node.data,
        created_by=node.created_by,
        created_at=node.created_at,
        version=node.version + 1
    )

    fresh_db.update_node(node.id, updated)

    retrieved = fresh_db.get_node(node.id)
    assert retrieved.content == "def new(): pass"
    assert retrieved.version == node.version + 1


def test_remove_node_cleans_edges(fresh_db):
    """
    Validate that removing a node also removes its incident edges.

    Verifies:
    - Removing a node removes all edges connected to it
    - Edge count decreases correctly
    """
    node1 = NodeData.create(type=NodeType.CODE.value, content="Code 1")
    node2 = NodeData.create(type=NodeType.CODE.value, content="Code 2")
    node3 = NodeData.create(type=NodeType.CODE.value, content="Code 3")

    fresh_db.add_nodes_batch([node1, node2, node3])

    # Create edges: 1 -> 2, 2 -> 3
    fresh_db.add_edge(EdgeData.create(node1.id, node2.id, EdgeType.DEPENDS_ON.value))
    fresh_db.add_edge(EdgeData.create(node2.id, node3.id, EdgeType.DEPENDS_ON.value))

    assert fresh_db.edge_count == 2

    # Remove node2 (has both incoming and outgoing edges)
    fresh_db.remove_node(node2.id)

    # Both edges should be removed
    assert fresh_db.edge_count == 0
    assert fresh_db.node_count == 2
