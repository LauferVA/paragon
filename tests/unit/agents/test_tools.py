"""
Unit tests for agents/tools.py - Agent Tool Functions

Tests the LangGraph tools for graph operations including:
- Node and edge creation tools
- Query and analysis tools
- Layer 7B auditor tools (syntax checking, safe node creation)
- Graph statistics and cycle detection
"""
import pytest
from agents.tools import (
    add_node,
    add_node_safe,
    add_edge,
    query_nodes,
    get_node,
    get_waves,
    get_descendants,
    check_syntax,
    get_graph_stats,
    check_cycle,
    get_db,
)
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, EdgeType, NodeStatus


# =============================================================================
# NODE TOOL TESTS
# =============================================================================

def test_add_node_tool(fresh_db):
    """
    Validate that add_node tool creates a node successfully.

    Verifies:
    - Tool returns success=True
    - Node ID is returned
    - Node can be retrieved from database
    - Message describes the operation
    """
    result = add_node(
        node_type=NodeType.CODE.value,
        content="def example(): pass",
        created_by="test_agent"
    )

    assert result.success is True
    assert result.node_id != ""
    assert "Created node" in result.message

    # Verify node exists in DB
    db = get_db()
    node = db.get_node(result.node_id)
    assert node.type == NodeType.CODE.value
    assert node.content == "def example(): pass"


def test_add_node_tool_with_metadata(fresh_db):
    """
    Validate that add_node tool preserves metadata.

    Verifies:
    - Metadata dictionary is stored in node.data
    - All metadata fields are preserved
    """
    metadata = {
        "filename": "example.py",
        "line_number": 42,
        "author": "test_user"
    }

    result = add_node(
        node_type=NodeType.CODE.value,
        content="def example(): pass",
        data=metadata
    )

    assert result.success is True

    db = get_db()
    node = db.get_node(result.node_id)
    assert node.data["filename"] == "example.py"
    assert node.data["line_number"] == 42


# =============================================================================
# EDGE TOOL TESTS
# =============================================================================

def test_add_edge_tool(fresh_db):
    """
    Validate that add_edge tool creates an edge successfully.

    Verifies:
    - Tool returns success=True
    - Edge is created between nodes
    - Message describes the operation
    """
    # Create two nodes first
    node1_result = add_node(NodeType.SPEC.value, "Specification")
    node2_result = add_node(NodeType.CODE.value, "Implementation")

    # Create edge
    result = add_edge(
        source_id=node2_result.node_id,
        target_id=node1_result.node_id,
        edge_type=EdgeType.IMPLEMENTS.value
    )

    assert result.success is True
    assert result.count == 1
    assert "->" in result.message

    # Verify edge exists
    db = get_db()
    assert db.has_edge(node2_result.node_id, node1_result.node_id)


def test_add_edge_tool_invalid_nodes(fresh_db):
    """
    Validate that add_edge tool fails gracefully with invalid node IDs.

    Verifies:
    - Tool returns success=False
    - No edge is created
    - Error message is informative
    """
    result = add_edge(
        source_id="non-existent-1",
        target_id="non-existent-2",
        edge_type=EdgeType.DEPENDS_ON.value
    )

    assert result.success is False
    assert "Failed" in result.message
    assert result.count == 0


# =============================================================================
# QUERY TOOL TESTS
# =============================================================================

def test_query_nodes_by_type(fresh_db):
    """
    Validate that query_nodes correctly filters by node type.

    Verifies:
    - Filtering returns only matching nodes
    - Count matches number of results
    - Empty result for non-existent type
    """
    # Add nodes of different types
    add_node(NodeType.CODE.value, "Code 1")
    add_node(NodeType.CODE.value, "Code 2")
    add_node(NodeType.SPEC.value, "Spec 1")

    # Query for CODE nodes
    result = query_nodes(node_type=NodeType.CODE.value)

    assert result.success is True
    assert result.count == 2
    assert len(result.node_ids) == 2

    # Query for DOC nodes (should be empty)
    doc_result = query_nodes(node_type=NodeType.DOC.value)
    assert doc_result.count == 0


def test_query_nodes_by_status(fresh_db):
    """
    Validate that query_nodes correctly filters by status.

    Verifies:
    - Filtering by status returns correct nodes
    - Different statuses return different results
    """
    # Add nodes with different statuses
    node1 = add_node(NodeType.CODE.value, "Pending code")
    node2 = add_node(NodeType.CODE.value, "Verified code")

    # Update status of node2
    db = get_db()
    node = db.get_node(node2.node_id)
    node.set_status(NodeStatus.VERIFIED.value)
    db.update_node(node2.node_id, node)

    # Query by status
    pending_result = query_nodes(status=NodeStatus.PENDING.value)
    verified_result = query_nodes(status=NodeStatus.VERIFIED.value)

    assert pending_result.count >= 1
    assert verified_result.count >= 1


def test_get_node_tool(fresh_db):
    """
    Validate that get_node tool retrieves node data correctly.

    Verifies:
    - Tool returns complete node data
    - All fields are present in result
    """
    node_result = add_node(
        NodeType.CODE.value,
        "def test(): pass",
        data={"filename": "test.py"}
    )

    result = get_node(node_result.node_id)

    assert "id" in result
    assert "type" in result
    assert "content" in result
    assert result["id"] == node_result.node_id
    assert result["type"] == NodeType.CODE.value
    assert result["content"] == "def test(): pass"


# =============================================================================
# WAVE ANALYSIS TESTS
# =============================================================================

def test_get_waves_tool(db_with_sample_nodes):
    """
    Validate that get_waves tool computes execution layers correctly.

    Uses the db_with_sample_nodes fixture which has a REQ, SPEC, CODE structure.

    Verifies:
    - Tool returns success=True
    - Layer count is correct
    - Layers contain node IDs
    """
    from agents.tools import set_db

    db, nodes = db_with_sample_nodes

    # Add edges to create dependency chain: REQ -> SPEC -> CODE
    from core.schemas import EdgeData
    from core.ontology import EdgeType

    db.add_edge(EdgeData.create(nodes["req"].id, nodes["spec"].id, EdgeType.TRACES_TO.value))
    db.add_edge(EdgeData.create(nodes["spec"].id, nodes["code"].id, EdgeType.TRACES_TO.value))

    # Set this db as global
    set_db(db)

    result = get_waves()

    assert result.success is True
    assert result.layer_count > 0
    assert len(result.layers) > 0

    # Verify all layers contain node IDs (strings)
    for layer in result.layers:
        assert isinstance(layer, list)
        for node_id in layer:
            assert isinstance(node_id, str)


def test_get_waves_tool_empty_graph(fresh_db):
    """
    Validate that get_waves handles empty graph correctly.

    Verifies:
    - Returns success=True
    - Layer count is 0
    - Layers list is empty
    """
    result = get_waves()

    assert result.success is True
    assert result.layer_count == 0
    assert result.layers == []


# =============================================================================
# DESCENDANT ANALYSIS TESTS
# =============================================================================

def test_get_descendants_tool(fresh_db):
    """
    Validate that get_descendants tool finds transitive successors.

    Creates chain: A -> B -> C

    Verifies:
    - Tool returns success=True
    - Descendants are correct
    - Count matches number of descendants
    """
    # Create chain
    nodeA = add_node(NodeType.REQ.value, "A")
    nodeB = add_node(NodeType.SPEC.value, "B")
    nodeC = add_node(NodeType.CODE.value, "C")

    add_edge(nodeA.node_id, nodeB.node_id, EdgeType.DEPENDS_ON.value)
    add_edge(nodeB.node_id, nodeC.node_id, EdgeType.DEPENDS_ON.value)

    # Get descendants of A
    result = get_descendants(nodeA.node_id)

    assert result.success is True
    assert result.count >= 2
    assert nodeB.node_id in result.node_ids
    assert nodeC.node_id in result.node_ids


def test_get_descendants_tool_leaf_node(fresh_db):
    """
    Validate that get_descendants returns empty list for leaf nodes.

    Verifies:
    - Leaf node has no descendants
    - Tool returns success=True with count=0
    """
    node = add_node(NodeType.CODE.value, "Leaf node")

    result = get_descendants(node.node_id)

    assert result.success is True
    assert result.count == 0
    assert len(result.node_ids) == 0


# =============================================================================
# SYNTAX CHECKING TESTS (Layer 7B)
# =============================================================================

def test_check_syntax_valid_code(fresh_db):
    """
    Validate that check_syntax accepts valid Python code.

    Verifies:
    - Tool returns success=True
    - valid=True for correct Python
    - No errors reported
    """
    valid_code = """
def hello_world():
    '''Simple function.'''
    print("Hello, world!")
    return 42
"""

    result = check_syntax(valid_code, "python")

    assert result.success is True
    assert result.valid is True
    assert len(result.errors) == 0
    assert result.language == "python"


def test_check_syntax_invalid_code(fresh_db):
    """
    Validate that check_syntax detects syntax errors in Python code.

    Verifies:
    - Tool returns success=True (operation succeeded)
    - valid=False (code is invalid)
    - Errors list is not empty
    """
    invalid_code = """
def broken_function(
    # Missing closing parenthesis and colon
    print("This won't work"
"""

    result = check_syntax(invalid_code, "python")

    assert result.success is True  # Operation succeeded
    assert result.valid is False   # But code is invalid
    assert len(result.errors) > 0  # Should report errors


def test_check_syntax_unsupported_language(fresh_db):
    """
    Validate that check_syntax handles unsupported languages gracefully.

    Verifies:
    - Tool doesn't crash on unsupported languages
    - Returns success with warning
    """
    code = "SELECT * FROM users;"

    result = check_syntax(code, "sql")

    # Should succeed but skip validation
    assert result.success is True
    # May have warnings about unsupported language
    assert len(result.warnings) > 0 or result.valid is True


# =============================================================================
# GRAPH STATISTICS TESTS
# =============================================================================

def test_get_graph_stats_tool(fresh_db):
    """
    Validate that get_graph_stats returns accurate statistics.

    Verifies:
    - Stats include node_count, edge_count, has_cycle, is_empty
    - Values are correct
    """
    stats = get_graph_stats()

    assert "node_count" in stats
    assert "edge_count" in stats
    assert "has_cycle" in stats
    assert "is_empty" in stats

    # Initially empty
    assert stats["node_count"] == 0
    assert stats["edge_count"] == 0
    assert stats["is_empty"] is True

    # Add nodes and edges
    node1 = add_node(NodeType.CODE.value, "Code 1")
    node2 = add_node(NodeType.CODE.value, "Code 2")
    add_edge(node1.node_id, node2.node_id, EdgeType.DEPENDS_ON.value)

    stats = get_graph_stats()
    assert stats["node_count"] == 2
    assert stats["edge_count"] == 1
    assert stats["is_empty"] is False


# =============================================================================
# CYCLE DETECTION TESTS
# =============================================================================

def test_check_cycle_no_cycle(fresh_db):
    """
    Validate that check_cycle correctly identifies acyclic graphs.

    Verifies:
    - Tool returns success=True
    - has_cycle=False for DAG
    - Message indicates acyclic graph
    """
    # Create acyclic chain
    node1 = add_node(NodeType.REQ.value, "REQ")
    node2 = add_node(NodeType.SPEC.value, "SPEC")

    add_edge(node1.node_id, node2.node_id, EdgeType.DEPENDS_ON.value)

    result = check_cycle()

    assert result.success is True
    assert result.has_cycle is False
    assert "acyclic" in result.message.lower() or "DAG" in result.message


def test_check_cycle_with_cycle(fresh_db):
    """
    Validate that cycle-creating edges are rejected at write time.

    The graph has built-in cycle prevention - edges that would create
    cycles are rejected in add_edge. This test verifies that:
    - Cycle-creating edges return success=False
    - The graph remains acyclic after the rejection
    """
    # Create chain: A -> B -> C
    nodeA = add_node(NodeType.CODE.value, "A")
    nodeB = add_node(NodeType.CODE.value, "B")
    nodeC = add_node(NodeType.CODE.value, "C")

    add_edge(nodeA.node_id, nodeB.node_id, EdgeType.DEPENDS_ON.value)
    add_edge(nodeB.node_id, nodeC.node_id, EdgeType.DEPENDS_ON.value)

    # Attempt to create cycle: C -> A (should be rejected)
    result = add_edge(nodeC.node_id, nodeA.node_id, EdgeType.DEPENDS_ON.value)

    # Edge creation should fail due to cycle prevention
    assert result.success is False
    assert "cycle" in result.message.lower()

    # Graph should still be acyclic
    cycle_result = check_cycle()
    assert cycle_result.success is True
    assert cycle_result.has_cycle is False


# =============================================================================
# SAFE NODE CREATION TESTS (Layer 7B)
# =============================================================================

def test_add_node_safe_rejects_invalid_syntax(fresh_db):
    """
    Validate that add_node_safe rejects code with syntax errors.

    This is the core Layer 7B auditor functionality - preventing invalid
    code from entering the graph.

    Verifies:
    - Tool returns success=False
    - syntax_valid=False
    - Violations list contains error
    - Node is NOT added to graph
    """
    invalid_code = """
def broken(:
    print("Invalid"
"""

    result = add_node_safe(
        node_type=NodeType.CODE.value,
        content=invalid_code
    )

    assert result.success is False
    assert result.syntax_valid is False
    assert len(result.violations) > 0

    # Verify node was NOT added
    db = get_db()
    assert db.node_count == 0


def test_add_node_safe_accepts_valid_code(fresh_db):
    """
    Validate that add_node_safe accepts syntactically valid code.

    Verifies:
    - Tool returns success=True
    - syntax_valid=True
    - Node is added to graph
    - Node ID is returned
    """
    valid_code = """
def valid_function(x: int) -> int:
    '''A valid function.'''
    return x * 2
"""

    result = add_node_safe(
        node_type=NodeType.CODE.value,
        content=valid_code
    )

    assert result.success is True
    assert result.syntax_valid is True
    assert result.node_id != ""

    # Verify node was added
    db = get_db()
    assert db.node_count == 1
    node = db.get_node(result.node_id)
    assert node.content == valid_code


def test_add_node_safe_non_code_skips_syntax_check(fresh_db):
    """
    Validate that add_node_safe only checks syntax for CODE nodes.

    Non-CODE nodes (SPEC, REQ, etc.) should bypass syntax validation.

    Verifies:
    - Non-CODE nodes don't get syntax checked
    - Tool succeeds for SPEC nodes
    """
    result = add_node_safe(
        node_type=NodeType.SPEC.value,
        content="This is a specification, not code"
    )

    assert result.success is True
    # For non-CODE nodes, syntax check is skipped (valid by default)
    assert result.node_id != ""

    db = get_db()
    assert db.node_count == 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_tool_workflow_create_and_query(fresh_db):
    """
    Integration test: Create nodes via tools and query them.

    Verifies complete workflow:
    1. Create multiple nodes
    2. Create edges between them
    3. Query by type
    4. Get graph statistics
    """
    # Create nodes
    req = add_node(NodeType.REQ.value, "User requirement")
    spec = add_node(NodeType.SPEC.value, "Specification")
    code = add_node(NodeType.CODE.value, "def impl(): pass")

    assert req.success
    assert spec.success
    assert code.success

    # Create edges
    edge1 = add_edge(spec.node_id, req.node_id, EdgeType.TRACES_TO.value)
    edge2 = add_edge(code.node_id, spec.node_id, EdgeType.IMPLEMENTS.value)

    assert edge1.success
    assert edge2.success

    # Query
    code_nodes = query_nodes(node_type=NodeType.CODE.value)
    assert code_nodes.count == 1

    # Stats
    stats = get_graph_stats()
    assert stats["node_count"] == 3
    assert stats["edge_count"] == 2


def test_tool_workflow_safe_creation_with_validation(fresh_db):
    """
    Integration test: Use add_node_safe with syntax validation.

    Verifies Layer 7B auditor workflow:
    1. Try to add invalid code (should fail)
    2. Add valid code (should succeed)
    3. Query and verify only valid code exists
    """
    # Try to add invalid code
    invalid_result = add_node_safe(
        NodeType.CODE.value,
        "def broken(: pass"
    )
    assert invalid_result.success is False

    # Add valid code
    valid_result = add_node_safe(
        NodeType.CODE.value,
        "def working(): return 42"
    )
    assert valid_result.success is True

    # Verify only valid code is in graph
    stats = get_graph_stats()
    assert stats["node_count"] == 1

    db = get_db()
    node = db.get_node(valid_result.node_id)
    assert "working" in node.content
