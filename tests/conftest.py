"""
Pytest configuration and shared fixtures for Paragon tests.

This file provides common test fixtures that are available to all test files:
- fresh_db: Provides a clean ParagonDB instance for each test
- mock_llm: Mocks LLM responses for testing agent logic
- sample_graph: Pre-populated REQ->SPEC->CODE chain for integration tests
"""
import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from core.graph_db import ParagonDB
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, EdgeType, NodeStatus
from agents.tools import set_db


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest.fixture
def fresh_db():
    """
    Provide a clean ParagonDB instance for each test.

    This fixture creates a new empty graph database and injects it into
    the global tools context so that all tool functions operate on this
    test database.

    Yields:
        ParagonDB: A fresh, empty database instance

    Example:
        def test_add_node(fresh_db):
            node = NodeData.create(type="CODE", content="def foo(): pass")
            fresh_db.add_node(node)
            assert fresh_db.node_count == 1
    """
    db = ParagonDB()
    # Inject into tools module so tool functions use this test DB
    set_db(db)
    yield db
    # Cleanup: Clear the global DB reference
    set_db(None)


@pytest.fixture
def sample_graph(fresh_db):
    """
    Provide a pre-populated graph with REQ->SPEC->CODE chain.

    Creates a realistic test graph structure:
    - 1 REQ node (requirement)
    - 2 SPEC nodes (specifications) depending on REQ
    - 2 CODE nodes (implementations) implementing the SPEC nodes
    - All appropriate edges (TRACES_TO, DEPENDS_ON, IMPLEMENTS)

    Returns:
        Dict[str, Any]: Dictionary with:
            - 'db': The ParagonDB instance
            - 'req_id': ID of the requirement node
            - 'spec_ids': List of SPEC node IDs
            - 'code_ids': List of CODE node IDs

    Example:
        def test_wavefront(sample_graph):
            db = sample_graph['db']
            waves = db.get_waves()
            assert len(waves) >= 2  # At least REQ and SPEC layers
    """
    # Create REQ node
    req = NodeData.create(
        type=NodeType.REQ.value,
        content="Implement a cryptographic hash function module",
        created_by="test_user",
        status=NodeStatus.VERIFIED.value
    )
    fresh_db.add_node(req)

    # Create SPEC nodes
    spec1 = NodeData.create(
        type=NodeType.SPEC.value,
        content="Implement SHA256 hash function",
        created_by="architect",
        status=NodeStatus.VERIFIED.value
    )
    spec2 = NodeData.create(
        type=NodeType.SPEC.value,
        content="Implement hash verification utility",
        created_by="architect",
        status=NodeStatus.VERIFIED.value
    )
    fresh_db.add_node(spec1)
    fresh_db.add_node(spec2)

    # Create CODE nodes
    code1 = NodeData.create(
        type=NodeType.CODE.value,
        content="def sha256(data: bytes) -> str:\n    import hashlib\n    return hashlib.sha256(data).hexdigest()",
        created_by="builder",
        status=NodeStatus.VERIFIED.value,
        data={"filename": "crypto.py"}
    )
    code2 = NodeData.create(
        type=NodeType.CODE.value,
        content="def verify_hash(data: bytes, expected: str) -> bool:\n    return sha256(data) == expected",
        created_by="builder",
        status=NodeStatus.PENDING.value,
        data={"filename": "crypto.py"}
    )
    fresh_db.add_node(code1)
    fresh_db.add_node(code2)

    # Create edges: SPEC -> REQ (TRACES_TO)
    fresh_db.add_edge(EdgeData.create(
        source_id=spec1.id,
        target_id=req.id,
        type=EdgeType.TRACES_TO.value
    ))
    fresh_db.add_edge(EdgeData.create(
        source_id=spec2.id,
        target_id=req.id,
        type=EdgeType.TRACES_TO.value
    ))

    # Create edges: CODE -> SPEC (IMPLEMENTS)
    fresh_db.add_edge(EdgeData.create(
        source_id=code1.id,
        target_id=spec1.id,
        type=EdgeType.IMPLEMENTS.value
    ))
    fresh_db.add_edge(EdgeData.create(
        source_id=code2.id,
        target_id=spec2.id,
        type=EdgeType.IMPLEMENTS.value
    ))

    # Create dependency: spec2 depends on spec1
    fresh_db.add_edge(EdgeData.create(
        source_id=spec2.id,
        target_id=spec1.id,
        type=EdgeType.DEPENDS_ON.value
    ))

    # Create dependency: code2 depends on code1
    fresh_db.add_edge(EdgeData.create(
        source_id=code2.id,
        target_id=code1.id,
        type=EdgeType.DEPENDS_ON.value
    ))

    return {
        'db': fresh_db,
        'req_id': req.id,
        'spec_ids': [spec1.id, spec2.id],
        'code_ids': [code1.id, code2.id],
        'nodes': {
            'req': req,
            'spec1': spec1,
            'spec2': spec2,
            'code1': code1,
            'code2': code2,
        }
    }


# =============================================================================
# LLM MOCKING FIXTURES
# =============================================================================

@pytest.fixture
def mock_llm():
    """
    Provide a mock LLM client for testing agent logic without API calls.

    This fixture patches the LLM generation to return predefined responses,
    allowing us to test agent behavior without making actual API calls.

    Returns:
        Mock: A mock object that can be configured to return specific responses

    Example:
        def test_agent_generation(mock_llm):
            mock_llm.return_value = '{"code": "def test(): pass"}'
            result = agent.generate_code(spec)
            assert "def test():" in result
    """
    with patch('core.llm.generate_structured') as mock:
        # Default response is a simple success
        mock.return_value = {"success": True, "message": "Mock LLM response"}
        yield mock


@pytest.fixture
def mock_llm_code_generation(mock_llm):
    """
    Specialized mock for code generation that returns valid Python code.

    Returns:
        Mock: Configured to return valid code generation responses
    """
    mock_llm.return_value = {
        "filename": "test_module.py",
        "code": "def example_function():\n    '''Example function.'''\n    return 42",
        "imports": ["from typing import Any"],
        "description": "Example function implementation",
        "language": "python"
    }
    return mock_llm


# =============================================================================
# HELPER FIXTURES
# =============================================================================

@pytest.fixture
def sample_nodes():
    """
    Provide a list of sample NodeData objects for batch operations.

    Returns:
        List[NodeData]: A list of 5 sample nodes of various types
    """
    return [
        NodeData.create(type=NodeType.REQ.value, content="Requirement 1"),
        NodeData.create(type=NodeType.SPEC.value, content="Spec 1"),
        NodeData.create(type=NodeType.CODE.value, content="def func1(): pass"),
        NodeData.create(type=NodeType.TEST.value, content="def test_func1(): assert True"),
        NodeData.create(type=NodeType.DOC.value, content="Documentation for func1"),
    ]


@pytest.fixture
def sample_edges(sample_nodes):
    """
    Provide a list of sample EdgeData objects connecting sample_nodes.

    Args:
        sample_nodes: Fixture providing sample nodes

    Returns:
        List[EdgeData]: A list of edges forming a dependency chain
    """
    return [
        EdgeData.create(
            source_id=sample_nodes[1].id,  # SPEC
            target_id=sample_nodes[0].id,  # -> REQ
            type=EdgeType.TRACES_TO.value
        ),
        EdgeData.create(
            source_id=sample_nodes[2].id,  # CODE
            target_id=sample_nodes[1].id,  # -> SPEC
            type=EdgeType.IMPLEMENTS.value
        ),
        EdgeData.create(
            source_id=sample_nodes[3].id,  # TEST
            target_id=sample_nodes[2].id,  # -> CODE
            type=EdgeType.VERIFIES.value
        ),
    ]


# =============================================================================
# ASSERTION HELPERS
# =============================================================================

def assert_graph_has_nodes(db: ParagonDB, expected_count: int):
    """Helper to assert graph has expected number of nodes."""
    assert db.node_count == expected_count, f"Expected {expected_count} nodes, got {db.node_count}"


def assert_graph_has_edges(db: ParagonDB, expected_count: int):
    """Helper to assert graph has expected number of edges."""
    assert db.edge_count == expected_count, f"Expected {expected_count} edges, got {db.edge_count}"


def assert_node_exists(db: ParagonDB, node_id: str):
    """Helper to assert a node exists in the graph."""
    assert db.has_node(node_id), f"Node {node_id} does not exist in graph"


def assert_edge_exists(db: ParagonDB, source_id: str, target_id: str):
    """Helper to assert an edge exists in the graph."""
    assert db.has_edge(source_id, target_id), f"Edge {source_id} -> {target_id} does not exist"


# Export helpers for use in test files
__all__ = [
    'fresh_db',
    'sample_graph',
    'mock_llm',
    'mock_llm_code_generation',
    'sample_nodes',
    'sample_edges',
    'assert_graph_has_nodes',
    'assert_graph_has_edges',
    'assert_node_exists',
    'assert_edge_exists',
]
