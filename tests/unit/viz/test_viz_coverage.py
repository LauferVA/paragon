"""
Unit tests for viz/core.py - Visualization Module

Tests the visualization data structures, serialization, and graph export functionality:
- VizNode: Lightweight node representation with color/layout
- VizEdge: Lightweight edge representation
- GraphSnapshot: Full graph state for rendering
- GraphDelta: Incremental updates for real-time streaming
- MutationEvent: Individual mutation tracking
- VizGraph: Visualization state management
- Arrow IPC serialization
- Database snapshot creation
- Graph comparison utilities
"""
import pytest
import msgspec
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch
import io

import polars as pl

from viz.core import (
    VizNode,
    VizEdge,
    VizGraph,
    GraphSnapshot,
    GraphDelta,
    MutationEvent,
    MutationType,
    GraphComparison,
    serialize_to_arrow,
    create_snapshot_from_db,
    compare_snapshots,
    NODE_COLORS,
    STATUS_COLORS,
    EDGE_COLORS,
)
from core.ontology import NodeType, EdgeType, NodeStatus
from core.schemas import NodeData, EdgeData


# =============================================================================
# VIZNODE TESTS
# =============================================================================

def test_viznode_creation():
    """
    Validate VizNode can be created with all fields.

    Verifies:
    - All required fields are set correctly
    - Optional fields default correctly
    - msgspec.Struct serialization works
    """
    node = VizNode(
        id="test-node-123",
        type=NodeType.CODE.value,
        status=NodeStatus.VERIFIED.value,
        label="Test Node",
        color="#2A9D8F",
        size=1.5,
        x=100.0,
        y=200.0,
        created_by="test_user",
        created_at="2025-12-07T00:00:00Z",
        teleology_status="verified",
        layer=2,
        is_root=False,
        is_leaf=True,
    )

    assert node.id == "test-node-123"
    assert node.type == NodeType.CODE.value
    assert node.status == NodeStatus.VERIFIED.value
    assert node.label == "Test Node"
    assert node.color == "#2A9D8F"
    assert node.size == 1.5
    assert node.x == 100.0
    assert node.y == 200.0
    assert node.created_by == "test_user"
    assert node.created_at == "2025-12-07T00:00:00Z"
    assert node.teleology_status == "verified"
    assert node.layer == 2
    assert node.is_root is False
    assert node.is_leaf is True


def test_viznode_from_node_data_type_coloring():
    """
    Validate VizNode.from_node_data creates correct node with type-based coloring.

    Verifies:
    - Node data is correctly transformed
    - Type-based color mapping works
    - Label is created from node data
    """
    node_data = NodeData.create(
        type=NodeType.REQ.value,
        content="Test requirement",
        status=NodeStatus.PENDING.value,
        created_by="user1",
    )
    node_data.data = {"name": "My Requirement Name"}

    viz_node = VizNode.from_node_data(
        node_data,
        color_mode="type",
        layer=1,
        is_root=True,
        is_leaf=False,
        x=50.0,
        y=100.0,
    )

    assert viz_node.id == node_data.id
    assert viz_node.type == NodeType.REQ.value
    assert viz_node.status == NodeStatus.PENDING.value
    assert viz_node.color == NODE_COLORS[NodeType.REQ.value]
    assert viz_node.label == "My Requirement Name"[:20]
    assert viz_node.created_by == "user1"
    assert viz_node.layer == 1
    assert viz_node.is_root is True
    assert viz_node.is_leaf is False
    assert viz_node.x == 50.0
    assert viz_node.y == 100.0


def test_viznode_from_node_data_status_coloring():
    """
    Validate VizNode.from_node_data with status-based coloring.

    Verifies:
    - Status-based color mapping works
    - All other fields are preserved
    """
    node_data = NodeData.create(
        type=NodeType.CODE.value,
        content="def test(): pass",
        status=NodeStatus.VERIFIED.value,
    )

    viz_node = VizNode.from_node_data(
        node_data,
        color_mode="status",
    )

    assert viz_node.color == STATUS_COLORS[NodeStatus.VERIFIED.value]
    assert viz_node.type == NodeType.CODE.value


def test_viznode_from_node_data_default_label():
    """
    Validate VizNode creates default label when no name in data.

    Verifies:
    - Default label format is type[:4]:id[:8]
    """
    node_data = NodeData.create(
        type=NodeType.SPEC.value,
        content="Test spec",
    )

    viz_node = VizNode.from_node_data(node_data)

    # Default label should be type[:4]:id[:8]
    expected_label = f"{NodeType.SPEC.value[:4]}:{node_data.id[:8]}"
    assert viz_node.label == expected_label


def test_viznode_from_node_data_unknown_type():
    """
    Validate VizNode handles unknown node type with default color.

    Verifies:
    - Unknown types get default color
    """
    node_data = NodeData.create(
        type="UNKNOWN_TYPE",
        content="Test",
    )

    viz_node = VizNode.from_node_data(node_data)

    assert viz_node.color == NODE_COLORS["default"]


def test_viznode_msgspec_serialization():
    """
    Validate VizNode can be serialized/deserialized with msgspec.

    Verifies:
    - Round-trip serialization preserves all data
    """
    original = VizNode(
        id="node-1",
        type=NodeType.CODE.value,
        status=NodeStatus.VERIFIED.value,
        label="Test",
        color="#FFFFFF",
        size=2.0,
        x=10.0,
        y=20.0,
    )

    # Serialize and deserialize
    encoded = msgspec.json.encode(original)
    decoded = msgspec.json.decode(encoded, type=VizNode)

    assert decoded.id == original.id
    assert decoded.type == original.type
    assert decoded.status == original.status
    assert decoded.label == original.label
    assert decoded.color == original.color
    assert decoded.size == original.size
    assert decoded.x == original.x
    assert decoded.y == original.y


# =============================================================================
# VIZEDGE TESTS
# =============================================================================

def test_vizedge_creation():
    """
    Validate VizEdge can be created with all fields.

    Verifies:
    - All required fields are set correctly
    - Default weight is 1.0
    """
    edge = VizEdge(
        source="node-1",
        target="node-2",
        type=EdgeType.IMPLEMENTS.value,
        color="#2A9D8F",
        weight=2.0,
    )

    assert edge.source == "node-1"
    assert edge.target == "node-2"
    assert edge.type == EdgeType.IMPLEMENTS.value
    assert edge.color == "#2A9D8F"
    assert edge.weight == 2.0


def test_vizedge_from_edge_data():
    """
    Validate VizEdge.from_edge_data creates correct edge.

    Verifies:
    - Edge data is correctly transformed
    - Color mapping works for edge types
    """
    edge_data = EdgeData(
        source_id="node-1",
        target_id="node-2",
        type=EdgeType.TRACES_TO.value,
        weight=1.5,
    )

    viz_edge = VizEdge.from_edge_data(edge_data)

    assert viz_edge.source == "node-1"
    assert viz_edge.target == "node-2"
    assert viz_edge.type == EdgeType.TRACES_TO.value
    assert viz_edge.color == EDGE_COLORS[EdgeType.TRACES_TO.value]
    assert viz_edge.weight == 1.5


def test_vizedge_from_edge_data_unknown_type():
    """
    Validate VizEdge handles unknown edge type with default color.

    Verifies:
    - Unknown edge types get default color
    """
    edge_data = EdgeData(
        source_id="node-1",
        target_id="node-2",
        type="UNKNOWN_EDGE",
    )

    viz_edge = VizEdge.from_edge_data(edge_data)

    assert viz_edge.color == EDGE_COLORS["default"]


def test_vizedge_msgspec_serialization():
    """
    Validate VizEdge can be serialized/deserialized with msgspec.

    Verifies:
    - Round-trip serialization preserves all data
    """
    original = VizEdge(
        source="n1",
        target="n2",
        type=EdgeType.IMPLEMENTS.value,
        color="#FFFFFF",
        weight=3.0,
    )

    encoded = msgspec.json.encode(original)
    decoded = msgspec.json.decode(encoded, type=VizEdge)

    assert decoded.source == original.source
    assert decoded.target == original.target
    assert decoded.type == original.type
    assert decoded.color == original.color
    assert decoded.weight == original.weight


# =============================================================================
# GRAPHSNAPSHOT TESTS
# =============================================================================

def test_graphsnapshot_creation():
    """
    Validate GraphSnapshot can be created with all fields.

    Verifies:
    - All fields are set correctly
    - Default values work as expected
    """
    node1 = VizNode(
        id="n1",
        type=NodeType.REQ.value,
        status=NodeStatus.VERIFIED.value,
        label="Node 1",
        color="#FF0000",
    )
    edge1 = VizEdge(
        source="n1",
        target="n2",
        type=EdgeType.IMPLEMENTS.value,
        color="#00FF00",
    )

    snapshot = GraphSnapshot(
        timestamp="2025-12-07T00:00:00Z",
        node_count=1,
        edge_count=1,
        nodes=[node1],
        edges=[edge1],
        layer_count=3,
        has_cycle=False,
        root_count=1,
        leaf_count=0,
        version="v1.0",
        label="Test Snapshot",
    )

    assert snapshot.timestamp == "2025-12-07T00:00:00Z"
    assert snapshot.node_count == 1
    assert snapshot.edge_count == 1
    assert len(snapshot.nodes) == 1
    assert len(snapshot.edges) == 1
    assert snapshot.layer_count == 3
    assert snapshot.has_cycle is False
    assert snapshot.root_count == 1
    assert snapshot.leaf_count == 0
    assert snapshot.version == "v1.0"
    assert snapshot.label == "Test Snapshot"


def test_graphsnapshot_to_dict():
    """
    Validate GraphSnapshot.to_dict() converts to dictionary correctly.

    Verifies:
    - All fields are present in dict
    - Nested structures are converted to builtins
    """
    node1 = VizNode(
        id="n1",
        type=NodeType.CODE.value,
        status=NodeStatus.VERIFIED.value,
        label="Node",
        color="#FF0000",
    )
    edge1 = VizEdge(
        source="n1",
        target="n2",
        type=EdgeType.IMPLEMENTS.value,
        color="#00FF00",
    )

    snapshot = GraphSnapshot(
        timestamp="2025-12-07T00:00:00Z",
        node_count=1,
        edge_count=1,
        nodes=[node1],
        edges=[edge1],
    )

    result = snapshot.to_dict()

    assert isinstance(result, dict)
    assert result["timestamp"] == "2025-12-07T00:00:00Z"
    assert result["node_count"] == 1
    assert result["edge_count"] == 1
    assert len(result["nodes"]) == 1
    assert len(result["edges"]) == 1
    assert result["layer_count"] == 0
    assert result["has_cycle"] is False
    assert isinstance(result["nodes"][0], dict)
    assert isinstance(result["edges"][0], dict)


# =============================================================================
# GRAPHDELTA TESTS
# =============================================================================

def test_graphdelta_creation():
    """
    Validate GraphDelta can be created with all change types.

    Verifies:
    - All fields are set correctly
    - Default empty lists work
    """
    node1 = VizNode(
        id="n1",
        type=NodeType.CODE.value,
        status=NodeStatus.VERIFIED.value,
        label="Node",
        color="#FF0000",
    )
    edge1 = VizEdge(
        source="n1",
        target="n2",
        type=EdgeType.IMPLEMENTS.value,
        color="#00FF00",
    )

    delta = GraphDelta(
        timestamp="2025-12-07T00:00:00Z",
        sequence=1,
        nodes_added=[node1],
        nodes_updated=[],
        nodes_removed=["n3"],
        edges_added=[edge1],
        edges_removed=[{"source": "n4", "target": "n5"}],
    )

    assert delta.timestamp == "2025-12-07T00:00:00Z"
    assert delta.sequence == 1
    assert len(delta.nodes_added) == 1
    assert len(delta.nodes_updated) == 0
    assert delta.nodes_removed == ["n3"]
    assert len(delta.edges_added) == 1
    assert delta.edges_removed == [{"source": "n4", "target": "n5"}]


def test_graphdelta_is_empty_true():
    """
    Validate GraphDelta.is_empty() returns True for empty delta.

    Verifies:
    - Empty delta is correctly identified
    """
    delta = GraphDelta(
        timestamp="2025-12-07T00:00:00Z",
        sequence=1,
    )

    assert delta.is_empty() is True


def test_graphdelta_is_empty_false():
    """
    Validate GraphDelta.is_empty() returns False for non-empty delta.

    Verifies:
    - Non-empty delta is correctly identified
    """
    node1 = VizNode(
        id="n1",
        type=NodeType.CODE.value,
        status=NodeStatus.VERIFIED.value,
        label="Node",
        color="#FF0000",
    )

    delta = GraphDelta(
        timestamp="2025-12-07T00:00:00Z",
        sequence=1,
        nodes_added=[node1],
    )

    assert delta.is_empty() is False


def test_graphdelta_to_dict():
    """
    Validate GraphDelta.to_dict() converts to dictionary correctly.

    Verifies:
    - All fields are present in dict
    - Nested structures are converted
    """
    node1 = VizNode(
        id="n1",
        type=NodeType.CODE.value,
        status=NodeStatus.VERIFIED.value,
        label="Node",
        color="#FF0000",
    )

    delta = GraphDelta(
        timestamp="2025-12-07T00:00:00Z",
        sequence=5,
        nodes_added=[node1],
        nodes_removed=["n2"],
    )

    result = delta.to_dict()

    assert isinstance(result, dict)
    assert result["timestamp"] == "2025-12-07T00:00:00Z"
    assert result["sequence"] == 5
    assert len(result["nodes_added"]) == 1
    assert result["nodes_removed"] == ["n2"]
    assert isinstance(result["nodes_added"][0], dict)


# =============================================================================
# MUTATIONEVENT TESTS
# =============================================================================

def test_mutationevent_node_created():
    """
    Validate MutationEvent for NODE_CREATED.

    Verifies:
    - All node creation fields are set correctly
    """
    event = MutationEvent(
        timestamp="2025-12-07T00:00:00Z",
        sequence=1,
        mutation_type=MutationType.NODE_CREATED.value,
        node_id="n1",
        node_type=NodeType.CODE.value,
        agent_id="builder-1",
        agent_role="Builder",
        correlation_id="corr-123",
    )

    assert event.timestamp == "2025-12-07T00:00:00Z"
    assert event.sequence == 1
    assert event.mutation_type == MutationType.NODE_CREATED.value
    assert event.node_id == "n1"
    assert event.node_type == NodeType.CODE.value
    assert event.agent_id == "builder-1"
    assert event.agent_role == "Builder"
    assert event.correlation_id == "corr-123"


def test_mutationevent_status_changed():
    """
    Validate MutationEvent for STATUS_CHANGED.

    Verifies:
    - Status change fields are set correctly
    """
    event = MutationEvent(
        timestamp="2025-12-07T00:00:00Z",
        sequence=2,
        mutation_type=MutationType.STATUS_CHANGED.value,
        node_id="n1",
        old_status=NodeStatus.PENDING.value,
        new_status=NodeStatus.VERIFIED.value,
    )

    assert event.mutation_type == MutationType.STATUS_CHANGED.value
    assert event.old_status == NodeStatus.PENDING.value
    assert event.new_status == NodeStatus.VERIFIED.value


def test_mutationevent_edge_created():
    """
    Validate MutationEvent for EDGE_CREATED.

    Verifies:
    - Edge creation fields are set correctly
    """
    event = MutationEvent(
        timestamp="2025-12-07T00:00:00Z",
        sequence=3,
        mutation_type=MutationType.EDGE_CREATED.value,
        source_id="n1",
        target_id="n2",
        edge_type=EdgeType.IMPLEMENTS.value,
    )

    assert event.mutation_type == MutationType.EDGE_CREATED.value
    assert event.source_id == "n1"
    assert event.target_id == "n2"
    assert event.edge_type == EdgeType.IMPLEMENTS.value


def test_mutationevent_context_pruned():
    """
    Validate MutationEvent for CONTEXT_PRUNED.

    Verifies:
    - Context pruning fields are set correctly
    """
    event = MutationEvent(
        timestamp="2025-12-07T00:00:00Z",
        sequence=4,
        mutation_type=MutationType.CONTEXT_PRUNED.value,
        nodes_considered=100,
        nodes_selected=10,
        token_usage=5000,
    )

    assert event.mutation_type == MutationType.CONTEXT_PRUNED.value
    assert event.nodes_considered == 100
    assert event.nodes_selected == 10
    assert event.token_usage == 5000


# =============================================================================
# VIZGRAPH TESTS
# =============================================================================

def test_vizgraph_initialization():
    """
    Validate VizGraph initializes correctly.

    Verifies:
    - Empty graph has zero nodes and edges
    - Sequence starts at 0
    """
    graph = VizGraph()

    assert graph.node_count == 0
    assert graph.edge_count == 0


def test_vizgraph_add_node():
    """
    Validate VizGraph.add_node() adds node correctly.

    Verifies:
    - Node is added to internal storage
    - Node count increases
    """
    graph = VizGraph()
    node = VizNode(
        id="n1",
        type=NodeType.CODE.value,
        status=NodeStatus.VERIFIED.value,
        label="Test",
        color="#FF0000",
    )

    graph.add_node(node)

    assert graph.node_count == 1
    assert "n1" in graph._nodes


def test_vizgraph_add_node_updates_existing():
    """
    Validate VizGraph.add_node() updates existing node.

    Verifies:
    - Adding node with same ID updates it
    - Node count doesn't increase
    """
    graph = VizGraph()
    node1 = VizNode(
        id="n1",
        type=NodeType.CODE.value,
        status=NodeStatus.PENDING.value,
        label="Test1",
        color="#FF0000",
    )
    node2 = VizNode(
        id="n1",
        type=NodeType.CODE.value,
        status=NodeStatus.VERIFIED.value,
        label="Test2",
        color="#00FF00",
    )

    graph.add_node(node1)
    graph.add_node(node2)

    assert graph.node_count == 1
    assert graph._nodes["n1"].status == NodeStatus.VERIFIED.value
    assert graph._nodes["n1"].label == "Test2"


def test_vizgraph_remove_node():
    """
    Validate VizGraph.remove_node() removes node correctly.

    Verifies:
    - Node is removed from storage
    - Node count decreases
    - Returns removed node
    """
    graph = VizGraph()
    node = VizNode(
        id="n1",
        type=NodeType.CODE.value,
        status=NodeStatus.VERIFIED.value,
        label="Test",
        color="#FF0000",
    )

    graph.add_node(node)
    removed = graph.remove_node("n1")

    assert graph.node_count == 0
    assert removed.id == "n1"
    assert "n1" not in graph._nodes


def test_vizgraph_remove_node_removes_incident_edges():
    """
    Validate VizGraph.remove_node() removes incident edges.

    Verifies:
    - Removing a node also removes its edges
    - Edge count decreases correctly
    """
    graph = VizGraph()
    node1 = VizNode(id="n1", type=NodeType.CODE.value, status=NodeStatus.VERIFIED.value, label="N1", color="#FF0000")
    node2 = VizNode(id="n2", type=NodeType.CODE.value, status=NodeStatus.VERIFIED.value, label="N2", color="#FF0000")
    edge1 = VizEdge(source="n1", target="n2", type=EdgeType.IMPLEMENTS.value, color="#00FF00")
    edge2 = VizEdge(source="n2", target="n1", type=EdgeType.IMPLEMENTS.value, color="#00FF00")

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    assert graph.edge_count == 2

    graph.remove_node("n1")

    assert graph.edge_count == 0


def test_vizgraph_remove_nonexistent_node():
    """
    Validate VizGraph.remove_node() returns None for non-existent node.

    Verifies:
    - Removing non-existent node returns None
    - No errors raised
    """
    graph = VizGraph()

    removed = graph.remove_node("nonexistent")

    assert removed is None


def test_vizgraph_add_edge():
    """
    Validate VizGraph.add_edge() adds edge correctly.

    Verifies:
    - Edge is added to internal storage
    - Edge count increases
    """
    graph = VizGraph()
    edge = VizEdge(
        source="n1",
        target="n2",
        type=EdgeType.IMPLEMENTS.value,
        color="#00FF00",
    )

    graph.add_edge(edge)

    assert graph.edge_count == 1
    assert ("n1", "n2") in graph._edges


def test_vizgraph_add_edge_updates_existing():
    """
    Validate VizGraph.add_edge() updates existing edge.

    Verifies:
    - Adding edge with same source/target updates it
    - Edge count doesn't increase
    """
    graph = VizGraph()
    edge1 = VizEdge(source="n1", target="n2", type=EdgeType.IMPLEMENTS.value, color="#FF0000", weight=1.0)
    edge2 = VizEdge(source="n1", target="n2", type=EdgeType.TRACES_TO.value, color="#00FF00", weight=2.0)

    graph.add_edge(edge1)
    graph.add_edge(edge2)

    assert graph.edge_count == 1
    assert graph._edges[("n1", "n2")].type == EdgeType.TRACES_TO.value
    assert graph._edges[("n1", "n2")].weight == 2.0


def test_vizgraph_remove_edge():
    """
    Validate VizGraph.remove_edge() removes edge correctly.

    Verifies:
    - Edge is removed from storage
    - Edge count decreases
    - Returns removed edge
    """
    graph = VizGraph()
    edge = VizEdge(source="n1", target="n2", type=EdgeType.IMPLEMENTS.value, color="#00FF00")

    graph.add_edge(edge)
    removed = graph.remove_edge("n1", "n2")

    assert graph.edge_count == 0
    assert removed.source == "n1"
    assert removed.target == "n2"
    assert ("n1", "n2") not in graph._edges


def test_vizgraph_remove_nonexistent_edge():
    """
    Validate VizGraph.remove_edge() returns None for non-existent edge.

    Verifies:
    - Removing non-existent edge returns None
    - No errors raised
    """
    graph = VizGraph()

    removed = graph.remove_edge("n1", "n2")

    assert removed is None


def test_vizgraph_get_snapshot():
    """
    Validate VizGraph.get_snapshot() creates correct snapshot.

    Verifies:
    - Snapshot contains all nodes and edges
    - Metadata is calculated correctly
    """
    graph = VizGraph()
    node1 = VizNode(id="n1", type=NodeType.CODE.value, status=NodeStatus.VERIFIED.value, label="N1", color="#FF0000", is_root=True, layer=0)
    node2 = VizNode(id="n2", type=NodeType.CODE.value, status=NodeStatus.VERIFIED.value, label="N2", color="#FF0000", is_leaf=True, layer=1)
    edge = VizEdge(source="n1", target="n2", type=EdgeType.IMPLEMENTS.value, color="#00FF00")

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_edge(edge)

    snapshot = graph.get_snapshot(version="v1.0", label="Test")

    assert snapshot.node_count == 2
    assert snapshot.edge_count == 1
    assert len(snapshot.nodes) == 2
    assert len(snapshot.edges) == 1
    assert snapshot.root_count == 1
    assert snapshot.leaf_count == 1
    assert snapshot.layer_count == 2  # layers 0 and 1 = 2 layers
    assert snapshot.version == "v1.0"
    assert snapshot.label == "Test"
    assert snapshot.has_cycle is False


def test_vizgraph_create_delta():
    """
    Validate VizGraph.create_delta() creates correct delta.

    Verifies:
    - Delta contains specified changes
    - Sequence number increments
    """
    graph = VizGraph()
    node1 = VizNode(id="n1", type=NodeType.CODE.value, status=NodeStatus.VERIFIED.value, label="N1", color="#FF0000")
    edge1 = VizEdge(source="n1", target="n2", type=EdgeType.IMPLEMENTS.value, color="#00FF00")

    delta1 = graph.create_delta(
        added_nodes=[node1],
        added_edges=[edge1],
    )

    assert delta1.sequence == 1
    assert len(delta1.nodes_added) == 1
    assert len(delta1.edges_added) == 1

    delta2 = graph.create_delta(
        removed_node_ids=["n1"],
    )

    assert delta2.sequence == 2
    assert delta2.nodes_removed == ["n1"]


# =============================================================================
# SERIALIZE_TO_ARROW TESTS
# =============================================================================

def test_serialize_to_arrow_success():
    """
    Validate serialize_to_arrow() creates valid Arrow IPC bytes.

    Verifies:
    - Returns tuple of (nodes_bytes, edges_bytes)
    - Bytes can be deserialized back to polars DataFrames
    - Data is preserved
    """
    node1 = VizNode(
        id="n1",
        type=NodeType.CODE.value,
        status=NodeStatus.VERIFIED.value,
        label="Node1",
        color="#FF0000",
        size=1.0,
        x=10.0,
        y=20.0,
        layer=0,
        is_root=True,
        is_leaf=False,
    )
    edge1 = VizEdge(
        source="n1",
        target="n2",
        type=EdgeType.IMPLEMENTS.value,
        color="#00FF00",
        weight=1.5,
    )

    snapshot = GraphSnapshot(
        timestamp="2025-12-07T00:00:00Z",
        node_count=1,
        edge_count=1,
        nodes=[node1],
        edges=[edge1],
    )

    nodes_bytes, edges_bytes = serialize_to_arrow(snapshot)

    # Verify we got bytes back
    assert isinstance(nodes_bytes, bytes)
    assert isinstance(edges_bytes, bytes)
    assert len(nodes_bytes) > 0
    assert len(edges_bytes) > 0

    # Deserialize and verify data
    nodes_df = pl.read_ipc(io.BytesIO(nodes_bytes))
    edges_df = pl.read_ipc(io.BytesIO(edges_bytes))

    assert len(nodes_df) == 1
    assert nodes_df["id"][0] == "n1"
    assert nodes_df["type"][0] == NodeType.CODE.value
    assert nodes_df["label"][0] == "Node1"

    assert len(edges_df) == 1
    assert edges_df["source"][0] == "n1"
    assert edges_df["target"][0] == "n2"
    assert edges_df["type"][0] == EdgeType.IMPLEMENTS.value


def test_serialize_to_arrow_empty_graph():
    """
    Validate serialize_to_arrow() handles empty graph.

    Verifies:
    - Empty snapshot serializes without errors
    - Returns valid but empty Arrow bytes
    """
    snapshot = GraphSnapshot(
        timestamp="2025-12-07T00:00:00Z",
        node_count=0,
        edge_count=0,
        nodes=[],
        edges=[],
    )

    nodes_bytes, edges_bytes = serialize_to_arrow(snapshot)

    assert isinstance(nodes_bytes, bytes)
    assert isinstance(edges_bytes, bytes)

    nodes_df = pl.read_ipc(io.BytesIO(nodes_bytes))
    edges_df = pl.read_ipc(io.BytesIO(edges_bytes))

    assert len(nodes_df) == 0
    assert len(edges_df) == 0


# =============================================================================
# CREATE_SNAPSHOT_FROM_DB TESTS
# =============================================================================

def test_create_snapshot_from_db_success():
    """
    Validate create_snapshot_from_db() creates snapshot from ParagonDB.

    Verifies:
    - Snapshot contains all nodes and edges from DB
    - Layout hints are computed
    - Metadata is calculated
    """
    # Mock ParagonDB
    mock_db = Mock()

    node1 = NodeData.create(type=NodeType.REQ.value, content="Requirement")
    node2 = NodeData.create(type=NodeType.CODE.value, content="Code")
    edge1 = EdgeData(source_id=node1.id, target_id=node2.id, type=EdgeType.IMPLEMENTS.value)

    mock_db.get_all_nodes.return_value = [node1, node2]
    mock_db.get_all_edges.return_value = [edge1]
    mock_db.get_waves.return_value = [[node1], [node2]]
    mock_db.get_root_nodes.return_value = [node1]
    mock_db.get_leaf_nodes.return_value = [node2]
    mock_db.has_cycle.return_value = False

    snapshot = create_snapshot_from_db(
        mock_db,
        color_mode="type",
        version="v1.0",
        label="Test DB",
    )

    assert snapshot.node_count == 2
    assert snapshot.edge_count == 1
    assert snapshot.layer_count == 2
    assert snapshot.root_count == 1
    assert snapshot.leaf_count == 1
    assert snapshot.has_cycle is False
    assert snapshot.version == "v1.0"
    assert snapshot.label == "Test DB"

    # Verify layout hints were computed
    assert all(n.x is not None for n in snapshot.nodes)
    assert all(n.y is not None for n in snapshot.nodes)


def test_create_snapshot_from_db_handles_exceptions():
    """
    Validate create_snapshot_from_db() handles get_waves() exceptions gracefully.

    Verifies:
    - Exceptions in get_waves() don't crash
    - Falls back to default layer 0
    """
    mock_db = Mock()

    node1 = NodeData.create(type=NodeType.CODE.value, content="Code")

    mock_db.get_all_nodes.return_value = [node1]
    mock_db.get_all_edges.return_value = []
    mock_db.get_waves.side_effect = Exception("Wave computation failed")
    mock_db.get_root_nodes.return_value = [node1]
    mock_db.get_leaf_nodes.return_value = [node1]
    mock_db.has_cycle.return_value = False

    snapshot = create_snapshot_from_db(mock_db)

    assert snapshot.node_count == 1
    # Should still work even if waves failed
    assert len(snapshot.nodes) == 1


def test_create_snapshot_from_db_status_coloring():
    """
    Validate create_snapshot_from_db() with status-based coloring.

    Verifies:
    - Nodes are colored by status when color_mode="status"
    """
    mock_db = Mock()

    node1 = NodeData.create(type=NodeType.CODE.value, content="Code", status=NodeStatus.VERIFIED.value)

    mock_db.get_all_nodes.return_value = [node1]
    mock_db.get_all_edges.return_value = []
    mock_db.get_waves.return_value = [[node1]]
    mock_db.get_root_nodes.return_value = [node1]
    mock_db.get_leaf_nodes.return_value = [node1]
    mock_db.has_cycle.return_value = False

    snapshot = create_snapshot_from_db(mock_db, color_mode="status")

    # Node should be colored by status
    assert snapshot.nodes[0].color == STATUS_COLORS[NodeStatus.VERIFIED.value]


# =============================================================================
# GRAPHCOMPARISON TESTS
# =============================================================================

def test_graphcomparison_initialization():
    """
    Validate GraphComparison computes deltas on initialization.

    Verifies:
    - Delta metrics are computed
    - Node sets are identified correctly
    """
    node1 = VizNode(id="n1", type=NodeType.CODE.value, status=NodeStatus.VERIFIED.value, label="N1", color="#FF0000")
    node2 = VizNode(id="n2", type=NodeType.CODE.value, status=NodeStatus.VERIFIED.value, label="N2", color="#FF0000")
    node3 = VizNode(id="n3", type=NodeType.SPEC.value, status=NodeStatus.PENDING.value, label="N3", color="#00FF00")

    edge1 = VizEdge(source="n1", target="n2", type=EdgeType.IMPLEMENTS.value, color="#0000FF")
    edge2 = VizEdge(source="n2", target="n3", type=EdgeType.TRACES_TO.value, color="#0000FF")

    baseline = GraphSnapshot(
        timestamp="2025-12-07T00:00:00Z",
        node_count=2,
        edge_count=1,
        nodes=[node1, node2],
        edges=[edge1],
    )

    treatment = GraphSnapshot(
        timestamp="2025-12-07T01:00:00Z",
        node_count=2,
        edge_count=1,
        nodes=[node2, node3],
        edges=[edge2],
    )

    comparison = GraphComparison(baseline=baseline, treatment=treatment)

    assert comparison.node_count_delta == 0  # 2 - 2 = 0
    assert comparison.edge_count_delta == 0  # 1 - 1 = 0
    assert "n1" in comparison.nodes_only_in_baseline
    assert "n3" in comparison.nodes_only_in_treatment
    assert "n2" in comparison.nodes_in_both


def test_graphcomparison_type_deltas():
    """
    Validate GraphComparison computes type deltas correctly.

    Verifies:
    - Type count changes are calculated
    """
    node1 = VizNode(id="n1", type=NodeType.CODE.value, status=NodeStatus.VERIFIED.value, label="N1", color="#FF0000")
    node2 = VizNode(id="n2", type=NodeType.CODE.value, status=NodeStatus.VERIFIED.value, label="N2", color="#FF0000")
    node3 = VizNode(id="n3", type=NodeType.SPEC.value, status=NodeStatus.PENDING.value, label="N3", color="#00FF00")

    baseline = GraphSnapshot(
        timestamp="2025-12-07T00:00:00Z",
        node_count=1,
        edge_count=0,
        nodes=[node1],
        edges=[],
    )

    treatment = GraphSnapshot(
        timestamp="2025-12-07T01:00:00Z",
        node_count=2,
        edge_count=0,
        nodes=[node2, node3],
        edges=[],
    )

    comparison = GraphComparison(baseline=baseline, treatment=treatment)

    # CODE: 1 -> 1 = 0 delta
    assert comparison.type_deltas[NodeType.CODE.value] == 0
    # SPEC: 0 -> 1 = +1 delta
    assert comparison.type_deltas[NodeType.SPEC.value] == 1


def test_graphcomparison_status_deltas():
    """
    Validate GraphComparison computes status deltas correctly.

    Verifies:
    - Status count changes are calculated
    """
    node1 = VizNode(id="n1", type=NodeType.CODE.value, status=NodeStatus.PENDING.value, label="N1", color="#FF0000")
    node2 = VizNode(id="n2", type=NodeType.CODE.value, status=NodeStatus.VERIFIED.value, label="N2", color="#FF0000")

    baseline = GraphSnapshot(
        timestamp="2025-12-07T00:00:00Z",
        node_count=1,
        edge_count=0,
        nodes=[node1],
        edges=[],
    )

    treatment = GraphSnapshot(
        timestamp="2025-12-07T01:00:00Z",
        node_count=1,
        edge_count=0,
        nodes=[node2],
        edges=[],
    )

    comparison = GraphComparison(baseline=baseline, treatment=treatment)

    # PENDING: 1 -> 0 = -1 delta
    assert comparison.status_deltas[NodeStatus.PENDING.value] == -1
    # VERIFIED: 0 -> 1 = +1 delta
    assert comparison.status_deltas[NodeStatus.VERIFIED.value] == 1


def test_graphcomparison_edge_type_deltas():
    """
    Validate GraphComparison computes edge type deltas correctly.

    Verifies:
    - Edge type count changes are calculated
    """
    edge1 = VizEdge(source="n1", target="n2", type=EdgeType.IMPLEMENTS.value, color="#0000FF")
    edge2 = VizEdge(source="n2", target="n3", type=EdgeType.TRACES_TO.value, color="#0000FF")
    edge3 = VizEdge(source="n3", target="n4", type=EdgeType.TRACES_TO.value, color="#0000FF")

    baseline = GraphSnapshot(
        timestamp="2025-12-07T00:00:00Z",
        node_count=0,
        edge_count=2,
        nodes=[],
        edges=[edge1, edge2],
    )

    treatment = GraphSnapshot(
        timestamp="2025-12-07T01:00:00Z",
        node_count=0,
        edge_count=1,
        nodes=[],
        edges=[edge3],
    )

    comparison = GraphComparison(baseline=baseline, treatment=treatment)

    # IMPLEMENTS: 1 -> 0 = -1 delta
    assert comparison.edge_type_deltas[EdgeType.IMPLEMENTS.value] == -1
    # TRACES_TO: 1 -> 1 = 0 delta
    assert comparison.edge_type_deltas[EdgeType.TRACES_TO.value] == 0


def test_graphcomparison_to_dict():
    """
    Validate GraphComparison.to_dict() converts to dictionary correctly.

    Verifies:
    - All comparison fields are present
    - Nested snapshots are converted
    """
    node1 = VizNode(id="n1", type=NodeType.CODE.value, status=NodeStatus.VERIFIED.value, label="N1", color="#FF0000")

    baseline = GraphSnapshot(
        timestamp="2025-12-07T00:00:00Z",
        node_count=1,
        edge_count=0,
        nodes=[node1],
        edges=[],
    )

    treatment = GraphSnapshot(
        timestamp="2025-12-07T01:00:00Z",
        node_count=1,
        edge_count=0,
        nodes=[node1],
        edges=[],
    )

    comparison = GraphComparison(baseline=baseline, treatment=treatment)
    result = comparison.to_dict()

    assert isinstance(result, dict)
    assert "baseline" in result
    assert "treatment" in result
    assert "node_count_delta" in result
    assert "edge_count_delta" in result
    assert "type_deltas" in result
    assert "status_deltas" in result
    assert "edge_type_deltas" in result
    assert "nodes_only_in_baseline" in result
    assert "nodes_only_in_treatment" in result
    assert "nodes_in_both_count" in result


# =============================================================================
# COMPARE_SNAPSHOTS TESTS
# =============================================================================

def test_compare_snapshots_creates_comparison():
    """
    Validate compare_snapshots() creates GraphComparison correctly.

    Verifies:
    - Returns GraphComparison instance
    - Baseline and treatment are preserved
    """
    node1 = VizNode(id="n1", type=NodeType.CODE.value, status=NodeStatus.VERIFIED.value, label="N1", color="#FF0000")

    baseline = GraphSnapshot(
        timestamp="2025-12-07T00:00:00Z",
        node_count=1,
        edge_count=0,
        nodes=[node1],
        edges=[],
    )

    treatment = GraphSnapshot(
        timestamp="2025-12-07T01:00:00Z",
        node_count=1,
        edge_count=0,
        nodes=[node1],
        edges=[],
    )

    comparison = compare_snapshots(baseline, treatment)

    assert isinstance(comparison, GraphComparison)
    assert comparison.baseline == baseline
    assert comparison.treatment == treatment


# =============================================================================
# COLOR PALETTE TESTS
# =============================================================================

def test_node_colors_complete():
    """
    Validate NODE_COLORS contains all NodeType values.

    Verifies:
    - All node types have color mappings
    - Default color exists
    """
    assert "default" in NODE_COLORS

    # Check common node types
    assert NodeType.REQ.value in NODE_COLORS
    assert NodeType.CODE.value in NODE_COLORS
    assert NodeType.SPEC.value in NODE_COLORS
    assert NodeType.TEST.value in NODE_COLORS


def test_status_colors_complete():
    """
    Validate STATUS_COLORS contains all NodeStatus values.

    Verifies:
    - All status values have color mappings
    - Default color exists
    """
    assert "default" in STATUS_COLORS

    # Check common statuses
    assert NodeStatus.PENDING.value in STATUS_COLORS
    assert NodeStatus.VERIFIED.value in STATUS_COLORS
    assert NodeStatus.FAILED.value in STATUS_COLORS


def test_edge_colors_complete():
    """
    Validate EDGE_COLORS contains all EdgeType values.

    Verifies:
    - All edge types have color mappings
    - Default color exists
    """
    assert "default" in EDGE_COLORS

    # Check common edge types
    assert EdgeType.IMPLEMENTS.value in EDGE_COLORS
    assert EdgeType.TRACES_TO.value in EDGE_COLORS
    assert EdgeType.DEPENDS_ON.value in EDGE_COLORS


# =============================================================================
# MUTATIONTYPE ENUM TESTS
# =============================================================================

def test_mutation_type_enum_values():
    """
    Validate MutationType enum contains expected values.

    Verifies:
    - All expected mutation types are defined
    """
    assert MutationType.NODE_CREATED == "NODE_CREATED"
    assert MutationType.NODE_UPDATED == "NODE_UPDATED"
    assert MutationType.NODE_DELETED == "NODE_DELETED"
    assert MutationType.EDGE_CREATED == "EDGE_CREATED"
    assert MutationType.EDGE_DELETED == "EDGE_DELETED"
    assert MutationType.STATUS_CHANGED == "STATUS_CHANGED"
    assert MutationType.CONTEXT_PRUNED == "CONTEXT_PRUNED"
    assert MutationType.BATCH_UPDATE == "BATCH_UPDATE"
