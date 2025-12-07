"""
Unit tests for the Adversary module (EntropyModule).

Tests all 8 error types with deterministic seeding.
"""

import pytest
import json
import tempfile
from pathlib import Path

from forge.adversary import (
    EntropyModule,
    AdversaryConfig,
    Modification,
    Manifest,
    create_adversary,
)
from core.graph_db import ParagonDB
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, EdgeType, NodeStatus


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    db = ParagonDB()

    # Create nodes
    nodes = [
        NodeData.create(
            type=NodeType.REQ.value,
            content="Build a web scraper for product prices",
            created_by="test",
        ),
        NodeData.create(
            type=NodeType.SPEC.value,
            content="Fetch HTML from target website",
            status=NodeStatus.VERIFIED.value,
            created_by="test",
        ),
        NodeData.create(
            type=NodeType.SPEC.value,
            content="Parse HTML to extract prices",
            status=NodeStatus.PENDING.value,
            created_by="test",
        ),
        NodeData.create(
            type=NodeType.CODE.value,
            content="def fetch_html(url): return requests.get(url).text",
            status=NodeStatus.PENDING.value,
            created_by="test",
        ),
    ]

    db.add_nodes_batch(nodes)

    # Create edges
    edges = [
        EdgeData.create(
            source_id=nodes[1].id,
            target_id=nodes[0].id,
            type=EdgeType.TRACES_TO.value,
        ),
        EdgeData.create(
            source_id=nodes[2].id,
            target_id=nodes[1].id,
            type=EdgeType.DEPENDS_ON.value,
        ),
        EdgeData.create(
            source_id=nodes[3].id,
            target_id=nodes[1].id,
            type=EdgeType.IMPLEMENTS.value,
        ),
    ]

    for edge in edges:
        db.add_edge(edge, check_cycle=False)

    return db


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

def test_entropy_module_initialization():
    """Test basic initialization."""
    adversary = EntropyModule(seed=42)
    assert adversary.seed == 42
    assert adversary.world_id == "world_42"
    assert len(adversary.error_configs) == 0


def test_entropy_module_custom_world_id():
    """Test custom world ID."""
    adversary = EntropyModule(seed=42, world_id="test_world")
    assert adversary.world_id == "test_world"


def test_add_error_valid():
    """Test adding a valid error configuration."""
    adversary = EntropyModule(seed=42)
    config = AdversaryConfig(error_type="drop_edges", rate=0.05)
    adversary.add_error(config)
    assert len(adversary.error_configs) == 1


def test_add_error_invalid_type():
    """Test adding an invalid error type."""
    adversary = EntropyModule(seed=42)
    with pytest.raises(ValueError, match="Invalid error type"):
        adversary.add_error(AdversaryConfig(error_type="invalid_error", rate=0.05))


def test_add_error_invalid_rate():
    """Test adding an error with invalid rate."""
    adversary = EntropyModule(seed=42)
    with pytest.raises(ValueError, match="Error rate must be between"):
        adversary.add_error(AdversaryConfig(error_type="drop_edges", rate=1.5))


# =============================================================================
# ERROR TYPE TESTS
# =============================================================================

def test_drop_edges(sample_graph):
    """Test edge dropping."""
    original_edge_count = sample_graph.edge_count

    adversary = EntropyModule(seed=42)
    adversary.add_error(AdversaryConfig(error_type="drop_edges", rate=1.0))  # Drop all

    corrupted = adversary.corrupt(sample_graph)

    # Should have dropped all edges
    assert corrupted.edge_count == 0
    assert corrupted.node_count == sample_graph.node_count  # Nodes unchanged

    # Check manifest
    manifest = adversary.get_manifest()
    assert manifest.total_modifications == original_edge_count
    assert manifest.error_summary["drop_edges"] == original_edge_count


def test_drop_nodes(sample_graph):
    """Test node dropping."""
    original_node_count = sample_graph.node_count

    adversary = EntropyModule(seed=42)
    adversary.add_error(AdversaryConfig(error_type="drop_nodes", rate=0.5))

    corrupted = adversary.corrupt(sample_graph)

    # Should have dropped some nodes
    assert corrupted.node_count < original_node_count

    # Check manifest
    manifest = adversary.get_manifest()
    assert manifest.error_summary["drop_nodes"] > 0


def test_mutate_strings(sample_graph):
    """Test string mutation."""
    adversary = EntropyModule(seed=42)
    adversary.add_error(AdversaryConfig(
        error_type="mutate_strings",
        rate=1.0,  # Mutate all
        params={"property": "content", "mutation_type": "typo"}
    ))

    original_contents = [n.content for n in sample_graph.get_all_nodes()]
    corrupted = adversary.corrupt(sample_graph)
    corrupted_contents = [n.content for n in corrupted.get_all_nodes()]

    # Some contents should have changed
    assert original_contents != corrupted_contents

    # Check manifest
    manifest = adversary.get_manifest()
    assert manifest.error_summary["mutate_strings"] > 0


def test_mutate_numbers(sample_graph):
    """Test numeric mutation."""
    adversary = EntropyModule(seed=42)
    adversary.add_error(AdversaryConfig(
        error_type="mutate_numbers",
        rate=1.0,
        params={"property": "version", "noise_factor": 2.0}  # Larger noise to ensure changes
    ))

    original_versions = [n.version for n in sample_graph.get_all_nodes()]
    corrupted = adversary.corrupt(sample_graph)
    corrupted_versions = [n.version for n in corrupted.get_all_nodes()]

    # Some versions should have changed (with larger noise factor)
    # Due to rounding, we check that at least some have changed
    assert any(o != c for o, c in zip(original_versions, corrupted_versions))

    # Check manifest
    manifest = adversary.get_manifest()
    assert manifest.error_summary["mutate_numbers"] > 0


def test_swap_labels(sample_graph):
    """Test label swapping."""
    adversary = EntropyModule(seed=42)
    adversary.add_error(AdversaryConfig(error_type="swap_labels", rate=1.0))

    original_types = [n.type for n in sample_graph.get_all_nodes()]
    corrupted = adversary.corrupt(sample_graph)
    corrupted_types = [n.type for n in corrupted.get_all_nodes()]

    # Types should have been swapped (same set but different assignment)
    assert sorted(original_types) == sorted(corrupted_types)
    # But not in the same order
    assert original_types != corrupted_types

    # Check manifest
    manifest = adversary.get_manifest()
    assert manifest.error_summary["swap_labels"] > 0


def test_lag_timestamps(sample_graph):
    """Test timestamp lagging."""
    adversary = EntropyModule(seed=42)
    adversary.add_error(AdversaryConfig(
        error_type="lag_timestamps",
        rate=1.0,
        params={"lag_seconds": 3600}
    ))

    original_timestamps = [n.created_at for n in sample_graph.get_all_nodes()]
    corrupted = adversary.corrupt(sample_graph)
    corrupted_timestamps = [n.created_at for n in corrupted.get_all_nodes()]

    # Timestamps should have changed
    assert original_timestamps != corrupted_timestamps

    # Check manifest
    manifest = adversary.get_manifest()
    assert manifest.error_summary["lag_timestamps"] > 0


def test_duplicate_nodes(sample_graph):
    """Test node duplication."""
    original_node_count = sample_graph.node_count

    adversary = EntropyModule(seed=42)
    adversary.add_error(AdversaryConfig(error_type="duplicate_nodes", rate=0.5))

    corrupted = adversary.corrupt(sample_graph)

    # Should have more nodes
    assert corrupted.node_count > original_node_count

    # Check manifest
    manifest = adversary.get_manifest()
    assert manifest.error_summary["duplicate_nodes"] > 0
    # Verify metadata tracks original
    for mod in manifest.modifications:
        if mod.error_type == "duplicate_nodes":
            assert "duplicate_of" in mod.metadata


def test_null_properties(sample_graph):
    """Test property nullification."""
    adversary = EntropyModule(seed=42)
    adversary.add_error(AdversaryConfig(
        error_type="null_properties",
        rate=1.0,
        params={"properties": ["content"]}
    ))

    corrupted = adversary.corrupt(sample_graph)

    # All contents should be empty
    for node in corrupted.get_all_nodes():
        assert node.content == ""

    # Check manifest
    manifest = adversary.get_manifest()
    assert manifest.error_summary["null_properties"] > 0


# =============================================================================
# MANIFEST TESTS
# =============================================================================

def test_manifest_creation(sample_graph):
    """Test manifest structure."""
    adversary = EntropyModule(seed=42)
    adversary.add_error(AdversaryConfig(error_type="drop_edges", rate=0.5))
    corrupted = adversary.corrupt(sample_graph)

    manifest = adversary.get_manifest()

    assert manifest.world_id == "world_42"
    assert manifest.seed == 42
    assert manifest.total_modifications > 0
    assert len(manifest.modifications) == manifest.total_modifications
    assert "drop_edges" in manifest.error_summary


def test_manifest_serialization(sample_graph):
    """Test manifest JSON serialization."""
    adversary = EntropyModule(seed=42)
    adversary.add_error(AdversaryConfig(error_type="drop_edges", rate=0.5))
    corrupted = adversary.corrupt(sample_graph)

    manifest = adversary.get_manifest()

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        manifest.to_json(temp_path)

        # Load and verify
        loaded = Manifest.from_json(temp_path)
        assert loaded.world_id == manifest.world_id
        assert loaded.seed == manifest.seed
        assert loaded.total_modifications == manifest.total_modifications
    finally:
        Path(temp_path).unlink()


def test_modification_record(sample_graph):
    """Test individual modification records."""
    adversary = EntropyModule(seed=42)
    adversary.add_error(AdversaryConfig(
        error_type="mutate_strings",
        rate=1.0,
        params={"property": "content", "mutation_type": "typo"}
    ))

    corrupted = adversary.corrupt(sample_graph)
    manifest = adversary.get_manifest()

    # Verify modification structure
    for mod in manifest.modifications:
        assert mod.error_type == "mutate_strings"
        assert mod.target_type == "node"
        assert mod.target_id is not None
        assert mod.original_value is not None
        assert mod.corrupted_value is not None
        assert mod.timestamp is not None
        assert "property" in mod.metadata
        assert "mutation_type" in mod.metadata


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

def test_corruption_is_deterministic(sample_graph):
    """Test that same seed produces same corruption."""
    # First run
    adversary1 = EntropyModule(seed=42)
    adversary1.add_error(AdversaryConfig(error_type="drop_edges", rate=0.5))
    adversary1.add_error(AdversaryConfig(error_type="mutate_strings", rate=0.3,
                                        params={"property": "content"}))
    corrupted1 = adversary1.corrupt(sample_graph)
    manifest1 = adversary1.get_manifest()

    # Second run with same seed
    adversary2 = EntropyModule(seed=42)
    adversary2.add_error(AdversaryConfig(error_type="drop_edges", rate=0.5))
    adversary2.add_error(AdversaryConfig(error_type="mutate_strings", rate=0.3,
                                        params={"property": "content"}))
    corrupted2 = adversary2.corrupt(sample_graph)
    manifest2 = adversary2.get_manifest()

    # Should produce identical results
    assert manifest1.total_modifications == manifest2.total_modifications
    assert corrupted1.node_count == corrupted2.node_count
    assert corrupted1.edge_count == corrupted2.edge_count


def test_different_seeds_produce_different_results(sample_graph):
    """Test that different seeds produce different corruption."""
    adversary1 = EntropyModule(seed=42)
    adversary1.add_error(AdversaryConfig(error_type="drop_edges", rate=0.5))
    corrupted1 = adversary1.corrupt(sample_graph)

    adversary2 = EntropyModule(seed=123)
    adversary2.add_error(AdversaryConfig(error_type="drop_edges", rate=0.5))
    corrupted2 = adversary2.corrupt(sample_graph)

    # Results should differ
    assert adversary1.get_manifest().total_modifications != adversary2.get_manifest().total_modifications


# =============================================================================
# PRESET TESTS
# =============================================================================

def test_create_adversary_light():
    """Test light preset."""
    adversary = create_adversary(seed=42, preset="light")
    assert len(adversary.error_configs) > 0


def test_create_adversary_medium():
    """Test medium preset."""
    adversary = create_adversary(seed=42, preset="medium")
    assert len(adversary.error_configs) > 0


def test_create_adversary_heavy():
    """Test heavy preset."""
    adversary = create_adversary(seed=42, preset="heavy")
    assert len(adversary.error_configs) > 0


def test_create_adversary_realistic():
    """Test realistic preset."""
    adversary = create_adversary(seed=42, preset="realistic")
    assert len(adversary.error_configs) > 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_corrupt_empty_graph():
    """Test corruption of empty graph."""
    empty_db = ParagonDB()

    adversary = EntropyModule(seed=42)
    adversary.add_error(AdversaryConfig(error_type="drop_edges", rate=0.5))

    corrupted = adversary.corrupt(empty_db)

    assert corrupted.node_count == 0
    assert corrupted.edge_count == 0
    assert adversary.get_manifest().total_modifications == 0


def test_zero_rate_produces_no_changes(sample_graph):
    """Test that rate=0.0 produces no changes."""
    adversary = EntropyModule(seed=42)
    adversary.add_error(AdversaryConfig(error_type="drop_edges", rate=0.0))

    corrupted = adversary.corrupt(sample_graph)

    assert corrupted.node_count == sample_graph.node_count
    assert corrupted.edge_count == sample_graph.edge_count
    assert adversary.get_manifest().total_modifications == 0


def test_multiple_error_types_combined(sample_graph):
    """Test combining multiple error types."""
    adversary = EntropyModule(seed=42)
    adversary.add_error(AdversaryConfig(error_type="drop_edges", rate=0.2))
    adversary.add_error(AdversaryConfig(error_type="mutate_strings", rate=0.1,
                                       params={"property": "content"}))
    adversary.add_error(AdversaryConfig(error_type="duplicate_nodes", rate=0.1))

    corrupted = adversary.corrupt(sample_graph)
    manifest = adversary.get_manifest()

    # Should have modifications from all error types
    assert len(manifest.error_summary) > 1
    assert manifest.total_modifications > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_recovery_accuracy_grading(sample_graph):
    """Test using manifest to grade recovery accuracy."""
    # Corrupt the graph
    adversary = EntropyModule(seed=42)
    adversary.add_error(AdversaryConfig(
        error_type="mutate_strings",
        rate=0.5,
        params={"property": "content", "mutation_type": "typo"}
    ))

    corrupted = adversary.corrupt(sample_graph)
    manifest = adversary.get_manifest()

    # Simulate a recovery algorithm that fixes some errors
    # In reality, this would be a sophisticated recovery algorithm
    # For testing, we'll just check we can access the original values

    correct_recoveries = 0
    total_corruptions = 0

    for mod in manifest.modifications:
        if mod.error_type == "mutate_strings":
            total_corruptions += 1
            # In a real recovery algorithm, we would:
            # 1. Get the corrupted node
            # 2. Attempt to recover its original value
            # 3. Compare against mod.original_value
            # For now, just verify we have access to the answer key
            assert mod.original_value is not None
            assert mod.corrupted_value is not None

    assert total_corruptions > 0
