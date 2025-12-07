#!/usr/bin/env python3
"""
TESTS: Paragon.Forge Linguist

Quick validation tests for the MaskingLayer.

Usage:
    python forge/test_linguist.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph_db import ParagonDB
from core.schemas import NodeData, EdgeData
from forge.linguist import (
    MaskingLayer,
    Themes,
    ThemeConfig,
    list_available_themes,
)


def test_basic_transformation():
    """Test basic graph transformation."""
    print("TEST 1: Basic transformation")

    # Create simple graph
    db = ParagonDB()
    node1 = NodeData.create(type="ENTITY", content="Test node 1")
    node2 = NodeData.create(type="ENTITY", content="Test node 2")
    db.add_node(node1)
    db.add_node(node2)
    db.add_edge(EdgeData.depends_on(node2.id, node1.id), check_cycle=False)

    # Apply theme
    masker = MaskingLayer(seed=42)
    result = masker.apply(db, theme=Themes.GENOMICS)

    # Verify structure preserved
    assert result.node_count == 2, f"Expected 2 nodes, got {result.node_count}"
    assert result.edge_count == 1, f"Expected 1 edge, got {result.edge_count}"

    # Verify properties exist
    for node in result.get_all_nodes():
        assert "label" in node.data, "Node missing 'label' property"
        assert "type" in node.data, "Node missing 'type' property"

    print("  ✓ PASSED")


def test_all_themes():
    """Test that all built-in themes work."""
    print("\nTEST 2: All built-in themes")

    # Create simple graph
    db = ParagonDB()
    node1 = NodeData.create(type="ENTITY", content="Test")
    db.add_node(node1)

    themes = list_available_themes()
    for theme_name in themes:
        print(f"  Testing theme: {theme_name}...")
        masker = MaskingLayer(seed=42)
        result = masker.apply(db, theme=theme_name)
        assert result.node_count == 1, f"Theme {theme_name} failed"

    print("  ✓ PASSED")


def test_custom_theme():
    """Test custom theme creation."""
    print("\nTEST 3: Custom theme")

    # Create graph
    db = ParagonDB()
    node1 = NodeData.create(type="ENTITY", content="Test")
    db.add_node(node1)

    # Create custom theme
    custom = ThemeConfig(
        theme_name="test_theme",
        node_name_pattern="Node_{id}",
        edge_name_pattern="edge",
        node_properties={"name": "name", "value": "random_int"},
        edge_properties={"weight": "random_int"},
    )

    # Apply theme
    masker = MaskingLayer(seed=42)
    result = masker.apply(db, theme=custom)

    # Verify custom properties
    node = list(result.iter_nodes())[0]
    assert "name" in node.data, "Custom property 'name' missing"
    assert "value" in node.data, "Custom property 'value' missing"

    print("  ✓ PASSED")


def test_determinism():
    """Test that same seed produces same results."""
    print("\nTEST 4: Deterministic transformation")

    # Create graph
    db = ParagonDB()
    node1 = NodeData.create(type="ENTITY", content="Test")
    db.add_node(node1)

    # Apply theme twice with same seed
    masker1 = MaskingLayer(seed=42)
    result1 = masker1.apply(db, theme=Themes.GENOMICS)

    masker2 = MaskingLayer(seed=42)
    result2 = masker2.apply(db, theme=Themes.GENOMICS)

    # Compare results
    node1 = list(result1.iter_nodes())[0]
    node2 = list(result2.iter_nodes())[0]

    assert node1.data["label"] == node2.data["label"], "Labels differ with same seed"

    print("  ✓ PASSED")


def test_preserve_original():
    """Test original data preservation."""
    print("\nTEST 5: Preserve original data")

    # Create graph
    db = ParagonDB()
    node1 = NodeData.create(type="ENTITY", content="Original content")
    db.add_node(node1)

    # Apply theme with preservation
    masker = MaskingLayer(seed=42)
    result = masker.apply(db, theme=Themes.GENOMICS, preserve_original=True)

    # Verify original data preserved
    node = list(result.iter_nodes())[0]
    assert "original" in node.data, "Original data not preserved"
    assert node.data["original"]["type"] == "ENTITY", "Original type incorrect"

    print("  ✓ PASSED")


def test_structure_preservation():
    """Test that complex graph structure is preserved."""
    print("\nTEST 6: Structure preservation")

    # Create complex graph
    db = ParagonDB()
    nodes = [NodeData.create(type="ENTITY", content=f"Node {i}") for i in range(5)]
    for node in nodes:
        db.add_node(node)

    # Create various edge patterns
    edges = [
        EdgeData.depends_on(nodes[1].id, nodes[0].id),
        EdgeData.depends_on(nodes[2].id, nodes[0].id),
        EdgeData.depends_on(nodes[3].id, nodes[1].id),
        EdgeData.depends_on(nodes[3].id, nodes[2].id),
        EdgeData.depends_on(nodes[4].id, nodes[3].id),
    ]
    for edge in edges:
        db.add_edge(edge, check_cycle=False)

    # Apply theme
    masker = MaskingLayer(seed=42)
    result = masker.apply(db, theme=Themes.NETWORK)

    # Verify structure
    assert result.node_count == 5, "Node count changed"
    assert result.edge_count == 5, "Edge count changed"

    # Verify it's still a DAG
    assert not result.has_cycle(), "Introduced cycles"

    # Verify waves are same depth
    original_waves = db.get_waves()
    themed_waves = result.get_waves()
    assert len(original_waves) == len(themed_waves), "Wave structure changed"

    print("  ✓ PASSED")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*70)
    print("PARAGON.FORGE LINGUIST - TEST SUITE")
    print("="*70 + "\n")

    try:
        test_basic_transformation()
        test_all_themes()
        test_custom_theme()
        test_determinism()
        test_preserve_original()
        test_structure_preservation()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70 + "\n")
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
