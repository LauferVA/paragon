#!/usr/bin/env python3
"""
DEMONSTRATION: Paragon.Forge Linguist

This script demonstrates the MaskingLayer's ability to transform
generic graphs into domain-specific datasets.

Usage:
    python forge/demo_linguist.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph_db import ParagonDB
from core.schemas import NodeData, EdgeData
from forge.linguist import MaskingLayer, Themes, ThemeConfig, list_available_themes


def create_sample_graph() -> ParagonDB:
    """
    Create a generic graph structure for demonstration.

    This represents a simple dependency graph that could be
    from any domain - we'll transform it with different themes.
    """
    db = ParagonDB()

    # Create nodes
    node1 = NodeData.create(type="ENTITY", content="Root entity")
    node2 = NodeData.create(type="ENTITY", content="Child entity A")
    node3 = NodeData.create(type="ENTITY", content="Child entity B")
    node4 = NodeData.create(type="ENTITY", content="Grandchild entity")
    node5 = NodeData.create(type="RESOURCE", content="Shared resource")

    # Add nodes to graph
    for node in [node1, node2, node3, node4, node5]:
        db.add_node(node)

    # Create edges (dependencies)
    edges = [
        EdgeData.depends_on(node2.id, node1.id),  # node2 depends on node1
        EdgeData.depends_on(node3.id, node1.id),  # node3 depends on node1
        EdgeData.depends_on(node4.id, node2.id),  # node4 depends on node2
        EdgeData.depends_on(node4.id, node3.id),  # node4 depends on both node2 and node3
        EdgeData.create(node5.id, node1.id, type="USES"),  # node5 uses node1
        EdgeData.create(node5.id, node4.id, type="USES"),  # node5 uses node4
    ]

    for edge in edges:
        db.add_edge(edge, check_cycle=False)

    return db


def demo_genomics_theme():
    """Demonstrate GENOMICS theme."""
    print("\n" + "="*70)
    print("DEMONSTRATION: GENOMICS THEME")
    print("="*70)

    # Create sample graph
    print("\n1. Creating generic graph structure...")
    graph = create_sample_graph()
    print(f"   - Nodes: {graph.node_count}")
    print(f"   - Edges: {graph.edge_count}")

    # Apply genomics theme
    print("\n2. Applying GENOMICS theme...")
    masker = MaskingLayer(seed=42)
    themed_graph = masker.apply(graph, theme=Themes.GENOMICS, preserve_original=True)

    # Display results
    print("\n3. Results:")
    print("\n   NODES:")
    for node in themed_graph.get_all_nodes():
        label = node.data.get("label", "N/A")
        organism = node.data.get("organism", "N/A")
        print(f"   - {label}")
        print(f"     Type: {node.data.get('type', 'N/A')}")
        print(f"     Organism: {organism}")
        print(f"     Original: {node.data.get('original', {}).get('type', 'N/A')}")
        print()

    print("   EDGES:")
    for edge in themed_graph.get_all_edges():
        src_node = themed_graph.get_node(edge.source_id)
        tgt_node = themed_graph.get_node(edge.target_id)
        src_label = src_node.data.get("label", "N/A")
        tgt_label = tgt_node.data.get("label", "N/A")
        print(f"   - {src_label} --[{edge.type}]--> {tgt_label}")
        if "confidence_score" in edge.metadata:
            print(f"     Confidence: {edge.metadata['confidence_score']}")


def demo_logistics_theme():
    """Demonstrate LOGISTICS theme."""
    print("\n" + "="*70)
    print("DEMONSTRATION: LOGISTICS THEME")
    print("="*70)

    # Create sample graph
    print("\n1. Creating generic graph structure...")
    graph = create_sample_graph()

    # Apply logistics theme
    print("\n2. Applying LOGISTICS theme...")
    masker = MaskingLayer(seed=123)
    themed_graph = masker.apply(graph, theme=Themes.LOGISTICS)

    # Display results
    print("\n3. Results:")
    print("\n   NODES:")
    for node in themed_graph.get_all_nodes():
        label = node.data.get("label", "N/A")
        location = node.data.get("location", "N/A")
        capacity = node.data.get("capacity", "N/A")
        print(f"   - {label}")
        print(f"     Location: {location}")
        print(f"     Capacity: {capacity}")
        print()

    print("   EDGES:")
    for edge in themed_graph.get_all_edges():
        src_node = themed_graph.get_node(edge.source_id)
        tgt_node = themed_graph.get_node(edge.target_id)
        src_label = src_node.data.get("label", "N/A")
        tgt_label = tgt_node.data.get("label", "N/A")
        print(f"   - {src_label} --[{edge.type}]--> {tgt_label}")
        if "distance_km" in edge.metadata:
            print(f"     Distance: {edge.metadata['distance_km']} km")


def demo_custom_theme():
    """Demonstrate custom theme creation."""
    print("\n" + "="*70)
    print("DEMONSTRATION: CUSTOM THEME")
    print("="*70)

    # Create sample graph
    print("\n1. Creating generic graph structure...")
    graph = create_sample_graph()

    # Create custom theme
    print("\n2. Creating custom ACADEMIC theme...")
    custom_theme = ThemeConfig(
        theme_name="academic",
        description="Academic research domain",
        node_name_pattern="Researcher_{name}",
        edge_name_pattern="{edge_type}",
        node_properties={
            "name": "name",
            "email": "email",
            "institution": "company",
            "field": "job",
            "h_index": "random_int",
        },
        edge_properties={
            "collaboration_count": "random_int",
            "joint_papers": "random_int",
        },
    )

    # Apply custom theme
    print("\n3. Applying custom theme...")
    masker = MaskingLayer(seed=456)
    themed_graph = masker.apply(graph, theme=custom_theme)

    # Display results
    print("\n4. Results:")
    print("\n   NODES:")
    for node in themed_graph.get_all_nodes():
        label = node.data.get("label", "N/A")
        institution = node.data.get("institution", "N/A")
        field = node.data.get("field", "N/A")
        print(f"   - {label}")
        print(f"     Institution: {institution}")
        print(f"     Field: {field}")
        print()


def demo_all_themes():
    """Show a quick overview of all themes."""
    print("\n" + "="*70)
    print("AVAILABLE THEMES")
    print("="*70)

    print("\nBuilt-in themes:")
    for theme_name in list_available_themes():
        print(f"  - {theme_name}")

    print("\n" + "="*70)


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "PARAGON.FORGE LINGUIST DEMO" + " "*26 + "║")
    print("║" + " "*12 + "Semantic Masking for Graph Datasets" + " "*20 + "║")
    print("╚" + "="*68 + "╝")

    # Show available themes
    demo_all_themes()

    # Run theme demonstrations
    demo_genomics_theme()
    demo_logistics_theme()
    demo_custom_theme()

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nThe same graph structure has been transformed into three")
    print("different domain-specific datasets while preserving topology.")
    print()


if __name__ == "__main__":
    main()
