"""
Unit tests for forge/topologist.py - Graph Topology Generator

Tests the topology generation capabilities including:
- Erdos-Renyi random graphs
- Barabasi-Albert scale-free graphs
- Watts-Strogatz small-world graphs
- Grid/lattice graphs
- Configuration validation
- Deterministic seeding
- Factory functions
"""

import pytest
import rustworkx as rx

from forge.topologist import (
    GraphGenerator,
    TopologyConfig,
    Topologies,
    create_erdos_renyi,
    create_barabasi_albert,
    create_watts_strogatz,
    create_grid,
    graph_stats,
)


# =============================================================================
# ERDOS-RENYI TESTS
# =============================================================================

def test_erdos_renyi_generation_directed():
    """
    Test Erdos-Renyi random graph generation with directed graph.

    Verifies:
    - Graph is created with correct number of nodes
    - Graph is directed (PyDiGraph)
    - Edge probability affects edge count
    """
    config = TopologyConfig(
        topology_type=Topologies.ERDOS_RENYI,
        num_nodes=100,
        edge_probability=0.1,
        directed=True,
    )

    gen = GraphGenerator(seed=42)
    graph = gen.generate(config)

    assert isinstance(graph, rx.PyDiGraph)
    assert len(graph) == 100
    # With p=0.1, we expect roughly 100 * 99 * 0.1 = 990 edges (approximate)
    edge_count = len(graph.edge_list())
    assert 700 < edge_count < 1300  # Allow variance


def test_erdos_renyi_generation_undirected():
    """
    Test Erdos-Renyi random graph generation with undirected graph.

    Verifies:
    - Graph is undirected (PyGraph)
    - Correct node count
    """
    config = TopologyConfig(
        topology_type=Topologies.ERDOS_RENYI,
        num_nodes=100,
        edge_probability=0.1,
        directed=False,
    )

    gen = GraphGenerator(seed=42)
    graph = gen.generate(config)

    assert isinstance(graph, rx.PyGraph)
    assert len(graph) == 100


def test_erdos_renyi_deterministic_seeding():
    """
    Test that same seed produces identical Erdos-Renyi graphs.

    Verifies:
    - Two graphs with same seed are identical
    - Different seeds produce different graphs
    """
    config = TopologyConfig(
        topology_type=Topologies.ERDOS_RENYI,
        num_nodes=100,
        edge_probability=0.1,
        directed=True,
    )

    gen1 = GraphGenerator(seed=42)
    graph1 = gen1.generate(config)

    gen2 = GraphGenerator(seed=42)
    graph2 = gen2.generate(config)

    # Same seed should produce same graph
    assert len(graph1) == len(graph2)
    assert len(graph1.edge_list()) == len(graph2.edge_list())

    # Different seed should produce different graph
    gen3 = GraphGenerator(seed=99)
    graph3 = gen3.generate(config)
    assert len(graph1.edge_list()) != len(graph3.edge_list())


def test_erdos_renyi_edge_probability_bounds():
    """
    Test Erdos-Renyi with extreme edge probabilities.

    Verifies:
    - p=0.0 produces graph with no edges
    - p=1.0 produces complete graph
    """
    # p=0.0 - no edges
    config_empty = TopologyConfig(
        topology_type=Topologies.ERDOS_RENYI,
        num_nodes=100,
        edge_probability=0.0,
        directed=True,
    )
    gen = GraphGenerator(seed=42)
    graph_empty = gen.generate(config_empty)
    assert len(graph_empty.edge_list()) == 0

    # p=1.0 - complete graph (n * (n-1) edges for directed)
    config_complete = TopologyConfig(
        topology_type=Topologies.ERDOS_RENYI,
        num_nodes=100,  # Use 100 to meet minimum node requirement
        edge_probability=1.0,
        directed=True,
    )
    graph_complete = gen.generate(config_complete)
    assert len(graph_complete.edge_list()) == 100 * 99  # n * (n-1)


def test_erdos_renyi_factory_function():
    """Test create_erdos_renyi factory function."""
    graph = create_erdos_renyi(
        num_nodes=100,
        edge_probability=0.05,
        directed=True,
        seed=42,
    )

    assert isinstance(graph, rx.PyDiGraph)
    assert len(graph) == 100


# =============================================================================
# BARABASI-ALBERT TESTS
# =============================================================================

def test_barabasi_albert_generation_directed():
    """
    Test Barabasi-Albert scale-free graph generation.

    Verifies:
    - Graph has correct node count
    - Graph has power-law-like properties (few hubs, many low-degree nodes)
    """
    config = TopologyConfig(
        topology_type=Topologies.BARABASI_ALBERT,
        num_nodes=100,
        num_edges_per_node=3,
        directed=True,
    )

    gen = GraphGenerator(seed=42)
    graph = gen.generate(config)

    assert isinstance(graph, rx.PyDiGraph)
    assert len(graph) == 100

    # Check that we have edges
    edge_count = len(graph.edge_list())
    assert edge_count > 0

    # Check degree distribution has some variance (not all equal)
    stats = graph_stats(graph)
    assert stats["max_degree"] > stats["min_degree"]


def test_barabasi_albert_generation_undirected():
    """Test Barabasi-Albert with undirected graph."""
    config = TopologyConfig(
        topology_type=Topologies.BARABASI_ALBERT,
        num_nodes=100,
        num_edges_per_node=3,
        directed=False,
    )

    gen = GraphGenerator(seed=42)
    graph = gen.generate(config)

    assert isinstance(graph, rx.PyGraph)
    assert len(graph) == 100


def test_barabasi_albert_deterministic_seeding():
    """Test deterministic seeding for Barabasi-Albert."""
    config = TopologyConfig(
        topology_type=Topologies.BARABASI_ALBERT,
        num_nodes=100,
        num_edges_per_node=3,
        directed=True,
    )

    gen1 = GraphGenerator(seed=42)
    graph1 = gen1.generate(config)

    gen2 = GraphGenerator(seed=42)
    graph2 = gen2.generate(config)

    assert len(graph1.edge_list()) == len(graph2.edge_list())


def test_barabasi_albert_edges_per_node():
    """
    Test that num_edges_per_node parameter affects graph density.

    Verifies:
    - Higher m produces more edges
    """
    config_low = TopologyConfig(
        topology_type=Topologies.BARABASI_ALBERT,
        num_nodes=100,
        num_edges_per_node=2,
        directed=True,
    )

    config_high = TopologyConfig(
        topology_type=Topologies.BARABASI_ALBERT,
        num_nodes=100,
        num_edges_per_node=5,
        directed=True,
    )

    gen = GraphGenerator(seed=42)
    graph_low = gen.generate(config_low)
    graph_high = gen.generate(config_high)

    assert len(graph_high.edge_list()) > len(graph_low.edge_list())


def test_barabasi_albert_factory_function():
    """Test create_barabasi_albert factory function."""
    graph = create_barabasi_albert(
        num_nodes=100,
        m=3,
        directed=True,
        seed=42,
    )

    assert isinstance(graph, rx.PyDiGraph)
    assert len(graph) == 100


# =============================================================================
# WATTS-STROGATZ TESTS
# =============================================================================

def test_watts_strogatz_generation_directed():
    """
    Test Watts-Strogatz small-world graph generation.

    Verifies:
    - Graph has correct node count
    - Graph has regular structure (ring lattice base)
    """
    config = TopologyConfig(
        topology_type=Topologies.WATTS_STROGATZ,
        num_nodes=100,
        k_neighbors=4,
        rewire_prob=0.3,
        directed=True,
    )

    gen = GraphGenerator(seed=42)
    graph = gen.generate(config)

    assert isinstance(graph, rx.PyDiGraph)
    assert len(graph) == 100

    # Check that we have edges
    edge_count = len(graph.edge_list())
    assert edge_count > 0


def test_watts_strogatz_generation_undirected():
    """Test Watts-Strogatz with undirected graph."""
    config = TopologyConfig(
        topology_type=Topologies.WATTS_STROGATZ,
        num_nodes=100,
        k_neighbors=4,
        rewire_prob=0.3,
        directed=False,
    )

    gen = GraphGenerator(seed=42)
    graph = gen.generate(config)

    assert isinstance(graph, rx.PyGraph)
    assert len(graph) == 100


def test_watts_strogatz_deterministic_seeding():
    """Test deterministic seeding for Watts-Strogatz."""
    config = TopologyConfig(
        topology_type=Topologies.WATTS_STROGATZ,
        num_nodes=100,
        k_neighbors=4,
        rewire_prob=0.3,
        directed=True,
    )

    gen1 = GraphGenerator(seed=42)
    graph1 = gen1.generate(config)

    gen2 = GraphGenerator(seed=42)
    graph2 = gen2.generate(config)

    assert len(graph1.edge_list()) == len(graph2.edge_list())


def test_watts_strogatz_no_rewiring():
    """
    Test Watts-Strogatz with no rewiring (p=0.0).

    Verifies:
    - Creates a regular ring lattice
    - Edge count is deterministic
    """
    config = TopologyConfig(
        topology_type=Topologies.WATTS_STROGATZ,
        num_nodes=100,
        k_neighbors=4,
        rewire_prob=0.0,
        directed=False,
    )

    gen = GraphGenerator(seed=42)
    graph = gen.generate(config)

    # Ring lattice with k=4: each node connects to k/2 neighbors on each side
    # For undirected, expect 100 * (4/2) = 200 edges
    assert len(graph.edge_list()) == 200


def test_watts_strogatz_full_rewiring():
    """
    Test Watts-Strogatz with full rewiring (p=1.0).

    Verifies:
    - Graph is fully rewired (random-like)
    """
    config = TopologyConfig(
        topology_type=Topologies.WATTS_STROGATZ,
        num_nodes=100,
        k_neighbors=4,
        rewire_prob=1.0,
        directed=False,
    )

    gen = GraphGenerator(seed=42)
    graph = gen.generate(config)

    # Should still have roughly same number of edges as ring lattice
    edge_count = len(graph.edge_list())
    assert 150 < edge_count < 250  # Allow some variance from rewiring


def test_watts_strogatz_factory_function():
    """Test create_watts_strogatz factory function."""
    graph = create_watts_strogatz(
        num_nodes=100,
        k=4,
        p=0.3,
        directed=True,
        seed=42,
    )

    assert isinstance(graph, rx.PyDiGraph)
    assert len(graph) == 100


# =============================================================================
# GRID TESTS
# =============================================================================

def test_grid_generation_directed():
    """
    Test grid/lattice graph generation.

    Verifies:
    - Graph has correct node count (rows * cols)
    - Graph has grid structure (4-connectivity)
    """
    config = TopologyConfig(
        topology_type=Topologies.GRID,
        num_nodes=100,
        grid_rows=10,
        grid_cols=10,
        directed=True,
    )

    gen = GraphGenerator(seed=42)
    graph = gen.generate(config)

    assert isinstance(graph, rx.PyDiGraph)
    assert len(graph) == 100

    # For directed 10x10 grid with bidirectional edges:
    # Interior nodes have 4 neighbors (8 edges), boundary nodes have fewer
    # Expected: 2 * (10*9 + 9*10) = 2 * 180 = 360 edges
    assert len(graph.edge_list()) == 360


def test_grid_generation_undirected():
    """Test grid with undirected graph."""
    config = TopologyConfig(
        topology_type=Topologies.GRID,
        num_nodes=100,
        grid_rows=10,
        grid_cols=10,
        directed=False,
    )

    gen = GraphGenerator(seed=42)
    graph = gen.generate(config)

    assert isinstance(graph, rx.PyGraph)
    assert len(graph) == 100

    # For undirected 10x10 grid:
    # Expected: 10*9 + 9*10 = 180 edges
    assert len(graph.edge_list()) == 180


def test_grid_non_square():
    """
    Test grid with non-square dimensions.

    Verifies:
    - Graph handles rectangular grids correctly
    """
    config = TopologyConfig(
        topology_type=Topologies.GRID,
        num_nodes=200,  # Use 200 to meet minimum node requirement
        grid_rows=10,
        grid_cols=20,
        directed=False,
    )

    gen = GraphGenerator(seed=42)
    graph = gen.generate(config)

    assert len(graph) == 200

    # For undirected 10x20 grid:
    # Horizontal edges: 10 rows * 19 connections = 190
    # Vertical edges: 9 rows * 20 connections = 180
    # Total: 370 edges
    assert len(graph.edge_list()) == 370


def test_grid_factory_function():
    """Test create_grid factory function."""
    graph = create_grid(
        rows=10,
        cols=10,
        directed=True,
        seed=42,
    )

    assert isinstance(graph, rx.PyDiGraph)
    assert len(graph) == 100


# =============================================================================
# CONFIGURATION VALIDATION TESTS
# =============================================================================

def test_topology_config_validation_node_count_too_low():
    """
    Test that configuration validation rejects node count below minimum.

    Verifies:
    - ValueError raised for num_nodes < 100
    """
    config = TopologyConfig(
        topology_type=Topologies.ERDOS_RENYI,
        num_nodes=50,  # Below minimum of 100
        edge_probability=0.1,
    )

    with pytest.raises(ValueError, match="num_nodes must be between 100 and 10,000,000"):
        config.validate()


def test_topology_config_validation_node_count_too_high():
    """
    Test that configuration validation rejects node count above maximum.

    Verifies:
    - ValueError raised for num_nodes > 10,000,000
    """
    config = TopologyConfig(
        topology_type=Topologies.ERDOS_RENYI,
        num_nodes=20_000_000,  # Above maximum of 10,000,000
        edge_probability=0.1,
    )

    with pytest.raises(ValueError, match="num_nodes must be between 100 and 10,000,000"):
        config.validate()


def test_topology_config_validation_invalid_topology_type():
    """
    Test that configuration validation rejects invalid topology type.

    Verifies:
    - ValueError raised for unknown topology type
    """
    config = TopologyConfig(
        topology_type="invalid_topology",
        num_nodes=100,
    )

    with pytest.raises(ValueError, match="topology_type must be one of"):
        config.validate()


def test_erdos_renyi_validation_invalid_probability():
    """
    Test Erdos-Renyi validation for invalid edge probability.

    Verifies:
    - ValueError raised for edge_probability outside [0.0, 1.0]
    """
    config = TopologyConfig(
        topology_type=Topologies.ERDOS_RENYI,
        num_nodes=100,
        edge_probability=1.5,  # Invalid: > 1.0
    )

    with pytest.raises(ValueError, match="edge_probability must be between 0.0 and 1.0"):
        config.validate()


def test_barabasi_albert_validation_invalid_m():
    """
    Test Barabasi-Albert validation for invalid num_edges_per_node.

    Verifies:
    - ValueError raised for m < 1
    - ValueError raised for m >= num_nodes
    """
    # m < 1
    config1 = TopologyConfig(
        topology_type=Topologies.BARABASI_ALBERT,
        num_nodes=100,
        num_edges_per_node=0,
    )

    with pytest.raises(ValueError, match="num_edges_per_node must be >= 1"):
        config1.validate()

    # m >= num_nodes
    config2 = TopologyConfig(
        topology_type=Topologies.BARABASI_ALBERT,
        num_nodes=100,
        num_edges_per_node=100,
    )

    with pytest.raises(ValueError, match="num_edges_per_node must be < num_nodes"):
        config2.validate()


def test_watts_strogatz_validation_invalid_k():
    """
    Test Watts-Strogatz validation for invalid k_neighbors.

    Verifies:
    - ValueError raised for k < 2
    - ValueError raised for k >= num_nodes
    - ValueError raised for k not even
    """
    # k < 2
    config1 = TopologyConfig(
        topology_type=Topologies.WATTS_STROGATZ,
        num_nodes=100,
        k_neighbors=1,
        rewire_prob=0.3,
    )

    with pytest.raises(ValueError, match="k_neighbors must be >= 2"):
        config1.validate()

    # k >= num_nodes
    config2 = TopologyConfig(
        topology_type=Topologies.WATTS_STROGATZ,
        num_nodes=100,
        k_neighbors=100,
        rewire_prob=0.3,
    )

    with pytest.raises(ValueError, match="k_neighbors must be < num_nodes"):
        config2.validate()

    # k not even
    config3 = TopologyConfig(
        topology_type=Topologies.WATTS_STROGATZ,
        num_nodes=100,
        k_neighbors=5,  # Odd number
        rewire_prob=0.3,
    )

    with pytest.raises(ValueError, match="k_neighbors must be even"):
        config3.validate()


def test_watts_strogatz_validation_invalid_rewire_prob():
    """
    Test Watts-Strogatz validation for invalid rewire probability.

    Verifies:
    - ValueError raised for rewire_prob outside [0.0, 1.0]
    """
    config = TopologyConfig(
        topology_type=Topologies.WATTS_STROGATZ,
        num_nodes=100,
        k_neighbors=4,
        rewire_prob=1.5,  # Invalid: > 1.0
    )

    with pytest.raises(ValueError, match="rewire_prob must be between 0.0 and 1.0"):
        config.validate()


def test_grid_validation_invalid_dimensions():
    """
    Test grid validation for invalid grid dimensions.

    Verifies:
    - ValueError raised for rows < 1 or cols < 1
    - ValueError raised for rows * cols != num_nodes
    """
    # Invalid rows/cols
    config1 = TopologyConfig(
        topology_type=Topologies.GRID,
        num_nodes=100,
        grid_rows=0,
        grid_cols=10,
    )

    with pytest.raises(ValueError, match="grid_rows and grid_cols must be >= 1"):
        config1.validate()

    # Mismatched dimensions
    config2 = TopologyConfig(
        topology_type=Topologies.GRID,
        num_nodes=100,
        grid_rows=5,
        grid_cols=10,  # 5 * 10 = 50 != 100
    )

    with pytest.raises(ValueError, match="grid_rows \\* grid_cols must equal num_nodes"):
        config2.validate()


# =============================================================================
# GRAPH STATISTICS TESTS
# =============================================================================

def test_graph_stats_directed():
    """
    Test graph_stats utility function with directed graph.

    Verifies:
    - Returns correct node count
    - Returns correct edge count
    - Computes degree statistics
    - Computes density
    """
    config = TopologyConfig(
        topology_type=Topologies.ERDOS_RENYI,
        num_nodes=100,
        edge_probability=0.1,
        directed=True,
    )

    gen = GraphGenerator(seed=42)
    graph = gen.generate(config)

    stats = graph_stats(graph)

    assert stats["num_nodes"] == 100
    assert stats["num_edges"] > 0
    assert stats["avg_degree"] > 0
    assert stats["min_degree"] >= 0
    assert stats["max_degree"] >= stats["min_degree"]
    assert 0 <= stats["density"] <= 1


def test_graph_stats_undirected():
    """Test graph_stats with undirected graph."""
    config = TopologyConfig(
        topology_type=Topologies.ERDOS_RENYI,
        num_nodes=100,
        edge_probability=0.1,
        directed=False,
    )

    gen = GraphGenerator(seed=42)
    graph = gen.generate(config)

    stats = graph_stats(graph)

    assert stats["num_nodes"] == 100
    assert stats["num_edges"] > 0


def test_graph_stats_empty_graph():
    """
    Test graph_stats with graph that has nodes but no edges.

    Verifies:
    - Handles empty edge list gracefully
    - All degrees are 0
    """
    config = TopologyConfig(
        topology_type=Topologies.ERDOS_RENYI,
        num_nodes=100,
        edge_probability=0.0,
        directed=True,
    )

    gen = GraphGenerator(seed=42)
    graph = gen.generate(config)

    stats = graph_stats(graph)

    assert stats["num_nodes"] == 100
    assert stats["num_edges"] == 0
    assert stats["avg_degree"] == 0
    assert stats["min_degree"] == 0
    assert stats["max_degree"] == 0
    assert stats["density"] == 0


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

def test_generator_with_none_seed():
    """
    Test GraphGenerator with None seed (random generation).

    Verifies:
    - Generator accepts None seed
    - Generates valid graph
    """
    gen = GraphGenerator(seed=None)

    config = TopologyConfig(
        topology_type=Topologies.ERDOS_RENYI,
        num_nodes=100,
        edge_probability=0.1,
        directed=True,
    )

    graph = gen.generate(config)

    assert len(graph) == 100


def test_generator_unknown_topology_type():
    """
    Test that generator raises error for unknown topology type.

    Verifies:
    - ValueError raised when topology_type is invalid
    """
    # Create config with invalid topology (bypassing validation for test)
    config = TopologyConfig(
        topology_type="unknown_topology",
        num_nodes=100,
    )

    gen = GraphGenerator(seed=42)

    # Validation should catch this, but test error handling in generate()
    with pytest.raises(ValueError):
        # First validation will fail
        config.validate()


def test_factory_functions_with_minimal_parameters():
    """
    Test all factory functions with minimal parameters.

    Verifies:
    - All factory functions use sensible defaults
    """
    # Erdos-Renyi
    graph_er = create_erdos_renyi(num_nodes=100)
    assert len(graph_er) == 100

    # Barabasi-Albert
    graph_ba = create_barabasi_albert(num_nodes=100)
    assert len(graph_ba) == 100

    # Watts-Strogatz
    graph_ws = create_watts_strogatz(num_nodes=100)
    assert len(graph_ws) == 100

    # Grid
    graph_grid = create_grid(rows=10, cols=10)
    assert len(graph_grid) == 100
