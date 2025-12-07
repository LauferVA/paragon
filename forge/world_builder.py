"""
WorldBuilder - Fluent API for Paragon.Forge

Provides a unified, chainable interface for generating complete synthetic graph datasets.

Usage:
    from forge import WorldBuilder, Themes, Topologies

    world = (WorldBuilder(seed=42)
        .set_topology(Topologies.BARABASI_ALBERT, nodes=10_000, m=3)
        .apply_theme(Themes.LOGISTICS)
        .add_node_property("latency", distribution="lognormal", mean=20, sigma=5)
        .add_adversary("drop_edges", rate=0.05)
        .add_adversary("mutate_strings", rate=0.02)
        .build())

    # Access results
    graph = world.graph
    manifest = world.manifest  # Answer key for corruptions
    stats = world.stats

Design Philosophy:
    - NO PYDANTIC: All schemas use msgspec.Struct
    - FLUENT API: Method chaining for readability
    - COMPOSABLE: Each step is optional and order-independent (mostly)
    - DETERMINISTIC: Same seed = same world
"""

from typing import Any, Dict, List, Optional, Union
import msgspec
import time
import rustworkx as rx

# Local imports
from core.graph_db import ParagonDB
from core.schemas import NodeData, EdgeData
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
from forge.linguist import MaskingLayer, Themes, ThemeConfig
from forge.statistician import DistributionEngine, DistributionSpec
from forge.adversary import EntropyModule, AdversaryConfig, Manifest


class WorldStats(msgspec.Struct, frozen=True):
    """Statistics about the generated world."""
    node_count: int
    edge_count: int
    topology_type: str
    theme_applied: Optional[str]
    properties_added: int
    corruptions_applied: int
    generation_time_ms: float
    seed: int


class WorldResult(msgspec.Struct):
    """Result of world generation."""
    graph: Any  # rustworkx PyDiGraph
    manifest: Optional[Manifest]  # Answer key for corruptions
    stats: WorldStats
    config: Dict[str, Any]  # Full configuration used


class WorldBuilder:
    """
    Fluent builder for synthetic graph worlds.

    Chains together:
    1. Topologist - Graph structure generation
    2. Statistician - Property distributions
    3. Linguist - Domain theming
    4. Adversary - Controlled corruption

    Example:
        world = (WorldBuilder(seed=42)
            .set_topology(Topologies.BARABASI_ALBERT, nodes=10_000)
            .apply_theme(Themes.NETWORK)
            .add_node_property("cpu_usage", distribution="uniform", low=0.1, high=0.9)
            .add_adversary("drop_nodes", rate=0.01)
            .build())
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize WorldBuilder.

        Args:
            seed: Master seed for reproducibility. If None, uses current time.
        """
        self._seed = seed if seed is not None else int(time.time())

        # Configuration storage
        self._topology_config: Optional[TopologyConfig] = None
        self._theme: Optional[Union[Themes, ThemeConfig]] = None
        self._node_properties: List[DistributionSpec] = []
        self._edge_properties: List[DistributionSpec] = []
        self._adversary_configs: List[AdversaryConfig] = []

        # State
        self._graph = None
        self._built = False

    # =========================================================================
    # TOPOLOGY (Step 1)
    # =========================================================================

    def set_topology(
        self,
        topology: Union[str, Topologies],
        nodes: int,
        **kwargs
    ) -> "WorldBuilder":
        """
        Set the graph topology.

        Args:
            topology: Topology type (Topologies enum or string)
            nodes: Number of nodes
            **kwargs: Topology-specific parameters:
                - Erdos-Renyi: edge_probability (default: 0.1)
                - Barabasi-Albert: m (edges per new node, default: 3)
                - Watts-Strogatz: k (neighbors), rewire_prob (default: 0.3)
                - Grid: cols (default: sqrt(nodes))

        Returns:
            self for chaining
        """
        if isinstance(topology, Topologies):
            topo_type = topology.value
        else:
            topo_type = topology

        # Build config based on topology type
        if topo_type == "erdos_renyi":
            self._topology_config = TopologyConfig(
                topology_type=topo_type,
                num_nodes=nodes,
                edge_probability=kwargs.get("edge_probability", 0.1),
            )
        elif topo_type == "barabasi_albert":
            self._topology_config = TopologyConfig(
                topology_type=topo_type,
                num_nodes=nodes,
                num_edges_per_node=kwargs.get("m", kwargs.get("num_edges_per_node", 3)),
            )
        elif topo_type == "watts_strogatz":
            self._topology_config = TopologyConfig(
                topology_type=topo_type,
                num_nodes=nodes,
                k_neighbors=kwargs.get("k", kwargs.get("k_neighbors", 4)),
                rewire_prob=kwargs.get("rewire_prob", 0.3),
            )
        elif topo_type == "grid":
            import math
            cols = kwargs.get("cols", int(math.sqrt(nodes)))
            rows = nodes // cols
            self._topology_config = TopologyConfig(
                topology_type=topo_type,
                num_nodes=rows * cols,
                grid_rows=rows,
                grid_cols=cols,
            )
        else:
            raise ValueError(f"Unknown topology: {topo_type}")

        return self

    # Convenience topology methods
    def scale_free(self, nodes: int, m: int = 3) -> "WorldBuilder":
        """Create scale-free (Barabasi-Albert) topology."""
        return self.set_topology(Topologies.BARABASI_ALBERT, nodes, m=m)

    def random(self, nodes: int, edge_probability: float = 0.1) -> "WorldBuilder":
        """Create random (Erdos-Renyi) topology."""
        return self.set_topology(Topologies.ERDOS_RENYI, nodes, edge_probability=edge_probability)

    def small_world(self, nodes: int, k: int = 4, rewire: float = 0.3) -> "WorldBuilder":
        """Create small-world (Watts-Strogatz) topology."""
        return self.set_topology(Topologies.WATTS_STROGATZ, nodes, k=k, rewire_prob=rewire)

    def grid(self, rows: int, cols: int) -> "WorldBuilder":
        """Create grid topology."""
        return self.set_topology(Topologies.GRID, rows * cols, grid_rows=rows, grid_cols=cols)

    # =========================================================================
    # THEME (Step 2)
    # =========================================================================

    def apply_theme(self, theme: Union[Themes, ThemeConfig]) -> "WorldBuilder":
        """
        Apply a domain theme to the graph.

        Args:
            theme: Built-in theme (Themes enum) or custom ThemeConfig

        Returns:
            self for chaining
        """
        self._theme = theme
        return self

    # Convenience theme methods
    def genomics(self) -> "WorldBuilder":
        """Apply genomics theme."""
        return self.apply_theme(Themes.GENOMICS)

    def logistics(self) -> "WorldBuilder":
        """Apply logistics theme."""
        return self.apply_theme(Themes.LOGISTICS)

    def social(self) -> "WorldBuilder":
        """Apply social network theme."""
        return self.apply_theme(Themes.SOCIAL)

    def finance(self) -> "WorldBuilder":
        """Apply finance theme."""
        return self.apply_theme(Themes.FINANCE)

    def network(self) -> "WorldBuilder":
        """Apply network infrastructure theme."""
        return self.apply_theme(Themes.NETWORK)

    # =========================================================================
    # PROPERTIES (Step 3)
    # =========================================================================

    def add_node_property(
        self,
        name: str,
        distribution: str = "normal",
        **params
    ) -> "WorldBuilder":
        """
        Add a statistical property to nodes.

        Args:
            name: Property name
            distribution: Distribution type (normal, lognormal, uniform, etc.)
            **params: Distribution parameters (mean, std, low, high, etc.)

        Returns:
            self for chaining
        """
        # Convert params to float values for DistributionSpec
        float_params = {k: float(v) for k, v in params.items()}
        spec = DistributionSpec(
            name=name,
            distribution=distribution,
            params=float_params,
            target="nodes",
        )
        self._node_properties.append(spec)
        return self

    def add_edge_property(
        self,
        name: str,
        distribution: str = "normal",
        **params
    ) -> "WorldBuilder":
        """
        Add a statistical property to edges.

        Args:
            name: Property name
            distribution: Distribution type
            **params: Distribution parameters

        Returns:
            self for chaining
        """
        # Convert params to float values for DistributionSpec
        float_params = {k: float(v) for k, v in params.items()}
        spec = DistributionSpec(
            name=name,
            distribution=distribution,
            params=float_params,
            target="edges",
        )
        self._edge_properties.append(spec)
        return self

    # =========================================================================
    # ADVERSARY (Step 4)
    # =========================================================================

    def add_adversary(
        self,
        error_type: str,
        rate: float,
        **params
    ) -> "WorldBuilder":
        """
        Add controlled corruption.

        Args:
            error_type: Type of error (drop_edges, drop_nodes, mutate_strings, etc.)
            rate: Probability of corruption (0.0 to 1.0)
            **params: Error-specific parameters

        Returns:
            self for chaining
        """
        config = AdversaryConfig(
            error_type=error_type,
            rate=rate,
            params=params if params else {},
        )
        self._adversary_configs.append(config)
        return self

    # Convenience adversary methods
    def with_packet_loss(self, rate: float = 0.05) -> "WorldBuilder":
        """Add edge dropping (packet loss simulation)."""
        return self.add_adversary("drop_edges", rate)

    def with_node_failures(self, rate: float = 0.01) -> "WorldBuilder":
        """Add node dropping (failure simulation)."""
        return self.add_adversary("drop_nodes", rate)

    def with_typos(self, rate: float = 0.02, property: str = "label") -> "WorldBuilder":
        """Add string mutations (typo simulation)."""
        return self.add_adversary("mutate_strings", rate, property=property, mutation_type="typo")

    def with_noise(self, rate: float = 0.1, property: str = "value", magnitude: float = 0.1) -> "WorldBuilder":
        """Add numeric noise."""
        return self.add_adversary("mutate_numbers", rate, property=property, noise_magnitude=magnitude)

    def with_missing_data(self, rate: float = 0.05, properties: Optional[List[str]] = None) -> "WorldBuilder":
        """Add null properties (missing data simulation)."""
        return self.add_adversary("null_properties", rate, properties=properties or [])

    # =========================================================================
    # BUILD
    # =========================================================================

    def _convert_to_paragon_db(self, rx_graph: rx.PyDiGraph) -> ParagonDB:
        """Convert a raw rustworkx graph to ParagonDB format."""
        db = ParagonDB()

        # Map old indices to new node IDs
        index_to_id = {}

        # Add nodes
        for idx in rx_graph.node_indices():
            node_data = rx_graph[idx]
            # Handle both dict and None node data
            if node_data is None:
                node_data = {}
            elif not isinstance(node_data, dict):
                node_data = {"value": node_data}

            node_id = f"node_{idx}"
            node = NodeData(
                id=node_id,
                type="GENERATED",
                content=node_data.get("label", f"Node {idx}"),
                data=node_data,
            )
            db.add_node(node)
            # Store the string UUID, not the rustworkx index
            index_to_id[idx] = node_id

        # Add edges
        for source, target, raw_edge_data in rx_graph.edge_index_map().values():
            if raw_edge_data is None:
                raw_edge_data = {}
            elif not isinstance(raw_edge_data, dict):
                raw_edge_data = {"value": raw_edge_data}

            edge = EdgeData(
                source_id=index_to_id[source],
                target_id=index_to_id[target],
                type="CONNECTS",
                metadata=raw_edge_data,
            )
            db.add_edge(edge, check_cycle=False)

        return db

    def build(self) -> WorldResult:
        """
        Build the synthetic world.

        Executes the pipeline:
        1. Generate topology (Topologist)
        2. Apply statistical properties (Statistician)
        3. Convert to ParagonDB
        4. Apply theme (Linguist)
        5. Apply corruption (Adversary)

        Returns:
            WorldResult with graph, manifest, stats, and config
        """
        start_time = time.time()

        # Validate
        if self._topology_config is None:
            raise ValueError("Topology not set. Call set_topology() or a convenience method first.")

        # Step 1: Generate topology (returns raw rustworkx graph)
        generator = GraphGenerator(seed=self._seed)
        rx_graph = generator.generate(self._topology_config)

        # Step 2: Apply statistical properties (works on raw rustworkx)
        if self._node_properties or self._edge_properties:
            stats_engine = DistributionEngine(seed=self._seed + 1)
            for prop_spec in self._node_properties:
                stats_engine.add_property(prop_spec)
            for prop_spec in self._edge_properties:
                stats_engine.add_property(prop_spec)
            rx_graph = stats_engine.apply(rx_graph)

        # Step 3: Convert to ParagonDB for Linguist/Adversary
        graph = self._convert_to_paragon_db(rx_graph)

        # Step 4: Apply theme
        theme_name = None
        if self._theme is not None:
            masker = MaskingLayer(seed=self._seed + 2)
            graph = masker.apply(graph, theme=self._theme)
            if isinstance(self._theme, Themes):
                theme_name = self._theme.value
            elif hasattr(self._theme, 'name'):
                theme_name = self._theme.name

        # Step 5: Apply adversary
        manifest = None
        if self._adversary_configs:
            adversary = EntropyModule(seed=self._seed + 3)
            for config in self._adversary_configs:
                adversary.add_error(config)
            graph = adversary.corrupt(graph)
            manifest = adversary.get_manifest()

        # Compute stats
        elapsed_ms = (time.time() - start_time) * 1000
        stats = WorldStats(
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            topology_type=self._topology_config.topology_type,
            theme_applied=theme_name,
            properties_added=len(self._node_properties) + len(self._edge_properties),
            corruptions_applied=manifest.total_modifications if manifest else 0,
            generation_time_ms=elapsed_ms,
            seed=self._seed,
        )

        # Build config dict
        config = {
            "seed": self._seed,
            "topology": msgspec.to_builtins(self._topology_config),
            "theme": theme_name,
            "node_properties": [msgspec.to_builtins(p) for p in self._node_properties],
            "edge_properties": [msgspec.to_builtins(p) for p in self._edge_properties],
            "adversary_configs": [msgspec.to_builtins(c) for c in self._adversary_configs],
        }

        self._built = True
        self._graph = graph

        return WorldResult(
            graph=graph,
            manifest=manifest,
            stats=stats,
            config=config,
        )

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def clone(self) -> "WorldBuilder":
        """Create a copy of this builder with same configuration."""
        new = WorldBuilder(seed=self._seed)
        new._topology_config = self._topology_config
        new._theme = self._theme
        new._node_properties = list(self._node_properties)
        new._edge_properties = list(self._edge_properties)
        new._adversary_configs = list(self._adversary_configs)
        return new

    def with_seed(self, seed: int) -> "WorldBuilder":
        """Create a copy with a different seed."""
        clone = self.clone()
        clone._seed = seed
        return clone


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_world(
    topology: Union[str, Topologies],
    nodes: int,
    theme: Optional[Union[Themes, ThemeConfig]] = None,
    seed: Optional[int] = None,
    **topology_params
) -> WorldResult:
    """
    Quick factory for common world generation.

    Args:
        topology: Topology type
        nodes: Number of nodes
        theme: Optional theme to apply
        seed: Random seed
        **topology_params: Topology-specific parameters

    Returns:
        WorldResult with generated graph
    """
    builder = WorldBuilder(seed=seed).set_topology(topology, nodes, **topology_params)
    if theme:
        builder.apply_theme(theme)
    return builder.build()


def create_test_dataset(
    nodes: int = 1000,
    theme: Themes = Themes.NETWORK,
    corruption_rate: float = 0.05,
    seed: int = 42,
) -> WorldResult:
    """
    Create a standard test dataset with corruption.

    Args:
        nodes: Number of nodes (default: 1000)
        theme: Domain theme (default: NETWORK)
        corruption_rate: Overall corruption rate (default: 0.05)
        seed: Random seed (default: 42)

    Returns:
        WorldResult with corrupted graph and manifest
    """
    return (WorldBuilder(seed=seed)
        .scale_free(nodes, m=3)
        .apply_theme(theme)
        .add_adversary("drop_edges", rate=corruption_rate)
        .add_adversary("mutate_strings", rate=corruption_rate / 2, property="label")
        .add_adversary("null_properties", rate=corruption_rate / 2)
        .build())


def create_benchmark_dataset(
    nodes: int = 10000,
    topology: Topologies = Topologies.BARABASI_ALBERT,
    seed: int = 42,
) -> WorldResult:
    """
    Create a large benchmark dataset (no corruption).

    Args:
        nodes: Number of nodes (default: 10000)
        topology: Topology type (default: Barabasi-Albert)
        seed: Random seed

    Returns:
        WorldResult with clean graph
    """
    return (WorldBuilder(seed=seed)
        .set_topology(topology, nodes, m=3)
        .add_node_property("latency", distribution="lognormal", mean=20, sigma=5)
        .add_node_property("throughput", distribution="uniform", low=100, high=1000)
        .build())
