"""
Paragon.Forge - Synthetic Graph Dataset Generator

The Forge module transforms generic graph structures into domain-specific datasets
by applying semantic masking and realistic data generation.

Components:
- topologist: Graph skeleton generation (Erdos-Renyi, Barabasi-Albert, etc.)
- linguist: Semantic masking and theme application
- statistician: Statistical distribution and correlation engine
- adversary: Controlled corruption engine for testing recovery algorithms

Design Philosophy:
1. NO PYDANTIC: All schemas use msgspec.Struct
2. GRAPH-NATIVE: Works directly with rustworkx graphs
3. DETERMINISTIC: Reproducible via seeds
4. COMPOSABLE: Engines can be chained and combined
"""

from forge.linguist import MaskingLayer, Themes, ThemeConfig, Theme
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
from forge.statistician import (
    DistributionSpec,
    CorrelationSpec,
    DistributionEngine,
    apply_normal,
    apply_lognormal,
    apply_uniform,
    validate_distribution_spec,
)
from forge.adversary import (
    EntropyModule,
    AdversaryConfig,
    Modification,
    Manifest,
    create_adversary,
)
from forge.world_builder import (
    WorldBuilder,
    WorldResult,
    WorldStats,
    create_world,
    create_test_dataset,
    create_benchmark_dataset,
)

__all__ = [
    # Linguist exports
    "MaskingLayer",
    "Themes",
    "ThemeConfig",
    "Theme",
    # Topologist exports
    "GraphGenerator",
    "TopologyConfig",
    "Topologies",
    "create_erdos_renyi",
    "create_barabasi_albert",
    "create_watts_strogatz",
    "create_grid",
    "graph_stats",
    # Statistician exports
    "DistributionSpec",
    "CorrelationSpec",
    "DistributionEngine",
    "apply_normal",
    "apply_lognormal",
    "apply_uniform",
    "validate_distribution_spec",
    # Adversary exports
    "EntropyModule",
    "AdversaryConfig",
    "Modification",
    "Manifest",
    "create_adversary",
    # WorldBuilder exports
    "WorldBuilder",
    "WorldResult",
    "WorldStats",
    "create_world",
    "create_test_dataset",
    "create_benchmark_dataset",
]
