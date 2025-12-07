"""
PARAGON.FORGE - The Statistician (Agent B)
Distribution Engine for Graph Property Hydration

This module provides statistical distribution capabilities for rustworkx graphs,
enabling probabilistic properties with multivariate correlations.

Architecture:
- Layer 1: Distribution Specs (msgspec.Struct schemas)
- Layer 2: Distribution Engine (applies statistical properties)
- Layer 3: Correlation Engine (multivariate dependencies)

Design Principles:
1. NO PYDANTIC: All schemas use msgspec.Struct
2. GRAPH-NATIVE: Works directly with rustworkx.PyDiGraph
3. DETERMINISTIC SEED: Reproducible random generation
4. VECTORIZED: Uses numpy for batch operations

Performance Characteristics:
- O(N) for applying distributions to N nodes/edges
- O(C*N) for applying C correlations to N nodes
- In-place graph modification (zero-copy where possible)

Usage:
    from forge.statistician import DistributionEngine, DistributionSpec

    engine = DistributionEngine(seed=42)

    # Add latency distribution to edges
    engine.add_property(DistributionSpec(
        name="latency",
        distribution="lognormal",
        params={"mean": 20, "sigma": 2},
        target="edges"
    ))

    # Apply to graph
    hydrated_graph = engine.apply(graph)
"""

import msgspec
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Union, Literal
import rustworkx as rx
from scipy import stats


# =============================================================================
# SCHEMAS (msgspec.Struct - NO Pydantic)
# =============================================================================

class DistributionSpec(msgspec.Struct, kw_only=True, frozen=True):
    """
    Specification for a statistical distribution to apply to graph properties.

    Defines what property to create, what distribution to use, and where to apply it.

    Attributes:
        name: Property name to create (e.g., "latency", "age", "failure_rate")
        distribution: Distribution type - one of:
            - "normal": Normal/Gaussian distribution
            - "lognormal": Log-normal distribution (always positive)
            - "uniform": Uniform distribution
            - "exponential": Exponential distribution
            - "poisson": Poisson distribution (integer values)
            - "binomial": Binomial distribution (integer values)
            - "beta": Beta distribution (values in [0, 1])
            - "custom": Custom callable distribution
        params: Distribution parameters (varies by distribution type)
            - normal: {"mean": float, "std": float}
            - lognormal: {"mean": float, "sigma": float}
            - uniform: {"low": float, "high": float}
            - exponential: {"scale": float}
            - poisson: {"lam": float}
            - binomial: {"n": int, "p": float}
            - beta: {"a": float, "b": float}
        target: Where to apply - "nodes" or "edges"
        custom_fn: Optional custom distribution function (samples, seed) -> array
    """
    name: str
    distribution: Literal[
        "normal", "lognormal", "uniform", "exponential",
        "poisson", "binomial", "beta", "custom"
    ]
    params: Dict[str, float]
    target: Literal["nodes", "edges"] = "nodes"


class CorrelationSpec(msgspec.Struct, kw_only=True, frozen=True):
    """
    Specification for a multivariate correlation between properties.

    Defines conditional relationships: "If property X satisfies condition Y,
    then modify property Z".

    Example:
        "If age > 60, then failure_rate *= 2.0"
        CorrelationSpec(
            condition_property="age",
            condition_op=">",
            condition_value=60,
            effect_property="failure_rate",
            effect_modifier="multiply",
            effect_value=2.0
        )

    Attributes:
        condition_property: Property name to check (e.g., "age")
        condition_op: Comparison operator: ">", "<", "==", ">=", "<=", "!="
        condition_value: Value to compare against
        effect_property: Property name to modify (e.g., "failure_rate")
        effect_modifier: How to modify - "multiply", "add", "set", "power"
        effect_value: Value to use in modification
        target: Where to apply - "nodes" or "edges"
    """
    condition_property: str
    condition_op: Literal[">", "<", "==", ">=", "<=", "!="]
    condition_value: float
    effect_property: str
    effect_modifier: Literal["multiply", "add", "set", "power"]
    effect_value: float
    target: Literal["nodes", "edges"] = "nodes"


# =============================================================================
# DISTRIBUTION ENGINE
# =============================================================================

class DistributionEngine:
    """
    Engine for applying statistical distributions to graph properties.

    Supports:
    1. Standard statistical distributions (normal, lognormal, etc.)
    2. Custom callable distributions
    3. Multivariate correlations
    4. Reproducible random generation (via seed)

    Architecture:
    - Builds up a list of distribution specs and correlation specs
    - Applies them in order when apply() is called
    - Modifies graph in-place (stores properties in node/edge data dicts)

    Performance:
    - Vectorized operations via numpy
    - Single pass through graph per distribution
    - O(N) for N nodes/edges
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the distribution engine.

        Args:
            seed: Random seed for reproducibility. If None, uses system entropy.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Specs to apply
        self.distribution_specs: List[DistributionSpec] = []
        self.correlation_specs: List[CorrelationSpec] = []

        # Custom distribution functions
        self.custom_distributions: Dict[str, Callable] = {}

    def add_property(self, spec: DistributionSpec) -> "DistributionEngine":
        """
        Add a distribution specification to apply.

        Args:
            spec: Distribution specification

        Returns:
            Self for method chaining
        """
        self.distribution_specs.append(spec)
        return self

    def add_correlation(self, spec: CorrelationSpec) -> "DistributionEngine":
        """
        Add a correlation specification to apply.

        Args:
            spec: Correlation specification

        Returns:
            Self for method chaining
        """
        self.correlation_specs.append(spec)
        return self

    def register_custom_distribution(
        self,
        name: str,
        fn: Callable[[int, np.random.Generator], np.ndarray]
    ) -> "DistributionEngine":
        """
        Register a custom distribution function.

        Args:
            name: Name to reference in DistributionSpec
            fn: Function with signature (n_samples, rng) -> array of samples

        Returns:
            Self for method chaining

        Example:
            def my_dist(n, rng):
                return rng.exponential(scale=10, size=n) + 5

            engine.register_custom_distribution("my_dist", my_dist)
            engine.add_property(DistributionSpec(
                name="custom_prop",
                distribution="custom",
                params={"custom_name": "my_dist"},
                target="nodes"
            ))
        """
        self.custom_distributions[name] = fn
        return self

    def apply(self, graph: rx.PyDiGraph) -> rx.PyDiGraph:
        """
        Apply all distributions and correlations to the graph.

        This is the main entry point. It:
        1. Applies all distribution specs (adds properties)
        2. Applies all correlation specs (modifies properties)

        Args:
            graph: rustworkx graph to hydrate

        Returns:
            The modified graph (modified in-place, but returned for chaining)

        Raises:
            ValueError: If distribution type is unknown or params are invalid
        """
        # Apply distributions
        for spec in self.distribution_specs:
            self._apply_distribution(graph, spec)

        # Apply correlations
        for spec in self.correlation_specs:
            self._apply_correlation(graph, spec)

        return graph

    def _apply_distribution(self, graph: rx.PyDiGraph, spec: DistributionSpec) -> None:
        """
        Apply a single distribution spec to the graph.

        Args:
            graph: Graph to modify
            spec: Distribution specification
        """
        # Determine target collection
        if spec.target == "nodes":
            indices = graph.node_indices()
            get_data = graph.get_node_data
            # Note: rustworkx doesn't have set_node_data, so we modify in-place
        else:  # edges
            indices = graph.edge_indices()
            get_data = graph.get_edge_data

        # Generate samples
        n_samples = len(indices)
        if n_samples == 0:
            return

        samples = self._generate_samples(spec, n_samples)

        # Apply to graph
        for idx, value in zip(indices, samples):
            data = get_data(idx)

            # Store property in data dict
            # For NodeData/EdgeData, check if it has a 'data' attribute
            if hasattr(data, 'data') and isinstance(data.data, dict):
                data.data[spec.name] = float(value)
            # Fallback: if data is a dict directly
            elif isinstance(data, dict):
                data[spec.name] = float(value)
            # Fallback: add as attribute if possible
            else:
                try:
                    setattr(data, spec.name, float(value))
                except AttributeError:
                    # Last resort: create a data dict if the object supports it
                    if not hasattr(data, 'data'):
                        data.data = {}
                    data.data[spec.name] = float(value)

    def _apply_correlation(self, graph: rx.PyDiGraph, spec: CorrelationSpec) -> None:
        """
        Apply a single correlation spec to the graph.

        Args:
            graph: Graph to modify
            spec: Correlation specification
        """
        # Determine target collection
        if spec.target == "nodes":
            indices = graph.node_indices()
            get_data = graph.get_node_data
        else:  # edges
            indices = graph.edge_indices()
            get_data = graph.get_edge_data

        # Apply correlation condition-by-condition
        for idx in indices:
            data = get_data(idx)

            # Extract condition property value
            condition_value = self._get_property(data, spec.condition_property)
            if condition_value is None:
                continue

            # Check condition
            if not self._check_condition(
                condition_value,
                spec.condition_op,
                spec.condition_value
            ):
                continue

            # Extract effect property value
            effect_value = self._get_property(data, spec.effect_property)
            if effect_value is None:
                continue

            # Apply effect
            new_value = self._apply_effect(
                effect_value,
                spec.effect_modifier,
                spec.effect_value
            )

            # Store modified value
            self._set_property(data, spec.effect_property, new_value)

    def _generate_samples(self, spec: DistributionSpec, n_samples: int) -> np.ndarray:
        """
        Generate random samples from the specified distribution.

        Args:
            spec: Distribution specification
            n_samples: Number of samples to generate

        Returns:
            Array of samples

        Raises:
            ValueError: If distribution type is unknown or params are invalid
        """
        dist_type = spec.distribution
        params = spec.params

        if dist_type == "normal":
            return self.rng.normal(
                loc=params["mean"],
                scale=params["std"],
                size=n_samples
            )

        elif dist_type == "lognormal":
            return self.rng.lognormal(
                mean=params["mean"],
                sigma=params["sigma"],
                size=n_samples
            )

        elif dist_type == "uniform":
            return self.rng.uniform(
                low=params["low"],
                high=params["high"],
                size=n_samples
            )

        elif dist_type == "exponential":
            return self.rng.exponential(
                scale=params["scale"],
                size=n_samples
            )

        elif dist_type == "poisson":
            return self.rng.poisson(
                lam=params["lam"],
                size=n_samples
            ).astype(float)

        elif dist_type == "binomial":
            return self.rng.binomial(
                n=int(params["n"]),
                p=params["p"],
                size=n_samples
            ).astype(float)

        elif dist_type == "beta":
            return self.rng.beta(
                a=params["a"],
                b=params["b"],
                size=n_samples
            )

        elif dist_type == "custom":
            custom_name = params.get("custom_name")
            if not custom_name or custom_name not in self.custom_distributions:
                raise ValueError(
                    f"Custom distribution '{custom_name}' not registered. "
                    f"Use register_custom_distribution() first."
                )
            fn = self.custom_distributions[custom_name]
            return fn(n_samples, self.rng)

        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    def _get_property(self, data: Any, prop_name: str) -> Optional[float]:
        """
        Extract a property value from node/edge data.

        Args:
            data: Node or edge data object
            prop_name: Property name to extract

        Returns:
            Property value as float, or None if not found
        """
        # Try data.data dict first (NodeData/EdgeData)
        if hasattr(data, 'data') and isinstance(data.data, dict):
            if prop_name in data.data:
                return float(data.data[prop_name])

        # Try direct dict access
        if isinstance(data, dict):
            if prop_name in data:
                return float(data[prop_name])

        # Try attribute access
        if hasattr(data, prop_name):
            return float(getattr(data, prop_name))

        return None

    def _set_property(self, data: Any, prop_name: str, value: float) -> None:
        """
        Set a property value in node/edge data.

        Args:
            data: Node or edge data object
            prop_name: Property name to set
            value: Value to set
        """
        # Try data.data dict first (NodeData/EdgeData)
        if hasattr(data, 'data') and isinstance(data.data, dict):
            data.data[prop_name] = value
            return

        # Try direct dict access
        if isinstance(data, dict):
            data[prop_name] = value
            return

        # Try attribute access
        try:
            setattr(data, prop_name, value)
        except AttributeError:
            # Last resort: create data dict
            if not hasattr(data, 'data'):
                data.data = {}
            data.data[prop_name] = value

    def _check_condition(
        self,
        value: float,
        op: str,
        threshold: float
    ) -> bool:
        """
        Check if a condition is satisfied.

        Args:
            value: Value to check
            op: Comparison operator
            threshold: Threshold value

        Returns:
            True if condition is satisfied
        """
        if op == ">":
            return value > threshold
        elif op == "<":
            return value < threshold
        elif op == "==":
            return abs(value - threshold) < 1e-9  # Float equality tolerance
        elif op == ">=":
            return value >= threshold
        elif op == "<=":
            return value <= threshold
        elif op == "!=":
            return abs(value - threshold) >= 1e-9
        else:
            raise ValueError(f"Unknown operator: {op}")

    def _apply_effect(
        self,
        value: float,
        modifier: str,
        effect_value: float
    ) -> float:
        """
        Apply an effect modifier to a value.

        Args:
            value: Current value
            modifier: Modification type
            effect_value: Value to use in modification

        Returns:
            Modified value
        """
        if modifier == "multiply":
            return value * effect_value
        elif modifier == "add":
            return value + effect_value
        elif modifier == "set":
            return effect_value
        elif modifier == "power":
            return value ** effect_value
        else:
            raise ValueError(f"Unknown modifier: {modifier}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def apply_normal(
    graph: rx.PyDiGraph,
    property_name: str,
    mean: float,
    std: float,
    target: Literal["nodes", "edges"] = "nodes",
    seed: Optional[int] = None
) -> rx.PyDiGraph:
    """
    Convenience function to apply a normal distribution to a graph.

    Args:
        graph: Graph to modify
        property_name: Name of property to create
        mean: Mean of distribution
        std: Standard deviation
        target: "nodes" or "edges"
        seed: Random seed

    Returns:
        Modified graph
    """
    engine = DistributionEngine(seed=seed)
    engine.add_property(DistributionSpec(
        name=property_name,
        distribution="normal",
        params={"mean": mean, "std": std},
        target=target
    ))
    return engine.apply(graph)


def apply_lognormal(
    graph: rx.PyDiGraph,
    property_name: str,
    mean: float,
    sigma: float,
    target: Literal["nodes", "edges"] = "nodes",
    seed: Optional[int] = None
) -> rx.PyDiGraph:
    """
    Convenience function to apply a lognormal distribution to a graph.

    Args:
        graph: Graph to modify
        property_name: Name of property to create
        mean: Mean of underlying normal distribution
        sigma: Standard deviation of underlying normal distribution
        target: "nodes" or "edges"
        seed: Random seed

    Returns:
        Modified graph
    """
    engine = DistributionEngine(seed=seed)
    engine.add_property(DistributionSpec(
        name=property_name,
        distribution="lognormal",
        params={"mean": mean, "sigma": sigma},
        target=target
    ))
    return engine.apply(graph)


def apply_uniform(
    graph: rx.PyDiGraph,
    property_name: str,
    low: float,
    high: float,
    target: Literal["nodes", "edges"] = "nodes",
    seed: Optional[int] = None
) -> rx.PyDiGraph:
    """
    Convenience function to apply a uniform distribution to a graph.

    Args:
        graph: Graph to modify
        property_name: Name of property to create
        low: Lower bound (inclusive)
        high: Upper bound (exclusive)
        target: "nodes" or "edges"
        seed: Random seed

    Returns:
        Modified graph
    """
    engine = DistributionEngine(seed=seed)
    engine.add_property(DistributionSpec(
        name=property_name,
        distribution="uniform",
        params={"low": low, "high": high},
        target=target
    ))
    return engine.apply(graph)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_distribution_spec(spec: DistributionSpec) -> List[str]:
    """
    Validate a distribution specification.

    Args:
        spec: Distribution spec to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Validate required params for each distribution type
    required_params = {
        "normal": {"mean", "std"},
        "lognormal": {"mean", "sigma"},
        "uniform": {"low", "high"},
        "exponential": {"scale"},
        "poisson": {"lam"},
        "binomial": {"n", "p"},
        "beta": {"a", "b"},
        "custom": {"custom_name"},
    }

    dist_type = spec.distribution
    if dist_type not in required_params:
        errors.append(f"Unknown distribution type: {dist_type}")
        return errors

    missing_params = required_params[dist_type] - set(spec.params.keys())
    if missing_params:
        errors.append(
            f"Missing required params for {dist_type}: {missing_params}"
        )

    # Validate param values
    if dist_type == "normal" and spec.params.get("std", 0) <= 0:
        errors.append("std must be > 0 for normal distribution")

    if dist_type == "lognormal" and spec.params.get("sigma", 0) <= 0:
        errors.append("sigma must be > 0 for lognormal distribution")

    if dist_type == "uniform":
        low = spec.params.get("low", 0)
        high = spec.params.get("high", 0)
        if low >= high:
            errors.append("low must be < high for uniform distribution")

    if dist_type == "exponential" and spec.params.get("scale", 0) <= 0:
        errors.append("scale must be > 0 for exponential distribution")

    if dist_type == "poisson" and spec.params.get("lam", 0) <= 0:
        errors.append("lam must be > 0 for poisson distribution")

    if dist_type == "binomial":
        n = spec.params.get("n", 0)
        p = spec.params.get("p", 0)
        if n < 0 or int(n) != n:
            errors.append("n must be a non-negative integer for binomial distribution")
        if not 0 <= p <= 1:
            errors.append("p must be in [0, 1] for binomial distribution")

    if dist_type == "beta":
        a = spec.params.get("a", 0)
        b = spec.params.get("b", 0)
        if a <= 0:
            errors.append("a must be > 0 for beta distribution")
        if b <= 0:
            errors.append("b must be > 0 for beta distribution")

    return errors
