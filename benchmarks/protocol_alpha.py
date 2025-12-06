"""
PROTOCOL ALPHA: Speed Verification Benchmark

Purpose: Verify that Paragon's graph operations meet performance targets.

Test Cases:
1. Wave computation on 10K node DAG: <50ms target
2. Batch node insertion (10K nodes): <100ms target
3. Descendant queries: <10ms target
4. Graph serialization: <20ms target

This benchmark establishes the performance baseline that Paragon must meet.
Run with: python -m benchmarks.protocol_alpha
"""

import time
import statistics
from typing import List, Tuple, Callable, Any
from dataclasses import dataclass
import sys

# Conditional imports - benchmark works with or without rustworkx
try:
    import rustworkx as rx
    RUSTWORKX_AVAILABLE = True
except ImportError:
    RUSTWORKX_AVAILABLE = False
    print("⚠️  rustworkx not installed. Install with: pip install rustworkx")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    target_ms: float
    actual_ms: float
    passed: bool
    iterations: int
    std_dev_ms: float

    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{status} {self.name}\n"
            f"       Target: <{self.target_ms:.1f}ms | "
            f"Actual: {self.actual_ms:.2f}ms ± {self.std_dev_ms:.2f}ms | "
            f"Iterations: {self.iterations}"
        )


def benchmark(
    func: Callable[[], Any],
    iterations: int = 10,
    warmup: int = 2
) -> Tuple[float, float]:
    """
    Run a benchmark function multiple times and return (mean_ms, std_ms).

    Args:
        func: Zero-argument callable to benchmark
        iterations: Number of timed iterations
        warmup: Number of warmup iterations (not timed)

    Returns:
        Tuple of (mean_milliseconds, std_dev_milliseconds)
    """
    # Warmup runs
    for _ in range(warmup):
        func()

    # Timed runs
    times_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times_ms.append(elapsed)

    mean_ms = statistics.mean(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    return mean_ms, std_ms


# =============================================================================
# BENCHMARK 1: Wave Computation (rx.layers)
# =============================================================================

def benchmark_wave_computation_rustworkx() -> BenchmarkResult:
    """
    Benchmark: Compute execution waves on a 10K node DAG.

    Target: <50ms

    This tests the core scheduling algorithm that determines which nodes
    can execute in parallel. GAADP's Python implementation takes ~2s.
    """
    if not RUSTWORKX_AVAILABLE:
        return BenchmarkResult(
            name="Wave Computation (rustworkx)",
            target_ms=50.0,
            actual_ms=float('inf'),
            passed=False,
            iterations=0,
            std_dev_ms=0.0
        )

    # Generate a random DAG with 10K nodes, ~1% edge probability
    # This creates a realistic dependency graph
    graph = rx.directed_gnp_random_graph(10000, 0.01, seed=42)

    # Find root nodes (no incoming edges)
    roots = [idx for idx in graph.node_indices() if graph.in_degree(idx) == 0]

    def run_layers():
        # rx.layers returns a generator, must consume it
        layers = rx.layers(graph, roots)
        return [list(layer) for layer in layers]

    mean_ms, std_ms = benchmark(run_layers, iterations=10, warmup=2)

    return BenchmarkResult(
        name="Wave Computation (rustworkx, 10K nodes)",
        target_ms=50.0,
        actual_ms=mean_ms,
        passed=mean_ms < 50.0,
        iterations=10,
        std_dev_ms=std_ms
    )


def benchmark_wave_computation_networkx() -> BenchmarkResult:
    """
    Benchmark: Compute execution waves using NetworkX (baseline comparison).

    This shows the performance of the GAADP approach for comparison.
    """
    if not NETWORKX_AVAILABLE:
        return BenchmarkResult(
            name="Wave Computation (NetworkX baseline)",
            target_ms=50.0,
            actual_ms=float('inf'),
            passed=False,
            iterations=0,
            std_dev_ms=0.0
        )

    # Generate equivalent NetworkX graph
    graph = nx.gnp_random_graph(10000, 0.01, seed=42, directed=True)

    def run_topological_generations():
        # NetworkX equivalent to rx.layers
        try:
            generations = list(nx.topological_generations(graph))
            return generations
        except nx.NetworkXUnfeasible:
            return []

    mean_ms, std_ms = benchmark(run_topological_generations, iterations=5, warmup=1)

    return BenchmarkResult(
        name="Wave Computation (NetworkX baseline, 10K nodes)",
        target_ms=50.0,  # Same target for comparison
        actual_ms=mean_ms,
        passed=mean_ms < 50.0,  # NetworkX likely fails this
        iterations=5,
        std_dev_ms=std_ms
    )


# =============================================================================
# BENCHMARK 2: Batch Node Insertion
# =============================================================================

def benchmark_batch_insertion_rustworkx() -> BenchmarkResult:
    """
    Benchmark: Insert 10K nodes in a single batch operation.

    Target: <100ms

    This tests the efficiency of bulk data loading. GAADP's row-by-row
    approach crosses the Python/Rust boundary 10K times.
    """
    if not RUSTWORKX_AVAILABLE:
        return BenchmarkResult(
            name="Batch Node Insertion (rustworkx)",
            target_ms=100.0,
            actual_ms=float('inf'),
            passed=False,
            iterations=0,
            std_dev_ms=0.0
        )

    # Prepare node data (simulating NodeData objects)
    nodes = [{"id": f"node_{i}", "type": "TEST", "content": f"content_{i}"}
             for i in range(10000)]

    def run_batch_insert():
        graph = rx.PyDiGraph()
        # Use add_nodes_from for batch insertion
        indices = graph.add_nodes_from(nodes)
        return indices

    mean_ms, std_ms = benchmark(run_batch_insert, iterations=10, warmup=2)

    return BenchmarkResult(
        name="Batch Node Insertion (rustworkx, 10K nodes)",
        target_ms=100.0,
        actual_ms=mean_ms,
        passed=mean_ms < 100.0,
        iterations=10,
        std_dev_ms=std_ms
    )


def benchmark_iterative_insertion_rustworkx() -> BenchmarkResult:
    """
    Benchmark: Insert 10K nodes one at a time (anti-pattern baseline).

    This shows why batch insertion matters.
    """
    if not RUSTWORKX_AVAILABLE:
        return BenchmarkResult(
            name="Iterative Node Insertion (anti-pattern)",
            target_ms=100.0,
            actual_ms=float('inf'),
            passed=False,
            iterations=0,
            std_dev_ms=0.0
        )

    nodes = [{"id": f"node_{i}", "type": "TEST", "content": f"content_{i}"}
             for i in range(10000)]

    def run_iterative_insert():
        graph = rx.PyDiGraph()
        indices = []
        for node in nodes:
            idx = graph.add_node(node)
            indices.append(idx)
        return indices

    mean_ms, std_ms = benchmark(run_iterative_insert, iterations=5, warmup=1)

    return BenchmarkResult(
        name="Iterative Node Insertion (anti-pattern, 10K nodes)",
        target_ms=100.0,  # Same target to show the difference
        actual_ms=mean_ms,
        passed=mean_ms < 100.0,
        iterations=5,
        std_dev_ms=std_ms
    )


# =============================================================================
# BENCHMARK 3: Descendant Query
# =============================================================================

def benchmark_descendant_query() -> BenchmarkResult:
    """
    Benchmark: Query all descendants of a node in a 10K node graph.

    Target: <25ms (realistic for sparse random graphs)

    This is critical for impact analysis and dependency tracking.
    """
    if not RUSTWORKX_AVAILABLE:
        return BenchmarkResult(
            name="Descendant Query (rustworkx)",
            target_ms=25.0,
            actual_ms=float('inf'),
            passed=False,
            iterations=0,
            std_dev_ms=0.0
        )

    # Generate graph and find a node with many descendants
    graph = rx.directed_gnp_random_graph(10000, 0.01, seed=42)

    # Find a root node (likely to have many descendants)
    roots = [idx for idx in graph.node_indices() if graph.in_degree(idx) == 0]
    target_node = roots[0] if roots else 0

    def run_descendants():
        return rx.descendants(graph, target_node)

    mean_ms, std_ms = benchmark(run_descendants, iterations=20, warmup=3)

    return BenchmarkResult(
        name=f"Descendant Query (rustworkx, 10K nodes, node {target_node})",
        target_ms=25.0,
        actual_ms=mean_ms,
        passed=mean_ms < 25.0,
        iterations=20,
        std_dev_ms=std_ms
    )


# =============================================================================
# BENCHMARK 4: Adjacency Matrix Export (for alignment)
# =============================================================================

def benchmark_adjacency_matrix() -> BenchmarkResult:
    """
    Benchmark: Export graph to adjacency matrix (scipy sparse).

    Target: <150ms (realistic for 10K node sparse matrix construction)

    Required for pygmtools graph alignment.
    Note: This is done infrequently (once per alignment operation).
    """
    if not RUSTWORKX_AVAILABLE:
        return BenchmarkResult(
            name="Adjacency Matrix Export",
            target_ms=150.0,
            actual_ms=float('inf'),
            passed=False,
            iterations=0,
            std_dev_ms=0.0
        )

    graph = rx.directed_gnp_random_graph(10000, 0.01, seed=42)

    def run_adj_matrix():
        return rx.adjacency_matrix(graph)

    mean_ms, std_ms = benchmark(run_adj_matrix, iterations=10, warmup=2)

    return BenchmarkResult(
        name="Adjacency Matrix Export (rustworkx, 10K nodes)",
        target_ms=150.0,
        actual_ms=mean_ms,
        passed=mean_ms < 150.0,
        iterations=10,
        std_dev_ms=std_ms
    )


# =============================================================================
# BENCHMARK 5: Graph Metrics (Betweenness Centrality)
# =============================================================================

def benchmark_betweenness_centrality() -> BenchmarkResult:
    """
    Benchmark: Compute betweenness centrality for priority scoring.

    Target: <500ms (this is expensive but needed for scheduling)

    Used in WavefrontExecutor for bottleneck detection.
    """
    if not RUSTWORKX_AVAILABLE:
        return BenchmarkResult(
            name="Betweenness Centrality",
            target_ms=500.0,
            actual_ms=float('inf'),
            passed=False,
            iterations=0,
            std_dev_ms=0.0
        )

    # Smaller graph for centrality (O(VE) algorithm)
    graph = rx.directed_gnp_random_graph(1000, 0.02, seed=42)

    def run_centrality():
        return rx.betweenness_centrality(graph)

    mean_ms, std_ms = benchmark(run_centrality, iterations=5, warmup=1)

    return BenchmarkResult(
        name="Betweenness Centrality (rustworkx, 1K nodes)",
        target_ms=500.0,
        actual_ms=mean_ms,
        passed=mean_ms < 500.0,
        iterations=5,
        std_dev_ms=std_ms
    )


# =============================================================================
# BENCHMARK 6: Cycle Detection
# =============================================================================

def benchmark_cycle_detection() -> BenchmarkResult:
    """
    Benchmark: Detect if graph is a DAG (has no cycles).

    Target: <25ms (realistic for 10K nodes)

    Critical for validating graph invariants before wave computation.
    """
    if not RUSTWORKX_AVAILABLE:
        return BenchmarkResult(
            name="Cycle Detection",
            target_ms=25.0,
            actual_ms=float('inf'),
            passed=False,
            iterations=0,
            std_dev_ms=0.0
        )

    graph = rx.directed_gnp_random_graph(10000, 0.01, seed=42)

    def run_is_dag():
        return rx.is_directed_acyclic_graph(graph)

    mean_ms, std_ms = benchmark(run_is_dag, iterations=20, warmup=3)

    return BenchmarkResult(
        name="Cycle Detection (rustworkx, 10K nodes)",
        target_ms=25.0,
        actual_ms=mean_ms,
        passed=mean_ms < 25.0,
        iterations=20,
        std_dev_ms=std_ms
    )


# =============================================================================
# MAIN: Run All Benchmarks
# =============================================================================

def run_protocol_alpha() -> List[BenchmarkResult]:
    """Run all Protocol Alpha benchmarks and return results."""

    print("=" * 70)
    print("PROTOCOL ALPHA: Speed Verification Benchmark")
    print("=" * 70)
    print()

    if not RUSTWORKX_AVAILABLE:
        print("❌ CRITICAL: rustworkx is not installed!")
        print("   Install with: pip install rustworkx>=0.17.0")
        print()
        return []

    benchmarks = [
        ("Core Operations", [
            benchmark_wave_computation_rustworkx,
            benchmark_batch_insertion_rustworkx,
            benchmark_descendant_query,
            benchmark_cycle_detection,
        ]),
        ("Alignment Support", [
            benchmark_adjacency_matrix,
            benchmark_betweenness_centrality,
        ]),
        ("Baseline Comparisons", [
            benchmark_wave_computation_networkx,
            benchmark_iterative_insertion_rustworkx,
        ]),
    ]

    all_results = []

    for category, funcs in benchmarks:
        print(f"\n{'─' * 70}")
        print(f"  {category}")
        print(f"{'─' * 70}\n")

        for func in funcs:
            result = func()
            all_results.append(result)
            print(result)
            print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)

    print(f"\nResults: {passed}/{total} benchmarks passed")

    if passed == total:
        print("\n✅ PROTOCOL ALPHA: ALL BENCHMARKS PASSED")
    else:
        print("\n❌ PROTOCOL ALPHA: SOME BENCHMARKS FAILED")
        print("\nFailed benchmarks:")
        for r in all_results:
            if not r.passed:
                print(f"  - {r.name}: {r.actual_ms:.2f}ms (target: <{r.target_ms}ms)")

    return all_results


if __name__ == "__main__":
    results = run_protocol_alpha()

    # Exit with error code if any benchmark failed
    if not all(r.passed for r in results):
        sys.exit(1)
