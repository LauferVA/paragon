"""
PARAGON PERFORMANCE TEST SUITE
===============================

Comprehensive performance tests across all modules.
Tests verify that the system meets speed and scalability requirements.

Test Categories:
1. Graph Operations Performance (Protocol Alpha compliance)
2. Wave Computation & Topological Sort
3. Memory Efficiency
4. Concurrent Operations
5. Serialization Performance (msgspec, Polars, Arrow)
6. Infrastructure Performance (Metrics, Logging)

Performance Targets:
- get_waves() < 50ms for 1000 nodes (Protocol Alpha)
- Batch insert 10K nodes < 100ms
- Query operations < 25ms
- Serialization < 150ms for 10K nodes

Run with: pytest tests/performance/test_performance.py -v
"""
import time
import statistics
import gc
import sys
from typing import List, Callable, Any, Tuple
from dataclasses import dataclass

import pytest
import rustworkx as rx

from core.graph_db import ParagonDB, create_empty_db
from core.schemas import NodeData, EdgeData, NodePayload, EdgePayload
from core.ontology import NodeType, NodeStatus, EdgeType
from infrastructure.metrics import MetricsCollector, NodeMetric


# =============================================================================
# BENCHMARK UTILITIES
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    name: str
    target_ms: float
    actual_ms: float
    std_dev_ms: float
    iterations: int
    passed: bool
    scale: int  # Number of items tested

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.name} (n={self.scale})\n"
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
    # Warmup runs (JIT, cache warming)
    for _ in range(warmup):
        func()

    # Collect garbage before timing
    gc.collect()

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


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def empty_db():
    """Provide a fresh empty database."""
    return create_empty_db()


@pytest.fixture
def small_dag(empty_db):
    """Create a small DAG (100 nodes) for quick tests."""
    db = empty_db
    nodes = []
    for i in range(100):
        node = NodeData.create(
            type=NodeType.CODE.value,
            content=f"def func_{i}(): pass"
        )
        db.add_node(node)
        nodes.append(node)

    # Create some edges (10% connectivity)
    for i in range(10, 100):
        for j in range(max(0, i - 10), i):
            if (i * j) % 10 == 0:  # Deterministic sparsity
                edge = EdgeData.depends_on(nodes[j].id, nodes[i].id)
                db.add_edge(edge, check_cycle=False)

    return db, nodes


@pytest.fixture
def medium_dag(empty_db):
    """Create a medium DAG (1000 nodes) for Protocol Alpha tests."""
    db = empty_db
    nodes = []
    for i in range(1000):
        node = NodeData.create(
            type=NodeType.CODE.value,
            content=f"def func_{i}(): return {i}"
        )
        db.add_node(node)
        nodes.append(node)

    # Create edges (sparse DAG, ~1% connectivity)
    for i in range(10, 1000):
        for j in range(max(0, i - 20), i):
            if (i * j) % 100 == 0:
                edge = EdgeData.depends_on(nodes[j].id, nodes[i].id)
                db.add_edge(edge, check_cycle=False)

    return db, nodes


@pytest.fixture
def large_dag(empty_db):
    """Create a large DAG (10000 nodes) for scalability tests."""
    db = empty_db
    nodes = []
    for i in range(10000):
        node = NodeData.create(
            type=NodeType.CODE.value,
            content=f"def func_{i}(): return {i}"
        )
        db.add_node(node)
        nodes.append(node)

    # Create sparse edges
    for i in range(10, 10000):
        for j in range(max(0, i - 10), i):
            if (i * j) % 1000 == 0:
                edge = EdgeData.depends_on(nodes[j].id, nodes[i].id)
                db.add_edge(edge, check_cycle=False)

    return db, nodes


# =============================================================================
# GRAPH OPERATIONS PERFORMANCE TESTS
# =============================================================================

class TestGraphOperationsPerformance:
    """Test performance of core graph operations."""

    def test_add_node_single_small_scale(self, empty_db):
        """Test single node addition performance (100 nodes)."""
        db = empty_db
        nodes = [
            NodeData.create(type=NodeType.CODE.value, content=f"code_{i}")
            for i in range(100)
        ]

        def add_nodes():
            test_db = create_empty_db()
            for node in nodes:
                test_db.add_node(node, allow_duplicate=True)

        mean_ms, std_ms = benchmark(add_nodes, iterations=20)

        result = BenchmarkResult(
            name="add_node single (small)",
            target_ms=10.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=20,
            passed=mean_ms < 10.0,
            scale=100
        )
        print(f"\n{result}")
        assert result.passed, f"Single node addition too slow: {mean_ms:.2f}ms"

    def test_add_node_single_medium_scale(self, empty_db):
        """Test single node addition performance (1000 nodes)."""
        db = empty_db
        nodes = [
            NodeData.create(type=NodeType.CODE.value, content=f"code_{i}")
            for i in range(1000)
        ]

        def add_nodes():
            test_db = create_empty_db()
            for node in nodes:
                test_db.add_node(node, allow_duplicate=True)

        mean_ms, std_ms = benchmark(add_nodes, iterations=10)

        result = BenchmarkResult(
            name="add_node single (medium)",
            target_ms=100.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=10,
            passed=mean_ms < 100.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Single node addition too slow: {mean_ms:.2f}ms"

    def test_add_nodes_batch_small_scale(self, empty_db):
        """Test batch node addition performance (100 nodes)."""
        nodes = [
            NodeData.create(type=NodeType.CODE.value, content=f"code_{i}")
            for i in range(100)
        ]

        def batch_add():
            test_db = create_empty_db()
            test_db.add_nodes_batch(nodes)

        mean_ms, std_ms = benchmark(batch_add, iterations=20)

        result = BenchmarkResult(
            name="add_nodes_batch (small)",
            target_ms=5.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=20,
            passed=mean_ms < 5.0,
            scale=100
        )
        print(f"\n{result}")
        assert result.passed, f"Batch node addition too slow: {mean_ms:.2f}ms"

    def test_add_nodes_batch_medium_scale(self, empty_db):
        """Test batch node addition performance (1000 nodes)."""
        nodes = [
            NodeData.create(type=NodeType.CODE.value, content=f"code_{i}")
            for i in range(1000)
        ]

        def batch_add():
            test_db = create_empty_db()
            test_db.add_nodes_batch(nodes)

        mean_ms, std_ms = benchmark(batch_add, iterations=10)

        result = BenchmarkResult(
            name="add_nodes_batch (medium)",
            target_ms=50.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=10,
            passed=mean_ms < 50.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Batch node addition too slow: {mean_ms:.2f}ms"

    def test_add_nodes_batch_large_scale(self, empty_db):
        """Test batch node addition performance (10000 nodes) - Protocol Alpha."""
        nodes = [
            NodeData.create(type=NodeType.CODE.value, content=f"code_{i}")
            for i in range(10000)
        ]

        def batch_add():
            test_db = create_empty_db()
            test_db.add_nodes_batch(nodes)

        mean_ms, std_ms = benchmark(batch_add, iterations=5)

        result = BenchmarkResult(
            name="add_nodes_batch (large) - Protocol Alpha",
            target_ms=100.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=5,
            passed=mean_ms < 100.0,
            scale=10000
        )
        print(f"\n{result}")
        assert result.passed, f"Large batch addition too slow: {mean_ms:.2f}ms"

    def test_get_node_performance(self, medium_dag):
        """Test node retrieval performance."""
        db, nodes = medium_dag
        node_ids = [n.id for n in nodes[:100]]  # Test first 100

        def get_nodes():
            for node_id in node_ids:
                _ = db.get_node(node_id)

        mean_ms, std_ms = benchmark(get_nodes, iterations=20)

        result = BenchmarkResult(
            name="get_node (100 lookups)",
            target_ms=5.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=20,
            passed=mean_ms < 5.0,
            scale=100
        )
        print(f"\n{result}")
        assert result.passed, f"Node retrieval too slow: {mean_ms:.2f}ms"

    def test_query_nodes_by_type(self, medium_dag):
        """Test query_nodes by type performance."""
        db, nodes = medium_dag

        def query_nodes():
            _ = db.get_nodes_by_type(NodeType.CODE.value)

        mean_ms, std_ms = benchmark(query_nodes, iterations=20)

        result = BenchmarkResult(
            name="get_nodes_by_type (1000 nodes)",
            target_ms=10.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=20,
            passed=mean_ms < 10.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Query by type too slow: {mean_ms:.2f}ms"

    def test_add_edge_performance(self, medium_dag):
        """Test edge addition performance."""
        db, nodes = medium_dag
        edges = []
        for i in range(100, 200):
            edges.append(EdgeData.depends_on(nodes[i].id, nodes[i + 100].id))

        def add_edges():
            for edge in edges:
                try:
                    db.add_edge(edge, check_cycle=True)
                except Exception:
                    pass  # May have cycles

        mean_ms, std_ms = benchmark(add_edges, iterations=10)

        result = BenchmarkResult(
            name="add_edge with cycle check (100 edges)",
            target_ms=50.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=10,
            passed=mean_ms < 50.0,
            scale=100
        )
        print(f"\n{result}")
        assert result.passed, f"Edge addition too slow: {mean_ms:.2f}ms"


# =============================================================================
# WAVE COMPUTATION PERFORMANCE (PROTOCOL ALPHA)
# =============================================================================

class TestWaveComputationPerformance:
    """Test wave computation performance - critical for Protocol Alpha."""

    def test_get_waves_small_scale(self, small_dag):
        """Test get_waves performance (100 nodes)."""
        db, nodes = small_dag

        def compute_waves():
            _ = db.get_waves()

        mean_ms, std_ms = benchmark(compute_waves, iterations=20)

        result = BenchmarkResult(
            name="get_waves (100 nodes)",
            target_ms=5.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=20,
            passed=mean_ms < 5.0,
            scale=100
        )
        print(f"\n{result}")
        assert result.passed, f"Wave computation too slow: {mean_ms:.2f}ms"

    def test_get_waves_medium_scale_protocol_alpha(self, medium_dag):
        """Test get_waves performance (1000 nodes) - PROTOCOL ALPHA CRITICAL."""
        db, nodes = medium_dag

        def compute_waves():
            _ = db.get_waves()

        mean_ms, std_ms = benchmark(compute_waves, iterations=10)

        result = BenchmarkResult(
            name="get_waves (1000 nodes) - PROTOCOL ALPHA",
            target_ms=50.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=10,
            passed=mean_ms < 50.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Protocol Alpha violation: {mean_ms:.2f}ms > 50ms target"

    def test_topological_sort_performance(self, medium_dag):
        """Test topological sort performance."""
        db, nodes = medium_dag

        def topo_sort():
            _ = db.topological_sort()

        mean_ms, std_ms = benchmark(topo_sort, iterations=10)

        result = BenchmarkResult(
            name="topological_sort (1000 nodes)",
            target_ms=30.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=10,
            passed=mean_ms < 30.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Topological sort too slow: {mean_ms:.2f}ms"

    def test_cycle_detection_performance(self, medium_dag):
        """Test cycle detection performance - Protocol Alpha."""
        db, nodes = medium_dag

        def check_cycles():
            _ = db.has_cycle()

        mean_ms, std_ms = benchmark(check_cycles, iterations=20)

        result = BenchmarkResult(
            name="has_cycle (1000 nodes)",
            target_ms=25.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=20,
            passed=mean_ms < 25.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Cycle detection too slow: {mean_ms:.2f}ms"


# =============================================================================
# GRAPH TRAVERSAL PERFORMANCE
# =============================================================================

class TestGraphTraversalPerformance:
    """Test graph traversal operations at scale."""

    def test_get_descendants_small(self, small_dag):
        """Test get_descendants performance (small graph)."""
        db, nodes = small_dag
        root_id = nodes[0].id

        def get_desc():
            _ = db.get_descendants(root_id)

        mean_ms, std_ms = benchmark(get_desc, iterations=20)

        result = BenchmarkResult(
            name="get_descendants (100 nodes)",
            target_ms=5.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=20,
            passed=mean_ms < 5.0,
            scale=100
        )
        print(f"\n{result}")
        assert result.passed, f"Descendants query too slow: {mean_ms:.2f}ms"

    def test_get_descendants_medium(self, medium_dag):
        """Test get_descendants performance (medium graph) - Protocol Alpha."""
        db, nodes = medium_dag
        root_id = nodes[0].id

        def get_desc():
            _ = db.get_descendants(root_id)

        mean_ms, std_ms = benchmark(get_desc, iterations=20)

        result = BenchmarkResult(
            name="get_descendants (1000 nodes) - Protocol Alpha",
            target_ms=25.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=20,
            passed=mean_ms < 25.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Descendants query too slow: {mean_ms:.2f}ms"

    def test_get_ancestors_performance(self, medium_dag):
        """Test get_ancestors performance."""
        db, nodes = medium_dag
        leaf_id = nodes[-1].id

        def get_anc():
            _ = db.get_ancestors(leaf_id)

        mean_ms, std_ms = benchmark(get_anc, iterations=20)

        result = BenchmarkResult(
            name="get_ancestors (1000 nodes)",
            target_ms=25.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=20,
            passed=mean_ms < 25.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Ancestors query too slow: {mean_ms:.2f}ms"

    def test_get_successors_performance(self, medium_dag):
        """Test get_successors performance."""
        db, nodes = medium_dag
        node_id = nodes[500].id

        def get_succ():
            _ = db.get_successors(node_id)

        mean_ms, std_ms = benchmark(get_succ, iterations=50)

        result = BenchmarkResult(
            name="get_successors (1000 nodes)",
            target_ms=1.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=50,
            passed=mean_ms < 1.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Successors query too slow: {mean_ms:.2f}ms"

    def test_get_predecessors_performance(self, medium_dag):
        """Test get_predecessors performance."""
        db, nodes = medium_dag
        node_id = nodes[500].id

        def get_pred():
            _ = db.get_predecessors(node_id)

        mean_ms, std_ms = benchmark(get_pred, iterations=50)

        result = BenchmarkResult(
            name="get_predecessors (1000 nodes)",
            target_ms=1.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=50,
            passed=mean_ms < 1.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Predecessors query too slow: {mean_ms:.2f}ms"


# =============================================================================
# MEMORY EFFICIENCY TESTS
# =============================================================================

class TestMemoryEfficiency:
    """Test memory usage during bulk operations."""

    def test_batch_insert_memory_usage(self):
        """Test memory usage during batch insert."""
        mem_before = get_memory_usage_mb()

        db = create_empty_db()
        nodes = [
            NodeData.create(type=NodeType.CODE.value, content=f"code_{i}" * 100)
            for i in range(10000)
        ]

        db.add_nodes_batch(nodes)

        mem_after = get_memory_usage_mb()
        mem_delta = mem_after - mem_before

        print(f"\nMemory usage for 10K nodes: {mem_delta:.2f} MB")
        # Should be < 100MB for 10K nodes with msgspec
        assert mem_delta < 100, f"Memory usage too high: {mem_delta:.2f}MB"

    def test_wave_computation_memory_stability(self, medium_dag):
        """Test that wave computation doesn't leak memory."""
        db, nodes = medium_dag

        mem_before = get_memory_usage_mb()

        # Run waves 100 times
        for _ in range(100):
            _ = db.get_waves()

        gc.collect()
        mem_after = get_memory_usage_mb()
        mem_delta = mem_after - mem_before

        print(f"\nMemory delta after 100 wave computations: {mem_delta:.2f} MB")
        # Should not grow significantly
        assert mem_delta < 10, f"Memory leak detected: {mem_delta:.2f}MB"

    def test_graph_traversal_memory_efficiency(self, medium_dag):
        """Test memory usage during graph traversal."""
        db, nodes = medium_dag

        mem_before = get_memory_usage_mb()

        # Perform many traversal operations
        for node in nodes[:100]:
            _ = db.get_descendants(node.id)
            _ = db.get_ancestors(node.id)

        gc.collect()
        mem_after = get_memory_usage_mb()
        mem_delta = mem_after - mem_before

        print(f"\nMemory delta after 200 traversals: {mem_delta:.2f} MB")
        assert mem_delta < 20, f"Traversal memory usage too high: {mem_delta:.2f}MB"


# =============================================================================
# SERIALIZATION PERFORMANCE
# =============================================================================

class TestSerializationPerformance:
    """Test serialization and deserialization performance."""

    def test_msgspec_encode_decode_small(self):
        """Test msgspec encoding/decoding (100 nodes)."""
        import msgspec

        nodes = [
            NodeData.create(type=NodeType.CODE.value, content=f"code_{i}")
            for i in range(100)
        ]

        def encode_decode():
            encoded = msgspec.json.encode(nodes)
            decoded = msgspec.json.decode(encoded, type=List[NodeData])
            return decoded

        mean_ms, std_ms = benchmark(encode_decode, iterations=20)

        result = BenchmarkResult(
            name="msgspec encode/decode (100 nodes)",
            target_ms=5.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=20,
            passed=mean_ms < 5.0,
            scale=100
        )
        print(f"\n{result}")
        assert result.passed, f"msgspec too slow: {mean_ms:.2f}ms"

    def test_msgspec_encode_decode_large(self):
        """Test msgspec encoding/decoding (10000 nodes)."""
        import msgspec

        nodes = [
            NodeData.create(type=NodeType.CODE.value, content=f"code_{i}")
            for i in range(10000)
        ]

        def encode_decode():
            encoded = msgspec.json.encode(nodes)
            decoded = msgspec.json.decode(encoded, type=List[NodeData])
            return decoded

        mean_ms, std_ms = benchmark(encode_decode, iterations=5)

        result = BenchmarkResult(
            name="msgspec encode/decode (10K nodes)",
            target_ms=100.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=5,
            passed=mean_ms < 100.0,
            scale=10000
        )
        print(f"\n{result}")
        assert result.passed, f"msgspec too slow: {mean_ms:.2f}ms"

    def test_polars_export_performance(self, medium_dag):
        """Test Polars DataFrame export performance."""
        db, nodes = medium_dag

        def export_polars():
            _ = db.to_polars_nodes()
            _ = db.to_polars_edges()

        mean_ms, std_ms = benchmark(export_polars, iterations=10)

        result = BenchmarkResult(
            name="Polars export (1000 nodes)",
            target_ms=50.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=10,
            passed=mean_ms < 50.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Polars export too slow: {mean_ms:.2f}ms"

    def test_arrow_serialization_performance(self, medium_dag):
        """Test Arrow IPC serialization performance."""
        import tempfile
        from pathlib import Path

        db, nodes = medium_dag

        def arrow_roundtrip():
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "test"
                db.save_arrow(path)
                # Note: We don't test loading as that would require
                # implementing load_arrow method

        mean_ms, std_ms = benchmark(arrow_roundtrip, iterations=10)

        result = BenchmarkResult(
            name="Arrow IPC save (1000 nodes)",
            target_ms=100.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=10,
            passed=mean_ms < 100.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Arrow serialization too slow: {mean_ms:.2f}ms"

    def test_parquet_serialization_performance(self, medium_dag):
        """Test Parquet serialization performance."""
        import tempfile
        from pathlib import Path

        db, nodes = medium_dag

        def parquet_roundtrip():
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "test"
                db.save_parquet(path)

        mean_ms, std_ms = benchmark(parquet_roundtrip, iterations=10)

        result = BenchmarkResult(
            name="Parquet save (1000 nodes)",
            target_ms=100.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=10,
            passed=mean_ms < 100.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Parquet serialization too slow: {mean_ms:.2f}ms"


# =============================================================================
# INFRASTRUCTURE PERFORMANCE
# =============================================================================

class TestInfrastructurePerformance:
    """Test infrastructure component performance."""

    def test_metrics_collector_record_performance(self):
        """Test metrics collector recording throughput."""
        collector = MetricsCollector()

        def record_metrics():
            for i in range(1000):
                node_id = f"node_{i}"
                collector.record_node_created(
                    node_id=node_id,
                    node_type=NodeType.CODE.value,
                    created_by="test"
                )
                collector.record_node_start(
                    node_id=node_id,
                    agent_id="agent_1",
                    agent_role="BUILDER",
                    operation="build"
                )
                collector.record_node_complete(
                    node_id=node_id,
                    status=NodeStatus.VERIFIED.value,
                    tokens=100
                )

        mean_ms, std_ms = benchmark(record_metrics, iterations=5)

        result = BenchmarkResult(
            name="MetricsCollector record (1000 events)",
            target_ms=100.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=5,
            passed=mean_ms < 100.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Metrics recording too slow: {mean_ms:.2f}ms"

    def test_metrics_collector_query_performance(self):
        """Test metrics collector query performance."""
        collector = MetricsCollector()

        # Populate with 1000 metrics
        for i in range(1000):
            node_id = f"node_{i}"
            collector.record_node_created(
                node_id=node_id,
                node_type=NodeType.CODE.value if i % 2 == 0 else NodeType.SPEC.value,
                created_by="test",
                traces_to_req="req_1" if i < 500 else "req_2"
            )
            collector.record_node_complete(
                node_id=node_id,
                status=NodeStatus.VERIFIED.value if i % 3 == 0 else NodeStatus.PENDING.value,
                tokens=100
            )

        def query_metrics():
            _ = collector.query_by_traceability(req_id="req_1")
            _ = collector.query_by_status(NodeStatus.VERIFIED.value)
            _ = collector.query_failures()

        mean_ms, std_ms = benchmark(query_metrics, iterations=20)

        result = BenchmarkResult(
            name="MetricsCollector query (1000 metrics)",
            target_ms=10.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=20,
            passed=mean_ms < 10.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Metrics query too slow: {mean_ms:.2f}ms"

    def test_metrics_collector_aggregation_performance(self):
        """Test metrics aggregation (Polars conversion) performance."""
        collector = MetricsCollector()

        # Populate with 1000 metrics
        for i in range(1000):
            node_id = f"node_{i}"
            collector.record_node_created(
                node_id=node_id,
                node_type=NodeType.CODE.value,
                created_by="test"
            )
            collector.record_node_complete(
                node_id=node_id,
                status=NodeStatus.VERIFIED.value,
                tokens=100,
                input_tokens=50,
                output_tokens=50
            )

        def aggregate():
            _ = collector.to_dataframe()
            _ = collector.get_summary()
            _ = collector.get_success_patterns()

        mean_ms, std_ms = benchmark(aggregate, iterations=10)

        result = BenchmarkResult(
            name="MetricsCollector aggregation (1000 metrics)",
            target_ms=50.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=10,
            passed=mean_ms < 50.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Metrics aggregation too slow: {mean_ms:.2f}ms"


# =============================================================================
# CONCURRENT OPERATIONS (Basic Tests)
# =============================================================================

class TestConcurrentOperations:
    """Test performance under concurrent access patterns."""

    def test_parallel_read_operations(self, medium_dag):
        """Test parallel read operations don't degrade performance."""
        db, nodes = medium_dag
        node_ids = [n.id for n in nodes[:100]]

        def parallel_reads():
            # Simulate parallel reads by doing many sequential reads
            # (True parallelism would require threading, which is complex)
            for _ in range(10):
                for node_id in node_ids:
                    _ = db.get_node(node_id)
                    _ = db.get_successors(node_id)

        mean_ms, std_ms = benchmark(parallel_reads, iterations=5)

        result = BenchmarkResult(
            name="Parallel reads (1000 operations)",
            target_ms=100.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=5,
            passed=mean_ms < 100.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Parallel reads too slow: {mean_ms:.2f}ms"

    def test_mixed_read_write_performance(self, small_dag):
        """Test mixed read/write operations."""
        db, nodes = small_dag

        def mixed_ops():
            # Mix of reads and writes
            for i in range(50):
                # Read
                _ = db.get_node(nodes[i].id)

                # Write
                new_node = NodeData.create(
                    type=NodeType.CODE.value,
                    content=f"new_code_{i}"
                )
                db.add_node(new_node, allow_duplicate=True)

                # Traversal
                _ = db.get_successors(nodes[i].id)

        mean_ms, std_ms = benchmark(mixed_ops, iterations=10)

        result = BenchmarkResult(
            name="Mixed read/write (150 operations)",
            target_ms=50.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=10,
            passed=mean_ms < 50.0,
            scale=150
        )
        print(f"\n{result}")
        assert result.passed, f"Mixed operations too slow: {mean_ms:.2f}ms"


# =============================================================================
# EDGE CASE PERFORMANCE
# =============================================================================

class TestEdgeCasePerformance:
    """Test performance on edge cases."""

    def test_empty_graph_operations(self):
        """Test operations on empty graph are fast."""
        db = create_empty_db()

        def empty_ops():
            _ = db.get_waves()
            _ = db.get_all_nodes()
            _ = db.get_all_edges()
            _ = db.has_cycle()
            _ = db.to_polars_nodes()

        mean_ms, std_ms = benchmark(empty_ops, iterations=50)

        result = BenchmarkResult(
            name="Empty graph operations",
            target_ms=1.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=50,
            passed=mean_ms < 1.0,
            scale=0
        )
        print(f"\n{result}")
        assert result.passed, f"Empty graph ops too slow: {mean_ms:.2f}ms"

    def test_single_node_graph_operations(self):
        """Test operations on single-node graph."""
        db = create_empty_db()
        node = NodeData.create(type=NodeType.CODE.value, content="test")
        db.add_node(node)

        def single_node_ops():
            _ = db.get_waves()
            _ = db.get_descendants(node.id)
            _ = db.get_ancestors(node.id)
            _ = db.topological_sort()

        mean_ms, std_ms = benchmark(single_node_ops, iterations=50)

        result = BenchmarkResult(
            name="Single node graph operations",
            target_ms=1.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=50,
            passed=mean_ms < 1.0,
            scale=1
        )
        print(f"\n{result}")
        assert result.passed, f"Single node ops too slow: {mean_ms:.2f}ms"

    def test_deep_chain_traversal(self):
        """Test traversal on a deep linear chain."""
        db = create_empty_db()
        nodes = []

        # Create a chain of 1000 nodes
        for i in range(1000):
            node = NodeData.create(type=NodeType.CODE.value, content=f"code_{i}")
            db.add_node(node)
            nodes.append(node)

            if i > 0:
                edge = EdgeData.depends_on(nodes[i-1].id, nodes[i].id)
                db.add_edge(edge, check_cycle=False)

        def traverse_chain():
            _ = db.get_descendants(nodes[0].id)
            _ = db.get_ancestors(nodes[-1].id)

        mean_ms, std_ms = benchmark(traverse_chain, iterations=10)

        result = BenchmarkResult(
            name="Deep chain traversal (1000 depth)",
            target_ms=50.0,
            actual_ms=mean_ms,
            std_dev_ms=std_ms,
            iterations=10,
            passed=mean_ms < 50.0,
            scale=1000
        )
        print(f"\n{result}")
        assert result.passed, f"Deep chain traversal too slow: {mean_ms:.2f}ms"


# =============================================================================
# PERFORMANCE SUMMARY
# =============================================================================

def test_performance_summary():
    """
    Print performance test summary.

    This test always passes - it's just for reporting.
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE TEST SUITE SUMMARY")
    print("=" * 70)
    print("\nAll performance tests completed.")
    print("\nKey Performance Targets:")
    print("  - Protocol Alpha: get_waves() < 50ms for 1000 nodes")
    print("  - Batch Insert: 10K nodes < 100ms")
    print("  - Graph Queries: < 25ms")
    print("  - Serialization: < 150ms for 10K nodes")
    print("\nTo run specific test categories:")
    print("  pytest tests/performance/test_performance.py::TestGraphOperationsPerformance -v")
    print("  pytest tests/performance/test_performance.py::TestWaveComputationPerformance -v")
    print("  pytest tests/performance/test_performance.py::TestMemoryEfficiency -v")
    print("=" * 70)
