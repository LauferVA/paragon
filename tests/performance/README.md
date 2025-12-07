# Paragon Performance Test Suite

Comprehensive performance tests verifying that the Paragon system meets speed and scalability requirements.

## Overview

**Total Tests: 34**

This test suite validates performance across all critical system components, ensuring compliance with Protocol Alpha and other performance requirements.

## Test Categories

### 1. Graph Operations Performance (8 tests)
Tests core graph database operations at multiple scales (100, 1000, 10000 nodes).

**Tests:**
- `test_add_node_single_small_scale` - Single node addition (100 nodes) < 10ms
- `test_add_node_single_medium_scale` - Single node addition (1000 nodes) < 100ms
- `test_add_nodes_batch_small_scale` - Batch insert (100 nodes) < 5ms
- `test_add_nodes_batch_medium_scale` - Batch insert (1000 nodes) < 50ms
- `test_add_nodes_batch_large_scale` - Batch insert (10K nodes) < 100ms **[Protocol Alpha]**
- `test_get_node_performance` - Node retrieval (100 lookups) < 5ms
- `test_query_nodes_by_type` - Query by type (1000 nodes) < 10ms
- `test_add_edge_performance` - Edge addition with cycle check (100 edges) < 50ms

### 2. Wave Computation Performance (4 tests)
**Critical for Protocol Alpha compliance.**

**Tests:**
- `test_get_waves_small_scale` - Wave computation (100 nodes) < 5ms
- `test_get_waves_medium_scale_protocol_alpha` - **[PROTOCOL ALPHA]** Wave computation (1000 nodes) < 50ms
- `test_topological_sort_performance` - Topological sort (1000 nodes) < 30ms
- `test_cycle_detection_performance` - Cycle detection (1000 nodes) < 25ms

**Protocol Alpha Target:** `get_waves()` must complete in < 50ms for 1000 nodes.

### 3. Graph Traversal Performance (5 tests)
Tests graph navigation operations.

**Tests:**
- `test_get_descendants_small` - Descendants query (100 nodes) < 5ms
- `test_get_descendants_medium` - Descendants query (1000 nodes) < 25ms **[Protocol Alpha]**
- `test_get_ancestors_performance` - Ancestors query (1000 nodes) < 25ms
- `test_get_successors_performance` - Immediate successors (1000 nodes) < 1ms
- `test_get_predecessors_performance` - Immediate predecessors (1000 nodes) < 1ms

### 4. Memory Efficiency (3 tests)
Tests memory usage and leak prevention.

**Tests:**
- `test_batch_insert_memory_usage` - 10K nodes should use < 100MB
- `test_wave_computation_memory_stability` - 100 wave computations should not leak (< 10MB delta)
- `test_graph_traversal_memory_efficiency` - 200 traversals should not leak (< 20MB delta)

### 5. Serialization Performance (5 tests)
Tests msgspec, Polars, Arrow, and Parquet serialization.

**Tests:**
- `test_msgspec_encode_decode_small` - msgspec (100 nodes) < 5ms
- `test_msgspec_encode_decode_large` - msgspec (10K nodes) < 100ms
- `test_polars_export_performance` - Polars export (1000 nodes) < 50ms
- `test_arrow_serialization_performance` - Arrow IPC (1000 nodes) < 100ms
- `test_parquet_serialization_performance` - Parquet (1000 nodes) < 100ms

### 6. Infrastructure Performance (3 tests)
Tests metrics collection and aggregation.

**Tests:**
- `test_metrics_collector_record_performance` - Record 1000 events < 100ms
- `test_metrics_collector_query_performance` - Query metrics (1000 entries) < 10ms
- `test_metrics_collector_aggregation_performance` - Aggregation (1000 entries) < 50ms

### 7. Concurrent Operations (2 tests)
Tests performance under concurrent access patterns.

**Tests:**
- `test_parallel_read_operations` - 1000 parallel reads < 100ms
- `test_mixed_read_write_performance` - 150 mixed operations < 50ms

### 8. Edge Case Performance (3 tests)
Tests performance on boundary conditions.

**Tests:**
- `test_empty_graph_operations` - Operations on empty graph < 1ms
- `test_single_node_graph_operations` - Operations on single-node graph < 1ms
- `test_deep_chain_traversal` - Traversal on 1000-depth chain < 50ms

### 9. Summary (1 test)
- `test_performance_summary` - Always passes, prints summary information

## Running the Tests

### Run All Performance Tests
```bash
pytest tests/performance/test_performance.py -v
```

### Run with Verbose Output (shows timing results)
```bash
pytest tests/performance/test_performance.py -v -s
```

### Run Specific Test Category
```bash
# Graph operations only
pytest tests/performance/test_performance.py::TestGraphOperationsPerformance -v

# Wave computation (Protocol Alpha critical)
pytest tests/performance/test_performance.py::TestWaveComputationPerformance -v

# Memory efficiency
pytest tests/performance/test_performance.py::TestMemoryEfficiency -v

# Serialization
pytest tests/performance/test_performance.py::TestSerializationPerformance -v

# Infrastructure
pytest tests/performance/test_performance.py::TestInfrastructurePerformance -v
```

### Run Single Test
```bash
pytest tests/performance/test_performance.py::TestWaveComputationPerformance::test_get_waves_medium_scale_protocol_alpha -v -s
```

## Performance Targets Summary

| Component | Target | Scale |
|-----------|--------|-------|
| **Protocol Alpha: Wave Computation** | < 50ms | 1000 nodes |
| Batch Node Insert | < 100ms | 10K nodes |
| Graph Queries (descendants, ancestors) | < 25ms | 1000 nodes |
| Topological Sort | < 30ms | 1000 nodes |
| Cycle Detection | < 25ms | 1000 nodes |
| msgspec Serialization | < 100ms | 10K nodes |
| Polars Export | < 50ms | 1000 nodes |
| Metrics Collection | < 100ms | 1000 events |
| Memory (batch insert) | < 100MB | 10K nodes |

## Interpreting Results

Each test prints results in the format:
```
[PASS] test_name (n=scale)
       Target: <XXms | Actual: YY.YYms ± Z.ZZms | Iterations: N
```

- **Target**: Maximum acceptable time
- **Actual**: Mean execution time from N iterations
- **±**: Standard deviation (consistency measure)
- **Scale**: Number of items tested

### Example Output
```
[PASS] get_waves (1000 nodes) - PROTOCOL ALPHA (n=1000)
       Target: <50.0ms | Actual: 0.20ms ± 0.02ms | Iterations: 10
```

This shows the test completed in 0.20ms (well under the 50ms target) with low variance (±0.02ms).

## Benchmark Methodology

All tests use the following methodology:

1. **Warmup Runs**: 2 iterations to warm up JIT, caches
2. **Timed Runs**: 5-50 iterations (depending on test duration)
3. **Garbage Collection**: Full GC between benchmark runs
4. **Statistical Analysis**: Mean and standard deviation calculated
5. **Pass/Fail**: Test passes if mean < target

## Performance Characteristics

### Expected Performance (as of Dec 2025)

Based on test results:

- **Wave Computation (1000 nodes)**: ~0.2ms (250x faster than target)
- **Batch Insert (10K nodes)**: ~10-50ms (2-10x faster than target)
- **Graph Traversal**: < 1ms for most operations
- **msgspec Serialization**: ~13ms for 10K nodes (8x faster than target)
- **Polars Export**: ~1ms for 1000 nodes (50x faster than target)

### Why Rust Matters

Paragon uses `rustworkx` for graph operations, providing:

- **50-250x** speedup over NetworkX for wave computation
- **O(V+E)** complexity vs **O(V²)** for Python recursion
- **60%** lower memory usage vs NetworkX
- **Single boundary crossing** for batch operations

## Troubleshooting

### Tests Failing

If performance tests fail:

1. **Check System Load**: Ensure no other heavy processes running
2. **Run Multiple Times**: Performance can vary, especially on laptops
3. **Check Python Version**: Python 3.11+ recommended for best performance
4. **Check Dependencies**: Ensure latest rustworkx, polars, msgspec installed

### Memory Tests Failing

Memory tests are sensitive to:

- Background processes
- Garbage collection timing
- OS memory management

If memory tests fail intermittently, try:
```bash
# Run with explicit GC
pytest tests/performance/test_performance.py::TestMemoryEfficiency -v --forceGC
```

## CI/CD Integration

These tests are suitable for CI/CD pipelines but note:

- **Duration**: Full suite takes ~5-10 seconds
- **Variability**: Performance can vary ±20% between runs
- **Resource Sensitive**: May need dedicated CI runners for consistent results

### Recommended CI Configuration

```yaml
- name: Run Performance Tests
  run: |
    pytest tests/performance/test_performance.py -v --tb=short
  timeout-minutes: 5
  # Consider running on dedicated performance runners
```

## Adding New Performance Tests

Follow this pattern:

```python
def test_new_operation_performance(self, medium_dag):
    """Test description and target."""
    db, nodes = medium_dag

    def operation():
        # Code to benchmark
        _ = db.some_operation()

    mean_ms, std_ms = benchmark(operation, iterations=10)

    result = BenchmarkResult(
        name="Operation name",
        target_ms=50.0,  # Set realistic target
        actual_ms=mean_ms,
        std_dev_ms=std_ms,
        iterations=10,
        passed=mean_ms < 50.0,
        scale=1000  # Number of items
    )
    print(f"\n{result}")
    assert result.passed, f"Description: {mean_ms:.2f}ms"
```

## References

- **Protocol Alpha Specification**: See `/Users/lauferva/paragon/CLAUDE.md` section 3
- **Graph Database**: `/Users/lauferva/paragon/core/graph_db.py`
- **Schemas**: `/Users/lauferva/paragon/core/schemas.py`
- **Metrics**: `/Users/lauferva/paragon/infrastructure/metrics.py`

## License

Part of the Paragon project. See main project LICENSE.
