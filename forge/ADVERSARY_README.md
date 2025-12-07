# Paragon Forge - Adversary Module

**Agent D: The Adversary** - Controlled corruption engine for testing recovery algorithms.

## Overview

The Adversary module introduces controlled errors into graph data while maintaining complete provenance of every modification. The **Manifest** is the Answer Key that enables grading any recovery algorithm.

## Design Philosophy

- **REPRODUCIBLE**: All corruption is seeded for determinism
- **TRACEABLE**: Every single modification is logged with before/after state
- **VARIED**: 8 different error types covering common real-world failure modes
- **REALISTIC**: Error rates and patterns match observed data quality issues
- **NO PYDANTIC**: All schemas use `msgspec.Struct` for performance

## Error Types

### 1. `drop_edges`
Remove edges to simulate packet loss or broken links.

```python
AdversaryConfig(error_type="drop_edges", rate=0.05)
```

### 2. `drop_nodes`
Remove nodes to simulate failures or deletions.

```python
AdversaryConfig(error_type="drop_nodes", rate=0.02)
```

### 3. `mutate_strings`
Character-level mutations to simulate OCR/sequencing errors.

```python
AdversaryConfig(
    error_type="mutate_strings",
    rate=0.02,
    params={
        "property": "content",  # Which property to mutate
        "mutation_type": "typo"  # typo, swap, delete, or insert
    }
)
```

### 4. `mutate_numbers`
Add noise to numeric values.

```python
AdversaryConfig(
    error_type="mutate_numbers",
    rate=0.01,
    params={
        "property": "version",
        "noise_factor": 0.1  # ±10% noise
    }
)
```

### 5. `swap_labels`
Swap labels/types between nodes to simulate mislabeling.

```python
AdversaryConfig(error_type="swap_labels", rate=0.03)
```

### 6. `lag_timestamps`
Shift timestamps to simulate out-of-order delivery.

```python
AdversaryConfig(
    error_type="lag_timestamps",
    rate=0.02,
    params={"lag_seconds": 3600}  # ±1 hour
)
```

### 7. `duplicate_nodes`
Create duplicates to simulate data entry errors.

```python
AdversaryConfig(error_type="duplicate_nodes", rate=0.01)
```

### 8. `null_properties`
Set properties to None/empty to simulate missing data.

```python
AdversaryConfig(
    error_type="null_properties",
    rate=0.02,
    params={"properties": ["content", "status"]}
)
```

## Usage

### Basic Example

```python
from forge.adversary import EntropyModule, AdversaryConfig
from core.graph_db import ParagonDB

# Create adversary with seed for reproducibility
adversary = EntropyModule(seed=42)

# Add error types
adversary.add_error(AdversaryConfig(
    error_type="drop_edges",
    rate=0.05  # 5% of edges
))

adversary.add_error(AdversaryConfig(
    error_type="mutate_strings",
    rate=0.02,
    params={"property": "content", "mutation_type": "typo"}
))

# Apply to graph
original_graph = ParagonDB()
# ... populate graph ...

corrupted_graph = adversary.corrupt(original_graph)

# Get the Answer Key
manifest = adversary.get_manifest()
manifest.to_json("answer_key.json")

print(f"Applied {manifest.total_modifications} corruptions")
print(f"Error breakdown: {manifest.error_summary}")
```

### Using Presets

```python
from forge.adversary import create_adversary

# Light corruption (subtle errors)
adversary = create_adversary(seed=42, preset="light")

# Medium corruption (balanced testing)
adversary = create_adversary(seed=42, preset="medium")

# Heavy corruption (stress testing)
adversary = create_adversary(seed=42, preset="heavy")

# Realistic (based on observed data quality issues)
adversary = create_adversary(seed=42, preset="realistic")

corrupted = adversary.corrupt(graph)
```

### Grading Recovery Algorithms

```python
# 1. Corrupt the graph
adversary = EntropyModule(seed=42)
adversary.add_error(AdversaryConfig(
    error_type="mutate_strings",
    rate=0.20
))

corrupted = adversary.corrupt(original_graph)
manifest = adversary.get_manifest()

# 2. Run your recovery algorithm
recovered = my_recovery_algorithm(corrupted)

# 3. Grade accuracy using the Answer Key
correct_recoveries = 0
total_corruptions = 0

for mod in manifest.modifications:
    total_corruptions += 1

    # Get the corrupted node
    node_id = mod.target_id
    recovered_value = get_node_property(recovered, node_id, "content")

    # Compare against original
    if recovered_value == mod.original_value:
        correct_recoveries += 1

accuracy = correct_recoveries / total_corruptions
print(f"Recovery accuracy: {accuracy * 100:.1f}%")
```

## Manifest Structure

The Manifest is the complete audit trail of all corruptions:

```python
class Manifest(msgspec.Struct):
    world_id: str              # Identifier for this corrupted world
    seed: int                  # RNG seed for reproducibility
    total_modifications: int   # Count of changes
    modifications: List[Modification]  # All changes
    error_summary: Dict[str, int]      # error_type -> count
    created_at: str           # ISO8601 timestamp
```

### Modification Record

Each modification contains:

```python
class Modification(msgspec.Struct):
    error_type: str           # Type of corruption
    target_type: str          # "node" or "edge"
    target_id: str           # UUID of affected element
    original_value: Any      # Pre-corruption state
    corrupted_value: Any     # Post-corruption state
    timestamp: str           # ISO8601 timestamp
    metadata: Dict[str, Any] # Extra context
```

## Deterministic Corruption

Same seed = same corruption:

```python
# Run 1
adversary1 = EntropyModule(seed=42)
adversary1.add_error(AdversaryConfig(error_type="drop_edges", rate=0.5))
corrupted1 = adversary1.corrupt(graph)

# Run 2 (same seed)
adversary2 = EntropyModule(seed=42)
adversary2.add_error(AdversaryConfig(error_type="drop_edges", rate=0.5))
corrupted2 = adversary2.corrupt(graph)

# Identical results
assert adversary1.get_manifest().total_modifications == \
       adversary2.get_manifest().total_modifications
```

## Use Cases

### 1. Testing Graph Repair Algorithms

```python
# Create controlled corruption
adversary = EntropyModule(seed=42)
adversary.add_error(AdversaryConfig(error_type="drop_edges", rate=0.20))
corrupted = adversary.corrupt(graph)

# Run repair algorithm
repaired = graph_repair_algorithm(corrupted)

# Grade accuracy
manifest = adversary.get_manifest()
# Compare repaired edges against manifest.modifications
```

### 2. Validating Invariant Checking

```python
# Corrupt a graph
adversary = create_adversary(seed=42, preset="medium")
corrupted = adversary.corrupt(graph)

# Run invariant checker
violations = check_graph_invariants(corrupted)

# Verify checker caught the corruptions
manifest = adversary.get_manifest()
# violations should detect errors logged in manifest
```

### 3. Simulating Data Degradation

```python
# Simulate data quality issues over time
for day in range(30):
    adversary = EntropyModule(seed=day)
    adversary.add_error(AdversaryConfig(
        error_type="mutate_strings",
        rate=0.001  # 0.1% per day
    ))
    graph = adversary.corrupt(graph)

# After 30 days, accumulated corruption
```

### 4. Benchmarking Recovery Accuracy

```python
presets = ["light", "medium", "heavy"]
results = {}

for preset in presets:
    adversary = create_adversary(seed=42, preset=preset)
    corrupted = adversary.corrupt(graph)
    recovered = recovery_algorithm(corrupted)

    accuracy = grade_recovery(recovered, adversary.get_manifest())
    results[preset] = accuracy

# Plot accuracy vs corruption level
```

## Performance Characteristics

- **Corruption**: O(N) where N = nodes + edges
- **Manifest size**: Linear in total modifications
- **Memory**: Creates deep copy of graph (2x memory usage during corruption)
- **Determinism**: Perfect reproducibility via RNG seeding

## Testing

Run the test suite:

```bash
pytest tests/unit/forge/test_adversary.py -v
```

All 26 tests should pass, covering:
- All 8 error types
- Manifest serialization
- Deterministic corruption
- Preset configurations
- Edge cases (empty graphs, zero rates)
- Recovery grading workflow

## Integration with Paragon

The Adversary module is part of the Paragon.Forge suite:

```python
from forge import (
    EntropyModule,      # This module
    AdversaryConfig,
    create_adversary,

    # Other Forge components
    GraphGenerator,     # Topologist
    MaskingLayer,       # Linguist
    DistributionEngine, # Statistician
)
```

## Architecture Notes

### msgspec.Struct (NOT Pydantic)

All schemas use `msgspec.Struct` for:
- ~3x less memory than dict
- ~10x faster serialization than Pydantic
- O(1) attribute access
- Type safety without runtime overhead

### Graph-Native Design

Works directly with `rustworkx.PyDiGraph` via `ParagonDB`:
- No conversion to NetworkX
- Preserves graph performance characteristics
- Maintains all Paragon node/edge metadata

### Immutable Manifests

Modifications are append-only:
- Complete audit trail
- No overwrites
- Cryptographically sign manifest for tamper detection (future)

## Future Enhancements

- [ ] Cryptographic signing of manifests
- [ ] Differential privacy guarantees
- [ ] Adaptive corruption rates based on graph structure
- [ ] Time-series corruption patterns
- [ ] Multi-graph corruption (corrupt graph family together)
- [ ] Recovery hints in manifest (guide algorithms)

## References

- [Paragon Protocol v4.0](../CLAUDE.md)
- [Core Schemas](../core/schemas.py)
- [Graph Database](../core/graph_db.py)
- [Forge Test Suite](../tests/unit/forge/test_adversary.py)
