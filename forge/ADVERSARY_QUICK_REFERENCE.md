# Adversary Module - Quick Reference

## Import

```python
from forge.adversary import EntropyModule, AdversaryConfig, create_adversary
```

## Error Types Cheat Sheet

| Error Type | Simulates | Example Rate |
|------------|-----------|--------------|
| `drop_edges` | Packet loss, broken links | 0.05 (5%) |
| `drop_nodes` | Node failures, deletions | 0.02 (2%) |
| `mutate_strings` | OCR errors, typos | 0.02 (2%) |
| `mutate_numbers` | Sensor noise | 0.01 (1%) |
| `swap_labels` | Mislabeling | 0.03 (3%) |
| `lag_timestamps` | Out-of-order delivery | 0.02 (2%) |
| `duplicate_nodes` | Data entry errors | 0.01 (1%) |
| `null_properties` | Missing data | 0.02 (2%) |

## Quick Start

```python
# Create adversary
adversary = EntropyModule(seed=42)

# Add errors
adversary.add_error(AdversaryConfig(error_type="drop_edges", rate=0.05))

# Corrupt graph
corrupted = adversary.corrupt(original_graph)

# Get answer key
manifest = adversary.get_manifest()
manifest.to_json("answer_key.json")
```

## Presets

```python
# Light (subtle)
adversary = create_adversary(seed=42, preset="light")

# Medium (balanced)
adversary = create_adversary(seed=42, preset="medium")

# Heavy (stress test)
adversary = create_adversary(seed=42, preset="heavy")

# Realistic (observed patterns)
adversary = create_adversary(seed=42, preset="realistic")
```

## Error Type Parameters

### mutate_strings
```python
params={
    "property": "content",  # Which property to corrupt
    "mutation_type": "typo"  # typo|swap|delete|insert
}
```

### mutate_numbers
```python
params={
    "property": "version",   # Which numeric property
    "noise_factor": 0.1      # ±10% noise
}
```

### lag_timestamps
```python
params={
    "lag_seconds": 3600  # ±1 hour in seconds
}
```

### null_properties
```python
params={
    "properties": ["content", "status"]  # List of properties to null
}
```

## Manifest Structure

```python
manifest.world_id              # Unique identifier
manifest.seed                  # RNG seed
manifest.total_modifications   # Total changes
manifest.modifications         # List[Modification]
manifest.error_summary         # Dict[error_type, count]
```

## Modification Record

```python
mod.error_type        # Type of corruption
mod.target_type       # "node" or "edge"
mod.target_id         # UUID of target
mod.original_value    # Before corruption
mod.corrupted_value   # After corruption
mod.timestamp         # When it happened
mod.metadata          # Extra context
```

## Recovery Grading Template

```python
# 1. Corrupt
adversary = EntropyModule(seed=42)
adversary.add_error(AdversaryConfig(error_type="mutate_strings", rate=0.20))
corrupted = adversary.corrupt(graph)
manifest = adversary.get_manifest()

# 2. Recover
recovered = your_recovery_algorithm(corrupted)

# 3. Grade
correct = sum(
    1 for mod in manifest.modifications
    if get_value(recovered, mod.target_id) == mod.original_value
)
accuracy = correct / len(manifest.modifications)
print(f"Accuracy: {accuracy * 100:.1f}%")
```

## Common Patterns

### Test All Error Types
```python
adversary = EntropyModule(seed=42)
for error_type in ["drop_edges", "drop_nodes", "mutate_strings",
                   "mutate_numbers", "swap_labels", "lag_timestamps",
                   "duplicate_nodes", "null_properties"]:
    adversary.add_error(AdversaryConfig(error_type=error_type, rate=0.05))
```

### Deterministic Testing
```python
# Run 1
adversary1 = EntropyModule(seed=999)
adversary1.add_error(AdversaryConfig(error_type="drop_edges", rate=0.5))
result1 = adversary1.corrupt(graph)

# Run 2 (identical)
adversary2 = EntropyModule(seed=999)
adversary2.add_error(AdversaryConfig(error_type="drop_edges", rate=0.5))
result2 = adversary2.corrupt(graph)

# Results are identical
assert adversary1.get_manifest().total_modifications == \
       adversary2.get_manifest().total_modifications
```

### Progressive Corruption
```python
graph = original_graph
for day in range(30):
    adversary = EntropyModule(seed=day)
    adversary.add_error(AdversaryConfig(error_type="mutate_strings", rate=0.001))
    graph = adversary.corrupt(graph)
# Simulates 30 days of gradual degradation
```

## Testing Commands

```bash
# Unit tests
pytest tests/unit/forge/test_adversary.py -v

# Integration tests
pytest tests/integration/test_adversary_integration.py -v

# All adversary tests
pytest tests/unit/forge/test_adversary.py \
       tests/integration/test_adversary_integration.py -v
```

## File Locations

| File | Path |
|------|------|
| Implementation | `forge/adversary.py` |
| Unit Tests | `tests/unit/forge/test_adversary.py` |
| Integration Tests | `tests/integration/test_adversary_integration.py` |
| Full Documentation | `forge/ADVERSARY_README.md` |
| Examples | `examples/adversary_demo.py` |

## Performance Notes

- **Complexity**: O(N) where N = nodes + edges
- **Memory**: 2x graph size (creates copy)
- **Determinism**: Perfect via seeding
- **Thread Safety**: Not thread-safe (use per-thread instances)
