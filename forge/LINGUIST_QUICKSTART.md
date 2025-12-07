# Linguist Quick Start Guide

Get started with Paragon.Forge.Linguist in 5 minutes!

## Installation

```bash
# Install dependencies
pip install faker>=20.0.0

# Verify installation
python -c "from forge import MaskingLayer, Themes; print('✓ Linguist ready!')"
```

## 30-Second Example

```python
from core.graph_db import ParagonDB
from core.schemas import NodeData, EdgeData
from forge.linguist import MaskingLayer, Themes

# 1. Create a simple graph
graph = ParagonDB()
node1 = NodeData.create(type="ENTITY", content="Entity 1")
node2 = NodeData.create(type="ENTITY", content="Entity 2")
graph.add_node(node1)
graph.add_node(node2)
graph.add_edge(EdgeData.depends_on(node2.id, node1.id))

# 2. Apply a theme
masker = MaskingLayer(seed=42)
genomics_graph = masker.apply(graph, theme=Themes.GENOMICS)

# 3. View results
for node in genomics_graph.get_all_nodes():
    print(f"{node.data['label']}: {node.data['organism']}")
```

Output:
```
Gene_TP53: Homo sapiens
Protein_BRCA1: Mus musculus
```

## Common Use Cases

### Use Case 1: Generate Test Dataset

```python
from forge import create_barabasi_albert, MaskingLayer, Themes

# Generate scale-free network (1000 nodes)
graph = create_barabasi_albert(num_nodes=1000, num_edges=3, seed=42)

# Transform to genomics domain
masker = MaskingLayer(seed=42)
genomics_graph = masker.apply(graph, theme=Themes.GENOMICS)

# Export for testing
genomics_graph.save_parquet("test_genomics_network")
```

### Use Case 2: Multi-Domain Testing

```python
# Test your algorithm on different domains
source_graph = load_my_graph()

themes = [Themes.GENOMICS, Themes.LOGISTICS, Themes.NETWORK]
for theme in themes:
    masker = MaskingLayer(seed=42)
    themed = masker.apply(source_graph, theme=theme)

    # Run your algorithm
    results = my_algorithm(themed)
    print(f"{theme}: {results}")
```

### Use Case 3: Custom Academic Network

```python
from forge.linguist import ThemeConfig, MaskingLayer

# Define custom theme
academic = ThemeConfig(
    theme_name="academic",
    node_name_pattern="Prof_{name}",
    edge_name_pattern="collaborates_with",
    node_properties={
        "name": "name",
        "university": "company",
        "field": "job",
        "h_index": "random_int",
    },
    edge_properties={
        "papers": "random_int",
        "years": "random_int",
    },
)

# Apply custom theme
masker = MaskingLayer(seed=42)
academic_graph = masker.apply(source_graph, theme=academic)
```

## All Built-in Themes

### GENOMICS - Molecular Biology
```python
genomics = masker.apply(graph, theme=Themes.GENOMICS)
# Nodes: Gene, Protein, Pathway, Organism, Mutation
# Edges: expresses, regulates, interacts_with, inhibits, activates
# Props: sequence, organism, gene_symbol, expression_level
```

### LOGISTICS - Supply Chain
```python
logistics = masker.apply(graph, theme=Themes.LOGISTICS)
# Nodes: Warehouse, Factory, Distribution_Center, Store, Port
# Edges: ships_to, supplies, receives_from, transports
# Props: location, capacity, facility_code, inventory_count
```

### SOCIAL - Social Networks
```python
social = masker.apply(graph, theme=Themes.SOCIAL)
# Nodes: User, Group, Post, Event, Organization
# Edges: follows, friends_with, likes, shares, mentions
# Props: username, email, bio, follower_count, verified
```

### FINANCE - Financial Networks
```python
finance = masker.apply(graph, theme=Themes.FINANCE)
# Nodes: Account, Company, Transaction, Portfolio, Asset
# Edges: transfers_to, owns, invests_in, borrows_from
# Props: account_number, balance, currency, risk_rating
```

### NETWORK - Computer Networks
```python
network = masker.apply(graph, theme=Themes.NETWORK)
# Nodes: Server, Router, Switch, Firewall, Device, Database
# Edges: connects_to, routes_through, backs_up_to, monitors
# Props: hostname, ip_address, os, cpu_cores, bandwidth_mbps
```

## Common Patterns

### Pattern 1: Reproducible Datasets

```python
# Same seed = identical output
masker1 = MaskingLayer(seed=42)
masker2 = MaskingLayer(seed=42)

graph1 = masker1.apply(source, theme=Themes.GENOMICS)
graph2 = masker2.apply(source, theme=Themes.GENOMICS)

# graph1 == graph2 (exactly the same)
```

### Pattern 2: Preserve Original Data

```python
# Keep original for debugging
themed = masker.apply(
    graph,
    theme=Themes.LOGISTICS,
    preserve_original=True
)

for node in themed.get_all_nodes():
    print(f"Themed: {node.data['label']}")
    print(f"Original: {node.data['original']['type']}")
```

### Pattern 3: Pipeline Integration

```python
# Complete Forge pipeline
from forge import (
    create_erdos_renyi,    # Topologist
    MaskingLayer,          # Linguist
    Themes,
)

# 1. Generate structure
graph = create_erdos_renyi(num_nodes=100, edge_probability=0.1, seed=42)

# 2. Apply semantics
masker = MaskingLayer(seed=42)
themed = masker.apply(graph, theme=Themes.SOCIAL)

# 3. Use in application
analyze_social_network(themed)
```

## API Cheat Sheet

### Import
```python
from forge.linguist import MaskingLayer, Themes, ThemeConfig, list_available_themes
```

### Create Masker
```python
masker = MaskingLayer(seed=42)  # seed is optional but recommended
```

### Apply Theme
```python
# Built-in theme
themed = masker.apply(graph, theme=Themes.GENOMICS)

# Custom theme
themed = masker.apply(graph, theme=my_config)

# With original data preserved
themed = masker.apply(graph, theme=Themes.LOGISTICS, preserve_original=True)
```

### List Themes
```python
themes = list_available_themes()
print(themes)  # ['finance', 'genomics', 'logistics', 'network', 'social']
```

### Access Themed Data
```python
for node in themed.get_all_nodes():
    label = node.data["label"]          # Human-readable name
    node_type = node.data["type"]       # Domain type (e.g., "Gene")
    # ... theme-specific properties ...
    organism = node.data.get("organism")  # For GENOMICS theme

for edge in themed.get_all_edges():
    edge_type = edge.type               # Domain edge type (e.g., "regulates")
    # ... theme-specific properties ...
    confidence = edge.metadata.get("confidence_score")  # For GENOMICS
```

## Quick Tips

### Tip 1: Always Use a Seed
```python
# Good - reproducible
masker = MaskingLayer(seed=42)

# Bad - different results each run
masker = MaskingLayer()  # Random seed
```

### Tip 2: Structure Never Changes
```python
# These are ALWAYS equal
assert original.node_count == themed.node_count
assert original.edge_count == themed.edge_count
assert len(original.get_waves()) == len(themed.get_waves())

# Only semantics change!
```

### Tip 3: Custom Properties Use Faker
```python
# See Faker docs for available methods:
# https://faker.readthedocs.io/

custom = ThemeConfig(
    node_properties={
        "name": "name",          # Person name
        "email": "email",        # Email address
        "city": "city",          # City name
        "value": "random_int",   # Random integer
        "date": "date",          # Date object
        # ... many more ...
    }
)
```

## Testing

### Run Tests
```bash
python forge/test_linguist.py
```

### Run Demo
```bash
python forge/demo_linguist.py
```

## Troubleshooting

### Issue: "Unknown theme"
```python
# Wrong
masker.apply(graph, theme="genomic")  # Typo!

# Right
masker.apply(graph, theme=Themes.GENOMICS)
# or
masker.apply(graph, theme="genomics")
```

### Issue: "Faker has no method"
```python
# Wrong
ThemeConfig(node_properties={"x": "not_a_real_method"})

# Right - check Faker docs
ThemeConfig(node_properties={"x": "name"})  # Valid method
```

### Issue: Different results with same seed
```python
# Wrong - seed not set
masker = MaskingLayer()

# Right - seed set for reproducibility
masker = MaskingLayer(seed=42)
```

## Next Steps

- 📖 Read full documentation: `forge/README_LINGUIST.md`
- 🏗️ Explore architecture: `forge/LINGUIST_ARCHITECTURE.md`
- 🎯 See implementation details: `forge/LINGUIST_IMPLEMENTATION_SUMMARY.md`
- 💻 Check source code: `forge/linguist.py`

## Getting Help

1. Check the README: `forge/README_LINGUIST.md`
2. Run the demo: `python forge/demo_linguist.py`
3. Explore examples in `forge/test_linguist.py`

## Complete Example

```python
#!/usr/bin/env python3
"""Complete example: Generate and theme a graph."""

from forge import create_barabasi_albert, MaskingLayer, Themes

# 1. Generate structure (scale-free network)
print("1. Generating graph structure...")
graph = create_barabasi_albert(
    num_nodes=100,
    num_edges=3,
    seed=42
)
print(f"   Created graph: {graph.node_count} nodes, {graph.edge_count} edges")

# 2. Apply genomics theme
print("\n2. Applying GENOMICS theme...")
masker = MaskingLayer(seed=42)
genomics_graph = masker.apply(graph, theme=Themes.GENOMICS)

# 3. Explore results
print("\n3. Sample themed nodes:")
for node in list(genomics_graph.get_all_nodes())[:5]:
    print(f"   - {node.data['label']}")
    print(f"     Organism: {node.data['organism']}")
    print(f"     Type: {node.data['type']}")

print("\n4. Sample themed edges:")
for edge in list(genomics_graph.get_all_edges())[:5]:
    src = genomics_graph.get_node(edge.source_id)
    tgt = genomics_graph.get_node(edge.target_id)
    print(f"   - {src.data['label']} --[{edge.type}]--> {tgt.data['label']}")

# 4. Export dataset
print("\n5. Exporting dataset...")
genomics_graph.save_parquet("genomics_network")
print("   Saved to genomics_network.nodes.parquet and .edges.parquet")

print("\n✓ Complete!")
```

---

**You're ready to use the Linguist!** 🎉

Transform your graphs into realistic domain-specific datasets in just a few lines of code.
