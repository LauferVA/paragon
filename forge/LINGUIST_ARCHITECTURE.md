# Linguist Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PARAGON.FORGE.LINGUIST                          │
│                  Semantic Masking Layer (Agent C)                   │
└─────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │  User Input  │
                              │  - Graph     │
                              │  - Theme     │
                              │  - Seed      │
                              └──────┬───────┘
                                     │
                                     ▼
                          ┌──────────────────┐
                          │  MaskingLayer    │
                          │  (Main API)      │
                          └────────┬─────────┘
                                   │
                     ┌─────────────┼─────────────┐
                     ▼             ▼             ▼
              ┌───────────┐ ┌───────────┐ ┌──────────────┐
              │ Theme     │ │   Faker   │ │  ParagonDB   │
              │ Factory   │ │ Generator │ │   Bridge     │
              └─────┬─────┘ └─────┬─────┘ └──────┬───────┘
                    │             │              │
                    │ Compile     │ Generate     │ Transform
                    │ Theme       │ Properties   │ Structure
                    │             │              │
                    └─────────────┼──────────────┘
                                  │
                                  ▼
                         ┌────────────────┐
                         │  Themed Graph  │
                         │  (ParagonDB)   │
                         └────────────────┘
```

## Component Architecture

### 1. MaskingLayer (Main Engine)

```
MaskingLayer
├── __init__(seed)
│   └── Creates ThemeFactory with seed
│
├── apply(graph, theme, preserve_original)
│   ├── Build/compile theme
│   ├── Build node index map (deterministic)
│   ├── Transform all nodes → themed nodes
│   ├── Transform all edges → themed edges
│   └── Return new ParagonDB
│
├── _transform_node(node, theme, preserve_original)
│   ├── Generate themed properties (Faker)
│   ├── Assign domain-specific type
│   ├── Generate label from pattern
│   └── Create new NodeData
│
└── _transform_edge(edge, theme, node_id_map, preserve_original)
    ├── Generate themed properties (Faker)
    ├── Assign domain-specific edge type
    └── Create new EdgeData
```

### 2. ThemeFactory (Theme Builder)

```
ThemeFactory
├── __init__(seed)
│   └── Creates Faker instance with seed
│
├── build(theme_name: str) → Theme
│   ├── "genomics"   → build_genomics_theme()
│   ├── "logistics"  → build_logistics_theme()
│   ├── "social"     → build_social_theme()
│   ├── "finance"    → build_finance_theme()
│   └── "network"    → build_network_theme()
│
├── build_from_config(ThemeConfig) → Theme
│   ├── Compile node property generators
│   ├── Compile edge property generators
│   └── Return Theme object
│
└── Helper methods
    ├── _compile_faker_method(method_name)
    ├── _generate_dna_sequence()
    └── _generate_gene_symbol()
```

### 3. Schema Layer (msgspec.Struct)

```
ThemeConfig (User-facing)
├── theme_name: str
├── node_name_pattern: str
├── edge_name_pattern: str
├── node_properties: Dict[str, str]  # prop → faker_method
├── edge_properties: Dict[str, str]
└── description: str

Theme (Internal)
├── name: str
├── description: str
├── node_types: List[str]           # Domain types
├── edge_types: List[str]           # Domain edge types
├── node_generators: Dict[str, Callable]  # Compiled functions
├── edge_generators: Dict[str, Callable]
├── node_name_pattern: str
└── edge_name_pattern: str
```

## Data Flow

### Transformation Pipeline

```
┌─────────────┐
│ Source Node │  Generic node with UUID, type, content
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│ Theme Selection                 │
│ - Choose domain (e.g. GENOMICS) │
│ - Select node type (e.g. Gene)  │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ Property Generation (Faker)     │
│ - sequence: ATCGGCTA...         │
│ - organism: Homo sapiens        │
│ - gene_symbol: TP53             │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ Label Generation                │
│ - Pattern: "Gene_{gene_symbol}" │
│ - Result: "Gene_TP53"           │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│ New NodeData Creation           │
│ - type: "Gene"                  │
│ - data["label"]: "Gene_TP53"    │
│ - data["sequence"]: "ATC..."    │
│ - content: themed description   │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────┐
│ Themed Node │  Domain-specific node ready for use
└─────────────┘
```

### Graph Transformation Flow

```
INPUT: Generic Graph               OUTPUT: Themed Graph
┌───────────────────┐              ┌───────────────────┐
│  Node1 (ENTITY)   │              │  TP53 (Gene)      │
│  - id: abc123     │   ────▶      │  - id: xyz789     │
│  - content: "..." │              │  - organism: ...  │
└─────────┬─────────┘              └─────────┬─────────┘
          │ depends_on                       │ regulates
          ▼                                  ▼
┌───────────────────┐              ┌───────────────────┐
│  Node2 (ENTITY)   │   ────▶      │  BRCA1 (Gene)     │
│  - id: def456     │              │  - id: uvw012     │
│  - content: "..." │              │  - sequence: ...  │
└───────────────────┘              └───────────────────┘

PRESERVED:                         TRANSFORMED:
✓ Node count: 2                    ✗ Node IDs
✓ Edge count: 1                    ✗ Node types
✓ Graph topology (DAG)             ✗ Node content
✓ Wave structure                   ✗ Edge types
✓ Connectivity pattern             ✗ Property names/values
```

## Theme Structure

### Built-in Theme Hierarchy

```
Themes (Registry)
├── GENOMICS
│   ├── Node Types: [Gene, Protein, Pathway, Organism, Mutation]
│   ├── Edge Types: [expresses, regulates, interacts_with, ...]
│   ├── Node Props: {sequence, organism, chromosome, ...}
│   └── Edge Props: {confidence_score, interaction_type, ...}
│
├── LOGISTICS
│   ├── Node Types: [Warehouse, Factory, Distribution_Center, ...]
│   ├── Edge Types: [ships_to, supplies, receives_from, ...]
│   ├── Node Props: {location, capacity, facility_code, ...}
│   └── Edge Props: {distance_km, transit_time_hours, ...}
│
├── SOCIAL
│   ├── Node Types: [User, Group, Post, Event, Organization]
│   ├── Edge Types: [follows, friends_with, likes, shares, ...]
│   ├── Node Props: {username, email, bio, follower_count, ...}
│   └── Edge Props: {timestamp, interaction_count, ...}
│
├── FINANCE
│   ├── Node Types: [Account, Company, Transaction, Portfolio, Asset]
│   ├── Edge Types: [transfers_to, owns, invests_in, ...]
│   ├── Node Props: {account_number, balance, currency, ...}
│   └── Edge Props: {amount, transaction_date, status, ...}
│
└── NETWORK
    ├── Node Types: [Server, Router, Switch, Firewall, Device, Database]
    ├── Edge Types: [connects_to, routes_through, backs_up_to, ...]
    ├── Node Props: {hostname, ip_address, os, cpu_cores, ...}
    └── Edge Props: {bandwidth_mbps, latency_ms, protocol, ...}
```

## Integration Points

### 1. Paragon Core Integration

```
┌─────────────────────┐
│   ParagonDB         │  Graph database (rustworkx)
│   - add_node()      │
│   - add_edge()      │
│   - get_waves()     │ ◄─── Used by Linguist
│   - topological_... │
└─────────────────────┘
         ▲
         │ Creates new instance
         │
┌─────────────────────┐
│   MaskingLayer      │  Semantic transformer
│   - apply()         │
│   - _transform_...  │
└─────────────────────┘
```

### 2. Paragon.Forge Integration

```
┌─────────────────────┐
│   Topologist        │  Structure generation
│   - create_erdos... │
│   - create_barabasi │
└─────────┬───────────┘
          │ Generates
          ▼
     [Generic Graph]
          │
          │ Transform
          ▼
┌─────────────────────┐
│   Linguist          │  Semantic masking
│   - apply()         │ ◄─── YOU ARE HERE
└─────────┬───────────┘
          │ Produces
          ▼
    [Themed Graph]
          │
          │ Corrupt (future)
          ▼
┌─────────────────────┐
│   Adversary         │  Controlled corruption
│   - corrupt()       │
└─────────────────────┘
```

### 3. Schema Integration

```
core/schemas.py
├── NodeData (msgspec.Struct)
│   ├── id: str
│   ├── type: str
│   ├── content: str
│   ├── data: Dict[str, Any] ◄─── Linguist adds themed properties here
│   └── ...
│
└── EdgeData (msgspec.Struct)
    ├── source_id: str
    ├── target_id: str
    ├── type: str
    ├── metadata: Dict[str, Any] ◄─── Linguist adds themed properties here
    └── ...

forge/linguist.py
├── ThemeConfig (msgspec.Struct)
│   └── User-facing configuration
│
└── Theme (msgspec.Struct)
    └── Compiled theme with generators
```

## Execution Flow Example

### Apply GENOMICS Theme

```
1. User Call
   └─▶ masker.apply(graph, theme=Themes.GENOMICS)

2. Theme Factory
   ├─▶ factory.build("genomics")
   ├─▶ build_genomics_theme()
   └─▶ Returns Theme with:
       - node_types: [Gene, Protein, ...]
       - node_generators: {sequence: λ, organism: λ, ...}

3. Node Index Map
   └─▶ Assigns sequential index to each node (deterministic)

4. For each node in source graph:
   ├─▶ Select node type: node_types[index % len(node_types)]
   │   Example: Gene (index 0), Protein (index 1), ...
   │
   ├─▶ Generate properties:
   │   ├─▶ sequence = _generate_dna_sequence() → "ATCGGCTA..."
   │   ├─▶ organism = faker.random_element([...]) → "Homo sapiens"
   │   └─▶ gene_symbol = _generate_gene_symbol() → "TP53"
   │
   ├─▶ Generate label:
   │   └─▶ Pattern: "Gene_{gene_symbol}" → "Gene_TP53"
   │
   └─▶ Create new NodeData:
       ├─▶ type = "Gene"
       ├─▶ data["label"] = "Gene_TP53"
       ├─▶ data["sequence"] = "ATCGGCTA..."
       └─▶ data["organism"] = "Homo sapiens"

5. For each edge in source graph:
   ├─▶ Select edge type: edge_types[hash(orig_type) % len(edge_types)]
   │   Example: "regulates"
   │
   ├─▶ Generate edge properties:
   │   ├─▶ confidence_score = random.uniform(0.5, 1.0) → 0.847
   │   └─▶ interaction_type = faker.random_element([...]) → "physical"
   │
   └─▶ Create new EdgeData:
       ├─▶ type = "regulates"
       ├─▶ metadata["confidence_score"] = 0.847
       └─▶ metadata["interaction_type"] = "physical"

6. Return new ParagonDB
   └─▶ Themed graph with same structure, different semantics
```

## Performance Model

### Time Complexity

```
apply(graph, theme):
  O(1)   - Build/compile theme
  O(V)   - Build node index map
  O(V)   - Transform all nodes
  O(E)   - Transform all edges
  ────────
  O(V+E) - Total complexity
```

### Space Complexity

```
Memory Usage:
  Input Graph:        V × NodeSize + E × EdgeSize
  Output Graph:       V × NodeSize + E × EdgeSize
  Node Index Map:     V × (UUID + int)
  Theme Generators:   O(1)
  ─────────────────────────────────────────────
  Total:              ~2 × (V + E)
```

### Benchmark Expectations

```
Graph Size    │ Expected Time │ Memory Usage
─────────────┼───────────────┼──────────────
100 nodes     │     <10ms     │    ~1 MB
1,000 nodes   │    <100ms     │   ~10 MB
10,000 nodes  │     ~1s       │  ~100 MB
100,000 nodes │    ~10s       │   ~1 GB
```

## Error Handling

```
Error Flow:

User Input
    │
    ├─▶ Invalid theme name
    │   └─▶ ValueError: "Unknown theme: ..."
    │
    ├─▶ Invalid ThemeConfig
    │   ├─▶ Invalid faker method
    │   │   └─▶ ValueError: "Faker has no method: ..."
    │   └─▶ Invalid pattern format
    │       └─▶ Graceful fallback to default pattern
    │
    └─▶ Invalid graph
        ├─▶ Empty graph
        │   └─▶ Returns empty themed graph
        └─▶ Invalid node/edge data
            └─▶ Propagates ParagonDB error
```

## Determinism Guarantee

```
Seed Management:

User provides seed → MaskingLayer(seed=42)
                          │
                          ▼
                    ThemeFactory(seed=42)
                          │
                ┌─────────┼─────────┐
                ▼         ▼         ▼
           Faker.seed(42)          random.seed(42)
           random.seed(42)
                │
                ▼
         All random generation
         uses seeded RNG
                │
                ▼
         Deterministic output
         (same seed = same result)
```

## Extension Points

### Adding New Themes

```
1. Add to Themes registry:
   class Themes:
       NEW_THEME = "new_theme"

2. Implement builder:
   def build_new_theme(self) -> Theme:
       return Theme(
           name="new_theme",
           node_types=[...],
           edge_types=[...],
           node_generators={...},
           edge_generators={...},
           ...
       )

3. Update dispatcher:
   def build(self, theme_name: str) -> Theme:
       builders = {
           ...
           Themes.NEW_THEME: self.build_new_theme,
       }
```

### Custom Property Generators

```
class ThemeFactory:
    def _custom_generator(self) -> Any:
        """Custom generator beyond Faker."""
        # Custom logic here
        return custom_value

    def build_custom_theme(self) -> Theme:
        return Theme(
            node_generators={
                "custom_prop": lambda: self._custom_generator()
            }
        )
```

## Summary

The Linguist architecture is designed for:

✓ **Simplicity**: Single main class (MaskingLayer) with clear API
✓ **Extensibility**: Easy to add new themes or customize existing ones
✓ **Performance**: O(V+E) with minimal memory overhead
✓ **Determinism**: Seed-based reproducibility
✓ **Integration**: Seamless with Paragon core and Forge ecosystem
✓ **Type Safety**: Full msgspec.Struct schemas
✓ **Maintainability**: Clear separation of concerns

The system transforms graph semantics while preserving topology, enabling realistic domain-specific test datasets for algorithm development and validation.
