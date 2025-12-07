# Paragon.Forge.Linguist

**Agent C: The Linguist** - Semantic Masking Layer for Graph Datasets

## Overview

The Linguist module transforms generic graph structures into domain-specific datasets through **semantic masking**. It preserves the exact graph topology (nodes, edges, structure) while applying thematic overlays that change the semantics and generate realistic synthetic data.

## Core Concept

Think of it like translating a story into different languages - the plot (structure) stays the same, but the words (semantics) change completely.

```
Generic Graph           GENOMICS Theme         LOGISTICS Theme
    ┌───────┐              ┌───────┐              ┌───────┐
    │ Node1 │              │ TP53  │              │ WHX-01│
    └───┬───┘              └───┬───┘              └───┬───┘
        │ depends_on           │ regulates            │ ships_to
    ┌───▼───┐              ┌───▼───┐              ┌───▼───┐
    │ Node2 │              │ BRCA1 │              │ WHX-02│
    └───────┘              └───────┘              └───────┘
```

Same structure, different domain semantics!

## Features

- **Structure Preservation**: Graph topology remains identical (isomorphic)
- **Theme-Based Transformation**: Apply pre-built or custom domain themes
- **Realistic Data Generation**: Uses Faker for authentic-looking synthetic data
- **Deterministic**: Same seed produces same output (reproducible datasets)
- **Performance**: O(V+E) transformation with minimal memory overhead
- **msgspec Schemas**: Fast, type-safe configuration (NO PYDANTIC per CLAUDE.md)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MaskingLayer                         │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │   Source    │───▶│ ThemeFactory │───▶│  Themed  │  │
│  │   Graph     │    │  + Faker     │    │  Graph   │  │
│  └─────────────┘    └──────────────┘    └──────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Built-in Themes

### 1. GENOMICS
**Domain**: Molecular biology and genetics

**Node Types**: Gene, Protein, Pathway, Organism, Mutation

**Edge Types**: expresses, regulates, interacts_with, inhibits, activates

**Properties**:
- `sequence`: DNA sequence (e.g., "ATCGGCTA...")
- `organism`: Species name (e.g., "Homo sapiens")
- `chromosome`: Chromosome location (e.g., "chr7")
- `gene_symbol`: Gene identifier (e.g., "TP53")
- `protein_family`: Protein class (e.g., "Kinase")
- `expression_level`: Expression value (0.1-100.0)

**Edge Properties**:
- `confidence_score`: Interaction confidence (0.5-1.0)
- `interaction_type`: physical, genetic, regulatory, predicted
- `evidence`: experimental, computational, literature, database

### 2. LOGISTICS
**Domain**: Supply chain and transportation

**Node Types**: Warehouse, Factory, Distribution_Center, Store, Port

**Edge Types**: ships_to, supplies, receives_from, transports, routes_through

**Properties**:
- `location`: City and country (e.g., "Tokyo, Japan")
- `capacity`: Storage capacity (1000-100000)
- `inventory_count`: Current inventory (0-50000)
- `facility_code`: Unique code (e.g., "AB-1234")
- `operating_hours`: Daily hours (e.g., "24 hours/day")
- `manager`: Manager name

**Edge Properties**:
- `distance_km`: Route distance (10-5000 km)
- `transit_time_hours`: Transit time (1-72 hours)
- `shipping_cost`: Cost in USD (100-10000)
- `transport_mode`: truck, rail, ship, air, pipeline

### 3. SOCIAL
**Domain**: Social networks and user interaction

**Node Types**: User, Group, Post, Event, Organization

**Edge Types**: follows, friends_with, likes, shares, mentions, belongs_to

**Properties**:
- `username`: User handle (e.g., "john_doe_42")
- `display_name`: Display name (e.g., "John Doe")
- `email`: Email address
- `bio`: User bio (100 chars)
- `follower_count`: Number of followers (0-1000000)
- `verified`: Verified badge (boolean)
- `join_date`: Account creation date

**Edge Properties**:
- `timestamp`: Interaction time
- `interaction_count`: Number of interactions (1-1000)
- `relationship_type`: close_friend, acquaintance, family, colleague

### 4. FINANCE
**Domain**: Financial transactions and accounts

**Node Types**: Account, Company, Transaction, Portfolio, Asset

**Edge Types**: transfers_to, owns, invests_in, borrows_from, pays

**Properties**:
- `account_number`: Bank account number
- `company_name`: Company name
- `balance`: Account balance (0-1000000)
- `currency`: Currency code (e.g., "USD")
- `account_type`: checking, savings, investment, credit, loan
- `risk_rating`: low, medium, high, critical
- `created_date`: Account creation date

**Edge Properties**:
- `amount`: Transaction amount (10-100000)
- `transaction_date`: Transaction timestamp
- `status`: pending, completed, failed, reversed
- `transaction_type`: wire, ach, check, cash, credit

### 5. NETWORK
**Domain**: Computer network infrastructure

**Node Types**: Server, Router, Switch, Firewall, Device, Database

**Edge Types**: connects_to, routes_through, backs_up_to, monitors, secures

**Properties**:
- `hostname`: Server hostname (e.g., "web-server-AB12")
- `ip_address`: IPv4 address (e.g., "192.168.1.100")
- `mac_address`: MAC address
- `os`: Operating system (Linux, Windows Server, etc.)
- `cpu_cores`: Number of CPU cores (2-64)
- `memory_gb`: RAM in GB (8-256)
- `uptime_days`: Days since last reboot (0-365)
- `location`: Data center location

**Edge Properties**:
- `bandwidth_mbps`: Link bandwidth (100-40000 Mbps)
- `latency_ms`: Network latency (0.1-100 ms)
- `packet_loss`: Packet loss percentage (0-5%)
- `protocol`: TCP, UDP, ICMP, HTTP, HTTPS, SSH

## Usage

### Basic Usage

```python
from core.graph_db import ParagonDB
from forge.linguist import MaskingLayer, Themes

# Create or load a graph
graph = ParagonDB()
# ... populate graph with nodes and edges ...

# Apply a theme
masker = MaskingLayer(seed=42)
themed_graph = masker.apply(graph, theme=Themes.GENOMICS)

# Access themed data
for node in themed_graph.get_all_nodes():
    print(f"Label: {node.data['label']}")
    print(f"Type: {node.data['type']}")
    print(f"Organism: {node.data.get('organism', 'N/A')}")
```

### Custom Themes

```python
from forge.linguist import ThemeConfig, MaskingLayer

# Define custom theme
academic_theme = ThemeConfig(
    theme_name="academic",
    description="Academic research domain",
    node_name_pattern="Researcher_{name}",
    edge_name_pattern="{edge_type}",
    node_properties={
        "name": "name",
        "email": "email",
        "institution": "company",
        "field": "job",
        "h_index": "random_int",
    },
    edge_properties={
        "collaboration_count": "random_int",
        "joint_papers": "random_int",
    },
)

# Apply custom theme
masker = MaskingLayer(seed=42)
themed_graph = masker.apply(graph, theme=academic_theme)
```

### Preserve Original Data

```python
# Keep original data in node.data["original"]
themed_graph = masker.apply(
    graph,
    theme=Themes.LOGISTICS,
    preserve_original=True
)

for node in themed_graph.get_all_nodes():
    print(f"Themed: {node.data['label']}")
    print(f"Original type: {node.data['original']['type']}")
```

### Deterministic Generation

```python
# Same seed = same results
masker1 = MaskingLayer(seed=42)
graph1 = masker1.apply(source_graph, theme=Themes.SOCIAL)

masker2 = MaskingLayer(seed=42)
graph2 = masker2.apply(source_graph, theme=Themes.SOCIAL)

# graph1 and graph2 are identical!
```

## API Reference

### `MaskingLayer`

Main class for applying semantic masking.

**Constructor**:
```python
MaskingLayer(seed: Optional[int] = None)
```
- `seed`: Random seed for reproducible transformations

**Methods**:

#### `apply()`
```python
apply(
    graph: ParagonDB,
    theme: str | ThemeConfig,
    preserve_original: bool = False
) -> ParagonDB
```
Apply a theme to a graph.

- `graph`: Source graph to transform
- `theme`: Theme name (string) or ThemeConfig object
- `preserve_original`: If True, preserve original data in `node.data["original"]`
- Returns: New ParagonDB with theme applied

### `ThemeConfig`

Configuration for custom themes (msgspec.Struct).

**Fields**:
- `theme_name`: str - Theme identifier
- `node_name_pattern`: str - Format string (e.g., `"Gene_{id}"`)
- `edge_name_pattern`: str - Format string for edges
- `node_properties`: Dict[str, str] - property_name -> faker_method
- `edge_properties`: Dict[str, str] - property_name -> faker_method
- `description`: str - Human-readable description

### `Themes`

Registry of built-in theme names.

**Constants**:
- `Themes.GENOMICS`
- `Themes.LOGISTICS`
- `Themes.SOCIAL`
- `Themes.FINANCE`
- `Themes.NETWORK`

**Methods**:
- `Themes.all_themes()` -> Set[str]: Return all available theme names

### Helper Functions

#### `apply_theme()`
```python
apply_theme(
    graph: ParagonDB,
    theme: str | ThemeConfig,
    seed: Optional[int] = None,
    preserve_original: bool = False
) -> ParagonDB
```
Convenience function for one-line theme application.

#### `list_available_themes()`
```python
list_available_themes() -> List[str]
```
Return list of all built-in theme names.

## Faker Methods Reference

Common Faker methods available for custom themes:

**People**:
- `name`: Full name
- `first_name`: First name only
- `last_name`: Last name only
- `email`: Email address
- `phone_number`: Phone number
- `job`: Job title
- `company`: Company name

**Locations**:
- `address`: Full address
- `city`: City name
- `country`: Country name
- `street_address`: Street address

**Numbers**:
- `random_int`: Random integer
- `random_digit`: Single digit (0-9)
- `random_number`: Random number with digits

**Dates/Times**:
- `date`: Date object
- `date_time`: DateTime object
- `past_date`: Date in the past
- `future_date`: Date in the future

**Text**:
- `text`: Random text (default 200 chars)
- `sentence`: Random sentence
- `word`: Single word
- `paragraph`: Random paragraph

**Technical**:
- `ipv4`: IPv4 address
- `ipv6`: IPv6 address
- `mac_address`: MAC address
- `url`: URL
- `user_name`: Username
- `password`: Password

**Financial**:
- `credit_card_number`: Credit card number
- `currency_code`: Currency code (USD, EUR, etc.)
- `bban`: Bank account number

See [Faker documentation](https://faker.readthedocs.io/) for complete list.

## Performance

- **Time Complexity**: O(V + E) where V = nodes, E = edges
- **Space Complexity**: O(V + E) for new graph
- **Memory Overhead**: ~2x original graph size (new graph + original)
- **Typical Performance**:
  - 1000 nodes: < 100ms
  - 10000 nodes: < 1s
  - 100000 nodes: < 10s

## Design Principles

1. **Structure is Sacred**: Never modify graph topology, only semantics
2. **Deterministic**: Same seed produces identical results
3. **Composable**: Themes can be mixed or layered (future)
4. **Performant**: O(V+E) with minimal memory overhead
5. **Type-Safe**: msgspec.Struct for all schemas (NO PYDANTIC)
6. **Graph-Native**: Works directly with ParagonDB

## Testing

Run the test suite:
```bash
python forge/test_linguist.py
```

Run the demonstration:
```bash
python forge/demo_linguist.py
```

## Integration with Paragon.Forge

The Linguist is part of the Paragon.Forge ecosystem:

- **Topologist**: Generates graph skeletons (structure)
- **Linguist**: Applies semantic themes (semantics) ← YOU ARE HERE
- **Adversary**: Injects controlled corruption (testing)

**Typical Pipeline**:
```python
from forge import (
    create_barabasi_albert,  # Topologist
    MaskingLayer,            # Linguist
    create_adversary,        # Adversary (future)
)

# 1. Generate structure
graph = create_barabasi_albert(num_nodes=1000, num_edges=3, seed=42)

# 2. Apply semantics
masker = MaskingLayer(seed=42)
themed_graph = masker.apply(graph, theme=Themes.GENOMICS)

# 3. Inject corruption (for testing recovery algorithms)
# adversary = create_adversary(corruption_rate=0.1)
# corrupted_graph = adversary.corrupt(themed_graph)
```

## Examples

### Example 1: Generate Genomics Test Dataset

```python
from forge import create_barabasi_albert, MaskingLayer, Themes

# Create scale-free network (common in biology)
graph = create_barabasi_albert(num_nodes=500, num_edges=3, seed=42)

# Apply genomics theme
masker = MaskingLayer(seed=42)
genomics_graph = masker.apply(graph, theme=Themes.GENOMICS)

# Export for analysis
genomics_graph.save_parquet("genomics_network")
```

### Example 2: Generate Logistics Network

```python
from forge import create_watts_strogatz, MaskingLayer, Themes

# Create small-world network (common in logistics)
graph = create_watts_strogatz(num_nodes=100, k=6, p=0.1, seed=42)

# Apply logistics theme
masker = MaskingLayer(seed=42)
logistics_graph = masker.apply(graph, theme=Themes.LOGISTICS)

# Analyze supply chain
waves = logistics_graph.get_waves()
print(f"Supply chain depth: {len(waves)} levels")
```

### Example 3: Multi-Theme Comparison

```python
# Same structure, different themes
graph = create_erdos_renyi(num_nodes=50, edge_probability=0.1, seed=42)

themes = [Themes.GENOMICS, Themes.SOCIAL, Themes.NETWORK]
for theme in themes:
    masker = MaskingLayer(seed=42)
    themed = masker.apply(graph, theme=theme)
    print(f"\n{theme.upper()} THEME:")
    for node in list(themed.get_all_nodes())[:3]:
        print(f"  {node.data['label']}")
```

## Troubleshooting

### "Unknown theme" Error
```python
# Bad:
masker.apply(graph, theme="genomic")  # Typo!

# Good:
masker.apply(graph, theme=Themes.GENOMICS)
# Or:
masker.apply(graph, theme="genomics")
```

### "Faker has no method" Error
```python
# Bad:
ThemeConfig(
    node_properties={"invalid": "not_a_method"}  # Typo!
)

# Good - check Faker docs for valid methods:
ThemeConfig(
    node_properties={"name": "name", "email": "email"}
)
```

### Different Results with Same Seed
```python
# Make sure to set seed on MaskingLayer, not just theme:
masker = MaskingLayer(seed=42)  # ✓ Correct
themed = masker.apply(graph, theme=Themes.GENOMICS)
```

## Future Enhancements

- [ ] Theme composition (mix multiple themes)
- [ ] Conditional property generation based on node type
- [ ] Domain-specific constraints (e.g., valid IP ranges for NETWORK theme)
- [ ] Theme marketplace / plugin system
- [ ] Performance optimization for 1M+ node graphs
- [ ] Theme validation and testing framework

## Contributing

To add a new built-in theme:

1. Add theme constant to `Themes` class
2. Implement `build_<theme_name>_theme()` in `ThemeFactory`
3. Update `ThemeFactory.build()` dispatcher
4. Add documentation to this README
5. Add test cases to `test_linguist.py`

## License

Part of Project Paragon. See parent LICENSE file.
