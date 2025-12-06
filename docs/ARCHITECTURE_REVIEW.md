# PARAGON Architecture Review

## Executive Summary

This document analyzes the architectural transformation from GAADP (Python/NetworkX) to Paragon (Rust-accelerated/Graph-Native), examining tradeoffs, critical decisions, and potential pitfalls.

---

## 1. THE CORE TRANSFORMATION

### 1.1 The Python-Rust Sandwich

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION (Python)                       │
│  LangGraph state machines, AsyncIO, Agent dispatch              │
│  ↓ Crosses boundary ONCE per batch ↓                            │
├─────────────────────────────────────────────────────────────────┤
│                    COMPUTE (Rust via PyO3)                      │
│  rustworkx graph storage, Polars DataFrames, pygmtools          │
│  ↓ Zero-copy Arrow IPC ↓                                        │
├─────────────────────────────────────────────────────────────────┤
│                    VISUALIZATION (WebGPU)                       │
│  Cosmograph, Arrow streams, real-time rendering                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight**: The boundary between Python and Rust is the critical performance chokepoint. Every crossing has overhead. The design minimizes crossings via:

1. **Batch operations**: `add_nodes_from([...])` not `for n: add_node(n)`
2. **Lazy evaluation**: Polars `scan_csv()` pushes filters to Rust
3. **Arrow IPC**: Zero-copy data transfer to visualization

### 1.2 What We're Replacing vs Keeping

| Component | GAADP | Paragon | Action |
|-----------|-------|---------|--------|
| Graph storage | NetworkX | rustworkx | **REPLACE** |
| Wave computation | Python recursion | `rx.layers()` | **REPLACE** |
| Data loading | iterative rows | Polars batch | **REPLACE** |
| Code parsing | `ast` module | tree-sitter | **REPLACE** |
| Graph alignment | numpy manual | pygmtools | **REPLACE** |
| Ontology/Protocols | Pydantic | msgspec (hot) + Pydantic (cold) | **ADAPT** |
| Transition Matrix | Dict[Tuple] | Dict[Tuple] | **KEEP** |
| Agent Dispatch | Dict[Tuple] | Dict[Tuple] | **KEEP** |
| Agent Manifest | YAML | YAML | **KEEP** |
| LangGraph adapter | asyncio | asyncio | **KEEP** |

---

## 2. CRITICAL ARCHITECTURAL DECISIONS

### 2.1 Decision: Dual Schema Strategy (msgspec + Pydantic)

**Problem**: Pydantic is convenient but slow for high-frequency operations.

**Analysis**:
- `NodeData` and `EdgeData` are touched on every graph operation
- Agent outputs (`ArchitectOutput`, etc.) are touched once per LLM call
- LLM calls cost ~$0.01 and take ~2s; schema overhead is negligible
- Graph operations happen 10,000x per second; overhead dominates

**Decision**:
```python
# HOT PATH: msgspec for graph primitives
class NodeData(msgspec.Struct):
    id: str
    type: str
    content: Optional[str]
    metadata: Dict[str, Any]
    status: str

# COLD PATH: Pydantic for LLM protocols (keep validation richness)
class ArchitectOutput(BaseModel):
    reasoning: Optional[str]
    new_nodes: List[NodeSpec]
    ...
```

**Tradeoff**:
- ✅ 3-10x faster serialization on hot path
- ✅ Keep Pydantic's rich validation for LLM outputs
- ⚠️ Two schema systems to maintain
- ⚠️ Conversion layer needed at boundary

**Mitigation**: Create explicit `NodeSpec.to_node_data()` converters.

---

### 2.2 Decision: Index Map Bridge Pattern

**Problem**: rustworkx uses integer indices, our domain uses UUID strings.

**Analysis**:
- UUIDs are essential for distributed systems, provenance, merge
- Integer indices are essential for Rust performance
- Can't have both natively

**Decision**: The "Bridge" pattern with bidirectional maps:
```python
class ParagonDB:
    _graph: rx.PyDiGraph
    _node_map: Dict[str, int]   # UUID → index
    _inv_map: Dict[int, str]    # index → UUID
```

**Tradeoff**:
- ✅ Clean API with string IDs
- ✅ Rust-speed graph operations
- ⚠️ Memory overhead (~100 bytes per node for maps)
- ⚠️ Map synchronization complexity

**Mitigation**:
- Use `__slots__` or frozen dataclasses for map entries
- Consider `bidict` library for guaranteed consistency
- For 100K nodes: ~10MB overhead (acceptable)

---

### 2.3 Decision: Lazy Evaluation (Polars)

**Problem**: Loading large CSVs blocks the event loop.

**Analysis**:
- GAADP iterates rows: `for row in csv: add_node(row)`
- Each iteration crosses Python/Rust boundary
- 100K rows = 100K boundary crossings

**Decision**: Polars lazy evaluation:
```python
# OLD: N boundary crossings
for row in csv_reader:
    graph.add_node(NodeData(**row))

# NEW: 1 boundary crossing
lf = pl.scan_csv(path)
lf = lf.filter(pl.col("status") != "deleted")  # Pushed to Rust
nodes = [NodeData(**row) for row in lf.collect().to_dicts()]
graph.add_nodes_from(nodes)
```

**Tradeoff**:
- ✅ 10-100x faster for large datasets
- ✅ Filter pushdown reduces memory
- ⚠️ Delayed error detection (fails at `.collect()`)
- ⚠️ Schema validation timing changes

**Mitigation**: Validate schema on LazyFrame before collect:
```python
def validate_before_collect(lf: pl.LazyFrame, schema: pa.DataFrameSchema):
    # Collect small sample to validate
    sample = lf.head(100).collect()
    schema.validate(sample)
    return lf  # Return original lazy frame
```

---

### 2.4 Decision: tree-sitter over ast

**Problem**: Python's `ast` module is limited to Python and loses source locations.

**Analysis**:
- `ast` works only for syntactically valid Python
- `ast` doesn't preserve byte offsets for incremental parsing
- tree-sitter supports 100+ languages
- tree-sitter provides incremental parsing (edit detection)

**Decision**: Use tree-sitter with S-expression queries:
```scheme
; Find all function definitions
(function_definition
  name: (identifier) @func.name
  parameters: (parameters) @func.params
) @func.def

; Find all class definitions
(class_definition
  name: (identifier) @class.name
  body: (block) @class.body
) @class.def
```

**Tradeoff**:
- ✅ Multi-language support (Python, JS, Rust, Go...)
- ✅ Incremental parsing for real-time analysis
- ✅ Concrete syntax tree (preserves comments, whitespace)
- ⚠️ Steeper learning curve (S-expressions)
- ⚠️ Grammar files are language-specific

**Mitigation**: Create query library for common patterns:
```python
QUERIES = {
    "python_functions": "(function_definition name: (identifier) @name)",
    "python_classes": "(class_definition name: (identifier) @name)",
    "python_imports": "(import_statement) @import",
}
```

---

### 2.5 Decision: rx.layers() for Wavefront

**Problem**: Python recursion for DAG layering is slow and stack-limited.

**Analysis**:
- GAADP's `identify_waves()` uses Python loops over NetworkX
- For 10K nodes with avg degree 10, this takes ~2 seconds
- rustworkx's `layers()` is Rust-native topological generation

**Decision**: Direct replacement:
```python
# OLD: Python recursion (wavefront.py:261-295)
def identify_waves(self, graph: nx.DiGraph) -> List[Set[str]]:
    waves = []
    pending_nodes = {...}
    while pending_nodes:
        current_wave = set()
        for node in pending_nodes:
            dependencies = [...]  # O(E) per node
            if set(dependencies).issubset(completed):
                current_wave.add(node)
        ...

# NEW: Rust native
def get_waves(self) -> List[List[NodeData]]:
    layers = rx.layers(self._graph, first_layer=self._get_roots())
    return [[self._graph[idx] for idx in layer] for layer in layers]
```

**Tradeoff**:
- ✅ 50-100x faster (Rust vs Python loops)
- ✅ No stack overflow risk
- ✅ Handles cycles gracefully (raises `DAGHasCycle`)
- ⚠️ Different API (returns generators, not sets)
- ⚠️ `first_layer` parameter required (must identify roots)

**Mitigation**: Helper to find DAG roots:
```python
def _get_roots(self) -> List[int]:
    return [idx for idx in self._graph.node_indices()
            if self._graph.in_degree(idx) == 0]
```

---

### 2.6 Decision: Keep Transition Matrix as Data

**Problem**: Should we compile the transition matrix into code?

**Analysis**:
- Current: `TRANSITION_MATRIX: Dict[Tuple[str, str], List[TransitionRule]]`
- Could compile to: `match (status, type): case (...): ...`
- Compilation would be faster but lose introspection

**Decision**: Keep as data structure (no compilation).

**Rationale**:
1. **Debuggability**: Can inspect valid transitions at runtime
2. **Validation**: `validate_transition_matrix()` catches errors at load
3. **Hot reload**: Could update rules without restart (future)
4. **Performance**: Dict lookup is O(1), already fast enough

**The lookup cost is negligible**:
- One dict lookup per state transition
- Transitions happen ~10-100x per task execution
- LLM calls dominate by 1000x

---

## 3. RISK ANALYSIS

### 3.1 High Risk: Index Map Synchronization

**Scenario**: Map gets out of sync with graph during concurrent operations.

**Mitigation**:
```python
class ParagonDB:
    _lock: asyncio.Lock

    async def add_node(self, data: NodeData) -> int:
        async with self._lock:
            if data.id in self._node_map:
                return self._node_map[data.id]
            idx = self._graph.add_node(data)
            self._node_map[data.id] = idx
            self._inv_map[idx] = data.id
            return idx
```

### 3.2 Medium Risk: msgspec/Pydantic Boundary

**Scenario**: Data loss or type coercion at conversion boundary.

**Mitigation**: Explicit, tested converters:
```python
def nodespec_to_nodedata(spec: NodeSpec) -> NodeData:
    """Convert Pydantic NodeSpec to msgspec NodeData."""
    return NodeData(
        id=spec.id or uuid4().hex,
        type=spec.type,
        content=spec.content,
        metadata=spec.metadata or {},
        status=NodeStatus.PENDING.value,
    )

# Test coverage required for all conversion paths
```

### 3.3 Low Risk: tree-sitter Query Complexity

**Scenario**: Complex queries become unmaintainable.

**Mitigation**: Query library with tests and documentation.

---

## 4. PERFORMANCE TARGETS

| Operation | GAADP (measured) | Paragon (target) | Speedup |
|-----------|------------------|------------------|---------|
| 10K node wave computation | ~2000ms | <50ms | 40x |
| 100K node batch insert | ~5000ms | <100ms | 50x |
| Graph serialization (10K) | ~500ms | <20ms | 25x |
| Code parse (1000 LOC) | ~100ms | <10ms | 10x |

---

## 5. VERIFICATION PROTOCOLS

### Protocol Alpha: Speed
```python
# Generate random DAG with 10K nodes
g = rx.directed_gnp_random_graph(10000, 0.01, seed=42)

# Measure wave computation
start = time.perf_counter()
layers = rx.layers(g, first_layer=[...])
waves = list(layers)  # Force evaluation
elapsed = time.perf_counter() - start

assert elapsed < 0.050, f"Wave computation took {elapsed:.3f}s, target <50ms"
```

### Protocol Beta: Integrity
```python
# Ingest Paragon codebase into itself
parser = CodeParser()
db = ParagonDB()

for py_file in glob("paragon/**/*.py"):
    nodes = parser.parse_file(py_file)
    db.add_nodes_batch(nodes)

# Verify dependency tracking
schemas_id = db.find_node_by_path("core/schemas.py")
descendants = db.get_descendants(schemas_id)
descendant_paths = [d.metadata.get("path") for d in descendants]

assert "core/graph_db.py" in descendant_paths, \
    "graph_db.py should depend on schemas.py"
```

---

## 6. MIGRATION PATH

### Phase 1: Foundation (Current)
- [x] Project structure created
- [ ] `core/schemas.py` - NodeData, EdgeData (msgspec)
- [ ] `core/graph_db.py` - ParagonDB with index maps

### Phase 2: Ingestion
- [ ] `infrastructure/data_loader.py` - Polars lazy loader
- [ ] `domain/code_parser.py` - tree-sitter integration

### Phase 3: Intelligence
- [ ] `core/alignment.py` - pygmtools integration
- [ ] `core/ontology.py` - Port transition matrix

### Phase 4: Orchestration
- [ ] `agents/orchestrator.py` - Port LangGraph adapter
- [ ] Port agent manifest and protocols

### Phase 5: Serving
- [ ] `api/routes.py` - Cosmograph endpoints
- [ ] `main.py` - Granian entry point

---

## 7. APPENDIX: GAADP NOVEL IP TO PRESERVE

These intellectual innovations must be preserved exactly:

1. **Transition Matrix** (`ontology.py:293-602`)
   - State machine as declarative data
   - Priority-ordered rules with named conditions

2. **Agent Dispatch** (`ontology.py:612-674`)
   - Condition-based routing table
   - Gen-2 TDD loop encoding

3. **Research Standard v1.0** (`protocols.py:296-510`)
   - Sufficient statistic for autonomous generation
   - 10-criterion verification checklist

4. **Dialector Pipeline** (`agent_manifest.yaml:425-653`)
   - Pre-research ambiguity detection
   - Blocking vs clarifying classification

5. **GraphMetrics** (`wavefront.py:39-168`)
   - Articulation points, bridge edges
   - Priority scoring formula

6. **Protocol Schema Inlining** (`protocols.py:705-740`)
   - `_inline_refs()` for LLM compatibility
   - Forced `tool_choice` pattern
