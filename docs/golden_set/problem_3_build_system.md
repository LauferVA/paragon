# GOLDEN SET PROBLEM 3: Incremental Build System

**Problem ID:** `GOLDEN-003`
**Category:** Complex Orchestration - Incremental Computation
**Difficulty:** High
**Estimated Implementation Time:** 3-5 days
**Date Created:** 2025-12-07

---

## EXECUTIVE SUMMARY

Build an incremental build system that tracks file dependencies, computes minimal rebuild sets, and parallelizes compilation tasks. This problem tests the orchestrator's ability to:
- Manage complex dependency graphs with cycles detection
- Implement change detection and cache invalidation strategies
- Parallelize independent tasks while respecting dependencies
- Handle both incremental and full build scenarios
- Maintain build artifact consistency across rebuilds

This is a canonical example of a system requiring both graph-based dependency tracking and intelligent work scheduling.

---

## 1. PROBLEM STATEMENT

### 1.1 User Requirement (What a User Would Submit)

```
I need an incremental build system for my project that:
1. Tracks dependencies between source files and build outputs
2. Only rebuilds files that have changed or whose dependencies changed
3. Parallelizes independent compilation tasks to maximize throughput
4. Supports both incremental rebuilds and full clean builds
5. Handles circular dependency detection
6. Caches build artifacts and invalidates them intelligently
7. Provides clear build reports showing what was rebuilt and why

The system should work for a multi-language project (Python, Rust, TypeScript)
and support different build tools (cargo, tsc, pytest). It should be smart enough
to know that if A depends on B and B changes, then A must be rebuilt.
```

### 1.2 Success Criteria

**Functional Requirements:**
- Dependency graph correctly models file relationships
- Change detection identifies modified files accurately
- Minimal rebuild set computation is optimal (rebuilds only necessary files)
- Parallel execution respects dependency constraints
- Cache invalidation is correct (no stale artifacts)
- Supports multiple build tool backends

**Quality Requirements:**
- All graph invariants maintained (DAG, no orphans)
- Test coverage ≥ 90%
- Build correctness: 100% (no missed rebuilds, no unnecessary rebuilds)
- Performance: Incremental build ≥ 10x faster than full rebuild for 1% file changes
- Handles ≥ 1000 file projects efficiently

---

## 2. CORE COMPONENTS TO IMPLEMENT

### 2.1 Component Breakdown

#### Component 1: Dependency Graph Manager
**Type:** Core data structure
**File Path:** `build_system/dependency_graph.py`
**Description:** Manages the directed acyclic graph of file dependencies

**Key Responsibilities:**
- Add/remove file nodes with metadata (path, hash, timestamp)
- Add/remove dependency edges (A depends on B)
- Detect circular dependencies
- Compute transitive closure (all dependencies of a file)
- Export graph for visualization

**Key Methods:**
```python
def add_file(path: Path, content_hash: str, metadata: Dict) -> str  # Returns node_id
def add_dependency(source: str, target: str) -> bool
def detect_cycles() -> List[List[str]]  # Returns list of cycle paths
def get_transitive_dependencies(file_id: str) -> Set[str]
def get_dependents(file_id: str) -> Set[str]  # Files that depend on this one
```

**Schema (msgspec.Struct):**
```python
class FileNode(msgspec.Struct, kw_only=True, frozen=True):
    file_id: str  # UUID
    path: str  # Absolute file path
    content_hash: str  # SHA256 of file content
    last_modified: float  # Timestamp
    file_type: str  # "source", "generated", "artifact"
    language: Optional[str]  # "python", "rust", "typescript"

class DependencyEdge(msgspec.Struct, kw_only=True, frozen=True):
    source_id: str
    target_id: str
    dependency_type: str  # "import", "include", "transitive"
```

#### Component 2: Change Detection Algorithm
**Type:** Core algorithm
**File Path:** `build_system/change_detector.py`
**Description:** Identifies which files have changed since last build

**Key Responsibilities:**
- Compute content hashes (SHA256) for files
- Compare current hashes with previous build snapshot
- Detect new files, deleted files, modified files
- Handle filesystem timestamp caching
- Support .gitignore-style ignore patterns

**Key Methods:**
```python
def scan_directory(root: Path, ignore_patterns: List[str]) -> Dict[str, str]  # path -> hash
def detect_changes(previous_snapshot: Dict, current_snapshot: Dict) -> ChangeSet
def compute_file_hash(path: Path) -> str
def is_ignored(path: Path, patterns: List[str]) -> bool
```

**Schema:**
```python
class ChangeSet(msgspec.Struct, kw_only=True):
    added: List[str]  # New files
    modified: List[str]  # Changed files
    deleted: List[str]  # Removed files
    unchanged: List[str]  # No changes

class BuildSnapshot(msgspec.Struct, kw_only=True):
    timestamp: float
    file_hashes: Dict[str, str]  # path -> hash
    build_artifacts: Dict[str, str]  # artifact_path -> source_path
```

#### Component 3: Rebuild Set Computation
**Type:** Core algorithm
**File Path:** `build_system/rebuild_planner.py`
**Description:** Computes minimal set of files to rebuild based on changes

**Key Responsibilities:**
- Given a ChangeSet, compute all files that must be rebuilt
- Include direct changes and transitive dependents
- Respect dependency graph topology
- Optimize for minimal rebuild set
- Generate rebuild order (topological sort)

**Key Methods:**
```python
def compute_rebuild_set(changes: ChangeSet, dep_graph: DependencyGraph) -> RebuildPlan
def compute_rebuild_order(rebuild_set: Set[str], dep_graph: DependencyGraph) -> List[str]
def estimate_rebuild_time(plan: RebuildPlan) -> float
```

**Schema:**
```python
class RebuildPlan(msgspec.Struct, kw_only=True):
    rebuild_targets: List[str]  # Ordered list of files to rebuild
    rebuild_count: int
    total_file_count: int
    affected_percentage: float
    estimated_time_seconds: float
    rebuild_reasons: Dict[str, str]  # file_id -> reason ("modified", "dependency_changed")
```

#### Component 4: Parallel Task Scheduler
**Type:** Core execution engine
**File Path:** `build_system/task_scheduler.py`
**Description:** Executes build tasks in parallel while respecting dependencies

**Key Responsibilities:**
- Execute build tasks in topological order
- Parallelize independent tasks (no shared dependencies)
- Handle task failures and retries
- Collect build outputs and errors
- Report progress in real-time

**Key Methods:**
```python
async def execute_build_plan(plan: RebuildPlan, build_config: BuildConfig) -> BuildResult
async def execute_task(task: BuildTask, semaphore: asyncio.Semaphore) -> TaskResult
def compute_task_waves(tasks: List[BuildTask], dep_graph: DependencyGraph) -> List[List[BuildTask]]
```

**Schema:**
```python
class BuildTask(msgspec.Struct, kw_only=True):
    task_id: str
    file_id: str
    command: str  # Shell command to execute
    dependencies: List[str]  # Task IDs that must complete first
    timeout_seconds: int

class TaskResult(msgspec.Struct, kw_only=True):
    task_id: str
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float

class BuildResult(msgspec.Struct, kw_only=True):
    success: bool
    total_tasks: int
    succeeded_tasks: int
    failed_tasks: int
    skipped_tasks: int
    total_duration_seconds: float
    task_results: List[TaskResult]
```

#### Component 5: Build Cache Manager
**Type:** Infrastructure
**File Path:** `build_system/cache_manager.py`
**Description:** Manages build artifact caching and invalidation

**Key Responsibilities:**
- Store build artifacts keyed by content hash
- Retrieve cached artifacts when inputs haven't changed
- Invalidate cache entries when dependencies change
- Implement cache eviction policies (LRU, size-based)
- Support persistent cache across builds

**Key Methods:**
```python
def get_cached_artifact(file_id: str, content_hash: str) -> Optional[bytes]
def store_artifact(file_id: str, content_hash: str, artifact: bytes) -> None
def invalidate_artifact(file_id: str) -> None
def prune_cache(max_size_bytes: int, strategy: str) -> None
```

**Schema:**
```python
class CacheEntry(msgspec.Struct, kw_only=True):
    file_id: str
    content_hash: str
    artifact_path: str
    created_at: float
    last_accessed: float
    size_bytes: int

class CacheStats(msgspec.Struct, kw_only=True):
    total_entries: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    hit_rate: float
```

#### Component 6: Build Configuration
**Type:** Configuration schema
**File Path:** `build_system/config.py`
**Description:** Configuration for build system behavior

**Schema:**
```python
class BuildToolConfig(msgspec.Struct, kw_only=True):
    name: str  # "cargo", "tsc", "pytest"
    command_template: str  # "cargo build {file}"
    file_pattern: str  # "*.rs", "*.ts"
    output_pattern: str  # "target/debug/{name}", "dist/{name}.js"

class BuildConfig(msgspec.Struct, kw_only=True):
    project_root: str
    build_tools: List[BuildToolConfig]
    max_parallel_tasks: int
    cache_enabled: bool
    cache_max_size_mb: int
    ignore_patterns: List[str]
```

### 2.2 Component Dependency Graph

```
BuildConfiguration
         ↓
DependencyGraphManager ← ChangeDetector
         ↓                      ↓
   RebuildPlanner ←────────────┘
         ↓
   TaskScheduler → CacheManager
         ↓
   BuildResult
```

**Wave 0 (No Dependencies):**
- BuildConfiguration
- FileNode/Edge schemas

**Wave 1 (Depends on Wave 0):**
- DependencyGraphManager
- ChangeDetector
- CacheManager

**Wave 2 (Depends on Wave 1):**
- RebuildPlanner

**Wave 3 (Depends on Wave 2):**
- TaskScheduler

---

## 3. DEPENDENCY GRAPH MANAGEMENT

### 3.1 Graph Structure

**Node Types:**
- `SOURCE_FILE`: Original source code files
- `GENERATED_FILE`: Files produced by build tools
- `BUILD_ARTIFACT`: Final build outputs (binaries, libraries)

**Edge Types:**
- `IMPORTS`: Direct code imports (Python: import, Rust: use, TS: import)
- `INCLUDES`: File includes (C/C++: #include)
- `GENERATES`: Build tool produces artifact from source
- `TRANSITIVE`: Computed transitive dependency

**Graph Invariants:**
- Must be a DAG (no circular dependencies in build graph)
- Every BUILD_ARTIFACT must have at least one GENERATES edge
- Transitive edges must be consistent with path compression

### 3.2 Cycle Detection Algorithm

**Strategy:** Depth-first search with recursion stack tracking

**Pseudocode:**
```python
def detect_cycles(graph: DependencyGraph) -> List[List[str]]:
    visited = set()
    rec_stack = set()
    cycles = []

    def dfs(node_id: str, path: List[str]):
        visited.add(node_id)
        rec_stack.add(node_id)
        path.append(node_id)

        for neighbor in graph.get_successors(node_id):
            if neighbor not in visited:
                dfs(neighbor, path.copy())
            elif neighbor in rec_stack:
                cycle_start = path.index(neighbor)
                cycles.append(path[cycle_start:] + [neighbor])

        rec_stack.remove(node_id)

    for node in graph.nodes():
        if node not in visited:
            dfs(node, [])

    return cycles
```

**Test Cases:**
- Single file self-dependency (A → A)
- Two-file cycle (A → B → A)
- Three-file cycle (A → B → C → A)
- Multiple independent cycles
- No cycles (valid DAG)

---

## 4. CHANGE DETECTION ALGORITHMS

### 4.1 Content-Based Detection

**Strategy:** SHA256 hashing of file contents

**Advantages:**
- Detects actual content changes (not just timestamp updates)
- Handles file moves/renames correctly
- Cross-platform consistency

**Implementation:**
```python
def compute_content_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()
```

### 4.2 Timestamp Optimization

**Strategy:** Use filesystem mtime as fast pre-check

**Algorithm:**
1. Compare file mtime with last build snapshot
2. If mtime unchanged, assume content unchanged (skip hash)
3. If mtime changed, compute hash to confirm actual change
4. Handle clock skew and filesystem precision issues

**Edge Cases:**
- File touched but not modified (mtime changes, hash same)
- File restored from backup (mtime newer, hash unchanged)
- Network filesystem timestamp inconsistencies

### 4.3 Ignore Patterns

**Strategy:** .gitignore-style glob patterns

**Supported Patterns:**
```
*.pyc              # All .pyc files
__pycache__/       # Directory and contents
target/            # Rust build directory
node_modules/      # Node.js dependencies
!important.log     # Negation (don't ignore)
```

**Implementation:** Use `pathspec` library for efficient matching

---

## 5. PARALLEL TASK SCHEDULING

### 5.1 Wave-Based Parallelization

**Strategy:** Group tasks into waves based on dependency depth

**Algorithm:**
```python
def compute_task_waves(tasks: List[BuildTask], dep_graph: DependencyGraph) -> List[List[BuildTask]]:
    # Wave 0: Tasks with no dependencies
    # Wave 1: Tasks depending only on Wave 0
    # Wave N: Tasks depending on Wave 0..N-1

    waves = []
    completed = set()
    remaining = set(task.task_id for task in tasks)

    while remaining:
        wave = []
        for task_id in remaining:
            task = get_task(task_id)
            if all(dep in completed for dep in task.dependencies):
                wave.append(task)

        if not wave:
            raise CircularDependencyError("Cannot schedule tasks")

        waves.append(wave)
        completed.update(t.task_id for t in wave)
        remaining -= completed

    return waves
```

**This is IDENTICAL to ParagonDB.get_waves()** - excellent testing opportunity for graph layer!

### 5.2 Concurrency Control

**Strategy:** asyncio with semaphore-based rate limiting

**Implementation:**
```python
async def execute_wave(wave: List[BuildTask], max_parallel: int) -> List[TaskResult]:
    semaphore = asyncio.Semaphore(max_parallel)

    async def execute_with_limit(task: BuildTask):
        async with semaphore:
            return await execute_task(task)

    results = await asyncio.gather(
        *[execute_with_limit(task) for task in wave],
        return_exceptions=True
    )
    return results
```

**Tuning Parameters:**
- `max_parallel`: CPU count or user-configured limit
- Task timeout: Per-task timeout for hangs
- Retry policy: Retry transient failures (network, resource)

---

## 6. CACHE INVALIDATION STRATEGIES

### 6.1 Content-Addressed Caching

**Strategy:** Key artifacts by (file_id, content_hash) tuple

**Cache Key:** `f"{file_id}:{content_hash}"`

**Invalidation Rules:**
1. Content hash changes → invalidate cached artifact
2. Any dependency's hash changes → invalidate dependent artifacts
3. Build tool version changes → invalidate all artifacts for that tool

### 6.2 Transitive Invalidation

**Strategy:** Invalidate all downstream artifacts when upstream changes

**Algorithm:**
```python
def invalidate_recursive(file_id: str, cache: CacheManager, dep_graph: DependencyGraph):
    cache.invalidate_artifact(file_id)

    dependents = dep_graph.get_dependents(file_id)
    for dependent in dependents:
        invalidate_recursive(dependent, cache, dep_graph)
```

**Optimization:** Batch invalidation to avoid redundant work

### 6.3 Cache Eviction Policies

**LRU (Least Recently Used):**
- Track last access timestamp for each cache entry
- Evict oldest entries when cache exceeds size limit

**Size-Based:**
- Track total cache size in bytes
- Evict largest entries first when space needed

**Hybrid:**
- Combine LRU + size (score = age * size)
- Configurable weights

---

## 7. BUILD ARTIFACT MANAGEMENT

### 7.1 Artifact Storage

**Structure:**
```
.build_cache/
├── artifacts/
│   ├── abc123_def456.o         # Object file (file_id_hash.ext)
│   ├── abc123_def456.wasm      # WebAssembly module
│   └── ...
├── metadata/
│   ├── abc123.json             # Artifact metadata
│   └── ...
└── build_snapshot.json         # Last successful build state
```

### 7.2 Artifact Metadata

**Schema:**
```python
class ArtifactMetadata(msgspec.Struct, kw_only=True):
    artifact_id: str
    source_file_id: str
    content_hash: str
    build_tool: str
    created_at: float
    size_bytes: int
    dependencies: List[str]  # Input file IDs
```

### 7.3 Snapshot Persistence

**build_snapshot.json:**
```json
{
  "timestamp": 1733567890.123,
  "file_hashes": {
    "src/main.rs": "abc123...",
    "src/lib.rs": "def456..."
  },
  "artifacts": {
    "target/debug/myapp": "abc123"
  }
}
```

**Recovery:** Load snapshot on startup, compare with current filesystem state

---

## 8. TEST SCENARIOS

### 8.1 Incremental Build Scenarios

#### Scenario 1: Single File Change
**Setup:**
- Project with 10 files (A-J)
- Dependency chain: A→B→C, D→E→F, G→H→I→J
- Modify file B

**Expected Behavior:**
- Rebuild set: {B, C} (A unchanged, D-J independent)
- Parallel execution: B first, then C
- Cache hits: {A, D, E, F, G, H, I, J}

**Assertions:**
```python
assert rebuild_set == {"B", "C"}
assert rebuild_order[0] == "B"
assert rebuild_order[1] == "C"
assert cache_hits == 8
assert build_time < full_build_time / 5
```

#### Scenario 2: Transitive Dependency Change
**Setup:**
- Linear chain: A→B→C→D→E
- Modify file A

**Expected Behavior:**
- Rebuild set: {A, B, C, D, E} (all downstream)
- Sequential execution (no parallelism due to linear chain)
- Full rebuild time (no cache benefits)

**Assertions:**
```python
assert rebuild_set == {"A", "B", "C", "D", "E"}
assert len(waves) == 5  # Sequential waves
```

#### Scenario 3: Independent Subtrees
**Setup:**
- Two independent trees: A→B→C and X→Y→Z
- Modify A and X

**Expected Behavior:**
- Rebuild set: {A, B, C, X, Y, Z}
- Parallel execution: Two waves of 2 tasks each
  - Wave 0: {A, X}
  - Wave 1: {B, Y}
  - Wave 2: {C, Z}

**Assertions:**
```python
assert len(waves) == 3
assert set(waves[0]) == {"A", "X"}
assert total_time < 2 * sequential_time
```

### 8.2 Full Build Scenarios

#### Scenario 4: Clean Build (No Cache)
**Setup:**
- Empty cache directory
- Project with 100 files

**Expected Behavior:**
- Rebuild all files
- Populate cache with all artifacts
- Establish baseline timing

**Assertions:**
```python
assert rebuild_set == set(all_files)
assert cache.stats().total_entries == 100
assert all artifact exists in .build_cache
```

#### Scenario 5: Full Rebuild (Cache Present)
**Setup:**
- Full cache from previous build
- No file changes
- User requests full rebuild (e.g., `make clean && make`)

**Expected Behavior:**
- All files rebuilt despite cache
- Cache updated with fresh artifacts
- Verify idempotency (same output as before)

**Assertions:**
```python
assert rebuild_count == 100
assert new_hashes == old_hashes  # Deterministic builds
```

### 8.3 Cache Invalidation Scenarios

#### Scenario 6: Dependency Hash Change
**Setup:**
- File A depends on B
- B's content changes (new hash)
- A's content unchanged

**Expected Behavior:**
- B rebuild (content changed)
- A rebuild (dependency hash changed)
- Cache invalidation for both A and B

**Assertions:**
```python
assert "B" in rebuild_set  # Direct change
assert "A" in rebuild_set  # Transitive invalidation
assert cache.get_cached_artifact("A", old_hash_A) is None
```

#### Scenario 7: Cache Eviction
**Setup:**
- Cache size limit: 100MB
- Build generates 120MB of artifacts

**Expected Behavior:**
- Oldest/largest 20MB evicted
- Remaining 100MB in cache
- Next build cache misses for evicted artifacts

**Assertions:**
```python
assert cache.stats().total_size_bytes <= 100 * 1024 * 1024
assert cache.stats().total_entries < initial_entries
```

### 8.4 Error Handling Scenarios

#### Scenario 8: Build Task Failure
**Setup:**
- File C has syntax error
- Dependency chain: A→B→C→D

**Expected Behavior:**
- A, B build successfully
- C build fails
- D skipped (dependency failed)
- Overall build fails with clear error

**Assertions:**
```python
assert build_result.success == False
assert task_result["C"].exit_code != 0
assert "D" in build_result.skipped_tasks
```

#### Scenario 9: Circular Dependency
**Setup:**
- User creates cycle: A→B→C→A

**Expected Behavior:**
- Cycle detection before build starts
- Clear error message with cycle path
- No build attempted

**Assertions:**
```python
with pytest.raises(CircularDependencyError) as exc:
    compute_rebuild_plan(...)
assert "A -> B -> C -> A" in str(exc.value)
```

### 8.5 Performance Benchmarks

#### Scenario 10: Large Project (1000 Files)
**Setup:**
- 1000 source files
- Average 5 dependencies per file
- Change 1% of files (10 files)

**Expected Behavior:**
- Incremental build 10-50x faster than full build
- Memory usage < 500MB
- Wave computation < 100ms

**Performance Targets:**
```python
assert incremental_time < full_build_time / 10
assert memory_usage_mb < 500
assert wave_computation_time < 0.1  # 100ms
```

---

## 9. TESTING STRATEGY

### 9.1 Unit Tests

**DependencyGraphManager (15 tests):**
- `test_add_file_creates_node`
- `test_add_duplicate_file_raises_error`
- `test_add_dependency_creates_edge`
- `test_add_dependency_invalid_nodes_raises_error`
- `test_detect_cycle_single_node`
- `test_detect_cycle_two_nodes`
- `test_detect_cycle_complex_graph`
- `test_get_transitive_dependencies_empty`
- `test_get_transitive_dependencies_single_level`
- `test_get_transitive_dependencies_multi_level`
- `test_get_dependents_returns_reverse_deps`
- `test_remove_file_cascades_edges`
- `test_export_graph_to_paragon_db`
- `test_graph_stats_accuracy`
- `test_concurrent_access_thread_safety`

**ChangeDetector (10 tests):**
- `test_scan_directory_finds_all_files`
- `test_scan_directory_respects_ignore_patterns`
- `test_detect_changes_identifies_added_files`
- `test_detect_changes_identifies_modified_files`
- `test_detect_changes_identifies_deleted_files`
- `test_compute_file_hash_consistency`
- `test_is_ignored_matches_patterns`
- `test_timestamp_optimization_skips_hash`
- `test_timestamp_handles_clock_skew`
- `test_large_file_hashing_performance`

**RebuildPlanner (12 tests):**
- `test_compute_rebuild_set_single_change`
- `test_compute_rebuild_set_transitive_deps`
- `test_compute_rebuild_set_multiple_changes`
- `test_compute_rebuild_set_independent_subtrees`
- `test_compute_rebuild_order_topological`
- `test_compute_rebuild_order_respects_deps`
- `test_estimate_rebuild_time_accuracy`
- `test_rebuild_reasons_correct`
- `test_empty_changeset_no_rebuild`
- `test_full_rebuild_all_files`
- `test_rebuild_plan_serialization`
- `test_rebuild_plan_caching`

**TaskScheduler (10 tests):**
- `test_execute_task_success`
- `test_execute_task_failure_exit_code`
- `test_execute_task_timeout`
- `test_execute_wave_parallel_execution`
- `test_compute_task_waves_correct_grouping`
- `test_semaphore_limits_concurrency`
- `test_task_retry_on_transient_failure`
- `test_collect_stdout_stderr`
- `test_build_result_aggregation`
- `test_progress_reporting`

**CacheManager (10 tests):**
- `test_store_artifact_creates_entry`
- `test_get_cached_artifact_hit`
- `test_get_cached_artifact_miss`
- `test_invalidate_artifact_removes_entry`
- `test_prune_cache_lru_policy`
- `test_prune_cache_size_policy`
- `test_cache_stats_accuracy`
- `test_cache_persistence_across_runs`
- `test_concurrent_cache_access`
- `test_cache_corruption_recovery`

### 9.2 Integration Tests

**Build System Integration (8 tests):**
- `test_full_workflow_single_file_change`
- `test_full_workflow_transitive_rebuild`
- `test_cache_invalidation_on_dependency_change`
- `test_parallel_build_correctness`
- `test_build_failure_propagation`
- `test_circular_dependency_detection_integration`
- `test_multi_language_project_build`
- `test_incremental_vs_full_build_equivalence`

### 9.3 Property-Based Tests (Hypothesis)

**Graph Properties:**
```python
@given(st.lists(st.tuples(st.text(), st.text()), min_size=10, max_size=100))
def test_rebuild_set_is_minimal(dependencies):
    """Rebuild set should be minimal (no unnecessary rebuilds)"""
    graph = build_graph_from_edges(dependencies)
    assume(is_dag(graph))  # Only test valid DAGs

    change = random.choice(list(graph.nodes()))
    rebuild_set = compute_rebuild_set({change}, graph)

    # Property: Rebuild set contains only changed file + transitive dependents
    assert change in rebuild_set
    for node in rebuild_set:
        assert node == change or has_path(change, node, graph)
```

**Cache Correctness:**
```python
@given(st.lists(st.text(), min_size=5, max_size=50))
def test_cache_invalidation_is_sound(file_changes):
    """Cache must be invalidated for all affected files"""
    cache = CacheManager()
    graph = build_random_dag()

    for changed_file in file_changes:
        invalidated = cache.invalidate_recursive(changed_file, graph)

        # Property: All dependents must be invalidated
        dependents = graph.get_transitive_dependents(changed_file)
        assert dependents.issubset(invalidated)
```

---

## 10. PARAGON-SPECIFIC INTEGRATION

### 10.1 Graph Database Integration

**Map Build System to ParagonDB:**
- File nodes → `NodeType.CODE` or `NodeType.ARTIFACT`
- Dependency edges → `EdgeType.DEPENDS_ON`
- Build metadata → `NodePayload.metadata`

**Benefits:**
- Reuse `get_waves()` for wave computation
- Leverage teleology for build traceability
- Use Merkle hashing for content addressing
- Export graph for visualization in Rerun

### 10.2 Orchestrator Integration

**TDD Cycle Mapping:**
1. **DIALECTIC:** Clarify build configuration (which tools, which files)
2. **RESEARCH:** Discover file dependencies (static analysis of imports)
3. **PLAN:** Architect creates dependency graph structure
4. **BUILD:** Builder implements build tool wrappers
5. **TEST:** Tester verifies incremental correctness

**Checkpoint Integration:**
- Save dependency graph to graph_db
- Checkpoint build snapshots via SqliteSaver
- Resume interrupted builds from last successful wave

### 10.3 Rerun Visualization

**Visualizations:**
- 3D dependency graph (files as nodes, dependencies as edges)
- Build wave animation (highlight active tasks in each wave)
- Cache hit/miss heatmap
- Build time waterfall chart
- Incremental vs full build comparison

---

## 11. EXTENSION POINTS

### 11.1 Advanced Features (Out of Scope for MVP)

**Distributed Builds:**
- Distribute tasks across multiple machines
- Shared artifact cache (Redis, S3)
- Remote execution API (gRPC)

**Predictive Rebuilds:**
- Machine learning model predicts likely changes
- Pre-warm cache for predicted files
- Speculative parallel builds

**Build Optimization:**
- Profile build times per file
- Identify slowest tasks for optimization
- Suggest dependency refactoring to improve parallelism

### 11.2 Language-Specific Integrations

**Python:**
- Parse imports from AST
- Detect dynamic imports (importlib)
- Handle conditional imports

**Rust:**
- Parse `use` statements and `mod` declarations
- Cargo.toml dependency tracking
- Proc macro dependencies

**TypeScript:**
- Parse import/export statements
- Handle type-only imports
- tsconfig.json path resolution

---

## 12. EVALUATION CRITERIA

### 12.1 Orchestrator Evaluation

**How well does the orchestrator handle this problem?**
- Does it correctly decompose into independent components?
- Does it identify the graph-based nature early?
- Does it recognize the need for parallelism?
- Does it generate correct schemas?
- Does it handle cycle detection edge case?

### 12.2 Code Quality Metrics

**Generated Code Quality:**
- Test coverage ≥ 90%
- No circular imports
- Correct use of async/await
- Proper error handling
- Schema validation with msgspec

### 12.3 Performance Validation

**Benchmark Targets:**
```python
# Protocol Alpha extension
def test_build_system_performance():
    """Test incremental build system performance"""
    # 1000 file project, 10 file changes
    assert incremental_build_time < 5.0  # seconds
    assert wave_computation_time < 0.1  # 100ms
    assert cache_hit_rate > 0.9  # 90%
```

---

## 13. IMPLEMENTATION CHECKLIST

### Phase 1: Foundation (Day 1)
- [ ] Define all msgspec schemas (FileNode, Edge, ChangeSet, etc.)
- [ ] Implement DependencyGraphManager with cycle detection
- [ ] Implement ChangeDetector with content hashing
- [ ] Unit tests for both (20 tests)

### Phase 2: Core Logic (Day 2)
- [ ] Implement RebuildPlanner with minimal set computation
- [ ] Implement wave-based topological sort
- [ ] Unit tests (12 tests)
- [ ] Integration test: single file change scenario

### Phase 3: Execution (Day 3)
- [ ] Implement TaskScheduler with async execution
- [ ] Implement semaphore-based rate limiting
- [ ] Implement CacheManager with LRU eviction
- [ ] Unit tests (20 tests)
- [ ] Integration test: parallel build scenario

### Phase 4: Integration (Day 4)
- [ ] Integrate with ParagonDB
- [ ] Implement build snapshot persistence
- [ ] Add Rerun visualization hooks
- [ ] End-to-end test: full TDD cycle

### Phase 5: Validation (Day 5)
- [ ] Property-based tests (Hypothesis)
- [ ] Performance benchmarks (1000 file project)
- [ ] Security review (command injection, path traversal)
- [ ] Documentation and examples

---

## 14. EXPECTED LEARNING OUTCOMES

### For the Orchestrator:
1. **Graph algorithm recognition:** Identifying when a problem requires DAG operations
2. **Parallelism planning:** Understanding dependency-based concurrency constraints
3. **Cache invalidation:** Reasoning about transitive effects of changes
4. **Schema design:** Creating efficient data structures for complex state

### For the Paragon System:
1. **Reusability:** Demonstrating how `get_waves()` solves real problems
2. **Schema discipline:** Enforcing msgspec for all data structures
3. **Testing rigor:** Comprehensive test coverage including property-based tests
4. **Performance awareness:** Optimizing for O(V+E) graph operations

---

## 15. DELIVERABLES

### Code Artifacts:
- `build_system/` module with 6 Python files
- `tests/build_system/` with 60+ unit tests
- `tests/integration/test_build_system.py` with 8 integration tests
- Example project in `examples/build_system_demo/`

### Documentation:
- API documentation (docstrings)
- User guide (how to configure build system)
- Architecture diagram (component interactions)

### Metrics:
- Test coverage report (≥90% target)
- Performance benchmark results
- Build time comparison (incremental vs full)

---

## APPENDIX A: Reference Implementations

### Real-World Build Systems to Study:
1. **Bazel** (Google): Content-addressed caching, distributed builds
2. **Buck2** (Facebook): Graph-based incremental builds
3. **Ninja** (Google): Low-level build file with explicit dependencies
4. **Make** (GNU): Classic dependency-based build system
5. **Turborepo** (Vercel): Monorepo build orchestration with caching

### Key Learnings:
- Bazel's content-addressing is very similar to Merkle hashing
- Buck2's graph model maps directly to ParagonDB
- Ninja's wave-based execution is identical to `get_waves()`

---

## APPENDIX B: Prompt Template for Orchestrator

**When submitting this problem to Paragon:**

```markdown
I need an incremental build system for my multi-language project.

REQUIREMENTS:
1. Track file dependencies (imports, includes, etc.)
2. Only rebuild files that changed or whose dependencies changed
3. Run independent builds in parallel
4. Cache build artifacts and invalidate smartly
5. Support Python, Rust, and TypeScript projects
6. Detect circular dependencies and fail fast

EXAMPLE WORKFLOW:
- I modify src/utils.py
- System detects that main.py imports utils.py
- System rebuilds: utils.py → main.py
- System skips: tests.py, config.py (no dependency)
- Build completes 10x faster than full rebuild

CONSTRAINTS:
- Must handle 1000+ file projects
- Must be correct (no stale artifacts)
- Must use graph-based dependency tracking
- Must use content hashing for change detection

Please implement this system using the Paragon TDD workflow.
```

**Expected Orchestrator Behavior:**
1. DIALECTIC identifies ambiguities (which build tools? which languages?)
2. RESEARCH investigates build system patterns (Bazel, Make)
3. ARCHITECT creates component breakdown (matches Section 2.1)
4. BUILDER generates code with msgspec schemas
5. TESTER verifies all scenarios from Section 8

---

**END OF RESEARCH DOCUMENT**
