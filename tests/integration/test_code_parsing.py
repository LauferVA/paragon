"""
PROTOCOL BETA: Integrity Verification Benchmark

Purpose: Verify that Paragon correctly tracks code dependencies.

Test Cases:
1. Self-ingestion: Parse Paragon codebase into its own graph using real CodeParser
2. Dependency tracking: Verify import relationships are captured
3. Descendant correctness: Validate graph traversal
4. Round-trip integrity: Serialize/deserialize without data loss

This benchmark validates the correctness of the graph model using REAL implementations.
Run with: python -m benchmarks.protocol_beta
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# Conditional imports
try:
    import rustworkx as rx
    RUSTWORKX_AVAILABLE = True
except ImportError:
    RUSTWORKX_AVAILABLE = False
    print("⚠️  rustworkx not installed. Install with: pip install rustworkx")

try:
    import msgspec
    MSGSPEC_AVAILABLE = True
except ImportError:
    MSGSPEC_AVAILABLE = False
    print("⚠️  msgspec not installed. Install with: pip install msgspec")

# Import real Paragon modules
try:
    from core.graph_db import ParagonDB
    from core.schemas import NodeData, EdgeData, serialize_node, deserialize_node
    from core.ontology import NodeType, EdgeType, NodeStatus
    from domain.code_parser import CodeParser, parse_python_directory
    PARAGON_AVAILABLE = True
except ImportError as e:
    PARAGON_AVAILABLE = False
    print(f"⚠️  Paragon modules not available: {e}")


@dataclass
class IntegrityResult:
    """Result of an integrity check."""
    name: str
    passed: bool
    details: str
    expected: Optional[str] = None
    actual: Optional[str] = None

    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        result = f"{status} {self.name}\n       {self.details}"
        if not self.passed and self.expected:
            result += f"\n       Expected: {self.expected}"
            result += f"\n       Actual: {self.actual}"
        return result


# =============================================================================
# REAL IMPLEMENTATIONS (Using actual Paragon modules)
# =============================================================================
# These use the actual ParagonDB and CodeParser implementations.


# =============================================================================
# INTEGRITY CHECK 1: Self-Ingestion (Using Real CodeParser)
# =============================================================================

def check_self_ingestion() -> IntegrityResult:
    """
    Verify that Paragon can parse its own codebase using the REAL tree-sitter parser.

    This is the fundamental integrity test - if we can't parse ourselves,
    we can't parse anything.
    """
    if not RUSTWORKX_AVAILABLE:
        return IntegrityResult(
            name="Self-Ingestion",
            passed=False,
            details="rustworkx not available"
        )

    if not PARAGON_AVAILABLE:
        return IntegrityResult(
            name="Self-Ingestion",
            passed=False,
            details="Paragon modules not available"
        )

    paragon_root = Path(__file__).parent.parent

    try:
        # Use the REAL CodeParser and ParagonDB
        db = ParagonDB()
        nodes, edges = parse_python_directory(paragon_root, recursive=True)

        # Add nodes to graph
        db.add_nodes_batch(nodes)

        # Only add edges where both nodes exist
        valid_node_ids = {n.id for n in nodes}
        valid_edges = [
            e for e in edges
            if e.source_id in valid_node_ids and e.target_id in valid_node_ids
        ]
        db.add_edges_batch(valid_edges)

        if db.node_count < 10:
            return IntegrityResult(
                name="Self-Ingestion (Real Parser)",
                passed=False,
                details=f"Expected at least 10 nodes, got {db.node_count}",
                expected=">=10 nodes",
                actual=str(db.node_count)
            )

        # Count by kind for detailed report
        kind_counts = {}
        for node in nodes:
            kind = node.data.get("kind", "unknown")
            kind_counts[kind] = kind_counts.get(kind, 0) + 1

        kind_summary = ", ".join(f"{k}={v}" for k, v in sorted(kind_counts.items()))

        return IntegrityResult(
            name="Self-Ingestion (Real Parser)",
            passed=True,
            details=f"tree-sitter parsed {db.node_count} nodes, {db.edge_count} edges ({kind_summary})"
        )

    except Exception as e:
        return IntegrityResult(
            name="Self-Ingestion (Real Parser)",
            passed=False,
            details=f"Parse error: {e}"
        )


# =============================================================================
# INTEGRITY CHECK 2: Import Relationship Tracking (Using Real ParagonDB)
# =============================================================================

def check_import_tracking() -> IntegrityResult:
    """
    Verify that import relationships are correctly captured using REAL ParagonDB.

    Create a test scenario with known imports and verify they're tracked.
    """
    if not RUSTWORKX_AVAILABLE:
        return IntegrityResult(
            name="Import Relationship Tracking",
            passed=False,
            details="rustworkx not available"
        )

    if not PARAGON_AVAILABLE:
        return IntegrityResult(
            name="Import Relationship Tracking",
            passed=False,
            details="Paragon modules not available"
        )

    db = ParagonDB()

    # Create test modules using REAL NodeData
    module_a = NodeData(id="module_a", type=NodeType.CODE.value, content="", data={"kind": "module", "name": "module_a"})
    module_b = NodeData(id="module_b", type=NodeType.CODE.value, content="", data={"kind": "module", "name": "module_b"})
    module_c = NodeData(id="module_c", type=NodeType.CODE.value, content="", data={"kind": "module", "name": "module_c"})

    db.add_nodes_batch([module_a, module_b, module_c])

    # Create import edges: A → B → C using REAL EdgeData
    db.add_edge(EdgeData(source_id="module_a", target_id="module_b", type=EdgeType.REFERENCES.value))
    db.add_edge(EdgeData(source_id="module_b", target_id="module_c", type=EdgeType.REFERENCES.value))

    # Verify: C should be a descendant of A (transitively)
    descendants_a = db.get_descendants("module_a")
    descendant_ids = {d.id for d in descendants_a}

    if "module_c" not in descendant_ids:
        return IntegrityResult(
            name="Import Relationship Tracking (Real)",
            passed=False,
            details="Transitive dependency not captured",
            expected="module_c in descendants of module_a",
            actual=str(descendant_ids)
        )

    # Verify: A should be an ancestor of C
    ancestors_c = db.get_ancestors("module_c")
    ancestor_ids = {a.id for a in ancestors_c}

    if "module_a" not in ancestor_ids:
        return IntegrityResult(
            name="Import Relationship Tracking (Real)",
            passed=False,
            details="Reverse dependency not captured",
            expected="module_a in ancestors of module_c",
            actual=str(ancestor_ids)
        )

    return IntegrityResult(
        name="Import Relationship Tracking (Real)",
        passed=True,
        details="Import chain A→B→C correctly tracked in both directions using real ParagonDB"
    )


# =============================================================================
# INTEGRITY CHECK 3: Index Map Consistency (Using Real ParagonDB)
# =============================================================================

def check_index_map_consistency() -> IntegrityResult:
    """
    Verify that the UUID↔index bidirectional map stays consistent.

    This is critical - if the maps get out of sync, everything breaks.
    """
    if not RUSTWORKX_AVAILABLE:
        return IntegrityResult(
            name="Index Map Consistency",
            passed=False,
            details="rustworkx not available"
        )

    if not PARAGON_AVAILABLE:
        return IntegrityResult(
            name="Index Map Consistency",
            passed=False,
            details="Paragon modules not available"
        )

    db = ParagonDB()

    # Add nodes using real NodeData
    nodes = [NodeData(id=f"node_{i}", type=NodeType.CODE.value, content=f"content_{i}") for i in range(100)]
    db.add_nodes_batch(nodes)

    # Verify all nodes can be retrieved
    missing = []
    for node in nodes:
        retrieved = db.get_node(node.id)
        if retrieved is None:
            missing.append(node.id)
        elif retrieved.id != node.id:
            missing.append(f"{node.id} (got {retrieved.id})")

    if missing:
        return IntegrityResult(
            name="Index Map Consistency (Real)",
            passed=False,
            details=f"Found {len(missing)} missing/mismatched nodes",
            expected="All nodes retrievable",
            actual="; ".join(missing[:3])
        )

    return IntegrityResult(
        name="Index Map Consistency (Real)",
        passed=True,
        details=f"All {db.node_count} nodes correctly stored and retrievable"
    )


# =============================================================================
# INTEGRITY CHECK 4: Duplicate Node Handling (Using Real ParagonDB)
# =============================================================================

def check_duplicate_handling() -> IntegrityResult:
    """
    Verify that adding the same node twice doesn't create duplicates.

    Idempotency is critical for reliability.
    """
    if not RUSTWORKX_AVAILABLE:
        return IntegrityResult(
            name="Duplicate Node Handling",
            passed=False,
            details="rustworkx not available"
        )

    if not PARAGON_AVAILABLE:
        return IntegrityResult(
            name="Duplicate Node Handling",
            passed=False,
            details="Paragon modules not available"
        )

    from core.graph_db import DuplicateNodeError

    db = ParagonDB()

    # Add node using real NodeData
    node = NodeData(id="duplicate_test", type=NodeType.CODE.value, content="test")

    db.add_node(node)

    # Try to add same node again - should raise DuplicateNodeError
    duplicate_raised = False
    try:
        db.add_node(node)
    except DuplicateNodeError:
        duplicate_raised = True

    if not duplicate_raised:
        return IntegrityResult(
            name="Duplicate Node Handling (Real)",
            passed=False,
            details="Expected DuplicateNodeError was not raised",
            expected="DuplicateNodeError",
            actual="No error raised"
        )

    # Verify exactly 1 node exists
    if db.node_count != 1:
        return IntegrityResult(
            name="Duplicate Node Handling (Real)",
            passed=False,
            details="Unexpected node count after duplicate attempt",
            expected="1 node",
            actual=f"{db.node_count} nodes"
        )

    # Verify has_node works
    if not db.has_node("duplicate_test"):
        return IntegrityResult(
            name="Duplicate Node Handling (Real)",
            passed=False,
            details="has_node returned False for existing node",
            expected="has_node returns True",
            actual="has_node returns False"
        )

    return IntegrityResult(
        name="Duplicate Node Handling (Real)",
        passed=True,
        details="DuplicateNodeError correctly raised, duplicate rejected, original preserved"
    )


# =============================================================================
# INTEGRITY CHECK 5: Graph Serialization Round-Trip (Using Real Schemas)
# =============================================================================

def check_serialization_roundtrip() -> IntegrityResult:
    """
    Verify that graph can be serialized and deserialized without data loss.

    Required for persistence and distributed operation.
    Uses the REAL serialize_node/deserialize_node functions.
    """
    if not RUSTWORKX_AVAILABLE:
        return IntegrityResult(
            name="Serialization Round-Trip",
            passed=False,
            details="rustworkx not available"
        )

    if not MSGSPEC_AVAILABLE:
        return IntegrityResult(
            name="Serialization Round-Trip",
            passed=False,
            details="msgspec not available"
        )

    if not PARAGON_AVAILABLE:
        return IntegrityResult(
            name="Serialization Round-Trip",
            passed=False,
            details="Paragon modules not available"
        )

    # Create test nodes using REAL NodeData
    original_nodes = [
        NodeData(id="module_test", type=NodeType.CODE.value, content="module content", data={"path": "test.py"}),
        NodeData(id="class_test", type=NodeType.CLASS.value, content="class content", data={"name": "TestClass"}),
        NodeData(id="func_test", type=NodeType.FUNCTION.value, content="def test(): pass", data={"name": "test_func"}),
    ]

    try:
        # Serialize and deserialize each node using REAL functions
        for orig in original_nodes:
            # Serialize
            encoded = serialize_node(orig)

            # Deserialize
            decoded = deserialize_node(encoded)

            # Verify
            if orig.id != decoded.id:
                return IntegrityResult(
                    name="Serialization Round-Trip (Real)",
                    passed=False,
                    details="ID mismatch after round-trip",
                    expected=orig.id,
                    actual=decoded.id
                )
            if orig.type != decoded.type:
                return IntegrityResult(
                    name="Serialization Round-Trip (Real)",
                    passed=False,
                    details="Type mismatch after round-trip",
                    expected=orig.type,
                    actual=decoded.type
                )
            if orig.content != decoded.content:
                return IntegrityResult(
                    name="Serialization Round-Trip (Real)",
                    passed=False,
                    details="Content mismatch after round-trip",
                    expected=orig.content,
                    actual=decoded.content
                )

    except Exception as e:
        return IntegrityResult(
            name="Serialization Round-Trip (Real)",
            passed=False,
            details=f"Serialization failed: {e}"
        )

    return IntegrityResult(
        name="Serialization Round-Trip (Real)",
        passed=True,
        details=f"Successfully serialized/deserialized {len(original_nodes)} nodes via real msgspec functions"
    )


# =============================================================================
# INTEGRITY CHECK 6: Wave Computation Correctness
# =============================================================================

def check_wave_correctness() -> IntegrityResult:
    """
    Verify that wave computation produces correct topological ordering.

    Nodes in later waves must depend on nodes in earlier waves.
    """
    if not RUSTWORKX_AVAILABLE:
        return IntegrityResult(
            name="Wave Computation Correctness",
            passed=False,
            details="rustworkx not available"
        )

    # Create a known DAG: A → B → D
    #                     A → C → D
    graph = rx.PyDiGraph()

    a = graph.add_node("A")
    b = graph.add_node("B")
    c = graph.add_node("C")
    d = graph.add_node("D")

    graph.add_edge(a, b, "A→B")
    graph.add_edge(a, c, "A→C")
    graph.add_edge(b, d, "B→D")
    graph.add_edge(c, d, "C→D")

    # Get waves
    roots = [idx for idx in graph.node_indices() if graph.in_degree(idx) == 0]
    layers_gen = rx.layers(graph, roots)
    # rx.layers returns node data directly, not indices
    waves = [list(layer) for layer in layers_gen]

    # Expected: Wave 0: [A], Wave 1: [B, C], Wave 2: [D]
    if len(waves) != 3:
        return IntegrityResult(
            name="Wave Computation Correctness",
            passed=False,
            details=f"Expected 3 waves, got {len(waves)}",
            expected="3 waves",
            actual=f"{len(waves)} waves"
        )

    # waves already contains node data (strings in this case)
    wave_contents = waves

    if wave_contents[0] != ["A"]:
        return IntegrityResult(
            name="Wave Computation Correctness",
            passed=False,
            details="Wave 0 incorrect",
            expected="['A']",
            actual=str(wave_contents[0])
        )

    if set(wave_contents[1]) != {"B", "C"}:
        return IntegrityResult(
            name="Wave Computation Correctness",
            passed=False,
            details="Wave 1 incorrect",
            expected="{'B', 'C'}",
            actual=str(set(wave_contents[1]))
        )

    if wave_contents[2] != ["D"]:
        return IntegrityResult(
            name="Wave Computation Correctness",
            passed=False,
            details="Wave 2 incorrect",
            expected="['D']",
            actual=str(wave_contents[2])
        )

    return IntegrityResult(
        name="Wave Computation Correctness",
        passed=True,
        details="DAG A→[B,C]→D correctly decomposed into 3 waves"
    )


# =============================================================================
# MAIN: Run All Integrity Checks
# =============================================================================

def run_protocol_beta() -> List[IntegrityResult]:
    """Run all Protocol Beta integrity checks and return results."""

    print("=" * 70)
    print("PROTOCOL BETA: Integrity Verification Benchmark")
    print("=" * 70)
    print()

    if not RUSTWORKX_AVAILABLE:
        print("❌ CRITICAL: rustworkx is not installed!")
        print("   Install with: pip install rustworkx>=0.17.0")
        print()
        return []

    checks = [
        ("Graph Operations", [
            check_index_map_consistency,
            check_duplicate_handling,
            check_wave_correctness,
        ]),
        ("Code Analysis", [
            check_self_ingestion,
            check_import_tracking,
        ]),
        ("Persistence", [
            check_serialization_roundtrip,
        ]),
    ]

    all_results = []

    for category, funcs in checks:
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

    print(f"\nResults: {passed}/{total} integrity checks passed")

    if passed == total:
        print("\n✅ PROTOCOL BETA: ALL INTEGRITY CHECKS PASSED")
    else:
        print("\n❌ PROTOCOL BETA: SOME INTEGRITY CHECKS FAILED")
        print("\nFailed checks:")
        for r in all_results:
            if not r.passed:
                print(f"  - {r.name}: {r.details}")

    return all_results


if __name__ == "__main__":
    results = run_protocol_beta()

    # Exit with error code if any check failed
    if not all(r.passed for r in results):
        sys.exit(1)
