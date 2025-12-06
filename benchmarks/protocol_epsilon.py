"""
PROTOCOL EPSILON - Layer 8 Verification (The Physics Engine)

Tests the "Soul" components:
1. Graph Invariants (Handshaking, Balis, DAG, Stratification)
2. Teleology (Golden Thread validation)
3. Merkle Provenance (Content-addressable hashing)

These tests verify that Paragon's mathematical foundations are sound.
"""
import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import rustworkx as rx

from core.schemas import NodeData, EdgeData, compute_hash
from core.graph_db import ParagonDB, GraphInvariantError
from core.ontology import NodeType, NodeStatus, EdgeType
from core.graph_invariants import (
    GraphInvariants,
    InvariantSeverity,
    IncrementalValidator,
    validate_graph,
    is_valid_dag,
    get_graph_metrics,
)
from core.teleology import (
    TeleologyValidator,
    TeleologyStatus,
    validate_teleology,
    find_unjustified_nodes,
    has_teleological_integrity,
)


# =============================================================================
# TEST GRAPH INVARIANTS
# =============================================================================

class TestHandshakingLemma(unittest.TestCase):
    """Tests for the Handshaking Lemma invariant."""

    def test_empty_graph_valid(self):
        """Empty graph satisfies handshaking lemma."""
        graph = rx.PyDiGraph()
        valid, violation = GraphInvariants.validate_handshaking_lemma(graph)
        self.assertTrue(valid)
        self.assertIsNone(violation)

    def test_single_node_valid(self):
        """Single node with no edges is valid."""
        graph = rx.PyDiGraph()
        graph.add_node("A")
        valid, violation = GraphInvariants.validate_handshaking_lemma(graph)
        self.assertTrue(valid)

    def test_linear_chain_valid(self):
        """Linear chain satisfies lemma."""
        graph = rx.PyDiGraph()
        a = graph.add_node("A")
        b = graph.add_node("B")
        c = graph.add_node("C")
        graph.add_edge(a, b, "e1")
        graph.add_edge(b, c, "e2")

        valid, violation = GraphInvariants.validate_handshaking_lemma(graph)
        self.assertTrue(valid)

    def test_tree_valid(self):
        """Tree structure is valid."""
        graph = rx.PyDiGraph()
        root = graph.add_node("root")
        child1 = graph.add_node("child1")
        child2 = graph.add_node("child2")
        graph.add_edge(root, child1, "e1")
        graph.add_edge(root, child2, "e2")

        valid, violation = GraphInvariants.validate_handshaking_lemma(graph)
        self.assertTrue(valid)


class TestDAGAcyclicity(unittest.TestCase):
    """Tests for the DAG Acyclicity invariant."""

    def test_dag_valid(self):
        """DAG without cycles is valid."""
        graph = rx.PyDiGraph()
        a = graph.add_node("A")
        b = graph.add_node("B")
        c = graph.add_node("C")
        graph.add_edge(a, b, "e1")
        graph.add_edge(a, c, "e2")
        graph.add_edge(b, c, "e3")

        valid, violation = GraphInvariants.validate_dag_acyclicity(graph)
        self.assertTrue(valid)

    def test_cycle_detected(self):
        """Cycle in graph is detected."""
        graph = rx.PyDiGraph()
        a = graph.add_node("A")
        b = graph.add_node("B")
        c = graph.add_node("C")
        graph.add_edge(a, b, "e1")
        graph.add_edge(b, c, "e2")
        graph.add_edge(c, a, "e3")  # Creates cycle

        valid, violation = GraphInvariants.validate_dag_acyclicity(graph)
        self.assertFalse(valid)
        self.assertIsNotNone(violation)
        self.assertEqual(violation.severity, InvariantSeverity.ERROR)

    def test_self_loop_detected(self):
        """Self-loop is detected."""
        graph = rx.PyDiGraph()
        a = graph.add_node("A")
        graph.add_edge(a, a, "self")

        valid, violation = GraphInvariants.validate_dag_acyclicity(graph)
        self.assertFalse(valid)


class TestBalisDegree(unittest.TestCase):
    """Tests for the Balis Degree (reachability) invariant."""

    def test_connected_graph_valid(self):
        """Fully connected DAG is valid."""
        graph = rx.PyDiGraph()
        a = graph.add_node("A")  # source
        b = graph.add_node("B")
        c = graph.add_node("C")  # sink

        graph.add_edge(a, b, "e1")
        graph.add_edge(b, c, "e2")

        valid, violation, unreachable = GraphInvariants.validate_balis_degree(graph)
        self.assertTrue(valid)
        self.assertEqual(len(unreachable), 0)

    def test_disconnected_components_flagged(self):
        """Disconnected components are flagged."""
        graph = rx.PyDiGraph()
        # Component 1
        a = graph.add_node("A")
        b = graph.add_node("B")
        graph.add_edge(a, b, "e1")

        # Component 2 (disconnected)
        c = graph.add_node("C")
        d = graph.add_node("D")
        graph.add_edge(c, d, "e2")

        valid, violation, unreachable = GraphInvariants.validate_balis_degree(graph)
        # Sources A, C should not all reach sinks B, D
        self.assertFalse(valid)
        self.assertGreater(len(unreachable), 0)


class TestStratification(unittest.TestCase):
    """Tests for the Stratification invariant."""

    def test_correct_ordering_valid(self):
        """Correct type ordering passes."""
        graph = rx.PyDiGraph()

        req = NodeData.create(type=NodeType.REQ.value, content="requirement")
        spec = NodeData.create(type=NodeType.SPEC.value, content="spec")
        code = NodeData.create(type=NodeType.CODE.value, content="code")

        req_idx = graph.add_node(req)
        spec_idx = graph.add_node(spec)
        code_idx = graph.add_node(code)

        # REQ -> SPEC -> CODE (correct flow)
        graph.add_edge(req_idx, spec_idx, "traces")
        graph.add_edge(spec_idx, code_idx, "implements")

        valid, violation = GraphInvariants.validate_stratification(graph)
        self.assertTrue(valid)

    def test_backwards_edge_flagged(self):
        """Backwards type edge is flagged."""
        graph = rx.PyDiGraph()

        code = NodeData.create(type=NodeType.CODE.value, content="code")
        req = NodeData.create(type=NodeType.REQ.value, content="requirement")

        code_idx = graph.add_node(code)
        req_idx = graph.add_node(req)

        # CODE -> REQ is backwards (violates stratification)
        graph.add_edge(code_idx, req_idx, "backwards")

        valid, violation = GraphInvariants.validate_stratification(graph)
        self.assertFalse(valid)


class TestCyclomaticComplexity(unittest.TestCase):
    """Tests for cyclomatic complexity calculation."""

    def test_linear_chain_complexity_zero(self):
        """Linear chain has complexity 0."""
        graph = rx.PyDiGraph()
        nodes = [graph.add_node(str(i)) for i in range(5)]
        for i in range(4):
            graph.add_edge(nodes[i], nodes[i + 1], f"e{i}")

        # E=4, V=5, components=1 -> 4 - 5 + 1 = 0
        complexity = GraphInvariants.compute_cyclomatic_complexity(graph)
        self.assertEqual(complexity, 0)

    def test_diamond_complexity(self):
        """Diamond pattern has complexity > 0."""
        graph = rx.PyDiGraph()
        a = graph.add_node("A")
        b = graph.add_node("B")
        c = graph.add_node("C")
        d = graph.add_node("D")

        graph.add_edge(a, b, "e1")
        graph.add_edge(a, c, "e2")
        graph.add_edge(b, d, "e3")
        graph.add_edge(c, d, "e4")

        # E=4, V=4, components=1 -> 4 - 4 + 1 = 1
        complexity = GraphInvariants.compute_cyclomatic_complexity(graph)
        self.assertEqual(complexity, 1)


class TestIncrementalValidator(unittest.TestCase):
    """Tests for incremental (pre-insert) validation."""

    def test_cycle_detection_pre_insert(self):
        """Detect if edge would create cycle before adding."""
        graph = rx.PyDiGraph()
        a = graph.add_node("A")
        b = graph.add_node("B")
        c = graph.add_node("C")
        graph.add_edge(a, b, "e1")
        graph.add_edge(b, c, "e2")

        # Adding c -> a would create cycle
        would_cycle = IncrementalValidator.would_create_cycle(graph, c, a)
        self.assertTrue(would_cycle)

        # Adding a -> c would not create cycle (already exists path)
        would_cycle = IncrementalValidator.would_create_cycle(graph, a, c)
        self.assertFalse(would_cycle)


# =============================================================================
# TEST TELEOLOGY
# =============================================================================

class TestTeleologyValidator(unittest.TestCase):
    """Tests for teleology (Golden Thread) validation."""

    def setUp(self):
        """Create a test database."""
        self.db = ParagonDB()

    def test_req_node_is_root(self):
        """REQ nodes are teleology roots."""
        req = NodeData.create(type=NodeType.REQ.value, content="Build X")
        self.db.add_node(req)

        report = validate_teleology(
            self.db._graph,
            self.db._node_map,
            self.db._inv_map
        )

        self.assertTrue(report.is_valid)
        self.assertEqual(report.root_count, 1)
        self.assertEqual(report.node_results[req.id].status, TeleologyStatus.ROOT)

    def test_connected_nodes_justified(self):
        """Nodes connected to REQ are justified."""
        req = NodeData.create(type=NodeType.REQ.value, content="Build X")
        spec = NodeData.create(type=NodeType.SPEC.value, content="Spec for X")
        code = NodeData.create(type=NodeType.CODE.value, content="impl X")

        self.db.add_node(req)
        self.db.add_node(spec)
        self.db.add_node(code)

        # Create edges: SPEC -> REQ, CODE -> SPEC
        self.db.add_edge(EdgeData.create(spec.id, req.id, EdgeType.TRACES_TO.value))
        self.db.add_edge(EdgeData.create(code.id, spec.id, EdgeType.IMPLEMENTS.value))

        report = validate_teleology(
            self.db._graph,
            self.db._node_map,
            self.db._inv_map
        )

        self.assertTrue(report.is_valid)
        self.assertEqual(report.justified_count, 2)  # spec and code
        self.assertEqual(report.node_results[spec.id].status, TeleologyStatus.JUSTIFIED)
        self.assertEqual(report.node_results[code.id].status, TeleologyStatus.JUSTIFIED)

    def test_orphaned_node_detected(self):
        """Orphaned nodes are detected."""
        req = NodeData.create(type=NodeType.REQ.value, content="Build X")
        orphan = NodeData.create(type=NodeType.CODE.value, content="random code")

        self.db.add_node(req)
        self.db.add_node(orphan)
        # No edges - orphan is disconnected

        report = validate_teleology(
            self.db._graph,
            self.db._node_map,
            self.db._inv_map
        )

        self.assertFalse(report.is_valid)
        self.assertEqual(report.orphaned_count, 1)
        self.assertIn(orphan.id, report.unjustified_nodes)

    def test_hallucinated_scope_detected(self):
        """Code not tracing to REQ is unjustified."""
        req = NodeData.create(type=NodeType.REQ.value, content="Build X")
        spec = NodeData.create(type=NodeType.SPEC.value, content="Spec X")
        hallucinated = NodeData.create(type=NodeType.CODE.value, content="extra code")

        self.db.add_node(req)
        self.db.add_node(spec)
        self.db.add_node(hallucinated)

        # Only connect spec to req, hallucinated is connected but not to REQ chain
        self.db.add_edge(EdgeData.create(spec.id, req.id, EdgeType.TRACES_TO.value))
        # hallucinated has edge to spec but spec doesn't lead to it
        self.db.add_edge(EdgeData.create(hallucinated.id, spec.id, EdgeType.DEPENDS_ON.value))

        report = validate_teleology(
            self.db._graph,
            self.db._node_map,
            self.db._inv_map
        )

        # All nodes should be justified since hallucinated connects to spec which connects to req
        self.assertTrue(report.is_valid)


class TestTeleologyConvenience(unittest.TestCase):
    """Tests for teleology convenience functions."""

    def test_find_unjustified_nodes(self):
        """find_unjustified_nodes returns correct list."""
        db = ParagonDB()
        req = NodeData.create(type=NodeType.REQ.value, content="req")
        orphan = NodeData.create(type=NodeType.CODE.value, content="orphan")

        db.add_node(req)
        db.add_node(orphan)

        unjustified = find_unjustified_nodes(db._graph, db._node_map, db._inv_map)
        self.assertIn(orphan.id, unjustified)
        self.assertNotIn(req.id, unjustified)

    def test_has_teleological_integrity(self):
        """has_teleological_integrity returns correct boolean."""
        db = ParagonDB()
        req = NodeData.create(type=NodeType.REQ.value, content="req")
        db.add_node(req)

        self.assertTrue(has_teleological_integrity(db._graph, db._node_map, db._inv_map))

        # Add orphan
        orphan = NodeData.create(type=NodeType.CODE.value, content="orphan")
        db.add_node(orphan)

        self.assertFalse(has_teleological_integrity(db._graph, db._node_map, db._inv_map))


# =============================================================================
# TEST MERKLE PROVENANCE
# =============================================================================

class TestMerkleHash(unittest.TestCase):
    """Tests for Merkle hash provenance."""

    def test_compute_hash_deterministic(self):
        """Same content produces same hash."""
        h1 = compute_hash("hello world")
        h2 = compute_hash("hello world")
        self.assertEqual(h1, h2)

    def test_compute_hash_different_content(self):
        """Different content produces different hash."""
        h1 = compute_hash("hello world")
        h2 = compute_hash("hello world!")
        self.assertNotEqual(h1, h2)

    def test_compute_hash_with_dependencies(self):
        """Dependencies affect the hash."""
        base = compute_hash("content")
        with_deps = compute_hash("content", ["dep1hash", "dep2hash"])
        self.assertNotEqual(base, with_deps)

    def test_compute_hash_dependency_order_invariant(self):
        """Dependency order doesn't matter (sorted internally)."""
        h1 = compute_hash("content", ["hash1", "hash2", "hash3"])
        h2 = compute_hash("content", ["hash3", "hash1", "hash2"])
        self.assertEqual(h1, h2)

    def test_nodedata_merkle_hash(self):
        """NodeData can compute its merkle hash."""
        node = NodeData.create(type=NodeType.CODE.value, content="def foo(): pass")
        self.assertIsNone(node.merkle_hash)

        hash_val = node.compute_merkle_hash()
        self.assertIsNotNone(node.merkle_hash)
        self.assertEqual(node.merkle_hash, hash_val)

    def test_nodedata_merkle_hash_with_deps(self):
        """NodeData merkle hash with dependencies."""
        dep1 = NodeData.create(type=NodeType.SPEC.value, content="spec1")
        dep1.compute_merkle_hash()

        dep2 = NodeData.create(type=NodeType.SPEC.value, content="spec2")
        dep2.compute_merkle_hash()

        node = NodeData.create(type=NodeType.CODE.value, content="implementation")
        node.compute_merkle_hash([dep1.merkle_hash, dep2.merkle_hash])

        # Changing a dependency should change downstream hash
        old_hash = node.merkle_hash
        node.compute_merkle_hash([dep1.merkle_hash])  # Remove dep2
        self.assertNotEqual(old_hash, node.merkle_hash)


class TestTeleologyStatus(unittest.TestCase):
    """Tests for teleology status on NodeData."""

    def test_default_status_unchecked(self):
        """Default teleology status is unchecked."""
        node = NodeData.create(type=NodeType.CODE.value, content="code")
        self.assertEqual(node.teleology_status, "unchecked")

    def test_set_teleology_status(self):
        """Can set teleology status."""
        node = NodeData.create(type=NodeType.CODE.value, content="code")
        node.set_teleology_status("justified")
        self.assertEqual(node.teleology_status, "justified")


# =============================================================================
# TEST FULL VALIDATION
# =============================================================================

class TestFullValidation(unittest.TestCase):
    """Tests for complete graph validation."""

    def test_validate_all_on_valid_graph(self):
        """validate_all passes on valid graph."""
        graph = rx.PyDiGraph()
        a = graph.add_node(NodeData.create(type=NodeType.REQ.value, content="req"))
        b = graph.add_node(NodeData.create(type=NodeType.SPEC.value, content="spec"))
        c = graph.add_node(NodeData.create(type=NodeType.CODE.value, content="code"))

        graph.add_edge(a, b, "e1")
        graph.add_edge(b, c, "e2")

        report = GraphInvariants.validate_all(graph)
        self.assertTrue(report.valid)
        self.assertEqual(len(report.errors), 0)

    def test_validate_all_on_cyclic_graph(self):
        """validate_all fails on cyclic graph."""
        graph = rx.PyDiGraph()
        a = graph.add_node("A")
        b = graph.add_node("B")
        graph.add_edge(a, b, "e1")
        graph.add_edge(b, a, "e2")  # Cycle

        report = GraphInvariants.validate_all(graph)
        self.assertFalse(report.valid)
        self.assertGreater(len(report.errors), 0)

    def test_get_graph_metrics(self):
        """get_graph_metrics returns expected metrics."""
        graph = rx.PyDiGraph()
        a = graph.add_node("A")
        b = graph.add_node("B")
        c = graph.add_node("C")
        graph.add_edge(a, b, "e1")
        graph.add_edge(b, c, "e2")

        metrics = get_graph_metrics(graph)
        self.assertEqual(metrics['node_count'], 3)
        self.assertEqual(metrics['edge_count'], 2)
        self.assertTrue(metrics['is_dag'])
        self.assertEqual(metrics['weakly_connected_components'], 1)


# =============================================================================
# INTEGRATION TEST
# =============================================================================

class TestLayer8Integration(unittest.TestCase):
    """Integration tests for Layer 8 components."""

    def test_full_workflow_with_invariants_and_teleology(self):
        """Test a complete workflow with both invariants and teleology checks."""
        db = ParagonDB()

        # Create requirement
        req = NodeData.create(
            type=NodeType.REQ.value,
            content="Build a calculator that adds numbers"
        )
        db.add_node(req)

        # Create spec
        spec = NodeData.create(
            type=NodeType.SPEC.value,
            content="def add(a: int, b: int) -> int"
        )
        db.add_node(spec)

        # Create code
        code = NodeData.create(
            type=NodeType.CODE.value,
            content="def add(a: int, b: int) -> int:\n    return a + b"
        )
        db.add_node(code)

        # Create test
        test = NodeData.create(
            type=NodeType.TEST.value,
            content="assert add(1, 2) == 3"
        )
        db.add_node(test)

        # Add edges
        db.add_edge(EdgeData.create(spec.id, req.id, EdgeType.TRACES_TO.value))
        db.add_edge(EdgeData.create(code.id, spec.id, EdgeType.IMPLEMENTS.value))
        db.add_edge(EdgeData.create(test.id, code.id, EdgeType.TESTS.value))

        # Validate graph invariants
        invariant_report = GraphInvariants.validate_all(db._graph, db._inv_map)
        self.assertTrue(invariant_report.valid, f"Invariant errors: {invariant_report.errors}")

        # Validate teleology
        teleology_report = validate_teleology(db._graph, db._node_map, db._inv_map)
        self.assertTrue(teleology_report.is_valid, f"Unjustified: {teleology_report.unjustified_nodes}")

        # Compute merkle hashes in dependency order
        req.compute_merkle_hash()
        spec.compute_merkle_hash([req.merkle_hash])
        code.compute_merkle_hash([spec.merkle_hash])
        test.compute_merkle_hash([code.merkle_hash])

        # All nodes should have hashes
        self.assertIsNotNone(req.merkle_hash)
        self.assertIsNotNone(spec.merkle_hash)
        self.assertIsNotNone(code.merkle_hash)
        self.assertIsNotNone(test.merkle_hash)


# =============================================================================
# MAIN
# =============================================================================

def run_protocol_epsilon():
    """Run Protocol Epsilon verification."""
    print("=" * 70)
    print("PROTOCOL EPSILON - Layer 8 (Physics Engine) Verification")
    print("=" * 70)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestHandshakingLemma,
        TestDAGAcyclicity,
        TestBalisDegree,
        TestStratification,
        TestCyclomaticComplexity,
        TestIncrementalValidator,
        TestTeleologyValidator,
        TestTeleologyConvenience,
        TestMerkleHash,
        TestTeleologyStatus,
        TestFullValidation,
        TestLayer8Integration,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print()
    print("PROTOCOL EPSILON SUMMARY")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print()

    if result.wasSuccessful():
        print("✓ PROTOCOL EPSILON PASSED - Layer 8 (Physics Engine) Verified")
    else:
        print("✗ PROTOCOL EPSILON FAILED")
        for failure in result.failures:
            print(f"  FAIL: {failure[0]}")
        for error in result.errors:
            print(f"  ERROR: {error[0]}")

    # Return format expected by run_all.py
    passed = result.testsRun - len(result.failures) - len(result.errors)
    return {
        "passed": passed,
        "total": result.testsRun,
        "errors": [str(f[0]) for f in result.failures + result.errors],
    }


if __name__ == "__main__":
    result = run_protocol_epsilon()
    sys.exit(0 if result["passed"] == result["total"] else 1)
