"""
PARAGON END-TO-END TESTS

Tests complete workflows through the system:
1. TDD Cycle: REQ → SPEC → CODE → TEST → PASSED
2. Teleology: All nodes trace to requirements (no hallucinated scope)
3. Graph Invariants: DAG, handshaking, Balis degree
4. Code Ingestion: Parse codebase into graph
5. Quality Gate: Full validation pipeline

Run: pytest tests/e2e/ -v
"""
import sys
import pytest
from pathlib import Path
from typing import Dict, Any

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.graph_db import ParagonDB
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, EdgeType, NodeStatus
from core.graph_invariants import GraphInvariants, validate_graph, is_valid_dag
from core.teleology import validate_teleology, has_teleological_integrity, find_unjustified_nodes
from agents.tools import set_db, get_db, add_node, add_node_safe, add_edge


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fresh_db():
    """Provide a fresh ParagonDB instance."""
    db = ParagonDB()
    set_db(db)
    yield db
    set_db(None)


@pytest.fixture
def tdd_graph(fresh_db):
    """Create a complete TDD graph: REQ → SPEC → CODE → TEST."""
    db = fresh_db
    
    # Create requirement
    req = NodeData.create(
        type=NodeType.REQ.value,
        content="Build a function that adds two numbers and returns the sum",
        created_by="test",
    )
    db.add_node(req)
    
    # Create specification
    spec = NodeData.create(
        type=NodeType.SPEC.value,
        content="def add(a: int, b: int) -> int: Return sum of a and b",
        created_by="test",
    )
    db.add_node(spec)
    
    # Create code
    code = NodeData.create(
        type=NodeType.CODE.value,
        content="def add(a: int, b: int) -> int:\n    return a + b",
        created_by="test",
    )
    db.add_node(code)
    
    # Create test
    test = NodeData.create(
        type=NodeType.TEST.value,
        content="def test_add():\n    assert add(1, 2) == 3\n    assert add(-1, 1) == 0",
        created_by="test",
    )
    db.add_node(test)
    
    # Create edges (SPEC traces to REQ, CODE implements SPEC, TEST tests CODE)
    db.add_edge(EdgeData.create(spec.id, req.id, EdgeType.TRACES_TO.value))
    db.add_edge(EdgeData.create(code.id, spec.id, EdgeType.IMPLEMENTS.value))
    db.add_edge(EdgeData.create(test.id, code.id, EdgeType.TESTS.value))
    
    return db, {"req": req, "spec": spec, "code": code, "test": test}


# =============================================================================
# E2E TEST 1: COMPLETE TDD WORKFLOW
# =============================================================================

class TestTDDWorkflow:
    """Test the complete TDD cycle from requirement to passing tests."""
    
    def test_full_tdd_graph_structure(self, tdd_graph):
        """Graph has correct structure: 4 nodes, 3 edges."""
        db, nodes = tdd_graph
        
        assert db.node_count == 4
        assert db.edge_count == 3
        
    def test_wave_computation_correct_order(self, tdd_graph):
        """Waves reflect dependency order: TEST first (source), REQ last (sink).

        Edge direction: child → parent (SPEC→REQ, CODE→SPEC, TEST→CODE).
        So TEST has no incoming edges and is in the first wave.
        """
        db, nodes = tdd_graph

        waves = db.get_waves()
        assert len(waves) >= 3  # At least 3 waves

        # First wave should contain TEST (no incoming edges = source)
        first_wave_ids = {n.id for n in waves[0]}
        assert nodes["test"].id in first_wave_ids

    def test_descendant_tracking(self, tdd_graph):
        """TEST has all other nodes as descendants (following edge direction).

        Edge direction: child → parent. TEST → CODE → SPEC → REQ.
        """
        db, nodes = tdd_graph

        descendants = db.get_descendants(nodes["test"].id)
        descendant_ids = {d.id for d in descendants}

        assert nodes["code"].id in descendant_ids
        assert nodes["spec"].id in descendant_ids
        assert nodes["req"].id in descendant_ids

    def test_ancestor_tracking(self, tdd_graph):
        """REQ has all other nodes as ancestors (reverse edge direction).

        Edge direction: child → parent. TEST → CODE → SPEC → REQ.
        So REQ's ancestors are SPEC, CODE, TEST.
        """
        db, nodes = tdd_graph

        ancestors = db.get_ancestors(nodes["req"].id)
        ancestor_ids = {a.id for a in ancestors}

        assert nodes["spec"].id in ancestor_ids
        assert nodes["code"].id in ancestor_ids
        assert nodes["test"].id in ancestor_ids


# =============================================================================
# E2E TEST 2: TELEOLOGY (GOLDEN THREAD)
# =============================================================================

class TestTeleology:
    """Test that all nodes trace back to requirements (no hallucinated scope)."""
    
    def test_all_nodes_justified(self, tdd_graph):
        """All nodes in TDD graph are teleologically justified."""
        db, nodes = tdd_graph
        
        report = validate_teleology(db._graph, db._node_map, db._inv_map)
        
        assert report.is_valid
        assert report.orphaned_count == 0
        assert len(report.unjustified_nodes) == 0
        
    def test_orphan_detected(self, fresh_db):
        """Orphan nodes (not connected to REQ) are detected."""
        db = fresh_db
        
        # Create REQ
        req = NodeData.create(type=NodeType.REQ.value, content="Build X")
        db.add_node(req)
        
        # Create orphan CODE (not connected to anything)
        orphan = NodeData.create(type=NodeType.CODE.value, content="orphan code")
        db.add_node(orphan)
        
        # Validate
        report = validate_teleology(db._graph, db._node_map, db._inv_map)
        
        assert not report.is_valid
        assert orphan.id in report.unjustified_nodes
        
    def test_has_teleological_integrity_helper(self, tdd_graph):
        """has_teleological_integrity returns True for valid graph."""
        db, nodes = tdd_graph
        
        result = has_teleological_integrity(db._graph, db._node_map, db._inv_map)
        assert result is True
        

# =============================================================================
# E2E TEST 3: GRAPH INVARIANTS (PHYSICS ENGINE)
# =============================================================================

class TestGraphInvariants:
    """Test graph mathematical invariants."""
    
    def test_dag_acyclicity(self, tdd_graph):
        """TDD graph is a valid DAG (no cycles)."""
        db, nodes = tdd_graph
        
        valid, violation = GraphInvariants.validate_dag_acyclicity(db._graph)
        assert valid
        assert violation is None
        
    def test_cycle_detection(self, fresh_db):
        """Cycles are prevented at insert time (GraphInvariantError raised)."""
        from core.graph_db import GraphInvariantError

        db = fresh_db

        # Create nodes: A → B → C
        a = NodeData.create(type=NodeType.CODE.value, content="A")
        b = NodeData.create(type=NodeType.CODE.value, content="B")
        c = NodeData.create(type=NodeType.CODE.value, content="C")

        db.add_node(a)
        db.add_node(b)
        db.add_node(c)

        db.add_edge(EdgeData.create(a.id, b.id, EdgeType.DEPENDS_ON.value))
        db.add_edge(EdgeData.create(b.id, c.id, EdgeType.DEPENDS_ON.value))

        # Attempting to create cycle C → A should raise GraphInvariantError
        with pytest.raises(GraphInvariantError) as excinfo:
            db.add_edge(EdgeData.create(c.id, a.id, EdgeType.DEPENDS_ON.value))

        assert "cycle" in str(excinfo.value).lower()
        
    def test_handshaking_lemma(self, tdd_graph):
        """Handshaking lemma holds: sum(in_degrees) == sum(out_degrees) == |E|."""
        db, nodes = tdd_graph
        
        valid, violation = GraphInvariants.validate_handshaking_lemma(db._graph)
        assert valid
        
    def test_full_validation(self, tdd_graph):
        """Full graph validation passes."""
        db, nodes = tdd_graph
        
        report = GraphInvariants.validate_all(db._graph)
        assert report.valid
        assert len(report.errors) == 0


# =============================================================================
# E2E TEST 4: CODE INGESTION PIPELINE
# =============================================================================

class TestCodeIngestion:
    """Test code parsing and graph ingestion."""
    
    def test_self_ingestion(self, fresh_db):
        """Paragon can parse its own codebase."""
        from domain.code_parser import parse_python_directory
        
        db = fresh_db
        paragon_root = Path(__file__).parent.parent.parent
        
        nodes, edges = parse_python_directory(paragon_root, recursive=True)
        
        # Should have parsed many nodes
        assert len(nodes) >= 10
        
        # Add to graph
        db.add_nodes_batch(nodes)
        
        # Filter valid edges
        valid_node_ids = {n.id for n in nodes}
        valid_edges = [e for e in edges 
                       if e.source_id in valid_node_ids 
                       and e.target_id in valid_node_ids]
        db.add_edges_batch(valid_edges)
        
        # Verify structure
        assert db.node_count >= 10
        
    def test_code_node_syntax_validation(self, fresh_db):
        """add_node_safe rejects invalid Python syntax."""
        result = add_node_safe(
            node_type="CODE",
            content="def broken(:\n    pass",  # Invalid syntax
            created_by="test",
        )
        
        assert not result.success
        assert not result.syntax_valid


# =============================================================================
# E2E TEST 5: QUALITY GATE PIPELINE
# =============================================================================

class TestQualityGate:
    """Test quality gate enforcement."""
    
    def test_valid_code_passes_gate(self, fresh_db):
        """Valid code passes all quality checks."""
        result = add_node_safe(
            node_type="CODE",
            content="def add(a: int, b: int) -> int:\n    return a + b",
            created_by="test",
        )
        
        assert result.success
        assert result.syntax_valid
        assert result.node_id != ""
        
    def test_graph_invariants_after_batch_ops(self, fresh_db):
        """Graph invariants hold after batch operations."""
        db = fresh_db
        
        # Create batch of nodes
        nodes = [
            NodeData.create(type=NodeType.CODE.value, content=f"node_{i}")
            for i in range(100)
        ]
        db.add_nodes_batch(nodes)
        
        # Create chain of edges
        edges = [
            EdgeData.create(nodes[i].id, nodes[i+1].id, EdgeType.DEPENDS_ON.value)
            for i in range(99)
        ]
        db.add_edges_batch(edges)
        
        # Validate
        report = GraphInvariants.validate_all(db._graph)
        assert report.valid


# =============================================================================
# E2E TEST 6: ORCHESTRATOR INTEGRATION
# =============================================================================

class TestOrchestratorIntegration:
    """Test orchestrator runs complete workflows."""
    
    def test_orchestrator_completes_cycle(self, fresh_db):
        """Orchestrator runs through all phases."""
        import agents.orchestrator as orch
        from agents.orchestrator import TDDOrchestrator
        from core.llm import reset_llm
        
        # Reset state
        reset_llm()
        
        # Disable LLM for deterministic testing
        original_llm = orch.LLM_AVAILABLE
        orch.LLM_AVAILABLE = False
        
        try:
            orchestrator = TDDOrchestrator(enable_checkpointing=False)
            
            result = orchestrator.run(
                session_id="e2e-test",
                task_id="e2e-task",
                spec="Create a simple greeting function",
                requirements=["Must return a string"],
                max_iterations=3,
            )
            
            assert result is not None
            assert result.get("phase") in ["passed", "failed"]
            
        finally:
            orch.LLM_AVAILABLE = original_llm


# =============================================================================
# E2E TEST 7: CROSS-MODULE INTEGRATION
# =============================================================================

class TestCrossModuleIntegration:
    """Test integration between multiple modules."""
    
    def test_tools_create_valid_graph(self, fresh_db):
        """Tools create graphs that pass all validations."""
        # Create nodes via tools
        req_result = add_node("REQ", "Build a calculator", created_by="test")
        spec_result = add_node("SPEC", "def add(a, b): return a + b", created_by="test")
        code_result = add_node_safe(
            node_type="CODE",
            content="def add(a, b):\n    return a + b",
            created_by="test",
        )
        
        assert req_result.success
        assert spec_result.success
        assert code_result.success
        
        # Create edges
        add_edge(spec_result.node_id, req_result.node_id, EdgeType.TRACES_TO.value)
        add_edge(code_result.node_id, spec_result.node_id, EdgeType.IMPLEMENTS.value)
        
        # Validate
        db = get_db()
        invariant_report = GraphInvariants.validate_all(db._graph)
        assert invariant_report.valid
        
        teleology_report = validate_teleology(db._graph, db._node_map, db._inv_map)
        assert teleology_report.is_valid


# =============================================================================
# MAIN
# =============================================================================

def run_e2e_tests():
    """Run all e2e tests and return summary."""
    import unittest
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestTDDWorkflow,
        TestTeleology,
        TestGraphInvariants,
        TestCodeIngestion,
        TestQualityGate,
        TestOrchestratorIntegration,
        TestCrossModuleIntegration,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("E2E TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    passed = result.testsRun - len(result.failures) - len(result.errors)
    return passed, result.testsRun, [str(f[0]) for f in result.failures + result.errors]


if __name__ == "__main__":
    passed, total, errors = run_e2e_tests()
    sys.exit(0 if passed == total else 1)
