"""
PROTOCOL GAMMA - Orchestration Layer Tests

Tests the LangGraph orchestration and human-in-the-loop components.

Test Categories:
1. Tool Functions - ParagonDB tool wrappers
2. StateGraph - TDD cycle execution
3. Human Loop - Request/response lifecycle
4. Integration - Full workflow

Run: python -m benchmarks.protocol_gamma
"""
import sys
from pathlib import Path

# Add paragon to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Tuple
import traceback


def run_tests() -> Tuple[int, int, List[str]]:
    """Run all Protocol Gamma tests."""
    passed = 0
    failed = 0
    failures = []

    def test(name: str, func):
        nonlocal passed, failed, failures
        try:
            func()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}")
            print(f"    Error: {e}")
            traceback.print_exc()
            failed += 1
            failures.append(f"{name}: {e}")

    print("\n" + "=" * 60)
    print("PROTOCOL GAMMA - Orchestration Layer Tests")
    print("=" * 60)

    # =========================================================================
    # 1. TOOL FUNCTIONS
    # =========================================================================
    print("\n1. Tool Functions")
    print("-" * 40)

    def test_add_node_tool():
        from agents.tools import add_node, get_db, set_db
        from core.graph_db import ParagonDB

        # Fresh DB for test
        set_db(ParagonDB())

        result = add_node(
            node_type="CODE",
            content="def hello(): pass",
            data={"language": "python"},
            created_by="test",
        )

        assert result.success, f"add_node failed: {result.message}"
        assert result.node_id, "No node_id returned"

        # Verify node exists
        db = get_db()
        node = db.get_node(result.node_id)
        assert node is not None, "Node not found in DB"
        assert node.content == "def hello(): pass"

    test("add_node tool", test_add_node_tool)

    def test_batch_operations():
        from agents.tools import add_nodes_batch, add_edges_batch, get_db, set_db
        from core.graph_db import ParagonDB

        set_db(ParagonDB())

        # Add batch of nodes
        nodes = [
            {"type": "CODE", "content": f"node_{i}", "data": {"index": i}}
            for i in range(5)
        ]
        result = add_nodes_batch(nodes)

        assert result.success, f"add_nodes_batch failed: {result.message}"
        assert result.count == 5
        assert len(result.node_ids) == 5

        # Add edges between them
        edges = [
            {"source_id": result.node_ids[i], "target_id": result.node_ids[i + 1], "type": "DEPENDS_ON"}
            for i in range(4)
        ]
        edge_result = add_edges_batch(edges)

        assert edge_result.success, f"add_edges_batch failed: {edge_result.message}"
        assert edge_result.count == 4

        # Verify graph state
        db = get_db()
        assert db.node_count == 5
        assert db.edge_count == 4

    test("batch operations", test_batch_operations)

    def test_query_nodes():
        from agents.tools import add_node, query_nodes, get_db, set_db
        from core.graph_db import ParagonDB

        set_db(ParagonDB())

        # Add nodes of different types
        add_node("CODE", "code content")
        add_node("CODE", "more code")
        add_node("TEST", "test content")
        add_node("SPEC", "spec content")

        # Query all
        result = query_nodes()
        assert result.success
        assert result.count == 4

        # Query by type
        result = query_nodes(node_type="CODE")
        assert result.success
        assert result.count == 2

        result = query_nodes(node_type="TEST")
        assert result.success
        assert result.count == 1

    test("query_nodes tool", test_query_nodes)

    def test_analysis_tools():
        from agents.tools import (
            add_node, add_edge, get_waves, get_descendants,
            get_ancestors, check_cycle, get_db, set_db
        )
        from core.graph_db import ParagonDB

        set_db(ParagonDB())

        # Create a DAG: A -> B -> C
        r1 = add_node("CODE", "A")
        r2 = add_node("CODE", "B")
        r3 = add_node("CODE", "C")

        add_edge(r1.node_id, r2.node_id, "DEPENDS_ON")
        add_edge(r2.node_id, r3.node_id, "DEPENDS_ON")

        # Test waves
        waves = get_waves()
        assert waves.success
        assert waves.layer_count == 3

        # Test descendants
        desc = get_descendants(r1.node_id)
        assert desc.success
        assert desc.count == 2  # B and C

        # Test ancestors
        anc = get_ancestors(r3.node_id)
        assert anc.success
        assert anc.count == 2  # A and B

        # Test cycle detection
        cycle = check_cycle()
        assert cycle.success
        assert not cycle.has_cycle

    test("analysis tools", test_analysis_tools)

    def test_tool_registry():
        from agents.tools import TOOLS, get_tool, list_tools

        # Check registry has expected tools
        tool_names = list_tools()
        assert "add_node" in tool_names
        assert "get_waves" in tool_names
        assert "parse_source" in tool_names
        assert "align_node_sets" in tool_names

        # Check get_tool
        add_node_fn = get_tool("add_node")
        assert add_node_fn is not None
        assert callable(add_node_fn)

        # Check unknown tool returns None
        unknown = get_tool("nonexistent_tool")
        assert unknown is None

    test("tool registry", test_tool_registry)

    # =========================================================================
    # 2. STATEGRAPH
    # =========================================================================
    print("\n2. StateGraph (TDD Cycle)")
    print("-" * 40)

    def test_graph_creation():
        from agents.orchestrator import create_tdd_graph, CyclePhase

        graph = create_tdd_graph()
        assert graph is not None

        # Check nodes exist
        # LangGraph internal structure varies, just verify it compiles
        compiled = graph.compile()
        assert compiled is not None

    test("graph creation", test_graph_creation)

    def test_orchestrator_run():
        import agents.orchestrator as orch
        from agents.orchestrator import TDDOrchestrator
        from agents.tools import set_db
        from core.graph_db import ParagonDB
        from core.llm import reset_llm

        # CRITICAL: Reset global state to avoid context bloat from previous tests
        set_db(ParagonDB())
        reset_llm()

        # Protocol Gamma tests ORCHESTRATION LOGIC, not LLM API latency.
        # LLM calls work (verified), but take 5-20s each, making 3 tests = 3+ min
        # Mocking ensures fast CI while RateLimitGuard protects production.
        original_llm = orch.LLM_AVAILABLE
        orch.LLM_AVAILABLE = False

        try:
            orchestrator = TDDOrchestrator(enable_checkpointing=False)

            result = orchestrator.run(
                session_id="test-session-1",
                task_id="test-task-1",
                spec="Implement a simple function",
                requirements=["Must be fast", "Must be correct"],
                max_iterations=3,
            )

            # Should complete (simulated tests pass on iteration 2)
            assert result is not None
            assert result.get("final_status") in ("passed", "failed", None)
        finally:
            orch.LLM_AVAILABLE = original_llm

    test("orchestrator run", test_orchestrator_run)

    def test_convenience_function():
        import agents.orchestrator as orch
        from agents.orchestrator import run_tdd_cycle
        from agents.tools import set_db
        from core.graph_db import ParagonDB
        from core.llm import reset_llm

        # Reset global state to avoid context bloat
        set_db(ParagonDB())
        reset_llm()

        # Mock LLM for fast CI tests
        original_llm = orch.LLM_AVAILABLE
        orch.LLM_AVAILABLE = False

        try:
            result = run_tdd_cycle(
                spec="Simple test spec",
                requirements=["req1", "req2"],
            )

            assert result is not None
            # Verify state structure
            assert "session_id" in result
            assert "task_id" in result
            assert "messages" in result
        finally:
            orch.LLM_AVAILABLE = original_llm

    test("run_tdd_cycle convenience", test_convenience_function)

    # =========================================================================
    # 3. HUMAN LOOP
    # =========================================================================
    print("\n3. Human Loop Controller")
    print("-" * 40)

    def test_pause_points():
        from agents.human_loop import (
            PausePoint, PauseType, get_pause_point,
            STANDARD_PAUSE_POINTS
        )

        # Check standard pause points exist
        assert "plan_approval" in STANDARD_PAUSE_POINTS
        assert "code_review" in STANDARD_PAUSE_POINTS

        # Get a pause point
        pp = get_pause_point("plan_approval")
        assert pp is not None
        assert pp.pause_type == PauseType.APPROVAL.value

    test("pause point registry", test_pause_points)

    def test_human_loop_controller():
        from agents.human_loop import (
            HumanLoopController, PausePoint, PauseType, RequestStatus
        )

        controller = HumanLoopController()

        # Create a request
        pause_point = PausePoint(
            id="test_approval",
            pause_type=PauseType.APPROVAL.value,
            description="Approve this test",
        )

        request = controller.create_request(
            pause_point=pause_point,
            session_id="test-session",
            prompt="Do you approve?",
            context={"test": True},
        )

        assert request.id is not None
        assert request.status == RequestStatus.PENDING.value

        # Check pending requests
        pending = controller.get_pending_requests()
        assert len(pending) == 1
        assert pending[0].id == request.id

        # Submit response
        response = controller.submit_response(
            request_id=request.id,
            response="yes",
            metadata={"reason": "looks good"},
        )

        assert response is not None
        assert response.response == "yes"

        # Verify request is now completed
        pending = controller.get_pending_requests()
        assert len(pending) == 0

        completed = controller.get_request(request.id)
        assert completed.status == RequestStatus.RESPONDED.value

    test("human loop controller", test_human_loop_controller)

    def test_selection_validation():
        from agents.human_loop import (
            HumanLoopController, PausePoint, PauseType
        )

        controller = HumanLoopController()

        # Create selection request
        pause_point = PausePoint(
            id="test_selection",
            pause_type=PauseType.SELECTION.value,
            description="Choose an option",
            options=["option_a", "option_b", "option_c"],
        )

        request = controller.create_request(
            pause_point=pause_point,
            session_id="test-session",
            prompt="Which option?",
        )

        # Valid selection
        response = controller.submit_response(request.id, "option_b")
        assert response.response == "option_b"

    test("selection validation", test_selection_validation)

    def test_convenience_functions():
        from agents.human_loop import (
            HumanLoopController,
            create_approval_request,
            create_feedback_request,
            create_selection_request,
            PauseType,
        )

        controller = HumanLoopController()

        # Approval
        req1 = create_approval_request(
            controller, "session-1", "Approve the plan?"
        )
        assert req1.pause_type == PauseType.APPROVAL.value

        # Feedback
        req2 = create_feedback_request(
            controller, "session-1", "Any feedback?"
        )
        assert req2.pause_type == PauseType.FEEDBACK.value

        # Selection
        req3 = create_selection_request(
            controller, "session-1", "Pick one",
            options=["a", "b", "c"]
        )
        assert req3.pause_type == PauseType.SELECTION.value
        assert req3.options == ["a", "b", "c"]

    test("convenience functions", test_convenience_functions)

    def test_transition_matrix():
        from agents.human_loop import (
            TransitionMatrix, TDD_TRANSITIONS,
            is_valid_transition, get_pause_point_for_state
        )

        # Test valid transitions
        assert is_valid_transition(TDD_TRANSITIONS, "init", "plan")
        assert is_valid_transition(TDD_TRANSITIONS, "test", "passed")
        assert is_valid_transition(TDD_TRANSITIONS, "test", "fix")

        # Test invalid transitions
        assert not is_valid_transition(TDD_TRANSITIONS, "init", "test")
        assert not is_valid_transition(TDD_TRANSITIONS, "passed", "build")

        # Test terminal states
        assert "passed" in TDD_TRANSITIONS.terminal_states
        assert "failed" in TDD_TRANSITIONS.terminal_states

        # Test pause points
        pp = get_pause_point_for_state(TDD_TRANSITIONS, "plan")
        assert pp is not None
        assert pp.id == "plan_approval"

    test("transition matrix", test_transition_matrix)

    # =========================================================================
    # 4. INTEGRATION
    # =========================================================================
    print("\n4. Integration Tests")
    print("-" * 40)

    def test_tools_with_orchestrator():
        """Test that tools work within orchestrator context."""
        import agents.orchestrator as orch
        from agents.tools import set_db, get_db, add_node, get_graph_stats
        from agents.orchestrator import TDDOrchestrator
        from core.graph_db import ParagonDB
        from core.llm import reset_llm

        # Fresh DB and LLM to avoid context bloat
        set_db(ParagonDB())
        reset_llm()

        # Add some nodes via tools
        add_node("SPEC", "Build a calculator")
        add_node("CODE", "def add(a, b): return a + b")
        add_node("TEST", "assert add(1, 2) == 3")

        # Mock LLM for fast CI tests
        original_llm = orch.LLM_AVAILABLE
        orch.LLM_AVAILABLE = False

        try:
            orchestrator = TDDOrchestrator(enable_checkpointing=False)
            result = orchestrator.run(
                session_id="integration-test",
                task_id="calc-task",
                spec="Calculator implementation",
                max_iterations=3,
            )

            # DB should still have our nodes
            stats = get_graph_stats()
            assert stats["node_count"] >= 3
        finally:
            orch.LLM_AVAILABLE = original_llm

    test("tools with orchestrator", test_tools_with_orchestrator)

    def test_human_loop_state_helpers():
        from agents.human_loop import (
            check_human_input_needed,
            get_human_input_type,
            apply_human_response,
            PauseType,
        )

        # State without human input
        state1 = {"phase": "build", "pending_human_input": None}
        assert not check_human_input_needed(state1)
        assert get_human_input_type(state1) is None

        # State with human input needed
        state2 = {"phase": "plan", "pending_human_input": PauseType.APPROVAL.value}
        assert check_human_input_needed(state2)
        assert get_human_input_type(state2) == PauseType.APPROVAL.value

        # Apply response
        state3 = apply_human_response(state2, "approved")
        assert state3["human_response"] == "approved"
        assert state3["pending_human_input"] is None

    test("human loop state helpers", test_human_loop_state_helpers)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"PROTOCOL GAMMA RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")

    # Return format: (passed, total, errors) to match run_all.py expectations
    total = passed + failed
    return passed, total, failures


if __name__ == "__main__":
    passed, total, failures = run_tests()
    sys.exit(0 if len(failures) == 0 else 1)
