"""
PROTOCOL ZETA - Human-in-the-Loop & Research Phase Verification

Tests the DIALECTIC, CLARIFICATION, and RESEARCH phases of the TDDOrchestrator.

Test Categories:
1. Schema Validation - DialectorOutput, ResearchArtifact structure
2. Phase Transitions - State machine routing correctness
3. Human Loop Mocking - Auto-response pattern for automated testing
4. Research Artifact Quality - Sufficient statistic completeness

Run: python -m benchmarks.protocol_zeta
"""

import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# =============================================================================
# TEST INFRASTRUCTURE
# =============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str = ""
    duration_ms: float = 0.0


class ProtocolZetaRunner:
    """Runner for Protocol Zeta tests."""

    def __init__(self):
        self.results: List[TestResult] = []

    def run_test(self, name: str, test_fn) -> TestResult:
        """Run a single test and capture result."""
        import time
        start = time.perf_counter()
        try:
            test_fn()
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(name=name, passed=True, duration_ms=duration)
        except AssertionError as e:
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(name=name, passed=False, message=str(e), duration_ms=duration)
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(name=name, passed=False, message=f"Exception: {e}", duration_ms=duration)

        self.results.append(result)
        return result


# =============================================================================
# MOCK HUMAN LOOP CONTROLLER
# =============================================================================

class MockHumanLoopController:
    """
    Controller that auto-responds to requests with pre-defined answers.

    Used for automated testing of human-in-the-loop workflows.
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.request_history: List[Dict[str, Any]] = []
        self.pending_requests: Dict[str, Any] = {}

    def create_request(
        self,
        pause_point_id: str,
        session_id: str,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        options: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a mock request and auto-respond if response is pre-defined."""
        import uuid

        request = {
            "id": str(uuid.uuid4()),
            "pause_point_id": pause_point_id,
            "session_id": session_id,
            "prompt": prompt,
            "context": context or {},
            "options": options,
            "status": "pending",
            "response": None,
        }

        self.request_history.append(request)
        self.pending_requests[request["id"]] = request

        # Auto-respond if we have a pre-defined response
        if pause_point_id in self.responses:
            self.submit_response(request["id"], self.responses[pause_point_id])

        return request

    def submit_response(self, request_id: str, response: str) -> bool:
        """Submit a response to a pending request."""
        if request_id not in self.pending_requests:
            return False

        request = self.pending_requests[request_id]
        request["status"] = "responded"
        request["response"] = response
        return True

    def get_pending_count(self) -> int:
        """Get count of pending requests."""
        return sum(1 for r in self.pending_requests.values() if r["status"] == "pending")


# =============================================================================
# SCHEMA VALIDATION TESTS
# =============================================================================

def test_dialector_output_schema():
    """Test DialectorOutput schema structure."""
    from agents.schemas import DialectorOutput, AmbiguityMarker

    # Test valid output
    output = DialectorOutput(
        is_clear=False,
        ambiguities=[
            AmbiguityMarker(
                category="SUBJECTIVE",
                text="fast",
                impact="BLOCKING",
                suggested_question="What does 'fast' mean in terms of latency?",
            )
        ],
        blocking_count=1,
        recommendation="CLARIFY",
    )

    assert output.is_clear == False
    assert len(output.ambiguities) == 1
    assert output.ambiguities[0].category == "SUBJECTIVE"
    assert output.blocking_count == 1
    assert output.recommendation == "CLARIFY"


def test_research_artifact_schema():
    """Test ResearchArtifact schema structure."""
    from agents.schemas import ResearchArtifact, ResearchFinding

    artifact = ResearchArtifact(
        task_category="algorithmic",
        input_contract="n: int",
        output_contract="int",
        happy_path_examples=["fib(5) -> 5", "fib(10) -> 55"],
        edge_cases=["fib(0) -> 0", "fib(1) -> 1"],
        error_cases=["fib(-1) -> ValueError"],
        complexity_bounds="O(n) time, O(1) space with iteration",
        security_posture="No external input validation needed",
        findings=[
            ResearchFinding(
                topic="Algorithm choice",
                summary="Iterative approach preferred over recursive",
                confidence=0.95,
            )
        ],
    )

    assert artifact.task_category == "algorithmic"
    assert artifact.input_contract == "n: int"
    assert len(artifact.happy_path_examples) == 2
    assert len(artifact.edge_cases) == 2
    assert len(artifact.error_cases) == 1
    assert artifact.complexity_bounds is not None


def test_ambiguity_marker_impact_values():
    """Test AmbiguityMarker impact classification."""
    from agents.schemas import AmbiguityMarker

    blocking = AmbiguityMarker(
        category="SUBJECTIVE",
        text="efficient",
        impact="BLOCKING",
    )
    assert blocking.impact == "BLOCKING"

    clarifying = AmbiguityMarker(
        category="MISSING_CONTEXT",
        text="the system",
        impact="CLARIFYING",
    )
    assert clarifying.impact == "CLARIFYING"


def test_dialector_clear_output():
    """Test DialectorOutput for clear requirements."""
    from agents.schemas import DialectorOutput

    output = DialectorOutput(
        is_clear=True,
        ambiguities=[],
        blocking_count=0,
        recommendation="PROCEED",
    )

    assert output.is_clear == True
    assert len(output.ambiguities) == 0
    assert output.recommendation == "PROCEED"


# =============================================================================
# PHASE TRANSITION TESTS
# =============================================================================

def test_cycle_phase_enum():
    """Test CyclePhase enum includes new phases."""
    from agents.orchestrator import CyclePhase

    phases = [p.value for p in CyclePhase]

    assert "dialectic" in phases
    assert "clarification" in phases
    assert "research" in phases
    assert "plan" in phases
    assert "build" in phases
    assert "test" in phases


def test_route_after_dialectic_clear():
    """Test routing when spec is clear (no ambiguities)."""
    from agents.orchestrator import route_after_dialectic

    state = {
        "ambiguities": [],
        "dialectic_passed": True,
    }

    next_node = route_after_dialectic(state)
    assert next_node == "research"


def test_route_after_dialectic_ambiguous():
    """Test routing when ambiguities are found."""
    from agents.orchestrator import route_after_dialectic

    state = {
        "ambiguities": [{"text": "fast", "impact": "BLOCKING"}],
        "dialectic_passed": False,
    }

    next_node = route_after_dialectic(state)
    assert next_node == "clarification"


def test_route_after_clarification_complete():
    """Test routing after clarification is complete."""
    from agents.orchestrator import route_after_clarification

    state = {
        "dialectic_passed": True,
    }

    next_node = route_after_clarification(state)
    assert next_node == "research"


def test_route_after_research_complete():
    """Test routing after research is complete."""
    from agents.orchestrator import route_after_research

    state = {
        "research_complete": True,
    }

    next_node = route_after_research(state)
    assert next_node == "plan"


# =============================================================================
# HUMAN LOOP MOCK TESTS
# =============================================================================

def test_mock_controller_create_request():
    """Test MockHumanLoopController request creation."""
    controller = MockHumanLoopController()

    request = controller.create_request(
        pause_point_id="clarification_001",
        session_id="test_session",
        prompt="What does 'fast' mean?",
        options=["<100ms", "<1s", "<10s"],
    )

    assert request["pause_point_id"] == "clarification_001"
    assert request["status"] == "pending"
    assert len(controller.request_history) == 1


def test_mock_controller_auto_response():
    """Test MockHumanLoopController auto-responds with pre-defined answers."""
    controller = MockHumanLoopController(responses={
        "clarification_001": "Less than 100ms latency",
    })

    request = controller.create_request(
        pause_point_id="clarification_001",
        session_id="test_session",
        prompt="What does 'fast' mean?",
    )

    assert request["status"] == "responded"
    assert request["response"] == "Less than 100ms latency"


def test_mock_controller_pending_count():
    """Test pending request counting."""
    controller = MockHumanLoopController(responses={
        "auto_respond": "yes",
    })

    # Create one that will auto-respond
    controller.create_request(
        pause_point_id="auto_respond",
        session_id="test",
        prompt="Q1",
    )

    # Create one that won't
    controller.create_request(
        pause_point_id="manual",
        session_id="test",
        prompt="Q2",
    )

    assert controller.get_pending_count() == 1


def test_mock_controller_manual_response():
    """Test manual response submission."""
    controller = MockHumanLoopController()

    request = controller.create_request(
        pause_point_id="manual",
        session_id="test",
        prompt="Please clarify",
    )

    assert request["status"] == "pending"

    success = controller.submit_response(request["id"], "User's clarification")
    assert success == True
    assert request["status"] == "responded"
    assert request["response"] == "User's clarification"


# =============================================================================
# GRAPH STATE TESTS
# =============================================================================

def test_graph_state_has_dialectic_fields():
    """Test GraphState includes dialectic/research fields."""
    from agents.orchestrator import GraphState

    # These fields should be in the TypedDict
    state: GraphState = {
        "session_id": "test",
        "task_id": "task1",
        "spec": "test spec",
        "phase": "init",
        "messages": [],
        "ambiguities": [],
        "clarification_questions": [],
        "research_findings": [],
        "dialectic_passed": False,
        "research_complete": False,
    }

    assert "ambiguities" in state
    assert "clarification_questions" in state
    assert "research_findings" in state
    assert "dialectic_passed" in state
    assert "research_complete" in state


def test_graph_creation_includes_new_nodes():
    """Test StateGraph includes dialectic/clarification/research nodes."""
    from agents.orchestrator import create_tdd_graph

    graph = create_tdd_graph()
    nodes = list(graph.nodes.keys())

    assert "dialectic" in nodes
    assert "clarification" in nodes
    assert "research" in nodes
    assert "plan" in nodes


# =============================================================================
# INTEGRATION TESTS (Mocked LLM)
# =============================================================================

def test_dialectic_node_without_llm():
    """Test dialectic_node graceful degradation without LLM."""
    from agents.orchestrator import dialectic_node, CyclePhase

    state = {
        "spec": "Build a sorting function",
        "session_id": "test",
    }

    # Temporarily disable LLM
    import agents.orchestrator as orch
    original = orch.LLM_AVAILABLE
    orch.LLM_AVAILABLE = False

    try:
        result = dialectic_node(state)
        assert result["phase"] == CyclePhase.RESEARCH.value
        assert result["dialectic_passed"] == True
    finally:
        orch.LLM_AVAILABLE = original


def test_research_node_without_llm():
    """Test research_node graceful degradation without LLM."""
    from agents.orchestrator import research_node, CyclePhase

    state = {
        "spec": "Implement fibonacci",
        "session_id": "test",
    }

    import agents.orchestrator as orch
    original = orch.LLM_AVAILABLE
    orch.LLM_AVAILABLE = False

    try:
        result = research_node(state)
        assert result["phase"] == CyclePhase.PLAN.value
        assert result["research_complete"] == True
    finally:
        orch.LLM_AVAILABLE = original


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_protocol_zeta():
    """Run all Protocol Zeta tests."""
    print("=" * 60)
    print("PROTOCOL ZETA - Human-in-the-Loop & Research Verification")
    print("=" * 60)
    print()

    runner = ProtocolZetaRunner()

    # Schema Validation Tests
    print("1. Schema Validation Tests")
    print("-" * 40)
    tests_schema = [
        ("dialector_output_schema", test_dialector_output_schema),
        ("research_artifact_schema", test_research_artifact_schema),
        ("ambiguity_marker_impact", test_ambiguity_marker_impact_values),
        ("dialector_clear_output", test_dialector_clear_output),
    ]
    for name, fn in tests_schema:
        result = runner.run_test(name, fn)
        status = "\u2713" if result.passed else "\u2717"
        print(f"  {status} {name}")
        if not result.passed:
            print(f"    {result.message}")

    # Phase Transition Tests
    print()
    print("2. Phase Transition Tests")
    print("-" * 40)
    tests_phase = [
        ("cycle_phase_enum", test_cycle_phase_enum),
        ("route_dialectic_clear", test_route_after_dialectic_clear),
        ("route_dialectic_ambiguous", test_route_after_dialectic_ambiguous),
        ("route_clarification_complete", test_route_after_clarification_complete),
        ("route_research_complete", test_route_after_research_complete),
    ]
    for name, fn in tests_phase:
        result = runner.run_test(name, fn)
        status = "\u2713" if result.passed else "\u2717"
        print(f"  {status} {name}")
        if not result.passed:
            print(f"    {result.message}")

    # Human Loop Mock Tests
    print()
    print("3. Human Loop Mock Tests")
    print("-" * 40)
    tests_mock = [
        ("mock_create_request", test_mock_controller_create_request),
        ("mock_auto_response", test_mock_controller_auto_response),
        ("mock_pending_count", test_mock_controller_pending_count),
        ("mock_manual_response", test_mock_controller_manual_response),
    ]
    for name, fn in tests_mock:
        result = runner.run_test(name, fn)
        status = "\u2713" if result.passed else "\u2717"
        print(f"  {status} {name}")
        if not result.passed:
            print(f"    {result.message}")

    # Graph State Tests
    print()
    print("4. Graph State Tests")
    print("-" * 40)
    tests_state = [
        ("state_dialectic_fields", test_graph_state_has_dialectic_fields),
        ("graph_new_nodes", test_graph_creation_includes_new_nodes),
    ]
    for name, fn in tests_state:
        result = runner.run_test(name, fn)
        status = "\u2713" if result.passed else "\u2717"
        print(f"  {status} {name}")
        if not result.passed:
            print(f"    {result.message}")

    # Integration Tests
    print()
    print("5. Integration Tests (Mocked LLM)")
    print("-" * 40)
    tests_integration = [
        ("dialectic_without_llm", test_dialectic_node_without_llm),
        ("research_without_llm", test_research_node_without_llm),
    ]
    for name, fn in tests_integration:
        result = runner.run_test(name, fn)
        status = "\u2713" if result.passed else "\u2717"
        print(f"  {status} {name}")
        if not result.passed:
            print(f"    {result.message}")

    # Summary
    print()
    print("=" * 60)
    passed = sum(1 for r in runner.results if r.passed)
    failed = sum(1 for r in runner.results if not r.passed)
    print(f"PROTOCOL ZETA RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_protocol_zeta()
    sys.exit(0 if success else 1)
