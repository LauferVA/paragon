"""
Unit tests for the Paragon Orchestrator.

Focuses on the new DIALECTIC, CLARIFICATION, and RESEARCH phases
introduced in the TDD orchestration workflow.

Test Coverage:
1. Phase Enum Tests - Verify all phases are defined
2. State Tests - Verify GraphState has all required fields
3. Node Function Tests - Test each node function with mocked LLM
4. Routing Function Tests - Test conditional edge logic
5. Graph Structure Tests - Verify graph construction

Run: pytest tests/unit/agents/test_orchestrator.py -v
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def base_state() -> Dict[str, Any]:
    """Base state dictionary for testing."""
    return {
        "session_id": "test_session",
        "task_id": "test_task",
        "spec": "Implement a function to calculate fibonacci numbers",
        "requirements": ["Must handle edge cases", "Must be efficient"],
        "phase": "init",
        "messages": [],
        "artifacts": [],
        "code_node_ids": [],
        "test_node_ids": [],
        "test_results": [],
        "errors": [],
        "iteration": 0,
        "max_iterations": 3,
        "last_test_passed": False,
        "pending_human_input": None,
        "human_response": None,
        "ambiguities": [],
        "clarification_questions": [],
        "research_findings": [],
        "dialectic_passed": False,
        "research_complete": False,
        "final_status": None,
    }


@pytest.fixture
def mock_dialector_output():
    """Mock DialectorOutput for testing."""
    from agents.schemas import DialectorOutput, AmbiguityMarker

    return DialectorOutput(
        is_clear=False,
        ambiguities=[
            AmbiguityMarker(
                category="SUBJECTIVE",
                text="efficient",
                impact="BLOCKING",
                suggested_question="What does 'efficient' mean in terms of time complexity?",
            )
        ],
        blocking_count=1,
        recommendation="CLARIFY",
    )


@pytest.fixture
def mock_dialector_output_clear():
    """Mock DialectorOutput for clear requirements."""
    from agents.schemas import DialectorOutput

    return DialectorOutput(
        is_clear=True,
        ambiguities=[],
        blocking_count=0,
        recommendation="PROCEED",
    )


@pytest.fixture
def mock_research_artifact():
    """Mock ResearchArtifact for testing."""
    from agents.schemas import ResearchArtifact, ResearchFinding

    return ResearchArtifact(
        task_category="algorithmic",
        input_contract="n: int",
        output_contract="int",
        happy_path_examples=["fib(5) -> 5", "fib(10) -> 55"],
        edge_cases=["fib(0) -> 0", "fib(1) -> 1"],
        error_cases=["fib(-1) -> ValueError"],
        complexity_bounds="O(n) time, O(1) space",
        security_posture="Input validation required",
        findings=[
            ResearchFinding(
                topic="Algorithm",
                summary="Use iterative approach",
                confidence=0.95,
            )
        ],
    )


@pytest.fixture
def mock_implementation_plan():
    """Mock ImplementationPlan for testing."""
    from agents.schemas import ImplementationPlan, ComponentSpec, DependencyEdge

    return ImplementationPlan(
        explanation="Implement fibonacci function with input validation",
        components=[
            ComponentSpec(
                name="fibonacci",
                type="function",
                description="Calculate nth fibonacci number",
                dependencies=[],
                file_path="fibonacci.py",
            )
        ],
        dependencies=[],
        estimated_complexity="low",
    )


@pytest.fixture
def mock_code_generation():
    """Mock CodeGeneration for testing."""
    from agents.schemas import CodeGeneration

    return CodeGeneration(
        filename="fibonacci.py",
        code="""def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
""",
        imports=[],
        description="Iterative fibonacci implementation",
        language="python",
    )


# =============================================================================
# 1. PHASE ENUM TESTS
# =============================================================================

def test_cycle_phase_includes_all_phases():
    """Test that CyclePhase enum includes all required phases."""
    from agents.orchestrator import CyclePhase

    phases = [p.value for p in CyclePhase]

    # Original phases
    assert "init" in phases
    assert "plan" in phases
    assert "build" in phases
    assert "test" in phases
    assert "fix" in phases
    assert "passed" in phases
    assert "failed" in phases

    # New phases
    assert "dialectic" in phases
    assert "clarification" in phases
    assert "research" in phases


def test_phase_values_are_strings():
    """Test that all phase values are strings."""
    from agents.orchestrator import CyclePhase

    for phase in CyclePhase:
        assert isinstance(phase.value, str)
        assert len(phase.value) > 0


# =============================================================================
# 2. STATE TESTS
# =============================================================================

def test_graph_state_has_required_fields():
    """Test that GraphState TypedDict has all required fields."""
    from agents.orchestrator import GraphState

    # Get the annotations from TypedDict
    annotations = GraphState.__annotations__

    # Core fields
    assert "session_id" in annotations
    assert "task_id" in annotations
    assert "phase" in annotations
    assert "spec" in annotations

    # New dialectic/research fields
    assert "ambiguities" in annotations
    assert "clarification_questions" in annotations
    assert "research_findings" in annotations
    assert "dialectic_passed" in annotations
    assert "research_complete" in annotations

    # Human loop fields
    assert "pending_human_input" in annotations
    assert "human_response" in annotations


def test_state_reducers_append_correctly():
    """Test that list append reducer works correctly."""
    from agents.orchestrator import list_append_reducer

    # Test appending to existing list
    existing = [1, 2, 3]
    new = [4, 5]
    result = list_append_reducer(existing, new)
    assert result == [1, 2, 3, 4, 5]

    # Test with None existing
    result = list_append_reducer(None, [1, 2])
    assert result == [1, 2]

    # Test with None new
    result = list_append_reducer([1, 2], None)
    assert result == [1, 2]

    # Test with both None
    result = list_append_reducer(None, None)
    assert result == []


# =============================================================================
# 3. NODE FUNCTION TESTS (with mocked LLM)
# =============================================================================

def test_init_node_sets_phase_to_dialectic(base_state):
    """Test init_node transitions to DIALECTIC phase."""
    from agents.orchestrator import init_node, CyclePhase

    result = init_node(base_state)

    assert result["phase"] == CyclePhase.DIALECTIC.value
    assert result["iteration"] == 0
    assert result["dialectic_passed"] == False
    assert result["research_complete"] == False
    assert "messages" in result
    assert len(result["messages"]) > 0


def test_dialectic_node_clear_spec_proceeds_to_research(base_state, mock_dialector_output_clear):
    """Test dialectic_node with clear spec proceeds to RESEARCH."""
    from agents.orchestrator import dialectic_node, CyclePhase

    # Mock LLM to return clear output
    with patch("agents.orchestrator.LLM_AVAILABLE", True):
        with patch("agents.orchestrator.get_llm") as mock_get_llm:
            mock_llm = Mock()
            mock_llm.generate.return_value = mock_dialector_output_clear
            mock_get_llm.return_value = mock_llm

            result = dialectic_node(base_state)

            assert result["phase"] == CyclePhase.RESEARCH.value
            assert result["dialectic_passed"] == True
            assert len(result["ambiguities"]) == 0
            assert len(result["clarification_questions"]) == 0


def test_dialectic_node_ambiguous_spec_goes_to_clarification(base_state, mock_dialector_output):
    """Test dialectic_node with ambiguous spec goes to CLARIFICATION."""
    from agents.orchestrator import dialectic_node, CyclePhase

    with patch("agents.orchestrator.LLM_AVAILABLE", True):
        with patch("agents.orchestrator.get_llm") as mock_get_llm:
            mock_llm = Mock()
            mock_llm.generate.return_value = mock_dialector_output
            mock_get_llm.return_value = mock_llm

            result = dialectic_node(base_state)

            assert result["phase"] == CyclePhase.CLARIFICATION.value
            assert result["dialectic_passed"] == False
            assert len(result["ambiguities"]) > 0
            assert len(result["clarification_questions"]) > 0
            assert result["pending_human_input"] == "clarification"


def test_clarification_node_with_response_proceeds(base_state):
    """Test clarification_node with human response proceeds to RESEARCH."""
    from agents.orchestrator import clarification_node, CyclePhase

    state = base_state.copy()
    state["clarification_questions"] = [
        {"question": "What does 'efficient' mean?", "category": "SUBJECTIVE"}
    ]
    state["human_response"] = "Efficient means O(n) time complexity or better"

    result = clarification_node(state)

    assert result["phase"] == CyclePhase.RESEARCH.value
    assert result["dialectic_passed"] == True
    assert result["pending_human_input"] is None
    assert "User clarification:" in result["messages"][0]["content"]


def test_clarification_node_without_response_waits(base_state):
    """Test clarification_node without response stays in CLARIFICATION."""
    from agents.orchestrator import clarification_node, CyclePhase

    state = base_state.copy()
    state["clarification_questions"] = [
        {"question": "What does 'efficient' mean?", "category": "SUBJECTIVE"}
    ]
    state["human_response"] = None

    result = clarification_node(state)

    assert result["phase"] == CyclePhase.CLARIFICATION.value
    assert result["pending_human_input"] == "clarification"
    assert len(result["messages"]) > 0


def test_research_node_creates_artifact(base_state, mock_research_artifact):
    """Test research_node creates ResearchArtifact."""
    from agents.orchestrator import research_node, CyclePhase

    with patch("agents.orchestrator.LLM_AVAILABLE", True):
        with patch("agents.orchestrator.get_llm") as mock_get_llm:
            mock_llm = Mock()
            mock_llm.generate.return_value = mock_research_artifact
            mock_get_llm.return_value = mock_llm

            result = research_node(base_state)

            assert result["phase"] == CyclePhase.PLAN.value
            assert result["research_complete"] == True
            assert len(result["research_findings"]) > 0
            assert "spec" in result
            assert "Research Artifact" in result["spec"]


def test_research_node_without_llm_proceeds(base_state):
    """Test research_node without LLM proceeds to PLAN."""
    from agents.orchestrator import research_node, CyclePhase

    with patch("agents.orchestrator.LLM_AVAILABLE", False):
        result = research_node(base_state)

        assert result["phase"] == CyclePhase.PLAN.value
        assert result["research_complete"] == True


def test_plan_node_creates_spec_nodes(base_state, mock_implementation_plan):
    """Test plan_node creates SPEC nodes."""
    from agents.orchestrator import plan_node, CyclePhase

    with patch("agents.orchestrator.LLM_AVAILABLE", True):
        with patch("agents.orchestrator.get_llm") as mock_get_llm:
            with patch("agents.orchestrator.add_node") as mock_add_node:
                with patch("agents.orchestrator.get_graph_stats") as mock_stats:
                    # Setup mocks
                    mock_llm = Mock()
                    mock_llm.generate.return_value = mock_implementation_plan
                    mock_get_llm.return_value = mock_llm
                    mock_stats.return_value = {"node_count": 0}

                    mock_result = Mock()
                    mock_result.success = True
                    mock_result.node_id = "spec_001"
                    mock_add_node.return_value = mock_result

                    result = plan_node(base_state)

                    assert result["phase"] == CyclePhase.BUILD.value
                    assert len(result["artifacts"]) > 0
                    # Verify add_node was called
                    assert mock_add_node.called


def test_build_node_generates_code(base_state, mock_code_generation):
    """Test build_node generates code artifacts."""
    from agents.orchestrator import build_node, CyclePhase

    state = base_state.copy()
    state["artifacts"] = [{"type": "spec_nodes", "node_ids": ["spec_001"]}]

    with patch("agents.orchestrator.LLM_AVAILABLE", True):
        with patch("agents.orchestrator.get_llm") as mock_get_llm:
            with patch("agents.orchestrator.get_node") as mock_get_node:
                with patch("agents.orchestrator.add_node_safe") as mock_add_safe:
                    # Setup mocks
                    mock_llm = Mock()
                    mock_llm.generate.return_value = mock_code_generation
                    mock_get_llm.return_value = mock_llm

                    mock_get_node.return_value = {
                        "content": "Test spec",
                        "data": {"component_name": "fibonacci"},
                    }

                    mock_safe_result = Mock()
                    mock_safe_result.success = True
                    mock_safe_result.node_id = "code_001"
                    mock_safe_result.syntax_valid = True
                    mock_safe_result.topology_valid = True
                    mock_add_safe.return_value = mock_safe_result

                    result = build_node(state)

                    assert result["phase"] == CyclePhase.TEST.value
                    assert len(result["code_node_ids"]) > 0


def test_test_node_executes_tests(base_state):
    """Test test_node executes tests and returns results."""
    from agents.orchestrator import test_node, CyclePhase

    result = test_node(base_state)

    # Should have phase set
    assert result["phase"] in [CyclePhase.PASSED.value, CyclePhase.FIX.value]
    assert "test_results" in result
    assert len(result["test_results"]) > 0
    assert "last_test_passed" in result


# =============================================================================
# 4. ROUTING FUNCTION TESTS
# =============================================================================

def test_route_after_dialectic_clear():
    """Test routing after dialectic with clear spec."""
    from agents.orchestrator import route_after_dialectic

    state = {
        "ambiguities": [],
        "dialectic_passed": True,
    }

    next_node = route_after_dialectic(state)
    assert next_node == "research"


def test_route_after_dialectic_ambiguous():
    """Test routing after dialectic with ambiguities."""
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


def test_route_after_test_passed():
    """Test routing when tests pass."""
    from agents.orchestrator import route_after_test

    state = {
        "last_test_passed": True,
    }

    next_node = route_after_test(state)
    assert next_node == "passed"


def test_route_after_test_failed():
    """Test routing when tests fail."""
    from agents.orchestrator import route_after_test

    state = {
        "last_test_passed": False,
    }

    next_node = route_after_test(state)
    assert next_node == "fix"


def test_route_after_fix_within_limit():
    """Test routing after fix when under iteration limit."""
    from agents.orchestrator import route_after_fix

    state = {
        "iteration": 1,
        "max_iterations": 3,
    }

    next_node = route_after_fix(state)
    assert next_node == "build"


def test_route_after_fix_exceeded_limit():
    """Test routing after fix when iteration limit exceeded."""
    from agents.orchestrator import route_after_fix

    state = {
        "iteration": 3,
        "max_iterations": 3,
    }

    next_node = route_after_fix(state)
    assert next_node == "failed"


# =============================================================================
# 5. GRAPH STRUCTURE TESTS
# =============================================================================

def test_create_tdd_graph_has_all_nodes():
    """Test that created graph has all required nodes."""
    from agents.orchestrator import create_tdd_graph

    graph = create_tdd_graph()
    nodes = list(graph.nodes.keys())

    # Check all nodes exist
    assert "init" in nodes
    assert "dialectic" in nodes
    assert "clarification" in nodes
    assert "research" in nodes
    assert "plan" in nodes
    assert "build" in nodes
    assert "test" in nodes
    assert "fix" in nodes
    assert "passed" in nodes
    assert "failed" in nodes


def test_graph_compiles_successfully():
    """Test that graph compiles without errors."""
    from agents.orchestrator import create_orchestrator

    # Should compile without raising
    orchestrator = create_orchestrator(checkpointer=None)
    assert orchestrator is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_full_workflow_without_llm(base_state):
    """Test full workflow without LLM (graceful degradation)."""
    from agents.orchestrator import (
        init_node, dialectic_node, research_node, plan_node,
        CyclePhase
    )

    with patch("agents.orchestrator.LLM_AVAILABLE", False):
        # Run through init -> dialectic -> research -> plan
        result = init_node(base_state)
        assert result["phase"] == CyclePhase.DIALECTIC.value

        state = base_state.copy()
        state.update(result)
        result = dialectic_node(state)
        assert result["phase"] == CyclePhase.RESEARCH.value

        state.update(result)
        result = research_node(state)
        assert result["phase"] == CyclePhase.PLAN.value


def test_orchestrator_initialization():
    """Test TDDOrchestrator class initialization."""
    from agents.orchestrator import TDDOrchestrator

    # Test with checkpointing disabled
    orchestrator = TDDOrchestrator(enable_checkpointing=False)
    assert orchestrator.checkpointer is None
    assert orchestrator.graph is not None

    # Test with memory checkpointing
    orchestrator = TDDOrchestrator(
        enable_checkpointing=True,
        persist_to_sqlite=False
    )
    assert orchestrator.checkpointer is not None
    assert orchestrator.graph is not None


def test_run_tdd_cycle_convenience_function():
    """Test run_tdd_cycle convenience function."""
    from agents.orchestrator import run_tdd_cycle

    with patch("agents.orchestrator.LLM_AVAILABLE", False):
        result = run_tdd_cycle(
            spec="Test spec",
            requirements=["req1"],
            session_id="test_session",
            task_id="test_task",
        )

        assert result is not None
        assert "session_id" in result
        assert "final_status" in result


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

def test_dialectic_node_handles_llm_error(base_state):
    """Test dialectic_node handles LLM errors gracefully."""
    from agents.orchestrator import dialectic_node, CyclePhase

    with patch("agents.orchestrator.LLM_AVAILABLE", True):
        with patch("agents.orchestrator.get_llm") as mock_get_llm:
            mock_llm = Mock()
            mock_llm.generate.side_effect = Exception("LLM Error")
            mock_get_llm.return_value = mock_llm

            result = dialectic_node(base_state)

            # Should proceed to research despite error
            assert result["phase"] == CyclePhase.RESEARCH.value
            assert any("unavailable" in msg["content"].lower() for msg in result["messages"])


def test_research_node_handles_llm_error(base_state):
    """Test research_node handles LLM errors gracefully."""
    from agents.orchestrator import research_node, CyclePhase

    with patch("agents.orchestrator.LLM_AVAILABLE", True):
        with patch("agents.orchestrator.get_llm") as mock_get_llm:
            mock_llm = Mock()
            mock_llm.generate.side_effect = Exception("LLM Error")
            mock_get_llm.return_value = mock_llm

            result = research_node(base_state)

            # Should proceed to plan despite error
            assert result["phase"] == CyclePhase.PLAN.value
            assert result["research_complete"] == True


def test_fix_node_respects_max_iterations(base_state):
    """Test fix_node respects max_iterations limit."""
    from agents.orchestrator import fix_node, CyclePhase

    state = base_state.copy()
    state["iteration"] = 2
    state["max_iterations"] = 3
    state["test_results"] = [{"passed": False, "errors": ["test error"]}]

    result = fix_node(state)

    # Should go to FAILED when max iterations reached
    assert result["iteration"] == 3
    assert result["phase"] == CyclePhase.FAILED.value
    assert len(result["errors"]) > 0
