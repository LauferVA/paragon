"""
Forensic Analysis Engine - Demonstration Script

This script demonstrates the ForensicAnalyzer's ability to:
1. Classify failures (F1-F5)
2. Trace failures to responsible agents
3. Calculate attribution confidence
4. Analyze complex multi-phase failures

Run this script to see example output from the attribution engine.

Usage:
    python examples/forensic_analysis_demo.py
"""
import tempfile
from pathlib import Path
from datetime import datetime

from infrastructure.attribution import ForensicAnalyzer
from infrastructure.training_store import TrainingStore
from agents.schemas import (
    AgentSignature,
    SignatureChain,
    CyclePhase,
    SignatureAction,
)


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_attribution_result(result, scenario_name: str) -> None:
    """Print a formatted attribution result."""
    print(f"Scenario: {scenario_name}")
    print(f"Failure Code: {result.failure_code.value}")
    print(f"Attributed Agent: {result.attributed_agent_id}")
    print(f"Attributed Model: {result.attributed_model_id}")
    print(f"Attributed Phase: {result.attributed_phase.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Contributing Agents: {', '.join(result.contributing_agents) if result.contributing_agents else 'None'}")
    print(f"\nReasoning:\n  {result.reasoning}")
    print("-" * 80)


def scenario_1_simple_syntax_error(analyzer: ForensicAnalyzer) -> None:
    """Scenario 1: Simple syntax error with no signature chain."""
    print_header("Scenario 1: Simple Syntax Error (No Signature Chain)")

    result = analyzer.analyze_failure(
        session_id="demo_session_1",
        error_type="SyntaxError",
        error_message="invalid syntax: expected ':' at line 42 in function calculate_total()",
    )

    print_attribution_result(result, "Simple Syntax Error")


def scenario_2_build_failure_with_chain(analyzer: ForensicAnalyzer) -> None:
    """Scenario 2: Build failure with complete signature chain."""
    print_header("Scenario 2: Build Failure with Signature Chain")

    # Create a signature chain simulating a typical build process
    chain = SignatureChain(
        node_id="code_node_42",
        state_id="state_abc123",
        signatures=[
            AgentSignature(
                agent_id="architect_agent_v1",
                model_id="claude-sonnet-4-5-20250929",
                phase=CyclePhase.PLAN,
                action=SignatureAction.CREATED,
                temperature=0.7,
                context_constraints={"max_tokens": 4000},
                timestamp="2025-12-06T10:00:00Z",
            ),
            AgentSignature(
                agent_id="builder_agent_v2",
                model_id="claude-sonnet-4-5-20250929",
                phase=CyclePhase.BUILD,
                action=SignatureAction.CREATED,
                temperature=0.3,
                context_constraints={"max_tokens": 8000},
                timestamp="2025-12-06T10:05:00Z",
            ),
            AgentSignature(
                agent_id="tester_agent_v1",
                model_id="claude-sonnet-4-5-20250929",
                phase=CyclePhase.TEST,
                action=SignatureAction.VERIFIED,
                temperature=0.0,
                context_constraints={"max_tokens": 4000},
                timestamp="2025-12-06T10:10:00Z",
            ),
        ],
    )

    result = analyzer.analyze_failure(
        session_id="demo_session_2",
        error_type="NameError",
        error_message="name 'undefined_variable' is not defined in function process_data()",
        failed_node_id="code_node_42",
        signature_chain=chain,
    )

    print_attribution_result(result, "Build Failure with Signature Chain")


def scenario_3_test_failure(analyzer: ForensicAnalyzer) -> None:
    """Scenario 3: Test failure - inadequate test coverage."""
    print_header("Scenario 3: Test Failure (Inadequate Coverage)")

    chain = SignatureChain(
        node_id="test_node_99",
        state_id="state_xyz789",
        signatures=[
            AgentSignature(
                agent_id="builder_agent_v2",
                model_id="claude-sonnet-4-5-20250929",
                phase=CyclePhase.BUILD,
                action=SignatureAction.CREATED,
                temperature=0.3,
                context_constraints={},
                timestamp="2025-12-06T11:00:00Z",
            ),
            AgentSignature(
                agent_id="tester_agent_v1",
                model_id="claude-sonnet-4-5-20250929",
                phase=CyclePhase.TEST,
                action=SignatureAction.CREATED,
                temperature=0.2,
                context_constraints={},
                timestamp="2025-12-06T11:05:00Z",
            ),
        ],
    )

    result = analyzer.analyze_failure(
        session_id="demo_session_3",
        error_type="AssertionError",
        error_message="test_edge_case_handling failed: expected result to be None for empty input, got []",
        failed_node_id="test_node_99",
        signature_chain=chain,
    )

    print_attribution_result(result, "Test Failure (Inadequate Coverage)")


def scenario_4_external_failure(analyzer: ForensicAnalyzer) -> None:
    """Scenario 4: External failure - API timeout."""
    print_header("Scenario 4: External Failure (API Timeout)")

    chain = SignatureChain(
        node_id="api_integration_node",
        state_id="state_ext123",
        signatures=[
            AgentSignature(
                agent_id="builder_agent_v2",
                model_id="claude-sonnet-4-5-20250929",
                phase=CyclePhase.BUILD,
                action=SignatureAction.CREATED,
                temperature=0.3,
                context_constraints={},
                timestamp="2025-12-06T12:00:00Z",
            ),
        ],
    )

    result = analyzer.analyze_failure(
        session_id="demo_session_4",
        error_type="TimeoutError",
        error_message="HTTPError: Connection to https://api.example.com/v1/data timed out after 30 seconds",
        failed_node_id="api_integration_node",
        signature_chain=chain,
    )

    print_attribution_result(result, "External Failure (API Timeout)")


def scenario_5_research_failure(analyzer: ForensicAnalyzer) -> None:
    """Scenario 5: Research failure - ambiguous requirements."""
    print_header("Scenario 5: Research Failure (Ambiguous Requirements)")

    chain = SignatureChain(
        node_id="req_node_55",
        state_id="state_req456",
        signatures=[
            AgentSignature(
                agent_id="dialector_agent_v1",
                model_id="claude-sonnet-4-5-20250929",
                phase=CyclePhase.DIALECTIC,
                action=SignatureAction.CREATED,
                temperature=0.8,
                context_constraints={},
                timestamp="2025-12-06T09:00:00Z",
            ),
            AgentSignature(
                agent_id="researcher_agent_v1",
                model_id="claude-sonnet-4-5-20250929",
                phase=CyclePhase.RESEARCH,
                action=SignatureAction.MODIFIED,
                temperature=0.7,
                context_constraints={},
                timestamp="2025-12-06T09:15:00Z",
            ),
        ],
    )

    result = analyzer.analyze_failure(
        session_id="demo_session_5",
        error_type="ValidationError",
        error_message="Requirement validation failed: 'user-friendly interface' is too subjective - needs concrete acceptance criteria",
        failed_node_id="req_node_55",
        signature_chain=chain,
    )

    print_attribution_result(result, "Research Failure (Ambiguous Requirements)")


def scenario_6_multi_agent_complex_failure(analyzer: ForensicAnalyzer) -> None:
    """Scenario 6: Complex failure involving multiple agents and phases."""
    print_header("Scenario 6: Complex Multi-Agent Failure")

    chain = SignatureChain(
        node_id="complex_feature_node",
        state_id="state_complex",
        signatures=[
            AgentSignature(
                agent_id="dialector_agent_v1",
                model_id="claude-sonnet-4-5-20250929",
                phase=CyclePhase.DIALECTIC,
                action=SignatureAction.CREATED,
                temperature=0.8,
                context_constraints={},
                timestamp="2025-12-06T08:00:00Z",
            ),
            AgentSignature(
                agent_id="researcher_agent_v1",
                model_id="claude-sonnet-4-5-20250929",
                phase=CyclePhase.RESEARCH,
                action=SignatureAction.VERIFIED,
                temperature=0.7,
                context_constraints={},
                timestamp="2025-12-06T08:30:00Z",
            ),
            AgentSignature(
                agent_id="architect_agent_v1",
                model_id="claude-opus-4-5-20251101",
                phase=CyclePhase.PLAN,
                action=SignatureAction.CREATED,
                temperature=0.6,
                context_constraints={},
                timestamp="2025-12-06T09:00:00Z",
            ),
            AgentSignature(
                agent_id="builder_agent_v2",
                model_id="claude-sonnet-4-5-20250929",
                phase=CyclePhase.BUILD,
                action=SignatureAction.CREATED,
                temperature=0.3,
                context_constraints={"max_tokens": 16000},
                timestamp="2025-12-06T10:00:00Z",
            ),
            AgentSignature(
                agent_id="builder_agent_v2",
                model_id="claude-sonnet-4-5-20250929",
                phase=CyclePhase.BUILD,
                action=SignatureAction.REJECTED,
                temperature=0.3,
                context_constraints={},
                timestamp="2025-12-06T10:30:00Z",
            ),
            AgentSignature(
                agent_id="builder_agent_v3",
                model_id="claude-opus-4-5-20251101",
                phase=CyclePhase.BUILD,
                action=SignatureAction.MODIFIED,
                temperature=0.2,
                context_constraints={},
                timestamp="2025-12-06T11:00:00Z",
            ),
        ],
    )

    result = analyzer.analyze_failure(
        session_id="demo_session_6",
        error_type="TypeError",
        error_message="TypeError: unsupported operand type(s) for +: 'NoneType' and 'int' at line 156",
        failed_node_id="complex_feature_node",
        signature_chain=chain,
    )

    print_attribution_result(result, "Complex Multi-Agent Failure")


def scenario_7_multi_failure_session(analyzer: ForensicAnalyzer) -> None:
    """Scenario 7: Multiple failures in a single session."""
    print_header("Scenario 7: Multiple Failures in Single Session")

    failures = [
        {
            "error_type": "SyntaxError",
            "error_message": "Missing closing parenthesis in function call",
        },
        {
            "error_type": "ImportError",
            "error_message": "Cannot import name 'NonExistentClass' from module 'utils'",
        },
        {
            "error_type": "AssertionError",
            "error_message": "Test failed: expected status 200, got 404",
        },
        {
            "error_type": "ConnectionError",
            "error_message": "Failed to connect to database: connection refused",
        },
    ]

    results = analyzer.analyze_session_failures(
        session_id="demo_session_7",
        failures=failures,
    )

    print(f"Analyzed {len(results)} failures in session demo_session_7:\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.failure_code.value} - {result.attributed_phase.value} phase")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Agent: {result.attributed_agent_id}")
        print()


def main():
    """Run all demonstration scenarios."""
    print_header("FORENSIC ANALYSIS ENGINE - DEMONSTRATION")
    print("This demonstration shows the ForensicAnalyzer's ability to classify")
    print("failures and attribute them to responsible agents with confidence scores.")

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    try:
        # Initialize store and analyzer
        store = TrainingStore(db_path=db_path)
        analyzer = ForensicAnalyzer(store=store)

        # Run all scenarios
        scenario_1_simple_syntax_error(analyzer)
        scenario_2_build_failure_with_chain(analyzer)
        scenario_3_test_failure(analyzer)
        scenario_4_external_failure(analyzer)
        scenario_5_research_failure(analyzer)
        scenario_6_multi_agent_complex_failure(analyzer)
        scenario_7_multi_failure_session(analyzer)

        print_header("DEMONSTRATION COMPLETE")
        print("The ForensicAnalyzer successfully:")
        print("  ✓ Classified failures using the F1-F5 taxonomy")
        print("  ✓ Attributed failures to specific agents and phases")
        print("  ✓ Calculated confidence scores based on signature chains")
        print("  ✓ Identified contributing agents in multi-phase workflows")
        print("  ✓ Handled edge cases (missing data, external failures, etc.)")
        print("\nThis data can be used for:")
        print("  - Learning which agents perform best under which constraints")
        print("  - Improving model routing decisions")
        print("  - Debugging complex failures")
        print("  - Quality improvement feedback loops")

    finally:
        # Cleanup
        db_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
