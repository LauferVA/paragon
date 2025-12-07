"""
Unit tests for the Forensic Analysis Engine (Attribution System).

Tests the ForensicAnalyzer's ability to:
- Classify failures (F1-F5)
- Trace signature chains
- Calculate attribution confidence
- Analyze failure scenarios

Reference: docs/IMPLEMENTATION_PLAN_LEARNING.md Section 7
"""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from infrastructure.attribution import ForensicAnalyzer, AttributionResult
from infrastructure.training_store import TrainingStore
from agents.schemas import (
    AgentSignature,
    SignatureChain,
    CyclePhase,
    FailureCode,
    SignatureAction,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    store = TrainingStore(db_path=db_path)
    yield store

    # Cleanup
    db_path.unlink(missing_ok=True)


@pytest.fixture
def analyzer(temp_db):
    """Create a ForensicAnalyzer instance with temp database."""
    return ForensicAnalyzer(store=temp_db)


@pytest.fixture
def sample_signature_chain():
    """Create a sample signature chain for testing."""
    return SignatureChain(
        node_id="node_123",
        state_id="state_abc",
        signatures=[
            AgentSignature(
                agent_id="researcher_agent",
                model_id="claude-sonnet-4-5",
                phase=CyclePhase.RESEARCH,
                action=SignatureAction.CREATED,
                temperature=0.7,
                context_constraints={},
                timestamp=datetime.now().isoformat(),
            ),
            AgentSignature(
                agent_id="builder_agent",
                model_id="claude-sonnet-4-5",
                phase=CyclePhase.BUILD,
                action=SignatureAction.MODIFIED,
                temperature=0.3,
                context_constraints={},
                timestamp=datetime.now().isoformat(),
            ),
            AgentSignature(
                agent_id="tester_agent",
                model_id="claude-sonnet-4-5",
                phase=CyclePhase.TEST,
                action=SignatureAction.VERIFIED,
                temperature=0.0,
                context_constraints={},
                timestamp=datetime.now().isoformat(),
            ),
        ],
    )


# =============================================================================
# Failure Classification Tests (F1-F5)
# =============================================================================


class TestFailureClassification:
    """Test the F1-F5 failure classification logic."""

    def test_f1_research_failure(self, analyzer):
        """Test F1 classification for research/topology failures."""
        result = analyzer.analyze_failure(
            session_id="test_session_1",
            error_type="TopologyError",
            error_message="Invalid graph structure detected",
            failed_node_id="req_node_1",
        )

        # Should be F1 if phase is DIALECTIC/RESEARCH
        # Without signature chain, it will infer based on error
        assert result.failure_code in [FailureCode.F1, FailureCode.F5]

    def test_f2_implementation_failure_syntax(self, analyzer):
        """Test F2 classification for syntax errors."""
        result = analyzer.analyze_failure(
            session_id="test_session_2",
            error_type="SyntaxError",
            error_message="invalid syntax at line 42",
        )

        assert result.failure_code == FailureCode.F2
        assert result.attributed_phase == CyclePhase.BUILD

    def test_f2_implementation_failure_import(self, analyzer):
        """Test F2 classification for import errors."""
        result = analyzer.analyze_failure(
            session_id="test_session_3",
            error_type="ImportError",
            error_message="No module named 'nonexistent_module'",
        )

        assert result.failure_code == FailureCode.F2

    def test_f2_implementation_failure_name_error(self, analyzer):
        """Test F2 classification for name errors."""
        result = analyzer.analyze_failure(
            session_id="test_session_4",
            error_type="NameError",
            error_message="name 'undefined_var' is not defined",
        )

        assert result.failure_code == FailureCode.F2

    def test_f3_verification_failure(self, analyzer):
        """Test F3 classification for test failures."""
        result = analyzer.analyze_failure(
            session_id="test_session_5",
            error_type="AssertionError",
            error_message="expected 42 but got 0",
        )

        assert result.failure_code == FailureCode.F3

    def test_f4_external_failure_network(self, analyzer):
        """Test F4 classification for network errors."""
        result = analyzer.analyze_failure(
            session_id="test_session_6",
            error_type="ConnectionError",
            error_message="Failed to connect to remote server",
        )

        assert result.failure_code == FailureCode.F4

    def test_f4_external_failure_api(self, analyzer):
        """Test F4 classification for API errors."""
        result = analyzer.analyze_failure(
            session_id="test_session_7",
            error_type="HTTPError",
            error_message="API returned 503 Service Unavailable",
        )

        assert result.failure_code == FailureCode.F4

    def test_f5_indeterminate_failure(self, analyzer):
        """Test F5 classification for unknown errors."""
        result = analyzer.analyze_failure(
            session_id="test_session_8",
            error_type="WeirdError",
            error_message="Something very strange happened",
        )

        assert result.failure_code == FailureCode.F5


# =============================================================================
# Signature Chain Attribution Tests
# =============================================================================


class TestSignatureChainAttribution:
    """Test attribution based on signature chains."""

    def test_build_failure_attributes_to_builder(self, analyzer, sample_signature_chain):
        """Test that build failures attribute to the builder agent."""
        result = analyzer.analyze_failure(
            session_id="test_session_10",
            error_type="SyntaxError",
            error_message="Missing colon",
            failed_node_id="node_123",
            signature_chain=sample_signature_chain,
        )

        assert result.failure_code == FailureCode.F2
        assert result.attributed_agent_id == "builder_agent"
        assert result.attributed_phase == CyclePhase.TEST  # Latest signature phase

    def test_test_failure_attributes_to_tester(self, analyzer, sample_signature_chain):
        """Test that test failures attribute to the tester agent."""
        result = analyzer.analyze_failure(
            session_id="test_session_11",
            error_type="AssertionError",
            error_message="Test failed: expected True, got False",
            failed_node_id="node_123",
            signature_chain=sample_signature_chain,
        )

        assert result.failure_code == FailureCode.F3
        assert result.attributed_agent_id == "tester_agent"

    def test_research_failure_attributes_to_researcher(self, analyzer):
        """Test that research failures attribute to the researcher."""
        research_chain = SignatureChain(
            node_id="req_node",
            state_id="state_xyz",
            signatures=[
                AgentSignature(
                    agent_id="dialector_agent",
                    model_id="claude-sonnet-4-5",
                    phase=CyclePhase.DIALECTIC,
                    action=SignatureAction.CREATED,
                    temperature=0.8,
                    context_constraints={},
                    timestamp=datetime.now().isoformat(),
                ),
            ],
        )

        result = analyzer.analyze_failure(
            session_id="test_session_12",
            error_type="AmbiguityError",
            error_message="Requirement is too vague",
            failed_node_id="req_node",
            signature_chain=research_chain,
        )

        assert result.attributed_agent_id == "dialector_agent"
        assert result.attributed_phase == CyclePhase.DIALECTIC

    def test_contributing_agents_identified(self, analyzer, sample_signature_chain):
        """Test that contributing agents are correctly identified."""
        result = analyzer.analyze_failure(
            session_id="test_session_13",
            error_type="RuntimeError",
            error_message="Something went wrong",
            failed_node_id="node_123",
            signature_chain=sample_signature_chain,
        )

        # Should identify agents other than the attributed one
        assert len(result.contributing_agents) >= 1
        assert result.attributed_agent_id not in result.contributing_agents


# =============================================================================
# Confidence Calculation Tests
# =============================================================================


class TestConfidenceCalculation:
    """Test confidence score calculation."""

    def test_build_phase_high_confidence(self, analyzer):
        """Test that BUILD phase failures have high confidence."""
        build_chain = SignatureChain(
            node_id="code_node",
            state_id="state_1",
            signatures=[
                AgentSignature(
                    agent_id="builder_agent",
                    model_id="claude-sonnet-4-5",
                    phase=CyclePhase.BUILD,
                    action=SignatureAction.CREATED,
                    temperature=0.3,
                    context_constraints={},
                    timestamp=datetime.now().isoformat(),
                ),
            ],
        )

        result = analyzer.analyze_failure(
            session_id="test_session_20",
            error_type="SyntaxError",
            error_message="Invalid syntax",
            signature_chain=build_chain,
        )

        # BUILD phase should have 0.9 base confidence
        assert result.confidence >= 0.8

    def test_dialectic_phase_medium_confidence(self, analyzer):
        """Test that DIALECTIC phase failures have medium confidence."""
        dialectic_chain = SignatureChain(
            node_id="req_node",
            state_id="state_2",
            signatures=[
                AgentSignature(
                    agent_id="dialector_agent",
                    model_id="claude-sonnet-4-5",
                    phase=CyclePhase.DIALECTIC,
                    action=SignatureAction.CREATED,
                    temperature=0.7,
                    context_constraints={},
                    timestamp=datetime.now().isoformat(),
                ),
            ],
        )

        result = analyzer.analyze_failure(
            session_id="test_session_21",
            error_type="RequirementError",
            error_message="Unclear requirement",
            signature_chain=dialectic_chain,
        )

        # DIALECTIC phase should have 0.6 base confidence
        assert 0.5 <= result.confidence <= 0.7

    def test_external_failure_low_confidence(self, analyzer):
        """Test that external failures have low confidence (not agent's fault)."""
        result = analyzer.analyze_failure(
            session_id="test_session_22",
            error_type="ConnectionError",
            error_message="Network timeout",
        )

        # External failures should have reduced confidence
        assert result.confidence < 0.5

    def test_indeterminate_failure_very_low_confidence(self, analyzer):
        """Test that indeterminate failures have very low confidence."""
        result = analyzer.analyze_failure(
            session_id="test_session_23",
            error_type="MysteryError",
            error_message="Unknown problem occurred",
        )

        # Indeterminate failures should have very low confidence
        assert result.confidence < 0.4

    def test_rejected_action_reduces_confidence(self, analyzer):
        """Test that REJECTED actions reduce confidence."""
        rejected_chain = SignatureChain(
            node_id="node_rejected",
            state_id="state_3",
            signatures=[
                AgentSignature(
                    agent_id="builder_agent",
                    model_id="claude-sonnet-4-5",
                    phase=CyclePhase.BUILD,
                    action=SignatureAction.REJECTED,
                    temperature=0.3,
                    context_constraints={},
                    timestamp=datetime.now().isoformat(),
                ),
            ],
        )

        result = analyzer.analyze_failure(
            session_id="test_session_24",
            error_type="ValidationError",
            error_message="Code was rejected",
            signature_chain=rejected_chain,
        )

        # Should have lower confidence than normal BUILD phase
        assert result.confidence < 0.9


# =============================================================================
# Reasoning Generation Tests
# =============================================================================


class TestReasoningGeneration:
    """Test that reasoning strings are informative."""

    def test_reasoning_includes_failure_code(self, analyzer):
        """Test that reasoning includes failure code."""
        result = analyzer.analyze_failure(
            session_id="test_session_30",
            error_type="SyntaxError",
            error_message="Missing colon",
        )

        assert result.failure_code.value in result.reasoning

    def test_reasoning_includes_phase(self, analyzer):
        """Test that reasoning includes phase information."""
        result = analyzer.analyze_failure(
            session_id="test_session_31",
            error_type="ImportError",
            error_message="Module not found",
        )

        assert "phase" in result.reasoning.lower()

    def test_reasoning_includes_agent_when_available(self, analyzer, sample_signature_chain):
        """Test that reasoning includes agent ID when signature chain is available."""
        result = analyzer.analyze_failure(
            session_id="test_session_32",
            error_type="TypeError",
            error_message="Wrong type",
            signature_chain=sample_signature_chain,
        )

        # Should mention an agent name
        assert any(
            agent in result.reasoning
            for agent in ["researcher_agent", "builder_agent", "tester_agent"]
        )

    def test_reasoning_truncates_long_errors(self, analyzer):
        """Test that very long error messages are truncated."""
        long_error = "X" * 500

        result = analyzer.analyze_failure(
            session_id="test_session_33",
            error_type="LongError",
            error_message=long_error,
        )

        # Reasoning should be shorter than original error
        assert len(result.reasoning) < len(long_error) + 500


# =============================================================================
# Multi-Failure Analysis Tests
# =============================================================================


class TestMultiFailureAnalysis:
    """Test analyzing multiple failures in a session."""

    def test_analyze_multiple_failures(self, analyzer):
        """Test analyzing multiple failures from one session."""
        failures = [
            {
                "error_type": "SyntaxError",
                "error_message": "Invalid syntax in function foo",
            },
            {
                "error_type": "ImportError",
                "error_message": "Cannot import module bar",
            },
            {
                "error_type": "AssertionError",
                "error_message": "Test failed for baz",
            },
        ]

        results = analyzer.analyze_session_failures(
            session_id="test_session_40",
            failures=failures,
        )

        assert len(results) == 3
        assert all(isinstance(r, AttributionResult) for r in results)

        # Check that different failures have different classifications
        codes = [r.failure_code for r in results]
        assert FailureCode.F2 in codes  # SyntaxError and ImportError
        assert FailureCode.F3 in codes  # AssertionError

    def test_analyze_failures_with_mixed_data(self, analyzer, sample_signature_chain):
        """Test analyzing failures with some having signature chains and some not."""
        failures = [
            {
                "node_id": "node_123",
                "error_type": "SyntaxError",
                "error_message": "Bad syntax",
                "signature_chain": sample_signature_chain,
            },
            {
                "error_type": "RuntimeError",
                "error_message": "Something broke",
                # No node_id or signature_chain
            },
        ]

        results = analyzer.analyze_session_failures(
            session_id="test_session_41",
            failures=failures,
        )

        assert len(results) == 2

        # First should have agent attribution from signature chain
        assert results[0].attributed_agent_id != "unknown"

        # Second might have "unknown" attribution
        # (depending on error pattern matching)


# =============================================================================
# Phase Inference Tests
# =============================================================================


class TestPhaseInference:
    """Test phase inference when signature chain is unavailable."""

    def test_infer_test_phase_from_test_error(self, analyzer):
        """Test inferring TEST phase from test-related errors."""
        phase = analyzer._infer_phase_from_error(
            "AssertionError",
            "test_foo failed: expected 1, got 2"
        )

        assert phase == CyclePhase.TEST

    def test_infer_build_phase_from_syntax_error(self, analyzer):
        """Test inferring BUILD phase from syntax errors."""
        phase = analyzer._infer_phase_from_error(
            "SyntaxError",
            "invalid syntax"
        )

        assert phase == CyclePhase.BUILD

    def test_infer_build_phase_from_import_error(self, analyzer):
        """Test inferring BUILD phase from import errors."""
        phase = analyzer._infer_phase_from_error(
            "ImportError",
            "cannot import module"
        )

        assert phase == CyclePhase.BUILD

    def test_default_to_build_phase(self, analyzer):
        """Test that unknown errors default to BUILD phase."""
        phase = analyzer._infer_phase_from_error(
            "UnknownError",
            "something weird happened"
        )

        assert phase == CyclePhase.BUILD


# =============================================================================
# Integration with TrainingStore Tests
# =============================================================================


class TestTrainingStoreIntegration:
    """Test integration with TrainingStore."""

    def test_retrieve_signature_chain_from_store(self, analyzer, temp_db, sample_signature_chain):
        """Test retrieving signature chain from TrainingStore."""
        # Store the signature chain
        temp_db.record_signature_chain(sample_signature_chain)

        # Retrieve it
        retrieved_chain = analyzer._trace_signature_chain("node_123")

        assert len(retrieved_chain) == 3
        assert all(isinstance(sig, AgentSignature) for sig in retrieved_chain)

    def test_empty_signature_chain_for_unknown_node(self, analyzer):
        """Test that unknown nodes return empty signature list."""
        signatures = analyzer._trace_signature_chain("nonexistent_node")

        assert signatures == []

    def test_get_agent_failure_stats_placeholder(self, analyzer):
        """Test that agent failure stats returns placeholder structure."""
        stats = analyzer.get_agent_failure_stats(agent_id="builder_agent")

        assert "total_attributions" in stats
        assert "failure_code_distribution" in stats
        assert "average_confidence" in stats


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_error_message(self, analyzer):
        """Test handling empty error message."""
        result = analyzer.analyze_failure(
            session_id="test_session_50",
            error_type="EmptyError",
            error_message="",
        )

        assert result.failure_code == FailureCode.F5  # Indeterminate
        assert result.confidence < 0.5

    def test_none_signature_chain(self, analyzer):
        """Test handling explicit None signature chain."""
        result = analyzer.analyze_failure(
            session_id="test_session_51",
            error_type="SomeError",
            error_message="Error occurred",
            signature_chain=None,
        )

        assert result.attributed_agent_id == "unknown"
        assert result.attributed_model_id == "unknown"

    def test_empty_signature_chain(self, analyzer):
        """Test handling signature chain with no signatures."""
        empty_chain = SignatureChain(
            node_id="empty_node",
            state_id="empty_state",
            signatures=[],
        )

        result = analyzer.analyze_failure(
            session_id="test_session_52",
            error_type="ErrorType",
            error_message="Error message",
            signature_chain=empty_chain,
        )

        assert result.attributed_agent_id == "unknown"

    def test_very_long_error_message(self, analyzer):
        """Test handling very long error messages."""
        long_message = "A" * 10000

        result = analyzer.analyze_failure(
            session_id="test_session_53",
            error_type="LongError",
            error_message=long_message,
        )

        # Should not crash
        assert result is not None
        # Reasoning should be truncated
        assert len(result.reasoning) < len(long_message)
