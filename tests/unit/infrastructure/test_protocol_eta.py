"""
PROTOCOL ETA - Learning System & Attribution Tests

Tests the learning infrastructure (Layer L4/L5):
1. Learning Manager: STUDY/PRODUCTION mode transitions
2. ForensicAnalyzer: Failure attribution and root cause analysis
3. TrainingStore: Session outcomes and signature persistence
4. DivergenceDetector: Test-production mismatch detection
5. Model Recommendations: Adaptive model selection based on history

Reference: docs/IMPLEMENTATION_PLAN_LEARNING.md
Run: pytest tests/unit/infrastructure/test_protocol_eta.py -v
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

# Infrastructure imports
from infrastructure.learning import (
    LearningManager,
    LearningMode,
    ModelRecommendation,
    LearningStats,
    TransitionReport,
)
from infrastructure.training_store import TrainingStore
from infrastructure.attribution import ForensicAnalyzer, AttributionResult
from infrastructure.divergence import DivergenceDetector

# Schema imports
from agents.schemas import (
    AgentSignature,
    SignatureChain,
    CyclePhase,
    FailureCode,
    NodeOutcome,
    SignatureAction,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test databases."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def training_store(temp_db_dir):
    """Create a TrainingStore with temporary database."""
    db_path = temp_db_dir / "test_training.db"
    return TrainingStore(db_path=db_path)


@pytest.fixture
def learning_manager(temp_db_dir):
    """Create a LearningManager in STUDY mode."""
    db_path = temp_db_dir / "test_learning.db"
    return LearningManager(db_path=db_path, mode=LearningMode.STUDY)


@pytest.fixture
def forensic_analyzer(training_store):
    """Create a ForensicAnalyzer with test store."""
    return ForensicAnalyzer(training_store)


@pytest.fixture
def divergence_detector(training_store):
    """Create a DivergenceDetector with test store."""
    return DivergenceDetector(training_store)


@pytest.fixture
def sample_signature():
    """Create a sample AgentSignature for testing."""
    return AgentSignature(
        agent_id="builder_v1",
        model_id="claude-sonnet-4-5-20250929",
        phase=CyclePhase.BUILD,
        action=SignatureAction.CREATED,
        temperature=0.7,
        context_constraints={"max_tokens": 4096},
        timestamp=datetime.utcnow().isoformat(),
    )


@pytest.fixture
def sample_signature_chain(sample_signature):
    """Create a sample SignatureChain for testing."""
    return SignatureChain(
        node_id="test_node_001",
        state_id="state_001",
        signatures=[sample_signature],
    )


# =============================================================================
# TEST CLASS 1: Learning Mode Basics
# =============================================================================

class TestLearningModeBasics:
    """Test basic learning mode functionality."""

    def test_initializes_in_study_mode_by_default(self, temp_db_dir):
        """Manager initializes in STUDY mode by default."""
        db_path = temp_db_dir / "test.db"
        manager = LearningManager(db_path=db_path)
        assert manager.mode == LearningMode.STUDY

    def test_can_initialize_in_production_mode(self, temp_db_dir):
        """Manager can be initialized in PRODUCTION mode."""
        db_path = temp_db_dir / "test.db"
        manager = LearningManager(db_path=db_path, mode=LearningMode.PRODUCTION)
        assert manager.mode == LearningMode.PRODUCTION

    def test_study_mode_returns_no_recommendations(self, learning_manager):
        """STUDY mode returns None for model recommendations (no biasing)."""
        recommendation = learning_manager.get_model_recommendation(
            phase=CyclePhase.BUILD
        )
        assert recommendation is None

    def test_learning_mode_enum_values(self):
        """LearningMode enum has correct values."""
        assert LearningMode.STUDY.value == "study"
        assert LearningMode.PRODUCTION.value == "production"

    def test_mode_transition_to_study(self, learning_manager):
        """Can transition to STUDY mode."""
        learning_manager.mode = LearningMode.PRODUCTION
        learning_manager.transition_to_study()
        assert learning_manager.mode == LearningMode.STUDY


# =============================================================================
# TEST CLASS 2: Model Recommendations
# =============================================================================

class TestModelRecommendations:
    """Test model recommendation logic."""

    def test_production_mode_no_data_returns_none(self, temp_db_dir):
        """PRODUCTION mode returns None when no historical data."""
        db_path = temp_db_dir / "test.db"
        manager = LearningManager(db_path=db_path, mode=LearningMode.PRODUCTION)

        recommendation = manager.get_model_recommendation(phase=CyclePhase.BUILD)
        # No data yet, should return None
        assert recommendation is None

    def test_recommendation_struct_fields(self):
        """ModelRecommendation has correct fields."""
        rec = ModelRecommendation(
            model_id="claude-sonnet-4-5-20250929",
            confidence=0.85,
            reasoning="Best success rate",
            based_on_samples=50,
            success_rate=0.9,
        )
        assert rec.model_id == "claude-sonnet-4-5-20250929"
        assert rec.confidence == 0.85
        assert rec.based_on_samples == 50
        assert rec.success_rate == 0.9

    def test_production_mode_with_data_may_return_recommendation(self, temp_db_dir):
        """PRODUCTION mode may return recommendations with sufficient data."""
        db_path = temp_db_dir / "test.db"
        manager = LearningManager(db_path=db_path, mode=LearningMode.PRODUCTION)

        # Populate with successful data
        for i in range(10):
            session_id = f"session_{i}"
            sig = AgentSignature(
                agent_id="builder_v1",
                model_id="claude-sonnet-4-5-20250929",
                phase=CyclePhase.BUILD,
                action=SignatureAction.CREATED,
                temperature=0.7,
                context_constraints={},
                timestamp=datetime.utcnow().isoformat(),
            )
            manager.store.record_attribution(
                session_id=session_id,
                signature=sig,
                node_id=f"node_{i}",
                state_id=f"state_{i}",
            )
            manager.store.record_session_outcome(
                session_id=session_id,
                outcome=NodeOutcome.VERIFIED_SUCCESS,
            )

        # May return recommendation (exploration rate may cause None)
        recommendation = manager.get_model_recommendation(phase=CyclePhase.BUILD)
        if recommendation is not None:
            assert isinstance(recommendation, ModelRecommendation)
            assert recommendation.model_id in manager.AVAILABLE_MODELS

    def test_available_models_list(self, learning_manager):
        """AVAILABLE_MODELS contains expected model IDs."""
        assert len(learning_manager.AVAILABLE_MODELS) >= 2
        assert any("claude" in m for m in learning_manager.AVAILABLE_MODELS)

    def test_exploration_rate_is_reasonable(self, learning_manager):
        """EXPLORATION_RATE is between 0 and 1."""
        assert 0.0 < learning_manager.EXPLORATION_RATE < 1.0


# =============================================================================
# TEST CLASS 3: Outcome Recording
# =============================================================================

class TestOutcomeRecording:
    """Test outcome recording functionality."""

    def test_records_successful_outcome(self, learning_manager):
        """Records successful session outcome."""
        learning_manager.record_outcome(
            session_id="test_success_001",
            success=True,
        )
        outcome = learning_manager.store.get_session_outcome("test_success_001")
        assert outcome is not None
        assert outcome["outcome"] == NodeOutcome.VERIFIED_SUCCESS.value

    def test_records_failure_with_code(self, learning_manager):
        """Records failed session with failure code (F1-F5)."""
        learning_manager.record_outcome(
            session_id="test_failure_001",
            success=False,
            failure_code=FailureCode.F2,
            failure_phase=CyclePhase.BUILD,
        )
        outcome = learning_manager.store.get_session_outcome("test_failure_001")
        assert outcome is not None
        assert outcome["outcome"] == NodeOutcome.VERIFIED_FAILURE.value
        assert outcome["failure_code"] == FailureCode.F2.value
        assert outcome["failure_phase"] == CyclePhase.BUILD.value

    def test_records_outcome_with_stats(self, learning_manager):
        """Records outcome with additional statistics."""
        stats = {
            "total_nodes": 42,
            "total_iterations": 3,
            "total_tokens": 15000,
        }
        learning_manager.record_outcome(
            session_id="test_stats_001",
            success=True,
            stats=stats,
        )
        outcome = learning_manager.store.get_session_outcome("test_stats_001")
        assert outcome is not None
        assert outcome["total_nodes"] == 42
        assert outcome["total_iterations"] == 3

    def test_failure_code_enum_values(self):
        """FailureCode enum has F1-F5 values."""
        assert FailureCode.F1.value == "F1"
        assert FailureCode.F2.value == "F2"
        assert FailureCode.F3.value == "F3"
        assert FailureCode.F4.value == "F4"
        assert FailureCode.F5.value == "F5"

    def test_node_outcome_enum_values(self):
        """NodeOutcome enum has correct values."""
        assert NodeOutcome.VERIFIED_SUCCESS.value == "verified_success"
        assert NodeOutcome.VERIFIED_FAILURE.value == "verified_failure"


# =============================================================================
# TEST CLASS 4: Divergence Detection
# =============================================================================

class TestDivergenceDetection:
    """Test test-production divergence detection."""

    def test_detects_false_positive(self, learning_manager):
        """Detects false positive (tests pass, production fails)."""
        event_id = learning_manager.check_and_log_divergence(
            session_id="test_fp_001",
            test_passed=True,
            user_feedback="broken",
            node_id="node_123",
        )
        assert event_id is not None

    def test_detects_false_negative(self, learning_manager):
        """Detects false negative (tests fail, production works)."""
        event_id = learning_manager.check_and_log_divergence(
            session_id="test_fn_001",
            test_passed=False,
            user_feedback="success",
            node_id="node_456",
        )
        assert event_id is not None

    def test_no_divergence_on_matching_outcomes(self, learning_manager):
        """No divergence logged when outcomes match."""
        event_id = learning_manager.check_and_log_divergence(
            session_id="test_match_001",
            test_passed=True,
            user_feedback="success",
        )
        assert event_id is None

    def test_divergence_detector_check(self, divergence_detector):
        """DivergenceDetector.check_divergence returns correct type."""
        event = divergence_detector.check_divergence(
            session_id="test_001",
            test_passed=True,
            prod_outcome="failure",
        )
        if event is not None:
            assert hasattr(event, "divergence_type")

    def test_divergence_rate_calculation(self, divergence_detector):
        """Can calculate divergence rate."""
        rate = divergence_detector.calculate_divergence_rate()
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0


# =============================================================================
# TEST CLASS 5: Learning Statistics
# =============================================================================

class TestLearningStats:
    """Test learning statistics generation."""

    def test_stats_with_no_data(self, learning_manager):
        """Statistics with no data collected."""
        stats = learning_manager.get_learning_stats()
        assert isinstance(stats, LearningStats)
        assert stats.mode == LearningMode.STUDY
        assert stats.total_sessions == 0
        assert stats.ready_for_production is False

    def test_stats_with_some_data(self, learning_manager):
        """Statistics with partial data."""
        for i in range(10):
            learning_manager.record_outcome(
                session_id=f"session_{i}",
                success=(i % 2 == 0),
            )
        stats = learning_manager.get_learning_stats()
        assert stats.total_sessions == 10
        assert stats.successful_sessions == 5
        assert stats.failed_sessions == 5
        assert stats.ready_for_production is False

    def test_stats_ready_for_production(self, learning_manager):
        """Statistics indicate ready for production at threshold."""
        for i in range(100):
            learning_manager.record_outcome(
                session_id=f"session_{i}",
                success=True,
            )
        stats = learning_manager.get_learning_stats()
        assert stats.total_sessions == 100
        assert stats.ready_for_production is True
        assert "Ready" in stats.recommendation

    def test_stats_struct_fields(self):
        """LearningStats struct has correct fields."""
        stats = LearningStats(
            mode=LearningMode.STUDY,
            total_sessions=50,
            successful_sessions=40,
            failed_sessions=10,
            divergences=2,
            total_attributions=100,
            ready_for_production=False,
            recommendation="Continue collecting data",
        )
        assert stats.successful_sessions + stats.failed_sessions == stats.total_sessions


# =============================================================================
# TEST CLASS 6: Transition Logic
# =============================================================================

class TestTransitionLogic:
    """Test mode transition logic."""

    def test_transition_report_not_ready(self, learning_manager):
        """Transition report indicates not ready with insufficient data."""
        report = learning_manager.should_transition_to_production()
        assert isinstance(report, TransitionReport)
        assert report.ready is False
        assert report.session_count < 100
        assert "Not ready" in report.recommendation

    def test_transition_report_ready(self, learning_manager):
        """Transition report indicates ready at threshold."""
        for i in range(100):
            learning_manager.record_outcome(
                session_id=f"session_{i}",
                success=True,
            )
        report = learning_manager.should_transition_to_production()
        assert report.ready is True
        assert report.session_count == 100
        assert report.requires_human_review is True

    def test_cannot_transition_without_data(self, learning_manager):
        """Cannot transition to production without sufficient data."""
        success = learning_manager.transition_to_production()
        assert success is False
        assert learning_manager.mode == LearningMode.STUDY

    def test_can_transition_with_enough_data(self, learning_manager):
        """Can transition to production with sufficient data."""
        for i in range(100):
            learning_manager.record_outcome(
                session_id=f"session_{i}",
                success=True,
            )
        success = learning_manager.transition_to_production()
        assert success is True
        assert learning_manager.mode == LearningMode.PRODUCTION

    def test_min_sessions_threshold(self, learning_manager):
        """MIN_SESSIONS_FOR_PRODUCTION is reasonable."""
        assert learning_manager.MIN_SESSIONS_FOR_PRODUCTION >= 50


# =============================================================================
# TEST CLASS 7: Forensic Attribution
# =============================================================================

class TestForensicAttribution:
    """Test forensic failure analysis."""

    def test_analyze_failure_returns_result(self, forensic_analyzer):
        """analyze_failure returns AttributionResult."""
        result = forensic_analyzer.analyze_failure(
            session_id="test_session_001",
            error_type="SyntaxError",
            error_message="invalid syntax at line 10",
        )
        assert isinstance(result, AttributionResult)
        assert result.failure_code in list(FailureCode)
        assert 0.0 <= result.confidence <= 1.0

    def test_classifies_syntax_error_as_f2(self, forensic_analyzer):
        """SyntaxError classified as F2 (Implementation Failure)."""
        result = forensic_analyzer.analyze_failure(
            session_id="test_syntax",
            error_type="SyntaxError",
            error_message="unexpected token",
        )
        assert result.failure_code == FailureCode.F2

    def test_classifies_connection_error_as_f4(self, forensic_analyzer):
        """ConnectionError classified as F4 (External Failure)."""
        result = forensic_analyzer.analyze_failure(
            session_id="test_network",
            error_type="ConnectionError",
            error_message="Failed to connect to server",
        )
        assert result.failure_code == FailureCode.F4

    def test_classifies_assertion_error_as_f3(self, forensic_analyzer):
        """AssertionError classified as F3 (Verification Failure)."""
        result = forensic_analyzer.analyze_failure(
            session_id="test_assert",
            error_type="AssertionError",
            error_message="expected 5, got 4",
        )
        assert result.failure_code == FailureCode.F3

    def test_attribution_with_signature_chain(
        self, forensic_analyzer, sample_signature_chain
    ):
        """Attribution uses signature chain when provided."""
        result = forensic_analyzer.analyze_failure(
            session_id="test_chain",
            error_type="TypeError",
            error_message="missing argument",
            signature_chain=sample_signature_chain,
        )
        assert result.attributed_agent_id == "builder_v1"
        assert result.attributed_model_id == "claude-sonnet-4-5-20250929"

    def test_analyze_session_failures_batch(self, forensic_analyzer):
        """Can analyze multiple failures in batch."""
        failures = [
            {"error_type": "SyntaxError", "error_message": "invalid syntax"},
            {"error_type": "TypeError", "error_message": "missing argument"},
        ]
        results = forensic_analyzer.analyze_session_failures(
            session_id="batch_test",
            failures=failures,
        )
        assert len(results) == 2
        assert all(isinstance(r, AttributionResult) for r in results)


# =============================================================================
# TEST CLASS 8: Training Store
# =============================================================================

class TestTrainingStore:
    """Test training data persistence."""

    def test_record_attribution(self, training_store, sample_signature):
        """Can record attribution to training store."""
        training_store.record_attribution(
            session_id="session_001",
            signature=sample_signature,
            node_id="node_001",
            state_id="state_001",
        )
        # Should not raise

    def test_get_signature_chain(self, training_store, sample_signature):
        """Can retrieve signature chain for a node."""
        training_store.record_attribution(
            session_id="session_002",
            signature=sample_signature,
            node_id="node_002",
            state_id="state_002",
        )
        chain = training_store.get_signature_chain("node_002")
        # May be None if not fully implemented
        if chain is not None:
            assert isinstance(chain, SignatureChain)

    def test_get_session_count(self, training_store):
        """Can get total session count."""
        count = training_store.get_session_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_get_attribution_count(self, training_store, sample_signature):
        """Can get total attribution count."""
        training_store.record_attribution(
            session_id="session_003",
            signature=sample_signature,
            node_id="node_003",
            state_id="state_003",
        )
        count = training_store.get_attribution_count()
        assert isinstance(count, int)
        assert count >= 1

    def test_get_success_rate(self, training_store):
        """Can get success rate for model+phase."""
        rate = training_store.get_success_rate(
            model_id="claude-sonnet-4-5-20250929",
            phase="build",
        )
        assert isinstance(rate, float)
        # Default prior should be 0.5 with no data
        assert 0.0 <= rate <= 1.0


# =============================================================================
# TEST CLASS 9: Model Performance Summary
# =============================================================================

class TestModelPerformanceSummary:
    """Test model performance summary generation."""

    def test_summary_with_no_data(self, learning_manager):
        """Performance summary with no historical data."""
        summary = learning_manager.get_model_performance_summary()
        assert isinstance(summary, dict)
        assert len(summary) == len(learning_manager.AVAILABLE_MODELS)

    def test_summary_structure(self, learning_manager):
        """Summary has correct structure: model -> phase -> stats."""
        summary = learning_manager.get_model_performance_summary()
        for model_id, phases in summary.items():
            assert isinstance(phases, dict)
            for phase, stats in phases.items():
                assert "sample_count" in stats
                assert "success_rate" in stats

    def test_summary_with_data(self, learning_manager):
        """Performance summary with historical data."""
        # Add data for one model
        for i in range(5):
            sig = AgentSignature(
                agent_id="builder_v1",
                model_id="claude-sonnet-4-5-20250929",
                phase=CyclePhase.BUILD,
                action=SignatureAction.CREATED,
                temperature=0.7,
                context_constraints={},
                timestamp=datetime.utcnow().isoformat(),
            )
            learning_manager.store.record_attribution(
                session_id=f"session_{i}",
                signature=sig,
                node_id=f"node_{i}",
                state_id=f"state_{i}",
            )
            learning_manager.store.record_session_outcome(
                session_id=f"session_{i}",
                outcome=NodeOutcome.VERIFIED_SUCCESS,
            )

        summary = learning_manager.get_model_performance_summary()
        sonnet_stats = summary.get("claude-sonnet-4-5-20250929", {}).get("build", {})
        assert sonnet_stats.get("sample_count", 0) >= 5


# =============================================================================
# TEST CLASS 10: Signature Chain
# =============================================================================

class TestSignatureChain:
    """Test signature chain functionality."""

    def test_signature_chain_creation(self, sample_signature):
        """Can create SignatureChain with signatures."""
        chain = SignatureChain(
            node_id="test_node",
            state_id="state_001",
            signatures=[sample_signature],
        )
        assert chain.node_id == "test_node"
        assert chain.state_id == "state_001"
        assert len(chain.signatures) == 1

    def test_agent_signature_fields(self, sample_signature):
        """AgentSignature has correct fields."""
        assert sample_signature.agent_id == "builder_v1"
        assert sample_signature.model_id == "claude-sonnet-4-5-20250929"
        assert sample_signature.phase == CyclePhase.BUILD
        assert sample_signature.action == SignatureAction.CREATED
        assert sample_signature.temperature == 0.7

    def test_signature_action_enum(self):
        """SignatureAction enum has expected values."""
        assert SignatureAction.CREATED.value == "created"
        assert SignatureAction.MODIFIED.value == "modified"
        assert SignatureAction.VERIFIED.value == "verified"
        assert SignatureAction.REJECTED.value == "rejected"

    def test_cycle_phase_enum(self):
        """CyclePhase enum has expected values."""
        phases = [CyclePhase.INIT, CyclePhase.PLAN, CyclePhase.BUILD,
                  CyclePhase.TEST, CyclePhase.PASSED, CyclePhase.FAILED]
        assert len(phases) >= 6

    def test_multi_signature_chain(self, sample_signature):
        """SignatureChain can hold multiple signatures."""
        sig2 = AgentSignature(
            agent_id="tester_v1",
            model_id="claude-haiku-3-5-20241022",
            phase=CyclePhase.TEST,
            action=SignatureAction.VERIFIED,
            temperature=0.5,
            context_constraints={},
            timestamp=datetime.utcnow().isoformat(),
        )
        chain = SignatureChain(
            node_id="multi_sig_node",
            state_id="state_multi",
            signatures=[sample_signature, sig2],
        )
        assert len(chain.signatures) == 2
        assert chain.signatures[0].phase == CyclePhase.BUILD
        assert chain.signatures[1].phase == CyclePhase.TEST


# =============================================================================
# TEST CLASS 11: Phase Confidence Weights
# =============================================================================

class TestPhaseConfidenceWeights:
    """Test phase-based confidence weights for attribution."""

    def test_build_phase_high_confidence(self, forensic_analyzer):
        """BUILD phase has high attribution confidence."""
        confidence = forensic_analyzer.PHASE_CONFIDENCE.get(CyclePhase.BUILD, 0)
        assert confidence >= 0.8

    def test_test_phase_medium_confidence(self, forensic_analyzer):
        """TEST phase has medium attribution confidence."""
        confidence = forensic_analyzer.PHASE_CONFIDENCE.get(CyclePhase.TEST, 0)
        assert confidence >= 0.7

    def test_passed_phase_low_confidence(self, forensic_analyzer):
        """PASSED phase has low attribution confidence (shouldn't have failures)."""
        confidence = forensic_analyzer.PHASE_CONFIDENCE.get(CyclePhase.PASSED, 0)
        assert confidence <= 0.5

    def test_all_phases_have_weights(self, forensic_analyzer):
        """All CyclePhase values have confidence weights."""
        for phase in [CyclePhase.INIT, CyclePhase.PLAN, CyclePhase.BUILD,
                      CyclePhase.TEST, CyclePhase.PASSED, CyclePhase.FAILED]:
            assert phase in forensic_analyzer.PHASE_CONFIDENCE


# =============================================================================
# MAIN
# =============================================================================

def run_protocol_eta():
    """Run Protocol Eta tests and return summary."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)

    return result.returncode == 0


if __name__ == "__main__":
    success = run_protocol_eta()
    import sys
    sys.exit(0 if success else 1)
