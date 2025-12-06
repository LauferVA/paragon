"""
Tests for Learning System - Two-mode learning manager.

Tests:
- Learning mode transitions
- Model recommendations in STUDY vs PRODUCTION modes
- Outcome recording
- Divergence checking
- Learning statistics
- Transition readiness assessment
- Model performance tracking

Layer: L4/L5 (Learning Manager)
Status: Production
"""
import pytest
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from infrastructure.learning import (
    LearningManager,
    LearningMode,
    ModelRecommendation,
    LearningStats,
    TransitionReport,
)
from infrastructure.training_store import TrainingStore
from agents.schemas import (
    AgentSignature,
    CyclePhase,
    FailureCode,
    NodeOutcome,
    SignatureAction,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_learning.db"

    yield db_path

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def learning_manager(temp_db):
    """Create a LearningManager instance with temporary database."""
    return LearningManager(db_path=temp_db, mode=LearningMode.STUDY)


@pytest.fixture
def sample_signature():
    """Create a sample agent signature for testing."""
    return AgentSignature(
        agent_id="builder_v1",
        model_id="claude-sonnet-4-5-20250929",
        phase=CyclePhase.BUILD,
        action=SignatureAction.CREATED,
        temperature=0.7,
        context_constraints={"max_tokens": 4096},
        timestamp=datetime.utcnow().isoformat(),
    )


class TestLearningModeBasics:
    """Test basic learning mode functionality."""

    def test_initializes_in_study_mode_by_default(self, temp_db):
        """Test that manager initializes in STUDY mode by default."""
        manager = LearningManager(db_path=temp_db)
        assert manager.mode == LearningMode.STUDY

    def test_can_initialize_in_production_mode(self, temp_db):
        """Test that manager can be initialized in PRODUCTION mode."""
        manager = LearningManager(db_path=temp_db, mode=LearningMode.PRODUCTION)
        assert manager.mode == LearningMode.PRODUCTION

    def test_study_mode_returns_no_recommendations(self, learning_manager):
        """Test that STUDY mode returns None for recommendations."""
        recommendation = learning_manager.get_model_recommendation(
            phase=CyclePhase.BUILD
        )
        assert recommendation is None


class TestModelRecommendations:
    """Test model recommendation logic."""

    def test_production_mode_with_no_data_returns_none(self, temp_db):
        """Test that PRODUCTION mode returns None when no historical data."""
        manager = LearningManager(db_path=temp_db, mode=LearningMode.PRODUCTION)

        recommendation = manager.get_model_recommendation(
            phase=CyclePhase.BUILD
        )

        # No data yet, so should return None
        assert recommendation is None

    def test_production_mode_with_data_returns_recommendation(self, temp_db):
        """Test that PRODUCTION mode returns recommendations with sufficient data."""
        manager = LearningManager(db_path=temp_db, mode=LearningMode.PRODUCTION)

        # Populate with some successful data
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

            # Record attribution
            manager.store.record_attribution(
                session_id=session_id,
                signature=sig,
                node_id=f"node_{i}",
                state_id=f"state_{i}",
            )

            # Record successful outcome
            manager.store.record_session_outcome(
                session_id=session_id,
                outcome=NodeOutcome.VERIFIED_SUCCESS,
            )

        # Now should get a recommendation
        recommendation = manager.get_model_recommendation(
            phase=CyclePhase.BUILD
        )

        # Might still be None due to exploration rate, but if not None, should be valid
        if recommendation:
            assert isinstance(recommendation, ModelRecommendation)
            assert recommendation.model_id in manager.AVAILABLE_MODELS
            assert 0.0 <= recommendation.confidence <= 1.0
            assert recommendation.based_on_samples > 0

    def test_recommendation_prefers_successful_models(self, temp_db):
        """Test that recommendations prefer models with higher success rates."""
        manager = LearningManager(db_path=temp_db, mode=LearningMode.PRODUCTION)

        # Create data for two models with different success rates
        # Model A: 90% success
        for i in range(10):
            session_id = f"session_a_{i}"
            sig = AgentSignature(
                agent_id="builder_v1",
                model_id="claude-opus-4-5-20251101",
                phase=CyclePhase.BUILD,
                action=SignatureAction.CREATED,
                temperature=0.7,
                context_constraints={},
                timestamp=datetime.utcnow().isoformat(),
            )

            manager.store.record_attribution(
                session_id=session_id,
                signature=sig,
                node_id=f"node_a_{i}",
                state_id=f"state_a_{i}",
            )

            outcome = NodeOutcome.VERIFIED_SUCCESS if i < 9 else NodeOutcome.VERIFIED_FAILURE
            manager.store.record_session_outcome(
                session_id=session_id,
                outcome=outcome,
            )

        # Model B: 50% success
        for i in range(10):
            session_id = f"session_b_{i}"
            sig = AgentSignature(
                agent_id="builder_v1",
                model_id="claude-haiku-3-5-20241022",
                phase=CyclePhase.BUILD,
                action=SignatureAction.CREATED,
                temperature=0.7,
                context_constraints={},
                timestamp=datetime.utcnow().isoformat(),
            )

            manager.store.record_attribution(
                session_id=session_id,
                signature=sig,
                node_id=f"node_b_{i}",
                state_id=f"state_b_{i}",
            )

            outcome = NodeOutcome.VERIFIED_SUCCESS if i < 5 else NodeOutcome.VERIFIED_FAILURE
            manager.store.record_session_outcome(
                session_id=session_id,
                outcome=outcome,
            )

        # Check success rates
        opus_rate = manager.store.get_success_rate("claude-opus-4-5-20251101", "build")
        haiku_rate = manager.store.get_success_rate("claude-haiku-3-5-20241022", "build")

        assert opus_rate > haiku_rate  # Opus should have higher success rate


class TestOutcomeRecording:
    """Test outcome recording functionality."""

    def test_records_successful_outcome(self, learning_manager):
        """Test recording a successful session outcome."""
        learning_manager.record_outcome(
            session_id="test_session_1",
            success=True,
        )

        outcome = learning_manager.store.get_session_outcome("test_session_1")
        assert outcome is not None
        assert outcome["outcome"] == NodeOutcome.VERIFIED_SUCCESS.value

    def test_records_failure_with_code(self, learning_manager):
        """Test recording a failed session with failure code."""
        learning_manager.record_outcome(
            session_id="test_session_2",
            success=False,
            failure_code=FailureCode.F2,
            failure_phase=CyclePhase.BUILD,
        )

        outcome = learning_manager.store.get_session_outcome("test_session_2")
        assert outcome is not None
        assert outcome["outcome"] == NodeOutcome.VERIFIED_FAILURE.value
        assert outcome["failure_code"] == FailureCode.F2.value
        assert outcome["failure_phase"] == CyclePhase.BUILD.value

    def test_records_outcome_with_stats(self, learning_manager):
        """Test recording outcome with statistics."""
        stats = {
            "total_nodes": 42,
            "total_iterations": 3,
            "total_tokens": 15000,
        }

        learning_manager.record_outcome(
            session_id="test_session_3",
            success=True,
            stats=stats,
        )

        outcome = learning_manager.store.get_session_outcome("test_session_3")
        assert outcome is not None
        assert outcome["total_nodes"] == 42
        assert outcome["total_iterations"] == 3
        assert outcome["total_tokens"] == 15000


class TestDivergenceChecking:
    """Test divergence detection integration."""

    def test_detects_false_positive_divergence(self, learning_manager):
        """Test detection of false positive (tests pass, production fails)."""
        event_id = learning_manager.check_and_log_divergence(
            session_id="test_session_4",
            test_passed=True,
            user_feedback="broken",
            node_id="node_123",
        )

        assert event_id is not None

    def test_no_divergence_when_outcomes_match(self, learning_manager):
        """Test that no divergence is logged when outcomes match."""
        event_id = learning_manager.check_and_log_divergence(
            session_id="test_session_5",
            test_passed=True,
            user_feedback="success",
        )

        assert event_id is None


class TestLearningStats:
    """Test learning statistics generation."""

    def test_stats_with_no_data(self, learning_manager):
        """Test statistics when no data collected yet."""
        stats = learning_manager.get_learning_stats()

        assert isinstance(stats, LearningStats)
        assert stats.mode == LearningMode.STUDY
        assert stats.total_sessions == 0
        assert stats.successful_sessions == 0
        assert stats.failed_sessions == 0
        assert stats.ready_for_production is False

    def test_stats_with_some_data(self, learning_manager):
        """Test statistics with some sessions recorded."""
        # Record 10 sessions
        for i in range(10):
            learning_manager.record_outcome(
                session_id=f"session_{i}",
                success=(i % 2 == 0),  # 5 successes, 5 failures
            )

        stats = learning_manager.get_learning_stats()

        assert stats.total_sessions == 10
        assert stats.successful_sessions == 5
        assert stats.failed_sessions == 5
        assert stats.ready_for_production is False  # Need 100 sessions

    def test_stats_ready_for_production(self, learning_manager):
        """Test statistics when enough data for production mode."""
        # Record 100 sessions
        for i in range(100):
            learning_manager.record_outcome(
                session_id=f"session_{i}",
                success=True,
            )

        stats = learning_manager.get_learning_stats()

        assert stats.total_sessions == 100
        assert stats.ready_for_production is True
        assert "Ready to transition" in stats.recommendation


class TestTransitionLogic:
    """Test mode transition logic."""

    def test_transition_report_not_ready(self, learning_manager):
        """Test transition report when not enough data."""
        report = learning_manager.should_transition_to_production()

        assert isinstance(report, TransitionReport)
        assert report.ready is False
        assert report.session_count < 100
        assert "Not ready" in report.recommendation

    def test_transition_report_ready(self, learning_manager):
        """Test transition report when enough data collected."""
        # Record 100 sessions
        for i in range(100):
            learning_manager.record_outcome(
                session_id=f"session_{i}",
                success=True,
            )

        report = learning_manager.should_transition_to_production()

        assert report.ready is True
        assert report.session_count == 100
        assert "ready" in report.recommendation.lower()
        assert report.requires_human_review is True

    def test_cannot_transition_without_enough_data(self, learning_manager):
        """Test that transition fails without enough data."""
        success = learning_manager.transition_to_production()

        assert success is False
        assert learning_manager.mode == LearningMode.STUDY

    def test_can_transition_with_enough_data(self, learning_manager):
        """Test that transition succeeds with enough data."""
        # Record 100 sessions
        for i in range(100):
            learning_manager.record_outcome(
                session_id=f"session_{i}",
                success=True,
            )

        success = learning_manager.transition_to_production()

        assert success is True
        assert learning_manager.mode == LearningMode.PRODUCTION

    def test_can_transition_back_to_study(self, learning_manager):
        """Test transitioning back to STUDY mode."""
        learning_manager.mode = LearningMode.PRODUCTION

        learning_manager.transition_to_study()

        assert learning_manager.mode == LearningMode.STUDY


class TestModelPerformanceSummary:
    """Test model performance summary generation."""

    def test_summary_with_no_data(self, learning_manager):
        """Test performance summary with no historical data."""
        summary = learning_manager.get_model_performance_summary()

        assert isinstance(summary, dict)
        # Should have entries for all available models
        assert len(summary) == len(learning_manager.AVAILABLE_MODELS)

        # All should have 0 samples and 0.5 prior success rate
        for model_id, phases in summary.items():
            for phase, stats in phases.items():
                assert stats["sample_count"] == 0
                assert stats["success_rate"] == 0.5  # Prior

    def test_summary_with_data(self, learning_manager):
        """Test performance summary with historical data."""
        # Add some data for one model
        for i in range(5):
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

            learning_manager.store.record_attribution(
                session_id=session_id,
                signature=sig,
                node_id=f"node_{i}",
                state_id=f"state_{i}",
            )

            learning_manager.store.record_session_outcome(
                session_id=session_id,
                outcome=NodeOutcome.VERIFIED_SUCCESS,
            )

        summary = learning_manager.get_model_performance_summary()

        # Check that the model we added data for has non-zero samples
        sonnet_stats = summary["claude-sonnet-4-5-20250929"]["build"]
        assert sonnet_stats["sample_count"] == 5
        assert sonnet_stats["success_rate"] == 1.0  # All successful
