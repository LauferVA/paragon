"""
Learning System - Two-mode learning manager.

Study Mode: Collect clean data without biasing decisions
Production Mode: Apply learned patterns to optimize

Reference: docs/IMPLEMENTATION_PLAN_LEARNING.md Sections 5, 9

Layer: L4/L5 (Learning Manager)
Status: Production
"""
import msgspec
import random
from enum import Enum
from typing import Optional, List, Dict
from pathlib import Path

from agents.schemas import CyclePhase, FailureCode, NodeOutcome
from infrastructure.training_store import TrainingStore
from infrastructure.attribution import ForensicAnalyzer
from infrastructure.divergence import DivergenceDetector


class LearningMode(str, Enum):
    """
    Learning mode controls whether system collects data or applies learned patterns.

    STUDY: Collect data, no biasing, random exploration
    PRODUCTION: Apply learned patterns, optimize based on history
    """
    STUDY = "study"
    PRODUCTION = "production"


class ModelRecommendation(msgspec.Struct, kw_only=True, frozen=True):
    """
    Recommendation for which model to use based on historical performance.

    Fields:
        model_id: Recommended model identifier
        confidence: Confidence in recommendation (0.0-1.0)
        reasoning: Human-readable explanation
        based_on_samples: Number of historical samples used
        success_rate: Historical success rate for this model
    """
    model_id: str
    confidence: float
    reasoning: str
    based_on_samples: int
    success_rate: float = 0.5


class LearningStats(msgspec.Struct, kw_only=True):
    """
    Statistics about learning data collected.

    Fields:
        mode: Current learning mode
        total_sessions: Total number of sessions recorded
        successful_sessions: Number of successful sessions
        failed_sessions: Number of failed sessions
        divergences: Number of test-production divergences
        total_attributions: Total node attributions recorded
        ready_for_production: Whether enough data for production mode
        recommendation: Recommended action
    """
    mode: LearningMode
    total_sessions: int
    successful_sessions: int
    failed_sessions: int
    divergences: int
    total_attributions: int
    ready_for_production: bool
    recommendation: str


class TransitionReport(msgspec.Struct, kw_only=True, frozen=True):
    """
    Report on whether system can transition from STUDY to PRODUCTION.

    Criteria (from IMPLEMENTATION_PLAN_LEARNING.md Section 2.2):
    - Minimum 100 sessions completed
    - Statistical significance on at least 3 constraint dimensions (future)
    - Human review of learned priors (manual gate)
    """
    ready: bool
    session_count: int
    success_rate: float
    divergence_rate: float
    recommendation: str
    requires_human_review: bool = True


class LearningManager:
    """
    Manages learning modes and applies learned patterns.

    In Study Mode:
    - Logs everything
    - Makes no recommendations (random exploration)
    - No biasing of model selection
    - Builds the training dataset

    In Production Mode:
    - Uses historical data to optimize decisions
    - Routes to better-performing models
    - Applies learned patterns
    - Continues exploration at 10% rate (epsilon-greedy)

    Reference: docs/IMPLEMENTATION_PLAN_LEARNING.md Sections 5, 9
    """

    # Threshold for transitioning to production mode
    MIN_SESSIONS_FOR_PRODUCTION = 100

    # Exploration rate in production mode (epsilon-greedy)
    EXPLORATION_RATE = 0.1

    # Available models (in priority order - best first)
    AVAILABLE_MODELS = [
        "claude-opus-4-5-20251101",      # Best quality, highest cost
        "claude-sonnet-4-5-20250929",    # Good quality, medium cost
        "claude-sonnet-3-7-20250219",    # Decent quality, lower cost
        "claude-haiku-3-5-20241022",     # Fast, cheapest
    ]

    def __init__(
        self,
        store: Optional[TrainingStore] = None,
        mode: LearningMode = LearningMode.STUDY,
        db_path: Optional[Path] = None,
    ):
        """
        Initialize the learning manager.

        Args:
            store: Optional TrainingStore instance (creates one if not provided)
            mode: Initial learning mode (defaults to STUDY for safety)
            db_path: Optional path to training database
        """
        self.store = store or TrainingStore(db_path=db_path)
        self.mode = mode
        self.analyzer = ForensicAnalyzer(self.store)
        self.divergence = DivergenceDetector(self.store)

        # Warn if starting in production mode without sufficient data
        if mode == LearningMode.PRODUCTION:
            stats = self.get_learning_stats()
            if not stats.ready_for_production:
                print(f"WARNING: Starting in PRODUCTION mode with only {stats.total_sessions} sessions")
                print(f"Recommendation: {stats.recommendation}")

    def get_model_recommendation(
        self,
        phase: CyclePhase,
        task_type: Optional[str] = None,
        constraints: Optional[Dict[str, str]] = None,
    ) -> Optional[ModelRecommendation]:
        """
        Get model recommendation based on historical performance.

        Returns None in STUDY mode (no biasing).

        Args:
            phase: Current cycle phase
            task_type: Optional task type for more specific recommendations
            constraints: Optional constraint filters (not yet implemented)

        Returns:
            ModelRecommendation if in PRODUCTION mode with data, None otherwise
        """
        if self.mode == LearningMode.STUDY:
            return None  # No biasing in study mode

        # Epsilon-greedy exploration: random selection some of the time
        if random.random() < self.EXPLORATION_RATE:
            # Explore: return None to allow random selection
            return None

        # Query success rates for each model in this phase
        candidates = []
        for model_id in self.AVAILABLE_MODELS:
            success_rate = self.store.get_success_rate(
                model_id=model_id,
                phase=phase.value,
                constraints=constraints,
            )

            # Get sample count for confidence calculation
            sample_count = self._get_sample_count(model_id, phase.value)

            # Calculate confidence based on sample size
            # More samples = higher confidence (sigmoid-like curve)
            confidence = min(0.9, sample_count / (sample_count + 10))

            candidates.append({
                "model_id": model_id,
                "success_rate": success_rate,
                "sample_count": sample_count,
                "confidence": confidence,
            })

        # If no candidates have samples, return None
        if all(c["sample_count"] == 0 for c in candidates):
            return None

        # Select best model by success rate (weighted by confidence)
        best = max(candidates, key=lambda c: c["success_rate"] * c["confidence"])

        # Build reasoning
        reasoning = (
            f"Model {best['model_id']} selected for {phase.value} phase "
            f"with {best['success_rate']:.1%} success rate "
            f"based on {best['sample_count']} historical samples"
        )

        return ModelRecommendation(
            model_id=best["model_id"],
            confidence=best["confidence"],
            reasoning=reasoning,
            based_on_samples=best["sample_count"],
            success_rate=best["success_rate"],
        )

    def record_outcome(
        self,
        session_id: str,
        success: bool,
        failure_code: Optional[FailureCode] = None,
        failure_phase: Optional[CyclePhase] = None,
        stats: Optional[Dict] = None,
    ) -> None:
        """
        Record session outcome for learning.

        Args:
            session_id: Session identifier
            success: Whether session succeeded
            failure_code: Optional failure classification (F1-F5)
            failure_phase: Optional phase where failure occurred
            stats: Optional statistics dict (total_nodes, total_iterations, total_tokens)
        """
        outcome = NodeOutcome.VERIFIED_SUCCESS if success else NodeOutcome.VERIFIED_FAILURE

        self.store.record_session_outcome(
            session_id=session_id,
            outcome=outcome,
            failure_code=failure_code,
            failure_phase=failure_phase,
            stats=stats,
        )

    def check_and_log_divergence(
        self,
        session_id: str,
        test_passed: bool,
        user_feedback: str,
        node_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Check for test-production divergence and log if found.

        Args:
            session_id: Session identifier
            test_passed: Whether tests passed
            user_feedback: User feedback on actual production outcome
            node_id: Optional node to attribute divergence to

        Returns:
            Divergence event ID if divergence detected, None otherwise
        """
        event = self.divergence.check_divergence(
            session_id=session_id,
            test_passed=test_passed,
            prod_outcome=user_feedback,
            node_id=node_id,
        )

        if event:
            self.divergence.log_divergence(event)
            return event.event_id

        return None

    def get_learning_stats(self) -> LearningStats:
        """
        Get statistics about learning data collected.

        Returns:
            LearningStats with current status
        """
        total_sessions = self.store.get_session_count()

        # Get success/failure breakdown
        import sqlite3
        with sqlite3.connect(self.store.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN outcome = 'verified_success' THEN 1 ELSE 0 END) as successes,
                    SUM(CASE WHEN outcome = 'verified_failure' THEN 1 ELSE 0 END) as failures
                FROM session_outcomes
            """
            )
            row = cursor.fetchone()
            successful_sessions = row[0] or 0
            failed_sessions = row[1] or 0

        # Get divergence count
        divergence_report = self.divergence.get_divergence_report()
        divergences = divergence_report.total_divergences

        # Get attribution count
        total_attributions = self.store.get_attribution_count()

        # Determine if ready for production
        ready_for_production = total_sessions >= self.MIN_SESSIONS_FOR_PRODUCTION

        # Build recommendation
        if ready_for_production and self.mode == LearningMode.STUDY:
            recommendation = (
                f"Ready to transition to PRODUCTION mode "
                f"({total_sessions} sessions collected). "
                f"Review learned patterns before transitioning."
            )
        elif not ready_for_production:
            remaining = self.MIN_SESSIONS_FOR_PRODUCTION - total_sessions
            recommendation = (
                f"Continue in STUDY mode. "
                f"Need {remaining} more sessions to reach minimum threshold."
            )
        else:
            recommendation = "Currently operating in PRODUCTION mode with learned patterns."

        return LearningStats(
            mode=self.mode,
            total_sessions=total_sessions,
            successful_sessions=successful_sessions,
            failed_sessions=failed_sessions,
            divergences=divergences,
            total_attributions=total_attributions,
            ready_for_production=ready_for_production,
            recommendation=recommendation,
        )

    def should_transition_to_production(self) -> TransitionReport:
        """
        Check if enough data collected to transition from STUDY to PRODUCTION.

        Threshold: 100 sessions (per IMPLEMENTATION_PLAN_LEARNING.md Section 2.2)

        Returns:
            TransitionReport with readiness assessment
        """
        total_sessions = self.store.get_session_count()

        # Calculate success rate
        import sqlite3
        with sqlite3.connect(self.store.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN outcome = 'verified_success' THEN 1 ELSE 0 END) as successes
                FROM session_outcomes
            """
            )
            row = cursor.fetchone()
            total = row[0] or 0
            successes = row[1] or 0

        success_rate = (successes / total) if total > 0 else 0.0

        # Calculate divergence rate
        divergence_rate = self.divergence.calculate_divergence_rate()

        # Determine readiness
        ready = total_sessions >= self.MIN_SESSIONS_FOR_PRODUCTION

        # Build recommendation
        if ready:
            recommendation = (
                f"System ready for PRODUCTION mode. "
                f"Collected {total_sessions} sessions with {success_rate:.1%} success rate. "
                f"Divergence rate: {divergence_rate:.1%}. "
                f"IMPORTANT: Perform human review of learned patterns before transitioning."
            )
        else:
            remaining = self.MIN_SESSIONS_FOR_PRODUCTION - total_sessions
            recommendation = (
                f"Not ready for PRODUCTION mode. "
                f"Need {remaining} more sessions ({total_sessions}/{self.MIN_SESSIONS_FOR_PRODUCTION})."
            )

        return TransitionReport(
            ready=ready,
            session_count=total_sessions,
            success_rate=success_rate,
            divergence_rate=divergence_rate,
            recommendation=recommendation,
            requires_human_review=True,
        )

    def transition_to_production(self) -> bool:
        """
        Transition from STUDY to PRODUCTION mode.

        Returns:
            True if transition successful, False if not ready
        """
        report = self.should_transition_to_production()

        if not report.ready:
            print(f"Cannot transition: {report.recommendation}")
            return False

        print(f"Transitioning to PRODUCTION mode. {report.recommendation}")
        self.mode = LearningMode.PRODUCTION
        return True

    def transition_to_study(self) -> None:
        """
        Transition from PRODUCTION back to STUDY mode.

        Useful for collecting more diverse data or when learned patterns
        are producing poor results.
        """
        print("Transitioning to STUDY mode for data collection.")
        self.mode = LearningMode.STUDY

    def _get_sample_count(self, model_id: str, phase: str) -> int:
        """
        Get number of samples for a model in a phase.

        Args:
            model_id: Model identifier
            phase: Phase name

        Returns:
            Number of historical samples
        """
        import sqlite3

        with sqlite3.connect(self.store.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*)
                FROM node_attributions
                WHERE model_id = ? AND phase = ?
            """,
                (model_id, phase),
            )
            return cursor.fetchone()[0]

    def get_model_performance_summary(self) -> Dict[str, Dict]:
        """
        Get performance summary for all models across all phases.

        Returns:
            Dict mapping model_id -> phase -> stats
        """
        summary = {}

        for model_id in self.AVAILABLE_MODELS:
            model_stats = {}

            for phase in CyclePhase:
                success_rate = self.store.get_success_rate(
                    model_id=model_id,
                    phase=phase.value,
                )
                sample_count = self._get_sample_count(model_id, phase.value)

                model_stats[phase.value] = {
                    "success_rate": success_rate,
                    "sample_count": sample_count,
                }

            summary[model_id] = model_stats

        return summary
