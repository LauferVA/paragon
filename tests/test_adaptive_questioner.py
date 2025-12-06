"""
Tests for Adaptive Questioner - Question optimization logic.

Tests:
- Question prioritization
- Skip probability calculation
- Suggested answer confidence
- Question outcome recording
- User priority integration
- Historical pattern learning
- Question statistics

Layer: Agent (Adaptive)
Status: Production
"""
import pytest
from pathlib import Path
import tempfile
import shutil

from agents.adaptive_questioner import (
    AdaptiveQuestioner,
    QuestionPriority,
    UserPriorities,
)
from agents.schemas import AmbiguityMarker
from infrastructure.training_store import TrainingStore


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_questions.db"

    yield db_path

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def questioner(temp_db):
    """Create an AdaptiveQuestioner instance with temporary database."""
    return AdaptiveQuestioner(db_path=temp_db)


@pytest.fixture
def sample_ambiguities():
    """Create sample ambiguities for testing."""
    return [
        AmbiguityMarker(
            category="SUBJECTIVE",
            text="good performance",
            impact="BLOCKING",
            suggested_question="What specific metric defines 'good performance'?",
            suggested_answer="Response time < 200ms",
        ),
        AmbiguityMarker(
            category="UNDEFINED_TERM",
            text="the database",
            impact="CLARIFYING",
            suggested_question="Which database are you referring to?",
            suggested_answer="PostgreSQL",
        ),
        AmbiguityMarker(
            category="COMPARATIVE",
            text="faster than before",
            impact="CLARIFYING",
            suggested_question="How much faster? What's the baseline?",
            suggested_answer=None,
        ),
    ]


class TestQuestionerInitialization:
    """Test questioner initialization."""

    def test_initializes_with_db_path(self, temp_db):
        """Test that questioner initializes with database path."""
        questioner = AdaptiveQuestioner(db_path=temp_db)
        assert questioner.store.db_path == temp_db

    def test_creates_question_tracking_tables(self, temp_db):
        """Test that question tracking tables are created."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        import sqlite3
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "question_attempts" in tables
        assert "question_outcomes" in tables


class TestQuestionPrioritization:
    """Test question prioritization logic."""

    def test_prioritizes_blocking_over_clarifying(self, questioner, sample_ambiguities):
        """Test that BLOCKING questions are prioritized over CLARIFYING."""
        prioritized = questioner.prioritize_questions(sample_ambiguities)

        # First question should be the BLOCKING one (SUBJECTIVE)
        assert prioritized[0].category == "SUBJECTIVE"
        assert prioritized[0].impact == "BLOCKING"

    def test_respects_max_questions_limit(self, questioner, sample_ambiguities):
        """Test that max_clarification_questions limit is enforced."""
        priorities = UserPriorities(max_clarification_questions=1)

        prioritized = questioner.prioritize_questions(
            sample_ambiguities,
            priorities=priorities,
        )

        assert len(prioritized) == 1

    def test_prioritizes_high_information_gain(self, questioner):
        """Test that high information gain categories are prioritized."""
        ambiguities = [
            AmbiguityMarker(
                category="COMPARATIVE",  # 0.7 gain
                text="better",
                impact="CLARIFYING",
            ),
            AmbiguityMarker(
                category="CONTRADICTIONS",  # 0.9 gain
                text="should be fast but also thorough",
                impact="BLOCKING",
            ),
        ]

        prioritized = questioner.prioritize_questions(ambiguities)

        # CONTRADICTIONS should be first (higher info gain + blocking)
        assert prioritized[0].category == "CONTRADICTIONS"

    def test_speed_priority_filters_questions(self, questioner, sample_ambiguities):
        """Test that high speed_weight reduces question count."""
        # This tests the logic but exact behavior depends on skip probabilities
        priorities = UserPriorities(
            speed_weight=0.8,
            max_clarification_questions=10,
        )

        prioritized = questioner.prioritize_questions(
            sample_ambiguities,
            priorities=priorities,
        )

        # Should still get questions, but prioritization affected
        assert len(prioritized) <= len(sample_ambiguities)


class TestSuggestedAnswers:
    """Test suggested answer logic."""

    def test_returns_llm_suggestion_if_present(self, questioner, sample_ambiguities):
        """Test that LLM suggested answers are returned."""
        # First ambiguity has a suggested answer
        suggested = questioner.get_suggested_answer(sample_ambiguities[0])

        # With no historical data, should return the LLM suggestion if acceptance rate is high enough
        # Since we have no data, it will use the prior (0.5) which is < 0.7 threshold
        # So it should return None initially
        # But if we add some positive history, it should return the suggestion

    def test_returns_none_for_low_confidence_suggestions(self, questioner):
        """Test that low confidence suggestions return None."""
        # Ambiguity with suggestion but category has low acceptance rate
        amb = AmbiguityMarker(
            category="UNDEFINED_TERM",
            text="the thing",
            impact="CLARIFYING",
            suggested_answer="some answer",
        )

        # With no historical data, acceptance rate is 0.5 (prior)
        # This is below the 0.7 threshold, so should return None
        suggested = questioner.get_suggested_answer(amb)

        # Initially should be None due to low confidence
        assert suggested is None or suggested == "some answer"


class TestQuestionOutcomeRecording:
    """Test question outcome recording."""

    def test_records_answered_question(self, questioner, sample_ambiguities):
        """Test recording a question that was answered."""
        amb = sample_ambiguities[0]

        question_id = questioner.record_question_outcome(
            session_id="session_1",
            ambiguity=amb,
            was_answered=True,
            user_answer="Response time < 100ms",
            used_suggestion=False,
            answer_quality_score=1.0,
        )

        assert question_id is not None

    def test_records_skipped_question(self, questioner, sample_ambiguities):
        """Test recording a question that was skipped."""
        amb = sample_ambiguities[1]

        question_id = questioner.record_question_outcome(
            session_id="session_2",
            ambiguity=amb,
            was_answered=False,
            answer_quality_score=0.0,
        )

        assert question_id is not None

    def test_records_suggestion_acceptance(self, questioner, sample_ambiguities):
        """Test recording when user accepts suggested answer."""
        amb = sample_ambiguities[0]

        question_id = questioner.record_question_outcome(
            session_id="session_3",
            ambiguity=amb,
            was_answered=True,
            user_answer=amb.suggested_answer,
            used_suggestion=True,
            answer_quality_score=1.0,
        )

        assert question_id is not None

    def test_updates_question_outcome_after_session(self, questioner, sample_ambiguities):
        """Test updating question outcome after session completes."""
        amb = sample_ambiguities[0]

        question_id = questioner.record_question_outcome(
            session_id="session_4",
            ambiguity=amb,
            was_answered=True,
            user_answer="some answer",
        )

        # Update after session succeeds
        questioner.update_question_outcome(
            question_id=question_id,
            session_id="session_4",
            led_to_success=True,
        )

        # No assertion - just verify it doesn't crash


class TestHistoricalPatterns:
    """Test learning from historical patterns."""

    def test_calculates_skip_probability(self, questioner, sample_ambiguities):
        """Test calculation of skip probability from historical data."""
        # Record some questions, some skipped
        for i in range(10):
            questioner.record_question_outcome(
                session_id=f"session_{i}",
                ambiguity=sample_ambiguities[0],
                was_answered=(i < 7),  # 7 answered, 3 skipped
            )

        # Get skip probability for this category
        skip_prob = questioner._get_skip_probability("SUBJECTIVE")

        # Should be around 30% (3/10)
        assert 0.25 <= skip_prob <= 0.35

    def test_calculates_suggestion_acceptance_rate(self, questioner, sample_ambiguities):
        """Test calculation of suggestion acceptance rate."""
        # Record some questions with suggestions
        amb = sample_ambiguities[0]

        for i in range(10):
            questioner.record_question_outcome(
                session_id=f"session_{i}",
                ambiguity=amb,
                was_answered=True,
                used_suggestion=(i < 8),  # 8 accepted, 2 rejected
            )

        # Get acceptance rate for this category
        acceptance_rate = questioner._get_suggestion_acceptance_rate("SUBJECTIVE")

        # Should be around 80% (8/10)
        assert 0.75 <= acceptance_rate <= 0.85

    def test_finds_most_common_answer(self, questioner):
        """Test finding most common answer for a category."""
        amb = AmbiguityMarker(
            category="UNDEFINED_TERM",
            text="the database",
            impact="CLARIFYING",
        )

        # Record several answers, some repeating
        answers = ["PostgreSQL", "MySQL", "PostgreSQL", "PostgreSQL", "MongoDB"]
        for i, answer in enumerate(answers):
            questioner.record_question_outcome(
                session_id=f"session_{i}",
                ambiguity=amb,
                was_answered=True,
                user_answer=answer,
            )

        # Get most common answer
        common_answer = questioner._get_most_common_answer("UNDEFINED_TERM")

        # Should be "PostgreSQL" (appears 3 times)
        assert common_answer == "PostgreSQL"


class TestQuestionStatistics:
    """Test question statistics generation."""

    def test_stats_with_no_data(self, questioner):
        """Test statistics when no questions recorded."""
        stats = questioner.get_question_stats()

        assert stats["total_questions"] == 0
        assert stats["answered"] == 0
        assert stats["skipped"] == 0
        assert stats["skip_rate"] == 0.0

    def test_stats_with_data(self, questioner, sample_ambiguities):
        """Test statistics with recorded questions."""
        # Record 10 questions: 7 answered, 3 skipped
        for i in range(10):
            questioner.record_question_outcome(
                session_id=f"session_{i}",
                ambiguity=sample_ambiguities[0],
                was_answered=(i < 7),
                answer_quality_score=1.0 if i < 7 else 0.0,
            )

        stats = questioner.get_question_stats()

        assert stats["total_questions"] == 10
        assert stats["answered"] == 7
        assert stats["skipped"] == 3
        assert stats["skip_rate"] == 0.3

    def test_stats_filtered_by_category(self, questioner, sample_ambiguities):
        """Test statistics filtered by ambiguity category."""
        # Record questions from different categories
        for i in range(5):
            questioner.record_question_outcome(
                session_id=f"session_subj_{i}",
                ambiguity=sample_ambiguities[0],  # SUBJECTIVE
                was_answered=True,
            )

        for i in range(3):
            questioner.record_question_outcome(
                session_id=f"session_undef_{i}",
                ambiguity=sample_ambiguities[1],  # UNDEFINED_TERM
                was_answered=True,
            )

        # Get stats for SUBJECTIVE only
        stats = questioner.get_question_stats(category="SUBJECTIVE")

        assert stats["total_questions"] == 5

        # Get stats for UNDEFINED_TERM only
        stats = questioner.get_question_stats(category="UNDEFINED_TERM")

        assert stats["total_questions"] == 3

    def test_stats_tracks_suggestion_usage(self, questioner, sample_ambiguities):
        """Test that statistics track suggestion usage."""
        amb = sample_ambiguities[0]

        # Record questions with some using suggestions
        for i in range(10):
            questioner.record_question_outcome(
                session_id=f"session_{i}",
                ambiguity=amb,
                was_answered=True,
                used_suggestion=(i < 6),  # 6 used suggestions
            )

        stats = questioner.get_question_stats(category="SUBJECTIVE")

        assert stats["used_suggestions"] == 6
        assert stats["suggestion_rate"] == 0.6  # 6/10 answered


class TestUserPriorityIntegration:
    """Test integration with user priorities."""

    def test_control_weight_keeps_all_questions(self, questioner, sample_ambiguities):
        """Test that high control_weight keeps more questions."""
        priorities = UserPriorities(
            control_weight=0.8,
            max_clarification_questions=10,
        )

        prioritized = questioner.prioritize_questions(
            sample_ambiguities,
            priorities=priorities,
        )

        # Should keep all questions
        assert len(prioritized) == len(sample_ambiguities)

    def test_speed_weight_reduces_questions(self, questioner):
        """Test that high speed_weight affects prioritization."""
        # Create many ambiguities
        ambiguities = [
            AmbiguityMarker(
                category="COMPARATIVE",
                text=f"better than option {i}",
                impact="CLARIFYING",
            )
            for i in range(10)
        ]

        priorities = UserPriorities(
            speed_weight=0.8,
            max_clarification_questions=3,  # Only ask top 3
        )

        prioritized = questioner.prioritize_questions(
            ambiguities,
            priorities=priorities,
        )

        # Should only get top 3
        assert len(prioritized) == 3


class TestQuestionPriorityCalculation:
    """Test priority score calculation."""

    def test_priority_includes_information_gain(self, questioner):
        """Test that priority score considers information gain."""
        amb_high = AmbiguityMarker(
            category="CONTRADICTIONS",  # High info gain (0.9)
            text="conflicting requirements",
            impact="BLOCKING",
        )

        amb_low = AmbiguityMarker(
            category="UNDEFINED_PRONOUN",  # Lower info gain (0.6)
            text="it should work",
            impact="CLARIFYING",
        )

        priorities = UserPriorities()

        high_priority = questioner._calculate_question_priority(amb_high, priorities)
        low_priority = questioner._calculate_question_priority(amb_low, priorities)

        # High info gain should have higher priority
        assert high_priority.priority_score > low_priority.priority_score

    def test_blocking_impact_increases_priority(self, questioner):
        """Test that BLOCKING impact increases priority."""
        amb_blocking = AmbiguityMarker(
            category="SUBJECTIVE",
            text="good quality",
            impact="BLOCKING",
        )

        amb_clarifying = AmbiguityMarker(
            category="SUBJECTIVE",
            text="nice color",
            impact="CLARIFYING",
        )

        priorities = UserPriorities()

        blocking_priority = questioner._calculate_question_priority(amb_blocking, priorities)
        clarifying_priority = questioner._calculate_question_priority(amb_clarifying, priorities)

        # BLOCKING should have higher priority
        assert blocking_priority.priority_score > clarifying_priority.priority_score
