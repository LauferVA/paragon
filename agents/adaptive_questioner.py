"""
Adaptive Questioner - Optimizes question asking strategy.

Uses learned patterns to:
- Skip low-value questions based on historical data
- Suggest better answers based on common patterns
- Respect user priorities and preferences
- Maximize information gain while minimizing user burden

Reference: docs/RESEARCH_ADAPTIVE_QUESTIONING.md

Layer: Agent (Adaptive)
Status: Production
"""
import msgspec
import sqlite3
from typing import List, Optional, Dict
from pathlib import Path

from agents.schemas import CyclePhase, AmbiguityMarker
from infrastructure.training_store import TrainingStore


class QuestionPriority(msgspec.Struct, kw_only=True, frozen=True):
    """
    Priority score for a clarification question.

    Fields:
        question_id: Unique identifier for the question
        expected_information_gain: Expected reduction in implementation uncertainty (0.0-1.0)
        skip_probability: Likelihood user will skip this question (0.0-1.0)
        suggested_answer_confidence: Confidence in suggested answer (0.0-1.0)
        historical_acceptance_rate: How often suggested answers were accepted (0.0-1.0)
        priority_score: Combined score for ranking (higher = more important)
    """
    question_id: str
    expected_information_gain: float  # 0.0-1.0
    skip_probability: float           # 0.0-1.0
    suggested_answer_confidence: float  # 0.0-1.0
    historical_acceptance_rate: float = 0.5  # Default prior
    priority_score: float = 0.5


class QuestionOutcome(msgspec.Struct, kw_only=True, frozen=True):
    """
    Outcome of a clarification question.

    Used for learning which questions are valuable and which can be skipped.
    """
    question_id: str
    session_id: str
    ambiguity_category: str
    question_text: str
    was_answered: bool
    user_answer: Optional[str] = None
    suggested_answer: Optional[str] = None
    used_suggestion: bool = False
    answer_quality_score: float = 0.5  # 0.0-1.0
    led_to_success: Optional[bool] = None  # Determined post-session


class UserPriorities(msgspec.Struct, kw_only=True):
    """
    User's priorities for the questioning strategy.

    Reference: docs/RESEARCH_ADAPTIVE_QUESTIONING.md Section 2

    Fields:
        speed_weight: Weight for speed optimization (0.0-1.0)
        cost_weight: Weight for cost optimization (0.0-1.0)
        control_weight: Weight for user control (0.0-1.0)
        quality_mode: "production" (strict quality) or "experimental" (lenient)
        max_clarification_questions: Maximum questions to ask
        auto_proceed_confidence: Confidence threshold for auto-proceeding (0.0-1.0)
    """
    speed_weight: float = 0.33
    cost_weight: float = 0.33
    control_weight: float = 0.34
    quality_mode: str = "production"  # "production" or "experimental"
    max_clarification_questions: int = 5
    auto_proceed_confidence: float = 0.85


class AdaptiveQuestioner:
    """
    Optimizes question-asking based on learned patterns.

    Uses historical data about:
    - Which questions users skip
    - Which suggested answers are accepted
    - Which ambiguities cause failures
    - Which questions lead to successful outcomes

    This implements the learning strategy from RESEARCH_ADAPTIVE_QUESTIONING.md
    to balance information gain with user experience.
    """

    # Default information gain estimates for ambiguity categories
    # (from RESEARCH_ADAPTIVE_QUESTIONING.md)
    DEFAULT_INFORMATION_GAIN = {
        "SUBJECTIVE": 0.8,           # High gain - critical for quality
        "COMPARATIVE": 0.7,          # High gain - affects behavior
        "UNDEFINED_PRONOUN": 0.6,    # Medium gain - affects correctness
        "UNDEFINED_TERM": 0.75,      # High gain - affects implementation
        "MISSING_CONTEXT": 0.65,     # Medium-high gain - affects scope
        "IMPLICIT_ASSUMPTIONS": 0.7, # High gain - hidden requirements
        "CONTRADICTIONS": 0.9,       # Very high gain - blocking issue
    }

    def __init__(self, store: Optional[TrainingStore] = None, db_path: Optional[Path] = None):
        """
        Initialize the adaptive questioner.

        Args:
            store: Optional TrainingStore for historical data
            db_path: Optional path to training database
        """
        self.store = store or TrainingStore(db_path=db_path)
        self._init_question_tracking()

    def _init_question_tracking(self) -> None:
        """Initialize question tracking tables in the database."""
        with sqlite3.connect(self.store.db_path) as conn:
            conn.executescript(
                """
                -- Question attempts tracking
                CREATE TABLE IF NOT EXISTS question_attempts (
                    question_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    ambiguity_category TEXT NOT NULL,
                    question_text TEXT NOT NULL,
                    suggested_answer TEXT,
                    user_answer TEXT,
                    was_answered INTEGER NOT NULL,
                    used_suggestion INTEGER DEFAULT 0,
                    answer_quality_score REAL DEFAULT 0.5,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Question outcomes (post-session analysis)
                CREATE TABLE IF NOT EXISTS question_outcomes (
                    question_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    led_to_success INTEGER,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (question_id) REFERENCES question_attempts(question_id)
                );

                -- Indexes for fast queries
                CREATE INDEX IF NOT EXISTS idx_qa_session ON question_attempts(session_id);
                CREATE INDEX IF NOT EXISTS idx_qa_category ON question_attempts(ambiguity_category);
                CREATE INDEX IF NOT EXISTS idx_qo_success ON question_outcomes(led_to_success);
            """
            )

    def prioritize_questions(
        self,
        ambiguities: List[AmbiguityMarker],
        priorities: Optional[UserPriorities] = None,
    ) -> List[AmbiguityMarker]:
        """
        Reorder questions by expected information gain and user priorities.

        High-value questions first, skippable questions last.

        Args:
            ambiguities: List of detected ambiguities
            priorities: Optional user priorities (uses defaults if not provided)

        Returns:
            Reordered list of ambiguities by priority
        """
        priorities = priorities or UserPriorities()

        # Calculate priority score for each ambiguity
        scored_ambiguities = []

        for amb in ambiguities:
            priority = self._calculate_question_priority(amb, priorities)
            scored_ambiguities.append((priority.priority_score, amb))

        # Sort by priority score (descending - highest first)
        scored_ambiguities.sort(key=lambda x: x[0], reverse=True)

        # Extract just the ambiguities
        prioritized = [amb for score, amb in scored_ambiguities]

        # Apply max_clarification_questions limit
        if len(prioritized) > priorities.max_clarification_questions:
            prioritized = prioritized[:priorities.max_clarification_questions]

        return prioritized

    def _calculate_question_priority(
        self,
        ambiguity: AmbiguityMarker,
        priorities: UserPriorities,
    ) -> QuestionPriority:
        """
        Calculate priority score for a single ambiguity.

        Algorithm:
        1. Get expected information gain (from category + historical data)
        2. Get skip probability (from historical skip rates)
        3. Get suggested answer confidence (from acceptance rates)
        4. Combine with user priorities to get final score

        Args:
            ambiguity: Ambiguity to score
            priorities: User priorities

        Returns:
            QuestionPriority with calculated scores
        """
        question_id = f"{ambiguity.category}_{hash(ambiguity.text)}"

        # Get base information gain for this category
        base_info_gain = self.DEFAULT_INFORMATION_GAIN.get(
            ambiguity.category, 0.6
        )

        # Adjust for impact (BLOCKING > CLARIFYING)
        if ambiguity.impact == "BLOCKING":
            expected_info_gain = min(1.0, base_info_gain * 1.2)
        else:
            expected_info_gain = base_info_gain

        # Get historical skip probability for this category
        skip_prob = self._get_skip_probability(ambiguity.category)

        # Get suggested answer confidence
        suggested_answer_conf = self._get_suggested_answer_confidence(
            ambiguity.category,
            ambiguity.suggested_answer,
        )

        # Get historical acceptance rate for suggested answers
        acceptance_rate = self._get_suggestion_acceptance_rate(ambiguity.category)

        # Calculate combined priority score based on user priorities
        priority_score = self._compute_priority_score(
            info_gain=expected_info_gain,
            skip_prob=skip_prob,
            suggested_conf=suggested_answer_conf,
            priorities=priorities,
        )

        return QuestionPriority(
            question_id=question_id,
            expected_information_gain=expected_info_gain,
            skip_probability=skip_prob,
            suggested_answer_confidence=suggested_answer_conf,
            historical_acceptance_rate=acceptance_rate,
            priority_score=priority_score,
        )

    def _compute_priority_score(
        self,
        info_gain: float,
        skip_prob: float,
        suggested_conf: float,
        priorities: UserPriorities,
    ) -> float:
        """
        Compute final priority score from components and user priorities.

        Higher score = higher priority = ask this question first

        Args:
            info_gain: Expected information gain
            skip_prob: Probability user will skip
            suggested_conf: Confidence in suggested answer
            priorities: User priorities

        Returns:
            Priority score (0.0-1.0)
        """
        # Base score is information gain
        score = info_gain

        # Apply user priorities
        if priorities.speed_weight > 0.6:
            # User wants speed - penalize questions with high skip probability
            # (they likely don't want to answer many questions)
            score *= (1.0 - skip_prob * 0.5)

        if priorities.control_weight > 0.6:
            # User wants control - boost all questions
            # (they want to answer everything)
            score *= 1.2

        if priorities.cost_weight > 0.6:
            # User wants low cost - boost questions with high suggested answer confidence
            # (we can auto-fill these)
            score *= (1.0 + suggested_conf * 0.3)

        # Cap at 1.0
        return min(1.0, score)

    def get_suggested_answer(
        self,
        ambiguity: AmbiguityMarker,
    ) -> Optional[str]:
        """
        Get suggested answer based on historical patterns.

        Returns None if no confident suggestion available.

        Args:
            ambiguity: Ambiguity to suggest answer for

        Returns:
            Suggested answer string, or None
        """
        # If ambiguity already has a suggested answer from LLM, use it
        if ambiguity.suggested_answer:
            # Check confidence based on historical acceptance rates
            acceptance_rate = self._get_suggestion_acceptance_rate(ambiguity.category)

            # Only return if confidence is high enough
            if acceptance_rate >= 0.7:
                return ambiguity.suggested_answer

        # Query historical patterns for this category
        historical_answer = self._get_most_common_answer(ambiguity.category)

        return historical_answer

    def record_question_outcome(
        self,
        session_id: str,
        ambiguity: AmbiguityMarker,
        was_answered: bool,
        user_answer: Optional[str] = None,
        used_suggestion: bool = False,
        answer_quality_score: float = 0.5,
    ) -> str:
        """
        Record question outcome for learning.

        Args:
            session_id: Session identifier
            ambiguity: The ambiguity/question asked
            was_answered: Whether user provided an answer
            user_answer: Optional user's answer
            used_suggestion: Whether user accepted suggested answer
            answer_quality_score: Quality of answer (0.0=skip, 0.5=brief, 1.0=substantive)

        Returns:
            question_id for tracking
        """
        question_id = f"{session_id}_{ambiguity.category}_{hash(ambiguity.text)}"

        with sqlite3.connect(self.store.db_path) as conn:
            conn.execute(
                """
                INSERT INTO question_attempts (
                    question_id, session_id, ambiguity_category, question_text,
                    suggested_answer, user_answer, was_answered,
                    used_suggestion, answer_quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    question_id,
                    session_id,
                    ambiguity.category,
                    ambiguity.suggested_question or ambiguity.text,
                    ambiguity.suggested_answer,
                    user_answer,
                    1 if was_answered else 0,
                    1 if used_suggestion else 0,
                    answer_quality_score,
                ),
            )

        return question_id

    def update_question_outcome(
        self,
        question_id: str,
        session_id: str,
        led_to_success: bool,
    ) -> None:
        """
        Update question outcome after session completes.

        This connects question quality to session success for learning.

        Args:
            question_id: Question identifier
            session_id: Session identifier
            led_to_success: Whether session succeeded
        """
        with sqlite3.connect(self.store.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO question_outcomes (
                    question_id, session_id, led_to_success, updated_at
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (question_id, session_id, 1 if led_to_success else 0),
            )

    def _get_skip_probability(self, category: str) -> float:
        """
        Get historical skip probability for an ambiguity category.

        Args:
            category: Ambiguity category

        Returns:
            Skip probability (0.0-1.0), defaults to 0.3 if no data
        """
        with sqlite3.connect(self.store.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN was_answered = 0 THEN 1 ELSE 0 END) as skipped
                FROM question_attempts
                WHERE ambiguity_category = ?
            """,
                (category,),
            )
            row = cursor.fetchone()

            if row and row[0] > 0:
                total, skipped = row
                return float(skipped) / float(total)

        # Prior: assume 30% skip rate
        return 0.3

    def _get_suggested_answer_confidence(
        self,
        category: str,
        suggested_answer: Optional[str],
    ) -> float:
        """
        Get confidence in suggested answer based on historical acceptance.

        Args:
            category: Ambiguity category
            suggested_answer: The suggested answer (if any)

        Returns:
            Confidence score (0.0-1.0)
        """
        if not suggested_answer:
            return 0.0

        acceptance_rate = self._get_suggestion_acceptance_rate(category)

        # Confidence is based on how often suggestions are accepted
        return acceptance_rate

    def _get_suggestion_acceptance_rate(self, category: str) -> float:
        """
        Get rate at which suggested answers are accepted for a category.

        Args:
            category: Ambiguity category

        Returns:
            Acceptance rate (0.0-1.0), defaults to 0.5 if no data
        """
        with sqlite3.connect(self.store.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN used_suggestion = 1 THEN 1 ELSE 0 END) as accepted
                FROM question_attempts
                WHERE ambiguity_category = ?
                  AND suggested_answer IS NOT NULL
            """,
                (category,),
            )
            row = cursor.fetchone()

            if row and row[0] > 0:
                total, accepted = row
                return float(accepted) / float(total)

        # Prior: assume 50% acceptance
        return 0.5

    def _get_most_common_answer(self, category: str) -> Optional[str]:
        """
        Get the most common user answer for a category.

        Useful for suggesting answers based on patterns.

        Args:
            category: Ambiguity category

        Returns:
            Most common answer, or None if no data
        """
        with sqlite3.connect(self.store.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT user_answer, COUNT(*) as count
                FROM question_attempts
                WHERE ambiguity_category = ?
                  AND user_answer IS NOT NULL
                  AND was_answered = 1
                GROUP BY user_answer
                ORDER BY count DESC
                LIMIT 1
            """,
                (category,),
            )
            row = cursor.fetchone()

            if row:
                return row[0]

        return None

    def get_question_stats(self, category: Optional[str] = None) -> Dict:
        """
        Get statistics about question asking patterns.

        Args:
            category: Optional category filter

        Returns:
            Dict with statistics
        """
        query = """
            SELECT
                COUNT(*) as total_questions,
                SUM(CASE WHEN was_answered = 1 THEN 1 ELSE 0 END) as answered,
                SUM(CASE WHEN was_answered = 0 THEN 1 ELSE 0 END) as skipped,
                SUM(CASE WHEN used_suggestion = 1 THEN 1 ELSE 0 END) as used_suggestions,
                AVG(answer_quality_score) as avg_quality
            FROM question_attempts
        """
        params = []

        if category:
            query += " WHERE ambiguity_category = ?"
            params.append(category)

        with sqlite3.connect(self.store.db_path) as conn:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()

            if row and row[0] > 0:
                total, answered, skipped, suggestions, avg_quality = row
                return {
                    "total_questions": total,
                    "answered": answered,
                    "skipped": skipped,
                    "skip_rate": float(skipped) / float(total) if total > 0 else 0.0,
                    "used_suggestions": suggestions,
                    "suggestion_rate": float(suggestions) / float(answered) if answered > 0 else 0.0,
                    "avg_quality_score": avg_quality or 0.0,
                }

        return {
            "total_questions": 0,
            "answered": 0,
            "skipped": 0,
            "skip_rate": 0.0,
            "used_suggestions": 0,
            "suggestion_rate": 0.0,
            "avg_quality_score": 0.0,
        }
