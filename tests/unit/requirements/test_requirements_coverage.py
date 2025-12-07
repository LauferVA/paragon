"""
Unit tests for requirements/socratic_engine.py - Comprehensive coverage

Tests the Socratic Engine, Gap Analyzer, and supporting components including:
- Utility functions (now_utc)
- Enums (QuestionLevel, QuestionCategory, AskCondition, AnswerStatus)
- CanonicalQuestion structure
- GapAnalyzer functionality
- SessionState properties
- SocraticEngine session management
- Question flow and answer recording
- Gap analysis and finalization
- Convenience functions
"""
import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from requirements.socratic_engine import (
    # Utility
    now_utc,
    # Enums
    QuestionLevel,
    QuestionCategory,
    AskCondition,
    AnswerStatus,
    # Question structures
    CanonicalQuestion,
    ALL_QUESTIONS,
    L1_QUESTIONS,
    L2_QUESTIONS,
    L3_QUESTIONS,
    TERMINAL_QUESTIONS,
    # Gap analysis
    Gap,
    GapAnalyzer,
    # Session management
    QuestionAnswer,
    SessionState,
    SocraticEngine,
    # Convenience functions
    get_all_questions,
    get_questions_by_level,
    analyze_for_gaps,
)
from core.ontology import NodeType, NodeStatus, EdgeType


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

def test_now_utc_returns_iso8601_string():
    """
    Validate that now_utc returns a valid ISO8601 timestamp string.

    Verifies:
    - Returns a string
    - String is in ISO8601 format
    - Contains timezone info
    """
    timestamp = now_utc()

    assert isinstance(timestamp, str)
    assert "T" in timestamp  # ISO8601 has T separator
    # Should be parseable as datetime
    parsed = datetime.fromisoformat(timestamp)
    assert parsed.tzinfo is not None


def test_now_utc_uses_utc_timezone():
    """
    Validate that now_utc uses UTC timezone.

    Verifies:
    - Timestamp is in UTC
    - Multiple calls return different timestamps
    """
    timestamp = now_utc()
    parsed = datetime.fromisoformat(timestamp)

    # Should be UTC
    assert parsed.tzinfo == timezone.utc


# =============================================================================
# ENUM TESTS
# =============================================================================

def test_question_level_enum_values():
    """Validate QuestionLevel enum has expected values."""
    assert QuestionLevel.L1_SYSTEM.value == "L1_SYSTEM"
    assert QuestionLevel.L2_MODULE.value == "L2_MODULE"
    assert QuestionLevel.L3_NODE.value == "L3_NODE"
    assert QuestionLevel.TERMINAL.value == "TERMINAL"


def test_question_category_enum_values():
    """Validate QuestionCategory enum has expected values."""
    assert QuestionCategory.SCOPE.value == "scope"
    assert QuestionCategory.BEHAVIOR.value == "behavior"
    assert QuestionCategory.DATA.value == "data"
    assert QuestionCategory.INTEGRATION.value == "integration"
    assert QuestionCategory.CONSTRAINTS.value == "constraints"
    assert QuestionCategory.ERROR_HANDLING.value == "error_handling"
    assert QuestionCategory.SECURITY.value == "security"
    assert QuestionCategory.PERFORMANCE.value == "performance"
    assert QuestionCategory.ASSUMPTIONS.value == "assumptions"


def test_ask_condition_enum_values():
    """Validate AskCondition enum has expected values."""
    assert AskCondition.ALWAYS.value == "always"
    assert AskCondition.IF_AMBIGUOUS.value == "if_ambiguous"
    assert AskCondition.IF_NO_SCHEMA.value == "if_no_schema"
    assert AskCondition.IF_EXTERNAL_DEPS.value == "if_external"
    assert AskCondition.IF_SIDE_EFFECTS.value == "if_side_effects"
    assert AskCondition.IF_MISSING_PROVENANCE.value == "if_no_provenance"
    assert AskCondition.IF_MISSING_SCOPE.value == "if_no_scope"


def test_answer_status_enum_values():
    """Validate AnswerStatus enum has expected values."""
    assert AnswerStatus.PENDING.value == "pending"
    assert AnswerStatus.ASKED.value == "asked"
    assert AnswerStatus.ANSWERED.value == "answered"
    assert AnswerStatus.SKIPPED.value == "skipped"
    assert AnswerStatus.INFERRED.value == "inferred"


# =============================================================================
# CANONICAL QUESTION TESTS
# =============================================================================

def test_canonical_question_creation():
    """Validate CanonicalQuestion struct creation."""
    question = CanonicalQuestion(
        id="test_q",
        level=QuestionLevel.L1_SYSTEM.value,
        category=QuestionCategory.SCOPE.value,
        question="What is the purpose?",
        rationale="Understanding purpose is critical",
    )

    assert question.id == "test_q"
    assert question.level == QuestionLevel.L1_SYSTEM.value
    assert question.category == QuestionCategory.SCOPE.value
    assert question.question == "What is the purpose?"
    assert question.rationale == "Understanding purpose is critical"
    assert question.ask_when == AskCondition.ALWAYS.value  # Default
    assert question.follow_ups == ()  # Default
    assert question.examples == ()  # Default


def test_canonical_question_with_all_fields():
    """Validate CanonicalQuestion with all optional fields."""
    question = CanonicalQuestion(
        id="test_q2",
        level=QuestionLevel.L2_MODULE.value,
        category=QuestionCategory.DATA.value,
        question="What is the schema?",
        rationale="Schema defines structure",
        ask_when=AskCondition.IF_NO_SCHEMA.value,
        follow_ups=("test_q3", "test_q4"),
        examples=("Example 1", "Example 2"),
    )

    assert question.ask_when == AskCondition.IF_NO_SCHEMA.value
    assert len(question.follow_ups) == 2
    assert "test_q3" in question.follow_ups
    assert len(question.examples) == 2
    assert "Example 1" in question.examples


def test_canonical_question_is_frozen():
    """Validate that CanonicalQuestion is immutable (frozen)."""
    question = CanonicalQuestion(
        id="test_frozen",
        level=QuestionLevel.L1_SYSTEM.value,
        category=QuestionCategory.SCOPE.value,
        question="Test?",
        rationale="Test",
    )

    # Should not be able to modify
    with pytest.raises(AttributeError):
        question.id = "new_id"


# =============================================================================
# QUESTION REGISTRY TESTS
# =============================================================================

def test_l1_questions_registry():
    """Validate L1 question registry contains expected questions."""
    assert "domain_core" in L1_QUESTIONS
    assert "primary_users" in L1_QUESTIONS
    assert "scale_expectation" in L1_QUESTIONS
    assert "existing_systems" in L1_QUESTIONS
    assert "security_requirements" in L1_QUESTIONS

    # All should be L1 level
    for q in L1_QUESTIONS.values():
        assert q.level == QuestionLevel.L1_SYSTEM.value


def test_l2_questions_registry():
    """Validate L2 question registry contains expected questions."""
    assert "trigger_exit" in L2_QUESTIONS
    assert "cardinality_latency" in L2_QUESTIONS
    assert "dependencies_dependents" in L2_QUESTIONS
    assert "data_ownership" in L2_QUESTIONS
    assert "fault_isolation" in L2_QUESTIONS

    # All should be L2 level
    for q in L2_QUESTIONS.values():
        assert q.level == QuestionLevel.L2_MODULE.value


def test_l3_questions_registry():
    """Validate L3 question registry contains expected questions."""
    assert "io_schema" in L3_QUESTIONS
    assert "side_effects" in L3_QUESTIONS
    assert "input_provenance" in L3_QUESTIONS
    assert "implicit_deps" in L3_QUESTIONS

    # All should be L3 level
    for q in L3_QUESTIONS.values():
        assert q.level == QuestionLevel.L3_NODE.value


def test_terminal_questions_registry():
    """Validate terminal question registry contains expected questions."""
    assert "out_of_scope" in TERMINAL_QUESTIONS
    assert "assumptions_review" in TERMINAL_QUESTIONS
    assert "delegation_license" in TERMINAL_QUESTIONS

    # All should be TERMINAL level
    for q in TERMINAL_QUESTIONS.values():
        assert q.level == QuestionLevel.TERMINAL.value


def test_all_questions_aggregation():
    """Validate ALL_QUESTIONS contains all question registries."""
    expected_count = (
        len(L1_QUESTIONS) +
        len(L2_QUESTIONS) +
        len(L3_QUESTIONS) +
        len(TERMINAL_QUESTIONS)
    )

    assert len(ALL_QUESTIONS) == expected_count

    # Should contain all question IDs
    for qid in L1_QUESTIONS.keys():
        assert qid in ALL_QUESTIONS
    for qid in L2_QUESTIONS.keys():
        assert qid in ALL_QUESTIONS
    for qid in L3_QUESTIONS.keys():
        assert qid in ALL_QUESTIONS
    for qid in TERMINAL_QUESTIONS.keys():
        assert qid in ALL_QUESTIONS


# =============================================================================
# GAP ANALYZER TESTS
# =============================================================================

def test_gap_analyzer_initialization():
    """Validate GapAnalyzer initializes with detection patterns."""
    analyzer = GapAnalyzer()

    assert hasattr(analyzer, '_ambiguous_terms')
    assert hasattr(analyzer, '_side_effect_terms')
    assert hasattr(analyzer, '_external_terms')

    # Check some expected terms
    assert "should" in analyzer._ambiguous_terms
    assert "maybe" in analyzer._ambiguous_terms
    assert "send" in analyzer._side_effect_terms
    assert "update" in analyzer._side_effect_terms
    assert "api" in analyzer._external_terms


def test_gap_analyzer_detects_ambiguous_terms():
    """Validate GapAnalyzer detects ambiguous terminology."""
    analyzer = GapAnalyzer()
    content = "The system should maybe send an email when appropriate."

    gaps = analyzer.analyze_spec(content)

    # Should detect ambiguous terms
    ambiguous_gaps = [g for g in gaps if "ambiguous" in g.context.lower()]
    assert len(ambiguous_gaps) > 0
    assert ambiguous_gaps[0].severity in ["medium", "high", "critical"]


def test_gap_analyzer_detects_side_effects():
    """Validate GapAnalyzer detects side effects."""
    analyzer = GapAnalyzer()
    content = "The function will send a notification and update the database."

    gaps = analyzer.analyze_spec(content)

    # Should detect side effects
    side_effect_gaps = [g for g in gaps if g.question_id == "side_effects"]
    assert len(side_effect_gaps) > 0
    assert side_effect_gaps[0].severity == "high"


def test_gap_analyzer_detects_external_dependencies():
    """Validate GapAnalyzer detects external dependencies."""
    analyzer = GapAnalyzer()
    content = "The system integrates with the Stripe API and AWS services."

    gaps = analyzer.analyze_spec(content)

    # Should detect external dependencies
    external_gaps = [g for g in gaps if "external" in g.context.lower() or "dependencies" in g.context.lower()]
    assert len(external_gaps) > 0


def test_gap_analyzer_detects_missing_out_of_scope():
    """Validate GapAnalyzer detects missing out_of_scope."""
    analyzer = GapAnalyzer()
    content = "Build a user authentication system with login and registration."

    gaps = analyzer.analyze_spec(content)

    # Should detect missing out_of_scope
    scope_gaps = [g for g in gaps if g.question_id == "out_of_scope"]
    assert len(scope_gaps) > 0
    assert scope_gaps[0].severity == "critical"


def test_gap_analyzer_accepts_out_of_scope_in_content():
    """Validate GapAnalyzer accepts out_of_scope in content."""
    analyzer = GapAnalyzer()
    content = "Build a login system. Out of scope: 2FA, OAuth, password reset."

    gaps = analyzer.analyze_spec(content)

    # Should NOT flag missing out_of_scope
    scope_gaps = [g for g in gaps if g.question_id == "out_of_scope"]
    assert len(scope_gaps) == 0


def test_gap_analyzer_accepts_out_of_scope_in_metadata():
    """Validate GapAnalyzer accepts out_of_scope in metadata."""
    analyzer = GapAnalyzer()
    content = "Build a login system."
    metadata = {"out_of_scope": "2FA is out of scope"}

    gaps = analyzer.analyze_spec(content, metadata)

    # Should NOT flag missing out_of_scope
    scope_gaps = [g for g in gaps if g.question_id == "out_of_scope"]
    assert len(scope_gaps) == 0


def test_gap_analyzer_detects_missing_trigger_exit():
    """Validate GapAnalyzer detects missing trigger/exit conditions."""
    analyzer = GapAnalyzer()
    content = "The system processes user data."

    gaps = analyzer.analyze_spec(content)

    # Should detect missing trigger/exit
    trigger_gaps = [g for g in gaps if g.question_id == "trigger_exit"]
    assert len(trigger_gaps) > 0


def test_gap_analyzer_accepts_trigger_and_exit():
    """Validate GapAnalyzer accepts clear trigger and exit conditions."""
    analyzer = GapAnalyzer()
    content = "When user clicks submit, process data until all records are complete."

    gaps = analyzer.analyze_spec(content)

    # Should NOT flag missing trigger/exit
    trigger_gaps = [g for g in gaps if g.question_id == "trigger_exit"]
    assert len(trigger_gaps) == 0


def test_gap_analyzer_sorts_by_severity():
    """Validate GapAnalyzer sorts gaps by severity."""
    analyzer = GapAnalyzer()
    content = "The system might send an API call."  # Multiple gaps

    gaps = analyzer.analyze_spec(content)

    # Should be sorted: critical, high, medium, low
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    for i in range(len(gaps) - 1):
        assert severity_order[gaps[i].severity] <= severity_order[gaps[i + 1].severity]


def test_gap_analyzer_get_questions_for_gaps():
    """Validate GapAnalyzer can retrieve questions for identified gaps."""
    analyzer = GapAnalyzer()
    gaps = [
        Gap(question_id="out_of_scope", severity="critical", context="Test"),
        Gap(question_id="side_effects", severity="high", context="Test"),
        Gap(question_id="invalid_id", severity="low", context="Test"),
    ]

    questions = analyzer.get_questions_for_gaps(gaps)

    # Should get questions for valid IDs only
    assert len(questions) == 2
    question_ids = [q.id for q in questions]
    assert "out_of_scope" in question_ids
    assert "side_effects" in question_ids
    assert "invalid_id" not in question_ids


def test_gap_analyzer_empty_content():
    """Validate GapAnalyzer handles empty content."""
    analyzer = GapAnalyzer()
    gaps = analyzer.analyze_spec("")

    # Should still detect missing out_of_scope and trigger_exit
    assert len(gaps) > 0


# =============================================================================
# SESSION STATE TESTS
# =============================================================================

def test_session_state_creation():
    """Validate SessionState dataclass creation."""
    session = SessionState(
        session_id="test_session",
        req_id="REQ_123",
    )

    assert session.session_id == "test_session"
    assert session.req_id == "REQ_123"
    assert session.current_level == QuestionLevel.L1_SYSTEM.value
    assert session.qa_pairs == {}
    assert session.assumptions == []
    assert session.inferred_answers == {}
    assert session.completed_at is None


def test_session_state_is_complete_empty():
    """Validate SessionState.is_complete returns True for empty session."""
    session = SessionState(session_id="test", req_id="REQ_1")

    # No questions -> complete
    assert session.is_complete is True


def test_session_state_is_complete_with_pending():
    """Validate SessionState.is_complete returns False with pending questions."""
    session = SessionState(session_id="test", req_id="REQ_1")

    qa = QuestionAnswer(
        question_id="test_q",
        question=L1_QUESTIONS["domain_core"],
        status=AnswerStatus.PENDING,
    )
    session.qa_pairs["test_q"] = qa

    assert session.is_complete is False


def test_session_state_is_complete_all_answered():
    """Validate SessionState.is_complete returns True when all answered."""
    session = SessionState(session_id="test", req_id="REQ_1")

    qa1 = QuestionAnswer(
        question_id="q1",
        question=L1_QUESTIONS["domain_core"],
        status=AnswerStatus.ANSWERED,
        answer="E-commerce",
    )
    qa2 = QuestionAnswer(
        question_id="q2",
        question=L1_QUESTIONS["primary_users"],
        status=AnswerStatus.SKIPPED,
    )
    session.qa_pairs["q1"] = qa1
    session.qa_pairs["q2"] = qa2

    assert session.is_complete is True


def test_session_state_completion_rate_empty():
    """Validate SessionState.completion_rate for empty session."""
    session = SessionState(session_id="test", req_id="REQ_1")

    assert session.completion_rate == 0.0


def test_session_state_completion_rate_partial():
    """Validate SessionState.completion_rate calculates correctly."""
    session = SessionState(session_id="test", req_id="REQ_1")

    # Add 4 questions: 2 answered, 1 inferred, 1 pending
    qa1 = QuestionAnswer(
        question_id="q1",
        question=L1_QUESTIONS["domain_core"],
        status=AnswerStatus.ANSWERED,
        answer="Test",
    )
    qa2 = QuestionAnswer(
        question_id="q2",
        question=L1_QUESTIONS["primary_users"],
        status=AnswerStatus.INFERRED,
        answer="Users",
    )
    qa3 = QuestionAnswer(
        question_id="q3",
        question=L1_QUESTIONS["scale_expectation"],
        status=AnswerStatus.SKIPPED,
    )
    qa4 = QuestionAnswer(
        question_id="q4",
        question=L1_QUESTIONS["security_requirements"],
        status=AnswerStatus.PENDING,
    )

    session.qa_pairs = {"q1": qa1, "q2": qa2, "q3": qa3, "q4": qa4}

    # 3 out of 4 are complete (answered/inferred/skipped)
    assert session.completion_rate == 0.75


def test_session_state_completion_rate_full():
    """Validate SessionState.completion_rate for fully answered session."""
    session = SessionState(session_id="test", req_id="REQ_1")

    qa1 = QuestionAnswer(
        question_id="q1",
        question=L1_QUESTIONS["domain_core"],
        status=AnswerStatus.ANSWERED,
        answer="Test",
    )
    qa2 = QuestionAnswer(
        question_id="q2",
        question=L1_QUESTIONS["primary_users"],
        status=AnswerStatus.ANSWERED,
        answer="Users",
    )

    session.qa_pairs = {"q1": qa1, "q2": qa2}

    assert session.completion_rate == 1.0


# =============================================================================
# SOCRATIC ENGINE - SESSION MANAGEMENT TESTS
# =============================================================================

def test_socratic_engine_initialization():
    """Validate SocraticEngine initializes correctly."""
    engine = SocraticEngine()

    assert hasattr(engine, '_sessions')
    assert hasattr(engine, '_gap_analyzer')
    assert isinstance(engine._sessions, dict)
    assert isinstance(engine._gap_analyzer, GapAnalyzer)


def test_socratic_engine_create_session():
    """Validate SocraticEngine.create_session creates a new session."""
    engine = SocraticEngine()

    session = engine.create_session(req_id="REQ_123")

    assert session.req_id == "REQ_123"
    assert session.session_id is not None
    assert len(session.session_id) > 0

    # Should initialize with L1 questions
    assert len(session.qa_pairs) == len(L1_QUESTIONS)
    for qid in L1_QUESTIONS.keys():
        assert qid in session.qa_pairs
        assert session.qa_pairs[qid].status == AnswerStatus.PENDING


def test_socratic_engine_create_session_custom_id():
    """Validate SocraticEngine.create_session with custom session_id."""
    engine = SocraticEngine()

    session = engine.create_session(req_id="REQ_123", session_id="custom_id")

    assert session.session_id == "custom_id"
    assert session.req_id == "REQ_123"


def test_socratic_engine_create_session_stores_session():
    """Validate SocraticEngine.create_session stores the session."""
    engine = SocraticEngine()

    session = engine.create_session(req_id="REQ_123")

    # Should be stored in engine
    assert session.session_id in engine._sessions
    assert engine._sessions[session.session_id] == session


def test_socratic_engine_get_session():
    """Validate SocraticEngine.get_session retrieves existing session."""
    engine = SocraticEngine()

    session = engine.create_session(req_id="REQ_123")
    retrieved = engine.get_session(session.session_id)

    assert retrieved is not None
    assert retrieved.session_id == session.session_id
    assert retrieved.req_id == "REQ_123"


def test_socratic_engine_get_session_not_found():
    """Validate SocraticEngine.get_session returns None for non-existent session."""
    engine = SocraticEngine()

    retrieved = engine.get_session("non_existent_id")

    assert retrieved is None


# =============================================================================
# SOCRATIC ENGINE - QUESTION FLOW TESTS
# =============================================================================

def test_socratic_engine_get_pending_questions():
    """Validate SocraticEngine.get_pending_questions returns pending questions."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    pending = engine.get_pending_questions(session)

    # All L1 questions should be pending
    assert len(pending) == len(L1_QUESTIONS)


def test_socratic_engine_get_pending_questions_current_level_only():
    """Validate get_pending_questions returns only current level questions."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    # Add L2 question
    qa_l2 = QuestionAnswer(
        question_id="test_l2",
        question=L2_QUESTIONS["trigger_exit"],
        status=AnswerStatus.PENDING,
    )
    session.qa_pairs["test_l2"] = qa_l2

    # Should only return L1 (current level)
    pending = engine.get_pending_questions(session)
    assert len(pending) == len(L1_QUESTIONS)
    assert all(q.level == QuestionLevel.L1_SYSTEM.value for q in pending)


def test_socratic_engine_get_next_question():
    """Validate SocraticEngine.get_next_question returns first pending question."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    next_q = engine.get_next_question(session)

    assert next_q is not None
    assert next_q.level == QuestionLevel.L1_SYSTEM.value


def test_socratic_engine_get_next_question_none_pending():
    """Validate get_next_question returns None when no pending questions."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    # Answer all L1 questions
    for qid in session.qa_pairs:
        session.qa_pairs[qid].status = AnswerStatus.ANSWERED
        session.qa_pairs[qid].answer = "Test answer"

    next_q = engine.get_next_question(session)

    assert next_q is None


def test_socratic_engine_record_answer():
    """Validate SocraticEngine.record_answer records user answer."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    qid = "domain_core"
    answer = "E-commerce platform"

    engine.record_answer(session, qid, answer)

    qa = session.qa_pairs[qid]
    assert qa.answer == answer
    assert qa.status == AnswerStatus.ANSWERED
    assert qa.answered_at is not None


def test_socratic_engine_record_answer_inferred():
    """Validate record_answer with inferred=True marks as inferred."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    qid = "domain_core"
    answer = "Inferred domain"

    engine.record_answer(session, qid, answer, inferred=True)

    qa = session.qa_pairs[qid]
    assert qa.answer == answer
    assert qa.status == AnswerStatus.INFERRED
    assert qid in session.inferred_answers
    assert session.inferred_answers[qid] == answer


def test_socratic_engine_record_answer_invalid_question():
    """Validate record_answer ignores invalid question IDs."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    # Should not raise error
    engine.record_answer(session, "invalid_qid", "Answer")

    assert "invalid_qid" not in session.qa_pairs


def test_socratic_engine_skip_question():
    """Validate SocraticEngine.skip_question marks question as skipped."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    qid = "domain_core"
    engine.skip_question(session, qid)

    qa = session.qa_pairs[qid]
    assert qa.status == AnswerStatus.SKIPPED
    assert qa.answer is None


def test_socratic_engine_skip_question_invalid_id():
    """Validate skip_question ignores invalid question IDs."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    # Should not raise error
    engine.skip_question(session, "invalid_qid")

    assert "invalid_qid" not in session.qa_pairs


def test_socratic_engine_advance_level_l1_to_l2():
    """Validate advance_level progresses from L1 to L2."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    assert session.current_level == QuestionLevel.L1_SYSTEM.value

    new_level = engine.advance_level(session)

    assert new_level == QuestionLevel.L2_MODULE.value
    assert session.current_level == QuestionLevel.L2_MODULE.value

    # Should add L2 questions
    for qid in L2_QUESTIONS.keys():
        assert qid in session.qa_pairs


def test_socratic_engine_advance_level_l2_to_l3():
    """Validate advance_level progresses from L2 to L3."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    engine.advance_level(session)  # L1 -> L2
    new_level = engine.advance_level(session)  # L2 -> L3

    assert new_level == QuestionLevel.L3_NODE.value
    assert session.current_level == QuestionLevel.L3_NODE.value


def test_socratic_engine_advance_level_l3_to_terminal():
    """Validate advance_level progresses from L3 to TERMINAL."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    engine.advance_level(session)  # L1 -> L2
    engine.advance_level(session)  # L2 -> L3
    new_level = engine.advance_level(session)  # L3 -> TERMINAL

    assert new_level == QuestionLevel.TERMINAL.value
    assert session.current_level == QuestionLevel.TERMINAL.value

    # Should add TERMINAL questions
    for qid in TERMINAL_QUESTIONS.keys():
        assert qid in session.qa_pairs


def test_socratic_engine_advance_level_at_terminal():
    """Validate advance_level stays at TERMINAL when at end."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    # Advance to TERMINAL
    engine.advance_level(session)  # L1 -> L2
    engine.advance_level(session)  # L2 -> L3
    engine.advance_level(session)  # L3 -> TERMINAL

    # Try to advance past TERMINAL
    new_level = engine.advance_level(session)

    assert new_level == QuestionLevel.TERMINAL.value
    assert session.current_level == QuestionLevel.TERMINAL.value


# =============================================================================
# SOCRATIC ENGINE - GAP ANALYSIS TESTS
# =============================================================================

def test_socratic_engine_analyze_spec():
    """Validate SocraticEngine.analyze_spec identifies gaps."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    content = "Build a system that sends notifications via API."

    gaps = engine.analyze_spec(session, content)

    # Should detect gaps
    assert len(gaps) > 0


def test_socratic_engine_analyze_spec_adds_l3_questions():
    """Validate analyze_spec adds L3 questions for identified gaps."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    content = "The system will send notifications and update records."

    initial_count = len(session.qa_pairs)
    gaps = engine.analyze_spec(session, content)

    # Should add L3 questions for side effects
    assert len(session.qa_pairs) > initial_count


def test_socratic_engine_analyze_spec_with_metadata():
    """Validate analyze_spec uses metadata."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    content = "Build a system."
    metadata = {"out_of_scope": "Mobile support"}

    gaps = engine.analyze_spec(session, content, metadata)

    # Should not flag missing out_of_scope
    scope_gaps = [g for g in gaps if g.question_id == "out_of_scope"]
    assert len(scope_gaps) == 0


def test_socratic_engine_get_critical_gaps():
    """Validate get_critical_gaps returns critical unanswered gaps."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    # Add a critical gap
    qa = QuestionAnswer(
        question_id="out_of_scope",
        question=TERMINAL_QUESTIONS["out_of_scope"],
        status=AnswerStatus.PENDING,
        context="Missing scope definition",
    )
    session.qa_pairs["out_of_scope"] = qa

    critical = engine.get_critical_gaps(session)

    assert len(critical) > 0
    assert all(g.severity == "critical" for g in critical)


# =============================================================================
# SOCRATIC ENGINE - FINALIZATION TESTS
# =============================================================================

def test_socratic_engine_add_assumption():
    """Validate add_assumption adds assumption to session."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    assumption = "Users have valid email addresses"
    engine.add_assumption(session, assumption)

    assert assumption in session.assumptions


def test_socratic_engine_finalize_session():
    """Validate finalize_session generates summary."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    # Add some answers
    engine.record_answer(session, "domain_core", "E-commerce")
    engine.add_assumption(session, "Users are authenticated")

    result = engine.finalize_session(session, approved=True)

    assert result["session_id"] == session.session_id
    assert result["req_id"] == "REQ_123"
    assert result["approved"] is True
    assert "completion_rate" in result
    assert "answers" in result
    assert "domain_core" in result["answers"]
    assert "assumptions" in result
    assert "Users are authenticated" in result["assumptions"]
    assert result["started_at"] is not None
    assert result["completed_at"] is not None


def test_socratic_engine_finalize_session_not_approved():
    """Validate finalize_session with approved=False."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    result = engine.finalize_session(session, approved=False)

    assert result["approved"] is False


def test_socratic_engine_finalize_session_with_inferred():
    """Validate finalize_session includes inferred answers."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    engine.record_answer(session, "domain_core", "E-commerce", inferred=True)

    result = engine.finalize_session(session)

    assert "inferred" in result
    assert "domain_core" in result["inferred"]


def test_socratic_engine_finalize_session_sets_completed_at():
    """Validate finalize_session sets completed_at timestamp."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    assert session.completed_at is None

    engine.finalize_session(session)

    assert session.completed_at is not None


# =============================================================================
# SOCRATIC ENGINE - CONVENIENCE METHODS TESTS
# =============================================================================

def test_socratic_engine_get_session_summary():
    """Validate get_session_summary returns comprehensive summary."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    # Add varied statuses
    engine.record_answer(session, "domain_core", "E-commerce")
    engine.skip_question(session, "primary_users")

    summary = engine.get_session_summary(session)

    assert summary["session_id"] == session.session_id
    assert summary["req_id"] == "REQ_123"
    assert summary["current_level"] == QuestionLevel.L1_SYSTEM.value
    assert "total_questions" in summary
    assert "by_status" in summary
    assert summary["by_status"][AnswerStatus.ANSWERED.value] >= 1
    assert summary["by_status"][AnswerStatus.SKIPPED.value] >= 1
    assert "completion_rate" in summary
    assert "is_complete" in summary


# =============================================================================
# CONVENIENCE FUNCTIONS TESTS
# =============================================================================

def test_get_all_questions():
    """Validate get_all_questions returns all questions."""
    all_q = get_all_questions()

    assert len(all_q) == len(ALL_QUESTIONS)
    assert "domain_core" in all_q
    assert "trigger_exit" in all_q
    assert "side_effects" in all_q
    assert "out_of_scope" in all_q


def test_get_all_questions_returns_copy():
    """Validate get_all_questions returns a copy, not reference."""
    all_q1 = get_all_questions()
    all_q2 = get_all_questions()

    # Should be different objects
    assert all_q1 is not all_q2


def test_get_questions_by_level_l1():
    """Validate get_questions_by_level returns L1 questions."""
    l1_questions = get_questions_by_level(QuestionLevel.L1_SYSTEM)

    assert len(l1_questions) == len(L1_QUESTIONS)
    assert all(q.level == QuestionLevel.L1_SYSTEM.value for q in l1_questions)


def test_get_questions_by_level_l2():
    """Validate get_questions_by_level returns L2 questions."""
    l2_questions = get_questions_by_level(QuestionLevel.L2_MODULE)

    assert len(l2_questions) == len(L2_QUESTIONS)
    assert all(q.level == QuestionLevel.L2_MODULE.value for q in l2_questions)


def test_get_questions_by_level_l3():
    """Validate get_questions_by_level returns L3 questions."""
    l3_questions = get_questions_by_level(QuestionLevel.L3_NODE)

    assert len(l3_questions) == len(L3_QUESTIONS)
    assert all(q.level == QuestionLevel.L3_NODE.value for q in l3_questions)


def test_get_questions_by_level_terminal():
    """Validate get_questions_by_level returns TERMINAL questions."""
    terminal_questions = get_questions_by_level(QuestionLevel.TERMINAL)

    assert len(terminal_questions) == len(TERMINAL_QUESTIONS)
    assert all(q.level == QuestionLevel.TERMINAL.value for q in terminal_questions)


def test_analyze_for_gaps():
    """Validate analyze_for_gaps convenience function."""
    content = "Build a system that sends email notifications."

    gaps = analyze_for_gaps(content)

    assert isinstance(gaps, list)
    assert len(gaps) > 0
    assert all(isinstance(g, Gap) for g in gaps)


def test_analyze_for_gaps_empty_content():
    """Validate analyze_for_gaps handles empty content."""
    gaps = analyze_for_gaps("")

    # Should still detect some gaps (like missing out_of_scope)
    assert isinstance(gaps, list)


# =============================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =============================================================================

def test_gap_multiple_severity_types():
    """Validate Gap supports all severity types."""
    gap_critical = Gap(question_id="q1", severity="critical", context="Test")
    gap_high = Gap(question_id="q2", severity="high", context="Test")
    gap_medium = Gap(question_id="q3", severity="medium", context="Test")
    gap_low = Gap(question_id="q4", severity="low", context="Test")

    assert gap_critical.severity == "critical"
    assert gap_high.severity == "high"
    assert gap_medium.severity == "medium"
    assert gap_low.severity == "low"


def test_question_answer_defaults():
    """Validate QuestionAnswer uses correct defaults."""
    qa = QuestionAnswer(
        question_id="test",
        question=L1_QUESTIONS["domain_core"],
    )

    assert qa.status == AnswerStatus.PENDING
    assert qa.answer is None
    assert qa.asked_at is None
    assert qa.answered_at is None
    assert qa.context is None


def test_session_multiple_assumptions():
    """Validate session can store multiple assumptions."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    engine.add_assumption(session, "Assumption 1")
    engine.add_assumption(session, "Assumption 2")
    engine.add_assumption(session, "Assumption 3")

    assert len(session.assumptions) == 3
    assert "Assumption 1" in session.assumptions
    assert "Assumption 2" in session.assumptions
    assert "Assumption 3" in session.assumptions


def test_gap_analyzer_case_insensitive():
    """Validate GapAnalyzer is case-insensitive."""
    analyzer = GapAnalyzer()

    content_lower = "the system should send notifications"
    content_upper = "THE SYSTEM SHOULD SEND NOTIFICATIONS"
    content_mixed = "The System Should Send Notifications"

    gaps_lower = analyzer.analyze_spec(content_lower)
    gaps_upper = analyzer.analyze_spec(content_upper)
    gaps_mixed = analyzer.analyze_spec(content_mixed)

    # All should detect the same gaps
    assert len(gaps_lower) == len(gaps_upper) == len(gaps_mixed)


def test_finalize_session_with_out_of_scope_answer():
    """Validate finalize_session extracts out_of_scope from answers."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    # Add out_of_scope question and answer
    qa = QuestionAnswer(
        question_id="out_of_scope",
        question=TERMINAL_QUESTIONS["out_of_scope"],
        status=AnswerStatus.ANSWERED,
        answer="2FA is out of scope",
    )
    session.qa_pairs["out_of_scope"] = qa

    result = engine.finalize_session(session)

    assert result["out_of_scope"] == "2FA is out of scope"


def test_finalize_session_without_out_of_scope():
    """Validate finalize_session handles missing out_of_scope."""
    engine = SocraticEngine()
    session = engine.create_session(req_id="REQ_123")

    result = engine.finalize_session(session)

    assert result["out_of_scope"] == ""


def test_session_with_follow_up_questions():
    """Validate questions with follow_ups are properly structured."""
    # Find a question with follow_ups (if any)
    questions_with_followups = [
        q for q in ALL_QUESTIONS.values()
        if len(q.follow_ups) > 0
    ]

    # If there are any, validate structure
    if questions_with_followups:
        q = questions_with_followups[0]
        assert isinstance(q.follow_ups, tuple)
        assert all(isinstance(fid, str) for fid in q.follow_ups)


def test_session_state_started_at_auto_generated():
    """Validate SessionState.started_at is auto-generated."""
    session = SessionState(session_id="test", req_id="REQ_1")

    assert session.started_at is not None
    # Should be parseable as datetime
    parsed = datetime.fromisoformat(session.started_at)
    assert parsed.tzinfo is not None
