"""
PARAGON SOCRATIC ENGINE - The Dialectic Specification System

AI-driven gap analysis and clarification for requirements.
Implements the Socratic method: questioning to surface hidden assumptions.

Architecture:
- CanonicalQuestion: Curated questions with level, category, conditions
- GapAnalyzer: Identifies missing information per feature
- SocraticEngine: Manages Q&A session state and flow
- QuestionLevel: L1 (System), L2 (Module), L3 (Node)

Design Philosophy:
1. QUESTION HIERARCHY: O(1) system questions, O(n) feature questions, O(m) conditional
2. STRUCTURAL VALIDATORS: Some questions only trigger on specific graph patterns
3. TERMINAL EDGE: Every spec must define out_of_scope (the Negative Space)
4. MINIMAL SUFFICIENT STATISTIC: Only ask what's necessary to disambiguate

Question Hierarchy:
- L1 (System): Asked once in intake. "What is the core domain?"
- L2 (Module): Asked per feature. "What triggers this feature?"
- L3 (Node): Conditional on validators. "What's the input provenance?"
- Terminal: Assumptions review + delegation license.
"""
import msgspec
from typing import Optional, Dict, Any, List, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

from core.ontology import NodeType, NodeStatus, EdgeType


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def now_utc() -> str:
    """Fast UTC timestamp as ISO8601 string."""
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# QUESTION LEVELS AND CATEGORIES
# =============================================================================

class QuestionLevel(str, Enum):
    """
    Hierarchy of question levels.

    L1: Asked once per system - O(1) cost
    L2: Asked per feature - O(n) cost where n = features
    L3: Conditional on structural validators - O(m) cost where m = triggered
    TERMINAL: Final review before delegation
    """
    L1_SYSTEM = "L1_SYSTEM"
    L2_MODULE = "L2_MODULE"
    L3_NODE = "L3_NODE"
    TERMINAL = "TERMINAL"


class QuestionCategory(str, Enum):
    """Categories for organizing questions."""
    SCOPE = "scope"                     # What's in/out of scope
    BEHAVIOR = "behavior"               # How should it behave
    DATA = "data"                       # Data structures and flow
    INTEGRATION = "integration"         # External dependencies
    CONSTRAINTS = "constraints"         # Limits and requirements
    ERROR_HANDLING = "error_handling"   # Failure modes
    SECURITY = "security"               # Auth, permissions
    PERFORMANCE = "performance"         # Speed, scale
    ASSUMPTIONS = "assumptions"         # Hidden assumptions


class AskCondition(str, Enum):
    """When to ask a question (structural triggers)."""
    ALWAYS = "always"                   # Always ask this question
    IF_AMBIGUOUS = "if_ambiguous"       # If spec has ambiguous terms
    IF_NO_SCHEMA = "if_no_schema"       # If I/O schema not derivable
    IF_EXTERNAL_DEPS = "if_external"    # If external dependencies
    IF_SIDE_EFFECTS = "if_side_effects" # If side effects mentioned
    IF_MISSING_PROVENANCE = "if_no_provenance"  # No input provenance
    IF_MISSING_SCOPE = "if_no_scope"    # No out_of_scope defined


# =============================================================================
# CANONICAL QUESTIONS
# =============================================================================

class CanonicalQuestion(msgspec.Struct, kw_only=True, frozen=True):
    """
    A curated question in the Socratic dialogue.

    Each question has:
    - A level (L1/L2/L3/Terminal)
    - A category for organization
    - An ask_when condition (when to trigger)
    - The question text itself
    - Optional follow-up questions
    """
    id: str                             # Unique identifier
    level: str                          # QuestionLevel.value
    category: str                       # QuestionCategory.value
    question: str                       # The question text
    rationale: str                      # Why this question matters
    ask_when: str = AskCondition.ALWAYS.value  # When to ask
    follow_ups: Tuple[str, ...] = ()    # Follow-up question IDs
    examples: Tuple[str, ...] = ()      # Example answers


# =============================================================================
# QUESTION REGISTRY
# =============================================================================

# L1 Questions - Asked once per system intake
L1_QUESTIONS: Dict[str, CanonicalQuestion] = {
    "domain_core": CanonicalQuestion(
        id="domain_core",
        level=QuestionLevel.L1_SYSTEM.value,
        category=QuestionCategory.SCOPE.value,
        question="What is the core business domain this system operates in?",
        rationale="Understanding the domain shapes all downstream decisions.",
        examples=("E-commerce", "Healthcare records", "Financial trading"),
    ),
    "primary_users": CanonicalQuestion(
        id="primary_users",
        level=QuestionLevel.L1_SYSTEM.value,
        category=QuestionCategory.SCOPE.value,
        question="Who are the primary users of this system?",
        rationale="User types determine interfaces and access patterns.",
        examples=("Internal employees", "External customers", "API consumers"),
    ),
    "scale_expectation": CanonicalQuestion(
        id="scale_expectation",
        level=QuestionLevel.L1_SYSTEM.value,
        category=QuestionCategory.PERFORMANCE.value,
        question="What scale is this system expected to handle (users, requests, data)?",
        rationale="Scale affects architecture, storage, and technology choices.",
        examples=("100 users/month", "1M requests/day", "10TB data"),
    ),
    "existing_systems": CanonicalQuestion(
        id="existing_systems",
        level=QuestionLevel.L1_SYSTEM.value,
        category=QuestionCategory.INTEGRATION.value,
        question="What existing systems must this integrate with?",
        rationale="Integration constraints shape API design and data formats.",
        ask_when=AskCondition.IF_EXTERNAL_DEPS.value,
    ),
    "security_requirements": CanonicalQuestion(
        id="security_requirements",
        level=QuestionLevel.L1_SYSTEM.value,
        category=QuestionCategory.SECURITY.value,
        question="What are the security and compliance requirements?",
        rationale="Security requirements affect architecture at every layer.",
        examples=("HIPAA", "SOC2", "GDPR", "Internal only"),
    ),
}

# L2 Questions - Asked per feature/module
L2_QUESTIONS: Dict[str, CanonicalQuestion] = {
    "trigger_exit": CanonicalQuestion(
        id="trigger_exit",
        level=QuestionLevel.L2_MODULE.value,
        category=QuestionCategory.BEHAVIOR.value,
        question="What triggers this feature, and what is the exit condition?",
        rationale="Clear boundaries prevent scope creep and define done.",
        examples=(
            "Triggered by user click, exits when data saved",
            "Triggered by cron job, exits when all records processed",
        ),
    ),
    "cardinality_latency": CanonicalQuestion(
        id="cardinality_latency",
        level=QuestionLevel.L2_MODULE.value,
        category=QuestionCategory.BEHAVIOR.value,
        question="Is this synchronous or async? Batch or stream? Single or bulk?",
        rationale="Cardinality and latency shape the implementation pattern.",
        examples=("Sync single request", "Async batch job", "Streaming pipeline"),
    ),
    "dependencies_dependents": CanonicalQuestion(
        id="dependencies_dependents",
        level=QuestionLevel.L2_MODULE.value,
        category=QuestionCategory.INTEGRATION.value,
        question="What does this feature depend on, and what depends on it?",
        rationale="Dependency graph determines build order and failure propagation.",
    ),
    "data_ownership": CanonicalQuestion(
        id="data_ownership",
        level=QuestionLevel.L2_MODULE.value,
        category=QuestionCategory.DATA.value,
        question="Is the data sovereign (owned) or borrowed (from another service)?",
        rationale="Ownership determines storage, caching, and consistency strategy.",
        examples=("Sovereign - we are source of truth", "Borrowed - cached from API"),
    ),
    "fault_isolation": CanonicalQuestion(
        id="fault_isolation",
        level=QuestionLevel.L2_MODULE.value,
        category=QuestionCategory.ERROR_HANDLING.value,
        question="If this feature fails, what is the blast radius and recovery path?",
        rationale="Fault isolation prevents cascading failures.",
        ask_when=AskCondition.IF_AMBIGUOUS.value,
    ),
}

# L3 Questions - Conditional on structural validators
L3_QUESTIONS: Dict[str, CanonicalQuestion] = {
    "io_schema": CanonicalQuestion(
        id="io_schema",
        level=QuestionLevel.L3_NODE.value,
        category=QuestionCategory.DATA.value,
        question="What is the exact input and output schema for this operation?",
        rationale="Explicit schemas enable type checking and validation.",
        ask_when=AskCondition.IF_NO_SCHEMA.value,
        examples=('{"user_id": "string", "amount": "decimal"}'),
    ),
    "side_effects": CanonicalQuestion(
        id="side_effects",
        level=QuestionLevel.L3_NODE.value,
        category=QuestionCategory.BEHAVIOR.value,
        question="What side effects does this operation have, and is it idempotent?",
        rationale="Side effects and idempotency affect retry and consistency.",
        ask_when=AskCondition.IF_SIDE_EFFECTS.value,
    ),
    "input_provenance": CanonicalQuestion(
        id="input_provenance",
        level=QuestionLevel.L3_NODE.value,
        category=QuestionCategory.DATA.value,
        question="Where does the input data originate, and how is it validated?",
        rationale="STRUCTURAL_VALIDATOR: Never guess input provenance!",
        ask_when=AskCondition.IF_MISSING_PROVENANCE.value,
    ),
    "implicit_deps": CanonicalQuestion(
        id="implicit_deps",
        level=QuestionLevel.L3_NODE.value,
        category=QuestionCategory.INTEGRATION.value,
        question="Are there implicit dependencies not visible in the spec (env vars, config)?",
        rationale="STRUCTURAL_VALIDATOR: Hidden dependencies cause deployment failures.",
        ask_when=AskCondition.ALWAYS.value,
    ),
}

# Terminal Questions - Final review before delegation
TERMINAL_QUESTIONS: Dict[str, CanonicalQuestion] = {
    "out_of_scope": CanonicalQuestion(
        id="out_of_scope",
        level=QuestionLevel.TERMINAL.value,
        category=QuestionCategory.SCOPE.value,
        question="What is explicitly OUT OF SCOPE for this requirement?",
        rationale="The Terminal Edge: define the Negative Space.",
        ask_when=AskCondition.IF_MISSING_SCOPE.value,
        examples=(
            "2FA is out of scope for v1",
            "Mobile support is out of scope",
            "Real-time sync is out of scope",
        ),
    ),
    "assumptions_review": CanonicalQuestion(
        id="assumptions_review",
        level=QuestionLevel.TERMINAL.value,
        category=QuestionCategory.ASSUMPTIONS.value,
        question="Do you agree with these assumptions? [List assumptions]",
        rationale="Surface hidden assumptions before they become bugs.",
    ),
    "delegation_license": CanonicalQuestion(
        id="delegation_license",
        level=QuestionLevel.TERMINAL.value,
        category=QuestionCategory.SCOPE.value,
        question="May I proceed with implementation based on this specification?",
        rationale="Explicit delegation prevents scope disputes.",
    ),
}

# All questions indexed by ID
ALL_QUESTIONS: Dict[str, CanonicalQuestion] = {
    **L1_QUESTIONS,
    **L2_QUESTIONS,
    **L3_QUESTIONS,
    **TERMINAL_QUESTIONS,
}


# =============================================================================
# GAP ANALYZER
# =============================================================================

@dataclass
class Gap:
    """A gap identified in the specification."""
    question_id: str                    # Question that would fill this gap
    severity: str                       # "critical", "high", "medium", "low"
    context: str                        # Where the gap was detected
    suggestion: Optional[str] = None    # Suggested resolution


class GapAnalyzer:
    """
    Identifies missing information in a specification.

    Analyzes requirement/spec content against the canonical questions
    to find gaps that need clarification.

    Usage:
        analyzer = GapAnalyzer()
        gaps = analyzer.analyze_spec(spec_content, spec_data)
    """

    def __init__(self):
        """Initialize the analyzer with detection patterns."""
        # Patterns that indicate specific gaps
        self._ambiguous_terms = {
            "should", "might", "could", "possibly", "maybe",
            "appropriate", "reasonable", "as needed", "etc",
            "various", "some", "many", "few", "large", "small",
        }
        self._side_effect_terms = {
            "send", "notify", "email", "webhook", "call",
            "update", "write", "save", "delete", "create",
            "log", "audit", "track", "record",
        }
        self._external_terms = {
            "api", "service", "endpoint", "webhook", "integration",
            "third-party", "external", "oauth", "stripe", "aws",
        }

    def analyze_spec(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Gap]:
        """
        Analyze a specification for gaps.

        Args:
            content: The spec content text
            metadata: Optional metadata from the spec node

        Returns:
            List of identified gaps
        """
        gaps = []
        content_lower = content.lower()
        metadata = metadata or {}

        # Check for ambiguous terms
        found_ambiguous = [
            term for term in self._ambiguous_terms
            if term in content_lower
        ]
        if found_ambiguous:
            gaps.append(Gap(
                question_id="fault_isolation",
                severity="medium",
                context=f"Ambiguous terms found: {', '.join(found_ambiguous)}",
                suggestion="Replace with specific values or ranges",
            ))

        # Check for side effects
        found_side_effects = [
            term for term in self._side_effect_terms
            if term in content_lower
        ]
        if found_side_effects:
            gaps.append(Gap(
                question_id="side_effects",
                severity="high",
                context=f"Side effects implied: {', '.join(found_side_effects)}",
                suggestion="Document side effects and idempotency",
            ))

        # Check for external dependencies
        found_external = [
            term for term in self._external_terms
            if term in content_lower
        ]
        if found_external:
            gaps.append(Gap(
                question_id="dependencies_dependents",
                severity="high",
                context=f"External dependencies: {', '.join(found_external)}",
                suggestion="Document integration requirements",
            ))

        # Check for missing out_of_scope
        if "out of scope" not in content_lower and "out-of-scope" not in content_lower:
            if not metadata.get("out_of_scope"):
                gaps.append(Gap(
                    question_id="out_of_scope",
                    severity="critical",
                    context="No out_of_scope defined",
                    suggestion="Define the Terminal Edge (Negative Space)",
                ))

        # Check for missing trigger/exit
        has_trigger = any(term in content_lower for term in ["when", "trigger", "on"])
        has_exit = any(term in content_lower for term in ["until", "complete", "finish", "done"])
        if not (has_trigger and has_exit):
            gaps.append(Gap(
                question_id="trigger_exit",
                severity="high",
                context="Trigger or exit condition unclear",
                suggestion="Define what starts and ends this feature",
            ))

        return sorted(gaps, key=lambda g: {
            "critical": 0, "high": 1, "medium": 2, "low": 3
        }.get(g.severity, 4))

    def get_questions_for_gaps(self, gaps: List[Gap]) -> List[CanonicalQuestion]:
        """Get the canonical questions that would fill the gaps."""
        question_ids = set(gap.question_id for gap in gaps)
        return [ALL_QUESTIONS[qid] for qid in question_ids if qid in ALL_QUESTIONS]


# =============================================================================
# SOCRATIC SESSION STATE
# =============================================================================

class AnswerStatus(str, Enum):
    """Status of an answer in the session."""
    PENDING = "pending"         # Not yet asked
    ASKED = "asked"             # Question presented to user
    ANSWERED = "answered"       # User provided answer
    SKIPPED = "skipped"         # User skipped the question
    INFERRED = "inferred"       # Answer inferred from context


@dataclass
class QuestionAnswer:
    """A question-answer pair in the session."""
    question_id: str
    question: CanonicalQuestion
    status: AnswerStatus = AnswerStatus.PENDING
    answer: Optional[str] = None
    asked_at: Optional[str] = None
    answered_at: Optional[str] = None
    context: Optional[str] = None       # Where this question arose


@dataclass
class SessionState:
    """Complete state of a Socratic session."""
    session_id: str
    req_id: str                         # REQ node being refined
    started_at: str = field(default_factory=now_utc)
    completed_at: Optional[str] = None
    current_level: str = QuestionLevel.L1_SYSTEM.value
    qa_pairs: Dict[str, QuestionAnswer] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    inferred_answers: Dict[str, str] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if all required questions are answered."""
        for qa in self.qa_pairs.values():
            if qa.status == AnswerStatus.PENDING:
                return False
        return True

    @property
    def completion_rate(self) -> float:
        """Percentage of questions answered."""
        if not self.qa_pairs:
            return 0.0
        answered = sum(
            1 for qa in self.qa_pairs.values()
            if qa.status in (AnswerStatus.ANSWERED, AnswerStatus.INFERRED, AnswerStatus.SKIPPED)
        )
        return answered / len(self.qa_pairs)


# =============================================================================
# SOCRATIC ENGINE
# =============================================================================

class SocraticEngine:
    """
    Manages the Socratic dialogue for requirement clarification.

    Drives the Q&A flow from L1 (system) through L2 (feature)
    to L3 (conditional) and Terminal (final review).

    Usage:
        engine = SocraticEngine()
        session = engine.create_session(req_id)

        # L1 Questions
        questions = engine.get_l1_questions(session)
        for q in questions:
            engine.record_answer(session, q.id, user_answer)

        # Analyze for gaps
        gaps = engine.analyze_spec(session, spec_content)
        additional = engine.get_gap_questions(session, gaps)

        # Terminal
        engine.finalize_session(session, assumptions=[...])
    """

    def __init__(self):
        """Initialize the engine."""
        self._sessions: Dict[str, SessionState] = {}
        self._gap_analyzer = GapAnalyzer()

    def create_session(self, req_id: str, session_id: Optional[str] = None) -> SessionState:
        """
        Create a new Socratic session for a requirement.

        Args:
            req_id: The REQ node ID to refine
            session_id: Optional custom session ID

        Returns:
            New SessionState
        """
        import uuid
        session_id = session_id or uuid.uuid4().hex

        session = SessionState(
            session_id=session_id,
            req_id=req_id,
        )

        # Initialize with L1 questions
        for qid, question in L1_QUESTIONS.items():
            session.qa_pairs[qid] = QuestionAnswer(
                question_id=qid,
                question=question,
            )

        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get an existing session."""
        return self._sessions.get(session_id)

    # =========================================================================
    # QUESTION FLOW
    # =========================================================================

    def get_pending_questions(self, session: SessionState) -> List[CanonicalQuestion]:
        """Get all pending questions for the current level."""
        pending = []
        for qa in session.qa_pairs.values():
            if qa.status == AnswerStatus.PENDING:
                if qa.question.level == session.current_level:
                    pending.append(qa.question)
        return pending

    def get_next_question(self, session: SessionState) -> Optional[CanonicalQuestion]:
        """Get the next unanswered question."""
        pending = self.get_pending_questions(session)
        return pending[0] if pending else None

    def record_answer(
        self,
        session: SessionState,
        question_id: str,
        answer: str,
        inferred: bool = False,
    ) -> None:
        """
        Record an answer to a question.

        Args:
            session: The session to update
            question_id: ID of the question being answered
            answer: The answer text
            inferred: Whether the answer was inferred (not user-provided)
        """
        if question_id in session.qa_pairs:
            qa = session.qa_pairs[question_id]
            qa.answer = answer
            qa.status = AnswerStatus.INFERRED if inferred else AnswerStatus.ANSWERED
            qa.answered_at = now_utc()

            if inferred:
                session.inferred_answers[question_id] = answer

    def skip_question(self, session: SessionState, question_id: str) -> None:
        """Mark a question as skipped."""
        if question_id in session.qa_pairs:
            session.qa_pairs[question_id].status = AnswerStatus.SKIPPED

    def advance_level(self, session: SessionState) -> str:
        """
        Advance to the next question level.

        Returns the new level.
        """
        level_order = [
            QuestionLevel.L1_SYSTEM.value,
            QuestionLevel.L2_MODULE.value,
            QuestionLevel.L3_NODE.value,
            QuestionLevel.TERMINAL.value,
        ]

        current_idx = level_order.index(session.current_level)
        if current_idx < len(level_order) - 1:
            session.current_level = level_order[current_idx + 1]

            # Add questions for the new level
            if session.current_level == QuestionLevel.L2_MODULE.value:
                for qid, q in L2_QUESTIONS.items():
                    if qid not in session.qa_pairs:
                        session.qa_pairs[qid] = QuestionAnswer(question_id=qid, question=q)

            elif session.current_level == QuestionLevel.TERMINAL.value:
                for qid, q in TERMINAL_QUESTIONS.items():
                    if qid not in session.qa_pairs:
                        session.qa_pairs[qid] = QuestionAnswer(question_id=qid, question=q)

        return session.current_level

    # =========================================================================
    # GAP ANALYSIS
    # =========================================================================

    def analyze_spec(
        self,
        session: SessionState,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Gap]:
        """
        Analyze a spec for gaps and add L3 questions as needed.

        Args:
            session: The current session
            content: Spec content to analyze
            metadata: Optional spec metadata

        Returns:
            List of identified gaps
        """
        gaps = self._gap_analyzer.analyze_spec(content, metadata)

        # Add L3 questions for gaps
        for gap in gaps:
            if gap.question_id in L3_QUESTIONS:
                qid = gap.question_id
                if qid not in session.qa_pairs:
                    session.qa_pairs[qid] = QuestionAnswer(
                        question_id=qid,
                        question=L3_QUESTIONS[qid],
                        context=gap.context,
                    )

        return gaps

    def get_critical_gaps(self, session: SessionState) -> List[Gap]:
        """Get gaps that are critical and unanswered."""
        # Re-analyze with current state
        gaps = []
        for qa in session.qa_pairs.values():
            if qa.status == AnswerStatus.PENDING and qa.context:
                question = qa.question
                if question.ask_when == AskCondition.IF_MISSING_SCOPE.value:
                    gaps.append(Gap(
                        question_id=qa.question_id,
                        severity="critical",
                        context=qa.context or "Missing scope",
                    ))
        return gaps

    # =========================================================================
    # FINALIZATION
    # =========================================================================

    def add_assumption(self, session: SessionState, assumption: str) -> None:
        """Add an assumption to the session."""
        session.assumptions.append(assumption)

    def finalize_session(
        self,
        session: SessionState,
        approved: bool = True,
    ) -> Dict[str, Any]:
        """
        Finalize the session and generate the clarified specification.

        Args:
            session: The session to finalize
            approved: Whether the user approved proceeding

        Returns:
            Dictionary with session summary and gathered information
        """
        session.completed_at = now_utc()

        # Collect all answers
        answers = {}
        for qid, qa in session.qa_pairs.items():
            if qa.answer:
                answers[qid] = qa.answer

        # Build the clarified spec data
        return {
            "session_id": session.session_id,
            "req_id": session.req_id,
            "approved": approved,
            "completion_rate": session.completion_rate,
            "answers": answers,
            "assumptions": session.assumptions,
            "inferred": session.inferred_answers,
            "out_of_scope": answers.get("out_of_scope", ""),
            "started_at": session.started_at,
            "completed_at": session.completed_at,
        }

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def get_session_summary(self, session: SessionState) -> Dict[str, Any]:
        """Get a summary of the session state."""
        by_status = {
            AnswerStatus.PENDING.value: 0,
            AnswerStatus.ANSWERED.value: 0,
            AnswerStatus.SKIPPED.value: 0,
            AnswerStatus.INFERRED.value: 0,
        }
        for qa in session.qa_pairs.values():
            by_status[qa.status.value] = by_status.get(qa.status.value, 0) + 1

        return {
            "session_id": session.session_id,
            "req_id": session.req_id,
            "current_level": session.current_level,
            "total_questions": len(session.qa_pairs),
            "by_status": by_status,
            "completion_rate": session.completion_rate,
            "assumptions_count": len(session.assumptions),
            "is_complete": session.is_complete,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_all_questions() -> Dict[str, CanonicalQuestion]:
    """Get all canonical questions."""
    return ALL_QUESTIONS.copy()


def get_questions_by_level(level: QuestionLevel) -> List[CanonicalQuestion]:
    """Get questions for a specific level."""
    return [q for q in ALL_QUESTIONS.values() if q.level == level.value]


def analyze_for_gaps(content: str) -> List[Gap]:
    """Convenience function to analyze content for gaps."""
    analyzer = GapAnalyzer()
    return analyzer.analyze_spec(content)
