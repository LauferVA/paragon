"""
INITIAL CONVERSATION - Startup experience for Paragon

Handles the initial conversation flow when no spec file is provided.
Guides users through initial questions to understand their project.
"""
import logging
from typing import List, Optional, Dict, Any
import msgspec

from agents.schemas import ParsedSpec
from agents.tools import add_node

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMAS
# =============================================================================

class ConversationQuestion(msgspec.Struct, kw_only=True, frozen=True):
    """A question in the initial conversation."""
    id: str
    question: str
    context: str = ""
    required: bool = True
    category: str = "general"  # "general", "technical", "user", "features"


class ConversationAnswer(msgspec.Struct, kw_only=True):
    """An answer to a conversation question."""
    question_id: str
    answer: str
    confidence: float = 1.0


class ConversationState(msgspec.Struct, kw_only=True):
    """State of the initial conversation."""
    questions_asked: List[str] = []
    answers: List[ConversationAnswer] = []
    current_question_index: int = 0
    is_complete: bool = False


# =============================================================================
# QUESTION BANK
# =============================================================================

STANDARD_QUESTIONS = [
    ConversationQuestion(
        id="project_goal",
        question="What do you want to build today?",
        context="Describe your project idea or what you're trying to accomplish.",
        required=True,
        category="general",
    ),
    ConversationQuestion(
        id="target_user",
        question="Who is the target user for this project?",
        context="Who will use this? What are their needs?",
        required=False,
        category="user",
    ),
    ConversationQuestion(
        id="must_have_features",
        question="What are the must-have features?",
        context="List the core features that are absolutely essential.",
        required=True,
        category="features",
    ),
    ConversationQuestion(
        id="technical_constraints",
        question="Are there any technical constraints or requirements?",
        context="Technology stack, performance requirements, platform constraints, etc.",
        required=False,
        category="technical",
    ),
    ConversationQuestion(
        id="success_criteria",
        question="How will you know when this is successful?",
        context="What does 'done' look like? What are your success metrics?",
        required=False,
        category="general",
    ),
]


# =============================================================================
# INITIAL CONVERSATION
# =============================================================================

class InitialConversation:
    """
    Manages the initial conversation flow for new projects.

    Guides users through standard questions to understand their project
    and converts answers into a structured ParsedSpec.
    """

    def __init__(self, questions: Optional[List[ConversationQuestion]] = None):
        """
        Initialize the conversation.

        Args:
            questions: Optional custom question list (uses STANDARD_QUESTIONS if not provided)
        """
        self.questions = questions or STANDARD_QUESTIONS
        self.state = ConversationState()

    def get_next_question(self) -> Optional[ConversationQuestion]:
        """
        Get the next question to ask.

        Returns:
            Next question or None if conversation is complete
        """
        if self.state.current_question_index >= len(self.questions):
            self.state.is_complete = True
            return None

        question = self.questions[self.state.current_question_index]
        return question

    def record_answer(self, question_id: str, answer: str) -> None:
        """
        Record an answer to a question.

        Args:
            question_id: ID of the question being answered
            answer: User's answer
        """
        self.state.answers.append(
            ConversationAnswer(
                question_id=question_id,
                answer=answer,
            )
        )
        self.state.questions_asked.append(question_id)
        self.state.current_question_index += 1

    def skip_question(self) -> None:
        """Skip the current question (for non-required questions)."""
        current = self.get_next_question()
        if current and not current.required:
            self.state.questions_asked.append(current.id)
            self.state.current_question_index += 1

    def is_complete(self) -> bool:
        """Check if the conversation is complete."""
        return self.state.is_complete

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current conversation progress.

        Returns:
            Dict with progress information
        """
        return {
            "total_questions": len(self.questions),
            "answered": len(self.state.answers),
            "current_index": self.state.current_question_index,
            "is_complete": self.state.is_complete,
            "completion_percentage": (len(self.state.answers) / len(self.questions)) * 100,
        }

    def build_spec_from_answers(self) -> ParsedSpec:
        """
        Build a ParsedSpec from the conversation answers.

        Returns:
            ParsedSpec suitable for initializing the orchestrator
        """
        # Extract answers by category
        answers_by_id = {a.question_id: a.answer for a in self.state.answers}

        # Extract project goal as title and description
        project_goal = answers_by_id.get("project_goal", "New Project")
        title_parts = project_goal.split('\n', 1)
        title = title_parts[0].strip()
        description = title_parts[1].strip() if len(title_parts) > 1 else project_goal

        # Build requirements from must-have features
        must_have = answers_by_id.get("must_have_features", "")
        requirements = [
            req.strip()
            for req in must_have.split('\n')
            if req.strip()
        ]

        # Extract features list
        features = requirements  # For now, features = requirements

        # Build technical details
        technical_details = answers_by_id.get("technical_constraints")
        success_criteria = answers_by_id.get("success_criteria")

        if success_criteria:
            if technical_details:
                technical_details += f"\n\nSuccess Criteria:\n{success_criteria}"
            else:
                technical_details = f"Success Criteria:\n{success_criteria}"

        # Build full description including all context
        full_description = description
        if answers_by_id.get("target_user"):
            full_description += f"\n\nTarget User: {answers_by_id['target_user']}"

        return ParsedSpec(
            title=title,
            description=full_description,
            requirements=requirements,
            technical_details=technical_details,
            target_user=answers_by_id.get("target_user"),
            must_have_features=features,
            constraints=[],
            raw_content=self._build_raw_content(),
            file_format="conversation",
        )

    def _build_raw_content(self) -> str:
        """Build raw content from all Q&A pairs."""
        lines = []
        answers_by_id = {a.question_id: a.answer for a in self.state.answers}

        for question in self.questions:
            if question.id in answers_by_id:
                lines.append(f"Q: {question.question}")
                lines.append(f"A: {answers_by_id[question.id]}")
                lines.append("")

        return '\n'.join(lines)

    def persist_to_graph(self, session_id: str) -> List[str]:
        """
        Persist the conversation to the graph as REQ nodes.

        Args:
            session_id: Current session ID

        Returns:
            List of created node IDs
        """
        node_ids = []
        answers_by_id = {a.question_id: a.answer for a in self.state.answers}

        # Create a REQ node for the main project goal
        if "project_goal" in answers_by_id:
            result = add_node(
                node_type="REQ",
                content=answers_by_id["project_goal"],
                data={
                    "category": "project_goal",
                    "session_id": session_id,
                    "from_conversation": True,
                },
                created_by="initial_conversation",
            )
            if result.success:
                node_ids.append(result.node_id)

        # Create REQ nodes for must-have features
        if "must_have_features" in answers_by_id:
            features = [
                f.strip()
                for f in answers_by_id["must_have_features"].split('\n')
                if f.strip()
            ]
            for feature in features:
                result = add_node(
                    node_type="REQ",
                    content=feature,
                    data={
                        "category": "feature",
                        "session_id": session_id,
                        "from_conversation": True,
                    },
                    created_by="initial_conversation",
                )
                if result.success:
                    node_ids.append(result.node_id)

        logger.info(f"Persisted {len(node_ids)} REQ nodes from initial conversation")
        return node_ids


# =============================================================================
# SPEC-BASED CONVERSATION STARTER
# =============================================================================

def determine_starting_phase(spec: ParsedSpec) -> str:
    """
    Determine which phase to start in based on the spec content.

    Args:
        spec: Parsed specification

    Returns:
        Phase name to start in ("dialectic", "research", "plan")
    """
    # If we have detailed requirements and technical details, can start at planning
    if spec.requirements and spec.technical_details and len(spec.requirements) >= 3:
        return "plan"

    # If we have some requirements but missing details, start at research
    if spec.requirements or spec.must_have_features:
        return "research"

    # Otherwise, start at dialectic to identify ambiguities
    return "dialectic"


def create_initial_message_for_spec(spec: ParsedSpec) -> str:
    """
    Create an initial message for the conversation when starting with a spec.

    Args:
        spec: Parsed specification

    Returns:
        Initial message string
    """
    phase = determine_starting_phase(spec)

    messages = {
        "dialectic": f"I've loaded your specification for '{spec.title}'. Let me analyze it for any ambiguities before we proceed...",
        "research": f"I've loaded your specification for '{spec.title}'. I have {len(spec.requirements)} requirement(s). Let me conduct research to create a detailed implementation plan...",
        "plan": f"I've loaded your detailed specification for '{spec.title}'. I have all the information I need. Let me create an implementation plan...",
    }

    return messages.get(phase, f"I've loaded your specification for '{spec.title}'. Let's get started...")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def start_fresh_conversation() -> InitialConversation:
    """
    Start a fresh conversation with standard questions.

    Returns:
        InitialConversation instance
    """
    return InitialConversation()


def get_greeting_message() -> str:
    """Get the initial greeting message for fresh starts."""
    return """Welcome to Paragon! I'm here to help you build high-quality software.

What would you like to build today?

(You can describe your project idea, or I can guide you through some questions to understand what you need.)"""
