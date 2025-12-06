"""
PARAGON HUMAN LOOP - Human-in-the-Loop Controller

Manages human interaction points in agent workflows.

Pause Points:
- APPROVAL: Binary yes/no decisions
- FEEDBACK: Free-form text input
- SELECTION: Choose from options
- REVIEW: Code/artifact review

Design:
- Declarative pause point definitions
- Request/response pattern
- Integration with LangGraph checkpointing
- Support for async and sync workflows

Based on legacy: gaadp-constructor/orchestration/human_loop.py
"""
from typing import List, Dict, Any, Optional, Callable, Awaitable
from enum import Enum
from datetime import datetime
import uuid
import msgspec


# =============================================================================
# TYPES AND ENUMS
# =============================================================================

class PauseType(str, Enum):
    """Types of pause points requiring human input."""
    APPROVAL = "approval"
    FEEDBACK = "feedback"
    SELECTION = "selection"
    REVIEW = "review"
    ESCALATION = "escalation"


class RequestStatus(str, Enum):
    """Status of a human input request."""
    PENDING = "pending"
    RESPONDED = "responded"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    """Priority levels for human requests."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class PausePoint(msgspec.Struct):
    """
    Definition of a pause point in the workflow.

    Pause points are declarative definitions of where
    human intervention may be needed.
    """
    id: str
    pause_type: str
    description: str
    required: bool = True
    timeout_seconds: Optional[int] = None
    options: Optional[List[str]] = None  # For SELECTION type
    default: Optional[str] = None
    metadata: Dict[str, Any] = {}


class HumanRequest(msgspec.Struct):
    """
    A request for human input.

    Created when a pause point is triggered and
    waiting for human response.
    """
    id: str
    pause_point_id: str
    session_id: str
    pause_type: str
    prompt: str
    context: Dict[str, Any]
    options: Optional[List[str]] = None
    priority: str = Priority.NORMAL.value
    status: str = RequestStatus.PENDING.value
    created_at: str = ""
    responded_at: Optional[str] = None
    response: Optional[str] = None
    timeout_seconds: Optional[int] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


class HumanResponse(msgspec.Struct):
    """
    A response from a human.

    Captures the human's decision, feedback, or selection.
    """
    request_id: str
    response: str
    responded_at: str = ""
    metadata: Dict[str, Any] = {}

    def __post_init__(self):
        if not self.responded_at:
            self.responded_at = datetime.utcnow().isoformat()


# =============================================================================
# PAUSE POINT REGISTRY
# =============================================================================

# Common pause points that can be referenced by ID
STANDARD_PAUSE_POINTS: Dict[str, PausePoint] = {
    "plan_approval": PausePoint(
        id="plan_approval",
        pause_type=PauseType.APPROVAL.value,
        description="Approve the implementation plan before proceeding",
        required=True,
    ),
    "code_review": PausePoint(
        id="code_review",
        pause_type=PauseType.REVIEW.value,
        description="Review generated code before testing",
        required=False,
    ),
    "test_failure_decision": PausePoint(
        id="test_failure_decision",
        pause_type=PauseType.SELECTION.value,
        description="Decide how to handle test failures",
        options=["retry", "modify_approach", "escalate", "abort"],
        default="retry",
    ),
    "final_approval": PausePoint(
        id="final_approval",
        pause_type=PauseType.APPROVAL.value,
        description="Final approval before completing the task",
        required=True,
    ),
    "clarification": PausePoint(
        id="clarification",
        pause_type=PauseType.FEEDBACK.value,
        description="Request clarification on requirements",
        required=True,
    ),
}


def get_pause_point(pause_point_id: str) -> Optional[PausePoint]:
    """Get a standard pause point by ID."""
    return STANDARD_PAUSE_POINTS.get(pause_point_id)


def register_pause_point(pause_point: PausePoint) -> None:
    """Register a custom pause point."""
    STANDARD_PAUSE_POINTS[pause_point.id] = pause_point


# =============================================================================
# HUMAN LOOP CONTROLLER
# =============================================================================

class HumanLoopController:
    """
    Controller for human-in-the-loop interactions.

    Manages the lifecycle of human input requests:
    1. Create request from pause point
    2. Wait for human response
    3. Process and validate response
    4. Resume workflow
    """

    def __init__(
        self,
        on_request: Optional[Callable[[HumanRequest], None]] = None,
        on_response: Optional[Callable[[HumanResponse], None]] = None,
    ):
        """
        Initialize the controller.

        Args:
            on_request: Callback when a request is created
            on_response: Callback when a response is received
        """
        self.pending_requests: Dict[str, HumanRequest] = {}
        self.completed_requests: Dict[str, HumanRequest] = {}
        self.on_request = on_request
        self.on_response = on_response

    def create_request(
        self,
        pause_point: PausePoint,
        session_id: str,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        priority: Priority = Priority.NORMAL,
    ) -> HumanRequest:
        """
        Create a new human input request.

        Args:
            pause_point: The pause point definition
            session_id: Current session ID
            prompt: The prompt to show the human
            context: Additional context
            priority: Request priority

        Returns:
            The created HumanRequest
        """
        request = HumanRequest(
            id=str(uuid.uuid4()),
            pause_point_id=pause_point.id,
            session_id=session_id,
            pause_type=pause_point.pause_type,
            prompt=prompt,
            context=context or {},
            options=pause_point.options,
            priority=priority.value,
            timeout_seconds=pause_point.timeout_seconds,
        )

        self.pending_requests[request.id] = request

        if self.on_request:
            self.on_request(request)

        return request

    def submit_response(
        self,
        request_id: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[HumanResponse]:
        """
        Submit a response to a pending request.

        Args:
            request_id: The request to respond to
            response: The human's response
            metadata: Optional metadata

        Returns:
            The created HumanResponse, or None if request not found
        """
        request = self.pending_requests.get(request_id)
        if not request:
            return None

        # Validate response for selection type
        if request.pause_type == PauseType.SELECTION.value:
            if request.options and response not in request.options:
                # Invalid selection - could raise or use default
                if request.options:
                    response = request.options[0]

        # Update request
        request.status = RequestStatus.RESPONDED.value
        request.responded_at = datetime.utcnow().isoformat()
        request.response = response

        # Move to completed
        del self.pending_requests[request_id]
        self.completed_requests[request_id] = request

        # Create response object
        human_response = HumanResponse(
            request_id=request_id,
            response=response,
            metadata=metadata or {},
        )

        if self.on_response:
            self.on_response(human_response)

        return human_response

    def get_pending_requests(
        self,
        session_id: Optional[str] = None,
    ) -> List[HumanRequest]:
        """
        Get pending requests, optionally filtered by session.

        Args:
            session_id: Optional session filter

        Returns:
            List of pending requests
        """
        requests = list(self.pending_requests.values())
        if session_id:
            requests = [r for r in requests if r.session_id == session_id]
        return requests

    def get_request(self, request_id: str) -> Optional[HumanRequest]:
        """Get a request by ID (pending or completed)."""
        return (
            self.pending_requests.get(request_id) or
            self.completed_requests.get(request_id)
        )

    def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a pending request.

        Args:
            request_id: The request to cancel

        Returns:
            True if cancelled, False if not found
        """
        request = self.pending_requests.get(request_id)
        if not request:
            return False

        request.status = RequestStatus.CANCELLED.value
        del self.pending_requests[request_id]
        self.completed_requests[request_id] = request
        return True

    def clear_session(self, session_id: str) -> int:
        """
        Clear all requests for a session.

        Args:
            session_id: Session to clear

        Returns:
            Number of requests cleared
        """
        to_remove = [
            rid for rid, req in self.pending_requests.items()
            if req.session_id == session_id
        ]
        for rid in to_remove:
            del self.pending_requests[rid]
        return len(to_remove)


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_approval_request(
    controller: HumanLoopController,
    session_id: str,
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
) -> HumanRequest:
    """
    Convenience function to create an approval request.

    Args:
        controller: The human loop controller
        session_id: Current session
        prompt: What to approve
        context: Additional context

    Returns:
        The created request
    """
    pause_point = PausePoint(
        id=f"approval_{uuid.uuid4().hex[:8]}",
        pause_type=PauseType.APPROVAL.value,
        description=prompt,
    )
    return controller.create_request(pause_point, session_id, prompt, context)


def create_feedback_request(
    controller: HumanLoopController,
    session_id: str,
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
) -> HumanRequest:
    """
    Convenience function to create a feedback request.

    Args:
        controller: The human loop controller
        session_id: Current session
        prompt: What feedback is needed
        context: Additional context

    Returns:
        The created request
    """
    pause_point = PausePoint(
        id=f"feedback_{uuid.uuid4().hex[:8]}",
        pause_type=PauseType.FEEDBACK.value,
        description=prompt,
    )
    return controller.create_request(pause_point, session_id, prompt, context)


def create_selection_request(
    controller: HumanLoopController,
    session_id: str,
    prompt: str,
    options: List[str],
    context: Optional[Dict[str, Any]] = None,
) -> HumanRequest:
    """
    Convenience function to create a selection request.

    Args:
        controller: The human loop controller
        session_id: Current session
        prompt: What to select
        options: Available options
        context: Additional context

    Returns:
        The created request
    """
    pause_point = PausePoint(
        id=f"selection_{uuid.uuid4().hex[:8]}",
        pause_type=PauseType.SELECTION.value,
        description=prompt,
        options=options,
    )
    return controller.create_request(pause_point, session_id, prompt, context)


# =============================================================================
# STATE INTEGRATION
# =============================================================================

def check_human_input_needed(state: Dict[str, Any]) -> bool:
    """
    Check if a state requires human input.

    Args:
        state: Orchestrator state

    Returns:
        True if human input is pending
    """
    return state.get("pending_human_input") is not None


def get_human_input_type(state: Dict[str, Any]) -> Optional[str]:
    """
    Get the type of human input needed.

    Args:
        state: Orchestrator state

    Returns:
        Pause type or None
    """
    return state.get("pending_human_input")


def apply_human_response(
    state: Dict[str, Any],
    response: str,
) -> Dict[str, Any]:
    """
    Apply a human response to state.

    Args:
        state: Current state
        response: Human response

    Returns:
        Updated state
    """
    return {
        **state,
        "human_response": response,
        "pending_human_input": None,
    }


# =============================================================================
# TRANSITION MATRIX
# =============================================================================

class TransitionMatrix(msgspec.Struct):
    """
    Declarative state machine as data.

    Defines valid transitions and conditions for the workflow.
    Can be used to validate state changes or generate documentation.
    """
    states: List[str]
    transitions: Dict[str, List[str]]  # from_state -> [to_states]
    pause_points: Dict[str, str]  # state -> pause_point_id (if any)
    terminal_states: List[str]


# Default TDD transition matrix
TDD_TRANSITIONS = TransitionMatrix(
    states=["init", "plan", "build", "test", "fix", "passed", "failed"],
    transitions={
        "init": ["plan"],
        "plan": ["build"],
        "build": ["test"],
        "test": ["passed", "fix"],
        "fix": ["build", "failed"],
        "passed": [],
        "failed": [],
    },
    pause_points={
        "plan": "plan_approval",
        "build": "code_review",
        "failed": "test_failure_decision",
    },
    terminal_states=["passed", "failed"],
)


def is_valid_transition(
    matrix: TransitionMatrix,
    from_state: str,
    to_state: str,
) -> bool:
    """
    Check if a state transition is valid.

    Args:
        matrix: The transition matrix
        from_state: Current state
        to_state: Target state

    Returns:
        True if transition is valid
    """
    valid_targets = matrix.transitions.get(from_state, [])
    return to_state in valid_targets


def get_pause_point_for_state(
    matrix: TransitionMatrix,
    state: str,
) -> Optional[PausePoint]:
    """
    Get the pause point for a state, if any.

    Args:
        matrix: The transition matrix
        state: Current state

    Returns:
        PausePoint if defined for this state
    """
    pause_point_id = matrix.pause_points.get(state)
    if pause_point_id:
        return get_pause_point(pause_point_id)
    return None
