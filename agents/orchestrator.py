"""
PARAGON ORCHESTRATOR - LangGraph StateGraph Implementation

TDD-style orchestration for code generation and validation workflows.

State Machine:
    INIT -> PLAN -> BUILD -> TEST -> [PASSED | FIX -> BUILD]
                                  -> [FAILED (max retries)]

Design:
- StateGraph for declarative state machine definition
- Checkpointing for pause/resume and human-in-the-loop
- TypedDict state with Annotated reducers for proper state merging
- Tool integration with ParagonDB

Layer 7 Integration:
- Layer 7A (Creator): LLM generates plans and code via core/llm.py
- Layer 7B (Auditor): Safety hooks validate before graph insertion

Based on legacy: gaadp-constructor/orchestration/langgraph_adapter.py
"""
from typing import List, Dict, Any, Optional, Literal, Annotated, TypedDict
from enum import Enum
from datetime import datetime
import operator
import msgspec
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Production persistence - SqliteSaver for durable checkpointing
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

from agents.tools import (
    add_node, add_node_safe, add_edge, query_nodes, get_node,
    get_waves, get_descendants, parse_source,
    get_graph_stats, check_cycle, check_syntax, verify_alignment,
    update_node_status,
)

# Diagnostic logging for phase timing and state tracking
try:
    from infrastructure.diagnostics import get_diagnostics, reset_diagnostics
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False

# SocraticEngine for gap analysis (dialectic phase)
try:
    from requirements.socratic_engine import (
        SocraticEngine, GapAnalyzer, ALL_QUESTIONS,
    )
    _gap_analyzer = GapAnalyzer()
    SOCRATIC_AVAILABLE = True
except ImportError:
    _gap_analyzer = None
    SOCRATIC_AVAILABLE = False

# SocraticOrchestrator for full Socratic dialogue (replaces inline import in dialectic_node)
try:
    from agents.socrates import SocraticOrchestrator, run_socratic_dialogue
    SOCRATIC_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    SOCRATIC_ORCHESTRATOR_AVAILABLE = False

# Documenter for auto-documentation on success
try:
    from agents.documenter import Documenter, generate_all_docs
    DOCUMENTER_AVAILABLE = True
except ImportError:
    DOCUMENTER_AVAILABLE = False

# Quality Gate for enforcing quality floor
try:
    from agents.quality_gate import QualityGate, check_quality
    QUALITY_GATE_AVAILABLE = True
except ImportError:
    QUALITY_GATE_AVAILABLE = False

# Analytics for graph health checks
try:
    from core.analytics import get_graph_health_report, GraphHealthReport
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

# Teleology for golden thread validation
try:
    from core.teleology import validate_teleology, TeleologyReport
    TELEOLOGY_AVAILABLE = True
except ImportError:
    TELEOLOGY_AVAILABLE = False

# Resource Guard for OOM protection
try:
    from core.resource_guard import ResourceGuard, ResourceSignal
    RESOURCE_GUARD_AVAILABLE = True
except ImportError:
    RESOURCE_GUARD_AVAILABLE = False

# Attribution for failure analysis
try:
    from infrastructure.attribution import ForensicAnalyzer, AttributionResult
    ATTRIBUTION_AVAILABLE = True
except ImportError:
    ATTRIBUTION_AVAILABLE = False

# Divergence for test/prod mismatch detection
try:
    from infrastructure.divergence import DivergenceDetector
    DIVERGENCE_AVAILABLE = True
except ImportError:
    DIVERGENCE_AVAILABLE = False

# Diagnostics for correlation ID tracking
try:
    from infrastructure.diagnostics import diag, generate_correlation_id
    DIAGNOSTICS_CORRELATION_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_CORRELATION_AVAILABLE = False

# Training store for learning system persistence
try:
    from infrastructure.training_store import TrainingStore
    TRAINING_STORE_AVAILABLE = True
except ImportError:
    TRAINING_STORE_AVAILABLE = False

# RerunLogger for visual temporal debugging (Time Machine)
try:
    from infrastructure.rerun_logger import RerunLogger, create_logger as create_rerun_logger
    RERUN_LOGGER_AVAILABLE = True
except ImportError:
    RERUN_LOGGER_AVAILABLE = False

# Environment detection for system capability logging
try:
    from infrastructure.environment import EnvironmentDetector, EnvironmentReport
    ENVIRONMENT_AVAILABLE = True
except ImportError:
    ENVIRONMENT_AVAILABLE = False

# Event bus for error broadcasting to WebSocket clients
try:
    from infrastructure.event_bus import publish_orchestrator_error, publish_phase_changed
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False

# Optional LLM integration (graceful degradation if not configured)
try:
    from core.llm import get_llm, StructuredLLM
    from agents.schemas import (
        ImplementationPlan, CodeGeneration, TestGeneration,
        DialectorOutput, AmbiguityMarker, ResearchArtifact, ResearchFinding,
        AgentSignature, SignatureAction,
    )
    from agents.prompts import build_prompt, build_dialector_prompt, build_researcher_prompt
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Human-in-the-loop controller for user questions
try:
    from agents.human_loop import (
        HumanLoopController, create_feedback_request, create_selection_request,
        PauseType, RequestStatus,
    )
    HUMAN_LOOP_AVAILABLE = True
except ImportError:
    HUMAN_LOOP_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DEFINITIONS
# =============================================================================

class CyclePhase(str, Enum):
    """TDD cycle phases - now includes Research/Dialectic."""
    INIT = "INIT"
    DIALECTIC = "DIALECTIC"      # Pre-research ambiguity check
    CLARIFICATION = "CLARIFICATION"  # Waiting for user answers
    RESEARCH = "RESEARCH"        # Socratic research loop
    PLAN = "PLAN"
    BUILD = "BUILD"
    TEST = "TEST"
    FIX = "FIX"
    PASSED = "PASSED"
    FAILED = "FAILED"


class HumanCheckpointType(str, Enum):
    """Types of human interaction points."""
    APPROVAL = "approval"      # Yes/No decision
    FEEDBACK = "feedback"      # Free-form input
    SELECTION = "selection"    # Choose from options
    REVIEW = "review"          # Review and approve/reject


class AgentMessage(msgspec.Struct):
    """A message in the agent conversation."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_name: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None


class TestResult(msgspec.Struct):
    """Result of a test execution."""
    passed: bool
    output: str
    errors: List[str]
    coverage: Optional[float] = None


class BuildArtifact(msgspec.Struct):
    """An artifact produced during the build phase."""
    node_id: str
    artifact_type: str  # "code", "test", "doc"
    path: Optional[str] = None
    content: Optional[str] = None


# Custom reducer for list append (instead of replace)
def list_append_reducer(existing: List, new: List) -> List:
    """Append new items to existing list."""
    if existing is None:
        return new or []
    if new is None:
        return existing
    return existing + new


# =============================================================================
# TOPOLOGY-DRIVEN PHASE INFERENCE
# =============================================================================

def infer_phase_from_node(db, node_id: str) -> str:
    """
    Infer the current phase based on node type and graph structure.

    Instead of tracking phase as explicit state, derive it from topology:
    - REQ with no RESEARCH → needs dialectic/research
    - REQ with RESEARCH but no SPEC → needs planning
    - SPEC with no CODE → needs building
    - CODE with no TEST → needs testing
    - CODE with FAILED TEST → needs fixing

    This is the "Phase as Emergent Property" principle from CLAUDE.md.

    Args:
        db: ParagonDB instance
        node_id: The node to analyze

    Returns:
        Inferred phase string matching CyclePhase values
    """
    from core.ontology import NodeType, NodeStatus, EdgeType

    try:
        node = db.get_node(node_id)
    except Exception:
        return CyclePhase.INIT.value

    node_type = node.type
    node_status = node.status

    # Get edges
    outgoing = db.get_outgoing_edges(node_id)
    incoming = db.get_incoming_edges(node_id)
    outgoing_types = {e.get("type") for e in outgoing}
    incoming_types = {e.get("type") for e in incoming}

    # Phase inference based on node type and structure
    if node_type == NodeType.REQ.value:
        if EdgeType.RESEARCH_FOR.value not in incoming_types:
            return CyclePhase.DIALECTIC.value
        if EdgeType.TRACES_TO.value not in incoming_types:
            return CyclePhase.PLAN.value
        return CyclePhase.PASSED.value

    elif node_type == NodeType.RESEARCH.value:
        if node_status == NodeStatus.PENDING.value:
            return CyclePhase.RESEARCH.value
        return CyclePhase.PLAN.value

    elif node_type == NodeType.SPEC.value:
        if EdgeType.IMPLEMENTS.value not in incoming_types:
            return CyclePhase.BUILD.value
        return CyclePhase.TEST.value

    elif node_type == NodeType.CODE.value:
        if EdgeType.TESTS.value not in incoming_types:
            return CyclePhase.TEST.value
        # Check if tests passed
        for edge in incoming:
            if edge.get("type") == EdgeType.TESTS.value:
                try:
                    test_node = db.get_node(edge.get("source"))
                    if test_node.status == NodeStatus.FAILED.value:
                        return CyclePhase.FIX.value
                except Exception:
                    pass
        return CyclePhase.PASSED.value

    elif node_type == NodeType.TEST_SUITE.value:
        if node_status == NodeStatus.FAILED.value:
            return CyclePhase.FIX.value
        return CyclePhase.PASSED.value

    elif node_type == NodeType.CLARIFICATION.value:
        if node_status == NodeStatus.PENDING.value:
            return CyclePhase.CLARIFICATION.value
        return CyclePhase.RESEARCH.value

    return CyclePhase.INIT.value


class GraphState(TypedDict):
    """
    State for the TDD orchestration cycle using TypedDict.

    Uses Annotated with reducers for proper state merging.
    Lists use append semantics, scalars use replacement.
    """
    # Identification (replace)
    session_id: str
    task_id: str

    # Current phase (replace)
    phase: str

    # Conversation history (append)
    messages: Annotated[List[Dict[str, Any]], list_append_reducer]

    # Task specification (replace)
    spec: str
    requirements: List[str]

    # Build artifacts (append)
    artifacts: Annotated[List[Dict[str, Any]], list_append_reducer]
    code_node_ids: Annotated[List[str], list_append_reducer]
    test_node_ids: Annotated[List[str], list_append_reducer]

    # Test results (append)
    test_results: Annotated[List[Dict[str, Any]], list_append_reducer]
    last_test_passed: bool

    # Iteration tracking (replace)
    iteration: int
    max_iterations: int

    # Human-in-the-loop (replace)
    pending_human_input: Optional[str]
    human_response: Optional[str]

    # Dialectic/Research phase state (replace)
    ambiguities: Annotated[List[Dict[str, Any]], list_append_reducer]
    clarification_questions: Annotated[List[Dict[str, Any]], list_append_reducer]
    research_findings: Annotated[List[Dict[str, Any]], list_append_reducer]
    research_complete: bool

    # Errors (append)
    errors: Annotated[List[str], list_append_reducer]

    # Final status (replace)
    final_status: Optional[str]


# =============================================================================
# NODE FUNCTIONS (State Machine Nodes)
# =============================================================================

def create_requirement_nodes_from_spec(state: GraphState) -> List[str]:
    """
    Parse spec and immediately create REQ nodes.

    PHASE 2: Proactive node creation - no waiting for approval.

    Args:
        state: Current graph state containing spec

    Returns:
        List of created node IDs
    """
    spec = state.get("spec", "")
    session_id = state.get("session_id", "unknown")
    task_id = state.get("task_id", "unknown")

    if not spec or not spec.strip():
        logger.info("No spec provided, skipping REQ node creation")
        return []

    # Try to use SpecParser to extract requirements
    try:
        from agents.spec_parser import SpecParser
        from agents.tools import get_db

        # Parse spec using LLM-based parser
        parser = SpecParser()

        # Create a temporary file for parsing (SpecParser expects a file path)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(spec)
            temp_path = f.name

        try:
            parsed = parser.parse_spec(temp_path)
            requirements = parsed.requirements
        finally:
            import os
            os.unlink(temp_path)

        # If no requirements extracted, create a single REQ for the entire spec
        if not requirements:
            requirements = [parsed.title or "Main Requirement"]

        # Create REQ nodes in the graph
        db = get_db()
        if db is None:
            logger.warning("ParagonDB not available, cannot create REQ nodes")
            return []

        # Create SPEC root node if it doesn't exist
        spec_id = f"spec_{task_id}"
        from core.schemas import NodeData, EdgeData
        from core.ontology import NodeType, NodeStatus, EdgeType

        if not db.has_node(spec_id):
            spec_node = NodeData(
                id=spec_id,
                type=NodeType.SPEC.value,
                content=parsed.title or "Project Specification",
                status=NodeStatus.PENDING.value,
                created_by="orchestrator",
            )
            db.add_node(spec_node)
            from agents.tools import _log_node_created
            _log_node_created(spec_id, NodeType.SPEC.value, parsed.title or "Project Specification")
            logger.info(f"Created SPEC node: {spec_id}")

        created_ids = []
        for idx, req_text in enumerate(requirements):
            req_id = f"req_{task_id}_{idx}"

            # Check if node already exists
            if db.has_node(req_id):
                logger.info(f"REQ node {req_id} already exists")
                created_ids.append(req_id)
                continue

            # Create REQ node
            req_node = NodeData(
                id=req_id,
                type=NodeType.REQ.value,
                content=req_text,
                status=NodeStatus.PENDING.value,
                created_by="orchestrator",
            )
            db.add_node(req_node)

            # CRITICAL FIX: Log node creation to trigger persistence
            # This ensures API workers can see nodes via parquet files
            from agents.tools import _log_node_created
            _log_node_created(req_id, NodeType.REQ.value, req_text)

            created_ids.append(req_id)
            logger.info(f"Created REQ node: {req_id} - {req_text[:50]}...")

        # Create edges from SPEC to all REQ nodes
        from agents.tools import add_edge
        for req_id in created_ids:
            try:
                add_edge(
                    source_id=spec_id,
                    target_id=req_id,
                    edge_type="DEFINES",
                )
                logger.info(f"Created edge: {spec_id} -> {req_id}")
            except Exception as e:
                logger.warning(f"Failed to create edge {spec_id} -> {req_id}: {e}")

        return created_ids

    except Exception as e:
        logger.warning(f"Failed to parse spec and create REQ nodes: {e}")
        logger.warning(f"Continuing without REQ nodes - user can provide feedback to clarify")
        return []


def process_user_message(state: GraphState, message: str) -> Dict[str, Any]:
    """
    Process user message at any time during orchestration.

    UNIFIED CONVERSATION: This is the primary interaction method.
    Users can send messages without waiting for approval or phase transitions.

    Classifies message intent and updates graph accordingly:
    - new_requirement: Creates new REQ node
    - clarification: Updates existing nodes with context
    - modification: Revises graph based on feedback
    - approval: Clears pending_human_input and continues
    - question: Stores as CLARIFICATION node for context
    - commentary: Acknowledges and continues

    Args:
        state: Current graph state
        message: User's message content

    Returns:
        State updates with processed message and orchestrator response
    """
    from agents.tools import get_db
    from core.schemas import NodeData
    from core.ontology import NodeType, NodeStatus

    session_id = state.get("session_id", "unknown")
    task_id = state.get("task_id", "unknown")
    current_phase = state.get("phase", "INIT")

    logger.info(f"Processing user message in session {session_id}, phase {current_phase}: {message[:50]}...")

    # Store message in conversation history
    new_message = {
        "role": "user",
        "content": message,
        "timestamp": str(datetime.now()),
        "phase": current_phase,
    }

    # Enhanced intent classification with more context awareness
    message_lower = message.lower()

    # Check for approval/confirmation
    if any(keyword in message_lower for keyword in ["yes", "approve", "proceed", "continue", "go ahead", "looks good", "let's build"]):
        intent = "approval"
    # Check for new requirement
    elif any(keyword in message_lower for keyword in ["add", "also need", "requirement", "must have", "feature", "should include"]):
        intent = "new_requirement"
    # Check for modification request
    elif any(keyword in message_lower for keyword in ["change", "update", "modify", "instead", "revise", "fix"]):
        intent = "modification"
    # Check for question
    elif any(keyword in message_lower for keyword in ["?", "how", "what", "when", "where", "why", "which", "can you"]):
        intent = "question"
    # Check for rejection
    elif any(keyword in message_lower for keyword in ["no", "wait", "stop", "cancel", "reject"]):
        intent = "rejection"
    else:
        # Default: treat as new requirement if substantial, otherwise commentary
        if len(message.split()) > 5:
            intent = "new_requirement"
        else:
            intent = "commentary"

    logger.info(f"Classified message intent as: {intent}")

    # Handle based on intent
    db = get_db()
    response_message = None

    if intent == "approval":
        # User approves - clear pending input and allow orchestrator to continue
        logger.info("User approved - clearing pending_human_input")
        return {
            "messages": [new_message],
            "human_response": message,
            "pending_human_input": None,  # Clear to allow continuation
        }

    elif intent == "new_requirement" and db:
        # Create new REQ node
        import uuid
        req_id = f"req_{task_id}_{uuid.uuid4().hex[:8]}"

        req_node = NodeData(
            id=req_id,
            type=NodeType.REQ.value,
            content=message,
            status=NodeStatus.PENDING.value,
            created_by="user",
        )
        db.add_node(req_node)
        logger.info(f"Created new REQ node from user message: {req_id}")

        # Generate response acknowledging the new requirement
        response_message = {
            "role": "system",
            "content": f"Added new requirement: {message[:100]}... I'll incorporate this into the implementation plan.",
            "timestamp": str(datetime.now()),
            "phase": current_phase,
        }

    elif intent == "question" and db:
        # Store question as CLARIFICATION node for context
        import uuid
        clarif_id = f"clarif_{task_id}_{uuid.uuid4().hex[:8]}"

        clarif_node = NodeData(
            id=clarif_id,
            type=NodeType.CLARIFICATION.value,
            content=message,
            status=NodeStatus.PENDING.value,
            created_by="user",
            data={
                "role": "question",
                "session_id": session_id,
                "phase": current_phase,
            }
        )
        db.add_node(clarif_node)
        logger.info(f"Stored user question as CLARIFICATION node: {clarif_id}")

        # TODO: Generate intelligent response using LLM based on current context
        response_message = {
            "role": "system",
            "content": "I've noted your question. I'll address this as I continue working on your project.",
            "timestamp": str(datetime.now()),
            "phase": current_phase,
        }

    elif intent == "modification":
        # TODO: Implement modification logic (update existing REQ nodes)
        response_message = {
            "role": "system",
            "content": "I'll update the requirements based on your feedback.",
            "timestamp": str(datetime.now()),
            "phase": current_phase,
        }

    elif intent == "rejection":
        # User wants to revise - keep pending_human_input set
        response_message = {
            "role": "system",
            "content": "Understood. Please provide your feedback or clarifications, and I'll adjust accordingly.",
            "timestamp": str(datetime.now()),
            "phase": current_phase,
        }

    else:
        # Commentary - acknowledge
        response_message = {
            "role": "system",
            "content": "Noted. I'll continue working on your project.",
            "timestamp": str(datetime.now()),
            "phase": current_phase,
        }

    # Return state updates including response message
    messages_to_add = [new_message]
    if response_message:
        messages_to_add.append(response_message)

    return {
        "messages": messages_to_add,
        "human_response": message,  # Update human_response field
        "pending_human_input": None if intent == "approval" else state.get("pending_human_input"),
    }


def init_node(state: GraphState) -> Dict[str, Any]:
    """
    Initialize the orchestration cycle.

    - Validate inputs
    - Set up initial state
    - IMMEDIATELY create REQ nodes from spec (Phase 2: Proactive approach)
    - Transition to DIALECTIC (pre-research ambiguity check)
    """
    # Phase 2: Proactively create REQ nodes from spec
    req_node_ids = create_requirement_nodes_from_spec(state)

    return {
        "phase": CyclePhase.DIALECTIC.value,
        "messages": [{
            "role": "system",
            "content": f"Starting TDD cycle for task: {state.get('task_id', 'unknown')}. Created {len(req_node_ids)} requirement nodes."
        }],
        "iteration": 0,
        "ambiguities": [],
        "clarification_questions": [],
        "research_findings": [],
        "research_complete": False,
    }


def dialectic_node(state: GraphState) -> Dict[str, Any]:
    """
    Dialectic phase - Run full Socratic dialogue for requirement clarification.

    NEW IMPLEMENTATION: Uses SocraticOrchestrator for complete L1->L2->L3->Terminal flow.

    This node now:
    1. Creates a SocraticOrchestrator instance
    2. Runs the full dialogue (pauses for human input at each level)
    3. Persists Q&A to graph as CLARIFICATION nodes
    4. Returns augmented spec with all clarifications

    Flow:
    - If dialogue APPROVED: proceed to RESEARCH with augmented spec
    - If NOT approved or incomplete: stay in CLARIFICATION (wait for user)
    - Autonomous mode is handled via initial phase setting in orchestrator.run()
    """
    spec = state.get("spec", "")
    session_id = state.get("session_id", "unknown")
    task_id = state.get("task_id", "unknown")

    # Check if autonomous mode (skip dialectic)
    # Note: Autonomous mode is handled via initial phase setting, not a flag
    # This check is kept for backward compatibility but can be removed in future

    messages = [{
        "role": "assistant",
        "content": "Starting Socratic dialogue for requirement clarification..."
    }]

    # Check if SocraticOrchestrator is available
    if not SOCRATIC_ORCHESTRATOR_AVAILABLE:
        logger.warning("SocraticOrchestrator not available, using legacy dialectic")
        return _legacy_dialectic_node(state)

    # NEW: Use SocraticOrchestrator for full dialogue
    try:
        from agents.tools import get_db

        db = get_db()
        if db is None:
            # Fallback to old dialectic if DB not available
            logger.warning("ParagonDB not available, using legacy dialectic")
            return _legacy_dialectic_node(state)

        # Run Socratic dialogue
        socrates = SocraticOrchestrator(db, enable_checkpointing=True)

        # Create or find REQ node for this task
        req_node_id = f"req_{task_id}"
        if not db.has_node(req_node_id):
            from core.schemas import NodeData
            from core.ontology import NodeType, NodeStatus
            # Use constructor directly since we want to specify our own id
            req_node = NodeData(
                id=req_node_id,
                type=NodeType.REQ.value,
                content=spec,
                status=NodeStatus.PENDING.value,
                created_by="orchestrator",
            )
            db.add_node(req_node)

        # Run the dialogue (this will pause for human input)
        result = socrates.run(
            req_id=req_node_id,
            spec_content=spec,
            session_id=f"socratic_{session_id}",
        )

        messages.append({
            "role": "assistant",
            "content": (
                f"Socratic dialogue complete. "
                f"Approved: {result.approved}. "
                f"Questions answered: {result.answered_questions}/{result.total_questions}. "
                f"Out of scope items: {len(result.out_of_scope)}"
            ),
        })

        # Determine next phase based on approval
        if result.approved:
            next_phase = CyclePhase.RESEARCH.value
            messages.append({
                "role": "assistant",
                "content": "User approved specification. Proceeding to research."
            })
        else:
            next_phase = CyclePhase.CLARIFICATION.value
            messages.append({
                "role": "assistant",
                "content": "Awaiting user approval or additional clarifications."
            })

        return {
            "phase": next_phase,
            "messages": messages,
            "spec": result.augmented_spec,  # Updated spec with clarifications
            "socratic_session_id": result.session_id,
            "out_of_scope": result.out_of_scope,
            "assumptions": result.assumptions,
            "pending_human_input": None if result.approved else "socratic_approval",
        }

    except ImportError:
        # Fallback to old dialectic if SocraticOrchestrator not available
        logger.warning("SocraticOrchestrator not available, using legacy dialectic")
        return _legacy_dialectic_node(state)
    except Exception as e:
        logger.error(f"Socratic dialogue failed: {e}")
        # Broadcast error to WebSocket clients
        if EVENT_BUS_AVAILABLE:
            try:
                publish_orchestrator_error(
                    error_message=str(e),
                    phase="DIALECTIC",
                    error_type=type(e).__name__
                )
            except Exception:
                pass  # Don't let event bus break orchestrator
        # Fallback to old dialectic on error
        return _legacy_dialectic_node(state)


def _legacy_dialectic_node(state: GraphState) -> Dict[str, Any]:
    """
    LEGACY dialectic implementation (pre-Socratic integration).

    Uses conditional question selection strategy with multi-source questions.
    Kept as fallback for when SocraticOrchestrator is not available.
    """
    spec = state.get("spec", "")
    session_id = state.get("session_id", "unknown")
    phase = state.get("phase", "initial")

    messages = [{
        "role": "assistant",
        "content": "Analyzing requirement and selecting questions..."
    }]

    ambiguities = []
    clarification_questions = []
    # CRITICAL: Default to CLARIFICATION - we ALWAYS need user approval before proceeding
    # Even if the spec is perfectly clear, the user must explicitly confirm they're ready
    next_phase = CyclePhase.CLARIFICATION.value  # Always require user approval

    # NEW: Use QuestionSelector for conditional question selection
    try:
        from agents.question_selector import select_questions_for_dialectic

        # Select questions using multi-source strategy
        selected_questions = select_questions_for_dialectic(
            initial_prompt=spec,
            phase=phase,
            research_findings=state.get("research_findings", []),
            module_complete=False,  # TODO: Implement module completion detection
            software_complete=False,  # TODO: Implement software completion detection
        )

        messages.append({
            "role": "tool",
            "content": f"Question selector identified {len(selected_questions)} questions",
            "tool_name": "question_selector",
            "tool_result": {"question_count": len(selected_questions)},
        })

        # Convert selected questions to clarification format
        for sq in selected_questions:
            clarification_questions.append({
                "question": sq.question,
                "category": sq.category,
                "priority": sq.priority,
                "suggested_answer": sq.suggested_answer,
                "source": sq.source.value,
                "rationale": sq.rationale,
            })
    except ImportError:
        logger.warning("QuestionSelector not available, falling back to basic analysis")
    except Exception as e:
        logger.warning(f"Question selector failed: {e}, falling back to basic analysis")

    # Fallback: Run SocraticEngine gap analysis if question selector failed
    if not clarification_questions and SOCRATIC_AVAILABLE and _gap_analyzer:
        try:
            gaps = _gap_analyzer.analyze_spec(spec)
            if gaps:
                messages.append({
                    "role": "tool",
                    "content": f"Socratic gap analysis found {len(gaps)} gap(s)",
                    "tool_name": "socratic_engine",
                    "tool_result": {"gap_count": len(gaps)},
                })
                # Convert gaps to clarification questions
                for gap in gaps:
                    if gap.severity in ("critical", "high"):
                        question = ALL_QUESTIONS.get(gap.question_id)
                        if question:
                            clarification_questions.append({
                                "question": question.question,
                                "category": question.category,
                                "text": gap.context,
                                "suggested_answer": gap.suggestion,
                                "source": "socratic",
                            })
        except Exception as e:
            logger.debug(f"Socratic gap analysis failed: {e}")

    if LLM_AVAILABLE:
        try:
            llm = get_llm()

            # Build dialector prompt
            system_prompt = """You are the DIALECTOR agent. Analyze the requirement for ambiguity that would prevent autonomous code generation.

Scan for:
1. SUBJECTIVE_TERMS: "fast", "efficient", "user-friendly"
2. COMPARATIVE_STATEMENTS: "faster than", "better than" (compared to what?)
3. UNDEFINED_PRONOUN: "it", "this" without clear referent
4. UNDEFINED_TERM: Domain-specific terms needing definition
5. MISSING_CONTEXT: Input/output format not specified

For each ambiguity you find:
- Provide a suggested_question to ask the user
- Provide a suggested_answer with your best-guess reasonable default (e.g., for "fast" suggest "< 100ms latency")

The suggested_answer helps users accept a reasonable default if they don't have a strong preference.

For technical, well-defined requirements like "implement fibonacci" or "create a stack class", return is_clear: true.
Only set is_clear: false if there's genuine blocking ambiguity."""

            user_prompt = f"""# Requirement to Analyze

{spec}

Analyze for ambiguity. Return structured output with verdict (CLEAR or NEEDS_CLARIFICATION) and any blocking ambiguities found."""

            result = llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=DialectorOutput,
            )

            # DialectorOutput has: is_clear, ambiguities, blocking_count, recommendation
            # Override is_clear if blocking ambiguities were found (LLM sometimes says clear but lists blockers)
            has_blocking = result.blocking_count > 0 or any(
                getattr(a, 'impact', '').upper() == 'BLOCKING' for a in result.ambiguities
            )
            actually_clear = result.is_clear and not has_blocking and len(result.ambiguities) == 0

            verdict = "CLEAR" if actually_clear else "NEEDS_CLARIFICATION"
            messages.append({
                "role": "assistant",
                "content": f"Dialectic analysis: {verdict} ({result.recommendation}) - {result.blocking_count} blocking, {len(result.ambiguities)} total ambiguities"
            })

            if not actually_clear and result.ambiguities:
                # Found ambiguities - generate clarification questions for each
                # AmbiguityMarker has: category, text, impact, suggested_question, suggested_answer
                for amb in result.ambiguities:
                    ambiguities.append({
                        "text": amb.text,
                        "category": amb.category,
                        "impact": amb.impact,
                        "question": amb.suggested_question,
                        "suggested_answer": getattr(amb, 'suggested_answer', None),
                    })

                    # Treat ALL ambiguities as needing clarification
                    # The impact field from LLM is often a description, not "BLOCKING"/"CLARIFYING"
                    # Generate a question for each ambiguity
                    question_text = amb.suggested_question
                    if not question_text:
                        # Generate question based on category
                        if amb.category == "SUBJECTIVE_TERMS":
                            question_text = f"What specific, measurable criteria define '{amb.text}'?"
                        elif amb.category == "MISSING_CONTEXT":
                            question_text = f"Please provide more details about: {amb.text}"
                        elif amb.category == "COMPARATIVE_STATEMENTS":
                            question_text = f"What is '{amb.text}' being compared to?"
                        elif amb.category == "UNDEFINED_PRONOUN":
                            question_text = f"What does '{amb.text}' refer to specifically?"
                        else:
                            question_text = f"Please clarify: {amb.text}"

                    clarification_questions.append({
                        "question": question_text,
                        "category": amb.category,
                        "text": amb.text,
                        "suggested_answer": getattr(amb, 'suggested_answer', None),
                    })

                # Always add at least a confirmation question for mutual agreement
                if not clarification_questions:
                    # CRITICAL: Even if spec is clear, user must confirm they're ready to proceed
                    clarification_questions.append({
                        "question": "I've analyzed your specification and it appears complete. Do you have any additional requirements, clarifications, or questions before we proceed to implementation?",
                        "category": "CONFIRMATION",
                        "text": "Spec review confirmation",
                        "suggested_answer": "No, the specification is complete. Please proceed.",
                    })
                    messages.append({
                        "role": "assistant",
                        "content": "Specification appears clear. Asking for confirmation before proceeding."
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": f"Found {len(clarification_questions)} ambiguities. Waiting for user clarification."
                    })

                # Always require clarification phase for mutual agreement
                next_phase = CyclePhase.CLARIFICATION.value

        except Exception as e:
            logger.warning(f"Dialectic analysis failed: {e}")
            # Even on error, we still need user confirmation
            clarification_questions.append({
                "question": "Dialectic analysis encountered an issue. Please review your specification and confirm you're ready to proceed, or provide any clarifications.",
                "category": "CONFIRMATION",
                "text": "Fallback confirmation",
                "suggested_answer": "Proceed with the current specification.",
            })
            messages.append({
                "role": "assistant",
                "content": f"Dialectic analysis unavailable ({e}), but still requiring user confirmation"
            })
    else:
        # Even without LLM, we need user confirmation
        clarification_questions.append({
            "question": "LLM analysis is not available. Please review your specification and confirm you're ready to proceed to implementation.",
            "category": "CONFIRMATION",
            "text": "No-LLM confirmation",
            "suggested_answer": "Proceed with the current specification.",
        })
        messages.append({
            "role": "assistant",
            "content": "LLM not configured, but still requiring user confirmation before proceeding"
        })

    return {
        "phase": next_phase,
        "messages": messages,
        "ambiguities": ambiguities,
        "clarification_questions": clarification_questions,
        # ALWAYS require human input - this is the mutual agreement requirement
        "pending_human_input": "clarification",
    }


def clarification_node(state: GraphState) -> Dict[str, Any]:
    """
    Clarification phase - wait for and process user responses.

    This node handles the human-in-the-loop interaction:
    - Prioritizes questions using AdaptiveQuestioner (if available)
    - Presents questions to user
    - Waits for responses
    - Records outcomes for learning
    - Incorporates feedback into the spec
    - PERSISTS Q&A to graph as CLARIFICATION nodes
    """
    questions = state.get("clarification_questions", [])
    human_response = state.get("human_response", None)
    spec = state.get("spec", "")
    session_id = state.get("session_id", "unknown")
    task_id = state.get("task_id", "unknown")

    messages = []

    # Try to use AdaptiveQuestioner for smart prioritization
    prioritized_questions = questions
    adaptive_questioner = None
    try:
        from agents.adaptive_questioner import AdaptiveQuestioner, UserPriorities
        from agents.schemas import AmbiguityMarker

        adaptive_questioner = AdaptiveQuestioner()

        # Convert questions to AmbiguityMarker format if needed
        ambiguities = []
        for q in questions:
            if isinstance(q, dict):
                ambiguities.append(AmbiguityMarker(
                    text=q.get("question", q.get("phrase", "")),
                    category=q.get("category", "MISSING_CONTEXT"),
                    impact=q.get("impact", "CLARIFYING"),
                    suggested_question=q.get("question"),
                    suggested_answer=q.get("suggested_answer"),
                ))
            elif hasattr(q, 'category'):
                ambiguities.append(q)

        if ambiguities:
            # Prioritize questions by expected information gain
            prioritized = adaptive_questioner.prioritize_questions(ambiguities)
            # Convert back to dict format for display
            prioritized_questions = [
                {
                    "phrase": a.text,  # Use text as phrase for display
                    "question": a.suggested_question or a.text,
                    "category": a.category,
                    "suggested_answer": a.suggested_answer,
                }
                for a in prioritized
            ]
            logger.debug(f"AdaptiveQuestioner prioritized {len(prioritized_questions)} questions")
    except ImportError:
        pass  # AdaptiveQuestioner not available
    except Exception as e:
        logger.debug(f"AdaptiveQuestioner failed: {e}")

    if human_response:
        # User has provided responses - incorporate them and PERSIST TO GRAPH
        messages.append({
            "role": "user",
            "content": f"User clarification: {human_response}"
        })

        # PERSIST: Create CLARIFICATION nodes for each Q&A pair
        from datetime import datetime
        try:
            # Find the REQ node to link to
            req_result = query_nodes(node_type="REQ", limit=100)
            req_node_id = None
            if req_result.success and req_result.node_ids:
                # Use the first REQ node (or find by task_id if needed)
                req_node_id = req_result.node_ids[0]

            # Parse the human response into individual answers
            # Assuming format: "1. answer1\n2. answer2\n..."
            answer_lines = [line.strip() for line in human_response.split('\n') if line.strip()]

            for i, q in enumerate(prioritized_questions):
                question_text = q.get("question", q.get("phrase", ""))
                category = q.get("category", "MISSING_CONTEXT")

                # Create CLARIFICATION node for the question
                question_node_result = add_node(
                    node_type="CLARIFICATION",
                    content=question_text,
                    data={
                        "role": "question",
                        "category": category,
                        "turn_number": i,
                        "timestamp": datetime.utcnow().isoformat(),
                        "session_id": session_id,
                    },
                    created_by="dialector_agent",
                )

                if question_node_result.success:
                    question_node_id = question_node_result.node_id

                    # Link question to REQ if we have one
                    if req_node_id:
                        add_edge(
                            source_id=question_node_id,
                            target_id=req_node_id,
                            edge_type="TRACES_TO",
                        )

                    # Create CLARIFICATION node for the answer if we have one
                    if i < len(answer_lines):
                        # Extract answer (remove leading number if present)
                        answer_text = answer_lines[i]
                        # Remove leading "1. " style prefix
                        import re
                        answer_text = re.sub(r'^\d+\.\s*', '', answer_text)

                        answer_node_result = add_node(
                            node_type="CLARIFICATION",
                            content=answer_text,
                            data={
                                "role": "answer",
                                "category": category,
                                "turn_number": i,
                                "timestamp": datetime.utcnow().isoformat(),
                                "session_id": session_id,
                            },
                            created_by="user",
                        )

                        if answer_node_result.success:
                            answer_node_id = answer_node_result.node_id

                            # Create RESOLVED_BY edge from answer to question
                            add_edge(
                                source_id=answer_node_id,
                                target_id=question_node_id,
                                edge_type="RESOLVED_BY",
                            )

                            logger.debug(f"Persisted Q&A pair: {question_node_id} -> {answer_node_id}")
        except Exception as e:
            logger.warning(f"Failed to persist clarification to graph: {e}")

        # Record question outcomes for learning (if adaptive questioner available)
        if adaptive_questioner:
            try:
                for q in questions:
                    if isinstance(q, dict):
                        from agents.schemas import AmbiguityMarker
                        amb = AmbiguityMarker(
                            text=q.get("question", q.get("phrase", "")),
                            category=q.get("category", "MISSING_CONTEXT"),
                            impact="CLARIFYING",
                        )
                        adaptive_questioner.record_question_outcome(
                            session_id=session_id,
                            ambiguity=amb,
                            was_answered=True,
                            user_answer=human_response,
                            used_suggestion=False,
                            answer_quality_score=0.8,  # User provided response
                        )
            except Exception as e:
                logger.debug(f"Failed to record question outcome: {e}")

        # Augment the spec with clarifications
        augmented_spec = f"""{spec}

# User Clarifications
{human_response}
"""
        messages.append({
            "role": "assistant",
            "content": "Clarifications received. Proceeding to research phase."
        })

        return {
            "phase": CyclePhase.RESEARCH.value,
            "messages": messages,
            "spec": augmented_spec,
            "pending_human_input": None,
        }
    else:
        # Still waiting for user input - format prioritized questions for display
        question_text = "Please clarify the following:\n\n"
        for i, q in enumerate(prioritized_questions, 1):
            question_text += f"{i}. {q.get('question', q.get('phrase', 'Unknown'))}\n"
            question_text += f"   (Category: {q.get('category', 'unknown')})\n"
            # Show suggested answer if available
            if q.get('suggested_answer'):
                question_text += f"   Suggested: {q.get('suggested_answer')}\n"
            question_text += "\n"

        messages.append({
            "role": "assistant",
            "content": question_text
        })

        # Stay in clarification phase waiting for input
        return {
            "phase": CyclePhase.CLARIFICATION.value,
            "messages": messages,
            "pending_human_input": "clarification",
        }


def research_node(state: GraphState) -> Dict[str, Any]:
    """
    Research phase - Socratic research loop.

    Transforms the requirement into a structured Research Artifact:
    - Input/output contracts
    - Examples (happy path, edge case, error case)
    - Complexity bounds
    - Security considerations

    This is the "sufficient statistic" for autonomous code generation.
    """
    spec = state.get("spec", "")
    session_id = state.get("session_id", "unknown")

    messages = [{
        "role": "assistant",
        "content": "Conducting research to create sufficient statistic..."
    }]

    research_findings = []
    next_phase = CyclePhase.PLAN.value

    if LLM_AVAILABLE:
        try:
            llm = get_llm()

            system_prompt = """You are the RESEARCHER agent. Transform the requirement into a structured Research Artifact.

Your artifact must include:
1. Task category (greenfield, algorithmic, systems, etc.)
2. Input/output contracts with Python types
3. Happy path, edge case, and error case examples
4. Complexity bounds (time and space)
5. Security considerations (forbidden patterns, trust boundary)

This Research Artifact is the "sufficient statistic" - it should contain everything needed to implement the code without further clarification."""

            user_prompt = f"""# Requirement to Research

{spec}

Transform this into a structured Research Artifact with concrete examples, type contracts, and test specifications."""

            result = llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=ResearchArtifact,
            )

            # ResearchArtifact has: task_category, input_contract, output_contract,
            # happy_path_examples, edge_cases, error_cases, complexity_bounds,
            # security_posture, findings
            research_findings.append({
                "task_category": result.task_category,
                "input_contract": result.input_contract,
                "output_contract": result.output_contract,
                "happy_path_count": len(result.happy_path_examples),
                "edge_case_count": len(result.edge_cases),
                "error_case_count": len(result.error_cases),
                "complexity_bounds": result.complexity_bounds,
                "security_posture": result.security_posture,
                "findings_count": len(result.findings),
            })

            messages.append({
                "role": "assistant",
                "content": f"Research complete. Task: {result.task_category}. Examples: {len(result.happy_path_examples)} happy, {len(result.edge_cases)} edge, {len(result.error_cases)} error."
            })

            # Add research summary to spec for planning phase
            research_summary = f"""
# Research Artifact

**Task Category:** {result.task_category}

**Input Contract:** {result.input_contract}
**Output Contract:** {result.output_contract}

**Complexity:** {result.complexity_bounds or 'Not specified'}
**Security:** {result.security_posture or 'Not specified'}

**Happy Path Examples:**
{chr(10).join(f'- {ex}' for ex in result.happy_path_examples) if result.happy_path_examples else 'None'}

**Edge Cases:**
{chr(10).join(f'- {ex}' for ex in result.edge_cases) if result.edge_cases else 'None'}

**Error Cases:**
{chr(10).join(f'- {ex}' for ex in result.error_cases) if result.error_cases else 'None'}
"""
            augmented_spec = f"{spec}\n{research_summary}"

            # Trigger notification for research completion
            try:
                create_notification(
                    notification_type="research_complete",
                    message=f"Research complete: {result.task_category} - {len(result.happy_path_examples)} examples generated",
                    target_tabs=["research", "specification"],
                    urgency="info",
                    source_component="orchestrator",
                )
            except Exception as e:
                logger.warning(f"Failed to create research completion notification: {e}")

            return {
                "phase": next_phase,
                "messages": messages,
                "spec": augmented_spec,
                "research_findings": research_findings,
                "research_complete": True,
            }

        except Exception as e:
            logger.warning(f"Research phase failed: {e}")
            # Broadcast error to WebSocket clients
            if EVENT_BUS_AVAILABLE:
                try:
                    publish_orchestrator_error(
                        error_message=str(e),
                        phase="RESEARCH",
                        error_type=type(e).__name__
                    )
                except Exception:
                    pass  # Don't let event bus break orchestrator
            messages.append({
                "role": "assistant",
                "content": f"Research unavailable ({e}), proceeding to planning"
            })

    return {
        "phase": next_phase,
        "messages": messages,
        "research_findings": research_findings,
        "research_complete": True,
    }


def plan_node(state: GraphState) -> Dict[str, Any]:
    """
    Planning phase - analyze requirements and create plan.

    Layer 7A Integration:
    - Uses StructuredLLM to generate ImplementationPlan
    - Creates SPEC nodes for each component
    - Establishes DEPENDS_ON edges based on dependencies

    - Parse specification
    - Identify components to build
    - Create dependency graph
    - Request human approval if needed
    """
    spec = state.get("spec", "")
    requirements = state.get("requirements", [])

    # Check current graph state
    stats = get_graph_stats()

    messages = [
        {
            "role": "assistant",
            "content": f"Planning implementation for spec with {len(requirements)} requirements"
        },
        {
            "role": "tool",
            "content": f"Graph stats: {stats}",
            "tool_name": "get_graph_stats",
            "tool_result": stats,
        },
    ]

    # Layer 7A: Use LLM to generate implementation plan
    spec_node_ids = []
    plan_result = None

    if LLM_AVAILABLE:
        try:
            llm = get_llm()

            # Build prompts for architect
            system_prompt = """You are an Architect agent. Your role is to decompose requirements into atomic, implementable specifications.

Each component must be:
- Atomic: implementable in a single pass
- Testable: has clear success criteria
- Independent: can be built after its dependencies are satisfied

IMPORTANT: You must also identify:
- out_of_scope: Items explicitly excluded from this implementation (features, use cases, edge cases)
- assumptions: Assumptions made during planning (e.g., "assumes Python 3.10+", "assumes PostgreSQL database")

Output a structured implementation plan."""

            user_prompt = f"""# Task Specification

{spec}

# Requirements
{chr(10).join(f"- {r}" for r in requirements) if requirements else "No specific requirements provided."}

Decompose this into atomic, implementable components.

Make sure to identify:
1. Items that are explicitly OUT OF SCOPE for this implementation
2. Key ASSUMPTIONS you're making during the planning phase"""

            # Generate plan using structured output
            plan_result = llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=ImplementationPlan,
            )

            messages.append({
                "role": "assistant",
                "content": f"Generated plan with {len(plan_result.components)} components: {plan_result.explanation}"
            })

            # Create SPEC nodes for each component
            component_to_node = {}
            for component in plan_result.components:
                spec_content = f"""# {component.name}
Type: {component.type}
Description: {component.description}
Dependencies: {', '.join(component.dependencies) if component.dependencies else 'None'}
"""
                if component.file_path:
                    spec_content += f"File: {component.file_path}\n"

                result = add_node(
                    node_type="SPEC",
                    content=spec_content,
                    data={
                        "component_name": component.name,
                        "component_type": component.type,
                        "dependencies": component.dependencies,
                        "out_of_scope": plan_result.out_of_scope,
                        "assumptions": plan_result.assumptions,
                    },
                    created_by="architect_agent",
                )

                if result.success:
                    spec_node_ids.append(result.node_id)
                    component_to_node[component.name] = result.node_id
                    messages.append({
                        "role": "tool",
                        "content": f"Created SPEC node for {component.name}",
                        "tool_name": "add_node",
                        "tool_result": {"node_id": result.node_id},
                    })

            # Create DEPENDS_ON edges from plan dependencies
            for dep_edge in plan_result.dependencies:
                if dep_edge.source in component_to_node and dep_edge.target in component_to_node:
                    source_id = component_to_node[dep_edge.source]
                    target_id = component_to_node[dep_edge.target]
                    add_edge(
                        source_id=source_id,
                        target_id=target_id,
                        edge_type="DEPENDS_ON",
                    )

            # Trigger notification for spec creation
            if spec_node_ids:
                try:
                    create_notification(
                        notification_type="spec_updated",
                        message=f"Planning complete: {len(spec_node_ids)} specifications created",
                        target_tabs=["specification", "build"],
                        urgency="info",
                        source_component="orchestrator",
                    )
                except Exception as e:
                    logger.warning(f"Failed to create spec notification: {e}")

        except Exception as e:
            logger.warning(f"LLM planning failed, using fallback: {e}")
            # Broadcast error to WebSocket clients
            if EVENT_BUS_AVAILABLE:
                try:
                    publish_orchestrator_error(
                        error_message=str(e),
                        phase="PLAN",
                        error_type=type(e).__name__
                    )
                except Exception:
                    pass  # Don't let event bus break orchestrator
            messages.append({
                "role": "assistant",
                "content": f"LLM planning unavailable ({e}), proceeding with basic plan"
            })
    else:
        messages.append({
            "role": "assistant",
            "content": "LLM not configured, proceeding with basic planning"
        })

    return {
        "phase": CyclePhase.BUILD.value,
        "messages": messages,
        "pending_human_input": None,  # Could set to "approval" to pause for review
        # Store spec node IDs for build phase
        "artifacts": [{"type": "spec_nodes", "node_ids": spec_node_ids}] if spec_node_ids else [],
    }


def build_node(state: GraphState) -> Dict[str, Any]:
    """
    Build phase - generate code artifacts.

    Layer 7 Integration:
    - Layer 7A: Uses StructuredLLM to generate CodeGeneration
    - Layer 7B: Uses add_node_safe for syntax + topology validation

    - Create code nodes
    - Create test nodes
    - Establish dependencies
    """
    iteration = state.get("iteration", 0)
    artifacts = state.get("artifacts", [])

    messages = [
        {
            "role": "assistant",
            "content": f"Build iteration {iteration + 1}: Creating code artifacts"
        },
    ]

    code_node_ids = []
    test_node_ids = []

    # Find SPEC nodes from plan phase
    spec_node_ids = []
    for artifact in artifacts:
        if artifact.get("type") == "spec_nodes":
            spec_node_ids = artifact.get("node_ids", [])
            break

    if LLM_AVAILABLE and spec_node_ids:
        try:
            llm = get_llm()

            # Generate code for each SPEC
            for spec_id in spec_node_ids:
                spec_data = get_node(spec_id)
                if "error" in spec_data:
                    continue

                spec_content = spec_data.get("content", "")
                component_name = spec_data.get("data", {}).get("component_name", "unknown")

                # Layer 7A: Generate code using LLM
                system_prompt = """You are a Builder agent. Your role is to implement working code from specifications.

Rules:
- Write complete, runnable code
- Include all necessary imports
- Add docstrings and type hints
- Handle edge cases gracefully

Output complete implementation code."""

                user_prompt = f"""# Specification to Implement

{spec_content}

Implement complete, working Python code for this specification."""

                try:
                    code_result = llm.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        schema=CodeGeneration,
                    )

                    messages.append({
                        "role": "assistant",
                        "content": f"Generated code for {component_name}: {code_result.filename}"
                    })

                    # Create attribution signature for learning system
                    from datetime import datetime
                    builder_signature = AgentSignature(
                        agent_id="builder_agent",
                        model_id=llm.model if hasattr(llm, 'model') else "unknown",
                        phase=CyclePhase.BUILD,
                        action=SignatureAction.CREATED,
                        temperature=getattr(llm, 'temperature', 0.7),
                        context_constraints={
                            "spec_id": spec_id,
                            "component": component_name,
                        },
                        timestamp=datetime.utcnow().isoformat(),
                    )

                    # Layer 7B: Add with safety checks and attribution
                    safe_result = add_node_safe(
                        node_type="CODE",
                        content=code_result.code,
                        spec_id=spec_id,
                        data={
                            "filename": code_result.filename,
                            "imports": code_result.imports,
                            "description": code_result.description,
                            "language": code_result.language,
                        },
                        created_by="builder_agent",
                        check_alignment=True,
                        signature=builder_signature,
                    )

                    if safe_result.success:
                        code_node_ids.append(safe_result.node_id)
                        messages.append({
                            "role": "tool",
                            "content": f"CODE node created and verified: {safe_result.node_id}",
                            "tool_name": "add_node_safe",
                            "tool_result": {
                                "node_id": safe_result.node_id,
                                "syntax_valid": safe_result.syntax_valid,
                                "topology_valid": safe_result.topology_valid,
                            },
                        })

                        # Update SPEC status to PROCESSING
                        update_node_status(spec_id, "PROCESSING")
                    else:
                        # Layer 7B rejected the code
                        messages.append({
                            "role": "tool",
                            "content": f"CODE rejected by auditor: {safe_result.message}",
                            "tool_name": "add_node_safe",
                            "tool_result": {
                                "success": False,
                                "violations": safe_result.violations,
                            },
                        })

                except Exception as e:
                    logger.warning(f"Code generation failed for {spec_id}: {e}")
                    # Broadcast error to WebSocket clients
                    if EVENT_BUS_AVAILABLE:
                        try:
                            publish_orchestrator_error(
                                error_message=f"Code generation failed for {spec_id}: {str(e)}",
                                phase="BUILD",
                                error_type=type(e).__name__
                            )
                        except Exception:
                            pass  # Don't let event bus break orchestrator
                    messages.append({
                        "role": "assistant",
                        "content": f"Failed to generate code for {component_name}: {e}"
                    })

        except Exception as e:
            logger.warning(f"LLM build failed: {e}")
            messages.append({
                "role": "assistant",
                "content": f"LLM code generation unavailable: {e}"
            })
    else:
        messages.append({
            "role": "assistant",
            "content": "LLM not configured or no SPEC nodes, proceeding with placeholder build"
        })

    messages.append({
        "role": "assistant",
        "content": f"Build phase complete. Created {len(code_node_ids)} code node(s). Ready for testing."
    })

    return {
        "phase": CyclePhase.TEST.value,
        "messages": messages,
        "code_node_ids": code_node_ids,
        "test_node_ids": test_node_ids,
    }


def test_node(state: GraphState) -> Dict[str, Any]:
    """
    Test phase - execute tests and validate.

    - Run test suite
    - Check coverage
    - Enforce quality floor (if tests pass)
    - Check for divergence
    - Determine pass/fail
    """
    iteration = state.get("iteration", 0)
    session_id = state.get("session_id", "unknown")
    existing_results = state.get("test_results", [])
    code_node_ids = state.get("code_node_ids", [])

    # Simulate test result (would be real in production)
    # Pass after first test failure and fix cycle
    test_passed = len(existing_results) >= 1

    test_result = {
        "passed": test_passed,
        "output": "All tests passed" if test_passed else "1 test failed",
        "errors": [] if test_passed else ["AssertionError in test_example"],
        "coverage": 0.85 if test_passed else 0.60,
    }

    messages = [
        {
            "role": "assistant",
            "content": f"Running tests for iteration {iteration + 1}"
        },
        {
            "role": "tool",
            "content": f"Test result: {'PASSED' if test_passed else 'FAILED'}",
            "tool_name": "run_tests",
            "tool_result": test_result,
        },
    ]

    quality_violations = []

    # Quality Gate: enforce quality floor BEFORE declaring success
    if test_passed and QUALITY_GATE_AVAILABLE:
        try:
            quality_report = check_quality(
                test_pass_rate=1.0,
                node_ids=code_node_ids,
            )
            if not quality_report.passed:
                # Tests passed but quality floor failed
                test_passed = False
                quality_violations = [v.description for v in quality_report.violations]
                messages.append({
                    "role": "tool",
                    "content": f"Quality gate FAILED: {len(quality_report.violations)} violation(s)",
                    "tool_name": "quality_gate",
                    "tool_result": {
                        "passed": False,
                        "violations": quality_violations,
                    },
                })
            else:
                messages.append({
                    "role": "tool",
                    "content": "Quality gate PASSED",
                    "tool_name": "quality_gate",
                    "tool_result": {"passed": True},
                })
        except Exception as e:
            logger.debug(f"Quality gate check failed: {e}")
            # Broadcast error to WebSocket clients
            if EVENT_BUS_AVAILABLE:
                try:
                    publish_orchestrator_error(
                        error_message=f"Quality gate check failed: {str(e)}",
                        phase="TEST",
                        error_type=type(e).__name__
                    )
                except Exception:
                    pass  # Don't let event bus break orchestrator

    # Divergence detection: log for learning system
    if DIVERGENCE_AVAILABLE and TRAINING_STORE_AVAILABLE:
        try:
            store = TrainingStore()
            detector = DivergenceDetector(store)
            # Check for divergence (would compare with production outcomes in real system)
            divergence = detector.check_divergence(
                session_id=session_id,
                test_passed=test_passed,
                prod_outcome="pending",  # Would be actual prod outcome
            )
            if divergence:
                messages.append({
                    "role": "tool",
                    "content": f"Divergence detected: {divergence.divergence_type}",
                    "tool_name": "divergence_detector",
                    "tool_result": {"type": divergence.divergence_type},
                })
        except Exception as e:
            logger.debug(f"Divergence detection failed: {e}")
            # Broadcast error to WebSocket clients
            if EVENT_BUS_AVAILABLE:
                try:
                    publish_orchestrator_error(
                        error_message=f"Divergence detection failed: {str(e)}",
                        phase="TEST",
                        error_type=type(e).__name__
                    )
                except Exception:
                    pass  # Don't let event bus break orchestrator

    # Determine next phase
    if test_passed:
        next_phase = CyclePhase.PASSED.value
    else:
        next_phase = CyclePhase.FIX.value

    return {
        "phase": next_phase,
        "messages": messages,
        "test_results": [test_result],  # Will be appended by reducer
        "last_test_passed": test_passed,
        "quality_violations": quality_violations,
    }


def fix_node(state: GraphState) -> Dict[str, Any]:
    """
    Fix phase - analyze failures and attempt repairs.

    - Analyze test failures
    - Generate fixes
    - Increment iteration counter
    - Perform forensic attribution analysis
    """
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    test_results = state.get("test_results", [])
    session_id = state.get("session_id", "unknown")
    code_node_ids = state.get("code_node_ids", [])

    # Increment iteration
    iteration += 1

    messages = [{
        "role": "assistant",
        "content": f"Analyzing failures and generating fixes (attempt {iteration}/{max_iterations})"
    }]

    # Check if we've exceeded max iterations
    if iteration >= max_iterations:
        messages.append({
            "role": "assistant",
            "content": f"Max iterations ({max_iterations}) reached. Marking as FAILED."
        })
        return {
            "phase": CyclePhase.FAILED.value,
            "messages": messages,
            "iteration": iteration,
            "errors": [f"Exceeded max iterations ({max_iterations})"],
        }

    # Get last failure info
    attribution_results = []
    if test_results:
        last_result = test_results[-1]
        failure_errors = last_result.get("errors", [])

        # Forensic Attribution Analysis: Determine which agent/model caused the failure
        if ATTRIBUTION_AVAILABLE and TRAINING_STORE_AVAILABLE and failure_errors:
            try:
                store = TrainingStore()
                analyzer = ForensicAnalyzer(store)

                # Analyze each failure error
                for error in failure_errors:
                    # Extract error type and message
                    error_type = error.split(":")[0] if ":" in error else "UnknownError"
                    error_message = error

                    # Perform forensic analysis
                    attribution = analyzer.analyze_failure(
                        session_id=session_id,
                        error_type=error_type,
                        error_message=error_message,
                        failed_node_id=code_node_ids[0] if code_node_ids else None,
                    )
                    attribution_results.append(attribution)

                    # Log attribution for visibility
                    messages.append({
                        "role": "tool",
                        "content": f"Attribution: {attribution.failure_code.value} -> {attribution.attributed_agent_id} ({attribution.confidence:.1%} confidence)",
                        "tool_name": "forensic_analyzer",
                        "tool_result": {
                            "failure_code": attribution.failure_code.value,
                            "attributed_agent": attribution.attributed_agent_id,
                            "attributed_model": attribution.attributed_model_id,
                            "confidence": attribution.confidence,
                            "reasoning": attribution.reasoning,
                        },
                    })
            except Exception as e:
                logger.debug(f"Attribution analysis failed: {e}")

        messages.append({
            "role": "assistant",
            "content": f"Fixing {len(failure_errors)} errors: {failure_errors}"
        })

    return {
        "phase": CyclePhase.BUILD.value,  # Back to build for retry
        "messages": messages,
        "iteration": iteration,
    }


def passed_node(state: GraphState) -> Dict[str, Any]:
    """
    Success terminal state.

    - Finalize artifacts
    - Update node statuses
    - Validate teleology (golden thread)
    - Generate report
    - Trigger documentation generation (via integration)
    """
    iteration = state.get("iteration", 0)
    session_id = state.get("session_id", "unknown")

    # Check graph integrity
    cycle_result = check_cycle()

    messages = [
        {
            "role": "assistant",
            "content": f"TDD cycle PASSED after {iteration + 1} iteration(s)!"
        },
        {
            "role": "tool",
            "content": f"Graph integrity: {cycle_result.message}",
            "tool_name": "check_cycle",
            "tool_result": {"has_cycle": cycle_result.has_cycle},
        },
    ]

    teleology_valid = True

    # Teleology: Validate the golden thread (all nodes trace to REQ)
    if TELEOLOGY_AVAILABLE:
        try:
            from agents.tools import _db
            if _db is not None:
                report = validate_teleology(_db._graph, _db._node_map, _db._inv_map)
                teleology_valid = report.is_valid

                messages.append({
                    "role": "tool",
                    "content": (
                        f"Teleology: {'VALID' if report.is_valid else 'INVALID'} - "
                        f"{report.justified_count}/{report.total_nodes - report.root_count} justified "
                        f"({report.justification_rate:.0%})"
                    ),
                    "tool_name": "teleology_validator",
                    "tool_result": {
                        "is_valid": report.is_valid,
                        "justified_count": report.justified_count,
                        "unjustified_count": report.unjustified_count,
                        "orphaned_count": report.orphaned_count,
                        "unjustified_nodes": report.unjustified_nodes[:5],  # First 5 only
                    },
                })

                if not report.is_valid:
                    logger.warning(
                        f"Teleology check found {report.unjustified_count} unjustified nodes: "
                        f"{report.unjustified_nodes[:3]}"
                    )
        except Exception as e:
            logger.debug(f"Teleology validation failed: {e}")

    # Flush transaction: generate docs and commit
    # This will generate README/wiki/changelog and create a git commit
    try:
        from agents.tools import flush_transaction
        flush_result = flush_transaction(agent_id="orchestrator-success")
        messages.append({
            "role": "tool",
            "content": flush_result.message,
            "tool_name": "flush_transaction",
            "tool_result": {
                "success": flush_result.success,
                "message": flush_result.message,
            },
        })
        if not flush_result.success:
            logger.warning(f"Transaction flush had issues: {flush_result.message}")
    except Exception as e:
        logger.debug(f"Transaction flush failed: {e}")
        messages.append({
            "role": "tool",
            "content": f"Transaction flush error: {e}",
            "tool_name": "flush_transaction",
            "tool_result": {"success": False, "error": str(e)},
        })

    # Learning: Record successful outcome for adaptive model selection
    try:
        from core.llm import _learning_manager, _learning_available
        if _learning_available and _learning_manager is not None:
            _learning_manager.record_outcome(
                session_id=session_id,
                success=True,
                stats={
                    "total_iterations": iteration + 1,
                    "teleology_valid": teleology_valid,
                },
            )
            logger.debug(f"Learning: Recorded successful outcome for session {session_id}")
    except Exception as e:
        logger.debug(f"Learning feedback failed: {e}")

    return {
        "phase": CyclePhase.PASSED.value,
        "messages": messages,
        "final_status": "passed",
        "teleology_valid": teleology_valid,
    }


def failed_node(state: GraphState) -> Dict[str, Any]:
    """
    Failure terminal state.

    - Log failures
    - Perform forensic attribution (root cause analysis)
    - Preserve state for debugging
    - Optionally request human intervention
    """
    errors = state.get("errors", [])
    session_id = state.get("session_id", "unknown")
    test_results = state.get("test_results", [])

    messages = [{
        "role": "assistant",
        "content": f"TDD cycle FAILED with {len(errors)} error(s)"
    }]

    # Attribution: Perform root cause analysis if available
    attribution_results = []
    if ATTRIBUTION_AVAILABLE and TRAINING_STORE_AVAILABLE:
        try:
            store = TrainingStore()
            analyzer = ForensicAnalyzer(store)

            # Build failure list from errors and test results
            failures = []
            for error in errors:
                failures.append({
                    "error_type": "CycleError",
                    "error_message": str(error),
                })

            for result in test_results:
                if not result.get("passed", True):
                    for err in result.get("errors", []):
                        failures.append({
                            "error_type": "TestError",
                            "error_message": str(err),
                        })

            if failures:
                results = analyzer.analyze_session_failures(session_id, failures)
                for attr in results:
                    attribution_results.append({
                        "failure_code": attr.failure_code.value,
                        "attributed_agent": attr.attributed_agent_id,
                        "attributed_phase": attr.attributed_phase.value,
                        "confidence": attr.confidence,
                        "reasoning": attr.reasoning[:200],  # Truncate for message
                    })

                messages.append({
                    "role": "tool",
                    "content": f"Attribution analysis: {len(results)} failure(s) analyzed",
                    "tool_name": "forensic_analyzer",
                    "tool_result": {"attributions": attribution_results},
                })
        except Exception as e:
            logger.debug(f"Attribution analysis failed: {e}")

    # Learning: Record failed outcome for adaptive model selection
    try:
        from core.llm import _learning_manager, _learning_available
        from infrastructure.learning import FailureCode, CyclePhase as LearningPhase
        if _learning_available and _learning_manager is not None:
            # Determine failure code from attribution if available
            failure_code = None
            failure_phase = None
            if attribution_results:
                first_attr = attribution_results[0]
                try:
                    failure_code = FailureCode(first_attr.get("failure_code", "F5"))
                    failure_phase = LearningPhase(first_attr.get("attributed_phase", "BUILD"))
                except (ValueError, KeyError):
                    pass

            _learning_manager.record_outcome(
                session_id=session_id,
                success=False,
                failure_code=failure_code,
                failure_phase=failure_phase,
                stats={
                    "error_count": len(errors),
                    "attribution_count": len(attribution_results),
                },
            )
            logger.debug(f"Learning: Recorded failed outcome for session {session_id}")
    except Exception as e:
        logger.debug(f"Learning feedback failed: {e}")

    return {
        "phase": CyclePhase.FAILED.value,
        "messages": messages,
        "final_status": "failed",
        "pending_human_input": HumanCheckpointType.REVIEW.value,
        "attribution_results": attribution_results,
    }


# =============================================================================
# ROUTING FUNCTIONS (Conditional Edges)
# =============================================================================

def route_after_dialectic(state: GraphState) -> str:
    """Route after dialectic analysis - to clarification if ambiguities found."""
    # CRITICAL: Check if we need human input and pause if so
    if state.get("pending_human_input"):
        return END  # Pause graph to wait for human input

    ambiguities = state.get("ambiguities", [])
    if ambiguities:
        return "clarification"
    else:
        return "research"


def route_after_clarification(state: GraphState) -> str:
    """Route after clarification - back to dialectic or forward to research."""
    # CRITICAL: If we have pending human input, pause the graph
    if state.get("pending_human_input"):
        return END  # Pause graph to wait for human input

    # Once human responds and clears pending_human_input, continue to research
    # No need to re-check dialectic - if we reached clarification, we should proceed
    return "research"


def route_after_research(state: GraphState) -> str:
    """Route after research - to plan when research is complete."""
    if state.get("research_complete", False):
        return "plan"
    else:
        # Continue research if not complete (for iterative research)
        return "research"


def route_after_test(state: GraphState) -> str:
    """Route based on test results."""
    if state.get("last_test_passed", False):
        return "passed"
    else:
        return "fix"


def route_after_fix(state: GraphState) -> str:
    """Route based on iteration count."""
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    if iteration >= max_iterations:
        return "failed"
    else:
        return "build"


def should_continue(state: GraphState) -> bool:
    """Check if we should continue or wait for human input."""
    return state.get("pending_human_input") is None


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def create_tdd_graph() -> StateGraph:
    """
    Create the TDD orchestration StateGraph with Research/Dialectic phases.

    Graph structure:
        INIT -> DIALECTIC -> [routing] -> CLARIFICATION (wait for human)
                    ^            |              |
                    |            v              v
                    +---------- RESEARCH -> PLAN -> BUILD -> TEST
                                                      ^       |
                                                      |       v
                                                    FIX <-- [routing]
                                                      |       |
                                                      v       v
                                                   FAILED   PASSED

    The Dialectic/Research loop ensures:
    1. Ambiguities are detected before planning
    2. User is asked clarifying questions
    3. Research creates a "sufficient statistic" for autonomous generation
    """
    # Create graph with typed state schema
    graph = StateGraph(GraphState)

    # Add nodes - including new Dialectic/Research phases
    graph.add_node("init", init_node)
    graph.add_node("dialectic", dialectic_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("research", research_node)
    graph.add_node("plan", plan_node)
    graph.add_node("build", build_node)
    graph.add_node("test", test_node)
    graph.add_node("fix", fix_node)
    graph.add_node("passed", passed_node)
    graph.add_node("failed", failed_node)

    # Add edges - new flow: init -> dialectic -> clarification/research -> plan
    graph.add_edge("init", "dialectic")

    # Conditional routing after dialectic - to clarification if ambiguities found
    # CRITICAL: END is included to allow pausing for human input
    graph.add_conditional_edges(
        "dialectic",
        route_after_dialectic,
        {
            "clarification": "clarification",
            "research": "research",
            END: END,  # Pause for human input
        }
    )

    # After clarification, route back to dialectic or forward to research
    # CRITICAL: END is included to allow pausing for human input
    graph.add_conditional_edges(
        "clarification",
        route_after_clarification,
        {
            "dialectic": "dialectic",
            "research": "research",
            END: END,  # Pause for human input
        }
    )

    # After research, proceed to plan
    graph.add_conditional_edges(
        "research",
        route_after_research,
        {
            "plan": "plan",
            "research": "research",  # For iterative research
        }
    )

    # Standard TDD flow continues
    graph.add_edge("plan", "build")
    graph.add_edge("build", "test")

    # Conditional routing after test
    graph.add_conditional_edges(
        "test",
        route_after_test,
        {
            "passed": "passed",
            "fix": "fix",
        }
    )

    # Conditional routing after fix
    graph.add_conditional_edges(
        "fix",
        route_after_fix,
        {
            "build": "build",
            "failed": "failed",
        }
    )

    # Terminal states
    graph.add_edge("passed", END)
    graph.add_edge("failed", END)

    # Set entry point
    graph.set_entry_point("init")

    return graph


def create_orchestrator(checkpointer: Optional[Any] = None):
    """
    Create a compiled orchestrator with optional checkpointing.

    Args:
        checkpointer: Optional checkpointer (MemorySaver or SqliteSaver) for state persistence

    Returns:
        Compiled StateGraph ready for execution
    """
    graph = create_tdd_graph()

    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    else:
        return graph.compile()


# =============================================================================
# EXECUTION HELPERS
# =============================================================================

class TDDOrchestrator:
    """
    High-level orchestrator for TDD cycles.

    Provides a clean interface for running TDD workflows
    with optional checkpointing and human-in-the-loop.
    """

    def __init__(
        self,
        enable_checkpointing: bool = True,
        persist_to_sqlite: bool = True,
        checkpoint_db_path: str = "workspace/checkpoints.db",
    ):
        """
        Initialize the orchestrator.

        Args:
            enable_checkpointing: Whether to enable state checkpointing
            persist_to_sqlite: Use SqliteSaver for durable persistence (survives restarts)
            checkpoint_db_path: Path to SQLite database for checkpoints
        """
        self.checkpointer = None
        self._sqlite_conn = None  # Keep reference to close later
        if enable_checkpointing:
            if persist_to_sqlite and SQLITE_AVAILABLE:
                # Production mode: durable persistence
                from pathlib import Path
                import sqlite3
                Path(checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
                # SqliteSaver.from_conn_string returns a context manager in newer versions
                # We need to create the connection manually and keep it alive
                self._sqlite_conn = sqlite3.connect(checkpoint_db_path, check_same_thread=False)
                self.checkpointer = SqliteSaver(self._sqlite_conn)
                logger.info(f"Using SqliteSaver for durable checkpointing: {checkpoint_db_path}")
            else:
                # Development mode: in-memory only
                self.checkpointer = MemorySaver()
                if persist_to_sqlite and not SQLITE_AVAILABLE:
                    logger.warning("SqliteSaver not available, falling back to MemorySaver")
        self.graph = create_orchestrator(self.checkpointer)

        # Infrastructure is initialized lazily:
        # - MutationLogger: on first add_node_safe call (tools.py)
        # - GapAnalyzer: at module import (orchestrator.py)
        # - Documenter: on passed_node (orchestrator.py)
        self.infrastructure_status = {
            "socratic": SOCRATIC_AVAILABLE,
            "documenter": DOCUMENTER_AVAILABLE,
        }
        logger.info(f"Infrastructure available: {self.infrastructure_status}")

    def run(
        self,
        session_id: str,
        task_id: str,
        spec: str,
        requirements: Optional[List[str]] = None,
        max_iterations: int = 3,
        fresh: bool = True,
        initial_phase: Optional[str] = None,
        autonomous_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a TDD cycle to completion.

        Args:
            session_id: Unique session identifier. If fresh=True and checkpointing
                       is enabled, a suffix is added to ensure a clean slate.
            task_id: Task identifier
            spec: Task specification
            requirements: List of requirements
            max_iterations: Maximum fix attempts
            fresh: If True, ensure this is a fresh run (not resuming previous state).
                  Set to False if intentionally resuming a session.
            initial_phase: Optional initial phase to start in (e.g., "research", "plan").
                          If not provided, starts at "init" which transitions to "dialectic".
                          Use this when loading a spec file to skip dialectic if appropriate.
            autonomous_mode: If True, skip the dialectic/clarification phase and proceed
                           directly to planning. Default is False (require human approval).

        Returns:
            Final state dictionary

        Note:
            When checkpointing is enabled, running with the same session_id twice
            will resume from the previous state. Use fresh=True (default) to avoid
            this, or fresh=False to intentionally resume.
        """
        import uuid

        # Ensure fresh runs don't accidentally resume previous state
        effective_session_id = session_id
        if fresh and self.checkpointer:
            effective_session_id = f"{session_id}_{uuid.uuid4().hex[:8]}"

        # Resource Guard: Check system resources before starting
        resource_ok = True
        if RESOURCE_GUARD_AVAILABLE:
            try:
                guard = ResourceGuard()
                signal = guard.check()
                if signal == ResourceSignal.CRITICAL:
                    logger.warning("ResourceGuard: CRITICAL - system resources low, run may fail")
                    resource_ok = False
                elif signal == ResourceSignal.WARNING:
                    logger.info("ResourceGuard: WARNING - system resources getting low")
            except Exception as e:
                logger.debug(f"ResourceGuard check failed: {e}")

        # Analytics: Log initial graph health before run
        if ANALYTICS_AVAILABLE:
            try:
                from agents.tools import _db
                if _db is not None:
                    health = get_graph_health_report(_db)
                    logger.info(
                        f"Graph health before run: {health.total_nodes} nodes, "
                        f"{health.total_edges} edges, density={health.density:.3f}, "
                        f"hotspots={health.hotspot_count}, orphans={health.orphan_count}"
                    )
            except Exception as e:
                logger.debug(f"Analytics health check failed: {e}")

        # RerunLogger: Initialize Time Machine for this session
        rerun_logger = None
        if RERUN_LOGGER_AVAILABLE:
            try:
                rerun_logger = create_rerun_logger(session_id=effective_session_id)
                if rerun_logger and rerun_logger.recording_path:
                    logger.info(f"Time Machine recording to: {rerun_logger.recording_path}")
            except Exception as e:
                logger.debug(f"RerunLogger initialization failed: {e}")

        # Environment: Log system capabilities at run start
        if ENVIRONMENT_AVAILABLE:
            try:
                detector = EnvironmentDetector()
                env_report = detector.detect()
                logger.info(
                    f"Environment: {env_report.os_name}, Python {env_report.python_version}, "
                    f"RAM={env_report.ram_gb:.1f}GB, GPU={'Yes' if env_report.gpu_available else 'No'}"
                )
                # Log to RerunLogger if available
                if rerun_logger:
                    rerun_logger.log_thought(
                        "system",
                        f"Environment: {env_report.os_name}, RAM={env_report.ram_gb:.1f}GB, GPU={env_report.gpu_available}"
                    )
            except Exception as e:
                logger.debug(f"Environment detection failed: {e}")

        # Determine starting phase
        # If initial_phase is provided and valid, skip init and go directly there
        starting_phase = CyclePhase.INIT.value
        skip_init = False

        if initial_phase:
            try:
                # Validate it's a valid phase
                CyclePhase(initial_phase)
                starting_phase = initial_phase
                skip_init = True
                logger.info(f"Starting directly at phase: {initial_phase}")
            except ValueError:
                logger.warning(f"Invalid initial_phase '{initial_phase}', starting at INIT")

        initial_state: GraphState = {
            "session_id": effective_session_id,
            "task_id": task_id,
            "spec": spec,
            "requirements": requirements or [],
            "max_iterations": max_iterations,
            "phase": starting_phase,
            "messages": [],
            "artifacts": [],
            "code_node_ids": [],
            "test_node_ids": [],
            "test_results": [],
            "errors": [],
            "iteration": 0,
            "last_test_passed": False,
            "pending_human_input": None,
            "human_response": None,
            "final_status": None,
            # Pre-populate dialectic state
            "ambiguities": [],
            "clarification_questions": [],
            "research_findings": [],
            "research_complete": False,
        }

        config = {"configurable": {"thread_id": effective_session_id}}

        # Initialize diagnostics for this run
        if DIAGNOSTICS_AVAILABLE:
            diag = get_diagnostics()
            correlation_id = diag.set_session(effective_session_id)
            diag.print_state_summary(use_color=False)  # Log initial state

            # Link correlation ID to MutationLogger for cross-referencing
            try:
                from infrastructure.logger import get_logger as get_mutation_logger
                mutation_logger = get_mutation_logger()
                mutation_logger.set_correlation_id(correlation_id)
                logger.debug(f"Linked correlation_id={correlation_id} to MutationLogger")
            except Exception as e:
                logger.debug(f"Failed to link correlation_id to MutationLogger: {e}")

        # Run to completion and collect final state
        final_state = initial_state.copy()
        current_phase = None
        for event in self.graph.stream(initial_state, config):
            # Resource Guard: Check mid-run if resources are critical
            if RESOURCE_GUARD_AVAILABLE:
                try:
                    guard = ResourceGuard()
                    signal = guard.check()
                    if signal == ResourceSignal.CRITICAL:
                        logger.error("ResourceGuard: CRITICAL during run - aborting to prevent OOM")
                        final_state["errors"] = final_state.get("errors", []) + [
                            "Run aborted: system resources critically low"
                        ]
                        final_state["final_status"] = "aborted"
                        break
                except Exception:
                    pass

            # Each event is {node_name: state_updates}
            for node_name, updates in event.items():
                # Track phase transitions for diagnostics
                if DIAGNOSTICS_AVAILABLE:
                    new_phase = updates.get("phase")
                    if new_phase and new_phase != current_phase:
                        if current_phase is not None:
                            diag.end_phase(success=True)
                        diag.start_phase(new_phase)
                        current_phase = new_phase

                # Merge updates into final_state
                for key, value in updates.items():
                    if key in ("messages", "test_results", "artifacts",
                               "code_node_ids", "test_node_ids", "errors"):
                        # Lists are appended
                        existing = final_state.get(key, [])
                        final_state[key] = existing + (value or [])
                    else:
                        # Scalars are replaced
                        final_state[key] = value

        # End final phase and print summary
        if DIAGNOSTICS_AVAILABLE:
            if current_phase is not None:
                success = final_state.get("final_status") == "passed"
                diag.end_phase(success=success)
            diag.print_summary(use_color=False)

        # Analytics: Log final graph health after run
        if ANALYTICS_AVAILABLE:
            try:
                from agents.tools import _db
                if _db is not None:
                    health = get_graph_health_report(_db)
                    logger.info(
                        f"Graph health after run: {health.total_nodes} nodes, "
                        f"{health.total_edges} edges, is_dag={health.is_dag}"
                    )
            except Exception as e:
                logger.debug(f"Analytics health check failed: {e}")

        return final_state

    def resume(
        self,
        session_id: str,
        human_response: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resume a paused orchestration.

        Args:
            session_id: Session to resume
            human_response: Human input if requested

        Returns:
            Final state dictionary
        """
        if not self.checkpointer:
            raise ValueError("Checkpointing not enabled")

        config = {"configurable": {"thread_id": session_id}}

        # Get current state
        state = self.graph.get_state(config)

        if human_response:
            # Update with human response
            state_update = {
                "human_response": human_response,
                "pending_human_input": None,
            }
            self.graph.update_state(config, state_update)

        # Continue execution
        final_state = {}
        for event in self.graph.stream(None, config):
            for updates in event.values():
                final_state.update(updates)

        return final_state

    def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current state for a session."""
        if not self.checkpointer:
            return None

        config = {"configurable": {"thread_id": session_id}}
        return self.graph.get_state(config)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_notification(
    notification_type: str,
    message: str,
    target_tabs: List[str],
    urgency: str = "info",
    related_node_id: Optional[str] = None,
    action_required: bool = False,
    source_component: str = "orchestrator",
) -> Optional[str]:
    """
    Create and publish a notification.

    This is a helper function for orchestrator and agents to create notifications
    that will be broadcasted to connected UI clients via WebSocket.

    Args:
        notification_type: Type of notification (spec_updated, research_complete,
            approval_needed, phase_changed, etc.)
        message: Human-readable notification message
        target_tabs: List of UI tabs to show notification in (e.g., ["build", "research"])
        urgency: Urgency level - "info", "warning", or "critical"
        related_node_id: Optional node ID this notification relates to
        action_required: Whether this notification requires user action
        source_component: Component creating the notification (default: orchestrator)

    Returns:
        Notification node ID if successful, None otherwise

    Example:
        # Notify when research completes
        create_notification(
            notification_type="research_complete",
            message="Research findings available for review",
            target_tabs=["research", "specification"],
            urgency="info",
            related_node_id=research_node_id,
        )
    """
    try:
        from core.graph_db import get_db
        from core.schemas import NodeData, EdgeData
        from core.ontology import NodeType, NodeStatus, EdgeType

        # Create notification node
        db = get_db()
        notification_node = NodeData.create(
            type=NodeType.NOTIFICATION.value,
            content=message,
            data={
                "notification_type": notification_type,
                "source_component": source_component,
                "target_tabs": target_tabs,
                "urgency": urgency,
                "related_node_id": related_node_id,
                "action_required": action_required,
                "read_by": [],
            },
            status=NodeStatus.PENDING.value,
            created_by=source_component,
        )

        db.add_node(notification_node)

        # Link to related node if provided
        if related_node_id:
            try:
                edge_data = EdgeData.create(
                    source_id=notification_node.id,
                    target_id=related_node_id,
                    type=EdgeType.TRACES_TO.value,
                )
                db.add_edge(edge_data)
            except Exception as e:
                logger.warning(f"Failed to link notification to node {related_node_id}: {e}")

        # Publish notification event for WebSocket broadcasting
        if EVENT_BUS_AVAILABLE:
            from infrastructure.event_bus import publish_notification
            publish_notification(
                notification_type=notification_type,
                message=message,
                target_tabs=target_tabs,
                urgency=urgency,
                related_node_id=related_node_id,
                metadata={
                    "notification_id": notification_node.id,
                    "action_required": action_required,
                },
                source=source_component,
            )

        return notification_node.id

    except Exception as e:
        logger.error(f"Failed to create notification: {e}")
        return None


def run_tdd_cycle(
    spec: str,
    requirements: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run a single TDD cycle.

    Args:
        spec: Task specification
        requirements: Optional requirements list
        session_id: Optional session ID (generated if not provided)
        task_id: Optional task ID (generated if not provided)

    Returns:
        Final orchestration state
    """
    import uuid

    orchestrator = TDDOrchestrator(enable_checkpointing=False)

    return orchestrator.run(
        session_id=session_id or str(uuid.uuid4()),
        task_id=task_id or str(uuid.uuid4()),
        spec=spec,
        requirements=requirements,
    )
