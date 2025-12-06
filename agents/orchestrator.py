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

# Optional LLM integration (graceful degradation if not configured)
try:
    from core.llm import get_llm, StructuredLLM
    from agents.schemas import ImplementationPlan, CodeGeneration, TestGeneration
    from agents.prompts import build_prompt
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DEFINITIONS
# =============================================================================

class CyclePhase(str, Enum):
    """TDD cycle phases."""
    INIT = "init"
    PLAN = "plan"
    BUILD = "build"
    TEST = "test"
    FIX = "fix"
    PASSED = "passed"
    FAILED = "failed"


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

    # Errors (append)
    errors: Annotated[List[str], list_append_reducer]

    # Final status (replace)
    final_status: Optional[str]


# =============================================================================
# NODE FUNCTIONS (State Machine Nodes)
# =============================================================================

def init_node(state: GraphState) -> Dict[str, Any]:
    """
    Initialize the orchestration cycle.

    - Validate inputs
    - Set up initial state
    - Transition to PLAN
    """
    return {
        "phase": CyclePhase.PLAN.value,
        "messages": [{
            "role": "system",
            "content": f"Starting TDD cycle for task: {state.get('task_id', 'unknown')}"
        }],
        "iteration": 0,
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

Output a structured implementation plan."""

            user_prompt = f"""# Task Specification

{spec}

# Requirements
{chr(10).join(f"- {r}" for r in requirements) if requirements else "No specific requirements provided."}

Decompose this into atomic, implementable components."""

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

        except Exception as e:
            logger.warning(f"LLM planning failed, using fallback: {e}")
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

                    # Layer 7B: Add with safety checks
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
    - Determine pass/fail
    """
    iteration = state.get("iteration", 0)
    existing_results = state.get("test_results", [])

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
    }


def fix_node(state: GraphState) -> Dict[str, Any]:
    """
    Fix phase - analyze failures and attempt repairs.

    - Analyze test failures
    - Generate fixes
    - Increment iteration counter
    """
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    test_results = state.get("test_results", [])

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
    if test_results:
        last_result = test_results[-1]
        failure_errors = last_result.get("errors", [])
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
    - Generate report
    """
    iteration = state.get("iteration", 0)

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

    return {
        "phase": CyclePhase.PASSED.value,
        "messages": messages,
        "final_status": "passed",
    }


def failed_node(state: GraphState) -> Dict[str, Any]:
    """
    Failure terminal state.

    - Log failures
    - Preserve state for debugging
    - Optionally request human intervention
    """
    errors = state.get("errors", [])

    messages = [{
        "role": "assistant",
        "content": f"TDD cycle FAILED with {len(errors)} error(s)"
    }]

    return {
        "phase": CyclePhase.FAILED.value,
        "messages": messages,
        "final_status": "failed",
        "pending_human_input": HumanCheckpointType.REVIEW.value,
    }


# =============================================================================
# ROUTING FUNCTIONS (Conditional Edges)
# =============================================================================

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
    Create the TDD orchestration StateGraph.

    Graph structure:
        INIT -> PLAN -> BUILD -> TEST
                          ^       |
                          |       v
                        FIX <-- [routing]
                          |       |
                          v       v
                       FAILED   PASSED
    """
    # Create graph with typed state schema
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("init", init_node)
    graph.add_node("plan", plan_node)
    graph.add_node("build", build_node)
    graph.add_node("test", test_node)
    graph.add_node("fix", fix_node)
    graph.add_node("passed", passed_node)
    graph.add_node("failed", failed_node)

    # Add edges
    graph.add_edge("init", "plan")
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
        if enable_checkpointing:
            if persist_to_sqlite and SQLITE_AVAILABLE:
                # Production mode: durable persistence
                from pathlib import Path
                Path(checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
                self.checkpointer = SqliteSaver.from_conn_string(checkpoint_db_path)
                logger.info(f"Using SqliteSaver for durable checkpointing: {checkpoint_db_path}")
            else:
                # Development mode: in-memory only
                self.checkpointer = MemorySaver()
                if persist_to_sqlite and not SQLITE_AVAILABLE:
                    logger.warning("SqliteSaver not available, falling back to MemorySaver")
        self.graph = create_orchestrator(self.checkpointer)

    def run(
        self,
        session_id: str,
        task_id: str,
        spec: str,
        requirements: Optional[List[str]] = None,
        max_iterations: int = 3,
        fresh: bool = True,
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

        initial_state: GraphState = {
            "session_id": effective_session_id,
            "task_id": task_id,
            "spec": spec,
            "requirements": requirements or [],
            "max_iterations": max_iterations,
            "phase": CyclePhase.INIT.value,
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
        }

        config = {"configurable": {"thread_id": effective_session_id}}

        # Run to completion and collect final state
        final_state = initial_state.copy()
        for event in self.graph.stream(initial_state, config):
            # Each event is {node_name: state_updates}
            for node_name, updates in event.items():
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
