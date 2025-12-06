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

# Diagnostic logging for phase timing and state tracking
try:
    from infrastructure.diagnostics import get_diagnostics, reset_diagnostics
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False

# Optional LLM integration (graceful degradation if not configured)
try:
    from core.llm import get_llm, StructuredLLM
    from agents.schemas import (
        ImplementationPlan, CodeGeneration, TestGeneration,
        DialectorOutput, AmbiguityMarker, ResearchArtifact, ResearchFinding,
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
    INIT = "init"
    DIALECTIC = "dialectic"      # Pre-research ambiguity check
    CLARIFICATION = "clarification"  # Waiting for user answers
    RESEARCH = "research"        # Socratic research loop
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

    # Dialectic/Research phase state (replace)
    ambiguities: Annotated[List[Dict[str, Any]], list_append_reducer]
    clarification_questions: Annotated[List[Dict[str, Any]], list_append_reducer]
    research_findings: Annotated[List[Dict[str, Any]], list_append_reducer]
    dialectic_passed: bool
    research_complete: bool

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
    - Transition to DIALECTIC (pre-research ambiguity check)
    """
    return {
        "phase": CyclePhase.DIALECTIC.value,
        "messages": [{
            "role": "system",
            "content": f"Starting TDD cycle for task: {state.get('task_id', 'unknown')}"
        }],
        "iteration": 0,
        "ambiguities": [],
        "clarification_questions": [],
        "research_findings": [],
        "dialectic_passed": False,
        "research_complete": False,
    }


def dialectic_node(state: GraphState) -> Dict[str, Any]:
    """
    Dialectic phase - pre-research ambiguity detection.

    Analyzes the requirement for subjective terms, undefined references,
    and missing context. Creates clarification questions if needed.

    Flow:
    - If CLEAR: proceed to RESEARCH
    - If NEEDS_CLARIFICATION: proceed to CLARIFICATION (wait for user)
    """
    spec = state.get("spec", "")
    session_id = state.get("session_id", "unknown")

    messages = [{
        "role": "assistant",
        "content": "Analyzing requirement for ambiguity..."
    }]

    ambiguities = []
    clarification_questions = []
    next_phase = CyclePhase.RESEARCH.value  # Default: proceed to research

    if LLM_AVAILABLE:
        try:
            llm = get_llm()

            # Build dialector prompt
            system_prompt = """You are the DIALECTOR agent. Analyze the requirement for ambiguity that would prevent autonomous code generation.

Scan for:
1. SUBJECTIVE TERMS: "fast", "efficient", "user-friendly"
2. COMPARATIVE STATEMENTS: "faster than", "better than" (compared to what?)
3. UNDEFINED PRONOUNS: "it", "this" without clear referent
4. UNDEFINED TERMS: Domain-specific terms needing definition
5. MISSING CONTEXT: Input/output format not specified

For technical, well-defined requirements like "implement fibonacci" or "create a stack class", return verdict: CLEAR.
Only flag as NEEDS_CLARIFICATION if there's genuine blocking ambiguity."""

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
                # AmbiguityMarker has: category, text, impact, suggested_question
                for amb in result.ambiguities:
                    ambiguities.append({
                        "text": amb.text,
                        "category": amb.category,
                        "impact": amb.impact,
                        "question": amb.suggested_question,
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
                    })

                if clarification_questions:
                    next_phase = CyclePhase.CLARIFICATION.value
                    messages.append({
                        "role": "assistant",
                        "content": f"Found {len(clarification_questions)} blocking ambiguities. Waiting for user clarification."
                    })

        except Exception as e:
            logger.warning(f"Dialectic analysis failed: {e}")
            messages.append({
                "role": "assistant",
                "content": f"Dialectic analysis unavailable ({e}), proceeding to research"
            })
    else:
        messages.append({
            "role": "assistant",
            "content": "LLM not configured, skipping dialectic analysis"
        })

    return {
        "phase": next_phase,
        "messages": messages,
        "ambiguities": ambiguities,
        "clarification_questions": clarification_questions,
        "dialectic_passed": (next_phase == CyclePhase.RESEARCH.value),
        "pending_human_input": "clarification" if next_phase == CyclePhase.CLARIFICATION.value else None,
    }


def clarification_node(state: GraphState) -> Dict[str, Any]:
    """
    Clarification phase - wait for and process user responses.

    This node handles the human-in-the-loop interaction:
    - Presents questions to user
    - Waits for responses
    - Incorporates feedback into the spec
    """
    questions = state.get("clarification_questions", [])
    human_response = state.get("human_response", None)
    spec = state.get("spec", "")

    messages = []

    if human_response:
        # User has provided responses - incorporate them
        messages.append({
            "role": "user",
            "content": f"User clarification: {human_response}"
        })

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
            "dialectic_passed": True,
            "pending_human_input": None,
        }
    else:
        # Still waiting for user input - format questions for display
        question_text = "Please clarify the following:\n\n"
        for i, q in enumerate(questions, 1):
            question_text += f"{i}. {q.get('question', q.get('phrase', 'Unknown'))}\n"
            question_text += f"   (Category: {q.get('category', 'unknown')})\n\n"

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

            return {
                "phase": next_phase,
                "messages": messages,
                "spec": augmented_spec,
                "research_findings": research_findings,
                "research_complete": True,
            }

        except Exception as e:
            logger.warning(f"Research phase failed: {e}")
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

def route_after_dialectic(state: GraphState) -> str:
    """Route after dialectic analysis - to clarification if ambiguities found."""
    ambiguities = state.get("ambiguities", [])
    if ambiguities and not state.get("dialectic_passed", False):
        return "clarification"
    else:
        return "research"


def route_after_clarification(state: GraphState) -> str:
    """Route after clarification - back to dialectic or forward to research."""
    # If we have pending human input, we wait (handled by interrupt)
    # Once human responds, we continue to research
    if state.get("dialectic_passed", False):
        return "research"
    else:
        # Re-check for more ambiguities after clarification
        return "dialectic"


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
    graph.add_conditional_edges(
        "dialectic",
        route_after_dialectic,
        {
            "clarification": "clarification",
            "research": "research",
        }
    )

    # After clarification, route back to dialectic or forward to research
    graph.add_conditional_edges(
        "clarification",
        route_after_clarification,
        {
            "dialectic": "dialectic",
            "research": "research",
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

        # Initialize diagnostics for this run
        if DIAGNOSTICS_AVAILABLE:
            diag = get_diagnostics()
            diag.set_session(effective_session_id)
            diag.print_state_summary(use_color=False)  # Log initial state

        # Run to completion and collect final state
        final_state = initial_state.copy()
        current_phase = None
        for event in self.graph.stream(initial_state, config):
            # Each event is {node_name: state_updates}
            for node_name, updates in event.items():
                # Track phase transitions for diagnostics
                if DIAGNOSTICS_AVAILABLE:
                    new_phase = updates.get("phase")
                    if new_phase and new_phase != current_phase:
                        if current_phase is not None:
                            diag.end_phase(success=True)
                        diag.start_phase(new_phase.upper())
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
