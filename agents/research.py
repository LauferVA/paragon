"""
PARAGON RESEARCHER - Socratic Research Loop

Implements the RESEARCHER agent with nested LangGraph state machine:
    RESEARCH <-> CRITIQUE -> SYNTHESIZE

The Dialector persona finds ambiguity in requirements.
Socrates formulates precise questions.
The loop continues until out_of_scope items are identified.

Design Philosophy:
- Nested state machine for research refinement
- Socratic method: Question assumptions, find edge cases
- Stops when clarity is achieved (out_of_scope populated)

Layer 7 Integration:
- Uses StructuredLLM for generating research artifacts
- Integrates with Tavily for web search
- Creates RESEARCH nodes linked to REQ nodes
"""
from typing import List, Dict, Any, Optional, Literal, Annotated, TypedDict
from enum import Enum
import msgspec
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.tools_web import search_web, SearchResult, SearchResponse
from agents.tools import get_db, NodeResult
from core.ontology import NodeType, EdgeType, NodeStatus

# LLM integration (graceful degradation if not available)
try:
    from core.llm import get_llm, StructuredLLM
    from agents.schemas import ResearchArtifact, ResearchFinding, DialectorOutput, AmbiguityMarker
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DEFINITIONS
# =============================================================================

class ResearchPhase(str, Enum):
    """Phases in the research cycle."""
    INIT = "init"
    RESEARCH = "research"
    CRITIQUE = "critique"
    SYNTHESIZE = "synthesize"
    COMPLETE = "complete"


def list_append_reducer(existing: List, new: List) -> List:
    """Append new items to existing list."""
    if existing is None:
        return new or []
    if new is None:
        return existing
    return existing + new


class ResearchState(TypedDict):
    """
    State for the research cycle.

    Uses Annotated with reducers for proper state merging.
    """
    # Identification
    req_node_id: str
    requirement_text: str

    # Current phase
    phase: str

    # Research accumulation
    query: str
    findings: Annotated[List[Dict[str, Any]], list_append_reducer]
    search_results: Annotated[List[Dict[str, Any]], list_append_reducer]

    # Critique accumulation
    critiques: Annotated[List[str], list_append_reducer]
    ambiguities: Annotated[List[Dict[str, Any]], list_append_reducer]

    # Synthesis output
    synthesis: str
    out_of_scope: List[str]

    # Control flow
    iteration: int
    max_iterations: int
    is_complete: bool

    # Messages (for debugging/logging)
    messages: Annotated[List[Dict[str, Any]], list_append_reducer]


# =============================================================================
# NODE FUNCTIONS (State Machine Nodes)
# =============================================================================

def init_node(state: ResearchState) -> Dict[str, Any]:
    """
    Initialize the research cycle.

    Sets up initial state and transitions to RESEARCH.
    """
    return {
        "phase": ResearchPhase.RESEARCH.value,
        "iteration": 0,
        "is_complete": False,
        "messages": [{
            "role": "system",
            "content": f"Starting research for requirement: {state.get('req_node_id', 'unknown')}"
        }],
    }


def research_node(state: ResearchState) -> Dict[str, Any]:
    """
    Research phase - perform web searches and gather information.

    Uses Tavily to search for relevant information based on the requirement.
    Stores findings for critique phase.
    """
    iteration = state.get("iteration", 0)
    requirement_text = state.get("requirement_text", "")
    query = state.get("query", "")

    messages = [{
        "role": "assistant",
        "content": f"Research iteration {iteration + 1}: Searching for information"
    }]

    findings = []
    search_results = []

    # Generate research query from requirement
    if not query:
        # First iteration - use requirement text as base query
        query = requirement_text[:200]  # Truncate if too long
    else:
        # Later iterations - refine based on critiques
        critiques = state.get("critiques", [])
        if critiques:
            # Use latest critique to refine search
            query = f"{requirement_text} {critiques[-1]}"

    # Perform web search
    try:
        response = search_web(
            query=query,
            context=requirement_text,
            max_results=5,
            search_depth="advanced",
        )

        # Store search results
        for result in response.results:
            search_results.append({
                "title": result.title,
                "url": result.url,
                "content": result.content,
                "score": result.score,
            })

            # Convert to research findings
            findings.append({
                "topic": result.title,
                "summary": result.content[:500],  # Truncate for storage
                "source": result.url,
                "confidence": result.score,
            })

        messages.append({
            "role": "tool",
            "content": f"Found {len(search_results)} results for query: {query}",
            "tool_name": "search_web",
            "tool_result": {"total_results": response.total_results},
        })

    except Exception as e:
        logger.error(f"Research search failed: {e}")
        messages.append({
            "role": "assistant",
            "content": f"Search failed: {str(e)}. Proceeding with available information."
        })

    return {
        "phase": ResearchPhase.CRITIQUE.value,
        "query": query,
        "findings": findings,
        "search_results": search_results,
        "messages": messages,
    }


def critique_node(state: ResearchState) -> Dict[str, Any]:
    """
    Critique phase - The Dialector finds ambiguity.

    Analyzes the requirement and research findings to identify:
    - Subjective terms without definition
    - Missing units or measurements
    - Ambiguous pronouns or references
    - Undefined technical terms
    """
    requirement_text = state.get("requirement_text", "")
    findings = state.get("findings", [])
    iteration = state.get("iteration", 0)

    messages = [{
        "role": "assistant",
        "content": f"Critique iteration {iteration + 1}: Analyzing for ambiguities"
    }]

    critiques = []
    ambiguities = []

    if LLM_AVAILABLE:
        try:
            llm = get_llm()

            # Build Dialector prompt
            system_prompt = """You are the Dialector, an expert at finding ambiguity in requirements.

Your role is to identify:
1. SUBJECTIVE terms (e.g., "fast", "simple", "good") without quantification
2. COMPARATIVE statements (e.g., "better", "more") without baseline
3. UNDEFINED pronouns (e.g., "it", "they") without clear referent
4. UNDEFINED technical terms or acronyms
5. MISSING context (e.g., "when X happens" but X is not defined)

For each ambiguity, categorize its IMPACT:
- BLOCKING: Must be resolved to proceed
- CLARIFYING: Would improve quality but not blocking

Output structured ambiguity markers."""

            findings_text = "\n".join([
                f"- {f.get('topic', 'Unknown')}: {f.get('summary', '')[:200]}"
                for f in findings[:5]  # Limit to 5 most recent
            ])

            user_prompt = f"""# Requirement to Analyze

{requirement_text}

# Research Findings

{findings_text if findings_text else "No findings yet."}

# Task

Identify ALL ambiguities in the requirement that could lead to implementation confusion.
Focus on terms that need precise definition or measurement."""

            # Generate critique using structured output
            result = llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=DialectorOutput,
            )

            # Extract ambiguities
            for ambiguity in result.ambiguities:
                ambiguities.append({
                    "category": ambiguity.category,
                    "text": ambiguity.text,
                    "impact": ambiguity.impact,
                    "suggested_question": ambiguity.suggested_question,
                })

                # Add critique text
                critique_text = f"{ambiguity.category}: '{ambiguity.text}' - {ambiguity.suggested_question or 'Needs clarification'}"
                critiques.append(critique_text)

            messages.append({
                "role": "assistant",
                "content": f"Found {len(ambiguities)} ambiguities. Is clear: {result.is_clear}"
            })

            # Check if we should stop (requirement is clear enough)
            if result.is_clear or result.recommendation == "PROCEED":
                # Move to synthesis
                return {
                    "phase": ResearchPhase.SYNTHESIZE.value,
                    "critiques": critiques,
                    "ambiguities": ambiguities,
                    "messages": messages,
                }

        except Exception as e:
            logger.error(f"LLM critique failed: {e}")
            messages.append({
                "role": "assistant",
                "content": f"Critique generation failed: {str(e)}. Using heuristics."
            })

    # Fallback: Simple heuristic critique
    if not critiques:
        # Look for common ambiguity patterns
        subjective_terms = ["fast", "simple", "good", "better", "easy", "complex", "large", "small"]
        for term in subjective_terms:
            if term.lower() in requirement_text.lower():
                critiques.append(f"SUBJECTIVE: '{term}' needs quantification")
                ambiguities.append({
                    "category": "SUBJECTIVE",
                    "text": term,
                    "impact": "CLARIFYING",
                    "suggested_question": f"What specific metric defines '{term}'?",
                })

    return {
        "phase": ResearchPhase.SYNTHESIZE.value,
        "critiques": critiques,
        "ambiguities": ambiguities,
        "messages": messages,
    }


def synthesize_node(state: ResearchState) -> Dict[str, Any]:
    """
    Synthesis phase - Combine findings into coherent output.

    Creates a structured research artifact following Research Standard v1.0.
    Populates out_of_scope with items that need clarification.
    """
    requirement_text = state.get("requirement_text", "")
    findings = state.get("findings", [])
    ambiguities = state.get("ambiguities", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    req_node_id = state.get("req_node_id", "")

    messages = [{
        "role": "assistant",
        "content": f"Synthesis iteration {iteration + 1}: Creating research artifact"
    }]

    # Increment iteration counter
    iteration += 1

    # Determine out_of_scope items (blocking ambiguities)
    out_of_scope = []
    blocking_ambiguities = [
        a for a in ambiguities
        if a.get("impact") == "BLOCKING"
    ]

    if blocking_ambiguities:
        # We have blocking ambiguities - these are out of scope
        for ambiguity in blocking_ambiguities:
            out_of_scope.append(
                f"{ambiguity.get('text', 'Unknown')}: {ambiguity.get('suggested_question', 'Needs clarification')}"
            )

    # Create synthesis text
    synthesis_lines = [
        f"# Research Synthesis for Requirement\n",
        f"**Iterations:** {iteration}/{max_iterations}\n",
        f"\n## Original Requirement\n",
        f"{requirement_text}\n",
        f"\n## Key Findings ({len(findings)})\n",
    ]

    for i, finding in enumerate(findings[:10], 1):  # Limit to 10 most relevant
        synthesis_lines.append(f"\n### {i}. {finding.get('topic', 'Unknown')}")
        synthesis_lines.append(f"**Confidence:** {finding.get('confidence', 0.0):.2f}")
        if finding.get('source'):
            synthesis_lines.append(f"**Source:** {finding.get('source')}")
        synthesis_lines.append(f"\n{finding.get('summary', '')}\n")

    synthesis_lines.append(f"\n## Ambiguities Identified ({len(ambiguities)})\n")
    for ambiguity in ambiguities:
        synthesis_lines.append(
            f"- **{ambiguity.get('category')}** [{ambiguity.get('impact')}]: "
            f"'{ambiguity.get('text')}' - {ambiguity.get('suggested_question', 'Needs clarification')}"
        )

    if out_of_scope:
        synthesis_lines.append(f"\n## Out of Scope (Requires Clarification)\n")
        for item in out_of_scope:
            synthesis_lines.append(f"- {item}")

    synthesis = "\n".join(synthesis_lines)

    # Create RESEARCH node in graph
    db = get_db()
    try:
        node = db.add_node(
            node_type=NodeType.RESEARCH.value,
            content=synthesis,
            data={
                "iteration": iteration,
                "total_findings": len(findings),
                "total_ambiguities": len(ambiguities),
                "blocking_count": len(blocking_ambiguities),
                "out_of_scope": out_of_scope,
            },
            created_by="researcher_agent",
            status=NodeStatus.PENDING.value,
        )

        # Link to REQ node
        if req_node_id:
            db.add_edge(
                source_id=node.id,
                target_id=req_node_id,
                edge_type=EdgeType.RESEARCH_FOR.value,
                created_by="researcher_agent",
            )

        messages.append({
            "role": "tool",
            "content": f"Created RESEARCH node: {node.id}",
            "tool_name": "add_node",
            "tool_result": {"node_id": node.id},
        })

    except Exception as e:
        logger.error(f"Failed to create RESEARCH node: {e}")
        messages.append({
            "role": "assistant",
            "content": f"Failed to create node: {str(e)}"
        })

    # Determine if we're complete
    # Complete if: (1) we have out_of_scope items, OR (2) max iterations reached
    is_complete = (len(out_of_scope) > 0) or (iteration >= max_iterations)

    if is_complete:
        messages.append({
            "role": "assistant",
            "content": f"Research complete. Out of scope items: {len(out_of_scope)}"
        })

    return {
        "phase": ResearchPhase.COMPLETE.value if is_complete else ResearchPhase.RESEARCH.value,
        "synthesis": synthesis,
        "out_of_scope": out_of_scope,
        "iteration": iteration,
        "is_complete": is_complete,
        "messages": messages,
    }


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def should_continue(state: ResearchState) -> str:
    """Route based on completion status."""
    if state.get("is_complete", False):
        return "complete"
    else:
        # Loop back to research for another iteration
        return "research"


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def create_research_graph() -> StateGraph:
    """
    Create the research orchestration StateGraph.

    Graph structure:
        INIT -> RESEARCH -> CRITIQUE -> SYNTHESIZE
                  ^                         |
                  |                         v
                  +---- [routing] -----> COMPLETE
    """
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("init", init_node)
    graph.add_node("research", research_node)
    graph.add_node("critique", critique_node)
    graph.add_node("synthesize", synthesize_node)

    # Add edges
    graph.add_edge("init", "research")
    graph.add_edge("research", "critique")
    graph.add_edge("critique", "synthesize")

    # Conditional routing after synthesis
    graph.add_conditional_edges(
        "synthesize",
        should_continue,
        {
            "research": "research",  # Loop back for another iteration
            "complete": END,         # Terminal state
        }
    )

    # Set entry point
    graph.set_entry_point("init")

    return graph


# =============================================================================
# RESEARCH ORCHESTRATOR
# =============================================================================

class ResearchOrchestrator:
    """
    High-level orchestrator for research cycles.

    Provides a clean interface for running research workflows.
    """

    def __init__(self, enable_checkpointing: bool = False):
        """
        Initialize the research orchestrator.

        Args:
            enable_checkpointing: Whether to enable state checkpointing
        """
        self.checkpointer = MemorySaver() if enable_checkpointing else None
        self.graph = create_research_graph()
        if self.checkpointer:
            self.compiled = self.graph.compile(checkpointer=self.checkpointer)
        else:
            self.compiled = self.graph.compile()

    def run(
        self,
        req_node_id: str,
        requirement_text: str,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Run a research cycle to completion.

        Args:
            req_node_id: The REQ node ID this research is for
            requirement_text: The requirement text to research
            max_iterations: Maximum research iterations (default: 3)

        Returns:
            Final state dictionary
        """
        initial_state: ResearchState = {
            "req_node_id": req_node_id,
            "requirement_text": requirement_text,
            "phase": ResearchPhase.INIT.value,
            "query": "",
            "findings": [],
            "search_results": [],
            "critiques": [],
            "ambiguities": [],
            "synthesis": "",
            "out_of_scope": [],
            "iteration": 0,
            "max_iterations": max_iterations,
            "is_complete": False,
            "messages": [],
        }

        config = {"configurable": {"thread_id": req_node_id}}

        # Run to completion and collect final state
        final_state = initial_state.copy()
        for event in self.compiled.stream(initial_state, config):
            # Each event is {node_name: state_updates}
            for node_name, updates in event.items():
                # Merge updates into final_state
                for key, value in updates.items():
                    if key in ("findings", "search_results", "critiques", "ambiguities", "messages"):
                        # Lists are appended
                        existing = final_state.get(key, [])
                        final_state[key] = existing + (value or [])
                    else:
                        # Scalars are replaced
                        final_state[key] = value

        return final_state


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def research_requirement(
    req_node_id: str,
    requirement_text: str,
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """
    Convenience function to run research for a requirement.

    Args:
        req_node_id: The REQ node ID
        requirement_text: The requirement text
        max_iterations: Maximum iterations (default: 3)

    Returns:
        Final research state including synthesis and out_of_scope items
    """
    orchestrator = ResearchOrchestrator(enable_checkpointing=False)
    return orchestrator.run(
        req_node_id=req_node_id,
        requirement_text=requirement_text,
        max_iterations=max_iterations,
    )
