"""
PARAGON INTELLIGENCE - Prompt Context Builders

Transforms graph state into structured prompts for LLM generation.
Each builder creates the context needed for a specific agent type.

Design:
- Builders pull relevant nodes/edges from ParagonDB
- Context is injected into system prompts defined in agents.yaml
- User prompts are constructed from node content + dependencies

Layer 7 Phase 1: Context building for structured generation.
"""
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Import will be from relative path when used
# from .tools import get_db
# For now, we accept db as parameter to avoid circular imports


# =============================================================================
# CONFIG LOADER
# =============================================================================

_agent_config: Optional[Dict[str, Any]] = None


def get_agent_config() -> Dict[str, Any]:
    """Load agent configuration from agents.yaml."""
    global _agent_config
    if _agent_config is None:
        config_path = Path(__file__).parent.parent / "config" / "agents.yaml"
        with open(config_path, "r") as f:
            _agent_config = yaml.safe_load(f)
    return _agent_config


def get_agent_system_prompt(agent_role: str) -> str:
    """Get the system prompt for a specific agent role."""
    config = get_agent_config()
    agent_key = agent_role.lower()
    if agent_key not in config:
        raise ValueError(f"Unknown agent role: {agent_role}")
    return config[agent_key].get("system_prompt", "")


# =============================================================================
# CONTEXT EXTRACTORS
# =============================================================================

def extract_node_context(db: Any, node_id: str) -> Dict[str, Any]:
    """
    Extract full context for a node including its content and metadata.

    Args:
        db: ParagonDB instance
        node_id: The node to extract context for

    Returns:
        Dict with node data, content, and metadata
    """
    try:
        node = db.get_node(node_id)
    except Exception:
        return {"error": f"Node {node_id} not found"}

    if node is None:
        return {"error": f"Node {node_id} not found"}

    # NodeData is a msgspec.Struct, access as attributes
    return {
        "id": node_id,
        "type": getattr(node, "type", "UNKNOWN"),
        "status": getattr(node, "status", "UNKNOWN"),
        "content": getattr(node, "content", ""),
        "metadata": getattr(node, "data", {}),
    }


def extract_predecessor_context(db: Any, node_id: str) -> List[Dict[str, Any]]:
    """
    Extract context from all predecessor nodes.

    Used by Builder to understand specs, by Verifier to check against specs, etc.
    """
    predecessors = []

    # Get incoming edges
    incoming = db.get_incoming_edges(node_id)
    for edge in incoming:
        source_id = edge.get("source")
        if source_id:
            pred_context = extract_node_context(db, source_id)
            pred_context["edge_type"] = edge.get("type", "UNKNOWN")
            predecessors.append(pred_context)

    return predecessors


def extract_dependency_chain(
    db: Any,
    node_id: str,
    max_depth: int = 5,
    use_dominators: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extract the dependency chain for a node using Economic Topology.

    Two modes:
    1. use_dominators=True (default): Uses rx.immediate_dominators to get
       ONLY the control-flow gates - nodes that MUST be understood.
       This is O(V+E) and produces minimal context for token efficiency.

    2. use_dominators=False: Falls back to naive BFS traversal of
       DEPENDS_ON edges. Can explode on deep graphs.

    Args:
        db: ParagonDB instance
        node_id: The target node
        max_depth: Maximum BFS depth (only used if use_dominators=False)
        use_dominators: Use Dominator Tree pruning (recommended)

    Returns:
        List of dependency context dicts
    """
    # Try dominator-based pruning first (Economic Topology)
    if use_dominators:
        try:
            # get_dominators returns NodeData objects
            dominators = db.get_dominators(node_id)
            chain = []
            for node_data in dominators:
                chain.append({
                    "id": node_data.id,
                    "type": node_data.type,
                    "status": node_data.status,
                    "content": node_data.content,
                    "metadata": node_data.data,
                })
            return chain
        except Exception:
            # Fall back to BFS if dominators fail
            pass

    # Fallback: naive BFS traversal (original logic)
    chain = []
    visited = set()
    to_visit = [node_id]
    depth = 0

    while to_visit and depth < max_depth:
        current_id = to_visit.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)

        incoming = db.get_incoming_edges(current_id)
        for edge in incoming:
            if edge.get("type") == "DEPENDS_ON":
                source_id = edge.get("source")
                if source_id and source_id not in visited:
                    to_visit.append(source_id)
                    chain.append(extract_node_context(db, source_id))

        depth += 1

    return chain


# =============================================================================
# ARCHITECT PROMPT BUILDER
# =============================================================================

def build_architect_prompt(
    db: Any,
    req_node_id: str,
) -> Tuple[str, str]:
    """
    Build prompts for the Architect agent.

    Args:
        db: ParagonDB instance
        req_node_id: The REQ node to decompose

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = get_agent_system_prompt("architect")

    # Extract REQ content
    req_context = extract_node_context(db, req_node_id)

    # Get any research artifacts
    research_context = []
    incoming = db.get_incoming_edges(req_node_id)
    for edge in incoming:
        if edge.get("type") == "RESEARCH_FOR":
            research_node = extract_node_context(db, edge.get("source"))
            if research_node.get("type") == "RESEARCH":
                research_context.append(research_node)

    # Build user prompt
    user_prompt = f"""# Requirement to Decompose

**ID:** {req_node_id}
**Content:**
{req_context.get('content', 'No content provided')}

"""

    if research_context:
        user_prompt += "# Research Findings\n\n"
        for research in research_context:
            user_prompt += f"**Research ID:** {research.get('id')}\n"
            user_prompt += f"{research.get('content', '')}\n\n"

    user_prompt += """
# Your Task

Decompose this requirement into atomic, implementable specifications.
Output a structured plan with components and their dependencies.
"""

    return system_prompt, user_prompt


# =============================================================================
# BUILDER PROMPT BUILDER
# =============================================================================

def build_builder_prompt(
    db: Any,
    spec_node_id: str,
    existing_code_context: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """
    Build prompts for the Builder agent.

    Args:
        db: ParagonDB instance
        spec_node_id: The SPEC node to implement
        existing_code_context: Optional list of file paths to include as context

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = get_agent_system_prompt("builder")

    # Extract SPEC content
    spec_context = extract_node_context(db, spec_node_id)

    # Get the parent REQ for broader context
    req_context = None
    incoming = db.get_incoming_edges(spec_node_id)
    for edge in incoming:
        if edge.get("type") == "TRACES_TO":
            source_context = extract_node_context(db, edge.get("source"))
            if source_context.get("type") == "PLAN":
                # Trace through PLAN to REQ
                plan_incoming = db.get_incoming_edges(edge.get("source"))
                for plan_edge in plan_incoming:
                    if plan_edge.get("type") == "TRACES_TO":
                        req_context = extract_node_context(db, plan_edge.get("source"))
                        break

    # Get dependency context
    deps = extract_dependency_chain(db, spec_node_id)
    verified_deps = [d for d in deps if d.get("status") == "VERIFIED"]

    # Build user prompt
    user_prompt = f"""# Specification to Implement

**SPEC ID:** {spec_node_id}
**Content:**
{spec_context.get('content', 'No content provided')}

"""

    if req_context:
        user_prompt += f"""# Original Requirement Context

{req_context.get('content', '')}

"""

    if verified_deps:
        user_prompt += "# Verified Dependencies (You can use these)\n\n"
        for dep in verified_deps:
            user_prompt += f"**{dep.get('id')}** ({dep.get('type')}):\n"
            user_prompt += f"{dep.get('content', '')[:500]}...\n\n"

    user_prompt += """
# Your Task

Implement complete, working code that satisfies this specification.
Include all necessary imports and follow existing codebase patterns.
"""

    return system_prompt, user_prompt


# =============================================================================
# TESTER PROMPT BUILDER
# =============================================================================

def build_tester_prompt(
    db: Any,
    code_node_id: str,
) -> Tuple[str, str]:
    """
    Build prompts for the Tester agent.

    Args:
        db: ParagonDB instance
        code_node_id: The CODE node to test

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = get_agent_system_prompt("tester")

    # Extract CODE content
    code_context = extract_node_context(db, code_node_id)

    # Get the SPEC this implements
    spec_context = None
    incoming = db.get_incoming_edges(code_node_id)
    for edge in incoming:
        if edge.get("type") == "IMPLEMENTS":
            spec_context = extract_node_context(db, edge.get("source"))
            break

    # Build user prompt
    user_prompt = f"""# Code to Test

**CODE ID:** {code_node_id}
**Filename:** {code_context.get('metadata', {}).get('filename', 'unknown')}

```python
{code_context.get('content', 'No code provided')}
```

"""

    if spec_context:
        user_prompt += f"""# Specification (What it should do)

{spec_context.get('content', 'No specification found')}

"""

    user_prompt += """
# Your Task

Generate comprehensive tests following the four testing layers:
1. Unit Tests - Known inputs/outputs
2. Property-Based Tests - Random valid inputs, verify invariants
3. Contract Tests - Inputs/outputs match specification
4. Static Analysis - Security and forbidden patterns

Provide a verdict: PASS, NEEDS_REVISION, or FAIL.
"""

    return system_prompt, user_prompt


# =============================================================================
# VERIFIER PROMPT BUILDER
# =============================================================================

def build_verifier_prompt(
    db: Any,
    code_node_id: str,
) -> Tuple[str, str]:
    """
    Build prompts for the Verifier agent.

    Args:
        db: ParagonDB instance
        code_node_id: The CODE node to verify

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = get_agent_system_prompt("verifier")

    # Extract CODE content
    code_context = extract_node_context(db, code_node_id)

    # Get the SPEC
    spec_context = None
    incoming = db.get_incoming_edges(code_node_id)
    for edge in incoming:
        if edge.get("type") == "IMPLEMENTS":
            spec_context = extract_node_context(db, edge.get("source"))
            break

    # Get test results
    test_results = []
    for edge in incoming:
        if edge.get("type") == "TESTS":
            test_context = extract_node_context(db, edge.get("source"))
            test_results.append(test_context)

    # Build user prompt
    user_prompt = f"""# Code Under Review

**CODE ID:** {code_node_id}

```python
{code_context.get('content', 'No code provided')}
```

"""

    if spec_context:
        user_prompt += f"""# Specification Requirements

{spec_context.get('content', 'No specification found')}

"""

    if test_results:
        user_prompt += "# Test Results\n\n"
        for test in test_results:
            status = test.get('status', 'UNKNOWN')
            user_prompt += f"**{test.get('id')}**: {status}\n"
            user_prompt += f"{test.get('content', '')[:300]}...\n\n"

    user_prompt += """
# Your Task

Verify that the code correctly implements the specification.
Provide a verdict: PASS or FAIL with clear reasoning.
"""

    return system_prompt, user_prompt


# =============================================================================
# RESEARCHER PROMPT BUILDER
# =============================================================================

def build_researcher_prompt(
    db: Any,
    req_node_id: str,
) -> Tuple[str, str]:
    """
    Build prompts for the Researcher agent.

    Args:
        db: ParagonDB instance
        req_node_id: The REQ node to research

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = get_agent_system_prompt("researcher")

    # Extract REQ content
    req_context = extract_node_context(db, req_node_id)

    # Check if dialectic analysis exists
    dialectic_context = None
    incoming = db.get_incoming_edges(req_node_id)
    for edge in incoming:
        source_context = extract_node_context(db, edge.get("source"))
        if source_context.get("type") == "CLARIFICATION":
            dialectic_context = source_context
            break

    # Build user prompt
    user_prompt = f"""# Requirement to Research

**REQ ID:** {req_node_id}
**Content:**
{req_context.get('content', 'No content provided')}

"""

    if dialectic_context:
        user_prompt += f"""# Dialectic Analysis

The following ambiguities were identified and resolved:
{dialectic_context.get('content', '')}

"""

    user_prompt += """
# Your Task

Transform this requirement into a structured Research Artifact following Research Standard v1.0:

1. Determine task category (greenfield, brownfield, algorithmic, systems, debug)
2. Define input/output contracts with Python type annotations
3. Provide happy path, edge case, and error case examples
4. Specify complexity bounds
5. Note security considerations
"""

    return system_prompt, user_prompt


# =============================================================================
# DIALECTOR PROMPT BUILDER
# =============================================================================

def build_dialector_prompt(
    db: Any,
    req_node_id: str,
) -> Tuple[str, str]:
    """
    Build prompts for the Dialector agent.

    Args:
        db: ParagonDB instance
        req_node_id: The REQ node to analyze for ambiguity

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = get_agent_system_prompt("dialector")

    # Extract REQ content
    req_context = extract_node_context(db, req_node_id)

    # Build user prompt
    user_prompt = f"""# Requirement to Analyze

**REQ ID:** {req_node_id}
**Content:**
{req_context.get('content', 'No content provided')}

# Your Task

Analyze this requirement for ambiguity markers:

1. **SUBJECTIVE TERMS**: "fast", "efficient", "user-friendly"
2. **COMPARATIVE STATEMENTS**: "faster than", "better than"
3. **UNDEFINED PRONOUNS**: "it", "this" without clear referent
4. **UNDEFINED TERMS**: Domain-specific terms needing definition
5. **MISSING CONTEXT**: Input/output format not specified

For each ambiguity found, classify impact as BLOCKING or CLARIFYING.
Provide a recommendation: PROCEED, CLARIFY, or BLOCK.
"""

    return system_prompt, user_prompt


# =============================================================================
# SOCRATES PROMPT BUILDER
# =============================================================================

def build_socrates_prompt(
    db: Any,
    clarification_node_id: str,
) -> Tuple[str, str]:
    """
    Build prompts for the Socrates agent.

    Args:
        db: ParagonDB instance
        clarification_node_id: The CLARIFICATION node to resolve

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = get_agent_system_prompt("socrates")

    # Extract CLARIFICATION content
    clarification_context = extract_node_context(db, clarification_node_id)

    # Get the original REQ
    req_context = None
    outgoing = db.get_outgoing_edges(clarification_node_id)
    for edge in outgoing:
        if edge.get("type") == "BLOCKS":
            req_context = extract_node_context(db, edge.get("target"))
            break

    # Build user prompt
    user_prompt = f"""# Clarification Needed

**CLARIFICATION ID:** {clarification_node_id}
**Ambiguity:**
{clarification_context.get('content', 'No content provided')}

"""

    if req_context:
        user_prompt += f"""# Original Requirement

{req_context.get('content', '')}

"""

    user_prompt += """
# Your Task

1. Research this ambiguity (standards, best practices, common patterns)
2. Formulate precise questions to resolve it
3. If possible, resolve through research alone
4. Provide questions and any default assumptions that could be used
"""

    return system_prompt, user_prompt


# =============================================================================
# UNIFIED PROMPT BUILDER
# =============================================================================

PROMPT_BUILDERS = {
    "architect": build_architect_prompt,
    "builder": build_builder_prompt,
    "tester": build_tester_prompt,
    "verifier": build_verifier_prompt,
    "researcher": build_researcher_prompt,
    "dialector": build_dialector_prompt,
    "socrates": build_socrates_prompt,
}


def build_prompt(
    db: Any,
    agent_role: str,
    node_id: str,
    **kwargs,
) -> Tuple[str, str]:
    """
    Unified prompt builder dispatcher.

    Args:
        db: ParagonDB instance
        agent_role: The agent role (architect, builder, etc.)
        node_id: The target node ID
        **kwargs: Additional arguments for specific builders

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    role_key = agent_role.lower()
    if role_key not in PROMPT_BUILDERS:
        raise ValueError(f"Unknown agent role: {agent_role}")

    builder = PROMPT_BUILDERS[role_key]
    return builder(db, node_id, **kwargs)
