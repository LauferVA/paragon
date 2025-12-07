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


def get_relevant_specs(db: Any, code_node_id: str) -> List[Dict[str, Any]]:
    """
    Get the SPEC nodes that a CODE node implements.

    Follows IMPLEMENTS edges to find specifications.
    Used by Builder to understand what it's implementing.

    Args:
        db: ParagonDB instance
        code_node_id: CODE node ID

    Returns:
        List of SPEC node context dicts
    """
    specs = []
    outgoing = db.get_outgoing_edges(code_node_id)

    for edge in outgoing:
        if edge.get("type") == "IMPLEMENTS":
            spec_context = extract_node_context(db, edge.get("target"))
            if spec_context.get("type") == "SPEC":
                specs.append(spec_context)

    return specs


def get_relevant_tests(db: Any, code_node_id: str) -> List[Dict[str, Any]]:
    """
    Get the TEST nodes that test a CODE node.

    Follows TESTS edges to find test suites.
    Used by Tester to find existing tests.

    Args:
        db: ParagonDB instance
        code_node_id: CODE node ID

    Returns:
        List of TEST node context dicts
    """
    tests = []
    incoming = db.get_incoming_edges(code_node_id)

    for edge in incoming:
        if edge.get("type") == "TESTS":
            test_context = extract_node_context(db, edge.get("source"))
            if test_context.get("type") in ("TEST", "TEST_SUITE"):
                tests.append(test_context)

    return tests


def get_requirement_chain(db: Any, node_id: str) -> List[Dict[str, Any]]:
    """
    Trace a node back to its originating REQ nodes.

    Useful for teleological validation - every artifact should
    trace back to a requirement ("golden thread").

    Args:
        db: ParagonDB instance
        node_id: Starting node

    Returns:
        List of REQ node context dicts in the ancestry chain
    """
    reqs = []
    visited = set()
    to_visit = [node_id]

    while to_visit:
        current_id = to_visit.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)

        current_context = extract_node_context(db, current_id)

        # Check if this is a REQ
        if current_context.get("type") == "REQ":
            reqs.append(current_context)
            continue  # Don't traverse past REQ nodes

        # Get predecessors via outgoing edges (TRACES_TO, IMPLEMENTS, etc.)
        outgoing = db.get_outgoing_edges(current_id)
        for edge in outgoing:
            target_id = edge.get("target")
            if target_id and target_id not in visited:
                to_visit.append(target_id)

    return reqs


# =============================================================================
# HYBRID CONTEXT ASSEMBLY (Compiler + Fuzzy)
# =============================================================================

def assemble_hybrid_context(
    db: Any,
    node_id: str,
    context_type: str = "code_generation",
    fuzzy_threshold: float = 0.6,
    fuzzy_limit: int = 3,
    include_fuzzy: bool = True,
) -> Dict[str, Any]:
    """
    Assemble context using two-layer hybrid approach:

    Layer 1 (Compiler): Graph edges provide deterministic relationships
    - Dominators: Control-flow gates that MUST be understood
    - Predecessors: Direct dependencies
    - Requirements: Golden thread traceability

    Layer 2 (Fuzzy): Semantic similarity for enrichment
    - Similar implementations: Patterns to follow
    - Similar tests: Testing strategies to adopt
    - Similar specs: Related specifications

    Architecture:
    - Compiler layer is PRIMARY (100% precision, O(V+E) via rustworkx)
    - Fuzzy layer is SECONDARY (enrichment, threshold >= 0.6)
    - Fuzzy results never contradict compiler results

    Args:
        db: ParagonDB instance
        node_id: Target node to assemble context for
        context_type: One of:
            - "code_generation": Building code from specs
            - "test_generation": Creating tests for code
            - "research": Exploring requirements/concepts
            - "attribution": Root cause analysis
            - "quality_gate": Quality assessment
            - "clarification": Resolving ambiguities
        fuzzy_threshold: Minimum similarity score (0.6 recommended)
        fuzzy_limit: Max fuzzy results per category
        include_fuzzy: Whether to include fuzzy layer (True by default)

    Returns:
        Dict with:
        - compiler_context: Deterministic graph-derived context
        - fuzzy_context: Semantic similarity-derived context
        - merged_context: Unified context for prompts
        - metadata: Assembly statistics
    """
    result = {
        "compiler_context": {},
        "fuzzy_context": {},
        "merged_context": {},
        "metadata": {
            "node_id": node_id,
            "context_type": context_type,
            "fuzzy_enabled": include_fuzzy,
        },
    }

    # Verify node exists
    try:
        target_node = db.get_node(node_id)
    except Exception:
        result["metadata"]["error"] = f"Node {node_id} not found"
        return result

    # =================================================================
    # LAYER 1: COMPILER CONTEXT (Deterministic Graph Traversal)
    # =================================================================

    compiler_ctx = {}

    # Get dominators (economic topology - minimal essential context)
    dominators = extract_dependency_chain(db, node_id, use_dominators=True)
    compiler_ctx["dominators"] = dominators

    # Get direct predecessors
    predecessors = extract_predecessor_context(db, node_id)
    compiler_ctx["predecessors"] = predecessors

    # Context-specific compiler extraction
    if context_type == "code_generation":
        # Get specs this code should implement
        specs = get_relevant_specs(db, node_id)
        compiler_ctx["specs"] = specs
        # Get requirement chain for traceability
        reqs = get_requirement_chain(db, node_id)
        compiler_ctx["requirements"] = reqs

    elif context_type == "test_generation":
        # Get the code to test
        code_ctx = extract_node_context(db, node_id)
        compiler_ctx["code"] = code_ctx
        # Get specs for expected behavior
        specs = get_relevant_specs(db, node_id)
        compiler_ctx["specs"] = specs
        # Get existing tests for patterns
        existing_tests = get_relevant_tests(db, node_id)
        compiler_ctx["existing_tests"] = existing_tests

    elif context_type == "research":
        # Get any existing clarifications
        incoming = db.get_incoming_edges(node_id)
        clarifications = []
        for edge in incoming:
            if edge.get("type") in ("CLARIFIES", "RESEARCH_FOR"):
                clarifications.append(extract_node_context(db, edge.get("source")))
        compiler_ctx["clarifications"] = clarifications

    elif context_type == "attribution":
        # Get failure context (descendants that failed)
        descendants = db.get_descendants(node_id)
        failed = [extract_node_context(db, d.id) for d in descendants
                  if d.status == "FAILED"]
        compiler_ctx["failed_descendants"] = failed

    elif context_type == "quality_gate":
        # Get all artifacts to assess
        code_ctx = extract_node_context(db, node_id)
        compiler_ctx["code"] = code_ctx
        specs = get_relevant_specs(db, node_id)
        compiler_ctx["specs"] = specs
        tests = get_relevant_tests(db, node_id)
        compiler_ctx["tests"] = tests

    elif context_type == "clarification":
        # Get the requirement being clarified
        outgoing = db.get_outgoing_edges(node_id)
        for edge in outgoing:
            if edge.get("type") == "BLOCKS":
                compiler_ctx["blocked_req"] = extract_node_context(db, edge.get("target"))
                break

    result["compiler_context"] = compiler_ctx
    result["metadata"]["compiler_node_count"] = (
        len(dominators) + len(predecessors) +
        sum(len(v) for v in compiler_ctx.values() if isinstance(v, list))
    )

    # =================================================================
    # LAYER 2: FUZZY CONTEXT (Semantic Similarity)
    # =================================================================

    fuzzy_ctx = {}

    if include_fuzzy:
        # Get IDs already in compiler context to avoid duplicates
        compiler_ids = set()
        for nodes in compiler_ctx.values():
            if isinstance(nodes, list):
                for n in nodes:
                    if isinstance(n, dict) and "id" in n:
                        compiler_ids.add(n["id"])
            elif isinstance(nodes, dict) and "id" in nodes:
                compiler_ids.add(nodes["id"])
        compiler_ids.add(node_id)

        # Use target node's content as query
        query_text = target_node.content

        if query_text:
            # Find similar nodes based on context type
            if context_type == "code_generation":
                # Find similar CODE implementations
                similar_code = db.find_similar_nodes(
                    query=query_text,
                    threshold=fuzzy_threshold,
                    limit=fuzzy_limit,
                    node_type="CODE",
                    exclude_ids=compiler_ids,
                )
                fuzzy_ctx["similar_implementations"] = [
                    {"id": n.id, "content": n.content[:500], "score": score}
                    for n, score in similar_code
                ]

            elif context_type == "test_generation":
                # Find similar TEST nodes
                similar_tests = db.find_similar_nodes(
                    query=query_text,
                    threshold=fuzzy_threshold,
                    limit=fuzzy_limit,
                    node_type="TEST_SUITE",
                    exclude_ids=compiler_ids,
                )
                fuzzy_ctx["similar_tests"] = [
                    {"id": n.id, "content": n.content[:500], "score": score}
                    for n, score in similar_tests
                ]

            elif context_type == "research":
                # Find similar RESEARCH nodes
                similar_research = db.find_similar_nodes(
                    query=query_text,
                    threshold=fuzzy_threshold,
                    limit=fuzzy_limit,
                    node_type="RESEARCH",
                    exclude_ids=compiler_ids,
                )
                fuzzy_ctx["similar_research"] = [
                    {"id": n.id, "content": n.content[:500], "score": score}
                    for n, score in similar_research
                ]

            elif context_type == "attribution":
                # Find similar failure patterns
                similar_failures = db.find_similar_nodes(
                    query=query_text,
                    threshold=fuzzy_threshold,
                    limit=fuzzy_limit,
                    exclude_ids=compiler_ids,
                )
                # Filter to only failed nodes
                fuzzy_ctx["similar_failures"] = [
                    {"id": n.id, "content": n.content[:500], "score": score, "status": n.status}
                    for n, score in similar_failures
                    if n.status == "FAILED"
                ]

            elif context_type == "quality_gate":
                # Find similar verified code for comparison
                similar_verified = db.find_similar_nodes(
                    query=query_text,
                    threshold=fuzzy_threshold,
                    limit=fuzzy_limit,
                    node_type="CODE",
                    exclude_ids=compiler_ids,
                )
                fuzzy_ctx["similar_verified"] = [
                    {"id": n.id, "content": n.content[:500], "score": score}
                    for n, score in similar_verified
                    if n.status == "VERIFIED"
                ]

            elif context_type == "clarification":
                # Find similar clarifications that were resolved
                similar_clarifications = db.find_similar_nodes(
                    query=query_text,
                    threshold=fuzzy_threshold,
                    limit=fuzzy_limit,
                    node_type="CLARIFICATION",
                    exclude_ids=compiler_ids,
                )
                fuzzy_ctx["similar_clarifications"] = [
                    {"id": n.id, "content": n.content[:500], "score": score}
                    for n, score in similar_clarifications
                    if n.status == "VERIFIED"  # Resolved clarifications
                ]

    result["fuzzy_context"] = fuzzy_ctx
    result["metadata"]["fuzzy_node_count"] = sum(
        len(v) for v in fuzzy_ctx.values() if isinstance(v, list)
    )

    # =================================================================
    # MERGE: Combine Compiler + Fuzzy into Unified Context
    # =================================================================

    merged = {
        "target": extract_node_context(db, node_id),
        "dependencies": compiler_ctx.get("dominators", []),
        "direct_dependencies": compiler_ctx.get("predecessors", []),
    }

    # Add context-specific fields
    for key, value in compiler_ctx.items():
        if key not in ("dominators", "predecessors"):
            merged[key] = value

    # Add fuzzy enrichment (clearly labeled)
    if fuzzy_ctx:
        merged["similar_examples"] = fuzzy_ctx

    result["merged_context"] = merged

    return result


def format_hybrid_context_for_prompt(
    context: Dict[str, Any],
    max_chars: int = 8000,
) -> str:
    """
    Format hybrid context into a string suitable for LLM prompts.

    Prioritizes compiler context, adds fuzzy examples as "inspiration".

    Args:
        context: Result from assemble_hybrid_context()
        max_chars: Maximum characters for context string

    Returns:
        Formatted context string
    """
    parts = []
    chars_used = 0

    merged = context.get("merged_context", {})

    # 1. Target node
    target = merged.get("target", {})
    if target:
        section = f"## Target Node\n**ID:** {target.get('id')}\n**Type:** {target.get('type')}\n**Content:**\n{target.get('content', '')[:1000]}\n"
        parts.append(section)
        chars_used += len(section)

    # 2. Dependencies (compiler layer - always include)
    deps = merged.get("dependencies", [])
    if deps:
        section = "## Dependencies (Required Context)\n"
        for dep in deps[:5]:  # Limit to 5 dependencies
            section += f"- **{dep.get('id')}** ({dep.get('type')}): {dep.get('content', '')[:200]}...\n"
        parts.append(section)
        chars_used += len(section)

    # 3. Context-specific compiler context
    for key in ["specs", "code", "requirements", "tests"]:
        if key in merged and chars_used < max_chars * 0.7:
            items = merged[key]
            if isinstance(items, list):
                section = f"## {key.title()}\n"
                for item in items[:3]:
                    section += f"- **{item.get('id')}**: {item.get('content', '')[:300]}...\n"
                parts.append(section)
                chars_used += len(section)
            elif isinstance(items, dict):
                section = f"## {key.title()}\n{items.get('content', '')[:500]}\n"
                parts.append(section)
                chars_used += len(section)

    # 4. Fuzzy examples (enrichment - only if space remains)
    similar = merged.get("similar_examples", {})
    if similar and chars_used < max_chars * 0.85:
        section = "## Similar Examples (For Inspiration)\n"
        for category, items in similar.items():
            if items and chars_used < max_chars:
                section += f"### {category.replace('_', ' ').title()}\n"
                for item in items[:2]:
                    section += f"- [{item.get('score', 0):.2f}] {item.get('content', '')[:200]}...\n"
        parts.append(section)
        chars_used += len(section)

    return "\n".join(parts)


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
