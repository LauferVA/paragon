"""
PARAGON AGENT DISPATCH - Graph-Native Agent Coordination

Wave 4 Refactor: Replace hardcoded phase routing with topology-driven dispatch

Instead of:
    if phase == "BUILD": return BuilderAgent()

Use:
    agent_role = get_triggered_agent(db, node_id)  # Returns "BUILDER" based on graph shape

This module provides helpers for:
1. Deriving the current "phase" from graph topology
2. Matching structural triggers to determine which agent should run
3. Context assembly for agents based on graph neighbors

Design Philosophy:
- Phase is an EMERGENT property of graph state, not an explicit state machine
- Agent dispatch is determined by STRUCTURAL_TRIGGERS in ontology.py
- Context is assembled via graph traversal, not manual file listing
"""
from typing import Optional, List, Dict, Any, Tuple
from core.ontology import (
    NodeType, NodeStatus, EdgeType,
    STRUCTURAL_TRIGGERS, StructuralTrigger,
)
from core.schemas import NodeData


# =============================================================================
# PHASE DETECTION FROM GRAPH STATE
# =============================================================================

def infer_phase_from_node(db, node_id: str) -> str:
    """
    Infer the current "phase" based on node type and graph structure.

    Instead of tracking phase as explicit state, we derive it from topology:
    - REQ with no RESEARCH → needs dialectic/research
    - REQ with RESEARCH but no PLAN → needs planning
    - SPEC with no CODE → needs building
    - CODE with no TEST_SUITE → needs testing
    - CODE with FAILED TEST_SUITE → needs fixing

    Args:
        db: ParagonDB instance
        node_id: The node to analyze

    Returns:
        Inferred phase string (e.g., "dialectic", "research", "build", "test")
    """
    try:
        node = db.get_node(node_id)
    except Exception:
        return "unknown"

    node_type = node.type
    node_status = node.status

    # Get outgoing edges from this node
    outgoing_edges = db.get_outgoing_edges(node_id)
    edge_types = {e["type"] for e in outgoing_edges}

    # Get incoming edges to this node
    incoming_edges = db.get_incoming_edges(node_id)
    incoming_types = {e["type"] for e in incoming_edges}

    # Phase inference based on node type and structure
    if node_type == NodeType.REQ.value:
        # REQ → check if has research/plan
        if EdgeType.RESEARCH_FOR.value not in incoming_types:
            return "dialectic"  # Needs ambiguity check and research
        if EdgeType.TRACES_TO.value not in incoming_types:
            return "planning"  # Has research, needs plan
        return "verified"  # Has been processed

    elif node_type == NodeType.RESEARCH.value:
        if node_status == NodeStatus.PENDING.value:
            return "research"
        return "research_complete"

    elif node_type == NodeType.SPEC.value:
        # SPEC → check if has implementing code
        if EdgeType.IMPLEMENTS.value not in incoming_types:
            return "build"  # Needs implementation
        return "implemented"

    elif node_type == NodeType.CODE.value:
        # CODE → check if has tests
        if EdgeType.TESTS.value not in incoming_types:
            return "test"  # Needs testing
        # Check if tests passed
        test_edges = [e for e in incoming_edges if e["type"] == EdgeType.TESTS.value]
        for test_edge in test_edges:
            test_node = db.get_node(test_edge["source"])
            if test_node.status == NodeStatus.FAILED.value:
                return "fix"  # Tests failed, needs fixing
        return "tested"

    elif node_type == NodeType.TEST_SUITE.value:
        if node_status == NodeStatus.FAILED.value:
            return "fix"
        return "verified"

    elif node_type == NodeType.CLARIFICATION.value:
        if node_status == NodeStatus.PENDING.value:
            return "clarification"  # Waiting for human input
        return "clarified"

    return "unknown"


def get_session_phase(db, session_id: Optional[str] = None) -> str:
    """
    Determine the overall phase for a session based on graph state.

    Looks at all active (non-VERIFIED, non-FAILED) nodes to determine
    what kind of work is pending.

    Args:
        db: ParagonDB instance
        session_id: Optional session ID to filter nodes

    Returns:
        Aggregate phase string
    """
    # Find all pending/processing nodes
    pending_nodes = db.find_nodes(status=NodeStatus.PENDING.value)
    processing_nodes = db.find_nodes(status=NodeStatus.PROCESSING.value)

    active_nodes = pending_nodes + processing_nodes

    if not active_nodes:
        return "idle"

    # Count phases
    phase_counts: Dict[str, int] = {}
    for node in active_nodes:
        phase = infer_phase_from_node(db, node.id)
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    # Return most common phase
    if phase_counts:
        return max(phase_counts.keys(), key=lambda k: phase_counts[k])

    return "idle"


# =============================================================================
# STRUCTURAL TRIGGER MATCHING
# =============================================================================

def get_triggered_agent(db, node_id: str) -> Optional[str]:
    """
    Determine which agent should process a node based on structural triggers.

    This replaces hardcoded phase→agent mapping with graph-based dispatch:
    - Examines the node's type, status, and edge patterns
    - Matches against STRUCTURAL_TRIGGERS in ontology.py
    - Returns the agent role that should handle this node

    Args:
        db: ParagonDB instance
        node_id: The node to analyze

    Returns:
        Agent role string (e.g., "BUILDER", "TESTER"), or None if no trigger matches
    """
    try:
        node = db.get_node(node_id)
    except Exception:
        return None

    # Try each structural trigger
    for trigger_id, trigger in STRUCTURAL_TRIGGERS.items():
        if _matches_trigger(db, node_id, node, trigger):
            return trigger.agent_role

    return None


def _matches_trigger(
    db,
    node_id: str,
    node: NodeData,
    trigger: StructuralTrigger,
) -> bool:
    """
    Check if a node matches a structural trigger.

    Args:
        db: ParagonDB instance
        node_id: Node ID
        node: NodeData
        trigger: StructuralTrigger to match against

    Returns:
        True if all trigger conditions are met
    """
    # Check node type
    if node.type != trigger.target_node_type:
        return False

    # Check status patterns
    for sp in trigger.status_patterns:
        if sp.is_not:
            if node.status == sp.status:
                return False
        else:
            if node.status != sp.status:
                return False

    # Check required edges
    outgoing = db.get_outgoing_edges(node_id)
    incoming = db.get_incoming_edges(node_id)

    for edge_pattern in trigger.required_edges:
        if not _edge_pattern_exists(outgoing, incoming, edge_pattern):
            return False

    # Check forbidden edges
    for edge_pattern in trigger.forbidden_edges:
        if _edge_pattern_exists(outgoing, incoming, edge_pattern):
            return False

    # Check predecessor requirements
    if trigger.all_predecessors_status:
        predecessors = db.get_predecessors(node_id)
        for pred in predecessors:
            if pred.status != trigger.all_predecessors_status:
                return False

    return True


def _edge_pattern_exists(
    outgoing: List[Dict],
    incoming: List[Dict],
    pattern,
) -> bool:
    """Check if an edge matching the pattern exists."""
    edges = []
    if pattern.direction in ("outgoing", "any"):
        edges.extend(outgoing)
    if pattern.direction in ("incoming", "any"):
        edges.extend(incoming)

    for edge in edges:
        if edge["type"] == pattern.edge_type:
            if pattern.exists:
                return True

    return False


def get_all_triggered_nodes(db) -> List[Tuple[str, str]]:
    """
    Find all nodes that currently match a structural trigger.

    Returns:
        List of (node_id, agent_role) tuples for nodes ready for processing
    """
    result = []

    # Check pending and processing nodes
    for status in [NodeStatus.PENDING.value, NodeStatus.PROCESSING.value]:
        nodes = db.find_nodes(status=status)
        for node in nodes:
            agent_role = get_triggered_agent(db, node.id)
            if agent_role:
                result.append((node.id, agent_role))

    return result


# =============================================================================
# CONTEXT ASSEMBLY
# =============================================================================

def assemble_context(
    db,
    node_id: str,
    max_depth: int = 2,
    include_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Assemble context for an agent from graph neighbors.

    Instead of manually specifying which files to include, we traverse
    the graph to find relevant nodes.

    Args:
        db: ParagonDB instance
        node_id: The focus node
        max_depth: Maximum traversal depth
        include_types: Optional list of node types to include

    Returns:
        Dict with context information:
        - focus_node: The target node
        - ancestors: Nodes this depends on
        - descendants: Nodes that depend on this
        - related: Nodes connected by any edge
    """
    try:
        focus_node = db.get_node(node_id)
    except Exception:
        return {"error": f"Node {node_id} not found"}

    context = {
        "focus_node": _node_to_dict(focus_node),
        "ancestors": [],
        "descendants": [],
        "related": [],
    }

    # Get ancestors (what this depends on)
    try:
        ancestors = db.get_ancestors(node_id)
        for ancestor in ancestors[:10]:  # Limit for context window
            if include_types is None or ancestor.type in include_types:
                context["ancestors"].append(_node_to_dict(ancestor))
    except Exception:
        pass

    # Get descendants (what depends on this)
    try:
        descendants = db.get_descendants(node_id)
        for desc in descendants[:10]:
            if include_types is None or desc.type in include_types:
                context["descendants"].append(_node_to_dict(desc))
    except Exception:
        pass

    # Get directly connected nodes
    try:
        outgoing = db.get_outgoing_edges(node_id)
        incoming = db.get_incoming_edges(node_id)

        seen_ids = {node_id}
        for edge in outgoing + incoming:
            target_id = edge.get("target") or edge.get("source")
            if target_id and target_id not in seen_ids:
                seen_ids.add(target_id)
                try:
                    related_node = db.get_node(target_id)
                    if include_types is None or related_node.type in include_types:
                        context["related"].append({
                            "node": _node_to_dict(related_node),
                            "edge_type": edge["type"],
                        })
                except Exception:
                    pass
    except Exception:
        pass

    return context


def _node_to_dict(node: NodeData) -> Dict[str, Any]:
    """Convert NodeData to a dict for context."""
    return {
        "id": node.id,
        "type": node.type,
        "status": node.status,
        "content": node.content[:500] if len(node.content) > 500 else node.content,
        "created_by": node.created_by,
    }


def get_relevant_specs(db, code_node_id: str) -> List[NodeData]:
    """
    Get the SPEC nodes that a CODE node implements.

    Follows IMPLEMENTS edges to find specifications.

    Args:
        db: ParagonDB instance
        code_node_id: CODE node ID

    Returns:
        List of SPEC nodes
    """
    outgoing = db.get_outgoing_edges(code_node_id)
    specs = []

    for edge in outgoing:
        if edge["type"] == EdgeType.IMPLEMENTS.value:
            try:
                spec_node = db.get_node(edge["target"])
                if spec_node.type == NodeType.SPEC.value:
                    specs.append(spec_node)
            except Exception:
                pass

    return specs


def get_relevant_tests(db, code_node_id: str) -> List[NodeData]:
    """
    Get the TEST_SUITE nodes that test a CODE node.

    Follows TESTS edges to find test suites.

    Args:
        db: ParagonDB instance
        code_node_id: CODE node ID

    Returns:
        List of TEST_SUITE nodes
    """
    incoming = db.get_incoming_edges(code_node_id)
    tests = []

    for edge in incoming:
        if edge["type"] == EdgeType.TESTS.value:
            try:
                test_node = db.get_node(edge["source"])
                if test_node.type == NodeType.TEST_SUITE.value:
                    tests.append(test_node)
            except Exception:
                pass

    return tests
