"""
PARAGON ONTOLOGY - The Dictionary of the System (Graph-Native Edition)

If schemas.py is the Grammar (how we structure sentences),
ontology.py is the Dictionary (the words we can use).

This module defines:
- Enums: The vocabulary (NodeType, EdgeType, NodeStatus)
- TopologyConstraint: Graph shape validation (replaces TRANSITION_MATRIX)
- StructuralTrigger: Pattern-based agent dispatch (replaces magic strings)

Key Principle: Graph topology IS the state machine.
We don't encode state transitions in tables - we define valid SHAPES
that the graph must maintain. The structure itself enforces invariants.

DESIGN PHILOSOPHY (Physics vs Policy):
- PHYSICS: A node cannot have an incoming edge from a non-existent node
- POLICY: A node "should" have approval before processing

This module encodes PHYSICS. Policy is derived from graph queries.
"""
from typing import Dict, List, Tuple, Optional, Set, Literal
from enum import Enum
import warnings

import msgspec


# =============================================================================
# ENUMS (The Vocabulary)
# =============================================================================

class NodeType(str, Enum):
    """Types of nodes in the graph."""
    # Primary workflow types
    REQ = "REQ"                      # User requirement (entry point)
    RESEARCH = "RESEARCH"            # Research artifact (Research Standard v1.0)
    CLARIFICATION = "CLARIFICATION"  # Ambiguity needing human input
    SPEC = "SPEC"                    # Atomic specification
    PLAN = "PLAN"                    # Decomposition strategy
    CODE = "CODE"                    # Implementation artifact
    TEST = "TEST"                    # Verification result (legacy)
    TEST_SUITE = "TEST_SUITE"        # Gen-2 TDD: Comprehensive test results from TESTER
    DOC = "DOC"                      # Documentation artifact
    ESCALATION = "ESCALATION"        # Failure requiring intervention
    # CPG (Code Property Graph) types for static analysis
    CLASS = "CLASS"                  # Class definition
    FUNCTION = "FUNCTION"            # Function/method definition
    CALL = "CALL"                    # Function call site
    # Graph-Native Configuration Types (Wave 1 Refactor)
    CONFIG = "CONFIG"                # System configuration node
    AGENT_CONFIG = "AGENT_CONFIG"    # Agent-specific configuration
    RESOURCE_POLICY = "RESOURCE_POLICY"  # Resource constraint policies
    SESSION = "SESSION"              # Session context holder
    RATE_LIMIT_EVENT = "RATE_LIMIT_EVENT"  # Rate limit tracking event


class EdgeType(str, Enum):
    """Types of edges between nodes."""
    TRACES_TO = "TRACES_TO"          # Provenance: any -> REQ
    DEPENDS_ON = "DEPENDS_ON"        # Ordering: SPEC -> SPEC (must complete first)
    IMPLEMENTS = "IMPLEMENTS"        # Realization: CODE -> SPEC
    VERIFIES = "VERIFIES"            # Validation: TEST -> CODE
    TESTS = "TESTS"                  # TDD: TEST_SUITE -> CODE (Gen-2)
    DEFINES = "DEFINES"              # Decomposition: PLAN -> SPEC
    BLOCKS = "BLOCKS"                # Blocking: any -> CLARIFICATION/ESCALATION
    FEEDBACK = "FEEDBACK"            # Critique: failed CODE -> SPEC (or Tester -> Builder)
    RESOLVED_BY = "RESOLVED_BY"      # Answer: CLARIFICATION -> response
    RESEARCH_FOR = "RESEARCH_FOR"    # Research artifact for: RESEARCH -> REQ
    # Extended relationships (for AST/CPG features)
    CONTAINS = "CONTAINS"            # Containment: parent -> child
    REFERENCES = "REFERENCES"        # Reference: caller -> callee
    INHERITS = "INHERITS"            # Inheritance: subclass -> superclass
    # Graph-Native Configuration Edges (Wave 1 Refactor)
    CONFIGURED_BY = "CONFIGURED_BY"  # Configuration: node -> CONFIG/AGENT_CONFIG
    ENFORCES = "ENFORCES"            # Policy: RESOURCE_POLICY -> constraint target
    APPLIES_TO = "APPLIES_TO"        # Scope: CONFIG -> target nodes/types
    RATE_LIMITED_BY = "RATE_LIMITED_BY"  # Rate limit: request -> RATE_LIMIT_EVENT
    SESSION_CONTAINS = "SESSION_CONTAINS"  # Session membership: SESSION -> nodes


class NodeStatus(str, Enum):
    """
    Status states for nodes.

    NOTE: BLOCKED is retained as an EXPLICIT override status (human-initiated block).
    The computed property `is_blocked_by_dependencies` on NodeData handles
    structural blocking (when predecessors aren't complete).
    """
    PENDING = "PENDING"              # Waiting to be processed
    PROCESSING = "PROCESSING"        # Currently being processed
    BLOCKED = "BLOCKED"              # Explicitly blocked (human override)
    TESTING = "TESTING"              # Being tested by TESTER (Gen-2 TDD)
    TESTED = "TESTED"                # Tests passed, awaiting verification
    VERIFIED = "VERIFIED"            # Successfully completed
    FAILED = "FAILED"                # Terminal failure


class TransitionTrigger(str, Enum):
    """What triggered a state transition (for telemetry)."""
    AGENT_COMPLETION = "agent_completion"
    USER_ACTION = "user_action"
    TIMEOUT = "timeout"
    ERROR = "error"
    DEPENDENCY_RESOLVED = "dependency_resolved"
    SYSTEM_OVERRIDE = "system_override"


# =============================================================================
# Type Aliases
# =============================================================================

ApprovalStatus = Literal["none", "pending", "approved", "rejected"]
EdgeDirection = Literal["incoming", "outgoing", "any"]


# =============================================================================
# TOPOLOGY CONSTRAINTS (The Physics of Graph Shapes)
# =============================================================================

class ConstraintMode(str, Enum):
    """
    When to enforce a constraint.

    HARD: Enforced on every write operation. Violation = rejection.
    SOFT: Checked during validation queries. Violation = warning/incomplete.
    """
    HARD = "hard"
    SOFT = "soft"


class EdgeConstraint(msgspec.Struct, kw_only=True, frozen=True):
    """
    Defines a required or forbidden edge pattern for a node type.

    Examples:
    - CODE node MUST have an outgoing IMPLEMENTS edge (to SPEC)
    - CLARIFICATION node MUST have an incoming BLOCKS edge
    - TEST_SUITE MUST target a CODE node via TESTS edge
    """
    edge_type: str                           # EdgeType.value (e.g., "IMPLEMENTS")
    direction: EdgeDirection = "incoming"    # incoming, outgoing, or any
    target_node_type: Optional[str] = None   # Required type of connected node (None = any)
    min_count: int = 0                       # Minimum edges of this type (0 = optional)
    max_count: Optional[int] = None          # Maximum edges (None = unlimited)
    mode: str = ConstraintMode.SOFT.value    # hard or soft enforcement


class TopologyConstraint(msgspec.Struct, kw_only=True, frozen=True):
    """
    Defines valid "shapes" for a node type in the graph.

    Instead of encoding transitions in a matrix, we define the STRUCTURE
    that must exist. The graph topology itself acts as the state machine.

    Example: A complete SPEC node should have:
    - An incoming DEFINES edge from a PLAN
    - An outgoing IMPLEMENTS edge to CODE (when implemented)
    - Its predecessors should all be VERIFIED (for it to be ready)
    """
    node_type: str                           # NodeType this constraint applies to
    description: str = ""                    # Human-readable explanation
    edge_constraints: Tuple[EdgeConstraint, ...] = ()  # Required edge patterns
    allowed_statuses: Tuple[str, ...] = ()   # Valid statuses for this node type


# =============================================================================
# TOPOLOGY CONSTRAINT REGISTRY
# =============================================================================

# These define the valid "shapes" in the graph.
# The graph engine validates against these on write operations.

TOPOLOGY_CONSTRAINTS: Dict[str, TopologyConstraint] = {

    # =========================================================================
    # REQ (Requirement) - Entry Point
    # =========================================================================
    NodeType.REQ.value: TopologyConstraint(
        node_type=NodeType.REQ.value,
        description="Requirements are entry points. They spawn RESEARCH, PLAN, or CLARIFICATION nodes.",
        edge_constraints=(
            EdgeConstraint(
                edge_type=EdgeType.TRACES_TO.value,
                direction="incoming",
                mode=ConstraintMode.SOFT.value,
                min_count=0,  # REQs can be roots
            ),
        ),
        allowed_statuses=(
            NodeStatus.PENDING.value,
            NodeStatus.PROCESSING.value,
            NodeStatus.BLOCKED.value,
            NodeStatus.VERIFIED.value,
            NodeStatus.FAILED.value,
        ),
    ),

    # =========================================================================
    # RESEARCH - Research Artifact
    # =========================================================================
    NodeType.RESEARCH.value: TopologyConstraint(
        node_type=NodeType.RESEARCH.value,
        description="Research artifacts trace back to a REQ.",
        edge_constraints=(
            EdgeConstraint(
                edge_type=EdgeType.RESEARCH_FOR.value,
                direction="outgoing",
                target_node_type=NodeType.REQ.value,
                min_count=1,  # Must be for a REQ
                mode=ConstraintMode.SOFT.value,
            ),
        ),
        allowed_statuses=(
            NodeStatus.PENDING.value,
            NodeStatus.PROCESSING.value,
            NodeStatus.VERIFIED.value,
            NodeStatus.FAILED.value,
        ),
    ),

    # =========================================================================
    # SPEC - Atomic Specification
    # =========================================================================
    NodeType.SPEC.value: TopologyConstraint(
        node_type=NodeType.SPEC.value,
        description="Specs are defined by PLANs and implemented by CODE.",
        edge_constraints=(
            EdgeConstraint(
                edge_type=EdgeType.DEFINES.value,
                direction="incoming",
                target_node_type=NodeType.PLAN.value,
                min_count=0,  # Soft: should have a defining PLAN
                mode=ConstraintMode.SOFT.value,
            ),
            EdgeConstraint(
                edge_type=EdgeType.DEPENDS_ON.value,
                direction="outgoing",
                target_node_type=NodeType.SPEC.value,
                min_count=0,  # Optional dependencies
                mode=ConstraintMode.SOFT.value,
            ),
        ),
        allowed_statuses=(
            NodeStatus.PENDING.value,
            NodeStatus.PROCESSING.value,
            NodeStatus.BLOCKED.value,
            NodeStatus.VERIFIED.value,
            NodeStatus.FAILED.value,
        ),
    ),

    # =========================================================================
    # CODE - Implementation Artifact
    # =========================================================================
    NodeType.CODE.value: TopologyConstraint(
        node_type=NodeType.CODE.value,
        description="Code implements a SPEC and is verified by TEST/TEST_SUITE.",
        edge_constraints=(
            EdgeConstraint(
                edge_type=EdgeType.IMPLEMENTS.value,
                direction="outgoing",
                target_node_type=NodeType.SPEC.value,
                min_count=1,  # Code MUST implement a spec
                mode=ConstraintMode.SOFT.value,  # Soft: allow creation before linking
            ),
            EdgeConstraint(
                edge_type=EdgeType.VERIFIES.value,
                direction="incoming",
                min_count=0,  # Will have verification when tested
                mode=ConstraintMode.SOFT.value,
            ),
            EdgeConstraint(
                edge_type=EdgeType.TESTS.value,
                direction="incoming",
                target_node_type=NodeType.TEST_SUITE.value,
                min_count=0,  # Will have tests when TDD completes
                mode=ConstraintMode.SOFT.value,
            ),
        ),
        allowed_statuses=(
            NodeStatus.PENDING.value,
            NodeStatus.PROCESSING.value,
            NodeStatus.TESTING.value,
            NodeStatus.TESTED.value,
            NodeStatus.VERIFIED.value,
            NodeStatus.FAILED.value,
        ),
    ),

    # =========================================================================
    # TEST_SUITE - TDD Test Results
    # =========================================================================
    NodeType.TEST_SUITE.value: TopologyConstraint(
        node_type=NodeType.TEST_SUITE.value,
        description="Test suites test CODE nodes.",
        edge_constraints=(
            EdgeConstraint(
                edge_type=EdgeType.TESTS.value,
                direction="outgoing",
                target_node_type=NodeType.CODE.value,
                min_count=1,  # Must test some code
                mode=ConstraintMode.HARD.value,  # Hard: can't create orphan tests
            ),
        ),
        allowed_statuses=(
            NodeStatus.PENDING.value,
            NodeStatus.PROCESSING.value,
            NodeStatus.VERIFIED.value,
            NodeStatus.FAILED.value,
        ),
    ),

    # =========================================================================
    # CLARIFICATION - Ambiguity Node
    # =========================================================================
    NodeType.CLARIFICATION.value: TopologyConstraint(
        node_type=NodeType.CLARIFICATION.value,
        description="Clarifications block other nodes until resolved.",
        edge_constraints=(
            EdgeConstraint(
                edge_type=EdgeType.BLOCKS.value,
                direction="outgoing",
                min_count=0,  # May or may not block something
                mode=ConstraintMode.SOFT.value,
            ),
            EdgeConstraint(
                edge_type=EdgeType.RESOLVED_BY.value,
                direction="outgoing",
                min_count=0,  # Resolved when answered
                mode=ConstraintMode.SOFT.value,
            ),
        ),
        allowed_statuses=(
            NodeStatus.PENDING.value,
            NodeStatus.PROCESSING.value,
            NodeStatus.BLOCKED.value,
            NodeStatus.VERIFIED.value,
        ),
    ),

    # =========================================================================
    # PLAN - Decomposition Strategy
    # =========================================================================
    NodeType.PLAN.value: TopologyConstraint(
        node_type=NodeType.PLAN.value,
        description="Plans define SPECs through decomposition.",
        edge_constraints=(
            EdgeConstraint(
                edge_type=EdgeType.DEFINES.value,
                direction="outgoing",
                target_node_type=NodeType.SPEC.value,
                min_count=0,  # Will define specs when decomposition runs
                mode=ConstraintMode.SOFT.value,
            ),
        ),
        allowed_statuses=(
            NodeStatus.PENDING.value,
            NodeStatus.VERIFIED.value,
        ),
    ),

    # =========================================================================
    # ESCALATION - Failure Node
    # =========================================================================
    NodeType.ESCALATION.value: TopologyConstraint(
        node_type=NodeType.ESCALATION.value,
        description="Escalations block progress and require intervention.",
        edge_constraints=(
            EdgeConstraint(
                edge_type=EdgeType.BLOCKS.value,
                direction="outgoing",
                min_count=1,  # Must block something
                mode=ConstraintMode.SOFT.value,
            ),
        ),
        allowed_statuses=(
            NodeStatus.PENDING.value,
            NodeStatus.PROCESSING.value,
            NodeStatus.VERIFIED.value,
            NodeStatus.FAILED.value,
        ),
    ),

    # =========================================================================
    # GRAPH-NATIVE CONFIGURATION TYPES (Wave 1 Refactor)
    # =========================================================================

    # CONFIG - System Configuration Node
    NodeType.CONFIG.value: TopologyConstraint(
        node_type=NodeType.CONFIG.value,
        description="System configuration stored as graph node. Content is JSON of settings.",
        edge_constraints=(
            EdgeConstraint(
                edge_type=EdgeType.APPLIES_TO.value,
                direction="outgoing",
                min_count=0,  # Optional - can be global config
                mode=ConstraintMode.SOFT.value,
            ),
        ),
        allowed_statuses=(
            NodeStatus.VERIFIED.value,  # Config should be stable
        ),
    ),

    # AGENT_CONFIG - Agent-Specific Configuration
    NodeType.AGENT_CONFIG.value: TopologyConstraint(
        node_type=NodeType.AGENT_CONFIG.value,
        description="Agent-specific configuration. Overrides system CONFIG for specific agents.",
        edge_constraints=(
            EdgeConstraint(
                edge_type=EdgeType.APPLIES_TO.value,
                direction="outgoing",
                min_count=1,  # Must specify which agent(s) this applies to
                mode=ConstraintMode.SOFT.value,
            ),
        ),
        allowed_statuses=(
            NodeStatus.VERIFIED.value,
        ),
    ),

    # RESOURCE_POLICY - Resource Constraint Policies
    NodeType.RESOURCE_POLICY.value: TopologyConstraint(
        node_type=NodeType.RESOURCE_POLICY.value,
        description="Resource constraints (RAM, CPU thresholds). ENFORCES edge to constrained nodes.",
        edge_constraints=(
            EdgeConstraint(
                edge_type=EdgeType.ENFORCES.value,
                direction="outgoing",
                min_count=0,  # Can be global policy
                mode=ConstraintMode.SOFT.value,
            ),
        ),
        allowed_statuses=(
            NodeStatus.VERIFIED.value,
        ),
    ),

    # SESSION - Session Context Holder
    NodeType.SESSION.value: TopologyConstraint(
        node_type=NodeType.SESSION.value,
        description="Session context. Contains nodes created during this session.",
        edge_constraints=(
            EdgeConstraint(
                edge_type=EdgeType.SESSION_CONTAINS.value,
                direction="outgoing",
                min_count=0,  # Empty sessions allowed
                mode=ConstraintMode.SOFT.value,
            ),
        ),
        allowed_statuses=(
            NodeStatus.PROCESSING.value,  # Active session
            NodeStatus.VERIFIED.value,    # Completed session
            NodeStatus.FAILED.value,      # Failed session
        ),
    ),

    # RATE_LIMIT_EVENT - Rate Limit Tracking
    NodeType.RATE_LIMIT_EVENT.value: TopologyConstraint(
        node_type=NodeType.RATE_LIMIT_EVENT.value,
        description="Immutable record of rate limit events. Append-only for tracking.",
        edge_constraints=(),  # No required edges - standalone events
        allowed_statuses=(
            NodeStatus.VERIFIED.value,  # Events are immutable once created
        ),
    ),
}


# =============================================================================
# STRUCTURAL TRIGGERS (Pattern-Based Agent Dispatch)
# =============================================================================

class EdgePattern(msgspec.Struct, kw_only=True, frozen=True):
    """
    Pattern matching for edge presence/absence.

    Used by StructuralTrigger to define graph-based conditions.
    """
    edge_type: str                           # EdgeType.value
    direction: EdgeDirection = "incoming"    # incoming, outgoing, any
    exists: bool = True                      # True = must exist, False = must NOT exist
    target_node_type: Optional[str] = None   # Filter by connected node type
    target_node_status: Optional[str] = None # Filter by connected node status


class StatusPattern(msgspec.Struct, kw_only=True, frozen=True):
    """Pattern for matching node status."""
    status: str                              # NodeStatus.value
    is_not: bool = False                     # True = status must NOT be this value


class StructuralTrigger(msgspec.Struct, kw_only=True, frozen=True):
    """
    Defines when an agent should be triggered based on graph structure.

    Instead of magic strings like "needs_testing", we define structural patterns:
    - A CODE node that lacks an incoming VERIFIES edge needs verification
    - A SPEC node where all DEPENDS_ON targets are VERIFIED is ready for build

    This makes agent dispatch purely a function of graph topology.
    """
    trigger_id: str                          # Unique identifier
    description: str = ""                    # Human-readable explanation
    target_node_type: str                    # NodeType this trigger applies to
    agent_role: str                          # Agent to dispatch (e.g., "TESTER")

    # Structural patterns (all must match)
    required_edges: Tuple[EdgePattern, ...] = ()     # Edges that MUST exist
    forbidden_edges: Tuple[EdgePattern, ...] = ()    # Edges that must NOT exist
    status_patterns: Tuple[StatusPattern, ...] = ()  # Status requirements

    # Predecessor requirements (graph-native "dependencies_met" check)
    all_predecessors_status: Optional[str] = None    # All predecessors must have this status
    any_predecessor_status: Optional[str] = None     # At least one predecessor has this status


# =============================================================================
# STRUCTURAL TRIGGER REGISTRY
# =============================================================================

STRUCTURAL_TRIGGERS: Dict[str, StructuralTrigger] = {

    # =========================================================================
    # DIALECTIC PIPELINE
    # =========================================================================
    "req_needs_dialectic": StructuralTrigger(
        trigger_id="req_needs_dialectic",
        description="REQ node that hasn't been analyzed for ambiguity",
        target_node_type=NodeType.REQ.value,
        agent_role="DIALECTOR",
        status_patterns=(
            StatusPattern(status=NodeStatus.PENDING.value),
        ),
        forbidden_edges=(
            # No outgoing edge to RESEARCH or PLAN yet
            EdgePattern(
                edge_type=EdgeType.RESEARCH_FOR.value,
                direction="incoming",
                exists=True,
            ),
        ),
    ),

    # =========================================================================
    # RESEARCH PIPELINE
    # =========================================================================
    "req_needs_research": StructuralTrigger(
        trigger_id="req_needs_research",
        description="REQ that passed dialectic and needs research",
        target_node_type=NodeType.REQ.value,
        agent_role="RESEARCHER",
        status_patterns=(
            StatusPattern(status=NodeStatus.PROCESSING.value),
        ),
        forbidden_edges=(
            EdgePattern(
                edge_type=EdgeType.RESEARCH_FOR.value,
                direction="incoming",
                exists=True,
            ),
        ),
    ),

    "research_needs_verification": StructuralTrigger(
        trigger_id="research_needs_verification",
        description="RESEARCH artifact that needs verification",
        target_node_type=NodeType.RESEARCH.value,
        agent_role="RESEARCH_VERIFIER",
        status_patterns=(
            StatusPattern(status=NodeStatus.PENDING.value),
        ),
    ),

    # =========================================================================
    # SPEC → CODE PIPELINE
    # =========================================================================
    "req_needs_decomposition": StructuralTrigger(
        trigger_id="req_needs_decomposition",
        description="REQ with verified research that needs decomposition into SPECs",
        target_node_type=NodeType.REQ.value,
        agent_role="ARCHITECT",
        status_patterns=(
            StatusPattern(status=NodeStatus.PROCESSING.value),
        ),
        required_edges=(
            # Has research
            EdgePattern(
                edge_type=EdgeType.RESEARCH_FOR.value,
                direction="incoming",
                target_node_status=NodeStatus.VERIFIED.value,
            ),
        ),
        forbidden_edges=(
            # No PLAN yet
            EdgePattern(
                edge_type=EdgeType.TRACES_TO.value,
                direction="incoming",
                target_node_type=NodeType.PLAN.value,
            ),
        ),
    ),

    "spec_ready_for_build": StructuralTrigger(
        trigger_id="spec_ready_for_build",
        description="SPEC with all dependencies verified, ready for implementation",
        target_node_type=NodeType.SPEC.value,
        agent_role="BUILDER",
        status_patterns=(
            StatusPattern(status=NodeStatus.PENDING.value),
        ),
        forbidden_edges=(
            # No CODE implementing it yet
            EdgePattern(
                edge_type=EdgeType.IMPLEMENTS.value,
                direction="incoming",
                exists=True,
            ),
        ),
        all_predecessors_status=NodeStatus.VERIFIED.value,  # All deps verified
    ),

    # =========================================================================
    # TDD LOOP
    # =========================================================================
    "code_needs_testing": StructuralTrigger(
        trigger_id="code_needs_testing",
        description="CODE node without test coverage",
        target_node_type=NodeType.CODE.value,
        agent_role="TESTER",
        status_patterns=(
            StatusPattern(status=NodeStatus.PENDING.value),
        ),
        forbidden_edges=(
            # No TEST_SUITE testing it yet
            EdgePattern(
                edge_type=EdgeType.TESTS.value,
                direction="incoming",
                exists=True,
            ),
        ),
    ),

    "code_needs_verification": StructuralTrigger(
        trigger_id="code_needs_verification",
        description="CODE that passed tests and needs final verification",
        target_node_type=NodeType.CODE.value,
        agent_role="VERIFIER",
        status_patterns=(
            StatusPattern(status=NodeStatus.TESTED.value),
        ),
        required_edges=(
            # Has passing tests
            EdgePattern(
                edge_type=EdgeType.TESTS.value,
                direction="incoming",
                target_node_status=NodeStatus.VERIFIED.value,
            ),
        ),
        forbidden_edges=(
            # No VERIFIES edge yet
            EdgePattern(
                edge_type=EdgeType.VERIFIES.value,
                direction="incoming",
                exists=True,
            ),
        ),
    ),

    # =========================================================================
    # CLARIFICATION & ESCALATION
    # =========================================================================
    "clarification_needs_resolution": StructuralTrigger(
        trigger_id="clarification_needs_resolution",
        description="CLARIFICATION node awaiting resolution",
        target_node_type=NodeType.CLARIFICATION.value,
        agent_role="SOCRATES",
        status_patterns=(
            StatusPattern(status=NodeStatus.PENDING.value),
        ),
        forbidden_edges=(
            EdgePattern(
                edge_type=EdgeType.RESOLVED_BY.value,
                direction="outgoing",
                exists=True,
            ),
        ),
    ),

    "escalation_needs_replan": StructuralTrigger(
        trigger_id="escalation_needs_replan",
        description="ESCALATION node requiring architect intervention",
        target_node_type=NodeType.ESCALATION.value,
        agent_role="ARCHITECT",
        status_patterns=(
            StatusPattern(status=NodeStatus.PENDING.value),
        ),
    ),

    # =========================================================================
    # BLOCKING DETECTION (Graph-Native)
    # =========================================================================
    "req_has_blocking_clarification": StructuralTrigger(
        trigger_id="req_has_blocking_clarification",
        description="REQ blocked by unresolved clarification",
        target_node_type=NodeType.REQ.value,
        agent_role="SOCRATES",
        required_edges=(
            EdgePattern(
                edge_type=EdgeType.BLOCKS.value,
                direction="incoming",
                target_node_type=NodeType.CLARIFICATION.value,
                target_node_status=NodeStatus.PENDING.value,
            ),
        ),
    ),
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def get_constraint(node_type: str) -> Optional[TopologyConstraint]:
    """Get topology constraint for a node type."""
    return TOPOLOGY_CONSTRAINTS.get(node_type)


def get_triggers_for_node_type(node_type: str) -> List[StructuralTrigger]:
    """Get all structural triggers that apply to a node type."""
    return [
        trigger for trigger in STRUCTURAL_TRIGGERS.values()
        if trigger.target_node_type == node_type
    ]


def validate_status_for_type(node_type: str, status: str) -> bool:
    """Check if a status is valid for a given node type."""
    constraint = TOPOLOGY_CONSTRAINTS.get(node_type)
    if constraint is None:
        return True  # No constraint = all statuses allowed
    if not constraint.allowed_statuses:
        return True  # Empty = all allowed
    return status in constraint.allowed_statuses


def get_required_edges(node_type: str, mode: str = "all") -> List[EdgeConstraint]:
    """
    Get edge constraints for a node type.

    Args:
        node_type: The node type to query
        mode: "all", "hard", or "soft" - which constraint modes to include

    Returns:
        List of EdgeConstraint that apply
    """
    constraint = TOPOLOGY_CONSTRAINTS.get(node_type)
    if constraint is None:
        return []

    if mode == "all":
        return list(constraint.edge_constraints)
    elif mode == "hard":
        return [ec for ec in constraint.edge_constraints if ec.mode == ConstraintMode.HARD.value]
    elif mode == "soft":
        return [ec for ec in constraint.edge_constraints if ec.mode == ConstraintMode.SOFT.value]
    else:
        return []


# =============================================================================
# DEPRECATED: TRANSITION MATRIX (Removed)
# =============================================================================
# The TRANSITION_MATRIX has been removed in favor of TopologyConstraints.
# Status transitions are now implicit in graph structure:
# - A node is "blocked" if it has unverified predecessors (computed property)
# - A node is "ready" if structural triggers match (pattern query)
#
# If you need the old behavior, query the graph structure directly.
# =============================================================================


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def _validate_constraints() -> List[str]:
    """Validate that all constraints reference valid types."""
    errors = []

    valid_node_types = {nt.value for nt in NodeType}
    valid_edge_types = {et.value for et in EdgeType}
    valid_statuses = {ns.value for ns in NodeStatus}

    # Validate TopologyConstraints
    for node_type, constraint in TOPOLOGY_CONSTRAINTS.items():
        if node_type not in valid_node_types:
            errors.append(f"Invalid node type in constraint: {node_type}")

        for ec in constraint.edge_constraints:
            if ec.edge_type not in valid_edge_types:
                errors.append(f"Invalid edge type in constraint for {node_type}: {ec.edge_type}")
            if ec.target_node_type and ec.target_node_type not in valid_node_types:
                errors.append(f"Invalid target node type in constraint for {node_type}: {ec.target_node_type}")

        for status in constraint.allowed_statuses:
            if status not in valid_statuses:
                errors.append(f"Invalid status in constraint for {node_type}: {status}")

    # Validate StructuralTriggers
    for trigger_id, trigger in STRUCTURAL_TRIGGERS.items():
        if trigger.target_node_type not in valid_node_types:
            errors.append(f"Invalid node type in trigger {trigger_id}: {trigger.target_node_type}")

        for ep in trigger.required_edges + trigger.forbidden_edges:
            if ep.edge_type not in valid_edge_types:
                errors.append(f"Invalid edge type in trigger {trigger_id}: {ep.edge_type}")
            if ep.target_node_type and ep.target_node_type not in valid_node_types:
                errors.append(f"Invalid target node type in trigger {trigger_id}: {ep.target_node_type}")
            if ep.target_node_status and ep.target_node_status not in valid_statuses:
                errors.append(f"Invalid target status in trigger {trigger_id}: {ep.target_node_status}")

        for sp in trigger.status_patterns:
            if sp.status not in valid_statuses:
                errors.append(f"Invalid status in trigger {trigger_id}: {sp.status}")

        if trigger.all_predecessors_status and trigger.all_predecessors_status not in valid_statuses:
            errors.append(f"Invalid predecessor status in trigger {trigger_id}: {trigger.all_predecessors_status}")

    return errors


# Run validation on module load
_validation_errors = _validate_constraints()
if _validation_errors:
    for err in _validation_errors:
        warnings.warn(f"Ontology validation: {err}")
