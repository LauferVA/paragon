"""
PARAGON INTELLIGENCE - Agent Output Schemas

Defines the msgspec Structs that agents must produce.
These are the "contracts" between agents and the graph.

Design:
- All outputs use msgspec.Struct for performance (3-10x faster than Pydantic)
- Each agent type has a dedicated output schema
- Schemas are designed to be directly insertable into ParagonDB

Layer 7 Phase 1: The Creator output structures.
"""
import msgspec
from typing import List, Optional, Tuple, Literal
from enum import Enum


# =============================================================================
# SPEC PARSING
# =============================================================================

class ParsedSpec(msgspec.Struct, kw_only=True):
    """
    Structured output from spec parsing.

    Contains extracted information from a spec file.
    """
    title: str
    description: str
    requirements: List[str] = []
    technical_details: Optional[str] = None
    target_user: Optional[str] = None
    must_have_features: List[str] = []
    constraints: List[str] = []
    raw_content: str = ""
    file_format: str = ""


# =============================================================================
# COMPONENT SPECIFICATIONS
# =============================================================================

class ComponentSpec(msgspec.Struct, kw_only=True, frozen=True):
    """
    A single component in an implementation plan.

    Represents a class, function, or module that needs to be created.
    """
    name: str
    type: str  # "class", "function", "module"
    description: str
    dependencies: List[str] = []
    file_path: Optional[str] = None


class DependencyEdge(msgspec.Struct, kw_only=True, frozen=True):
    """
    Represents a dependency relationship between components.

    Used by the Architect to specify execution order.
    """
    source: str  # Component name that depends on target
    target: str  # Component name being depended upon
    edge_type: str = "DEPENDS_ON"


# =============================================================================
# ARCHITECT OUTPUT
# =============================================================================

class ImplementationPlan(msgspec.Struct, kw_only=True, frozen=True):
    """
    Output schema for the Architect agent.

    Decomposes a requirement into atomic, implementable components.
    """
    explanation: str
    components: List[ComponentSpec]
    dependencies: List[DependencyEdge] = []
    estimated_complexity: str = "medium"  # "low", "medium", "high"
    notes: Optional[str] = None


# =============================================================================
# BUILDER OUTPUT
# =============================================================================

class CodeGeneration(msgspec.Struct, kw_only=True, frozen=True):
    """
    Output schema for the Builder agent.

    Contains the actual code implementation for a SPEC node.
    """
    filename: str
    code: str
    imports: List[str] = []
    description: str = ""
    language: str = "python"
    target_spec_id: Optional[str] = None


# =============================================================================
# TESTER OUTPUT
# =============================================================================

class TestCase(msgspec.Struct, kw_only=True, frozen=True):
    """
    A single test case within a test suite.
    """
    name: str
    test_type: str  # "unit", "property", "contract", "integration"
    code: str
    expected_outcome: str = "pass"


class TestGeneration(msgspec.Struct, kw_only=True, frozen=True):
    """
    Output schema for the Tester agent.

    Contains test code and metadata for a CODE node.
    """
    filename: str
    test_code: str
    target_node_id: str
    test_cases: List[TestCase] = []
    coverage_estimate: float = 0.0


# =============================================================================
# VERIFIER OUTPUT
# =============================================================================

class VerificationResult(msgspec.Struct, kw_only=True, frozen=True):
    """
    Output schema for the Verifier agent.

    Contains the verdict on whether code meets its specification.
    """
    verdict: str  # "PASS", "FAIL", "NEEDS_REVISION"
    reasoning: str
    issues_found: List[str] = []
    suggestions: List[str] = []
    target_code_id: str = ""


# =============================================================================
# RESEARCHER OUTPUT
# =============================================================================

class ResearchFinding(msgspec.Struct, kw_only=True, frozen=True):
    """
    A single finding from research.
    """
    topic: str
    summary: str
    source: Optional[str] = None
    confidence: float = 1.0


class ResearchArtifact(msgspec.Struct, kw_only=True, frozen=True):
    """
    Output schema for the Researcher agent.

    Structured research artifact following Research Standard v1.0.
    """
    task_category: str  # "greenfield", "brownfield", "algorithmic", "systems", "debug"
    input_contract: str  # Python type annotation
    output_contract: str  # Python type annotation
    happy_path_examples: List[str] = []
    edge_cases: List[str] = []
    error_cases: List[str] = []
    complexity_bounds: Optional[str] = None
    security_posture: Optional[str] = None
    findings: List[ResearchFinding] = []


# =============================================================================
# DIALECTOR OUTPUT
# =============================================================================

class AmbiguityMarker(msgspec.Struct, kw_only=True, frozen=True):
    """
    A detected ambiguity in user requirements.
    """
    category: str  # "SUBJECTIVE", "COMPARATIVE", "UNDEFINED_PRONOUN", "UNDEFINED_TERM", "MISSING_CONTEXT"
    text: str  # The ambiguous text
    impact: str  # "BLOCKING", "CLARIFYING"
    suggested_question: Optional[str] = None
    suggested_answer: Optional[str] = None  # LLM's best-guess answer for automated/assisted mode


class DialectorOutput(msgspec.Struct, kw_only=True, frozen=True):
    """
    Output schema for the Dialector agent.

    Identifies ambiguities that need resolution before research.
    """
    is_clear: bool
    ambiguities: List[AmbiguityMarker] = []
    blocking_count: int = 0
    recommendation: str = ""  # "PROCEED", "CLARIFY", "BLOCK"


# =============================================================================
# SOCRATES OUTPUT
# =============================================================================

class ClarificationQuestion(msgspec.Struct, kw_only=True, frozen=True):
    """
    A clarifying question to resolve ambiguity.
    """
    question: str
    context: str
    options: List[str] = []  # Suggested answers if applicable
    priority: str = "medium"  # "low", "medium", "high"


class SocratesOutput(msgspec.Struct, kw_only=True, frozen=True):
    """
    Output schema for the Socrates agent.

    Contains questions to resolve ambiguities.
    """
    questions: List[ClarificationQuestion]
    research_summary: Optional[str] = None
    can_proceed_with_defaults: bool = False
    default_assumptions: List[str] = []


# =============================================================================
# GENERIC TOOL RESULTS (for agents/tools.py)
# =============================================================================

class ToolResult(msgspec.Struct, kw_only=True):
    """
    Generic result from any tool invocation.
    """
    success: bool
    message: str
    data: Optional[str] = None


class NodeResult(msgspec.Struct, kw_only=True):
    """
    Result of a node operation (add, update, query).
    """
    success: bool
    node_id: str
    node_type: Optional[str] = None
    status: Optional[str] = None
    message: str = ""


class EdgeResult(msgspec.Struct, kw_only=True):
    """
    Result of an edge operation.
    """
    success: bool
    source_id: str
    target_id: str
    edge_type: str
    message: str = ""


class QueryResult(msgspec.Struct, kw_only=True):
    """
    Result of a graph query operation.
    """
    success: bool
    nodes: List[str] = []
    count: int = 0
    message: str = ""


# =============================================================================
# ALIGNMENT VERIFICATION (Layer 7B - The Auditor)
# =============================================================================

class AlignmentCheck(msgspec.Struct, kw_only=True, frozen=True):
    """
    Result of alignment verification between generated code and spec.
    """
    aligned: bool
    spec_id: str
    code_id: str
    missing_requirements: List[str] = []
    extra_functionality: List[str] = []
    confidence: float = 1.0
    notes: Optional[str] = None


class SyntaxCheck(msgspec.Struct, kw_only=True, frozen=True):
    """
    Result of syntax validation via tree-sitter.
    """
    valid: bool
    language: str
    errors: List[str] = []
    warnings: List[str] = []


class AuditorReport(msgspec.Struct, kw_only=True, frozen=True):
    """
    Combined output from all Layer 7B auditor checks.
    """
    syntax_check: SyntaxCheck
    alignment_check: Optional[AlignmentCheck] = None
    approved: bool = False
    rejection_reasons: List[str] = []


# =============================================================================
# TYPESCRIPT/REACT CODE GENERATION SCHEMAS
# =============================================================================

class TypeScriptProp(msgspec.Struct, kw_only=True, frozen=True):
    """
    A single prop in a TypeScript interface or React component.
    """
    name: str
    type: str  # TypeScript type annotation
    required: bool = True
    description: str = ""
    default_value: Optional[str] = None


class TypeScriptInterface(msgspec.Struct, kw_only=True, frozen=True):
    """
    A TypeScript interface definition.
    """
    name: str
    description: str = ""
    props: List[TypeScriptProp] = []
    extends: List[str] = []  # Parent interfaces


class TypeScriptType(msgspec.Struct, kw_only=True, frozen=True):
    """
    A TypeScript type alias definition.
    """
    name: str
    definition: str  # e.g., '"primary" | "secondary"'
    description: str = ""


class ReactHook(msgspec.Struct, kw_only=True, frozen=True):
    """
    A React hook used by a component.
    """
    name: str  # e.g., "useState", "useEffect", "useGraphWebSocket"
    import_from: str  # e.g., "react", "./hooks/useGraphWebSocket"
    description: str = ""


class ReactComponentSpec(msgspec.Struct, kw_only=True, frozen=True):
    """
    Specification for a React component.

    This is the detailed spec that drives code generation.
    """
    name: str  # PascalCase component name
    description: str
    file_path: str  # e.g., "src/components/GraphViewer.tsx"

    # Props interface
    props_interface: Optional[TypeScriptInterface] = None

    # State management
    hooks: List[ReactHook] = []
    local_state: List[TypeScriptProp] = []  # useState variables

    # Behavior
    event_handlers: List[str] = []  # e.g., ["onClick", "onHover", "onNodeSelect"]
    side_effects: List[str] = []  # e.g., ["Fetch data on mount", "Subscribe to WebSocket"]

    # Rendering
    renders: str = ""  # High-level description of what it renders
    children_components: List[str] = []  # Child components it renders

    # Styling
    styling_approach: str = "tailwind"  # "tailwind", "css-modules", "styled-components"
    css_classes: List[str] = []

    # Dependencies
    imports: List[str] = []  # External imports needed
    internal_dependencies: List[str] = []  # Other components/hooks it depends on


class WebSocketMessageSchema(msgspec.Struct, kw_only=True, frozen=True):
    """
    Schema for a WebSocket message type.
    """
    message_type: str  # e.g., "snapshot", "delta", "heartbeat"
    direction: str  # "inbound", "outbound", "bidirectional"
    payload_interface: TypeScriptInterface
    description: str = ""


class ReactHookSpec(msgspec.Struct, kw_only=True, frozen=True):
    """
    Specification for a custom React hook.
    """
    name: str  # e.g., "useGraphWebSocket"
    description: str
    file_path: str  # e.g., "src/hooks/useGraphWebSocket.ts"

    # Parameters
    parameters: List[TypeScriptProp] = []

    # Return value
    return_type: str  # TypeScript type annotation for return
    return_description: str = ""

    # Dependencies
    hooks_used: List[str] = []  # React hooks this hook uses internally
    imports: List[str] = []


class TypeScriptCodeGeneration(msgspec.Struct, kw_only=True, frozen=True):
    """
    Output schema for TypeScript code generation.

    Similar to CodeGeneration but with TypeScript-specific fields.
    """
    filename: str
    code: str
    language: str = "typescript"  # "typescript" or "tsx"
    imports: List[str] = []
    exports: List[str] = []  # Named exports
    default_export: Optional[str] = None  # Default export name
    description: str = ""
    target_spec_id: Optional[str] = None

    # TypeScript-specific
    interfaces_defined: List[str] = []  # Names of interfaces in this file
    types_defined: List[str] = []  # Names of type aliases in this file
    components_defined: List[str] = []  # Names of React components in this file


class ReactAppSpec(msgspec.Struct, kw_only=True, frozen=True):
    """
    High-level specification for a React application.

    Defines the overall structure and components.
    """
    app_name: str
    description: str

    # Structure
    components: List[ReactComponentSpec] = []
    hooks: List[ReactHookSpec] = []
    types: List[TypeScriptInterface] = []
    type_aliases: List[TypeScriptType] = []

    # WebSocket schemas (for real-time features)
    websocket_messages: List[WebSocketMessageSchema] = []

    # Configuration
    entry_point: str = "src/App.tsx"
    styling_framework: str = "tailwind"
    state_management: str = "zustand"  # "zustand", "redux", "context"

    # Dependencies
    npm_dependencies: List[str] = []
    npm_dev_dependencies: List[str] = []


class FrontendImplementationPlan(msgspec.Struct, kw_only=True, frozen=True):
    """
    Output schema for frontend architecture planning.

    Produced by the Architect when planning a React frontend.
    """
    explanation: str
    app_spec: ReactAppSpec

    # File structure
    file_structure: List[str] = []  # List of file paths to create

    # Implementation order (topological sort)
    implementation_order: List[str] = []  # Component names in order to implement

    # API integration points
    api_endpoints_used: List[str] = []  # Backend endpoints the frontend will call
    websocket_endpoints: List[str] = []  # WebSocket endpoints

    # Estimated complexity
    estimated_complexity: str = "medium"
    notes: Optional[str] = None


# =============================================================================
# LEARNING SYSTEM SCHEMAS (Branch A: Schema Foundation)
# =============================================================================

class SignatureAction(str, Enum):
    """
    Actions that can be recorded in an agent signature.

    Tracks what the agent did to/with a node.
    """
    CREATED = "created"
    MODIFIED = "modified"
    VERIFIED = "verified"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    SUPERSEDED = "superseded"


class CyclePhase(str, Enum):
    """
    Phases in the Paragon development cycle.

    Maps to the orchestrator's main loop phases.
    """
    INIT = "init"
    DIALECTIC = "dialectic"
    RESEARCH = "research"
    CLARIFICATION = "clarification"
    PLAN = "plan"
    BUILD = "build"
    TEST = "test"
    PASSED = "passed"
    FAILED = "failed"


class FailureCode(str, Enum):
    """
    Categorization of failure modes for learning.

    Allows the system to understand patterns in failures.
    """
    F1 = "F1"  # Research Failure
    F2 = "F2"  # Implementation Failure
    F3 = "F3"  # Verification Failure
    F4 = "F4"  # External Failure (API, network, etc.)
    F5 = "F5"  # Indeterminate


class NodeOutcome(str, Enum):
    """
    Final outcome classification for a node after testing/verification.

    Used for learning from node histories.
    """
    VERIFIED_SUCCESS = "verified_success"
    VERIFIED_FAILURE = "verified_failure"
    TEST_PROD_DIVERGENCE = "test_prod_divergence"
    UNEXERCISED = "unexercised"
    INDETERMINATE = "indeterminate"


class AgentSignature(msgspec.Struct, frozen=True, kw_only=True):
    """
    Immutable signature recording an agent's interaction with a node.

    Every modification to the graph is signed by the agent that made it,
    creating a complete audit trail for learning and debugging.

    Fields:
        agent_id: Identifier of the agent (e.g., "Builder-v2", "Tester-v1")
        model_id: LLM model used (e.g., "claude-sonnet-4-5-20250929")
        phase: Which cycle phase this action occurred in
        action: What the agent did (created, modified, verified, etc.)
        temperature: LLM temperature parameter used
        context_constraints: Additional context (token limits, cost caps, etc.)
        timestamp: ISO 8601 timestamp of the signature
    """
    agent_id: str
    model_id: str
    phase: CyclePhase
    action: SignatureAction
    temperature: float
    context_constraints: dict
    timestamp: str


class SignatureChain(msgspec.Struct, kw_only=True):
    """
    Complete history of agent signatures for a node.

    Tracks the evolution of a node through its lifecycle, including
    replacements (when a node is superseded by a new implementation).

    Fields:
        node_id: Current node ID
        state_id: Changes when node content evolves (for versioning)
        signatures: Chronological list of all agent interactions
        is_replacement: True if this node replaced a previous one
        replaced_node_id: ID of the node this replaces (if any)
    """
    node_id: str
    state_id: str  # Changes on evolution
    signatures: List[AgentSignature]
    is_replacement: bool = False
    replaced_node_id: Optional[str] = None


# =============================================================================
# QUALITY GATE SCHEMAS (Branch C: Quality Gate)
# =============================================================================

class QualityViolation(msgspec.Struct, frozen=True, kw_only=True):
    """
    A single quality violation detected by the quality gate.

    Tracks violations against quality floor metrics including:
    - Test pass rate
    - Static analysis issues
    - Graph invariants
    - Cyclomatic complexity
    """
    metric: str
    threshold: str
    actual: str
    severity: Literal["critical", "warning", "info"]
    node_id: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    description: str = ""


class QualityReport(msgspec.Struct, kw_only=True):
    """
    Complete quality gate assessment report.

    This report determines if generated code meets Paragon's quality floor.
    Quality metrics have PRIMACY - they are hard constraints, not tradeoffs.

    Fields:
        passed: Overall pass/fail status
        violations: List of all violations found
        test_pass_rate: 0.0 to 1.0 (target: 1.0)
        static_analysis_criticals: Number of critical issues (target: 0)
        graph_invariant_compliance: 0.0 to 1.0 (target: 1.0)
        max_cyclomatic_complexity: Maximum complexity found (target: ≤ 15)
        quality_mode: "production" (strict) or "experimental" (lenient)
        total_nodes_checked: Number of nodes analyzed
        summary: Human-readable summary
    """
    passed: bool
    violations: List[QualityViolation]
    test_pass_rate: float
    static_analysis_criticals: int
    graph_invariant_compliance: float
    max_cyclomatic_complexity: int
    quality_mode: Literal["production", "experimental"]
    total_nodes_checked: int = 0
    summary: str = ""
