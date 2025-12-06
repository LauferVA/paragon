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
from typing import List, Optional, Tuple


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
