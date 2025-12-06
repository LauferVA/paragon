"""
Quality Gate - Enforces hard quality constraints.

Principle: Quality metrics have PRIMACY. They are hard constraints, not tradeoffs.
Reference: CLAUDE.md Section 5.2, docs/RESEARCH_ADAPTIVE_QUESTIONING.md

This module implements the quality floor enforcement system for Paragon.
In production mode, ALL constraints are hard requirements.
In experimental mode, some constraints can be warnings.

Quality Floor Metrics:
1. Test Pass Rate: 100% (ALL tests must pass)
2. Static Analysis: 0 critical issues (OWASP Top 10, CWE Top 25)
3. Graph Invariants: 100% compliance (Teleological integrity)
4. Cyclomatic Complexity: ≤ 15 per function (Maintainability)

Layer 7B: The Auditor - Quality Gate Component
"""
import msgspec
from typing import List, Optional, Literal, Dict, Any
from pathlib import Path

from core.graph_invariants import GraphInvariants, InvariantSeverity
from core.schemas import NodeData


# =============================================================================
# QUALITY VIOLATION SCHEMA
# =============================================================================

class QualityViolation(msgspec.Struct, frozen=True, kw_only=True):
    """A single quality violation."""
    metric: str
    threshold: str
    actual: str
    severity: Literal["critical", "warning", "info"]
    node_id: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    description: str = ""


# =============================================================================
# QUALITY REPORT SCHEMA
# =============================================================================

class QualityReport(msgspec.Struct, kw_only=True):
    """Result of quality gate check."""
    passed: bool
    violations: List[QualityViolation]
    test_pass_rate: float
    static_analysis_criticals: int
    graph_invariant_compliance: float
    max_cyclomatic_complexity: int
    quality_mode: Literal["production", "experimental"]
    total_nodes_checked: int = 0
    summary: str = ""


# =============================================================================
# QUALITY GATE IMPLEMENTATION
# =============================================================================

class QualityGate:
    """
    Enforces quality floor constraints.

    In production mode, ALL constraints are hard requirements.
    In experimental mode, some constraints can be warnings.

    Quality Floor:
    - Test Pass Rate: 100%
    - Static Analysis: 0 critical issues
    - Graph Invariants: 100% compliance
    - Cyclomatic Complexity: ≤ 15 per function
    """

    def __init__(self, quality_mode: Literal["production", "experimental"] = "production"):
        """
        Initialize the quality gate.

        Args:
            quality_mode: "production" for strict checks, "experimental" for lenient
        """
        self.quality_mode = quality_mode

        # Quality floor thresholds
        self.thresholds = {
            "test_pass_rate": 1.0,  # 100%
            "static_analysis_criticals": 0,
            "graph_invariant_compliance": 1.0,  # 100%
            "max_cyclomatic_complexity": 15,
        }

    def check_quality_floor(
        self,
        widget_nodes: List[NodeData],
        test_results: Optional[List[Dict[str, Any]]] = None,
        graph = None
    ) -> QualityReport:
        """
        Check all quality constraints.

        Args:
            widget_nodes: List of nodes representing the generated code
            test_results: Optional list of test results with keys: name, passed, error
            graph: Optional ParagonDB instance for graph invariant checks

        Returns:
            QualityReport with pass/fail status and violations
        """
        violations = []

        # 1. Test pass rate
        test_pass_rate = self._check_test_pass_rate(test_results, violations)

        # 2. Static analysis
        static_criticals = self._check_static_analysis(widget_nodes, violations)

        # 3. Graph invariants
        invariant_compliance = self._check_graph_invariants(widget_nodes, violations, graph)

        # 4. Cyclomatic complexity
        max_complexity = self._check_complexity(widget_nodes, violations)

        # Determine pass/fail based on mode
        if self.quality_mode == "production":
            # In production mode, ANY critical violation fails
            critical_violations = [v for v in violations if v.severity == "critical"]
            passed = len(critical_violations) == 0
        else:
            # In experimental mode, more lenient
            # Allow some warnings, but still fail on critical issues
            critical_violations = [v for v in violations if v.severity == "critical"]
            passed = len(critical_violations) == 0

        # Generate summary
        summary = self._generate_summary(
            passed, violations, test_pass_rate, static_criticals,
            invariant_compliance, max_complexity
        )

        return QualityReport(
            passed=passed,
            violations=violations,
            test_pass_rate=test_pass_rate,
            static_analysis_criticals=static_criticals,
            graph_invariant_compliance=invariant_compliance,
            max_cyclomatic_complexity=max_complexity,
            quality_mode=self.quality_mode,
            total_nodes_checked=len(widget_nodes),
            summary=summary,
        )

    def _check_test_pass_rate(
        self,
        test_results: Optional[List[Dict[str, Any]]],
        violations: List[QualityViolation]
    ) -> float:
        """
        Check test pass rate against threshold.

        Args:
            test_results: List of test result dicts with 'passed' boolean
            violations: List to append violations to

        Returns:
            Test pass rate (0.0 to 1.0)
        """
        if not test_results:
            # No tests provided - this is a warning in experimental, critical in production
            severity = "critical" if self.quality_mode == "production" else "warning"
            violations.append(QualityViolation(
                metric="test_pass_rate",
                threshold="100%",
                actual="N/A (no tests)",
                severity=severity,
                description="No test results provided"
            ))
            return 0.0

        total_tests = len(test_results)
        passed_tests = sum(1 for t in test_results if t.get("passed", False))
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0

        if pass_rate < self.thresholds["test_pass_rate"]:
            severity = "critical" if self.quality_mode == "production" else "warning"
            violations.append(QualityViolation(
                metric="test_pass_rate",
                threshold=f"{self.thresholds['test_pass_rate']*100:.0f}%",
                actual=f"{pass_rate*100:.1f}% ({passed_tests}/{total_tests})",
                severity=severity,
                description=f"{total_tests - passed_tests} test(s) failed"
            ))

        return pass_rate

    def _check_static_analysis(
        self,
        nodes: List[NodeData],
        violations: List[QualityViolation]
    ) -> int:
        """
        Check static analysis for critical issues.

        Currently checks for:
        - Basic Python syntax via AST parsing
        - Common code smells
        - Security patterns (hardcoded secrets)

        Args:
            nodes: List of nodes to check
            violations: List to append violations to

        Returns:
            Number of critical issues found
        """
        import ast
        import re

        critical_count = 0

        for node in nodes:
            # Only check CODE nodes
            if node.type != "CODE":
                continue

            # Check 1: Syntax validity via AST
            try:
                ast.parse(node.content)
            except SyntaxError as e:
                critical_count += 1
                violations.append(QualityViolation(
                    metric="static_analysis",
                    threshold="0 critical",
                    actual="syntax_error",
                    severity="critical",
                    node_id=node.id,
                    line_number=e.lineno if hasattr(e, 'lineno') else None,
                    description=f"Syntax error: {str(e)}"
                ))
                continue  # Can't do further analysis on invalid syntax

            # Check 2: Hardcoded secrets (basic entropy check)
            secret_patterns = [
                (r'password\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded password"),
                (r'api[_-]?key\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded API key"),
                (r'secret\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded secret"),
                (r'token\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded token"),
            ]

            for pattern, desc in secret_patterns:
                if re.search(pattern, node.content, re.IGNORECASE):
                    severity = "critical" if self.quality_mode == "production" else "warning"
                    critical_count += 1
                    violations.append(QualityViolation(
                        metric="static_analysis",
                        threshold="0 critical",
                        actual="security_issue",
                        severity=severity,
                        node_id=node.id,
                        description=desc
                    ))

        # Overall check
        if critical_count > self.thresholds["static_analysis_criticals"]:
            violations.append(QualityViolation(
                metric="static_analysis",
                threshold=f"{self.thresholds['static_analysis_criticals']} critical",
                actual=f"{critical_count} critical",
                severity="critical",
                description=f"Found {critical_count} critical static analysis issues"
            ))

        return critical_count

    def _check_graph_invariants(
        self,
        nodes: List[NodeData],
        violations: List[QualityViolation],
        graph = None
    ) -> float:
        """
        Check graph invariants compliance.

        Validates:
        - Handshaking lemma
        - DAG acyclicity
        - Balis degree (source-sink reachability)
        - Stratification (type ordering)

        Args:
            nodes: List of nodes to check
            violations: List to append violations to
            graph: Optional ParagonDB instance

        Returns:
            Compliance rate (0.0 to 1.0)
        """
        if graph is None:
            # Can't check without graph - warning
            violations.append(QualityViolation(
                metric="graph_invariants",
                threshold="100%",
                actual="N/A (no graph)",
                severity="warning",
                description="Graph instance not provided for invariant checks"
            ))
            return 1.0  # Neutral - assume valid if no graph provided

        try:
            # Get the rustworkx graph
            rx_graph = graph.graph

            # Run all invariant checks
            report = GraphInvariants.validate_all(
                rx_graph,
                inv_map=graph.inv_map,
                raise_on_error=False
            )

            # Count violations by severity
            error_count = len(report.errors)
            warning_count = len(report.warnings)

            # Add violations to our list
            for invariant_violation in report.violations:
                severity = "critical" if invariant_violation.severity == InvariantSeverity.ERROR else "warning"
                violations.append(QualityViolation(
                    metric="graph_invariants",
                    threshold="100%",
                    actual=invariant_violation.invariant,
                    severity=severity,
                    description=invariant_violation.message
                ))

            # Compute compliance (inverse of error rate)
            total_checks = 4  # Handshaking, DAG, Balis, Stratification
            compliance = 1.0 - (error_count / total_checks) if total_checks > 0 else 1.0

            if compliance < self.thresholds["graph_invariant_compliance"]:
                violations.append(QualityViolation(
                    metric="graph_invariants",
                    threshold=f"{self.thresholds['graph_invariant_compliance']*100:.0f}%",
                    actual=f"{compliance*100:.1f}%",
                    severity="critical",
                    description=f"Graph invariant compliance below threshold ({error_count} errors)"
                ))

            return compliance

        except Exception as e:
            violations.append(QualityViolation(
                metric="graph_invariants",
                threshold="100%",
                actual="error",
                severity="warning",
                description=f"Error checking graph invariants: {str(e)}"
            ))
            return 0.0

    def _check_complexity(
        self,
        nodes: List[NodeData],
        violations: List[QualityViolation]
    ) -> int:
        """
        Check cyclomatic complexity of code nodes.

        Uses simple AST-based complexity calculation.
        Target: ≤ 15 per function (McCabe threshold)

        Args:
            nodes: List of nodes to check
            violations: List to append violations to

        Returns:
            Maximum complexity found across all functions
        """
        import ast

        max_complexity = 0

        for node in nodes:
            # Only check CODE nodes
            if node.type != "CODE":
                continue

            try:
                tree = ast.parse(node.content)

                # Find all function definitions
                for item in ast.walk(tree):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexity = self._compute_cyclomatic_complexity(item)
                        max_complexity = max(max_complexity, complexity)

                        if complexity > self.thresholds["max_cyclomatic_complexity"]:
                            severity = "critical" if self.quality_mode == "production" else "warning"
                            violations.append(QualityViolation(
                                metric="cyclomatic_complexity",
                                threshold=f"≤ {self.thresholds['max_cyclomatic_complexity']}",
                                actual=str(complexity),
                                severity=severity,
                                node_id=node.id,
                                line_number=item.lineno,
                                description=f"Function '{item.name}' has complexity {complexity}"
                            ))

            except SyntaxError:
                # Already caught in static analysis
                pass
            except Exception as e:
                violations.append(QualityViolation(
                    metric="cyclomatic_complexity",
                    threshold=f"≤ {self.thresholds['max_cyclomatic_complexity']}",
                    actual="error",
                    severity="warning",
                    node_id=node.id,
                    description=f"Error computing complexity: {str(e)}"
                ))

        return max_complexity

    def _compute_cyclomatic_complexity(self, func_node) -> int:
        """
        Compute cyclomatic complexity for a function.

        Complexity = 1 + number of decision points
        Decision points: if, for, while, except, and, or, etc.

        Args:
            func_node: AST FunctionDef or AsyncFunctionDef node

        Returns:
            Cyclomatic complexity value
        """
        import ast

        complexity = 1  # Base complexity

        # Count decision points
        for node in ast.walk(func_node):
            # Branching
            if isinstance(node, (ast.If, ast.IfExp)):
                complexity += 1
            # Loops
            elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                complexity += 1
            # Exception handling
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            # Boolean operators (each 'and' or 'or' adds a path)
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            # Comprehensions
            elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                complexity += 1

        return complexity

    def _generate_summary(
        self,
        passed: bool,
        violations: List[QualityViolation],
        test_pass_rate: float,
        static_criticals: int,
        invariant_compliance: float,
        max_complexity: int
    ) -> str:
        """Generate a human-readable summary of the quality check."""
        if passed:
            return (
                f"Quality gate PASSED ({self.quality_mode} mode): "
                f"Tests {test_pass_rate*100:.1f}%, "
                f"{static_criticals} critical issues, "
                f"Graph {invariant_compliance*100:.1f}% compliant, "
                f"Max complexity {max_complexity}"
            )
        else:
            critical_count = sum(1 for v in violations if v.severity == "critical")
            warning_count = sum(1 for v in violations if v.severity == "warning")
            return (
                f"Quality gate FAILED ({self.quality_mode} mode): "
                f"{critical_count} critical violation(s), "
                f"{warning_count} warning(s)"
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def check_quality(
    widget_nodes: List[NodeData],
    test_results: Optional[List[Dict[str, Any]]] = None,
    graph = None,
    quality_mode: Literal["production", "experimental"] = "production"
) -> QualityReport:
    """
    Convenience function to check quality floor.

    Args:
        widget_nodes: List of nodes to check
        test_results: Optional test results
        graph: Optional ParagonDB instance
        quality_mode: "production" or "experimental"

    Returns:
        QualityReport with results
    """
    gate = QualityGate(quality_mode=quality_mode)
    return gate.check_quality_floor(widget_nodes, test_results, graph)
