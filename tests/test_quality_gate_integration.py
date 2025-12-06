"""
Integration test demonstrating quality gate usage.

This test shows how the quality gate would be integrated into
the orchestrator's BUILD -> TEST -> QUALITY_GATE workflow.
"""
import pytest
from agents.quality_gate import QualityGate, check_quality
from core.schemas import NodeData
from core.ontology import NodeType
from core.graph_db import ParagonDB


def test_quality_gate_end_to_end_passing():
    """
    Demonstrate quality gate in a successful scenario.

    Simulates:
    1. Builder generates code
    2. Tester runs tests
    3. Quality gate validates all metrics
    4. Everything passes
    """
    # Step 1: Simulated builder output (valid code)
    widget_nodes = [
        NodeData.create(
            type=NodeType.CODE.value,
            content="""
def calculate_sum(numbers):
    \"\"\"Calculate sum of numbers.\"\"\"
    return sum(numbers)
""",
            data={"file_path": "widget/calculator.py"},
            created_by="builder_agent"
        ),
        NodeData.create(
            type=NodeType.CODE.value,
            content="""
def format_result(value):
    \"\"\"Format result as string.\"\"\"
    return f"Result: {value}"
""",
            data={"file_path": "widget/formatter.py"},
            created_by="builder_agent"
        ),
    ]

    # Step 2: Simulated test results (all passing)
    test_results = [
        {"name": "test_calculate_sum_basic", "passed": True},
        {"name": "test_calculate_sum_empty", "passed": True},
        {"name": "test_format_result", "passed": True},
    ]

    # Step 3: Simulated graph (valid topology)
    db = ParagonDB()
    for node in widget_nodes:
        db.add_node(node)

    # Step 4: Run quality gate
    report = check_quality(
        widget_nodes=widget_nodes,
        test_results=test_results,
        graph=db,
        quality_mode="production"
    )

    # Step 5: Verify quality gate passes
    assert report.passed is True
    assert report.test_pass_rate == 1.0
    assert report.static_analysis_criticals == 0
    assert report.total_nodes_checked == 2
    assert "PASSED" in report.summary

    print(f"\n{report.summary}")
    print(f"Nodes checked: {report.total_nodes_checked}")
    print(f"Test pass rate: {report.test_pass_rate * 100:.0f}%")
    print(f"Static analysis: {report.static_analysis_criticals} critical issues")
    print(f"Graph compliance: {report.graph_invariant_compliance * 100:.0f}%")
    print(f"Max complexity: {report.max_cyclomatic_complexity}")


def test_quality_gate_end_to_end_failing():
    """
    Demonstrate quality gate in a failure scenario.

    Simulates:
    1. Builder generates code with issues
    2. Tests fail
    3. Quality gate catches violations
    4. Fails appropriately
    """
    # Step 1: Simulated builder output (code with issues)
    widget_nodes = [
        NodeData.create(
            type=NodeType.CODE.value,
            content="""
def connect_to_db():
    password = "hardcoded_secret123"
    return connect(password)
""",
            data={"file_path": "widget/db.py"},
            created_by="builder_agent"
        ),
        NodeData.create(
            type=NodeType.CODE.value,
            content="""
def bad_syntax(
    return "missing colon"
""",
            data={"file_path": "widget/broken.py"},
            created_by="builder_agent"
        ),
    ]

    # Step 2: Simulated test results (some failing)
    test_results = [
        {"name": "test_connect", "passed": False, "error": "Connection failed"},
        {"name": "test_basic", "passed": True},
    ]

    # Step 3: Run quality gate
    report = check_quality(
        widget_nodes=widget_nodes,
        test_results=test_results,
        quality_mode="production"
    )

    # Step 4: Verify quality gate fails
    assert report.passed is False
    assert report.test_pass_rate < 1.0
    assert report.static_analysis_criticals > 0
    assert "FAILED" in report.summary

    # Step 5: Examine violations
    critical_violations = [v for v in report.violations if v.severity == "critical"]
    assert len(critical_violations) >= 2  # At least test failures + syntax error

    # Find specific violation types
    syntax_violations = [v for v in report.violations if "syntax" in v.description.lower()]
    secret_violations = [v for v in report.violations if "password" in v.description.lower()]
    test_violations = [v for v in report.violations if v.metric == "test_pass_rate"]

    assert len(syntax_violations) >= 1
    assert len(secret_violations) >= 1
    assert len(test_violations) >= 1

    print(f"\n{report.summary}")
    print(f"Violations found: {len(report.violations)}")
    for violation in report.violations[:5]:  # Show first 5
        print(f"  - [{violation.severity.upper()}] {violation.metric}: {violation.description}")


def test_quality_gate_experimental_mode_more_lenient():
    """
    Demonstrate that experimental mode is more lenient.

    Shows that the same violations that fail in production
    may still pass in experimental mode (with warnings).
    """
    # Code with no tests
    widget_nodes = [
        NodeData.create(
            type=NodeType.CODE.value,
            content="def simple(): return 42",
            created_by="builder_agent"
        ),
    ]

    # No tests
    test_results = None

    # Production mode: should fail
    report_prod = check_quality(
        widget_nodes=widget_nodes,
        test_results=test_results,
        quality_mode="production"
    )
    assert report_prod.passed is False
    assert any(v.severity == "critical" for v in report_prod.violations)

    # Experimental mode: may have warnings but less strict
    report_exp = check_quality(
        widget_nodes=widget_nodes,
        test_results=test_results,
        quality_mode="experimental"
    )
    # In experimental mode, no tests is a warning not critical
    warning_violations = [v for v in report_exp.violations if v.severity == "warning"]
    assert len(warning_violations) >= 1

    print(f"\nProduction mode: {report_prod.summary}")
    print(f"Experimental mode: {report_exp.summary}")
