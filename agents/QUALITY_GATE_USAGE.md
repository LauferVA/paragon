# Quality Gate Usage Guide

## Overview

The Quality Gate enforces hard quality constraints on generated code. Quality metrics have **PRIMACY** - they are hard constraints, not tradeoffs.

## Quality Floor Metrics

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Test Pass Rate | 100% | ALL tests must pass |
| Static Analysis | 0 critical | OWASP Top 10, CWE Top 25 |
| Graph Invariants | 100% | Teleological integrity |
| Cyclomatic Complexity | ≤ 15/function | Maintainability |

## Basic Usage

```python
from agents.quality_gate import check_quality
from core.schemas import NodeData
from core.ontology import NodeType

# 1. Collect generated code nodes
widget_nodes = [
    NodeData.create(
        type=NodeType.CODE.value,
        content="def hello(): return 'world'",
        created_by="builder_agent"
    ),
]

# 2. Collect test results
test_results = [
    {"name": "test_hello", "passed": True},
]

# 3. Run quality gate
report = check_quality(
    widget_nodes=widget_nodes,
    test_results=test_results,
    graph=db,  # Optional ParagonDB instance
    quality_mode="production"  # or "experimental"
)

# 4. Check result
if report.passed:
    print(f"Quality gate PASSED: {report.summary}")
    # Proceed to deployment
else:
    print(f"Quality gate FAILED: {report.summary}")
    # Review violations
    for violation in report.violations:
        if violation.severity == "critical":
            print(f"  CRITICAL: {violation.description}")
```

## Quality Modes

### Production Mode (Default)
- Strict enforcement of all quality constraints
- Any critical violation fails the gate
- No tests = FAIL
- Syntax errors = FAIL
- Security issues = FAIL

### Experimental Mode
- More lenient enforcement
- Some violations are warnings instead of critical
- No tests = WARNING
- Useful for prototyping and development

```python
# Production mode (strict)
report_prod = check_quality(
    widget_nodes=nodes,
    test_results=tests,
    quality_mode="production"
)

# Experimental mode (lenient)
report_exp = check_quality(
    widget_nodes=nodes,
    test_results=tests,
    quality_mode="experimental"
)
```

## Quality Report Structure

```python
class QualityReport:
    passed: bool                      # Overall pass/fail
    violations: List[QualityViolation]  # All violations
    test_pass_rate: float             # 0.0 to 1.0
    static_analysis_criticals: int    # Number of critical issues
    graph_invariant_compliance: float # 0.0 to 1.0
    max_cyclomatic_complexity: int    # Maximum found
    quality_mode: str                 # "production" or "experimental"
    total_nodes_checked: int          # Number of nodes analyzed
    summary: str                      # Human-readable summary
```

## Quality Violation Structure

```python
class QualityViolation:
    metric: str          # Which metric failed
    threshold: str       # Expected threshold
    actual: str          # Actual value
    severity: str        # "critical", "warning", or "info"
    node_id: str | None  # Node that failed (if applicable)
    file_path: str | None  # File path (if applicable)
    line_number: int | None  # Line number (if applicable)
    description: str     # Human-readable description
```

## Integration with Orchestrator

```python
# In orchestrator.py BUILD -> TEST -> QUALITY_GATE workflow

# 1. BUILD phase
builder_result = builder_agent.build(spec_nodes)
widget_nodes = builder_result.nodes

# 2. TEST phase
tester_result = tester_agent.test(widget_nodes)
test_results = tester_result.results

# 3. QUALITY_GATE phase
from agents.quality_gate import check_quality

quality_report = check_quality(
    widget_nodes=widget_nodes,
    test_results=test_results,
    graph=self.db,
    quality_mode=self.config.quality_mode
)

if not quality_report.passed:
    # Handle failure
    self.log_failure(quality_report)

    # Classify failure type (F1-F5)
    failure_code = self.classify_failure(quality_report)

    # Store for learning
    self.store_failure_for_learning(quality_report, failure_code)

    # Decide: retry, escalate, or fail
    if self.should_retry(quality_report):
        return self.retry_build(with_feedback=quality_report.violations)
    else:
        return self.escalate_to_user(quality_report)
else:
    # Success - proceed
    self.mark_widget_verified()
    return widget_nodes
```

## Customizing Thresholds

```python
from agents.quality_gate import QualityGate

# Create gate with custom thresholds
gate = QualityGate(quality_mode="production")

# Override thresholds (if needed)
gate.thresholds["max_cyclomatic_complexity"] = 10  # Stricter
gate.thresholds["test_pass_rate"] = 0.95  # Allow 5% failures (not recommended!)

# Run check
report = gate.check_quality_floor(
    widget_nodes=nodes,
    test_results=tests,
    graph=db
)
```

## Checked Violations

### Test Pass Rate
- **Critical**: Test pass rate < 100%
- **Critical (production)**: No tests provided
- **Warning (experimental)**: No tests provided

### Static Analysis
- **Critical**: Syntax errors
- **Critical**: Hardcoded passwords
- **Critical**: Hardcoded API keys
- **Critical**: Hardcoded secrets/tokens

### Graph Invariants
- **Critical**: Handshaking lemma violation
- **Critical**: Cycle in DAG
- **Warning**: Balis degree (disconnected subgraphs)
- **Warning**: Stratification violations

### Cyclomatic Complexity
- **Critical (production)**: Function complexity > 15
- **Warning (experimental)**: Function complexity > 15

## Best Practices

1. **Always run quality gate in production mode for final deployment**
2. **Use experimental mode only during active development**
3. **Review all violations, not just critical ones**
4. **Fix violations at the source (in specs/architecture) not in post-processing**
5. **Track violation patterns over time for learning**

## Example Output

```
Quality gate PASSED (production mode): Tests 100.0%, 0 critical issues, Graph 100.0% compliant, Max complexity 8
```

```
Quality gate FAILED (production mode): 3 critical violation(s), 2 warning(s)
  - [CRITICAL] test_pass_rate: 2 test(s) failed
  - [CRITICAL] static_analysis: Syntax error: '(' was never closed
  - [CRITICAL] cyclomatic_complexity: Function 'process_data' has complexity 18
  - [WARNING] graph_invariants: stratification
  - [WARNING] static_analysis: security_issue
```

## Related Documentation

- [CLAUDE.md Section 5.2](../CLAUDE.md#52-quality-floor-primacy) - Quality floor definition
- [RESEARCH_ADAPTIVE_QUESTIONING.md](../docs/RESEARCH_ADAPTIVE_QUESTIONING.md) - Research on quality metrics
- [core/graph_invariants.py](../core/graph_invariants.py) - Graph invariant validation
