"""
Unit tests for edge case collection system.

Tests:
- EdgeCaseClassifier detection criteria
- EdgeCaseStore persistence
- EdgeCaseCollector integration
- Parser divergence detection
"""
import pytest
import tempfile
from pathlib import Path

from infrastructure.edge_cases import (
    EdgeCaseClassifier,
    EdgeCaseStore,
    EdgeCaseCollector,
    EdgeCaseObservation,
    EdgeCase,
    check_parser_divergence,
)


# =============================================================================
# CLASSIFIER TESTS
# =============================================================================

class TestEdgeCaseClassifier:
    """Test edge case classification criteria."""

    def test_parser_divergence_detected(self):
        """Detect when tree-sitter and ast give different results."""
        classifier = EdgeCaseClassifier()

        obs = EdgeCaseObservation(
            node_id="test-1",
            tree_sitter_valid=True,
            ast_valid=False,
            ast_errors=["IndentationError: expected an indented block"],
        )

        categories, details = classifier.classify(obs)

        assert "parser_divergence" in categories
        assert "parser_divergence" in details
        assert details["parser_divergence"]["tree_sitter_valid"] is True
        assert details["parser_divergence"]["ast_valid"] is False

    def test_no_divergence_when_parsers_agree(self):
        """No detection when both parsers agree."""
        classifier = EdgeCaseClassifier()

        obs = EdgeCaseObservation(
            node_id="test-2",
            tree_sitter_valid=True,
            ast_valid=True,
        )

        categories, details = classifier.classify(obs)

        assert "parser_divergence" not in categories

    def test_exec_mismatch_detected(self):
        """Detect when syntax valid but execution fails."""
        classifier = EdgeCaseClassifier()

        obs = EdgeCaseObservation(
            node_id="test-3",
            syntax_valid=True,
            exec_success=False,
            exec_error="NameError: name 'undefined' is not defined",
        )

        categories, details = classifier.classify(obs)

        assert "exec_mismatch" in categories
        assert details["exec_mismatch"]["exec_error"] == "NameError: name 'undefined' is not defined"

    def test_boundary_hit_empty_string(self):
        """Detect empty string boundary."""
        classifier = EdgeCaseClassifier()

        obs = EdgeCaseObservation(
            node_id="test-4",
            input_value="",
        )

        categories, details = classifier.classify(obs)

        assert "boundary_hit" in categories
        assert details["boundary_hit"]["boundary_type"] == "empty"

    def test_boundary_hit_zero(self):
        """Detect zero boundary."""
        classifier = EdgeCaseClassifier()

        obs = EdgeCaseObservation(
            node_id="test-5",
            input_value=0,
        )

        categories, details = classifier.classify(obs)

        assert "boundary_hit" in categories
        assert details["boundary_hit"]["boundary_type"] == "zero"

    def test_retry_success_detected(self):
        """Detect successful retry."""
        classifier = EdgeCaseClassifier()

        obs = EdgeCaseObservation(
            node_id="test-6",
            retry_count=3,
            success=True,
        )

        categories, details = classifier.classify(obs)

        assert "retry_success" in categories
        assert details["retry_success"]["retry_count"] == 3

    def test_confidence_anomaly_detected(self):
        """Detect confidence vs outcome mismatch."""
        classifier = EdgeCaseClassifier()

        obs = EdgeCaseObservation(
            node_id="test-7",
            confidence=0.95,
            actual_outcome=0.0,  # Failed despite high confidence
        )

        categories, details = classifier.classify(obs)

        assert "confidence_anomaly" in categories
        assert details["confidence_anomaly"]["gap"] == 0.95

    def test_type_coercion_detected(self):
        """Detect type coercion edge case."""
        classifier = EdgeCaseClassifier()

        obs = EdgeCaseObservation(
            node_id="test-8",
            input_type="str",
            expected_type="int",
            success=True,
        )

        categories, details = classifier.classify(obs)

        assert "type_coercion" in categories

    def test_severity_ranking(self):
        """Test severity is correctly ranked."""
        classifier = EdgeCaseClassifier()

        # Parser divergence is "high"
        severity = classifier.get_highest_severity(["parser_divergence", "boundary_hit"])
        assert severity == "high"

        # Only low severity
        severity = classifier.get_highest_severity(["boundary_hit", "timing_outlier"])
        assert severity == "low"


# =============================================================================
# STORE TESTS
# =============================================================================

class TestEdgeCaseStore:
    """Test edge case persistence."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_edge_cases.db"
            store = EdgeCaseStore(db_path)
            yield store

    def test_store_and_retrieve(self, temp_store):
        """Store and retrieve an edge case."""
        case = EdgeCase(
            edge_case_id="ec-123",
            node_id="node-456",
            session_id="session-789",
            project_id="project-abc",
            categories=["parser_divergence"],
            severity="high",
            source="auto",
            code_snippet="if True:",
            description="Parser divergence detected",
            detection_details={"tree_sitter_valid": True, "ast_valid": False},
            detected_at="2025-12-07T12:00:00Z",
        )

        temp_store.store(case)

        retrieved = temp_store.get("ec-123")
        assert retrieved is not None
        assert retrieved.edge_case_id == "ec-123"
        assert retrieved.categories == ["parser_divergence"]
        assert retrieved.severity == "high"

    def test_query_by_category(self, temp_store):
        """Query edge cases by category."""
        # Store two edge cases
        case1 = EdgeCase(
            edge_case_id="ec-1",
            node_id="n1",
            session_id="s1",
            project_id="p1",
            categories=["parser_divergence"],
            severity="high",
            source="auto",
            code_snippet="code1",
            description="desc1",
            detection_details={},
            detected_at="2025-12-07T12:00:00Z",
        )
        case2 = EdgeCase(
            edge_case_id="ec-2",
            node_id="n2",
            session_id="s1",
            project_id="p1",
            categories=["boundary_hit"],
            severity="low",
            source="auto",
            code_snippet="code2",
            description="desc2",
            detection_details={},
            detected_at="2025-12-07T12:00:00Z",
        )

        temp_store.store(case1)
        temp_store.store(case2)

        # Query by category
        results = temp_store.query(category="parser_divergence")
        assert len(results) == 1
        assert results[0].edge_case_id == "ec-1"

    def test_mark_resolved(self, temp_store):
        """Mark an edge case as resolved."""
        case = EdgeCase(
            edge_case_id="ec-resolve",
            node_id="n1",
            session_id="s1",
            project_id="p1",
            categories=["test"],
            severity="medium",
            source="auto",
            code_snippet="code",
            description="desc",
            detection_details={},
            detected_at="2025-12-07T12:00:00Z",
        )

        temp_store.store(case)

        success = temp_store.mark_resolved("ec-resolve", notes="Fixed in commit abc123")
        assert success

        retrieved = temp_store.get("ec-resolve")
        assert retrieved.resolved is True
        assert retrieved.resolution_notes == "Fixed in commit abc123"

    def test_summary(self, temp_store):
        """Get summary statistics."""
        # Store some edge cases
        for i, (cat, sev) in enumerate([
            (["parser_divergence"], "high"),
            (["parser_divergence"], "high"),
            (["boundary_hit"], "low"),
        ]):
            case = EdgeCase(
                edge_case_id=f"ec-{i}",
                node_id=f"n{i}",
                session_id="s1",
                project_id="p1",
                categories=cat,
                severity=sev,
                source="auto",
                code_snippet="code",
                description="desc",
                detection_details={},
                detected_at="2025-12-07T12:00:00Z",
            )
            temp_store.store(case)

        summary = temp_store.get_summary()

        assert summary["total"] == 3
        assert summary["by_severity"]["high"] == 2
        assert summary["by_severity"]["low"] == 1
        assert summary["by_category"]["parser_divergence"] == 2


# =============================================================================
# COLLECTOR TESTS
# =============================================================================

class TestEdgeCaseCollector:
    """Test the main collector interface."""

    @pytest.fixture
    def temp_collector(self):
        """Create a collector with temporary store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_edge_cases.db"
            store = EdgeCaseStore(db_path)
            collector = EdgeCaseCollector(store)
            yield collector

    def test_check_and_store_auto(self, temp_collector):
        """Auto-detect and store edge case."""
        obs = EdgeCaseObservation(
            node_id="test-node",
            tree_sitter_valid=True,
            ast_valid=False,
            code_snippet="if True:",
        )

        result = temp_collector.check_and_store(obs)

        assert result is not None
        assert "parser_divergence" in result.categories
        assert result.source == "auto"

        # Verify stored
        cases = temp_collector.query()
        assert len(cases) == 1

    def test_flag_manually(self, temp_collector):
        """Manually flag an edge case."""
        case = temp_collector.flag_manually(
            node_id="manual-node",
            code_snippet="some weird code",
            reason="This behaves unexpectedly",
            flagged_by="developer",
        )

        assert case is not None
        assert "manual_flag" in case.categories
        assert case.source == "manual"
        assert case.flagged_by == "developer"
        assert case.flag_reason == "This behaves unexpectedly"

    def test_no_edge_case_for_normal_observation(self, temp_collector):
        """No edge case created for normal observations."""
        obs = EdgeCaseObservation(
            node_id="normal-node",
            tree_sitter_valid=True,
            ast_valid=True,
            syntax_valid=True,
        )

        result = temp_collector.check_and_store(obs)

        assert result is None


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestHelperFunctions:
    """Test helper functions."""

    def test_check_parser_divergence_both_valid(self):
        """Both parsers agree on valid code."""
        code = "def hello():\n    return 'world'"
        ts_valid, ast_valid, ts_errors, ast_errors = check_parser_divergence(code)

        assert ts_valid is True
        assert ast_valid is True
        assert len(ast_errors) == 0

    def test_check_parser_divergence_both_invalid(self):
        """Both parsers agree on invalid code."""
        code = "def broken(:"
        ts_valid, ast_valid, ts_errors, ast_errors = check_parser_divergence(code)

        # Both should detect the error
        assert ast_valid is False
        assert len(ast_errors) > 0

    def test_check_parser_divergence_incomplete_block(self):
        """Incomplete block case - may diverge."""
        code = "if True:"
        ts_valid, ast_valid, ts_errors, ast_errors = check_parser_divergence(code)

        # ast should fail
        assert ast_valid is False
        # tree-sitter may be more lenient (depends on tree-sitter-python version)
