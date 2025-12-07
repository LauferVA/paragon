"""
Tests for Divergence Detector - Test/production mismatch detection.

Tests:
- Divergence detection logic
- False positive detection (tests passed, prod failed)
- False negative detection (tests failed, prod passed)
- Flaky test detection
- Divergence rate calculation
- Severity assessment
- Report generation

Layer: L3 (Divergence Detection)
Status: Production
"""
import pytest
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta

from infrastructure.training_store import TrainingStore
from infrastructure.divergence import DivergenceDetector, DivergenceEvent, DivergenceReport
from agents.schemas import NodeOutcome


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_divergence.db"

    yield db_path

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def store(temp_db):
    """Create a TrainingStore instance with temporary database."""
    return TrainingStore(db_path=temp_db)


@pytest.fixture
def detector(store):
    """Create a DivergenceDetector instance."""
    return DivergenceDetector(store=store)


class TestDivergenceDetection:
    """Test divergence detection logic."""

    def test_no_divergence_both_success(self, detector):
        """Test no divergence when both test and prod succeed."""
        event = detector.check_divergence(
            session_id="s1",
            test_passed=True,
            prod_outcome="success",
        )
        assert event is None

    def test_no_divergence_both_failure(self, detector):
        """Test no divergence when both test and prod fail."""
        event = detector.check_divergence(
            session_id="s1",
            test_passed=False,
            prod_outcome="failure",
        )
        assert event is None

    def test_false_positive_critical(self, detector):
        """Test false positive with critical severity."""
        event = detector.check_divergence(
            session_id="s1",
            test_passed=True,
            prod_outcome="broken",
            node_id="node_123",
        )

        assert event is not None
        assert event.session_id == "s1"
        assert event.node_id == "node_123"
        assert event.test_outcome == "passed"
        assert event.prod_outcome == "broken"
        assert event.divergence_type == "false_positive"
        assert event.severity == "critical"
        assert event.event_id is not None

    def test_false_positive_high(self, detector):
        """Test false positive with high severity."""
        event = detector.check_divergence(
            session_id="s2",
            test_passed=True,
            prod_outcome="wrong",
        )

        assert event is not None
        assert event.divergence_type == "false_positive"
        assert event.severity == "high"

    def test_false_positive_medium(self, detector):
        """Test false positive with medium severity."""
        event = detector.check_divergence(
            session_id="s3",
            test_passed=True,
            prod_outcome="rejected",
        )

        assert event is not None
        assert event.divergence_type == "false_positive"
        assert event.severity in ["high", "medium"]  # "rejected" is high severity

    def test_false_negative(self, detector):
        """Test false negative (tests too strict)."""
        event = detector.check_divergence(
            session_id="s4",
            test_passed=False,
            prod_outcome="actually_works",
        )

        assert event is not None
        assert event.test_outcome == "failed"
        assert event.prod_outcome == "actually_works"
        assert event.divergence_type == "false_negative"
        assert event.severity == "medium"

    def test_default_node_id(self, detector):
        """Test that missing node_id defaults to 'unknown'."""
        event = detector.check_divergence(
            session_id="s5",
            test_passed=True,
            prod_outcome="broken",
        )

        assert event is not None
        assert event.node_id == "unknown"

    def test_context_preserved(self, detector):
        """Test that context information is preserved."""
        event = detector.check_divergence(
            session_id="s6",
            test_passed=True,
            prod_outcome="wrong",
        )

        assert event is not None
        assert "test_passed" in event.context
        assert event.context["test_passed"] is True
        assert "prod_outcome_raw" in event.context


class TestSeverityAssessment:
    """Test severity assessment logic."""

    def test_critical_severity_broken(self, detector):
        """Test critical severity for 'broken' outcome."""
        severity = detector._assess_severity("broken")
        assert severity == "critical"

    def test_critical_severity_critical(self, detector):
        """Test critical severity for 'critical' outcome."""
        severity = detector._assess_severity("critical")
        assert severity == "critical"

    def test_critical_severity_unusable(self, detector):
        """Test critical severity for 'unusable' outcome."""
        severity = detector._assess_severity("unusable")
        assert severity == "critical"

    def test_critical_severity_crash(self, detector):
        """Test critical severity for 'crash' outcome."""
        severity = detector._assess_severity("crash")
        assert severity == "critical"

    def test_high_severity_wrong(self, detector):
        """Test high severity for 'wrong' outcome."""
        severity = detector._assess_severity("wrong")
        assert severity == "high"

    def test_high_severity_incorrect(self, detector):
        """Test high severity for 'incorrect' outcome."""
        severity = detector._assess_severity("incorrect")
        assert severity == "high"

    def test_high_severity_rejected(self, detector):
        """Test high severity for 'rejected' outcome."""
        severity = detector._assess_severity("rejected")
        assert severity == "high"

    def test_medium_severity_default(self, detector):
        """Test medium severity for unknown outcome."""
        severity = detector._assess_severity("some_other_issue")
        assert severity == "medium"

    def test_case_insensitive(self, detector):
        """Test that severity assessment is case-insensitive."""
        assert detector._assess_severity("BROKEN") == "critical"
        assert detector._assess_severity("Broken") == "critical"
        assert detector._assess_severity("WrOnG") == "high"


class TestDivergenceLogging:
    """Test divergence event logging."""

    def test_log_divergence(self, detector, store):
        """Test logging a divergence event."""
        event = DivergenceEvent(
            event_id="evt_123",
            session_id="s1",
            node_id="node_1",
            test_outcome="passed",
            prod_outcome="broken",
            divergence_type="false_positive",
            severity="critical",
            detected_at=datetime.utcnow().isoformat(),
            context={},
        )

        detector.log_divergence(event)

        # Verify it was logged
        import sqlite3

        with sqlite3.connect(store.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM divergence_events WHERE event_id = ?", ("evt_123",)
            )
            row = cursor.fetchone()

        assert row is not None
        assert row[1] == "s1"  # session_id
        assert row[2] == "node_1"  # node_id
        assert row[3] == "passed"  # test_outcome
        assert row[4] == "broken"  # prod_outcome
        assert row[5] == "false_positive"  # divergence_type
        assert row[6] == "critical"  # severity

    def test_log_multiple_events(self, detector):
        """Test logging multiple divergence events."""
        for i in range(3):
            event = DivergenceEvent(
                event_id=f"evt_{i}",
                session_id=f"s{i}",
                node_id=f"node_{i}",
                test_outcome="passed",
                prod_outcome="broken",
                divergence_type="false_positive",
                severity="critical",
                detected_at=datetime.utcnow().isoformat(),
                context={},
            )
            detector.log_divergence(event)

        # Verify all were logged
        report = detector.get_divergence_report()
        assert report.total_divergences == 3


class TestDivergenceReport:
    """Test divergence report generation."""

    def test_get_divergence_report_empty(self, detector):
        """Test report with no divergences."""
        report = detector.get_divergence_report()

        assert report.total_divergences == 0
        assert report.false_positives == 0
        assert report.false_negatives == 0
        assert report.flaky_tests == 0
        assert report.divergence_rate == 0.0
        assert len(report.events) == 0

    def test_get_divergence_report_with_events(self, detector, store):
        """Test report with divergence events."""
        # Log some divergences
        event1 = DivergenceEvent(
            event_id="evt_1",
            session_id="s1",
            node_id="node_1",
            test_outcome="passed",
            prod_outcome="broken",
            divergence_type="false_positive",
            severity="critical",
            detected_at=datetime.utcnow().isoformat(),
            context={},
        )
        detector.log_divergence(event1)

        event2 = DivergenceEvent(
            event_id="evt_2",
            session_id="s2",
            node_id="node_2",
            test_outcome="failed",
            prod_outcome="actually_works",
            divergence_type="false_negative",
            severity="medium",
            detected_at=datetime.utcnow().isoformat(),
            context={},
        )
        detector.log_divergence(event2)

        # Create session outcomes for rate calculation
        store.record_session_outcome("s1", NodeOutcome.TEST_PROD_DIVERGENCE)
        store.record_session_outcome("s2", NodeOutcome.TEST_PROD_DIVERGENCE)

        # Get report
        report = detector.get_divergence_report()

        assert report.total_divergences == 2
        assert report.false_positives == 1
        assert report.false_negatives == 1
        assert report.flaky_tests == 0
        assert len(report.events) == 2

    def test_get_divergence_report_by_session(self, detector):
        """Test report filtered by session."""
        # Log divergences for different sessions
        event1 = DivergenceEvent(
            event_id="evt_1",
            session_id="s1",
            node_id="node_1",
            test_outcome="passed",
            prod_outcome="broken",
            divergence_type="false_positive",
            severity="critical",
            detected_at=datetime.utcnow().isoformat(),
            context={},
        )
        detector.log_divergence(event1)

        event2 = DivergenceEvent(
            event_id="evt_2",
            session_id="s2",
            node_id="node_2",
            test_outcome="passed",
            prod_outcome="wrong",
            divergence_type="false_positive",
            severity="high",
            detected_at=datetime.utcnow().isoformat(),
            context={},
        )
        detector.log_divergence(event2)

        # Get report for session s1 only
        report = detector.get_divergence_report(session_id="s1")

        assert report.total_divergences == 1
        assert report.events[0].session_id == "s1"

    def test_get_divergence_report_time_window(self, detector):
        """Test report filtered by time window."""
        # Log old divergence
        old_event = DivergenceEvent(
            event_id="evt_old",
            session_id="s_old",
            node_id="node_old",
            test_outcome="passed",
            prod_outcome="broken",
            divergence_type="false_positive",
            severity="critical",
            detected_at=(datetime.utcnow() - timedelta(hours=25)).isoformat(),
            context={},
        )
        detector.log_divergence(old_event)

        # Log recent divergence
        recent_event = DivergenceEvent(
            event_id="evt_recent",
            session_id="s_recent",
            node_id="node_recent",
            test_outcome="passed",
            prod_outcome="broken",
            divergence_type="false_positive",
            severity="critical",
            detected_at=datetime.utcnow().isoformat(),
            context={},
        )
        detector.log_divergence(recent_event)

        # Get report for last 24 hours
        report = detector.get_divergence_report(time_window_hours=24)

        assert report.total_divergences == 1
        assert report.events[0].event_id == "evt_recent"


class TestDivergenceRate:
    """Test divergence rate calculation."""

    def test_calculate_divergence_rate_no_data(self, detector):
        """Test divergence rate with no data."""
        rate = detector.calculate_divergence_rate()
        assert rate == 0.0

    def test_calculate_divergence_rate_with_data(self, detector, store):
        """Test divergence rate calculation."""
        # Create 10 sessions, 3 with divergences
        for i in range(10):
            session_id = f"s{i}"
            store.record_session_outcome(
                session_id,
                NodeOutcome.VERIFIED_SUCCESS if i >= 3 else NodeOutcome.TEST_PROD_DIVERGENCE,
            )

            if i < 3:
                event = DivergenceEvent(
                    event_id=f"evt_{i}",
                    session_id=session_id,
                    node_id=f"node_{i}",
                    test_outcome="passed",
                    prod_outcome="broken",
                    divergence_type="false_positive",
                    severity="critical",
                    detected_at=datetime.utcnow().isoformat(),
                    context={},
                )
                detector.log_divergence(event)

        rate = detector.calculate_divergence_rate()

        # 3 sessions with divergences out of 10 total
        assert rate == pytest.approx(0.3, abs=0.01)

    def test_calculate_divergence_rate_all_divergent(self, detector, store):
        """Test divergence rate when all sessions have divergences."""
        for i in range(5):
            session_id = f"s{i}"
            store.record_session_outcome(session_id, NodeOutcome.TEST_PROD_DIVERGENCE)

            event = DivergenceEvent(
                event_id=f"evt_{i}",
                session_id=session_id,
                node_id=f"node_{i}",
                test_outcome="passed",
                prod_outcome="broken",
                divergence_type="false_positive",
                severity="critical",
                detected_at=datetime.utcnow().isoformat(),
                context={},
            )
            detector.log_divergence(event)

        rate = detector.calculate_divergence_rate()
        assert rate == 1.0

    def test_calculate_divergence_rate_none_divergent(self, detector, store):
        """Test divergence rate when no sessions have divergences."""
        for i in range(5):
            store.record_session_outcome(f"s{i}", NodeOutcome.VERIFIED_SUCCESS)

        rate = detector.calculate_divergence_rate()
        assert rate == 0.0


class TestFlakyTests:
    """Test flaky test detection."""

    def test_get_flaky_tests_none(self, detector):
        """Test getting flaky tests when none exist."""
        flaky = detector.get_flaky_tests()
        assert len(flaky) == 0

    def test_get_flaky_tests_with_flaky(self, detector):
        """Test getting flaky tests."""
        # Log multiple flaky events for same node
        for i in range(3):
            event = DivergenceEvent(
                event_id=f"evt_{i}",
                session_id=f"s{i}",
                node_id="flaky_node_1",
                test_outcome="passed",
                prod_outcome="broken",
                divergence_type="flaky",
                severity="medium",
                detected_at=datetime.utcnow().isoformat(),
                context={},
            )
            detector.log_divergence(event)

        # Log one flaky event for another node (should not be included - needs > 1)
        event = DivergenceEvent(
            event_id="evt_single",
            session_id="s_single",
            node_id="flaky_node_2",
            test_outcome="passed",
            prod_outcome="broken",
            divergence_type="flaky",
            severity="medium",
            detected_at=datetime.utcnow().isoformat(),
            context={},
        )
        detector.log_divergence(event)

        flaky = detector.get_flaky_tests()

        assert len(flaky) == 1
        assert "flaky_node_1" in flaky

    def test_get_flaky_tests_ignores_non_flaky(self, detector):
        """Test that non-flaky divergence types are ignored."""
        # Log false_positive events for same node
        for i in range(3):
            event = DivergenceEvent(
                event_id=f"evt_{i}",
                session_id=f"s{i}",
                node_id="node_1",
                test_outcome="passed",
                prod_outcome="broken",
                divergence_type="false_positive",
                severity="critical",
                detected_at=datetime.utcnow().isoformat(),
                context={},
            )
            detector.log_divergence(event)

        flaky = detector.get_flaky_tests()
        assert len(flaky) == 0


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow_false_positive(self, detector, store):
        """Test full workflow: detect -> log -> report."""
        # Detect divergence
        event = detector.check_divergence(
            session_id="s1",
            test_passed=True,
            prod_outcome="broken",
            node_id="node_1",
        )

        assert event is not None

        # Log it
        detector.log_divergence(event)

        # Create session outcome
        store.record_session_outcome("s1", NodeOutcome.TEST_PROD_DIVERGENCE)

        # Get report
        report = detector.get_divergence_report()

        assert report.total_divergences == 1
        assert report.false_positives == 1
        assert report.divergence_rate > 0.0

    def test_multiple_divergence_types(self, detector, store):
        """Test workflow with multiple divergence types."""
        # False positive
        event1 = detector.check_divergence(
            session_id="s1", test_passed=True, prod_outcome="broken", node_id="n1"
        )
        detector.log_divergence(event1)
        store.record_session_outcome("s1", NodeOutcome.TEST_PROD_DIVERGENCE)

        # False negative
        event2 = detector.check_divergence(
            session_id="s2", test_passed=False, prod_outcome="actually_works", node_id="n2"
        )
        detector.log_divergence(event2)
        store.record_session_outcome("s2", NodeOutcome.VERIFIED_SUCCESS)

        # Get report
        report = detector.get_divergence_report()

        assert report.total_divergences == 2
        assert report.false_positives == 1
        assert report.false_negatives == 1
        assert report.flaky_tests == 0
