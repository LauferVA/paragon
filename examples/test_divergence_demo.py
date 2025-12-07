"""
Divergence Detector Demo Test - Example usage as a test.

This test demonstrates the Divergence Detector in action.
Run with: pytest tests/test_divergence_demo.py -v -s
"""
import pytest
from pathlib import Path
import tempfile
import shutil

from infrastructure.training_store import TrainingStore
from infrastructure.divergence import DivergenceDetector
from agents.schemas import NodeOutcome


def test_divergence_detector_demo():
    """
    Demonstrate the Divergence Detector catching test/production mismatches.

    This is a working example showing how to detect and track divergences.
    """
    print("\n" + "=" * 80)
    print("DIVERGENCE DETECTOR DEMONSTRATION")
    print("=" * 80)

    # Setup
    temp_dir = tempfile.mkdtemp()
    temp_db = Path(temp_dir) / "demo.db"
    store = TrainingStore(db_path=temp_db)
    detector = DivergenceDetector(store=store)

    try:
        # DEMO 1: False Positive (CRITICAL - tests passed, production failed)
        print("\n1. FALSE POSITIVE - Tests Passed, Production Failed")
        print("-" * 80)
        print("Scenario: Widget verification passed all tests, but user reports it's broken.")

        session_id = "widget_session_1"
        store.record_session_outcome(session_id, NodeOutcome.TEST_PROD_DIVERGENCE)

        event = detector.check_divergence(
            session_id=session_id,
            test_passed=True,
            prod_outcome="broken",
            node_id="widget_verification_123",
        )

        assert event is not None
        assert event.divergence_type == "false_positive"
        assert event.severity == "critical"

        detector.log_divergence(event)

        print(f"✗ DIVERGENCE DETECTED!")
        print(f"  Type:      {event.divergence_type}")
        print(f"  Severity:  {event.severity}")
        print(f"  Test:      {event.test_outcome} → Production: {event.prod_outcome}")

        # DEMO 2: False Negative (tests too strict)
        print("\n2. FALSE NEGATIVE - Tests Failed, Production Works")
        print("-" * 80)
        print("Scenario: Auth flow failed tests, but user says it works fine.")

        session_id = "auth_session_2"
        store.record_session_outcome(session_id, NodeOutcome.VERIFIED_SUCCESS)

        event = detector.check_divergence(
            session_id=session_id,
            test_passed=False,
            prod_outcome="actually_works",
            node_id="auth_flow_456",
        )

        assert event is not None
        assert event.divergence_type == "false_negative"
        assert event.severity == "medium"

        detector.log_divergence(event)

        print(f"⚠ DIVERGENCE DETECTED!")
        print(f"  Type:      {event.divergence_type}")
        print(f"  Severity:  {event.severity}")
        print(f"  Test:      {event.test_outcome} → Production: {event.prod_outcome}")

        # DEMO 3: Multiple divergences across sessions
        print("\n3. TRACKING MULTIPLE DIVERGENCES")
        print("-" * 80)
        print("Scenario: Tracking divergences across 10 sessions.\n")

        scenarios = [
            ("s3", True, "critical", NodeOutcome.TEST_PROD_DIVERGENCE, "node_3"),
            ("s4", True, "success", NodeOutcome.VERIFIED_SUCCESS, "node_4"),
            ("s5", False, "failure", NodeOutcome.VERIFIED_FAILURE, "node_5"),
            ("s6", True, "wrong", NodeOutcome.TEST_PROD_DIVERGENCE, "node_6"),
            ("s7", True, "success", NodeOutcome.VERIFIED_SUCCESS, "node_7"),
            ("s8", False, "actually_works", NodeOutcome.VERIFIED_SUCCESS, "node_8"),
            ("s9", True, "success", NodeOutcome.VERIFIED_SUCCESS, "node_9"),
            ("s10", True, "unusable", NodeOutcome.TEST_PROD_DIVERGENCE, "node_10"),
        ]

        for session_id, test_passed, prod_outcome, outcome, node_id in scenarios:
            store.record_session_outcome(session_id, outcome)
            event = detector.check_divergence(
                session_id=session_id,
                test_passed=test_passed,
                prod_outcome=prod_outcome,
                node_id=node_id,
            )
            if event:
                detector.log_divergence(event)
                print(f"  {session_id}: {event.divergence_type} ({event.severity})")

        # Generate comprehensive report
        print("\n" + "=" * 80)
        print("DIVERGENCE REPORT")
        print("=" * 80)

        report = detector.get_divergence_report()

        print(f"\nMetrics:")
        print(f"  Total Sessions:     10")
        print(f"  Total Divergences:  {report.total_divergences}")
        print(f"  False Positives:    {report.false_positives}")
        print(f"  False Negatives:    {report.false_negatives}")
        print(f"  Flaky Tests:        {report.flaky_tests}")
        print(f"  Divergence Rate:    {report.divergence_rate:.1%}")

        # Verify the report
        assert report.total_divergences == 5  # 2 from initial demos + 3 from scenarios
        assert report.false_positives == 3  # s6 (wrong), s10 (unusable), widget_session_1 (broken)
        assert report.false_negatives == 2  # s8 (actually_works), auth_session_2 (actually_works)
        assert report.divergence_rate > 0.0

        print(f"\nTarget Metrics:")
        print(f"  Divergence Rate: < 5% (current: {report.divergence_rate:.1%})")

        if report.divergence_rate > 0.05:
            print(f"\n⚠ WARNING: Divergence rate exceeds 5% threshold!")
        else:
            print(f"\n✓ Divergence rate within acceptable range")

        # Show event details
        print(f"\nDivergence Events:")
        for i, event in enumerate(report.events[:5], 1):  # Show first 5
            print(f"  {i}. {event.session_id}: {event.divergence_type} ({event.severity})")

        print("\n" + "=" * 80)
        print("Key Takeaways:")
        print("  • False positives are CRITICAL - they indicate broken verification")
        print("  • Track divergence rate across all sessions (target: < 5%)")
        print("  • Use divergence events to improve test quality over time")
        print("=" * 80 + "\n")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
