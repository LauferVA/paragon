"""
Divergence Detector Demo - Example usage scenarios.

Demonstrates the Divergence Detector catching test/production mismatches.

Usage:
    python examples/divergence_demo.py
"""
import tempfile
from pathlib import Path
from datetime import datetime

from infrastructure.training_store import TrainingStore
from infrastructure.divergence import DivergenceDetector
from agents.schemas import NodeOutcome


def demo_false_positive():
    """
    Demonstrate the most dangerous case: tests passed but production failed.

    This is the "false confidence" state - our verification system lied to us.
    """
    print("\n" + "=" * 80)
    print("DEMO 1: FALSE POSITIVE (Tests Passed, Production Failed)")
    print("=" * 80)
    print("\nScenario: Widget verification passed all tests, but user reports it's broken.")
    print("This is CRITICAL - our testing didn't catch a real production issue.\n")

    # Create temporary store and detector
    temp_db = Path(tempfile.mkdtemp()) / "demo.db"
    store = TrainingStore(db_path=temp_db)
    detector = DivergenceDetector(store=store)

    # Record session outcome
    session_id = "widget_session_1"
    store.record_session_outcome(session_id, NodeOutcome.TEST_PROD_DIVERGENCE)

    # Check for divergence
    event = detector.check_divergence(
        session_id=session_id,
        test_passed=True,  # Tests said everything was fine
        prod_outcome="broken",  # But user says it's broken
        node_id="widget_verification_123",
    )

    if event:
        print(f"✗ DIVERGENCE DETECTED!")
        print(f"  Event ID:         {event.event_id}")
        print(f"  Session:          {event.session_id}")
        print(f"  Node:             {event.node_id}")
        print(f"  Test Outcome:     {event.test_outcome}")
        print(f"  Production:       {event.prod_outcome}")
        print(f"  Type:             {event.divergence_type}")
        print(f"  Severity:         {event.severity}")
        print(f"\nAction: This divergence should trigger:")
        print("  1. Immediate investigation of test suite")
        print("  2. Review of verification criteria")
        print("  3. Possible rollback of widget changes")

        # Log the divergence
        detector.log_divergence(event)

        # Generate report
        report = detector.get_divergence_report()
        print(f"\nDivergence Report:")
        print(f"  Total Divergences:  {report.total_divergences}")
        print(f"  False Positives:    {report.false_positives}")
        print(f"  Divergence Rate:    {report.divergence_rate:.1%}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_db.parent)


def demo_false_negative():
    """
    Demonstrate false negative: tests failed but production actually works.

    This means our tests are too strict or testing the wrong things.
    """
    print("\n" + "=" * 80)
    print("DEMO 2: FALSE NEGATIVE (Tests Failed, Production Works)")
    print("=" * 80)
    print("\nScenario: Auth flow failed strict tests, but user says it works fine.")
    print("This suggests our tests may be overly strict or testing wrong criteria.\n")

    # Create temporary store and detector
    temp_db = Path(tempfile.mkdtemp()) / "demo.db"
    store = TrainingStore(db_path=temp_db)
    detector = DivergenceDetector(store=store)

    # Record session outcome
    session_id = "auth_session_2"
    store.record_session_outcome(session_id, NodeOutcome.VERIFIED_SUCCESS)

    # Check for divergence
    event = detector.check_divergence(
        session_id=session_id,
        test_passed=False,  # Tests failed
        prod_outcome="actually_works",  # But it works in production
        node_id="auth_flow_456",
    )

    if event:
        print(f"⚠ DIVERGENCE DETECTED!")
        print(f"  Event ID:         {event.event_id}")
        print(f"  Session:          {event.session_id}")
        print(f"  Node:             {event.node_id}")
        print(f"  Test Outcome:     {event.test_outcome}")
        print(f"  Production:       {event.prod_outcome}")
        print(f"  Type:             {event.divergence_type}")
        print(f"  Severity:         {event.severity}")
        print(f"\nAction: This divergence should trigger:")
        print("  1. Review of test requirements")
        print("  2. Possible relaxation of test criteria")
        print("  3. Better alignment between tests and real-world usage")

        # Log the divergence
        detector.log_divergence(event)

    # Cleanup
    import shutil
    shutil.rmtree(temp_db.parent)


def demo_multiple_divergences():
    """
    Demonstrate tracking multiple divergences across sessions.

    Shows how divergence rate is calculated and reported.
    """
    print("\n" + "=" * 80)
    print("DEMO 3: MULTIPLE DIVERGENCES ACROSS SESSIONS")
    print("=" * 80)
    print("\nScenario: Tracking divergences across 10 sessions.\n")

    # Create temporary store and detector
    temp_db = Path(tempfile.mkdtemp()) / "demo.db"
    store = TrainingStore(db_path=temp_db)
    detector = DivergenceDetector(store=store)

    # Simulate 10 sessions with varying outcomes
    scenarios = [
        ("s1", True, "broken", NodeOutcome.TEST_PROD_DIVERGENCE, "widget_1"),
        ("s2", True, "success", NodeOutcome.VERIFIED_SUCCESS, "widget_2"),
        ("s3", False, "failure", NodeOutcome.VERIFIED_FAILURE, "widget_3"),
        ("s4", True, "wrong", NodeOutcome.TEST_PROD_DIVERGENCE, "widget_4"),
        ("s5", True, "success", NodeOutcome.VERIFIED_SUCCESS, "widget_5"),
        ("s6", False, "actually_works", NodeOutcome.VERIFIED_SUCCESS, "widget_6"),
        ("s7", True, "success", NodeOutcome.VERIFIED_SUCCESS, "widget_7"),
        ("s8", True, "critical", NodeOutcome.TEST_PROD_DIVERGENCE, "widget_8"),
        ("s9", True, "success", NodeOutcome.VERIFIED_SUCCESS, "widget_9"),
        ("s10", True, "success", NodeOutcome.VERIFIED_SUCCESS, "widget_10"),
    ]

    print("Processing sessions...")
    divergence_count = 0

    for session_id, test_passed, prod_outcome, outcome, node_id in scenarios:
        # Record session outcome
        store.record_session_outcome(session_id, outcome)

        # Check for divergence
        event = detector.check_divergence(
            session_id=session_id,
            test_passed=test_passed,
            prod_outcome=prod_outcome,
            node_id=node_id,
        )

        if event:
            detector.log_divergence(event)
            divergence_count += 1
            print(f"  {session_id}: {event.divergence_type} ({event.severity})")

    # Generate comprehensive report
    print("\n" + "-" * 80)
    report = detector.get_divergence_report()

    print(f"\nFINAL DIVERGENCE REPORT:")
    print(f"  Total Sessions:     10")
    print(f"  Total Divergences:  {report.total_divergences}")
    print(f"  False Positives:    {report.false_positives} (tests passed, prod failed)")
    print(f"  False Negatives:    {report.false_negatives} (tests failed, prod worked)")
    print(f"  Flaky Tests:        {report.flaky_tests}")
    print(f"  Divergence Rate:    {report.divergence_rate:.1%}")

    print(f"\nTARGET METRICS:")
    print(f"  Divergence Rate:    < 5% (current: {report.divergence_rate:.1%})")

    if report.divergence_rate > 0.05:
        print(f"\n⚠ WARNING: Divergence rate exceeds 5% threshold!")
        print(f"  Recommended actions:")
        print(f"  1. Review test coverage and quality")
        print(f"  2. Improve verification criteria")
        print(f"  3. Add integration tests for real-world scenarios")
    else:
        print(f"\n✓ Divergence rate within acceptable range")

    # Show details of each divergence event
    print(f"\nDIVERGENCE EVENTS:")
    for i, event in enumerate(report.events, 1):
        print(f"\n  {i}. {event.session_id}")
        print(f"     Node:           {event.node_id}")
        print(f"     Type:           {event.divergence_type}")
        print(f"     Severity:       {event.severity}")
        print(f"     Test→Prod:      {event.test_outcome} → {event.prod_outcome}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_db.parent)


def main():
    """Run all demo scenarios."""
    print("\n" + "=" * 80)
    print("DIVERGENCE DETECTOR DEMONSTRATION")
    print("=" * 80)
    print("\nThe Divergence Detector catches the most dangerous state:")
    print("When tests pass but production fails (false confidence).")
    print("\nThis demo shows three scenarios:\n")
    print("  1. False Positive - Tests passed, production failed (CRITICAL)")
    print("  2. False Negative - Tests failed, production worked (Overly strict)")
    print("  3. Multiple Divergences - Tracking and reporting across sessions")

    demo_false_positive()
    demo_false_negative()
    demo_multiple_divergences()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  • False positives are CRITICAL - they indicate broken verification")
    print("  • Track divergence rate across all sessions (target: < 5%)")
    print("  • Use divergence events to improve test quality over time")
    print("  • Flaky tests indicate non-deterministic behavior requiring investigation")
    print("\n")


if __name__ == "__main__":
    main()
