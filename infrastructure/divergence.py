"""
Divergence Detector - Catches test/production mismatches.

The most dangerous state is when tests pass but production fails.
This indicates flawed verification, which must be detected and logged.

Reference: docs/IMPLEMENTATION_PLAN_LEARNING.md Section 6

Layer: L3 (Divergence Detection)
Status: Production
"""
import msgspec
import uuid
from datetime import datetime, timedelta
from typing import List, Optional
from agents.schemas import NodeOutcome, CyclePhase, FailureCode


class DivergenceEvent(msgspec.Struct, kw_only=True, frozen=True):
    """
    A detected divergence between test and production.

    Represents the dangerous state where our verification process
    gave us false confidence (tests passed but production failed)
    or was overly strict (tests failed but production worked).
    """
    event_id: str
    session_id: str
    node_id: str
    test_outcome: str  # What tests said
    prod_outcome: str  # What production showed
    divergence_type: str  # "false_positive", "false_negative", "flaky"
    severity: str  # "critical", "high", "medium", "low"
    detected_at: str
    context: dict  # Additional context


class DivergenceReport(msgspec.Struct, kw_only=True):
    """
    Summary of divergence analysis.

    Provides metrics and insights about test-production divergence
    across sessions.
    """
    total_divergences: int
    false_positives: int  # Tests passed, prod failed
    false_negatives: int  # Tests failed, prod passed
    flaky_tests: int      # Inconsistent results
    divergence_rate: float
    events: List[DivergenceEvent]


class DivergenceDetector:
    """
    Detects mismatches between test and production outcomes.

    Monitors for the dangerous TEST_PROD_DIVERGENCE state and
    logs events for learning and improvement.

    This is the "canary in the coal mine" - when tests pass but
    production fails, it means our verification process is broken
    and needs immediate attention.
    """

    def __init__(self, store):
        """
        Initialize the divergence detector.

        Args:
            store: TrainingStore instance for persistence
        """
        self.store = store

    def check_divergence(
        self,
        session_id: str,
        test_passed: bool,
        prod_outcome: str,
        node_id: Optional[str] = None,
    ) -> Optional[DivergenceEvent]:
        """
        Check if there's a divergence between test and production.

        Args:
            session_id: Session to check
            test_passed: Whether tests passed
            prod_outcome: What happened in production
                         ("success", "failure", "rejected", "broken", "wrong")
            node_id: Optional specific node to attribute divergence to

        Returns:
            DivergenceEvent if divergence detected, None otherwise
        """
        # Normalize production outcome
        prod_success = prod_outcome in ("success", "verified_success", "acceptable", "actually_works")
        prod_failure = prod_outcome in ("failure", "rejected", "broken", "wrong", "incorrect", "unusable")

        # No divergence if outcomes match
        if test_passed and prod_success:
            return None
        if not test_passed and prod_failure:
            return None

        # Determine divergence type
        if test_passed and prod_failure:
            # CRITICAL: Tests lied to us - false confidence
            divergence_type = "false_positive"
            severity = self._assess_severity(prod_outcome)
            test_outcome_str = "passed"
        elif not test_passed and prod_success:
            # Tests were too strict
            divergence_type = "false_negative"
            severity = "medium"
            test_outcome_str = "failed"
        else:
            # Indeterminate production outcome
            return None

        # Create divergence event
        event = DivergenceEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            node_id=node_id or "unknown",
            test_outcome=test_outcome_str,
            prod_outcome=prod_outcome,
            divergence_type=divergence_type,
            severity=severity,
            detected_at=datetime.utcnow().isoformat(),
            context={
                "test_passed": test_passed,
                "prod_outcome_raw": prod_outcome,
            },
        )

        return event

    def log_divergence(self, event: DivergenceEvent) -> None:
        """
        Log a divergence event to the training store.

        Args:
            event: DivergenceEvent to log
        """
        import sqlite3

        with sqlite3.connect(self.store.db_path) as conn:
            conn.execute(
                """
                INSERT INTO divergence_events (
                    event_id, session_id, node_id, test_outcome,
                    prod_outcome, divergence_type, severity, detected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.event_id,
                    event.session_id,
                    event.node_id,
                    event.test_outcome,
                    event.prod_outcome,
                    event.divergence_type,
                    event.severity,
                    event.detected_at,
                ),
            )

    def get_divergence_report(
        self,
        session_id: Optional[str] = None,
        time_window_hours: Optional[int] = None,
    ) -> DivergenceReport:
        """
        Generate a report of divergences.

        Args:
            session_id: Optional filter by session
            time_window_hours: Optional time window in hours

        Returns:
            DivergenceReport with analysis
        """
        import sqlite3

        query = "SELECT * FROM divergence_events WHERE 1=1"
        params = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        if time_window_hours:
            cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
            query += " AND detected_at >= ?"
            params.append(cutoff.isoformat())

        query += " ORDER BY detected_at DESC"

        with sqlite3.connect(self.store.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        # Build events list
        events = []
        false_positives = 0
        false_negatives = 0
        flaky_tests = 0

        for row in rows:
            event = DivergenceEvent(
                event_id=row["event_id"],
                session_id=row["session_id"],
                node_id=row["node_id"],
                test_outcome=row["test_outcome"],
                prod_outcome=row["prod_outcome"],
                divergence_type=row["divergence_type"],
                severity=row["severity"],
                detected_at=row["detected_at"],
                context={},
            )
            events.append(event)

            if event.divergence_type == "false_positive":
                false_positives += 1
            elif event.divergence_type == "false_negative":
                false_negatives += 1
            elif event.divergence_type == "flaky":
                flaky_tests += 1

        # Calculate divergence rate
        total_divergences = len(events)
        divergence_rate = self.calculate_divergence_rate(
            session_id=session_id, time_window_hours=time_window_hours
        )

        return DivergenceReport(
            total_divergences=total_divergences,
            false_positives=false_positives,
            false_negatives=false_negatives,
            flaky_tests=flaky_tests,
            divergence_rate=divergence_rate,
            events=events,
        )

    def calculate_divergence_rate(
        self,
        session_id: Optional[str] = None,
        time_window_hours: Optional[int] = None,
    ) -> float:
        """
        Calculate overall divergence rate.

        Divergence rate = (divergences) / (total sessions)

        Args:
            session_id: Optional filter by session
            time_window_hours: Optional time window in hours

        Returns:
            Divergence rate as float 0.0-1.0
        """
        import sqlite3

        # Count divergences
        query_divs = "SELECT COUNT(DISTINCT session_id) as div_count FROM divergence_events WHERE 1=1"
        params_divs = []

        if session_id:
            query_divs += " AND session_id = ?"
            params_divs.append(session_id)

        if time_window_hours:
            cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
            query_divs += " AND detected_at >= ?"
            params_divs.append(cutoff.isoformat())

        # Count total sessions
        query_total = "SELECT COUNT(*) as total FROM session_outcomes WHERE 1=1"
        params_total = []

        if session_id:
            query_total += " AND session_id = ?"
            params_total.append(session_id)

        if time_window_hours:
            cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
            query_total += " AND created_at >= ?"
            params_total.append(cutoff.isoformat())

        with sqlite3.connect(self.store.db_path) as conn:
            div_count = conn.execute(query_divs, params_divs).fetchone()[0]
            total_count = conn.execute(query_total, params_total).fetchone()[0]

        if total_count == 0:
            return 0.0

        return float(div_count) / float(total_count)

    def get_flaky_tests(self) -> List[str]:
        """
        Get list of nodes with inconsistent test results.

        A flaky test is one that sometimes passes and sometimes fails
        on the same code, or shows different test/prod outcomes across runs.

        Returns:
            List of node IDs that have shown flaky behavior
        """
        import sqlite3

        # Find nodes with multiple divergence events
        query = """
            SELECT node_id, COUNT(*) as event_count
            FROM divergence_events
            WHERE divergence_type = 'flaky'
            GROUP BY node_id
            HAVING event_count > 1
            ORDER BY event_count DESC
        """

        with sqlite3.connect(self.store.db_path) as conn:
            cursor = conn.execute(query)
            return [row[0] for row in cursor.fetchall()]

    def _assess_severity(self, prod_outcome: str) -> str:
        """
        Assess severity based on production outcome language.

        Args:
            prod_outcome: Production outcome string

        Returns:
            Severity level: "critical", "high", "medium", or "low"
        """
        prod_outcome_lower = prod_outcome.lower()

        if any(word in prod_outcome_lower for word in ["broken", "critical", "unusable", "crash"]):
            return "critical"
        if any(word in prod_outcome_lower for word in ["wrong", "incorrect", "rejected"]):
            return "high"

        return "medium"
