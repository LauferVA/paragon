"""
Edge Case Collector - Captures interesting anomalies for learning.

Detects and stores edge cases for post-hoc analysis:
- Parser divergences (tree-sitter vs ast)
- Validation mismatches (syntax valid but exec fails)
- Boundary conditions
- Retry patterns
- Confidence anomalies
- Timing outliers

Supports both auto-detection and manual flagging.

Layer: Infrastructure
Status: Production
"""
import ast
import sqlite3
import uuid
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import msgspec


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class EdgeCaseCategory(msgspec.Struct, frozen=True):
    """Category definition for edge cases."""
    name: str
    description: str
    severity: str  # "low", "medium", "high", "critical"


class EdgeCaseObservation(msgspec.Struct, kw_only=True):
    """
    Raw observation that may contain edge cases.

    Populated during code validation/execution.
    """
    # Identity
    node_id: str
    session_id: str = ""
    project_id: str = ""

    # Code context
    code_snippet: str = ""
    code_hash: str = ""

    # Parser results
    tree_sitter_valid: Optional[bool] = None
    ast_valid: Optional[bool] = None
    tree_sitter_errors: List[str] = msgspec.field(default_factory=list)
    ast_errors: List[str] = msgspec.field(default_factory=list)

    # Execution results
    syntax_valid: bool = True
    exec_success: Optional[bool] = None
    exec_error: str = ""

    # Validation context
    input_value: Any = None
    input_type: str = ""
    expected_type: str = ""

    # Performance
    duration_ms: float = 0.0
    mean_duration_ms: float = 0.0  # Historical mean for comparison

    # Retry tracking
    retry_count: int = 0
    success: bool = True

    # Confidence tracking (for LLM outputs)
    confidence: float = 0.0
    actual_outcome: float = 0.0  # 1.0 = success, 0.0 = failure

    # Metadata
    timestamp: str = ""
    created_by: str = ""
    extra: Dict[str, Any] = msgspec.field(default_factory=dict)


class EdgeCase(msgspec.Struct, kw_only=True):
    """
    A detected or flagged edge case.

    Stored for learning and analysis.
    """
    # Identity
    edge_case_id: str
    node_id: str
    session_id: str
    project_id: str

    # Classification
    categories: List[str]  # Which criteria matched
    severity: str  # Highest severity among matched categories
    source: str  # "auto" or "manual"

    # Context
    code_snippet: str
    description: str

    # Detection details
    detection_details: Dict[str, Any]  # What triggered the detection

    # Timing
    detected_at: str

    # Manual flagging
    flagged_by: Optional[str] = None
    flag_reason: Optional[str] = None

    # Resolution tracking
    resolved: bool = False
    resolution_notes: str = ""
    resolved_at: Optional[str] = None


# =============================================================================
# EDGE CASE CLASSIFIER
# =============================================================================

class EdgeCaseClassifier:
    """
    Determines if an observation qualifies as an edge case.

    Uses explicit criteria to classify observations.
    """

    # Category definitions with severity
    CATEGORIES = {
        "parser_divergence": EdgeCaseCategory(
            name="parser_divergence",
            description="tree-sitter and ast.parse() give different results",
            severity="high",
        ),
        "exec_mismatch": EdgeCaseCategory(
            name="exec_mismatch",
            description="Syntax valid but execution fails",
            severity="medium",
        ),
        "boundary_hit": EdgeCaseCategory(
            name="boundary_hit",
            description="Value at exact boundary (empty, zero, max)",
            severity="low",
        ),
        "retry_success": EdgeCaseCategory(
            name="retry_success",
            description="Failed initially but succeeded on retry",
            severity="medium",
        ),
        "confidence_anomaly": EdgeCaseCategory(
            name="confidence_anomaly",
            description="Large gap between confidence and actual outcome",
            severity="high",
        ),
        "timing_outlier": EdgeCaseCategory(
            name="timing_outlier",
            description="Processing time significantly above average",
            severity="low",
        ),
        "type_coercion": EdgeCaseCategory(
            name="type_coercion",
            description="Input type differs from expected but works",
            severity="low",
        ),
        "manual_flag": EdgeCaseCategory(
            name="manual_flag",
            description="Manually flagged as interesting",
            severity="medium",
        ),
    }

    # Severity ranking
    SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}

    def __init__(self):
        """Initialize the classifier."""
        self._timing_history: List[float] = []
        self._timing_mean: float = 0.0
        self._timing_std: float = 0.0

    def update_timing_stats(self, duration_ms: float) -> None:
        """Update timing statistics for outlier detection."""
        self._timing_history.append(duration_ms)
        # Keep last 100 samples
        if len(self._timing_history) > 100:
            self._timing_history = self._timing_history[-100:]

        if len(self._timing_history) >= 3:
            self._timing_mean = statistics.mean(self._timing_history)
            self._timing_std = statistics.stdev(self._timing_history)

    def classify(self, obs: EdgeCaseObservation) -> tuple[List[str], Dict[str, Any]]:
        """
        Classify an observation and return matched categories with details.

        Returns:
            Tuple of (list of category names, dict of detection details)
        """
        matched = []
        details = {}

        # 1. Parser divergence
        if self._check_parser_divergence(obs):
            matched.append("parser_divergence")
            details["parser_divergence"] = {
                "tree_sitter_valid": obs.tree_sitter_valid,
                "ast_valid": obs.ast_valid,
                "tree_sitter_errors": obs.tree_sitter_errors,
                "ast_errors": obs.ast_errors,
            }

        # 2. Exec mismatch
        if self._check_exec_mismatch(obs):
            matched.append("exec_mismatch")
            details["exec_mismatch"] = {
                "syntax_valid": obs.syntax_valid,
                "exec_success": obs.exec_success,
                "exec_error": obs.exec_error,
            }

        # 3. Boundary hit
        if self._check_boundary_hit(obs):
            matched.append("boundary_hit")
            details["boundary_hit"] = {
                "input_value": str(obs.input_value)[:100],
                "boundary_type": self._identify_boundary_type(obs.input_value),
            }

        # 4. Retry success
        if self._check_retry_success(obs):
            matched.append("retry_success")
            details["retry_success"] = {
                "retry_count": obs.retry_count,
                "final_success": obs.success,
            }

        # 5. Confidence anomaly
        if self._check_confidence_anomaly(obs):
            matched.append("confidence_anomaly")
            details["confidence_anomaly"] = {
                "confidence": obs.confidence,
                "actual_outcome": obs.actual_outcome,
                "gap": abs(obs.confidence - obs.actual_outcome),
            }

        # 6. Timing outlier
        if self._check_timing_outlier(obs):
            matched.append("timing_outlier")
            details["timing_outlier"] = {
                "duration_ms": obs.duration_ms,
                "mean_ms": self._timing_mean,
                "std_ms": self._timing_std,
                "z_score": (obs.duration_ms - self._timing_mean) / self._timing_std if self._timing_std > 0 else 0,
            }

        # 7. Type coercion
        if self._check_type_coercion(obs):
            matched.append("type_coercion")
            details["type_coercion"] = {
                "input_type": obs.input_type,
                "expected_type": obs.expected_type,
            }

        return matched, details

    def _check_parser_divergence(self, obs: EdgeCaseObservation) -> bool:
        """Check if tree-sitter and ast give different results."""
        if obs.tree_sitter_valid is None or obs.ast_valid is None:
            return False
        return obs.tree_sitter_valid != obs.ast_valid

    def _check_exec_mismatch(self, obs: EdgeCaseObservation) -> bool:
        """Check if syntax valid but execution failed."""
        if obs.exec_success is None:
            return False
        return obs.syntax_valid and not obs.exec_success

    def _check_boundary_hit(self, obs: EdgeCaseObservation) -> bool:
        """Check if input is at a boundary value."""
        val = obs.input_value
        if val is None:
            return False

        # Empty checks
        if val == "" or val == [] or val == {} or val == set():
            return True

        # Zero checks
        if val == 0 or val == 0.0:
            return True

        # String length boundaries
        if isinstance(val, str) and len(val) in (1, 255, 256, 1024, 4096):
            return True

        # List length boundaries
        if isinstance(val, (list, tuple)) and len(val) in (0, 1, 100, 1000):
            return True

        return False

    def _identify_boundary_type(self, val: Any) -> str:
        """Identify what type of boundary value this is."""
        if val is None:
            return "null"
        if val == "" or val == [] or val == {} or val == set():
            return "empty"
        if val == 0 or val == 0.0:
            return "zero"
        if isinstance(val, str):
            return f"string_len_{len(val)}"
        if isinstance(val, (list, tuple)):
            return f"collection_len_{len(val)}"
        return "unknown"

    def _check_retry_success(self, obs: EdgeCaseObservation) -> bool:
        """Check if operation succeeded after retries."""
        return obs.retry_count > 0 and obs.success

    def _check_confidence_anomaly(self, obs: EdgeCaseObservation) -> bool:
        """Check if confidence significantly differs from outcome."""
        if obs.confidence == 0.0 and obs.actual_outcome == 0.0:
            return False  # No confidence data
        return abs(obs.confidence - obs.actual_outcome) > 0.5

    def _check_timing_outlier(self, obs: EdgeCaseObservation) -> bool:
        """Check if timing is > 3 standard deviations from mean."""
        if self._timing_std == 0 or len(self._timing_history) < 10:
            return False
        z_score = (obs.duration_ms - self._timing_mean) / self._timing_std
        return z_score > 3.0

    def _check_type_coercion(self, obs: EdgeCaseObservation) -> bool:
        """Check if input type differs from expected."""
        if not obs.input_type or not obs.expected_type:
            return False
        return obs.input_type != obs.expected_type and obs.success

    def get_highest_severity(self, categories: List[str]) -> str:
        """Get the highest severity among matched categories."""
        if not categories:
            return "low"

        severities = [
            self.CATEGORIES[cat].severity
            for cat in categories
            if cat in self.CATEGORIES
        ]

        if not severities:
            return "low"

        return max(severities, key=lambda s: self.SEVERITY_RANK.get(s, 0))


# =============================================================================
# EDGE CASE STORE
# =============================================================================

class EdgeCaseStore:
    """SQLite-backed storage for edge cases."""

    DB_PATH = Path("data/edge_cases.db")

    def __init__(self, db_path: Path | str | None = None):
        """Initialize the edge case store."""
        self.db_path = Path(db_path) if db_path else self.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Edge cases table
                CREATE TABLE IF NOT EXISTS edge_cases (
                    edge_case_id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    categories TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    code_snippet TEXT,
                    description TEXT,
                    detection_details TEXT,
                    detected_at TEXT NOT NULL,
                    flagged_by TEXT,
                    flag_reason TEXT,
                    resolved INTEGER DEFAULT 0,
                    resolution_notes TEXT,
                    resolved_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_edge_cases_session
                    ON edge_cases(session_id);
                CREATE INDEX IF NOT EXISTS idx_edge_cases_project
                    ON edge_cases(project_id);
                CREATE INDEX IF NOT EXISTS idx_edge_cases_severity
                    ON edge_cases(severity);
                CREATE INDEX IF NOT EXISTS idx_edge_cases_category
                    ON edge_cases(categories);
                CREATE INDEX IF NOT EXISTS idx_edge_cases_resolved
                    ON edge_cases(resolved);
            """)

    def store(self, edge_case: EdgeCase) -> str:
        """Store an edge case and return its ID."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO edge_cases (
                    edge_case_id, node_id, session_id, project_id,
                    categories, severity, source, code_snippet,
                    description, detection_details, detected_at,
                    flagged_by, flag_reason, resolved, resolution_notes, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge_case.edge_case_id,
                    edge_case.node_id,
                    edge_case.session_id,
                    edge_case.project_id,
                    json.dumps(edge_case.categories),
                    edge_case.severity,
                    edge_case.source,
                    edge_case.code_snippet[:10000] if edge_case.code_snippet else "",
                    edge_case.description,
                    json.dumps(edge_case.detection_details),
                    edge_case.detected_at,
                    edge_case.flagged_by,
                    edge_case.flag_reason,
                    1 if edge_case.resolved else 0,
                    edge_case.resolution_notes,
                    edge_case.resolved_at,
                ),
            )

        return edge_case.edge_case_id

    def get(self, edge_case_id: str) -> Optional[EdgeCase]:
        """Retrieve an edge case by ID."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM edge_cases WHERE edge_case_id = ?",
                (edge_case_id,),
            ).fetchone()

        if not row:
            return None

        return EdgeCase(
            edge_case_id=row["edge_case_id"],
            node_id=row["node_id"],
            session_id=row["session_id"],
            project_id=row["project_id"],
            categories=json.loads(row["categories"]),
            severity=row["severity"],
            source=row["source"],
            code_snippet=row["code_snippet"],
            description=row["description"],
            detection_details=json.loads(row["detection_details"]),
            detected_at=row["detected_at"],
            flagged_by=row["flagged_by"],
            flag_reason=row["flag_reason"],
            resolved=bool(row["resolved"]),
            resolution_notes=row["resolution_notes"] or "",
            resolved_at=row["resolved_at"],
        )

    def query(
        self,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 100,
    ) -> List[EdgeCase]:
        """Query edge cases with filters."""
        import json

        conditions = []
        params = []

        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if category:
            conditions.append("categories LIKE ?")
            params.append(f'%"{category}"%')
        if severity:
            conditions.append("severity = ?")
            params.append(severity)
        if resolved is not None:
            conditions.append("resolved = ?")
            params.append(1 if resolved else 0)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT * FROM edge_cases
                WHERE {where_clause}
                ORDER BY detected_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

        return [
            EdgeCase(
                edge_case_id=row["edge_case_id"],
                node_id=row["node_id"],
                session_id=row["session_id"],
                project_id=row["project_id"],
                categories=json.loads(row["categories"]),
                severity=row["severity"],
                source=row["source"],
                code_snippet=row["code_snippet"],
                description=row["description"],
                detection_details=json.loads(row["detection_details"]),
                detected_at=row["detected_at"],
                flagged_by=row["flagged_by"],
                flag_reason=row["flag_reason"],
                resolved=bool(row["resolved"]),
                resolution_notes=row["resolution_notes"] or "",
                resolved_at=row["resolved_at"],
            )
            for row in rows
        ]

    def mark_resolved(
        self,
        edge_case_id: str,
        notes: str = "",
    ) -> bool:
        """Mark an edge case as resolved."""
        now = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE edge_cases
                SET resolved = 1, resolution_notes = ?, resolved_at = ?
                WHERE edge_case_id = ?
                """,
                (notes, now, edge_case_id),
            )

        return cursor.rowcount > 0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of edge cases."""
        with sqlite3.connect(self.db_path) as conn:
            # Total counts
            total = conn.execute("SELECT COUNT(*) FROM edge_cases").fetchone()[0]
            resolved = conn.execute(
                "SELECT COUNT(*) FROM edge_cases WHERE resolved = 1"
            ).fetchone()[0]

            # By severity
            severity_counts = dict(conn.execute(
                "SELECT severity, COUNT(*) FROM edge_cases GROUP BY severity"
            ).fetchall())

            # By category (approximate since categories is JSON)
            category_counts = {}
            for cat in EdgeCaseClassifier.CATEGORIES:
                count = conn.execute(
                    "SELECT COUNT(*) FROM edge_cases WHERE categories LIKE ?",
                    (f'%"{cat}"%',),
                ).fetchone()[0]
                if count > 0:
                    category_counts[cat] = count

            # By source
            source_counts = dict(conn.execute(
                "SELECT source, COUNT(*) FROM edge_cases GROUP BY source"
            ).fetchall())

        return {
            "total": total,
            "resolved": resolved,
            "unresolved": total - resolved,
            "by_severity": severity_counts,
            "by_category": category_counts,
            "by_source": source_counts,
        }

    def clear_all(self) -> int:
        """Clear all edge cases. Returns count deleted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM edge_cases")
        return cursor.rowcount


# =============================================================================
# EDGE CASE COLLECTOR (Main Interface)
# =============================================================================

class EdgeCaseCollector:
    """
    Main interface for edge case collection.

    Combines classifier and store with convenience methods.
    """

    def __init__(self, store: Optional[EdgeCaseStore] = None):
        """Initialize the collector."""
        self.classifier = EdgeCaseClassifier()
        self.store = store or EdgeCaseStore()

    def check_and_store(
        self,
        obs: EdgeCaseObservation,
        auto_store: bool = True,
    ) -> Optional[EdgeCase]:
        """
        Check observation for edge cases and optionally store.

        Returns EdgeCase if detected, None otherwise.
        """
        # Update timing stats
        if obs.duration_ms > 0:
            self.classifier.update_timing_stats(obs.duration_ms)

        # Classify
        categories, details = self.classifier.classify(obs)

        if not categories:
            return None

        # Create edge case
        edge_case = EdgeCase(
            edge_case_id=str(uuid.uuid4()),
            node_id=obs.node_id,
            session_id=obs.session_id,
            project_id=obs.project_id,
            categories=categories,
            severity=self.classifier.get_highest_severity(categories),
            source="auto",
            code_snippet=obs.code_snippet,
            description=self._generate_description(categories, details),
            detection_details=details,
            detected_at=obs.timestamp or datetime.now(timezone.utc).isoformat(),
        )

        if auto_store:
            self.store.store(edge_case)

        return edge_case

    def flag_manually(
        self,
        node_id: str,
        code_snippet: str,
        reason: str,
        flagged_by: str,
        session_id: str = "",
        project_id: str = "",
    ) -> EdgeCase:
        """
        Manually flag something as an edge case.

        For "this is weird but I can't articulate why" situations.
        """
        edge_case = EdgeCase(
            edge_case_id=str(uuid.uuid4()),
            node_id=node_id,
            session_id=session_id,
            project_id=project_id,
            categories=["manual_flag"],
            severity="medium",
            source="manual",
            code_snippet=code_snippet,
            description=f"Manually flagged: {reason}",
            detection_details={"manual_reason": reason},
            detected_at=datetime.now(timezone.utc).isoformat(),
            flagged_by=flagged_by,
            flag_reason=reason,
        )

        self.store.store(edge_case)
        return edge_case

    def _generate_description(
        self,
        categories: List[str],
        details: Dict[str, Any],
    ) -> str:
        """Generate a human-readable description of the edge case."""
        parts = []

        for cat in categories:
            cat_def = self.classifier.CATEGORIES.get(cat)
            if cat_def:
                parts.append(f"[{cat}] {cat_def.description}")

        return "; ".join(parts)

    def query(self, **kwargs) -> List[EdgeCase]:
        """Query edge cases (delegates to store)."""
        return self.store.query(**kwargs)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics (delegates to store)."""
        return self.store.get_summary()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_parser_divergence(code: str) -> tuple[bool, bool, List[str], List[str]]:
    """
    Check if tree-sitter and ast.parse() give different results.

    Returns: (tree_sitter_valid, ast_valid, ts_errors, ast_errors)
    """
    # Check ast.parse()
    ast_valid = True
    ast_errors = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        ast_valid = False
        ast_errors.append(str(e))
    except Exception as e:
        ast_valid = False
        ast_errors.append(f"Unexpected: {e}")

    # Check tree-sitter (if available)
    tree_sitter_valid = True
    ts_errors = []
    try:
        import tree_sitter_python as tspython
        import tree_sitter as ts

        parser = ts.Parser(ts.Language(tspython.language()))
        tree = parser.parse(code.encode())

        # Check for ERROR nodes
        def has_error(node):
            if node.type == "ERROR":
                return True
            return any(has_error(child) for child in node.children)

        if has_error(tree.root_node):
            tree_sitter_valid = False
            ts_errors.append("tree-sitter found ERROR nodes")

    except ImportError:
        # tree-sitter not available, assume same as ast
        tree_sitter_valid = ast_valid
        ts_errors = ast_errors.copy()
    except Exception as e:
        tree_sitter_valid = False
        ts_errors.append(str(e))

    return tree_sitter_valid, ast_valid, ts_errors, ast_errors


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_collector: Optional[EdgeCaseCollector] = None


def get_edge_case_collector() -> EdgeCaseCollector:
    """Get or create the global EdgeCaseCollector instance."""
    global _collector
    if _collector is None:
        _collector = EdgeCaseCollector()
    return _collector


def log_edge_case_observation(
    node_id: str,
    code_snippet: str,
    tree_sitter_valid: Optional[bool] = None,
    ast_valid: Optional[bool] = None,
    syntax_valid: bool = True,
    exec_success: Optional[bool] = None,
    exec_error: str = "",
    session_id: str = "",
    project_id: str = "",
    duration_ms: float = 0.0,
    retry_count: int = 0,
    success: bool = True,
    **extra,
) -> Optional[EdgeCase]:
    """
    Convenience function to log an observation and detect edge cases.

    Returns EdgeCase if one was detected, None otherwise.
    """
    obs = EdgeCaseObservation(
        node_id=node_id,
        session_id=session_id,
        project_id=project_id,
        code_snippet=code_snippet,
        tree_sitter_valid=tree_sitter_valid,
        ast_valid=ast_valid,
        syntax_valid=syntax_valid,
        exec_success=exec_success,
        exec_error=exec_error,
        duration_ms=duration_ms,
        retry_count=retry_count,
        success=success,
        timestamp=datetime.now(timezone.utc).isoformat(),
        extra=extra,
    )

    return get_edge_case_collector().check_and_store(obs)
