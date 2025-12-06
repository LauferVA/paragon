"""
Training Store - SQLite persistence for learning system.

Stores:
- Node attributions (which agent created/modified each node)
- Session outcomes (success/failure per session)
- Signature chains (full history of agent interactions)
- Failure classifications (F1-F5 taxonomy)

Reference: docs/IMPLEMENTATION_PLAN_LEARNING.md Section 4

Layer: L1 (Database)
Status: Production
"""
import sqlite3
import uuid
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import msgspec

from agents.schemas import (
    AgentSignature,
    SignatureChain,
    CyclePhase,
    FailureCode,
    NodeOutcome,
    SignatureAction,
)


class TrainingStore:
    """SQLite-backed storage for learning system data."""

    DB_PATH = Path("data/training.db")

    def __init__(self, db_path: Path | str | None = None):
        """
        Initialize the training store.

        Args:
            db_path: Optional path to database file (defaults to data/training.db)
        """
        self.db_path = Path(db_path) if db_path else self.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                -- Core attribution table
                CREATE TABLE IF NOT EXISTS node_attributions (
                    attribution_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    state_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    action TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Session outcomes for learning
                CREATE TABLE IF NOT EXISTS session_outcomes (
                    session_id TEXT PRIMARY KEY,
                    outcome TEXT NOT NULL,
                    failure_code TEXT,
                    failure_phase TEXT,
                    total_nodes INTEGER DEFAULT 0,
                    total_iterations INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Signature chains (full history)
                CREATE TABLE IF NOT EXISTS signature_chains (
                    chain_id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    state_id TEXT NOT NULL,
                    is_replacement INTEGER DEFAULT 0,
                    replaced_node_id TEXT,
                    signatures_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Divergence events (test-production mismatches)
                CREATE TABLE IF NOT EXISTS divergence_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    test_outcome TEXT NOT NULL,
                    prod_outcome TEXT NOT NULL,
                    divergence_type TEXT NOT NULL CHECK(divergence_type IN ('false_positive', 'false_negative', 'flaky')),
                    severity TEXT NOT NULL CHECK(severity IN ('critical', 'high', 'medium', 'low')),
                    detected_at TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_attr_session ON node_attributions(session_id);
                CREATE INDEX IF NOT EXISTS idx_attr_node ON node_attributions(node_id);
                CREATE INDEX IF NOT EXISTS idx_attr_agent ON node_attributions(agent_id);
                CREATE INDEX IF NOT EXISTS idx_attr_phase ON node_attributions(phase);
                CREATE INDEX IF NOT EXISTS idx_chains_node ON signature_chains(node_id);
                CREATE INDEX IF NOT EXISTS idx_divergence_session ON divergence_events(session_id);
                CREATE INDEX IF NOT EXISTS idx_divergence_type ON divergence_events(divergence_type);
                CREATE INDEX IF NOT EXISTS idx_divergence_node ON divergence_events(node_id);
            """
            )

    # === Write Methods ===

    def record_attribution(
        self, session_id: str, signature: AgentSignature, node_id: str, state_id: str
    ) -> str:
        """
        Record a single agent attribution for a node.

        Args:
            session_id: Session this attribution belongs to
            signature: Agent signature containing attribution details
            node_id: ID of the node being attributed
            state_id: State ID of the node at time of attribution

        Returns:
            attribution_id: UUID of the created attribution record
        """
        attribution_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO node_attributions (
                    attribution_id, session_id, node_id, state_id,
                    agent_id, model_id, phase, action, temperature, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    attribution_id,
                    session_id,
                    node_id,
                    state_id,
                    signature.agent_id,
                    signature.model_id,
                    signature.phase.value,
                    signature.action.value,
                    signature.temperature,
                    signature.timestamp,
                ),
            )

        return attribution_id

    def record_signature_chain(self, chain: SignatureChain) -> str:
        """
        Store a complete signature chain.

        Args:
            chain: SignatureChain to store

        Returns:
            chain_id: UUID of the created chain record
        """
        chain_id = str(uuid.uuid4())

        # Serialize signatures using msgspec
        signatures_json = msgspec.json.encode(chain.signatures).decode("utf-8")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO signature_chains (
                    chain_id, node_id, state_id, is_replacement,
                    replaced_node_id, signatures_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    chain_id,
                    chain.node_id,
                    chain.state_id,
                    1 if chain.is_replacement else 0,
                    chain.replaced_node_id,
                    signatures_json,
                ),
            )

        return chain_id

    def record_session_outcome(
        self,
        session_id: str,
        outcome: NodeOutcome,
        failure_code: FailureCode | None = None,
        failure_phase: CyclePhase | None = None,
        stats: dict | None = None,
    ) -> None:
        """
        Record the final outcome of a session.

        Args:
            session_id: Session identifier
            outcome: Final outcome classification
            failure_code: Optional failure code (F1-F5)
            failure_phase: Optional phase where failure occurred
            stats: Optional statistics dict with keys:
                   total_nodes, total_iterations, total_tokens
        """
        stats = stats or {}

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO session_outcomes (
                    session_id, outcome, failure_code, failure_phase,
                    total_nodes, total_iterations, total_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    outcome.value,
                    failure_code.value if failure_code else None,
                    failure_phase.value if failure_phase else None,
                    stats.get("total_nodes", 0),
                    stats.get("total_iterations", 0),
                    stats.get("total_tokens", 0),
                ),
            )

    # === Read Methods ===

    def get_attributions_by_session(self, session_id: str) -> List[dict]:
        """
        Get all attributions for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of attribution dicts with keys:
            attribution_id, session_id, node_id, state_id, agent_id,
            model_id, phase, action, temperature, timestamp
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM node_attributions
                WHERE session_id = ?
                ORDER BY created_at
            """,
                (session_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_attributions_by_node(self, node_id: str) -> List[dict]:
        """
        Get attribution history for a node.

        Args:
            node_id: Node identifier

        Returns:
            List of attribution dicts ordered by timestamp
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM node_attributions
                WHERE node_id = ?
                ORDER BY timestamp
            """,
                (node_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_signature_chain(self, node_id: str) -> SignatureChain | None:
        """
        Get the signature chain for a node.

        Args:
            node_id: Node identifier

        Returns:
            SignatureChain if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT node_id, state_id, is_replacement,
                       replaced_node_id, signatures_json
                FROM signature_chains
                WHERE node_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """,
                (node_id,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # Deserialize signatures using msgspec
            signatures = msgspec.json.decode(row[4], type=List[AgentSignature])

            return SignatureChain(
                node_id=row[0],
                state_id=row[1],
                is_replacement=bool(row[2]),
                replaced_node_id=row[3],
                signatures=signatures,
            )

    def get_success_rate(
        self, model_id: str, phase: str, constraints: dict | None = None
    ) -> float:
        """
        Calculate success rate for a model in a phase.

        Args:
            model_id: Model identifier (e.g., "claude-sonnet-4-5")
            phase: Phase name (e.g., "build", "test")
            constraints: Optional constraint filters (not yet implemented)

        Returns:
            Success rate as float 0.0-1.0, or 0.5 if no data
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN so.outcome = 'verified_success' THEN 1 ELSE 0 END) as successes
                FROM node_attributions na
                JOIN session_outcomes so ON na.session_id = so.session_id
                WHERE na.model_id = ? AND na.phase = ?
            """,
                (model_id, phase),
            )
            row = cursor.fetchone()

            if row and row[0] > 0:
                total, successes = row
                return float(successes) / float(total)

        # Prior: 0.5 if no data
        return 0.5

    def get_failure_distribution(self, session_id: str | None = None) -> dict:
        """
        Get distribution of failure codes.

        Args:
            session_id: Optional session to filter by

        Returns:
            Dict mapping failure codes to counts
        """
        query = """
            SELECT failure_code, COUNT(*) as count
            FROM session_outcomes
            WHERE failure_code IS NOT NULL
        """
        params = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " GROUP BY failure_code"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_session_outcome(self, session_id: str) -> dict | None:
        """
        Get the outcome record for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dict with session outcome data, or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM session_outcomes
                WHERE session_id = ?
            """,
                (session_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_session_count(self) -> int:
        """
        Get total number of sessions recorded.

        Returns:
            Count of sessions
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM session_outcomes")
            return cursor.fetchone()[0]

    def get_attribution_count(self) -> int:
        """
        Get total number of attributions recorded.

        Returns:
            Count of attributions
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM node_attributions")
            return cursor.fetchone()[0]

    def get_chain_count(self) -> int:
        """
        Get total number of signature chains recorded.

        Returns:
            Count of chains
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM signature_chains")
            return cursor.fetchone()[0]

    def clear_all(self) -> None:
        """
        Clear all data from all tables.

        WARNING: This is destructive and cannot be undone!
        Only use for testing.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM node_attributions")
            conn.execute("DELETE FROM session_outcomes")
            conn.execute("DELETE FROM signature_chains")
            conn.execute("DELETE FROM divergence_events")
