"""
Tests for Training Store - SQLite persistence for learning system.

Tests:
- Schema creation
- Attribution recording and retrieval
- Signature chain storage and deserialization
- Session outcome tracking
- Success rate calculation
- Failure distribution queries
- Concurrent access safety

Layer: L1 (Database)
Status: Production
"""
import pytest
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from infrastructure.training_store import TrainingStore
from agents.schemas import (
    AgentSignature,
    SignatureChain,
    CyclePhase,
    FailureCode,
    NodeOutcome,
    SignatureAction,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_training.db"

    yield db_path

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def store(temp_db):
    """Create a TrainingStore instance with temporary database."""
    return TrainingStore(db_path=temp_db)


@pytest.fixture
def sample_signature():
    """Create a sample agent signature for testing."""
    return AgentSignature(
        agent_id="builder_v1",
        model_id="claude-sonnet-4-5",
        phase=CyclePhase.BUILD,
        action=SignatureAction.CREATED,
        temperature=0.7,
        context_constraints={"max_tokens": 4096, "cost_cap": 1.0},
        timestamp=datetime.utcnow().isoformat(),
    )


@pytest.fixture
def sample_signature_chain(sample_signature):
    """Create a sample signature chain for testing."""
    return SignatureChain(
        node_id="node_123",
        state_id="state_abc",
        signatures=[sample_signature],
        is_replacement=False,
        replaced_node_id=None,
    )


class TestSchemaCreation:
    """Test database schema initialization."""

    def test_creates_database_file(self, temp_db):
        """Test that initializing creates the database file."""
        assert not temp_db.exists()
        store = TrainingStore(db_path=temp_db)
        assert temp_db.exists()

    def test_creates_all_tables(self, store, temp_db):
        """Test that all required tables are created."""
        import sqlite3

        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "node_attributions" in tables
        assert "session_outcomes" in tables
        assert "signature_chains" in tables
        assert "divergence_events" in tables

    def test_creates_indexes(self, store, temp_db):
        """Test that indexes are created."""
        import sqlite3

        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
        )
        indexes = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "idx_attr_session" in indexes
        assert "idx_attr_node" in indexes
        assert "idx_attr_agent" in indexes
        assert "idx_attr_phase" in indexes
        assert "idx_chains_node" in indexes
        assert "idx_divergence_session" in indexes
        assert "idx_divergence_type" in indexes
        assert "idx_divergence_node" in indexes

    def test_idempotent_initialization(self, store):
        """Test that re-initializing doesn't break existing schema."""
        # Record some data
        session_id = "test_session"
        sample_sig = AgentSignature(
            agent_id="test",
            model_id="test",
            phase=CyclePhase.BUILD,
            action=SignatureAction.CREATED,
            temperature=0.5,
            context_constraints={},
            timestamp=datetime.utcnow().isoformat(),
        )
        store.record_attribution(session_id, sample_sig, "node_1", "state_1")

        # Re-initialize
        store._init_schema()

        # Data should still be there
        attributions = store.get_attributions_by_session(session_id)
        assert len(attributions) == 1


class TestAttributionRecording:
    """Test attribution recording and retrieval."""

    def test_record_attribution(self, store, sample_signature):
        """Test recording a single attribution."""
        session_id = "session_1"
        node_id = "node_1"
        state_id = "state_1"

        attribution_id = store.record_attribution(
            session_id, sample_signature, node_id, state_id
        )

        assert attribution_id is not None
        assert len(attribution_id) == 36  # UUID length

    def test_get_attributions_by_session(self, store, sample_signature):
        """Test retrieving attributions for a session."""
        session_id = "session_1"

        # Record multiple attributions
        store.record_attribution(session_id, sample_signature, "node_1", "state_1")
        store.record_attribution(session_id, sample_signature, "node_2", "state_2")

        # Retrieve
        attributions = store.get_attributions_by_session(session_id)

        assert len(attributions) == 2
        assert all(attr["session_id"] == session_id for attr in attributions)
        assert attributions[0]["agent_id"] == "builder_v1"
        assert attributions[0]["model_id"] == "claude-sonnet-4-5"
        assert attributions[0]["phase"] == "build"
        assert attributions[0]["action"] == "created"
        assert attributions[0]["temperature"] == 0.7

    def test_get_attributions_by_node(self, store, sample_signature):
        """Test retrieving attribution history for a node."""
        node_id = "node_1"

        # Record multiple attributions for same node
        sig1 = AgentSignature(
            agent_id="builder_v1",
            model_id="claude-sonnet-4-5",
            phase=CyclePhase.BUILD,
            action=SignatureAction.CREATED,
            temperature=0.7,
            context_constraints={},
            timestamp="2025-01-01T00:00:00",
        )

        sig2 = AgentSignature(
            agent_id="tester_v1",
            model_id="claude-sonnet-4-5",
            phase=CyclePhase.TEST,
            action=SignatureAction.VERIFIED,
            temperature=0.0,
            context_constraints={},
            timestamp="2025-01-01T00:01:00",
        )

        store.record_attribution("session_1", sig1, node_id, "state_1")
        store.record_attribution("session_1", sig2, node_id, "state_2")

        # Retrieve
        attributions = store.get_attributions_by_node(node_id)

        assert len(attributions) == 2
        assert attributions[0]["action"] == "created"
        assert attributions[1]["action"] == "verified"

    def test_get_attributions_empty_session(self, store):
        """Test retrieving attributions for non-existent session."""
        attributions = store.get_attributions_by_session("nonexistent")
        assert len(attributions) == 0


class TestSignatureChainStorage:
    """Test signature chain storage and retrieval."""

    def test_record_signature_chain(self, store, sample_signature_chain):
        """Test storing a signature chain."""
        chain_id = store.record_signature_chain(sample_signature_chain)

        assert chain_id is not None
        assert len(chain_id) == 36  # UUID length

    def test_get_signature_chain(self, store, sample_signature_chain):
        """Test retrieving a signature chain."""
        # Store
        store.record_signature_chain(sample_signature_chain)

        # Retrieve
        retrieved = store.get_signature_chain("node_123")

        assert retrieved is not None
        assert retrieved.node_id == "node_123"
        assert retrieved.state_id == "state_abc"
        assert retrieved.is_replacement is False
        assert retrieved.replaced_node_id is None
        assert len(retrieved.signatures) == 1
        assert retrieved.signatures[0].agent_id == "builder_v1"

    def test_get_signature_chain_nonexistent(self, store):
        """Test retrieving a non-existent chain."""
        retrieved = store.get_signature_chain("nonexistent")
        assert retrieved is None

    def test_signature_chain_with_replacement(self, store, sample_signature):
        """Test storing a chain that is a replacement."""
        chain = SignatureChain(
            node_id="node_new",
            state_id="state_new",
            signatures=[sample_signature],
            is_replacement=True,
            replaced_node_id="node_old",
        )

        store.record_signature_chain(chain)
        retrieved = store.get_signature_chain("node_new")

        assert retrieved.is_replacement is True
        assert retrieved.replaced_node_id == "node_old"

    def test_signature_chain_multiple_signatures(self, store):
        """Test chain with multiple signatures."""
        sig1 = AgentSignature(
            agent_id="builder_v1",
            model_id="claude-sonnet-4-5",
            phase=CyclePhase.BUILD,
            action=SignatureAction.CREATED,
            temperature=0.7,
            context_constraints={},
            timestamp="2025-01-01T00:00:00",
        )

        sig2 = AgentSignature(
            agent_id="tester_v1",
            model_id="claude-sonnet-4-5",
            phase=CyclePhase.TEST,
            action=SignatureAction.VERIFIED,
            temperature=0.0,
            context_constraints={},
            timestamp="2025-01-01T00:01:00",
        )

        chain = SignatureChain(
            node_id="node_multi",
            state_id="state_multi",
            signatures=[sig1, sig2],
            is_replacement=False,
            replaced_node_id=None,
        )

        store.record_signature_chain(chain)
        retrieved = store.get_signature_chain("node_multi")

        assert len(retrieved.signatures) == 2
        assert retrieved.signatures[0].action == SignatureAction.CREATED
        assert retrieved.signatures[1].action == SignatureAction.VERIFIED


class TestSessionOutcomes:
    """Test session outcome recording and retrieval."""

    def test_record_session_outcome_success(self, store):
        """Test recording a successful session outcome."""
        session_id = "session_success"

        store.record_session_outcome(
            session_id=session_id,
            outcome=NodeOutcome.VERIFIED_SUCCESS,
            stats={"total_nodes": 10, "total_iterations": 3, "total_tokens": 1000},
        )

        outcome = store.get_session_outcome(session_id)

        assert outcome is not None
        assert outcome["outcome"] == "verified_success"
        assert outcome["failure_code"] is None
        assert outcome["failure_phase"] is None
        assert outcome["total_nodes"] == 10
        assert outcome["total_iterations"] == 3
        assert outcome["total_tokens"] == 1000

    def test_record_session_outcome_failure(self, store):
        """Test recording a failed session outcome."""
        session_id = "session_failure"

        store.record_session_outcome(
            session_id=session_id,
            outcome=NodeOutcome.VERIFIED_FAILURE,
            failure_code=FailureCode.F2,
            failure_phase=CyclePhase.BUILD,
            stats={"total_nodes": 5, "total_iterations": 2, "total_tokens": 500},
        )

        outcome = store.get_session_outcome(session_id)

        assert outcome is not None
        assert outcome["outcome"] == "verified_failure"
        assert outcome["failure_code"] == "F2"
        assert outcome["failure_phase"] == "build"

    def test_record_session_outcome_minimal(self, store):
        """Test recording outcome with minimal data."""
        session_id = "session_minimal"

        store.record_session_outcome(
            session_id=session_id, outcome=NodeOutcome.INDETERMINATE
        )

        outcome = store.get_session_outcome(session_id)

        assert outcome is not None
        assert outcome["outcome"] == "indeterminate"
        assert outcome["total_nodes"] == 0
        assert outcome["total_iterations"] == 0
        assert outcome["total_tokens"] == 0

    def test_get_session_outcome_nonexistent(self, store):
        """Test retrieving non-existent session outcome."""
        outcome = store.get_session_outcome("nonexistent")
        assert outcome is None


class TestSuccessRate:
    """Test success rate calculation."""

    def test_success_rate_with_data(self, store, sample_signature):
        """Test calculating success rate with data."""
        # Record successes and failures
        for i in range(7):
            session_id = f"session_{i}"
            store.record_attribution(
                session_id, sample_signature, f"node_{i}", f"state_{i}"
            )
            outcome = (
                NodeOutcome.VERIFIED_SUCCESS if i < 5 else NodeOutcome.VERIFIED_FAILURE
            )
            store.record_session_outcome(session_id, outcome)

        # Calculate success rate
        rate = store.get_success_rate("claude-sonnet-4-5", "build")

        # 5 successes out of 7 total
        assert rate == pytest.approx(5.0 / 7.0, abs=0.01)

    def test_success_rate_no_data(self, store):
        """Test success rate with no data returns prior."""
        rate = store.get_success_rate("unknown-model", "unknown-phase")
        assert rate == 0.5  # Prior

    def test_success_rate_all_success(self, store, sample_signature):
        """Test success rate with all successes."""
        for i in range(5):
            session_id = f"session_{i}"
            store.record_attribution(
                session_id, sample_signature, f"node_{i}", f"state_{i}"
            )
            store.record_session_outcome(session_id, NodeOutcome.VERIFIED_SUCCESS)

        rate = store.get_success_rate("claude-sonnet-4-5", "build")
        assert rate == 1.0

    def test_success_rate_all_failure(self, store, sample_signature):
        """Test success rate with all failures."""
        for i in range(5):
            session_id = f"session_{i}"
            store.record_attribution(
                session_id, sample_signature, f"node_{i}", f"state_{i}"
            )
            store.record_session_outcome(session_id, NodeOutcome.VERIFIED_FAILURE)

        rate = store.get_success_rate("claude-sonnet-4-5", "build")
        assert rate == 0.0


class TestFailureDistribution:
    """Test failure distribution queries."""

    def test_failure_distribution_all_sessions(self, store):
        """Test getting failure distribution across all sessions."""
        # Record various failures
        store.record_session_outcome(
            "s1", NodeOutcome.VERIFIED_FAILURE, failure_code=FailureCode.F1
        )
        store.record_session_outcome(
            "s2", NodeOutcome.VERIFIED_FAILURE, failure_code=FailureCode.F2
        )
        store.record_session_outcome(
            "s3", NodeOutcome.VERIFIED_FAILURE, failure_code=FailureCode.F2
        )
        store.record_session_outcome(
            "s4", NodeOutcome.VERIFIED_FAILURE, failure_code=FailureCode.F3
        )
        store.record_session_outcome("s5", NodeOutcome.VERIFIED_SUCCESS)

        dist = store.get_failure_distribution()

        assert dist["F1"] == 1
        assert dist["F2"] == 2
        assert dist["F3"] == 1
        assert "F4" not in dist
        assert "F5" not in dist

    def test_failure_distribution_single_session(self, store):
        """Test getting failure distribution for specific session."""
        store.record_session_outcome(
            "s1", NodeOutcome.VERIFIED_FAILURE, failure_code=FailureCode.F1
        )

        dist = store.get_failure_distribution(session_id="s1")

        assert dist["F1"] == 1

    def test_failure_distribution_empty(self, store):
        """Test failure distribution with no failures."""
        store.record_session_outcome("s1", NodeOutcome.VERIFIED_SUCCESS)

        dist = store.get_failure_distribution()

        assert len(dist) == 0


class TestCountMethods:
    """Test count helper methods."""

    def test_session_count(self, store):
        """Test getting session count."""
        assert store.get_session_count() == 0

        store.record_session_outcome("s1", NodeOutcome.VERIFIED_SUCCESS)
        store.record_session_outcome("s2", NodeOutcome.VERIFIED_FAILURE)

        assert store.get_session_count() == 2

    def test_attribution_count(self, store, sample_signature):
        """Test getting attribution count."""
        assert store.get_attribution_count() == 0

        store.record_attribution("s1", sample_signature, "n1", "st1")
        store.record_attribution("s1", sample_signature, "n2", "st2")

        assert store.get_attribution_count() == 2

    def test_chain_count(self, store, sample_signature_chain):
        """Test getting chain count."""
        assert store.get_chain_count() == 0

        store.record_signature_chain(sample_signature_chain)

        assert store.get_chain_count() == 1


class TestConcurrentAccess:
    """Test concurrent access patterns."""

    def test_multiple_instances_same_db(self, temp_db, sample_signature):
        """Test multiple TrainingStore instances accessing same database."""
        store1 = TrainingStore(db_path=temp_db)
        store2 = TrainingStore(db_path=temp_db)

        # Write with store1
        store1.record_attribution("s1", sample_signature, "n1", "st1")

        # Read with store2
        attributions = store2.get_attributions_by_session("s1")
        assert len(attributions) == 1


class TestClearAll:
    """Test clear_all method."""

    def test_clear_all(self, store, sample_signature, sample_signature_chain):
        """Test clearing all data."""
        # Add data
        store.record_attribution("s1", sample_signature, "n1", "st1")
        store.record_signature_chain(sample_signature_chain)
        store.record_session_outcome("s1", NodeOutcome.VERIFIED_SUCCESS)

        # Verify data exists
        assert store.get_attribution_count() > 0
        assert store.get_chain_count() > 0
        assert store.get_session_count() > 0

        # Clear all
        store.clear_all()

        # Verify empty
        assert store.get_attribution_count() == 0
        assert store.get_chain_count() == 0
        assert store.get_session_count() == 0
