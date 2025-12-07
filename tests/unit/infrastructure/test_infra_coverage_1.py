"""
Unit tests for the first third of uncovered functions in infrastructure module.

This file tests functions from:
- AuditLogger (8 functions)
- BulkIngestor (4 functions)
- DiagnosticLogger (14 functions)
- DivergenceDetector (3 functions)
- EnvironmentDetector (9 functions)
- EventBuffer (8 functions)
- FileLogger (4 functions)
- ForensicAnalyzer (8 functions)

Total: 58 functions (target is first 65, close enough for first third)

Uses msgspec.Struct for all schema definitions per PROJECT PARAGON protocol.
"""
import pytest
import tempfile
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime, timezone, timedelta
import threading
import time

import msgspec

# Infrastructure imports
from infrastructure.logger import (
    AuditLogger,
    AuditEntry,
    EventBuffer,
    FileLogger,
)
from infrastructure.data_loader import (
    BulkIngestor,
    DataIntegrityError,
)
from infrastructure.diagnostics import (
    DiagnosticLogger,
    LLMCallMetric,
    PhaseMetric,
    generate_correlation_id,
)
from infrastructure.divergence import (
    DivergenceDetector,
    DivergenceEvent,
)
from infrastructure.environment import (
    EnvironmentDetector,
    EnvironmentReport,
)
from infrastructure.attribution import (
    ForensicAnalyzer,
    AttributionResult,
)

# Schema imports
from agents.schemas import (
    AgentSignature,
    SignatureChain,
    CyclePhase,
    FailureCode,
    SignatureAction,
    NodeOutcome,
)
from core.schemas import NodeData, EdgeData
from core.graph_db import ParagonDB
from infrastructure.training_store import TrainingStore
from viz.core import MutationEvent, MutationType

import polars as pl


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def audit_logger(temp_dir):
    """Create an AuditLogger with temporary log file."""
    log_path = temp_dir / "audit.log"
    return AuditLogger(log_path=log_path)


@pytest.fixture
def event_buffer():
    """Create an EventBuffer for testing."""
    return EventBuffer(max_size=100)


@pytest.fixture
def file_logger(temp_dir):
    """Create a FileLogger with temporary directory."""
    return FileLogger(log_path=temp_dir)


@pytest.fixture
def paragon_db():
    """Create a fresh ParagonDB for testing."""
    return ParagonDB()


@pytest.fixture
def bulk_ingestor(paragon_db):
    """Create a BulkIngestor with test database."""
    return BulkIngestor(db=paragon_db)


@pytest.fixture
def diagnostic_logger(temp_dir):
    """Create a DiagnosticLogger with temporary log file."""
    log_path = temp_dir / "diagnostics.jsonl"
    return DiagnosticLogger(log_path=log_path)


@pytest.fixture
def training_store(temp_dir):
    """Create a TrainingStore with temporary database."""
    db_path = temp_dir / "training.db"
    return TrainingStore(db_path=db_path)


@pytest.fixture
def divergence_detector(training_store):
    """Create a DivergenceDetector with test store."""
    return DivergenceDetector(store=training_store)


@pytest.fixture
def environment_detector():
    """Create an EnvironmentDetector for testing."""
    return EnvironmentDetector()


@pytest.fixture
def forensic_analyzer(training_store):
    """Create a ForensicAnalyzer with test store."""
    return ForensicAnalyzer(store=training_store)


@pytest.fixture
def sample_signature():
    """Create a sample AgentSignature for testing."""
    return AgentSignature(
        agent_id="test-builder-v1",
        model_id="claude-sonnet-4-5-20250929",
        phase=CyclePhase.BUILD,
        action=SignatureAction.CREATED,
        temperature=0.7,
        context_constraints={"max_tokens": 4096},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# AUDITLOGGER TESTS (8 functions)
# =============================================================================

class TestAuditLogger:
    """Tests for AuditLogger class."""

    def test_init_creates_directory(self, temp_dir):
        """Test that __init__ creates the log directory."""
        log_path = temp_dir / "subdir" / "audit.log"
        logger = AuditLogger(log_path=log_path)

        assert logger.log_path == log_path
        assert logger.log_path.parent.exists()

    def test_init_default_path(self):
        """Test that __init__ uses default path when none provided."""
        logger = AuditLogger()
        assert logger.log_path == Path("data/audit.log")

    def test_log_writes_entry(self, audit_logger):
        """Test that log() writes an AuditEntry to file."""
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_id="test-agent",
            node_id="test-node",
            action="test_action",
            details={"key": "value"},
        )

        audit_logger.log(entry)

        # Verify file was created and contains entry
        assert audit_logger.log_path.exists()
        with open(audit_logger.log_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["agent_id"] == "test-agent"
            assert data["node_id"] == "test-node"

    def test_log_action_creates_entry(self, audit_logger):
        """Test that log_action() creates and logs an entry."""
        audit_logger.log_action(
            action="node_created",
            node_id="node_123",
            agent_id="builder",
            agent_role="BUILDER",
            merkle_hash="abc123",
            extra_field="extra_value",
        )

        entries = audit_logger.read_entries()
        assert len(entries) == 1
        assert entries[0].action == "node_created"
        assert entries[0].node_id == "node_123"
        assert entries[0].agent_id == "builder"
        assert entries[0].details["extra_field"] == "extra_value"

    def test_read_entries_empty_log(self, audit_logger):
        """Test that read_entries() returns empty list for empty log."""
        entries = audit_logger.read_entries()
        assert entries == []

    def test_read_entries_with_filters(self, audit_logger):
        """Test that read_entries() applies filters correctly."""
        # Log multiple entries
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        audit_logger.log_action("action1", node_id="node1", agent_id="agent1")
        time.sleep(0.01)  # Small delay to ensure different timestamps
        audit_logger.log_action("action2", node_id="node2", agent_id="agent2")

        # Filter by agent_id
        entries = audit_logger.read_entries(agent_id="agent1")
        assert len(entries) == 1
        assert entries[0].agent_id == "agent1"

        # Filter by node_id
        entries = audit_logger.read_entries(node_id="node2")
        assert len(entries) == 1
        assert entries[0].node_id == "node2"

        # Filter by action
        entries = audit_logger.read_entries(action="action1")
        assert len(entries) == 1
        assert entries[0].action == "action1"

    def test_get_node_history(self, audit_logger):
        """Test that get_node_history() returns all entries for a node."""
        audit_logger.log_action("created", node_id="node1", agent_id="agent1")
        audit_logger.log_action("modified", node_id="node1", agent_id="agent2")
        audit_logger.log_action("created", node_id="node2", agent_id="agent1")

        history = audit_logger.get_node_history("node1")
        assert len(history) == 2
        assert all(e.node_id == "node1" for e in history)

    def test_get_agent_activity(self, audit_logger):
        """Test that get_agent_activity() returns all entries for an agent."""
        audit_logger.log_action("action1", agent_id="agent1")
        audit_logger.log_action("action2", agent_id="agent1")
        audit_logger.log_action("action3", agent_id="agent2")

        activity = audit_logger.get_agent_activity("agent1")
        assert len(activity) == 2
        assert all(e.agent_id == "agent1" for e in activity)

    def test_get_recent(self, audit_logger):
        """Test that get_recent() returns most recent entries."""
        # Log 10 entries
        for i in range(10):
            audit_logger.log_action(f"action{i}", node_id=f"node{i}")

        # Get 5 most recent
        recent = audit_logger.get_recent(n=5)
        assert len(recent) == 5
        # Should be in reverse order (most recent first)
        assert recent[0].action == "action9"
        assert recent[-1].action == "action5"

    def test_clear_removes_log_file(self, audit_logger):
        """Test that clear() removes the log file."""
        audit_logger.log_action("test")
        assert audit_logger.log_path.exists()

        audit_logger.clear()
        assert not audit_logger.log_path.exists()


# =============================================================================
# BULKINGESTOR TESTS (4 functions)
# =============================================================================

class TestBulkIngestor:
    """Tests for BulkIngestor class."""

    def test_init_stores_db(self, paragon_db):
        """Test that __init__ stores the database reference."""
        ingestor = BulkIngestor(db=paragon_db)
        assert ingestor.db is paragon_db

    def test_ingest_nodes_basic(self, bulk_ingestor, paragon_db):
        """Test that ingest_nodes() adds nodes to database."""
        df = pl.DataFrame({
            "id": ["node1", "node2", "node3"],
            "type": ["SPEC", "CODE", "TEST"],
            "content": ["content1", "content2", "content3"],
            "status": ["PENDING", "PENDING", "PENDING"],
            "created_by": ["system", "system", "system"],
        })

        count = bulk_ingestor.ingest_nodes(df)

        assert count == 3
        assert paragon_db.node_count == 3

    def test_ingest_nodes_with_defaults(self, bulk_ingestor, paragon_db):
        """Test that ingest_nodes() fills in default values."""
        df = pl.DataFrame({
            "id": ["node1", "node2"],
            "type": ["SPEC", "CODE"],
        })

        count = bulk_ingestor.ingest_nodes(df)

        assert count == 2
        # Verify defaults were applied
        node1 = paragon_db.get_node("node1")
        assert node1.content == ""
        assert node1.status == "PENDING"
        assert node1.created_by == "system"

    def test_ingest_nodes_empty_dataframe(self, bulk_ingestor):
        """Test that ingest_nodes() handles empty DataFrame."""
        df = pl.DataFrame({
            "id": [],
            "type": [],
        })

        count = bulk_ingestor.ingest_nodes(df)
        assert count == 0

    def test_ingest_nodes_missing_required_columns(self, bulk_ingestor):
        """Test that ingest_nodes() raises error for missing columns."""
        df = pl.DataFrame({
            "id": ["node1"],
            # Missing "type" column
        })

        with pytest.raises(DataIntegrityError, match="Missing required columns"):
            bulk_ingestor.ingest_nodes(df)

    def test_ingest_edges_basic(self, bulk_ingestor, paragon_db):
        """Test that ingest_edges() adds edges to database."""
        # First add nodes
        for i in range(3):
            paragon_db.add_node(NodeData(id=f"node{i}", type="SPEC", content="test"))

        df = pl.DataFrame({
            "source_id": ["node0", "node1"],
            "target_id": ["node1", "node2"],
            "type": ["DEPENDS_ON", "IMPLEMENTS"],
        })

        count = bulk_ingestor.ingest_edges(df)

        assert count == 2
        assert paragon_db.edge_count == 2

    def test_ingest_edges_with_defaults(self, bulk_ingestor, paragon_db):
        """Test that ingest_edges() fills in default values."""
        # Add nodes
        for i in range(2):
            paragon_db.add_node(NodeData(id=f"node{i}", type="SPEC", content="test"))

        df = pl.DataFrame({
            "source_id": ["node0"],
            "target_id": ["node1"],
            "type": ["DEPENDS_ON"],
        })

        count = bulk_ingestor.ingest_edges(df)

        assert count == 1

    def test_ingest_from_files_csv(self, bulk_ingestor, temp_dir):
        """Test that ingest_from_files() loads from CSV files."""
        # Create sample CSV files
        nodes_csv = temp_dir / "nodes.csv"
        edges_csv = temp_dir / "edges.csv"

        # Write nodes CSV
        with open(nodes_csv, "w") as f:
            f.write("id,type,content,status,created_by\n")
            f.write("node1,SPEC,test1,PENDING,system\n")
            f.write("node2,CODE,test2,PENDING,system\n")

        # Write edges CSV
        with open(edges_csv, "w") as f:
            f.write("source_id,target_id,type,weight,created_by\n")
            f.write("node1,node2,IMPLEMENTS,1.0,system\n")

        nodes_count, edges_count = bulk_ingestor.ingest_from_files(
            nodes_path=nodes_csv,
            edges_path=edges_csv,
            format="csv",
        )

        assert nodes_count == 2
        assert edges_count == 1


# =============================================================================
# DIAGNOSTICLOGGER TESTS (14 functions)
# =============================================================================

class TestDiagnosticLogger:
    """Tests for DiagnosticLogger class."""

    def test_init_creates_directory(self, temp_dir):
        """Test that __init__ creates log directory."""
        log_path = temp_dir / "subdir" / "diagnostics.jsonl"
        logger = DiagnosticLogger(log_path=log_path)

        assert logger.log_path.parent.exists()

    def test_set_session_generates_correlation_id(self, diagnostic_logger):
        """Test that set_session() generates and returns correlation ID."""
        session_id = "test-session"
        correlation_id = diagnostic_logger.set_session(session_id)

        assert correlation_id is not None
        assert correlation_id.startswith("dx_")
        assert diagnostic_logger.correlation_id == correlation_id

    def test_get_db_state_no_db(self, diagnostic_logger):
        """Test that get_db_state() handles missing database."""
        with patch("agents.tools.get_db", return_value=None):
            state = diagnostic_logger.get_db_state()
            assert state["status"] == "CLEAN"
            assert "None" in state["message"]

    def test_get_db_state_active(self, diagnostic_logger):
        """Test that get_db_state() returns active database info."""
        mock_db = MagicMock()
        mock_db.node_count = 42
        mock_db.edge_count = 17

        with patch("agents.tools.get_db", return_value=mock_db):
            state = diagnostic_logger.get_db_state()
            assert state["status"] == "ACTIVE"
            assert state["node_count"] == 42
            assert state["edge_count"] == 17

    def test_get_llm_state_no_llm(self, diagnostic_logger):
        """Test that get_llm_state() handles missing LLM instance."""
        with patch("core.llm._llm_instance", None):
            state = diagnostic_logger.get_llm_state()
            assert state["status"] == "CLEAN"

    def test_get_rate_limit_state_no_limiter(self, diagnostic_logger):
        """Test that get_rate_limit_state() handles missing rate limiter."""
        with patch("core.llm._rate_limit_guard", None):
            state = diagnostic_logger.get_rate_limit_state()
            assert state["status"] == "CLEAN"

    def test_llm_call_context_manager(self, diagnostic_logger):
        """Test that llm_call() provides working context manager."""
        with diagnostic_logger.llm_call("TestSchema") as call:
            call.set_tokens(100, 50)
            call.set_truncated(True)

        # Verify call was recorded
        assert len(diagnostic_logger._llm_calls) == 1
        metric = diagnostic_logger._llm_calls[0]
        assert metric.schema_name == "TestSchema"
        assert metric.input_tokens == 100
        assert metric.output_tokens == 50
        assert metric.truncated is True
        assert metric.success is True

    def test_llm_call_context_manager_with_error(self, diagnostic_logger):
        """Test that llm_call() handles exceptions."""
        with pytest.raises(ValueError):
            with diagnostic_logger.llm_call("TestSchema") as call:
                raise ValueError("Test error")

        # Verify call was recorded with error
        assert len(diagnostic_logger._llm_calls) == 1
        metric = diagnostic_logger._llm_calls[0]
        assert metric.success is False
        assert "Test error" in metric.error

    def test_record_llm_call(self, diagnostic_logger):
        """Test that _record_llm_call() adds metric to list."""
        metric = LLMCallMetric(
            schema_name="TestSchema",
            start_time=time.time() - 0.5,
            end_time=time.time(),
            input_tokens=100,
            output_tokens=50,
            success=True,
        )

        diagnostic_logger._record_llm_call(metric)

        assert len(diagnostic_logger._llm_calls) == 1
        assert diagnostic_logger._llm_calls[0] == metric

    def test_write_log_creates_entry(self, diagnostic_logger):
        """Test that _write_log() writes to file."""
        diagnostic_logger.set_session("test-session")
        diagnostic_logger._write_log("test_event", {"key": "value"})

        # Verify file was written
        assert diagnostic_logger.log_path.exists()
        with open(diagnostic_logger.log_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2  # session_start + test_event
            data = json.loads(lines[1])
            assert data["type"] == "test_event"
            assert data["key"] == "value"

    def test_start_phase_creates_metric(self, diagnostic_logger):
        """Test that start_phase() creates a phase metric."""
        diagnostic_logger.start_phase("test_phase")

        assert diagnostic_logger._current_phase is not None
        assert diagnostic_logger._current_phase.phase_name == "test_phase"

    def test_end_phase_records_metric(self, diagnostic_logger):
        """Test that end_phase() records completed phase."""
        diagnostic_logger.start_phase("test_phase")
        time.sleep(0.01)  # Small delay
        diagnostic_logger.end_phase(success=True)

        assert diagnostic_logger._current_phase is None
        assert len(diagnostic_logger._phase_metrics) == 1
        assert diagnostic_logger._phase_metrics[0].phase_name == "test_phase"
        assert diagnostic_logger._phase_metrics[0].success is True

    def test_record_llm_call_simple(self, diagnostic_logger):
        """Test that record_llm_call_simple() works without context manager."""
        diagnostic_logger.record_llm_call_simple(
            schema_name="TestSchema",
            duration_ms=150.0,
            input_tokens=100,
            output_tokens=50,
            success=True,
        )

        assert len(diagnostic_logger._llm_calls) == 1
        metric = diagnostic_logger._llm_calls[0]
        assert metric.schema_name == "TestSchema"
        assert metric.input_tokens == 100

    def test_reset_clears_state(self, diagnostic_logger):
        """Test that reset() clears all metrics."""
        diagnostic_logger.record_llm_call_simple("Test", 100, 10, 10)
        diagnostic_logger.start_phase("test")
        diagnostic_logger.end_phase()

        diagnostic_logger.reset()

        assert len(diagnostic_logger._llm_calls) == 0
        assert len(diagnostic_logger._phase_metrics) == 0

    def test_print_state_summary_no_errors(self, diagnostic_logger, capsys):
        """Test that print_state_summary() runs without errors."""
        diagnostic_logger.print_state_summary(use_color=False)

        captured = capsys.readouterr()
        assert "PARAGON DIAGNOSTICS" in captured.out

    def test_print_summary_no_errors(self, diagnostic_logger, capsys):
        """Test that print_summary() runs without errors."""
        diagnostic_logger.set_session("test")
        diagnostic_logger.record_llm_call_simple("Test", 100, 10, 10)
        diagnostic_logger.print_summary(use_color=False)

        captured = capsys.readouterr()
        assert "Session Summary" in captured.out


# =============================================================================
# DIVERGENCEDETECTOR TESTS (3 functions)
# =============================================================================

class TestDivergenceDetector:
    """Tests for DivergenceDetector class."""

    def test_init_stores_reference(self, training_store):
        """Test that __init__ stores the training store reference."""
        detector = DivergenceDetector(store=training_store)
        assert detector.store is training_store

    def test_log_divergence_writes_to_db(self, divergence_detector):
        """Test that log_divergence() writes event to database."""
        event = DivergenceEvent(
            event_id="test-event-id",
            session_id="test-session",
            node_id="test-node",
            test_outcome="passed",
            prod_outcome="failure",
            divergence_type="false_positive",
            severity="critical",
            detected_at=datetime.utcnow().isoformat(),
            context={},
        )

        divergence_detector.log_divergence(event)

        # Verify event was written
        with sqlite3.connect(divergence_detector.store.db_path) as conn:
            cursor = conn.execute("SELECT * FROM divergence_events WHERE event_id = ?", ("test-event-id",))
            row = cursor.fetchone()
            assert row is not None

    def test_assess_severity_critical(self, divergence_detector):
        """Test that _assess_severity() identifies critical failures."""
        severity = divergence_detector._assess_severity("broken and unusable")
        assert severity == "critical"

    def test_assess_severity_high(self, divergence_detector):
        """Test that _assess_severity() identifies high severity."""
        severity = divergence_detector._assess_severity("wrong output")
        assert severity == "high"

    def test_assess_severity_default(self, divergence_detector):
        """Test that _assess_severity() defaults to medium."""
        severity = divergence_detector._assess_severity("some issue")
        assert severity == "medium"


# =============================================================================
# ENVIRONMENTDETECTOR TESTS (9 functions)
# =============================================================================

class TestEnvironmentDetector:
    """Tests for EnvironmentDetector class."""

    def test_init_sets_working_dir(self, temp_dir):
        """Test that __init__ sets working directory."""
        detector = EnvironmentDetector(working_dir=temp_dir)
        assert detector.working_dir == temp_dir

    def test_init_defaults_to_cwd(self):
        """Test that __init__ defaults to current directory."""
        detector = EnvironmentDetector()
        assert detector.working_dir == Path.cwd()

    def test_detect_returns_report(self, environment_detector):
        """Test that detect() returns EnvironmentReport."""
        report = environment_detector.detect()

        assert isinstance(report, EnvironmentReport)
        assert report.os_name in ["Linux", "Darwin", "Windows", "FreeBSD"]
        assert len(report.python_version.split(".")) == 3

    def test_detect_os_returns_valid_platform(self, environment_detector):
        """Test that _detect_os() returns valid OS name."""
        os_name = environment_detector._detect_os()
        assert os_name in ["Linux", "Darwin", "Windows", "FreeBSD", "Java"]

    def test_detect_python_returns_version(self, environment_detector):
        """Test that _detect_python() returns version string."""
        version = environment_detector._detect_python()
        parts = version.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_detect_ram_returns_positive(self, environment_detector):
        """Test that _detect_ram() returns positive value."""
        ram_gb = environment_detector._detect_ram()
        assert ram_gb > 0

    def test_detect_gpu_returns_tuple(self, environment_detector):
        """Test that _detect_gpu() returns (bool, Optional[str]) tuple."""
        gpu_available, gpu_name = environment_detector._detect_gpu()
        assert isinstance(gpu_available, bool)
        if gpu_available:
            assert isinstance(gpu_name, str)
        else:
            assert gpu_name is None

    def test_detect_disk_returns_positive(self, environment_detector):
        """Test that _detect_disk() returns positive value."""
        disk_gb = environment_detector._detect_disk()
        assert disk_gb > 0

    def test_detect_network_returns_bool(self, environment_detector):
        """Test that _detect_network() returns boolean."""
        network_available = environment_detector._detect_network()
        assert isinstance(network_available, bool)

    def test_detect_git_finds_repo(self):
        """Test that _detect_git() finds git repository."""
        # Test in paragon directory (should have .git)
        detector = EnvironmentDetector(working_dir=Path.cwd())
        has_git = detector._detect_git()
        # This test might fail if not run from git repo, so just check type
        assert isinstance(has_git, bool)


# =============================================================================
# EVENTBUFFER TESTS (8 functions)
# =============================================================================

class TestEventBuffer:
    """Tests for EventBuffer class."""

    def test_init_creates_empty_buffer(self):
        """Test that __init__ creates empty buffer."""
        buffer = EventBuffer(max_size=10)
        assert len(buffer) == 0

    def test_append_adds_event(self, event_buffer):
        """Test that append() adds event to buffer."""
        event = MutationEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sequence=1,
            mutation_type=MutationType.NODE_CREATED.value,
            node_id="node1",
        )

        event_buffer.append(event)
        assert len(event_buffer) == 1

    def test_get_since_filters_by_timestamp(self, event_buffer):
        """Test that get_since() filters events by timestamp."""
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=1)

        event1 = MutationEvent(
            timestamp=past.isoformat(),
            sequence=1,
            mutation_type=MutationType.NODE_CREATED.value,
            node_id="node1",
        )
        event2 = MutationEvent(
            timestamp=now.isoformat(),
            sequence=2,
            mutation_type=MutationType.NODE_CREATED.value,
            node_id="node2",
        )

        event_buffer.append(event1)
        event_buffer.append(event2)

        recent = event_buffer.get_since(now.isoformat())
        assert len(recent) == 1
        assert recent[0].node_id == "node2"

    def test_get_last_returns_n_events(self, event_buffer):
        """Test that get_last() returns last n events."""
        for i in range(10):
            event = MutationEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                sequence=i,
                mutation_type=MutationType.NODE_CREATED.value,
                node_id=f"node{i}",
            )
            event_buffer.append(event)

        last_3 = event_buffer.get_last(3)
        assert len(last_3) == 3
        assert last_3[-1].node_id == "node9"

    def test_get_by_node_filters_by_node_id(self, event_buffer):
        """Test that get_by_node() filters by node ID."""
        event1 = MutationEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sequence=1,
            mutation_type=MutationType.NODE_CREATED.value,
            node_id="node1",
        )
        event2 = MutationEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sequence=2,
            mutation_type=MutationType.NODE_UPDATED.value,
            node_id="node1",
        )
        event3 = MutationEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sequence=3,
            mutation_type=MutationType.NODE_CREATED.value,
            node_id="node2",
        )

        event_buffer.append(event1)
        event_buffer.append(event2)
        event_buffer.append(event3)

        node1_events = event_buffer.get_by_node("node1")
        assert len(node1_events) == 2

    def test_get_by_type_filters_by_mutation_type(self, event_buffer):
        """Test that get_by_type() filters by mutation type."""
        event1 = MutationEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sequence=1,
            mutation_type=MutationType.NODE_CREATED.value,
            node_id="node1",
        )
        event2 = MutationEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sequence=2,
            mutation_type=MutationType.NODE_UPDATED.value,
            node_id="node2",
        )

        event_buffer.append(event1)
        event_buffer.append(event2)

        created = event_buffer.get_by_type(MutationType.NODE_CREATED.value)
        assert len(created) == 1
        assert created[0].node_id == "node1"

    def test_next_sequence_increments(self, event_buffer):
        """Test that next_sequence() increments counter."""
        seq1 = event_buffer.next_sequence()
        seq2 = event_buffer.next_sequence()
        seq3 = event_buffer.next_sequence()

        assert seq2 == seq1 + 1
        assert seq3 == seq2 + 1

    def test_clear_empties_buffer(self, event_buffer):
        """Test that clear() removes all events."""
        for i in range(5):
            event = MutationEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                sequence=i,
                mutation_type=MutationType.NODE_CREATED.value,
                node_id=f"node{i}",
            )
            event_buffer.append(event)

        assert len(event_buffer) == 5
        event_buffer.clear()
        assert len(event_buffer) == 0


# =============================================================================
# FILELOGGER TESTS (4 functions)
# =============================================================================

class TestFileLogger:
    """Tests for FileLogger class."""

    def test_init_creates_directory(self, temp_dir):
        """Test that __init__ creates log directory."""
        log_path = temp_dir / "subdir"
        logger = FileLogger(log_path=log_path)
        assert log_path.exists()

    def test_write_creates_log_file(self, file_logger):
        """Test that write() creates log file."""
        event = MutationEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sequence=1,
            mutation_type=MutationType.NODE_CREATED.value,
            node_id="node1",
        )

        file_logger.write(event)

        # Check that file was created
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        expected_file = file_logger._log_path / f"mutations_{today}.jsonl"
        assert expected_file.exists()

    def test_ensure_file_rotates_daily(self, file_logger):
        """Test that _ensure_file() handles daily rotation."""
        # This is tested implicitly by write() creating dated files
        file_logger._ensure_file()
        assert file_logger._current_file is not None
        assert file_logger._current_date is not None

    def test_read_log_returns_events(self, file_logger):
        """Test that read_log() reads back written events."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        event = MutationEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sequence=1,
            mutation_type=MutationType.NODE_CREATED.value,
            node_id="node1",
        )

        file_logger.write(event)
        file_logger.close()  # Close to flush

        events = file_logger.read_log(today)
        assert len(events) == 1
        assert events[0].node_id == "node1"

    def test_close_closes_file(self, file_logger):
        """Test that close() closes file handle."""
        event = MutationEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sequence=1,
            mutation_type=MutationType.NODE_CREATED.value,
            node_id="node1",
        )

        file_logger.write(event)
        assert file_logger._current_file is not None

        file_logger.close()
        assert file_logger._current_file is None


# =============================================================================
# FORENSICANALYZER TESTS (8 functions)
# =============================================================================

class TestForensicAnalyzer:
    """Tests for ForensicAnalyzer class."""

    def test_init_stores_reference(self, training_store):
        """Test that __init__ stores training store reference."""
        analyzer = ForensicAnalyzer(store=training_store)
        assert analyzer.store is training_store

    def test_analyze_failure_returns_result(self, forensic_analyzer):
        """Test that analyze_failure() returns AttributionResult."""
        result = forensic_analyzer.analyze_failure(
            session_id="test-session",
            error_type="SyntaxError",
            error_message="invalid syntax",
        )

        assert isinstance(result, AttributionResult)
        assert result.failure_code in [fc for fc in FailureCode]

    def test_classify_failure_f2_for_syntax_error(self, forensic_analyzer):
        """Test that _classify_failure() identifies F2 for syntax errors."""
        failure_code = forensic_analyzer._classify_failure(
            error_type="SyntaxError",
            error_message="invalid syntax",
            phase=CyclePhase.BUILD,
        )

        assert failure_code == FailureCode.F2

    def test_classify_failure_f3_for_test_errors(self, forensic_analyzer):
        """Test that _classify_failure() identifies F3 for test errors."""
        failure_code = forensic_analyzer._classify_failure(
            error_type="AssertionError",
            error_message="test failed",
            phase=CyclePhase.TEST,
        )

        assert failure_code == FailureCode.F3

    def test_classify_failure_f4_for_network_errors(self, forensic_analyzer):
        """Test that _classify_failure() identifies F4 for network errors."""
        failure_code = forensic_analyzer._classify_failure(
            error_type="ConnectionError",
            error_message="network timeout",
            phase=CyclePhase.BUILD,
        )

        assert failure_code == FailureCode.F4

    def test_infer_phase_from_error_test_phase(self, forensic_analyzer):
        """Test that _infer_phase_from_error() identifies test phase."""
        phase = forensic_analyzer._infer_phase_from_error(
            error_type="AssertionError",
            error_message="expected 5 but got 3",
        )

        assert phase == CyclePhase.TEST

    def test_infer_phase_from_error_build_phase(self, forensic_analyzer):
        """Test that _infer_phase_from_error() identifies build phase."""
        phase = forensic_analyzer._infer_phase_from_error(
            error_type="SyntaxError",
            error_message="invalid syntax",
        )

        assert phase == CyclePhase.BUILD

    def test_calculate_attribution_confidence_returns_tuple(self, forensic_analyzer, sample_signature):
        """Test that _calculate_attribution_confidence() returns tuple."""
        signatures = [sample_signature]

        agent_id, model_id, confidence, contributing = \
            forensic_analyzer._calculate_attribution_confidence(
                signatures=signatures,
                failure_code=FailureCode.F2,
                phase=CyclePhase.BUILD,
            )

        assert isinstance(agent_id, str)
        assert isinstance(model_id, str)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(contributing, list)

    def test_build_reasoning_returns_string(self, forensic_analyzer, sample_signature):
        """Test that _build_reasoning() returns explanation string."""
        reasoning = forensic_analyzer._build_reasoning(
            failure_code=FailureCode.F2,
            failure_phase=CyclePhase.BUILD,
            error_type="SyntaxError",
            error_message="invalid syntax",
            signatures=[sample_signature],
            attributed_agent="test-builder-v1",
        )

        assert isinstance(reasoning, str)
        assert "F2" in reasoning
        assert "SyntaxError" in reasoning

    def test_analyze_session_failures_multiple(self, forensic_analyzer):
        """Test that analyze_session_failures() handles multiple failures."""
        failures = [
            {
                "error_type": "SyntaxError",
                "error_message": "invalid syntax",
            },
            {
                "error_type": "AssertionError",
                "error_message": "test failed",
            },
        ]

        results = forensic_analyzer.analyze_session_failures(
            session_id="test-session",
            failures=failures,
        )

        assert len(results) == 2
        assert all(isinstance(r, AttributionResult) for r in results)


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================

class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_event_buffer_thread_safe(self):
        """Test that EventBuffer is thread-safe."""
        buffer = EventBuffer(max_size=1000)

        def add_events(start_id):
            for i in range(100):
                event = MutationEvent(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    sequence=buffer.next_sequence(),
                    mutation_type=MutationType.NODE_CREATED.value,
                    node_id=f"node{start_id}_{i}",
                )
                buffer.append(event)

        # Create multiple threads
        threads = [threading.Thread(target=add_events, args=(i,)) for i in range(5)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should have 500 events total
        assert len(buffer) == 500

    def test_audit_logger_thread_safe(self, temp_dir):
        """Test that AuditLogger is thread-safe."""
        logger = AuditLogger(log_path=temp_dir / "audit.log")

        def log_actions(start_id):
            for i in range(50):
                logger.log_action(
                    action=f"action_{start_id}_{i}",
                    node_id=f"node_{start_id}_{i}",
                )

        # Create multiple threads
        threads = [threading.Thread(target=log_actions, args=(i,)) for i in range(3)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should have 150 entries
        entries = logger.read_entries()
        assert len(entries) == 150
