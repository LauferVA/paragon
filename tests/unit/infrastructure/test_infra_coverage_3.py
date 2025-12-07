"""
Unit tests for final third of uncovered infrastructure functions.

Coverage includes:
- RerunIntegration: __init__, close, initialize, log_event
- RerunLogger: __init__, _generate_session_id, _initialize, _next_sequence,
  log_code_diff, log_edge, log_graph_snapshot, log_node, log_thought, save
- TrainingStore: __init__, _init_schema, clear_all, record_attribution,
  record_session_outcome, record_signature_chain
- TeleologyChain.to_commit_message
- LLMCallContext: set_error, set_retry_count, set_tokens, set_truncated
- LLMCallMetric.to_dict
- Standalone functions: configure_logger, create_agent_config, create_logger,
  create_resource_policy, create_sample_edge_csv, create_sample_node_csv,
  detect_environment, generate_correlation_id, get_agent_config, get_all_config,
  get_audit_logger, get_collector, get_config, get_diagnostics, get_git_sync,
  get_logger, get_node_color, get_resource_policy, get_traceability_report,
  hex_to_rgb, initialize_config, load_git_config, load_git_config_from_graph,
  load_git_config_from_toml, load_graph_from_csv, load_graph_from_parquet,
  load_observability_config, load_observability_config_from_graph,
  load_observability_config_from_toml, load_toml_config, log_audit,
  log_edge_created, log_node_created, log_status_changed, now_utc,
  on_transaction_complete, print_session_summary, print_state_summary,
  record_node_complete, record_node_created, record_node_start,
  reset_diagnostics, update_config
"""
import pytest
import tempfile
import shutil
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock, mock_open
import uuid

# Import modules under test
from infrastructure.rerun_logger import (
    RerunLogger,
    create_logger,
    hex_to_rgb,
    get_node_color,
    load_observability_config,
    load_observability_config_from_graph,
    load_observability_config_from_toml,
)
from infrastructure.training_store import TrainingStore
from infrastructure.git_sync import (
    TeleologyChain,
    GitSync,
    get_git_sync,
    on_transaction_complete,
    load_git_config,
    load_git_config_from_graph,
    load_git_config_from_toml,
)
from infrastructure.logger import (
    RerunIntegration,
    configure_logger,
    LoggerConfig,
    get_logger,
    get_audit_logger,
    log_node_created,
    log_status_changed,
    log_edge_created,
    log_audit,
)
from infrastructure.diagnostics import (
    LLMCallMetric,
    LLMCallContext,
    DiagnosticLogger,
    generate_correlation_id,
    get_diagnostics,
    reset_diagnostics,
    print_state_summary,
    print_session_summary,
)
from infrastructure.metrics import (
    now_utc,
    get_collector,
    record_node_created as metrics_record_node_created,
    record_node_start,
    record_node_complete,
    get_traceability_report,
)
from infrastructure.config_graph import (
    load_toml_config,
    initialize_config,
    get_config,
    get_all_config,
    update_config,
    create_agent_config,
    get_agent_config,
    create_resource_policy,
    get_resource_policy,
)
from infrastructure.data_loader import (
    load_graph_from_csv,
    load_graph_from_parquet,
    create_sample_node_csv,
    create_sample_edge_csv,
)
from infrastructure.environment import detect_environment

from core.schemas import NodeData, generate_id
from core.ontology import NodeType, NodeStatus
from agents.schemas import (
    AgentSignature,
    SignatureChain,
    CyclePhase,
    FailureCode,
    NodeOutcome,
    SignatureAction,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def temp_db(temp_dir):
    """Create a temporary database path."""
    return temp_dir / "test.db"


@pytest.fixture
def mock_db():
    """Create a mock ParagonDB instance."""
    db = Mock()
    db.node_count = 10
    db.edge_count = 5
    db.find_nodes = Mock(return_value=[])
    db.get_node = Mock()
    db.get_outgoing_edges = Mock(return_value=[])
    db.add_node = Mock()
    db.add_edge = Mock()
    db.update_node = Mock()
    db.add_nodes_batch = Mock()
    db.add_edges_batch = Mock()
    return db


@pytest.fixture
def sample_signature():
    """Create a sample agent signature."""
    return AgentSignature(
        agent_id="builder_v1",
        model_id="claude-sonnet-4-5",
        phase=CyclePhase.BUILD,
        action=SignatureAction.CREATED,
        temperature=0.7,
        context_constraints={"max_tokens": 4096},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def sample_node_data():
    """Create sample node data."""
    return NodeData(
        id=generate_id(),
        type=NodeType.CODE.value,
        content="def foo(): pass",
        status=NodeStatus.VERIFIED.value,
        created_by="test_agent",
    )


# =============================================================================
# RERUN LOGGER TESTS
# =============================================================================

class TestRerunLogger:
    """Tests for RerunLogger class."""

    def test_init_with_session_id(self):
        """Test RerunLogger initialization with session ID."""
        logger = RerunLogger(session_id="test-session")
        assert logger.session_id == "test-session"
        assert logger.recording_path is None or isinstance(logger.recording_path, Path)
        assert logger._sequence == 0
        assert logger._node_positions == {}

    def test_init_without_session_id(self):
        """Test RerunLogger initialization without session ID generates one."""
        logger = RerunLogger()
        assert logger.session_id is not None
        assert len(logger.session_id) == 8  # UUID hex[:8]

    def test_generate_session_id(self):
        """Test session ID generation."""
        logger = RerunLogger()
        session_id = logger._generate_session_id()
        assert isinstance(session_id, str)
        assert len(session_id) == 8

    def test_next_sequence(self):
        """Test sequence counter increment."""
        logger = RerunLogger(session_id="test")
        assert logger._next_sequence() == 1
        assert logger._next_sequence() == 2
        assert logger._next_sequence() == 3

    @patch('infrastructure.rerun_logger.RERUN_AVAILABLE', False)
    def test_initialize_without_rerun(self):
        """Test initialization when Rerun SDK is not available."""
        logger = RerunLogger(session_id="test")
        result = logger._initialize()
        assert result is False
        assert logger._initialized is False

    def test_log_node_not_initialized(self):
        """Test log_node when logger is not initialized."""
        logger = RerunLogger(session_id="test")
        logger._initialized = False
        # Should not raise, just return early
        logger.log_node("node_1", "CODE", "def foo(): pass")

    def test_log_edge_not_initialized(self):
        """Test log_edge when logger is not initialized."""
        logger = RerunLogger(session_id="test")
        logger._initialized = False
        logger.log_edge("node_1", "node_2", "DEPENDS_ON")

    def test_log_code_diff_not_initialized(self):
        """Test log_code_diff when logger is not initialized."""
        logger = RerunLogger(session_id="test")
        logger._initialized = False
        logger.log_code_diff("test.py", "old", "new")

    def test_log_thought_not_initialized(self):
        """Test log_thought when logger is not initialized."""
        logger = RerunLogger(session_id="test")
        logger._initialized = False
        logger.log_thought("agent_1", "thinking...")

    def test_log_graph_snapshot_not_initialized(self):
        """Test log_graph_snapshot when logger is not initialized."""
        logger = RerunLogger(session_id="test")
        logger._initialized = False
        nodes = [{"id": "n1", "type": "CODE", "content": "test"}]
        edges = [{"source_id": "n1", "target_id": "n2", "type": "DEPENDS_ON"}]
        logger.log_graph_snapshot(nodes, edges)

    def test_save_not_initialized(self):
        """Test save when logger is not initialized."""
        logger = RerunLogger(session_id="test")
        logger._initialized = False
        result = logger.save()
        assert result is None

    def test_context_manager(self):
        """Test RerunLogger as context manager."""
        with RerunLogger(session_id="test") as logger:
            assert isinstance(logger, RerunLogger)


class TestRerunIntegration:
    """Tests for RerunIntegration class."""

    def test_init(self):
        """Test RerunIntegration initialization."""
        integration = RerunIntegration(app_id="test-app")
        assert integration._app_id == "test-app"
        assert integration._rr is None
        assert integration._initialized is False

    @patch('infrastructure.logger.rr', None)
    def test_initialize_without_rerun_sdk(self, capsys):
        """Test initialization without Rerun SDK installed."""
        integration = RerunIntegration()
        result = integration.initialize()
        assert result is False
        assert integration._initialized is False

    def test_log_event_not_initialized(self):
        """Test log_event when not initialized."""
        integration = RerunIntegration()
        from viz.core import MutationEvent, MutationType
        event = MutationEvent(
            timestamp="2024-01-01T00:00:00Z",
            sequence=1,
            mutation_type=MutationType.NODE_CREATED.value,
            node_id="node_1",
        )
        # Should not raise, just return early
        integration.log_event(event)

    def test_close_not_initialized(self):
        """Test close when not initialized."""
        integration = RerunIntegration()
        integration.close()  # Should not raise


# =============================================================================
# TRAINING STORE TESTS
# =============================================================================

class TestTrainingStore:
    """Tests for TrainingStore class."""

    def test_init_creates_schema(self, temp_db):
        """Test TrainingStore initialization creates schema."""
        store = TrainingStore(db_path=temp_db)
        assert temp_db.exists()

        import sqlite3
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "node_attributions" in tables
        assert "session_outcomes" in tables
        assert "signature_chains" in tables
        assert "divergence_events" in tables

    def test_record_attribution(self, temp_db, sample_signature):
        """Test recording a node attribution."""
        store = TrainingStore(db_path=temp_db)
        attr_id = store.record_attribution(
            session_id="session_1",
            signature=sample_signature,
            node_id="node_1",
            state_id="state_1",
        )
        assert isinstance(attr_id, str)

        # Verify it was stored
        attrs = store.get_attributions_by_node("node_1")
        assert len(attrs) == 1
        assert attrs[0]["agent_id"] == "builder_v1"

    def test_record_signature_chain(self, temp_db, sample_signature):
        """Test recording a signature chain."""
        store = TrainingStore(db_path=temp_db)
        chain = SignatureChain(
            node_id="node_1",
            state_id="state_1",
            signatures=[sample_signature],
            is_replacement=False,
            replaced_node_id=None,
        )
        chain_id = store.record_signature_chain(chain)
        assert isinstance(chain_id, str)

        # Verify retrieval
        retrieved = store.get_signature_chain("node_1")
        assert retrieved is not None
        assert retrieved.node_id == "node_1"
        assert len(retrieved.signatures) == 1

    def test_record_session_outcome(self, temp_db):
        """Test recording a session outcome."""
        store = TrainingStore(db_path=temp_db)
        store.record_session_outcome(
            session_id="session_1",
            outcome=NodeOutcome.VERIFIED_SUCCESS,
            failure_code=None,
            failure_phase=None,
            stats={"total_nodes": 5, "total_tokens": 1000},
        )

        outcome = store.get_session_outcome("session_1")
        assert outcome is not None
        assert outcome["outcome"] == NodeOutcome.VERIFIED_SUCCESS.value
        assert outcome["total_nodes"] == 5

    def test_clear_all(self, temp_db, sample_signature):
        """Test clearing all data."""
        store = TrainingStore(db_path=temp_db)

        # Add some data
        store.record_attribution("s1", sample_signature, "n1", "st1")
        store.record_session_outcome("s1", NodeOutcome.VERIFIED_SUCCESS)

        # Clear
        store.clear_all()

        # Verify empty
        assert store.get_session_count() == 0
        assert store.get_attribution_count() == 0


# =============================================================================
# TELEOLOGY CHAIN TESTS
# =============================================================================

class TestTeleologyChain:
    """Tests for TeleologyChain."""

    def test_to_commit_message_with_code(self, sample_node_data):
        """Test commit message generation with code node."""
        chain = TeleologyChain(code_node=sample_node_data)
        message = chain.to_commit_message()
        assert message.startswith("feat:")
        assert "def foo(): pass" in message

    def test_to_commit_message_with_req(self):
        """Test commit message generation with REQ node."""
        req = NodeData(
            id=generate_id(),
            type=NodeType.REQ.value,
            content="Build authentication system",
            status=NodeStatus.PENDING.value,
            created_by="user",
        )
        chain = TeleologyChain(req_node=req)
        message = chain.to_commit_message()
        assert message.startswith("chore:")
        assert "REQ-" in message

    def test_to_commit_message_with_spec(self):
        """Test commit message generation with SPEC node."""
        spec = NodeData(
            id=generate_id(),
            type=NodeType.SPEC.value,
            content="API specification for auth",
            status=NodeStatus.PENDING.value,
            created_by="user",
        )
        chain = TeleologyChain(spec_node=spec)
        message = chain.to_commit_message()
        assert message.startswith("docs:")

    def test_to_commit_message_empty_chain(self):
        """Test commit message generation with empty chain."""
        chain = TeleologyChain()
        message = chain.to_commit_message()
        assert message.startswith("chore:")
        assert "Update graph" in message


# =============================================================================
# LLM CALL METRICS TESTS
# =============================================================================

class TestLLMCallMetric:
    """Tests for LLMCallMetric."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metric = LLMCallMetric(
            schema_name="TestSchema",
            start_time=time.time(),
            end_time=time.time() + 1.5,
            input_tokens=100,
            output_tokens=50,
            success=True,
            retry_count=0,
            truncated=False,
        )
        data = metric.to_dict()
        assert data["schema"] == "TestSchema"
        assert data["input_tokens"] == 100
        assert data["output_tokens"] == 50
        assert data["success"] is True
        assert "duration_ms" in data


class TestLLMCallContext:
    """Tests for LLMCallContext."""

    def test_set_tokens(self):
        """Test setting token counts."""
        metric = LLMCallMetric("Test", time.time())
        diag = DiagnosticLogger()
        ctx = LLMCallContext(metric, diag)
        ctx.set_tokens(100, 50)
        assert metric.input_tokens == 100
        assert metric.output_tokens == 50

    def test_set_error(self):
        """Test setting error."""
        metric = LLMCallMetric("Test", time.time())
        diag = DiagnosticLogger()
        ctx = LLMCallContext(metric, diag)
        ctx.set_error("Test error")
        assert metric.error == "Test error"
        assert metric.success is False

    def test_set_truncated(self):
        """Test setting truncated flag."""
        metric = LLMCallMetric("Test", time.time())
        diag = DiagnosticLogger()
        ctx = LLMCallContext(metric, diag)
        ctx.set_truncated(True)
        assert metric.truncated is True

    def test_set_retry_count(self):
        """Test setting retry count."""
        metric = LLMCallMetric("Test", time.time())
        diag = DiagnosticLogger()
        ctx = LLMCallContext(metric, diag)
        ctx.set_retry_count(3)
        assert metric.retry_count == 3

    def test_context_manager_success(self):
        """Test context manager with successful execution."""
        diag = DiagnosticLogger()
        with diag.llm_call("TestSchema") as call:
            call.set_tokens(100, 50)
        assert len(diag._llm_calls) == 1
        assert diag._llm_calls[0].success is True

    def test_context_manager_failure(self):
        """Test context manager with exception."""
        diag = DiagnosticLogger()
        try:
            with diag.llm_call("TestSchema") as call:
                raise ValueError("Test error")
        except ValueError:
            pass
        assert len(diag._llm_calls) == 1
        assert diag._llm_calls[0].success is False


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Tests for standalone utility functions."""

    def test_now_utc(self):
        """Test UTC timestamp generation."""
        timestamp = now_utc()
        assert isinstance(timestamp, str)
        # Should be ISO8601 format
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        corr_id = generate_correlation_id()
        assert isinstance(corr_id, str)
        assert corr_id.startswith("dx_")

    def test_hex_to_rgb(self):
        """Test hex to RGB conversion."""
        rgb = hex_to_rgb("#FF0000")
        assert rgb == (255, 0, 0)

        rgb = hex_to_rgb("00FF00")
        assert rgb == (0, 255, 0)

    def test_get_node_color(self):
        """Test getting node color by type."""
        color = get_node_color("CODE")
        assert isinstance(color, tuple)
        assert len(color) == 3

        # Test unknown type
        color = get_node_color("UNKNOWN")
        assert isinstance(color, tuple)

    def test_detect_environment(self):
        """Test environment detection."""
        env = detect_environment()
        assert env in ["development", "testing", "production"]


# =============================================================================
# LOGGER CONVENIENCE FUNCTIONS
# =============================================================================

class TestLoggerConvenience:
    """Tests for logger convenience functions."""

    def test_create_logger(self):
        """Test create_logger factory function."""
        logger = create_logger(session_id="test")
        assert isinstance(logger, RerunLogger)
        assert logger.session_id == "test"

    def test_get_logger(self):
        """Test get_logger returns singleton."""
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2

    def test_configure_logger(self):
        """Test configure_logger."""
        config = LoggerConfig(enable_rerun=False, enable_file_log=False)
        logger = configure_logger(config)
        assert logger.config.enable_rerun is False

    def test_log_node_created_convenience(self):
        """Test log_node_created convenience function."""
        event = log_node_created("node_1", "CODE", "agent_1")
        assert event.node_id == "node_1"

    def test_log_status_changed_convenience(self):
        """Test log_status_changed convenience function."""
        event = log_status_changed("node_1", "PENDING", "VERIFIED")
        assert event.old_status == "PENDING"
        assert event.new_status == "VERIFIED"

    def test_log_edge_created_convenience(self):
        """Test log_edge_created convenience function."""
        event = log_edge_created("node_1", "node_2", "DEPENDS_ON")
        assert event.source_id == "node_1"

    def test_get_audit_logger(self):
        """Test get_audit_logger."""
        logger = get_audit_logger()
        assert logger is not None

    def test_log_audit(self, temp_dir):
        """Test log_audit convenience function."""
        audit_path = temp_dir / "audit.log"
        get_audit_logger(audit_path)
        log_audit("test_action", node_id="node_1", agent_id="agent_1")
        assert audit_path.exists()


# =============================================================================
# DIAGNOSTICS TESTS
# =============================================================================

class TestDiagnostics:
    """Tests for diagnostics functions."""

    def test_get_diagnostics(self):
        """Test get_diagnostics returns singleton."""
        diag1 = get_diagnostics()
        diag2 = get_diagnostics()
        assert diag1 is diag2

    def test_reset_diagnostics(self):
        """Test reset_diagnostics."""
        diag = get_diagnostics()
        diag._llm_calls.append(LLMCallMetric("Test", time.time()))
        reset_diagnostics()
        diag2 = get_diagnostics()
        assert len(diag2._llm_calls) == 0

    def test_print_state_summary(self, capsys):
        """Test print_state_summary."""
        print_state_summary(use_color=False)
        captured = capsys.readouterr()
        assert "PARAGON DIAGNOSTICS" in captured.out

    def test_print_session_summary(self, capsys):
        """Test print_session_summary."""
        diag = get_diagnostics()
        diag.set_session("test-session")
        print_session_summary(use_color=False)
        captured = capsys.readouterr()
        assert "Session Summary" in captured.out


# =============================================================================
# METRICS TESTS
# =============================================================================

class TestMetrics:
    """Tests for metrics functions."""

    def test_get_collector(self):
        """Test get_collector returns singleton."""
        collector1 = get_collector()
        collector2 = get_collector()
        assert collector1 is collector2

    def test_record_node_created_convenience(self):
        """Test record_node_created convenience function."""
        metric = metrics_record_node_created("node_1", "CODE")
        assert metric.node_id == "node_1"
        assert metric.node_type == "CODE"

    def test_record_node_start_convenience(self):
        """Test record_node_start convenience function."""
        collector = get_collector()
        collector.record_node_created("node_1", "CODE")
        record_node_start("node_1", "agent_1", "BUILDER", "build")
        metric = collector.get_metric("node_1")
        assert metric.agent_id == "agent_1"

    def test_record_node_complete_convenience(self):
        """Test record_node_complete convenience function."""
        collector = get_collector()
        collector.record_node_created("node_1", "CODE")
        metric = record_node_complete("node_1", "VERIFIED", tokens=100)
        assert metric is not None
        assert metric.status == "VERIFIED"

    def test_get_traceability_report_convenience(self):
        """Test get_traceability_report convenience function."""
        collector = get_collector()
        req_id = generate_id()
        collector.record_node_created(req_id, "REQ", traces_to_req=req_id)
        report = get_traceability_report(req_id)
        assert report.req_id == req_id


# =============================================================================
# CONFIG GRAPH TESTS
# =============================================================================

class TestConfigGraph:
    """Tests for config graph functions."""

    def test_load_toml_config(self):
        """Test loading TOML config."""
        config = load_toml_config()
        assert isinstance(config, dict)

    def test_initialize_config(self, mock_db):
        """Test initializing config in graph."""
        config_dict = {"system": {"name": "paragon"}}
        node_ids = initialize_config(mock_db, config_dict)
        assert "system" in node_ids
        assert mock_db.add_node.called

    def test_get_config(self, mock_db):
        """Test getting config from graph."""
        # Mock find_nodes to return a config node
        config_node = NodeData(
            id=generate_id(),
            type=NodeType.CONFIG.value,
            content='{"enabled": true}',
            status=NodeStatus.VERIFIED.value,
            created_by="system",
        )
        config_node.metadata.extra["config_section"] = "git"
        mock_db.find_nodes.return_value = [config_node]

        config = get_config(mock_db, "git")
        assert config == {"enabled": True}

    def test_get_all_config(self, mock_db):
        """Test getting all config from graph."""
        node1 = NodeData(
            id=generate_id(),
            type=NodeType.CONFIG.value,
            content='{"a": 1}',
            status=NodeStatus.VERIFIED.value,
            created_by="system",
        )
        node1.metadata.extra["config_section"] = "section1"

        node2 = NodeData(
            id=generate_id(),
            type=NodeType.CONFIG.value,
            content='{"b": 2}',
            status=NodeStatus.VERIFIED.value,
            created_by="system",
        )
        node2.metadata.extra["config_section"] = "section2"

        mock_db.find_nodes.return_value = [node1, node2]

        all_config = get_all_config(mock_db)
        assert "section1" in all_config
        assert "section2" in all_config

    def test_update_config(self, mock_db):
        """Test updating config in graph."""
        config_node = NodeData(
            id="node_1",
            type=NodeType.CONFIG.value,
            content='{"enabled": false}',
            status=NodeStatus.VERIFIED.value,
            created_by="system",
        )
        config_node.metadata.extra["config_section"] = "git"
        mock_db.find_nodes.return_value = [config_node]

        result = update_config(mock_db, "git", {"enabled": True})
        assert result is True
        assert mock_db.update_node.called

    def test_create_agent_config(self, mock_db):
        """Test creating agent config."""
        config = {"temperature": 0.7}
        node_id = create_agent_config(mock_db, "agent_1", config)
        assert isinstance(node_id, str)
        assert mock_db.add_node.called

    def test_get_agent_config(self, mock_db):
        """Test getting agent config."""
        config_node = NodeData(
            id=generate_id(),
            type=NodeType.AGENT_CONFIG.value,
            content='{"temperature": 0.7}',
            status=NodeStatus.VERIFIED.value,
            created_by="system",
        )
        config_node.metadata.extra["agent_id"] = "agent_1"
        mock_db.find_nodes.return_value = [config_node]

        config = get_agent_config(mock_db, "agent_1")
        assert config == {"temperature": 0.7}

    def test_create_resource_policy(self, mock_db):
        """Test creating resource policy."""
        node_id = create_resource_policy(mock_db, "default", 90.0, 95.0)
        assert isinstance(node_id, str)
        assert mock_db.add_node.called

    def test_get_resource_policy(self, mock_db):
        """Test getting resource policy."""
        policy_node = NodeData(
            id=generate_id(),
            type=NodeType.RESOURCE_POLICY.value,
            content='{"ram_threshold_percent": 90.0}',
            status=NodeStatus.VERIFIED.value,
            created_by="system",
        )
        policy_node.metadata.extra["policy_name"] = "default"
        mock_db.find_nodes.return_value = [policy_node]

        policy = get_resource_policy(mock_db, "default")
        assert policy["ram_threshold_percent"] == 90.0


# =============================================================================
# GIT SYNC TESTS
# =============================================================================

class TestGitSync:
    """Tests for git sync functions."""

    def test_load_git_config_from_toml(self):
        """Test loading git config from TOML."""
        config = load_git_config_from_toml()
        assert config is not None
        assert hasattr(config, 'enabled')

    def test_load_git_config_from_graph(self, mock_db):
        """Test loading git config from graph."""
        config_node = NodeData(
            id=generate_id(),
            type=NodeType.CONFIG.value,
            content='{"enabled": true, "repo_path": "."}',
            status=NodeStatus.VERIFIED.value,
            created_by="system",
        )
        config_node.metadata.extra["config_section"] = "git"
        mock_db.find_nodes.return_value = [config_node]

        config = load_git_config_from_graph(mock_db)
        assert config is not None

    def test_load_git_config(self, mock_db):
        """Test load_git_config with priority."""
        config = load_git_config()
        assert config is not None

    def test_get_git_sync(self):
        """Test get_git_sync returns singleton."""
        sync1 = get_git_sync()
        sync2 = get_git_sync()
        assert sync1 is sync2

    def test_on_transaction_complete(self):
        """Test on_transaction_complete convenience function."""
        result = on_transaction_complete(
            nodes_created=["node_1"],
            edges_created=[("node_1", "node_2", "DEPENDS_ON")],
        )
        # Result depends on whether git is enabled and repo exists
        assert isinstance(result, bool)


# =============================================================================
# OBSERVABILITY CONFIG TESTS
# =============================================================================

class TestObservabilityConfig:
    """Tests for observability configuration loading."""

    def test_load_observability_config_from_toml(self):
        """Test loading observability config from TOML."""
        config = load_observability_config_from_toml()
        assert isinstance(config, dict)
        assert "log_dir" in config or "rerun_enabled" in config

    def test_load_observability_config_from_graph(self, mock_db):
        """Test loading observability config from graph."""
        config_node = NodeData(
            id=generate_id(),
            type=NodeType.CONFIG.value,
            content='{"log_dir": "./data", "rerun_enabled": true}',
            status=NodeStatus.VERIFIED.value,
            created_by="system",
        )
        config_node.metadata.extra["config_section"] = "observability"
        mock_db.find_nodes.return_value = [config_node]

        config = load_observability_config_from_graph(mock_db)
        assert config is not None
        assert "log_dir" in config

    def test_load_observability_config(self, mock_db):
        """Test load_observability_config with priority."""
        config = load_observability_config()
        assert isinstance(config, dict)


# =============================================================================
# DATA LOADER TESTS
# =============================================================================

class TestDataLoader:
    """Tests for data loader functions."""

    def test_create_sample_node_csv(self, temp_dir):
        """Test creating sample node CSV."""
        csv_path = temp_dir / "nodes.csv"
        create_sample_node_csv(csv_path, num_nodes=10)
        assert csv_path.exists()

        # Verify content
        import polars as pl
        df = pl.read_csv(csv_path)
        assert len(df) == 10
        assert "id" in df.columns
        assert "type" in df.columns

    def test_create_sample_edge_csv(self, temp_dir):
        """Test creating sample edge CSV."""
        node_ids = [generate_id() for _ in range(5)]
        csv_path = temp_dir / "edges.csv"
        create_sample_edge_csv(csv_path, node_ids, edge_probability=0.5)
        assert csv_path.exists()

    def test_load_graph_from_csv(self, temp_dir):
        """Test loading graph from CSV files."""
        nodes_path = temp_dir / "nodes.csv"
        edges_path = temp_dir / "edges.csv"

        # Create sample files
        create_sample_node_csv(nodes_path, num_nodes=5)

        # Create edges CSV manually
        import polars as pl
        import uuid
        node_ids = [uuid.uuid4().hex for _ in range(3)]
        edges_df = pl.DataFrame({
            "source_id": [node_ids[0], node_ids[1]],
            "target_id": [node_ids[1], node_ids[2]],
            "type": ["DEPENDS_ON", "DEPENDS_ON"],
        })
        edges_df.write_csv(edges_path)

        # Load graph
        db = load_graph_from_csv(nodes_path, edges_path)
        assert db.node_count > 0

    def test_load_graph_from_parquet(self, temp_dir):
        """Test loading graph from Parquet files."""
        import polars as pl

        # Create sample Parquet files
        nodes_path = temp_dir / "nodes.parquet"
        nodes_df = pl.DataFrame({
            "id": [generate_id() for _ in range(5)],
            "type": ["CODE"] * 5,
            "content": ["test"] * 5,
            "status": ["PENDING"] * 5,
            "created_by": ["system"] * 5,
        })
        nodes_df.write_parquet(nodes_path)

        db = load_graph_from_parquet(nodes_path)
        assert db.node_count == 5


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests across multiple components."""

    def test_full_workflow(self, temp_dir, mock_db):
        """Test complete workflow of creating nodes, logging, and metrics."""
        # Set up diagnostics
        diag = get_diagnostics()
        diag.set_session("test-session")

        # Record LLM call
        with diag.llm_call("TestSchema") as call:
            call.set_tokens(100, 50)

        # Create node and record metrics
        node_id = generate_id()
        metric = metrics_record_node_created(node_id, "CODE")
        record_node_start(node_id, "agent_1", "BUILDER", "build")
        record_node_complete(node_id, "VERIFIED", tokens=150)

        # Verify everything was recorded
        assert len(diag._llm_calls) == 1
        collector = get_collector()
        node_metric = collector.get_metric(node_id)
        assert node_metric is not None
        assert node_metric.status == "VERIFIED"

    def test_config_initialization_and_retrieval(self, mock_db):
        """Test config initialization and retrieval."""
        # Initialize config
        config_dict = {
            "system": {"name": "paragon"},
            "git": {"enabled": True},
        }
        node_ids = initialize_config(mock_db, config_dict)

        # Mock the retrieval
        system_node = NodeData(
            id=node_ids["system"],
            type=NodeType.CONFIG.value,
            content='{"name": "paragon"}',
            status=NodeStatus.VERIFIED.value,
            created_by="system",
        )
        system_node.metadata.extra["config_section"] = "system"
        mock_db.find_nodes.return_value = [system_node]

        # Retrieve config
        config = get_config(mock_db, "system")
        assert config["name"] == "paragon"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
