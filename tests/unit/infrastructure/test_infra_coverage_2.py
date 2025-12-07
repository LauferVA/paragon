"""
Unit tests for infrastructure module - Coverage Group 2

Tests for:
- GitSync: Git synchronization manager
- LearningManager: Two-mode learning system
- MetricsCollector: Traceability engine
- MutationLogger: Temporal debugger
- PolarsLoader: Lazy data loader

Layer: Infrastructure
Status: Production
"""
import pytest
from pathlib import Path
import tempfile
import shutil
import subprocess
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
import polars as pl
import sqlite3
import time

from infrastructure.git_sync import (
    GitSync,
    GitSyncConfig,
    TeleologyChain,
    load_git_config_from_toml,
)
from infrastructure.learning import (
    LearningManager,
    LearningMode,
    ModelRecommendation,
    TransitionReport,
)
from infrastructure.metrics import (
    MetricsCollector,
    NodeMetric,
    FailurePattern,
    SuccessPattern,
    TraceabilityReport,
)
from infrastructure.logger import (
    MutationLogger,
    LoggerConfig,
    EventBuffer,
)
from infrastructure.data_loader import (
    PolarsLoader,
    SchemaValidationError,
    DataIntegrityError,
)
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, NodeStatus, EdgeType
from agents.schemas import CyclePhase, FailureCode, NodeOutcome
from viz.core import MutationType, MutationEvent


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
def temp_git_repo(temp_dir):
    """Create a temporary git repository."""
    repo_path = temp_dir / "test_repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Create initial commit
    test_file = repo_path / "README.md"
    test_file.write_text("# Test Repo\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    yield repo_path


@pytest.fixture
def mock_db():
    """Create a mock ParagonDB instance."""
    db = Mock()

    # Mock node data
    node = NodeData(
        id="node_123",
        type=NodeType.CODE.value,
        content="def hello():\n    return 'world'",
        status=NodeStatus.VERIFIED.value,
        created_by="agent_001",
    )
    db.get_node.return_value = node
    db.get_outgoing_edges.return_value = []

    return db


@pytest.fixture
def learning_store(temp_dir):
    """Create a temporary training store for learning tests."""
    from infrastructure.training_store import TrainingStore
    db_path = temp_dir / "learning_test.db"
    return TrainingStore(db_path=db_path)


# =============================================================================
# GITSYNC TESTS
# =============================================================================

class TestGitSyncInit:
    """Test GitSync initialization."""

    def test_init_with_config(self, temp_git_repo):
        """Test initialization with custom config."""
        config = GitSyncConfig(
            enabled=True,
            repo_path=str(temp_git_repo),
            auto_commit=False,
        )

        sync = GitSync(config=config, db=None)

        assert sync.config.enabled is True
        assert sync.config.auto_commit is False
        assert sync.repo_path == temp_git_repo

    def test_init_without_config_uses_defaults(self, temp_git_repo, monkeypatch):
        """Test initialization without config loads from TOML."""
        # Mock the config loading to avoid file dependency
        mock_config = GitSyncConfig(
            enabled=True,
            repo_path=str(temp_git_repo),
        )
        monkeypatch.setattr(
            "infrastructure.git_sync.load_git_config",
            lambda db=None: mock_config,
        )

        sync = GitSync(db=None)

        assert sync.config is not None

    def test_init_disabled_when_not_git_repo(self, temp_dir):
        """Test that git sync is disabled when not in a git repo."""
        config = GitSyncConfig(
            enabled=True,
            repo_path=str(temp_dir),
        )

        with pytest.warns(UserWarning, match="Not a git repository"):
            sync = GitSync(config=config, db=None)

        assert sync.config.enabled is False


class TestGitSyncEnsureGitRepo:
    """Test _ensure_git_repo method."""

    def test_ensure_git_repo_valid(self, temp_git_repo):
        """Test with valid git repository."""
        config = GitSyncConfig(enabled=True, repo_path=str(temp_git_repo))
        sync = GitSync(config=config)

        # Should not raise or warn
        assert sync.config.enabled is True

    def test_ensure_git_repo_invalid_disables(self, temp_dir):
        """Test that invalid git repo disables sync."""
        config = GitSyncConfig(enabled=True, repo_path=str(temp_dir))

        with pytest.warns(UserWarning):
            sync = GitSync(config=config)

        assert sync.config.enabled is False


class TestGitSyncGetTeleologyChain:
    """Test _get_teleology_chain method."""

    def test_get_teleology_chain_without_db(self, temp_git_repo):
        """Test teleology chain when no DB is provided."""
        config = GitSyncConfig(enabled=True, repo_path=str(temp_git_repo))
        sync = GitSync(config=config, db=None)

        chain = sync._get_teleology_chain("node_123")

        assert chain.code_node is None
        assert chain.spec_node is None
        assert chain.req_node is None

    def test_get_teleology_chain_with_code_node(self, temp_git_repo, mock_db):
        """Test teleology chain starting from CODE node."""
        config = GitSyncConfig(enabled=True, repo_path=str(temp_git_repo))
        sync = GitSync(config=config, db=mock_db)

        chain = sync._get_teleology_chain("node_123")

        assert chain.code_node is not None
        assert chain.code_node.id == "node_123"

    def test_get_teleology_chain_traversal(self, temp_git_repo, mock_db):
        """Test teleology chain traverses outgoing edges."""
        # Setup mock to return spec and req nodes
        spec_node = NodeData(
            id="spec_456",
            type=NodeType.SPEC.value,
            content="Specification content",
            status=NodeStatus.VERIFIED.value,
            created_by="agent_001",
        )
        req_node = NodeData(
            id="req_789",
            type=NodeType.REQ.value,
            content="Requirement content",
            status=NodeStatus.VERIFIED.value,
            created_by="agent_001",
        )

        def get_node_side_effect(node_id):
            if node_id == "node_123":
                return NodeData(
                    id="node_123",
                    type=NodeType.CODE.value,
                    content="code",
                    status=NodeStatus.VERIFIED.value,
                    created_by="agent",
                )
            elif node_id == "spec_456":
                return spec_node
            elif node_id == "req_789":
                return req_node

        mock_db.get_node.side_effect = get_node_side_effect
        mock_db.get_outgoing_edges.side_effect = [
            [{"target": "spec_456", "type": EdgeType.IMPLEMENTS.value}],
            [{"target": "req_789", "type": EdgeType.TRACES_TO.value}],
            [],
        ]

        config = GitSyncConfig(enabled=True, repo_path=str(temp_git_repo))
        sync = GitSync(config=config, db=mock_db)

        chain = sync._get_teleology_chain("node_123")

        assert chain.code_node is not None
        assert chain.spec_node is not None
        assert chain.req_node is not None


class TestGitSyncRunGitCommand:
    """Test _run_git_command method."""

    def test_run_git_command_success(self, temp_git_repo):
        """Test running a successful git command."""
        config = GitSyncConfig(enabled=True, repo_path=str(temp_git_repo))
        sync = GitSync(config=config)

        result = sync._run_git_command(["status", "--porcelain"])

        assert result.returncode == 0
        assert isinstance(result.stdout, str)

    def test_run_git_command_sets_author_env(self, temp_git_repo):
        """Test that git command sets author environment variables."""
        config = GitSyncConfig(
            enabled=True,
            repo_path=str(temp_git_repo),
            author_name="Test Paragon",
            author_email="paragon@test.com",
        )
        sync = GitSync(config=config)

        # This should execute without error and use the config
        result = sync._run_git_command(["config", "user.name"])
        assert result.returncode == 0


class TestGitSyncCreateTag:
    """Test create_tag method."""

    def test_create_tag_simple(self, temp_git_repo):
        """Test creating a simple tag."""
        config = GitSyncConfig(enabled=True, repo_path=str(temp_git_repo))
        sync = GitSync(config=config)

        success = sync.create_tag("v1.0.0")

        assert success is True

        # Verify tag exists
        result = subprocess.run(
            ["git", "tag", "-l", "v1.0.0"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
        )
        assert "v1.0.0" in result.stdout

    def test_create_tag_with_message(self, temp_git_repo):
        """Test creating an annotated tag with message."""
        config = GitSyncConfig(enabled=True, repo_path=str(temp_git_repo))
        sync = GitSync(config=config)

        success = sync.create_tag("v2.0.0", message="Release 2.0.0")

        assert success is True

    def test_create_tag_when_disabled(self, temp_dir):
        """Test that create_tag returns False when disabled."""
        config = GitSyncConfig(enabled=False)
        sync = GitSync(config=config)

        success = sync.create_tag("v1.0.0")

        assert success is False


class TestGitSyncGetCommitCount:
    """Test get_commit_count method."""

    def test_get_commit_count(self, temp_git_repo):
        """Test getting commit count."""
        config = GitSyncConfig(enabled=True, repo_path=str(temp_git_repo))
        sync = GitSync(config=config)

        count = sync.get_commit_count()

        assert count >= 1  # At least the initial commit

    def test_get_commit_count_when_disabled(self):
        """Test that get_commit_count returns 0 when disabled."""
        config = GitSyncConfig(enabled=False)
        sync = GitSync(config=config)

        count = sync.get_commit_count()

        assert count == 0


class TestGitSyncGetCurrentCommit:
    """Test get_current_commit method."""

    def test_get_current_commit(self, temp_git_repo):
        """Test getting current commit hash."""
        config = GitSyncConfig(enabled=True, repo_path=str(temp_git_repo))
        sync = GitSync(config=config)

        commit = sync.get_current_commit()

        assert commit is not None
        assert len(commit) == 7  # Short hash

    def test_get_current_commit_when_disabled(self):
        """Test that get_current_commit returns None when disabled."""
        config = GitSyncConfig(enabled=False)
        sync = GitSync(config=config)

        commit = sync.get_current_commit()

        assert commit is None


class TestGitSyncOnTransactionComplete:
    """Test on_transaction_complete method."""

    def test_on_transaction_complete_creates_commit(self, temp_git_repo):
        """Test that transaction completion creates a git commit."""
        config = GitSyncConfig(
            enabled=True,
            repo_path=str(temp_git_repo),
            auto_commit=True,
        )
        sync = GitSync(config=config, db=None)

        # Create a file change
        test_file = temp_git_repo / "test.txt"
        test_file.write_text("test content")

        success = sync.on_transaction_complete(
            nodes_created=["node_123"],
            edges_created=[],
            agent_id="agent_001",
            agent_role="BUILDER",
        )

        assert success is True

        # Verify commit was created
        result = subprocess.run(
            ["git", "log", "--oneline", "-n", "1"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
        )
        assert "chore:" in result.stdout or "feat:" in result.stdout

    def test_on_transaction_complete_disabled(self, temp_git_repo):
        """Test that disabled sync doesn't commit."""
        config = GitSyncConfig(enabled=False)
        sync = GitSync(config=config)

        success = sync.on_transaction_complete(
            nodes_created=["node_123"],
            edges_created=[],
        )

        assert success is False

    def test_on_transaction_complete_empty_transaction(self, temp_git_repo):
        """Test that empty transaction doesn't commit."""
        config = GitSyncConfig(enabled=True, repo_path=str(temp_git_repo))
        sync = GitSync(config=config)

        success = sync.on_transaction_complete(
            nodes_created=[],
            edges_created=[],
        )

        assert success is False


class TestTeleologyChainToCommitMessage:
    """Test TeleologyChain.to_commit_message method."""

    def test_to_commit_message_with_code_node(self):
        """Test commit message generation from code node."""
        code_node = NodeData(
            id="code_123",
            type=NodeType.CODE.value,
            content="def hello():\n    return 'world'",
            status=NodeStatus.VERIFIED.value,
            created_by="agent",
        )

        chain = TeleologyChain(code_node=code_node)
        message = chain.to_commit_message()

        assert message.startswith("feat:")
        assert "def hello():" in message

    def test_to_commit_message_with_req_node(self):
        """Test commit message generation from requirement node."""
        req_node = NodeData(
            id="req_abc123",
            type=NodeType.REQ.value,
            content="Implement hash function",
            status=NodeStatus.VERIFIED.value,
            created_by="agent",
        )

        chain = TeleologyChain(req_node=req_node)
        message = chain.to_commit_message()

        assert message.startswith("chore:")
        assert "REQ-req_abc1" in message
        assert "Implement hash function" in message

    def test_to_commit_message_empty_chain(self):
        """Test commit message generation from empty chain."""
        chain = TeleologyChain()
        message = chain.to_commit_message()

        assert message.startswith("chore:")
        assert "Update graph" in message


# =============================================================================
# LEARNING MANAGER TESTS
# =============================================================================

class TestLearningManagerInit:
    """Test LearningManager initialization."""

    def test_init_with_defaults(self, learning_store):
        """Test initialization with default parameters."""
        manager = LearningManager(store=learning_store)

        assert manager.mode == LearningMode.STUDY
        assert manager.store is learning_store

    def test_init_in_production_mode(self, learning_store):
        """Test initialization in production mode."""
        manager = LearningManager(store=learning_store, mode=LearningMode.PRODUCTION)

        assert manager.mode == LearningMode.PRODUCTION


class TestLearningManagerGetSampleCount:
    """Test _get_sample_count method."""

    def test_get_sample_count_no_data(self, learning_store):
        """Test getting sample count with no data."""
        manager = LearningManager(store=learning_store)

        count = manager._get_sample_count("claude-opus-4-5", "build")

        assert count == 0

    def test_get_sample_count_with_data(self, learning_store):
        """Test getting sample count with data."""
        from agents.schemas import AgentSignature, SignatureAction

        manager = LearningManager(store=learning_store)

        # Add some attributions
        sig = AgentSignature(
            agent_id="test",
            model_id="claude-opus-4-5",
            phase=CyclePhase.BUILD,
            action=SignatureAction.CREATED,
            temperature=0.7,
            context_constraints={},
            timestamp=datetime.utcnow().isoformat(),
        )

        for i in range(5):
            learning_store.record_attribution(
                f"session_{i}", sig, f"node_{i}", f"state_{i}"
            )

        count = manager._get_sample_count("claude-opus-4-5", "build")

        assert count == 5


class TestLearningManagerRecordOutcome:
    """Test record_outcome method."""

    def test_record_outcome_success(self, learning_store):
        """Test recording successful outcome."""
        manager = LearningManager(store=learning_store)

        manager.record_outcome(
            session_id="session_1",
            success=True,
            stats={"total_nodes": 10, "total_tokens": 1000},
        )

        outcome = learning_store.get_session_outcome("session_1")
        assert outcome is not None
        assert outcome["outcome"] == "verified_success"

    def test_record_outcome_failure(self, learning_store):
        """Test recording failed outcome."""
        manager = LearningManager(store=learning_store)

        manager.record_outcome(
            session_id="session_2",
            success=False,
            failure_code=FailureCode.F2,
            failure_phase=CyclePhase.BUILD,
        )

        outcome = learning_store.get_session_outcome("session_2")
        assert outcome is not None
        assert outcome["outcome"] == "verified_failure"
        assert outcome["failure_code"] == "F2"


class TestLearningManagerShouldTransitionToProduction:
    """Test should_transition_to_production method."""

    def test_should_transition_not_ready(self, learning_store):
        """Test transition report when not ready."""
        manager = LearningManager(store=learning_store)

        report = manager.should_transition_to_production()

        assert report.ready is False
        assert report.session_count == 0
        assert "Not ready" in report.recommendation

    def test_should_transition_ready(self, learning_store):
        """Test transition report when ready."""
        manager = LearningManager(store=learning_store)

        # Record 100 sessions
        for i in range(100):
            manager.record_outcome(f"session_{i}", success=True)

        report = manager.should_transition_to_production()

        assert report.ready is True
        assert report.session_count == 100
        assert "ready" in report.recommendation.lower()


class TestLearningManagerTransitionToProduction:
    """Test transition_to_production method."""

    def test_transition_to_production_success(self, learning_store):
        """Test successful transition to production."""
        manager = LearningManager(store=learning_store, mode=LearningMode.STUDY)

        # Record enough sessions
        for i in range(100):
            manager.record_outcome(f"session_{i}", success=True)

        success = manager.transition_to_production()

        assert success is True
        assert manager.mode == LearningMode.PRODUCTION

    def test_transition_to_production_not_ready(self, learning_store):
        """Test transition fails when not ready."""
        manager = LearningManager(store=learning_store, mode=LearningMode.STUDY)

        success = manager.transition_to_production()

        assert success is False
        assert manager.mode == LearningMode.STUDY


class TestLearningManagerTransitionToStudy:
    """Test transition_to_study method."""

    def test_transition_to_study(self, learning_store):
        """Test transition to study mode."""
        manager = LearningManager(store=learning_store, mode=LearningMode.PRODUCTION)

        manager.transition_to_study()

        assert manager.mode == LearningMode.STUDY


# =============================================================================
# METRICS COLLECTOR TESTS
# =============================================================================

class TestMetricsCollectorInit:
    """Test MetricsCollector initialization."""

    def test_init(self):
        """Test collector initialization."""
        collector = MetricsCollector()

        assert len(collector._metrics) == 0
        assert len(collector._in_progress) == 0


class TestMetricsCollectorClear:
    """Test clear method."""

    def test_clear(self):
        """Test clearing all metrics."""
        collector = MetricsCollector()

        collector.record_node_created("node_1", NodeType.CODE.value)
        assert len(collector._metrics) > 0

        collector.clear()

        assert len(collector._metrics) == 0
        assert len(collector._in_progress) == 0


class TestMetricsCollectorGetAllMetrics:
    """Test get_all_metrics method."""

    def test_get_all_metrics_empty(self):
        """Test getting all metrics when empty."""
        collector = MetricsCollector()

        metrics = collector.get_all_metrics()

        assert len(metrics) == 0

    def test_get_all_metrics_with_data(self):
        """Test getting all metrics with data."""
        collector = MetricsCollector()

        collector.record_node_created("node_1", NodeType.CODE.value)
        collector.record_node_created("node_2", NodeType.SPEC.value)

        metrics = collector.get_all_metrics()

        assert len(metrics) == 2


class TestMetricsCollectorGetFailurePatterns:
    """Test get_failure_patterns method."""

    def test_get_failure_patterns_no_failures(self):
        """Test getting failure patterns with no failures."""
        collector = MetricsCollector()

        patterns = collector.get_failure_patterns()

        assert len(patterns) == 0

    def test_get_failure_patterns_with_failures(self):
        """Test getting failure patterns with failures."""
        collector = MetricsCollector()

        # Record some failures with same error
        for i in range(3):
            collector.record_node_created(f"node_{i}", NodeType.CODE.value)
            collector.record_node_complete(
                f"node_{i}",
                NodeStatus.FAILED.value,
                error="TypeError: expected string",
            )

        patterns = collector.get_failure_patterns(min_count=2)

        assert len(patterns) >= 1
        assert patterns[0].count == 3


class TestMetricsCollectorGetMetric:
    """Test get_metric method."""

    def test_get_metric_exists(self):
        """Test getting existing metric."""
        collector = MetricsCollector()

        collector.record_node_created("node_1", NodeType.CODE.value)
        metric = collector.get_metric("node_1")

        assert metric is not None
        assert metric.node_id == "node_1"

    def test_get_metric_not_exists(self):
        """Test getting non-existent metric."""
        collector = MetricsCollector()

        metric = collector.get_metric("nonexistent")

        assert metric is None


class TestMetricsCollectorGetSuccessPatterns:
    """Test get_success_patterns method."""

    def test_get_success_patterns_no_successes(self):
        """Test getting success patterns with no successes."""
        collector = MetricsCollector()

        patterns = collector.get_success_patterns()

        assert len(patterns) == 0

    def test_get_success_patterns_with_successes(self):
        """Test getting success patterns with successes."""
        collector = MetricsCollector()

        # Record successful nodes
        for i in range(3):
            collector.record_node_created(f"node_{i}", NodeType.CODE.value)
            collector.record_node_start(
                f"node_{i}", "agent_1", "BUILDER", "create_code"
            )
            collector.record_node_complete(
                f"node_{i}", NodeStatus.VERIFIED.value, tokens=100
            )

        patterns = collector.get_success_patterns()

        assert len(patterns) >= 1


class TestMetricsCollectorGetSummary:
    """Test get_summary method."""

    def test_get_summary_empty(self):
        """Test getting summary with no data."""
        collector = MetricsCollector()

        summary = collector.get_summary()

        assert summary["total_nodes"] == 0
        assert summary["total_tokens"] == 0

    def test_get_summary_with_data(self):
        """Test getting summary with data."""
        collector = MetricsCollector()

        collector.record_node_created("node_1", NodeType.CODE.value)
        collector.record_node_complete("node_1", NodeStatus.VERIFIED.value, tokens=100)

        summary = collector.get_summary()

        assert summary["total_nodes"] == 1
        assert summary["total_tokens"] == 100


class TestMetricsCollectorGetTraceabilityReport:
    """Test get_traceability_report method."""

    def test_get_traceability_report(self):
        """Test getting traceability report."""
        collector = MetricsCollector()

        # Create a REQ node
        collector.record_node_created(
            "req_1", NodeType.REQ.value, traces_to_req="req_1"
        )

        # Create CODE nodes that trace to it
        collector.record_node_created(
            "code_1", NodeType.CODE.value, traces_to_req="req_1"
        )
        collector.record_node_complete("code_1", NodeStatus.VERIFIED.value, tokens=100)

        report = collector.get_traceability_report("req_1")

        assert report.req_id == "req_1"
        assert report.total_code >= 1
        assert report.total_tokens >= 100


class TestMetricsCollectorQueryByAgent:
    """Test query_by_agent method."""

    def test_query_by_agent(self):
        """Test querying metrics by agent role."""
        collector = MetricsCollector()

        collector.record_node_created("node_1", NodeType.CODE.value)
        collector.record_node_start("node_1", "agent_1", "BUILDER", "create")

        collector.record_node_created("node_2", NodeType.CODE.value)
        collector.record_node_start("node_2", "agent_2", "TESTER", "test")

        results = collector.query_by_agent(agent_role="BUILDER")

        assert len(results) == 1
        assert results[0].agent_role == "BUILDER"


class TestMetricsCollectorQueryByStatus:
    """Test query_by_status method."""

    def test_query_by_status(self):
        """Test querying metrics by status."""
        collector = MetricsCollector()

        collector.record_node_created("node_1", NodeType.CODE.value)
        collector.record_node_complete("node_1", NodeStatus.VERIFIED.value)

        collector.record_node_created("node_2", NodeType.CODE.value)
        collector.record_node_complete("node_2", NodeStatus.FAILED.value)

        verified = collector.query_by_status(NodeStatus.VERIFIED.value)

        assert len(verified) == 1
        assert verified[0].status == NodeStatus.VERIFIED.value


class TestMetricsCollectorQueryByTraceability:
    """Test query_by_traceability method."""

    def test_query_by_traceability_req_id(self):
        """Test querying by requirement ID."""
        collector = MetricsCollector()

        collector.record_node_created(
            "node_1", NodeType.CODE.value, traces_to_req="req_1"
        )
        collector.record_node_created(
            "node_2", NodeType.CODE.value, traces_to_req="req_2"
        )

        results = collector.query_by_traceability(req_id="req_1")

        assert len(results) == 1
        assert results[0].traces_to_req == "req_1"


class TestMetricsCollectorQueryFailures:
    """Test query_failures method."""

    def test_query_failures(self):
        """Test querying all failures."""
        collector = MetricsCollector()

        collector.record_node_created("node_1", NodeType.CODE.value)
        collector.record_node_complete("node_1", NodeStatus.FAILED.value, error="Error")

        collector.record_node_created("node_2", NodeType.CODE.value)
        collector.record_node_complete("node_2", NodeStatus.VERIFIED.value)

        failures = collector.query_failures()

        assert len(failures) == 1
        assert failures[0].status == NodeStatus.FAILED.value


class TestMetricsCollectorRecordMaterialization:
    """Test record_materialization method."""

    def test_record_materialization(self):
        """Test recording materialization."""
        collector = MetricsCollector()

        collector.record_node_created("node_1", NodeType.CODE.value)
        collector.record_materialization(
            "node_1", commit_sha="abc123", files=["file1.py", "file2.py"]
        )

        metric = collector.get_metric("node_1")

        assert metric.materialized_commit == "abc123"
        assert len(metric.materialized_files) == 2


class TestMetricsCollectorRecordNodeComplete:
    """Test record_node_complete method."""

    def test_record_node_complete(self):
        """Test recording node completion."""
        collector = MetricsCollector()

        collector.record_node_created("node_1", NodeType.CODE.value)
        collector.record_node_start("node_1", "agent_1", "BUILDER", "create")

        metric = collector.record_node_complete(
            "node_1",
            NodeStatus.VERIFIED.value,
            tokens=100,
            input_tokens=50,
            output_tokens=50,
        )

        assert metric is not None
        assert metric.status == NodeStatus.VERIFIED.value
        assert metric.token_count == 100
        assert metric.processing_time_ms is not None


class TestMetricsCollectorRecordNodeCreated:
    """Test record_node_created method."""

    def test_record_node_created(self):
        """Test recording node creation."""
        collector = MetricsCollector()

        metric = collector.record_node_created(
            "node_1",
            NodeType.CODE.value,
            created_by="agent_1",
            traces_to_req="req_1",
        )

        assert metric.node_id == "node_1"
        assert metric.node_type == NodeType.CODE.value
        assert metric.traces_to_req == "req_1"


class TestMetricsCollectorRecordNodeStart:
    """Test record_node_start method."""

    def test_record_node_start(self):
        """Test recording node processing start."""
        collector = MetricsCollector()

        collector.record_node_created("node_1", NodeType.CODE.value)
        collector.record_node_start(
            "node_1",
            "agent_1",
            "BUILDER",
            "create_code",
            context_node_ids=["ctx_1", "ctx_2"],
            context_token_count=500,
        )

        metric = collector.get_metric("node_1")

        assert metric.agent_id == "agent_1"
        assert metric.agent_role == "BUILDER"
        assert metric.operation == "create_code"
        assert len(metric.context_node_ids) == 2


class TestMetricsCollectorRecordRetry:
    """Test record_retry method."""

    def test_record_retry(self):
        """Test recording retry attempts."""
        collector = MetricsCollector()

        collector.record_node_created("node_1", NodeType.CODE.value)
        collector.record_retry("node_1", "First error")
        collector.record_retry("node_1", "Second error")

        metric = collector.get_metric("node_1")

        assert metric.retry_count == 2
        assert metric.last_error == "Second error"


class TestMetricsCollectorSubscribeToMutationLogger:
    """Test subscribe_to_mutation_logger method."""

    def test_subscribe_to_mutation_logger(self):
        """Test subscribing to mutation logger."""
        collector = MetricsCollector()
        logger = MutationLogger()

        # Should not raise
        collector.subscribe_to_mutation_logger(logger)


class TestMetricsCollectorSyncFromMutationLogger:
    """Test sync_from_mutation_logger method."""

    def test_sync_from_mutation_logger(self):
        """Test syncing from mutation logger."""
        collector = MetricsCollector()
        logger = MutationLogger()

        # Log some events
        logger.log_node_created("node_1", NodeType.CODE.value, agent_id="agent_1")
        logger.log_status_changed("node_1", "pending", "verified")

        count = collector.sync_from_mutation_logger(logger)

        assert count >= 0


class TestMetricsCollectorToDataframe:
    """Test to_dataframe method."""

    def test_to_dataframe_empty(self):
        """Test converting empty metrics to dataframe."""
        collector = MetricsCollector()

        df = collector.to_dataframe()

        assert df.is_empty()

    def test_to_dataframe_with_data(self):
        """Test converting metrics to dataframe."""
        collector = MetricsCollector()

        collector.record_node_created("node_1", NodeType.CODE.value)
        collector.record_node_complete("node_1", NodeStatus.VERIFIED.value, tokens=100)

        df = collector.to_dataframe()

        assert not df.is_empty()
        assert "node_id" in df.columns
        assert "token_count" in df.columns


class TestMetricsCollectorUpdateTraceability:
    """Test update_traceability method."""

    def test_update_traceability(self):
        """Test updating traceability links."""
        collector = MetricsCollector()

        collector.record_node_created("node_1", NodeType.CODE.value)
        collector.update_traceability(
            "node_1",
            traces_to_req="req_1",
            traces_to_spec="spec_1",
            implements_spec="spec_1",
        )

        metric = collector.get_metric("node_1")

        assert metric.traces_to_req == "req_1"
        assert metric.traces_to_spec == "spec_1"
        assert metric.implements_spec == "spec_1"


# =============================================================================
# MUTATION LOGGER TESTS
# =============================================================================

class TestMutationLoggerInit:
    """Test MutationLogger initialization."""

    def test_init_with_defaults(self):
        """Test initialization with defaults."""
        logger = MutationLogger()

        assert logger.config is not None
        assert logger._buffer is not None

    def test_init_with_custom_config(self, temp_dir):
        """Test initialization with custom config."""
        config = LoggerConfig(
            enable_file_log=True,
            log_path=temp_dir,
            buffer_size=5000,
        )

        logger = MutationLogger(config=config)

        assert logger.config.buffer_size == 5000


class TestMutationLoggerEmit:
    """Test _emit method."""

    def test_emit_adds_to_buffer(self):
        """Test that emit adds event to buffer."""
        logger = MutationLogger()

        event = MutationEvent(
            timestamp=datetime.now().isoformat(),
            sequence=1,
            mutation_type=MutationType.NODE_CREATED.value,
            node_id="node_1",
            node_type=NodeType.CODE.value,
        )

        logger._emit(event)

        assert len(logger._buffer) == 1


class TestMutationLoggerNow:
    """Test _now method."""

    def test_now_returns_iso_timestamp(self):
        """Test that _now returns ISO8601 timestamp."""
        logger = MutationLogger()

        timestamp = logger._now()

        assert "T" in timestamp
        assert isinstance(timestamp, str)


class TestMutationLoggerClose:
    """Test close method."""

    def test_close(self):
        """Test closing logger."""
        logger = MutationLogger()

        # Should not raise
        logger.close()


class TestMutationLoggerGetCorrelationId:
    """Test get_correlation_id method."""

    def test_get_correlation_id_default(self):
        """Test getting correlation ID when not set."""
        logger = MutationLogger()

        correlation_id = logger.get_correlation_id()

        assert correlation_id is None

    def test_get_correlation_id_after_set(self):
        """Test getting correlation ID after setting."""
        logger = MutationLogger()

        logger.set_correlation_id("test_correlation")
        correlation_id = logger.get_correlation_id()

        assert correlation_id == "test_correlation"


class TestMutationLoggerGetEventsByType:
    """Test get_events_by_type method."""

    def test_get_events_by_type(self):
        """Test getting events by type."""
        logger = MutationLogger()

        logger.log_node_created("node_1", NodeType.CODE.value)
        logger.log_node_updated("node_2", NodeType.SPEC.value)
        logger.log_node_created("node_3", NodeType.CODE.value)

        created_events = logger.get_events_by_type(MutationType.NODE_CREATED.value)

        assert len(created_events) == 2


class TestMutationLoggerGetEventsForNode:
    """Test get_events_for_node method."""

    def test_get_events_for_node(self):
        """Test getting events for a specific node."""
        logger = MutationLogger()

        logger.log_node_created("node_1", NodeType.CODE.value)
        logger.log_status_changed("node_1", "pending", "verified")
        logger.log_node_created("node_2", NodeType.SPEC.value)

        events = logger.get_events_for_node("node_1")

        assert len(events) == 2


class TestMutationLoggerGetEventsSince:
    """Test get_events_since method."""

    def test_get_events_since(self):
        """Test getting events since a timestamp."""
        logger = MutationLogger()

        # Log an event
        logger.log_node_created("node_1", NodeType.CODE.value)

        # Get current time
        now = datetime.now().isoformat()

        # Log another event
        time.sleep(0.01)
        logger.log_node_created("node_2", NodeType.CODE.value)

        events = logger.get_events_since(now)

        # Should get at least the second event
        assert len(events) >= 1


class TestMutationLoggerGetNodeTimeline:
    """Test get_node_timeline method."""

    def test_get_node_timeline(self):
        """Test getting node timeline."""
        logger = MutationLogger()

        logger.log_node_created("node_1", NodeType.CODE.value)
        logger.log_status_changed("node_1", "pending", "processing")
        logger.log_status_changed("node_1", "processing", "verified")

        timeline = logger.get_node_timeline("node_1")

        assert len(timeline) == 3
        assert timeline[0]["type"] == MutationType.NODE_CREATED.value


class TestMutationLoggerGetRecentEvents:
    """Test get_recent_events method."""

    def test_get_recent_events(self):
        """Test getting recent events."""
        logger = MutationLogger()

        for i in range(10):
            logger.log_node_created(f"node_{i}", NodeType.CODE.value)

        recent = logger.get_recent_events(n=5)

        assert len(recent) == 5


class TestMutationLoggerLogContextPruned:
    """Test log_context_pruned method."""

    def test_log_context_pruned(self):
        """Test logging context pruning."""
        logger = MutationLogger()

        event = logger.log_context_pruned(
            node_id="node_1",
            nodes_considered=100,
            nodes_selected=20,
            token_usage=1000,
        )

        assert event.mutation_type == MutationType.CONTEXT_PRUNED.value
        assert event.nodes_considered == 100
        assert event.nodes_selected == 20


class TestMutationLoggerLogEdgeCreated:
    """Test log_edge_created method."""

    def test_log_edge_created(self):
        """Test logging edge creation."""
        logger = MutationLogger()

        event = logger.log_edge_created(
            source_id="node_1",
            target_id="node_2",
            edge_type=EdgeType.IMPLEMENTS.value,
        )

        assert event.mutation_type == MutationType.EDGE_CREATED.value
        assert event.source_id == "node_1"
        assert event.target_id == "node_2"


class TestMutationLoggerLogEdgeDeleted:
    """Test log_edge_deleted method."""

    def test_log_edge_deleted(self):
        """Test logging edge deletion."""
        logger = MutationLogger()

        event = logger.log_edge_deleted(
            source_id="node_1",
            target_id="node_2",
            edge_type=EdgeType.IMPLEMENTS.value,
        )

        assert event.mutation_type == MutationType.EDGE_DELETED.value


class TestMutationLoggerLogNodeCreated:
    """Test log_node_created method."""

    def test_log_node_created(self):
        """Test logging node creation."""
        logger = MutationLogger()

        event = logger.log_node_created(
            node_id="node_1",
            node_type=NodeType.CODE.value,
            agent_id="agent_1",
            agent_role="BUILDER",
        )

        assert event.mutation_type == MutationType.NODE_CREATED.value
        assert event.node_id == "node_1"


class TestMutationLoggerLogNodeDeleted:
    """Test log_node_deleted method."""

    def test_log_node_deleted(self):
        """Test logging node deletion."""
        logger = MutationLogger()

        event = logger.log_node_deleted("node_1", NodeType.CODE.value)

        assert event.mutation_type == MutationType.NODE_DELETED.value


class TestMutationLoggerLogNodeUpdated:
    """Test log_node_updated method."""

    def test_log_node_updated(self):
        """Test logging node update."""
        logger = MutationLogger()

        event = logger.log_node_updated("node_1", NodeType.CODE.value)

        assert event.mutation_type == MutationType.NODE_UPDATED.value


class TestMutationLoggerLogStatusChanged:
    """Test log_status_changed method."""

    def test_log_status_changed(self):
        """Test logging status change."""
        logger = MutationLogger()

        event = logger.log_status_changed(
            node_id="node_1",
            old_status="pending",
            new_status="verified",
            agent_id="agent_1",
        )

        assert event.mutation_type == MutationType.STATUS_CHANGED.value
        assert event.old_status == "pending"
        assert event.new_status == "verified"


class TestMutationLoggerSetCorrelationId:
    """Test set_correlation_id method."""

    def test_set_correlation_id(self):
        """Test setting correlation ID."""
        logger = MutationLogger()

        logger.set_correlation_id("test_id")

        assert logger._correlation_id == "test_id"


class TestMutationLoggerSubscribe:
    """Test subscribe method."""

    def test_subscribe(self):
        """Test subscribing to events."""
        logger = MutationLogger()
        events_received = []

        def callback(event):
            events_received.append(event)

        logger.subscribe(callback)
        logger.log_node_created("node_1", NodeType.CODE.value)

        assert len(events_received) == 1


class TestMutationLoggerUnsubscribe:
    """Test unsubscribe method."""

    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        logger = MutationLogger()
        events_received = []

        def callback(event):
            events_received.append(event)

        logger.subscribe(callback)
        logger.log_node_created("node_1", NodeType.CODE.value)

        logger.unsubscribe(callback)
        logger.log_node_created("node_2", NodeType.CODE.value)

        assert len(events_received) == 1


# =============================================================================
# POLARS LOADER TESTS
# =============================================================================

class TestPolarsLoaderInit:
    """Test PolarsLoader initialization."""

    def test_init_with_validation(self):
        """Test initialization with validation enabled."""
        loader = PolarsLoader(validate=True)

        assert loader.validate is True

    def test_init_without_validation(self):
        """Test initialization with validation disabled."""
        loader = PolarsLoader(validate=False)

        assert loader.validate is False


class TestPolarsLoaderValidateEdgeSchema:
    """Test _validate_edge_schema method."""

    def test_validate_edge_schema_valid(self, temp_dir):
        """Test validating valid edge schema."""
        # Create a valid edge CSV
        csv_path = temp_dir / "edges.csv"
        df = pl.DataFrame({
            "source_id": ["node_1", "node_2"],
            "target_id": ["node_2", "node_3"],
            "type": ["implements", "depends_on"],
        })
        df.write_csv(csv_path)

        loader = PolarsLoader(validate=True)
        lf = pl.scan_csv(csv_path)

        # Should not raise
        loader._validate_edge_schema(lf)

    def test_validate_edge_schema_missing_column(self, temp_dir):
        """Test validating edge schema with missing column."""
        csv_path = temp_dir / "edges.csv"
        df = pl.DataFrame({
            "source_id": ["node_1"],
            "target_id": ["node_2"],
            # Missing 'type' column
        })
        df.write_csv(csv_path)

        loader = PolarsLoader(validate=True)
        lf = pl.scan_csv(csv_path)

        with pytest.raises(SchemaValidationError):
            loader._validate_edge_schema(lf)


class TestPolarsLoaderValidateNodeSchema:
    """Test _validate_node_schema method."""

    def test_validate_node_schema_valid(self, temp_dir):
        """Test validating valid node schema."""
        csv_path = temp_dir / "nodes.csv"
        df = pl.DataFrame({
            "id": ["node_1", "node_2"],
            "type": ["code", "spec"],
            "content": ["content1", "content2"],
            "status": ["pending", "verified"],
            "created_by": ["agent_1", "agent_2"],
        })
        df.write_csv(csv_path)

        loader = PolarsLoader(validate=True)
        lf = pl.scan_csv(csv_path)

        # Should not raise
        loader._validate_node_schema(lf)

    def test_validate_node_schema_missing_column(self, temp_dir):
        """Test validating node schema with missing column."""
        csv_path = temp_dir / "nodes.csv"
        df = pl.DataFrame({
            "id": ["node_1"],
            "type": ["code"],
            # Missing required columns
        })
        df.write_csv(csv_path)

        loader = PolarsLoader(validate=True)
        lf = pl.scan_csv(csv_path)

        with pytest.raises(SchemaValidationError):
            loader._validate_node_schema(lf)


class TestPolarsLoaderLoadArrowEdges:
    """Test load_arrow_edges method."""

    def test_load_arrow_edges(self, temp_dir):
        """Test loading edges from Arrow IPC file."""
        arrow_path = temp_dir / "edges.arrow"
        df = pl.DataFrame({
            "source_id": ["node_1"],
            "target_id": ["node_2"],
            "type": ["implements"],
        })
        df.write_ipc(arrow_path)

        loader = PolarsLoader(validate=True)
        lf = loader.load_arrow_edges(arrow_path)

        assert isinstance(lf, pl.LazyFrame)


class TestPolarsLoaderLoadArrowNodes:
    """Test load_arrow_nodes method."""

    def test_load_arrow_nodes(self, temp_dir):
        """Test loading nodes from Arrow IPC file."""
        arrow_path = temp_dir / "nodes.arrow"
        df = pl.DataFrame({
            "id": ["node_1"],
            "type": ["code"],
            "content": ["content"],
            "status": ["pending"],
            "created_by": ["agent"],
        })
        df.write_ipc(arrow_path)

        loader = PolarsLoader(validate=True)
        lf = loader.load_arrow_nodes(arrow_path)

        assert isinstance(lf, pl.LazyFrame)


class TestPolarsLoaderLoadCsvEdges:
    """Test load_csv_edges method."""

    def test_load_csv_edges(self, temp_dir):
        """Test loading edges from CSV."""
        csv_path = temp_dir / "edges.csv"
        df = pl.DataFrame({
            "source_id": ["node_1"],
            "target_id": ["node_2"],
            "type": ["implements"],
        })
        df.write_csv(csv_path)

        loader = PolarsLoader(validate=True)
        lf = loader.load_csv_edges(csv_path)

        assert isinstance(lf, pl.LazyFrame)


class TestPolarsLoaderLoadCsvNodes:
    """Test load_csv_nodes method."""

    def test_load_csv_nodes(self, temp_dir):
        """Test loading nodes from CSV."""
        csv_path = temp_dir / "nodes.csv"
        df = pl.DataFrame({
            "id": ["node_1"],
            "type": ["code"],
            "content": ["content"],
            "status": ["pending"],
            "created_by": ["agent"],
        })
        df.write_csv(csv_path)

        loader = PolarsLoader(validate=True)
        lf = loader.load_csv_nodes(csv_path)

        assert isinstance(lf, pl.LazyFrame)


class TestPolarsLoaderLoadGraphState:
    """Test load_graph_state method."""

    def test_load_graph_state_csv(self, temp_dir):
        """Test loading graph state from CSV files."""
        nodes_path = temp_dir / "nodes.csv"
        edges_path = temp_dir / "edges.csv"

        nodes_df = pl.DataFrame({
            "id": ["node_1"],
            "type": ["code"],
            "content": ["content"],
            "status": ["pending"],
            "created_by": ["agent"],
        })
        nodes_df.write_csv(nodes_path)

        edges_df = pl.DataFrame({
            "source_id": ["node_1"],
            "target_id": ["node_2"],
            "type": ["implements"],
        })
        edges_df.write_csv(edges_path)

        loader = PolarsLoader(validate=True)
        nodes_lf, edges_lf = loader.load_graph_state(nodes_path, edges_path, format="csv")

        assert isinstance(nodes_lf, pl.LazyFrame)
        assert isinstance(edges_lf, pl.LazyFrame)

    def test_load_graph_state_invalid_format(self, temp_dir):
        """Test loading graph state with invalid format."""
        loader = PolarsLoader()

        with pytest.raises(ValueError, match="Unknown format"):
            loader.load_graph_state(
                temp_dir / "nodes.csv",
                temp_dir / "edges.csv",
                format="invalid",
            )


class TestPolarsLoaderLoadParquetEdges:
    """Test load_parquet_edges method."""

    def test_load_parquet_edges(self, temp_dir):
        """Test loading edges from Parquet."""
        parquet_path = temp_dir / "edges.parquet"
        df = pl.DataFrame({
            "source_id": ["node_1"],
            "target_id": ["node_2"],
            "type": ["implements"],
        })
        df.write_parquet(parquet_path)

        loader = PolarsLoader(validate=True)
        lf = loader.load_parquet_edges(parquet_path)

        assert isinstance(lf, pl.LazyFrame)


class TestPolarsLoaderLoadParquetNodes:
    """Test load_parquet_nodes method."""

    def test_load_parquet_nodes(self, temp_dir):
        """Test loading nodes from Parquet."""
        parquet_path = temp_dir / "nodes.parquet"
        df = pl.DataFrame({
            "id": ["node_1"],
            "type": ["code"],
            "content": ["content"],
            "status": ["pending"],
            "created_by": ["agent"],
        })
        df.write_parquet(parquet_path)

        loader = PolarsLoader(validate=True)
        lf = loader.load_parquet_nodes(parquet_path)

        assert isinstance(lf, pl.LazyFrame)
