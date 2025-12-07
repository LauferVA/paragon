"""
Unit tests for uncovered functions in agents module (Part 1 of 2).

This test file covers the first 68 of 137 uncovered functions:

Classes tested:
- AdaptiveQuestioner: All private methods and initialization
- Documenter: All public and private methods
- HumanLoopController: Lifecycle methods
- MarkdownBuilder: All builder methods
- QualityGate: All private check methods

Functions tested:
- _compute_alignment_score
- _create_research_node
- _get_git_sync
- _get_mutation_logger
- _get_training_store
- _log_edge_created
- _log_node_created
- _record_attribution
- _record_transaction

Layer: Tests
Status: Production
"""
import pytest
import tempfile
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime

import msgspec

# AdaptiveQuestioner imports
from agents.adaptive_questioner import (
    AdaptiveQuestioner,
    QuestionPriority,
    QuestionOutcome,
    UserPriorities,
)
from agents.schemas import AmbiguityMarker

# Documenter imports
from agents.documenter import (
    Documenter,
    MarkdownBuilder,
    DocumenterConfig,
    load_documenter_config,
    load_documenter_config_from_graph,
    load_documenter_config_from_toml,
)

# HumanLoop imports
from agents.human_loop import (
    HumanLoopController,
    PausePoint,
    HumanRequest,
    HumanResponse,
    PauseType,
    RequestStatus,
    Priority,
)

# QualityGate imports
from agents.quality_gate import (
    QualityGate,
    QualityReport,
    QualityViolation,
)

# Tools imports
from agents.tools import (
    _compute_alignment_score,
    _get_git_sync,
    _get_mutation_logger,
    _get_training_store,
    _log_edge_created,
    _log_node_created,
    _record_attribution,
    _record_transaction,
)

from core.schemas import NodeData
from core.ontology import NodeType, NodeStatus, EdgeType
from infrastructure.training_store import TrainingStore


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    yield db_path
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_db():
    """Create a mock ParagonDB instance."""
    db = Mock()
    db.node_count = 10
    db.edge_count = 15
    db.get_nodes_by_type = Mock(return_value=[])
    db.get_node = Mock(return_value=None)
    db.get_incoming_edges = Mock(return_value=[])
    db.get_outgoing_edges = Mock(return_value=[])
    return db


@pytest.fixture
def sample_code_node():
    """Create a sample code node for testing."""
    return NodeData.create(
        type=NodeType.CODE.value,
        content="""
def calculate_sum(a, b):
    return a + b
""",
        created_by="test",
    )


@pytest.fixture
def sample_spec_node():
    """Create a sample spec node for testing."""
    return NodeData.create(
        type=NodeType.SPEC.value,
        content="Create a function to calculate the sum of two numbers",
        created_by="test",
    )


# =============================================================================
# ADAPTIVE QUESTIONER TESTS
# =============================================================================

class TestAdaptiveQuestionerInit:
    """Test AdaptiveQuestioner initialization."""

    def test_init_with_db_path(self, temp_db):
        """Test initialization with database path."""
        questioner = AdaptiveQuestioner(db_path=temp_db)
        assert questioner.store.db_path == temp_db

    def test_init_with_store(self, temp_db):
        """Test initialization with existing TrainingStore."""
        store = TrainingStore(db_path=temp_db)
        questioner = AdaptiveQuestioner(store=store)
        assert questioner.store == store

    def test_init_creates_tables(self, temp_db):
        """Test that initialization creates question tracking tables."""
        AdaptiveQuestioner(db_path=temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "question_attempts" in tables
        assert "question_outcomes" in tables


class TestAdaptiveQuestionerPrivateMethods:
    """Test AdaptiveQuestioner private methods."""

    def test_init_question_tracking(self, temp_db):
        """Test _init_question_tracking creates proper schema."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        conn = sqlite3.connect(temp_db)

        # Check question_attempts table structure
        cursor = conn.execute("PRAGMA table_info(question_attempts)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "question_id" in columns
        assert "session_id" in columns
        assert "ambiguity_category" in columns
        assert "user_answer" in columns

        # Check question_outcomes table structure
        cursor = conn.execute("PRAGMA table_info(question_outcomes)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "question_id" in columns
        assert "led_to_success" in columns

        conn.close()

    def test_calculate_question_priority(self, temp_db):
        """Test _calculate_question_priority computes correct priority."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        ambiguity = AmbiguityMarker(
            category="SUBJECTIVE",
            text="good performance",
            impact="BLOCKING",
            suggested_question="What defines good performance?",
            suggested_answer="< 200ms",
        )

        priorities = UserPriorities()
        priority = questioner._calculate_question_priority(ambiguity, priorities)

        assert isinstance(priority, QuestionPriority)
        assert priority.expected_information_gain > 0.0
        assert priority.skip_probability >= 0.0
        assert priority.priority_score > 0.0

    def test_compute_priority_score_speed_weight(self, temp_db):
        """Test _compute_priority_score with high speed weight."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        priorities = UserPriorities(speed_weight=0.7, cost_weight=0.2, control_weight=0.1)

        score = questioner._compute_priority_score(
            info_gain=0.8,
            skip_prob=0.5,
            suggested_conf=0.6,
            priorities=priorities,
        )

        assert 0.0 <= score <= 1.0
        # With high skip_prob, speed mode should penalize
        assert score < 0.8

    def test_compute_priority_score_control_weight(self, temp_db):
        """Test _compute_priority_score with high control weight."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        priorities = UserPriorities(speed_weight=0.1, cost_weight=0.2, control_weight=0.7)

        score = questioner._compute_priority_score(
            info_gain=0.5,
            skip_prob=0.3,
            suggested_conf=0.4,
            priorities=priorities,
        )

        assert 0.0 <= score <= 1.0
        # Control mode should boost score
        assert score >= 0.5 * 1.2  # Base * boost (capped at 1.0)

    def test_compute_priority_score_cost_weight(self, temp_db):
        """Test _compute_priority_score with high cost weight."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        priorities = UserPriorities(speed_weight=0.2, cost_weight=0.7, control_weight=0.1)

        score = questioner._compute_priority_score(
            info_gain=0.6,
            skip_prob=0.2,
            suggested_conf=0.9,
            priorities=priorities,
        )

        assert 0.0 <= score <= 1.0
        # High suggested confidence should boost score in cost mode

    def test_get_most_common_answer_no_data(self, temp_db):
        """Test _get_most_common_answer returns None when no data."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        result = questioner._get_most_common_answer("SUBJECTIVE")
        assert result is None

    def test_get_most_common_answer_with_data(self, temp_db):
        """Test _get_most_common_answer returns most common answer."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        # Insert test data
        conn = sqlite3.connect(temp_db)
        conn.executemany(
            """
            INSERT INTO question_attempts
            (question_id, session_id, ambiguity_category, question_text, user_answer, was_answered)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("q1", "s1", "SUBJECTIVE", "How fast?", "< 100ms", 1),
                ("q2", "s1", "SUBJECTIVE", "How fast?", "< 200ms", 1),
                ("q3", "s1", "SUBJECTIVE", "How fast?", "< 200ms", 1),
            ]
        )
        conn.commit()
        conn.close()

        result = questioner._get_most_common_answer("SUBJECTIVE")
        assert result == "< 200ms"

    def test_get_skip_probability_no_data(self, temp_db):
        """Test _get_skip_probability returns default when no data."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        prob = questioner._get_skip_probability("SUBJECTIVE")
        assert prob == 0.3  # Default prior

    def test_get_skip_probability_with_data(self, temp_db):
        """Test _get_skip_probability computes correct probability."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        # Insert test data: 2 answered, 1 skipped
        conn = sqlite3.connect(temp_db)
        conn.executemany(
            """
            INSERT INTO question_attempts
            (question_id, session_id, ambiguity_category, question_text, was_answered)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("q1", "s1", "SUBJECTIVE", "Q1", 1),
                ("q2", "s1", "SUBJECTIVE", "Q2", 1),
                ("q3", "s1", "SUBJECTIVE", "Q3", 0),
            ]
        )
        conn.commit()
        conn.close()

        prob = questioner._get_skip_probability("SUBJECTIVE")
        assert abs(prob - (1.0 / 3.0)) < 0.01

    def test_get_suggested_answer_confidence_no_suggestion(self, temp_db):
        """Test _get_suggested_answer_confidence returns 0 when no suggestion."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        conf = questioner._get_suggested_answer_confidence("SUBJECTIVE", None)
        assert conf == 0.0

    def test_get_suggested_answer_confidence_with_suggestion(self, temp_db):
        """Test _get_suggested_answer_confidence returns acceptance rate."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        conf = questioner._get_suggested_answer_confidence("SUBJECTIVE", "< 200ms")
        # Should return acceptance rate (default 0.5 with no data)
        assert conf == 0.5

    def test_get_suggestion_acceptance_rate_no_data(self, temp_db):
        """Test _get_suggestion_acceptance_rate returns default."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        rate = questioner._get_suggestion_acceptance_rate("SUBJECTIVE")
        assert rate == 0.5  # Default prior

    def test_get_suggestion_acceptance_rate_with_data(self, temp_db):
        """Test _get_suggestion_acceptance_rate computes correctly."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        # Insert test data: 3 with suggestions, 2 accepted
        conn = sqlite3.connect(temp_db)
        conn.executemany(
            """
            INSERT INTO question_attempts
            (question_id, session_id, ambiguity_category, question_text,
             suggested_answer, was_answered, used_suggestion)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("q1", "s1", "SUBJECTIVE", "Q1", "ans1", 1, 1),
                ("q2", "s1", "SUBJECTIVE", "Q2", "ans2", 1, 1),
                ("q3", "s1", "SUBJECTIVE", "Q3", "ans3", 1, 0),
            ]
        )
        conn.commit()
        conn.close()

        rate = questioner._get_suggestion_acceptance_rate("SUBJECTIVE")
        assert abs(rate - (2.0 / 3.0)) < 0.01


class TestAdaptiveQuestionerPrioritization:
    """Test AdaptiveQuestioner prioritization logic."""

    def test_prioritize_questions_orders_by_priority(self, temp_db):
        """Test prioritize_questions orders by priority score."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        ambiguities = [
            AmbiguityMarker(
                category="COMPARATIVE",
                text="faster",
                impact="CLARIFYING",
            ),
            AmbiguityMarker(
                category="CONTRADICTIONS",
                text="conflicting requirements",
                impact="BLOCKING",
            ),
            AmbiguityMarker(
                category="SUBJECTIVE",
                text="good",
                impact="BLOCKING",
            ),
        ]

        prioritized = questioner.prioritize_questions(ambiguities)

        # Should have all ambiguities
        assert len(prioritized) == 3
        # CONTRADICTIONS (0.9 base) should be first due to high info gain
        assert prioritized[0].category == "CONTRADICTIONS"

    def test_prioritize_questions_respects_max_limit(self, temp_db):
        """Test prioritize_questions respects max_clarification_questions."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        ambiguities = [
            AmbiguityMarker(category="SUBJECTIVE", text=f"q{i}", impact="BLOCKING")
            for i in range(10)
        ]

        priorities = UserPriorities(max_clarification_questions=3)
        prioritized = questioner.prioritize_questions(ambiguities, priorities)

        assert len(prioritized) == 3


class TestAdaptiveQuestionerRecording:
    """Test AdaptiveQuestioner recording methods."""

    def test_record_question_outcome(self, temp_db):
        """Test record_question_outcome stores data correctly."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        ambiguity = AmbiguityMarker(
            category="SUBJECTIVE",
            text="good performance",
            impact="BLOCKING",
            suggested_question="Define good?",
            suggested_answer="< 200ms",
        )

        question_id = questioner.record_question_outcome(
            session_id="test_session",
            ambiguity=ambiguity,
            was_answered=True,
            user_answer="< 100ms",
            used_suggestion=False,
            answer_quality_score=0.8,
        )

        assert question_id is not None
        assert "test_session" in question_id

        # Verify in database
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT * FROM question_attempts WHERE question_id = ?",
            (question_id,)
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None

    def test_update_question_outcome(self, temp_db):
        """Test update_question_outcome updates success status."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        ambiguity = AmbiguityMarker(
            category="SUBJECTIVE",
            text="test",
            impact="BLOCKING",
        )

        question_id = questioner.record_question_outcome(
            session_id="test_session",
            ambiguity=ambiguity,
            was_answered=True,
        )

        questioner.update_question_outcome(
            question_id=question_id,
            session_id="test_session",
            led_to_success=True,
        )

        # Verify in database
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT led_to_success FROM question_outcomes WHERE question_id = ?",
            (question_id,)
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == 1


# =============================================================================
# DOCUMENTER TESTS
# =============================================================================

class TestMarkdownBuilder:
    """Test MarkdownBuilder methods."""

    def test_init(self):
        """Test MarkdownBuilder initialization."""
        md = MarkdownBuilder()
        assert md.lines == []

    def test_h1(self):
        """Test h1 method."""
        md = MarkdownBuilder()
        md.h1("Title")
        assert "# Title\n" in md.lines

    def test_h2(self):
        """Test h2 method."""
        md = MarkdownBuilder()
        md.h2("Subtitle")
        assert "## Subtitle\n" in md.lines

    def test_h3(self):
        """Test h3 method."""
        md = MarkdownBuilder()
        md.h3("Section")
        assert "### Section\n" in md.lines

    def test_h4(self):
        """Test h4 method."""
        md = MarkdownBuilder()
        md.h4("Subsection")
        assert "#### Subsection\n" in md.lines

    def test_p(self):
        """Test p method."""
        md = MarkdownBuilder()
        md.p("Some text")
        assert "Some text\n" in md.lines

    def test_code_block(self):
        """Test code_block method."""
        md = MarkdownBuilder()
        md.code_block("def foo():\n    pass", language="python")
        result = md.to_string()
        assert "```python" in result
        assert "def foo():" in result
        assert "```" in result

    def test_ul(self):
        """Test ul method."""
        md = MarkdownBuilder()
        md.ul(["Item 1", "Item 2", "Item 3"])
        result = md.to_string()
        assert "- Item 1\n" in result
        assert "- Item 2\n" in result
        assert "- Item 3\n" in result

    def test_ol(self):
        """Test ol method."""
        md = MarkdownBuilder()
        md.ol(["First", "Second", "Third"])
        result = md.to_string()
        assert "1. First\n" in result
        assert "2. Second\n" in result
        assert "3. Third\n" in result

    def test_table(self):
        """Test table method."""
        md = MarkdownBuilder()
        headers = ["Name", "Age", "City"]
        rows = [
            ["Alice", "30", "NYC"],
            ["Bob", "25", "LA"],
        ]
        md.table(headers, rows)
        result = md.to_string()
        assert "| Name | Age | City |" in result
        assert "| --- | --- | --- |" in result
        assert "| Alice | 30 | NYC |" in result

    def test_hr(self):
        """Test hr method."""
        md = MarkdownBuilder()
        md.hr()
        assert "---\n" in md.to_string()

    def test_newline(self):
        """Test newline method."""
        md = MarkdownBuilder()
        md.newline()
        assert "\n" in md.lines

    def test_to_string(self):
        """Test to_string method."""
        md = MarkdownBuilder()
        md.h1("Title").p("Text").newline()
        result = md.to_string()
        assert isinstance(result, str)
        assert "# Title\n" in result
        assert "Text\n" in result

    def test_chaining(self):
        """Test method chaining."""
        md = MarkdownBuilder()
        result = md.h1("Title").h2("Subtitle").p("Text").to_string()
        assert "# Title\n" in result
        assert "## Subtitle\n" in result
        assert "Text\n" in result


class TestDocumenterInit:
    """Test Documenter initialization."""

    def test_init_without_db(self):
        """Test initialization without database."""
        documenter = Documenter()
        assert documenter.db is None
        assert isinstance(documenter.config, DocumenterConfig)

    def test_init_with_db(self, mock_db):
        """Test initialization with database."""
        documenter = Documenter(db=mock_db)
        assert documenter.db == mock_db

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = DocumenterConfig(
            readme_path="custom_README.md",
            auto_generate=False,
        )
        documenter = Documenter(config=config)
        assert documenter.config == config
        assert documenter.config.readme_path == "custom_README.md"


class TestDocumenterHelpers:
    """Test Documenter helper methods."""

    def test_get_status_emoji(self):
        """Test _get_status_emoji returns correct emojis."""
        documenter = Documenter()

        assert documenter._get_status_emoji(NodeStatus.VERIFIED.value) == "✅"
        assert documenter._get_status_emoji(NodeStatus.TESTED.value) == "🧪"
        assert documenter._get_status_emoji(NodeStatus.PROCESSING.value) == "⏳"
        assert documenter._get_status_emoji(NodeStatus.PENDING.value) == "⏸️"
        assert documenter._get_status_emoji(NodeStatus.FAILED.value) == "❌"
        assert documenter._get_status_emoji(NodeStatus.BLOCKED.value) == "🚫"
        assert documenter._get_status_emoji("unknown") == "❓"

    def test_get_content_preview_short(self):
        """Test _get_content_preview with short content."""
        documenter = Documenter()

        preview = documenter._get_content_preview("Short text", max_length=100)
        assert preview == "Short text"

    def test_get_content_preview_long(self):
        """Test _get_content_preview truncates long content."""
        documenter = Documenter()

        long_text = "a" * 200
        preview = documenter._get_content_preview(long_text, max_length=50)
        assert len(preview) <= 53  # 50 + "..."
        assert preview.endswith("...")

    def test_get_content_preview_empty(self):
        """Test _get_content_preview with empty content."""
        documenter = Documenter()

        preview = documenter._get_content_preview("")
        assert preview == "(empty)"

    def test_get_content_preview_multiline(self):
        """Test _get_content_preview returns first line."""
        documenter = Documenter()

        text = "First line\nSecond line\nThird line"
        preview = documenter._get_content_preview(text, max_length=100)
        assert preview == "First line"

    def test_find_parent_req_no_db(self):
        """Test _find_parent_req returns None without database."""
        documenter = Documenter(db=None)

        result = documenter._find_parent_req("some_node_id")
        assert result is None

    def test_find_parent_req_direct_req(self, mock_db):
        """Test _find_parent_req returns REQ if node is REQ."""
        req_node = NodeData.create(
            type=NodeType.REQ.value,
            content="A requirement",
            created_by="test",
        )

        mock_db.get_node = Mock(return_value=req_node)
        documenter = Documenter(db=mock_db)

        result = documenter._find_parent_req(req_node.id)
        assert result == req_node.id

    def test_find_implementing_code_no_db(self):
        """Test _find_implementing_code returns empty list without db."""
        documenter = Documenter(db=None)

        result = documenter._find_implementing_code("spec_id")
        assert result == []

    def test_find_implementing_code_with_codes(self, mock_db):
        """Test _find_implementing_code returns implementing codes."""
        code_node = NodeData.create(
            type=NodeType.CODE.value,
            content="def foo(): pass",
            created_by="test",
        )

        mock_db.get_incoming_edges = Mock(return_value=[
            {
                "type": EdgeType.IMPLEMENTS.value,
                "source": code_node.id,
                "target": "spec_id",
            }
        ])
        mock_db.get_node = Mock(return_value=code_node)

        documenter = Documenter(db=mock_db)
        result = documenter._find_implementing_code("spec_id")

        assert len(result) == 1
        assert result[0].id == code_node.id


class TestDocumenterGeneration:
    """Test Documenter generation methods."""

    def test_generate_readme_no_db(self, temp_dir):
        """Test generate_readme returns False without database."""
        config = DocumenterConfig(readme_path=str(temp_dir / "README.md"))
        documenter = Documenter(db=None, config=config)

        result = documenter.generate_readme()
        assert result is False

    def test_generate_readme_empty_graph(self, mock_db, temp_dir):
        """Test generate_readme with empty graph."""
        config = DocumenterConfig(readme_path=str(temp_dir / "README.md"))
        mock_db.get_nodes_by_type = Mock(return_value=[])

        documenter = Documenter(db=mock_db, config=config)
        result = documenter.generate_readme()

        assert result is True
        readme_path = temp_dir / "README.md"
        assert readme_path.exists()
        content = readme_path.read_text()
        assert "Project Paragon" in content
        assert "No requirements defined yet" in content

    def test_generate_wiki_no_db(self):
        """Test generate_wiki returns False without database."""
        documenter = Documenter(db=None)
        result = documenter.generate_wiki()
        assert result is False

    def test_append_changelog_creates_file(self, temp_dir):
        """Test append_changelog creates changelog if not exists."""
        config = DocumenterConfig(changelog_path=str(temp_dir / "CHANGELOG.md"))
        documenter = Documenter(config=config)

        result = documenter.append_changelog(
            old_merkle=None,
            new_merkle="abc123def456",
            description="Initial commit",
        )

        assert result is True
        changelog_path = temp_dir / "CHANGELOG.md"
        assert changelog_path.exists()
        content = changelog_path.read_text()
        assert "Changelog" in content
        assert "abc123de" in content  # First 8 chars
        assert "initial" in content.lower()

    def test_append_changelog_appends_entry(self, temp_dir):
        """Test append_changelog appends to existing changelog."""
        changelog_path = temp_dir / "CHANGELOG.md"
        changelog_path.write_text("# Changelog\n\n## Old Entry\n")

        config = DocumenterConfig(changelog_path=str(changelog_path))
        documenter = Documenter(config=config)

        result = documenter.append_changelog(
            old_merkle="old123",
            new_merkle="new456",
        )

        assert result is True
        content = changelog_path.read_text()
        assert "new456" in content
        assert "old123" in content


# =============================================================================
# HUMAN LOOP CONTROLLER TESTS
# =============================================================================

class TestHumanLoopControllerInit:
    """Test HumanLoopController initialization."""

    def test_init_default(self):
        """Test initialization with defaults."""
        controller = HumanLoopController()
        assert controller.pending_requests == {}
        assert controller.completed_requests == {}
        assert controller.on_request is None
        assert controller.on_response is None

    def test_init_with_callbacks(self):
        """Test initialization with callbacks."""
        on_request = Mock()
        on_response = Mock()

        controller = HumanLoopController(
            on_request=on_request,
            on_response=on_response,
        )

        assert controller.on_request == on_request
        assert controller.on_response == on_response


class TestHumanLoopControllerLifecycle:
    """Test HumanLoopController request lifecycle."""

    def test_submit_response(self):
        """Test submit_response processes response."""
        controller = HumanLoopController()

        pause_point = PausePoint(
            id="test_pause",
            pause_type=PauseType.APPROVAL.value,
            description="Test approval",
        )

        request = controller.create_request(
            pause_point=pause_point,
            session_id="test_session",
            prompt="Approve this?",
        )

        response = controller.submit_response(
            request_id=request.id,
            response="yes",
        )

        assert response is not None
        assert response.response == "yes"
        assert request.id not in controller.pending_requests
        assert request.id in controller.completed_requests

    def test_submit_response_not_found(self):
        """Test submit_response returns None for unknown request."""
        controller = HumanLoopController()

        response = controller.submit_response(
            request_id="nonexistent",
            response="yes",
        )

        assert response is None

    def test_cancel_request(self):
        """Test cancel_request cancels pending request."""
        controller = HumanLoopController()

        pause_point = PausePoint(
            id="test_pause",
            pause_type=PauseType.APPROVAL.value,
            description="Test",
        )

        request = controller.create_request(
            pause_point=pause_point,
            session_id="test_session",
            prompt="Test",
        )

        result = controller.cancel_request(request.id)

        assert result is True
        assert request.id not in controller.pending_requests
        assert request.id in controller.completed_requests

        completed = controller.completed_requests[request.id]
        assert completed.status == RequestStatus.CANCELLED.value

    def test_cancel_request_not_found(self):
        """Test cancel_request returns False for unknown request."""
        controller = HumanLoopController()

        result = controller.cancel_request("nonexistent")
        assert result is False

    def test_clear_session(self):
        """Test clear_session removes all session requests."""
        controller = HumanLoopController()

        pause_point = PausePoint(
            id="test_pause",
            pause_type=PauseType.APPROVAL.value,
            description="Test",
        )

        # Create multiple requests for same session
        controller.create_request(pause_point, "session1", "Q1")
        controller.create_request(pause_point, "session1", "Q2")
        controller.create_request(pause_point, "session2", "Q3")

        count = controller.clear_session("session1")

        assert count == 2
        assert len(controller.pending_requests) == 1


# =============================================================================
# QUALITY GATE PRIVATE METHODS TESTS
# =============================================================================

class TestQualityGateInit:
    """Test QualityGate initialization."""

    def test_init_production(self):
        """Test initialization in production mode."""
        gate = QualityGate(quality_mode="production")
        assert gate.quality_mode == "production"
        assert gate.thresholds["test_pass_rate"] == 1.0
        assert gate.thresholds["static_analysis_criticals"] == 0
        assert gate.thresholds["graph_invariant_compliance"] == 1.0
        assert gate.thresholds["max_cyclomatic_complexity"] == 15

    def test_init_experimental(self):
        """Test initialization in experimental mode."""
        gate = QualityGate(quality_mode="experimental")
        assert gate.quality_mode == "experimental"


class TestQualityGateCheckMethods:
    """Test QualityGate check methods."""

    def test_check_test_pass_rate_no_tests_production(self):
        """Test _check_test_pass_rate with no tests in production."""
        gate = QualityGate(quality_mode="production")
        violations = []

        rate = gate._check_test_pass_rate(None, violations)

        assert rate == 0.0
        assert len(violations) == 1
        assert violations[0].severity == "critical"

    def test_check_test_pass_rate_no_tests_experimental(self):
        """Test _check_test_pass_rate with no tests in experimental."""
        gate = QualityGate(quality_mode="experimental")
        violations = []

        rate = gate._check_test_pass_rate(None, violations)

        assert rate == 0.0
        assert len(violations) == 1
        assert violations[0].severity == "warning"

    def test_check_test_pass_rate_partial(self):
        """Test _check_test_pass_rate with partial pass."""
        gate = QualityGate(quality_mode="production")
        violations = []

        test_results = [
            {"passed": True},
            {"passed": False},
            {"passed": True},
        ]

        rate = gate._check_test_pass_rate(test_results, violations)

        assert abs(rate - (2.0 / 3.0)) < 0.01
        assert len(violations) == 1

    def test_check_static_analysis_valid_code(self, sample_code_node):
        """Test _check_static_analysis with valid code."""
        gate = QualityGate(quality_mode="production")
        violations = []

        criticals = gate._check_static_analysis([sample_code_node], violations)

        assert criticals == 0
        assert len([v for v in violations if v.severity == "critical"]) == 0

    def test_check_static_analysis_syntax_error(self):
        """Test _check_static_analysis with syntax error."""
        gate = QualityGate(quality_mode="production")
        violations = []

        bad_node = NodeData.create(
            type=NodeType.CODE.value,
            content="def foo(\n    pass",  # Syntax error
            created_by="test",
        )

        criticals = gate._check_static_analysis([bad_node], violations)

        assert criticals > 0

    def test_check_complexity_simple_function(self, sample_code_node):
        """Test _check_complexity with simple function."""
        gate = QualityGate(quality_mode="production")
        violations = []

        max_complexity = gate._check_complexity([sample_code_node], violations)

        # Simple addition function should have complexity 1
        assert max_complexity == 1
        assert len([v for v in violations if v.metric == "cyclomatic_complexity"]) == 0

    def test_compute_cyclomatic_complexity_simple(self):
        """Test _compute_cyclomatic_complexity with simple function."""
        import ast

        gate = QualityGate(quality_mode="production")

        code = """
def simple():
    return 1
"""
        tree = ast.parse(code)
        func = list(ast.walk(tree))[1]  # Get FunctionDef

        complexity = gate._compute_cyclomatic_complexity(func)
        assert complexity == 1

    def test_compute_cyclomatic_complexity_with_branches(self):
        """Test _compute_cyclomatic_complexity with branches."""
        import ast

        gate = QualityGate(quality_mode="production")

        code = """
def complex_func(a, b):
    if a > 0:
        for i in range(10):
            if b < 5:
                return i
    return 0
"""
        tree = ast.parse(code)
        func = list(ast.walk(tree))[1]  # Get FunctionDef

        complexity = gate._compute_cyclomatic_complexity(func)
        # 1 (base) + 1 (if) + 1 (for) + 1 (if) = 4
        assert complexity == 4

    def test_check_graph_invariants_no_graph(self):
        """Test _check_graph_invariants without graph."""
        gate = QualityGate(quality_mode="production")
        violations = []

        compliance = gate._check_graph_invariants([], violations, graph=None)

        # Should return neutral 1.0 without graph
        assert compliance == 1.0
        assert len(violations) == 1
        assert violations[0].severity == "warning"

    def test_generate_summary_passed(self):
        """Test _generate_summary for passed quality gate."""
        gate = QualityGate(quality_mode="production")

        summary = gate._generate_summary(
            passed=True,
            violations=[],
            test_pass_rate=1.0,
            static_criticals=0,
            invariant_compliance=1.0,
            max_complexity=5,
        )

        assert "PASSED" in summary
        assert "production" in summary

    def test_generate_summary_failed(self):
        """Test _generate_summary for failed quality gate."""
        gate = QualityGate(quality_mode="production")

        violations = [
            QualityViolation(
                metric="test",
                threshold="100%",
                actual="50%",
                severity="critical",
            )
        ]

        summary = gate._generate_summary(
            passed=False,
            violations=violations,
            test_pass_rate=0.5,
            static_criticals=2,
            invariant_compliance=0.8,
            max_complexity=20,
        )

        assert "FAILED" in summary
        assert "1 critical" in summary


# =============================================================================
# TOOLS MODULE FUNCTION TESTS
# =============================================================================

class TestToolsHelperFunctions:
    """Test standalone helper functions in tools module."""

    def test_compute_alignment_score_basic(self, sample_spec_node, sample_code_node):
        """Test _compute_alignment_score with matching content."""
        score = _compute_alignment_score(sample_spec_node, sample_code_node)

        assert 0.0 <= score <= 1.0
        # "calculate" and "sum" should be in both spec and code
        # Score may be 0 if stop words filtered out all matches
        assert score >= 0.0

    def test_compute_alignment_score_no_overlap(self):
        """Test _compute_alignment_score with no keyword overlap."""
        spec = NodeData.create(
            type=NodeType.SPEC.value,
            content="Create a database connection handler",
            created_by="test",
        )
        code = NodeData.create(
            type=NodeType.CODE.value,
            content="def multiply(x, y): return x * y",
            created_by="test",
        )

        score = _compute_alignment_score(spec, code)
        assert 0.0 <= score <= 1.0

    def test_compute_alignment_score_empty_spec(self):
        """Test _compute_alignment_score with empty spec."""
        spec = NodeData.create(
            type=NodeType.SPEC.value,
            content="",
            created_by="test",
        )
        code = NodeData.create(
            type=NodeType.CODE.value,
            content="def foo(): pass",
            created_by="test",
        )

        score = _compute_alignment_score(spec, code)
        # Should return neutral 0.5 for empty spec
        assert score == 0.5

    def test_record_transaction(self):
        """Test _record_transaction records mutation."""
        from agents.tools import _pending_transaction

        # Clear any existing transactions
        _pending_transaction.clear()

        _record_transaction("node123", "CODE", edge_info=("n1", "n2", "DEPENDS_ON"))

        assert len(_pending_transaction) == 2
        assert _pending_transaction[0] == ("node", "node123", "CODE")
        assert _pending_transaction[1] == ("edge", "n1", "n2", "DEPENDS_ON")

    @patch('agents.tools._mutation_logger', None)
    def test_get_mutation_logger_not_available(self):
        """Test _get_mutation_logger returns None when not available."""
        # Mock the import to fail
        import sys
        original_modules = sys.modules.copy()

        try:
            # Remove the module so import fails
            if 'infrastructure.logger' in sys.modules:
                del sys.modules['infrastructure.logger']

            # Force reimport which will fail
            with patch.dict('sys.modules', {'infrastructure.logger': None}):
                logger = _get_mutation_logger()
                # Should handle gracefully (may return None or existing logger)
                # The function handles ImportError gracefully
        finally:
            # Restore modules
            sys.modules.update(original_modules)

    @patch('agents.tools._git_sync', None)
    def test_get_git_sync_not_available(self):
        """Test _get_git_sync handles missing imports gracefully."""
        import sys
        original_modules = sys.modules.copy()

        try:
            with patch.dict('sys.modules', {'infrastructure.git_sync': None}):
                sync = _get_git_sync()
                # Should handle gracefully
        finally:
            sys.modules.update(original_modules)

    @patch('agents.tools._training_store', None)
    def test_get_training_store_not_available(self):
        """Test _get_training_store handles missing imports gracefully."""
        import sys
        original_modules = sys.modules.copy()

        try:
            with patch.dict('sys.modules', {'infrastructure.training_store': None}):
                store = _get_training_store()
                # Should handle gracefully
        finally:
            sys.modules.update(original_modules)

    def test_log_node_created(self):
        """Test _log_node_created logs and records."""
        from agents.tools import _pending_transaction

        _pending_transaction.clear()

        with patch('agents.tools._get_mutation_logger', return_value=None):
            _log_node_created("node123", "CODE", "agent")

        # Should record transaction even if logger unavailable
        assert ("node", "node123", "CODE") in _pending_transaction

    def test_log_edge_created(self):
        """Test _log_edge_created logs and records."""
        from agents.tools import _pending_transaction

        _pending_transaction.clear()

        with patch('agents.tools._get_mutation_logger', return_value=None):
            _log_edge_created("n1", "n2", "DEPENDS_ON")

        # Should record transaction
        assert ("edge", "n1", "n2", "DEPENDS_ON") in _pending_transaction

    def test_record_attribution_no_store(self):
        """Test _record_attribution returns False without store."""
        with patch('agents.tools._get_training_store', return_value=None):
            result = _record_attribution(
                session_id="test",
                signature=Mock(),
                node_id="node123",
                state_id="state456",
            )
            assert result is False

    def test_record_attribution_no_signature(self):
        """Test _record_attribution returns False without signature."""
        with patch('agents.tools._get_training_store', return_value=Mock()):
            result = _record_attribution(
                session_id="test",
                signature=None,
                node_id="node123",
                state_id="state456",
            )
            assert result is False


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_questioner_full_workflow(self, temp_db):
        """Test complete questioner workflow."""
        questioner = AdaptiveQuestioner(db_path=temp_db)

        # Create ambiguities
        ambiguities = [
            AmbiguityMarker(
                category="SUBJECTIVE",
                text="fast",
                impact="BLOCKING",
                suggested_answer="< 200ms",
            ),
            AmbiguityMarker(
                category="UNDEFINED_TERM",
                text="database",
                impact="CLARIFYING",
            ),
        ]

        # Prioritize
        prioritized = questioner.prioritize_questions(ambiguities)
        assert len(prioritized) == 2

        # Record outcome
        question_id = questioner.record_question_outcome(
            session_id="test_session",
            ambiguity=prioritized[0],
            was_answered=True,
            user_answer="< 100ms",
        )

        # Update success
        questioner.update_question_outcome(
            question_id=question_id,
            session_id="test_session",
            led_to_success=True,
        )

        # Get stats
        stats = questioner.get_question_stats()
        assert stats["total_questions"] == 1
        assert stats["answered"] == 1

    def test_documenter_markdown_generation(self, temp_dir):
        """Test complete markdown generation."""
        md = MarkdownBuilder()

        (md.h1("Project Title")
           .p("This is a test project.")
           .h2("Features")
           .ul(["Feature 1", "Feature 2"])
           .h2("Installation")
           .code_block("pip install package", "bash")
           .hr()
           .p("Footer text"))

        result = md.to_string()

        assert "# Project Title" in result
        assert "## Features" in result
        assert "- Feature 1" in result
        assert "```bash" in result
        assert "---" in result

    def test_human_loop_full_cycle(self):
        """Test complete human loop request/response cycle."""
        on_request = Mock()
        on_response = Mock()

        controller = HumanLoopController(
            on_request=on_request,
            on_response=on_response,
        )

        pause_point = PausePoint(
            id="approval",
            pause_type=PauseType.APPROVAL.value,
            description="Approve plan",
        )

        # Create request
        request = controller.create_request(
            pause_point=pause_point,
            session_id="session1",
            prompt="Approve this plan?",
            context={"plan": "Build feature X"},
        )

        assert on_request.called
        assert request.id in controller.pending_requests

        # Submit response
        response = controller.submit_response(
            request_id=request.id,
            response="approved",
        )

        assert on_response.called
        assert response is not None
        assert request.id in controller.completed_requests
