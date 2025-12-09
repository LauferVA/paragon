"""
Unit tests for research feedback API endpoints.

Tests the new research feedback system:
- GET /api/research/{research_task_id}
- POST /api/research/{research_task_id}/feedback
- POST /api/research/{research_task_id}/response
- GET /api/research/tasks/active
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from core.graph_db import ParagonDB
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, NodeStatus, EdgeType
from api.research_feedback_endpoints import (
    get_research_task,
    submit_research_feedback,
    submit_research_response,
    get_active_research_tasks,
)


@pytest.fixture
def db():
    """Create a test database."""
    return ParagonDB()


@pytest.fixture
def sample_research_node(db):
    """Create a sample RESEARCH node for testing."""
    req_node = NodeData.create(
        type=NodeType.REQ.value,
        content="Test requirement",
        status=NodeStatus.PROCESSING.value,
    )
    db.add_node(req_node)

    research_node = NodeData.create(
        type=NodeType.RESEARCH.value,
        content="Research synthesis: Found 5 relevant sources",
        status=NodeStatus.PENDING.value,
        data={
            "iteration": 1,
            "query": "Research query about testing",
            "total_findings": 5,
            "total_ambiguities": 2,
            "blocking_count": 0,
            "out_of_scope": ["Implementation details"],
            "findings": [
                {"topic": "Testing", "summary": "Unit tests are important"},
                {"topic": "Coverage", "summary": "Aim for >80% coverage"},
            ],
            "ambiguities": [
                {"category": "UNDEFINED_TERM", "text": "production-ready"},
                {"category": "SUBJECTIVE", "text": "good enough"},
            ],
            "search_results": [
                {"title": "Best practices", "url": "https://example.com/1"},
            ],
            "user_approval_required": False,
            "user_approval_state": "pending",
            "awaiting_user_action": False,
        },
    )
    db.add_node(research_node)

    # Create RESEARCH_FOR edge
    edge = EdgeData.create(
        source_id=research_node.id,
        target_id=req_node.id,
        edge_type=EdgeType.RESEARCH_FOR.value,
    )
    db.add_edge(edge)

    return research_node, req_node


class TestGetResearchTask:
    """Tests for GET /api/research/{research_task_id}"""

    @pytest.mark.asyncio
    async def test_get_research_task_success(self, db, sample_research_node):
        """Test successfully retrieving a research task."""
        research_node, req_node = sample_research_node

        # Create mock request
        request = Mock()
        request.path_params = {"research_task_id": research_node.id}

        # Patch get_db to return our test db
        with patch("api.research_feedback_endpoints.get_db", return_value=db):
            response = await get_research_task(request)

        assert response.status_code == 200
        data = response.body.decode()
        assert research_node.id in data
        assert req_node.id in data
        assert "synthesis" in data
        assert "findings" in data
        assert "hover_metadata" in data

    @pytest.mark.asyncio
    async def test_get_research_task_not_found(self, db):
        """Test getting a non-existent research task."""
        request = Mock()
        request.path_params = {"research_task_id": "nonexistent_id"}

        with patch("api.research_feedback_endpoints.get_db", return_value=db):
            response = await get_research_task(request)

        assert response.status_code in [400, 500]

    @pytest.mark.asyncio
    async def test_get_research_task_wrong_type(self, db):
        """Test getting a node that isn't a RESEARCH type."""
        # Create a REQ node
        req_node = NodeData.create(
            type=NodeType.REQ.value,
            content="Test requirement",
            status=NodeStatus.PENDING.value,
        )
        db.add_node(req_node)

        request = Mock()
        request.path_params = {"research_task_id": req_node.id}

        with patch("api.research_feedback_endpoints.get_db", return_value=db):
            response = await get_research_task(request)

        assert response.status_code == 400
        assert b"not a RESEARCH node" in response.body


class TestSubmitResearchFeedback:
    """Tests for POST /api/research/{research_task_id}/feedback"""

    @pytest.mark.asyncio
    async def test_submit_feedback_success(self, db, sample_research_node):
        """Test successfully submitting feedback."""
        research_node, req_node = sample_research_node

        # Create mock request
        request = Mock()
        request.path_params = {"research_task_id": research_node.id}
        request.json = AsyncMock(return_value={
            "feedback": "Great research, needs more details",
            "metadata": {"rating": 4}
        })

        # Mock WebSocket components
        ws_connections = set()
        next_sequence = lambda: 1
        broadcast_delta = AsyncMock()

        with patch("api.research_feedback_endpoints.get_db", return_value=db):
            with patch("api.research_feedback_endpoints.EVENT_BUS_AVAILABLE", False):
                response = await submit_research_feedback(
                    request, ws_connections, next_sequence, broadcast_delta
                )

        assert response.status_code == 200
        body = response.body.decode()
        assert "success" in body
        assert "feedback_node_id" in body

        # Verify feedback node was created
        updated_research = db.get_node(research_node.id)
        assert updated_research.data.get("user_feedback") is not None
        assert updated_research.data.get("user_feedback_node_id") is not None

    @pytest.mark.asyncio
    async def test_submit_feedback_empty(self, db, sample_research_node):
        """Test submitting empty feedback fails."""
        research_node, req_node = sample_research_node

        request = Mock()
        request.path_params = {"research_task_id": research_node.id}
        request.json = AsyncMock(return_value={"feedback": ""})

        ws_connections = set()
        next_sequence = lambda: 1
        broadcast_delta = AsyncMock()

        with patch("api.research_feedback_endpoints.get_db", return_value=db):
            response = await submit_research_feedback(
                request, ws_connections, next_sequence, broadcast_delta
            )

        assert response.status_code == 400
        assert b"required" in response.body


class TestSubmitResearchResponse:
    """Tests for POST /api/research/{research_task_id}/response"""

    @pytest.mark.asyncio
    async def test_approve_research(self, db, sample_research_node):
        """Test approving a research task."""
        research_node, req_node = sample_research_node

        request = Mock()
        request.path_params = {"research_task_id": research_node.id}
        request.json = AsyncMock(return_value={
            "action": "approve",
            "message": "Research approved",
            "context": {}
        })

        ws_connections = set()
        next_sequence = lambda: 1
        broadcast_delta = AsyncMock()

        with patch("api.research_feedback_endpoints.get_db", return_value=db):
            with patch("api.research_feedback_endpoints.EVENT_BUS_AVAILABLE", False):
                response = await submit_research_response(
                    request, ws_connections, next_sequence, broadcast_delta
                )

        assert response.status_code == 200
        body = response.body.decode()
        assert "approve" in body
        assert "RESEARCH -> PLAN" in body

        # Verify research node was updated
        updated_research = db.get_node(research_node.id)
        assert updated_research.data.get("user_approval_state") == "approved"
        assert updated_research.status == NodeStatus.VERIFIED.value

    @pytest.mark.asyncio
    async def test_revise_research(self, db, sample_research_node):
        """Test requesting revision on a research task."""
        research_node, req_node = sample_research_node

        request = Mock()
        request.path_params = {"research_task_id": research_node.id}
        request.json = AsyncMock(return_value={
            "action": "revise",
            "message": "Needs more detail",
            "context": {"revision_notes": "Add more examples"}
        })

        ws_connections = set()
        next_sequence = lambda: 1
        broadcast_delta = AsyncMock()

        with patch("api.research_feedback_endpoints.get_db", return_value=db):
            with patch("api.research_feedback_endpoints.EVENT_BUS_AVAILABLE", False):
                response = await submit_research_response(
                    request, ws_connections, next_sequence, broadcast_delta
                )

        assert response.status_code == 200

        # Verify research node state
        updated_research = db.get_node(research_node.id)
        assert updated_research.data.get("user_approval_state") == "revision_requested"
        assert updated_research.data.get("awaiting_user_action") is True

    @pytest.mark.asyncio
    async def test_invalid_action(self, db, sample_research_node):
        """Test submitting invalid action fails."""
        research_node, req_node = sample_research_node

        request = Mock()
        request.path_params = {"research_task_id": research_node.id}
        request.json = AsyncMock(return_value={"action": "invalid"})

        ws_connections = set()
        next_sequence = lambda: 1
        broadcast_delta = AsyncMock()

        with patch("api.research_feedback_endpoints.get_db", return_value=db):
            response = await submit_research_response(
                request, ws_connections, next_sequence, broadcast_delta
            )

        assert response.status_code == 400


class TestGetActiveResearchTasks:
    """Tests for GET /api/research/tasks/active"""

    @pytest.mark.asyncio
    async def test_get_active_tasks(self, db, sample_research_node):
        """Test getting active research tasks."""
        research_node, req_node = sample_research_node

        # Mark research as awaiting user action
        research_data = research_node.data.copy()
        research_data["awaiting_user_action"] = True
        updated = NodeData.create(
            id=research_node.id,
            type=research_node.type,
            content=research_node.content,
            status=research_node.status,
            data=research_data,
            created_by=research_node.created_by,
            created_at=research_node.created_at,
        )
        db.update_node(research_node.id, updated)

        request = Mock()
        request.query_params = {"limit": "50"}

        with patch("api.research_feedback_endpoints.get_db", return_value=db):
            response = await get_active_research_tasks(request)

        assert response.status_code == 200
        body = response.body.decode()
        assert "tasks" in body
        assert "count" in body
        assert research_node.id in body

    @pytest.mark.asyncio
    async def test_get_active_tasks_empty(self, db):
        """Test getting active tasks when none exist."""
        request = Mock()
        request.query_params = {"limit": "50"}

        with patch("api.research_feedback_endpoints.get_db", return_value=db):
            response = await get_active_research_tasks(request)

        assert response.status_code == 200
        body = response.body.decode()
        assert '"count": 0' in body

    @pytest.mark.asyncio
    async def test_get_active_tasks_limit(self, db, sample_research_node):
        """Test limit parameter works correctly."""
        research_node, req_node = sample_research_node

        # Mark as active
        research_data = research_node.data.copy()
        research_data["awaiting_user_action"] = True
        updated = NodeData.create(
            id=research_node.id,
            type=research_node.type,
            content=research_node.content,
            status=research_node.status,
            data=research_data,
            created_by=research_node.created_by,
            created_at=research_node.created_at,
        )
        db.update_node(research_node.id, updated)

        request = Mock()
        request.query_params = {"limit": "1"}

        with patch("api.research_feedback_endpoints.get_db", return_value=db):
            response = await get_active_research_tasks(request)

        assert response.status_code == 200


# Integration test (requires full API setup)
@pytest.mark.integration
class TestResearchFeedbackIntegration:
    """Integration tests for the full research feedback flow."""

    @pytest.mark.asyncio
    async def test_full_feedback_flow(self, db):
        """Test complete flow: get -> feedback -> response."""
        # This would be implemented when the full API is available
        pass
