"""
Unit tests for notification API endpoints.

Tests the notification system including:
- Creating notifications
- Retrieving pending notifications
- Marking notifications as read
- Notification data validation
"""
import pytest
from starlette.testclient import TestClient
from core.graph_db import get_db
from core.schemas import NodeData
from core.ontology import NodeType, NodeStatus
from api.routes import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def db():
    """Get database instance and clear it."""
    db_instance = get_db()
    # Clear database before each test
    db_instance._graph.clear()
    db_instance._node_index.clear()
    db_instance._edge_index.clear()
    return db_instance


def test_get_pending_notifications_empty(client, db):
    """Test getting notifications when none exist."""
    response = client.get("/api/notifications/pending")
    assert response.status_code == 200
    data = response.json()
    assert data["notifications"] == []
    assert data["unread_count"] == 0


def test_get_pending_notifications_with_data(client, db):
    """Test getting notifications when some exist."""
    # Create a notification node
    notification = NodeData.create(
        type=NodeType.NOTIFICATION.value,
        content="Test notification message",
        data={
            "notification_type": "test",
            "source_component": "test_suite",
            "target_tabs": ["build", "research"],
            "urgency": "info",
            "related_node_id": None,
            "action_required": False,
            "read_by": [],
        },
        status=NodeStatus.PENDING.value,
        created_by="test",
    )
    db.add_node(notification)

    response = client.get("/api/notifications/pending")
    assert response.status_code == 200
    data = response.json()
    assert len(data["notifications"]) == 1
    assert data["unread_count"] == 1
    assert data["notifications"][0]["notification_id"] == notification.id
    assert data["notifications"][0]["message"] == "Test notification message"
    assert data["notifications"][0]["type"] == "test"
    assert data["notifications"][0]["target_tabs"] == ["build", "research"]
    assert data["notifications"][0]["urgency"] == "info"
    assert data["notifications"][0]["read"] is False


def test_mark_notification_read(client, db):
    """Test marking a notification as read."""
    # Create a notification node
    notification = NodeData.create(
        type=NodeType.NOTIFICATION.value,
        content="Test notification",
        data={
            "notification_type": "test",
            "source_component": "test",
            "target_tabs": ["build"],
            "urgency": "info",
            "related_node_id": None,
            "action_required": False,
            "read_by": [],
        },
        status=NodeStatus.PENDING.value,
        created_by="test",
    )
    db.add_node(notification)

    # Mark as read
    response = client.post(f"/api/notifications/{notification.id}/mark-read")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["notification_id"] == notification.id

    # Verify status changed
    node = db.get_node(notification.id)
    assert node.status == NodeStatus.VERIFIED.value


def test_mark_notification_read_not_found(client, db):
    """Test marking non-existent notification as read."""
    response = client.post("/api/notifications/nonexistent/mark-read")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data


def test_mark_notification_read_wrong_type(client, db):
    """Test marking non-notification node as read."""
    # Create a non-notification node
    node = NodeData.create(
        type=NodeType.CODE.value,
        content="print('hello')",
        created_by="test",
    )
    db.add_node(node)

    response = client.post(f"/api/notifications/{node.id}/mark-read")
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "not a notification" in data["error"]


def test_create_notification_endpoint(client, db):
    """Test creating a notification via API."""
    request_body = {
        "notification_type": "spec_updated",
        "message": "New specification created",
        "target_tabs": ["build", "specification"],
        "urgency": "info",
        "source_component": "orchestrator",
        "action_required": True,
    }

    response = client.post("/api/notifications/create", json=request_body)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert "notification_id" in data

    # Verify notification was created in DB
    notification_id = data["notification_id"]
    node = db.get_node(notification_id)
    assert node is not None
    assert node.type == NodeType.NOTIFICATION.value
    assert node.content == "New specification created"
    assert node.data["notification_type"] == "spec_updated"
    assert node.data["target_tabs"] == ["build", "specification"]
    assert node.data["urgency"] == "info"
    assert node.data["action_required"] is True


def test_create_notification_missing_message(client, db):
    """Test creating notification without message."""
    request_body = {
        "target_tabs": ["build"],
    }

    response = client.post("/api/notifications/create", json=request_body)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "Message is required" in data["error"]


def test_create_notification_missing_target_tabs(client, db):
    """Test creating notification without target tabs."""
    request_body = {
        "message": "Test message",
    }

    response = client.post("/api/notifications/create", json=request_body)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "target_tabs is required" in data["error"]


def test_create_notification_invalid_urgency(client, db):
    """Test creating notification with invalid urgency."""
    request_body = {
        "message": "Test message",
        "target_tabs": ["build"],
        "urgency": "invalid_urgency",
    }

    response = client.post("/api/notifications/create", json=request_body)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "urgency must be one of" in data["error"]


def test_create_notification_with_related_node(client, db):
    """Test creating notification linked to another node."""
    # Create a related node
    related_node = NodeData.create(
        type=NodeType.SPEC.value,
        content="Test spec",
        created_by="test",
    )
    db.add_node(related_node)

    request_body = {
        "message": "Spec updated",
        "target_tabs": ["specification"],
        "related_node_id": related_node.id,
    }

    response = client.post("/api/notifications/create", json=request_body)
    assert response.status_code == 201
    data = response.json()
    notification_id = data["notification_id"]

    # Verify edge was created
    edges = db.get_outgoing_edges(notification_id)
    assert len(edges) == 1
    assert edges[0]["target_id"] == related_node.id
    assert edges[0]["type"] == "TRACES_TO"


def test_notifications_sorted_by_created_at(client, db):
    """Test that notifications are sorted by created_at descending."""
    import time

    # Create multiple notifications with slight delays
    for i in range(3):
        notification = NodeData.create(
            type=NodeType.NOTIFICATION.value,
            content=f"Notification {i}",
            data={
                "notification_type": "test",
                "source_component": "test",
                "target_tabs": ["build"],
                "urgency": "info",
                "related_node_id": None,
                "action_required": False,
                "read_by": [],
            },
            status=NodeStatus.PENDING.value,
            created_by="test",
        )
        db.add_node(notification)
        time.sleep(0.01)  # Small delay to ensure different timestamps

    response = client.get("/api/notifications/pending")
    assert response.status_code == 200
    data = response.json()
    notifications = data["notifications"]

    # Should be sorted descending (newest first)
    assert len(notifications) == 3
    assert notifications[0]["message"] == "Notification 2"
    assert notifications[1]["message"] == "Notification 1"
    assert notifications[2]["message"] == "Notification 0"


def test_notification_urgency_levels(client, db):
    """Test different urgency levels."""
    urgency_levels = ["info", "warning", "critical"]

    for urgency in urgency_levels:
        request_body = {
            "message": f"Test {urgency} message",
            "target_tabs": ["build"],
            "urgency": urgency,
        }

        response = client.post("/api/notifications/create", json=request_body)
        assert response.status_code == 201

    # Verify all were created
    response = client.get("/api/notifications/pending")
    data = response.json()
    assert len(data["notifications"]) == 3


def test_notification_multiple_target_tabs(client, db):
    """Test notification with multiple target tabs."""
    request_body = {
        "message": "Cross-tab notification",
        "target_tabs": ["build", "research", "specification"],
    }

    response = client.post("/api/notifications/create", json=request_body)
    assert response.status_code == 201
    data = response.json()

    # Verify notification
    notification_id = data["notification_id"]
    node = db.get_node(notification_id)
    assert len(node.data["target_tabs"]) == 3
    assert "build" in node.data["target_tabs"]
    assert "research" in node.data["target_tabs"]
    assert "specification" in node.data["target_tabs"]
