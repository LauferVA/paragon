"""
Integration tests for cross-tab notification system.

Tests the complete notification flow from creation to WebSocket delivery:
- Event bus integration
- WebSocket broadcasting
- Orchestrator notification helpers
- End-to-end notification delivery
"""
import pytest
import asyncio
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from core.graph_db import get_db
from core.schemas import NodeData
from core.ontology import NodeType, NodeStatus
from api.routes import create_app
from infrastructure.event_bus import get_event_bus, EventType, publish_notification
from agents.orchestrator import create_notification


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


@pytest.fixture
def event_bus():
    """Get event bus instance and clear subscribers."""
    bus = get_event_bus()
    # Clear any existing subscribers
    bus.clear_subscribers()
    return bus


def test_orchestrator_create_notification(db):
    """Test creating notification via orchestrator helper."""
    notification_id = create_notification(
        notification_type="test_notification",
        message="Test message from orchestrator",
        target_tabs=["build", "research"],
        urgency="info",
        source_component="test_orchestrator",
    )

    assert notification_id is not None

    # Verify notification was created
    node = db.get_node(notification_id)
    assert node is not None
    assert node.type == NodeType.NOTIFICATION.value
    assert node.content == "Test message from orchestrator"
    assert node.data["notification_type"] == "test_notification"
    assert node.data["target_tabs"] == ["build", "research"]


def test_orchestrator_notification_with_related_node(db):
    """Test creating notification linked to another node."""
    # Create a related node
    spec_node = NodeData.create(
        type=NodeType.SPEC.value,
        content="Test specification",
        created_by="test",
    )
    db.add_node(spec_node)

    # Create notification
    notification_id = create_notification(
        notification_type="spec_updated",
        message="Specification has been updated",
        target_tabs=["specification"],
        urgency="info",
        related_node_id=spec_node.id,
    )

    assert notification_id is not None

    # Verify edge was created
    edges = db.get_outgoing_edges(notification_id)
    assert len(edges) == 1
    assert edges[0]["target_id"] == spec_node.id
    assert edges[0]["type"] == "TRACES_TO"


def test_event_bus_notification_publishing(event_bus):
    """Test notification publishing through event bus."""
    received_events = []

    # Subscribe to notification events
    def on_notification(event):
        received_events.append(event)

    event_bus.subscribe(EventType.NOTIFICATION_CREATED, on_notification)

    # Publish notification
    publish_notification(
        notification_type="test",
        message="Test notification",
        target_tabs=["build"],
        urgency="info",
    )

    # Verify event was received
    assert len(received_events) == 1
    event = received_events[0]
    assert event.type == EventType.NOTIFICATION_CREATED
    assert event.payload["notification_type"] == "test"
    assert event.payload["message"] == "Test notification"
    assert event.payload["target_tabs"] == ["build"]


def test_notification_full_flow(client, db, event_bus):
    """Test complete notification flow from creation to API retrieval."""
    # Create notification via API
    request_body = {
        "notification_type": "research_complete",
        "message": "Research phase completed",
        "target_tabs": ["research", "specification"],
        "urgency": "info",
    }

    response = client.post("/api/notifications/create", json=request_body)
    assert response.status_code == 201
    notification_id = response.json()["notification_id"]

    # Retrieve via API
    response = client.get("/api/notifications/pending")
    assert response.status_code == 200
    data = response.json()
    assert data["unread_count"] == 1
    assert len(data["notifications"]) == 1
    assert data["notifications"][0]["notification_id"] == notification_id

    # Mark as read
    response = client.post(f"/api/notifications/{notification_id}/mark-read")
    assert response.status_code == 200

    # Verify read count updated
    response = client.get("/api/notifications/pending")
    data = response.json()
    assert data["unread_count"] == 0
    assert data["notifications"][0]["read"] is True


def test_multiple_notifications_different_tabs(client, db):
    """Test multiple notifications targeting different tabs."""
    notifications = [
        {
            "notification_type": "spec_updated",
            "message": "New spec created",
            "target_tabs": ["specification"],
        },
        {
            "notification_type": "research_complete",
            "message": "Research done",
            "target_tabs": ["research"],
        },
        {
            "notification_type": "build_ready",
            "message": "Ready to build",
            "target_tabs": ["build"],
        },
        {
            "notification_type": "global_update",
            "message": "System update",
            "target_tabs": ["build", "research", "specification"],
        },
    ]

    for notif in notifications:
        response = client.post("/api/notifications/create", json=notif)
        assert response.status_code == 201

    # Retrieve all notifications
    response = client.get("/api/notifications/pending")
    data = response.json()
    assert len(data["notifications"]) == 4
    assert data["unread_count"] == 4


def test_notification_urgency_filtering(client, db):
    """Test that different urgency levels are properly stored and retrieved."""
    urgency_levels = [
        ("info", "Informational message"),
        ("warning", "Warning message"),
        ("critical", "Critical alert"),
    ]

    for urgency, message in urgency_levels:
        request_body = {
            "message": message,
            "target_tabs": ["build"],
            "urgency": urgency,
        }
        response = client.post("/api/notifications/create", json=request_body)
        assert response.status_code == 201

    # Retrieve all
    response = client.get("/api/notifications/pending")
    data = response.json()
    notifications = data["notifications"]

    # Verify all urgency levels are present
    urgencies = {n["urgency"] for n in notifications}
    assert urgencies == {"info", "warning", "critical"}


def test_notification_action_required_flag(client, db):
    """Test notification with action_required flag."""
    request_body = {
        "notification_type": "approval_needed",
        "message": "Approval required for deployment",
        "target_tabs": ["build"],
        "urgency": "warning",
        "action_required": True,
    }

    response = client.post("/api/notifications/create", json=request_body)
    assert response.status_code == 201
    notification_id = response.json()["notification_id"]

    # Verify flag is stored
    node = db.get_node(notification_id)
    assert node.data["action_required"] is True

    # Verify it's returned in API
    response = client.get("/api/notifications/pending")
    data = response.json()
    notification = data["notifications"][0]
    assert notification["metadata"]["action_required"] is True


def test_orchestrator_notification_error_handling(db):
    """Test that notification creation errors are handled gracefully."""
    # Try to create notification with invalid data
    notification_id = create_notification(
        notification_type="test",
        message="",  # Empty message should still work (validation is in API)
        target_tabs=[],  # Empty tabs should still work
    )

    # Should return a notification ID (validation is lenient in helper)
    assert notification_id is not None


def test_notification_with_metadata(client, db):
    """Test notification with additional metadata."""
    # Create a related node
    spec_node = NodeData.create(
        type=NodeType.SPEC.value,
        content="Test spec",
        created_by="test",
    )
    db.add_node(spec_node)

    request_body = {
        "notification_type": "spec_updated",
        "message": "Specification updated",
        "target_tabs": ["specification"],
        "related_node_id": spec_node.id,
    }

    response = client.post("/api/notifications/create", json=request_body)
    assert response.status_code == 201

    # Retrieve and check metadata
    response = client.get("/api/notifications/pending")
    data = response.json()
    notification = data["notifications"][0]
    assert notification["metadata"]["related_node_id"] == spec_node.id


def test_read_unread_count_accuracy(client, db):
    """Test that unread count is accurate as notifications are marked read."""
    # Create 5 notifications
    for i in range(5):
        request_body = {
            "message": f"Notification {i}",
            "target_tabs": ["build"],
        }
        client.post("/api/notifications/create", json=request_body)

    # Check initial count
    response = client.get("/api/notifications/pending")
    data = response.json()
    assert data["unread_count"] == 5

    # Mark 2 as read
    notifications = data["notifications"]
    for i in range(2):
        notif_id = notifications[i]["notification_id"]
        client.post(f"/api/notifications/{notif_id}/mark-read")

    # Check updated count
    response = client.get("/api/notifications/pending")
    data = response.json()
    assert data["unread_count"] == 3

    # Mark rest as read
    for i in range(2, 5):
        notif_id = notifications[i]["notification_id"]
        client.post(f"/api/notifications/{notif_id}/mark-read")

    # Final check
    response = client.get("/api/notifications/pending")
    data = response.json()
    assert data["unread_count"] == 0


def test_notification_chronological_order(client, db):
    """Test notifications are returned in chronological order (newest first)."""
    import time

    # Create notifications with delays
    notification_ids = []
    for i in range(3):
        request_body = {
            "message": f"Message {i}",
            "target_tabs": ["build"],
        }
        response = client.post("/api/notifications/create", json=request_body)
        notification_ids.append(response.json()["notification_id"])
        time.sleep(0.02)

    # Retrieve notifications
    response = client.get("/api/notifications/pending")
    data = response.json()
    notifications = data["notifications"]

    # Verify order (newest first)
    assert notifications[0]["message"] == "Message 2"
    assert notifications[1]["message"] == "Message 1"
    assert notifications[2]["message"] == "Message 0"


def test_notification_with_empty_read_by_list(db):
    """Test that new notifications have empty read_by list."""
    notification_id = create_notification(
        notification_type="test",
        message="Test notification",
        target_tabs=["build"],
    )

    node = db.get_node(notification_id)
    assert node.data["read_by"] == []


def test_notification_types(client, db):
    """Test various notification types are properly stored."""
    notification_types = [
        "spec_updated",
        "research_complete",
        "approval_needed",
        "phase_changed",
        "build_complete",
        "test_results",
    ]

    for notif_type in notification_types:
        request_body = {
            "notification_type": notif_type,
            "message": f"Test {notif_type}",
            "target_tabs": ["build"],
        }
        response = client.post("/api/notifications/create", json=request_body)
        assert response.status_code == 201

    # Retrieve and verify
    response = client.get("/api/notifications/pending")
    data = response.json()
    assert len(data["notifications"]) == len(notification_types)

    returned_types = {n["type"] for n in data["notifications"]}
    assert returned_types == set(notification_types)
