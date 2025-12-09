# Cross-Tab Notification System

## Overview

The cross-tab notification system allows events in one part of the Paragon system (spec updates, research completion) to trigger notifications that appear across different UI tabs in real-time via WebSocket connections.

## Architecture

```
Orchestrator/Agent
    |
    v
create_notification() --> Graph DB (NOTIFICATION node)
    |                           |
    v                           v
Event Bus                  TRACES_TO edge (optional)
    |
    v
WebSocket Broadcast
    |
    v
UI Client (all connected tabs)
```

## Components

### 1. Node Type (core/ontology.py)

Added `NOTIFICATION` node type with topology constraint:

```python
NodeType.NOTIFICATION = "NOTIFICATION"  # Cross-tab notification

# Topology constraint
NodeType.NOTIFICATION.value: TopologyConstraint(
    node_type=NodeType.NOTIFICATION.value,
    description="Cross-tab notification for UI events. May link to source node via TRACES_TO.",
    edge_constraints=(
        EdgeConstraint(
            edge_type=EdgeType.TRACES_TO.value,
            direction="outgoing",
            min_count=0,  # Optional - may be global notification
            mode=ConstraintMode.SOFT.value,
        ),
    ),
    allowed_statuses=(
        NodeStatus.PENDING.value,   # Unread notification
        NodeStatus.VERIFIED.value,  # Read notification
    ),
)
```

### 2. Notification Node Schema

NOTIFICATION nodes have the following structure:

```python
NodeData(
    type="NOTIFICATION",
    content="Notification message text",
    data={
        "notification_type": str,  # "spec_updated" | "research_complete" | "approval_needed" | "phase_changed"
        "source_component": str,   # Component that created notification (e.g., "orchestrator")
        "target_tabs": List[str],  # ["build", "research", "specification"]
        "urgency": str,            # "info" | "warning" | "critical"
        "related_node_id": Optional[str],  # Optional related node ID
        "action_required": bool,   # Whether user action is required
        "read_by": List[str],      # List of users who have read this notification
    },
    status=NodeStatus.PENDING,  # PENDING = unread, VERIFIED = read
)
```

### 3. Event Types (infrastructure/event_bus.py)

Added notification event types:

```python
class EventType(str, Enum):
    NOTIFICATION_CREATED = "notification_created"
    CROSS_TAB_NOTIFICATION = "cross_tab_notification"
    MESSAGE_CREATED = "message_created"
    DIALOGUE_TURN_ADDED = "dialogue_turn_added"
```

### 4. API Endpoints (api/routes.py)

#### GET /api/notifications/pending

Get all pending notifications for the current user/session.

**Response:**
```json
{
  "notifications": [
    {
      "notification_id": "abc123...",
      "type": "spec_updated",
      "source": "orchestrator",
      "message": "New requirement added",
      "target_tabs": ["build", "specification"],
      "urgency": "info",
      "metadata": {
        "related_node_id": "node_id",
        "action_required": false
      },
      "created_at": "2025-01-15T10:30:00Z",
      "read": false
    }
  ],
  "unread_count": 5
}
```

#### POST /api/notifications/{notification_id}/mark-read

Mark a notification as read.

**Response:**
```json
{
  "status": "success",
  "notification_id": "abc123..."
}
```

#### POST /api/notifications/create

Programmatic endpoint for creating notifications from orchestrator/agents.

**Request Body:**
```json
{
  "notification_type": "spec_updated",
  "message": "Notification message text",
  "target_tabs": ["build", "research", "specification"],
  "urgency": "info",
  "source_component": "orchestrator",
  "related_node_id": "optional-node-id",
  "action_required": false
}
```

**Response:**
```json
{
  "status": "success",
  "notification_id": "abc123..."
}
```

### 5. WebSocket Integration

Notifications are broadcasted to all connected WebSocket clients via the `/api/viz/ws` endpoint.

**WebSocket Message Format:**
```json
{
  "type": "notification",
  "data": {
    "notification_id": "abc123...",
    "type": "spec_updated",
    "message": "New requirement added",
    "target_tabs": ["build", "specification"],
    "urgency": "info",
    "metadata": {
      "notification_id": "abc123...",
      "action_required": false
    },
    "related_node_id": "node_id",
    "timestamp": "2025-01-15T10:30:00Z",
    "source": "orchestrator"
  }
}
```

The WebSocket handler subscribes to `EventType.NOTIFICATION_CREATED` events on application startup:

```python
event_bus.subscribe_async(EventType.NOTIFICATION_CREATED, broadcast_graph_event)
```

### 6. Orchestrator Helper (agents/orchestrator.py)

The `create_notification()` helper function allows orchestrator and agents to easily create notifications:

```python
def create_notification(
    notification_type: str,
    message: str,
    target_tabs: List[str],
    urgency: str = "info",
    related_node_id: Optional[str] = None,
    action_required: bool = False,
    source_component: str = "orchestrator",
) -> Optional[str]:
    """
    Create and publish a notification.

    Returns:
        Notification node ID if successful, None otherwise
    """
```

**Example Usage:**

```python
from agents.orchestrator import create_notification

# Notify when research completes
create_notification(
    notification_type="research_complete",
    message="Research findings available for review",
    target_tabs=["research", "specification"],
    urgency="info",
    related_node_id=research_node_id,
)
```

### 7. Auto-Notification Triggers

The system automatically creates notifications for key events:

#### Research Completion

Triggered in `research_node()` when research phase completes:

```python
create_notification(
    notification_type="research_complete",
    message=f"Research complete: {result.task_category} - {len(result.happy_path_examples)} examples generated",
    target_tabs=["research", "specification"],
    urgency="info",
    source_component="orchestrator",
)
```

#### Specification Creation

Triggered in `plan_node()` when specs are created:

```python
create_notification(
    notification_type="spec_updated",
    message=f"Planning complete: {len(spec_node_ids)} specifications created",
    target_tabs=["specification", "build"],
    urgency="info",
    source_component="orchestrator",
)
```

## Notification Types

Supported notification types:

- **spec_updated**: Specification created or updated
- **research_complete**: Research phase completed
- **approval_needed**: User approval required
- **phase_changed**: Orchestrator phase changed
- **build_complete**: Build phase completed
- **test_results**: Test results available

## Urgency Levels

- **info**: Informational message (default)
- **warning**: Warning that requires attention
- **critical**: Critical alert requiring immediate action

## Target Tabs

Notifications can target one or more UI tabs:

- **build**: Build/implementation tab
- **research**: Research tab
- **specification**: Specification/planning tab

## Testing

### Unit Tests

Located in `/Users/lauferva/paragon/tests/unit/api/test_notifications.py`

Tests cover:
- Getting pending notifications
- Marking notifications as read
- Creating notifications via API
- Validation (missing fields, invalid urgency)
- Linking notifications to related nodes
- Sorting by created_at timestamp
- Multiple urgency levels and target tabs

Run tests:
```bash
pytest tests/unit/api/test_notifications.py -v
```

### Integration Tests

Located in `/Users/lauferva/paragon/tests/integration/test_cross_tab_notifications.py`

Tests cover:
- Orchestrator notification helper
- Event bus integration
- Complete notification flow (create -> retrieve -> mark read)
- Multiple notifications with different tabs
- Urgency filtering
- Action required flag
- Read/unread count accuracy
- Chronological ordering

Run tests:
```bash
pytest tests/integration/test_cross_tab_notifications.py -v
```

## Examples

### Creating a Notification from Orchestrator

```python
from agents.orchestrator import create_notification

# Simple info notification
notification_id = create_notification(
    notification_type="phase_changed",
    message="Entered BUILD phase",
    target_tabs=["build"],
    urgency="info",
)

# Warning with action required
notification_id = create_notification(
    notification_type="approval_needed",
    message="Deployment requires approval",
    target_tabs=["build", "specification"],
    urgency="warning",
    action_required=True,
)

# Critical alert linked to node
notification_id = create_notification(
    notification_type="test_failed",
    message="Critical test failure detected",
    target_tabs=["build"],
    urgency="critical",
    related_node_id=code_node_id,
    action_required=True,
)
```

### Creating a Notification via API

```bash
curl -X POST http://localhost:8000/api/notifications/create \
  -H "Content-Type: application/json" \
  -d '{
    "notification_type": "spec_updated",
    "message": "New specification created",
    "target_tabs": ["specification", "build"],
    "urgency": "info"
  }'
```

### Getting Pending Notifications

```bash
curl http://localhost:8000/api/notifications/pending
```

### Marking Notification as Read

```bash
curl -X POST http://localhost:8000/api/notifications/abc123.../mark-read
```

### Listening to WebSocket Notifications

```javascript
const ws = new WebSocket('ws://localhost:8000/api/viz/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'notification') {
    const notification = data.data;
    console.log('Received notification:', notification);

    // Check if this tab should show the notification
    if (notification.target_tabs.includes(currentTab)) {
      showNotification(notification);
    }
  }
};
```

## Implementation Checklist

- [x] Add NOTIFICATION NodeType to core/ontology.py
- [x] Add notification event types to infrastructure/event_bus.py
- [x] Create notification API endpoints in api/routes.py
- [x] Add WebSocket notification broadcasting
- [x] Create orchestrator notification helper
- [x] Add auto-notification triggers for key events
- [x] Write unit tests for notification system
- [x] Write integration tests for cross-tab notifications

## Future Enhancements

1. **Notification Filtering**: Allow clients to filter notifications by tab, urgency, or type
2. **Notification History**: Store read notifications for a configurable period
3. **Notification Preferences**: User preferences for notification delivery
4. **Batch Marking**: Mark multiple notifications as read in one request
5. **Push Notifications**: Integration with browser push notifications
6. **Notification Templates**: Predefined templates for common notification types
7. **Notification Scheduling**: Schedule notifications for future delivery
8. **Notification Grouping**: Group related notifications together
