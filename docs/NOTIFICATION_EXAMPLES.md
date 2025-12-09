# Notification System - Usage Examples

## Quick Start

### 1. Creating a Notification from Python

```python
from agents.orchestrator import create_notification

# Basic notification
notification_id = create_notification(
    notification_type="info",
    message="Task completed successfully",
    target_tabs=["build"],
    urgency="info",
)
```

### 2. Creating a Notification via HTTP API

```bash
curl -X POST http://localhost:8000/api/notifications/create \
  -H "Content-Type: application/json" \
  -d '{
    "notification_type": "build_complete",
    "message": "Build completed successfully",
    "target_tabs": ["build"],
    "urgency": "info"
  }'
```

### 3. Getting Pending Notifications

```bash
curl http://localhost:8000/api/notifications/pending
```

## Common Patterns

### Pattern 1: Research Completion Notification

When research completes, notify the research and specification tabs:

```python
from agents.orchestrator import create_notification

def on_research_complete(research_node_id: str, task_category: str, example_count: int):
    create_notification(
        notification_type="research_complete",
        message=f"Research complete: {task_category} - {example_count} examples generated",
        target_tabs=["research", "specification"],
        urgency="info",
        related_node_id=research_node_id,
        source_component="research_agent",
    )
```

### Pattern 2: Specification Update Notification

When specs are created or updated, notify specification and build tabs:

```python
def on_specs_created(spec_count: int):
    create_notification(
        notification_type="spec_updated",
        message=f"Planning complete: {spec_count} specifications created",
        target_tabs=["specification", "build"],
        urgency="info",
        source_component="architect_agent",
    )
```

### Pattern 3: Approval Required Notification

When user approval is needed, create a high-priority notification:

```python
def request_deployment_approval(deployment_id: str):
    create_notification(
        notification_type="approval_needed",
        message="Deployment to production requires your approval",
        target_tabs=["build"],
        urgency="warning",
        action_required=True,
        related_node_id=deployment_id,
        source_component="deployment_manager",
    )
```

### Pattern 4: Critical Alert Notification

For critical issues, broadcast to all tabs:

```python
def alert_critical_failure(error_message: str, node_id: str):
    create_notification(
        notification_type="critical_error",
        message=f"Critical failure: {error_message}",
        target_tabs=["build", "research", "specification"],
        urgency="critical",
        action_required=True,
        related_node_id=node_id,
        source_component="error_monitor",
    )
```

### Pattern 5: Phase Change Notification

Notify when orchestrator changes phases:

```python
def on_phase_change(new_phase: str, session_id: str):
    create_notification(
        notification_type="phase_changed",
        message=f"Orchestrator entered {new_phase} phase",
        target_tabs=["build", "research"],
        urgency="info",
        source_component="orchestrator",
    )
```

## Advanced Usage

### Using Event Bus Directly

If you need more control, publish notifications through the event bus:

```python
from infrastructure.event_bus import publish_notification

publish_notification(
    notification_type="custom_event",
    message="Custom notification message",
    target_tabs=["build"],
    urgency="info",
    related_node_id=node_id,
    metadata={
        "custom_field": "custom_value",
        "notification_id": notification_id,
    },
    source="custom_component",
)
```

### Creating Notifications with Related Nodes

Link notifications to the nodes they reference:

```python
# Create a spec node first
from agents.tools import add_node

spec_result = add_node(
    node_type="SPEC",
    content="User authentication specification",
    created_by="architect",
)

# Create notification linked to the spec
notification_id = create_notification(
    notification_type="spec_updated",
    message="New authentication spec created",
    target_tabs=["specification"],
    urgency="info",
    related_node_id=spec_result.node_id,  # Link to the spec
)
```

### Batch Notification Creation

Create multiple notifications for different targets:

```python
def notify_milestone_complete(milestone: str, affected_nodes: List[str]):
    # Notify research team
    create_notification(
        notification_type="milestone_complete",
        message=f"Milestone '{milestone}' completed",
        target_tabs=["research"],
        urgency="info",
    )

    # Notify build team
    create_notification(
        notification_type="milestone_complete",
        message=f"Ready to build: {milestone}",
        target_tabs=["build"],
        urgency="info",
    )

    # Notify specific nodes
    for node_id in affected_nodes:
        create_notification(
            notification_type="node_update",
            message=f"Node affected by milestone: {milestone}",
            target_tabs=["specification"],
            urgency="info",
            related_node_id=node_id,
        )
```

## Frontend Integration

### WebSocket Listener (JavaScript)

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/api/viz/ws');

// Track current tab
let currentTab = 'build';  // or 'research', 'specification'

// Listen for notifications
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'notification') {
    const notification = data.data;

    // Check if notification targets current tab
    if (notification.target_tabs.includes(currentTab)) {
      displayNotification(notification);
    }
  }
};

function displayNotification(notification) {
  // Create notification UI element
  const notificationEl = document.createElement('div');
  notificationEl.className = `notification notification-${notification.urgency}`;
  notificationEl.innerHTML = `
    <div class="notification-header">
      <span class="notification-type">${notification.type}</span>
      <span class="notification-time">${new Date(notification.timestamp).toLocaleTimeString()}</span>
    </div>
    <div class="notification-message">${notification.message}</div>
    ${notification.metadata.action_required ? '<button class="notification-action">Take Action</button>' : ''}
  `;

  // Add to notification area
  document.getElementById('notification-area').appendChild(notificationEl);

  // Auto-dismiss after 5 seconds (except critical)
  if (notification.urgency !== 'critical') {
    setTimeout(() => notificationEl.remove(), 5000);
  }
}
```

### Fetching Notifications on Page Load (JavaScript)

```javascript
async function loadPendingNotifications() {
  const response = await fetch('/api/notifications/pending');
  const data = await response.json();

  // Display unread count
  document.getElementById('notification-badge').textContent = data.unread_count;

  // Display notifications
  data.notifications.forEach(notification => {
    if (notification.target_tabs.includes(currentTab)) {
      displayNotification(notification);
    }
  });
}

// Load on page load
loadPendingNotifications();
```

### Marking Notifications as Read (JavaScript)

```javascript
async function markNotificationRead(notificationId) {
  const response = await fetch(`/api/notifications/${notificationId}/mark-read`, {
    method: 'POST',
  });

  if (response.ok) {
    // Update UI
    document.getElementById(`notification-${notificationId}`).classList.add('read');
    updateUnreadCount();
  }
}

async function updateUnreadCount() {
  const response = await fetch('/api/notifications/pending');
  const data = await response.json();
  document.getElementById('notification-badge').textContent = data.unread_count;
}
```

## Testing Notifications

### Manual Testing via cURL

```bash
# Create a test notification
curl -X POST http://localhost:8000/api/notifications/create \
  -H "Content-Type: application/json" \
  -d '{
    "notification_type": "test",
    "message": "Test notification",
    "target_tabs": ["build"],
    "urgency": "info"
  }'

# Get pending notifications
curl http://localhost:8000/api/notifications/pending | jq

# Mark as read (replace NOTIFICATION_ID)
curl -X POST http://localhost:8000/api/notifications/NOTIFICATION_ID/mark-read
```

### Testing with WebSocket

```python
import asyncio
import websockets
import json

async def test_websocket_notifications():
    uri = "ws://localhost:8000/api/viz/ws"

    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")

        # Listen for messages
        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data.get('type') == 'notification':
                print(f"Received notification: {data['data']['message']}")

asyncio.run(test_websocket_notifications())
```

## Best Practices

### 1. Choose Appropriate Urgency

```python
# Info: routine updates, completions
create_notification(urgency="info", ...)

# Warning: requires attention but not urgent
create_notification(urgency="warning", ...)

# Critical: requires immediate action
create_notification(urgency="critical", action_required=True, ...)
```

### 2. Target the Right Tabs

```python
# Research-related: notify research and spec tabs
create_notification(target_tabs=["research", "specification"], ...)

# Build-related: notify build tab
create_notification(target_tabs=["build"], ...)

# System-wide: notify all tabs
create_notification(target_tabs=["build", "research", "specification"], ...)
```

### 3. Link to Related Nodes

Always link notifications to related nodes when applicable:

```python
create_notification(
    related_node_id=node_id,  # Link to the relevant node
    ...
)
```

### 4. Use Descriptive Messages

```python
# Bad
create_notification(message="Done", ...)

# Good
create_notification(
    message="Research complete: algorithmic task - 5 examples generated",
    ...
)
```

### 5. Handle Errors Gracefully

```python
try:
    notification_id = create_notification(...)
    if notification_id:
        logger.info(f"Notification created: {notification_id}")
except Exception as e:
    logger.error(f"Failed to create notification: {e}")
    # Continue execution - don't let notification failures break core logic
```

## Notification Types Reference

| Type | Description | Suggested Tabs | Urgency |
|------|-------------|----------------|---------|
| `spec_updated` | Specification created/updated | specification, build | info |
| `research_complete` | Research phase completed | research, specification | info |
| `approval_needed` | User approval required | build | warning |
| `phase_changed` | Orchestrator phase changed | build, research | info |
| `build_complete` | Build phase completed | build | info |
| `test_results` | Test results available | build | info/warning |
| `critical_error` | Critical system error | all tabs | critical |
| `deployment_ready` | Ready for deployment | build | info |
| `feedback_required` | User feedback needed | research, specification | warning |

## Troubleshooting

### Notifications Not Appearing

1. Check WebSocket connection:
```javascript
ws.onopen = () => console.log('WebSocket connected');
ws.onerror = (error) => console.error('WebSocket error:', error);
```

2. Verify notification was created:
```bash
curl http://localhost:8000/api/notifications/pending
```

3. Check target tabs match current tab:
```javascript
console.log('Current tab:', currentTab);
console.log('Notification targets:', notification.target_tabs);
```

### Unread Count Not Updating

Ensure notifications are being marked as read:
```javascript
// After user dismisses notification
await markNotificationRead(notification.notification_id);
```

### WebSocket Disconnecting

Add reconnection logic:
```javascript
function connectWebSocket() {
  const ws = new WebSocket('ws://localhost:8000/api/viz/ws');

  ws.onclose = () => {
    console.log('WebSocket disconnected, reconnecting...');
    setTimeout(connectWebSocket, 1000);
  };

  return ws;
}
```
