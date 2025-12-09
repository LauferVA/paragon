# Research Feedback System Backend - Integration Guide

## Overview
This document describes the implementation of the research feedback system backend for Paragon, allowing users to provide feedback on research tasks from the Research Tab.

## Files Modified

### 1. `/Users/lauferva/paragon/core/ontology.py`
**Change:** Added new edge type `HAS_FEEDBACK`
```python
HAS_FEEDBACK = "HAS_FEEDBACK"    # User feedback: RESEARCH -> FEEDBACK node
```

### 2. `/Users/lauferva/paragon/infrastructure/event_bus.py`
**Change:** Added two new event types
```python
RESEARCH_FEEDBACK_RECEIVED = "research_feedback_received"
RESEARCH_TASK_COMPLETED = "research_task_completed"
```

## New Files Created

### 3. `/Users/lauferva/paragon/api/research_feedback_endpoints.py`
Contains four new endpoint implementations:
- `get_research_task(request)` - GET /api/research/{research_task_id}
- `submit_research_feedback(request, _ws_connections, _next_sequence, broadcast_delta)` - POST /api/research/{research_task_id}/feedback
- `submit_research_response(request, _ws_connections, _next_sequence, broadcast_delta)` - POST /api/research/{research_task_id}/response
- `get_active_research_tasks(request)` - GET /api/research/tasks/active

## Integration Steps

### Step 1: Import the new endpoints in `/Users/lauferva/paragon/api/routes.py`

Add at the top of the file (around line 55, after other imports):
```python
# Research feedback endpoints
from api.research_feedback_endpoints import (
    get_research_task,
    submit_research_feedback as _submit_research_feedback,
    submit_research_response as _submit_research_response,
    get_active_research_tasks,
)
```

### Step 2: Create wrapper functions in routes.py

Add these wrapper functions (after the existing endpoint functions, before `create_routes()`):
```python
# =============================================================================
# RESEARCH FEEDBACK ENDPOINTS
# =============================================================================

async def submit_research_feedback(request: Request) -> JSONResponse:
    """Wrapper for research feedback endpoint."""
    return await _submit_research_feedback(request, _ws_connections, _next_sequence, broadcast_delta)


async def submit_research_response(request: Request) -> JSONResponse:
    """Wrapper for research response endpoint."""
    return await _submit_research_response(request, _ws_connections, _next_sequence, broadcast_delta)
```

### Step 3: Register routes in `create_routes()` function

In the `create_routes()` function (around line 3561, after the existing research route), add:
```python
        Route("/api/research/nodes", get_research_nodes, methods=["GET"]),
        # Research feedback endpoints
        Route("/api/research/{research_task_id}", get_research_task, methods=["GET"]),
        Route("/api/research/{research_task_id}/feedback", submit_research_feedback, methods=["POST"]),
        Route("/api/research/{research_task_id}/response", submit_research_response, methods=["POST"]),
        Route("/api/research/tasks/active", get_active_research_tasks, methods=["GET"]),
```

### Step 4: Subscribe to research feedback events in WebSocket handler

In the `viz_websocket()` function (around line 1066, after sending initial snapshot), add event subscription:
```python
        # Subscribe to research feedback events
        if EVENT_BUS_AVAILABLE:
            event_bus = get_event_bus()

            async def on_research_feedback(event: GraphEvent):
                """Handle research feedback events and broadcast to clients."""
                await broadcast_json({
                    "type": "research_feedback",
                    "data": event.payload,
                    "timestamp": event.timestamp,
                })

            async def on_research_completed(event: GraphEvent):
                """Handle research completion events and broadcast to clients."""
                await broadcast_json({
                    "type": "research_completed",
                    "data": event.payload,
                    "timestamp": event.timestamp,
                })

            # Subscribe to events
            event_bus.subscribe_async(EventType.RESEARCH_FEEDBACK_RECEIVED, on_research_feedback)
            event_bus.subscribe_async(EventType.RESEARCH_TASK_COMPLETED, on_research_completed)
```

## API Endpoints Documentation

### 1. GET /api/research/{research_task_id}
Get a single research task with full metadata including user feedback state.

**Response:**
```json
{
  "node_id": "research_abc123",
  "req_node_id": "req_xyz789",
  "iteration": 1,
  "query": "Research query text",
  "total_findings": 5,
  "total_ambiguities": 2,
  "blocking_count": 0,
  "out_of_scope": [],
  "synthesis": "Research synthesis text",
  "findings": [...],
  "ambiguities": [...],
  "search_results": [...],
  "created_at": "2025-12-08T10:30:00Z",
  "status": "PENDING",
  "user_approval_required": false,
  "user_approval_state": "pending",
  "user_feedback": null,
  "user_feedback_timestamp": null,
  "user_feedback_node_id": null,
  "awaiting_user_action": false,
  "hover_metadata": {
    "findings_tooltip": "5 findings from research",
    "ambiguities_tooltip": "2 ambiguities detected",
    "synthesis_tooltip": "Research synthesis and conclusions"
  }
}
```

### 2. POST /api/research/{research_task_id}/feedback
Submit user feedback on a research task.

**Request Body:**
```json
{
  "feedback": "This research looks good but needs more details on X",
  "metadata": {
    "rating": 4,
    "issues": ["Missing detail on API contracts"],
    "suggestions": ["Add more examples"]
  }
}
```

**Response:**
```json
{
  "success": true,
  "feedback_node_id": "msg_feedback_abc123",
  "research_node_id": "research_abc123",
  "message": "Feedback submitted successfully"
}
```

**Side Effects:**
- Creates MESSAGE node with type "research_feedback"
- Creates HAS_FEEDBACK edge from RESEARCH -> MESSAGE
- Updates research node with feedback reference
- Publishes RESEARCH_FEEDBACK_RECEIVED event
- Broadcasts to WebSocket clients

### 3. POST /api/research/{research_task_id}/response
Submit structured response (approve/revise/clarify).

**Request Body:**
```json
{
  "action": "approve",
  "message": "Research looks good, proceed to planning",
  "context": {
    "notes": "All key requirements captured"
  }
}
```

**Actions:**
- `approve`: Marks research as VERIFIED, triggers phase transition to PLAN
- `revise`: Marks research as PENDING with revision_requested state
- `clarify`: Marks research as PENDING with denied state

**Response:**
```json
{
  "success": true,
  "action": "approve",
  "message_id": "msg_user_orch_abc123",
  "phase_transition": "RESEARCH -> PLAN"
}
```

**Side Effects:**
- Sends agent message to orchestrator inbox
- Updates research node status and approval state
- Publishes RESEARCH_TASK_COMPLETED event (if approved)
- Broadcasts to WebSocket clients

### 4. GET /api/research/tasks/active
Get all research tasks awaiting user feedback.

**Query Parameters:**
- `limit` (optional, default 50): Maximum number of tasks to return

**Response:**
```json
{
  "tasks": [
    {
      "node_id": "research_abc123",
      "req_node_id": "req_xyz789",
      "query": "Research query",
      "synthesis": "Research synthesis (truncated)...",
      "total_findings": 5,
      "total_ambiguities": 2,
      "awaiting_user_action": true,
      "user_approval_state": "pending",
      "created_at": "2025-12-08T10:30:00Z",
      "action_hint": "Review research findings and approve or request revision"
    }
  ],
  "count": 1
}
```

## Research Node State Fields

The following fields are added to RESEARCH node data dictionary:

```python
{
  "user_approval_required": bool,      # Whether user approval is needed
  "user_approval_state": str,          # "pending" | "approved" | "denied" | "revision_requested"
  "user_feedback": Optional[str],      # User's feedback text
  "user_feedback_timestamp": Optional[str],  # ISO 8601 timestamp
  "user_feedback_node_id": Optional[str],    # ID of feedback MESSAGE node
  "awaiting_user_action": bool,        # Whether waiting for user input
}
```

## WebSocket Message Types

### research_feedback
Sent when user submits feedback on a research task.
```json
{
  "type": "research_feedback",
  "data": {
    "research_node_id": "research_abc123",
    "feedback_node_id": "msg_feedback_abc123",
    "feedback_text": "...",
    "metadata": {...}
  },
  "timestamp": 1702123456.789
}
```

### research_completed
Sent when user approves a research task.
```json
{
  "type": "research_completed",
  "data": {
    "research_node_id": "research_abc123",
    "approved": true,
    "message_id": "msg_user_orch_abc123"
  },
  "timestamp": 1702123456.789
}
```

## Example API Usage

### Example 1: Get a research task
```bash
curl http://localhost:8000/api/research/research_abc123
```

### Example 2: Submit feedback
```bash
curl -X POST http://localhost:8000/api/research/research_abc123/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "feedback": "Great research, but please add more edge cases",
    "metadata": {
      "rating": 4,
      "issues": ["Missing edge cases"]
    }
  }'
```

### Example 3: Approve research
```bash
curl -X POST http://localhost:8000/api/research/research_abc123/response \
  -H "Content-Type: application/json" \
  -d '{
    "action": "approve",
    "message": "Research approved, proceed to planning"
  }'
```

### Example 4: Get active research tasks
```bash
curl http://localhost:8000/api/research/tasks/active?limit=10
```

## Testing

Test file location: `/Users/lauferva/paragon/tests/unit/api/test_research_feedback.py`

Run tests:
```bash
pytest tests/unit/api/test_research_feedback.py -v
```

## Architecture Notes

### Graph-Native Design
- Feedback is stored as MESSAGE nodes in the graph
- HAS_FEEDBACK edges create explicit relationships
- All state changes are persisted in the graph
- Event bus provides real-time notifications

### Bicameral Mind Pattern
- Endpoints (Layer 7A) handle probabilistic user input
- State validation (Layer 7B) enforces graph constraints
- Events flow through infrastructure layer for decoupling

### Integration with Agent Messages
The `/response` endpoint uses the inter-agent messaging system (`agents/agent_messages.py`) to communicate with the orchestrator, ensuring all agent communication flows through the graph.

## Future Enhancements

1. **Notification System**: Integrate with cross-tab notifications when research tasks need attention
2. **Analytics**: Track approval rates, common feedback patterns
3. **Batch Operations**: Allow approving/rejecting multiple research tasks at once
4. **Rich Feedback**: Support structured feedback with specific field annotations
5. **Audit Trail**: Query history of all feedback on a research task

## Troubleshooting

### Issue: Feedback not appearing in graph
- Check that HAS_FEEDBACK edge was created
- Verify feedback node was added successfully
- Check event bus logs for event publishing errors

### Issue: WebSocket not receiving updates
- Verify event bus subscription in viz_websocket
- Check that EVENT_BUS_AVAILABLE is True
- Ensure broadcast_delta is being called

### Issue: Phase transition not triggering
- Verify research node status was updated to VERIFIED
- Check that agent message was sent to orchestrator
- Review orchestrator logs for message processing
